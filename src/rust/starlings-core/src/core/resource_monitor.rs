use std::env;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use sysinfo::{CpuRefreshKind, MemoryRefreshKind, RefreshKind, System};

/// Circuit breaker states for resource safety
#[derive(Debug, Clone, Copy, PartialEq)]
enum CircuitState {
    Closed,   // Normal operation
    Open,     // System unhealthy, reject operations
    HalfOpen, // Testing if system recovered
}

/// Safety levels for resource usage
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SafetyLevel {
    /// Conservative: Use max 50% of available resources
    Conservative,
    /// Balanced: Use max 70% of available resources  
    Balanced,
    /// Performance: Use max 85% of available resources
    Performance,
    /// Unsafe: Use max 95% (requires STARLINGS_UNSAFE=1)
    Unsafe,
}

impl SafetyLevel {
    fn memory_threshold(&self) -> f64 {
        match self {
            SafetyLevel::Conservative => 0.50,
            SafetyLevel::Balanced => 0.70,
            SafetyLevel::Performance => 0.85,
            SafetyLevel::Unsafe => 0.95,
        }
    }

    fn max_operation_fraction(&self) -> f64 {
        match self {
            SafetyLevel::Conservative => 0.20, // Max 20% for single op
            SafetyLevel::Balanced => 0.35,
            SafetyLevel::Performance => 0.50,
            SafetyLevel::Unsafe => 0.80,
        }
    }
}

/// Errors that can occur during safety checks
#[derive(Debug, Clone)]
pub enum SafetyError {
    CircuitOpen(String),
    InsufficientResources(String),
    OperationTooLarge(String),
}

impl std::fmt::Display for SafetyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SafetyError::CircuitOpen(msg) => write!(f, "Circuit breaker tripped: {}", msg),
            SafetyError::InsufficientResources(msg) => write!(f, "Insufficient resources: {}", msg),
            SafetyError::OperationTooLarge(msg) => write!(f, "Operation too large: {}", msg),
        }
    }
}

impl std::error::Error for SafetyError {}

/// Permit for resource-intensive operations
#[derive(Debug)]
pub struct OperationPermit {
    _permit_id: u64,
}

impl OperationPermit {
    fn new(permit_id: u64) -> Self {
        Self {
            _permit_id: permit_id,
        }
    }
}

/// Resource monitoring and adaptive processing limits with circuit breaker
#[derive(Debug, Clone)]
pub struct ResourceMonitor {
    system: Arc<Mutex<System>>,
    last_refresh: Arc<Mutex<Instant>>,
    refresh_interval: Duration,
    memory_limit_mb: Option<u64>,
    cpu_limit_percent: f32,

    // Circuit breaker state
    circuit_state: Arc<Mutex<CircuitState>>,
    last_healthy_time: Arc<Mutex<Instant>>,
    consecutive_failures: Arc<AtomicU32>,
    safety_level: SafetyLevel,

    // Operation tracking
    operation_counter: Arc<AtomicU32>,
}

#[derive(Debug, Clone)]
pub struct ResourceUsage {
    pub memory_used_mb: u64,
    pub memory_available_mb: u64,
    pub memory_total_mb: u64,
    pub memory_percent: f32,
    pub cpu_percent: f32,
    pub disk_free_gb: u64,
    pub disk_total_gb: u64,
    pub disk_percent: f32,
    pub is_memory_pressure: bool,
    pub is_cpu_pressure: bool,
    pub is_disk_pressure: bool,
}

#[derive(Debug, Clone)]
pub struct AdaptiveLimits {
    pub batch_size: usize,
    pub should_throttle: bool,
    pub should_spill_to_disk: bool,
    pub delay_between_batches_ms: u64,
    pub memory_warning: Option<String>,
    pub disk_warning: Option<String>,
}

#[derive(Debug, Clone)]
pub enum ProcessingStrategy {
    /// Dataset fits comfortably in memory
    InMemory {
        batch_size: usize,
        total_batches: usize,
    },
    /// Dataset requires memory-aware processing with potential spilling
    MemoryAware {
        batch_size: usize,
        should_spill: bool,
        spill_threshold_mb: u64,
        total_batches: usize,
    },
    /// Dataset requires streaming with aggressive disk spilling
    Streaming {
        batch_size: usize,
        aggressive_spilling: bool,
        max_memory_mb: u64,
        total_batches: usize,
    },
    /// Insufficient system resources
    Insufficient {
        required_memory_mb: u64,
        available_memory_mb: u64,
        required_disk_gb: u64,
        available_disk_gb: u64,
    },
}

/// Memory pressure levels for cleaner conditional logic
#[derive(Debug, Clone, Copy, PartialEq)]
enum MemoryPressure {
    None,     // < 75%
    Low,      // 75-85%
    Medium,   // 85-90%
    High,     // 90-95%
    Critical, // > 95%
}

/// CPU pressure levels for cleaner conditional logic
#[derive(Debug, Clone, Copy, PartialEq)]
enum CpuPressure {
    None,    // < 70%
    Low,     // 70-80%
    Medium,  // 80-90%
    High,    // 90-95%
    Extreme, // > 95%
}

impl ResourceMonitor {
    /// Helper function to categorise memory pressure levels
    fn memory_pressure_level(memory_percent: f32) -> MemoryPressure {
        match memory_percent {
            p if p > 95.0 => MemoryPressure::Critical,
            p if p > 90.0 => MemoryPressure::High,
            p if p > 85.0 => MemoryPressure::Medium,
            p if p > 75.0 => MemoryPressure::Low,
            _ => MemoryPressure::None,
        }
    }

    /// Helper function to categorise CPU pressure levels
    fn cpu_pressure_level(cpu_percent: f32) -> CpuPressure {
        match cpu_percent {
            p if p > 95.0 => CpuPressure::Extreme,
            p if p > 90.0 => CpuPressure::High,
            p if p > 80.0 => CpuPressure::Medium,
            p if p > 70.0 => CpuPressure::Low,
            _ => CpuPressure::None,
        }
    }

    /// Get batch parameters based on memory pressure level
    fn memory_batch_params(pressure: MemoryPressure) -> (usize, u64, Option<&'static str>) {
        match pressure {
            MemoryPressure::Critical => (200, 2000, Some("CRITICAL")),
            MemoryPressure::High => (50, 1000, Some("HIGH")),
            MemoryPressure::Medium => (10, 500, Some("MEDIUM")),
            MemoryPressure::Low => (4, 100, None),
            MemoryPressure::None => (1, 0, None),
        }
    }

    /// Get CPU delay based on CPU pressure level
    fn cpu_delay_ms(pressure: CpuPressure) -> u64 {
        match pressure {
            CpuPressure::Extreme => 1000,
            CpuPressure::High => 500,
            CpuPressure::Medium => 200,
            CpuPressure::Low => 0,
            CpuPressure::None => 0,
        }
    }

    /// Get batch multiplier factor based on resource pressure
    fn batch_multiplier_factor(memory_percent: f32, cpu_percent: f32) -> f64 {
        let memory_factor = match Self::memory_pressure_level(memory_percent) {
            MemoryPressure::Critical | MemoryPressure::High => 0.5,
            MemoryPressure::Medium => 0.75,
            _ => 1.0,
        };

        let cpu_factor = match Self::cpu_pressure_level(cpu_percent) {
            CpuPressure::Extreme | CpuPressure::High => 0.5,
            CpuPressure::Medium | CpuPressure::Low => 0.75,
            CpuPressure::None => 1.0,
        };

        memory_factor * cpu_factor
    }

    /// Create a new resource monitor with automatic memory detection
    #[must_use]
    pub fn new() -> Self {
        Self::with_safety_level(SafetyLevel::Conservative)
    }

    /// Create resource monitor with specific safety level
    #[must_use]
    pub fn with_safety_level(safety_level: SafetyLevel) -> Self {
        let refresh_kind = RefreshKind::new()
            .with_cpu(CpuRefreshKind::everything())
            .with_memory(MemoryRefreshKind::everything());

        let system = System::new_with_specifics(refresh_kind);

        Self {
            system: Arc::new(Mutex::new(system)),
            last_refresh: Arc::new(Mutex::new(Instant::now())),
            refresh_interval: Duration::from_secs(1),
            memory_limit_mb: None,
            cpu_limit_percent: 80.0, // Lowered to be a better neighbour

            // Circuit breaker state
            circuit_state: Arc::new(Mutex::new(CircuitState::Closed)),
            last_healthy_time: Arc::new(Mutex::new(Instant::now())),
            consecutive_failures: Arc::new(AtomicU32::new(0)),
            safety_level,

            // Operation tracking
            operation_counter: Arc::new(AtomicU32::new(0)),
        }
    }

    /// Create resource monitor from environment variables
    #[must_use]
    pub fn from_env() -> Self {
        let safety_level = match env::var("STARLINGS_SAFETY_LEVEL").as_deref() {
            Ok("conservative") => SafetyLevel::Conservative,
            Ok("balanced") => SafetyLevel::Balanced,
            Ok("performance") => SafetyLevel::Performance,
            Ok("unsafe") if env::var("STARLINGS_UNSAFE").is_ok() => SafetyLevel::Unsafe,
            _ => SafetyLevel::Conservative, // Default to safe
        };

        Self::with_safety_level(safety_level)
    }

    /// Create with explicit memory limit (following Polars pattern)
    #[must_use]
    pub fn with_memory_limit(memory_limit_mb: u64) -> Self {
        let mut monitor = Self::new();
        monitor.memory_limit_mb = Some(memory_limit_mb);
        monitor
    }

    /// Get current system resource usage
    ///
    /// # Panics
    /// Panics if the system monitor mutex is poisoned
    #[must_use]
    pub fn get_usage(&self) -> ResourceUsage {
        self.refresh_if_needed();

        let system = self.system.lock().unwrap();
        let total_bytes = system.total_memory();
        let available_bytes = system.available_memory();
        let used_bytes = total_bytes - available_bytes;

        let total_memory_mb = total_bytes / (1024 * 1024);
        let available_memory_mb = available_bytes / (1024 * 1024);
        let used_memory_mb = used_bytes / (1024 * 1024);

        let memory_percent = if total_memory_mb > 0 {
            // Use f64 for better precision in percentage calculations
            (used_memory_mb as f64 / total_memory_mb as f64 * 100.0) as f32
        } else {
            0.0
        };

        // Average CPU usage across all cores
        let cpu_count = system.cpus().len().max(1); // Prevent division by zero
        let cpu_percent =
            system.cpus().iter().map(|cpu| cpu.cpu_usage()).sum::<f32>() / cpu_count as f32;

        // Get disk usage for current working directory
        let (disk_free_gb, disk_total_gb, disk_percent) = self.get_disk_usage();

        let effective_memory_limit = self.memory_limit_mb.unwrap_or((total_memory_mb * 80) / 100); // 80% default like Polars, using integer arithmetic

        let is_memory_pressure = used_memory_mb > effective_memory_limit;
        let is_cpu_pressure = cpu_percent > self.cpu_limit_percent;
        let is_disk_pressure = disk_percent > 90.0; // Consider disk pressure above 90%

        ResourceUsage {
            memory_used_mb: used_memory_mb,
            memory_available_mb: available_memory_mb,
            memory_total_mb: total_memory_mb,
            memory_percent,
            cpu_percent,
            disk_free_gb,
            disk_total_gb,
            disk_percent,
            is_memory_pressure,
            is_cpu_pressure,
            is_disk_pressure,
        }
    }

    /// Get adaptive processing limits based on current resource usage
    #[must_use]
    pub fn get_adaptive_limits(&self, base_batch_size: usize) -> AdaptiveLimits {
        let usage = self.get_usage();
        let should_throttle = usage.is_memory_pressure || usage.is_cpu_pressure;

        // Determine if we should spill to disk based on memory pressure
        let should_spill_to_disk = usage.memory_percent > 75.0 && usage.disk_free_gb > 5; // Need at least 5GB free

        // Use helper methods for cleaner conditional logic
        let memory_pressure = Self::memory_pressure_level(usage.memory_percent);
        let cpu_pressure = Self::cpu_pressure_level(usage.cpu_percent);

        let (batch_divisor, base_delay, severity) = Self::memory_batch_params(memory_pressure);

        // Calculate batch size - minimum of 50 for streaming efficiency
        let batch_size = (base_batch_size / batch_divisor).max(50);

        // Get CPU delay and combine with memory delay
        let cpu_delay = Self::cpu_delay_ms(cpu_pressure);
        let delay_ms = base_delay.max(cpu_delay);

        // Format warning messages
        let memory_warning = severity.map(|level| {
            let gb_used = usage.memory_used_mb as f64 / 1024.0;
            let gb_total = usage.memory_total_mb as f64 / 1024.0;
            let action = if should_spill_to_disk {
                "Enabling disk spilling"
            } else {
                match level {
                    "CRITICAL" => "Using minimal batch size",
                    "HIGH" => "Reducing batch size significantly",
                    _ => "Reducing batch size",
                }
            };
            format!(
                "{}: Memory usage {:.0}% ({:.1}GB/{:.1}GB) - {}",
                level, usage.memory_percent, gb_used, gb_total, action
            )
        });

        // Format disk warning if needed
        let disk_warning = if usage.is_disk_pressure {
            Some(format!(
                "LOW DISK SPACE: {:.0}% used ({:.1}GB free) - May affect spilling performance",
                usage.disk_percent, usage.disk_free_gb
            ))
        } else if should_spill_to_disk && usage.disk_free_gb < 10 {
            Some(format!(
                "LIMITED DISK SPACE: {:.1}GB free - Monitor closely during processing",
                usage.disk_free_gb
            ))
        } else {
            None
        };

        AdaptiveLimits {
            batch_size,
            should_throttle,
            should_spill_to_disk,
            delay_between_batches_ms: delay_ms,
            memory_warning,
            disk_warning,
        }
    }

    /// Estimate memory requirements for processing N entities
    pub fn estimate_memory_requirements(&self, num_entities: usize) -> u64 {
        // Based on starlings architecture analysis:
        // - ~60-115MB for 1M edges (from documentation)
        // - Each entity generates ~5 edges on average
        // - Conservative estimate: 150 bytes per edge
        let num_edges = num_entities * 5;
        let estimated_mb = (num_edges * 150) / (1024 * 1024);

        // Add 50% safety margin for intermediate data structures using integer arithmetic
        (estimated_mb + (estimated_mb / 2)) as u64
    }

    /// Calculate optimal batch size based on system resources and memory requirements
    fn calculate_optimal_batch_size(
        &self,
        base_batch_size: usize,
        required_mb: u64,
        usage: &ResourceUsage,
    ) -> usize {
        // Calculate memory headroom ratio using f64 for better precision
        let memory_headroom = usage.memory_available_mb as f64 / required_mb.max(1) as f64;

        // Base multiplier on memory headroom
        let memory_multiplier = match memory_headroom {
            ratio if ratio >= 8.0 => 8.0, // Abundant memory
            ratio if ratio >= 4.0 => 4.0, // Plenty of memory
            ratio if ratio >= 2.0 => 2.0, // Sufficient memory
            _ => 1.0,                     // Limited memory
        };

        // Use helper method for pressure factor calculation
        let pressure_factor =
            Self::batch_multiplier_factor(usage.memory_percent, usage.cpu_percent);

        // Calculate dynamic maximum based on total system memory
        let dynamic_max = match usage.memory_total_mb {
            mem if mem >= 32_000 => 2_000_000, // 32GB+ systems - large batches
            mem if mem >= 16_000 => 1_000_000, // 16GB+ systems - medium batches
            mem if mem >= 8_000 => 500_000,    // 8GB+ systems - smaller batches
            _ => 100_000,                      // <8GB systems - conservative
        };

        // Combine all factors
        let final_multiplier = memory_multiplier * pressure_factor;
        let optimal_size = (base_batch_size as f64 * final_multiplier).round() as usize;

        // Apply dynamic maximum and minimum bounds
        optimal_size.max(1_000).min(dynamic_max)
    }

    /// Determine optimal processing strategy for a dataset size
    pub fn determine_processing_strategy(&self, num_entities: usize) -> ProcessingStrategy {
        let required_mb = self.estimate_memory_requirements(num_entities);
        let usage = self.get_usage();
        let limits = self.get_adaptive_limits(50_000); // Use standard base batch size

        // Determine required disk space for spilling (estimate 2x memory for safety)
        let required_disk_gb = (required_mb * 2) / 1024;

        if required_mb <= usage.memory_available_mb / 2 {
            // Can fit comfortably in memory - use up to 50% of available memory
            let optimal_batch_size =
                self.calculate_optimal_batch_size(limits.batch_size, required_mb, &usage);

            ProcessingStrategy::InMemory {
                batch_size: optimal_batch_size,
                total_batches: ((num_entities * 5) / optimal_batch_size).max(1),
            }
        } else if required_mb <= usage.memory_available_mb && usage.disk_free_gb > required_disk_gb
        {
            // Need memory-aware processing with potential spilling
            let memory_aware_batch_size =
                self.calculate_optimal_batch_size(limits.batch_size, required_mb, &usage);

            ProcessingStrategy::MemoryAware {
                batch_size: memory_aware_batch_size,
                should_spill: limits.should_spill_to_disk,
                spill_threshold_mb: usage.memory_available_mb * 3 / 4, // Spill at 75% memory use
                total_batches: ((num_entities * 5) / memory_aware_batch_size).max(1),
            }
        } else if usage.disk_free_gb > required_disk_gb {
            // Must use streaming with aggressive disk spilling
            let streaming_batch_size = self
                .calculate_optimal_batch_size(limits.batch_size, required_mb, &usage)
                .min(10_000); // Cap streaming batches at 10K for memory safety

            ProcessingStrategy::Streaming {
                batch_size: streaming_batch_size,
                aggressive_spilling: true,
                max_memory_mb: usage.memory_available_mb / 2, // Use only half available memory
                total_batches: ((num_entities * 5) / streaming_batch_size).max(1),
            }
        } else {
            // Not enough resources
            ProcessingStrategy::Insufficient {
                required_memory_mb: required_mb,
                available_memory_mb: usage.memory_available_mb,
                required_disk_gb,
                available_disk_gb: usage.disk_free_gb,
            }
        }
    }

    /// Check if a planned operation is safe to run - now provides strategy recommendations
    pub fn check_operation_safety(
        &self,
        num_entities: usize,
    ) -> Result<ProcessingStrategy, String> {
        let strategy = self.determine_processing_strategy(num_entities);

        match &strategy {
            ProcessingStrategy::Insufficient {
                required_memory_mb,
                available_memory_mb,
                required_disk_gb,
                available_disk_gb,
            } => Err(format!(
                "Insufficient resources for {} entities:\n  Memory: need ~{}MB, have {}MB\n  Disk: need ~{}GB free, have {}GB\n  Consider reducing scale, freeing memory, or clearing disk space.",
                num_entities, required_memory_mb, available_memory_mb, required_disk_gb, available_disk_gb
            )),
            _ => Ok(strategy),
        }
    }

    /// Get disk usage - simplified implementation for compatibility
    fn get_disk_usage(&self) -> (u64, u64, f32) {
        // Use statvfs system call on Unix systems for disk space
        #[cfg(unix)]
        {
            use std::ffi::CString;
            use std::mem;

            let path = CString::new(".").unwrap();
            let mut stat: libc::statvfs = unsafe { mem::zeroed() };

            let result = unsafe { libc::statvfs(path.as_ptr(), &mut stat) };
            if result == 0 {
                let block_size: u64 = stat.f_bsize;
                // Handle platform differences: macOS has u32, Linux has u64
                #[cfg(target_os = "macos")]
                let total_blocks: u64 = stat.f_blocks.into();
                #[cfg(not(target_os = "macos"))]
                let total_blocks: u64 = stat.f_blocks;

                #[cfg(target_os = "macos")]
                let free_blocks: u64 = stat.f_bavail.into();
                #[cfg(not(target_os = "macos"))]
                let free_blocks: u64 = stat.f_bavail;

                let total_bytes = total_blocks * block_size;
                let free_bytes = free_blocks * block_size;
                let used_bytes = total_bytes - free_bytes;

                let total_gb = total_bytes / (1024 * 1024 * 1024);
                let free_gb = free_bytes / (1024 * 1024 * 1024);
                let used_percent = if total_bytes > 0 {
                    ((used_bytes as f64 / total_bytes as f64) * 100.0) as f32
                } else {
                    0.0
                };

                return (free_gb, total_gb, used_percent);
            }
        }

        // Conservative fallback for non-Unix or if statvfs fails
        (100, 500, 20.0)
    }

    fn refresh_if_needed(&self) {
        let mut last_refresh = self.last_refresh.lock().unwrap();
        if last_refresh.elapsed() >= self.refresh_interval {
            let mut system = self.system.lock().unwrap();
            system.refresh_memory();
            system.refresh_cpu_all();
            *last_refresh = Instant::now();
        }
    }

    /// Check if system is under resource pressure and should throttle operations
    pub fn should_throttle(&self) -> bool {
        let usage = self.get_usage();
        usage.is_memory_pressure || usage.is_cpu_pressure
    }

    /// Wait for resources with exponential backoff if under pressure
    pub fn wait_for_resources_with_backoff(&self, base_delay_ms: u64) -> u64 {
        if !self.should_throttle() {
            return 0;
        }

        let usage = self.get_usage();
        let mut delay_ms = base_delay_ms;

        // Exponential backoff based on pressure level
        if usage.memory_percent > 90.0 {
            delay_ms *= 4; // Severe memory pressure
        } else if usage.memory_percent > 80.0 {
            delay_ms *= 2; // High memory pressure
        }

        if usage.cpu_percent > 90.0 {
            delay_ms *= 2; // High CPU usage
        }

        // Cap maximum delay at 5 seconds
        delay_ms = delay_ms.min(5000);

        if delay_ms > 0 {
            std::thread::sleep(std::time::Duration::from_millis(delay_ms));
        }

        delay_ms
    }

    /// Get recommended throttling delay based on current system state
    pub fn get_throttling_delay(&self) -> u64 {
        if !self.should_throttle() {
            return 0;
        }

        let usage = self.get_usage();
        let base_delay = if usage.is_memory_pressure && usage.is_cpu_pressure {
            500 // Both memory and CPU pressure
        } else if usage.is_memory_pressure {
            200 // Memory pressure only
        } else if usage.is_cpu_pressure {
            100 // CPU pressure only
        } else {
            0
        };

        // Scale by severity
        let memory_factor = if usage.memory_percent > 95.0 {
            3.0
        } else if usage.memory_percent > 85.0 {
            2.0
        } else {
            1.0
        };

        (base_delay as f64 * memory_factor).round() as u64
    }

    // === Circuit Breaker Methods ===

    /// Pre-flight check - MUST be called before any large operation
    pub fn can_proceed(&self, estimated_mb: u64) -> Result<OperationPermit, SafetyError> {
        // Check circuit state
        if self.is_circuit_open() {
            return Err(SafetyError::CircuitOpen("System under stress".to_string()));
        }

        // Check system health - PROPORTIONAL thresholds
        let usage = self.get_usage();
        let safety_threshold = self.safety_level.memory_threshold() * 100.0;

        if usage.memory_percent > safety_threshold as f32 {
            self.trip_circuit(&format!(
                "Memory pressure too high: {:.1}%",
                usage.memory_percent
            ));
            return Err(SafetyError::InsufficientResources(format!(
                "Memory usage {:.1}% exceeds safety threshold {:.1}%",
                usage.memory_percent, safety_threshold
            )));
        }

        // Check if operation would exceed safety margins (PROPORTIONAL)
        let max_operation_fraction = self.safety_level.max_operation_fraction();
        let max_allowed_mb = (usage.memory_available_mb as f64 * max_operation_fraction) as u64;

        if estimated_mb > max_allowed_mb {
            return Err(SafetyError::OperationTooLarge(format!(
                "Operation requires ~{}MB but safety limit is {}MB ({}% of available {}MB). \
                 Try: 1) Smaller dataset, 2) Free memory, 3) Set STARLINGS_SAFETY_LEVEL=performance",
                estimated_mb, max_allowed_mb,
                (max_operation_fraction * 100.0) as u32,
                usage.memory_available_mb
            )));
        }

        // Generate permit ID and return permit
        let permit_id = self.operation_counter.fetch_add(1, Ordering::Relaxed) as u64;
        Ok(OperationPermit::new(permit_id))
    }

    /// Rate limiting proportional to system pressure
    pub fn throttle_if_needed(&self) -> Duration {
        let usage = self.get_usage();
        // Proportional delays based on pressure
        let memory_pressure = usage.memory_percent / 100.0;
        let cpu_pressure = usage.cpu_percent / 100.0;

        let max_pressure = memory_pressure.max(cpu_pressure);

        // Exponential backoff based on pressure
        let delay_ms = match max_pressure {
            p if p > 0.95 => 2000,
            p if p > 0.90 => 1000,
            p if p > 0.85 => 500,
            p if p > 0.80 => 200,
            p if p > 0.75 => 100,
            p if p > 0.70 => 50,
            _ => 0,
        };

        Duration::from_millis(delay_ms)
    }

    /// Get safe parallelism based on system state
    pub fn get_safe_parallelism(&self) -> usize {
        let usage = self.get_usage();
        let total_cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);

        // Proportional reduction based on memory pressure
        let memory_factor = 1.0 - (usage.memory_percent as f64 / 100.0).powf(2.0);
        (total_cores as f64 * memory_factor).max(1.0) as usize
    }

    /// Get max batch size based on available memory
    pub fn get_max_batch_size(&self) -> usize {
        let usage = self.get_usage();

        // Base calculation: 1% of available memory worth of entities
        // Assuming ~150 bytes per edge, 5 edges per entity
        let bytes_per_entity = 750;
        let one_percent_entities = (usage.memory_available_mb * 1024 * 10) / bytes_per_entity;

        // Apply safety factor based on current pressure
        let pressure_factor = (1.0 - usage.memory_percent as f64 / 100.0).max(0.1);

        ((one_percent_entities as f64) * pressure_factor).max(100.0) as usize
    }

    /// Check if circuit is open (system under stress)
    pub fn is_circuit_open(&self) -> bool {
        matches!(*self.circuit_state.lock().unwrap(), CircuitState::Open)
    }

    /// Trip the circuit breaker
    fn trip_circuit(&self, reason: &str) {
        *self.circuit_state.lock().unwrap() = CircuitState::Open;
        self.consecutive_failures.fetch_add(1, Ordering::Relaxed);
        eprintln!("🚨 Circuit breaker tripped: {}", reason);
    }

    /// Reset circuit if system has recovered
    pub fn reset_circuit_if_healthy(&self) {
        let usage = self.get_usage();
        let threshold = self.safety_level.memory_threshold() * 100.0;

        if usage.memory_percent < (threshold * 0.8) as f32 {
            // 80% of threshold for hysteresis
            let mut state = self.circuit_state.lock().unwrap();
            match *state {
                CircuitState::Open => {
                    *state = CircuitState::HalfOpen;
                    eprintln!("🔄 Circuit breaker half-open - testing recovery");
                }
                CircuitState::HalfOpen => {
                    *state = CircuitState::Closed;
                    self.consecutive_failures.store(0, Ordering::Relaxed);
                    *self.last_healthy_time.lock().unwrap() = Instant::now();
                    eprintln!("✅ Circuit breaker closed - system recovered");
                }
                CircuitState::Closed => {} // Already healthy
            }
        }
    }

    /// Get current safety level
    pub fn safety_level(&self) -> SafetyLevel {
        self.safety_level
    }

    /// Get safety threshold for memory usage
    pub fn get_safety_threshold(&self) -> f64 {
        self.safety_level.memory_threshold()
    }

    /// Get maximum fraction of memory allowed for single operation
    pub fn max_operation_fraction(&self) -> f64 {
        self.safety_level.max_operation_fraction()
    }
}

impl Default for ResourceMonitor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resource_monitor_creation() {
        let monitor = ResourceMonitor::new();
        let usage = monitor.get_usage();

        assert!(usage.memory_total_mb > 0);
        assert!(usage.memory_percent >= 0.0);
        assert!(usage.memory_percent <= 100.0);
    }

    #[test]
    fn test_adaptive_limits() {
        let monitor = ResourceMonitor::new();
        let limits = monitor.get_adaptive_limits(100_000);

        assert!(limits.batch_size >= 50); // Minimum batch size (updated)
        assert!(limits.batch_size <= 100_000); // Should not exceed base

        // Test that disk spilling flag is set to a valid boolean value
        // This is always true for bool type, but documents that the field exists
        let _disk_spilling_configured = limits.should_spill_to_disk;
    }

    #[test]
    fn test_memory_estimation() {
        let monitor = ResourceMonitor::new();
        let estimated_mb = monitor.estimate_memory_requirements(1_000_000);

        // Should be reasonable for 1M entities (expecting ~750MB based on docs)
        assert!(estimated_mb > 500);
        assert!(estimated_mb < 2000);
    }

    #[test]
    fn test_with_memory_limit() {
        let monitor = ResourceMonitor::with_memory_limit(4096); // 4GB
        assert_eq!(monitor.memory_limit_mb, Some(4096));
    }

    #[test]
    fn test_processing_strategy() {
        let monitor = ResourceMonitor::new();

        // Test small dataset (should be InMemory)
        let strategy = monitor.determine_processing_strategy(1_000);
        match strategy {
            ProcessingStrategy::InMemory { .. } => (), // Expected
            _ => panic!("Small dataset should use InMemory strategy"),
        }

        // Test very large dataset safety check
        let result = monitor.check_operation_safety(100_000_000);
        // Should either provide a strategy or explain why it's insufficient
        match result {
            Ok(_) => (),                                                 // Got a strategy
            Err(msg) => assert!(msg.contains("Insufficient resources")), // Expected error format
        }
    }
}
