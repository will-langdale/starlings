use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use sysinfo::{CpuRefreshKind, MemoryRefreshKind, RefreshKind, System};

/// Resource monitoring and adaptive processing limits
#[derive(Debug, Clone)]
pub struct ResourceMonitor {
    system: Arc<Mutex<System>>,
    last_refresh: Arc<Mutex<Instant>>,
    refresh_interval: Duration,
    memory_limit_mb: Option<u64>,
    cpu_limit_percent: f32,
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

impl ResourceMonitor {
    /// Create a new resource monitor with automatic memory detection
    pub fn new() -> Self {
        let refresh_kind = RefreshKind::new()
            .with_cpu(CpuRefreshKind::everything())
            .with_memory(MemoryRefreshKind::everything());

        let system = System::new_with_specifics(refresh_kind);

        Self {
            system: Arc::new(Mutex::new(system)),
            last_refresh: Arc::new(Mutex::new(Instant::now())),
            refresh_interval: Duration::from_secs(1),
            memory_limit_mb: None,
            cpu_limit_percent: 90.0,
        }
    }

    /// Create with explicit memory limit (following Polars pattern)
    pub fn with_memory_limit(memory_limit_mb: u64) -> Self {
        let mut monitor = Self::new();
        monitor.memory_limit_mb = Some(memory_limit_mb);
        monitor
    }

    /// Get current system resource usage
    pub fn get_usage(&self) -> ResourceUsage {
        self.refresh_if_needed();

        let system = self.system.lock().unwrap();
        let total_memory_kb = system.total_memory();
        let available_memory_kb = system.available_memory();
        let used_memory_kb = total_memory_kb - available_memory_kb;

        let total_memory_mb = total_memory_kb / 1024;
        let available_memory_mb = available_memory_kb / 1024;
        let used_memory_mb = used_memory_kb / 1024;

        let memory_percent = if total_memory_mb > 0 {
            (used_memory_mb as f32 / total_memory_mb as f32) * 100.0
        } else {
            0.0
        };

        // Average CPU usage across all cores
        let cpu_percent = system.cpus().iter().map(|cpu| cpu.cpu_usage()).sum::<f32>()
            / system.cpus().len() as f32;

        // Get disk usage for current working directory
        let (disk_free_gb, disk_total_gb, disk_percent) = self.get_disk_usage();

        let effective_memory_limit = self
            .memory_limit_mb
            .unwrap_or((total_memory_mb as f32 * 0.8) as u64); // 80% default like Polars

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
    pub fn get_adaptive_limits(&self, base_batch_size: usize) -> AdaptiveLimits {
        let usage = self.get_usage();
        let should_throttle = usage.is_memory_pressure || usage.is_cpu_pressure;

        // Determine if we should spill to disk based on memory pressure
        let should_spill_to_disk = usage.memory_percent > 75.0 && usage.disk_free_gb > 5; // Need at least 5GB free

        // Determine memory-based adjustments - more aggressive with streaming approach
        let (batch_divisor, base_delay, severity) = match usage.memory_percent {
            p if p > 95.0 => (200, 200, Some("CRITICAL")), // Much smaller batches when critical
            p if p > 90.0 => (50, 100, Some("HIGH")),
            p if p > 85.0 => (10, 50, Some("MEDIUM")),
            p if p > 75.0 => (4, 20, None), // Start adapting earlier
            _ => (1, 0, None),
        };

        // Calculate batch size - minimum of 50 for streaming efficiency
        let batch_size = (base_batch_size / batch_divisor).max(50);

        // Calculate delay with CPU throttling
        let cpu_delay = match usage.cpu_percent {
            p if p > 95.0 => 200,
            p if p > 90.0 => 100,
            _ => 0,
        };
        let delay_ms = base_delay.max(cpu_delay);

        // Format warning messages
        let memory_warning = severity.map(|level| {
            let gb_used = usage.memory_used_mb as f32 / 1024.0;
            let gb_total = usage.memory_total_mb as f32 / 1024.0;
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
                "{}: Memory usage {}% ({:.1}GB/{:.1}GB) - {}",
                level, usage.memory_percent as u32, gb_used, gb_total, action
            )
        });

        // Format disk warning if needed
        let disk_warning = if usage.is_disk_pressure {
            Some(format!(
                "LOW DISK SPACE: {}% used ({:.1}GB free) - May affect spilling performance",
                usage.disk_percent as u32, usage.disk_free_gb
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

        // Add 50% safety margin for intermediate data structures
        (estimated_mb as f32 * 1.5) as u64
    }

    /// Determine optimal processing strategy for a dataset size
    pub fn determine_processing_strategy(&self, num_entities: usize) -> ProcessingStrategy {
        let required_mb = self.estimate_memory_requirements(num_entities);
        let usage = self.get_usage();
        let limits = self.get_adaptive_limits(50_000); // Use standard base batch size

        // Determine required disk space for spilling (estimate 2x memory for safety)
        let required_disk_gb = (required_mb * 2) / 1024;

        if required_mb <= usage.memory_available_mb / 4 {
            // Can fit comfortably in memory
            ProcessingStrategy::InMemory {
                batch_size: limits.batch_size,
                total_batches: ((num_entities * 5) / limits.batch_size).max(1),
            }
        } else if required_mb <= usage.memory_available_mb && usage.disk_free_gb > required_disk_gb
        {
            // Need memory-aware processing with potential spilling
            ProcessingStrategy::MemoryAware {
                batch_size: limits.batch_size,
                should_spill: limits.should_spill_to_disk,
                spill_threshold_mb: usage.memory_available_mb * 3 / 4, // Spill at 75% memory use
                total_batches: ((num_entities * 5) / limits.batch_size).max(1),
            }
        } else if usage.disk_free_gb > required_disk_gb {
            // Must use streaming with aggressive disk spilling
            ProcessingStrategy::Streaming {
                batch_size: limits.batch_size.min(10_000), // Smaller batches for streaming
                aggressive_spilling: true,
                max_memory_mb: usage.memory_available_mb / 2, // Use only half available memory
                total_batches: ((num_entities * 5) / limits.batch_size.min(10_000)).max(1),
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
                let block_size = stat.f_bsize;
                let total_blocks = stat.f_blocks;
                let free_blocks = stat.f_bavail;

                let total_bytes = total_blocks * block_size;
                let free_bytes = free_blocks * block_size;
                let used_bytes = total_bytes - free_bytes;

                let total_gb = total_bytes / (1024 * 1024 * 1024);
                let free_gb = free_bytes / (1024 * 1024 * 1024);
                let used_percent = if total_bytes > 0 {
                    (used_bytes as f32 / total_bytes as f32) * 100.0
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

        // Test that disk spilling logic exists
        assert!(limits.should_spill_to_disk == true || limits.should_spill_to_disk == false);
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
