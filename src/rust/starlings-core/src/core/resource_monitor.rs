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
    pub is_memory_pressure: bool,
    pub is_cpu_pressure: bool,
}

#[derive(Debug, Clone)]
pub struct AdaptiveLimits {
    pub batch_size: usize,
    pub should_throttle: bool,
    pub delay_between_batches_ms: u64,
    pub memory_warning: Option<String>,
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

        let effective_memory_limit = self
            .memory_limit_mb
            .unwrap_or((total_memory_mb as f32 * 0.8) as u64); // 80% default like Polars

        let is_memory_pressure = used_memory_mb > effective_memory_limit;
        let is_cpu_pressure = cpu_percent > self.cpu_limit_percent;

        ResourceUsage {
            memory_used_mb: used_memory_mb,
            memory_available_mb: available_memory_mb,
            memory_total_mb: total_memory_mb,
            memory_percent,
            cpu_percent,
            is_memory_pressure,
            is_cpu_pressure,
        }
    }

    /// Get adaptive processing limits based on current resource usage
    pub fn get_adaptive_limits(&self, base_batch_size: usize) -> AdaptiveLimits {
        let usage = self.get_usage();
        let should_throttle = usage.is_memory_pressure || usage.is_cpu_pressure;

        // Determine memory-based adjustments
        let (batch_divisor, base_delay, severity) = match usage.memory_percent {
            p if p > 95.0 => (100, 100, Some("CRITICAL")),
            p if p > 90.0 => (20, 50, Some("HIGH")),
            p if p > 85.0 => (5, 20, Some("MEDIUM")),
            p if p > 80.0 => (2, 10, None),
            _ => (1, 0, None),
        };

        // Calculate batch size
        let batch_size = (base_batch_size / batch_divisor).max(100);

        // Calculate delay with CPU throttling
        let cpu_delay = match usage.cpu_percent {
            p if p > 95.0 => 200,
            p if p > 90.0 => 100,
            _ => 0,
        };
        let delay_ms = base_delay.max(cpu_delay);

        // Format warning message if needed
        let warning = severity.map(|level| {
            let gb_used = usage.memory_used_mb as f32 / 1024.0;
            let gb_total = usage.memory_total_mb as f32 / 1024.0;
            let message = match level {
                "CRITICAL" => "Using minimal batch size",
                "HIGH" => "Reducing batch size significantly",
                _ => "Reducing batch size",
            };
            format!(
                "{}: Memory usage {}% ({:.1}GB/{:.1}GB) - {}",
                level, usage.memory_percent as u32, gb_used, gb_total, message
            )
        });

        AdaptiveLimits {
            batch_size,
            should_throttle,
            delay_between_batches_ms: delay_ms,
            memory_warning: warning,
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

    /// Check if a planned operation is safe to run
    pub fn check_operation_safety(&self, num_entities: usize) -> Result<(), String> {
        let required_mb = self.estimate_memory_requirements(num_entities);
        let usage = self.get_usage();

        if required_mb > usage.memory_available_mb {
            return Err(format!(
                "Operation requires ~{}MB but only {}MB available. Consider reducing scale or freeing memory.",
                required_mb, usage.memory_available_mb
            ));
        }

        if required_mb > usage.memory_available_mb / 2 {
            return Err(format!(
                "WARNING: Operation requires ~{}MB, which is >50% of available memory ({}MB). This may cause system instability.",
                required_mb, usage.memory_available_mb
            ));
        }

        Ok(())
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

        assert!(limits.batch_size >= 100); // Minimum batch size
        assert!(limits.batch_size <= 100_000); // Should not exceed base
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
}
