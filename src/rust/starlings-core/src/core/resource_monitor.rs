//! Simple DuckDB-style resource monitor with a single memory limit.
//!
//! This module provides memory monitoring for operations, ensuring they
//! respect the configured memory limit. The partition cache manages its
//! own memory independently with LRU eviction.
//!
//! Memory limit is controlled via STARLINGS_MEMORY_LIMIT environment variable.
//!
//! ## Performance Design
//!
//! ResourceMonitor is designed for minimal overhead:
//! - <10ms initialization (memory-only refresh)
//! - <1ms per safety check (process memory only)
//! - ~1-2MB memory footprint (no CPU tracking)
//! - Only tracks current process memory (not system-wide CPU or all processes)

use std::env;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use sysinfo::{MemoryRefreshKind, Pid, ProcessRefreshKind, ProcessesToUpdate, RefreshKind, System};

/// Errors that can occur during memory checks
#[derive(Debug, Clone)]
pub enum SafetyError {
    InsufficientMemory {
        required_mb: u64,
        available_mb: u64,
        limit_mb: u64,
    },
}

impl std::fmt::Display for SafetyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SafetyError::InsufficientMemory {
                required_mb,
                available_mb,
                limit_mb,
            } => write!(
                f,
                "Operation requires {}MB but only {}MB available (limit: {}MB). \n                 Consider: 1) Smaller dataset, 2) Free memory, 3) Increase STARLINGS_MEMORY_LIMIT",
                required_mb,
                available_mb,
                limit_mb
            ),
        }
    }
}

impl std::error::Error for SafetyError {}

/// Simple DuckDB-style resource monitor with a single memory limit
#[derive(Debug, Clone)]
pub struct ResourceMonitor {
    system: Arc<Mutex<System>>,
    last_refresh: Arc<Mutex<Instant>>,
    refresh_interval: Duration,
    memory_limit_mb: u64,
    process_pid: Pid,
}

#[derive(Debug, Clone)]
pub struct ResourceUsage {
    pub memory_used_mb: u64,
    pub memory_available_mb: u64,
    pub memory_total_mb: u64,
    pub memory_percent: f32,
    pub memory_limit_mb: u64,
    pub memory_under_limit: bool,
}

impl ResourceMonitor {
    /// Create a new resource monitor with default memory limit (80% of RAM)
    #[must_use]
    pub fn new() -> Self {
        // Minimal refresh: only memory, no CPU (we never use CPU info)
        // This keeps initialization fast (<10ms) instead of slow (seconds)
        let refresh_kind = RefreshKind::new().with_memory(MemoryRefreshKind::everything());

        let mut system = System::new_with_specifics(refresh_kind);
        system.refresh_memory();

        let total_memory_mb = system.total_memory() / (1024 * 1024);
        let default_limit = (total_memory_mb * 80) / 100; // 80% default like DuckDB

        Self::with_memory_limit(default_limit)
    }

    /// Create resource monitor from environment variables
    #[must_use]
    pub fn from_env() -> Self {
        // Minimal refresh: only memory, no CPU (we never use CPU info)
        // This keeps initialization fast (<10ms) instead of slow (seconds)
        let refresh_kind = RefreshKind::new().with_memory(MemoryRefreshKind::everything());

        let mut system = System::new_with_specifics(refresh_kind);
        system.refresh_memory();

        let total_memory_mb = system.total_memory() / (1024 * 1024);

        // Parse STARLINGS_MEMORY_LIMIT environment variable
        let memory_limit_mb = if let Ok(limit_str) = env::var("STARLINGS_MEMORY_LIMIT") {
            crate::debug_println!("📊 STARLINGS_MEMORY_LIMIT env var: '{}'", limit_str);
            Self::parse_memory_limit(&limit_str, total_memory_mb)
        } else {
            let default_mb = (total_memory_mb * 80) / 100;
            crate::debug_println!(
                "📊 STARLINGS_MEMORY_LIMIT not set, using default: {}MB (80% of {}MB)",
                default_mb,
                total_memory_mb
            );
            default_mb
        };

        crate::debug_println!(
            "📊 Memory config: total={}MB, limit={}MB ({:.0}%)",
            total_memory_mb,
            memory_limit_mb,
            (memory_limit_mb as f64 / total_memory_mb as f64) * 100.0
        );

        Self::with_memory_limit(memory_limit_mb)
    }

    /// Create with explicit memory limit in MB
    #[must_use]
    pub fn with_memory_limit(memory_limit_mb: u64) -> Self {
        // Minimal refresh: only memory, no CPU (we never use CPU info)
        // This keeps initialization fast (<10ms) instead of slow (seconds)
        let refresh_kind = RefreshKind::new().with_memory(MemoryRefreshKind::everything());

        let mut system = System::new_with_specifics(refresh_kind);

        // Get current process PID for process-specific memory tracking
        let process_pid = sysinfo::get_current_pid().expect("Failed to get current process PID");

        // CRITICAL: Load process info immediately so first get_usage() call works
        // Without this, refresh_if_needed() won't refresh (elapsed=0) and process lookup fails
        // Lightweight refresh: only memory for our process (~1ms overhead)
        system.refresh_processes_specifics(
            ProcessesToUpdate::Some(&[process_pid]),
            false,
            ProcessRefreshKind::new().with_memory(),
        );

        Self {
            system: Arc::new(Mutex::new(system)),
            last_refresh: Arc::new(Mutex::new(Instant::now())),
            refresh_interval: Duration::from_secs(1),
            memory_limit_mb,
            process_pid,
        }
    }

    /// Parse memory limit string (e.g., "10GB", "50%", "1024")
    fn parse_memory_limit(limit_str: &str, total_memory_mb: u64) -> u64 {
        let limit_str = limit_str.trim();

        if let Some(percent_str) = limit_str.strip_suffix('%') {
            // Percentage of total RAM
            if let Ok(percent) = percent_str.parse::<u64>() {
                let percent = percent.min(100); // Cap at 100%
                return (total_memory_mb * percent) / 100;
            }
        } else if let Some(gb_str) = limit_str
            .strip_suffix("GB")
            .or_else(|| limit_str.strip_suffix("gb"))
        {
            // Gigabytes
            if let Ok(gb) = gb_str.trim().parse::<f64>() {
                return (gb * 1024.0) as u64;
            }
        } else if let Some(mb_str) = limit_str
            .strip_suffix("MB")
            .or_else(|| limit_str.strip_suffix("mb"))
        {
            // Megabytes
            if let Ok(mb) = mb_str.trim().parse::<u64>() {
                return mb;
            }
        } else if let Ok(mb) = limit_str.parse::<u64>() {
            // Plain number assumed to be MB
            return mb;
        }

        // Fallback to 80% if parsing fails
        (total_memory_mb * 80) / 100
    }

    /// Get current process resource usage
    #[must_use]
    pub fn get_usage(&self) -> ResourceUsage {
        self.refresh_if_needed();

        // Safe unwrap: We control this mutex and never panic whilst holding it
        let system = self.system.lock().unwrap();
        let total_bytes = system.total_memory();
        let total_memory_mb = total_bytes / (1024 * 1024);

        // Use PROCESS memory instead of SYSTEM memory
        // This prevents false positives when the OS is using lots of cache/buffers
        let used_memory_mb = if let Some(process) = system.process(self.process_pid) {
            let process_memory_bytes = process.memory();
            let process_mb = process_memory_bytes / (1024 * 1024);

            crate::debug_println!(
                "📊 Process memory: {}MB (limit: {}MB, available: {}MB)",
                process_mb,
                self.memory_limit_mb,
                self.memory_limit_mb.saturating_sub(process_mb)
            );

            process_mb
        } else {
            // Fallback: if we can't get process info, use system memory
            crate::debug_println!("⚠️  Could not get process memory, using system memory");
            let available_bytes = system.available_memory();
            let used_bytes = total_bytes - available_bytes;
            used_bytes / (1024 * 1024)
        };

        let available_memory_mb = self.memory_limit_mb.saturating_sub(used_memory_mb);

        let memory_percent = if self.memory_limit_mb > 0 {
            (used_memory_mb as f64 / self.memory_limit_mb as f64 * 100.0) as f32
        } else {
            0.0
        };

        let memory_under_limit = used_memory_mb < self.memory_limit_mb;

        ResourceUsage {
            memory_used_mb: used_memory_mb,
            memory_available_mb: available_memory_mb,
            memory_total_mb: total_memory_mb,
            memory_percent,
            memory_limit_mb: self.memory_limit_mb,
            memory_under_limit,
        }
    }

    /// Simple memory check - will operation fit within limit?
    pub fn can_proceed(&self, estimated_mb: u64) -> Result<(), SafetyError> {
        let usage = self.get_usage();

        // Cache manages its own memory via LRU eviction, no reservation needed
        let effective_limit = self.memory_limit_mb;

        // Check if operation would exceed limit
        let projected_usage = usage.memory_used_mb + estimated_mb;

        crate::debug_println!(
            "🔍 Memory check: need={}MB, used={}MB, limit={}MB, projected={}MB",
            estimated_mb,
            usage.memory_used_mb,
            self.memory_limit_mb,
            projected_usage
        );

        if projected_usage > effective_limit {
            let available = effective_limit.saturating_sub(usage.memory_used_mb);
            return Err(SafetyError::InsufficientMemory {
                required_mb: estimated_mb,
                available_mb: available,
                limit_mb: effective_limit,
            });
        }

        Ok(())
    }

    /// Get the memory limit
    pub fn get_memory_limit_mb(&self) -> u64 {
        self.memory_limit_mb
    }

    /// Get safety threshold as a fraction (for compatibility)
    pub fn get_safety_threshold(&self) -> f64 {
        // Return limit as percentage of total RAM
        let usage = self.get_usage();
        self.memory_limit_mb as f64 / usage.memory_total_mb as f64
    }

    fn refresh_if_needed(&self) {
        // Safe unwrap: We control this mutex and never panic whilst holding it
        let mut last_refresh = self.last_refresh.lock().unwrap();
        if last_refresh.elapsed() >= self.refresh_interval {
            // Safe unwrap: We control this mutex and never panic whilst holding it
            let mut system = self.system.lock().unwrap();
            system.refresh_memory();
            // Lightweight refresh: only memory for our process (~1ms overhead)
            // NO CPU refresh - we never use CPU info, and it would load all processes
            system.refresh_processes_specifics(
                ProcessesToUpdate::Some(&[self.process_pid]),
                false,
                ProcessRefreshKind::new().with_memory(),
            );
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
        assert_eq!(usage.memory_limit_mb, (usage.memory_total_mb * 80) / 100);
    }

    #[test]
    fn test_with_memory_limit() {
        let monitor = ResourceMonitor::with_memory_limit(4096); // 4GB
        assert_eq!(monitor.get_memory_limit_mb(), 4096);
    }

    #[test]
    fn test_parse_memory_limit() {
        let total_mb = 16384; // 16GB

        // Test percentage
        assert_eq!(ResourceMonitor::parse_memory_limit("50%", total_mb), 8192);
        assert_eq!(ResourceMonitor::parse_memory_limit("80%", total_mb), 13107); // 80% of 16384

        // Test GB
        assert_eq!(ResourceMonitor::parse_memory_limit("4GB", total_mb), 4096);
        assert_eq!(ResourceMonitor::parse_memory_limit("4gb", total_mb), 4096);
        assert_eq!(ResourceMonitor::parse_memory_limit("1.5GB", total_mb), 1536);

        // Test MB
        assert_eq!(
            ResourceMonitor::parse_memory_limit("2048MB", total_mb),
            2048
        );
        assert_eq!(
            ResourceMonitor::parse_memory_limit("2048mb", total_mb),
            2048
        );

        // Test plain number
        assert_eq!(ResourceMonitor::parse_memory_limit("1024", total_mb), 1024);

        // Test invalid falls back to 80%
        assert_eq!(
            ResourceMonitor::parse_memory_limit("invalid", total_mb),
            13107
        );
    }

    #[test]
    fn test_can_proceed() {
        // Use a large limit to ensure small allocations succeed
        let monitor = ResourceMonitor::with_memory_limit(100_000); // 100GB limit

        // Small allocation should succeed
        assert!(monitor.can_proceed(10).is_ok());

        // Allocation larger than limit should fail
        match monitor.can_proceed(200_000) {
            Err(SafetyError::InsufficientMemory { .. }) => (), // Expected
            _ => panic!("Should reject allocation larger than limit"),
        }
    }

    #[test]
    fn test_memory_over_limit_handling() {
        // Create a small limit to simulate system already over limit
        let monitor = ResourceMonitor::with_memory_limit(1); // 1MB limit (will be exceeded)

        // Small operations should still work if system has available memory
        let result = monitor.can_proceed(1);

        // Should either succeed (if system has memory) or fail gracefully
        match result {
            Ok(()) => (),                                      // Fine if system allows it
            Err(SafetyError::InsufficientMemory { .. }) => (), // Also fine
        }
    }
}
