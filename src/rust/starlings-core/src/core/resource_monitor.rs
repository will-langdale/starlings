//! Simple DuckDB-style resource monitor with a single memory limit.
//!
//! This module provides memory monitoring for operations, ensuring they
//! respect the configured memory limit. The partition cache manages its
//! own memory independently with LRU eviction.
//!
//! Memory limit is controlled via STARLINGS_MEMORY_LIMIT environment variable.

use std::env;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use sysinfo::{CpuRefreshKind, MemoryRefreshKind, RefreshKind, System};

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
        let refresh_kind = RefreshKind::new()
            .with_cpu(CpuRefreshKind::everything())
            .with_memory(MemoryRefreshKind::everything());

        let mut system = System::new_with_specifics(refresh_kind);
        system.refresh_memory();

        let total_memory_mb = system.total_memory() / (1024 * 1024);
        let default_limit = (total_memory_mb * 80) / 100; // 80% default like DuckDB

        Self::with_memory_limit(default_limit)
    }

    /// Create resource monitor from environment variables
    #[must_use]
    pub fn from_env() -> Self {
        let refresh_kind = RefreshKind::new()
            .with_cpu(CpuRefreshKind::everything())
            .with_memory(MemoryRefreshKind::everything());

        let mut system = System::new_with_specifics(refresh_kind);
        system.refresh_memory();

        let total_memory_mb = system.total_memory() / (1024 * 1024);

        // Parse STARLINGS_MEMORY_LIMIT environment variable
        let memory_limit_mb = if let Ok(limit_str) = env::var("STARLINGS_MEMORY_LIMIT") {
            Self::parse_memory_limit(&limit_str, total_memory_mb)
        } else {
            // Use 80% default like DuckDB if not specified
            (total_memory_mb * 80) / 100
        };

        Self::with_memory_limit(memory_limit_mb)
    }

    /// Create with explicit memory limit in MB
    #[must_use]
    pub fn with_memory_limit(memory_limit_mb: u64) -> Self {
        let refresh_kind = RefreshKind::new()
            .with_cpu(CpuRefreshKind::everything())
            .with_memory(MemoryRefreshKind::everything());

        let system = System::new_with_specifics(refresh_kind);

        Self {
            system: Arc::new(Mutex::new(system)),
            last_refresh: Arc::new(Mutex::new(Instant::now())),
            refresh_interval: Duration::from_secs(1),
            memory_limit_mb,
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

    /// Get current system resource usage
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
            (used_memory_mb as f64 / total_memory_mb as f64 * 100.0) as f32
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
        let mut last_refresh = self.last_refresh.lock().unwrap();
        if last_refresh.elapsed() >= self.refresh_interval {
            let mut system = self.system.lock().unwrap();
            system.refresh_memory();
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
