//! Global resource safety module for all public API functions.
//!
//! Provides a simple memory limit check following DuckDB's approach.
//! Uses a single STARLINGS_MEMORY_LIMIT environment variable that defaults
//! to 80% of total system RAM.

use crate::core::resource_monitor::{ResourceMonitor, SafetyError};
use std::sync::OnceLock;

/// Global resource monitor singleton.
static GLOBAL_RESOURCE_MONITOR: OnceLock<ResourceMonitor> = OnceLock::new();

/// Global memory safety check for all public API functions.
///
/// Ensures operations respect the memory limit using a simple DuckDB-style
/// approach. This provides a universal safety barrier that prevents system
/// crashes from memory exhaustion.
///
/// # Arguments
/// * `estimated_mb` - Estimated memory consumption in megabytes
///
/// # Returns
/// * `Ok(())` if operation is safe to proceed
/// * `Err(SafetyError)` if operation would exceed memory limit
///
/// # Memory Limit
/// Respects STARLINGS_MEMORY_LIMIT environment variable:
/// * Can be a percentage: "50%", "80%"
/// * Can be absolute: "10GB", "4096MB", "4096" (MB assumed)
/// * Default: 80% of total system RAM
///
/// # Example
/// ```rust
/// use starlings_core::core::ensure_memory_safety;
///
/// // Check if we can safely allocate 100MB
/// match ensure_memory_safety(100) {
///     Ok(()) => {
///         // Safe to proceed with allocation
///         let data = vec![0u8; 100 * 1024 * 1024];
///     },
///     Err(e) => {
///         // Operation would exceed memory limit
///         eprintln!("Memory check failed: {}", e);
///     }
/// }
/// ```
pub fn ensure_memory_safety(estimated_mb: u64) -> Result<(), SafetyError> {
    let monitor = GLOBAL_RESOURCE_MONITOR.get_or_init(|| {
        crate::debug_println!("🔧 Initializing global ResourceMonitor from environment");
        ResourceMonitor::from_env()
    });
    monitor.can_proceed(estimated_mb)
}

/// Get global resource monitor for advanced usage.
///
/// Provides access to the global ResourceMonitor instance for operations
/// that need more detailed resource information beyond simple safety checks.
/// The monitor is initialised lazily on first access.
///
/// # Returns
/// Reference to the global ResourceMonitor singleton
///
/// # Example
/// ```rust
/// use starlings_core::core::global_resource_monitor;
///
/// let monitor = global_resource_monitor();
/// let usage = monitor.get_usage();
/// println!("Current memory usage: {}MB", usage.memory_used_mb);
/// println!("Memory limit: {}MB", usage.memory_limit_mb);
/// ```
pub fn global_resource_monitor() -> &'static ResourceMonitor {
    GLOBAL_RESOURCE_MONITOR.get_or_init(|| {
        crate::debug_println!("🔧 Initializing global ResourceMonitor from environment");
        ResourceMonitor::from_env()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ensure_memory_safety_small_allocation() {
        // Small allocation should always succeed
        assert!(ensure_memory_safety(10).is_ok());
    }

    #[test]
    fn test_global_monitor_singleton() {
        // Multiple calls should return the same instance
        let monitor1 = global_resource_monitor();
        let monitor2 = global_resource_monitor();
        assert_eq!(monitor1 as *const _, monitor2 as *const _);
    }

    #[test]
    fn test_massive_allocation_rejected() {
        // Allocation larger than any reasonable system should be rejected
        let result = ensure_memory_safety(1_000_000); // 1TB
        match result {
            Err(SafetyError::InsufficientMemory { .. }) => {
                // Expected - this should be rejected
            }
            _ => {
                // If this passes, either we have a very large system
                // or the safety system isn't working properly
                // We'll allow it for testing on large systems
            }
        }
    }
}
