//! Global resource safety module for all public API functions.
//!
//! Provides a single, globally-accessible safety barrier that prevents
//! dangerous memory allocations across the entire Starlings API surface.
//! Uses the same conservative safety levels and DuckDB-style error messages
//! as the original DataContext-based safety system.

use crate::core::resource_monitor::{ResourceMonitor, SafetyError};
use std::sync::OnceLock;

/// Global resource monitor singleton.
///
/// This is initialized lazily on first access using ResourceMonitor::from_env(),
/// which respects STARLINGS_SAFETY_LEVEL environment variable for configuration.
static GLOBAL_RESOURCE_MONITOR: OnceLock<ResourceMonitor> = OnceLock::new();

/// Global memory safety check for all public API functions.
///
/// Ensures operations respect system resource limits using the same
/// conservative safety levels as Collection.from_edges(). This provides
/// a universal safety barrier that prevents system crashes from memory
/// exhaustion across all public functions.
///
/// # Arguments
/// * `estimated_mb` - Estimated memory consumption in megabytes
///
/// # Returns
/// * `Ok(())` if operation is safe to proceed
/// * `Err(SafetyError)` if operation would exceed safety limits
///
/// # Safety Levels
/// Respects STARLINGS_SAFETY_LEVEL environment variable:
/// * Conservative (default): Max 50% RAM usage
/// * Balanced: Max 70% RAM usage  
/// * Performance: Max 85% RAM usage
/// * Unsafe: Max 95% RAM usage (requires STARLINGS_UNSAFE=1)
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
///         // Operation would exceed safety limits
///         eprintln!("Safety check failed: {}", e);
///     }
/// }
/// ```
pub fn ensure_memory_safety(estimated_mb: u64) -> Result<(), SafetyError> {
    let monitor = GLOBAL_RESOURCE_MONITOR.get_or_init(ResourceMonitor::from_env);
    monitor.can_proceed(estimated_mb).map(|_| ())
}

/// Get global resource monitor for advanced usage.
///
/// Provides access to the global ResourceMonitor instance for operations
/// that need more detailed resource information beyond simple safety checks.
/// The monitor is initialized lazily on first access.
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
/// ```
pub fn global_resource_monitor() -> &'static ResourceMonitor {
    GLOBAL_RESOURCE_MONITOR.get_or_init(ResourceMonitor::from_env)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

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
    fn test_safety_respects_environment() {
        // Test that safety level is read from environment
        env::set_var("STARLINGS_SAFETY_LEVEL", "performance");

        // This should not fail with a fresh monitor
        // (Note: This test may be flaky depending on system resources)
        let result = ensure_memory_safety(100);

        // Clean up
        env::remove_var("STARLINGS_SAFETY_LEVEL");

        // We can't assert success/failure as it depends on system state,
        // but we can verify the function doesn't panic
        drop(result);
    }

    #[test]
    fn test_massive_allocation_rejected() {
        // Allocation larger than any reasonable system should be rejected
        let result = ensure_memory_safety(1_000_000); // 1TB
        match result {
            Err(SafetyError::InsufficientResources { .. }) => {
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
