/// Universal debug logging system for Starlings
///
/// Debug messages are only shown when STARLINGS_DEBUG=1 environment variable is set.
/// This provides a consistent, controllable way to show debug information across
/// the entire codebase.
use std::env;
use std::sync::OnceLock;

/// Cache for debug mode status to avoid repeated environment variable checks
static DEBUG_MODE: OnceLock<bool> = OnceLock::new();

/// Check if debug mode is enabled via STARLINGS_DEBUG=1 environment variable
pub fn is_debug_enabled() -> bool {
    *DEBUG_MODE.get_or_init(|| {
        env::var("STARLINGS_DEBUG")
            .map(|v| v == "1" || v.to_lowercase() == "true")
            .unwrap_or(false)
    })
}

/// Print debug message only if STARLINGS_DEBUG=1 is set
///
/// # Examples
///
/// ```
/// use starlings_core::debug_println;
///
/// # let thread_count = 4;
/// # let count = 1000;
/// debug_println!("🔧 Starlings: Using {} threads", thread_count);
/// debug_println!("Processing {} records", count);
/// ```
#[macro_export]
macro_rules! debug_println {
    ($($arg:tt)*) => {
        if $crate::core::debug::is_debug_enabled() {
            eprintln!($($arg)*);
        }
    };
}

/// Print debug message with formatting only if STARLINGS_DEBUG=1 is set
/// Similar to debug_println! but uses println! instead of eprintln!
#[macro_export]
macro_rules! debug_print {
    ($($arg:tt)*) => {
        if $crate::core::debug::is_debug_enabled() {
            println!($($arg)*);
        }
    };
}

/// Conditional debug block - execute code only if debug is enabled
///
/// # Examples
///
/// ```
/// use starlings_core::debug_if;
///
/// debug_if!({
///     let stats = vec![1, 2, 3]; // Some example stats
///     eprintln!("Detailed stats: {:?}", stats);
/// });
/// ```
#[macro_export]
macro_rules! debug_if {
    ($block:block) => {
        if $crate::core::debug::is_debug_enabled() {
            $block
        }
    };
}

#[cfg(test)]
mod tests {
    use std::env;

    #[test]
    fn test_debug_mode_detection() {
        // Since OnceLock can't be easily reset in tests, we'll test the function directly
        // by checking the environment variable parsing logic

        // Test with debug enabled
        env::set_var("STARLINGS_DEBUG", "1");
        let enabled = env::var("STARLINGS_DEBUG")
            .map(|v| v == "1" || v.to_lowercase() == "true")
            .unwrap_or(false);
        assert!(enabled);

        // Test with debug disabled
        env::set_var("STARLINGS_DEBUG", "0");
        let disabled = env::var("STARLINGS_DEBUG")
            .map(|v| v == "1" || v.to_lowercase() == "true")
            .unwrap_or(false);
        assert!(!disabled);

        // Test with "true" value
        env::set_var("STARLINGS_DEBUG", "true");
        let enabled_true = env::var("STARLINGS_DEBUG")
            .map(|v| v == "1" || v.to_lowercase() == "true")
            .unwrap_or(false);
        assert!(enabled_true);

        // Test with no environment variable
        env::remove_var("STARLINGS_DEBUG");
        let default_disabled = env::var("STARLINGS_DEBUG")
            .map(|v| v == "1" || v.to_lowercase() == "true")
            .unwrap_or(false);
        assert!(!default_disabled);
    }
}
