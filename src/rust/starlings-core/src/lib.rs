pub mod core;
pub mod hierarchy;
pub mod test_utils;

// Re-export commonly used types for easier access
pub use core::{DataContext, Key, ResourceMonitor};
pub use hierarchy::{MergeEvent, PartitionHierarchy, PartitionLevel};

#[cfg(test)]
mod tests {
    use super::core::{DataContext, Key};

    #[test]
    fn test_basic_functionality() {
        let ctx = DataContext::new();

        let id1 = ctx.ensure_record("test", Key::String("hello".to_string()));
        let id2 = ctx.ensure_record("test", Key::String("world".to_string()));
        let id3 = ctx.ensure_record("test", Key::String("hello".to_string()));

        assert_eq!(id1, 0);
        assert_eq!(id2, 1);
        assert_eq!(id1, id3);
        assert_eq!(ctx.len(), 2);
    }
}
