pub mod core;
pub mod expressions;
pub mod frame;
pub mod hierarchy;
pub mod metrics;
pub mod test_utils;

// Re-export commonly used types for easier access
pub use core::{DataContext, Key, ResourceMonitor};
pub use frame::EntityFrame;
pub use hierarchy::{MergeEvent, PartitionHierarchy, PartitionLevel};

#[cfg(test)]
mod tests {
    use super::core::{DataContext, Key};

    #[test]
    fn test_basic_functionality() {
        let ctx = DataContext::new();

        let hello_id = ctx.intern_string("hello");
        let world_id = ctx.intern_string("world");

        let id1 = ctx.ensure_record("test", Key::InternedString(hello_id));
        let id2 = ctx.ensure_record("test", Key::InternedString(world_id));
        let id3 = ctx.ensure_record("test", Key::InternedString(hello_id));

        assert_eq!(id1, 0);
        assert_eq!(id2, 1);
        assert_eq!(id1, id3);
        assert_eq!(ctx.len(), 2);
    }
}
