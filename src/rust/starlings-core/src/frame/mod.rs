pub mod translation;

use crate::core::DataContext;
use crate::hierarchy::PartitionHierarchy;
use roaring::RoaringBitmap;
use std::collections::HashMap;
use std::sync::Arc;

/// Multi-collection container that enables hierarchies to share DataContext
pub struct EntityFrame {
    /// Shared data context for all collections
    context: Arc<DataContext>,

    /// Named collections (hierarchies) in this frame
    collections: HashMap<String, PartitionHierarchy>,

    /// Track garbage records for future compaction support
    #[allow(dead_code)]
    garbage_records: RoaringBitmap,
}

impl EntityFrame {
    /// Create a new empty EntityFrame with fresh DataContext
    pub fn new() -> Self {
        EntityFrame {
            context: Arc::new(DataContext::new()),
            collections: HashMap::new(),
            garbage_records: RoaringBitmap::new(),
        }
    }

    /// Create EntityFrame with pre-allocated capacity
    pub fn with_capacity(estimated_records: usize) -> Self {
        EntityFrame {
            context: Arc::new(DataContext::with_capacity(estimated_records)),
            collections: HashMap::new(),
            garbage_records: RoaringBitmap::new(),
        }
    }

    /// Add a collection to the frame
    ///
    /// If the collection shares the same context (via Arc::ptr_eq), it's added directly.
    /// If it has a different context, it's assimilated (records merged into frame's context).
    pub fn add_collection(
        &mut self,
        name: String,
        hierarchy: PartitionHierarchy,
    ) -> Result<(), String> {
        // Check if this hierarchy shares our context
        if Arc::ptr_eq(&hierarchy.context, &self.context) {
            // Same context, just add directly
            self.collections.insert(name, hierarchy);
        } else {
            // Different context, need to assimilate
            let assimilated = translation::assimilate_hierarchy(hierarchy, &self.context)?;
            self.collections.insert(name, assimilated.hierarchy);
        }
        Ok(())
    }

    /// Get a collection by name
    pub fn get_collection(&self, name: &str) -> Option<&PartitionHierarchy> {
        self.collections.get(name)
    }

    /// Get mutable reference to a collection
    pub fn get_collection_mut(&mut self, name: &str) -> Option<&mut PartitionHierarchy> {
        self.collections.get_mut(name)
    }

    /// List all collection names
    pub fn collection_names(&self) -> Vec<String> {
        self.collections.keys().cloned().collect()
    }

    /// Get number of collections
    pub fn len(&self) -> usize {
        self.collections.len()
    }

    /// Check if frame is empty
    pub fn is_empty(&self) -> bool {
        self.collections.is_empty()
    }

    /// Get reference to the shared context
    pub fn context(&self) -> &Arc<DataContext> {
        &self.context
    }

    /// Remove a collection from the frame
    pub fn remove_collection(&mut self, name: &str) -> Option<PartitionHierarchy> {
        // In future, track removed records for garbage collection
        self.collections.remove(name)
    }
}

impl Default for EntityFrame {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Key;

    #[test]
    fn test_entity_frame_creation() {
        let frame = EntityFrame::new();
        assert_eq!(frame.len(), 0);
        assert!(frame.is_empty());
    }

    #[test]
    fn test_add_collection_same_context() {
        let mut frame = EntityFrame::new();
        let context = frame.context().clone();

        // Add records to context first
        context.ensure_record("source", Key::U32(1));
        context.ensure_record("source", Key::U32(2));
        context.ensure_record("source", Key::U32(3));

        // Create hierarchy with same context
        let edges = vec![(0, 1, 0.9), (1, 2, 0.8)];
        let hierarchy = PartitionHierarchy::from_edges(edges, context.clone(), 6, None).unwrap();

        frame.add_collection("test".to_string(), hierarchy).unwrap();
        assert_eq!(frame.len(), 1);
        assert!(frame.get_collection("test").is_some());
    }

    #[test]
    fn test_collection_names() {
        let mut frame = EntityFrame::new();
        let context = frame.context().clone();

        // Add records to context first
        context.ensure_record("source", Key::U32(1));
        context.ensure_record("source", Key::U32(2));

        for i in 0..3 {
            let edges = vec![(0, 1, 0.9)];
            let hierarchy =
                PartitionHierarchy::from_edges(edges, context.clone(), 6, None).unwrap();
            frame
                .add_collection(format!("collection_{}", i), hierarchy)
                .unwrap();
        }

        let names = frame.collection_names();
        assert_eq!(names.len(), 3);
        assert!(names.contains(&"collection_0".to_string()));
        assert!(names.contains(&"collection_1".to_string()));
        assert!(names.contains(&"collection_2".to_string()));
    }

    #[test]
    fn test_memory_sharing_verification() {
        let mut frame = EntityFrame::new();
        let context = frame.context().clone();

        // Add records to context first
        context.ensure_record("source", Key::U32(1)); // 0
        context.ensure_record("source", Key::U32(2)); // 1
        context.ensure_record("source", Key::U32(3)); // 2
        context.ensure_record("source", Key::U32(4)); // 3

        // Add two collections with same context
        let hierarchy1 =
            PartitionHierarchy::from_edges(vec![(0, 1, 0.9)], context.clone(), 6, None).unwrap();
        let hierarchy2 =
            PartitionHierarchy::from_edges(vec![(2, 3, 0.8)], context.clone(), 6, None).unwrap();

        frame
            .add_collection("col1".to_string(), hierarchy1)
            .unwrap();
        frame
            .add_collection("col2".to_string(), hierarchy2)
            .unwrap();

        // Verify they share the same context via Arc::ptr_eq
        let col1 = frame.get_collection("col1").unwrap();
        let col2 = frame.get_collection("col2").unwrap();
        assert!(Arc::ptr_eq(&col1.context, &col2.context));
    }
}
