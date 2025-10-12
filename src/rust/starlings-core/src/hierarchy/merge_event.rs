use roaring::RoaringBitmap;

/// Represents a binary merge event at a specific threshold
///
/// A merge event captures a single binary merge where one partition (child)
/// is absorbed into another partition (parent). This binary delta format
/// stores only the absorbed partition's nodes, not the full merged state.
///
/// # Memory Efficiency
///
/// By storing only the delta (child nodes), this achieves O(N) total memory
/// usage across all events, compared to O(N²) when storing full state.
#[derive(Debug, Clone, PartialEq)]
pub struct MergeEvent {
    /// The threshold at which this merge occurs
    pub threshold: f64,

    /// Canonical ID of the partition that survives (parent)
    pub parent_id: u32,

    /// Complete bitmap of the partition being absorbed (child)
    pub child_nodes: RoaringBitmap,
}

impl MergeEvent {
    /// Create a new binary delta merge event
    ///
    /// # Arguments
    /// * `threshold` - Similarity threshold at which merge occurs
    /// * `parent_id` - Canonical ID of surviving partition
    /// * `child_nodes` - Complete bitmap of absorbed partition
    ///
    /// # Panics
    /// Panics if child_nodes is empty (invalid merge)
    pub fn new(threshold: f64, parent_id: u32, child_nodes: RoaringBitmap) -> Self {
        assert!(
            !child_nodes.is_empty(),
            "child_nodes cannot be empty in merge event"
        );
        Self {
            threshold,
            parent_id,
            child_nodes,
        }
    }

    /// Get canonical ID of child partition (minimum record in child_nodes)
    pub fn child_id(&self) -> u32 {
        self.child_nodes
            .min()
            .expect("child_nodes validated in constructor")
    }

    /// Get approximate number of entities involved
    ///
    /// Returns child size + 1 (for parent). This is approximate because
    /// we don't store the parent's full size.
    pub fn total_entities(&self) -> u64 {
        self.child_nodes.len() + 1
    }

    /// Check if this merge affects a specific record
    ///
    /// Only checks child nodes. Records in parent partition are not
    /// stored in the event (delta representation).
    pub fn affects_record(&self, record_id: u32) -> bool {
        self.child_nodes.contains(record_id)
    }

    /// Get the threshold of this merge event
    pub fn threshold(&self) -> f64 {
        self.threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_merge_event_creation() {
        let mut child_bitmap = RoaringBitmap::new();
        child_bitmap.insert(10);
        child_bitmap.insert(11);

        let event = MergeEvent::new(0.8, 5, child_bitmap);

        assert_eq!(event.threshold, 0.8);
        assert_eq!(event.parent_id, 5);
        assert_eq!(event.child_id(), 10); // min of {10, 11}
        assert_eq!(event.total_entities(), 3); // child.len() + 1
    }

    #[test]
    fn test_affects_record() {
        let mut child_bitmap = RoaringBitmap::new();
        child_bitmap.insert(1);
        child_bitmap.insert(2);

        let merge = MergeEvent::new(0.7, 0, child_bitmap);

        assert!(merge.affects_record(1)); // In child
        assert!(merge.affects_record(2)); // In child
        assert!(!merge.affects_record(0)); // Parent not in child_nodes
        assert!(!merge.affects_record(3)); // Not involved
    }

    #[test]
    #[should_panic(expected = "child_nodes cannot be empty")]
    fn test_empty_child_nodes_panics() {
        let empty_bitmap = RoaringBitmap::new();
        MergeEvent::new(0.5, 0, empty_bitmap); // Should panic
    }
}
