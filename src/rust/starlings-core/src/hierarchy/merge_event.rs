use roaring::RoaringBitmap;

/// Represents a merge event at a specific threshold
///
/// A merge event captures the moment when multiple groups of entities
/// are merged together at a given threshold. Each group is represented
/// as a RoaringBitmap containing the record indices that belong to that group.
///
/// This supports n-way merges when multiple edges have the same threshold,
/// allowing multiple groups to merge simultaneously.
#[derive(Debug, Clone, PartialEq)]
pub struct MergeEvent {
    /// The threshold at which this merge occurs
    pub threshold: f64,

    /// Groups merging at this threshold (supports n-way merges)
    /// Each RoaringBitmap contains the record indices in that group
    pub merging_groups: Vec<RoaringBitmap>,
}

impl MergeEvent {
    /// Create a new merge event
    pub fn new(threshold: f64, merging_groups: Vec<RoaringBitmap>) -> Self {
        Self {
            threshold,
            merging_groups,
        }
    }

    /// Get the total number of entities involved in this merge
    pub fn total_entities(&self) -> u64 {
        self.merging_groups.iter().map(|group| group.len()).sum()
    }

    /// Check if this merge event affects a specific record
    pub fn affects_record(&self, record_id: u32) -> bool {
        self.merging_groups
            .iter()
            .any(|group| group.contains(record_id))
    }

    /// Get the threshold of this merge event
    pub fn threshold(&self) -> f64 {
        self.threshold
    }

    /// Get the merging groups
    pub fn merging_groups(&self) -> &[RoaringBitmap] {
        &self.merging_groups
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_merge_event_creation() {
        let mut group1 = RoaringBitmap::new();
        group1.insert(1);
        group1.insert(2);

        let mut group2 = RoaringBitmap::new();
        group2.insert(3);
        group2.insert(4);

        let merge = MergeEvent::new(0.8, vec![group1, group2]);

        assert_eq!(merge.threshold, 0.8);
        assert_eq!(merge.merging_groups.len(), 2);
        assert_eq!(merge.total_entities(), 4);
    }

    #[test]
    fn test_affects_record() {
        let mut group1 = RoaringBitmap::new();
        group1.insert(1);
        group1.insert(2);

        let merge = MergeEvent::new(0.7, vec![group1]);

        assert!(merge.affects_record(1));
        assert!(merge.affects_record(2));
        assert!(!merge.affects_record(3));
    }
}
