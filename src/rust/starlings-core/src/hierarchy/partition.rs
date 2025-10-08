use once_cell::sync::OnceCell;
use roaring::RoaringBitmap;
use std::fmt;
use std::sync::Arc;

/// Cached reverse index from record to entity
pub struct RecordToEntityIndex {
    /// Maps record index to entity index (None if record not in any entity)
    pub index: Vec<Option<usize>>,
    /// Generation when this index was built (for cache invalidation)
    pub generation: u64,
}

/// Represents a partition at a specific threshold level
///
/// A partition is a collection of entities (disjoint sets of records) at a given threshold.
/// Each entity is represented as a RoaringBitmap containing the indices of records that
/// belong together at this threshold level.
///
/// This structure includes pre-computed statistics like entity sizes for efficient access
/// to common metrics without recomputation.
#[derive(Clone)]
pub struct PartitionLevel {
    /// The threshold value for this partition
    threshold: f64,

    /// Entities as sets of record indices (RoaringBitmaps)
    /// Each bitmap represents one entity containing the record indices that belong to it
    entities: Vec<RoaringBitmap>,

    /// Pre-computed entity sizes for quick access
    /// Corresponds 1:1 with the entities vector
    entity_sizes: Vec<u32>,

    /// Cached reverse index for O(1) record->entity lookup
    /// Built lazily on first access
    record_to_entity_cache: Arc<OnceCell<RecordToEntityIndex>>,
}

impl PartitionLevel {
    /// Create a new partition level
    ///
    /// # Arguments
    /// * `threshold` - The threshold value for this partition
    /// * `entities` - Vector of RoaringBitmaps, each representing an entity
    ///
    /// # Returns
    /// A new PartitionLevel with pre-computed entity sizes
    pub fn new(threshold: f64, entities: Vec<RoaringBitmap>) -> Self {
        let entity_sizes = entities.iter().map(|e| e.len() as u32).collect();
        Self {
            threshold,
            entities,
            entity_sizes,
            record_to_entity_cache: Arc::new(OnceCell::new()),
        }
    }

    /// Build reverse index mapping record indices to entity indices
    fn build_record_to_entity_index(
        &self,
        num_records: usize,
        generation: u64,
    ) -> RecordToEntityIndex {
        let mut index = vec![None; num_records];

        for (entity_idx, entity_bitmap) in self.entities.iter().enumerate() {
            for record_idx in entity_bitmap.iter() {
                // Bounds check for safety after potential compaction
                if (record_idx as usize) < num_records {
                    index[record_idx as usize] = Some(entity_idx);
                }
            }
        }

        RecordToEntityIndex { index, generation }
    }

    /// Get the entity index for a given record (with generation tracking)
    /// Returns None if record not in any entity or if cache is stale
    pub fn get_entity_for_record_with_generation(
        &self,
        record_idx: usize,
        num_records: usize,
        current_generation: u64,
    ) -> Option<usize> {
        let cache = self
            .record_to_entity_cache
            .get_or_init(|| self.build_record_to_entity_index(num_records, current_generation));

        // Check if cache is still valid
        if cache.generation != current_generation {
            // Cache is stale, would need to rebuild
            // In production, we'd clear and rebuild here
            return None;
        }

        cache.index.get(record_idx).copied().flatten()
    }

    /// Get or build the reverse index for this partition
    pub fn get_record_to_entity_index(
        &self,
        num_records: usize,
        generation: u64,
    ) -> &RecordToEntityIndex {
        self.record_to_entity_cache
            .get_or_init(|| self.build_record_to_entity_index(num_records, generation))
    }

    /// Get the threshold for this partition
    pub fn threshold(&self) -> f64 {
        self.threshold
    }

    /// Get the entities in this partition
    pub fn entities(&self) -> &[RoaringBitmap] {
        &self.entities
    }

    /// Get the number of entities in this partition
    pub fn num_entities(&self) -> usize {
        self.entities.len()
    }

    /// Get pre-computed entity sizes
    pub fn entity_sizes(&self) -> &[u32] {
        &self.entity_sizes
    }

    /// Get the total number of records across all entities
    pub fn total_records(&self) -> u32 {
        self.entity_sizes.iter().sum()
    }

    /// Check if a specific record belongs to any entity
    pub fn contains_record(&self, record_id: u32) -> bool {
        self.entities
            .iter()
            .any(|entity| entity.contains(record_id))
    }

    /// Find which entity (if any) contains a specific record
    pub fn find_entity_for_record(&self, record_id: u32) -> Option<usize> {
        self.entities
            .iter()
            .position(|entity| entity.contains(record_id))
    }

    /// Estimate memory usage in bytes
    pub fn memory_usage_bytes(&self) -> u64 {
        let base_size = std::mem::size_of::<PartitionLevel>() as u64;
        let entities_size: u64 = self
            .entities
            .iter()
            .map(|b| b.serialized_size() as u64)
            .sum();
        base_size + entities_size
    }
}

impl fmt::Debug for PartitionLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PartitionLevel")
            .field("threshold", &self.threshold)
            .field("entities", &self.entities)
            .field("entity_sizes", &self.entity_sizes)
            .field(
                "has_cached_index",
                &self.record_to_entity_cache.get().is_some(),
            )
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_partition_level_creation() {
        let mut entity1 = RoaringBitmap::new();
        entity1.insert(0);
        entity1.insert(1);

        let mut entity2 = RoaringBitmap::new();
        entity2.insert(2);
        entity2.insert(3);
        entity2.insert(4);

        let partition = PartitionLevel::new(0.7, vec![entity1, entity2]);

        assert_eq!(partition.threshold(), 0.7);
        assert_eq!(partition.num_entities(), 2);
        assert_eq!(partition.entity_sizes(), &[2, 3]);
        assert_eq!(partition.total_records(), 5);
    }

    #[test]
    fn test_contains_record() {
        let mut entity = RoaringBitmap::new();
        entity.insert(1);
        entity.insert(2);

        let partition = PartitionLevel::new(0.5, vec![entity]);

        assert!(partition.contains_record(1));
        assert!(partition.contains_record(2));
        assert!(!partition.contains_record(0));
        assert!(!partition.contains_record(3));
    }

    #[test]
    fn test_find_entity_for_record() {
        let mut entity1 = RoaringBitmap::new();
        entity1.insert(0);
        entity1.insert(1);

        let mut entity2 = RoaringBitmap::new();
        entity2.insert(2);

        let partition = PartitionLevel::new(0.8, vec![entity1, entity2]);

        assert_eq!(partition.find_entity_for_record(0), Some(0));
        assert_eq!(partition.find_entity_for_record(1), Some(0));
        assert_eq!(partition.find_entity_for_record(2), Some(1));
        assert_eq!(partition.find_entity_for_record(3), None);
    }
}
