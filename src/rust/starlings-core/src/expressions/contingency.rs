//! Contingency table implementations for efficient metric computation

use crate::{DataContext, PartitionLevel};
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Entity ID for sparse contingency table
pub type EntityId = usize;

/// Sparse contingency table for efficient incremental updates
/// Only stores non-zero cells, enabling O(k) updates where k = affected entities
#[derive(Debug, Clone)]
pub struct SparseContingencyTable {
    /// Non-zero entity overlap counts: (entity1_id, entity2_id) -> overlap_count
    pub nonzero_cells: HashMap<(EntityId, EntityId), u32>,
    /// Row marginals: entity1_id -> total_records_in_entity1
    pub row_marginals: HashMap<EntityId, u32>,
    /// Column marginals: entity2_id -> total_records_in_entity2
    pub col_marginals: HashMap<EntityId, u32>,
    /// Total number of records across both partitions
    pub total_records: u32,
}

impl SparseContingencyTable {
    /// Create new empty sparse contingency table
    pub fn new() -> Self {
        Self {
            nonzero_cells: HashMap::new(),
            row_marginals: HashMap::new(),
            col_marginals: HashMap::new(),
            total_records: 0,
        }
    }

    /// Build initial sparse contingency table from two partitions
    /// O(k₁ × k₂) where k = number of entities (much better than O(n²))
    /// Uses parallel processing for large entity counts
    pub fn from_partitions(partition1: &PartitionLevel, partition2: &PartitionLevel) -> Self {
        let mut table = Self::new();

        // Calculate total records
        table.total_records = partition1.entities().iter().map(|e| e.len() as u32).sum();

        // Build row marginals (partition1 entity sizes)
        for (entity_id, entity) in partition1.entities().iter().enumerate() {
            table.row_marginals.insert(entity_id, entity.len() as u32);
        }

        // Build column marginals (partition2 entity sizes)
        for (entity_id, entity) in partition2.entities().iter().enumerate() {
            table.col_marginals.insert(entity_id, entity.len() as u32);
        }

        let n1 = partition1.entities().len();
        let n2 = partition2.entities().len();

        // Decide whether to use parallel or sequential processing
        // Parallel overhead is worth it for > 1000 entities in either partition
        let use_parallel = n1 > 1000 || n2 > 1000;

        if use_parallel {
            // Parallel processing for large entity counts
            let nonzero_cells = Mutex::new(HashMap::new());

            // Process entity pairs in parallel
            partition1
                .entities()
                .par_iter()
                .enumerate()
                .for_each(|(entity1_id, entity1)| {
                    // Local accumulator to reduce lock contention
                    let mut local_overlaps = Vec::new();

                    for (entity2_id, entity2) in partition2.entities().iter().enumerate() {
                        // Skip small entities early for optimisation
                        if entity1.len() < 2 && entity2.len() < 2 {
                            continue;
                        }

                        let overlap = entity1.intersection_len(entity2);
                        if overlap > 0 {
                            local_overlaps.push((entity1_id, entity2_id, overlap as u32));
                        }
                    }

                    // Batch insert to reduce lock contention
                    if !local_overlaps.is_empty() {
                        let mut cells = nonzero_cells.lock().unwrap();
                        for (e1, e2, overlap) in local_overlaps {
                            cells.insert((e1, e2), overlap);
                        }
                    }
                });

            table.nonzero_cells = nonzero_cells.into_inner().unwrap();
        } else {
            // Sequential processing for small entity counts
            for (entity1_id, entity1) in partition1.entities().iter().enumerate() {
                for (entity2_id, entity2) in partition2.entities().iter().enumerate() {
                    let overlap = entity1.intersection_len(entity2);
                    if overlap > 0 {
                        table
                            .nonzero_cells
                            .insert((entity1_id, entity2_id), overlap as u32);
                    }
                }
            }
        }

        table
    }

    /// Build contingency table using record-based algorithm (O(r) complexity)
    /// This is much faster than entity-based comparison when collections share the same context
    pub fn from_partitions_via_records(
        partition1: &PartitionLevel,
        partition2: &PartitionLevel,
        context: &Arc<DataContext>,
    ) -> Self {
        let mut table = Self::new();
        let num_records = context.len();
        let generation = context.generation();

        // Build reverse indices for both partitions
        let record_to_entity1 = partition1.get_record_to_entity_index(num_records, generation);
        let record_to_entity2 = partition2.get_record_to_entity_index(num_records, generation);

        // Check if indices are valid (same generation)
        if record_to_entity1.generation != generation || record_to_entity2.generation != generation
        {
            // Fall back to entity-based algorithm if caches are stale
            return Self::from_partitions(partition1, partition2);
        }

        // Calculate total records
        table.total_records = partition1.entities().iter().map(|e| e.len() as u32).sum();

        // Build row marginals (partition1 entity sizes)
        for (entity_id, entity) in partition1.entities().iter().enumerate() {
            table.row_marginals.insert(entity_id, entity.len() as u32);
        }

        // Build column marginals (partition2 entity sizes)
        for (entity_id, entity) in partition2.entities().iter().enumerate() {
            table.col_marginals.insert(entity_id, entity.len() as u32);
        }

        // Use parallel processing for large datasets
        if num_records > 10000 {
            // Parallel collection of entity pairs
            let pairs: Vec<(usize, usize)> = (0..num_records)
                .into_par_iter()
                .filter_map(|record_idx| {
                    let e1 = record_to_entity1.index.get(record_idx)?.as_ref()?;
                    let e2 = record_to_entity2.index.get(record_idx)?.as_ref()?;
                    Some((*e1, *e2))
                })
                .collect();

            // Aggregate into contingency table
            for (entity1_idx, entity2_idx) in pairs {
                *table
                    .nonzero_cells
                    .entry((entity1_idx, entity2_idx))
                    .or_insert(0) += 1;
            }
        } else {
            // Sequential processing for small datasets
            for record_idx in 0..num_records {
                if let (Some(Some(e1)), Some(Some(e2))) = (
                    record_to_entity1.index.get(record_idx),
                    record_to_entity2.index.get(record_idx),
                ) {
                    *table.nonzero_cells.entry((*e1, *e2)).or_insert(0) += 1;
                }
            }
        }

        table
    }

    /// Compute contingency table metrics from sparse representation
    pub fn to_contingency_table(&self) -> ContingencyTable {
        let mut true_positives = 0u64;
        let mut false_positives = 0u64;
        let mut false_negatives = 0u64;

        // Pre-compute pairs together for each entity to avoid O(k×m) complexity
        let mut entity1_pairs_together: HashMap<usize, u64> = HashMap::new();
        let mut entity2_pairs_together: HashMap<usize, u64> = HashMap::new();

        // Single pass through nonzero cells to compute all necessary sums
        for ((e1, e2), &overlap) in &self.nonzero_cells {
            if overlap > 1 {
                let pairs = (overlap as u64 * (overlap as u64 - 1)) / 2;
                true_positives += pairs;
                *entity1_pairs_together.entry(*e1).or_insert(0) += pairs;
                *entity2_pairs_together.entry(*e2).or_insert(0) += pairs;
            }
        }

        // False positives: pairs together in partition1, apart in partition2
        for (entity1_id, &size1) in &self.row_marginals {
            if size1 > 1 {
                let all_pairs_in_entity1 = (size1 as u64 * (size1 as u64 - 1)) / 2;
                let pairs_also_together =
                    entity1_pairs_together.get(entity1_id).copied().unwrap_or(0);
                false_positives += all_pairs_in_entity1 - pairs_also_together;
            }
        }

        // False negatives: pairs apart in partition1, together in partition2
        for (entity2_id, &size2) in &self.col_marginals {
            if size2 > 1 {
                let all_pairs_in_entity2 = (size2 as u64 * (size2 as u64 - 1)) / 2;
                let pairs_also_together =
                    entity2_pairs_together.get(entity2_id).copied().unwrap_or(0);
                false_negatives += all_pairs_in_entity2 - pairs_also_together;
            }
        }

        // True negatives: all possible pairs minus counted pairs
        let total_possible_pairs = if self.total_records > 1 {
            (self.total_records as u64 * (self.total_records as u64 - 1)) / 2
        } else {
            0
        };
        let true_negatives =
            total_possible_pairs.saturating_sub(true_positives + false_positives + false_negatives);

        ContingencyTable {
            true_positives: true_positives as u32,
            false_positives: false_positives as u32,
            false_negatives: false_negatives as u32,
            true_negatives: true_negatives as u32,
        }
    }
}

impl Default for SparseContingencyTable {
    fn default() -> Self {
        Self::new()
    }
}

/// Contingency table for comparing two partitions
#[derive(Debug)]
pub struct ContingencyTable {
    /// Number of record pairs that are in same entity in both partitions
    pub true_positives: u32,
    /// Number of record pairs that are in same entity in first but different in second
    pub false_positives: u32,
    /// Number of record pairs that are in different entities in first but same in second
    pub false_negatives: u32,
    /// Number of record pairs that are in different entities in both partitions
    pub true_negatives: u32,
}

impl ContingencyTable {
    /// Compute precision: TP / (TP + FP)
    pub fn precision(&self) -> f64 {
        let denominator = self.true_positives + self.false_positives;
        if denominator == 0 {
            // When no positive predictions are made, we have two conventions:
            // 1. Return 1.0 (no wrong predictions)
            // 2. Return 0.0 if there were true positives to find
            // We use convention 2: if there were positives to find (FN > 0) but we
            // predicted none, precision is 0.0
            if self.false_negatives > 0 {
                0.0
            } else {
                1.0 // No positives to find and none predicted - perfect precision
            }
        } else {
            self.true_positives as f64 / denominator as f64
        }
    }

    /// Compute recall: TP / (TP + FN)
    pub fn recall(&self) -> f64 {
        let denominator = self.true_positives + self.false_negatives;
        if denominator == 0 {
            1.0 // No true positives exist
        } else {
            self.true_positives as f64 / denominator as f64
        }
    }

    /// Compute F1 score: 2 * (precision * recall) / (precision + recall)
    pub fn f1_score(&self) -> f64 {
        let precision = self.precision();
        let recall = self.recall();

        if precision + recall == 0.0 {
            0.0
        } else {
            2.0 * (precision * recall) / (precision + recall)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use roaring::RoaringBitmap;

    fn create_test_partition(entities: Vec<Vec<u32>>) -> PartitionLevel {
        let bitmap_entities: Vec<RoaringBitmap> = entities
            .into_iter()
            .map(|records| records.into_iter().collect())
            .collect();
        PartitionLevel::new(0.0, bitmap_entities)
    }

    #[test]
    fn test_contingency_table_precision_recall_f1() {
        // Perfect match: both partitions identical
        let partition1 = create_test_partition(vec![vec![0, 1], vec![2, 3]]);
        let partition2 = create_test_partition(vec![vec![0, 1], vec![2, 3]]);

        let sparse_table = SparseContingencyTable::from_partitions(&partition1, &partition2);
        let table = sparse_table.to_contingency_table();

        assert_eq!(table.precision(), 1.0);
        assert_eq!(table.recall(), 1.0);
        assert_eq!(table.f1_score(), 1.0);

        // Complete mismatch: one partition has all singles, other has all together
        let partition1 = create_test_partition(vec![vec![0], vec![1], vec![2], vec![3]]);
        let partition2 = create_test_partition(vec![vec![0, 1, 2, 3]]);

        let sparse_table = SparseContingencyTable::from_partitions(&partition1, &partition2);
        let table = sparse_table.to_contingency_table();

        assert_eq!(table.precision(), 0.0); // No correct positive predictions
        assert_eq!(table.recall(), 0.0); // No true positives found
        assert_eq!(table.f1_score(), 0.0);
    }

    #[test]
    fn test_sparse_contingency_table() {
        let partition1 = create_test_partition(vec![vec![0, 1], vec![2]]);
        let partition2 = create_test_partition(vec![vec![0], vec![1, 2]]);

        let sparse_table = SparseContingencyTable::from_partitions(&partition1, &partition2);

        // Check marginals
        assert_eq!(sparse_table.row_marginals[&0], 2);
        assert_eq!(sparse_table.row_marginals[&1], 1);
        assert_eq!(sparse_table.col_marginals[&0], 1);
        assert_eq!(sparse_table.col_marginals[&1], 2);

        // Check nonzero cells
        assert!(sparse_table.nonzero_cells.contains_key(&(0, 0))); // entity1[0] overlaps with entity2[0]
        assert!(sparse_table.nonzero_cells.contains_key(&(0, 1))); // entity1[0] overlaps with entity2[1]
        assert!(sparse_table.nonzero_cells.contains_key(&(1, 1))); // entity1[1] overlaps with entity2[1]
    }
}
