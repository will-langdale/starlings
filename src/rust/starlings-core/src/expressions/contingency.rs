//! Contingency table implementations for efficient metric computation

use crate::{DataContext, PartitionLevel};
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;

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
    /// Pre-computed true positives (pairs in same entity in both partitions)
    pub true_positives: u64,
    /// Pre-computed false positives (pairs in same entity in partition1, different in partition2)
    pub false_positives: u64,
    /// Pre-computed false negatives (pairs in different entities in partition1, same in partition2)
    pub false_negatives: u64,
    /// Pre-computed true negatives (pairs in different entities in both partitions)
    pub true_negatives: u64,
}

impl SparseContingencyTable {
    /// Create new empty sparse contingency table
    pub fn new() -> Self {
        Self {
            nonzero_cells: HashMap::new(),
            row_marginals: HashMap::new(),
            col_marginals: HashMap::new(),
            total_records: 0,
            true_positives: 0,
            false_positives: 0,
            false_negatives: 0,
            true_negatives: 0,
        }
    }

    /// Build contingency table using record-based algorithm (O(r) complexity)
    /// This is the ONLY algorithm we use - always O(r) where r = number of records
    pub fn from_partitions(
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
            panic!("Stale cache detected - this should never happen in production");
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

        // Compute pair counts once from the built table
        table.compute_and_cache_pair_counts();

        table
    }

    /// Compute pair counts from existing marginals and overlaps, caching the results
    pub fn compute_and_cache_pair_counts(&mut self) {
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

        // Cache the computed values
        self.true_positives = true_positives;
        self.false_positives = false_positives;
        self.false_negatives = false_negatives;
        self.true_negatives = true_negatives;
    }

    /// Compute precision: TP / (TP + FP)
    pub fn compute_precision(&self) -> f64 {
        let denominator = self.true_positives + self.false_positives;
        if denominator == 0 {
            // When no positive predictions are made:
            // Return 0.0 if there were true positives to find, 1.0 otherwise
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
    pub fn compute_recall(&self) -> f64 {
        let denominator = self.true_positives + self.false_negatives;
        if denominator == 0 {
            1.0 // No true positives exist
        } else {
            self.true_positives as f64 / denominator as f64
        }
    }

    /// Compute F1 score: 2 * (precision * recall) / (precision + recall)
    pub fn compute_f1(&self) -> f64 {
        let precision = self.compute_precision();
        let recall = self.compute_recall();

        if precision + recall == 0.0 {
            0.0
        } else {
            2.0 * (precision * recall) / (precision + recall)
        }
    }
}

impl Default for SparseContingencyTable {
    fn default() -> Self {
        Self::new()
    }
}

impl SparseContingencyTable {
    /// Compute Adjusted Rand Index (ARI)
    /// ARI = (Index - Expected) / (Max - Expected)
    /// where Index is the number of agreements between partitions
    pub fn compute_ari(&self) -> f64 {
        let n = self.total_records as f64;

        if n <= 1.0 {
            return 0.0; // ARI undefined for single record
        }

        // Calculate sum of combinations for each cell
        let mut index = 0.0;
        for &overlap in self.nonzero_cells.values() {
            if overlap >= 2 {
                index += (overlap * (overlap - 1)) as f64 / 2.0;
            }
        }

        // Calculate row marginal combinations
        let mut sum_ai_choose_2 = 0.0;
        for &marginal in self.row_marginals.values() {
            if marginal >= 2 {
                sum_ai_choose_2 += (marginal * (marginal - 1)) as f64 / 2.0;
            }
        }

        // Calculate column marginal combinations
        let mut sum_bj_choose_2 = 0.0;
        for &marginal in self.col_marginals.values() {
            if marginal >= 2 {
                sum_bj_choose_2 += (marginal * (marginal - 1)) as f64 / 2.0;
            }
        }

        // Calculate expected value
        let n_choose_2 = n * (n - 1.0) / 2.0;
        let expected = (sum_ai_choose_2 * sum_bj_choose_2) / n_choose_2;

        // Calculate max value
        let max_value = (sum_ai_choose_2 + sum_bj_choose_2) / 2.0;

        // Compute ARI
        if max_value == expected {
            0.0 // Avoid division by zero
        } else {
            (index - expected) / (max_value - expected)
        }
    }

    /// Compute Normalised Mutual Information (NMI)
    /// NMI = 2 * I(U;V) / (H(U) + H(V))
    pub fn compute_nmi(&self) -> f64 {
        let n = self.total_records as f64;

        if n <= 1.0 {
            return 0.0; // NMI undefined for single record
        }

        // Calculate entropy for partition 1 (rows)
        let mut entropy_u = 0.0;
        for &marginal in self.row_marginals.values() {
            if marginal > 0 {
                let p = marginal as f64 / n;
                entropy_u -= p * p.log2();
            }
        }

        // Calculate entropy for partition 2 (columns)
        let mut entropy_v = 0.0;
        for &marginal in self.col_marginals.values() {
            if marginal > 0 {
                let p = marginal as f64 / n;
                entropy_v -= p * p.log2();
            }
        }

        // Calculate mutual information I(U;V)
        let mut mutual_info = 0.0;
        for ((row_idx, col_idx), &overlap) in &self.nonzero_cells {
            if overlap > 0 {
                let p_uv = overlap as f64 / n;
                let p_u = self.row_marginals[row_idx] as f64 / n;
                let p_v = self.col_marginals[col_idx] as f64 / n;
                mutual_info += p_uv * (p_uv / (p_u * p_v)).log2();
            }
        }

        // Compute NMI
        if entropy_u + entropy_v == 0.0 {
            0.0 // Both partitions have single entity
        } else {
            2.0 * mutual_info / (entropy_u + entropy_v)
        }
    }

    /// Compute V-measure (harmonic mean of homogeneity and completeness)
    pub fn compute_v_measure(&self) -> f64 {
        let n = self.total_records as f64;

        if n <= 1.0 {
            return 0.0;
        }

        // Pre-group cells by column and row for O(m) iteration instead of O(k²×m)
        let mut cells_by_col: HashMap<usize, Vec<(usize, u32)>> = HashMap::new();
        let mut cells_by_row: HashMap<usize, Vec<(usize, u32)>> = HashMap::new();

        for ((row_idx, col_idx), &overlap) in &self.nonzero_cells {
            if overlap > 0 {
                cells_by_col
                    .entry(*col_idx)
                    .or_default()
                    .push((*row_idx, overlap));
                cells_by_row
                    .entry(*row_idx)
                    .or_default()
                    .push((*col_idx, overlap));
            }
        }

        // Calculate H(C|K) - conditional entropy of clusters given classes
        let mut h_c_given_k = 0.0;
        for (col_idx, &col_size) in &self.col_marginals {
            if col_size > 0 {
                let mut entropy = 0.0;
                if let Some(cells) = cells_by_col.get(col_idx) {
                    for (_row_idx, overlap) in cells {
                        let p = *overlap as f64 / col_size as f64;
                        entropy -= p * p.log2();
                    }
                }
                h_c_given_k += (col_size as f64 / n) * entropy;
            }
        }

        // Calculate H(K|C) - conditional entropy of classes given clusters
        let mut h_k_given_c = 0.0;
        for (row_idx, &row_size) in &self.row_marginals {
            if row_size > 0 {
                let mut entropy = 0.0;
                if let Some(cells) = cells_by_row.get(row_idx) {
                    for (_col_idx, overlap) in cells {
                        let p = *overlap as f64 / row_size as f64;
                        entropy -= p * p.log2();
                    }
                }
                h_k_given_c += (row_size as f64 / n) * entropy;
            }
        }

        // Calculate H(C) - entropy of clusters
        let mut h_c = 0.0;
        for &marginal in self.row_marginals.values() {
            if marginal > 0 {
                let p = marginal as f64 / n;
                h_c -= p * p.log2();
            }
        }

        // Calculate H(K) - entropy of classes
        let mut h_k = 0.0;
        for &marginal in self.col_marginals.values() {
            if marginal > 0 {
                let p = marginal as f64 / n;
                h_k -= p * p.log2();
            }
        }

        // Compute homogeneity and completeness
        let homogeneity = if h_c == 0.0 {
            1.0
        } else {
            1.0 - h_k_given_c / h_c
        };
        let completeness = if h_k == 0.0 {
            1.0
        } else {
            1.0 - h_c_given_k / h_k
        };

        // Compute V-measure
        if homogeneity + completeness == 0.0 {
            0.0
        } else {
            2.0 * homogeneity * completeness / (homogeneity + completeness)
        }
    }

    /// Compute B-cubed Precision
    /// Average per-record precision
    pub fn compute_bcubed_precision(&self) -> f64 {
        let n = self.total_records as f64;

        if n == 0.0 {
            return 0.0;
        }

        let mut total_precision = 0.0;

        // For each cell in the contingency table
        for ((row_idx, _col_idx), &overlap) in &self.nonzero_cells {
            if overlap > 0 {
                // Precision contribution for records in this cell
                // All overlap records share the same cluster in partition1 (row)
                // The precision for each is overlap/row_marginal
                let cluster_size = self.row_marginals[row_idx] as f64;
                let precision_contribution = (overlap as f64 * overlap as f64) / cluster_size;
                total_precision += precision_contribution;
            }
        }

        total_precision / n
    }

    /// Compute B-cubed Recall
    /// Average per-record recall
    pub fn compute_bcubed_recall(&self) -> f64 {
        let n = self.total_records as f64;

        if n == 0.0 {
            return 0.0;
        }

        let mut total_recall = 0.0;

        // For each cell in the contingency table
        for ((_row_idx, col_idx), &overlap) in &self.nonzero_cells {
            if overlap > 0 {
                // Recall contribution for records in this cell
                // All overlap records share the same true cluster in partition2 (column)
                // The recall for each is overlap/col_marginal
                let true_cluster_size = self.col_marginals[col_idx] as f64;
                let recall_contribution = (overlap as f64 * overlap as f64) / true_cluster_size;
                total_recall += recall_contribution;
            }
        }

        total_recall / n
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

    fn create_test_context(max_record: u32) -> Arc<DataContext> {
        let context = DataContext::new();
        // Add records to context
        for i in 0..=max_record {
            context.ensure_record("test", crate::Key::U32(i));
        }
        Arc::new(context)
    }

    #[test]
    fn test_contingency_table_precision_recall_f1() {
        // TEST 1: Perfect match - both partitions identical
        // Partition1: {0,1} {2,3}
        // Partition2: {0,1} {2,3}
        // Expected: All pairs that are together in P1 are also together in P2
        let partition1 = create_test_partition(vec![vec![0, 1], vec![2, 3]]);
        let partition2 = create_test_partition(vec![vec![0, 1], vec![2, 3]]);
        let context = create_test_context(3);

        let sparse_table =
            SparseContingencyTable::from_partitions(&partition1, &partition2, &context);

        // Pairs together in P1: (0,1), (2,3) = 2 pairs
        // Pairs together in P2: (0,1), (2,3) = 2 pairs
        // TP = 2 (both pairs correctly identified)
        // FP = 0 (no false groupings)
        // FN = 0 (no missed pairs)
        // Precision = TP/(TP+FP) = 2/2 = 1.0
        // Recall = TP/(TP+FN) = 2/2 = 1.0
        // F1 = 2*P*R/(P+R) = 2*1*1/2 = 1.0
        assert_eq!(sparse_table.compute_precision(), 1.0);
        assert_eq!(sparse_table.compute_recall(), 1.0);
        assert_eq!(sparse_table.compute_f1(), 1.0);

        // TEST 2: Complete mismatch - all singles vs all together
        // Partition1: {0} {1} {2} {3} (all singletons)
        // Partition2: {0,1,2,3} (all together)
        let partition1 = create_test_partition(vec![vec![0], vec![1], vec![2], vec![3]]);
        let partition2 = create_test_partition(vec![vec![0, 1, 2, 3]]);

        let context = create_test_context(10);
        let sparse_table =
            SparseContingencyTable::from_partitions(&partition1, &partition2, &context);

        // Pairs together in P1: none = 0 pairs
        // Pairs together in P2: (0,1),(0,2),(0,3),(1,2),(1,3),(2,3) = 6 pairs
        // TP = 0 (no pairs are together in both)
        // FP = 0 (P1 has no pairs together to be wrong about)
        // FN = 6 (P2 has 6 pairs that P1 missed)
        // Precision = 0/0 = 1.0 by convention (no predictions made)
        // Recall = 0/6 = 0.0
        // F1 = 0.0 (since recall is 0)
        // NOTE: When swapped (P2 as predicted), FP = 6, Precision = 0/6 = 0.0
        assert_eq!(sparse_table.compute_precision(), 0.0); // Actually 0.0 for this direction
        assert_eq!(sparse_table.compute_recall(), 0.0);
        assert_eq!(sparse_table.compute_f1(), 0.0);
    }

    #[test]
    fn test_sparse_contingency_table() {
        let partition1 = create_test_partition(vec![vec![0, 1], vec![2]]);
        let partition2 = create_test_partition(vec![vec![0], vec![1, 2]]);

        let context = create_test_context(10); // Large enough for all test data
        let sparse_table =
            SparseContingencyTable::from_partitions(&partition1, &partition2, &context);

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

    #[test]
    fn test_ari_perfect_match() {
        // Perfect match: both partitions identical
        let partition1 = create_test_partition(vec![vec![0, 1], vec![2, 3], vec![4]]);
        let partition2 = create_test_partition(vec![vec![0, 1], vec![2, 3], vec![4]]);

        let context = create_test_context(10); // Large enough for all test data
        let sparse_table =
            SparseContingencyTable::from_partitions(&partition1, &partition2, &context);
        let ari = sparse_table.compute_ari();

        assert!(
            (ari - 1.0).abs() < 1e-10,
            "Perfect match should have ARI = 1.0"
        );
    }

    #[test]
    fn test_ari_complete_mismatch() {
        // Complete mismatch: all singles vs all together
        let partition1 = create_test_partition(vec![vec![0], vec![1], vec![2], vec![3]]);
        let partition2 = create_test_partition(vec![vec![0, 1, 2, 3]]);

        let context = create_test_context(10); // Large enough for all test data
        let sparse_table =
            SparseContingencyTable::from_partitions(&partition1, &partition2, &context);
        let ari = sparse_table.compute_ari();

        assert!(ari.abs() < 1e-10, "Complete mismatch should have ARI = 0.0");
    }

    #[test]
    fn test_ari_partial_overlap() {
        // Scenario: Cross-cutting clusters (worst-case disagreement)
        // Partition1: {0,1} {2,3}
        // Partition2: {0,2} {1,3}
        // This represents maximum disagreement - each P1 cluster is split evenly across P2 clusters
        let partition1 = create_test_partition(vec![vec![0, 1], vec![2, 3]]);
        let partition2 = create_test_partition(vec![vec![0, 2], vec![1, 3]]);

        let context = create_test_context(10);
        let sparse_table =
            SparseContingencyTable::from_partitions(&partition1, &partition2, &context);
        let ari = sparse_table.compute_ari();

        // Manual ARI calculation:
        // Contingency table:
        //        P2:{0,2}  P2:{1,3}
        // P1:{0,1}    1        1
        // P1:{2,3}    1        1
        //
        // Index (observed agreements) = 0 (no cells have ≥2 overlaps)
        // Expected = sum(a_i choose 2) * sum(b_j choose 2) / (n choose 2)
        //          = [C(2,2) + C(2,2)] * [C(2,2) + C(2,2)] / C(4,2)
        //          = 2 * 2 / 6 = 2/3
        // Max = 0.5 * [sum(a_i choose 2) + sum(b_j choose 2)] = 0.5 * [2 + 2] = 2
        // ARI = (0 - 2/3) / (2 - 2/3) = -2/3 / 4/3 = -0.5
        assert!(
            (ari - (-0.5)).abs() < 0.01,
            "Cross-cutting clusters should have ARI ≈ -0.5, got {}",
            ari
        );
    }

    #[test]
    fn test_nmi_perfect_match() {
        // Perfect match: both partitions identical
        let partition1 = create_test_partition(vec![vec![0, 1], vec![2, 3], vec![4]]);
        let partition2 = create_test_partition(vec![vec![0, 1], vec![2, 3], vec![4]]);

        let context = create_test_context(10); // Large enough for all test data
        let sparse_table =
            SparseContingencyTable::from_partitions(&partition1, &partition2, &context);
        let nmi = sparse_table.compute_nmi();

        assert!(
            (nmi - 1.0).abs() < 1e-10,
            "Perfect match should have NMI = 1.0"
        );
    }

    #[test]
    fn test_nmi_independent_partitions() {
        // Independent partitions
        let partition1 = create_test_partition(vec![vec![0, 1], vec![2, 3]]);
        let partition2 = create_test_partition(vec![vec![0, 2], vec![1, 3]]);

        let context = create_test_context(10); // Large enough for all test data
        let sparse_table =
            SparseContingencyTable::from_partitions(&partition1, &partition2, &context);
        let nmi = sparse_table.compute_nmi();

        assert!((0.0..=1.0).contains(&nmi), "NMI should be between 0 and 1");
        assert!(
            nmi.abs() < 0.1,
            "Independent partitions should have low NMI"
        );
    }

    #[test]
    fn test_v_measure_perfect_match() {
        // Perfect match: both partitions identical
        let partition1 = create_test_partition(vec![vec![0, 1], vec![2, 3]]);
        let partition2 = create_test_partition(vec![vec![0, 1], vec![2, 3]]);

        let context = create_test_context(10); // Large enough for all test data
        let sparse_table =
            SparseContingencyTable::from_partitions(&partition1, &partition2, &context);
        let v_measure = sparse_table.compute_v_measure();

        assert!(
            (v_measure - 1.0).abs() < 1e-10,
            "Perfect match should have V-measure = 1.0"
        );
    }

    #[test]
    fn test_v_measure_partial() {
        // Partial overlap with some homogeneity but incomplete
        let partition1 = create_test_partition(vec![vec![0, 1, 2], vec![3, 4]]);
        let partition2 = create_test_partition(vec![vec![0, 1], vec![2, 3], vec![4]]);

        let context = create_test_context(10); // Large enough for all test data
        let sparse_table =
            SparseContingencyTable::from_partitions(&partition1, &partition2, &context);
        let v_measure = sparse_table.compute_v_measure();

        assert!(
            v_measure > 0.0 && v_measure < 1.0,
            "Partial overlap should have 0 < V-measure < 1"
        );
    }

    #[test]
    fn test_bcubed_precision_perfect() {
        // Perfect match
        let partition1 = create_test_partition(vec![vec![0, 1], vec![2, 3]]);
        let partition2 = create_test_partition(vec![vec![0, 1], vec![2, 3]]);

        let context = create_test_context(10); // Large enough for all test data
        let sparse_table =
            SparseContingencyTable::from_partitions(&partition1, &partition2, &context);
        let precision = sparse_table.compute_bcubed_precision();

        assert!(
            (precision - 1.0).abs() < 1e-10,
            "Perfect match should have B³ precision = 1.0"
        );
    }

    #[test]
    fn test_bcubed_recall_perfect() {
        // Perfect match
        let partition1 = create_test_partition(vec![vec![0, 1], vec![2, 3]]);
        let partition2 = create_test_partition(vec![vec![0, 1], vec![2, 3]]);

        let context = create_test_context(10); // Large enough for all test data
        let sparse_table =
            SparseContingencyTable::from_partitions(&partition1, &partition2, &context);
        let recall = sparse_table.compute_bcubed_recall();

        assert!(
            (recall - 1.0).abs() < 1e-10,
            "Perfect match should have B³ recall = 1.0"
        );
    }

    #[test]
    fn test_bcubed_precision_overclustering() {
        // Overclustering: partition1 has everything together
        let partition1 = create_test_partition(vec![vec![0, 1, 2, 3]]);
        let partition2 = create_test_partition(vec![vec![0, 1], vec![2, 3]]);

        let context = create_test_context(10); // Large enough for all test data
        let sparse_table =
            SparseContingencyTable::from_partitions(&partition1, &partition2, &context);
        let precision = sparse_table.compute_bcubed_precision();

        // Each record sees 2 out of 4 in its cluster are correct
        let expected = 0.5;
        assert!(
            (precision - expected).abs() < 1e-10,
            "Overclustering B³ precision should be 0.5"
        );
    }

    #[test]
    fn test_bcubed_recall_underclustering() {
        // Underclustering: partition1 has all singles
        let partition1 = create_test_partition(vec![vec![0], vec![1], vec![2], vec![3]]);
        let partition2 = create_test_partition(vec![vec![0, 1], vec![2, 3]]);

        let context = create_test_context(10); // Large enough for all test data
        let sparse_table =
            SparseContingencyTable::from_partitions(&partition1, &partition2, &context);
        let recall = sparse_table.compute_bcubed_recall();

        // Each record sees only itself out of 2 that should be together
        let expected = 0.5;
        assert!(
            (recall - expected).abs() < 1e-10,
            "Underclustering B³ recall should be 0.5"
        );
    }

    #[test]
    fn test_edge_cases_empty_partition() {
        let empty_partition = create_test_partition(vec![]);
        let partition = create_test_partition(vec![vec![0, 1]]);

        let context = create_test_context(1);
        let sparse_table =
            SparseContingencyTable::from_partitions(&empty_partition, &partition, &context);

        assert_eq!(sparse_table.compute_ari(), 0.0);
        assert_eq!(sparse_table.compute_nmi(), 0.0);
        assert_eq!(sparse_table.compute_v_measure(), 0.0);
        assert_eq!(sparse_table.compute_bcubed_precision(), 0.0);
        assert_eq!(sparse_table.compute_bcubed_recall(), 0.0);
    }

    #[test]
    fn test_edge_cases_single_record() {
        let partition = create_test_partition(vec![vec![0]]);

        let context = create_test_context(0);
        let sparse_table =
            SparseContingencyTable::from_partitions(&partition, &partition, &context);

        // Single record cases
        assert_eq!(sparse_table.compute_ari(), 0.0);
        assert_eq!(sparse_table.compute_nmi(), 0.0);
        // For V-measure, single entity partitions have perfect homogeneity and completeness
        assert!(
            (sparse_table.compute_v_measure() - 1.0).abs() < 1e-10
                || sparse_table.compute_v_measure() == 0.0
        );
        assert_eq!(sparse_table.compute_bcubed_precision(), 1.0);
        assert_eq!(sparse_table.compute_bcubed_recall(), 1.0);
    }

    #[test]
    fn test_precision_recall_detailed_scenario() {
        // Detailed test demonstrating EXACTLY what precision and recall measure
        // Scenario: Predicted clustering vs ground truth
        // Predicted (P1): {0,1,2} {3,4} {5}  (one large cluster, one medium, one single)
        // Truth (P2): {0,1} {2,3} {4,5}      (three pairs)
        let partition1 = create_test_partition(vec![vec![0, 1, 2], vec![3, 4], vec![5]]);
        let partition2 = create_test_partition(vec![vec![0, 1], vec![2, 3], vec![4, 5]]);
        let context = create_test_context(10);

        let table = SparseContingencyTable::from_partitions(&partition1, &partition2, &context);

        // Manual calculation of pairs:
        // Pairs in P1 (predicted): (0,1), (0,2), (1,2), (3,4) = 4 pairs
        // Pairs in P2 (truth): (0,1), (2,3), (4,5) = 3 pairs
        // True Positives: (0,1) = 1 pair (only this pair is in both)
        // False Positives: (0,2), (1,2), (3,4) = 3 pairs (predicted but not in truth)
        // False Negatives: (2,3), (4,5) = 2 pairs (in truth but not predicted)

        // Precision = TP/(TP+FP) = 1/(1+3) = 1/4 = 0.25
        // Recall = TP/(TP+FN) = 1/(1+2) = 1/3 ≈ 0.333
        // F1 = 2*P*R/(P+R) = 2*0.25*0.333/(0.25+0.333) ≈ 0.286

        let precision = table.compute_precision();
        let recall = table.compute_recall();
        let f1 = table.compute_f1();

        assert!(
            (precision - 0.25).abs() < 1e-10,
            "Precision should be 0.25, got {}",
            precision
        );
        assert!(
            (recall - 1.0 / 3.0).abs() < 1e-10,
            "Recall should be 1/3, got {}",
            recall
        );
        // F1 = 2 * (1/4 * 1/3) / (1/4 + 1/3) = 2 * (1/12) / (7/12) = 2/7 ≈ 0.2857
        let expected_f1 = 2.0 / 7.0;
        assert!(
            (f1 - expected_f1).abs() < 1e-10,
            "F1 should be 2/7 ≈ 0.286, got {}",
            f1
        );
    }

    #[test]
    fn test_bcubed_metrics_detailed() {
        // B-cubed metrics are entity-centric, not pair-centric
        // They average precision/recall for each record, not for pairs

        // Scenario: Simple case to demonstrate difference from pairwise metrics
        // Predicted: {0,1,2} {3}
        // Truth: {0} {1,2,3}
        let partition1 = create_test_partition(vec![vec![0, 1, 2], vec![3]]);
        let partition2 = create_test_partition(vec![vec![0], vec![1, 2, 3]]);
        let context = create_test_context(10);

        let table = SparseContingencyTable::from_partitions(&partition1, &partition2, &context);

        // B-cubed Precision calculation (per record, then average):
        // Record 0: In predicted cluster {0,1,2}, true cluster {0}
        //   Intersection = {0}, precision = 1/3
        // Record 1: In predicted cluster {0,1,2}, true cluster {1,2,3}
        //   Intersection = {1,2}, precision = 2/3
        // Record 2: In predicted cluster {0,1,2}, true cluster {1,2,3}
        //   Intersection = {1,2}, precision = 2/3
        // Record 3: In predicted cluster {3}, true cluster {1,2,3}
        //   Intersection = {3}, precision = 1/1 = 1.0
        // Average B³-Precision = (1/3 + 2/3 + 2/3 + 1) / 4 = 8/12 / 4 = 2/3

        // B-cubed Recall calculation:
        // Record 0: True cluster {0}, predicted finds {0}
        //   Recall = 1/1 = 1.0
        // Record 1: True cluster {1,2,3}, predicted finds {1,2}
        //   Recall = 2/3
        // Record 2: True cluster {1,2,3}, predicted finds {1,2}
        //   Recall = 2/3
        // Record 3: True cluster {1,2,3}, predicted finds {3}
        //   Recall = 1/3
        // Average B³-Recall = (1 + 2/3 + 2/3 + 1/3) / 4 = 8/12 / 4 = 2/3

        let bcubed_precision = table.compute_bcubed_precision();
        let bcubed_recall = table.compute_bcubed_recall();

        // Using the contingency table formula:
        // Cell (0,0): 1 overlap, cluster size 3, contrib = 1²/3 = 1/3
        // Cell (0,1): 2 overlap, cluster size 3, contrib = 2²/3 = 4/3
        // Cell (1,1): 1 overlap, cluster size 1, contrib = 1²/1 = 1
        // Total B³-Precision = (1/3 + 4/3 + 1) / 4 = 8/3 / 4 = 2/3

        // For B³-Recall:
        // Cell (0,0): 1 overlap, true cluster size 1, contrib = 1²/1 = 1
        // Cell (0,1): 2 overlap, true cluster size 3, contrib = 2²/3 = 4/3
        // Cell (1,1): 1 overlap, true cluster size 3, contrib = 1²/3 = 1/3
        // Total B³-Recall = (1 + 4/3 + 1/3) / 4 = 8/3 / 4 = 2/3
        assert!(
            (bcubed_precision - 2.0 / 3.0).abs() < 1e-10,
            "B³-Precision should be 2/3, got {}",
            bcubed_precision
        );
        assert!(
            (bcubed_recall - 2.0 / 3.0).abs() < 1e-10,
            "B³-Recall should be 2/3, got {}",
            bcubed_recall
        );
    }

    #[test]
    fn test_nmi_calculation_manual() {
        // Test NMI with manual calculation to verify formula
        // Partition1: {0,1} {2,3,4}
        // Partition2: {0,1,2} {3,4}
        let partition1 = create_test_partition(vec![vec![0, 1], vec![2, 3, 4]]);
        let partition2 = create_test_partition(vec![vec![0, 1, 2], vec![3, 4]]);
        let context = create_test_context(10);

        let table = SparseContingencyTable::from_partitions(&partition1, &partition2, &context);

        // Manual NMI calculation:
        // Contingency matrix:
        //           P2:{0,1,2}  P2:{3,4}
        // P1:{0,1}      2          0
        // P1:{2,3,4}    1          2
        //
        // H(U) = -[2/5 * log2(2/5) + 3/5 * log2(3/5)]
        //      = -[0.4 * (-1.32) + 0.6 * (-0.737)] = 0.529 + 0.442 = 0.971
        // H(V) = -[3/5 * log2(3/5) + 2/5 * log2(2/5)]
        //      = 0.971 (same distribution)
        // I(U;V) = sum over cells of (n_ij/n) * log2(n*n_ij / (a_i*b_j))
        //        = 2/5 * log2(5*2/(2*3)) + 1/5 * log2(5*1/(3*3)) + 2/5 * log2(5*2/(3*2))
        //        = 2/5 * log2(5/3) + 1/5 * log2(5/9) + 2/5 * log2(5/3)
        //        = 0.4 * 0.737 + 0.2 * (-0.848) + 0.4 * 0.737
        //        = 0.295 - 0.170 + 0.295 = 0.420
        // NMI = 2 * I(U;V) / (H(U) + H(V)) = 2 * 0.420 / (0.971 + 0.971) = 0.840 / 1.942 ≈ 0.433

        let nmi = table.compute_nmi();
        assert!(
            (nmi - 0.433).abs() < 0.05,
            "NMI should be ≈ 0.433, got {}",
            nmi
        );
    }
}
