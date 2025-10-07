//! Record-based algorithm for O(r) metric computation
//!
//! ## Overview
//! The Record algorithm exploits spatial locality by iterating through all records
//! exactly once to build contingency tables. This single-pass approach is optimal
//! for cross-collection comparisons where incremental updates aren't possible.
//!
//! ## Direction
//! **Direction-agnostic**: The Record algorithm processes all records in a single pass
//! regardless of threshold order. It can handle sweeps in any direction (high to low,
//! low to high, or random access) with identical O(r) performance.
//!
//! ## How it Works
//! - **Input**: Any two partitions (possibly from different collections)
//! - **Process**: Single pass through all records, building contingency tables
//! - **Output**: Metrics computed in O(r) where r = total records
//!
//! ## When to Use
//! - ✅ Cross-collection comparisons
//! - ✅ Sweep × sweep operations
//! - ✅ One-off partition comparisons
//! - ✅ When partitions come from different hierarchies
//! - ❌ Same-hierarchy threshold sweeps (use Delta algorithm)
//!
//! ## Complexity
//! - Time: O(r) where r = number of records
//! - Memory: O(k₁ × k₂) where k₁, k₂ = entities in each partition
//! - Parallel: O(r/p) with p processors for large datasets
//!
//! ## Relationship to Delta Algorithm
//! Delta and Record are the two fundamental approaches to partition reconstruction:
//! - **Delta**: Exploits temporal locality (changes between thresholds)
//! - **Record**: Exploits spatial locality (all records in one pass)
//!   Together they exhaustively cover all metric computation scenarios.

use super::{ComparisonType, ComplexityEstimate, MetricAlgorithm, MetricResults, MetricType};
use crate::metrics::contingency::{
    choose_2, compute_conditional_entropy_flat, compute_entropy_from_sizes, ContingencyMetrics,
};
use crate::metrics::implementations::statistics::{compute_entity_count, compute_entropy};
use crate::{DataContext, PartitionLevel};
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;

/// Entity index (position in partition array) for contingency table
/// Used by Record algorithm for direct array access
pub type EntityIndex = usize;

/// Record-based state for efficient metric computation
/// Uses sequential entity indices for O(r) single-pass construction
#[derive(Debug, Clone)]
pub struct RecordState {
    /// Non-zero entity overlap counts: (entity1_idx, entity2_idx) -> overlap_count
    pub nonzero_cells: HashMap<(EntityIndex, EntityIndex), u32>,
    /// Row marginals: entity1_idx -> total_records_in_entity1
    pub row_marginals: HashMap<EntityIndex, u32>,
    /// Column marginals: entity2_idx -> total_records_in_entity2
    pub col_marginals: HashMap<EntityIndex, u32>,
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

impl RecordState {
    /// Create new empty record state
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
                let pairs = choose_2(overlap);
                true_positives += pairs;
                *entity1_pairs_together.entry(*e1).or_insert(0) += pairs;
                *entity2_pairs_together.entry(*e2).or_insert(0) += pairs;
            }
        }

        // False positives: pairs together in partition1, apart in partition2
        for (entity1_id, &size1) in &self.row_marginals {
            if size1 > 1 {
                let all_pairs_in_entity1 = choose_2(size1);
                let pairs_also_together =
                    entity1_pairs_together.get(entity1_id).copied().unwrap_or(0);
                false_positives += all_pairs_in_entity1 - pairs_also_together;
            }
        }

        // False negatives: pairs apart in partition1, together in partition2
        for (entity2_id, &size2) in &self.col_marginals {
            if size2 > 1 {
                let all_pairs_in_entity2 = choose_2(size2);
                let pairs_also_together =
                    entity2_pairs_together.get(entity2_id).copied().unwrap_or(0);
                false_negatives += all_pairs_in_entity2 - pairs_also_together;
            }
        }

        // True negatives: all possible pairs minus counted pairs
        let total_possible_pairs = if self.total_records > 1 {
            choose_2(self.total_records)
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
}

impl Default for RecordState {
    fn default() -> Self {
        Self::new()
    }
}

/// Implement ContingencyMetrics trait for RecordState
impl ContingencyMetrics for RecordState {
    fn compute_precision(&self) -> f64 {
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

    fn compute_recall(&self) -> f64 {
        let denominator = self.true_positives + self.false_negatives;
        if denominator == 0 {
            1.0 // No true positives exist
        } else {
            self.true_positives as f64 / denominator as f64
        }
    }

    fn compute_f1(&self) -> f64 {
        let precision = self.compute_precision();
        let recall = self.compute_recall();

        if precision + recall == 0.0 {
            0.0
        } else {
            2.0 * (precision * recall) / (precision + recall)
        }
    }

    fn compute_ari(&self) -> f64 {
        let n = self.total_records as f64;

        if n <= 1.0 {
            return 0.0; // ARI undefined for single record
        }

        // Calculate sum of combinations for each cell
        let mut index = 0.0;
        for &overlap in self.nonzero_cells.values() {
            if overlap >= 2 {
                index += choose_2(overlap) as f64;
            }
        }

        // Calculate row marginal combinations
        let mut sum_ai_choose_2 = 0.0;
        for &marginal in self.row_marginals.values() {
            if marginal >= 2 {
                sum_ai_choose_2 += choose_2(marginal) as f64;
            }
        }

        // Calculate column marginal combinations
        let mut sum_bj_choose_2 = 0.0;
        for &marginal in self.col_marginals.values() {
            if marginal >= 2 {
                sum_bj_choose_2 += choose_2(marginal) as f64;
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

    fn compute_nmi(&self) -> f64 {
        let n = self.total_records as f64;

        if n <= 1.0 {
            return 0.0; // NMI undefined for single record
        }

        // Calculate entropy for partition 1 (rows)
        let entropy_u = compute_entropy_from_sizes(self.row_marginals.values().copied(), n);

        // Calculate entropy for partition 2 (columns)
        let entropy_v = compute_entropy_from_sizes(self.col_marginals.values().copied(), n);

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

    fn compute_v_measure(&self) -> f64 {
        let n = self.total_records as f64;

        if n <= 1.0 {
            return 0.0;
        }

        // Calculate H(C|K) using our helper function
        let h_c_given_k =
            compute_conditional_entropy_flat(&self.nonzero_cells, &self.col_marginals, n);

        // Calculate H(K|C) using our helper function
        // Note: we need to flip the contingency table for this
        let mut flipped_contingency = HashMap::new();
        for ((row, col), count) in &self.nonzero_cells {
            flipped_contingency.insert((*col, *row), *count);
        }
        let h_k_given_c =
            compute_conditional_entropy_flat(&flipped_contingency, &self.row_marginals, n);

        // Calculate H(C) - entropy of clusters
        let h_c = compute_entropy_from_sizes(self.row_marginals.values().copied(), n);

        // Calculate H(K) - entropy of classes
        let h_k = compute_entropy_from_sizes(self.col_marginals.values().copied(), n);

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

    fn compute_bcubed_precision(&self) -> f64 {
        let n = self.total_records as f64;

        if n == 0.0 {
            return 0.0;
        }

        let mut total_precision = 0.0;

        // For each cell in the contingency table
        for ((row_idx, _), &overlap) in &self.nonzero_cells {
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

    fn compute_bcubed_recall(&self) -> f64 {
        let n = self.total_records as f64;

        if n == 0.0 {
            return 0.0;
        }

        let mut total_recall = 0.0;

        // For each cell in the contingency table
        for ((_, col_idx), &overlap) in &self.nonzero_cells {
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

/// Record-based algorithm that achieves O(r) complexity by iterating records once
///
/// This algorithm is one of two fundamental approaches (along with DeltaAlgorithm)
/// for computing metrics. It handles all cross-collection comparisons by iterating
/// through records once rather than maintaining incremental state.
pub struct RecordAlgorithm {}

impl Default for RecordAlgorithm {
    fn default() -> Self {
        Self::new()
    }
}

impl RecordAlgorithm {
    /// Create a new record-based algorithm instance
    pub fn new() -> Self {
        Self {}
    }

    /// Build all contingency tables for sweep × sweep in a single pass
    fn build_all_sweep_tables(
        &mut self,
        partitions1: &[PartitionLevel],
        partitions2: &[PartitionLevel],
        context: &Arc<DataContext>,
    ) -> Vec<Vec<RecordState>> {
        let num_records = context.len();
        let generation = context.generation();

        // Build reverse indices for all partitions
        let indices1: Vec<_> = partitions1
            .iter()
            .map(|p| p.get_record_to_entity_index(num_records, generation))
            .collect();
        let indices2: Vec<_> = partitions2
            .iter()
            .map(|p| p.get_record_to_entity_index(num_records, generation))
            .collect();

        // Initialise empty tables for all combinations
        let mut tables = vec![vec![RecordState::new(); partitions2.len()]; partitions1.len()];

        // Add marginals to all tables
        for (i, partition1) in partitions1.iter().enumerate() {
            for (j, partition2) in partitions2.iter().enumerate() {
                let table = &mut tables[i][j];
                table.total_records = partition1.entities().iter().map(|e| e.len() as u32).sum();

                for (entity_id, entity) in partition1.entities().iter().enumerate() {
                    table.row_marginals.insert(entity_id, entity.len() as u32);
                }
                for (entity_id, entity) in partition2.entities().iter().enumerate() {
                    table.col_marginals.insert(entity_id, entity.len() as u32);
                }
            }
        }

        // Single pass through all records, updating all tables
        for record_idx in 0..num_records {
            for (i, idx1) in indices1.iter().enumerate() {
                if let Some(entity1) = idx1.index.get(record_idx).and_then(|e| e.as_ref()) {
                    for (j, idx2) in indices2.iter().enumerate() {
                        if let Some(entity2) = idx2.index.get(record_idx).and_then(|e| e.as_ref()) {
                            // This record contributes to table[i][j]
                            *tables[i][j]
                                .nonzero_cells
                                .entry((*entity1, *entity2))
                                .or_insert(0) += 1;
                        }
                    }
                }
            }
        }

        // Compute pair counts for all tables
        for row in &mut tables {
            for table in row {
                table.compute_and_cache_pair_counts();
            }
        }

        tables
    }

    /// Compute metrics from a record state
    fn compute_metrics_from_state(
        &self,
        state: &RecordState,
        metrics: &[MetricType],
    ) -> MetricResults {
        let mut results = HashMap::new();

        for metric in metrics {
            let value = match metric {
                MetricType::F1 => state.compute_f1(),
                MetricType::Precision => state.compute_precision(),
                MetricType::Recall => state.compute_recall(),
                MetricType::ARI => state.compute_ari(),
                MetricType::NMI => state.compute_nmi(),
                MetricType::VMeasure => state.compute_v_measure(),
                MetricType::BCubedPrecision => state.compute_bcubed_precision(),
                MetricType::BCubedRecall => state.compute_bcubed_recall(),
                _ => 0.0, // Statistics metrics handled separately
            };
            results.insert(metric.name().to_string(), value);
        }

        results
    }
}

impl MetricAlgorithm for RecordAlgorithm {
    fn name(&self) -> &'static str {
        "Record-based (O(r) single-pass)"
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn can_handle(&self, comparison_type: &ComparisonType) -> bool {
        matches!(
            comparison_type,
            ComparisonType::SweepSweep { .. }
                | ComparisonType::PointPoint {
                    same_collection: false
                }
                | ComparisonType::SweepPoint {
                    same_collection: false
                }
        )
    }

    fn complexity(&self, comparison_type: &ComparisonType) -> ComplexityEstimate {
        match comparison_type {
            ComparisonType::SweepSweep { .. } => ComplexityEstimate {
                notation: "O(r)".to_string(),
                expected_ops: 1_000_000, // Rough estimate for 1M records
                description: "Single pass through all records".to_string(),
            },
            ComparisonType::PointPoint { .. } | ComparisonType::SweepPoint { .. } => {
                ComplexityEstimate {
                    notation: "O(r)".to_string(),
                    expected_ops: 1_000_000,
                    description: "Record iteration for cross-collection comparison".to_string(),
                }
            }
            _ => ComplexityEstimate {
                notation: "N/A".to_string(),
                expected_ops: 0,
                description: "Not handled by this algorithm".to_string(),
            },
        }
    }

    fn compute_single(
        &mut self,
        partition1: &PartitionLevel,
        partition2: Option<&PartitionLevel>,
        metrics: &[MetricType],
        context: &Arc<DataContext>,
    ) -> MetricResults {
        let mut results = HashMap::new();

        if let Some(partition2) = partition2 {
            // Comparison metrics
            let state = RecordState::from_partitions(partition1, partition2, context);
            results.extend(self.compute_metrics_from_state(&state, metrics));
        } else {
            // Single partition statistics
            for metric in metrics {
                let value = match metric {
                    MetricType::EntityCount => compute_entity_count(partition1),
                    MetricType::Entropy => compute_entropy(partition1),
                    _ => continue, // Skip comparison metrics
                };
                results.insert(metric.name().to_string(), value);
            }
        }

        results
    }

    fn compute_sweep(
        &mut self,
        partitions1: &[PartitionLevel],
        partitions2: Option<&[PartitionLevel]>,
        metrics: &[MetricType],
        context: &Arc<DataContext>,
    ) -> Vec<MetricResults> {
        if let Some(partitions2) = partitions2 {
            // Optimised sweep × sweep using single-pass algorithm
            let all_tables = self.build_all_sweep_tables(partitions1, partitions2, context);

            let mut results = Vec::new();
            for row in all_tables {
                for table in row {
                    results.push(self.compute_metrics_from_state(&table, metrics));
                }
            }
            results
        } else {
            // Single collection sweep - use default implementation
            let mut results = Vec::new();
            for p1 in partitions1 {
                results.push(self.compute_single(p1, None, metrics, context));
            }
            results
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Key;
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
            context.ensure_record("test", Key::U32(i));
        }
        Arc::new(context)
    }

    #[test]
    fn test_record_state_precision_recall_f1() {
        // TEST 1: Perfect match - both partitions identical
        // Partition1: {0,1} {2,3}
        // Partition2: {0,1} {2,3}
        let partition1 = create_test_partition(vec![vec![0, 1], vec![2, 3]]);
        let partition2 = create_test_partition(vec![vec![0, 1], vec![2, 3]]);
        let context = create_test_context(3);

        let state = RecordState::from_partitions(&partition1, &partition2, &context);

        assert_eq!(state.compute_precision(), 1.0);
        assert_eq!(state.compute_recall(), 1.0);
        assert_eq!(state.compute_f1(), 1.0);

        // TEST 2: Complete mismatch - all singles vs all together
        // Partition1: {0} {1} {2} {3} (all singletons)
        // Partition2: {0,1,2,3} (all together)
        let partition1 = create_test_partition(vec![vec![0], vec![1], vec![2], vec![3]]);
        let partition2 = create_test_partition(vec![vec![0, 1, 2, 3]]);

        let context = create_test_context(10);
        let state = RecordState::from_partitions(&partition1, &partition2, &context);

        assert_eq!(state.compute_precision(), 0.0);
        assert_eq!(state.compute_recall(), 0.0);
        assert_eq!(state.compute_f1(), 0.0);
    }

    #[test]
    fn test_record_state_ari() {
        // Perfect match: both partitions identical
        let partition1 = create_test_partition(vec![vec![0, 1], vec![2, 3], vec![4]]);
        let partition2 = create_test_partition(vec![vec![0, 1], vec![2, 3], vec![4]]);

        let context = create_test_context(10);
        let state = RecordState::from_partitions(&partition1, &partition2, &context);
        let ari = state.compute_ari();

        assert!(
            (ari - 1.0).abs() < 1e-10,
            "Perfect match should have ARI = 1.0"
        );
    }

    #[test]
    fn test_record_state_nmi() {
        // Perfect match: both partitions identical
        let partition1 = create_test_partition(vec![vec![0, 1], vec![2, 3], vec![4]]);
        let partition2 = create_test_partition(vec![vec![0, 1], vec![2, 3], vec![4]]);

        let context = create_test_context(10);
        let state = RecordState::from_partitions(&partition1, &partition2, &context);
        let nmi = state.compute_nmi();

        assert!(
            (nmi - 1.0).abs() < 1e-10,
            "Perfect match should have NMI = 1.0"
        );
    }

    #[test]
    fn test_record_state_v_measure() {
        // Perfect match: both partitions identical
        let partition1 = create_test_partition(vec![vec![0, 1], vec![2, 3]]);
        let partition2 = create_test_partition(vec![vec![0, 1], vec![2, 3]]);

        let context = create_test_context(10);
        let state = RecordState::from_partitions(&partition1, &partition2, &context);
        let v_measure = state.compute_v_measure();

        assert!(
            (v_measure - 1.0).abs() < 1e-10,
            "Perfect match should have V-measure = 1.0"
        );
    }

    #[test]
    fn test_record_state_bcubed() {
        // Perfect match
        let partition1 = create_test_partition(vec![vec![0, 1], vec![2, 3]]);
        let partition2 = create_test_partition(vec![vec![0, 1], vec![2, 3]]);

        let context = create_test_context(10);
        let state = RecordState::from_partitions(&partition1, &partition2, &context);
        let precision = state.compute_bcubed_precision();
        let recall = state.compute_bcubed_recall();

        assert!(
            (precision - 1.0).abs() < 1e-10,
            "Perfect match should have B³ precision = 1.0"
        );
        assert!(
            (recall - 1.0).abs() < 1e-10,
            "Perfect match should have B³ recall = 1.0"
        );
    }
}
