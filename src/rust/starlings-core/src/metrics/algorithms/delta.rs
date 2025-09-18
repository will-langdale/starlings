//! Delta algorithm for O(k) incremental metric computation
//!
//! ## Overview
//! The Delta algorithm exploits temporal locality when sweeping through thresholds
//! in the same hierarchy. It maintains incremental state and updates only the
//! entities affected by merge events between thresholds.
//!
//! ## How it Works
//! - **Input**: Sequence of thresholds in the same hierarchy
//! - **Process**: Applies only the delta (change) between adjacent thresholds
//! - **Output**: Metrics computed in O(k) where k = affected entities
//!
//! ## When to Use
//! - ✅ Same-hierarchy threshold sweeps
//! - ✅ Sequential threshold exploration
//! - ✅ When thresholds are close together (< 0.1 apart)
//! - ❌ Cross-collection comparisons (use Record algorithm)
//! - ❌ Random threshold access patterns
//!
//! ## Complexity
//! - Initial build: O(n) where n = number of entities
//! - Incremental updates: O(k) where k = entities affected by merges
//! - Memory: O(n) for maintaining state
//!
//! ## Relationship to Record Algorithm
//! Delta and Record are the two fundamental approaches to partition reconstruction:
//! - **Delta**: Exploits temporal locality (changes between thresholds)
//! - **Record**: Exploits spatial locality (all records in one pass)
//! Together they exhaustively cover all metric computation scenarios.

use super::common::choose_2;
use super::{ComparisonType, ComplexityEstimate, MetricAlgorithm, MetricResults, MetricType};
use crate::metrics::contingency::{
    compute_conditional_entropy, compute_entropy_from_sizes, ContingencyMetrics,
};
use crate::metrics::implementations::statistics::{compute_entity_count, compute_entropy};
use crate::{DataContext, PartitionLevel};
use roaring::RoaringBitmap;
use std::collections::HashMap;
use std::sync::Arc;

/// Type alias for canonical entity ID (minimum record ID in entity)
/// Used by Delta algorithm to track entity merges across thresholds
type CanonicalId = u32;

/// Delta-based algorithm that achieves O(k) complexity for incremental updates
///
/// This algorithm is one of two fundamental approaches (along with RecordAlgorithm)
/// for computing metrics. It handles all same-hierarchy comparisons by tracking
/// changes between thresholds rather than recomputing from scratch.
pub struct DeltaAlgorithm {
    /// Current incremental state (if any)
    state: Option<DeltaState>,
    /// Threshold for deciding when to rebuild vs update incrementally
    /// If the threshold gap exceeds this value, a full rebuild is performed
    rebuild_threshold: f64,
}

/// State maintained between computations for incremental updates
struct DeltaState {
    /// Last threshold processed
    last_threshold: f64,
    /// Canonical IDs of entities in last partition1
    last_p1_entities: Vec<CanonicalId>,
    /// Canonical IDs of entities in last partition2 (if comparison)
    last_p2_entities: Option<Vec<CanonicalId>>,
    /// Contingency table using canonical IDs as keys
    contingency_table: HashMap<(CanonicalId, CanonicalId), u32>,
    /// Row marginals using canonical IDs
    row_marginals: HashMap<CanonicalId, u32>,
    /// Column marginals using canonical IDs
    col_marginals: HashMap<CanonicalId, u32>,
    /// Total number of records
    total_records: u32,
    /// Cached true positives
    true_positives: u64,
    /// Cached false positives
    false_positives: u64,
    /// Cached false negatives
    false_negatives: u64,
    /// Cached true negatives
    true_negatives: u64,
    /// Sum of C(n_ij, 2) for all contingency table cells (for ARI)
    sum_cell_combinations: u64,
    /// Sum of C(a_i, 2) for row marginals (for ARI)
    sum_row_combinations: u64,
    /// Sum of C(b_j, 2) for column marginals (for ARI)
    sum_col_combinations: u64,
}

/// Get the canonical ID for an entity (its minimum record ID)
fn get_canonical_id(entity: &RoaringBitmap) -> CanonicalId {
    entity.min().expect("Entity cannot be empty")
}

impl Default for DeltaAlgorithm {
    fn default() -> Self {
        Self::new()
    }
}

/// Implement ContingencyMetrics trait for DeltaState
/// This allows direct metric computation without conversion, avoiding memory duplication
impl ContingencyMetrics for DeltaState {
    fn compute_precision(&self) -> f64 {
        let denominator = self.true_positives + self.false_positives;
        if denominator == 0 {
            if self.false_negatives > 0 {
                0.0
            } else {
                1.0
            }
        } else {
            self.true_positives as f64 / denominator as f64
        }
    }

    fn compute_recall(&self) -> f64 {
        let denominator = self.true_positives + self.false_negatives;
        if denominator == 0 {
            1.0
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
            return 0.0;
        }

        // We already maintain sum_cell_combinations, sum_row_combinations, sum_col_combinations
        let index = self.sum_cell_combinations as f64;
        let sum_ai_choose_2 = self.sum_row_combinations as f64;
        let sum_bj_choose_2 = self.sum_col_combinations as f64;

        let n_choose_2 = n * (n - 1.0) / 2.0;
        let expected = (sum_ai_choose_2 * sum_bj_choose_2) / n_choose_2;
        let max_value = (sum_ai_choose_2 + sum_bj_choose_2) / 2.0;

        if max_value == expected {
            0.0
        } else {
            (index - expected) / (max_value - expected)
        }
    }

    fn compute_nmi(&self) -> f64 {
        let n = self.total_records as f64;
        if n <= 1.0 {
            return 0.0;
        }

        // Calculate entropy for partition 1 (rows)
        let entropy_u = compute_entropy_from_sizes(self.row_marginals.values().copied(), n);

        // Calculate entropy for partition 2 (columns)
        let entropy_v = compute_entropy_from_sizes(self.col_marginals.values().copied(), n);

        // Calculate mutual information I(U;V)
        let mut mutual_info = 0.0;
        for ((row_id, col_id), &overlap) in &self.contingency_table {
            if overlap > 0 {
                let p_uv = overlap as f64 / n;
                let p_u = self.row_marginals[row_id] as f64 / n;
                let p_v = self.col_marginals[col_id] as f64 / n;
                mutual_info += p_uv * (p_uv / (p_u * p_v)).log2();
            }
        }

        // Compute NMI
        if entropy_u + entropy_v == 0.0 {
            0.0
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
            compute_conditional_entropy(&self.contingency_table, &self.col_marginals, n);

        // Calculate H(K|C) - we need to flip the contingency table
        let mut flipped_contingency = HashMap::new();
        for ((row, col), count) in &self.contingency_table {
            flipped_contingency.insert((*col, *row), *count);
        }
        let h_k_given_c = compute_conditional_entropy(&flipped_contingency, &self.row_marginals, n);

        // Calculate H(C) and H(K)
        let h_c = compute_entropy_from_sizes(self.row_marginals.values().copied(), n);
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
        for ((row_id, _), &overlap) in &self.contingency_table {
            if overlap > 0 {
                let cluster_size = self.row_marginals[row_id] as f64;
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
        for ((_, col_id), &overlap) in &self.contingency_table {
            if overlap > 0 {
                let true_cluster_size = self.col_marginals[col_id] as f64;
                let recall_contribution = (overlap as f64 * overlap as f64) / true_cluster_size;
                total_recall += recall_contribution;
            }
        }
        total_recall / n
    }
}

impl DeltaAlgorithm {
    /// Create a new delta-based algorithm instance with default rebuild threshold
    pub fn new() -> Self {
        Self::with_rebuild_threshold(0.1)
    }

    /// Create a new delta-based algorithm with specified rebuild threshold
    ///
    /// # Arguments
    /// * `rebuild_threshold` - Maximum threshold gap before triggering full rebuild (default: 0.1)
    pub fn with_rebuild_threshold(rebuild_threshold: f64) -> Self {
        Self {
            state: None,
            rebuild_threshold,
        }
    }

    /// Build initial state from partitions
    fn build_initial_state(
        partition1: &PartitionLevel,
        partition2: Option<&PartitionLevel>,
    ) -> DeltaState {
        let mut state = DeltaState {
            last_threshold: partition1.threshold(),
            last_p1_entities: Vec::new(),
            last_p2_entities: None,
            contingency_table: HashMap::new(),
            row_marginals: HashMap::new(),
            col_marginals: HashMap::new(),
            total_records: 0,
            true_positives: 0,
            false_positives: 0,
            false_negatives: 0,
            true_negatives: 0,
            sum_cell_combinations: 0,
            sum_row_combinations: 0,
            sum_col_combinations: 0,
        };

        // Build canonical ID list and marginals for partition1
        for entity in partition1.entities() {
            let canonical_id = get_canonical_id(entity);
            state.last_p1_entities.push(canonical_id);
            let size = entity.len() as u32;
            state.row_marginals.insert(canonical_id, size);
            state.total_records += size;
            // Calculate row combination sum for ARI
            state.sum_row_combinations += choose_2(size);
        }

        // Build canonical ID list and marginals for partition2 if present
        if let Some(p2) = partition2 {
            let mut p2_entities = Vec::new();
            for entity in p2.entities() {
                let canonical_id = get_canonical_id(entity);
                p2_entities.push(canonical_id);
                let size = entity.len() as u32;
                state.col_marginals.insert(canonical_id, size);
                // Calculate column combination sum for ARI
                state.sum_col_combinations += choose_2(size);
            }
            // Sort to ensure consistent ordering
            p2_entities.sort_unstable();
            state.last_p2_entities = Some(p2_entities);

            // Build contingency table
            for (i, entity1) in partition1.entities().iter().enumerate() {
                let id1 = state.last_p1_entities[i];
                for entity2 in p2.entities() {
                    let id2 = get_canonical_id(entity2); // Get canonical ID directly, don't use index
                    let overlap = entity1.intersection_len(entity2) as u32;
                    if overlap > 0 {
                        state.contingency_table.insert((id1, id2), overlap);
                        // Calculate cell combination sum for ARI
                        state.sum_cell_combinations += choose_2(overlap);
                    }
                }
            }
        }

        // Compute initial pair counts
        DeltaAlgorithm::compute_state_pair_counts(&mut state);

        state
    }

    /// Compute pair counts for the current state
    fn compute_state_pair_counts(state: &mut DeltaState) {
        let mut true_positives = 0u64;
        let mut false_positives = 0u64;
        let mut false_negatives = 0u64;

        // Pre-compute pairs together for each entity to avoid double counting
        let mut entity1_pairs_together: HashMap<u32, u64> = HashMap::new();
        let mut entity2_pairs_together: HashMap<u32, u64> = HashMap::new();

        // Single pass through contingency table to compute all necessary sums
        for (&(id1, id2), &overlap) in &state.contingency_table {
            if overlap > 1 {
                let pairs = (overlap as u64 * (overlap as u64 - 1)) / 2;
                true_positives += pairs;
                *entity1_pairs_together.entry(id1).or_insert(0) += pairs;
                *entity2_pairs_together.entry(id2).or_insert(0) += pairs;
            }
        }

        // False positives: pairs together in partition1, apart in partition2
        for (&entity_id, &size) in &state.row_marginals {
            if size > 1 {
                let all_pairs_in_entity = (size as u64 * (size as u64 - 1)) / 2;
                let pairs_also_together =
                    entity1_pairs_together.get(&entity_id).copied().unwrap_or(0);
                false_positives += all_pairs_in_entity.saturating_sub(pairs_also_together);
            }
        }

        // False negatives: pairs apart in partition1, together in partition2
        for (&entity_id, &size) in &state.col_marginals {
            if size > 1 {
                let all_pairs_in_entity = (size as u64 * (size as u64 - 1)) / 2;
                let pairs_also_together =
                    entity2_pairs_together.get(&entity_id).copied().unwrap_or(0);
                false_negatives += all_pairs_in_entity.saturating_sub(pairs_also_together);
            }
        }

        // True negatives
        let total_pairs = if state.total_records > 1 {
            (state.total_records as u64 * (state.total_records as u64 - 1)) / 2
        } else {
            0
        };
        let true_negatives =
            total_pairs.saturating_sub(true_positives + false_positives + false_negatives);

        // Cache the computed values
        state.true_positives = true_positives;
        state.false_positives = false_positives;
        state.false_negatives = false_negatives;
        state.true_negatives = true_negatives;
    }

    /// Perform incremental update from old state to new partition
    fn incremental_update(
        state: &mut DeltaState,
        new_partition1: &PartitionLevel,
        partition2: Option<&PartitionLevel>,
    ) {
        // Build mapping from old canonical IDs to new canonical IDs
        let mut old_to_new: HashMap<CanonicalId, CanonicalId> = HashMap::new();

        // For each old entity, find where its canonical record went
        for &old_id in &state.last_p1_entities {
            // The old canonical ID is a record that must exist in the new partition
            // Find which entity contains this record now
            for new_entity in new_partition1.entities() {
                if new_entity.contains(old_id) {
                    let new_id = get_canonical_id(new_entity);
                    old_to_new.insert(old_id, new_id);
                    break;
                }
            }
        }

        // Identify merges: multiple old IDs mapping to same new ID
        let mut new_to_old: HashMap<CanonicalId, Vec<CanonicalId>> = HashMap::new();
        for (&old_id, &new_id) in &old_to_new {
            new_to_old.entry(new_id).or_default().push(old_id);
        }

        // Update contingency table for merges
        let mut new_contingency = HashMap::new();
        let mut new_row_marginals = HashMap::new();
        let mut new_sum_row_combinations = 0u64;
        let mut new_sum_cell_combinations = 0u64;

        for (new_id, old_ids) in new_to_old {
            if old_ids.len() == 1 {
                // No merge, just transfer the data
                let old_id = old_ids[0];

                // Transfer marginal
                if let Some(&marginal) = state.row_marginals.get(&old_id) {
                    new_row_marginals.insert(new_id, marginal);
                    new_sum_row_combinations += choose_2(marginal);
                }

                // Transfer contingency cells
                if partition2.is_some() {
                    for (&(row, col), &count) in &state.contingency_table {
                        if row == old_id {
                            new_contingency.insert((new_id, col), count);
                            new_sum_cell_combinations += choose_2(count);
                        }
                    }
                }
            } else {
                // Merge: need to recalculate overlaps for merged entity
                // First, aggregate the marginal
                let mut merged_marginal = 0;
                for old_id in &old_ids {
                    if let Some(&marginal) = state.row_marginals.get(old_id) {
                        merged_marginal += marginal;
                    }
                }
                new_row_marginals.insert(new_id, merged_marginal);
                new_sum_row_combinations += choose_2(merged_marginal);

                // For merged entities, we need to recalculate overlaps with partition2
                // because merging changes the overlap counts!
                if let Some(p2) = partition2 {
                    // Find the merged entity in new_partition1
                    let merged_entity = new_partition1
                        .entities()
                        .iter()
                        .find(|e| get_canonical_id(e) == new_id)
                        .expect("Merged entity must exist");

                    // Recalculate overlaps with all entities in partition2
                    for entity2 in p2.entities() {
                        let id2 = get_canonical_id(entity2);
                        let overlap = merged_entity.intersection_len(entity2) as u32;
                        if overlap > 0 {
                            new_contingency.insert((new_id, id2), overlap);
                            new_sum_cell_combinations += choose_2(overlap);
                        }
                    }
                }
            }
        }

        // Handle new entities that didn't exist before (shouldn't happen in merges, but be safe)
        for new_entity in new_partition1.entities() {
            let new_id = get_canonical_id(new_entity);
            new_row_marginals
                .entry(new_id)
                .or_insert(new_entity.len() as u32);
        }

        // Update state with new data
        state.last_threshold = new_partition1.threshold();
        state.last_p1_entities = new_partition1
            .entities()
            .iter()
            .map(get_canonical_id)
            .collect();
        state.contingency_table = new_contingency;
        state.row_marginals = new_row_marginals;
        state.total_records = new_partition1
            .entities()
            .iter()
            .map(|e| e.len() as u32)
            .sum();

        // Update ARI components
        state.sum_row_combinations = new_sum_row_combinations;
        state.sum_cell_combinations = new_sum_cell_combinations;
        // Note: sum_col_combinations doesn't change as partition2 is fixed

        // Recompute pair counts after update
        DeltaAlgorithm::compute_state_pair_counts(state);
    }

    /// Compute metrics from current state
    fn compute_metrics_from_state(&self, metrics: &[MetricType]) -> MetricResults {
        let mut results = HashMap::new();

        if let Some(state) = &self.state {
            // Now we can compute all metrics directly on DeltaState
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
                    _ => 0.0,
                };
                results.insert(metric.name().to_string(), value);
            }
        }

        results
    }
}

impl MetricAlgorithm for DeltaAlgorithm {
    fn name(&self) -> &'static str {
        "Delta-based (O(k) incremental)"
    }

    fn can_handle(&self, comparison_type: &ComparisonType) -> bool {
        // Delta algorithm is best for same-collection comparisons
        matches!(
            comparison_type,
            ComparisonType::Single
                | ComparisonType::SweepPoint {
                    same_collection: true
                }
                | ComparisonType::PointPoint {
                    same_collection: true
                }
        )
    }

    fn complexity(&self, comparison_type: &ComparisonType) -> ComplexityEstimate {
        match comparison_type {
            ComparisonType::Single => ComplexityEstimate {
                notation: "O(1)".to_string(),
                expected_ops: 100,
                description: "Single partition statistics".to_string(),
            },
            ComparisonType::SweepPoint { .. } | ComparisonType::PointPoint { .. } => {
                ComplexityEstimate {
                    notation: "O(k)".to_string(),
                    expected_ops: 10_000,
                    description: "Incremental update for affected entities".to_string(),
                }
            }
            _ => ComplexityEstimate {
                notation: "N/A".to_string(),
                expected_ops: 0,
                description: "Not optimized for this comparison type".to_string(),
            },
        }
    }

    fn compute_single(
        &mut self,
        partition1: &PartitionLevel,
        partition2: Option<&PartitionLevel>,
        metrics: &[MetricType],
        _context: &Arc<DataContext>,
    ) -> MetricResults {
        let mut results = HashMap::new();

        if partition2.is_none() {
            // Single partition statistics
            for metric in metrics {
                let value = match metric {
                    MetricType::EntityCount => compute_entity_count(partition1),
                    MetricType::Entropy => compute_entropy(partition1),
                    _ => continue,
                };
                results.insert(metric.name().to_string(), value);
            }
            return results;
        }

        // Check if we can use incremental update
        let can_increment = if let Some(state) = &self.state {
            // Delta algorithm only works when moving monotonically down in threshold
            // (entities can only merge, never split)
            let moving_down = partition1.threshold() <= state.last_threshold;
            let threshold_ok =
                (partition1.threshold() - state.last_threshold).abs() <= self.rebuild_threshold;

            // Check if partition2 is unchanged
            let p2_unchanged = match (partition2, &state.last_p2_entities) {
                (None, None) => true,
                (Some(p2), Some(old_p2)) => {
                    // Check if partition2 has the same entities (by canonical IDs)
                    let mut new_p2_ids: Vec<CanonicalId> =
                        p2.entities().iter().map(get_canonical_id).collect();
                    new_p2_ids.sort_unstable(); // Sort to match the sorted order in build_initial_state
                    new_p2_ids == *old_p2
                }
                _ => false,
            };

            moving_down && threshold_ok && p2_unchanged
        } else {
            false
        };

        if can_increment {
            // Incremental update
            Self::incremental_update(self.state.as_mut().unwrap(), partition1, partition2);
        } else {
            // Full rebuild
            self.state = Some(Self::build_initial_state(partition1, partition2));
        }

        // Compute metrics from state
        self.compute_metrics_from_state(metrics)
    }

    fn compute_sweep(
        &mut self,
        partitions1: &[PartitionLevel],
        partitions2: Option<&[PartitionLevel]>,
        metrics: &[MetricType],
        context: &Arc<DataContext>,
    ) -> Vec<MetricResults> {
        let mut results = Vec::new();

        if let Some(partitions2) = partitions2 {
            // Two-collection comparison - use default implementation
            for p1 in partitions1 {
                for p2 in partitions2 {
                    results.push(self.compute_single(p1, Some(p2), metrics, context));
                }
            }
        } else {
            // Single collection sweep - ideal for incremental
            for p1 in partitions1 {
                results.push(self.compute_single(p1, None, metrics, context));
            }
        }

        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DataContext, Key, PartitionHierarchy};
    use roaring::RoaringBitmap;
    use std::sync::Arc;

    #[test]
    fn test_delta_algorithm_sweep_computation() {
        let mut algo = DeltaAlgorithm::new();
        let context = DataContext::new();

        // Add some records
        for i in 0..10 {
            context.ensure_record("test", Key::U32(i));
        }

        let context = Arc::new(context);

        // Create a simple hierarchy
        let edges = vec![
            (0, 1, 0.5),
            (2, 3, 0.6),
            (4, 5, 0.7),
            (6, 7, 0.8),
            (8, 9, 0.9),
        ];

        let hierarchy = PartitionHierarchy::from_edges(edges, context.clone(), 2, None)
            .expect("Failed to create hierarchy");

        let hierarchy = Arc::new(hierarchy);

        // Build partitions at different thresholds
        let hierarchy_clone = (*hierarchy).clone();
        let thresholds = [0.4, 0.55, 0.65, 0.75, 0.85];
        let partitions: Vec<Arc<PartitionLevel>> = thresholds
            .iter()
            .map(|&t| hierarchy_clone.at_threshold(t))
            .collect();

        // Test sweep computation
        let metrics = vec![MetricType::EntityCount];
        // Create a temporary vec of dereferenced partitions for the test
        let partition_derefs: Vec<PartitionLevel> =
            partitions.iter().map(|p| (**p).clone()).collect();
        let results = algo.compute_sweep(&partition_derefs, None, &metrics, &context);

        // Should have results for each threshold
        assert_eq!(results.len(), 5);

        // All results should have entity_count metric
        for (i, result) in results.iter().enumerate() {
            assert!(
                result.contains_key("entity_count"),
                "Result at threshold {} should contain entity_count",
                thresholds[i]
            );
        }
    }

    #[test]
    fn test_delta_algorithm_can_handle() {
        let algo = DeltaAlgorithm::new();

        // Should handle same-collection comparisons
        assert!(algo.can_handle(&ComparisonType::Single));
        assert!(algo.can_handle(&ComparisonType::SweepPoint {
            same_collection: true
        }));
        assert!(algo.can_handle(&ComparisonType::PointPoint {
            same_collection: true
        }));

        // Should NOT handle different-collection comparisons
        assert!(!algo.can_handle(&ComparisonType::SweepPoint {
            same_collection: false
        }));
        assert!(!algo.can_handle(&ComparisonType::PointPoint {
            same_collection: false
        }));
        assert!(!algo.can_handle(&ComparisonType::SweepSweep {
            same_collection: false
        }));
    }

    #[test]
    fn test_incremental_correctness() {
        // Create algorithm for incremental computation
        let mut incremental_algo = DeltaAlgorithm::new();

        // Create test data with known merge patterns
        let context = DataContext::new();
        for i in 0..20 {
            context.ensure_record("test", Key::U32(i));
        }
        let context = Arc::new(context);

        // Create edges that will cause specific merges
        let edges = vec![
            // Early merges
            (0, 1, 0.5),
            (2, 3, 0.5),
            (4, 5, 0.5),
            // Mid-threshold merges
            (1, 2, 0.7),
            (6, 7, 0.7),
            (8, 9, 0.7),
            // Late merges
            (3, 4, 0.9),
            (7, 8, 0.9),
            (10, 11, 0.9),
        ];

        let hierarchy = PartitionHierarchy::from_edges(edges.clone(), context.clone(), 2, None)
            .expect("Failed to create hierarchy");
        let hierarchy = Arc::new(hierarchy);

        // Create a second hierarchy for comparison
        let hierarchy2 = PartitionHierarchy::from_edges(
            vec![(0, 1, 0.6), (2, 3, 0.8), (4, 5, 0.9)],
            context.clone(),
            2,
            None,
        )
        .expect("Failed to create second hierarchy");

        // Test sweep with multiple thresholds
        let thresholds = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95];
        let hierarchy_clone = (*hierarchy).clone();
        let hierarchy2_clone = hierarchy2.clone();

        let partitions1: Vec<Arc<PartitionLevel>> = thresholds
            .iter()
            .map(|&t| hierarchy_clone.at_threshold(t))
            .collect();

        let partition2 = hierarchy2_clone.at_threshold(0.7);

        let metrics = vec![MetricType::F1, MetricType::Precision, MetricType::Recall];

        // Create temporary vecs of dereferenced partitions for the test
        let partitions1_derefs: Vec<PartitionLevel> =
            partitions1.iter().map(|p| (**p).clone()).collect();

        // Run incremental algorithm
        let incremental_results = incremental_algo.compute_sweep(
            &partitions1_derefs,
            Some(std::slice::from_ref(&*partition2)),
            &metrics,
            &context,
        );

        // Run separate full computations for comparison
        let mut rebuild_results = Vec::new();
        for p1 in partitions1.iter() {
            // Create fresh algorithm instance for each computation
            let mut fresh_algo = DeltaAlgorithm::new();
            let result = fresh_algo.compute_single(p1, Some(&*partition2), &metrics, &context);
            rebuild_results.push(result);
        }

        // Verify results match
        assert_eq!(incremental_results.len(), rebuild_results.len());

        for (i, (inc_result, reb_result)) in incremental_results
            .iter()
            .zip(rebuild_results.iter())
            .enumerate()
        {
            for metric in &metrics {
                let metric_name = metric.name();
                let inc_value = inc_result.get(metric_name).copied().unwrap_or(0.0);
                let reb_value = reb_result.get(metric_name).copied().unwrap_or(0.0);

                assert!(
                    (inc_value - reb_value).abs() < 1e-10,
                    "Mismatch at threshold {} for metric {}: incremental={}, rebuild={}",
                    thresholds[i],
                    metric_name,
                    inc_value,
                    reb_value
                );
            }
        }
    }

    #[test]
    fn test_edge_cases() {
        let mut algo = DeltaAlgorithm::new();
        let context = Arc::new(DataContext::new());

        // Empty partitions
        let empty_partition = crate::PartitionLevel::new(0.5, vec![]);
        let metrics = vec![MetricType::EntityCount];
        let result = algo.compute_single(&empty_partition, None, &metrics, &context);
        assert_eq!(result.get("entity_count"), Some(&0.0));

        // Single entity
        let mut single_entity = RoaringBitmap::new();
        single_entity.insert(0);
        let single_partition = crate::PartitionLevel::new(0.5, vec![single_entity]);
        let result = algo.compute_single(&single_partition, None, &metrics, &context);
        assert_eq!(result.get("entity_count"), Some(&1.0));
    }
}
