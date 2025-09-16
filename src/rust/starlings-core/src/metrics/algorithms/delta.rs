//! Delta algorithm for O(k) incremental metric computation
//!
//! This algorithm exploits the incremental nature of threshold changes to update
//! contingency tables efficiently when moving between adjacent thresholds in the
//! same hierarchy. Instead of recomputing everything, it tracks entity evolution
//! through merges and updates only affected cells.

use super::{ComparisonType, ComplexityEstimate, MetricAlgorithm, MetricResults, MetricType};
use crate::metrics::implementations::statistics::{compute_entity_count, compute_entropy};
use crate::{DataContext, PartitionLevel};
use roaring::RoaringBitmap;
use std::collections::HashMap;
use std::sync::Arc;

/// Delta-based algorithm that achieves O(k) complexity for incremental updates
pub struct DeltaAlgorithm {
    /// Current incremental state (if any)
    state: Option<IncrementalState>,
    /// Threshold for deciding when to rebuild vs update incrementally
    rebuild_threshold: f64,
}

/// State maintained between computations for incremental updates
struct IncrementalState {
    /// Last threshold processed
    last_threshold: f64,
    /// Canonical IDs of entities in last partition1
    last_p1_entities: Vec<u32>,
    /// Canonical IDs of entities in last partition2 (if comparison)
    last_p2_entities: Option<Vec<u32>>,
    /// Contingency table using canonical IDs as keys
    contingency_table: HashMap<(u32, u32), u32>,
    /// Row marginals using canonical IDs
    row_marginals: HashMap<u32, u32>,
    /// Column marginals using canonical IDs
    col_marginals: HashMap<u32, u32>,
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
}

/// Get the canonical ID for an entity (its minimum record ID)
fn get_canonical_id(entity: &RoaringBitmap) -> u32 {
    entity.min().expect("Entity cannot be empty")
}

impl Default for DeltaAlgorithm {
    fn default() -> Self {
        Self::new()
    }
}

impl DeltaAlgorithm {
    /// Create a new delta-based algorithm instance
    pub fn new() -> Self {
        Self {
            state: None,
            rebuild_threshold: 0.1, // Rebuild if threshold gap > 0.1
        }
    }

    /// Build initial state from partitions
    fn build_initial_state(
        partition1: &PartitionLevel,
        partition2: Option<&PartitionLevel>,
    ) -> IncrementalState {
        let mut state = IncrementalState {
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
        };

        // Build canonical ID list and marginals for partition1
        for entity in partition1.entities() {
            let canonical_id = get_canonical_id(entity);
            state.last_p1_entities.push(canonical_id);
            state
                .row_marginals
                .insert(canonical_id, entity.len() as u32);
            state.total_records += entity.len() as u32;
        }

        // Build canonical ID list and marginals for partition2 if present
        if let Some(p2) = partition2 {
            let mut p2_entities = Vec::new();
            for entity in p2.entities() {
                let canonical_id = get_canonical_id(entity);
                p2_entities.push(canonical_id);
                state
                    .col_marginals
                    .insert(canonical_id, entity.len() as u32);
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
                    }
                }
            }
        }

        // Compute initial pair counts
        DeltaAlgorithm::compute_state_pair_counts(&mut state);

        state
    }

    /// Compute pair counts for the current state
    fn compute_state_pair_counts(state: &mut IncrementalState) {
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
        state: &mut IncrementalState,
        new_partition1: &PartitionLevel,
        partition2: Option<&PartitionLevel>,
    ) {
        // Build mapping from old canonical IDs to new canonical IDs
        let mut old_to_new: HashMap<u32, u32> = HashMap::new();

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
        let mut new_to_old: HashMap<u32, Vec<u32>> = HashMap::new();
        for (&old_id, &new_id) in &old_to_new {
            new_to_old.entry(new_id).or_default().push(old_id);
        }

        // Update contingency table for merges
        let mut new_contingency = HashMap::new();
        let mut new_row_marginals = HashMap::new();

        for (new_id, old_ids) in new_to_old {
            if old_ids.len() == 1 {
                // No merge, just transfer the data
                let old_id = old_ids[0];

                // Transfer marginal
                if let Some(&marginal) = state.row_marginals.get(&old_id) {
                    new_row_marginals.insert(new_id, marginal);
                }

                // Transfer contingency cells
                if partition2.is_some() {
                    for (&(row, col), &count) in &state.contingency_table {
                        if row == old_id {
                            new_contingency.insert((new_id, col), count);
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

        // Recompute pair counts after update
        DeltaAlgorithm::compute_state_pair_counts(state);
    }

    /// Compute metrics from current state
    fn compute_metrics_from_state(&self, metrics: &[MetricType]) -> MetricResults {
        let mut results = HashMap::new();

        if let Some(state) = &self.state {
            for metric in metrics {
                let value = match metric {
                    MetricType::F1 => self.compute_f1_from_state(state),
                    MetricType::Precision => self.compute_precision_from_state(state),
                    MetricType::Recall => self.compute_recall_from_state(state),
                    // TODO: Implement other metrics
                    MetricType::ARI => 0.0,
                    MetricType::NMI => 0.0,
                    MetricType::VMeasure => 0.0,
                    MetricType::BCubedPrecision => 0.0,
                    MetricType::BCubedRecall => 0.0,
                    _ => 0.0,
                };
                results.insert(metric.name().to_string(), value);
            }
        }

        results
    }

    /// Compute precision directly from state
    fn compute_precision_from_state(&self, state: &IncrementalState) -> f64 {
        let denominator = state.true_positives + state.false_positives;
        if denominator == 0 {
            if state.false_negatives > 0 {
                0.0
            } else {
                1.0
            }
        } else {
            state.true_positives as f64 / denominator as f64
        }
    }

    /// Compute recall directly from state
    fn compute_recall_from_state(&self, state: &IncrementalState) -> f64 {
        let denominator = state.true_positives + state.false_negatives;
        if denominator == 0 {
            1.0
        } else {
            state.true_positives as f64 / denominator as f64
        }
    }

    /// Compute F1 score directly from state
    fn compute_f1_from_state(&self, state: &IncrementalState) -> f64 {
        let precision = self.compute_precision_from_state(state);
        let recall = self.compute_recall_from_state(state);

        if precision + recall == 0.0 {
            0.0
        } else {
            2.0 * (precision * recall) / (precision + recall)
        }
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
                    let mut new_p2_ids: Vec<u32> =
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
        let mut hierarchy_mut = (*hierarchy).clone();
        let thresholds = [0.4, 0.55, 0.65, 0.75, 0.85];
        let partitions: Vec<_> = thresholds
            .iter()
            .map(|&t| hierarchy_mut.at_threshold(t).clone())
            .collect();

        // Test sweep computation
        let metrics = vec![MetricType::EntityCount];
        let results = algo.compute_sweep(&partitions, None, &metrics, &context);

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
        let mut hierarchy_mut = (*hierarchy).clone();
        let mut hierarchy2_mut = hierarchy2.clone();

        let partitions1: Vec<_> = thresholds
            .iter()
            .map(|&t| hierarchy_mut.at_threshold(t).clone())
            .collect();

        let partition2 = hierarchy2_mut.at_threshold(0.7);

        let metrics = vec![MetricType::F1, MetricType::Precision, MetricType::Recall];

        // Run incremental algorithm
        let incremental_results = incremental_algo.compute_sweep(
            &partitions1,
            Some(std::slice::from_ref(partition2)),
            &metrics,
            &context,
        );

        // Run separate full computations for comparison
        let mut rebuild_results = Vec::new();
        for p1 in partitions1.iter() {
            // Create fresh algorithm instance for each computation
            let mut fresh_algo = DeltaAlgorithm::new();
            let result = fresh_algo.compute_single(p1, Some(partition2), &metrics, &context);
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
