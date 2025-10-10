//! High-performance metric computation engine with pluggable algorithms
//!
//! This module provides a unified interface for computing entity resolution metrics
//! using different algorithmic strategies optimised for specific comparison types.

pub mod algorithms;
pub mod contingency;
pub mod implementations;
pub mod types;

use crate::{DataContext, PartitionHierarchy, PartitionLevel};
use algorithms::{ComparisonType, DeltaAlgorithm, MetricAlgorithm, MetricType, RecordAlgorithm};
use std::sync::Arc;

pub use algorithms::{MetricResults, MetricType as CoreMetricType};

/// Main engine for metric computation with automatic algorithm selection
///
/// The engine uses exactly two algorithms that exhaustively cover all scenarios:
/// - **Delta**: For same-hierarchy threshold sweeps (incremental O(k) updates)
/// - **Record**: For cross-collection comparisons (single-pass O(r) iteration)
pub struct MetricEngine {
    /// Available algorithms in priority order
    /// Priority 1: Delta (for same-hierarchy)
    /// Priority 2: Record (for everything else)
    algorithms: Vec<Box<dyn MetricAlgorithm>>,
    /// Enable debug logging
    debug: bool,
}

impl MetricEngine {
    /// Create a new metric engine with the two fundamental algorithms
    pub fn new() -> Self {
        Self {
            algorithms: vec![
                Box::new(DeltaAlgorithm::new()), // Priority 1: Incremental O(k) for same-hierarchy
                Box::new(RecordAlgorithm::new()), // Priority 2: Single-pass O(r) for cross-collection
            ],
            // These two algorithms exhaustively partition the problem space:
            // - Delta handles temporal locality (threshold changes)
            // - Record handles spatial locality (record iteration)
            // No additional algorithms are needed or beneficial.
            debug: std::env::var("STARLINGS_DEBUG").is_ok(),
        }
    }

    /// Select the best algorithm for a given comparison type (returns index)
    ///
    /// Algorithm selection is deterministic:
    /// - Delta handles same-hierarchy comparisons
    /// - Record handles cross-collection comparisons
    ///   Exactly one algorithm will handle each comparison type.
    fn select_algorithm_index(&self, comparison_type: &ComparisonType) -> usize {
        // Find first algorithm that can handle this comparison
        for (i, algo) in self.algorithms.iter().enumerate() {
            if algo.can_handle(comparison_type) {
                if self.debug {
                    let complexity = algo.complexity(comparison_type);
                    eprintln!(
                        "Selected {} for {:?} (complexity: {}, ~{} ops)",
                        algo.name(),
                        comparison_type,
                        complexity.notation,
                        complexity.expected_ops
                    );
                }
                return i;
            }
        }

        panic!(
            "No algorithm can handle comparison type: {:?}",
            comparison_type
        );
    }

    /// Compute metrics for a single comparison
    pub fn compute_single(
        &mut self,
        partition1: &PartitionLevel,
        partition2: Option<&PartitionLevel>,
        metrics: &[MetricType],
        context: &Arc<DataContext>,
    ) -> MetricResults {
        self.compute_single_with_flag(partition1, partition2, metrics, context, false)
    }

    /// Compute metrics for a single comparison with explicit same_collection flag
    pub fn compute_single_with_flag(
        &mut self,
        partition1: &PartitionLevel,
        partition2: Option<&PartitionLevel>,
        metrics: &[MetricType],
        context: &Arc<DataContext>,
        same_collection: bool,
    ) -> MetricResults {
        let comparison_type = if partition2.is_none() {
            ComparisonType::Single
        } else {
            ComparisonType::PointPoint { same_collection }
        };

        let algo_index = self.select_algorithm_index(&comparison_type);
        let algorithm = &mut self.algorithms[algo_index];
        algorithm.compute_single(partition1, partition2, metrics, context)
    }

    /// Compute metrics for a sweep comparison
    pub fn compute_sweep(
        &mut self,
        partitions1: &[PartitionLevel],
        partitions2: Option<&[PartitionLevel]>,
        metrics: &[MetricType],
        context: &Arc<DataContext>,
        same_collection: bool,
    ) -> Vec<MetricResults> {
        let comparison_type = match (partitions1.len(), partitions2.map(|p| p.len())) {
            (1, Some(1)) => ComparisonType::PointPoint { same_collection },
            (1, None) => ComparisonType::Single,
            (_, Some(1)) | (_, None) => ComparisonType::SweepPoint { same_collection },
            (_, Some(_)) => ComparisonType::SweepSweep { same_collection },
        };

        if self.debug {
            eprintln!(
                "Computing {} metrics for {} × {} comparisons",
                metrics.len(),
                partitions1.len(),
                partitions2.map(|p| p.len()).unwrap_or(1)
            );
        }

        let algo_index = self.select_algorithm_index(&comparison_type);
        let algorithm = &mut self.algorithms[algo_index];
        algorithm.compute_sweep(partitions1, partitions2, metrics, context)
    }

    /// Compute metrics for a single comparison using Arc references
    pub fn compute_single_arc(
        &mut self,
        partition1: &Arc<PartitionLevel>,
        partition2: Option<&Arc<PartitionLevel>>,
        metrics: &[MetricType],
        context: &Arc<DataContext>,
        same_collection: bool,
    ) -> MetricResults {
        let comparison_type = if partition2.is_none() {
            ComparisonType::Single
        } else {
            ComparisonType::PointPoint { same_collection }
        };

        let algo_index = self.select_algorithm_index(&comparison_type);
        let algorithm = &mut self.algorithms[algo_index];
        algorithm.compute_single_arc(partition1, partition2, metrics, context)
    }

    /// Compute metrics for a sweep comparison using Arc references
    pub fn compute_sweep_arc(
        &mut self,
        partitions1: &[Arc<PartitionLevel>],
        partitions2: Option<&[Arc<PartitionLevel>]>,
        metrics: &[MetricType],
        context: &Arc<DataContext>,
        same_collection: bool,
    ) -> Vec<MetricResults> {
        let comparison_type = match (partitions1.len(), partitions2.map(|p| p.len())) {
            (1, Some(1)) => ComparisonType::PointPoint { same_collection },
            (1, None) => ComparisonType::Single,
            (_, Some(1)) | (_, None) => ComparisonType::SweepPoint { same_collection },
            (_, Some(_)) => ComparisonType::SweepSweep { same_collection },
        };

        if self.debug {
            eprintln!(
                "Computing {} metrics for {} × {} comparisons",
                metrics.len(),
                partitions1.len(),
                partitions2.map(|p| p.len()).unwrap_or(1)
            );
        }

        let algo_index = self.select_algorithm_index(&comparison_type);
        let algorithm = &mut self.algorithms[algo_index];
        algorithm.compute_sweep_arc(partitions1, partitions2, metrics, context)
    }

    /// Optimised single-collection sweep using only merge events (no partition building)
    /// This method builds ONLY the first partition and uses merge events for all others
    pub fn compute_single_sweep_with_merges(
        &mut self,
        first_partition: &Arc<PartitionLevel>,
        thresholds: &[f64],
        merge_events_between: &[Vec<crate::hierarchy::MergeEvent>],
        metrics: &[MetricType],
        context: &Arc<DataContext>,
    ) -> Vec<MetricResults> {
        use algorithms::delta::DeltaAlgorithm;

        if self.debug {
            eprintln!(
                "Computing {} metrics for single-collection sweep with {} thresholds using merge events",
                metrics.len(),
                thresholds.len()
            );
        }

        // For single collection sweeps, use the Delta algorithm with merge events
        // This achieves TRUE O(k) complexity by never building intermediate partitions
        if let Some(algorithm) = self.algorithms.get_mut(0) {
            if let Some(delta) = algorithm.as_any_mut().downcast_mut::<DeltaAlgorithm>() {
                return delta.compute_sweep_with_merges(
                    first_partition,
                    thresholds,
                    merge_events_between,
                    metrics,
                    context,
                );
            }
        }

        // This should never happen - Delta algorithm is always available
        panic!("Delta algorithm not available - this indicates a critical engine misconfiguration")
    }
}

impl Default for MetricEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Request for metric computation
pub struct MetricRequest {
    pub partitions1: Vec<PartitionLevel>,
    pub partitions2: Option<Vec<PartitionLevel>>,
    pub metrics: Vec<MetricType>,
    pub context: Arc<DataContext>,
    pub same_collection: bool,
}

impl MetricRequest {
    /// Determine the comparison type for this request
    pub fn comparison_type(&self) -> ComparisonType {
        match (
            self.partitions1.len(),
            self.partitions2.as_ref().map(|p| p.len()),
        ) {
            (1, Some(1)) => ComparisonType::PointPoint {
                same_collection: self.same_collection,
            },
            (1, None) => ComparisonType::Single,
            (_, Some(1)) | (_, None) => ComparisonType::SweepPoint {
                same_collection: self.same_collection,
            },
            (_, Some(_)) => ComparisonType::SweepSweep {
                same_collection: self.same_collection,
            },
        }
    }
}

/// Helper function to build partitions for a set of thresholds
pub fn build_partitions_for_thresholds(
    hierarchy: &PartitionHierarchy,
    thresholds: &[f64],
) -> Vec<Arc<PartitionLevel>> {
    // Use incremental building for multiple thresholds
    hierarchy.build_partitions_incrementally(thresholds)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Key;
    use roaring::RoaringBitmap;

    #[test]
    fn test_algorithm_selection() {
        let engine = MetricEngine::new();

        // Delta should handle same-collection sweep × point
        let algo_idx = engine.select_algorithm_index(&ComparisonType::SweepPoint {
            same_collection: true,
        });
        assert_eq!(
            engine.algorithms[algo_idx].name(),
            "Delta-based (O(k) incremental)"
        );

        // Record should handle sweep × sweep
        let algo_idx = engine.select_algorithm_index(&ComparisonType::SweepSweep {
            same_collection: false,
        });
        assert_eq!(
            engine.algorithms[algo_idx].name(),
            "Record-based (O(r) single-pass)"
        );
    }

    #[test]
    fn test_algorithms_are_exhaustive() {
        // This test proves that Delta and Record algorithms exhaustively cover
        // all possible comparison types with no gaps or overlaps.

        let delta = DeltaAlgorithm::new();
        let record = RecordAlgorithm::new();

        // All possible comparison types
        let all_comparisons = vec![
            ComparisonType::Single,
            ComparisonType::PointPoint {
                same_collection: true,
            },
            ComparisonType::PointPoint {
                same_collection: false,
            },
            ComparisonType::SweepPoint {
                same_collection: true,
            },
            ComparisonType::SweepPoint {
                same_collection: false,
            },
            ComparisonType::SweepSweep {
                same_collection: true,
            },
            ComparisonType::SweepSweep {
                same_collection: false,
            },
        ];

        for comparison_type in all_comparisons {
            let delta_handles = delta.can_handle(&comparison_type);
            let record_handles = record.can_handle(&comparison_type);

            // Exactly one algorithm must handle each comparison type
            assert!(
                delta_handles || record_handles,
                "No algorithm handles {:?}",
                comparison_type
            );

            // Verify no overlap - only one algorithm should handle each type
            assert!(
                !(delta_handles && record_handles),
                "Both algorithms handle {:?} - this is an overlap!",
                comparison_type
            );

            // Document which algorithm handles what
            if delta_handles {
                match comparison_type {
                    ComparisonType::Single
                    | ComparisonType::SweepPoint {
                        same_collection: true,
                    }
                    | ComparisonType::PointPoint {
                        same_collection: true,
                    } => {
                        // Delta correctly handles same-hierarchy comparisons
                    }
                    _ => panic!(
                        "Delta handles unexpected comparison type: {:?}",
                        comparison_type
                    ),
                }
            } else {
                match comparison_type {
                    ComparisonType::SweepSweep { .. }
                    | ComparisonType::PointPoint {
                        same_collection: false,
                    }
                    | ComparisonType::SweepPoint {
                        same_collection: false,
                    } => {
                        // Record correctly handles cross-collection comparisons
                    }
                    _ => panic!(
                        "Record handles unexpected comparison type: {:?}",
                        comparison_type
                    ),
                }
            }
        }
    }

    #[test]
    fn test_algorithm_partitioning_is_complete() {
        // This test mathematically proves that our two algorithms form a complete
        // partition of the problem space.

        let engine = MetricEngine::new();

        // Test that we can handle all comparison scenarios
        let test_scenarios = vec![
            ("Single partition analysis", ComparisonType::Single),
            (
                "Same hierarchy point-point",
                ComparisonType::PointPoint {
                    same_collection: true,
                },
            ),
            (
                "Different collections point-point",
                ComparisonType::PointPoint {
                    same_collection: false,
                },
            ),
            (
                "Same hierarchy threshold sweep",
                ComparisonType::SweepPoint {
                    same_collection: true,
                },
            ),
            (
                "Cross-collection sweep",
                ComparisonType::SweepPoint {
                    same_collection: false,
                },
            ),
            (
                "Same hierarchy sweep×sweep",
                ComparisonType::SweepSweep {
                    same_collection: true,
                },
            ),
            (
                "Cross-collection sweep×sweep",
                ComparisonType::SweepSweep {
                    same_collection: false,
                },
            ),
        ];

        for (scenario, comparison_type) in test_scenarios {
            // This will panic if no algorithm can handle the comparison type
            let algo_idx = engine.select_algorithm_index(&comparison_type);
            assert!(
                algo_idx < engine.algorithms.len(),
                "Failed to find algorithm for scenario: {}",
                scenario
            );

            // Verify the selected algorithm actually handles this type
            assert!(
                engine.algorithms[algo_idx].can_handle(&comparison_type),
                "Selected algorithm doesn't handle scenario: {}",
                scenario
            );
        }
    }

    #[test]
    fn test_delta_algorithm_basic() {
        let mut algo = DeltaAlgorithm::new();
        let context = Arc::new(DataContext::new());

        // Create simple partition
        let mut entities = Vec::new();
        let mut entity1 = RoaringBitmap::new();
        entity1.insert(0);
        entity1.insert(1);
        entities.push(entity1);

        let partition = PartitionLevel::new(0.5, entities);

        // Compute single partition metrics
        let metrics = vec![MetricType::EntityCount, MetricType::Entropy];
        let results = algo.compute_single(&partition, None, &metrics, &context);

        assert_eq!(results.get("entity_count"), Some(&1.0));
        assert!(results.contains_key("entropy"));
    }

    #[test]
    fn test_record_algorithm_basic() {
        let mut algo = RecordAlgorithm::new();
        let context = DataContext::new();

        // Add some records
        context.ensure_record("test", Key::U32(0));
        context.ensure_record("test", Key::U32(1));
        context.ensure_record("test", Key::U32(2));
        let context = Arc::new(context);

        // Create two partitions
        let mut entities1 = Vec::new();
        let mut entity1 = RoaringBitmap::new();
        entity1.insert(0);
        entity1.insert(1);
        entities1.push(entity1);

        let mut entities2 = Vec::new();
        let mut entity2 = RoaringBitmap::new();
        entity2.insert(0);
        entities2.push(entity2);
        let mut entity3 = RoaringBitmap::new();
        entity3.insert(1);
        entities2.push(entity3);

        let partition1 = PartitionLevel::new(0.8, entities1);
        let partition2 = PartitionLevel::new(0.9, entities2);

        // Compute comparison metrics
        let metrics = vec![MetricType::F1, MetricType::Precision, MetricType::Recall];
        let results = algo.compute_single(&partition1, Some(&partition2), &metrics, &context);

        // Should have computed metrics
        assert!(results.contains_key("f1"));
        assert!(results.contains_key("precision"));
        assert!(results.contains_key("recall"));
    }

    #[test]
    fn test_ari_computation() {
        let mut engine = MetricEngine::new();
        let context = DataContext::new();

        // Add records 0-5
        for i in 0..6 {
            context.ensure_record("test", Key::U32(i));
        }
        let context = Arc::new(context);

        // Create two partitions with known ARI
        // Partition1: {0,1}, {2,3}, {4,5}
        let mut entities1 = Vec::new();
        let mut e1 = RoaringBitmap::new();
        e1.insert(0);
        e1.insert(1);
        entities1.push(e1);
        let mut e2 = RoaringBitmap::new();
        e2.insert(2);
        e2.insert(3);
        entities1.push(e2);
        let mut e3 = RoaringBitmap::new();
        e3.insert(4);
        e3.insert(5);
        entities1.push(e3);

        // Partition2: {0,1}, {2,3}, {4,5} (identical, ARI should be 1.0)
        let entities2 = entities1.clone();

        let partition1 = PartitionLevel::new(0.8, entities1);
        let partition2 = PartitionLevel::new(0.9, entities2);

        // Test ARI computation
        let metrics = vec![MetricType::ARI];
        let results = engine.compute_single(&partition1, Some(&partition2), &metrics, &context);

        assert!(results.contains_key("ari"));
        let ari = results.get("ari").unwrap();
        assert!(
            (ari - 1.0).abs() < 1e-10,
            "ARI for identical partitions should be 1.0, got {}",
            ari
        );
    }

    #[test]
    fn test_ari_with_different_partitions() {
        let mut engine = MetricEngine::new();
        let context = DataContext::new();

        // Add records 0-5
        for i in 0..6 {
            context.ensure_record("test", Key::U32(i));
        }
        let context = Arc::new(context);

        // Create two different partitions
        // Partition1: {0,1,2}, {3,4,5}
        let mut entities1 = Vec::new();
        let mut e1 = RoaringBitmap::new();
        e1.insert(0);
        e1.insert(1);
        e1.insert(2);
        entities1.push(e1);
        let mut e2 = RoaringBitmap::new();
        e2.insert(3);
        e2.insert(4);
        e2.insert(5);
        entities1.push(e2);

        // Partition2: {0,1}, {2,3}, {4,5}
        let mut entities2 = Vec::new();
        let mut e3 = RoaringBitmap::new();
        e3.insert(0);
        e3.insert(1);
        entities2.push(e3);
        let mut e4 = RoaringBitmap::new();
        e4.insert(2);
        e4.insert(3);
        entities2.push(e4);
        let mut e5 = RoaringBitmap::new();
        e5.insert(4);
        e5.insert(5);
        entities2.push(e5);

        let partition1 = PartitionLevel::new(0.8, entities1);
        let partition2 = PartitionLevel::new(0.9, entities2);

        // Test ARI computation
        let metrics = vec![MetricType::ARI, MetricType::Precision, MetricType::Recall];
        let results = engine.compute_single(&partition1, Some(&partition2), &metrics, &context);

        assert!(results.contains_key("ari"));
        let ari = results.get("ari").unwrap();
        // ARI should be between -1 and 1, and for partial overlap should be between 0 and 1
        assert!(
            (-1.0..=1.0).contains(ari),
            "ARI should be in [-1, 1], got {}",
            ari
        );
        assert!(
            *ari > 0.0 && *ari < 1.0,
            "ARI for partial overlap should be between 0 and 1, got {}",
            ari
        );
    }
}
