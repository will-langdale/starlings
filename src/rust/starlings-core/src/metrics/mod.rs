//! High-performance metric computation engine with pluggable algorithms
//!
//! This module provides a unified interface for computing entity resolution metrics
//! using different algorithmic strategies optimised for specific comparison types.

pub mod algorithms;
pub mod implementations;
pub mod types;

use crate::{DataContext, PartitionHierarchy, PartitionLevel};
use algorithms::{
    ComparisonType, DeltaAlgorithm, MetricAlgorithm, MetricResults, MetricType, RecordAlgorithm,
};
use std::sync::Arc;

pub use algorithms::MetricType as CoreMetricType;

/// Main engine for metric computation with automatic algorithm selection
pub struct MetricEngine {
    /// Available algorithms in priority order
    algorithms: Vec<Box<dyn MetricAlgorithm>>,
    /// Enable debug logging
    debug: bool,
}

impl MetricEngine {
    /// Create a new metric engine with default algorithms
    pub fn new() -> Self {
        Self {
            algorithms: vec![
                Box::new(DeltaAlgorithm::new()),  // Priority 1: Incremental
                Box::new(RecordAlgorithm::new()), // Priority 2: Single-pass
            ],
            debug: std::env::var("STARLINGS_DEBUG").is_ok(),
        }
    }

    /// Select the best algorithm for a given comparison type (returns index)
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
        let comparison_type = if partition2.is_none() {
            ComparisonType::Single
        } else {
            ComparisonType::PointPoint {
                same_collection: false, // Will be determined by caller
            }
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
}
