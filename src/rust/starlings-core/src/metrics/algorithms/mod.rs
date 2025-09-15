//! Core trait and types for metric computation algorithms
//!
//! This module defines the fundamental abstraction for metric algorithms,
//! allowing different computational strategies (delta, record, etc.) to be
//! implemented as pluggable components with clear interfaces.

use crate::{DataContext, PartitionLevel};
use std::collections::HashMap;
use std::sync::Arc;

/// Type alias for metric results
pub type MetricResults = HashMap<String, f64>;

/// Core trait that all metric computation algorithms must implement
pub trait MetricAlgorithm: Send {
    /// Human-readable name for this algorithm (used in logging)
    fn name(&self) -> &'static str;

    /// Check if this algorithm can handle the given comparison type
    fn can_handle(&self, comparison_type: &ComparisonType) -> bool;

    /// Estimate the computational complexity for this comparison
    fn complexity(&self, comparison_type: &ComparisonType) -> ComplexityEstimate;

    /// Compute metrics for a single partition comparison
    fn compute_single(
        &mut self,
        partition1: &PartitionLevel,
        partition2: Option<&PartitionLevel>,
        metrics: &[MetricType],
        context: &Arc<DataContext>,
    ) -> MetricResults;

    /// Compute metrics for a sweep comparison (with potential optimisations)
    ///
    /// The default implementation calls compute_single for each pair,
    /// but algorithms can override this for better performance.
    fn compute_sweep(
        &mut self,
        partitions1: &[PartitionLevel],
        partitions2: Option<&[PartitionLevel]>,
        metrics: &[MetricType],
        context: &Arc<DataContext>,
    ) -> Vec<MetricResults> {
        let mut results = Vec::new();

        if let Some(partitions2) = partitions2 {
            // Two-collection comparison
            for p1 in partitions1 {
                for p2 in partitions2 {
                    results.push(self.compute_single(p1, Some(p2), metrics, context));
                }
            }
        } else {
            // Single-collection analysis
            for p1 in partitions1 {
                results.push(self.compute_single(p1, None, metrics, context));
            }
        }

        results
    }
}

/// Types of comparisons that can be performed
#[derive(Debug, Clone, PartialEq)]
pub enum ComparisonType {
    /// Single partition statistics (no comparison)
    Single,
    /// Point-to-point comparison
    PointPoint { same_collection: bool },
    /// Sweep against a single point
    SweepPoint { same_collection: bool },
    /// Sweep against sweep (cartesian product)
    SweepSweep { same_collection: bool },
}

/// Complexity estimate for an algorithm on a given comparison
#[derive(Debug, Clone)]
pub struct ComplexityEstimate {
    /// Big-O notation (e.g., "O(r)", "O(k)", "O(k₁ × k₂)")
    pub notation: String,
    /// Estimated number of operations
    pub expected_ops: u64,
    /// Human-readable description
    pub description: String,
}

/// Metric types that can be computed
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum MetricType {
    // Evaluation metrics (require comparison)
    F1,
    Precision,
    Recall,
    ARI,
    NMI,
    VMeasure,
    BCubedPrecision,
    BCubedRecall,

    // Statistics metrics (single partition)
    EntityCount,
    Entropy,
}

impl MetricType {
    /// Check if this metric requires comparison between two partitions
    pub fn requires_comparison(&self) -> bool {
        !matches!(self, MetricType::EntityCount | MetricType::Entropy)
    }

    /// Get the display name for this metric
    pub fn name(&self) -> &'static str {
        match self {
            MetricType::F1 => "f1",
            MetricType::Precision => "precision",
            MetricType::Recall => "recall",
            MetricType::ARI => "ari",
            MetricType::NMI => "nmi",
            MetricType::VMeasure => "v_measure",
            MetricType::BCubedPrecision => "bcubed_precision",
            MetricType::BCubedRecall => "bcubed_recall",
            MetricType::EntityCount => "entity_count",
            MetricType::Entropy => "entropy",
        }
    }
}

/// Re-export the algorithms
pub mod delta;
pub mod record;

pub use delta::DeltaAlgorithm;
pub use record::RecordAlgorithm;
