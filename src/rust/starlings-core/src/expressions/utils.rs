//! Utility functions for expressions

use super::metrics::{compute_entity_count, compute_entropy};
use super::types::MetricType;
use crate::PartitionLevel;

/// Generate all threshold values for a sweep expression
///
/// Generates thresholds from HIGH to LOW (e.g., 0.9 → 0.6) for optimal performance
/// with the Delta algorithm, which can incrementally track entity merges as thresholds decrease.
/// Ensures step sizes are quantized to 0.05 increments for performance.
pub fn generate_sweep_thresholds(start: f64, stop: f64, step: f64) -> Vec<f64> {
    // Enforce minimum step of 0.05 and round to nearest 0.05
    let step = if step < 0.05 {
        0.05
    } else {
        (step / 0.05).round() * 0.05
    };

    let mut thresholds = Vec::new();

    // ALWAYS generate from HIGH to LOW for Delta algorithm
    // This allows incremental tracking as entities merge with decreasing thresholds
    let mut current = stop;

    while current >= start - f64::EPSILON {
        thresholds.push(current);
        current -= step;
    }

    thresholds
}

/// Error types for metric computation
#[derive(Debug, Clone)]
pub enum MetricError {
    /// Tried to use a comparison metric for single partition analysis
    ComparisonMetricUsedForSingle(MetricType),
    /// Tried to use a single partition metric for comparison
    SingleMetricUsedForComparison(MetricType),
}

impl std::fmt::Display for MetricError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MetricError::ComparisonMetricUsedForSingle(metric) => {
                write!(
                    f,
                    "Metric {:?} requires two partitions but was used for single partition analysis",
                    metric
                )
            }
            MetricError::SingleMetricUsedForComparison(metric) => {
                write!(
                    f,
                    "Metric {:?} is for single partition analysis but was used for comparison",
                    metric
                )
            }
        }
    }
}

impl std::error::Error for MetricError {}

/// Compute specified metric for a single partition
pub fn compute_single_metric(
    partition: &PartitionLevel,
    metric: &MetricType,
) -> Result<f64, MetricError> {
    match metric {
        MetricType::EntityCount => Ok(compute_entity_count(partition)),
        MetricType::Entropy => Ok(compute_entropy(partition)),
        _ => {
            // Comparison metrics shouldn't be called with single partition
            Err(MetricError::ComparisonMetricUsedForSingle(metric.clone()))
        }
    }
}
