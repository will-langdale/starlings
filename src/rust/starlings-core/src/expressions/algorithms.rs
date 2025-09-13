//! Core algorithms for entity resolution analysis

use super::contingency::{ContingencyTable, SparseContingencyTable};
use super::metrics::{compute_entity_count, compute_entropy};
use super::types::MetricType;
use crate::{DataContext, PartitionLevel};
use std::sync::Arc;

/// Generate all threshold values for a sweep expression
/// Ensures step sizes are quantized to 0.05 increments for performance
pub fn generate_sweep_thresholds(start: f64, stop: f64, step: f64) -> Vec<f64> {
    // Enforce minimum step of 0.05 and round to nearest 0.05
    let step = if step < 0.05 {
        0.05
    } else {
        (step / 0.05).round() * 0.05
    };

    let mut thresholds = Vec::new();
    let mut current = start;

    while current <= stop + f64::EPSILON {
        // Add small epsilon to handle floating-point precision issues
        thresholds.push(current);
        current += step;
    }

    thresholds
}

/// Build all contingency tables for sweep × sweep in a single pass
/// This is the key optimisation: build indices once, process records once
pub fn build_all_sweep_tables(
    partitions1: &[PartitionLevel],
    partitions2: &[PartitionLevel],
    context: &Arc<DataContext>,
) -> Vec<Vec<SparseContingencyTable>> {
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
    let mut tables =
        vec![vec![SparseContingencyTable::new(); partitions2.len()]; partitions1.len()];

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

    // Process all records once, updating all tables
    for record_idx in 0..num_records {
        // For each record, check which entity it belongs to in each partition
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

    tables
}

/// Build contingency table using sparse representation for efficiency
/// Uses SparseContingencyTable for O(k₁ × k₂) complexity where k = entities
/// Automatically uses parallel processing for large partitions
pub fn build_contingency_table(
    partition1: &PartitionLevel,
    partition2: &PartitionLevel,
) -> ContingencyTable {
    // Fast path for identical partitions (same threshold comparison)
    if std::ptr::eq(partition1, partition2) {
        // Perfect match case - all pairs are true positives
        let mut true_positives = 0u64;
        for entity in partition1.entities() {
            let size = entity.len();
            if size > 1 {
                true_positives += (size * (size - 1)) / 2;
            }
        }

        let total_records = partition1.entities().iter().map(|e| e.len()).sum::<u64>();
        let total_possible_pairs = (total_records * (total_records - 1)) / 2;

        return ContingencyTable {
            true_positives: true_positives as u32,
            false_positives: 0,
            false_negatives: 0,
            true_negatives: (total_possible_pairs - true_positives) as u32,
        };
    }

    // Use sparse contingency table for efficient computation
    let sparse_table = SparseContingencyTable::from_partitions(partition1, partition2);
    sparse_table.to_contingency_table()
}

/// Error type for metric computation
#[derive(Debug)]
pub enum MetricError {
    /// Metric requires comparison but was used with single partition
    ComparisonMetricUsedForSingle(MetricType),
    /// Metric is for single partition but was used for comparison
    SingleMetricUsedForComparison(MetricType),
}

impl std::fmt::Display for MetricError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MetricError::ComparisonMetricUsedForSingle(metric) => {
                write!(f, "Metric {:?} requires comparison between two partitions but was used with a single partition", metric)
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

/// Compute specified metric for comparison between partitions
pub fn compute_comparison_metric(
    partition1: &PartitionLevel,
    partition2: &PartitionLevel,
    metric: &MetricType,
) -> Result<f64, MetricError> {
    match metric {
        MetricType::F1 => {
            let table = build_contingency_table(partition1, partition2);
            Ok(table.f1_score())
        }
        MetricType::Precision => {
            let table = build_contingency_table(partition1, partition2);
            Ok(table.precision())
        }
        MetricType::Recall => {
            let table = build_contingency_table(partition1, partition2);
            Ok(table.recall())
        }
        MetricType::ARI | MetricType::NMI | MetricType::VMeasure => {
            // Placeholder: These metrics require more complex implementations
            // For now, return 0.0 as they're not yet implemented
            Ok(0.0)
        }
        MetricType::BCubedPrecision | MetricType::BCubedRecall => {
            // Placeholder: B-cubed metrics require different approach
            // For now, return 0.0 as they're not yet implemented
            Ok(0.0)
        }
        MetricType::EntityCount | MetricType::Entropy => {
            // Single collection metrics shouldn't be called with comparison
            Err(MetricError::SingleMetricUsedForComparison(metric.clone()))
        }
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_sweep_thresholds() {
        let thresholds = generate_sweep_thresholds(0.5, 0.7, 0.1);
        assert_eq!(thresholds.len(), 3);
        assert!((thresholds[0] - 0.5).abs() < 1e-10);
        assert!((thresholds[1] - 0.6).abs() < 1e-10);
        assert!((thresholds[2] - 0.7).abs() < 1e-10);

        let thresholds = generate_sweep_thresholds(0.8, 0.9, 0.05);
        assert_eq!(thresholds.len(), 3);
        assert!((thresholds[0] - 0.8).abs() < 1e-10);
        assert!((thresholds[1] - 0.85).abs() < 1e-10);
        assert!((thresholds[2] - 0.9).abs() < 1e-10);

        // Test step rounding
        let thresholds = generate_sweep_thresholds(0.5, 0.6, 0.03);
        assert_eq!(thresholds.len(), 3); // 0.03 rounds to 0.05
        assert!((thresholds[0] - 0.5).abs() < 1e-10);
        assert!((thresholds[1] - 0.55).abs() < 1e-10);
        assert!((thresholds[2] - 0.6).abs() < 1e-10);
    }

    #[test]
    fn test_metric_error_handling() {
        use roaring::RoaringBitmap;

        // Create a test partition
        let entities = vec![RoaringBitmap::from_iter([0, 1])];
        let partition = PartitionLevel::new(0.5, entities);

        // Test single metric used in comparison context
        let result = compute_comparison_metric(&partition, &partition, &MetricType::EntityCount);
        assert!(result.is_err());
        match result {
            Err(MetricError::SingleMetricUsedForComparison(MetricType::EntityCount)) => {}
            _ => panic!("Expected SingleMetricUsedForComparison error"),
        }

        // Test comparison metric used in single context
        let result = compute_single_metric(&partition, &MetricType::F1);
        assert!(result.is_err());
        match result {
            Err(MetricError::ComparisonMetricUsedForSingle(MetricType::F1)) => {}
            _ => panic!("Expected ComparisonMetricUsedForSingle error"),
        }

        // Test error messages
        let err = MetricError::SingleMetricUsedForComparison(MetricType::EntityCount);
        assert!(err.to_string().contains("single partition analysis"));

        let err = MetricError::ComparisonMetricUsedForSingle(MetricType::F1);
        assert!(err.to_string().contains("requires comparison"));
    }
}
