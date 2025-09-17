//! Core algorithms for entity resolution analysis

use super::contingency::SparseContingencyTable;
use super::metrics::{compute_entity_count, compute_entropy};
use super::types::MetricType;
use crate::{DataContext, PartitionLevel};
use std::sync::Arc;

/// Pre-computed marginals for a partition to avoid redundant computation
struct PrecomputedMarginals {
    /// Entity sizes indexed by entity_id
    entity_sizes: Vec<u32>,
    /// Total number of records in the partition
    total_records: u32,
}

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
    #[cfg(debug_assertions)]
    use crate::debug_println;

    let num_records = context.len();
    let generation = context.generation();

    #[cfg(debug_assertions)]
    let start_time = std::time::Instant::now();

    // Build reverse indices for all partitions
    #[cfg(debug_assertions)]
    let index_start = std::time::Instant::now();

    let indices1: Vec<_> = partitions1
        .iter()
        .map(|p| p.get_record_to_entity_index(num_records, generation))
        .collect();
    let indices2: Vec<_> = partitions2
        .iter()
        .map(|p| p.get_record_to_entity_index(num_records, generation))
        .collect();

    #[cfg(debug_assertions)]
    {
        let index_time = index_start.elapsed();
        debug_println!(
            "      🔧 Index building: {:?} for {} indices",
            index_time,
            indices1.len() + indices2.len()
        );

        // Report index statistics
        let total_entities1: usize = partitions1.iter().map(|p| p.entities().len()).sum();
        let total_entities2: usize = partitions2.iter().map(|p| p.entities().len()).sum();
        debug_println!(
            "         Entities: {} in set 1, {} in set 2",
            total_entities1,
            total_entities2
        );
    }

    // Initialise empty tables for all combinations
    let mut tables =
        vec![vec![SparseContingencyTable::new(); partitions2.len()]; partitions1.len()];

    // Pre-compute marginals once per partition
    #[cfg(debug_assertions)]
    let marginal_start = std::time::Instant::now();

    let marginals1: Vec<PrecomputedMarginals> = partitions1
        .iter()
        .map(|partition| {
            let entity_sizes: Vec<u32> = partition
                .entities()
                .iter()
                .map(|e| e.len() as u32)
                .collect();
            let total_records = entity_sizes.iter().sum();
            PrecomputedMarginals {
                entity_sizes,
                total_records,
            }
        })
        .collect();

    let marginals2: Vec<PrecomputedMarginals> = partitions2
        .iter()
        .map(|partition| {
            let entity_sizes: Vec<u32> = partition
                .entities()
                .iter()
                .map(|e| e.len() as u32)
                .collect();
            let total_records = entity_sizes.iter().sum();
            PrecomputedMarginals {
                entity_sizes,
                total_records,
            }
        })
        .collect();

    // Initialize tables with pre-computed marginals
    for (i, marginal1) in marginals1.iter().enumerate() {
        for (j, marginal2) in marginals2.iter().enumerate() {
            let table = &mut tables[i][j];
            table.total_records = marginal1.total_records;

            // Copy row marginals
            for (entity_id, &size) in marginal1.entity_sizes.iter().enumerate() {
                if size > 0 {
                    table.row_marginals.insert(entity_id, size);
                }
            }

            // Copy column marginals
            for (entity_id, &size) in marginal2.entity_sizes.iter().enumerate() {
                if size > 0 {
                    table.col_marginals.insert(entity_id, size);
                }
            }
        }
    }

    #[cfg(debug_assertions)]
    {
        let marginal_time = marginal_start.elapsed();
        let num_tables = partitions1.len() * partitions2.len();
        debug_println!(
            "      🔧 Marginal pre-computation and table init: {:?} for {} tables",
            marginal_time,
            num_tables
        );
    }

    // Process all records once, updating all tables
    #[cfg(debug_assertions)]
    let record_start = std::time::Instant::now();

    #[cfg(debug_assertions)]
    let mut cells_added = 0usize;

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

                        #[cfg(debug_assertions)]
                        {
                            cells_added += 1;
                        }
                    }
                }
            }
        }
    }

    #[cfg(debug_assertions)]
    {
        let record_time = record_start.elapsed();
        debug_println!(
            "      🔧 Record processing: {:?} for {} records",
            record_time,
            num_records
        );
        debug_println!(
            "         Cells updated: {} ({:.1} per record)",
            cells_added,
            cells_added as f64 / num_records as f64
        );

        // Calculate table sparsity
        let total_cells: usize = tables
            .iter()
            .flat_map(|row| row.iter())
            .map(|t| t.nonzero_cells.len())
            .sum();
        let avg_cells_per_table =
            total_cells as f64 / (partitions1.len() * partitions2.len()) as f64;
        debug_println!(
            "         Average nonzero cells per table: {:.1}",
            avg_cells_per_table
        );

        let total_time = start_time.elapsed();
        debug_println!("      🔧 Total build_all_sweep_tables: {:?}", total_time);
    }

    tables
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
