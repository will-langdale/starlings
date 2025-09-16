//! Record-based algorithm for O(r) metric computation
//!
//! This algorithm iterates through records once to build contingency tables,
//! making it optimal for sweep × sweep comparisons where we need to compare
//! many partition pairs from collections sharing the same record space.

use super::{ComparisonType, ComplexityEstimate, MetricAlgorithm, MetricResults, MetricType};
use crate::expressions::contingency::SparseContingencyTable;
use crate::metrics::implementations::statistics::{compute_entity_count, compute_entropy};
use crate::{DataContext, PartitionLevel};
use std::collections::HashMap;
use std::sync::Arc;

/// Record-based algorithm that achieves O(r) complexity by iterating records once
pub struct RecordAlgorithm {
    /// Cache for record-to-entity indices to avoid recomputation
    #[allow(dead_code)]
    index_cache: HashMap<(u64, u64), RecordToEntityIndex>,
}

/// Cached index mapping records to entities
#[allow(dead_code)]
struct RecordToEntityIndex {
    index: Vec<Option<usize>>,
    generation: u64,
}

impl Default for RecordAlgorithm {
    fn default() -> Self {
        Self::new()
    }
}

impl RecordAlgorithm {
    /// Create a new record-based algorithm instance
    pub fn new() -> Self {
        Self {
            index_cache: HashMap::new(),
        }
    }

    /// Build all contingency tables for sweep × sweep in a single pass
    fn build_all_sweep_tables(
        &mut self,
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

    /// Compute metrics from a contingency table
    fn compute_metrics_from_table(
        &self,
        table: &SparseContingencyTable,
        metrics: &[MetricType],
    ) -> MetricResults {
        let mut results = HashMap::new();

        for metric in metrics {
            let value = match metric {
                MetricType::F1 => table.compute_f1(),
                MetricType::Precision => table.compute_precision(),
                MetricType::Recall => table.compute_recall(),
                MetricType::ARI => self.compute_ari(table),
                MetricType::NMI => self.compute_nmi(table),
                MetricType::VMeasure => self.compute_v_measure(table),
                MetricType::BCubedPrecision => self.compute_bcubed_precision(table),
                MetricType::BCubedRecall => self.compute_bcubed_recall(table),
                _ => 0.0, // Statistics metrics handled separately
            };
            results.insert(metric.name().to_string(), value);
        }

        results
    }

    fn compute_ari(&self, table: &SparseContingencyTable) -> f64 {
        table.compute_ari()
    }

    fn compute_nmi(&self, table: &SparseContingencyTable) -> f64 {
        table.compute_nmi()
    }

    fn compute_v_measure(&self, table: &SparseContingencyTable) -> f64 {
        table.compute_v_measure()
    }

    fn compute_bcubed_precision(&self, table: &SparseContingencyTable) -> f64 {
        table.compute_bcubed_precision()
    }

    fn compute_bcubed_recall(&self, table: &SparseContingencyTable) -> f64 {
        table.compute_bcubed_recall()
    }
}

impl MetricAlgorithm for RecordAlgorithm {
    fn name(&self) -> &'static str {
        "Record-based (O(r) single-pass)"
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
            let table = SparseContingencyTable::from_partitions(partition1, partition2, context);
            results.extend(self.compute_metrics_from_table(&table, metrics));
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
                    results.push(self.compute_metrics_from_table(&table, metrics));
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
