//! Expression API for entity resolution analysis
//!
//! This module implements the core algorithms for cross-collection comparison,
//! including the optimised record-based algorithm that reduces complexity from
//! O(k₁ × k₂) to O(r) where k = entities and r = records.
//!
//! The module provides:
//! - Expression types for point and sweep operations
//! - Metric definitions for evaluation and statistics
//! - Sparse contingency table implementation for efficient metric computation
//! - Record-based algorithm for large-scale comparisons

use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;

use starlings_core::{DataContext, PartitionLevel};

/// Represents an expression operation type
#[derive(Debug, Clone, PartialEq)]
pub enum ExpressionType {
    /// Single threshold query
    Point { collection: String, threshold: f64 },
    /// Threshold range query
    Sweep {
        collection: String,
        start: f64,
        stop: f64,
        step: f64,
    },
}

/// Represents a metric type for computation
#[derive(Debug, Clone, PartialEq)]
pub enum MetricType {
    // Evaluation metrics (require 2+ collections)
    F1,
    Precision,
    Recall,
    #[allow(clippy::upper_case_acronyms)]
    ARI,
    #[allow(clippy::upper_case_acronyms)]
    NMI,
    VMeasure,
    BCubedPrecision,
    BCubedRecall,

    // Statistics metrics (single collection)
    EntityCount,
    Entropy,
}

impl MetricType {
    /// Check if this metric requires multiple collections
    pub fn requires_comparison(&self) -> bool {
        matches!(
            self,
            MetricType::F1
                | MetricType::Precision
                | MetricType::Recall
                | MetricType::ARI
                | MetricType::NMI
                | MetricType::VMeasure
                | MetricType::BCubedPrecision
                | MetricType::BCubedRecall
        )
    }
}

/// Parse Python expression object into Rust expression type
pub fn parse_expression(py_expr: &Bound<'_, PyAny>) -> PyResult<ExpressionType> {
    // Extract expression_type attribute
    let expr_type: String = py_expr.getattr("expression_type")?.extract()?;

    match expr_type.as_str() {
        "point" => {
            let params = py_expr.getattr("params")?;
            let collection: String = params.get_item("collection")?.extract()?;
            let threshold: f64 = params.get_item("threshold")?.extract()?;

            Ok(ExpressionType::Point {
                collection,
                threshold,
            })
        }
        "sweep" => {
            let params = py_expr.getattr("params")?;
            let collection: String = params.get_item("collection")?.extract()?;
            let start: f64 = params.get_item("start")?.extract()?;
            let stop: f64 = params.get_item("stop")?.extract()?;
            let step: f64 = params.get_item("step")?.extract()?;

            Ok(ExpressionType::Sweep {
                collection,
                start,
                stop,
                step,
            })
        }
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Unknown expression type: {}",
            expr_type
        ))),
    }
}

/// Parse Python metric function into Rust metric type
pub fn parse_metric(py_metric: &Bound<'_, PyAny>) -> PyResult<MetricType> {
    let name: String = py_metric.getattr("name")?.extract()?;

    match name.as_str() {
        "f1" => Ok(MetricType::F1),
        "precision" => Ok(MetricType::Precision),
        "recall" => Ok(MetricType::Recall),
        "ari" => Ok(MetricType::ARI),
        "nmi" => Ok(MetricType::NMI),
        "v_measure" => Ok(MetricType::VMeasure),
        "bcubed_precision" => Ok(MetricType::BCubedPrecision),
        "bcubed_recall" => Ok(MetricType::BCubedRecall),
        "entity_count" => Ok(MetricType::EntityCount),
        "entropy" => Ok(MetricType::Entropy),
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Unknown metric: {}",
            name
        ))),
    }
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
/// This is the key optimization: build indices once, process records once
pub fn build_all_sweep_tables(
    partitions1: &[PartitionLevel],
    partitions2: &[PartitionLevel],
    context: &Arc<DataContext>,
) -> Vec<Vec<SparseContingencyTable>> {
    let num_records = context.len();
    let generation = context.generation();

    // Building all tables for sweep × sweep comparison
    let _start_time = std::time::Instant::now();

    // Build reverse indices for all partitions
    let indices1: Vec<_> = partitions1
        .iter()
        .map(|p| p.get_record_to_entity_index(num_records, generation))
        .collect();
    let indices2: Vec<_> = partitions2
        .iter()
        .map(|p| p.get_record_to_entity_index(num_records, generation))
        .collect();

    // All reverse indices built

    // Initialize empty tables for all combinations
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
    let _process_start = std::time::Instant::now();
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
    // All records processed for contingency tables

    tables
}

/// Compute entity count for a partition
pub fn compute_entity_count(partition: &PartitionLevel) -> f64 {
    partition.entities().len() as f64
}

/// Compute entropy for a partition
pub fn compute_entropy(partition: &PartitionLevel) -> f64 {
    let total_records: u32 = partition
        .entities()
        .iter()
        .map(|entity| entity.len() as u32)
        .sum();

    if total_records == 0 {
        return 0.0;
    }

    let mut entropy = 0.0;
    for entity in partition.entities() {
        if !entity.is_empty() {
            let proportion = entity.len() as f64 / total_records as f64;
            entropy -= proportion * proportion.log2();
        }
    }

    entropy
}

/// Entity ID for sparse contingency table
pub type EntityId = usize;

/// Sparse contingency table for efficient incremental updates
/// Only stores non-zero cells, enabling O(k) updates where k = affected entities
#[derive(Debug, Clone)]
pub struct SparseContingencyTable {
    /// Non-zero entity overlap counts: (entity1_id, entity2_id) -> overlap_count
    pub nonzero_cells: HashMap<(EntityId, EntityId), u32>,
    /// Row marginals: entity1_id -> total_records_in_entity1  
    pub row_marginals: HashMap<EntityId, u32>,
    /// Column marginals: entity2_id -> total_records_in_entity2
    pub col_marginals: HashMap<EntityId, u32>,
    /// Total number of records across both partitions
    pub total_records: u32,
}

impl SparseContingencyTable {
    /// Create new empty sparse contingency table
    pub fn new() -> Self {
        Self {
            nonzero_cells: HashMap::new(),
            row_marginals: HashMap::new(),
            col_marginals: HashMap::new(),
            total_records: 0,
        }
    }

    /// Build initial sparse contingency table from two partitions
    /// O(k₁ × k₂) where k = number of entities (much better than O(n²))
    /// Uses parallel processing for large entity counts
    /// Build contingency table using entity-based comparison
    pub fn from_partitions(partition1: &PartitionLevel, partition2: &PartitionLevel) -> Self {
        let mut table = Self::new();

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

        let n1 = partition1.entities().len();
        let n2 = partition2.entities().len();

        // Direct entity-based comparison
        // For very large comparisons, consider using from_partitions_via_records instead

        // Decide whether to use parallel or sequential processing
        // Parallel overhead is worth it for > 1000 entities in either partition
        let use_parallel = n1 > 1000 || n2 > 1000;

        if use_parallel {
            // Parallel processing for large entity counts
            let nonzero_cells = Mutex::new(HashMap::new());

            // Process entity pairs in parallel
            partition1
                .entities()
                .par_iter()
                .enumerate()
                .for_each(|(entity1_id, entity1)| {
                    // Local accumulator to reduce lock contention
                    let mut local_overlaps = Vec::new();

                    for (entity2_id, entity2) in partition2.entities().iter().enumerate() {
                        // Skip small entities early for optimization
                        if entity1.len() < 2 && entity2.len() < 2 {
                            continue;
                        }

                        let overlap = entity1.intersection_len(entity2);
                        if overlap > 0 {
                            local_overlaps.push((entity1_id, entity2_id, overlap as u32));
                        }
                    }

                    // Batch insert to reduce lock contention
                    if !local_overlaps.is_empty() {
                        let mut cells = nonzero_cells.lock().unwrap();
                        for (e1, e2, overlap) in local_overlaps {
                            cells.insert((e1, e2), overlap);
                        }
                    }
                });

            table.nonzero_cells = nonzero_cells.into_inner().unwrap();
        } else {
            // Sequential processing for small entity counts (original implementation)
            for (entity1_id, entity1) in partition1.entities().iter().enumerate() {
                for (entity2_id, entity2) in partition2.entities().iter().enumerate() {
                    let overlap = entity1.intersection_len(entity2);
                    if overlap > 0 {
                        table
                            .nonzero_cells
                            .insert((entity1_id, entity2_id), overlap as u32);
                    }
                }
            }
        }

        table
    }

    /// Build contingency table using record-based algorithm (O(r) complexity)
    /// This is much faster than entity-based comparison when collections share the same context
    pub fn from_partitions_via_records(
        partition1: &PartitionLevel,
        partition2: &PartitionLevel,
        context: &Arc<DataContext>,
    ) -> Self {
        let mut table = Self::new();
        let num_records = context.len();
        let generation = context.generation();

        // Using record-based algorithm for cross-collection comparison

        // Build reverse indices for both partitions
        let _start = std::time::Instant::now();
        let record_to_entity1 = partition1.get_record_to_entity_index(num_records, generation);
        let record_to_entity2 = partition2.get_record_to_entity_index(num_records, generation);
        // Reverse indices built

        // Check if indices are valid (same generation)
        if record_to_entity1.generation != generation || record_to_entity2.generation != generation
        {
            // Fall back to entity-based algorithm if caches are stale
            return Self::from_partitions(partition1, partition2);
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
        let _process_start = std::time::Instant::now();
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

            // Entity pairs collected

            // Aggregate into contingency table
            let _agg_start = std::time::Instant::now();
            for (entity1_idx, entity2_idx) in pairs {
                *table
                    .nonzero_cells
                    .entry((entity1_idx, entity2_idx))
                    .or_insert(0) += 1;
            }
            // Contingency table aggregation complete
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

        table
    }

    /// Compute contingency table metrics from sparse representation
    pub fn to_contingency_table(&self) -> ContingencyTable {
        let mut true_positives = 0u64;
        let mut false_positives = 0u64;
        let mut false_negatives = 0u64;

        // Pre-compute pairs together for each entity to avoid O(k×m) complexity
        // Build index: entity1_id -> sum of pairs together
        let mut entity1_pairs_together: HashMap<usize, u64> = HashMap::new();
        let mut entity2_pairs_together: HashMap<usize, u64> = HashMap::new();

        // Single pass through nonzero cells to compute all necessary sums
        for ((e1, e2), &overlap) in &self.nonzero_cells {
            if overlap > 1 {
                let pairs = (overlap as u64 * (overlap as u64 - 1)) / 2;
                true_positives += pairs;
                *entity1_pairs_together.entry(*e1).or_insert(0) += pairs;
                *entity2_pairs_together.entry(*e2).or_insert(0) += pairs;
            }
        }

        // False positives: pairs together in partition1, apart in partition2
        for (entity1_id, &size1) in &self.row_marginals {
            if size1 > 1 {
                let all_pairs_in_entity1 = (size1 as u64 * (size1 as u64 - 1)) / 2;
                let pairs_also_together =
                    entity1_pairs_together.get(entity1_id).copied().unwrap_or(0);
                false_positives += all_pairs_in_entity1 - pairs_also_together;
            }
        }

        // False negatives: pairs apart in partition1, together in partition2
        for (entity2_id, &size2) in &self.col_marginals {
            if size2 > 1 {
                let all_pairs_in_entity2 = (size2 as u64 * (size2 as u64 - 1)) / 2;
                let pairs_also_together =
                    entity2_pairs_together.get(entity2_id).copied().unwrap_or(0);
                false_negatives += all_pairs_in_entity2 - pairs_also_together;
            }
        }

        // True negatives: all possible pairs minus counted pairs
        let total_possible_pairs = if self.total_records > 1 {
            (self.total_records as u64 * (self.total_records as u64 - 1)) / 2
        } else {
            0
        };
        let true_negatives =
            total_possible_pairs.saturating_sub(true_positives + false_positives + false_negatives);

        ContingencyTable {
            true_positives: true_positives as u32,
            false_positives: false_positives as u32,
            false_negatives: false_negatives as u32,
            true_negatives: true_negatives as u32,
        }
    }
}

/// Contingency table for comparing two partitions
#[derive(Debug)]
#[allow(dead_code)]
pub struct ContingencyTable {
    /// Number of record pairs that are in same entity in both partitions
    pub true_positives: u32,
    /// Number of record pairs that are in same entity in first but different in second
    pub false_positives: u32,
    /// Number of record pairs that are in different entities in first but same in second
    pub false_negatives: u32,
    /// Number of record pairs that are in different entities in both partitions
    pub true_negatives: u32,
}

impl ContingencyTable {
    /// Compute precision: TP / (TP + FP)
    pub fn precision(&self) -> f64 {
        let denominator = self.true_positives + self.false_positives;
        if denominator == 0 {
            1.0 // No positive predictions made
        } else {
            self.true_positives as f64 / denominator as f64
        }
    }

    /// Compute recall: TP / (TP + FN)
    pub fn recall(&self) -> f64 {
        let denominator = self.true_positives + self.false_negatives;
        if denominator == 0 {
            1.0 // No true positives exist
        } else {
            self.true_positives as f64 / denominator as f64
        }
    }

    /// Compute F1 score: 2 * (precision * recall) / (precision + recall)
    pub fn f1_score(&self) -> f64 {
        let precision = self.precision();
        let recall = self.recall();

        if precision + recall == 0.0 {
            0.0
        } else {
            2.0 * (precision * recall) / (precision + recall)
        }
    }
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

/// Compute specified metric for comparison between partitions
pub fn compute_comparison_metric(
    partition1: &PartitionLevel,
    partition2: &PartitionLevel,
    metric: &MetricType,
) -> f64 {
    match metric {
        MetricType::F1 => {
            let table = build_contingency_table(partition1, partition2);
            table.f1_score()
        }
        MetricType::Precision => {
            let table = build_contingency_table(partition1, partition2);
            table.precision()
        }
        MetricType::Recall => {
            let table = build_contingency_table(partition1, partition2);
            table.recall()
        }
        MetricType::ARI | MetricType::NMI | MetricType::VMeasure => {
            // Placeholder: These metrics require more complex implementations
            // For now, return 0.0 as they're not yet implemented
            0.0
        }
        MetricType::BCubedPrecision | MetricType::BCubedRecall => {
            // Placeholder: B-cubed metrics require different approach
            // For now, return 0.0 as they're not yet implemented
            0.0
        }
        _ => {
            // Single collection metrics shouldn't be called with comparison
            panic!("Single collection metric used in comparison context")
        }
    }
}

/// Compute specified metric for a single partition
pub fn compute_single_metric(partition: &PartitionLevel, metric: &MetricType) -> f64 {
    match metric {
        MetricType::EntityCount => compute_entity_count(partition),
        MetricType::Entropy => compute_entropy(partition),
        _ => {
            // Comparison metrics shouldn't be called with single partition
            panic!("Comparison metric used in single partition context")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use roaring::RoaringBitmap;

    fn create_test_partition(entities: Vec<Vec<u32>>) -> PartitionLevel {
        let bitmap_entities: Vec<RoaringBitmap> = entities
            .into_iter()
            .map(|records| records.into_iter().collect())
            .collect();
        PartitionLevel::new(0.0, bitmap_entities)
    }

    #[test]
    fn test_generate_sweep_thresholds() {
        let thresholds = generate_sweep_thresholds(0.5, 0.7, 0.1);
        assert_eq!(thresholds, vec![0.5, 0.6, 0.7]);

        let thresholds = generate_sweep_thresholds(0.8, 0.9, 0.05);
        assert_eq!(thresholds, vec![0.8, 0.85, 0.9]);
    }

    #[test]
    fn test_compute_entity_count() {
        let partition = create_test_partition(vec![vec![0, 1], vec![2], vec![3, 4, 5]]);
        assert_eq!(compute_entity_count(&partition), 3.0);

        let empty_partition = create_test_partition(vec![]);
        assert_eq!(compute_entity_count(&empty_partition), 0.0);
    }

    #[test]
    fn test_compute_entropy() {
        // Uniform distribution: 2 entities of size 2 each
        let partition = create_test_partition(vec![vec![0, 1], vec![2, 3]]);
        let entropy = compute_entropy(&partition);
        assert!((entropy - 1.0).abs() < 1e-10); // -2 * (0.5 * log2(0.5)) = 1.0

        // Single entity
        let partition = create_test_partition(vec![vec![0, 1, 2, 3]]);
        let entropy = compute_entropy(&partition);
        assert!(entropy.abs() < 1e-10); // Should be 0.0
    }

    #[test]
    fn test_contingency_table() {
        // Perfect match: both partitions identical
        let partition1 = create_test_partition(vec![vec![0, 1], vec![2, 3]]);
        let partition2 = create_test_partition(vec![vec![0, 1], vec![2, 3]]);

        let table = build_contingency_table(&partition1, &partition2);
        assert_eq!(table.precision(), 1.0);
        assert_eq!(table.recall(), 1.0);
        assert_eq!(table.f1_score(), 1.0);

        // Complete mismatch: one partition has all singles, other has all together
        let partition1 = create_test_partition(vec![vec![0], vec![1], vec![2], vec![3]]);
        let partition2 = create_test_partition(vec![vec![0, 1, 2, 3]]);

        let table = build_contingency_table(&partition1, &partition2);
        assert_eq!(table.precision(), 0.0); // No correct positive predictions
        assert_eq!(table.recall(), 0.0); // No true positives found
        assert_eq!(table.f1_score(), 0.0);
    }

    #[test]
    fn test_metric_type_requires_comparison() {
        assert!(MetricType::F1.requires_comparison());
        assert!(MetricType::Precision.requires_comparison());
        assert!(MetricType::Recall.requires_comparison());

        assert!(!MetricType::EntityCount.requires_comparison());
        assert!(!MetricType::Entropy.requires_comparison());
    }
}
