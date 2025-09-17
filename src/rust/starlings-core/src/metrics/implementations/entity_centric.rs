//! Entity-centric metrics (B-cubed precision and recall)

use crate::expressions::contingency::SparseContingencyTable;

/// Compute B-cubed precision
///
/// B-cubed precision measures the average precision for each record within its cluster.
/// Returns a value between 0 and 1, where:
/// - 1.0 indicates perfect precision (each cluster contains only correct members)
/// - 0.0 indicates no correct clustering
pub fn compute_bcubed_precision(table: &SparseContingencyTable) -> f64 {
    // Delegate to the implementation on SparseContingencyTable
    table.compute_bcubed_precision()
}

/// Compute B-cubed recall
///
/// B-cubed recall measures the average recall for each record within its cluster.
/// Returns a value between 0 and 1, where:
/// - 1.0 indicates perfect recall (all correct members are in the same cluster)
/// - 0.0 indicates no correct clustering
pub fn compute_bcubed_recall(table: &SparseContingencyTable) -> f64 {
    // Delegate to the implementation on SparseContingencyTable
    table.compute_bcubed_recall()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DataContext, PartitionLevel};
    use roaring::RoaringBitmap;
    use std::sync::Arc;

    fn create_test_contingency_table(
        partition1_entities: Vec<Vec<u32>>,
        partition2_entities: Vec<Vec<u32>>,
    ) -> SparseContingencyTable {
        // Create data context with all records
        let context = DataContext::new();
        let mut max_record_id = 0;

        for entity in &partition1_entities {
            for &record_id in entity {
                if record_id > max_record_id {
                    max_record_id = record_id;
                }
            }
        }
        for entity in &partition2_entities {
            for &record_id in entity {
                if record_id > max_record_id {
                    max_record_id = record_id;
                }
            }
        }

        // Ensure all records exist in context
        for i in 0..=max_record_id {
            context.ensure_record("test", crate::core::Key::U32(i));
        }

        let context = Arc::new(context);

        // Create partitions
        let bitmap_entities1: Vec<RoaringBitmap> = partition1_entities
            .into_iter()
            .map(|records| records.into_iter().collect())
            .collect();
        let partition1 = PartitionLevel::new(0.5, bitmap_entities1);

        let bitmap_entities2: Vec<RoaringBitmap> = partition2_entities
            .into_iter()
            .map(|records| records.into_iter().collect())
            .collect();
        let partition2 = PartitionLevel::new(0.5, bitmap_entities2);

        // Build contingency table
        SparseContingencyTable::from_partitions(&partition1, &partition2, &context)
    }

    #[test]
    fn test_bcubed_precision_perfect() {
        // Scenario: Perfect clustering
        // Partition1: {0,1}, {2,3}, {4,5}
        // Partition2: {0,1}, {2,3}, {4,5}
        let table = create_test_contingency_table(
            vec![vec![0, 1], vec![2, 3], vec![4, 5]],
            vec![vec![0, 1], vec![2, 3], vec![4, 5]],
        );

        // Expected: B-cubed precision = 1.0
        let precision = compute_bcubed_precision(&table);
        assert!(
            (precision - 1.0).abs() < 1e-10,
            "Perfect clustering should yield B-cubed precision = 1.0, got {}",
            precision
        );
    }

    #[test]
    fn test_bcubed_recall_perfect() {
        // Scenario: Perfect clustering
        // Partition1: {0,1}, {2,3}, {4,5}
        // Partition2: {0,1}, {2,3}, {4,5}
        let table = create_test_contingency_table(
            vec![vec![0, 1], vec![2, 3], vec![4, 5]],
            vec![vec![0, 1], vec![2, 3], vec![4, 5]],
        );

        // Expected: B-cubed recall = 1.0
        let recall = compute_bcubed_recall(&table);
        assert!(
            (recall - 1.0).abs() < 1e-10,
            "Perfect clustering should yield B-cubed recall = 1.0, got {}",
            recall
        );
    }

    #[test]
    fn test_bcubed_partial_overlap() {
        // Scenario: Partial overlap
        // Partition1: {0,1,2}, {3,4,5}
        // Partition2: {0,1}, {2,3}, {4,5}
        let table = create_test_contingency_table(
            vec![vec![0, 1, 2], vec![3, 4, 5]],
            vec![vec![0, 1], vec![2, 3], vec![4, 5]],
        );

        // Both precision and recall should be between 0 and 1
        let precision = compute_bcubed_precision(&table);
        let recall = compute_bcubed_recall(&table);

        assert!(
            precision > 0.0 && precision < 1.0,
            "Partial overlap should yield 0 < B-cubed precision < 1, got {}",
            precision
        );
        assert!(
            recall > 0.0 && recall < 1.0,
            "Partial overlap should yield 0 < B-cubed recall < 1, got {}",
            recall
        );
    }

    #[test]
    fn test_bcubed_all_singletons() {
        // Scenario: All singletons (maximum precision)
        // Partition1: {0}, {1}, {2}, {3}
        // Partition2: {0,1}, {2,3}
        let table = create_test_contingency_table(
            vec![vec![0], vec![1], vec![2], vec![3]],
            vec![vec![0, 1], vec![2, 3]],
        );

        // Precision should be 1.0 (each singleton is pure)
        // Recall should be < 1.0 (not all true members are together)
        let precision = compute_bcubed_precision(&table);
        let recall = compute_bcubed_recall(&table);

        assert_eq!(
            precision, 1.0,
            "All singletons should yield B-cubed precision = 1.0, got {}",
            precision
        );
        assert!(
            recall < 1.0,
            "All singletons vs clusters should yield B-cubed recall < 1.0, got {}",
            recall
        );
    }
}
