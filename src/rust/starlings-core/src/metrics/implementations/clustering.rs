//! Clustering metrics (ARI, NMI, V-measure)

use crate::expressions::contingency::SparseContingencyTable;

/// Compute Adjusted Rand Index
///
/// ARI measures the similarity between two partitions, adjusted for chance.
/// Returns a value between -1 and 1, where:
/// - 1.0 indicates perfect agreement
/// - 0.0 indicates random clustering
/// - Negative values indicate worse than random agreement
pub fn compute_ari(table: &SparseContingencyTable) -> f64 {
    // Delegate to the implementation on SparseContingencyTable
    table.compute_ari()
}

/// Compute Normalised Mutual Information
///
/// NMI measures the mutual information between two partitions, normalised by their entropies.
/// Returns a value between 0 and 1, where:
/// - 1.0 indicates perfect agreement
/// - 0.0 indicates no mutual information
pub fn compute_nmi(table: &SparseContingencyTable) -> f64 {
    // Delegate to the implementation on SparseContingencyTable
    table.compute_nmi()
}

/// Compute V-measure (harmonic mean of homogeneity and completeness)
///
/// V-measure balances homogeneity (each cluster contains only members of a single class)
/// and completeness (all members of a class are assigned to the same cluster).
/// Returns a value between 0 and 1, where:
/// - 1.0 indicates perfect clustering
/// - 0.0 indicates poor clustering
pub fn compute_v_measure(table: &SparseContingencyTable) -> f64 {
    // Delegate to the implementation on SparseContingencyTable
    table.compute_v_measure()
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
    fn test_ari_perfect_agreement() {
        // Scenario: Both partitions are identical
        // Partition1: {0,1}, {2,3}, {4,5}
        // Partition2: {0,1}, {2,3}, {4,5}
        let table = create_test_contingency_table(
            vec![vec![0, 1], vec![2, 3], vec![4, 5]],
            vec![vec![0, 1], vec![2, 3], vec![4, 5]],
        );

        // Expected: ARI = 1.0 (perfect agreement)
        let ari = compute_ari(&table);
        assert!(
            (ari - 1.0).abs() < 1e-10,
            "Perfect agreement should yield ARI = 1.0, got {}",
            ari
        );
    }

    #[test]
    fn test_ari_all_singletons_vs_single_cluster() {
        // Scenario: Maximum disagreement
        // Partition1: {0}, {1}, {2}, {3}, {4}, {5} (all singletons)
        // Partition2: {0,1,2,3,4,5} (single cluster)
        let table = create_test_contingency_table(
            vec![vec![0], vec![1], vec![2], vec![3], vec![4], vec![5]],
            vec![vec![0, 1, 2, 3, 4, 5]],
        );

        // Calculation:
        // - Cell combinations: 6 cells with count=1 each, sum = 0
        // - Row combinations: 6 singletons, sum = 0
        // - Column combinations: 1 cluster of 6, choose_2(6) = 15
        // - Total combinations: choose_2(6) = 15
        // - Expected = (0 * 15) / 15 = 0
        // - Max = (0 + 15) / 2 = 7.5
        // - ARI = (0 - 0) / (7.5 - 0) = 0
        let ari = compute_ari(&table);
        assert!(
            ari.abs() < 1e-10,
            "All singletons vs single cluster should yield ARI ≈ 0, got {}",
            ari
        );
    }

    #[test]
    fn test_ari_partial_overlap() {
        // Scenario: Partial overlap between partitions
        // Partition1: {0,1,2}, {3,4,5}
        // Partition2: {0,1}, {2,3}, {4,5}
        let table = create_test_contingency_table(
            vec![vec![0, 1, 2], vec![3, 4, 5]],
            vec![vec![0, 1], vec![2, 3], vec![4, 5]],
        );

        // Manual calculation:
        // Contingency table:
        //           | {0,1} | {2,3} | {4,5} |
        // {0,1,2}   |   2   |   1   |   0   | = 3
        // {3,4,5}   |   0   |   1   |   2   | = 3
        //           |   2   |   2   |   2   | = 6
        //
        // Cell combinations: choose_2(2) + choose_2(1) + choose_2(1) + choose_2(2) = 1 + 0 + 0 + 1 = 2
        // Row combinations: choose_2(3) + choose_2(3) = 3 + 3 = 6
        // Column combinations: choose_2(2) + choose_2(2) + choose_2(2) = 1 + 1 + 1 = 3
        // Total combinations: choose_2(6) = 15
        // Expected = (6 * 3) / 15 = 18 / 15 = 1.2
        // Max = (6 + 3) / 2 = 4.5
        // ARI = (2 - 1.2) / (4.5 - 1.2) = 0.8 / 3.3 ≈ 0.242
        let ari = compute_ari(&table);
        let expected = 0.8 / 3.3;
        assert!(
            (ari - expected).abs() < 0.01,
            "Partial overlap ARI mismatch. Got {}, expected approximately {}",
            ari,
            expected
        );
    }

    #[test]
    fn test_ari_symmetry() {
        // Test that ARI(P1, P2) = ARI(P2, P1)
        let partition1 = vec![vec![0, 1], vec![2, 3, 4], vec![5]];
        let partition2 = vec![vec![0, 2], vec![1, 3], vec![4, 5]];

        let table1 = create_test_contingency_table(partition1.clone(), partition2.clone());
        let table2 = create_test_contingency_table(partition2, partition1);

        let ari1 = compute_ari(&table1);
        let ari2 = compute_ari(&table2);

        assert!(
            (ari1 - ari2).abs() < 1e-10,
            "ARI should be symmetric. ARI(P1,P2)={}, ARI(P2,P1)={}",
            ari1,
            ari2
        );
    }

    #[test]
    fn test_ari_empty_partition() {
        // Edge case: empty partition
        let table = create_test_contingency_table(vec![], vec![]);
        let ari = compute_ari(&table);
        assert_eq!(ari, 0.0, "Empty partitions should yield ARI = 0");
    }

    #[test]
    fn test_ari_single_record() {
        // Edge case: single record
        let table = create_test_contingency_table(vec![vec![0]], vec![vec![0]]);
        let ari = compute_ari(&table);
        assert_eq!(ari, 0.0, "Single record should yield ARI = 0");
    }

    #[test]
    fn test_ari_properties() {
        // Test that ARI satisfies its theoretical properties

        // Property 1: ARI is bounded between -1 and 1
        let test_cases = vec![
            // Various partition configurations
            (vec![vec![0, 1], vec![2, 3]], vec![vec![0, 2], vec![1, 3]]),
            (vec![vec![0], vec![1], vec![2]], vec![vec![0, 1, 2]]),
            (vec![vec![0, 1, 2]], vec![vec![0], vec![1], vec![2]]),
            (vec![vec![0, 1], vec![2]], vec![vec![0], vec![1, 2]]),
        ];

        for (partition1, partition2) in test_cases {
            let table = create_test_contingency_table(partition1, partition2);
            let ari = compute_ari(&table);
            assert!(
                (-1.0..=1.0).contains(&ari),
                "ARI should be in [-1, 1], got {}",
                ari
            );
        }

        // Property 2: ARI = 1 for identical partitions
        let identical_partition = vec![vec![0, 1], vec![2], vec![3, 4, 5]];
        let table = create_test_contingency_table(identical_partition.clone(), identical_partition);
        let ari = compute_ari(&table);
        assert!(
            (ari - 1.0).abs() < 1e-10,
            "Identical partitions should have ARI = 1.0, got {}",
            ari
        );
    }

    #[test]
    fn test_nmi_perfect_agreement() {
        // Scenario: Both partitions are identical
        // Partition1: {0,1}, {2,3}, {4,5}
        // Partition2: {0,1}, {2,3}, {4,5}
        let table = create_test_contingency_table(
            vec![vec![0, 1], vec![2, 3], vec![4, 5]],
            vec![vec![0, 1], vec![2, 3], vec![4, 5]],
        );

        // Expected: NMI = 1.0 (perfect agreement)
        let nmi = compute_nmi(&table);
        assert!(
            (nmi - 1.0).abs() < 1e-10,
            "Perfect agreement should yield NMI = 1.0, got {}",
            nmi
        );
    }

    #[test]
    fn test_nmi_no_information() {
        // Scenario: Partitions share no information
        // Partition1: {0}, {1}, {2}, {3}, {4}, {5} (all singletons)
        // Partition2: {0,1,2,3,4,5} (single cluster)
        let table = create_test_contingency_table(
            vec![vec![0], vec![1], vec![2], vec![3], vec![4], vec![5]],
            vec![vec![0, 1, 2, 3, 4, 5]],
        );

        // Expected: NMI = 0.0 (no mutual information)
        let nmi = compute_nmi(&table);
        assert_eq!(
            nmi, 0.0,
            "No mutual information should yield NMI = 0, got {}",
            nmi
        );
    }

    #[test]
    fn test_v_measure_perfect_clustering() {
        // Scenario: Perfect clustering
        // Partition1: {0,1}, {2,3}, {4,5}
        // Partition2: {0,1}, {2,3}, {4,5}
        let table = create_test_contingency_table(
            vec![vec![0, 1], vec![2, 3], vec![4, 5]],
            vec![vec![0, 1], vec![2, 3], vec![4, 5]],
        );

        // Expected: V-measure = 1.0 (perfect homogeneity and completeness)
        let v_measure = compute_v_measure(&table);
        assert!(
            (v_measure - 1.0).abs() < 1e-10,
            "Perfect clustering should yield V-measure = 1.0, got {}",
            v_measure
        );
    }

    #[test]
    fn test_v_measure_partial_overlap() {
        // Scenario: Partial overlap
        // Partition1: {0,1,2}, {3,4,5}
        // Partition2: {0,1}, {2,3}, {4,5}
        let table = create_test_contingency_table(
            vec![vec![0, 1, 2], vec![3, 4, 5]],
            vec![vec![0, 1], vec![2, 3], vec![4, 5]],
        );

        // V-measure should be between 0 and 1
        let v_measure = compute_v_measure(&table);
        assert!(
            v_measure > 0.0 && v_measure < 1.0,
            "Partial overlap should yield 0 < V-measure < 1, got {}",
            v_measure
        );
    }

    #[test]
    fn test_ari_random_like_partitions() {
        // Scenario: Two independent random-like partitions
        // Partition1: {0,2,4}, {1,3,5}
        // Partition2: {0,3}, {1,4}, {2,5}
        let table = create_test_contingency_table(
            vec![vec![0, 2, 4], vec![1, 3, 5]],
            vec![vec![0, 3], vec![1, 4], vec![2, 5]],
        );

        // Manual calculation:
        // Contingency table:
        //           | {0,3} | {1,4} | {2,5} |
        // {0,2,4}   |   1   |   1   |   1   | = 3
        // {1,3,5}   |   1   |   1   |   1   | = 3
        //           |   2   |   2   |   2   | = 6
        //
        // Cell combinations: 6 cells with count=1 each, sum = 0
        // Row combinations: choose_2(3) + choose_2(3) = 3 + 3 = 6
        // Column combinations: choose_2(2) + choose_2(2) + choose_2(2) = 1 + 1 + 1 = 3
        // Total combinations: choose_2(6) = 15
        // Expected = (6 * 3) / 15 = 18/15 = 1.2
        // Max = (6 + 3) / 2 = 4.5
        // ARI = (0 - 1.2) / (4.5 - 1.2) = -1.2 / 3.3 ≈ -0.364
        //
        // This negative value indicates the partitions are actually anti-correlated
        // (worse than random), which makes sense given the systematic misalignment
        let ari = compute_ari(&table);
        let expected = -1.2 / 3.3;
        assert!(
            (ari - expected).abs() < 0.01,
            "Anti-correlated partitions ARI mismatch. Got {}, expected approximately {}",
            ari,
            expected
        );
    }
}
