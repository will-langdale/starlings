//! Statistical metrics for single partitions

use crate::PartitionLevel;

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

#[cfg(test)]
mod tests {
    use super::*;
    use roaring::RoaringBitmap;

    fn create_test_partition(entities: Vec<Vec<u32>>) -> PartitionLevel {
        let bitmap_entities: Vec<RoaringBitmap> = entities
            .into_iter()
            .map(|records| records.into_iter().collect())
            .collect();
        PartitionLevel::new(0.5, bitmap_entities)
    }

    #[test]
    fn test_entity_count_empty_partition() {
        // Scenario: Empty partition with no entities
        let partition = create_test_partition(vec![]);

        // Expected: 0 entities
        let count = compute_entity_count(&partition);
        assert_eq!(count, 0.0, "Empty partition should have 0 entities");
    }

    #[test]
    fn test_entity_count_single_entity() {
        // Scenario: One entity containing records 0, 1, 2
        let partition = create_test_partition(vec![vec![0, 1, 2]]);

        // Expected: 1 entity (the count is number of entities, not records)
        let count = compute_entity_count(&partition);
        assert_eq!(
            count, 1.0,
            "Partition with one entity should return count of 1"
        );
    }

    #[test]
    fn test_entity_count_multiple_entities() {
        // Scenario: 4 entities with varying sizes
        // Entity 0: records [0, 1]
        // Entity 1: records [2, 3, 4]
        // Entity 2: records [5]
        // Entity 3: records [6, 7]
        let partition = create_test_partition(vec![vec![0, 1], vec![2, 3, 4], vec![5], vec![6, 7]]);

        // Expected: 4 entities (regardless of their sizes)
        let count = compute_entity_count(&partition);
        assert_eq!(count, 4.0, "Should count 4 distinct entities");
    }

    #[test]
    fn test_entropy_single_entity() {
        // Scenario: All 4 records in one entity [0, 1, 2, 3]
        // This represents complete clustering (minimum entropy)
        let partition = create_test_partition(vec![vec![0, 1, 2, 3]]);

        // Calculation:
        // - One entity with proportion 4/4 = 1.0
        // - Entropy = -1.0 * log2(1.0) = -1.0 * 0 = 0.0
        // Expected: 0.0 (minimum entropy - complete order)
        let entropy = compute_entropy(&partition);
        assert_eq!(entropy, 0.0, "Single entity should have zero entropy");
    }

    #[test]
    fn test_entropy_all_singletons() {
        // Scenario: 4 singleton entities (maximum entropy for 4 records)
        // Entity 0: [0], Entity 1: [1], Entity 2: [2], Entity 3: [3]
        let partition = create_test_partition(vec![vec![0], vec![1], vec![2], vec![3]]);

        // Calculation:
        // - 4 entities, each with proportion 1/4 = 0.25
        // - Each contributes: -0.25 * log2(0.25) = -0.25 * (-2) = 0.5
        // - Total entropy = 4 * 0.5 = 2.0
        // Expected: 2.0 (which equals log2(4) - maximum entropy)
        let entropy = compute_entropy(&partition);
        let expected = 2.0; // log2(4) = 2.0
        assert!(
            (entropy - expected).abs() < 1e-10,
            "All singletons should have maximum entropy of log2(n). Got {}, expected {}",
            entropy,
            expected
        );
    }

    #[test]
    fn test_entropy_uniform_distribution() {
        // Scenario: 2 entities of equal size
        // Entity 0: [0, 1], Entity 1: [2, 3]
        let partition = create_test_partition(vec![vec![0, 1], vec![2, 3]]);

        // Calculation:
        // - Entity 0: proportion = 2/4 = 0.5, contributes -0.5 * log2(0.5) = -0.5 * (-1) = 0.5
        // - Entity 1: proportion = 2/4 = 0.5, contributes -0.5 * log2(0.5) = -0.5 * (-1) = 0.5
        // - Total entropy = 0.5 + 0.5 = 1.0
        // Expected: 1.0
        let entropy = compute_entropy(&partition);
        assert!(
            (entropy - 1.0).abs() < 1e-10,
            "Two equal-sized entities should have entropy of 1.0. Got {}",
            entropy
        );
    }

    #[test]
    fn test_entropy_skewed_distribution() {
        // Scenario: Highly skewed distribution
        // Entity 0: [0, 1, 2, 3, 4, 5, 6] (7 records)
        // Entity 1: [7] (1 record)
        let partition = create_test_partition(vec![vec![0, 1, 2, 3, 4, 5, 6], vec![7]]);

        // Calculation:
        // - Entity 0: proportion = 7/8 = 0.875
        //   contributes: -0.875 * log2(0.875) ≈ -0.875 * (-0.193) ≈ 0.169
        // - Entity 1: proportion = 1/8 = 0.125
        //   contributes: -0.125 * log2(0.125) = -0.125 * (-3) = 0.375
        // - Total entropy ≈ 0.169 + 0.375 ≈ 0.544
        let entropy = compute_entropy(&partition);

        // Manual calculation for verification
        let p1: f64 = 7.0 / 8.0;
        let p2: f64 = 1.0 / 8.0;
        let expected = -(p1 * p1.log2() + p2 * p2.log2());

        assert!(
            (entropy - expected).abs() < 1e-10,
            "Skewed distribution entropy mismatch. Got {}, expected {}",
            entropy,
            expected
        );
    }

    #[test]
    fn test_entropy_three_entities_mixed_sizes() {
        // Scenario: 3 entities with different sizes for comprehensive test
        // Entity 0: [0, 1, 2] (3 records)
        // Entity 1: [3, 4] (2 records)
        // Entity 2: [5] (1 record)
        // Total: 6 records
        let partition = create_test_partition(vec![vec![0, 1, 2], vec![3, 4], vec![5]]);

        // Calculation:
        // - Entity 0: p = 3/6 = 0.5, contributes -0.5 * log2(0.5) = 0.5
        // - Entity 1: p = 2/6 = 1/3, contributes -(1/3) * log2(1/3) ≈ 0.528
        // - Entity 2: p = 1/6, contributes -(1/6) * log2(1/6) ≈ 0.431
        // - Total entropy ≈ 0.5 + 0.528 + 0.431 ≈ 1.459
        let entropy = compute_entropy(&partition);

        // Manual calculation
        let p1: f64 = 3.0 / 6.0;
        let p2: f64 = 2.0 / 6.0;
        let p3: f64 = 1.0 / 6.0;
        let expected = -(p1 * p1.log2() + p2 * p2.log2() + p3 * p3.log2());

        assert!(
            (entropy - expected).abs() < 1e-10,
            "Three entities entropy mismatch. Got {}, expected {}",
            entropy,
            expected
        );
    }

    #[test]
    fn test_entropy_empty_partition() {
        // Scenario: Empty partition (edge case)
        let partition = create_test_partition(vec![]);

        // Expected: 0.0 (by convention, empty partition has zero entropy)
        let entropy = compute_entropy(&partition);
        assert_eq!(entropy, 0.0, "Empty partition should have zero entropy");
    }

    #[test]
    fn test_entropy_properties() {
        // Test that entropy follows expected properties:
        // 1. Entropy is non-negative
        // 2. Entropy increases as distribution becomes more uniform
        // 3. Maximum entropy when all entities are singletons

        // Test various partition sizes
        for n in [2, 4, 8, 16] {
            // All singletons (maximum entropy)
            let singletons: Vec<Vec<u32>> = (0..n).map(|i| vec![i]).collect();
            let max_entropy_partition = create_test_partition(singletons);
            let max_entropy = compute_entropy(&max_entropy_partition);
            let expected_max = (n as f64).log2();

            assert!(
                (max_entropy - expected_max).abs() < 1e-10,
                "Maximum entropy for {} records should be log2({}) = {}. Got {}",
                n,
                n,
                expected_max,
                max_entropy
            );

            // Single entity (minimum entropy)
            let single_entity: Vec<u32> = (0..n).collect();
            let min_entropy_partition = create_test_partition(vec![single_entity]);
            let min_entropy = compute_entropy(&min_entropy_partition);

            assert_eq!(
                min_entropy, 0.0,
                "Minimum entropy should be 0 for single entity with {} records",
                n
            );

            // Verify max > min
            assert!(
                max_entropy > min_entropy,
                "Maximum entropy should exceed minimum entropy"
            );
        }
    }
}
