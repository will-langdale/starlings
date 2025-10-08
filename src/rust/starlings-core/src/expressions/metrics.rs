//! Metric computation functions for entity resolution analysis

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
        PartitionLevel::new(0.0, bitmap_entities)
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

        // Empty partition
        let empty_partition = create_test_partition(vec![]);
        assert_eq!(compute_entropy(&empty_partition), 0.0);
    }
}
