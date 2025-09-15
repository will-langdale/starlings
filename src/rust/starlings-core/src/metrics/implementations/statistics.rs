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
