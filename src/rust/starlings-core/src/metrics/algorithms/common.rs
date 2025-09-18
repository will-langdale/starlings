//! Common utilities shared between Delta and Record algorithms
//!
//! This module contains mathematical operations and helper functions
//! used by both fundamental metric algorithms.

use std::collections::HashMap;

/// Compute binomial coefficient "n choose 2"
///
/// This is used extensively in metric calculations for counting
/// pairs of records within entities.
#[inline]
pub fn choose_2(n: u32) -> u64 {
    if n < 2 {
        0
    } else {
        (n as u64) * ((n - 1) as u64) / 2
    }
}

/// Pre-compute pair counts for a set of entity sizes
///
/// Returns the sum of C(size, 2) for all entities, which is used
/// in ARI and other clustering metric calculations.
pub fn sum_entity_pair_counts(entity_sizes: &[u32]) -> u64 {
    entity_sizes.iter().map(|&size| choose_2(size)).sum()
}

/// Compute pair counts from marginals and overlaps
///
/// This is the core computation shared by both algorithms for
/// converting contingency tables into pair-based metrics.
pub struct PairCounts {
    pub true_positives: u64,
    pub false_positives: u64,
    pub false_negatives: u64,
    pub true_negatives: u64,
}

impl PairCounts {
    /// Compute pair counts from contingency table data
    pub fn from_contingency_data(
        overlaps: &HashMap<(usize, usize), u32>,
        row_marginals: &HashMap<usize, u32>,
        col_marginals: &HashMap<usize, u32>,
        total_records: u32,
    ) -> Self {
        let mut true_positives = 0u64;
        let mut entity1_pairs_together: HashMap<usize, u64> = HashMap::new();
        let mut entity2_pairs_together: HashMap<usize, u64> = HashMap::new();

        // Single pass through overlaps to compute true positives
        for ((e1, e2), &overlap) in overlaps {
            if overlap > 1 {
                let pairs = choose_2(overlap);
                true_positives += pairs;
                *entity1_pairs_together.entry(*e1).or_insert(0) += pairs;
                *entity2_pairs_together.entry(*e2).or_insert(0) += pairs;
            }
        }

        // False positives: pairs together in partition1, apart in partition2
        let false_positives: u64 = row_marginals
            .iter()
            .filter(|(_, &size)| size > 1)
            .map(|(entity_id, &size)| {
                let all_pairs = choose_2(size);
                let pairs_also_together =
                    entity1_pairs_together.get(entity_id).copied().unwrap_or(0);
                all_pairs.saturating_sub(pairs_also_together)
            })
            .sum();

        // False negatives: pairs apart in partition1, together in partition2
        let false_negatives: u64 = col_marginals
            .iter()
            .filter(|(_, &size)| size > 1)
            .map(|(entity_id, &size)| {
                let all_pairs = choose_2(size);
                let pairs_also_together =
                    entity2_pairs_together.get(entity_id).copied().unwrap_or(0);
                all_pairs.saturating_sub(pairs_also_together)
            })
            .sum();

        // True negatives: all remaining pairs
        let total_pairs = choose_2(total_records);
        let true_negatives =
            total_pairs.saturating_sub(true_positives + false_positives + false_negatives);

        Self {
            true_positives,
            false_positives,
            false_negatives,
            true_negatives,
        }
    }

    /// Compute F1 score from pair counts
    #[inline]
    pub fn f1_score(&self) -> f64 {
        let precision = self.precision();
        let recall = self.recall();
        if precision + recall == 0.0 {
            0.0
        } else {
            2.0 * (precision * recall) / (precision + recall)
        }
    }

    /// Compute precision from pair counts
    #[inline]
    pub fn precision(&self) -> f64 {
        let denominator = self.true_positives + self.false_positives;
        if denominator == 0 {
            if self.false_negatives > 0 {
                0.0
            } else {
                1.0
            }
        } else {
            self.true_positives as f64 / denominator as f64
        }
    }

    /// Compute recall from pair counts
    #[inline]
    pub fn recall(&self) -> f64 {
        let denominator = self.true_positives + self.false_negatives;
        if denominator == 0 {
            1.0
        } else {
            self.true_positives as f64 / denominator as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_choose_2() {
        assert_eq!(choose_2(0), 0);
        assert_eq!(choose_2(1), 0);
        assert_eq!(choose_2(2), 1);
        assert_eq!(choose_2(3), 3);
        assert_eq!(choose_2(4), 6);
        assert_eq!(choose_2(10), 45);
    }

    #[test]
    fn test_pair_counts_computation() {
        let mut overlaps = HashMap::new();
        overlaps.insert((0, 0), 3); // 3 records in entity 0,0
        overlaps.insert((0, 1), 1); // 1 record in entity 0,1
        overlaps.insert((1, 0), 2); // 2 records in entity 1,0

        let mut row_marginals = HashMap::new();
        row_marginals.insert(0, 4); // Entity 0 in partition1 has 4 records
        row_marginals.insert(1, 2); // Entity 1 in partition1 has 2 records

        let mut col_marginals = HashMap::new();
        col_marginals.insert(0, 5); // Entity 0 in partition2 has 5 records
        col_marginals.insert(1, 1); // Entity 1 in partition2 has 1 record

        let counts = PairCounts::from_contingency_data(
            &overlaps,
            &row_marginals,
            &col_marginals,
            6, // total records
        );

        // Verify counts are computed correctly
        assert_eq!(counts.true_positives, 4); // C(3,2) + C(2,2) = 3 + 1 = 4
                                              // Further validation would require manual calculation
    }
}
