//! Contingency-based metric computation
//!
//! This module defines a unified interface for computing entity resolution metrics
//! from contingency tables. We have TWO implementations because our algorithms
//! use different ID systems for optimal performance:
//!
//! - **DeltaState**: Uses canonical IDs (min record ID per entity) for O(k) incremental updates
//! - **RecordState**: Uses sequential indices for O(r) single-pass construction
//!
//! Both compute identical metrics but from different data structures.
//! This avoids expensive conversions and maintains optimal complexity for each algorithm.

use std::collections::HashMap;

/// Trait for computing metrics from contingency table data
///
/// Implemented by both DeltaState and RecordState to provide consistent
/// metric computation while allowing each to use its optimal data structure.
pub trait ContingencyMetrics {
    /// Compute precision (fraction of predicted pairs that are correct)
    fn compute_precision(&self) -> f64;

    /// Compute recall (fraction of true pairs that were found)
    fn compute_recall(&self) -> f64;

    /// Compute F1 score (harmonic mean of precision and recall)
    fn compute_f1(&self) -> f64;

    /// Compute Adjusted Rand Index
    fn compute_ari(&self) -> f64;

    /// Compute Normalised Mutual Information
    fn compute_nmi(&self) -> f64;

    /// Compute V-Measure (harmonic mean of homogeneity and completeness)
    fn compute_v_measure(&self) -> f64;

    /// Compute BCubed precision
    fn compute_bcubed_precision(&self) -> f64;

    /// Compute BCubed recall
    fn compute_bcubed_recall(&self) -> f64;
}

// Shared helper functions for entropy calculations used by both implementations

/// Compute entropy for a set of cluster sizes
pub fn compute_entropy_from_sizes<I>(sizes: I, total: f64) -> f64
where
    I: Iterator<Item = u32>,
{
    if total <= 0.0 {
        return 0.0;
    }

    let mut entropy = 0.0;
    for size in sizes {
        if size > 0 {
            let p = size as f64 / total;
            entropy -= p * p.log2();
        }
    }
    entropy
}

/// Compute conditional entropy H(X|Y) for nested HashMap structure (used by DeltaAlgorithm)
pub fn compute_conditional_entropy<K1, K2>(
    contingency: &HashMap<K1, HashMap<K2, u32>>,
    col_marginals: &HashMap<K2, u32>,
    total: f64,
) -> f64
where
    K1: Eq + std::hash::Hash + Copy,
    K2: Eq + std::hash::Hash + Copy,
{
    if total <= 0.0 {
        return 0.0;
    }

    let mut conditional_entropy = 0.0;

    // Group contingency cells by column (partition2)
    let mut cells_by_col: HashMap<K2, Vec<(K1, u32)>> = HashMap::new();
    for (&k1, row) in contingency {
        for (&k2, &count) in row {
            if count > 0 {
                cells_by_col.entry(k2).or_default().push((k1, count));
            }
        }
    }

    // Compute conditional entropy
    for (col_id, cells) in cells_by_col {
        if let Some(&col_size) = col_marginals.get(&col_id) {
            if col_size > 0 {
                let p_y = col_size as f64 / total;
                let mut h_x_given_y = 0.0;

                for (_, count) in cells {
                    let p_x_given_y = count as f64 / col_size as f64;
                    if p_x_given_y > 0.0 {
                        h_x_given_y -= p_x_given_y * p_x_given_y.log2();
                    }
                }

                conditional_entropy += p_y * h_x_given_y;
            }
        }
    }

    conditional_entropy
}

/// Compute conditional entropy H(X|Y) for flat HashMap structure (used by RecordAlgorithm)
pub fn compute_conditional_entropy_flat<K1, K2>(
    contingency: &HashMap<(K1, K2), u32>,
    col_marginals: &HashMap<K2, u32>,
    total: f64,
) -> f64
where
    K1: Eq + std::hash::Hash + Copy,
    K2: Eq + std::hash::Hash + Copy,
{
    if total <= 0.0 {
        return 0.0;
    }

    let mut conditional_entropy = 0.0;

    // Group contingency cells by column (partition2)
    let mut cells_by_col: HashMap<K2, Vec<(K1, u32)>> = HashMap::new();
    for (&(k1, k2), &count) in contingency {
        if count > 0 {
            cells_by_col.entry(k2).or_default().push((k1, count));
        }
    }

    // Compute conditional entropy
    for (col_id, cells) in cells_by_col {
        if let Some(&col_size) = col_marginals.get(&col_id) {
            if col_size > 0 {
                let p_y = col_size as f64 / total;
                let mut h_x_given_y = 0.0;

                for (_, count) in cells {
                    let p_x_given_y = count as f64 / col_size as f64;
                    if p_x_given_y > 0.0 {
                        h_x_given_y -= p_x_given_y * p_x_given_y.log2();
                    }
                }

                conditional_entropy += p_y * h_x_given_y;
            }
        }
    }

    conditional_entropy
}

/// Helper for computing the binomial coefficient "n choose 2"
#[inline]
pub fn choose_2(n: u32) -> u64 {
    if n < 2 {
        0
    } else {
        (n as u64 * (n as u64 - 1)) / 2
    }
}
