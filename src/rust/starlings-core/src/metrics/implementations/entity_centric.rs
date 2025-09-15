//! Entity-centric metrics (B-cubed precision and recall)

use crate::expressions::contingency::SparseContingencyTable;

/// Compute B-cubed precision
pub fn compute_bcubed_precision(_table: &SparseContingencyTable) -> f64 {
    // TODO: Implement B-cubed precision
    // Average per-record precision
    0.0
}

/// Compute B-cubed recall
pub fn compute_bcubed_recall(_table: &SparseContingencyTable) -> f64 {
    // TODO: Implement B-cubed recall
    // Average per-record recall
    0.0
}
