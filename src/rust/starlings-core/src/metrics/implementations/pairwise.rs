//! Pairwise metrics (precision, recall, F1)

use crate::expressions::contingency::SparseContingencyTable;

/// Compute F1 score from contingency table
pub fn compute_f1(table: &SparseContingencyTable) -> f64 {
    table.compute_f1()
}

/// Compute precision from contingency table
pub fn compute_precision(table: &SparseContingencyTable) -> f64 {
    table.compute_precision()
}

/// Compute recall from contingency table
pub fn compute_recall(table: &SparseContingencyTable) -> f64 {
    table.compute_recall()
}
