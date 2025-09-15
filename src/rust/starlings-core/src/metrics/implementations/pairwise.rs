//! Pairwise metrics (precision, recall, F1)

use crate::expressions::contingency::ContingencyTable;

/// Compute F1 score from contingency table
pub fn compute_f1(table: &ContingencyTable) -> f64 {
    table.f1_score()
}

/// Compute precision from contingency table
pub fn compute_precision(table: &ContingencyTable) -> f64 {
    table.precision()
}

/// Compute recall from contingency table
pub fn compute_recall(table: &ContingencyTable) -> f64 {
    table.recall()
}
