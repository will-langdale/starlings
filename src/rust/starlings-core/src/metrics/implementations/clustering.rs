//! Clustering metrics (ARI, NMI, V-measure)

use crate::expressions::contingency::SparseContingencyTable;

/// Compute Adjusted Rand Index
pub fn compute_ari(_table: &SparseContingencyTable) -> f64 {
    // TODO: Implement ARI calculation
    // ARI = (Index - Expected) / (Max - Expected)
    0.0
}

/// Compute Normalised Mutual Information
pub fn compute_nmi(_table: &SparseContingencyTable) -> f64 {
    // TODO: Implement NMI calculation
    // NMI = 2 * I(U;V) / (H(U) + H(V))
    0.0
}

/// Compute V-measure (harmonic mean of homogeneity and completeness)
pub fn compute_v_measure(_table: &SparseContingencyTable) -> f64 {
    // TODO: Implement V-measure calculation
    0.0
}
