//! Implementations of specific metric calculations
//!
//! This module contains the actual mathematical implementations of various metrics,
//! separated from the algorithmic strategies used to compute them efficiently.

pub mod clustering;
pub mod entity_centric;
pub mod pairwise;
pub mod statistics;

// Re-export commonly used functions
pub use clustering::{compute_ari, compute_nmi, compute_v_measure};
pub use entity_centric::{compute_bcubed_precision, compute_bcubed_recall};
pub use pairwise::{compute_f1, compute_precision, compute_recall};
pub use statistics::{compute_entity_count, compute_entropy};
