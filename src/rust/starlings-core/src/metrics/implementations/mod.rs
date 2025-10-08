//! Implementations of specific metric calculations
//!
//! This module contains the actual mathematical implementations of various metrics,
//! separated from the algorithmic strategies used to compute them efficiently.

pub mod statistics;

// Re-export commonly used functions
pub use statistics::{compute_entity_count, compute_entropy};
