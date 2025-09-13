//! Expression API for entity resolution analysis
//!
//! This module provides core algorithms and data structures for cross-collection comparison,
//! including the optimised record-based algorithm that reduces complexity from
//! O(k₁ × k₂) to O(r) where k = entities and r = records.

pub mod algorithms;
pub mod contingency;
pub mod metrics;
pub mod types;

// Re-export main types for convenience
pub use algorithms::{
    build_all_sweep_tables, build_contingency_table, compute_comparison_metric,
    compute_single_metric, generate_sweep_thresholds, MetricError,
};
pub use contingency::{ContingencyTable, SparseContingencyTable};
pub use metrics::{compute_entity_count, compute_entropy};
pub use types::{ExpressionType, MetricType};
