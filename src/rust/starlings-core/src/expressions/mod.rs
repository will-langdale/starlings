//! Expression API for entity resolution analysis
//!
//! This module provides core algorithms and data structures for cross-collection comparison,
//! including the optimised record-based algorithm that reduces complexity from
//! O(k₁ × k₂) to O(r) where k = entities and r = records.

pub mod metrics;
pub mod types;
pub mod utils;

// Re-export main types for convenience
pub use metrics::{compute_entity_count, compute_entropy};
pub use types::{ExpressionType, MetricType};
pub use utils::{compute_single_metric, generate_sweep_thresholds, MetricError};
