//! Core expression and metric types

/// Represents an expression operation type
#[derive(Debug, Clone, PartialEq)]
pub enum ExpressionType {
    /// Single threshold query
    Point {
        collection: String,
        threshold: f64,
        is_reference: bool,
    },
    /// Threshold range query
    Sweep {
        collection: String,
        start: f64,
        stop: f64,
        step: f64,
        is_reference: bool,
    },
}

impl ExpressionType {
    /// Get the collection name for this expression
    pub fn collection(&self) -> &str {
        match self {
            ExpressionType::Point { collection, .. } => collection,
            ExpressionType::Sweep { collection, .. } => collection,
        }
    }

    /// Check if this expression is marked as reference
    pub fn is_reference(&self) -> bool {
        match self {
            ExpressionType::Point { is_reference, .. } => *is_reference,
            ExpressionType::Sweep { is_reference, .. } => *is_reference,
        }
    }
}

/// Represents a metric type for computation
#[derive(Debug, Clone, PartialEq)]
pub enum MetricType {
    // Evaluation metrics (require 2+ collections)
    F1,
    Precision,
    Recall,
    #[allow(clippy::upper_case_acronyms)]
    ARI,
    #[allow(clippy::upper_case_acronyms)]
    NMI,
    VMeasure,
    BCubedPrecision,
    BCubedRecall,

    // Statistics metrics (single collection)
    EntityCount,
    Entropy,
}

impl MetricType {
    /// Check if this metric requires multiple collections
    pub fn requires_comparison(&self) -> bool {
        matches!(
            self,
            MetricType::F1
                | MetricType::Precision
                | MetricType::Recall
                | MetricType::ARI
                | MetricType::NMI
                | MetricType::VMeasure
                | MetricType::BCubedPrecision
                | MetricType::BCubedRecall
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metric_type_requires_comparison() {
        assert!(MetricType::F1.requires_comparison());
        assert!(MetricType::Precision.requires_comparison());
        assert!(MetricType::Recall.requires_comparison());

        assert!(!MetricType::EntityCount.requires_comparison());
        assert!(!MetricType::Entropy.requires_comparison());
    }

    #[test]
    fn test_expression_type_accessors() {
        let point = ExpressionType::Point {
            collection: "test".to_string(),
            threshold: 0.8,
            is_reference: true,
        };
        assert_eq!(point.collection(), "test");
        assert!(point.is_reference());

        let sweep = ExpressionType::Sweep {
            collection: "other".to_string(),
            start: 0.5,
            stop: 0.9,
            step: 0.1,
            is_reference: false,
        };
        assert_eq!(sweep.collection(), "other");
        assert!(!sweep.is_reference());
    }
}
