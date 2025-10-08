//! PyO3 wrapper for expression API - thin layer over starlings-core
//!
//! This module only handles Python object parsing and conversion.
//! All business logic is in starlings_core::expressions.

use pyo3::prelude::*;
use starlings_core::expressions::{ExpressionType, MetricType};

/// Parse Python expression object into Rust expression type
pub fn parse_expression(py_expr: &Bound<'_, PyAny>) -> PyResult<ExpressionType> {
    // Extract expression_type attribute
    let expr_type: String = py_expr.getattr("expression_type")?.extract()?;

    // Extract is_reference attribute (defaults to false if not present)
    let is_reference: bool = py_expr
        .getattr("is_reference")
        .and_then(|attr| attr.extract())
        .unwrap_or(false);

    match expr_type.as_str() {
        "point" => {
            let params = py_expr.getattr("params")?;
            let collection: String = params.get_item("collection")?.extract()?;
            let threshold: f64 = params.get_item("threshold")?.extract()?;

            Ok(ExpressionType::Point {
                collection,
                threshold,
                is_reference,
            })
        }
        "sweep" => {
            let params = py_expr.getattr("params")?;
            let collection: String = params.get_item("collection")?.extract()?;
            let start: f64 = params.get_item("start")?.extract()?;
            let stop: f64 = params.get_item("stop")?.extract()?;
            let step: f64 = params.get_item("step")?.extract()?;

            Ok(ExpressionType::Sweep {
                collection,
                start,
                stop,
                step,
                is_reference,
            })
        }
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Unknown expression type: {}",
            expr_type
        ))),
    }
}

/// Parse Python metric function into Rust metric type
pub fn parse_metric(py_metric: &Bound<'_, PyAny>) -> PyResult<MetricType> {
    // Handle both string metrics and metric objects with .name attribute
    let name: String = if let Ok(s) = py_metric.extract::<String>() {
        s
    } else {
        py_metric.getattr("name")?.extract()?
    };

    match name.as_str() {
        "f1" => Ok(MetricType::F1),
        "precision" => Ok(MetricType::Precision),
        "recall" => Ok(MetricType::Recall),
        "ari" => Ok(MetricType::ARI),
        "nmi" => Ok(MetricType::NMI),
        "v_measure" => Ok(MetricType::VMeasure),
        "bcubed_precision" => Ok(MetricType::BCubedPrecision),
        "bcubed_recall" => Ok(MetricType::BCubedRecall),
        "entity_count" => Ok(MetricType::EntityCount),
        "entropy" => Ok(MetricType::Entropy),
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Unknown metric: {}",
            name
        ))),
    }
}
