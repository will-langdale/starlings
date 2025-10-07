"""Expression API for entity resolution analysis.

This module provides the expression API for creating and composing
entity resolution analysis queries, inspired by polars.col() pattern.

Example:
    Basic usage with single collection::

        import starlings as sl

        # Single collection at specific threshold
        sl.col("splink").at(0.85)

        # Single collection across threshold range
        sl.col("splink").sweep(0.5, 0.95, 0.01)

    Cross-collection comparison::

        # Compare two collections
        ef.analyse(
            sl.col("splink").sweep(0.8, 0.9, 0.01),
            sl.col("dedupe").at(0.85),
            metrics=[sl.Metrics.eval.f1],
        )
"""

from __future__ import annotations

from typing import Any

# Constants for sweep constraints
MIN_SWEEP_STEP = 0.05  # Minimum step size for sweep operations
STEP_ROUNDING_FACTOR = 0.05  # Round steps to nearest 0.05


class Expression:
    """Represents a query expression for entity resolution analysis.

    An expression defines how to query a collection (either at a specific
    threshold or across a range) and can optionally be marked as a reference
    for asymmetric metrics.

    Attributes:
        expression_type: Type of expression ("point" or "sweep")
        params: Dictionary of parameters for the expression
        is_reference: Whether this collection is the ground truth reference
    """

    def __init__(self, expression_type: str, **params: Any) -> None:
        """Create an expression with the specified type and parameters.

        Args:
            expression_type: Either "point" or "sweep"
            **params: Expression-specific parameters (threshold, start/stop/step, etc.)
        """
        self.expression_type = expression_type
        self.params = params
        self.is_reference = False

    def reference(self) -> Expression:
        """Mark this collection as the reference (ground truth) for asymmetric metrics.

        When computing asymmetric metrics (precision, recall, F1), one collection
        must be designated as the reference or ground truth. This method marks
        the current expression as the reference.

        Returns:
            Self for method chaining

        Example:
            ```python
            # Explicit reference marking
            ef.analyse(
                sl.col("splink").sweep(0.8, 0.9, 0.1),
                sl.col("truth").at(1.0).reference(),
                metrics=[sl.Metrics.eval.recall],
            )

            # Implicit reference (last expression)
            ef.analyse(
                sl.col("splink").sweep(0.8, 0.9, 0.1),
                sl.col("truth").at(1.0),  # Becomes reference implicitly
                metrics=[sl.Metrics.eval.recall],
            )
            ```
        """
        self.is_reference = True
        return self

    def __repr__(self) -> str:
        """String representation for debugging."""
        ref_str = " [reference]" if self.is_reference else ""
        return f"Expression({self.expression_type}, {self.params}){ref_str}"


class ColExpression:
    """Column expression for referencing collections by name.

    Inspired by polars.col() for consistent API design. Provides methods
    to specify how the collection should be queried (at specific threshold
    or across a range).
    """

    def __init__(self, name: str) -> None:
        """Create a column expression for the named collection.

        Args:
            name: Name of the collection to reference
        """
        self.name = name

    def at(self, threshold: float) -> Expression:
        """Specify a single threshold for this collection.

        Args:
            threshold: Threshold value between 0.0 and 1.0

        Returns:
            Expression representing a point query at the specified threshold

        Example:
            ```python
            sl.col("splink").at(0.85)
            ```
        """
        return Expression("point", collection=self.name, threshold=threshold)

    def sweep(self, start: float, stop: float, step: float = 0.05) -> Expression:
        """Specify a threshold range for sweeping analysis.

        For performance at production scale (1M+ edges), step sizes are constrained:
        - Minimum step: 0.05
        - Steps are rounded to nearest 0.05 multiple
        - Example: step=0.01 becomes step=0.05, step=0.07 becomes step=0.05
        - Rationale: Prevents excessive threshold points in large-scale sweeps

        Args:
            start: Starting threshold (inclusive)
            stop: Ending threshold (inclusive)
            step: Step size between thresholds (minimum 0.05, default 0.05)

        Returns:
            Expression representing a sweep query across the threshold range

        Example:
            ```python
            sl.col("splink").sweep(0.5, 0.95, 0.05)  # Recommended for large datasets
            sl.col("splink").sweep(0.5, 0.95, 0.1)  # Faster, coarser granularity
            ```

        Note:
            Steps smaller than 0.05 are automatically rounded up to 0.05.
            This ensures reasonable performance at scale without excessive computations.
        """
        # Enforce minimum step for performance
        if step < MIN_SWEEP_STEP:
            step = MIN_SWEEP_STEP

        # Round step to nearest increment
        step = round(step / STEP_ROUNDING_FACTOR) * STEP_ROUNDING_FACTOR

        return Expression(
            "sweep", collection=self.name, start=start, stop=stop, step=step
        )

    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"col('{self.name}')"


def col(name: str) -> ColExpression:
    """Create a collection expression for analysis.

    This function follows the polars.col() pattern for consistent API design
    across data processing libraries.

    Args:
        name: Name of the collection to reference

    Returns:
        ColExpression that can be used with .at() or .sweep() methods

    Example:
        ```python
        # Reference a collection for analysis
        expr = sl.col("splink")

        # Use with specific threshold
        point_expr = sl.col("splink").at(0.85)

        # Use with threshold range
        sweep_expr = sl.col("splink").sweep(0.5, 0.95, 0.01)
        ```
    """
    return ColExpression(name)


class MetricFunction:
    """Base class for metric functions."""

    def __init__(self, name: str, metric_type: str) -> None:
        """Initialise metric function.

        Args:
            name: Name of the metric (e.g., "f1", "precision")
            metric_type: Type of metric ("evaluation" or "statistics")
        """
        self.name = name
        self.metric_type = metric_type

    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"Metric({self.name})"
