"""Expression API for Starlings analysis operations.

This module provides a Polars-inspired expression API for analysing entity resolution
results across different thresholds and collections. The API enables composable,
efficient operations through lazy evaluation.

Example:
    ```python
    import starlings as sl

    # Point comparison between collections
    result = ef.analyse(
        sl.col("splink").at(0.85),
        sl.col("truth").at(1.0),
        metrics=[sl.Metrics.eval.f1, sl.Metrics.eval.precision],
    )

    # Sweep operation for threshold exploration
    sweep_results = ef.analyse(
        sl.col("splink").sweep(0.5, 0.95, 0.01), metrics=[sl.Metrics.stats.entity_count]
    )
    ```
"""

from __future__ import annotations

from typing import Any

# Performance constants
MIN_SWEEP_STEP = 0.05  # Minimum step size for reasonable performance
STEP_ROUNDING_FACTOR = 0.05  # Round steps to this increment


class Expression:
    """Base class for analysis expressions.

    Expressions represent operations that can be applied to collections
    within an EntityFrame for analysis purposes.
    """

    def __init__(self, expression_type: str, **params: Any) -> None:
        """Initialise expression with type and parameters.

        Args:
            expression_type: Type of expression ("point" or "sweep")
            **params: Expression parameters
        """
        self.expression_type = expression_type
        self.params = params
        self.is_reference = False

    def reference(self) -> Expression:
        """Mark this collection as the reference (ground truth) for asymmetric metrics.

        When computing asymmetric comparison metrics like precision, recall, and f1,
        one collection must be designated as the reference (ground truth). Collections
        not marked with .reference() are treated as predictions.

        If no collection is explicitly marked as reference, the last expression
        passed to analyse() is implicitly used as the reference.

        Returns:
            Self for method chaining

        Example:
            ```python
            # Explicit reference
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

        For performance at scale, step sizes are constrained to multiples of 0.05.
        Smaller steps will be rounded up to 0.05.

        Args:
            start: Starting threshold (inclusive)
            stop: Ending threshold (inclusive)
            step: Step size between thresholds (minimum 0.05, default 0.05)

        Returns:
            Expression representing a sweep query across the threshold range

        Example:
            ```python
            sl.col("splink").sweep(0.5, 0.95, 0.05)  # Recommended
            sl.col("splink").sweep(0.5, 0.95, 0.1)  # Faster, coarser
            ```

        Note:
            For 1M-scale datasets, use step >= 0.05 to ensure reasonable performance.
            Steps smaller than 0.05 will be automatically adjusted to 0.05.
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


class EvaluationMetrics:
    """Evaluation metrics for comparing partitions.

    These metrics require two or more collections to compare partitions
    and compute agreement measures.
    """

    def __init__(self) -> None:
        """Initialise evaluation metrics."""
        self.f1 = MetricFunction("f1", "evaluation")
        self.precision = MetricFunction("precision", "evaluation")
        self.recall = MetricFunction("recall", "evaluation")
        self.ari = MetricFunction("ari", "evaluation")
        self.nmi = MetricFunction("nmi", "evaluation")
        self.v_measure = MetricFunction("v_measure", "evaluation")
        self.bcubed_precision = MetricFunction("bcubed_precision", "evaluation")
        self.bcubed_recall = MetricFunction("bcubed_recall", "evaluation")


class StatisticsMetrics:
    """Statistical metrics for single collections.

    These metrics can be computed on a single partition without requiring
    comparison to other collections.
    """

    def __init__(self) -> None:
        """Initialise statistics metrics."""
        self.entity_count = MetricFunction("entity_count", "statistics")
        self.entropy = MetricFunction("entropy", "statistics")


class Metrics:
    """Container for all available metrics.

    Provides organised access to different categories of metrics that can
    be used in the analyse() method.
    """

    def __init__(self) -> None:
        """Initialise metrics container."""
        self.eval = EvaluationMetrics()
        self.stats = StatisticsMetrics()


# Create module-level Metrics instance for easy access
Metrics = Metrics()  # type: ignore[assignment,misc]
