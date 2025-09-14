"""Metrics for entity resolution analysis.

This module provides metrics for evaluating entity resolution results.

Metrics are divided into two:

1. Statistics, which operate on a single collection
2. Evaluation, which operates between collections
    a. Symmetric metrics do not require a reference collection
    b. Asymmetric metrics require assigning a reference collection using
        `sl.col().reference()`

All metrics are defined in all comparative contexts.

Example:
    ```python
    import starlings as sl

    ef.analyse(
        sl.col("predicted").sweep(0.8, 0.9, 0.05),
        sl.col("truth").at(1.0).reference(),
        metrics=[
            sl.Metrics.eval.precision,
            sl.Metrics.eval.recall,
            sl.Metrics.stats.entity_count,
        ],
    )
    ```
"""

from . import eval, stats


class _MetricsContainer:
    """Container providing sl.Metrics.eval.* and sl.Metrics.stats.* access."""

    def __init__(self) -> None:
        """Initialise metrics container with eval and stats modules."""
        self.eval = eval
        self.stats = stats


# Create the instance that users access as sl.Metrics
Metrics = _MetricsContainer()

# Export for documentation and type hints
__all__ = ["Metrics", "eval", "stats"]
