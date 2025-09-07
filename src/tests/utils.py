"""Testing utilities for starlings entity resolution library."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def validate_entity_resolution_data(
    edges: list[tuple[int, int, float]], n: int
) -> bool:
    """Validate entity resolution data structure for development workflow.

    Quick validation function implementing the checks from the unified generator
    specification to ensure generated data meets structural requirements.

    Args:
        edges: List of (entity_id1, entity_id2, threshold) tuples
        n: Target entity count used for generation (actual effective_n may be n or n-1)

    Returns:
        True if validation passes (exactly n/2 entities at threshold 0.0),
        False otherwise

    Example:
        ```python
        from tests.utils import validate_entity_resolution_data
        import starlings

        edges = starlings.generate_entity_resolution_edges(n=100_000)
        if validate_entity_resolution_data(edges, 100_000):
            print("✅ Structure correct, scaling up...")
        ```
    """
    try:
        import starlings  # Import here to avoid circular imports  # noqa: PLC0415

        collection = starlings.Collection.from_edges(edges)
        count_1_0 = collection.at(1.0).num_entities
        count_0_0 = collection.at(0.0).num_entities

        # Calculate expected values based on unified generator algorithm
        effective_n = n if n % 2 == 0 else n - 1
        expected_1_0 = effective_n  # Exactly n entities at threshold 1.0 (fixed!)
        expected_0_0 = effective_n // 2  # Exactly n/2 entities at threshold 0.0

        logger.info(
            f"At 1.0: {count_1_0:,} (target: {expected_1_0:,}), "
            f"At 0.0: {count_0_0:,} (target: {expected_0_0:,})"
        )

        # Check exact endpoint requirements - both key guarantees
        if count_1_0 != expected_1_0:
            logger.error(
                f"Incorrect entity count at 1.0: {count_1_0} != {expected_1_0}"
            )
            return False

        if count_0_0 != expected_0_0:
            logger.error(
                f"Incorrect entity count at 0.0: {count_0_0} != {expected_0_0}"
            )
            return False

        # Check monotonic merging behaviour
        thresholds = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
        entity_counts = [collection.at(t).num_entities for t in thresholds]

        for i in range(len(entity_counts) - 1):
            if entity_counts[i] < entity_counts[i + 1]:
                logger.error(
                    f"Non-monotonic decrease: {entity_counts[i]} < "
                    f"{entity_counts[i + 1]} at thresholds {thresholds[i]} -> "
                    f"{thresholds[i + 1]}"
                )
                return False

        # Check for reasonable progression (allow larger jumps due to
        # pair-based algorithm)
        # The unified generator creates pairs that merge dramatically,
        # so allow 50% jumps
        max_jump = effective_n * 0.50  # Allow up to 50% jump (pair-based merging)
        for i in range(len(entity_counts) - 1):
            jump = entity_counts[i] - entity_counts[i + 1]
            if jump > max_jump:
                logger.error(
                    f"Excessive jump: {jump:,} > {max_jump:,} "
                    f"at {thresholds[i]} -> {thresholds[i + 1]}"
                )
                return False

        # The unified generator creates a simple n -> n/2 transition, so don't enforce
        # specific intermediate values - they depend on exact threshold distribution

        logger.info("✅ All validation checks passed")
        return True

    except Exception as e:  # noqa: BLE001
        logger.error(f"Validation failed with error: {e}")
        return False
