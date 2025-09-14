"""Test data generators for entity resolution benchmarking and evaluation.

This module provides robust generators for creating realistic entity resolution
test datasets with exact hierarchical guarantees.
"""

from __future__ import annotations

from .starlings import (
    generate_entity_resolution_edges as _generate_entity_resolution_edges,
)


def edges(n: int, *, num_thresholds: int | None = None) -> list[tuple[int, int, float]]:
    """Generate realistic entity resolution edges with exact hierarchical guarantees.

    Creates synthetic entity resolution test data using a 5-step constructive
    algorithm that guarantees precise entity counts at key thresholds. Produces
    exactly n entities at threshold 1.0 and n/2 entities at threshold 0.0.

    Args:
        n: Target number of entities for dataset sizing. The algorithm uses an
            effective count where effective_n = n if n is even, n-1 if n is odd.
        num_thresholds: Controls threshold distribution:
            - None (default): Applies ±0.001 random jitter to all thresholds,
              creating continuous variation ideal for PGO training diversity
            - Integer value: Snaps to exactly that many discrete, evenly-spaced
              thresholds (e.g., 5 gives [0.0, 0.25, 0.5, 0.75, 0.999])

    Returns:
        List of (entity_id1, entity_id2, threshold) tuples with approximately
        n * 5 total edges for realistic density.

    Mathematical guarantees:
        - Exact entity counts: n entities at threshold 1.0, n/2 at 0.0
        - Monotonic merging: entity counts decrease smoothly with threshold
        - Hierarchical consistency: no sudden clustering jumps

    Examples:
        ```python
        import starlings as sl
        from starlings import generators

        # Generate dataset for PGO training
        edges = generators.edges(100_000)
        collection = sl.Collection.from_edges(edges)

        # Verify exact guarantees
        assert collection.at(1.0).num_entities == 100_000
        assert collection.at(0.0).num_entities == 50_000

        # Generate controlled dataset for testing
        test_edges = generators.edges(10_000, num_thresholds=10)
        ```

    Note:
        This is the canonical entity resolution data generator for the starlings
        library, powering all testing, benchmarking, and evaluation workflows.
    """
    return _generate_entity_resolution_edges(n, num_thresholds)  # type: ignore[no-any-return]
