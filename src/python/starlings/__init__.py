"""Starlings: High-performance entity resolution evaluation for Python.

Starlings revolutionises entity resolution by preserving complete resolution hierarchies
rather than forcing threshold decisions. This enables instant exploration of any
threshold and provides 10-100x performance improvements through incremental computation.

Core Innovation:
    Instead of storing fixed clusters, Starlings stores merge events that can generate
    partitions at any threshold without recomputation. This achieves O(k) incremental
    metric updates where k = affected entities.

Key Features:
    - Instant threshold exploration: O(1) cached partition access
    - Incremental metrics: 10-100x faster than recomputing from scratch
    - Memory efficient: ~60-115MB for 1M edges using RoaringBitmaps
    - Type flexible: Handles int, str, bytes keys seamlessly

Performance Characteristics:
    - Hierarchy construction: O(m log m) where m = edges
    - Threshold query: O(m) first time, O(1) cached
    - Metric updates: O(k) incremental between thresholds

Example:
    ```python
    import starlings as sl

    # Create collection from edges
    edges = [
        ("record_1", "record_2", 0.95),
        ("record_2", "record_3", 0.85),
        ("record_4", "record_5", 0.75),
    ]
    collection = sl.Collection.from_edges(edges)

    # Get partition at specific threshold
    partition = collection.at(0.8)
    print(f"Entities: {len(partition.entities)}")
    ```
"""

from __future__ import annotations

import logging
import os
import time
from importlib.metadata import version  # noqa: PLC0415
from typing import Any, cast

from .debug import DebugTimer, get_memory_mb
from .starlings import Collection as PyCollection
from .starlings import Partition as PyPartition
from .starlings import (
    generate_entity_resolution_edges as _generate_entity_resolution_edges,
)

logger = logging.getLogger(__name__)

# Load debug flag once at module import time
_DEBUG_ENABLED = os.getenv("STARLINGS_DEBUG", "").lower() in ("1", "true", "on")


def generate_entity_resolution_edges(
    n: int, num_thresholds: int | None = None
) -> list[tuple[int, int, float]]:
    """Generate entity resolution edges using the unified constructive algorithm.

    Creates realistic entity resolution test data following a constructive approach
    that produces exactly n/2 entities at threshold 0.0 through systematic pair
    construction, with realistic hierarchical patterns for benchmarking.

    Args:
        n: Target number of entities for sizing (algorithm uses effective_n where
            effective_n = n if n is even, n-1 if n is odd)
        num_thresholds: If provided, snap thresholds to discrete values;
            if None, add continuous jitter for PGO training diversity

    Returns:
        List of (entity_id1, entity_id2, threshold) tuples with entity IDs as integers
        and thresholds between 0.0 and 1.0

    Algorithm:
        Implements the 5-step constructive approach:
        1. Create n/2 pairs of entities with high thresholds (>0.9) for merging
        2. Add noise edges within pairs for density and realistic patterns
        3. Apply jitter (continuous) or discrete threshold snapping
        4. Remove duplicate edges and shuffle for randomization

        Guarantees: exactly effective_n/2 entities at threshold 0.0

    Example:
        ```python
        # Generate dataset with PGO jitter for training
        edges = generate_entity_resolution_edges(100_000)
        collection = Collection.from_edges(edges)

        # Key guarantee: exactly n/2 entities at threshold 0.0
        assert collection.at(0.0).num_entities == 50_000

        # Generate dataset with discrete thresholds for testing
        edges = generate_entity_resolution_edges(100_000, num_thresholds=10)
        ```
    """
    result = _generate_entity_resolution_edges(n, num_thresholds)
    return result  # type: ignore[no-any-return]


__version__ = version("starlings")

Key = Any


class Partition:
    """A partition of records into entities at a specific threshold.

    A Partition represents a snapshot of resolved entities at a specific threshold,
    providing access to the resolved groups and their properties.

    Attributes:
        entities: List of entities, where each entity is a list of record indices.
        num_entities: Number of entities in this partition.

    Example:
        ```python
        partition = collection.at(0.8)
        entities = partition.entities
        # [[0, 1, 2], [3, 4], [5]]  # 3 entities
        print(f"Found {partition.num_entities} entities")
        ```
    """

    def __init__(self, _partition: PyPartition) -> None:
        """Initialize Partition wrapper."""
        self._partition = _partition

    @property
    def entities(self) -> list[list[int]]:
        """Get entities as list of lists of record indices.

        Returns resolved entities as a list where each entity is represented
        as a list of record indices that belong to that entity.

        Returns:
            List of entities, where each entity is a list of record indices.

        Example:
            ```python
            partition = collection.at(0.8)
            entities = partition.entities
            # [[0, 1, 2], [3, 4], [5]]  # 3 entities
            ```
        """
        return cast(list[list[int]], self._partition.entities)

    @property
    def num_entities(self) -> int:
        """Get the number of entities in this partition.

        Returns:
            Number of entities in this partition.

        Example:
            ```python
            partition = collection.at(0.8)
            print(f"Found {partition.num_entities} entities")
            ```
        """
        return cast(int, self._partition.num_entities)

    def __len__(self) -> int:
        """Get the number of entities in this partition."""
        return len(self._partition)

    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"Partition(entities={len(self)})"


class Collection:
    """Hierarchical partition structure that generates entities at any threshold.

    A Collection stores the complete hierarchy of merge events, enabling instant
    exploration of partitions at any threshold without recomputation. The first
    query at a threshold reconstructs the partition (O(m)), while subsequent
    queries use cached results (O(1)).

    Key Features:
        - Instant threshold exploration: O(1) cached partition access
        - Memory efficient: Uses RoaringBitmaps for compact entity storage
        - Type flexible: Handles int, str, bytes keys seamlessly

    Performance:
        - Hierarchy construction: O(m log m) where m = edges
        - First partition query: O(m) reconstruction
        - Cached partition query: O(1) from cache

    Example:
        ```python
        # Create collection from edges
        edges = [
            ("record_1", "record_2", 0.95),
            ("record_2", "record_3", 0.85),
            ("record_4", "record_5", 0.75),
        ]
        collection = Collection.from_edges(edges)

        # Get partition at specific threshold
        partition = collection.at(0.8)
        print(f"Entities: {len(partition.entities)}")
        ```
    """

    def __init__(self, _collection: PyCollection) -> None:
        """Initialize Collection wrapper."""
        self._collection = _collection

    @classmethod
    def from_edges(
        cls,
        edges: list[tuple[Key, Key, float]],
        *,
        source: str | None = None,
    ) -> Collection:
        """Build collection from weighted edges.

        Creates a hierarchical partition structure from similarity edges between
        records. Records can be any hashable Python type (int, str, bytes) and are
        automatically converted to internal indices for efficient processing.

        Args:
            edges: List of (record_i, record_j, similarity) tuples.
                Records can be any hashable type (int, str, bytes). Similarities
                should be between 0.0 and 1.0.
            source: Source name for record context. Defaults to "default".

        Returns:
            New Collection with hierarchy of merge events.

        Complexity:
            O(m log m) where m = len(edges)

        Example:
            ```python
            # Basic usage with different key types
            edges = [
                ("cust_123", "cust_456", 0.95),
                (123, 456, 0.85),
                (b"hash1", b"hash2", 0.75),
            ]
            collection = Collection.from_edges(edges)

            # Get partition at threshold
            partition = collection.at(0.8)
            print(f"Entities: {len(partition.entities)}")
            ```
        """
        start_time = time.perf_counter() if _DEBUG_ENABLED else 0.0
        start_memory = get_memory_mb() if _DEBUG_ENABLED else 0.0

        if _DEBUG_ENABLED:
            logger.debug(
                "Starting with %s edges, memory: %.1fMB",
                f"{len(edges):,}",
                start_memory,
            )

        with DebugTimer("Edge validation & preprocessing", _DEBUG_ENABLED):
            # Let Rust handle the actual processing, but we can measure overall time
            pass

        with DebugTimer("Hierarchy construction & partitioning", _DEBUG_ENABLED):
            rust_collection = PyCollection.from_edges(edges, source=source)

        if _DEBUG_ENABLED:
            total_time = time.perf_counter() - start_time
            final_memory = get_memory_mb()
            peak_delta = final_memory - start_memory
            logger.debug(
                "Total: %.3fs, %+.1fMB peak memory",
                total_time,
                peak_delta,
            )

        return cls(rust_collection)

    def at(self, threshold: float) -> Partition:
        """Get partition at specific threshold.

        Returns a Partition containing all entities that exist at the specified
        similarity threshold. The first call at a threshold reconstructs the partition
        from merge events (O(m)), while subsequent calls use cached results (O(1)).

        Args:
            threshold: Threshold value between 0.0 and 1.0. Records with
                similarity >= threshold will be merged into the same entity.

        Returns:
            Partition object with entities at the specified threshold.

        Complexity:
            First call at threshold: O(m) reconstruction
            Subsequent calls: O(1) from cache

        Example:
            ```python
            collection = Collection.from_edges(edges)

            # Get partition at different thresholds
            partition_low = collection.at(0.5)  # More, smaller entities
            partition_high = collection.at(0.9)  # Fewer, larger entities

            print(f"At 0.5: {len(partition_low.entities)} entities")
            print(f"At 0.9: {len(partition_high.entities)} entities")
            ```
        """
        rust_partition = self._collection.at(threshold)
        return Partition(rust_partition)

    def __repr__(self) -> str:
        """String representation for debugging."""
        return "Collection"
