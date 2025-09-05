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
from .starlings import GraphConfig as PyGraphConfig
from .starlings import Partition as PyPartition
from .starlings import generate_hierarchical_graph as _generate_hierarchical_graph
from .starlings import (
    generate_production_1m_randomized as _generate_production_1m_randomized,
)

logger = logging.getLogger(__name__)

# Load debug flag once at module import time
_DEBUG_ENABLED = os.getenv("STARLINGS_DEBUG", "").lower() in ("1", "true", "on")


def generate_hierarchical_graph(
    n_left: int,
    n_right: int,
    n_isolates: int,
    thresholds: list[tuple[float, int]],
) -> tuple[list[tuple[int, int, float]], int]:
    """Generate a hierarchical graph with realistic entity resolution patterns.

    Creates a bipartite graph with exact component counts at specified thresholds,
    using the hierarchical block construction method to ensure correct component
    distributions that mirror production entity resolution patterns.

    Args:
        n_left: Number of left-side records (e.g., customers)
        n_right: Number of right-side records (e.g., transactions)
        n_isolates: Number of isolated records (no edges)
        thresholds: List of (threshold, target_entities) pairs

    Returns:
        Tuple of (edges, total_nodes) where:
        - edges: List of (left_id, right_id, similarity) tuples
        - total_nodes: Total number of nodes in the graph

    Complexity:
        O(n + m) where n = nodes, m = edges

    Example:
        ```python
        # Custom graph generation
        edges, total_nodes = generate_hierarchical_graph(
            n_left=1000, n_right=1000, n_isolates=0, thresholds=[(0.9, 500), (0.7, 300)]
        )

        collection = Collection.from_edges(edges)
        partition = collection.at(0.9)
        print(f"Entities at 0.9: {len(partition.entities)}")
        ```
    """
    config = PyGraphConfig(n_left, n_right, n_isolates, thresholds)
    result = _generate_hierarchical_graph(config)
    return result  # type: ignore[no-any-return]


def generate_production_1m_graph() -> tuple[list[tuple[int, int, float]], int]:
    """Generate a production-scale 1M record graph with hierarchical thresholds.

    Convenience function that creates a realistic production-scale dataset
    with 1.1M records and multiple threshold levels.

    Returns:
        Tuple of (edges, total_nodes) with production-scale data

    Example:
        ```python
        edges, total_nodes = generate_production_1m_graph()
        collection = Collection.from_edges(edges)
        ```
    """
    config = PyGraphConfig.production_1m()
    result = _generate_hierarchical_graph(config)
    return result  # type: ignore[no-any-return]


def generate_production_1m_graph_randomized(
    seed: int | None = None, jitter_percent: float = 10.0
) -> tuple[list[tuple[int, int, float]], int]:
    """Generate a randomised production-scale 1M record graph for PGO training.

    Creates production-scale datasets with controlled randomness to prevent
    PGO overfitting whilst maintaining realistic test scenarios.

    Example:
        ```python
        edges, total_nodes = generate_production_1m_graph_randomized(None, 10.0)
        collection = Collection.from_edges(edges)
        ```
    """
    result = _generate_production_1m_randomized(seed, jitter_percent)
    return result  # type: ignore[no-any-return]


def generate_production_10m_graph() -> tuple[list[tuple[int, int, float]], int]:
    """Generate a large-scale 10M record graph with hierarchical thresholds.

    Convenience function that creates a very large production-scale dataset
    with 11M records and multiple threshold levels.

    Returns:
        Tuple of (edges, total_nodes) with large-scale data

    Example:
        ```python
        edges, total_nodes = generate_production_10m_graph()
        collection = Collection.from_edges(edges)
        ```
    """
    config = PyGraphConfig.production_10m()
    result = _generate_hierarchical_graph(config)
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


class GraphConfig:
    """Configuration for generating realistic entity resolution graphs.

    This mirrors production entity resolution patterns with hierarchical threshold
    structures and realistic component distributions for benchmarking and testing.

    Example:
        ```python
        # Custom configuration
        config = GraphConfig(1000, 1000, 0, [(0.9, 500), (0.7, 300)])

        # Production-scale configurations
        config_1m = GraphConfig.production_1m()
        config_10m = GraphConfig.production_10m()

        # Generate graph
        edges, total_nodes = generate_hierarchical_graph(config_1m)
        collection = Collection.from_edges(edges)
        ```
    """

    def __init__(
        self,
        n_left: int,
        n_right: int,
        n_isolates: int,
        thresholds: list[tuple[float, int]],
    ) -> None:
        """Create a new graph configuration.

        Args:
            n_left: Number of left-side records (e.g., customers)
            n_right: Number of right-side records (e.g., transactions)
            n_isolates: Number of isolated records (no edges)
            thresholds: List of (threshold, target_entities) pairs

        Example:
            ```python
            config = GraphConfig(1000, 1000, 0, [(0.9, 500), (0.7, 300)])
            ```
        """
        self._config = PyGraphConfig(n_left, n_right, n_isolates, thresholds)

    @classmethod
    def production_1m(cls) -> GraphConfig:
        """Create a production-scale configuration for million-record testing.

        Returns a pre-configured GraphConfig suitable for production-scale testing
        with 1.1M records and hierarchical thresholds.

        Returns:
            Production-scale configuration with:
            - 550k left + 550k right records
            - Thresholds: 0.9->200k entities, 0.7->100k entities, 0.5->50k entities

        Example:
            ```python
            config = GraphConfig.production_1m()
            edges, total_nodes = generate_hierarchical_graph(config)
            ```
        """
        instance = cls.__new__(cls)
        instance._config = PyGraphConfig.production_1m()
        return instance

    @classmethod
    def production_10m(cls) -> GraphConfig:
        """Create a large-scale configuration for 10M+ record testing.

        Returns a pre-configured GraphConfig suitable for very large-scale testing
        with 11M records and hierarchical thresholds.

        Returns:
            Large-scale configuration with 5.5M left + 5.5M right records

        Example:
            ```python
            config = GraphConfig.production_10m()
            edges, total_nodes = generate_hierarchical_graph(config)
            ```
        """
        instance = cls.__new__(cls)
        instance._config = PyGraphConfig.production_10m()
        return instance

    def __repr__(self) -> str:
        """String representation for debugging."""
        return repr(self._config)
