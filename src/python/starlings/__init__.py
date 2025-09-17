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
    # partition.entities contains the entity IDs at this threshold
    ```
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from importlib.metadata import version  # noqa: PLC0415
from typing import Any, cast

from tqdm import tqdm

from .config import DEBUG_ENABLED
from .expressions import col
from .metrics import Metrics
from .starlings import Collection as PyCollection
from .starlings import EntityFrame as PyEntityFrame
from .starlings import Partition as PyPartition

logger = logging.getLogger(__name__)

# Load debug flag once at module import time
_DEBUG_ENABLED = DEBUG_ENABLED


__version__ = version("starlings")

__all__ = [
    "Collection",
    "EntityFrame",
    "Partition",
    "Metrics",
    "col",
    "Key",
]

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
        # Access entity count: partition.num_entities
        ```
    """

    def __init__(self, _partition: PyPartition) -> None:
        """Initialise Partition wrapper."""
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
            # Access entity count: partition.num_entities
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
        # Access entities: len(partition.entities)
        ```
    """

    def __init__(self, _collection: PyCollection) -> None:
        """Initialise Collection wrapper."""
        self._collection = _collection

    @classmethod
    def from_edges(
        cls,
        edges: Iterable[tuple[Key, Key, float]],
        *,
        source: str | None = None,
        show_progress: bool = True,
    ) -> Collection:
        """Build collection from weighted edges with automatic resource management.

        Creates a hierarchical partition structure from similarity edges between
        records. Records can be any hashable Python type (int, str, bytes) and are
        automatically converted to internal indices for efficient processing.

        **Streaming Processing**: Automatically chooses the optimal processing strategy
        based on dataset size and available system resources:
        - **In-memory**: Small datasets that fit comfortably in RAM
        - **Memory-aware**: Medium datasets with potential disk spilling
        - **Streaming**: Large datasets with aggressive memory management

        Args:
            edges: Iterable of (record_i, record_j, similarity) tuples.
                Can be a list, generator, or any iterable. Records can be any hashable
                type (int, str, bytes). Similarities should be between 0.0 and 1.0.
            source: Source name for record context. Defaults to "default".
            show_progress: Whether to show tqdm progress bar. Defaults to True.

        Returns:
            New Collection with hierarchy of merge events.

        Complexity:
            O(m log m) where m = total number of edges

        Example:
            ```python
            # Automatic resource management - no manual tuning needed
            edges = [("cust_123", "cust_456", 0.95), (123, 456, 0.85)]
            collection = Collection.from_edges(edges)

            # Works seamlessly with generators of any size
            edge_gen = generate_entity_resolution_edges(100_000_000)  # 100M entities!
            collection = Collection.from_edges(edge_gen)  # Automatically uses streaming

            # Progress bars show processing strategy and resource usage
            ```
        """
        # Convert edges to list, handling both sequences and generators
        edge_list = cls._collect_edges(edges, show_progress)

        # Create progress callback if needed
        progress_bar = None
        progress_callback = None
        if show_progress:
            progress_bar = tqdm(
                total=len(edge_list),
                desc="Processing edges",
                unit="edges",
                unit_scale=True,
            )

            def progress_callback(progress: float, message: str) -> None:
                progress_bar.set_description(f"Processing edges - {message}")
                progress_bar.n = int(progress * len(edge_list))
                progress_bar.refresh()

        rust_collection = PyCollection.from_edges(
            edge_list,
            source=source,
            progress_callback=progress_callback,
        )

        if show_progress and progress_bar is not None:
            progress_bar.close()

        return cls(rust_collection)

    @staticmethod
    def _collect_edges(
        edges: Iterable[tuple[Key, Key, float]], show_progress: bool
    ) -> list[tuple[Key, Key, float]]:
        """Collect edges from any iterable into a list."""
        # Fast path for lists and sequences
        if hasattr(edges, "__len__"):
            return edges if isinstance(edges, list) else list(edges)

        # Handle generators and iterators
        edge_list = []
        progress_bar = None

        if show_progress:
            progress_bar = tqdm(desc="Loading edges", unit="edges")

        for item in edges:
            # Handle batched generators
            if isinstance(item, list):
                edge_list.extend(item)
                if progress_bar is not None:
                    progress_bar.set_description(f"Loaded {len(edge_list):,} edges")
                    progress_bar.update(len(item))
            else:
                edge_list.append(item)
                if progress_bar is not None and len(edge_list) % 10000 == 0:
                    progress_bar.set_description(f"Loaded {len(edge_list):,} edges")
                    progress_bar.update(10000)

        if progress_bar is not None:
            # Update remaining
            remaining = len(edge_list) % 10000
            if remaining:
                progress_bar.update(remaining)
            progress_bar.close()

        return edge_list

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

            # Compare entity counts at different thresholds:
            # partition_low.entities at threshold 0.5
            # partition_high.entities at threshold 0.9
            ```
        """
        rust_partition = self._collection.at(threshold)
        return Partition(rust_partition)

    def copy(self) -> Collection:
        """Create a deep copy of this collection with independent context.

        Creates a new collection that is completely independent of the original,
        allowing modifications without affecting the original collection.

        Returns:
            Collection: A new independent collection with the same data.

        Example:
            ```python
            original = Collection.from_edges(edges)
            independent_copy = original.copy()
            # independent_copy can be modified without affecting original
            ```
        """
        # Both views and regular collections use the same copy mechanism
        rust_copy = self._collection.copy()
        return Collection(rust_copy)

    def is_view(self) -> bool:
        """Check if this collection is an immutable view from a frame.

        Collections retrieved from EntityFrame are views and cannot be modified.
        Use copy() to create a mutable independent collection.

        Returns:
            bool: True if this is a view, False if it's an owned collection.

        Example:
            ```python
            frame = EntityFrame()
            frame.add_collection("test", collection)
            view = frame["test"]  # This is a view
            assert view.is_view() == True
            independent = view.copy()
            assert independent.is_view() == False
            ```
        """
        return bool(self._collection.is_view())

    def __repr__(self) -> str:
        """String representation for debugging."""
        return "Collection"


class EntityFrame:
    """Multi-collection container for managing multiple hierarchies with shared memory.

    An EntityFrame allows you to store and manage multiple Collections in a single
    container. Collections that share the same underlying data can share memory
    efficiently.

    Collections that share the same underlying data context will automatically
    share memory for efficient storage and processing.

    Example:
        ```python
        # Create an empty frame
        frame = EntityFrame()

        # Check collection names
        assert frame.collection_names() == []
        assert len(frame) == 0

        # Add collections to frame
        edges = [("a", "b", 0.9), ("b", "c", 0.8)]
        collection = Collection.from_edges(edges)
        frame.add_collection("dataset", collection)

        # Access collections as views
        view = frame["dataset"]
        assert view.is_view()
        ```
    """

    def __init__(self) -> None:
        """Create a new empty EntityFrame."""
        self._frame = PyEntityFrame()

    def add_collection(self, name: str, collection: Collection) -> None:
        """Add a collection to the frame.

        The collection will be cloned and integrated into the frame's shared
        memory context. Collections with the same underlying data context
        will share memory efficiently.

        Args:
            name: Name for the collection
            collection: Collection to add to the frame

        Raises:
            RuntimeError: If the collection cannot be added

        Example:
            ```python
            frame = EntityFrame()
            collection = Collection.from_edges(edges)
            frame.add_collection("my_collection", collection)

            # Access the collection as a view
            view = frame["my_collection"]
            assert view.is_view() == True
            ```
        """
        self._frame.add_collection(name, collection._collection)

    def has_collection(self, name: str) -> bool:
        """Check if a collection exists in the frame.

        Args:
            name: Name of the collection

        Returns:
            True if the collection exists, False otherwise
        """
        return bool(self._frame.has_collection(name))

    def collection_names(self) -> list[str]:
        """Get list of all collection names in the frame.

        Returns:
            List of collection names
        """
        return cast(list[str], self._frame.collection_names())

    def remove_collection(self, name: str) -> bool:
        """Remove a collection from the frame.

        Args:
            name: Name of the collection to remove

        Returns:
            True if collection was removed, False if it didn't exist
        """
        return bool(self._frame.remove_collection(name))

    def __len__(self) -> int:
        """Get number of collections in the frame."""
        return len(self._frame)

    def __contains__(self, name: str) -> bool:
        """Check if a collection exists using 'in' operator."""
        return name in self._frame

    def __getitem__(self, name: str) -> Collection:
        """Get a collection by name using dictionary-style access.

        Returns collections as immutable views. Use copy() to create
        a mutable independent collection.

        Args:
            name: Name of the collection

        Returns:
            Collection: The collection as a view (immutable)

        Raises:
            KeyError: If collection doesn't exist

        Example:
            ```python
            frame = EntityFrame()
            frame.add_collection("test", collection)

            view = frame["test"]  # Dictionary-style access
            assert view.is_view() == True

            independent = view.copy()
            assert independent.is_view() == False
            ```
        """
        rust_collection = self._frame.__getitem__(name)
        return Collection(rust_collection)

    def analyse(
        self, *expressions: Any, metrics: list[Any] | None = None
    ) -> list[dict[str, float]]:
        """Universal analysis method using expressions.

        Always returns List[Dict[str, float]] where each dict represents one
        measurement.

        This uniform format works seamlessly with DataFrame libraries.

        Args:
            *expressions: One or more sl.col() expressions
            metrics: List of metrics to compute. If None, defaults are:
                    - For comparisons (2+ collections): f1, precision, recall
                    - For single collection: entity_count, entropy

        Returns:
            List[Dict[str, float]]: Uniform format regardless of operation:
            - Point comparisons: Single dict in list
            - Sweeps: One dict per threshold point
            - Mixed operations: Cartesian product of sweep points

            Dictionary keys:
            - "{collection}_threshold" for all threshold values
            - Direct metric names ("f1", "precision", "recall", etc.)

        Example:
            ```python
            # Point comparison
            >>> result = ef.analyse(
            ...     sl.col("splink").at(0.85),
            ...     sl.col("truth").at(1.0),
            ...     metrics=[sl.Metrics.eval.f1, sl.Metrics.eval.precision]
            ... )
            [{"splink_threshold": 0.85, "truth_threshold": 1.0, "f1": 0.92, ...}]

            # Single collection sweep
            >>> result = ef.analyse(
            ...     sl.col("splink").sweep(0.7, 0.9, 0.1),
            ...     metrics=[sl.Metrics.stats.entity_count]
            ... )
            [{"splink_threshold": 0.7, "entity_count": 1250},
             {"splink_threshold": 0.8, "entity_count": 980},
             {"splink_threshold": 0.9, "entity_count": 750}]

            # Easy DataFrame conversion
            >>> import polars as pl
            >>> df = pl.from_dicts(result)
            ```
        """
        # Calculate total operations for progress bar
        # This requires introspecting the expressions to count threshold combinations
        num_combinations = 1
        for expr in expressions:
            # Check if expression is a sweep by looking for _sweep attribute
            if hasattr(expr, "_sweep") and expr._sweep:
                # Calculate number of points in sweep
                start, stop, step = expr._sweep
                num_points = int((stop - start) / step) + 1
                num_combinations *= num_points
            else:
                # Point expression contributes factor of 1
                num_combinations *= 1

        # Determine number of metrics (use defaults if None)
        if metrics is not None:
            num_metrics = len(metrics)
        else:
            # Default metrics based on number of expressions
            num_metrics = 3 if len(expressions) >= 2 else 2

        total_operations = num_combinations * num_metrics

        # Create progress bar
        progress_bar = tqdm(
            total=total_operations,
            desc="Analysing",
            unit="ops",
            unit_scale=True,
        )

        def progress_callback(progress: float, message: str) -> None:
            progress_bar.set_description(f"Analysing - {message}")
            progress_bar.n = int(progress * total_operations)
            progress_bar.refresh()

        try:
            result = self._frame.analyse(
                *expressions, metrics=metrics, progress_callback=progress_callback
            )
        finally:
            progress_bar.close()

        return result  # type: ignore[no-any-return]

    def __repr__(self) -> str:
        """String representation for debugging."""
        return repr(self._frame)
