"""Test large-scale cross-collection comparisons with record-based optimization.

This module tests the O(r) algorithm for cross-collection comparisons that
leverages shared DataContext between collections in an EntityFrame, reducing
complexity from O(k₁ × k₂) to O(r) where k = entities and r = records.
"""

import random
import time

import pytest
import starlings as sl


def generate_realistic_edges(
    n_edges: int, n_records: int | None = None
) -> list[tuple[str, str, float]]:
    """Generate realistic entity resolution edges for benchmarking.

    Args:
        n_edges: Number of edges to generate
        n_records: Approximate number of unique records (default: n_edges // 2)
    """
    random.seed(42)  # For reproducibility

    if n_records is None:
        n_records = n_edges // 2

    edges = []

    # Generate clusters of different sizes (power law distribution)
    cluster_sizes = []
    remaining_records = n_records
    cluster_id = 0

    while remaining_records > 0:
        # Power law: most clusters are small, few are large
        size = min(remaining_records, int(random.paretovariate(1.5) + 1))
        cluster_sizes.append(size)
        remaining_records -= size
        cluster_id += 1

    # Generate edges within clusters
    edge_count = 0
    for cluster_idx, cluster_size in enumerate(cluster_sizes):
        if edge_count >= n_edges:
            break

        records = [f"rec_{cluster_idx}_{i}" for i in range(cluster_size)]

        # Dense connections within small clusters, sparse in large ones
        density = min(1.0, 10.0 / cluster_size) if cluster_size > 1 else 0
        n_cluster_edges = min(
            int(cluster_size * (cluster_size - 1) * density / 2), n_edges - edge_count
        )

        for _ in range(n_cluster_edges):
            if edge_count >= n_edges:
                break
            i, j = random.sample(range(cluster_size), 2)
            similarity = 0.7 + random.random() * 0.3  # High similarity within cluster
            edges.append((records[i], records[j], similarity))
            edge_count += 1

    # Add some cross-cluster edges (potential false positives)
    n_cross_edges = min(n_edges // 20, n_edges - edge_count)
    for _ in range(n_cross_edges):
        if edge_count >= n_edges:
            break
        cluster1, cluster2 = random.sample(range(len(cluster_sizes)), 2)
        if cluster_sizes[cluster1] > 0 and cluster_sizes[cluster2] > 0:
            rec1 = f"rec_{cluster1}_0"
            rec2 = f"rec_{cluster2}_0"
            similarity = 0.5 + random.random() * 0.2  # Lower similarity across clusters
            edges.append((rec1, rec2, similarity))
            edge_count += 1

    return edges[:n_edges]


def generate_test_edges(n: int, seed: int = 42) -> list[tuple[str, str, float]]:
    """Generate deterministic test edges for reproducible testing."""
    random.seed(seed)
    edges = []

    # Create clusters of related records
    for cluster in range(max(1, n // 100)):
        cluster_size = random.randint(5, 15)
        cluster_records = [f"record_{cluster * 100 + i}" for i in range(cluster_size)]

        # Connect records within cluster with varying strengths
        for i in range(len(cluster_records)):
            for j in range(i + 1, min(i + 3, len(cluster_records))):
                similarity = 0.7 + random.random() * 0.3
                edges.append((cluster_records[i], cluster_records[j], similarity))

    # Add some cross-cluster connections with lower similarity
    for _ in range(n // 200):
        if n >= 200:  # Only if we have enough clusters
            cluster1 = random.randint(0, max(0, n // 100 - 1))
            cluster2 = random.randint(0, max(0, n // 100 - 1))
            if cluster1 != cluster2:
                record1 = f"record_{cluster1 * 100}"
                record2 = f"record_{cluster2 * 100}"
                similarity = 0.5 + random.random() * 0.2
                edges.append((record1, record2, similarity))

    return edges[:n]


class TestRecordBasedOptimization:
    """Test the record-based contingency table optimization.

    These tests validate both correctness and performance of the O(r) algorithm
    that replaces the O(k₁ × k₂) entity-based comparison.
    """

    def test_identical_results_small_dataset(self):
        """Verify that optimised algorithm produces correct results."""
        # Create test data
        edges = generate_test_edges(100)

        # Create EntityFrame with two collections from same edges
        ef = sl.EntityFrame()
        collection_a = sl.Collection.from_edges(edges)
        collection_b = sl.Collection.from_edges(edges)

        ef.add_collection("col_a", collection_a)
        ef.add_collection("col_b", collection_b)

        # Compare at various thresholds
        thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]

        for threshold in thresholds:
            # Same collections at same threshold should have perfect agreement
            result = ef.analyse(
                sl.col("col_a").at(threshold),
                sl.col("col_b").at(threshold),
                metrics=[
                    sl.Metrics.eval.f1,
                    sl.Metrics.eval.precision,
                    sl.Metrics.eval.recall,
                ],
            )

            assert len(result) == 1
            assert result[0]["f1"] == 1.0
            assert result[0]["precision"] == 1.0
            assert result[0]["recall"] == 1.0

    def test_different_thresholds_comparison(self):
        """Test comparison between different thresholds."""
        edges = generate_test_edges(200)

        ef = sl.EntityFrame()
        collection = sl.Collection.from_edges(edges)
        ef.add_collection("test", collection)
        ef.add_collection("test2", collection.copy())

        # Compare different thresholds
        result = ef.analyse(
            sl.col("test").at(0.8),
            sl.col("test2").at(0.7),
            metrics=[sl.Metrics.eval.f1],
        )

        assert len(result) == 1
        # Different thresholds should not have perfect agreement
        assert 0.0 < result[0]["f1"] < 1.0

    @pytest.mark.parametrize(
        ["n_edges", "max_time"],
        [
            pytest.param(1000, 2.0, id="1k_edges"),
            pytest.param(5000, 5.0, id="5k_edges"),
            pytest.param(10000, 10.0, id="10k_edges"),
        ],
    )
    def test_sweep_performance(self, n_edges: int, max_time: float):
        """Test that sweep operations complete efficiently.

        The record-based algorithm should make sweeps much faster by iterating
        records once rather than comparing all entity pairs.
        """
        edges = generate_realistic_edges(n_edges)

        ef = sl.EntityFrame()
        collection = sl.Collection.from_edges(edges)
        ef.add_collection("test1", collection)
        ef.add_collection("test2", collection.copy())

        start_time = time.time()

        # Perform a 3x3 sweep comparison
        results = ef.analyse(
            sl.col("test1").sweep(0.6, 0.8, 0.1),
            sl.col("test2").sweep(0.6, 0.8, 0.1),
            metrics=[sl.Metrics.eval.f1],
        )

        elapsed = time.time() - start_time

        # Check performance
        assert elapsed < max_time, f"Sweep took {elapsed:.2f}s, expected < {max_time}s"

        # Verify results
        assert len(results) == 9  # 3x3 grid
        for result in results:
            assert "test1_threshold" in result
            assert "test2_threshold" in result
            assert "f1" in result
            assert 0.0 <= result["f1"] <= 1.0

    def test_large_single_comparison(self):
        """Benchmark single threshold comparison at scale."""
        # Create a large dataset
        edges = generate_realistic_edges(50000, n_records=10000)

        ef = sl.EntityFrame()
        collection = sl.Collection.from_edges(edges)
        ef.add_collection("large1", collection)
        ef.add_collection("large2", collection.copy())

        start_time = time.time()

        result = ef.analyse(
            sl.col("large1").at(0.75),
            sl.col("large2").at(0.75),
            metrics=[
                sl.Metrics.eval.f1,
                sl.Metrics.eval.precision,
                sl.Metrics.eval.recall,
            ],
        )

        elapsed = time.time() - start_time

        # Should complete quickly even for large dataset
        assert elapsed < 5.0, f"Large comparison took {elapsed:.2f}s"

        # Verify results
        assert len(result) == 1
        assert result[0]["f1"] == 1.0  # Same data, same threshold
        assert result[0]["precision"] == 1.0
        assert result[0]["recall"] == 1.0

    def test_shared_context_detection(self):
        """Test that algorithm correctly detects shared context."""
        edges1 = generate_test_edges(100, seed=1)
        edges2 = generate_test_edges(100, seed=2)

        # Collections in same EntityFrame should share context
        ef = sl.EntityFrame()
        col1 = sl.Collection.from_edges(edges1)
        col2 = sl.Collection.from_edges(edges2)
        ef.add_collection("shared1", col1)
        ef.add_collection("shared2", col2)

        # This should use optimised algorithm (both in same frame)
        result = ef.analyse(
            sl.col("shared1").at(0.7),
            sl.col("shared2").at(0.7),
            metrics=[sl.Metrics.eval.f1],
        )
        assert len(result) == 1

    def test_cache_effectiveness(self):
        """Test that caching improves performance for repeated comparisons."""
        edges = generate_realistic_edges(5000)

        ef = sl.EntityFrame()
        collection = sl.Collection.from_edges(edges)
        ef.add_collection("cache_test", collection)

        # First run - builds caches
        start_time = time.time()
        result1 = ef.analyse(
            sl.col("cache_test").at(0.75),
            sl.col("cache_test").at(0.75),
            metrics=[sl.Metrics.eval.f1],
        )
        first_run_time = time.time() - start_time

        # Second run - should use cached indices
        start_time = time.time()
        result2 = ef.analyse(
            sl.col("cache_test").at(0.75),
            sl.col("cache_test").at(0.75),
            metrics=[sl.Metrics.eval.f1],
        )
        second_run_time = time.time() - start_time

        # Second run should be faster or similar
        assert second_run_time <= first_run_time * 1.5

        # Results should be identical
        assert result1[0]["f1"] == result2[0]["f1"]

    @pytest.mark.parametrize(
        ["num_edges", "expected_f1"],
        [
            pytest.param(50, 1.0, id="small_dataset"),
            pytest.param(200, 1.0, id="medium_dataset"),
            pytest.param(500, 1.0, id="large_dataset"),
        ],
    )
    def test_scaling_correctness(self, num_edges: int, expected_f1: float):
        """Test that optimization works correctly at different scales."""
        edges = generate_test_edges(num_edges)

        ef = sl.EntityFrame()
        collection = sl.Collection.from_edges(edges)
        ef.add_collection("scale_test", collection)

        # Same collection at same threshold should always have F1=1.0
        result = ef.analyse(
            sl.col("scale_test").at(0.75),
            sl.col("scale_test").at(0.75),
            metrics=[sl.Metrics.eval.f1],
        )

        assert abs(result[0]["f1"] - expected_f1) < 0.001
