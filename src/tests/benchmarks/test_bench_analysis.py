"""Minimal performance benchmarks for EntityFrame analysis optimisation.

This module contains focused benchmarks that measure the critical performance
characteristics needed to guide analysis method optimisation.
"""

import logging
import sys
import time

import pytest
import starlings as sl
from starlings import generators

logger = logging.getLogger(__name__)

# Mark all tests in this module as benchmarks to exclude from regular test runs
pytestmark = pytest.mark.benchmark


class TestAnalysisBenchmarks:
    """Focused benchmarks for analysis method optimisation."""

    # Scale parameter N (set via run_benchmarks or defaults to 1.0)
    n: float = 1.0

    @classmethod
    def setup_class(cls) -> None:
        """Set up minimal logging."""
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    def test_core_metrics_performance(self) -> None:
        """Benchmark core single-collection metrics to identify optimisation targets."""
        logger.info("\n=== CORE METRICS PERFORMANCE ===")

        # Create scaled dataset
        n_entities = int(self.n * 1_000_000)
        logger.info(f"Dataset: {n_entities:,} entities")

        edge_generator = generators.edges(n_entities)
        collection = sl.Collection.from_edges(edge_generator, show_progress=False)

        ef = sl.EntityFrame()
        ef.add_collection("test", collection)

        # Test sweep performance with both key metrics
        logger.info("\nSweep performance (0.6-0.9, step 0.1):")

        # Entity count sweep
        start = time.perf_counter()
        result = ef.analyse(
            sl.col("test").sweep(0.6, 0.9, 0.1), metrics=[sl.Metrics.stats.entity_count]
        )
        count_sweep_time = time.perf_counter() - start
        n_points = len(result)
        pts_per_sec = n_points / count_sweep_time
        logger.info(
            f"  Entity count: {count_sweep_time:.3f}s ({pts_per_sec:.1f} pts/s)"
        )

        # Entropy sweep (tests Delta algorithm)
        start = time.perf_counter()
        result = ef.analyse(
            sl.col("test").sweep(0.6, 0.9, 0.1), metrics=[sl.Metrics.stats.entropy]
        )
        entropy_sweep_time = time.perf_counter() - start
        pts_per_sec = n_points / entropy_sweep_time
        logger.info(
            f"  Entropy (Delta): {entropy_sweep_time:.3f}s ({pts_per_sec:.1f} pts/s)"
        )

        # Combined metrics (overhead test)
        start = time.perf_counter()
        result = ef.analyse(
            sl.col("test").sweep(0.6, 0.9, 0.1),
            metrics=[sl.Metrics.stats.entity_count, sl.Metrics.stats.entropy],
        )
        combined_time = time.perf_counter() - start
        overhead = (combined_time / max(count_sweep_time, entropy_sweep_time) - 1) * 100
        logger.info(f"  Combined: {combined_time:.3f}s (overhead: {overhead:+.1f}%)")

    def test_comparison_patterns(self) -> None:
        """Benchmark critical cross-collection comparison patterns."""
        logger.info("\n=== COMPARISON PATTERNS ===")

        # Create two scaled datasets
        n_entities = int(self.n * 1_000_000)
        logger.info(f"Dataset: {n_entities:,} entities per collection")

        edge_gen_a = generators.edges(n_entities)
        collection_a = sl.Collection.from_edges(edge_gen_a, show_progress=False)

        edge_gen_b = generators.edges(n_entities)
        collection_b = sl.Collection.from_edges(edge_gen_b, show_progress=False)

        ef = sl.EntityFrame()
        ef.add_collection("col_a", collection_a)
        ef.add_collection("col_b", collection_b)

        # Test key comparison patterns
        logger.info("\nComparison patterns:")

        # Point × Point (baseline)
        start = time.perf_counter()
        result = ef.analyse(
            sl.col("col_a").at(0.8),
            sl.col("col_b").at(0.8).reference(),
            metrics=[sl.Metrics.eval.f1],
        )
        point_point_time = time.perf_counter() - start
        logger.info(f"  Point × Point: {point_point_time * 1000:.1f}ms")

        # Point × Sweep (common pattern)
        start = time.perf_counter()
        result = ef.analyse(
            sl.col("col_a").at(0.8),
            sl.col("col_b").sweep(0.6, 0.9, 0.1).reference(),
            metrics=[sl.Metrics.eval.f1],
        )
        point_sweep_time = time.perf_counter() - start
        n_comparisons = len(result)
        cmp_per_sec = n_comparisons / point_sweep_time
        logger.info(
            f"  Point × Sweep ({n_comparisons} pts): "
            f"{point_sweep_time:.3f}s ({cmp_per_sec:.1f} cmp/s)"
        )

        # Sweep × Sweep (critical for optimisation)
        start = time.perf_counter()
        result = ef.analyse(
            sl.col("col_a").sweep(0.7, 0.8, 0.1),  # 2 points
            sl.col("col_b").sweep(0.7, 0.8, 0.1).reference(),  # 2 points
            metrics=[sl.Metrics.eval.f1],
        )
        sweep_sweep_time = time.perf_counter() - start
        n_comparisons = len(result)
        cmp_per_sec = n_comparisons / sweep_sweep_time
        logger.info(
            f"  Sweep × Sweep ({n_comparisons} grid): "
            f"{sweep_sweep_time:.3f}s ({cmp_per_sec:.1f} cmp/s)"
        )


def run_benchmarks(n: float = 1.0) -> None:
    """Run focused analysis benchmarks for optimisation insights."""
    logger.info(f"\nSTARLINGS ANALYSIS BENCHMARK (N={n})")
    logger.info("=" * 40)

    # Create test instance and set N parameter
    benchmark_tests = TestAnalysisBenchmarks()
    benchmark_tests.n = n
    benchmark_tests.setup_class()

    try:
        # Run only the essential benchmarks
        benchmark_tests.test_core_metrics_performance()
        benchmark_tests.test_comparison_patterns()

        logger.info("\n" + "=" * 40)
        logger.info("BENCHMARK COMPLETE")

    except Exception as e:
        logger.error(f"\nBENCHMARK FAILED: {e}")
        raise


if __name__ == "__main__":
    # Parse N parameter from command line (default to 1)
    n = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    run_benchmarks(n)
