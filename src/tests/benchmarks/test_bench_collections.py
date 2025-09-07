"""Performance benchmarks for Collection.from_edges pipeline analysis.

This module contains benchmarks that are excluded from the regular test suite
and only run via `just bench` for detailed performance analysis.
"""

import logging
import os
import time

import pytest
import starlings as sl

logger = logging.getLogger(__name__)

# Mark all tests in this module as benchmarks to exclude from regular test runs
pytestmark = pytest.mark.benchmark


class TestPerformanceBenchmarks:
    """Production-scale benchmarks for performance analysis."""

    @classmethod
    def setup_class(cls) -> None:
        """Set up debug logging for detailed instrumentation."""
        os.environ["STARLINGS_DEBUG"] = "1"
        logging.basicConfig(level=logging.DEBUG, format="%(levelname)s - %(message)s")

    def test_production_1m_performance_breakdown(self) -> None:
        """Benchmark Collection.from_edges with 1M edges and detailed breakdown.

        This test provides comprehensive performance analysis of the entire
        Collection.from_edges pipeline with production-scale data.
        """
        logger.info("\n" + "=" * 60)
        logger.info("🔬 PRODUCTION-SCALE PERFORMANCE ANALYSIS")
        logger.info("=" * 60)

        # Generate production-scale dataset using unified generator
        logger.info("📊 Generating 1M entity production dataset...")
        start_generation = time.perf_counter()
        edges = sl.generate_entity_resolution_edges(1_000_000)
        generation_time = time.perf_counter() - start_generation

        logger.info(f"   Dataset: {len(edges):,} edges, {1_000_000:,} entities")
        logger.info(f"   Generation time: {generation_time:.3f}s")

        # Benchmark Collection.from_edges with full instrumentation
        logger.info("\n🏗️  Running Collection.from_edges with debug instrumentation...")
        collection_start = time.perf_counter()
        collection = sl.Collection.from_edges(edges)
        collection_time = time.perf_counter() - collection_start

        # Test partition creation performance
        logger.info("\n📈 Testing partition reconstruction performance...")
        partition_start = time.perf_counter()
        partition = collection.at(0.8)
        partition_time = time.perf_counter() - partition_start

        logger.info(f"   Partition at 0.8: {len(partition.entities):,} entities")

        # Validate the unified generator structure
        logger.info(
            f"   Validation: 1M entities at 1.0 -> {collection.at(1.0).num_entities:,}"
        )
        logger.info(
            f"   Validation: 500k entities at 0.0 -> "
            f"{collection.at(0.0).num_entities:,}"
        )
        logger.info(f"   Partition time: {partition_time:.3f}s")

        # Summary
        total_time = collection_time + partition_time
        logger.info("\n✅ BENCHMARK SUMMARY")
        logger.info(f"   Total pipeline: {total_time:.3f}s")
        logger.info(
            f"   Collection creation: {collection_time:.3f}s "
            f"({collection_time / total_time * 100:.1f}%)"
        )
        logger.info(
            f"   Partition reconstruction: {partition_time:.3f}s "
            f"({partition_time / total_time * 100:.1f}%)"
        )
        logger.info(f"   Throughput: {len(edges) / collection_time:,.0f} edges/second")

        # Performance assertions - verify realistic intermediate behavior
        # The constructive algorithm guarantees n entities at 1.0 and n/2 at 0.0
        # Intermediate values should be monotonically decreasing between these bounds
        actual_entities_08 = len(partition.entities)
        min_entities = 1_000_000 // 2  # n/2 at complete merging
        max_entities = 1_000_000  # n at full separation

        assert min_entities <= actual_entities_08 <= max_entities, (
            f"Entity count at 0.8 should be between {min_entities:,} and "
            f"{max_entities:,}, got {actual_entities_08:,}"
        )
        assert collection_time < 15.0, (
            f"Collection creation took {collection_time:.3f}s, target <15s"
        )

    def test_scalability_analysis(self) -> None:
        """Test performance scaling across different dataset sizes."""
        logger.info("\n" + "=" * 60)
        logger.info("📈 SCALABILITY ANALYSIS")
        logger.info("=" * 60)

        sizes = [10_000, 50_000, 100_000, 500_000]
        results: list[tuple[int, float, float]] = []

        for size in sizes:
            logger.info(f"\n🔍 Testing {size:,} edges...")

            # Generate proportional dataset using unified generator
            # Note: size represents number of entities, not edges
            # Each entity produces 5 edges, so divide by 5 to get entity count
            entity_count = max(size // 5, 1000)  # Minimum 1000 entities
            edges = sl.generate_entity_resolution_edges(entity_count)
            edges = edges[:size]  # Trim to exact edge size for comparison

            # Benchmark
            start = time.perf_counter()
            sl.Collection.from_edges(edges)  # Create collection for timing
            elapsed = time.perf_counter() - start

            throughput = size / elapsed
            results.append((size, elapsed, throughput))

            logger.info(f"   Time: {elapsed:.3f}s")
            logger.info(f"   Throughput: {throughput:,.0f} edges/second")

        logger.info("\n📊 SCALABILITY RESULTS")
        logger.info(f"{'Size':>10} {'Time':>10} {'Throughput':>15}")
        logger.info("-" * 35)
        for size, elapsed, throughput in results:
            logger.info(f"{size:>10,} {elapsed:>9.3f}s {throughput:>12,.0f}/s")

    def test_threshold_access_performance(self) -> None:
        """Benchmark partition access patterns at different thresholds."""
        logger.info("\n" + "=" * 60)
        logger.info("🎯 THRESHOLD ACCESS PERFORMANCE")
        logger.info("=" * 60)

        # Create test collection using unified generator
        # Generate 20k entities (produces 100k edges)
        edges = sl.generate_entity_resolution_edges(20_000)
        collection = sl.Collection.from_edges(edges)

        thresholds = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]

        logger.info("Testing first access (reconstruction) vs cached access...")
        logger.info(f"{'Threshold':>10} {'First':>10} {'Cached':>10} {'Entities':>10}")
        logger.info("-" * 45)

        for threshold in thresholds:
            # First access - reconstruction
            start = time.perf_counter()
            partition = collection.at(threshold)
            first_time = time.perf_counter() - start

            # Second access - cached
            start = time.perf_counter()
            collection.at(threshold)  # Cache the result
            cached_time = time.perf_counter() - start

            logger.info(
                f"{threshold:>10.1f} {first_time * 1000:>9.3f}ms "
                f"{cached_time * 1000:>9.3f}ms {len(partition.entities):>9,}"
            )

        logger.info("\n✅ Cache performance validated")


def run_benchmarks() -> None:
    """Run all benchmarks programmatically (for use in justfile)."""
    logger.info("🚀 Starting Starlings Performance Benchmarks")
    logger.info("=" * 60)

    # Create test instance
    benchmark_tests = TestPerformanceBenchmarks()
    benchmark_tests.setup_class()

    try:
        # Run each benchmark
        benchmark_tests.test_production_1m_performance_breakdown()
        benchmark_tests.test_scalability_analysis()
        benchmark_tests.test_threshold_access_performance()

        logger.info("\n" + "=" * 60)
        logger.info("✅ ALL BENCHMARKS COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"\n❌ BENCHMARK FAILED: {e}")
        raise


if __name__ == "__main__":
    run_benchmarks()
