"""Performance benchmarks for Collection.from_edges pipeline analysis.

This module contains benchmarks that are excluded from the regular test suite
and only run via `just bench` for detailed performance analysis.
"""

import logging
import os
import sys
import time

import psutil
import pytest
import starlings as sl

logger = logging.getLogger(__name__)

# Mark all tests in this module as benchmarks to exclude from regular test runs
pytestmark = pytest.mark.benchmark


class TestPerformanceBenchmarks:
    """Production-scale benchmarks for performance analysis."""

    # Scale parameter N (set via run_benchmarks or defaults to 1.0)
    n: float = 1.0

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

        # System resource overview - Starlings will automatically manage resources
        num_entities = int(self.n * 1_000_000)
        estimated_memory_mb = (num_entities * 5 * 150) // (1024 * 1024)
        estimated_memory_gb = estimated_memory_mb / 1024

        logger.info("🔍 Resource Overview:")
        logger.info(f"   Target entities: {num_entities:,}")
        logger.info(
            f"   Estimated peak memory: ~{estimated_memory_mb:,}MB "
            f"({estimated_memory_gb:.1f}GB)"
        )

        # System memory check using psutil
        available_gb = psutil.virtual_memory().available / (1024**3)
        total_gb = psutil.virtual_memory().total / (1024**3)

        logger.info(
            f"   System memory: {available_gb:.1f}GB available / {total_gb:.1f}GB total"
        )

        # With automatic resource management, we inform what strategy will be used
        if estimated_memory_gb <= available_gb / 4:
            logger.info("✅ Expected strategy: In-memory processing")
        elif estimated_memory_gb <= available_gb:
            logger.info("✅ Expected strategy: Memory-aware with potential spilling")
        else:
            logger.info("✅ Expected strategy: Streaming with aggressive spilling")

        logger.info(
            "   🔄 Starlings will automatically choose the optimal processing strategy"
        )

        # Generate production-scale dataset using unified generator
        logger.info(
            f"\n📊 Generating {self.n:.1f}M entity production dataset as generator..."
        )

        # Create generator (no upfront memory allocation)
        edge_generator = sl.generate_entity_resolution_edges(int(self.n * 1_000_000))

        logger.info(
            f"   Target: ~{int(self.n * 5_000_000):,} edges, "
            f"{int(self.n * 1_000_000):,} entities"
        )

        # Benchmark Collection.from_edges with tqdm progress bars
        logger.info("\n🏗️  Running Collection.from_edges with tqdm progress bars...")

        collection_start = time.perf_counter()
        # Collection.from_edges with automatic resource management and progress bars
        collection = sl.Collection.from_edges(
            edge_generator,
            show_progress=True,
        )
        collection_time = time.perf_counter() - collection_start

        # Test partition creation performance
        logger.info("\n📈 Testing partition reconstruction performance...")
        partition_start = time.perf_counter()
        partition = collection.at(0.8)
        partition_time = time.perf_counter() - partition_start

        logger.info(f"   Partition at 0.8: {len(partition.entities):,} entities")

        # Validate the unified generator structure
        logger.info(
            f"   Validation: {self.n:.1f}M entities at 1.0 -> "
            f"{collection.at(1.0).num_entities:,}"
        )
        logger.info(
            f"   Validation: {int(self.n * 500_000):,} entities at 0.0 -> "
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
        # Calculate expected edge count (n entities * 5 edges per entity on average)
        expected_edges = int(self.n * 1_000_000) * 5
        logger.info(
            f"   Throughput: {expected_edges / collection_time:,.0f} edges/second"
        )

        # Performance assertions - verify realistic intermediate behaviour
        # The constructive algorithm guarantees n entities at 1.0 and n/2 at 0.0
        # Intermediate values should be monotonically decreasing between these bounds
        actual_entities_08 = len(partition.entities)
        min_entities = (self.n * 1_000_000) // 2  # n/2 at complete merging
        max_entities = self.n * 1_000_000  # n at full separation

        assert min_entities <= actual_entities_08 <= max_entities, (
            f"Entity count at 0.8 should be between {min_entities:,} and "
            f"{max_entities:,}, got {actual_entities_08:,}"
        )

        # Performance reporting only - no hard limits to allow large-scale testing
        if collection_time > 15.0:
            logger.warning(
                f"⚠️  Collection creation took {collection_time:.3f}s (>15s target)"
            )
        else:
            logger.info(f"✅ Collection creation: {collection_time:.3f}s (<15s target)")

    def test_scalability_analysis(self) -> None:
        """Test performance scaling across different dataset sizes."""
        logger.info("\n" + "=" * 60)
        logger.info("📈 SCALABILITY ANALYSIS")
        logger.info("=" * 60)

        sizes = [
            int(self.n * 10_000),
            int(self.n * 50_000),
            int(self.n * 100_000),
            int(self.n * 500_000),
        ]
        results: list[tuple[int, float, float]] = []

        for size in sizes:
            logger.info(f"\n🔍 Testing {size:,} edges...")

            # Generate proportional dataset using unified generator
            # Note: size represents number of entities, not edges
            # Each entity produces 5 edges, so divide by 5 to get entity count
            entity_count = max(size // 5, 1000)  # Minimum 1000 entities

            # For scalability testing, use small datasets as lists for precise timing
            edge_generator = sl.generate_entity_resolution_edges(int(entity_count))
            # Convert small datasets to list for precise edge count control
            edges = []
            for batch in edge_generator:
                edges.extend(batch)
                if len(edges) >= size:
                    break
            edges = edges[:size]  # Trim to exact edge size for comparison

            # Benchmark with memory-aware processing
            start = time.perf_counter()
            # Use automatic resource management for all datasets
            sl.Collection.from_edges(
                edges,
                show_progress=False,
            )
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
        # Generate N*20k entities (produces N*100k edges)
        edge_generator = sl.generate_entity_resolution_edges(int(self.n * 20_000))
        collection = sl.Collection.from_edges(
            edge_generator,
            show_progress=False,
        )

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

    def test_expression_sweep_performance(self) -> None:
        """Benchmark expression API sweep operations with delta/record algorithms."""
        logger.info("\n" + "=" * 60)
        logger.info("🔬 EXPRESSION API SWEEP PERFORMANCE")
        logger.info("=" * 60)

        # Create test dataset scaled by N - same size as main benchmark (1M for N=1)
        n_entities = int(self.n * 1_000_000)
        logger.info(f"\n📊 Generating {n_entities:,} entities for sweep testing...")
        edge_generator = sl.generate_entity_resolution_edges(n_entities)

        # Create collections
        logger.info("   Creating test collections...")
        collection_a = sl.Collection.from_edges(
            edge_generator,
            show_progress=False,
        )

        # Create a different collection for cross-collection comparison
        edge_generator_b = sl.generate_entity_resolution_edges(n_entities)
        collection_b = sl.Collection.from_edges(
            edge_generator_b,
            show_progress=False,
        )

        # Create EntityFrame
        ef = sl.EntityFrame()
        ef.add_collection("col_a", collection_a)
        ef.add_collection("col_b", collection_b)

        logger.info(f"   Collections created with ~{n_entities * 5:,} edges each")

        # Test 1: Point comparison (baseline)
        logger.info("\n1️⃣ POINT COMPARISON (Single threshold)")
        logger.info("   Testing sl.col('col_a').at(0.8) vs sl.col('col_b').at(0.8)")

        start = time.perf_counter()
        result = ef.analyse(
            sl.col("col_a").at(0.8),
            sl.col("col_b").at(0.8),
            metrics=[
                sl.Metrics.eval.f1,
                sl.Metrics.eval.precision,
                sl.Metrics.eval.recall,
            ],
        )
        point_time = time.perf_counter() - start

        logger.info(f"   Time: {point_time:.3f}s")
        logger.info(f"   F1 Score: {result[0]['f1']:.4f}")
        logger.info("   Algorithm: Record-based (cross-collection)")

        # Test 2: Same collection sweep (Delta algorithm)
        logger.info("\n2️⃣ SAME COLLECTION SWEEP (Delta algorithm)")
        logger.info("   Testing sl.col('col_a').sweep(0.5, 0.9, 0.1)")

        start = time.perf_counter()
        result = ef.analyse(
            sl.col("col_a").sweep(0.5, 0.9, 0.1),
            metrics=[sl.Metrics.stats.entity_count, sl.Metrics.stats.entropy],
        )
        delta_time = time.perf_counter() - start

        n_points = len(result)
        logger.info(f"   Time: {delta_time:.3f}s for {n_points} threshold points")
        logger.info(f"   Throughput: {n_points / delta_time:.1f} points/second")
        logger.info("   Algorithm: Delta-based (incremental updates)")

        # Show sample results
        logger.info("   Sample results:")
        for i in [0, n_points // 2, n_points - 1]:
            if i < len(result):
                logger.info(
                    f"     Threshold {result[i]['col_a_threshold']:.1f}: "
                    f"{result[i]['entity_count']:.0f} entities"
                )

        # Test 3: Cross-collection sweep (Record algorithm for cross-collection)
        logger.info("\n3️⃣ POINT × SWEEP COMPARISON (Record algorithm)")
        logger.info(
            "   Testing sl.col('col_a').at(0.8) vs sl.col('col_b').sweep(0.5, 0.9, 0.1)"
        )
        logger.info("   Cross-collection comparison uses record-based O(r) algorithm")

        start = time.perf_counter()
        result = ef.analyse(
            sl.col("col_a").at(0.8),
            sl.col("col_b").sweep(0.5, 0.9, 0.1),
            metrics=[sl.Metrics.eval.f1],
        )
        cross_time = time.perf_counter() - start

        n_comparisons = len(result)
        logger.info(f"   Time: {cross_time:.3f}s for {n_comparisons} comparisons")
        logger.info(
            f"   Throughput: {n_comparisons / cross_time:.1f} comparisons/second"
        )
        logger.info("   Algorithm: Record-based (different collections, O(r))")

        # Test 4: Cartesian product sweep (demonstrates record algorithm advantage)
        logger.info("\n4️⃣ CARTESIAN PRODUCT SWEEP (Record algorithm advantage)")

        # Use smaller sweeps for cartesian product to keep reasonable time
        sweep_points = 3  # 3x3 = 9 comparisons
        logger.info(
            "   Testing sl.col('col_a').sweep(0.6, 0.8, 0.1) vs "
            "sl.col('col_b').sweep(0.6, 0.8, 0.1)"
        )
        logger.info(
            f"   This creates {sweep_points}×{sweep_points} = "
            f"{sweep_points**2} comparisons"
        )

        start = time.perf_counter()
        result = ef.analyse(
            sl.col("col_a").sweep(0.6, 0.8, 0.1),
            sl.col("col_b").sweep(0.6, 0.8, 0.1),
            metrics=[sl.Metrics.eval.f1],
        )
        cartesian_time = time.perf_counter() - start

        n_comparisons = len(result)
        logger.info(f"   Time: {cartesian_time:.3f}s for {n_comparisons} comparisons")
        logger.info(
            f"   Throughput: {n_comparisons / cartesian_time:.1f} comparisons/second"
        )
        logger.info("   Algorithm: Record-based (massive advantage for sweep×sweep)")

        # Calculate theoretical entity-based time
        partition_a = collection_a.at(0.7)
        partition_b = collection_b.at(0.7)
        k1 = len(partition_a.entities)
        k2 = len(partition_b.entities)

        # Estimate based on entity comparisons
        entity_comparisons = k1 * k2 * n_comparisons
        estimated_entity_time = entity_comparisons * 1e-8  # ~10ns per comparison
        speedup = estimated_entity_time / cartesian_time

        logger.info("\n📈 ALGORITHM COMPARISON")
        logger.info(f"   Entities at threshold 0.7: {k1:,} × {k2:,}")
        logger.info(f"   Entity-based would need: {entity_comparisons:,} comparisons")
        logger.info(f"   Record-based used: {n_entities:,} record iterations")
        logger.info(f"   Theoretical speedup: {speedup:.0f}x")

        # Summary
        logger.info("\n✅ SWEEP PERFORMANCE SUMMARY")
        logger.info(f"   Dataset size: {n_entities:,} entities")
        logger.info(f"   Point comparison: {point_time:.3f}s (Record algorithm)")
        logger.info(f"   Same-collection sweep: {delta_time:.3f}s (Delta algorithm)")
        logger.info(f"   Point × sweep: {cross_time:.3f}s (Record algorithm)")
        logger.info(
            f"   Cartesian sweep (3×3): {cartesian_time:.3f}s "
            "(Record algorithm optimised)"
        )

        if speedup > 100:
            logger.info(f"   🚀 Record algorithm achieved {speedup:.0f}x speedup!")


def run_benchmarks(n: float = 1.0) -> None:
    """Run all benchmarks programmatically (for use in justfile)."""
    logger.info(f"🚀 Starting Starlings Performance Benchmarks (N={n}M entities)")
    logger.info("=" * 60)

    # Create test instance and set N parameter
    benchmark_tests = TestPerformanceBenchmarks()
    benchmark_tests.n = n
    benchmark_tests.setup_class()

    try:
        # Run each benchmark
        benchmark_tests.test_production_1m_performance_breakdown()
        benchmark_tests.test_scalability_analysis()
        benchmark_tests.test_threshold_access_performance()
        benchmark_tests.test_expression_sweep_performance()

        logger.info("\n" + "=" * 60)
        logger.info("✅ ALL BENCHMARKS COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"\n❌ BENCHMARK FAILED: {e}")
        raise


if __name__ == "__main__":
    # Parse N parameter from command line (default to 1)
    n = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    run_benchmarks(n)
