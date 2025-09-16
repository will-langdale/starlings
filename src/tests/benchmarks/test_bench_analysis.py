"""Performance benchmarks for EntityFrame analysis methods.

This module contains comprehensive benchmarks for all analytical methods available
in Starlings, testing performance across different dataset sizes and comparison types.
"""

import logging
import os
import sys
import time
from typing import Any

import psutil
import pytest
import starlings as sl
from starlings import generators

logger = logging.getLogger(__name__)

# Mark all tests in this module as benchmarks to exclude from regular test runs
pytestmark = pytest.mark.benchmark


class TestAnalysisBenchmarks:
    """Production-scale benchmarks for EntityFrame analysis methods."""

    # Scale parameter N (set via run_benchmarks or defaults to 1.0)
    n: float = 1.0

    @classmethod
    def setup_class(cls) -> None:
        """Set up debug logging for detailed instrumentation."""
        os.environ["STARLINGS_DEBUG"] = "1"
        logging.basicConfig(level=logging.DEBUG, format="%(levelname)s - %(message)s")

    def test_statistics_metrics_performance(self) -> None:
        """Benchmark single collection statistics metrics."""
        logger.info("\n" + "=" * 60)
        logger.info("📊 STATISTICS METRICS PERFORMANCE")
        logger.info("=" * 60)

        # Create scaled dataset
        n_entities = int(self.n * 100_000)
        logger.info(f"\n🔍 Creating dataset with {n_entities:,} entities...")

        edge_generator = generators.edges(n_entities)
        collection = sl.Collection.from_edges(edge_generator, show_progress=False)

        # Create EntityFrame
        ef = sl.EntityFrame()
        ef.add_collection("test", collection)

        # Test entity count metric
        logger.info("\n1️⃣ ENTITY COUNT METRIC")

        # Single point
        start = time.perf_counter()
        result = ef.analyse(
            sl.col("test").at(0.8), metrics=[sl.Metrics.stats.entity_count]
        )
        point_time = time.perf_counter() - start
        logger.info(f"   Single point: {point_time * 1000:.3f}ms")
        logger.info(f"   Entities at 0.8: {result[0]['entity_count']:.0f}")

        # Sweep
        start = time.perf_counter()
        result = ef.analyse(
            sl.col("test").sweep(0.0, 1.0, 0.1), metrics=[sl.Metrics.stats.entity_count]
        )
        sweep_time = time.perf_counter() - start
        n_points = len(result)
        logger.info(f"   Sweep ({n_points} points): {sweep_time:.3f}s")
        logger.info(f"   Throughput: {n_points / sweep_time:.1f} points/second")

        # Test entropy metric
        logger.info("\n2️⃣ ENTROPY METRIC")

        # Single point
        start = time.perf_counter()
        result = ef.analyse(sl.col("test").at(0.8), metrics=[sl.Metrics.stats.entropy])
        entropy_point_time = time.perf_counter() - start
        logger.info(f"   Single point: {entropy_point_time * 1000:.3f}ms")
        logger.info(f"   Entropy at 0.8: {result[0]['entropy']:.4f}")

        # Sweep with Delta algorithm (incremental)
        start = time.perf_counter()
        result = ef.analyse(
            sl.col("test").sweep(0.5, 0.9, 0.05), metrics=[sl.Metrics.stats.entropy]
        )
        entropy_sweep_time = time.perf_counter() - start
        n_points = len(result)
        logger.info(f"   Sweep ({n_points} points): {entropy_sweep_time:.3f}s")
        logger.info(f"   Throughput: {n_points / entropy_sweep_time:.1f} points/second")
        logger.info("   Algorithm: Delta-based (O(k) incremental)")

        # Combined metrics
        logger.info("\n3️⃣ COMBINED STATISTICS")
        start = time.perf_counter()
        result = ef.analyse(
            sl.col("test").sweep(0.5, 0.9, 0.1),
            metrics=[sl.Metrics.stats.entity_count, sl.Metrics.stats.entropy],
        )
        combined_time = time.perf_counter() - start
        n_points = len(result)
        logger.info(f"   Time for 2 metrics × {n_points} points: {combined_time:.3f}s")
        overhead_pct = (combined_time / entropy_sweep_time - 1) * 100
        logger.info(f"   Overhead vs single metric: {overhead_pct:.1f}%")

    def test_evaluation_metrics_performance(self) -> None:
        """Benchmark cross-collection evaluation metrics."""
        logger.info("\n" + "=" * 60)
        logger.info("🎯 EVALUATION METRICS PERFORMANCE")
        logger.info("=" * 60)

        # Create two scaled datasets
        n_entities = int(self.n * 100_000)
        logger.info(f"\n🔍 Creating two datasets with {n_entities:,} entities each...")

        edge_gen_a = generators.edges(n_entities)
        collection_a = sl.Collection.from_edges(edge_gen_a, show_progress=False)

        edge_gen_b = generators.edges(n_entities)
        collection_b = sl.Collection.from_edges(edge_gen_b, show_progress=False)

        # Create EntityFrame
        ef = sl.EntityFrame()
        ef.add_collection("col_a", collection_a)
        ef.add_collection("col_b", collection_b)

        # Test basic evaluation metrics
        logger.info("\n1️⃣ BASIC EVALUATION METRICS (F1, Precision, Recall)")

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
        basic_time = time.perf_counter() - start

        logger.info(f"   Time: {basic_time:.3f}s")
        logger.info(f"   F1: {result[0]['f1']:.4f}")
        logger.info(f"   Precision: {result[0]['precision']:.4f}")
        logger.info(f"   Recall: {result[0]['recall']:.4f}")
        logger.info("   Algorithm: Record-based (O(r) single-pass)")

        # Test advanced metrics
        logger.info("\n2️⃣ ADVANCED EVALUATION METRICS")

        # ARI
        start = time.perf_counter()
        result = ef.analyse(
            sl.col("col_a").at(0.8),
            sl.col("col_b").at(0.8),
            metrics=[sl.Metrics.eval.ari],
        )
        ari_time = time.perf_counter() - start
        logger.info(f"   ARI: {ari_time:.3f}s (value: {result[0]['ari']:.4f})")

        # NMI
        start = time.perf_counter()
        result = ef.analyse(
            sl.col("col_a").at(0.8),
            sl.col("col_b").at(0.8),
            metrics=[sl.Metrics.eval.nmi],
        )
        nmi_time = time.perf_counter() - start
        logger.info(f"   NMI: {nmi_time:.3f}s (value: {result[0]['nmi']:.4f})")

        # V-Measure
        start = time.perf_counter()
        result = ef.analyse(
            sl.col("col_a").at(0.8),
            sl.col("col_b").at(0.8),
            metrics=[sl.Metrics.eval.v_measure],
        )
        vmeasure_time = time.perf_counter() - start
        vmeasure_val = result[0]["v_measure"]
        logger.info(f"   V-Measure: {vmeasure_time:.3f}s (value: {vmeasure_val:.4f})")

        # BCubed metrics
        start = time.perf_counter()
        result = ef.analyse(
            sl.col("col_a").at(0.8),
            sl.col("col_b").at(0.8),
            metrics=[sl.Metrics.eval.bcubed_precision, sl.Metrics.eval.bcubed_recall],
        )
        bcubed_time = time.perf_counter() - start
        logger.info(f"   BCubed (P&R): {bcubed_time:.3f}s")
        logger.info(f"     Precision: {result[0]['bcubed_precision']:.4f}")
        logger.info(f"     Recall: {result[0]['bcubed_recall']:.4f}")

        # All metrics combined
        logger.info("\n3️⃣ ALL METRICS COMBINED")
        start = time.perf_counter()
        result = ef.analyse(
            sl.col("col_a").at(0.8),
            sl.col("col_b").at(0.8),
            metrics=[
                sl.Metrics.eval.f1,
                sl.Metrics.eval.precision,
                sl.Metrics.eval.recall,
                sl.Metrics.eval.ari,
                sl.Metrics.eval.nmi,
                sl.Metrics.eval.v_measure,
                sl.Metrics.eval.bcubed_precision,
                sl.Metrics.eval.bcubed_recall,
            ],
        )
        all_metrics_time = time.perf_counter() - start

        logger.info(f"   Time for 8 metrics: {all_metrics_time:.3f}s")
        logger.info(f"   Average per metric: {all_metrics_time / 8 * 1000:.1f}ms")

        # Calculate overhead
        individual_sum = basic_time + ari_time + nmi_time + vmeasure_time + bcubed_time
        efficiency = individual_sum / all_metrics_time
        logger.info(f"   Efficiency: {efficiency:.1f}x faster than individual calls")

    def test_sweep_comparison_performance(self) -> None:
        """Benchmark sweep comparison patterns and algorithm selection."""
        logger.info("\n" + "=" * 60)
        logger.info("🔄 SWEEP COMPARISON PERFORMANCE")
        logger.info("=" * 60)

        # Create scaled datasets
        n_entities = int(self.n * 500_000)  # 500k for sweep tests
        logger.info(f"\n🔍 Creating datasets with {n_entities:,} entities...")

        edge_gen_a = generators.edges(n_entities)
        collection_a = sl.Collection.from_edges(edge_gen_a, show_progress=False)

        edge_gen_b = generators.edges(n_entities)
        collection_b = sl.Collection.from_edges(edge_gen_b, show_progress=False)

        # Create EntityFrame
        ef = sl.EntityFrame()
        ef.add_collection("col_a", collection_a)
        ef.add_collection("col_b", collection_b)

        # Test 1: Same collection sweep (Delta algorithm)
        logger.info("\n1️⃣ SAME COLLECTION SWEEP (Delta algorithm)")

        start = time.perf_counter()
        result = ef.analyse(
            sl.col("col_a").sweep(0.5, 0.95, 0.05),
            metrics=[sl.Metrics.stats.entity_count, sl.Metrics.stats.entropy],
        )
        same_sweep_time = time.perf_counter() - start
        n_points = len(result)

        logger.info(f"   {n_points} points in {same_sweep_time:.3f}s")
        logger.info(f"   Throughput: {n_points / same_sweep_time:.1f} points/second")
        logger.info("   Algorithm: Delta-based O(k) incremental")

        # Show transition in entity counts
        counts = [r["entity_count"] for r in result]
        logger.info(f"   Entity count range: {min(counts):.0f} to {max(counts):.0f}")

        # Test 2: Point × Sweep comparison
        logger.info("\n2️⃣ POINT × SWEEP COMPARISON")

        start = time.perf_counter()
        result = ef.analyse(
            sl.col("col_a").at(0.8),
            sl.col("col_b").sweep(0.5, 0.95, 0.05),
            metrics=[sl.Metrics.eval.f1],
        )
        point_sweep_time = time.perf_counter() - start
        n_comparisons = len(result)

        logger.info(f"   {n_comparisons} comparisons in {point_sweep_time:.3f}s")
        throughput = n_comparisons / point_sweep_time
        logger.info(f"   Throughput: {throughput:.1f} comparisons/second")
        logger.info("   Algorithm: Record-based O(r)")

        # Test 3: Sweep × Point comparison (reverse)
        logger.info("\n3️⃣ SWEEP × POINT COMPARISON")

        start = time.perf_counter()
        result = ef.analyse(
            sl.col("col_a").sweep(0.5, 0.95, 0.05),
            sl.col("col_b").at(0.8),
            metrics=[sl.Metrics.eval.f1],
        )
        sweep_point_time = time.perf_counter() - start
        n_comparisons = len(result)

        logger.info(f"   {n_comparisons} comparisons in {sweep_point_time:.3f}s")
        is_symmetric = abs(sweep_point_time - point_sweep_time) < 0.5
        logger.info(f"   Symmetry check: {is_symmetric}")

        # Test 4: Sweep × Sweep comparison (Cartesian product)
        logger.info("\n4️⃣ SWEEP × SWEEP COMPARISON (Cartesian product)")

        # Use smaller sweeps for reasonable time
        start = time.perf_counter()
        result = ef.analyse(
            sl.col("col_a").sweep(0.6, 0.9, 0.1),  # 4 points
            sl.col("col_b").sweep(0.6, 0.9, 0.1),  # 4 points
            metrics=[sl.Metrics.eval.f1],
        )
        sweep_sweep_time = time.perf_counter() - start
        n_comparisons = len(result)

        logger.info(
            f"   {n_comparisons} comparisons (4×4 grid) in {sweep_sweep_time:.3f}s"
        )
        throughput = n_comparisons / sweep_sweep_time
        logger.info(f"   Throughput: {throughput:.1f} comparisons/second")
        logger.info("   Algorithm: Record-based O(r) - massive advantage")

        # Calculate theoretical speedup
        partition_a = collection_a.at(0.75)
        partition_b = collection_b.at(0.75)
        k1 = len(partition_a.entities)
        k2 = len(partition_b.entities)

        entity_comparisons = k1 * k2 * n_comparisons
        record_comparisons = n_entities * n_comparisons
        theoretical_speedup = entity_comparisons / record_comparisons

        logger.info("\n📈 ALGORITHM EFFICIENCY")
        logger.info(f"   Entity-based would need: {entity_comparisons:,} comparisons")
        logger.info(f"   Record-based needs: {record_comparisons:,} comparisons")
        logger.info(f"   Theoretical speedup: {theoretical_speedup:.0f}x")

    def test_reference_comparison_performance(self) -> None:
        """Benchmark reference collection comparisons."""
        logger.info("\n" + "=" * 60)
        logger.info("🎭 REFERENCE COMPARISON PERFORMANCE")
        logger.info("=" * 60)

        # Create datasets of different sizes
        n_entities_ref = int(self.n * 50_000)  # Smaller reference
        n_entities_test = int(self.n * 200_000)  # Larger test set

        logger.info(
            f"\n🔍 Creating reference ({n_entities_ref:,}) and "
            f"test ({n_entities_test:,}) datasets..."
        )

        edge_gen_ref = generators.edges(n_entities_ref)
        collection_ref = sl.Collection.from_edges(edge_gen_ref, show_progress=False)

        edge_gen_test = generators.edges(n_entities_test)
        collection_test = sl.Collection.from_edges(edge_gen_test, show_progress=False)

        # Create EntityFrame
        ef = sl.EntityFrame()
        ef.add_collection("reference", collection_ref)
        ef.add_collection("test", collection_test)

        # Test reference at fixed threshold vs test sweep
        logger.info("\n1️⃣ REFERENCE AT FIXED THRESHOLD")

        start = time.perf_counter()
        result = ef.analyse(
            sl.col("reference").at(0.9).reference(),  # Mark as reference
            sl.col("test").sweep(0.5, 0.95, 0.05),
            metrics=[
                sl.Metrics.eval.f1,
                sl.Metrics.eval.precision,
                sl.Metrics.eval.recall,
            ],
        )
        ref_fixed_time = time.perf_counter() - start
        n_comparisons = len(result)

        logger.info(f"   {n_comparisons} comparisons in {ref_fixed_time:.3f}s")
        ref_threshold = result[0].get("reference_threshold", "N/A")
        logger.info(f"   Reference marked: {ref_threshold}")

        # Find best threshold
        best_f1_idx = max(range(len(result)), key=lambda i: result[i]["f1"])
        best_result = result[best_f1_idx]
        logger.info(
            f"   Best F1: {best_result['f1']:.4f} at threshold "
            f"{best_result['test_threshold']:.2f}"
        )

        # Test reference sweep vs test at optimal
        logger.info("\n2️⃣ REFERENCE SWEEP VS TEST AT OPTIMAL")

        start = time.perf_counter()
        result = ef.analyse(
            sl.col("reference").sweep(0.7, 0.95, 0.05).reference(),
            sl.col("test").at(best_result["test_threshold"]),
            metrics=[sl.Metrics.eval.f1],
        )
        ref_sweep_time = time.perf_counter() - start
        n_comparisons = len(result)

        logger.info(f"   {n_comparisons} reference points in {ref_sweep_time:.3f}s")

        # Find reference threshold that gives best match
        best_ref_idx = max(range(len(result)), key=lambda i: result[i]["f1"])
        best_ref = result[best_ref_idx]
        logger.info(
            f"   Best reference threshold: {best_ref['reference_threshold']:.2f} "
            f"(F1: {best_ref['f1']:.4f})"
        )

    def test_memory_scaling_performance(self) -> None:
        """Benchmark memory usage and scaling behavior."""
        logger.info("\n" + "=" * 60)
        logger.info("💾 MEMORY SCALING PERFORMANCE")
        logger.info("=" * 60)

        # Test different dataset sizes
        sizes = [
            int(self.n * 10_000),
            int(self.n * 50_000),
            int(self.n * 100_000),
            int(self.n * 250_000),
        ]

        memory_results: list[dict[str, Any]] = []

        for n_entities in sizes:
            logger.info(f"\n🔍 Testing {n_entities:,} entities...")

            # Get baseline memory
            process = psutil.Process()
            baseline_memory = process.memory_info().rss / (1024 * 1024)  # MB

            # Create dataset
            edge_gen = generators.edges(n_entities)
            collection = sl.Collection.from_edges(edge_gen, show_progress=False)

            # Create EntityFrame
            ef = sl.EntityFrame()
            ef.add_collection("test", collection)

            # Measure memory after creation
            creation_memory = process.memory_info().rss / (1024 * 1024)  # MB

            # Run analysis
            start = time.perf_counter()
            result = ef.analyse(
                sl.col("test").sweep(0.5, 0.9, 0.1),
                metrics=[sl.Metrics.stats.entity_count, sl.Metrics.stats.entropy],
            )
            analysis_time = time.perf_counter() - start

            # Measure memory after analysis
            analysis_memory = process.memory_info().rss / (1024 * 1024)  # MB

            memory_delta = creation_memory - baseline_memory
            analysis_delta = analysis_memory - creation_memory

            memory_results.append({
                "entities": n_entities,
                "memory_mb": memory_delta,
                "analysis_overhead_mb": analysis_delta,
                "analysis_time": analysis_time,
                "points": len(result),
            })

            logger.info(f"   Memory used: {memory_delta:.1f}MB")
            logger.info(f"   Analysis overhead: {analysis_delta:.1f}MB")
            logger.info(f"   Analysis time: {analysis_time:.3f}s")

            # Clean up to free memory
            del ef, collection

        # Summary
        logger.info("\n📊 MEMORY SCALING SUMMARY")
        logger.info(f"{'Entities':>10} {'Memory':>10} {'Overhead':>10} {'Time':>10}")
        logger.info("-" * 42)

        for r in memory_results:
            logger.info(
                f"{r['entities']:>10,} {r['memory_mb']:>9.1f}MB "
                f"{r['analysis_overhead_mb']:>9.1f}MB {r['analysis_time']:>9.3f}s"
            )

        # Calculate scaling factor
        if len(memory_results) > 1:
            first = memory_results[0]
            last = memory_results[-1]
            entity_scale = last["entities"] / first["entities"]
            memory_scale = last["memory_mb"] / first["memory_mb"]
            time_scale = last["analysis_time"] / first["analysis_time"]

            logger.info(f"\n📈 SCALING ANALYSIS ({entity_scale:.1f}x entities)")
            logger.info(f"   Memory scaling: {memory_scale:.1f}x")
            logger.info(f"   Time scaling: {time_scale:.1f}x")
            efficiency_type = (
                "Linear" if memory_scale < entity_scale * 1.2 else "Superlinear"
            )
            logger.info(f"   Efficiency: {efficiency_type}")

    def test_large_scale_stress(self) -> None:
        """Stress test with maximum scale datasets."""
        logger.info("\n" + "=" * 60)
        logger.info("🚀 LARGE-SCALE STRESS TEST")
        logger.info("=" * 60)

        # Create production-scale dataset
        n_entities = int(self.n * 1_000_000)
        logger.info(f"\n🔍 Creating production dataset with {n_entities:,} entities...")
        logger.info("   This represents ~5M edges at production scale")

        # Check available memory
        available_gb = psutil.virtual_memory().available / (1024**3)
        required_gb = (n_entities * 5 * 150) / (1024**3)  # Rough estimate

        logger.info(f"   Available memory: {available_gb:.1f}GB")
        logger.info(f"   Estimated requirement: {required_gb:.1f}GB")

        if required_gb > available_gb * 0.8:
            logger.warning("   ⚠️  High memory usage expected - system may use swap")

        # Create dataset with progress
        start = time.perf_counter()
        edge_gen = generators.edges(n_entities)
        collection = sl.Collection.from_edges(edge_gen, show_progress=True)
        creation_time = time.perf_counter() - start

        logger.info(f"   Collection created in {creation_time:.2f}s")
        throughput = n_entities * 5 / creation_time
        logger.info(f"   Throughput: {throughput:,.0f} edges/second")

        # Create EntityFrame
        ef = sl.EntityFrame()
        ef.add_collection("large", collection)

        # Test 1: Full sweep analysis
        logger.info("\n1️⃣ FULL SWEEP ANALYSIS")

        start = time.perf_counter()
        result = ef.analyse(
            sl.col("large").sweep(0.0, 1.0, 0.1),
            metrics=[sl.Metrics.stats.entity_count],
        )
        sweep_time = time.perf_counter() - start

        logger.info(f"   11-point sweep in {sweep_time:.2f}s")
        logger.info(f"   Average per point: {sweep_time / 11:.3f}s")

        # Show entity count progression
        counts = [r["entity_count"] for r in result]
        logger.info(f"   Entity progression: {counts[0]:.0f} → {counts[-1]:.0f}")

        # Test 2: Cross-collection comparison at scale
        logger.info("\n2️⃣ CROSS-COLLECTION COMPARISON")

        # Create second large collection
        logger.info("   Creating second large collection...")
        edge_gen_2 = generators.edges(n_entities)
        collection_2 = sl.Collection.from_edges(edge_gen_2, show_progress=False)
        ef.add_collection("large_2", collection_2)

        start = time.perf_counter()
        result = ef.analyse(
            sl.col("large").at(0.8),
            sl.col("large_2").at(0.8),
            metrics=[
                sl.Metrics.eval.f1,
                sl.Metrics.eval.precision,
                sl.Metrics.eval.recall,
            ],
        )
        comparison_time = time.perf_counter() - start

        logger.info(
            f"   {n_entities:,} × {n_entities:,} comparison in {comparison_time:.2f}s"
        )
        logger.info(f"   F1: {result[0]['f1']:.4f}")

        # Test 3: Sweep × Sweep at scale (limited)
        logger.info("\n3️⃣ LIMITED SWEEP × SWEEP")

        start = time.perf_counter()
        result = ef.analyse(
            sl.col("large").sweep(0.7, 0.9, 0.1),  # 3 points
            sl.col("large_2").sweep(0.7, 0.9, 0.1),  # 3 points
            metrics=[sl.Metrics.eval.f1],
        )
        grid_time = time.perf_counter() - start

        logger.info(f"   3×3 grid (9 comparisons) in {grid_time:.2f}s")
        logger.info(f"   Average per comparison: {grid_time / 9:.3f}s")

        # Performance summary
        logger.info("\n✅ STRESS TEST SUMMARY")
        logger.info(f"   Dataset: {n_entities:,} entities (~{n_entities * 5:,} edges)")
        logger.info(f"   Creation: {creation_time:.2f}s")
        logger.info(f"   Full sweep: {sweep_time:.2f}s")
        logger.info(f"   Point comparison: {comparison_time:.2f}s")
        logger.info(f"   Grid comparison: {grid_time:.2f}s")

        # Check if within reasonable bounds
        if creation_time < 10:
            logger.info("   🚀 Excellent performance!")
        elif creation_time < 30:
            logger.info("   ✅ Good performance")
        else:
            logger.info("   ⚠️  Performance may need optimization")


def run_benchmarks(n: float = 1.0) -> None:
    """Run all analysis benchmarks programmatically."""
    logger.info(f"🚀 Starting Starlings Analysis Benchmarks (N={n})")
    logger.info("=" * 60)

    # Create test instance and set N parameter
    benchmark_tests = TestAnalysisBenchmarks()
    benchmark_tests.n = n
    benchmark_tests.setup_class()

    try:
        # Run each benchmark
        benchmark_tests.test_statistics_metrics_performance()
        benchmark_tests.test_evaluation_metrics_performance()
        benchmark_tests.test_sweep_comparison_performance()
        benchmark_tests.test_reference_comparison_performance()
        benchmark_tests.test_memory_scaling_performance()

        # Only run stress test for N >= 1
        if n >= 1.0:
            benchmark_tests.test_large_scale_stress()
        else:
            logger.info(f"\n⏭️  Skipping stress test for N={n} (only runs for N>=1)")

        logger.info("\n" + "=" * 60)
        logger.info("✅ ALL ANALYSIS BENCHMARKS COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"\n❌ BENCHMARK FAILED: {e}")
        raise


if __name__ == "__main__":
    # Parse N parameter from command line (default to 1)
    n = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    run_benchmarks(n)
