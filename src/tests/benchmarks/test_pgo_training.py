"""PGO training benchmarks with randomised data to prevent overfitting.

Benchmarks designed for Profile-Guided Optimisation training. Uses randomised
graph generation to create diverse workloads preventing PGO overfitting to
single data patterns.
"""

import logging
import time

import pytest
import starlings as sl

logger = logging.getLogger(__name__)

# Mark all tests in this module as benchmarks to exclude from regular test runs
pytestmark = pytest.mark.benchmark


class TestPGOTrainingBenchmarks:
    """Randomised benchmarks for PGO training to prevent overfitting."""

    @classmethod
    def setup_class(cls) -> None:
        """Set up debug logging for PGO profiling instrumentation."""
        logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    def run_randomized_workload(
        self, seed: int | None, jitter: float, size_desc: str
    ) -> None:
        """Run a single randomised workload for PGO profiling."""
        logger.info(f"🎲 Running {size_desc} workload (seed={seed}, jitter={jitter}%)")

        start_time = time.perf_counter()
        edges, total_nodes = sl.generate_production_1m_graph_randomized(seed, jitter)
        generation_time = time.perf_counter() - start_time

        logger.info(
            f"   Generated {len(edges):,} edges, {total_nodes:,} nodes "
            f"in {generation_time:.3f}s"
        )

        # Collection creation - the main target for PGO optimisation
        collection_start = time.perf_counter()
        collection = sl.Collection.from_edges(edges)
        collection_time = time.perf_counter() - collection_start

        # Partition access with varied thresholds - different from regular benchmarks
        partition_times = []
        thresholds = [0.95, 0.8, 0.65, 0.4, 0.2]

        for threshold in thresholds:
            partition_start = time.perf_counter()
            partition = collection.at(threshold)
            partition_time = time.perf_counter() - partition_start
            partition_times.append(partition_time)

            # Touch entity data to ensure full materialisation
            len(partition.entities)

        total_partition_time = sum(partition_times)

        logger.info(
            f"   Collection: {collection_time:.3f}s, "
            f"Partitions: {total_partition_time:.3f}s"
        )
        logger.info(f"   Throughput: {len(edges) / collection_time:,.0f} edges/second")

    def test_pgo_training_diverse_workloads(self) -> None:
        """Run multiple randomised workloads for comprehensive PGO training."""
        logger.info("\n" + "=" * 60)
        logger.info("🎯 PGO TRAINING - DIVERSE RANDOMISED WORKLOADS")
        logger.info("=" * 60)

        # Lighter workloads for PGO profiling - prioritise diversity over size
        workloads = [
            (None, 8.0, "Medium jitter (8%)"),
            (42, 12.0, "Seeded high jitter (12%)"),
            (None, 15.0, "High jitter (15%)"),
        ]

        for i, (seed, jitter, description) in enumerate(workloads, 1):
            logger.info(f"\n--- Workload {i}/{len(workloads)}: {description} ---")
            self.run_randomized_workload(seed, jitter, description)

        logger.info("\n✅ PGO training workloads completed")

    def test_pgo_training_scalability(self) -> None:
        """Test different dataset sizes with randomisation for PGO training."""
        logger.info("\n" + "=" * 60)
        logger.info("📈 PGO TRAINING - SCALABILITY WITH RANDOMISATION")
        logger.info("=" * 60)

        sizes_and_configs = [
            (50_000, "Small randomised"),
            (200_000, "Medium randomised"),
            (500_000, "Large randomised"),
        ]

        for size, description in sizes_and_configs:
            logger.info(f"\n🔍 Testing {description} ({size:,} edges)")

            # Generate randomised graph with slight jitter variation by size
            jitter = 8.0 + size * 0.00001
            edges, _ = sl.generate_production_1m_graph_randomized(None, jitter)
            edges = edges[:size]

            start = time.perf_counter()
            sl.Collection.from_edges(edges)
            elapsed = time.perf_counter() - start

            throughput = size / elapsed
            logger.info(
                f"   Time: {elapsed:.3f}s, Throughput: {throughput:,.0f} edges/second"
            )


def run_pgo_training() -> None:
    """Run PGO training benchmarks programmatically."""
    logger.info("🎯 Starting PGO Training Benchmarks")
    logger.info("=" * 60)

    benchmark_tests = TestPGOTrainingBenchmarks()
    benchmark_tests.setup_class()

    try:
        # Run diverse workloads for comprehensive PGO profiling
        benchmark_tests.test_pgo_training_diverse_workloads()
        benchmark_tests.test_pgo_training_scalability()

        logger.info("\n" + "=" * 60)
        logger.info("✅ PGO TRAINING BENCHMARKS COMPLETED")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"\n❌ PGO TRAINING FAILED: {e}")
        raise


if __name__ == "__main__":
    run_pgo_training()
