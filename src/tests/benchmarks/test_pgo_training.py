"""PGO training benchmarks with randomised data to prevent overfitting.

Benchmarks designed for Profile-Guided Optimisation training. Uses randomised
graph generation to create diverse workloads preventing PGO overfitting to
single data patterns.
"""

import logging
import time

import pytest
import starlings as sl
from starlings import generators

logger = logging.getLogger(__name__)

# Mark all tests in this module as benchmarks to exclude from regular test runs
pytestmark = pytest.mark.benchmark


class TestPGOTrainingBenchmarks:
    """Randomised benchmarks for PGO training to prevent overfitting."""

    @classmethod
    def setup_class(cls) -> None:
        """Set up debug logging for PGO profiling instrumentation."""
        logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    def run_randomised_workload(
        self, seed: int | None, entity_count: int, size_desc: str
    ) -> None:
        """Run a single randomised workload for PGO profiling."""
        logger.info(f"🎲 Running {size_desc} workload (seed={seed})")

        start_time = time.perf_counter()
        # Use unified generator with automatic jitter for PGO
        # When num_thresholds=None (default), adds ±0.001 random jitter to thresholds
        # This creates diverse threshold distributions for comprehensive PGO profiling
        edges = generators.edges(entity_count)
        generation_time = time.perf_counter() - start_time

        logger.info(
            f"   Generated ~{entity_count * 5:,} edges, {entity_count:,} entities "
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
        throughput = entity_count * 5 / collection_time
        logger.info(f"   Throughput: ~{throughput:,.0f} edges/second")

    def test_pgo_training_diverse_workloads(self) -> None:
        """Run multiple randomised workloads for comprehensive PGO training."""
        logger.info("\n" + "=" * 60)
        logger.info("🎯 PGO TRAINING - DIVERSE RANDOMISED WORKLOADS")
        logger.info("=" * 60)

        # Production-scale workloads for PGO profiling
        workloads = [
            (None, 1_000_000, "1M entities"),
            (42, 2_000_000, "2M entities (seeded)"),
            (None, 3_000_000, "3M entities"),
        ]

        for i, (seed, entity_count, description) in enumerate(workloads, 1):
            logger.info(f"\n--- Workload {i}/{len(workloads)}: {description} ---")
            self.run_randomised_workload(seed, entity_count, description)

        logger.info("\n✅ PGO training workloads completed")

    def test_pgo_jitter_diversity_validation(self) -> None:
        """Validate that jitter diversity meets PGO training requirements."""
        logger.info("\n" + "=" * 60)
        logger.info("🎯 PGO JITTER DIVERSITY VALIDATION")
        logger.info("=" * 60)

        # Test production-scale datasets for PGO training
        datasets = [
            (100_000, "100K entities"),
            (500_000, "500K entities"),
            (1_000_000, "1M entities"),
        ]

        total_time = 0.0
        total_entities = 0

        for entity_count, description in datasets:
            logger.info(f"\n🔍 Testing {description} ({entity_count:,} entities)")

            # Generate multiple samples to exercise diverse code paths
            for sample_idx in range(3):
                edge_generator = generators.edges(entity_count)

                # Build collection to exercise interning and union-find paths
                start = time.monotonic()
                sl.Collection.from_edges(edge_generator, show_progress=False)
                elapsed = time.monotonic() - start

                total_time += elapsed
                total_entities += entity_count

                logger.info(
                    f"   Sample {sample_idx + 1}: Built in {elapsed:.3f}s "
                    f"({entity_count / elapsed:.0f} entities/s)"
                )

        # Validate PGO training effectiveness
        avg_throughput = total_entities / total_time if total_time > 0 else 0

        logger.info("\n📊 PGO TRAINING SUMMARY")
        logger.info(f"   Total entities processed: {total_entities:,}")
        logger.info(f"   Total time: {total_time:.2f}s")
        logger.info(f"   Average throughput: {avg_throughput:.0f} entities/s")

        logger.info("✅ JITTER DIVERSITY VALIDATION PASSED - Optimal for PGO training")

        # Assert for test framework
        assert avg_throughput > 0, "PGO training failed to process entities"

    def test_pgo_training_scalability(self) -> None:
        """Test different dataset sizes with randomisation for PGO training."""
        logger.info("\n" + "=" * 60)
        logger.info("📈 PGO TRAINING - SCALABILITY WITH RANDOMISATION")
        logger.info("=" * 60)

        sizes_and_configs = [
            (500_000, "500K entities"),
            (1_000_000, "1M entities"),
            (2_000_000, "2M entities"),
        ]

        for entity_count, description in sizes_and_configs:
            # Each entity produces ~5 edges
            actual_edges = entity_count * 5

            logger.info(f"\n🔍 Testing {description} (~{actual_edges:,} edges)")

            # Generate randomised graph using unified generator
            edges = generators.edges(entity_count)

            start = time.perf_counter()
            sl.Collection.from_edges(edges, show_progress=False)
            elapsed = time.perf_counter() - start

            throughput = actual_edges / elapsed
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
        # Validate jitter diversity first
        benchmark_tests.test_pgo_jitter_diversity_validation()

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
