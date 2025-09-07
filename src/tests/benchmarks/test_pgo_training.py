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

    def _validate_jitter_diversity(
        self, edges: list[tuple[int, int, float]], workload_desc: str
    ) -> None:
        """Validate that jitter provides sufficient threshold diversity for PGO."""
        # Collect unique thresholds (rounded to 6 decimal places)
        unique_thresholds = set()
        for _, _, threshold in edges:
            rounded = round(threshold * 1_000_000)  # 6 decimal places
            unique_thresholds.add(rounded)

        diversity_count = len(unique_thresholds)

        # For effective PGO training, we need high threshold diversity
        # With jitter, we should see thousands of unique values
        min_expected_diversity = 1000
        excellent_diversity = 5000

        if diversity_count < min_expected_diversity:
            logger.warning(
                f"   ⚠️  LOW JITTER DIVERSITY: {diversity_count:,} unique thresholds "
                f"(expected >{min_expected_diversity:,}) - PGO training may overfit"
            )
        elif diversity_count >= excellent_diversity:
            logger.info(
                f"   ✅ EXCELLENT JITTER DIVERSITY: {diversity_count:,} unique "
                f"thresholds - optimal for PGO training"
            )
        else:
            logger.info(
                f"   ✅ GOOD JITTER DIVERSITY: {diversity_count:,} unique thresholds "
                f"- sufficient for PGO training"
            )

    def run_randomized_workload(
        self, seed: int | None, jitter: float, size_desc: str
    ) -> None:
        """Run a single randomised workload for PGO profiling."""
        logger.info(f"🎲 Running {size_desc} workload (seed={seed}, jitter={jitter}%)")

        start_time = time.perf_counter()
        # Use unified generator with automatic jitter for PGO
        # (seed affects internal randomness)
        edges = sl.generate_entity_resolution_edges(1_000_000, num_thresholds=None)
        generation_time = time.perf_counter() - start_time

        # Validate jitter diversity for PGO training effectiveness
        self._validate_jitter_diversity(edges, size_desc)

        logger.info(
            f"   Generated {len(edges):,} edges, {1_000_000:,} entities "
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

    def test_pgo_jitter_diversity_validation(self) -> None:
        """Validate that jitter diversity meets PGO training requirements."""
        logger.info("\n" + "=" * 60)
        logger.info("🎯 PGO JITTER DIVERSITY VALIDATION")
        logger.info("=" * 60)

        # Test multiple datasets to ensure consistent diversity
        datasets = [
            (10_000, "Small validation"),
            (100_000, "Medium validation"),
            (1_000_000, "Large validation"),
        ]

        all_thresholds = set()

        for entity_count, description in datasets:
            logger.info(f"\n🔍 Testing {description} ({entity_count:,} entities)")

            # Generate multiple samples to accumulate diverse thresholds
            for sample_idx in range(3):
                edges = sl.generate_entity_resolution_edges(
                    entity_count, num_thresholds=None
                )
                sample_thresholds = set()

                for _, _, threshold in edges:
                    rounded = round(threshold * 1_000_000)  # 6 decimal places
                    sample_thresholds.add(rounded)
                    all_thresholds.add(rounded)

                logger.info(
                    f"   Sample {sample_idx + 1}: {len(sample_thresholds):,} "
                    f"unique thresholds"
                )

        total_diversity = len(all_thresholds)

        # Validate overall diversity across all datasets
        min_required_diversity = 5000  # For effective PGO training

        logger.info(
            f"\n📊 TOTAL ACCUMULATED DIVERSITY: {total_diversity:,} unique thresholds"
        )

        if total_diversity >= min_required_diversity:
            logger.info(
                "✅ JITTER DIVERSITY VALIDATION PASSED - Optimal for PGO training"
            )
        else:
            logger.warning(
                f"⚠️  JITTER DIVERSITY BELOW MINIMUM: {total_diversity:,} < "
                f"{min_required_diversity:,}"
            )
            logger.warning("   PGO training may not achieve optimal performance gains")

        # Assert for test framework
        assert total_diversity >= min_required_diversity, (
            f"Insufficient jitter diversity for PGO training: {total_diversity:,} < "
            f"{min_required_diversity:,}"
        )

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

            # Generate randomised graph using unified generator
            # Calculate entity count from desired edge count
            entity_count = max(size // 5, 1000)  # Each entity produces ~5 edges
            edges = sl.generate_entity_resolution_edges(
                entity_count, num_thresholds=None
            )
            edges = edges[:size]  # Trim to exact edge size for comparison

            # Validate jitter diversity for PGO training effectiveness
            self._validate_jitter_diversity(edges, description)

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
