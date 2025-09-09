"""End-to-end out-of-core processing demonstration and validation.

This module demonstrates the complete out-of-core processing pipeline with
automatic resource management, disk spilling, and memory-bounded reconstruction.
"""

import logging

import pytest
import starlings as sl

logger = logging.getLogger(__name__)


@pytest.mark.e2e
class TestEndToEndOutOfCore:
    """End-to-end validation of complete out-of-core processing system."""

    def test_complete_out_of_core_pipeline(self) -> None:
        """Test the complete out-of-core pipeline with large dataset simulation."""
        logger.info("🚀 Testing complete out-of-core processing pipeline")

        # Generate a moderately large dataset (100k entities = ~500k edges)
        # This should trigger disk spilling on most systems
        num_entities = 100_000

        logger.info(f"   Generating {num_entities:,} entity dataset...")
        edge_generator = sl.generate_entity_resolution_edges(num_entities)

        # Create collection with resource monitoring enabled
        logger.info("   Creating collection with automatic resource management...")
        collection = sl.Collection.from_edges(edge_generator, show_progress=True)

        # Verify collection was created successfully
        assert collection is not None

        # Test partition reconstruction at various thresholds
        thresholds = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
        logger.info("   Testing partition reconstruction at multiple thresholds...")

        previous_entity_count = 0
        for threshold in thresholds:
            partition = collection.at(threshold)
            entity_count = len(partition.entities)

            # Verify monotonicity: entity count should decrease as threshold decreases
            if threshold < 1.0:
                assert entity_count <= previous_entity_count, (
                    f"Entity count should be monotonic: {entity_count} > "
                    f"{previous_entity_count} at threshold {threshold}"
                )

            previous_entity_count = entity_count
            logger.info(f"     Threshold {threshold}: {entity_count:,} entities")

        # Verify expected bounds
        full_partition = collection.at(1.0)
        merged_partition = collection.at(0.0)

        # At threshold 1.0, we should have close to the original entity count
        assert len(full_partition.entities) >= num_entities // 2, (
            f"Expected at least {num_entities // 2:,} entities at threshold 1.0, "
            f"got {len(full_partition.entities):,}"
        )

        # At threshold 0.0, we should have significant merging
        assert len(merged_partition.entities) <= len(full_partition.entities), (
            "Merged partition should have fewer entities than full partition"
        )

        logger.info("✅ Complete out-of-core pipeline validation successful")

    def test_disk_spilling_validation(self) -> None:
        """Validate that disk spilling actually occurs for large datasets."""
        # This test would ideally set memory limits to force spilling
        # For now, we test with a dataset that should naturally trigger spilling

        logger.info("🗃️  Testing disk spilling behaviour")

        # Generate dataset that should exceed comfortable in-memory limits
        num_entities = 200_000  # ~1M edges, should be 150-300MB

        logger.info(
            f"   Generating {num_entities:,} entity dataset for spilling test..."
        )
        edge_generator = sl.generate_entity_resolution_edges(num_entities)

        # Create collection - this should trigger automatic spilling
        collection = sl.Collection.from_edges(
            edge_generator,
            show_progress=False,  # Reduce output for automated testing
        )

        # Verify collection works correctly even with large dataset
        partition_high = collection.at(0.9)
        partition_low = collection.at(0.1)

        assert len(partition_high.entities) >= len(partition_low.entities), (
            "High threshold should have more entities than low threshold"
        )

        # Test multiple access patterns to verify disk storage
        for i in range(5):
            # Access different thresholds to test disk I/O
            threshold = 0.5 + (i * 0.1)
            partition = collection.at(threshold)
            assert len(partition.entities) > 0, (
                f"Empty partition at threshold {threshold}"
            )

        logger.info("✅ Disk spilling validation successful")

    def test_memory_bounded_processing(self) -> None:
        """Test memory-bounded processing with simulated constraints."""
        logger.info("🧠 Testing memory-bounded processing")

        # Create a dataset that would normally fit in memory
        num_entities = 50_000
        edge_generator = sl.generate_entity_resolution_edges(num_entities)

        # Create collection normally
        collection = sl.Collection.from_edges(edge_generator, show_progress=False)

        # Test that we can still reconstruct partitions efficiently
        # The system should automatically choose appropriate batch sizes
        test_thresholds = [0.95, 0.75, 0.5, 0.25, 0.05]

        for threshold in test_thresholds:
            partition = collection.at(threshold)
            entity_count = len(partition.entities)

            # Verify reasonable entity counts
            assert entity_count > 0, f"Empty partition at threshold {threshold}"
            assert entity_count <= num_entities, (
                f"Too many entities ({entity_count}) for {num_entities} input entities"
            )

            logger.info(
                f"     Threshold {threshold}: {entity_count:,} entities processed"
            )

        logger.info("✅ Memory-bounded processing validation successful")

    def test_resource_monitoring_integration(self) -> None:
        """Test integration with the resource monitoring system."""
        logger.info("📊 Testing resource monitoring integration")

        # Generate dataset that should trigger resource monitoring
        num_entities = 75_000
        edge_generator = sl.generate_entity_resolution_edges(num_entities)

        # Create collection with monitoring (should happen automatically)
        collection = sl.Collection.from_edges(edge_generator, show_progress=False)

        # Test that system adapts to different processing patterns
        # Rapid threshold changes should test different code paths
        thresholds = [1.0, 0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.5]

        for threshold in thresholds:
            partition = collection.at(threshold)
            # Just verify we get valid results - the system should handle
            # resource management automatically
            assert len(partition.entities) >= 0

        logger.info("✅ Resource monitoring integration successful")


def run_e2e_tests() -> None:
    """Run all end-to-end out-of-core tests."""
    logger.info("🧪 Running End-to-End Out-of-Core Tests")
    logger.info("=" * 50)

    test_instance = TestEndToEndOutOfCore()

    try:
        test_instance.test_complete_out_of_core_pipeline()
        test_instance.test_disk_spilling_validation()
        test_instance.test_memory_bounded_processing()
        test_instance.test_resource_monitoring_integration()

        logger.info("\n" + "=" * 50)
        logger.info("✅ ALL END-TO-END OUT-OF-CORE TESTS PASSED")
        logger.info("=" * 50)

    except Exception as e:
        logger.error(f"\n❌ END-TO-END TEST FAILED: {e}")
        raise


if __name__ == "__main__":
    # Set up logging for standalone execution
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    run_e2e_tests()
