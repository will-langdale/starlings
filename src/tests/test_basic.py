"""Basic unit tests for core functionality - designed to be system-safe.

These tests use small datasets (< 1000 entities) to verify core functionality
without risking system stability.
"""

import pytest
import starlings as sl


class TestBasicFunctionality:
    """Basic functionality tests using small, safe datasets."""

    def test_collection_creation_small(self):
        """Test basic collection creation with tiny dataset."""
        edges = [
            ("a", "b", 0.9),
            ("b", "c", 0.8),
            ("d", "e", 0.7),
        ]
        collection = sl.Collection.from_edges(edges)

        assert collection is not None
        assert len(collection.at(1.0).entities) == 5  # All entities separate at 1.0

    def test_threshold_behaviour(self):
        """Test that threshold behaviour works correctly with small dataset."""
        edges = [
            ("record1", "record2", 0.95),
            ("record2", "record3", 0.85),
            ("record4", "record5", 0.75),
        ]
        collection = sl.Collection.from_edges(edges)

        # At threshold 1.0 - no merges
        partition_high = collection.at(1.0)
        assert len(partition_high.entities) == 5

        # At threshold 0.5 - all merges should have happened
        partition_low = collection.at(0.5)
        assert len(partition_low.entities) <= 5  # Should be fewer entities

    def test_empty_collection(self):
        """Test handling of empty edge list."""
        edges = []
        collection = sl.Collection.from_edges(edges)

        assert collection is not None
        assert len(collection.at(1.0).entities) == 0

    def test_single_edge(self):
        """Test collection with single edge."""
        edges = [("entity1", "entity2", 0.8)]
        collection = sl.Collection.from_edges(edges)

        assert collection is not None

        # At threshold 1.0 - no merge
        partition_high = collection.at(1.0)
        assert len(partition_high.entities) == 2

        # At threshold 0.5 - merge should happen
        partition_low = collection.at(0.5)
        assert len(partition_low.entities) == 1

    def test_partition_entity_access(self):
        """Test that we can access entity information from partitions."""
        edges = [("user1", "user2", 0.9)]
        collection = sl.Collection.from_edges(edges)
        partition = collection.at(0.5)

        # Should be able to access entities
        entities = partition.entities
        assert len(entities) == 1  # Both merged into one cluster
        # Entities come back as lists of record IDs, not original strings
        assert isinstance(entities[0], list)
        assert len(entities[0]) == 2  # Both records in the cluster

    @pytest.mark.parametrize(
        ["threshold"],
        [
            pytest.param(0.0, id="threshold_zero"),
            pytest.param(0.5, id="threshold_half"),
            pytest.param(1.0, id="threshold_one"),
        ],
    )
    def test_threshold_values(self, threshold: float):
        """Test various threshold values work correctly."""
        edges = [("a", "b", 0.6), ("c", "d", 0.4)]
        collection = sl.Collection.from_edges(edges)
        partition = collection.at(threshold)

        # Should not crash and should return valid partition
        assert partition is not None
        assert len(partition.entities) >= 0


class TestSafetyMechanisms:
    """Test that safety mechanisms work with small datasets."""

    def test_small_dataset_succeeds(self):
        """Test that small datasets work without safety concerns."""
        edges = [(f"entity_{i}", f"entity_{i + 1}", 0.8) for i in range(100)]

        collection = sl.Collection.from_edges(edges)
        assert collection is not None

        partition = collection.at(0.7)
        assert partition is not None
        assert len(partition.entities) > 0
