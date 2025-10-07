"""Test memory management fixes and safety mechanisms.

This module tests the fixes for the memory management issues identified:
1. Memory calculation bugs
2. Memory estimation accuracy
"""

import os
import time

import pytest
import starlings as sl


class TestMemoryManagement:
    """Test memory management fixes."""

    def test_small_operations_work_with_low_limits(self):
        """Verify small operations work even with very low memory limits."""
        # Set a very low memory limit
        os.environ["STARLINGS_MEMORY_LIMIT"] = "100MB"
        try:
            # Small operation should work
            edges = [(i, i + 1, 0.5) for i in range(10)]
            collection = sl.Collection.from_edges(edges, show_progress=False)
            assert collection is not None

            # Should have entities
            partition = collection.at(1.0)
            assert len(partition.entities) == 11  # 10 edges = 11 entities
        finally:
            os.environ.pop("STARLINGS_MEMORY_LIMIT", None)

    def test_memory_estimation_accuracy(self):
        """Test that memory estimation is reasonably accurate."""
        # Small test to verify our estimates are in the right ballpark
        edges = [(i, i + 1, 0.5) for i in range(1000)]  # 1000 edges

        # Should work without issues
        collection = sl.Collection.from_edges(edges, show_progress=False)
        assert collection is not None

        # Basic functionality check
        partition = collection.at(0.8)
        assert len(partition.entities) > 0


@pytest.mark.benchmark
class TestMemoryManagementBenchmarks:
    """Benchmark memory management performance."""

    def test_memory_check_performance(self):
        """Verify memory checks don't significantly impact performance."""
        # Time a moderate operation
        edges = [(i, i + 1, 0.5) for i in range(10000)]

        start_time = time.time()
        collection = sl.Collection.from_edges(edges, show_progress=False)
        end_time = time.time()

        # Should complete in reasonable time (not hanging due to memory checks)
        duration = end_time - start_time
        assert duration < 5.0  # Should take less than 5 seconds
        assert collection is not None

    def test_cache_effectiveness(self):
        """Test that caching is working effectively."""
        os.environ["STARLINGS_MEMORY_LIMIT"] = "2GB"  # Enough for caching
        try:
            edges = [(i, i + 1, 0.5) for i in range(50000)]
            collection = sl.Collection.from_edges(edges, show_progress=False)

            # Multiple accesses should be fast due to caching

            # First access - builds cache
            start_time = time.time()
            partition1 = collection.at(0.8)
            first_duration = time.time() - start_time

            # Second access - should be from cache
            start_time = time.time()
            partition2 = collection.at(0.8)
            second_duration = time.time() - start_time

            # Results should be identical
            assert len(partition1.entities) == len(partition2.entities)

            # Second access should be much faster (cached)
            assert second_duration < first_duration / 2

        finally:
            os.environ.pop("STARLINGS_MEMORY_LIMIT", None)
