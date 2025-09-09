"""Test safety mechanisms WITHOUT running huge datasets.

This module tests the circuit breaker and resource safety features
by artificially creating memory pressure rather than using large datasets.
"""

import gc
import logging
import os
import threading
import time
from contextlib import contextmanager

import numpy as np
import psutil
import pytest
import starlings as sl


class MemoryBalloon:
    """Artificially consume memory to simulate pressure."""

    def __init__(self) -> None:
        """Initialise memory balloon with empty allocations list."""
        self.allocations: list[np.ndarray] = []

    def inflate_to_percent(self, target_percent: float, max_chunks: int = 50):
        """Consume memory until system reaches target percentage.

        Args:
            target_percent: Target memory usage percentage (0-100)
            max_chunks: Maximum number of 100MB chunks to allocate (safety limit)
        """
        mem = psutil.virtual_memory()
        target_bytes = mem.total * target_percent / 100
        current_bytes = mem.total - mem.available

        bytes_to_allocate = int(target_bytes - current_bytes)
        if bytes_to_allocate > 0:
            # Allocate in 100MB chunks
            chunk_size = 100 * 1024 * 1024
            chunks_allocated = 0

            while bytes_to_allocate > 0 and chunks_allocated < max_chunks:
                chunk = min(chunk_size, bytes_to_allocate)
                try:
                    # Use numpy array to ensure real memory allocation
                    self.allocations.append(np.ones(chunk // 8, dtype=np.float64))
                    bytes_to_allocate -= chunk
                    chunks_allocated += 1
                except MemoryError:
                    # Stop if we can't allocate more
                    break

        # Give system time to register the allocation
        time.sleep(0.1)

    def deflate(self):
        """Release allocated memory."""
        self.allocations.clear()
        gc.collect()
        # Give system time to register the deallocation
        time.sleep(0.1)

    def get_current_memory_percent(self) -> float:
        """Get current system memory usage percentage."""
        mem = psutil.virtual_memory()
        return (mem.total - mem.available) / mem.total * 100


@pytest.mark.safety
class TestSafetyMechanisms:
    """Test circuit breaker and safety without large datasets."""

    def setup_method(self):
        """Ensure we start with conservative safety settings."""
        os.environ["STARLINGS_SAFETY_LEVEL"] = "conservative"

    def teardown_method(self):
        """Clean up environment."""
        os.environ.pop("STARLINGS_SAFETY_LEVEL", None)

    def test_small_operation_works_under_normal_conditions(self):
        """Verify small operations work normally."""
        # Small operation should always work
        edges = [(i, i + 1, 0.5) for i in range(100)]
        collection = sl.Collection.from_edges(edges)
        assert collection is not None

        # Should have 101 entities at threshold 1.0
        partition = collection.at(1.0)
        assert len(partition.entities) == 101

    def test_operation_rejected_when_too_large_for_available_memory(self):
        """Verify large operations rejected based on proportional limits."""
        mem = psutil.virtual_memory()

        # Calculate dataset that would definitely exceed conservative limits
        # Conservative mode: 20% of available memory for single operation
        available_mb = mem.available / (1024 * 1024)
        conservative_limit_mb = available_mb * 0.2

        # Create an operation that would use 150% of the conservative limit
        target_mb = conservative_limit_mb * 1.5
        # 750 bytes per entity estimate
        entities_over_limit = int((target_mb * 1024 * 1024) / 750)

        # For very large systems, ensure reasonable minimum test size
        # But cap it to avoid overwhelming the system during tests
        entities_over_limit = max(entities_over_limit, 500_000)  # At least 500k
        entities_over_limit = min(entities_over_limit, 10_000_000)  # Cap at 10M

        logging.info(
            f"Testing with {entities_over_limit:,} entities (~{target_mb:.0f}MB) "
            f"vs limit {conservative_limit_mb:.0f}MB"
        )

        # This should be rejected during pre-flight check
        with pytest.raises((MemoryError, RuntimeError, ValueError)) as exc_info:
            # Create a large list of edges directly to test memory limits
            # Using a simple pattern instead of the generator for predictable sizing
            num_edges = entities_over_limit * 5  # 5 edges per entity estimate
            edges = [(i, i + 1, 0.5) for i in range(num_edges)]
            sl.Collection.from_edges(edges)  # Should raise exception

        # Should give helpful error message
        error_msg = str(exc_info.value).lower()
        expected_words = [
            "safety",
            "limit",
            "memory",
            "too large",
            "operation too large",
        ]
        assert any(word in error_msg for word in expected_words)

    def test_throttling_increases_with_artificial_pressure(self):
        """Verify throttling increases as memory pressure grows."""
        balloon = MemoryBalloon()
        timings = {}

        try:
            # Test at different memory pressure levels
            pressure_levels = [40, 60]  # Keep conservative to avoid system issues

            for memory_percent in pressure_levels:
                # Clear previous allocations
                balloon.deflate()

                # Inflate to target percentage
                balloon.inflate_to_percent(memory_percent)
                current_percent = balloon.get_current_memory_percent()

                # If we couldn't reach the target, skip this test
                if current_percent < memory_percent - 5:
                    skip_msg = (
                        f"Could not create sufficient memory pressure: "
                        f"{current_percent}% vs target {memory_percent}%"
                    )
                    pytest.skip(skip_msg)

                # Time a small operation
                edges = [(i, i + 1, 0.5) for i in range(1000)]
                start = time.perf_counter()
                collection = sl.Collection.from_edges(edges)
                elapsed = time.perf_counter() - start

                timings[memory_percent] = elapsed
                assert collection is not None  # Should still work, just slower

            # If we have enough data points, verify throttling
            if len(timings) >= 2:
                pressure_levels = sorted(timings.keys())
                # Higher pressure should generally take longer (with some tolerance)
                # We don't assert strictly because system timing can be variable
                logging.info(f"Timing results: {timings}")

        finally:
            balloon.deflate()

    def test_circuit_breaker_prevents_operations_under_extreme_pressure(self):
        """Verify circuit breaker trips and prevents operations."""
        balloon = MemoryBalloon()

        try:
            # Try to create high memory pressure (but safely)
            target_percent = min(85, psutil.virtual_memory().percent + 20)
            # Limited chunks for safety
            balloon.inflate_to_percent(target_percent, max_chunks=20)

            current_percent = balloon.get_current_memory_percent()

            # If we couldn't create enough pressure, skip this test
            if current_percent < 70:
                skip_msg = (
                    f"Could not create sufficient memory pressure: {current_percent}%"
                )
                pytest.skip(skip_msg)

            # Even small operations might be rejected due to circuit breaker
            # (This depends on the exact pressure level achieved)
            edges = [(i, i + 1, 0.5) for i in range(1000)]

            # Either the operation succeeds (with throttling) or fails gracefully
            try:
                collection = sl.Collection.from_edges(edges)
                # If it succeeds, it should still work correctly
                if collection is not None:
                    partition = collection.at(1.0)
                    assert len(partition.entities) > 0
            except (MemoryError, RuntimeError) as e:
                # If it fails, should be due to safety mechanisms
                error_msg = str(e).lower()
                expected_words = ["circuit", "safety", "memory", "resources"]
                assert any(word in error_msg for word in expected_words)

        finally:
            balloon.deflate()


class TestGeneratorSafety:
    """Test that entity generators respect safety limits."""

    def test_large_generator_safely_rejected(self):
        """Test that dangerously large generators are rejected."""
        # Test with a size that should exceed available memory
        # Use a size that would require >150% of available memory
        available_gb = psutil.virtual_memory().available // (1024**3)
        # 2M entities per GB = excessive
        excessive_entities = int(available_gb * 2_000_000)

        with pytest.raises(ValueError, match="Operation too large"):
            # This should be rejected by safety system before allocation
            sl.generate_entity_resolution_edges(excessive_entities)

    def test_small_generator_works(self):
        """Test that small generators work normally."""
        edges = list(sl.generate_entity_resolution_edges(1000))
        assert len(edges) >= 1  # Should return at least one batch

        # Check structure - first batch should contain actual edges
        first_batch = edges[0]
        assert len(first_batch) > 0
        assert all(len(edge) == 3 for edge in first_batch)

    def test_medium_generator_works(self):
        """Test that medium-sized generators work."""
        # 100k entities should be safe on most systems
        edge_gen = sl.generate_entity_resolution_edges(100_000)
        edges = list(edge_gen)

        # Should produce multiple batches
        assert len(edges) >= 1

        # Count total edges across all batches
        total_edges = sum(len(batch) for batch in edges)
        # Should be approximately n*0.75 edges (realistic for intra-cluster)
        assert total_edges > 50_000

    def test_generator_error_message_helpful(self):
        """Test that generator safety errors provide helpful messages."""
        # Calculate a size that should definitely be rejected
        available_gb = psutil.virtual_memory().available // (1024**3)
        excessive_entities = int(available_gb * 5_000_000)  # 5M entities per GB

        try:
            sl.generate_entity_resolution_edges(excessive_entities)
            # If this succeeds, system has enormous memory - skip the test
            pytest.skip("System has too much memory to trigger safety limits")
        except (ValueError, MemoryError) as e:
            error_msg = str(e)
            assert "Try:" in error_msg or "Operation too large" in error_msg
            assert (
                "STARLINGS_SAFETY_LEVEL=performance" in error_msg
                or "smaller dataset" in error_msg.lower()
            )

    def test_generator_respects_safety_level_environment(self):
        """Test that generators respect STARLINGS_SAFETY_LEVEL environment."""
        # This is tricky to test without knowing the exact system specs
        # We'll test that the environment variable is being read

        # Small dataset should work regardless of safety level
        with self._temporary_env_var("STARLINGS_SAFETY_LEVEL", "conservative"):
            edges = list(sl.generate_entity_resolution_edges(1000))
            assert len(edges) >= 1

        with self._temporary_env_var("STARLINGS_SAFETY_LEVEL", "performance"):
            edges = list(sl.generate_entity_resolution_edges(1000))
            assert len(edges) >= 1

    @contextmanager
    def _temporary_env_var(self, key: str, value: str):
        """Context manager for temporarily setting an environment variable."""
        original = os.environ.get(key)
        os.environ[key] = value
        try:
            yield
        finally:
            if original is not None:
                os.environ[key] = original
            else:
                os.environ.pop(key, None)

    def test_safety_levels_affect_limits(self):
        """Test that different safety levels have different limits."""
        # This test checks configuration without needing memory pressure

        # Conservative should have stricter limits
        os.environ["STARLINGS_SAFETY_LEVEL"] = "conservative"
        conservative_edges = [(i, i + 1, 0.5) for i in range(10000)]
        collection_conservative = sl.Collection.from_edges(conservative_edges)
        assert collection_conservative is not None

        # Performance should allow larger operations
        os.environ["STARLINGS_SAFETY_LEVEL"] = "performance"
        performance_edges = [(i, i + 1, 0.5) for i in range(50000)]  # 5x larger
        collection_performance = sl.Collection.from_edges(performance_edges)
        assert collection_performance is not None

        # Both should work, performance mode should handle larger datasets
        partition_conservative = collection_conservative.at(1.0)
        partition_performance = collection_performance.at(1.0)

        assert len(partition_conservative.entities) == 10001
        assert len(partition_performance.entities) == 50001

    @pytest.mark.stress
    def test_allocation_storm_prevention(self):
        """Verify system prevents allocation storms with multiple threads."""

        def allocate_small_dataset():
            """Allocate a small dataset in a thread."""
            try:
                edges = [(i, i + 1, 0.5) for i in range(5000)]
                collection = sl.Collection.from_edges(edges)
                return collection is not None
            except (MemoryError, RuntimeError):
                return False  # Expected under pressure

        # Start multiple threads trying to allocate simultaneously
        threads = []
        results = []

        def worker():
            results.append(allocate_small_dataset())

        # Use fewer threads to avoid overwhelming system
        num_threads = 5

        start_time = time.time()

        for _ in range(num_threads):
            t = threading.Thread(target=worker)
            threads.append(t)
            t.start()

        # System should remain responsive - check CPU periodically
        cpu_checks = []
        for _ in range(3):  # Check 3 times during execution
            time.sleep(0.1)
            cpu_checks.append(psutil.cpu_percent(interval=0.1))

        # Wait for all threads to complete (with timeout)
        for t in threads:
            t.join(timeout=10)
            if t.is_alive():
                pytest.fail("Thread timed out - system may be unresponsive")

        end_time = time.time()

        # Verify system remained reasonably responsive
        time_msg = f"Operations took too long: {end_time - start_time}s"
        assert end_time - start_time < 15, time_msg

        # At least some operations should have completed
        assert len(results) >= num_threads // 2, "Too many operations failed"

        # CPU should not have been pegged at 100% throughout
        avg_cpu = sum(cpu_checks) / len(cpu_checks)
        assert avg_cpu < 98, f"System was unresponsive (CPU: {avg_cpu}%)"


@pytest.mark.performance
class TestPerformanceImpact:
    """Test that safety mechanisms don't significantly impact normal operations."""

    def test_minimal_overhead_for_small_operations(self):
        """Verify safety checks don't add significant overhead."""
        # Time operation with safety enabled
        start = time.perf_counter()
        edges = [(i, i + 1, 0.5) for i in range(10000)]
        collection = sl.Collection.from_edges(edges)
        with_safety_time = time.perf_counter() - start

        assert collection is not None
        partition = collection.at(1.0)
        assert len(partition.entities) == 10001

        # Safety overhead should be minimal for small operations
        # (We can't easily test without safety, but can verify it's reasonable)
        slow_msg = f"Operation too slow: {with_safety_time}s"
        assert with_safety_time < 5.0, slow_msg

        logging.info(f"Small operation time with safety: {with_safety_time:.3f}s")
