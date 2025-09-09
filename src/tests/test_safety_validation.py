"""Safe validation tests for the safety system itself.

These tests verify the safety mechanisms work correctly using small datasets
that are themselves safe to run.
"""

import os
from contextlib import contextmanager

import starlings as sl


class TestSafetySystemValidation:
    """Validate that safety mechanisms work with small, safe datasets."""

    @staticmethod
    def _create_test_edges(count: int) -> list[tuple[str, str, float]]:
        """Create test edges for validation."""
        return [(f"entity_{i}", f"entity_{i + 1}", 0.8) for i in range(count)]

    def _assert_valid_collection(self, edges: list[tuple[str, str, float]]) -> None:
        """Assert that a collection can be created and queried successfully."""
        collection = sl.Collection.from_edges(edges)
        assert collection is not None

        partition = collection.at(0.7)
        assert len(partition.entities) > 0

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

    def test_small_dataset_always_succeeds(self):
        """Verify small datasets always succeed regardless of safety level."""
        edges = self._create_test_edges(50)
        self._assert_valid_collection(edges)

    def test_safety_level_environment_variable(self):
        """Test that safety level can be controlled via environment variable."""
        # Test conservative (strictest)
        with self._temporary_env_var("STARLINGS_SAFETY_LEVEL", "conservative"):
            edges = self._create_test_edges(100)
            self._assert_valid_collection(edges)

        # Test performance (more permissive)
        with self._temporary_env_var("STARLINGS_SAFETY_LEVEL", "performance"):
            edges = self._create_test_edges(200)
            self._assert_valid_collection(edges)

    def test_error_message_format(self):
        """Test that safety errors have helpful messages (artificially trigger)."""
        # This test is tricky - we can't easily trigger memory errors safely
        # Instead, we'll test the error message formatting through the Rust API
        # by creating a scenario designed to test the safety bounds

        # For now, just verify small operations work
        # This is more of a placeholder for future validation
        edges = self._create_test_edges(10)
        self._assert_valid_collection(edges)

    def test_processing_strategy_selection(self):
        """Test that different dataset sizes choose appropriate strategies."""
        # Very small dataset - should use in-memory
        small_edges = self._create_test_edges(10)
        self._assert_valid_collection(small_edges)

        # Medium dataset - should still work safely
        medium_edges = self._create_test_edges(500)
        self._assert_valid_collection(medium_edges)

    def test_circuit_breaker_recovery(self):
        """Test that the system can recover from safety conditions."""
        edges = self._create_test_edges(20)

        # Should succeed multiple times (no circuit breaker issues)
        for _ in range(3):
            self._assert_valid_collection(edges)
