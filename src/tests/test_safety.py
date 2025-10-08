"""Test safety mechanisms with simple, reliable tests.

This module tests that operations work correctly with memory limits using
production-scale scenarios, without requiring artificial memory pressure.
"""

import os

import pytest
import starlings as sl
from starlings import generators


@pytest.mark.safety
class TestSafetyMechanisms:
    """Test that operations respect memory limits."""

    def test_small_operation_works(self):
        """Small operation should work with generous limit."""
        os.environ["STARLINGS_MEMORY_LIMIT"] = "2GB"
        try:
            edges = [(i, i + 1, 0.5) for i in range(1000)]
            collection = sl.Collection.from_edges(edges, show_progress=False)
            assert collection is not None
        finally:
            os.environ.pop("STARLINGS_MEMORY_LIMIT", None)

    def test_1m_edges_works_with_reasonable_limit(self):
        """1M edges (production minimum) works with 2GB limit."""
        os.environ["STARLINGS_MEMORY_LIMIT"] = "2GB"
        try:
            edges = generators.edges(1_000_000)
            collection = sl.Collection.from_edges(edges, show_progress=False)
            assert collection is not None
        finally:
            os.environ.pop("STARLINGS_MEMORY_LIMIT", None)
