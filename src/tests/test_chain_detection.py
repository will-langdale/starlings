"""Integration tests for chain pattern detection."""

import pytest
import starlings as sl


class TestChainDetection:
    """Test chain pattern detection and error handling."""

    def test_basic_functionality(self):
        """Test that small chains still work (below detection threshold)."""
        # Small test to verify system works - below 10k threshold
        edges = [(i, i + 1, 0.85) for i in range(100)]
        c = sl.Collection.from_edges(edges)
        p = c.at(0.8)
        assert len(p.entities) == 1

    def test_cluster_pattern_no_error(self):
        """Cluster pattern should not trigger chain detection."""
        # Create cluster pattern (100 clusters of 50 records)
        # Total 5000 edges in cluster pattern (should not trigger)
        edges = []
        for cluster_id in range(100):
            base = cluster_id * 50
            for i in range(49):
                edges.append((base + i, base + i + 1, 0.85))

        c = sl.Collection.from_edges(edges)

        # Should have many entities (clusters)
        p = c.at(0.8)
        assert len(p.entities) > 50

    @pytest.mark.slow
    def test_large_chain_raises_error(self):
        """Large chain pattern should raise an error."""
        # Create 200k chain edges - should trigger detection and error
        edges = [(i, i + 1, 0.85) for i in range(200_000)]

        # Chain pattern should be rejected with an error
        with pytest.raises(Exception) as exc_info:
            sl.Collection.from_edges(edges)

        error_msg = str(exc_info.value)
        assert "UNSUPPORTED DATA PATTERN" in error_msg
        assert "sequential chain" in error_msg
