"""Tests for the Expression API."""

import logging
import time

import pytest
import starlings as sl


class TestExpressionAPI:
    """Test the expression API functionality."""

    def test_col_function_exists(self):
        """Test that the col() function is available."""
        col_expr = sl.col("test")
        assert str(col_expr) == "col('test')"

    def test_col_at_expression(self):
        """Test sl.col().at() creates point expressions."""
        expr = sl.col("splink").at(0.85)
        assert expr.expression_type == "point"
        assert expr.params["collection"] == "splink"
        assert expr.params["threshold"] == 0.85

    def test_col_sweep_expression(self):
        """Test sl.col().sweep() creates sweep expressions."""
        expr = sl.col("splink").sweep(0.5, 0.95, 0.05)
        assert expr.expression_type == "sweep"
        assert expr.params["collection"] == "splink"
        assert expr.params["start"] == 0.5
        assert expr.params["stop"] == 0.95
        assert expr.params["step"] == 0.05

    def test_reference_method_on_expression(self):
        """Test that .reference() method is available on expressions."""
        # Test with point expression
        expr = sl.col("truth").at(1.0)
        assert hasattr(expr, "reference")
        assert expr.is_reference is False

        # Mark as reference
        ref_expr = expr.reference()
        assert ref_expr.is_reference is True
        assert ref_expr is expr  # Should return self for chaining

        # Test with sweep expression
        sweep_expr = sl.col("splink").sweep(0.5, 0.9, 0.1)
        assert sweep_expr.is_reference is False

        ref_sweep = sweep_expr.reference()
        assert ref_sweep.is_reference is True

    def test_metrics_available(self):
        """Test that Metrics classes are available."""
        # Evaluation metrics
        assert hasattr(sl.Metrics.eval, "f1")
        assert hasattr(sl.Metrics.eval, "precision")
        assert hasattr(sl.Metrics.eval, "recall")

        # Statistics metrics
        assert hasattr(sl.Metrics.stats, "entity_count")
        assert hasattr(sl.Metrics.stats, "entropy")

        # Test metric objects have correct properties
        assert sl.Metrics.eval.f1.name == "f1"
        assert sl.Metrics.eval.f1.metric_type == "evaluation"
        assert sl.Metrics.stats.entity_count.name == "entity_count"
        assert sl.Metrics.stats.entity_count.metric_type == "statistics"


class TestExpressionIntegration:
    """Test the expression API integrated with EntityFrame."""

    @pytest.fixture
    def entity_frame_with_collections(self):
        """Create EntityFrame with test collections."""
        # Create test edges
        edges_a = [
            (0, 1, 0.9),
            (1, 2, 0.8),
            (3, 4, 0.85),
        ]
        edges_b = [
            (0, 1, 0.95),  # Same entity as A but higher threshold
            (2, 3, 0.7),  # Different grouping
        ]

        # Create collections
        collection_a = sl.Collection.from_edges(edges_a, show_progress=False)
        collection_b = sl.Collection.from_edges(edges_b, show_progress=False)

        # Create frame and add collections
        ef = sl.EntityFrame()
        ef.add_collection("a", collection_a)
        ef.add_collection("b", collection_b)

        return ef

    def test_point_comparison(self, entity_frame_with_collections):
        """Test point comparison using expression API."""
        ef = entity_frame_with_collections

        # Test that analyse method exists and can be called
        result = ef.analyse(
            sl.col("a").at(0.8),
            sl.col("b").at(0.8),
            metrics=[
                sl.Metrics.eval.f1,
                sl.Metrics.eval.precision,
                sl.Metrics.eval.recall,
            ],
        )

        # Check result format
        assert isinstance(result, list)
        assert len(result) == 1  # Single comparison point

        result_dict = result[0]
        assert isinstance(result_dict, dict)

        # Check threshold values are present
        assert "a_threshold" in result_dict
        assert "b_threshold" in result_dict
        assert result_dict["a_threshold"] == 0.8
        assert result_dict["b_threshold"] == 0.8

        # Check metrics are present
        assert "f1" in result_dict
        assert "precision" in result_dict
        assert "recall" in result_dict

        # Check metric values are reasonable
        for metric_name in ["f1", "precision", "recall"]:
            metric_value = result_dict[metric_name]
            assert isinstance(metric_value, int | float)
            assert 0.0 <= metric_value <= 1.0

    def test_single_collection_stats(self, entity_frame_with_collections):
        """Test single collection statistics using expression API."""
        ef = entity_frame_with_collections

        result = ef.analyse(
            sl.col("a").at(0.8),
            metrics=[sl.Metrics.stats.entity_count, sl.Metrics.stats.entropy],
        )

        assert isinstance(result, list)
        assert len(result) == 1

        result_dict = result[0]
        assert "a_threshold" in result_dict
        assert result_dict["a_threshold"] == 0.8

        assert "entity_count" in result_dict
        assert "entropy" in result_dict

        # Entity count should be positive integer
        assert result_dict["entity_count"] >= 0
        assert isinstance(result_dict["entity_count"], int | float)

        # Entropy should be non-negative float
        assert result_dict["entropy"] >= 0.0
        assert isinstance(result_dict["entropy"], int | float)

    def test_sweep_operation(self, entity_frame_with_collections):
        """Test sweep operation using expression API."""
        ef = entity_frame_with_collections

        result = ef.analyse(
            sl.col("a").sweep(0.7, 0.9, 0.1), metrics=[sl.Metrics.stats.entity_count]
        )

        # Should have multiple results for sweep
        assert isinstance(result, list)
        assert len(result) == 3  # 0.7, 0.8, 0.9

        # Check each threshold point
        expected_thresholds = [0.7, 0.8, 0.9]
        for i, result_dict in enumerate(result):
            assert "a_threshold" in result_dict
            assert abs(result_dict["a_threshold"] - expected_thresholds[i]) < 1e-10
            assert "entity_count" in result_dict
            assert result_dict["entity_count"] >= 0

    def test_default_metrics(self, entity_frame_with_collections):
        """Test that default metrics are applied when none specified."""
        ef = entity_frame_with_collections

        # Test multiple collections get comparison metrics by default
        result_comparison = ef.analyse(sl.col("a").at(0.8), sl.col("b").at(0.8))

        assert len(result_comparison) == 1
        result_dict = result_comparison[0]

        # Should include default comparison metrics
        assert "f1" in result_dict
        assert "precision" in result_dict
        assert "recall" in result_dict

        # Test single collection gets statistics metrics by default
        result_single = ef.analyse(sl.col("a").at(0.8))

        assert len(result_single) == 1
        result_dict = result_single[0]

        # Should include default statistics metrics
        assert "entity_count" in result_dict
        assert "entropy" in result_dict

    def test_sweep_step_constraint(self):
        """Test that sweep steps are constrained to 0.05 minimum."""
        # Test that small steps are rounded up to 0.05
        expr1 = sl.col("test").sweep(0.5, 0.7, 0.01)
        assert expr1.params["step"] == 0.05

        expr2 = sl.col("test").sweep(0.5, 0.7, 0.03)
        assert expr2.params["step"] == 0.05

        # Test that 0.05 and multiples are preserved
        expr3 = sl.col("test").sweep(0.5, 0.7, 0.05)
        assert expr3.params["step"] == 0.05

        expr4 = sl.col("test").sweep(0.5, 0.7, 0.1)
        assert expr4.params["step"] == 0.1

        expr5 = sl.col("test").sweep(0.5, 0.7, 0.15)
        assert abs(expr5.params["step"] - 0.15) < 1e-10

    def test_explicit_reference_marking(self, entity_frame_with_collections, caplog):
        """Test explicit reference marking with .reference()."""
        ef = entity_frame_with_collections

        with caplog.at_level(logging.INFO):
            # Explicitly mark truth as reference
            result = ef.analyse(
                sl.col("a").at(0.8),
                sl.col("b").at(0.9).reference(),  # Explicit reference
                metrics=[
                    sl.Metrics.eval.f1,
                    sl.Metrics.eval.precision,
                    sl.Metrics.eval.recall,
                ],
            )

        # Should not log about implicit reference since we provided explicit one
        assert "No explicit reference specified" not in caplog.text

        # Result should still work correctly
        assert len(result) == 1
        assert "f1" in result[0]
        assert "precision" in result[0]
        assert "recall" in result[0]

    def test_implicit_reference_marking(self, entity_frame_with_collections, caplog):
        """Test implicit reference marking (last expression)."""
        ef = entity_frame_with_collections

        with caplog.at_level(logging.INFO):
            # No explicit reference - b should become implicit reference
            result = ef.analyse(
                sl.col("a").at(0.8),
                sl.col("b").at(0.9),  # Will be implicit reference
                metrics=[sl.Metrics.eval.f1],
            )

        # Should log about using implicit reference
        assert "No explicit reference specified" in caplog.text
        assert "'b' (last expression) as implicit reference" in caplog.text

        # Result should work correctly
        assert len(result) == 1
        assert "f1" in result[0]

    def test_reference_with_sweep(self, entity_frame_with_collections, caplog):
        """Test reference marking with sweep operations."""
        ef = entity_frame_with_collections

        with caplog.at_level(logging.INFO):
            # Sweep vs point with explicit reference
            result = ef.analyse(
                sl.col("a").sweep(0.7, 0.9, 0.1),
                sl.col("b").at(0.85).reference(),  # Explicit reference
                metrics=[sl.Metrics.eval.precision, sl.Metrics.eval.recall],
            )

        # Should not log about implicit reference
        assert "No explicit reference specified" not in caplog.text

        # Should produce 3 results (one for each threshold in sweep)
        assert len(result) == 3
        for r in result:
            assert "precision" in r
            assert "recall" in r

    def test_multiple_references_error(self, entity_frame_with_collections):
        """Test that marking multiple collections as reference raises an error."""
        ef = entity_frame_with_collections

        # Attempting to mark both collections as reference should fail
        with pytest.raises(
            ValueError, match="Multiple collections marked as reference"
        ):
            ef.analyse(
                sl.col("a").at(0.8).reference(),  # First reference
                sl.col("b").at(0.9).reference(),  # Second reference - should error
                metrics=[sl.Metrics.eval.f1],
            )

    def test_single_collection_no_reference(self, entity_frame_with_collections):
        """Test that single collection analysis doesn't need reference."""
        ef = entity_frame_with_collections

        # Single collection with statistics metrics - no reference needed
        result = ef.analyse(
            sl.col("a").at(0.8),
            metrics=[sl.Metrics.stats.entity_count, sl.Metrics.stats.entropy],
        )

        assert len(result) == 1
        assert "entity_count" in result[0]
        assert "entropy" in result[0]

        # Marking reference on single collection should still work
        result_with_ref = ef.analyse(
            sl.col("a").at(0.8).reference(),  # Ignored for single collection
            metrics=[sl.Metrics.stats.entity_count],
        )

        assert len(result_with_ref) == 1
        assert "entity_count" in result_with_ref[0]


class TestLargeScaleExpressions:
    """Test expression API with large-scale datasets."""

    @pytest.mark.parametrize(
        ["scale", "max_duration"],
        [
            pytest.param(10_000, 15.0, id="10k_edges"),
            pytest.param(50_000, 30.0, id="50k_edges"),
            pytest.param(100_000, 60.0, id="100k_edges"),
        ],
    )
    def test_large_scale_cross_collection_comparison(
        self, scale: int, max_duration: float
    ):
        """Test cross-collection comparison at various scales."""
        # Generate test data with fewer entities for practical performance
        edges_a = sl.generate_entity_resolution_edges(scale, min(scale // 100, 10000))
        edges_b = sl.generate_entity_resolution_edges(scale, min(scale // 100, 10000))

        # Create collections
        collection_a = sl.Collection.from_edges(edges_a, show_progress=False)
        collection_b = sl.Collection.from_edges(edges_b, show_progress=False)

        # Create frame and add collections
        ef = sl.EntityFrame()
        ef.add_collection("a", collection_a)
        ef.add_collection("b", collection_b)

        # Test single point comparison
        start = time.time()
        result = ef.analyse(
            sl.col("a").at(0.8),
            sl.col("b").at(0.8),
            metrics=[
                sl.Metrics.eval.f1,
                sl.Metrics.eval.precision,
                sl.Metrics.eval.recall,
            ],
        )
        duration = time.time() - start

        # Verify result
        assert len(result) == 1
        assert "f1" in result[0]
        assert "precision" in result[0]
        assert "recall" in result[0]

        # Check performance - should complete within max duration
        assert duration < max_duration, f"Single point comparison took {duration:.2f}s"

    def test_large_scale_sweep_with_constraints(self):
        """Test that sweeps with 0.05 steps work efficiently at large scale."""
        # Generate 100k edges for practical test time
        edges = sl.generate_entity_resolution_edges(100_000, 10_000)
        collection = sl.Collection.from_edges(edges, show_progress=False)

        ef = sl.EntityFrame()
        ef.add_collection("large", collection)

        # Test sweep with 0.05 steps (5 thresholds)
        start = time.time()
        result = ef.analyse(
            sl.col("large").sweep(0.7, 0.9, 0.05),  # 5 thresholds
            metrics=[sl.Metrics.stats.entity_count],
        )
        duration = time.time() - start

        # Verify results
        assert len(result) == 5  # Should have 5 threshold points
        thresholds = [r["large_threshold"] for r in result]
        expected = [0.7, 0.75, 0.8, 0.85, 0.9]
        for actual, exp in zip(thresholds, expected, strict=False):
            assert abs(actual - exp) < 1e-10

        # All should have entity count
        for r in result:
            assert "entity_count" in r
            assert r["entity_count"] > 0

        # Performance check - should complete in reasonable time
        assert duration < 10.0, f"Sweep took {duration:.2f}s, expected < 10s"

    def test_cross_collection_sweep_performance(self):
        """Test cross-collection comparison with sweeps."""
        # Use smaller scale for cross-collection sweep test
        edges_a = sl.generate_entity_resolution_edges(1_000, 100)
        edges_b = sl.generate_entity_resolution_edges(1_000, 100)

        collection_a = sl.Collection.from_edges(edges_a, show_progress=False)
        collection_b = sl.Collection.from_edges(edges_b, show_progress=False)

        ef = sl.EntityFrame()
        ef.add_collection("a", collection_a)
        ef.add_collection("b", collection_b)

        # Test sweep vs point comparison
        start = time.time()
        result = ef.analyse(
            sl.col("a").sweep(0.8, 0.9, 0.05),  # 3 thresholds
            sl.col("b").at(0.85),  # Single point
            metrics=[sl.Metrics.eval.f1],
        )
        duration = time.time() - start

        # Should produce 3 results (cartesian product)
        assert len(result) == 3

        # Check all have correct structure
        for r in result:
            assert "a_threshold" in r
            assert "b_threshold" in r
            assert r["b_threshold"] == 0.85
            assert "f1" in r

        # Performance check - small scale should be fast
        assert duration < 2.0, f"Cross-collection sweep took {duration:.2f}s"
