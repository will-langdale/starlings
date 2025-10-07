"""End-to-end test mimicking real user exploratory data analysis workflow."""

import logging
import time

import pytest
import starlings as sl
from starlings import generators

logger = logging.getLogger(__name__)


@pytest.mark.e2e
def test_user_eda_workflow():
    """Production-scale EDA workflow: Million-record minimum scale for library."""
    # Million-scale is the MINIMUM expected dataset size for production use

    # Time graph generation using unified entity resolution generator
    start_time = time.monotonic()
    edges = generators.edges(1_000_000)
    total_nodes = 1_000_000  # Unified generator creates exactly n entities at
    # threshold 1.0
    graph_time = time.monotonic() - start_time

    # Time collection creation (optimised Python->Rust boundary)
    start_time = time.monotonic()
    collection = sl.Collection.from_edges(edges)
    collection_time = time.monotonic() - start_time

    # Test hierarchical behaviour: unified generator creates specific entity counts
    # At 1.0: exactly n entities, at ~0.9: exactly n/2 entities (pairs merge)

    # Test exact endpoints with corrected expectations
    # The unified generator guarantees exactly n entities at 1.0 and n/2 entities at 0.0
    entities_at_1_0 = collection.at(1.0).num_entities
    entities_at_0_0 = collection.at(0.0).num_entities

    assert entities_at_1_0 == total_nodes, (
        f"Expected {total_nodes} entities at 1.0, got {entities_at_1_0}"
    )
    assert entities_at_0_0 == total_nodes // 2, (
        f"Expected {total_nodes // 2} entities at 0.0, got {entities_at_0_0}"
    )

    # Test hierarchical behaviour around the main transition (~0.9)
    test_thresholds = [0.95, 0.9, 0.8, 0.5]
    entity_counts = [collection.at(t).num_entities for t in test_thresholds]

    # Should see transition from n entities towards n/2 entities as threshold decreases
    # Note: exact transition point depends on threshold distribution
    # The n/2 guarantee is only at exactly threshold 0.0, not at intermediate thresholds
    assert entity_counts[-1] >= total_nodes // 2, (
        "Should be transitioning towards half entity count at lower thresholds"
    )

    # Ensure monotonic decrease (entities can only decrease as threshold decreases)
    for i in range(len(entity_counts) - 1):
        assert entity_counts[i] >= entity_counts[i + 1], (
            f"Entity count should decrease: {entity_counts[i]} >= "
            f"{entity_counts[i + 1]} at thresholds {test_thresholds[i]} -> "
            f"{test_thresholds[i + 1]}"
        )

    # Test precision: at threshold 1.0 should have all singletons (guaranteed by fix)
    singleton_partition = collection.at(1.0)
    assert singleton_partition.num_entities == total_nodes, (
        f"Expected {total_nodes} singletons at 1.0"
    )

    # Quick EDA sweep across key thresholds for unified generator
    eda_thresholds = [1.0, 0.95, 0.9, 0.8, 0.5, 0.0]
    eda_counts = [collection.at(t).num_entities for t in eda_thresholds]

    # Calculate expected edge count (n entities * 5 edges per entity on average)
    expected_edges = total_nodes * 5

    logger.info(
        "EDA workflow: ~%d edges, %d total nodes. Graph: %.2fs, "
        "Collection: %.2fs. Entity counts: %s",
        expected_edges,
        total_nodes,
        graph_time,
        collection_time,
        dict(zip(eda_thresholds, eda_counts, strict=False)),
    )

    # Test EntityFrame with multiple collections for memory sharing
    frame = sl.EntityFrame()

    # Add the original collection to frame
    frame.add_collection("full_dataset", collection)

    # Create a simple second collection with different characteristics
    simple_edges = [(1, 2, 0.9), (2, 3, 0.8), (4, 5, 0.7)]
    simple_collection = sl.Collection.from_edges(simple_edges)
    frame.add_collection("simple", simple_collection)

    assert len(frame) == 2
    assert set(frame.collection_names()) == {"full_dataset", "simple"}

    # Test dictionary-style access returns views
    full_view = frame["full_dataset"]
    simple_view = frame["simple"]

    assert full_view.is_view()
    assert simple_view.is_view()

    # Verify data integrity - full view should match original
    full_entities_at_1_0 = full_view.at(1.0).num_entities
    assert full_entities_at_1_0 == total_nodes, "View should have same data as original"

    # Test copy functionality for creating independent collections
    independent_full = full_view.copy()
    independent_simple = simple_view.copy()

    assert not independent_full.is_view()
    assert not independent_simple.is_view()
    assert independent_full.at(1.0).num_entities == total_nodes

    logger.info(
        "EntityFrame workflow: Frame with %d collections. "
        "Views working correctly: full=%d entities, simple collection view exists",
        len(frame),
        full_view.at(1.0).num_entities,
    )

    # Test Expression API vs Direct Access (both APIs working together)
    # Demonstrate the distinction between collection.at() and sl.col().at()
    # Note: Operations on 1M records are optimised using the record-based algorithm

    # Direct partition access using collection.at() - gets actual entities
    # Use simple collection for performance testing
    partition_direct = frame["simple"].at(0.8)
    entities_direct = partition_direct.entities
    count_direct = len(entities_direct)

    # Expression API using ef.analyse() for metrics computation
    analysis_result = frame.analyse(
        sl.col("simple").at(0.8),
        metrics=[sl.Metrics.stats.entity_count, sl.Metrics.stats.entropy],
    )

    # Verify both approaches give consistent results
    assert len(analysis_result) == 1
    count_from_analysis = analysis_result[0]["entity_count"]
    assert abs(count_direct - count_from_analysis) < 0.01, (
        f"Direct access: {count_direct}, Analysis: {count_from_analysis}"
    )

    # Demonstrate expression API for threshold exploration (sweep)
    # Using simple collection for quick demonstration
    sweep_results = frame.analyse(
        sl.col("simple").sweep(0.7, 0.9, 0.1),
        metrics=[sl.Metrics.stats.entity_count],
    )

    # Should have results for 0.7, 0.8, 0.9
    assert len(sweep_results) == 3

    # Sort results by threshold since they may not be in order
    sweep_results_sorted = sorted(sweep_results, key=lambda r: r["simple_threshold"])
    sweep_thresholds = [r["simple_threshold"] for r in sweep_results_sorted]
    sweep_counts = [r["entity_count"] for r in sweep_results_sorted]

    expected_thresholds = [0.7, 0.8, 0.9]
    for actual, expected in zip(sweep_thresholds, expected_thresholds, strict=False):
        assert abs(actual - expected) < 0.01

    # Verify monotonic increase (higher thresholds should have more entities)
    assert sweep_counts[0] <= sweep_counts[1] <= sweep_counts[2]

    # Demonstrate large-scale comparison using expression API at 1M scale
    # With optimised record-based algorithm, we can handle 1M vs 1M comparisons

    # Create a second 1M collection for cross-collection comparison
    logger.info("Creating second 1M-edge collection for cross-collection comparison")
    start_time = time.monotonic()
    edges_2 = generators.edges(1_000_000)
    collection_2 = sl.Collection.from_edges(edges_2, show_progress=False)
    frame.add_collection("compare_1m", collection_2)
    setup_time = time.monotonic() - start_time
    logger.info("Second 1M collection created in %.2fs", setup_time)

    # Test 1M vs 1M comparison at threshold 0.8
    # At 0.8: each 1M collection has ~800k entities
    # This demonstrates the record-based O(r) algorithm handling production scale
    logger.info("Starting 1M vs 1M cross-collection comparison at threshold 0.8")
    start_time = time.monotonic()
    comparison_result = frame.analyse(
        sl.col("full_dataset").at(0.8),
        sl.col("compare_1m").at(0.8),
        metrics=[
            sl.Metrics.eval.f1,
            sl.Metrics.eval.precision,
            sl.Metrics.eval.recall,
        ],
    )
    comparison_time = time.monotonic() - start_time

    assert len(comparison_result) == 1
    result = comparison_result[0]

    # Check threshold values are preserved
    assert abs(result["full_dataset_threshold"] - 0.8) < 0.01
    assert abs(result["compare_1m_threshold"] - 0.8) < 0.01

    # Check comparison metrics are computed
    for metric in ["f1", "precision", "recall"]:
        assert metric in result
        metric_value = result[metric]
        assert isinstance(metric_value, int | float)
        assert 0.0 <= metric_value <= 1.0

    # Log performance results
    logger.info(
        "1M vs 1M comparison in %.2fs (F1=%.3f, Prec=%.3f, Rec=%.3f)",
        comparison_time,
        result["f1"],
        result["precision"],
        result["recall"],
    )

    # Performance assertion: record-based algorithm for single point comparison
    # Should complete within 15 seconds for 1M vs 1M (optimised O(r) algorithm)
    assert comparison_time < 15.0, (
        f"1M vs 1M comparison took {comparison_time:.2f}s, expected < 15s"
    )

    # Demonstrate sweep × point comparison at 1M scale
    logger.info("Starting 1M sweep × point comparison (3 thresholds)")
    start_time = time.monotonic()
    sweep_point_result = frame.analyse(
        sl.col("full_dataset").sweep(0.7, 0.9, 0.1),  # 3 thresholds
        sl.col("compare_1m").at(0.8),  # Single point
        metrics=[sl.Metrics.eval.f1],
    )
    sweep_point_time = time.monotonic() - start_time

    # Should produce 3 results (3 × 1)
    assert len(sweep_point_result) == 3
    for r in sweep_point_result:
        assert "full_dataset_threshold" in r
        assert "compare_1m_threshold" in r
        assert r["compare_1m_threshold"] == 0.8
        assert "f1" in r

    logger.info(
        "1M sweep × point comparison completed in %.2fs (3 comparisons)",
        sweep_point_time,
    )

    # Performance check - sweep should complete efficiently with record-based algorithm
    assert sweep_point_time < 30.0, (
        f"1M sweep × point took {sweep_point_time:.2f}s, expected < 30s"
    )

    # Demonstrate 1M-scale single collection sweep for entropy analysis
    logger.info("Starting 1M collection sweep for entropy analysis")
    start_time = time.monotonic()
    entropy_sweep_result = frame.analyse(
        sl.col("full_dataset").sweep(0.7, 0.95, 0.05),  # 6 thresholds
        metrics=[sl.Metrics.stats.entity_count, sl.Metrics.stats.entropy],
    )
    entropy_sweep_time = time.monotonic() - start_time

    assert len(entropy_sweep_result) == 6
    # Verify monotonic increase in entity count
    # Sort results by threshold since they may not be in order
    entropy_sweep_sorted = sorted(
        entropy_sweep_result, key=lambda r: r["full_dataset_threshold"]
    )
    entity_counts_1m = [r["entity_count"] for r in entropy_sweep_sorted]
    for i in range(len(entity_counts_1m) - 1):
        assert entity_counts_1m[i] <= entity_counts_1m[i + 1]

    logger.info(
        "1M entropy sweep completed in %.2fs (6 thresholds, entity counts: %s)",
        entropy_sweep_time,
        entity_counts_1m,
    )

    # Demonstrate sweep × sweep comparison at scale (limited for test time)
    logger.info("Starting sweep × sweep comparison (2×2 grid)")
    start_time = time.monotonic()
    sweep_sweep_result = frame.analyse(
        sl.col("full_dataset").sweep(0.8, 0.9, 0.1),  # 2 thresholds
        sl.col("compare_1m").sweep(0.8, 0.9, 0.1),  # 2 thresholds
        metrics=[sl.Metrics.eval.f1],
    )
    sweep_sweep_time = time.monotonic() - start_time

    # Should produce 4 results (2 × 2 grid)
    assert len(sweep_sweep_result) == 4
    for r in sweep_sweep_result:
        assert "full_dataset_threshold" in r
        assert "compare_1m_threshold" in r
        assert "f1" in r

    logger.info(
        "Sweep × sweep comparison completed in %.2fs (2×2 grid = 4 comparisons)",
        sweep_sweep_time,
    )

    # Performance check - sweep × sweep should complete efficiently with O(r) algorithm
    assert sweep_sweep_time < 45.0, (
        f"Sweep × sweep took {sweep_sweep_time:.2f}s, expected < 45s"
    )

    logger.info(
        "Expression API workflow: Direct access yielded %d entities at 0.8. "
        "Analysis API yielded %.0f entities. Sweep across [0.7, 0.8, 0.9] = %s. "
        "1M vs 1M comparison in %.2fs, sweep × point in %.2fs, sweep × sweep in %.2fs",
        count_direct,
        count_from_analysis,
        [int(c) for c in sweep_counts],
        comparison_time,
        sweep_point_time,
        sweep_sweep_time,
    )
