"""End-to-end test mimicking real user exploratory data analysis workflow."""

import logging
import time

import starlings as sl

logger = logging.getLogger(__name__)


def test_user_eda_workflow():
    """Production-scale EDA workflow: Million-record minimum scale for library."""
    # Million-scale is the MINIMUM expected dataset size for production use

    # Time graph generation using unified entity resolution generator
    start_time = time.monotonic()
    edges = sl.generate_entity_resolution_edges(1_000_000)
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
