# Starlings design documents overview

## What is Starlings?

Starlings is a Python library for systematically exploring and comparing entity resolution results across different thresholds and methods. Instead of forcing threshold decisions at processing time, Starlings preserves the complete resolution space as a hierarchy of merge events, enabling instant threshold exploration, efficient metric computation, and lossless data transport between pipeline stages.

## The core insight: merge events, not partitions

Traditional entity resolution tools force you to choose a threshold and compute clusters at that point. To explore different thresholds, you recompute from scratch each time. Starlings takes a different approach:

```python
import starlings as sl

# Traditional: Fixed clusters at each threshold
clusters_at_85 = compute_clusters(edges, 0.85)  # Full computation
clusters_at_90 = compute_clusters(edges, 0.90)  # Full computation again

# Starlings: Hierarchy generates any threshold
collection = sl.Collection.from_edges(edges)
partition_at_85 = collection.at(0.85)  # O(m) first time, O(1) cached
partition_at_90 = collection.at(0.90)  # O(1) from cache
partition_at_8739 = collection.at(0.8739)  # Any threshold, instantly
```

By storing the merge events—the moments when entities combine as thresholds change—we can generate any partition on demand. This isn't possible with standard dataframe libraries because they lack this hierarchical concept.

## Key capabilities

### Analyse: efficient threshold exploration
- Query any threshold without recomputation
- Update metrics incrementally (O(k) where k = affected entities)
- Compare multiple resolution methods systematically
- Sweep thousands of thresholds in seconds, not hours

### Represent: complete resolution space
- Store infinite thresholds in ~60-115MB (vs ~10GB for explicit storage)
- Support n-way merges naturally (when 5 entities merge simultaneously)
- Preserve all information for downstream decisions
- Handle both probabilistic scores and fixed clusters uniformly

### Transport: seamless integration
- Arrow format with dictionary encoding for efficient serialisation
- Clean decomposition to relational database tables
- Native adapters for Splink, er-evaluation, Matchbox
- Preserve complete resolution history across pipeline stages

## API design

The API borrows from Polars' expression-based approach, separating data containers from operations:

```python
import starlings as sl
import polars as pl

# Load data and add resolution attempts
ef = sl.from_records("customers", df)
ef.add_collection_from_edges("splink", splink_edges)
ef.add_collection_from_edges("dedupe", dedupe_edges)
ef.add_collection_from_entities("truth", truth_entities)

# Analyse with composable expressions
results = ef.analyse(
    sl.col("splink").sweep(0.5, 0.95, 0.01),
    sl.col("truth").at(1.0).reference(),  # Explicit reference for asymmetric metrics
    metrics=[sl.Metrics.eval.f1, sl.Metrics.eval.precision, sl.Metrics.eval.recall]
)
# Returns tidy-row format: [{"collection": "splink", "collection_threshold": 0.5,
#                            "reference": "truth", "reference_threshold": 1.0,
#                            "metric_name": "f1", "metric_value": 0.72}, ...]

# Convert to polars for analysis
df = pl.from_dicts(results)
f1_results = df.filter(pl.col("metric_name") == "f1")
optimal = f1_results.filter(pl.col("metric_value") == pl.col("metric_value").max()).row(0, named=True)

# Direct access for simple operations
partition = ef["splink"].at(optimal["collection_threshold"])
entities = partition.to_list()  # List of sets of (source, key) pairs

# Apply operations to entities
hashes = partition.map(sl.Ops.hash.sha256)  # Fast parallel hashing
sizes = partition.map(sl.Ops.compute.size)   # Entity sizes
```

## Technical implementation

### Memory architecture: contextual ownership
Collections use contextual ownership—they reference their DataContext which contains the complete record space. This ensures isolated records (those not appearing in any edges) are properly included as singleton entities. When collections are views from a frame, they share the context efficiently through Arc reference counting. When standalone, they own their context exclusively. This trade-off prioritises memory efficiency in the common read-heavy case whilst maintaining safety through immutable views.

### Performance characteristics
- **Construction**: O(m log m) where m = number of edges
- **Threshold query**: O(m) first time, O(1) when cached
- **Metric updates**: O(k) incremental between adjacent thresholds
- **Memory usage**: ~60-115MB for 1M edges
- **Assumptions**: Sparse graphs from blocking/LSH (m ~ n to n log n, not n²)

### Language stack
- **Rust core**: For performance-critical merge event processing with parallel execution via Rayon
- **Python interface**: For familiar data science workflows via PyO3 with automatic type conversion
- **Arrow integration**: For cross-language data transport with optimal dictionary encoding at the global level

### Robust threshold handling
Starlings uses fixed-point representation internally (multiply by 10^6) whilst maintaining a float API. This eliminates floating-point comparison bugs through mandatory quantisation (1-6 decimal places, default 6), ensuring exact threshold comparisons and predictable behaviour.

## Injectable operations pattern

Starlings uses a consistent pattern for extensible operations that separates concerns cleanly:

```python
# Metrics for comparing partitions (used in analyse)
sl.Metrics.eval.f1       # Evaluation against ground truth
sl.Metrics.eval.ari      # Adjusted Rand Index
sl.Metrics.eval.nmi      # Normalised Mutual Information
sl.Metrics.stats.entropy # Single partition statistics
sl.Metrics.stats.entity_count

# Operations for entities (used in partition.map)
sl.Ops.hash.sha256      # Hash entities for metadata attachment
sl.Ops.hash.blake3      # Alternative fast hashing
sl.Ops.compute.size     # Compute entity properties
sl.Ops.compute.density  # Internal connectivity metrics
```

These are marker types that route to optimised Rust implementations, providing both flexibility and performance. The separation makes it clear: metrics operate on partitions, operations operate on individual entities.

## Document structure

### Document audience

These design documents are technical specifications for engineers and AI agents implementing Starlings. They are NOT user documentation. References to "users" throughout these documents mean developers using the Starlings library, not end users of applications built with Starlings. The documents prioritise technical precision and implementation clarity over explanation or justification.

### [Document 1: Mathematical principles](principles.md)
Establishes the mathematical foundations, user requirements, and theoretical guarantees. Defines the multi-collection model as F = (R, {H₁, H₂, ..., Hₙ}, I) where collections ARE hierarchies. Proves the efficiency of incremental computation through additive metric properties and introduces the contextual ownership architecture where hierarchies reference their DataContext to ensure complete record coverage including isolates.

### [Document 2: Computer science constructs](algorithms.md)
Details the algorithms and data structures including the union-find based merge event extraction, efficient partition reconstruction from merge events, and incremental metric computation strategies. Shows how the contextual ownership model enables natural handling of isolated records using concrete Rust code to demonstrate the designs clearly. Includes complexity analysis for all core operations.

### [Document 3: Rust engine](engine.md)
Covers the Rust implementation including parallel processing with Rayon, memory-efficient data structures using RoaringBitmaps, Python bindings via PyO3, and performance optimisations. Details the Key conversion strategy where Python Keys convert directly to u32 indices at the boundary for maximum performance during hierarchy operations. Focuses on SIMD optimisations, cache strategies, and production deployment considerations.

### [Document 4: Python interface](interface.md)
Complete API specification with the polars-inspired expression system (sl.col().at(), sl.col().sweep()), integration examples for Splink/er-evaluation/Matchbox, and performance benchmarks. Provides migration guides showing the 10-100x speedup for threshold analysis, practical usage patterns including hierarchical resolution workflows, and demonstrates working with List[Dict] outputs using polars for analysis.

### [Document 5: Roadmap](roadmap.md)
A living roadmap for engineers and coding agents to implement Starlings from scratch, with explicit file paths, dependencies, and test requirements for each task. Organised into four milestones that each deliver working software, this document provides concrete implementation checkpoints that can be ticked off as work progresses, with continuous integration from the first Python MVP onwards.

## Why this matters

Entity resolution practitioners currently work with incomplete information, forced to make threshold decisions before understanding their impact. Consider a typical workflow:

1. **Today's reality**: Run Splink, get edges, pick threshold 0.85 because "it worked last time", generate clusters, evaluate against truth, realise 0.85 was suboptimal, re-run with 0.90, repeat.

2. **With Starlings**: Run Splink, get edges, load into Starlings, sweep 1000 thresholds in seconds, visualise F1/precision/recall curves, identify optimal threshold with data-driven confidence, export entities.

The difference isn't just speed—it's about making informed decisions. When you can see how metrics evolve across the entire threshold space, you understand not just what the optimal threshold is, but why it's optimal and how sensitive your results are to that choice.

Starlings also enables workflows that were previously impractical:
- **Ensemble resolution**: Compare multiple algorithms at their respective optimal thresholds
- **Stability analysis**: Find threshold ranges where small changes don't dramatically alter results  
- **Progressive refinement**: Start with rough blocking, analyse, refine, without losing previous work
- **Collaborative threshold selection**: Share complete resolution spaces, let downstream users choose thresholds
- **Hierarchical resolution**: Resolve within sources first, then link resolved entities across sources

This isn't about revolutionising the field—it's about providing a better tool for a specific problem. If you need to systematically explore thresholds, compare resolution methods, or preserve resolution information across pipeline stages, Starlings offers a purpose-built solution that traditional dataframe libraries cannot provide.
