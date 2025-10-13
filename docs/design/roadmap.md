# Starlings engineering build plan

**Audience**: Engineers and coding agents  
**Instruction**: Each task specifies exact files to create and what to implement. Refer to design docs for specifications.

## Milestone 1: Minimal Viable Collection

**Status**: ✅ COMPLETED (100% complete)

- All tasks (1.1-1.4) implemented with professional two-crate architecture
- Core types, hierarchy construction, and partition reconstruction fully working
- PyO3 Python bindings complete with comprehensive test coverage
- Benchmarks confirm O(1) cached partition access achieving ~7ns lookups

### Task 1.1: Core Types & Data Structures

**Create files**: ✅ COMPLETED

- ✅ `src/rust/starlings/src/core/key.rs`
- ✅ `src/rust/starlings/src/core/record.rs`
- ✅ `src/rust/starlings/src/core/data_context.rs`
- ✅ `src/rust/starlings/src/core/mod.rs`

**Implement**: ✅ COMPLETED

- ✅ Key enum with U32, U64, String, Bytes variants
- ✅ InternedRecord struct with source_id, key, attributes
- ✅ DataContext with records Vec, source_interner, identity_map, source_index
- ✅ DataContext::ensure_record() method for deduplication
- ✅ Unit tests: key equality, record deduplication, index stability
- ✅ Benchmark: 10k unique record insertions using criterion

**Dependencies**: ✅ COMPLETED - roaring, fxhash, string-interner  
**Reference**: `algorithms.md` - Core data structure architecture

### Task 1.2: Hierarchy Construction

**Create files**: ✅ COMPLETED

- ✅ `src/rust/starlings/src/hierarchy/mod.rs`
- ✅ `src/rust/starlings/src/hierarchy/builder.rs`
- ✅ `src/rust/starlings/src/hierarchy/merge_event.rs`

**Implement**: ✅ COMPLETED

- ✅ MergeEvent struct with threshold and RoaringBitmap merging_groups
- ✅ PartitionHierarchy struct with merges Vec, partition_cache LruCache, context Arc<DataContext>
- ✅ PartitionHierarchy::from_edges() using union-find (disjoint_sets crate)
- ✅ Quantisation enforcement (1-6 decimal places)
- ✅ Fixed-point threshold_to_key() conversion (multiply by 1_000_000)
- ✅ Tests: 3-node graph, disconnected components, same-threshold edges (n-way merge)
- ✅ Benchmark: from_edges with 1k and 10k edges

**Dependencies**: ✅ COMPLETED - disjoint-sets, lru  
**Reference**: `algorithms.md` - Connected components algorithm

### Task 1.3: Partition Reconstruction

**Create files**: ✅ COMPLETED

- ✅ `src/rust/starlings/src/hierarchy/partition.rs`

**Implement**: ✅ COMPLETED

- ✅ PartitionLevel struct with entities as Vec<RoaringBitmap>
- ✅ PartitionHierarchy::at_threshold() with cache check
- ✅ PartitionHierarchy::reconstruct_at_threshold() internal method
- ✅ Include all records from DataContext (handles isolates automatically)
- ✅ Tests: threshold 0.0 (one entity), 1.0 (all singletons), 0.5 (intermediate)
- ✅ Test: verify isolates appear as singletons
- ✅ Benchmark: cached vs uncached at_threshold() calls
  - Uncached: 25µs (100 edges), 168µs (1k edges), 1ms (10k edges)
  - Cached: ~7ns regardless of size (1600x speedup!)
  - Threshold sweep: 20 points across 5k edges in ~900µs

**Reference**: `algorithms.md` - Partition reconstruction from merge events

### Task 1.4: Python MVP

**Create files**: ✅ COMPLETED

- ✅ `src/rust/starlings-py/Cargo.toml` (Professional two-crate architecture)
- ✅ `src/rust/starlings-py/src/lib.rs` (PyO3 wrapper over starlings-core)
- ✅ `src/python/starlings/__init__.py` (Python package structure)
- ✅ `src/tests/test_collection_basic.py` (Comprehensive Python tests)

**Implement**: ✅ COMPLETED

- ✅ PyO3 project setup with maturin using two-crate architecture
- ✅ PyCollection struct wrapping PartitionHierarchy from starlings-core
- ✅ PyCollection::from_edges() classmethod with comprehensive error handling
- ✅ PyCollection::at() method returning PyPartition with cached performance
- ✅ Key conversion: Python int/str/bytes → Rust Key → u32 index (robust type handling)
- ✅ Python tests: 17 comprehensive tests covering edge cases, type conversions, error handling
- ✅ Two-layer testing strategy: 33 Rust core tests + 17 Python integration tests
- ✅ **Architecture follows Polars pattern - zero PyO3 dependencies in core crate**

**Dependencies**: ✅ COMPLETED - pyo3, maturin with clean separation  
**Reference**: `engine.md` - Python interface via PyO3, `contributing.md` - Architecture decisions

## Milestone 2: Multi-Collection Frame & Analysis

**Status**: ✅ COMPLETED (100% complete)

### Task 2.1: EntityFrame

**Status**: ✅ COMPLETED (9 of 9 items complete) - PR #27, commit 69872a1

**Create files**:

- ✅ `rust/starlings-core/src/frame/mod.rs`
- ✅ `rust/starlings-core/src/frame/translation.rs` (created instead of collection_map.rs)

**Implement**:

- ✅ EntityFrame struct with context Arc<DataContext>, collections HashMap
- ✅ EntityFrame::add_collection() with Arc::ptr_eq check for same context
- ✅ assimilate() method for different contexts with TranslationMap
- ✅ Collection view pattern with is_view flag (implemented in PyO3 layer)
- ✅ Collection::copy() creating deep copy with new DataContext (via PartitionHierarchy::clone())
- ✅ PyEntityFrame with __getitem__ for ef["name"] syntax
- ✅ Test: multiple collections share memory
- ✅ Test: view immutability (comprehensive tests in test_entity_frame.py)
- ✅ Update E2E test: Extended `test_user_eda_workflow()` to use EntityFrame with multiple collections, test memory sharing

**Key Implementation Notes**:
- Multi-collection container with memory sharing via Arc<DataContext>
- Added Clone implementation to PartitionHierarchy with clone_box trait method for HierarchyStorage
- Enhanced PyCollection with is_view field and copy() method
- PyEntityFrame now supports dictionary-style access: ef["collection_name"] returns view collections
- View collections are immutable references; use copy() to create independent collections
- All tests pass including comprehensive view semantics and E2E workflow validation
- Test file: `src/tests/test_entity_frame.py`

**Reference**: `algorithms.md` - Adding collections to frames section

### Task 2.2: Expression API

**Status**: ✅ COMPLETED (PR #28, commit 5d65596)

Fully implemented with:
- `sl.col()` expression builder
- `.at(threshold)` for point queries
- `.sweep(start, stop, step)` for threshold ranges
- `.reference()` for explicit ground truth marking
- `EntityFrame.analyse()` integration

**Test coverage**: `src/tests/test_expressions.py`
**Implementation**: `src/rust/starlings-core/src/expressions/*`

**Create files**:

- ✅ `python/starlings/expressions.py`
- ✅ `rust/starlings-core/src/expressions/mod.rs`
- ✅ `rust/starlings-core/src/expressions/builder.rs`
- ✅ `rust/starlings-core/src/expressions/evaluator.rs`

**Implement**:

- ✅ Python col() function returning ColExpression
- ✅ ColExpression.at() and .sweep() methods
- ✅ ColExpression.reference() method for marking ground truth
- ✅ Rust expression parsing distinguishing point vs sweep
- ✅ Determine reference collection (explicit via .reference() or implicit as last expression)
- ✅ EntityFrame.analyse() taking variable expressions
- ✅ Return type always List[Dict[str, Any]] with universal tidy-row schema:
  - "collection": str (primary collection name)
  - "collection_threshold": float
  - "reference": Optional[str] (None for single-collection)
  - "reference_threshold": Optional[float] (None for single-collection)
  - "metric_name": str
  - "metric_value": float
- ✅ Test: sl.col("a").at(0.8), sl.col("b").at(1.0).reference() with explicit reference
- ✅ Test: sl.col("a").sweep(0.5, 0.9, 0.1) single-collection output format
- ✅ Test: implicit reference behaviour (last expression becomes reference)
- ✅ Update E2E test: Replace manual `.at()` calls with expression API, add sweep testing with new output format

**Reference**: `interface.md` - Expression API section (updated with .reference() method and tidy-row format)

### Task 2.3: Core Metrics

**Status**: ✅ COMPLETED

Implemented with dual-algorithm metric engine:
- **Delta Algorithm**: O(k) incremental updates for same-collection sweeps
- **Record Algorithm**: O(r) single-pass for cross-collection comparisons

**Metrics implemented**:
- Pairwise: F1, Precision, Recall
- Cluster: ARI (Adjusted Rand Index)
- Statistics: Entity Count, Entropy

**Implementation files**:
- `src/rust/starlings-core/src/metrics/mod.rs` - Main engine
- `src/rust/starlings-core/src/metrics/algorithms/delta.rs` - Delta algorithm
- `src/rust/starlings-core/src/metrics/algorithms/record.rs` - Record algorithm
- `src/rust/starlings-core/src/metrics/contingency/mod.rs` - Contingency tables
- `src/rust/starlings-core/src/metrics/implementations/*` - Metric implementations

**Test coverage**: Tests in `src/rust/starlings-core/src/metrics/mod.rs`

**Create files**:

- ✅ `rust/starlings-core/src/metrics/mod.rs`
- ✅ `rust/starlings-core/src/metrics/algorithms/delta.rs`
- ✅ `rust/starlings-core/src/metrics/algorithms/record.rs`
- ✅ `rust/starlings-core/src/metrics/contingency/mod.rs`
- ✅ `rust/starlings-core/src/metrics/implementations/*`

**Implement**:

- ✅ compute_precision(), compute_recall(), compute_f1()
- ✅ Contingency table construction for two partitions
- ✅ Python sl.Metrics.eval.f1 etc. as marker classes
- ✅ Metric computation in analyse() method
- ✅ Test: known partitions with expected precision/recall
- ✅ Benchmark: metric computation for 1k entities
- ✅ Update E2E test: Add precision/recall/F1 computation against ground truth in workflow

**Reference**: `principles.md` - Pairwise classification metrics

## Features Implemented Beyond Original Roadmap

The following features were implemented beyond the original plan:

### Binary Delta MergeEvents (PR #31, commit 85bbf4b)
- **Innovation**: Changed MergeEvent from n-way full-state to binary delta representation
- **Impact**: Reduced memory from O(N²) to O(N) for merge event storage
- **Structure**: `parent_id: u32` + `child_nodes: RoaringBitmap` instead of `Vec<RoaringBitmap>`
- **File**: `src/rust/starlings-core/src/hierarchy/merge_event.rs`

### Disk Spilling Mechanisms (PR #29, commit 61f73d3)
- **Purpose**: Robust memory management for large-scale datasets
- **Features**:
  - Automatic spillover to disk when memory pressure detected
  - Spillable trait for operations >100MB
  - LRU cache management with eviction
- **Files**:
  - `src/rust/starlings-core/src/core/spilling.rs`
  - `src/rust/starlings-core/src/core/safety.rs`
  - `src/rust/starlings-core/src/core/resource_monitor.rs`
- **Test coverage**: `src/tests/test_safety.py`

### Advanced Performance Features
- **Incremental Builder**: `src/rust/starlings-core/src/hierarchy/incremental.rs`
- **Memory Cache**: `src/rust/starlings-core/src/hierarchy/memory_cache.rs`
- **Bitmap Pooling**: `src/rust/starlings-core/src/hierarchy/bitmap_pool.rs`
- **Chain Detection**: Tests in `src/tests/test_chain_detection.py`

## Milestone 3: Performance & Production

**Status**: 🚧 IN PROGRESS (~63% complete - 2.5 of 4 tasks complete)

### Task 3.1: Incremental Metrics

**Status**: ✅ COMPLETED

Delta algorithm provides O(k) incremental updates for same-collection sweeps. The metric engine automatically selects the appropriate algorithm (Delta or Record) based on query type.

**Implementation**: `src/rust/starlings-core/src/metrics/algorithms/delta.rs`

**Create files**:

- ✅ `rust/starlings-core/src/metrics/algorithms/delta.rs` (Delta algorithm state management)
- ✅ `rust/starlings-core/src/metrics/algorithms/record.rs` (Record-based computation)

**Implement**:

- ✅ IncrementalMetricState struct with last_threshold, contingency_table
- ✅ compute_delta() for metrics between adjacent thresholds
- ✅ O(k) update logic where k = affected entities
- ✅ Test: incremental result equals full recomputation
- ✅ Benchmark: 1000-threshold sweep time reduction
- ✅ Update E2E test: Add timing comparison between incremental vs full recomputation metrics

**Reference**: `principles.md` - Incremental metric computation

### Task 3.2: Parallelisation

**Status**: 🔄 PARTIALLY COMPLETE (~50% complete)

**Modify files**: Throughout hierarchy and metrics modules

**Implement**:

- [x] Rayon dependency added to starlings-core
- [x] `par_sort_unstable_by` implemented for edge sorting (`src/rust/starlings-core/src/test_utils.rs:71`)
- [x] `par_iter` used in record algorithm for independent operations (`src/rust/starlings-core/src/metrics/algorithms/record.rs:125`)
- [ ] Test: results identical with/without parallelisation (needs systematic testing)
- [ ] Benchmark: measure speedup on 1, 2, 4, 8 threads (needs formal benchmarks)

**Note**: Basic parallelisation is implemented and working, but systematic testing and benchmarking across all operations is incomplete.

**Dependencies**: rayon ✅ (already included)
**Reference**: `engine.md` - Parallel processing architecture

### Task 3.3: Arrow Serialisation

**Status**: ❌ NOT STARTED (0% complete)

**Create files**:

- `rust/starlings-core/src/io/mod.rs`
- `rust/starlings-core/src/io/arrow.rs`

**Implement**:

- [ ] Arrow schema with dictionary encoding for sources/keys
- [ ] EntityFrame::to_arrow() method
- [ ] EntityFrame::from_arrow() method
- [ ] RoaringBitmap serialisation as nested lists
- [ ] Test: round-trip preservation of all data
- [ ] Benchmark: serialisation of 100k records
- [ ] Update E2E test: Extend workflow to include EntityFrame.to_arrow() and round-trip testing

**Dependencies**: arrow  
**Reference**: `engine.md` - Arrow integration

### Task 3.4: Memory Management

**Status**: ✅ COMPLETED (PR #29, commit 61f73d3)

Disk spilling and safety mechanisms implemented, including:
- Automatic spillover to disk when memory pressure detected
- Spillable trait for operations >100MB
- LRU cache management with eviction
- Memory monitoring and resource management

**Implementation files**:
- `src/rust/starlings-core/src/core/spilling.rs` - Spilling mechanism
- `src/rust/starlings-core/src/core/safety.rs` - Safety checks
- `src/rust/starlings-core/src/core/resource_monitor.rs` - Resource monitoring

**Test coverage**: `src/tests/test_safety.py`

**Create files**:

- ✅ `rust/starlings-core/src/core/spilling.rs`
- ✅ `rust/starlings-core/src/core/safety.rs`
- ✅ `rust/starlings-core/src/core/resource_monitor.rs`
- ✅ `rust/starlings-core/src/hierarchy/storage/disk_storage.rs` (disk-backed storage)

**Implement**:

- ✅ Spillable trait for large operations
- ✅ Automatic spill_to_disk() when memory pressure detected
- ✅ LRU cache management with eviction
- ✅ Resource monitoring and safety checks
- ✅ Test: large operations handle memory pressure gracefully
- ✅ Test: data integrity after spilling/restoration

**Note**: Original plan included automatic compaction on EntityFrame::drop(). This may be added in future if needed for specific use cases.

**Reference**: `algorithms.md` - Automatic memory management

## Milestone 4: Complete Features

**Status**: 🚧 IN PROGRESS (25% complete - 1 of 4 tasks complete)

### Task 4.1: Advanced Metrics

**Status**: ✅ COMPLETED

**Created files**:

- ✅ `src/rust/starlings-core/src/metrics/contingency/mod.rs` - ContingencyMetrics trait with all advanced metrics

**Implement**:

- ✅ ARI (Adjusted Rand Index) - Fully implemented in both DeltaState and RecordState
- ✅ NMI (Normalised Mutual Information) - Fully implemented with entropy calculations
- ✅ V-Measure (harmonic mean of homogeneity and completeness) - Fully implemented
- ✅ BCubed Precision - Fully implemented
- ✅ BCubed Recall - Fully implemented
- ✅ Tests: Comprehensive test coverage in `src/rust/starlings-core/src/metrics/mod.rs:515-620`
- ✅ ContingencyMetrics trait provides unified interface for both Delta and Record algorithms

**Implementation notes**:
- All metrics implemented via the `ContingencyMetrics` trait in `src/rust/starlings-core/src/metrics/contingency/mod.rs:19-43`
- Both `DeltaState` and `RecordState` implement the trait for optimal performance with their respective data structures
- Includes helper functions for entropy calculations used by NMI and V-Measure
- Full integration with the metric engine's automatic algorithm selection

**Test coverage**:
- ARI tests with identical and different partitions (`test_ari_computation`, `test_ari_with_different_partitions`)
- NMI tests in `src/rust/starlings-core/src/metrics/algorithms/record.rs`
- All metrics validated against expected mathematical properties

**Reference**: `principles.md` - Complete mathematical operation space

### Task 4.2: Operations

**Status**: ❌ NOT STARTED (0% complete)

**Create files**:

- `rust/starlings-core/src/operations/mod.rs`
- `rust/starlings-core/src/operations/hash.rs`
- `rust/starlings-core/src/operations/compute.rs`

**Implement**:

- [ ] SHA256, Blake3, MD5 hash operations
- [ ] Size, density compute operations
- [ ] Partition::map() method with EntityProcessor enum
- [ ] Python sl.Ops.hash.sha256 etc. marker classes
- [ ] Test: hash consistency
- [ ] Benchmark: parallel map execution

**Dependencies**: sha2, blake3, md-5  
**Reference**: `algorithms.md` - Dual processing for entity operations

### Task 4.3: Hierarchical Resolution

**Status**: ❌ NOT STARTED (0% complete)

**Modify files**:
- `rust/starlings-core/src/hierarchy/builder.rs`

**Implement**:

- [ ] Accept Entity objects in records parameter of from_edges()
- [ ] entities_to_edges_internal() expanding entities to edges
- [ ] All pairs within entity get weight 1.0
- [ ] Collection::from_entities() constructor
- [ ] Test: two-stage resolution workflow
- [ ] Update E2E test: Add Collection.from_entities() workflow testing alongside from_edges()

**Reference**: `interface.md` - Hierarchical resolution workflow

### Task 4.4: Polish

**Status**: ❌ NOT STARTED (0% complete)

**Create files**:

- `python/starlings/_starlings.pyi`

**Implement**:

- [ ] Complete type stubs for all Python-visible classes
- [ ] Property tests with proptest (Rust) and hypothesis (Python)
- [ ] Memory leak detection with valgrind in CI
- [ ] Integration examples for Splink, er-evaluation
- [ ] Performance regression suite
- [ ] **Version bump and release notes for PyPI (existing CI/CD will handle publishing)**

**Reference**: `interface.md` - Integration sections

## Critical Constants

```rust
const CACHE_SIZE: usize = 10;  // LRU cache per hierarchy
const PRECISION_FACTOR: f64 = 1_000_000.0;  // Fixed-point conversion
const COMPACTION_THRESHOLD: f64 = 0.5;  // Garbage ratio trigger
const MAX_QUANTISE: u32 = 6;  // Maximum decimal places
```
