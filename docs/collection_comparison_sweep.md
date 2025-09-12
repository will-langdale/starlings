# Collection comparison optimisation: from O(k₁ × k₂) to O(r)

## Executive summary

We've identified a breakthrough optimisation for cross-collection comparisons in Starlings that reduces algorithmic complexity from O(k₁ × k₂) to O(r). By leveraging the fact that collections in an EntityFrame share the same underlying record space, we can build contingency tables by iterating over records rather than comparing entity pairs. 

**Realistic performance improvement**: 1,000-10,000x speedup for large-scale comparisons, making million-record sweeps feasible (seconds instead of hours).

**Critical requirement**: Proper handling of memory compaction and cache invalidation to prevent data corruption.

## The problem: infeasible large-scale comparisons

### Current implementation

When comparing two collections at specific thresholds (e.g., collection A at 0.8 vs collection B at 0.8), we currently:

1. Get partition A at threshold 0.8 (containing k₁ entities)
2. Get partition B at threshold 0.8 (containing k₂ entities)
3. Compare every entity pair to build a contingency table

```rust
// Current implementation in expressions.rs
for (entity1_id, entity1) in partition1.entities().iter().enumerate() {
    for (entity2_id, entity2) in partition2.entities().iter().enumerate() {
        let overlap = entity1.intersection_len(entity2);
        if overlap > 0 {
            table.nonzero_cells.insert((entity1_id, entity2_id), overlap as u32);
        }
    }
}
```

### The scale challenge

For 1M records at threshold 0.8:
- Each collection has approximately 796,000 entities
- Total comparisons needed: 796,000 × 796,000 = **633 billion**
- Even with RoaringBitmap's optimised `intersection_len()` at 10 nanoseconds per call
- Total time: 633 billion × 10ns = 6,330 seconds ≈ **1.76 hours**

### Why current optimisations don't help enough

1. **RoaringBitmap's container-level optimisations**: While `intersection_len()` skips non-overlapping containers internally, we still make 633 billion function calls
2. **Parallel processing**: Even with 100 cores, we're looking at 63 seconds minimum
3. **Sampling**: Our current solution - reduces accuracy for approximate results

## Algorithm Selection Strategy

We have two distinct algorithms for computing metrics, each optimal for different scenarios:

### Delta-based Algorithm (Incremental Updates)
- **Complexity**: O(k) updates between adjacent thresholds
- **Best for**: Single collection operations and single-dimension changes
- **Use cases**:
  1. Single collection, single point: Direct partition reconstruction
  2. Single collection sweep: Incremental updates between thresholds
  3. Sweep × point comparison: One fixed, one sweeping collection

### Record-based Algorithm (Full Record Iteration)
- **Complexity**: O(r) where r = number of records
- **Best for**: Cartesian product comparisons (sweep × sweep)
- **Use case**: Both collections sweeping creates n₁ × n₂ comparison points

### Decision Tree
```
if both expressions are sweeps:
    use record-based  # O(r) beats O(k₁ × k₂ × n₁ × n₂)
else:
    use delta-based   # O(k) incremental updates sufficient
```

### Why This Matters

For a 5×5 sweep comparison (25 total comparisons):
- **Delta-based**: Must compare k₁ × k₂ entities at EACH of 25 points = O(k₁ × k₂ × 25)
- **Record-based**: Iterate r records ONCE for all 25 comparisons = O(r)

With 80k entities per collection:
- Delta-based: 80k × 80k × 25 = 160 billion operations
- Record-based: 1M records × 1 pass = 1 million operations
- **Speedup**: 160,000×

## The key insight

**Collections in an EntityFrame share the same underlying record space!**

When we have two collections in an EntityFrame:
- Both partition the exact same set of records
- Each record appears in exactly one entity in collection A
- The same record appears in exactly one entity in collection B
- We're essentially asking: "How do records that are grouped together in A align with their groupings in B?"

This is fundamentally different from comparing arbitrary clusterings - we're comparing different ways of partitioning the same space.

## Critical safety considerations

### Memory compaction invalidation (SEVERE RISK)

The `auto_compact()` function in Task 3.4 will completely reindex all records, invalidating all cached reverse indices. **This will cause silent data corruption if not handled properly.**

**Solution**: Context generation tracking

```rust
pub struct DataContext {
    generation: u64,  // Increment on ANY modification
    records: Vec<InternedRecord>,
    // ... other fields
}

impl DataContext {
    pub fn compact(&mut self) {
        // ... perform compaction ...
        self.generation += 1;  // Invalidate all caches
    }
}

pub struct PartitionLevel {
    threshold: f64,
    entities: Vec<RoaringBitmap>,
    entity_sizes: Vec<u32>,
    
    // Cache with generation tracking
    context_generation: u64,  // Generation when built
    record_to_entity: OnceCell<Vec<Option<usize>>>,  // NOT EntityId!
}

impl PartitionLevel {
    pub fn get_entity_for_record(&self, record_idx: usize, context: &DataContext) -> Option<usize> {
        // Check if cache is still valid
        if self.context_generation != context.generation {
            self.record_to_entity.take();  // Clear stale cache
            self.context_generation = context.generation;
        }
        
        let index = self.record_to_entity.get_or_init(|| {
            self.build_record_to_entity_index(context.records.len())
        });
        
        index.get(record_idx).copied().flatten()
    }
}
```

### Record indices vs record IDs

**Critical clarification**: We iterate over **vector indices** in `DataContext::records`, not abstract record IDs.

```rust
// CORRECT: Iterate over vector indices
for record_idx in 0..context.records.len() {
    // record_idx is the position in Vec<InternedRecord>
    let entity_a = partition_a.get_entity_for_record(record_idx, &context);
    let entity_b = partition_b.get_entity_for_record(record_idx, &context);
}

// INCORRECT: Don't assume continuous "record IDs"
for record_id in 0..num_records {  // What is num_records?
    // This assumes IDs are 0..n which may not hold after deletion
}
```

### Entity ID instability across thresholds

Entity IDs are **positional indices** in the entities vector - they change completely between thresholds!

```rust
// At threshold 0.8: entities = [bitmap_A, bitmap_B, bitmap_C]
// Record 42 is in bitmap_B, so entity_id = 1

// At threshold 0.7: entities = [bitmap_X, bitmap_Y]  (after merges)
// Record 42 is now in bitmap_X, so entity_id = 0

// Entity IDs are NOT stable across thresholds!
```

**Implications**:
- Use `Vec<Option<usize>>` not `Vec<Option<EntityId>>` for reverse indices
- Indices are threshold-specific and cannot be incrementally updated
- Each threshold needs its own complete reverse index

## The record-based algorithm (corrected)

```rust
use rayon::prelude::*;
use std::sync::Mutex;
use std::collections::HashMap;

pub fn from_partitions_via_records(
    partition1: &PartitionLevel,
    partition2: &PartitionLevel,
    context: &DataContext,
) -> SparseContingencyTable {
    let num_records = context.records.len();
    
    // Build reverse indices with proper generation tracking
    let record_to_entity1 = build_record_to_entity_index(partition1, num_records);
    let record_to_entity2 = build_record_to_entity_index(partition2, num_records);
    
    // Parallel collection of entity pairs
    let pairs: Vec<(usize, usize)> = (0..num_records)
        .into_par_iter()
        .filter_map(|record_idx| {
            match (record_to_entity1[record_idx], record_to_entity2[record_idx]) {
                (Some(e1), Some(e2)) => Some((e1, e2)),
                _ => None,
            }
        })
        .collect();
    
    // Aggregate into contingency table
    let mut table = SparseContingencyTable::new();
    for (entity1_idx, entity2_idx) in pairs {
        *table.nonzero_cells
            .entry((entity1_idx, entity2_idx))
            .or_insert(0) += 1;
    }
    
    // Compute marginals
    for (entity_idx, entity) in partition1.entities().iter().enumerate() {
        table.row_marginals.insert(entity_idx, entity.len() as u32);
    }
    for (entity_idx, entity) in partition2.entities().iter().enumerate() {
        table.col_marginals.insert(entity_idx, entity.len() as u32);
    }
    
    table.total_records = num_records as u32;
    table
}

fn build_record_to_entity_index(
    partition: &PartitionLevel,
    num_records: usize,
) -> Vec<Option<usize>> {
    let mut index = vec![None; num_records];
    
    for (entity_idx, entity_bitmap) in partition.entities().iter().enumerate() {
        for record_idx in entity_bitmap.iter() {
            // Bounds check for safety after compaction
            if (record_idx as usize) < num_records {
                index[record_idx as usize] = Some(entity_idx);
            }
        }
    }
    
    index
}
```

## Complexity analysis (revised)

### Current approach
- **Time**: O(k₁ × k₂) entity comparisons
- **Space**: O(s) for sparse contingency table (s = non-zero cells)
- For 1M records at threshold 0.8: ~633 billion operations

### Record-based approach
- **Time**: O(r) + O(r) + O(r) + O(k₁ + k₂) = O(3r + k₁ + k₂)
  - O(r) to build first reverse index
  - O(r) to build second reverse index  
  - O(r) to iterate records and build contingency table (parallelisable)
  - O(k₁ + k₂) to compute marginals
- **Space**: O(2r) + O(s)
  - O(2r) for two reverse indices
  - O(s) for sparse contingency table

### When is it faster?

The record-based approach is faster when:
- k₁ × k₂ > 3r (accounting for index build costs)
- With caching: k₁ × k₂ > r (indices pre-built)
- For 1M records with 800k entities each: 640B > 3M ✓ (clearly better)

### Realistic performance improvement

- **Theoretical**: 633B / 3M = 211,000x (comparing operation counts)
- **With parallel processing**: 10,000-100,000x (Rayon on multi-core)
- **Real-world with overhead**: 1,000-10,000x (system overhead, memory access)
- **Key achievement**: Million-record sweeps become feasible (seconds not hours)

## Memory analysis (corrected)

### Per-partition reverse index

```rust
// Vec<Option<usize>> on 64-bit systems
// Option<usize> = 16 bytes (8 byte payload + 8 byte discriminant/padding)
```

- 1M records × 16 bytes = **16MB per partition**
- 10 cached partitions × 2 collections = **320MB additional memory**
- With memory alignment and allocator overhead: **~400MB total**

### Memory optimisation strategies

```rust
// For sparse datasets (many isolates)
type SparseIndex = HashMap<u32, u32>;  // Only store non-None entries

// For dense datasets (most records in entities)
type DenseIndex = Vec<Option<u32>>;  // Use u32 if <4B entities

// Adaptive selection
fn choose_index_type(partition: &PartitionLevel) -> IndexType {
    let coverage = partition.total_records() as f64 / partition.max_record_id() as f64;
    if coverage < 0.5 {
        IndexType::Sparse
    } else {
        IndexType::Dense
    }
}
```

## Implementation plan (phased rollout)

### Phase 1: Core algorithm with safety (PRIORITY)

**Goal**: Correct, safe implementation with explicit opt-in

```rust
pub struct ComparisonConfig {
    use_record_algorithm: bool,  // Explicit opt-in
    validate_results: bool,      // Compare with O(k₁×k₂) for testing
}
```

**Tasks**:
1. Implement `from_partitions_via_records()` with bounds checking
2. Add context generation tracking
3. Implement cache invalidation on compaction
4. Create comprehensive test suite:
   - Identical results vs O(k₁×k₂) algorithm
   - Correct behaviour after compaction
   - Handle isolates (records not in edges)
   - Thread safety with parallel implementation

### Phase 2: Automatic optimisation

**Goal**: Seamless performance improvement

```rust
impl EntityFrame {
    fn can_use_record_algorithm(&self, col1: &str, col2: &str) -> bool {
        // Must share exact same Arc, not just equal contexts
        match (self.collections.get(col1), self.collections.get(col2)) {
            (Some(h1), Some(h2)) => Arc::ptr_eq(&h1.context, &h2.context),
            _ => false,
        }
    }
}
```

**Tasks**:
1. Auto-detect shared context using `Arc::ptr_eq`
2. Parallel processing by default with Rayon
3. Memory accounting and pressure monitoring
4. Performance benchmarks at various scales

### Phase 3: Advanced caching (defer if complex)

**Goal**: Further optimisation for sweeps

**Note**: Incremental updates are complex due to entity renumbering. May not be worth the complexity.

**Alternative approach**: Cache pool
```rust
struct IndexCache {
    // Cache up to N indices, LRU eviction
    indices: LruCache<(CollectionId, Threshold), RecordToEntityIndex>,
    max_memory_mb: usize,
}
```

## Testing requirements

### Correctness tests

```rust
#[test]
fn test_identical_results() {
    let edges = generate_test_edges(10_000);
    let ef = EntityFrame::from_edges(edges);
    
    let result_old = compute_via_entities(&ef, 0.8, 0.8);
    let result_new = compute_via_records(&ef, 0.8, 0.8);
    
    assert_eq!(result_old.nonzero_cells, result_new.nonzero_cells);
    assert_eq!(result_old.metrics(), result_new.metrics());
}

#[test]
fn test_compaction_safety() {
    let mut ef = EntityFrame::from_edges(edges);
    let partition = ef.get_partition("col_a", 0.8);
    
    // Cache the index
    let idx1 = partition.get_record_to_entity_index();
    
    // Compact changes everything
    ef.compact();
    
    // Index should be invalidated and rebuilt
    let idx2 = partition.get_record_to_entity_index();
    assert_ne!(idx1.generation, idx2.generation);
}

#[test]
fn test_with_isolates() {
    // Records that appear in records but not in edges
    let edges = vec![(0, 1, 0.9), (2, 3, 0.8)];
    let all_records = vec![0, 1, 2, 3, 4, 5];  // 4, 5 are isolates
    
    let ef = EntityFrame::from_edges_and_records(edges, all_records);
    // Verify isolates are handled correctly
}
```

### Performance benchmarks

```rust
#[bench]
fn bench_1m_comparison_old(b: &mut Bencher) {
    let ef = create_1m_test_frame();
    b.iter(|| compute_via_entities(&ef, 0.8, 0.8));
    // Expected: ~1000+ seconds
}

#[bench]
fn bench_1m_comparison_new(b: &mut Bencher) {
    let ef = create_1m_test_frame();
    b.iter(|| compute_via_records(&ef, 0.8, 0.8));
    // Expected: ~0.1-1.0 seconds
}
```

## Migration guide

### For existing code

```rust
// Before
let table = SparseContingencyTable::from_partitions(partition1, partition2);

// After - with automatic detection
let table = if let Some(context) = detect_shared_context(partition1, partition2) {
    SparseContingencyTable::from_partitions_via_records(partition1, partition2, context)
} else {
    SparseContingencyTable::from_partitions_via_entities(partition1, partition2)
};

// Or with explicit config
let table = SparseContingencyTable::from_partitions(
    partition1,
    partition2, 
    ComparisonConfig {
        algorithm: if shared_context { Algorithm::Record } else { Algorithm::Entity },
        parallel: true,
    }
);
```

## Risk mitigation

### Critical risks and mitigations

1. **Memory compaction corruption**
   - Mitigation: Generation tracking, automatic cache invalidation
   - Test: Extensive compaction tests

2. **Memory explosion**
   - Mitigation: Memory monitoring, fallback to entity algorithm if pressure
   - Test: Benchmark memory usage at various scales

3. **Incorrect results**
   - Mitigation: Validation mode that runs both algorithms
   - Test: Property-based testing with random datasets

4. **Thread safety issues**
   - Mitigation: Careful use of Rayon, immutable data structures
   - Test: Stress tests with high concurrency

## Conclusion

This optimisation transforms cross-collection comparison from an O(k₁ × k₂) problem to an O(r) problem, providing a realistic 1,000-10,000x speedup for large-scale comparisons. The key insight - that collections in an EntityFrame partition the same record space - allows us to build contingency tables by iterating records rather than comparing entity pairs.

**Critical requirements**:
- Proper handling of memory compaction with generation tracking
- Clear understanding that we iterate vector indices, not abstract IDs
- Recognition that entity IDs are positional and threshold-specific
- Comprehensive testing to ensure correctness

With careful implementation following this specification, million-record comparisons that currently take hours will complete in seconds, making large-scale entity resolution analysis finally feasible.