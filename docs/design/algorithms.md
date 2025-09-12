# Starlings computer science constructs

This document details the algorithms, data structures, and computational techniques that implement Starlings' mathematical model. It provides concrete descriptions using Rust code to show exactly how the core algorithms work, with complexity analysis and design rationale.

## Core data structure architecture

### Multi-collection EntityFrame with contextual ownership

```rust
pub struct EntityFrame {
    // Shared data context (append-only arena)
    context: Arc<DataContext>,
    
    // Multiple hierarchies (collections) over shared record space
    collections: HashMap<CollectionId, PartitionHierarchy>,
    
    // Cross-collection comparison engine
    comparison_engine: CollectionComparer,
    
    // Track garbage for automatic compaction
    garbage_records: RoaringBitmap,
}

pub struct DataContext {
    // Append-only record storage
    records: Vec<InternedRecord>,
    
    // String interning for sources
    source_interner: StringInterner<FxHashMap>,
    
    // Canonical identity mapping
    identity_map: HashMap<(SourceId, Key), RecordIndex>,
    
    // Index for fast lookup
    source_index: HashMap<SourceId, RoaringBitmap>,
}

impl DataContext {
    /// Ensure a record exists in the context (deduplication)
    /// O(1) lookup and optional insertion
    /// Note: DataContext is append-only. Adding records after a hierarchy is built
    /// creates isolates (singleton entities) in any hierarchies that reference it.
    pub fn ensure_record(&mut self, source: &str, key: Key) -> RecordIndex {
        let source_id = self.source_interner.get_or_intern(source);
        let identity = (source_id, key.clone());
        
        if let Some(&idx) = self.identity_map.get(&identity) {
            idx
        } else {
            let idx = self.records.len() as RecordIndex;
            self.records.push(InternedRecord {
                source_id,
                key: key.clone(),
                attributes: None,
            });
            self.identity_map.insert(identity, idx);
            self.source_index.entry(source_id).or_default().insert(idx);
            idx
        }
    }
    
    /// Deep copy the context for collection copying
    pub fn deep_copy(&self) -> Self {
        DataContext {
            records: self.records.clone(),
            source_interner: self.source_interner.clone(),
            identity_map: self.identity_map.clone(),
            source_index: self.source_index.clone(),
        }
    }
}

pub struct InternedRecord {
    source_id: u16,     // Interned source identifier
    key: Key,           // Key within that source (flexible type)
    attributes: Option<RecordData>,  // Optional attribute data
}

/// Flexible key types for record identification
pub enum Key {
    U32(u32),
    U64(u64),
    String(String),  // Keys don't need interning - they're unique by definition
    Bytes(Vec<u8>),
}

/// Record specification for standalone collection construction
pub enum RecordSpec {
    Keys(Vec<Key>),           // Additional records (isolates)
    Entities(Vec<EntityData>), // Pre-grouped entities for hierarchical resolution
}

/// Entity data for hierarchical resolution
pub struct EntityData {
    id: u32,  // Temporary ID for edge references
    members: Vec<(SourceId, Key)>,
}
```

### Contextual ownership architecture

**Separation of data and logic with immutable views**

```rust
pub struct Collection {
    // Reference to data (may be shared or owned)
    context: Arc<DataContext>,
    
    // Logical structure (hierarchy of merge events)
    hierarchy: PartitionHierarchy,
    
    // Track if this is a view from a frame
    is_view: bool,
}

impl Collection {
    /// Create standalone collection with owned context
    pub fn from_edges(edges: Vec<(Key, Key, f64)>, 
                     records: Option<RecordSpec>) -> Self {
        let mut context = DataContext::new();
        
        // Add all records mentioned in edges
        for (key1, key2, _) in &edges {
            context.ensure_record("default", key1.clone());
            context.ensure_record("default", key2.clone());
        }
        
        // Add any additional records (isolates or all records - we dedupe)
        if let Some(records) = records {
            match records {
                RecordSpec::Keys(keys) => {
                    for key in keys {
                        context.ensure_record("default", key);  // Deduplicates automatically
                    }
                },
                RecordSpec::Entities(entities) => {
                    // Add all entity members to context
                    for entity in &entities {
                        for (source_id, key) in &entity.members {
                            context.ensure_record_with_source(*source_id, key.clone());
                        }
                    }
                    // Then convert entities to edges for hierarchy building
                    let entity_edges = Self::entities_to_edges_internal(&entities);
                    edges.extend(entity_edges);
                }
            }
        }
        
        let context_arc = Arc::new(context);
        
        // Convert edges to u32 indices for hierarchy construction
        let indexed_edges = edges_to_indices(&edges, &context_arc);
        let hierarchy = PartitionHierarchy::from_edges(indexed_edges, context_arc.clone(), 6);
        
        Collection {
            context: context_arc,
            hierarchy,
            is_view: false,
        }
    }
    
    /// Create standalone collection from pre-resolved entities
    pub fn from_entities(entities: Vec<EntitySet>) -> Self {
        let mut context = DataContext::new();
        
        // Add all records from entities to context
        for entity_set in &entities {
            for (source, key) in entity_set {
                context.ensure_record(source, key.clone());
            }
        }
        
        let context_arc = Arc::new(context);
        
        // Convert entities to edges at threshold 1.0 in Rust
        let edges = Self::entities_to_edges_internal(&entities);
        let hierarchy = PartitionHierarchy::from_edges(edges, context_arc.clone(), 6);
        
        Collection {
            context: context_arc,
            hierarchy,
            is_view: false,
        }
    }
    
    /// Convert entities to edges for hierarchy building (Rust implementation for performance)
    /// O(n²) within each entity, but entities are typically small
    /// All pairs within an entity get weight 1.0
    fn entities_to_edges_internal(entities: &[Vec<(SourceId, Key)>]) -> Vec<(u32, u32, f64)> {
        use rayon::prelude::*;
        
        entities.par_iter()
            .flat_map(|entity| {
                let members: Vec<_> = entity.iter().collect();
                let mut edges = Vec::with_capacity(members.len() * (members.len() - 1) / 2);
                
                for i in 0..members.len() {
                    for j in i+1..members.len() {
                        edges.push((
                            members[i].to_index(),
                            members[j].to_index(), 
                            1.0
                        ));
                    }
                }
                edges
            })
            .collect()
    }
    
    /// Create an explicit owned copy of this collection
    pub fn copy(&self) -> Self {
        let new_context = Arc::new(self.context.deep_copy());
        let mut new_hierarchy = self.hierarchy.clone();
        new_hierarchy.context = new_context.clone();  // Update to new context
        
        Collection {
            context: new_context,
            hierarchy: new_hierarchy,
            is_view: false,
        }
    }
    
    /// Check if this is an immutable view from a frame
    pub fn is_view(&self) -> bool {
        self.is_view
    }
}
```

## Efficient hierarchical data structures

### Merge-event based hierarchy with smart caching and fixed-point thresholds

```rust
pub struct PartitionHierarchy {
    // Reference to the data context that gives meaning to record indices
    context: Arc<DataContext>,
    
    // Primary: Merge events sorted by threshold (descending)
    merges: Vec<MergeEvent>,
    
    // Secondary: Cache for frequently accessed partitions
    // Using u32 keys (threshold * 1_000_000) for exact comparison
    partition_cache: LruCache<u32, PartitionLevel>,
    
    // Tertiary: State for incremental metric computation
    metric_state: Option<IncrementalMetricState>,
    
    // Index for binary search on thresholds (using integer keys)
    threshold_index: BTreeMap<u32, usize>,
    
    // Configuration
    cache_size: usize,  // Non-configurable size 10 per hierarchy for v1
}

impl PartitionHierarchy {
    const PRECISION_FACTOR: f64 = 1_000_000.0;  // Supports quantise up to 6 decimal places
    
    fn threshold_to_key(threshold: f64) -> u32 {
        (threshold.clamp(0.0, 1.0) * Self::PRECISION_FACTOR).round() as u32
    }
    
    fn key_to_threshold(key: u32) -> f64 {
        key as f64 / Self::PRECISION_FACTOR
    }
}

pub struct MergeEvent {
    threshold: f64,
    
    // Groups merging at this threshold (supports n-way)
    // Each RoaringBitmap contains the record indices in that group
    merging_groups: Vec<RoaringBitmap>,
}

pub struct PartitionLevel {
    threshold: f64,
    
    // Entities as sets of record indices (RoaringBitmaps)
    entities: Vec<RoaringBitmap>,
    
    // Pre-computed cheap statistics
    entity_sizes: Vec<u32>,
    
    // Lazy expensive metrics
    lazy_metrics: LazyMetricsCache,
}

pub struct IncrementalMetricState {
    last_threshold: f64,
    last_partition: PartitionLevel,
    contingency_table: SparseContingencyTable,
    affected_entities: RoaringBitmap,
}
```

**Memory analysis**

For 1M records with 1M edges in the Contextual Ownership model:
- DataContext: ~10MB for records + interning
- Merge events: 1M × 50-100 bytes = ~50-100MB (includes RoaringBitmaps)
- LRU cache (10 partitions): 10 × 500KB = ~5MB
- Total per collection: ~60-115MB
- Shared context when in frame: No duplication
- Total for 5 collections in frame: ~300-500MB (not 5× due to sharing)

## Connected components algorithm for hierarchy construction

### Union-find based merge event extraction with mandatory quantisation

```rust
use disjoint_sets::UnionFind;

impl PartitionHierarchy {
    pub fn from_edges(edges: Vec<(u32, u32, f64)>, 
                     context: Arc<DataContext>,
                     quantise: u32) -> Self {
        // Validate quantise is 1-6
        assert!(quantise >= 1 && quantise <= 6, "quantise must be between 1 and 6");
        
        // Apply quantisation
        let factor = 10_f64.powi(quantise as i32);
        let mut sorted_edges: Vec<_> = edges.into_iter()
            .map(|(i, j, w)| (i, j, (w * factor).round() / factor))
            .collect();
        
        // Sort edges by weight (descending) - O(m log m)
        sorted_edges.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
        
        // Group edges by threshold (handles quantisation naturally)
        let mut threshold_groups: Vec<(f64, Vec<(u32, u32)>)> = Vec::new();
        let mut current_threshold = sorted_edges[0].2;
        let mut current_group = Vec::new();
        
        for (src, dst, weight) in sorted_edges {
            if weight == current_threshold {
                current_group.push((src, dst));
            } else {
                threshold_groups.push((current_threshold, current_group));
                current_threshold = weight;
                current_group = vec![(src, dst)];
            }
        }
        if !current_group.is_empty() {
            threshold_groups.push((current_threshold, current_group));
        }
        
        // Build merge events directly with RoaringBitmaps
        let mut merges = Vec::new();
        let mut uf = UnionFind::new(num_records);
        
        for (threshold, edges_at_threshold) in threshold_groups {
            // Track which root components are merging
            let mut merging_roots: HashMap<usize, RoaringBitmap> = HashMap::new();
            
            // First pass: identify all merging groups
            for (src, dst) in &edges_at_threshold {
                let root_src = uf.find(*src as usize);
                let root_dst = uf.find(*dst as usize);
                
                if root_src != root_dst {
                    // Collect all records in each component
                    let mut src_group = RoaringBitmap::new();
                    let mut dst_group = RoaringBitmap::new();
                    
                    for i in 0..num_records {
                        if uf.find(i) == root_src {
                            src_group.insert(i as u32);
                        }
                        if uf.find(i) == root_dst {
                            dst_group.insert(i as u32);
                        }
                    }
                    
                    merging_roots.entry(root_src).or_insert(src_group.clone());
                    merging_roots.entry(root_dst).or_insert(dst_group.clone());
                }
            }
            
            // Apply merges
            for (src, dst) in &edges_at_threshold {
                uf.union(*src as usize, *dst as usize);
            }
            
            // Create merge events for connected groups
            let mut processed = HashSet::new();
            for root in merging_roots.keys() {
                if processed.contains(root) {
                    continue;
                }
                
                // Find all groups in this connected merge
                let mut merge_groups = Vec::new();
                let mut to_visit = vec![*root];
                
                while let Some(current) = to_visit.pop() {
                    if !processed.insert(current) {
                        continue;
                    }
                    
                    if let Some(group) = merging_roots.get(&current) {
                        merge_groups.push(group.clone());
                        
                        // Find other roots that merge with this one
                        for (other_root, _) in &merging_roots {
                            if !processed.contains(other_root) {
                                // Check if they're now in same component
                                if uf.find(*other_root) == uf.find(current) {
                                    to_visit.push(*other_root);
                                }
                            }
                        }
                    }
                }
                
                if merge_groups.len() > 1 {
                    merges.push(MergeEvent {
                        threshold,
                        merging_groups: merge_groups,
                    });
                }
            }
        }
        
        PartitionHierarchy {
            context,  // Store reference to context
            merges,
            partition_cache: LruCache::new(10),
            metric_state: None,
            threshold_index: Self::build_threshold_index(&merges),
            cache_size: 10,
        }
    }
    
    fn build_threshold_index(merges: &[MergeEvent]) -> BTreeMap<u32, usize> {
        merges.iter()
            .enumerate()
            .map(|(idx, merge)| (Self::threshold_to_key(merge.threshold), idx))
            .collect()
    }
}
```

## Partition reconstruction from merge events

### Efficient reconstruction algorithm with fixed-point thresholds

```rust
impl PartitionHierarchy {
    pub fn at_threshold(&mut self, threshold: f64) -> Result<&PartitionLevel, HierarchyError> {
        // Validate threshold
        if threshold < 0.0 || threshold > 1.0 {
            return Err(HierarchyError::InvalidThreshold(threshold));
        }
        
        let key = Self::threshold_to_key(threshold);
        
        // Check cache first using integer key
        if self.partition_cache.contains(&key) {
            return Ok(self.partition_cache.get(&key).unwrap());
        }
        
        // Check if cache is full and warn
        if self.partition_cache.len() >= self.cache_size {
            log::debug!("Cache full, evicting LRU partition");
        }
        
        // Reconstruct from merge events
        let partition = self.reconstruct_at_threshold(threshold)?;
        self.partition_cache.put(key, partition);
        Ok(self.partition_cache.get(&key).unwrap())
    }
    
    fn reconstruct_at_threshold(&self, threshold: f64) -> PartitionLevel {
        // Get complete record space from context
        let num_records = self.context.records.len();
        
        // Start with all singletons (including isolates)
        let mut uf = UnionFind::new(num_records);
        
        // Apply all merges with threshold >= t
        for merge in &self.merges {
            if merge.threshold >= threshold {
                // Union all groups in this merge
                let all_records: RoaringBitmap = merge.merging_groups
                    .iter()
                    .fold(RoaringBitmap::new(), |mut acc, group| {
                        acc.or_inplace(group);
                        acc
                    });
                
                // Pick first record as representative
                let mut iter = all_records.iter();
                if let Some(first) = iter.next() {
                    for record in iter {
                        uf.union(first as usize, record as usize);
                    }
                }
            } else {
                // Merges are sorted, so we can stop
                break;
            }
        }
        
        // Convert union-find to partition
        let mut entities: HashMap<usize, RoaringBitmap> = HashMap::new();
        for record in 0..num_records {
            let root = uf.find(record);
            entities.entry(root).or_default().insert(record as u32);
        }
        
        let entities: Vec<RoaringBitmap> = entities.into_values().collect();
        let entity_sizes = entities.iter().map(|e| e.cardinality()).collect();
        
        PartitionLevel {
            threshold,
            entities,
            entity_sizes,
            lazy_metrics: LazyMetricsCache::new(),
        }
    }
}
```

## Helper functions for hierarchy construction

```rust
/// Convert Key-based edges to u32 indices using DataContext
/// O(m) where m = number of edges
fn edges_to_indices(edges: &[(Key, Key, f64)], context: &DataContext) -> Vec<(u32, u32, f64)> {
    edges.iter().map(|(k1, k2, w)| {
        let idx1 = context.get_index(k1).expect("Key not in context");
        let idx2 = context.get_index(k2).expect("Key not in context");
        (idx1 as u32, idx2 as u32, *w)
    }).collect()
}

/// Translate all record indices in hierarchy according to map
/// O(m) where m = number of merge events
impl PartitionHierarchy {
    pub fn translate_indices(&mut self, translation_map: &TranslationMap) {
        for merge in &mut self.merges {
            // Translate each merging group
            let mut translated_groups = Vec::new();
            for group in &merge.merging_groups {
                let mut translated_group = RoaringBitmap::new();
                for old_idx in group.iter() {
                    if let Some(&new_idx) = translation_map.get(old_idx) {
                        translated_group.insert(new_idx as u32);
                    }
                }
                translated_groups.push(translated_group);
            }
            merge.merging_groups = translated_groups;
        }
    }
}
```

## Adding collections to frames with O(k + m) complexity

```rust
impl EntityFrame {
    pub fn add_collection(&mut self, name: &str, collection: Collection) {
        // Collections from frames are immutable views
        if collection.is_view {
            panic!("Cannot add a view collection to a frame. Use copy() first.");
        }
        
        if Arc::ptr_eq(&collection.context, &self.context) {
            // Same context, just add hierarchy
            self.collections.insert(name.into(), collection.hierarchy);
        } else {
            // Different context, need assimilation
            let translation = self.assimilate(collection);
            self.collections.insert(name.into(), translation.hierarchy);
        }
    }
    
    fn assimilate(&mut self, collection: Collection) -> AssimilatedCollection {
        // Complexity: O(k) where k = incoming collection's records
        //           + O(m) where m = merge events to rewrite
        let mut translation_map = TranslationMap::new();
        
        // Identity resolution: (source, key) determines identity
        // O(k) iterations with O(1) HashMap lookups
        for (idx, record) in collection.context.records.iter().enumerate() {
            let identity = (record.source_id, &record.key);
            
            if let Some(&existing_idx) = self.context.identity_map.get(&identity) {
                // Record exists, map to existing - O(1)
                translation_map.insert(idx, existing_idx);
            } else {
                // New record, append to context - O(1)
                let new_idx = self.context.records.len();
                self.context.records.push(record.clone());
                self.context.identity_map.insert(identity, new_idx);
                translation_map.insert(idx, new_idx);
            }
        }
        
        // Translate hierarchy to use new indices - O(m)
        let mut new_hierarchy = collection.hierarchy.clone();
        new_hierarchy.translate_indices(&translation_map);
        new_hierarchy.context = self.context.clone();  // Update context reference
        
        AssimilatedCollection {
            hierarchy: new_hierarchy,
            translation_map,
        }
    }
}
```

## Automatic memory management

### Compaction on collection removal

```rust
impl EntityFrame {
    pub fn drop(&mut self, name: &str) {
        if let Some(hierarchy) = self.collections.remove(name) {
            // Track which records were used by this collection
            let collection_records = hierarchy.get_all_record_indices();
            
            // Mark as garbage
            self.garbage_records.or_inplace(&collection_records);
            
            // Check if compaction needed
            let garbage_ratio = self.garbage_records.cardinality() as f64 / 
                              self.context.records.len() as f64;
            
            if garbage_ratio > 0.5 {
                self.auto_compact();
            }
        }
    }
    
    fn auto_compact(&mut self) {
        // Find live records
        let mut live_records = RoaringBitmap::new();
        for hierarchy in self.collections.values() {
            live_records.or_inplace(&hierarchy.get_all_record_indices());
        }
        
        // Build new context with only live records
        let mut new_context = DataContext::new();
        let mut translation_map = TranslationMap::new();
        
        for &old_idx in live_records.iter() {
            let record = &self.context.records[old_idx as usize];
            let new_idx = new_context.records.len();
            new_context.records.push(record.clone());
            translation_map.insert(old_idx as usize, new_idx);
        }
        
        // Update all hierarchies
        for hierarchy in self.collections.values_mut() {
            hierarchy.translate_indices(&translation_map);
        }
        
        // Swap contexts
        self.context = Arc::new(new_context);
        self.garbage_records.clear();
    }
}
```

## Lazy metric computation framework

### LazyMetric pattern implementation with complexity annotations

```rust
pub struct LazyMetricsCache {
    // Computed on first access, then cached
    precision: HashMap<OrderedFloat<f64>, OnceCell<f64>>,
    recall: HashMap<OrderedFloat<f64>, OnceCell<f64>>,
    f1_score: HashMap<OrderedFloat<f64>, OnceCell<f64>>,
    
    // LRU cache for expensive pairwise computations
    jaccard_cache: LruCache<(EntityId, EntityId), f64>,
    
    // Computation tracking
    computation_stats: ComputationStats,
}

pub struct ComputationStats {
    total_computations: usize,
    incremental_computations: usize,
    cache_hits: usize,
    avg_incremental_speedup: f64,
}

impl LazyMetricsCache {
    pub fn get_metric(&mut self, 
                      metric: MetricType, 
                      threshold: f64,
                      ground_truth: &PartitionLevel) -> f64 {
        let key = OrderedFloat(threshold);
        
        match metric {
            MetricType::Precision => {
                self.precision.entry(key).or_insert_with(|| {
                    self.compute_or_update_metric(
                        MetricType::Precision, 
                        threshold, 
                        ground_truth
                    ).into()
                }).get().copied().unwrap()
            },
            MetricType::BCubed => {
                // B-cubed is semi-incremental: O(k × avg_entity_size)
                // where k = affected entities
                self.compute_bcubed(threshold, ground_truth)
            },
            // Similar for other metrics...
        }
    }
    
    fn compute_or_update_metric(&mut self,
                                metric: MetricType,
                                threshold: f64,
                                truth: &PartitionLevel) -> f64 {
        // Find nearest computed threshold
        if let Some((prev_t, prev_val)) = self.find_nearest_computed(metric, threshold) {
            // Simple heuristic: if too far away, recompute from scratch
            if (threshold - prev_t).abs() > threshold * 0.1 {
                self.computation_stats.total_computations += 1;
                self.compute_from_scratch(metric, threshold, truth)
            } else {
                // Incremental update - O(k) where k = affected entities
                let delta = self.compute_delta(metric, prev_t, threshold, truth);
                self.computation_stats.incremental_computations += 1;
                prev_val + delta
            }
        } else {
            // First computation - O(n²) but only once
            self.computation_stats.total_computations += 1;
            self.compute_from_scratch(metric, threshold, truth)
        }
    }
}

// Metric computation complexity annotations
// Fully incremental O(k): Precision, Recall, F1, ARI, NMI
// Semi-incremental O(k × avg_entity_size): B-cubed metrics
// Note: All metrics benefit from caching between repeated queries
```

## Sparse structure exploitation

### Natural sparsity patterns

```rust
// 1. Sparse contingency tables
pub struct SparseContingencyTable {
    // Most entity pairs have zero overlap
    nonzero_cells: HashMap<(EntityId, EntityId), u32>,
    row_marginals: Vec<u32>,
    col_marginals: Vec<u32>,
    total: u32,
}

impl SparseContingencyTable {
    pub fn update_incremental(&mut self, merge: &MergeEvent) {
        // Only update cells affected by merge - O(k)
        for group in &merge.merging_groups {
            // Update only non-zero cells involving records in this group
            let affected_cells = self.nonzero_cells
                .keys()
                .filter(|(r, c)| {
                    // Check if either entity contains any record from the merging group
                    self.entity_contains_any(*r, group) || 
                    self.entity_contains_any(*c, group)
                })
                .cloned()
                .collect::<Vec<_>>();
            
            for cell in affected_cells {
                self.update_cell(cell, merge);
            }
        }
    }
}

// 2. Block structure for disconnected components
pub struct BlockedHierarchy {
    blocks: Vec<PartitionHierarchy>,
    block_assignments: Vec<BlockId>,
    
    pub fn process_parallel<T>(&self, op: impl Fn(&PartitionHierarchy) -> T + Sync) 
        -> Vec<T> 
    where T: Send 
    {
        self.blocks.par_iter().map(op).collect()
    }
}

// 3. Sparse edge representation
pub struct SparseEdgeList {
    // Only store actual edges, not full matrix
    edges: Vec<(u32, u32, f64)>,
    
    // For efficient lookup
    adjacency: HashMap<u32, Vec<(u32, f64)>>,
}
```

## Cross-collection comparison algorithms

### Algorithm selection strategy

When comparing collections, we have two distinct algorithms optimised for different scenarios:

**Delta-based algorithm** (incremental updates)
- **Complexity**: O(k) updates between adjacent thresholds where k = affected entities
- **Best for**: Single collection operations and single-dimension changes
- **Use cases**:
  - Single collection sweep: Incremental updates between thresholds
  - Sweep × point comparison: One collection fixed, one sweeping

**Record-based algorithm** (full record iteration)
- **Complexity**: O(r) where r = number of records
- **Best for**: Cartesian product comparisons (sweep × sweep)
- **Key insight**: Collections in an EntityFrame share the same underlying record space
- **Performance**: 1000-10000× speedup for large-scale comparisons

### The shared record space insight

Collections in an EntityFrame partition the exact same set of records:
- Each record appears in exactly one entity in collection A
- The same record appears in exactly one entity in collection B
- We're comparing different ways of partitioning the same space

This enables building contingency tables by iterating records rather than comparing entity pairs.

### Complexity analysis

For cross-collection comparison at single thresholds:

**Entity-based approach** (traditional):
- Time: O(k₁ × k₂) entity comparisons
- Example: 800k × 800k entities = 640 billion operations

**Record-based approach** (optimised):
- Time: O(r) record iterations + O(k₁ + k₂) marginals
- Example: 1M records = 1 million operations
- Speedup: 640,000× theoretical, 1000-10000× practical

### Record-based algorithm structure

```rust
pub fn from_partitions_via_records(
    partition1: &PartitionLevel,
    partition2: &PartitionLevel,
    context: &DataContext,
) -> SparseContingencyTable {
    // Build reverse indices: record → entity
    let record_to_entity1 = build_reverse_index(partition1);
    let record_to_entity2 = build_reverse_index(partition2);
    
    // Single pass over records (parallelisable)
    let pairs: Vec<(EntityId, EntityId)> = (0..num_records)
        .into_par_iter()
        .filter_map(|record_idx| {
            match (record_to_entity1[record_idx], record_to_entity2[record_idx]) {
                (Some(e1), Some(e2)) => Some((e1, e2)),
                _ => None,
            }
        })
        .collect();
    
    // Aggregate into contingency table
    build_contingency_table(pairs, partition1, partition2)
}
```

### When to use each algorithm

```
if both expressions are sweeps:
    use record-based  # O(r) beats O(k₁ × k₂ × n₁ × n₂)
else:
    use delta-based   # O(k) incremental updates sufficient
```

For a 5×5 sweep comparison (25 total comparisons):
- Delta-based: 80k × 80k × 25 = 160 billion operations
- Record-based: 1M records × 1 pass = 1 million operations
- Speedup: 160,000×

## Dual processing for entity operations

### Built-in operations with parallel Rust execution

```rust
pub enum EntityProcessor {
    // Fast built-in operations run in parallel
    BuiltIn(BuiltInOp),
    // Custom Python function - flexible but slower
    Python(PyObject),
}

pub enum BuiltInOp {
    Hash(HashAlgorithm),  // SHA256, MD5, Blake3, etc.
    Size,                 // Entity cardinality
    Density,              // Internal connectivity
    Fingerprint,          // MinHash or similar
}

impl PartitionLevel {
    pub fn map_over_entities<T>(&self, 
                               processor: EntityProcessor) -> HashMap<EntityId, T> 
    where T: Send + Sync 
    {
        match processor {
            EntityProcessor::BuiltIn(op) => {
                // Parallel execution with Rayon
                self.entities
                    .par_iter()
                    .enumerate()
                    .map(|(id, entity)| {
                        let result = match op {
                            BuiltInOp::Hash(algo) => compute_hash(entity, algo),
                            BuiltInOp::Size => entity.cardinality(),
                            BuiltInOp::Density => compute_density(entity),
                            BuiltInOp::Fingerprint => compute_fingerprint(entity),
                        };
                        (EntityId(id), result)
                    })
                    .collect()
            },
            EntityProcessor::Python(func) => {
                // Single-threaded Python execution
                self.entities
                    .iter()
                    .enumerate()
                    .map(|(id, entity)| {
                        let result = call_python_func(func, entity);
                        (EntityId(id), result)
                    })
                    .collect()
            }
        }
    }
}
```
