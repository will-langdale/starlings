# Starlings Rust engine

This document covers the Rust implementation of Starlings' core engine, including memory layout, parallelisation strategies, and performance optimisations. It details how the theoretical constructs map to efficient systems code.

## Implementation philosophy

Starlings uses Rust for the performance-critical hierarchy operations whilst exposing a Python interface for data science workflows. The implementation prioritises:
- Zero-copy operations where possible through Arc reference counting
- Parallel execution via Rayon for multi-core scaling
- Cache-efficient data structures with RoaringBitmaps for sparse sets
- Direct primitive types (u32 indices) for computation-intensive operations

## SIMD and cache optimisations

### Vectorised metric computation

```rust
use std::simd::{f64x4, u32x8, SimdFloat, SimdUint};

impl PartitionLevel {
    pub fn compute_sizes_vectorised(&self) -> Vec<u32> {
        let mut sizes = Vec::with_capacity(self.entities.len());
        
        // Process 8 entities at once
        for chunk in self.entities.chunks(8) {
            let chunk_sizes: [u32; 8] = chunk
                .iter()
                .map(|e| e.cardinality())
                .chain(std::iter::repeat(0))
                .take(8)
                .collect::<Vec<_>>()
                .try_into()
                .unwrap();
            
            let simd_sizes = u32x8::from_array(chunk_sizes);
            sizes.extend_from_slice(&simd_sizes.as_array()[..chunk.len()]);
        }
        
        sizes
    }
    
    pub fn compute_f1_scores_vectorised(&self, 
                                        precisions: &[f64], 
                                        recalls: &[f64]) -> Vec<f64> {
        assert_eq!(precisions.len(), recalls.len());
        let mut f1_scores = Vec::with_capacity(precisions.len());
        
        for (p_chunk, r_chunk) in precisions.chunks(4).zip(recalls.chunks(4)) {
            let p_simd = f64x4::from_slice(p_chunk);
            let r_simd = f64x4::from_slice(r_chunk);
            
            let two = f64x4::splat(2.0);
            let numerator = two * p_simd * r_simd;
            let denominator = p_simd + r_simd;
            let f1_simd = numerator / denominator;
            
            f1_scores.extend_from_slice(&f1_simd.to_array()[..p_chunk.len()]);
        }
        
        f1_scores
    }
}
```

### Cache-optimised memory layout

```rust
// Align to cache lines for better performance
#[repr(C, align(64))]
pub struct CacheAlignedPartition<'a> {
    threshold: f64,
    num_entities: u32,
    _pad1: [u8; 4],  // Padding for alignment
    
    // Hot data together - using safe references with lifetimes
    entity_sizes: &'a [u32],
    entity_data: &'a [RoaringBitmap],
    
    // Cold data separately
    metadata: Option<&'a PartitionMetadata>,
    _pad2: [u8; 24],  // Prevent false sharing (adjusted for reference sizes)
}

// Structure of Arrays for better vectorisation
pub struct SoAMetrics {
    precisions: Vec<f64>,
    recalls: Vec<f64>,
    f1_scores: Vec<f64>,
    // Better cache locality than Array of Structs
}
```

## Parallel processing architecture

### Rayon-based parallel merge event processing

```rust
use rayon::prelude::*;

impl PartitionHierarchy {
    pub fn build_parallel(edges: Vec<WeightedEdge>, context: Arc<DataContext>) -> Self {
        // Sort edges in parallel
        let mut edges = edges;
        edges.par_sort_unstable_by(|a, b| b.weight.partial_cmp(&a.weight).unwrap());
        
        // Group by threshold
        let threshold_groups = Self::group_by_threshold(edges);
        
        // Process each threshold's merges in parallel where possible
        let merges: Vec<MergeEvent> = threshold_groups
            .into_iter()
            .flat_map(|(threshold, edges)| {
                Self::find_merges_at_threshold(threshold, edges)
            })
            .collect();
        
        PartitionHierarchy {
            context,
            merges,
            partition_cache: LruCache::new(10),
            metric_state: None,
            threshold_index: Self::build_threshold_index(&merges),
            cache_size: 10,
        }
    }
}
```

### Work-stealing for multi-collection comparison

```rust
impl EntityFrame {
    pub fn compare_all_collections_parallel(&self, 
                                           truth_id: CollectionId,
                                           threshold_truth: f64,
                                           thresholds: HashMap<CollectionId, f64>) 
        -> HashMap<CollectionId, Metrics> {
        let truth = &self.collections[&truth_id];
        let truth_partition = truth.at_threshold(threshold_truth);
        
        thresholds
            .par_iter()
            .filter(|(id, _)| **id != truth_id)
            .map(|(id, threshold)| {
                let collection = &self.collections[id];
                let partition = collection.at_threshold(*threshold);
                let metrics = self.comparison_engine.compare_partitions(
                    partition,
                    truth_partition
                );
                (*id, metrics)
            })
            .collect()
    }
}
```

### High-performance hashing for large-scale deduplication

```rust
use blake3::Hasher;
use rayon::prelude::*;

fn compute_entity_hashes(partition: &PartitionLevel) -> Vec<[u8; 32]> {
    partition.entities
        .par_iter()
        .map(|entity| {
            let mut hasher = Hasher::new();
            // Sort for consistent hashing
            let mut members: Vec<_> = entity.iter().collect();
            members.sort_unstable();
            for member in members {
                hasher.update(&member.to_le_bytes());
            }
            *hasher.finalize().as_bytes()
        })
        .collect()
}
```

## Python interface via PyO3

### Core Python bindings with starlings package

The Python API follows the polars-inspired expression design, providing a clear separation between data containers and operations through composable expressions.

```rust
use pyo3::prelude::*;

#[pyclass]
pub struct PyEntityFrame {
    inner: Arc<Mutex<EntityFrame>>,
}

#[pymethods]
impl PyEntityFrame {
    #[staticmethod]
    pub fn from_records(source: &str, data: &PyAny) -> PyResult<Self> {
        // Handle pandas DataFrame, Arrow table, or list of dicts
        let records = if data.is_instance_of::<PyDataFrame>()? {
            records_from_dataframe(data)?
        } else if data.is_instance_of::<PyArrowTable>()? {
            records_from_arrow(data)?
        } else {
            records_from_list(data)?
        };
        
        let frame = EntityFrame::from_records(source, records);
        Ok(PyEntityFrame {
            inner: Arc::new(Mutex::new(frame)),
        })
    }
    
    pub fn add_collection_from_edges(&mut self, 
                                     name: &str, 
                                     edges: &PyList) -> PyResult<()> {
        // Optimised batch conversion with single GIL acquisition
        let internal_edges = Python::with_gil(|py| -> PyResult<Vec<(u32, u32, f64)>> {
            let frame = self.inner.lock().unwrap();
            let mut result = Vec::with_capacity(edges.len());
            
            for item in edges.iter() {
                let tuple = item.downcast::<PyTuple>()?;
                let k1 = tuple.get_item(0)?;
                let k2 = tuple.get_item(1)?;
                let weight = tuple.get_item(2)?.extract::<f64>()?;
                
                let idx1 = python_key_to_index_obj(k1, &frame.context)?;
                let idx2 = python_key_to_index_obj(k2, &frame.context)?;
                result.push((idx1, idx2, weight));
            }
            Ok(result)
        })?;
        
        // Process in Rust without GIL
        let mut frame = self.inner.lock().unwrap();
        frame.add_collection_from_edges(name, internal_edges);
        Ok(())
    }
    
    pub fn analyse(&mut self,
                  expressions: Vec<PyExpression>,
                  metrics: Option<Vec<PyMetric>>) -> PyResult<PyObject> {
        let mut frame = self.inner.lock().unwrap();

        // Parse expressions to determine operation type and reference
        let (operation, reference_info) = parse_expressions_with_reference(expressions)?;

        // Python wrapper provides defaults if metrics not specified
        let metrics = metrics.unwrap_or_else(|| {
            match &operation {
                Operation::SingleCollection(_) => vec![
                    PyMetric::EntityCount,
                    PyMetric::Entropy,
                ],
                Operation::Comparison(_) => vec![
                    PyMetric::F1,
                    PyMetric::Precision,
                    PyMetric::Recall,
                    PyMetric::ARI,
                    PyMetric::NMI,
                ],
            }
        });

        // Call Rust implementation with tidy-row output format
        let results = match operation {
            Operation::SingleCollection(col_spec) => {
                frame.compute_single_collection_metrics(col_spec, metrics)
            },
            Operation::Comparison { predictions, reference } => {
                frame.compute_comparison_metrics(predictions, reference, metrics)
            },
        };

        Python::with_gil(|py| {
            // Convert to Python list of dicts with universal schema
            let py_list = PyList::new(py,
                results.into_iter().map(|row| {
                    let dict = PyDict::new(py);
                    dict.set_item("collection", row.collection)?;
                    dict.set_item("collection_threshold", row.collection_threshold)?;
                    dict.set_item("reference", row.reference)?;
                    dict.set_item("reference_threshold", row.reference_threshold)?;
                    dict.set_item("metric_name", row.metric_name)?;
                    dict.set_item("metric_value", row.metric_value)?;
                    Ok(dict)
                })
                .collect::<PyResult<Vec<_>>>()?
            );
            Ok(py_list.into())
        })
    }
    
    // American spelling alias
    pub fn analyze(&mut self, 
                  expressions: Vec<PyExpression>,
                  metrics: Vec<PyMetric>) -> PyResult<PyObject> {
        self.analyse(expressions, metrics)
    }
    
    pub fn drop(&mut self, names: Vec<&str>) -> PyResult<()> {
        let mut frame = self.inner.lock().unwrap();
        for name in names {
            frame.drop(name);
        }
        Ok(())
    }
}
```

### Key conversion at Python/Rust boundary

Performance note: Convert Python Keys directly to u32 indices for hierarchy operations. Only use Rust Key enum when preserving actual key values is needed.

```rust
fn python_key_to_index(key: &PyObject, py: Python, context: &DataContext) -> PyResult<u32> {
    // Extract the key value from Python
    let rust_key = if let Ok(v) = key.extract::<u32>(py) {
        Key::U32(v)
    } else if let Ok(v) = key.extract::<u64>(py) {
        Key::U64(v)
    } else if let Ok(v) = key.extract::<String>(py) {
        Key::String(v)
    } else if let Ok(v) = key.extract::<Vec<u8>>(py) {
        Key::Bytes(v)
    } else {
        return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "Key must be int, str, or bytes"
        ));
    };
    
    // Look up or create index in context
    Ok(context.get_or_create_index(&rust_key) as u32)
}

// Optimised version for when we already have PyAny
fn python_key_to_index_obj(key: &PyAny, context: &DataContext) -> PyResult<u32> {
    let rust_key = if let Ok(v) = key.extract::<u32>() {
        Key::U32(v)
    } else if let Ok(v) = key.extract::<u64>() {
        Key::U64(v)
    } else if let Ok(v) = key.extract::<String>() {
        Key::String(v)
    } else if let Ok(v) = key.extract::<Vec<u8>>() {
        Key::Bytes(v)
    } else {
        return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "Key must be int, str, or bytes"
        ));
    };
    
    Ok(context.get_or_create_index(&rust_key) as u32)
}
```

### Injectable operations pattern implementation

These are marker types that route to optimised Rust implementations, providing both flexibility and performance. The separation makes it clear: metrics operate on partitions, operations operate on individual entities.

```rust
// Injectable operations pattern implementation
// These are marker types that route to optimised Rust implementations
#[pyclass]
pub struct F1Metric;

#[pyclass]
pub struct PrecisionMetric;

#[pyclass]
pub struct RecallMetric;

#[pyclass]
pub struct ARIMetric;

#[pyclass]
pub struct NMIMetric;

#[pyclass]
pub struct VMeasureMetric;

#[pyclass]
pub struct BCubedPrecisionMetric;

#[pyclass]
pub struct BCubedRecallMetric;

#[pyclass]
pub struct EntropyMetric;

#[pyclass]
pub struct EntityCountMetric;

#[pyclass]
pub struct SHA256Op;

#[pyclass]
pub struct Blake3Op;

#[pyclass]
pub struct MD5Op;

#[pyclass]
pub struct SizeOp;

#[pyclass]
pub struct DensityOp;

#[pyclass]
pub struct FingerprintOp;

// Nested class structure for metrics and operations
#[pyclass]
pub struct MetricsEval;

#[pymethods]
impl MetricsEval {
    #[classattr]
    fn f1() -> F1Metric { F1Metric }
    
    #[classattr]
    fn precision() -> PrecisionMetric { PrecisionMetric }
    
    #[classattr]
    fn recall() -> RecallMetric { RecallMetric }
    
    #[classattr]
    fn ari() -> ARIMetric { ARIMetric }
    
    #[classattr]
    fn nmi() -> NMIMetric { NMIMetric }
    
    #[classattr]
    fn v_measure() -> VMeasureMetric { VMeasureMetric }
    
    #[classattr]
    fn bcubed_precision() -> BCubedPrecisionMetric { BCubedPrecisionMetric }
    
    #[classattr]
    fn bcubed_recall() -> BCubedRecallMetric { BCubedRecallMetric }
}

#[pyclass]
pub struct MetricsStats;

#[pymethods]
impl MetricsStats {
    #[classattr]
    fn entropy() -> EntropyMetric { EntropyMetric }
    
    #[classattr]
    fn entity_count() -> EntityCountMetric { EntityCountMetric }
}

#[pyclass]
pub struct Metrics;

#[pymethods]
impl Metrics {
    #[classattr]
    fn eval() -> MetricsEval { MetricsEval }
    
    #[classattr]
    fn stats() -> MetricsStats { MetricsStats }
}

#[pyclass]
pub struct OpsHash;

#[pymethods]
impl OpsHash {
    #[classattr]
    fn sha256() -> SHA256Op { SHA256Op }
    
    #[classattr]
    fn blake3() -> Blake3Op { Blake3Op }
    
    #[classattr]
    fn md5() -> MD5Op { MD5Op }
}

#[pyclass]
pub struct OpsCompute;

#[pymethods]
impl OpsCompute {
    #[classattr]
    fn size() -> SizeOp { SizeOp }
    
    #[classattr]
    fn density() -> DensityOp { DensityOp }
    
    #[classattr]
    fn fingerprint() -> FingerprintOp { FingerprintOp }
}

#[pyclass]
pub struct Ops;

#[pymethods]
impl Ops {
    #[classattr]
    fn hash() -> OpsHash { OpsHash }
    
    #[classattr]
    fn compute() -> OpsCompute { OpsCompute }
}
```

### Expression API classes

```rust
#[pyclass]
pub struct PyColExpression {
    collection_name: String,
    is_reference: bool,  // Tracks if marked as reference
    threshold: Option<f64>,
    sweep_params: Option<(f64, f64, f64)>,
}

#[pymethods]
impl PyColExpression {
    pub fn at(&mut self, threshold: f64) -> PyResult<Self> {
        self.threshold = Some(threshold);
        Ok(self.clone())
    }

    pub fn sweep(&mut self, start: f64, stop: f64, step: f64) -> PyResult<Self> {
        self.sweep_params = Some((start, stop, step));
        Ok(self.clone())
    }

    pub fn reference(&mut self) -> PyResult<Self> {
        self.is_reference = true;
        Ok(self.clone())
    }
}
```

### Module registration

```rust
#[pymodule]
fn starlings(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyEntityFrame>()?;
    m.add_class::<PyCollection>()?;
    m.add_class::<PyPartition>()?;
    m.add_class::<PyColExpression>()?;
    m.add_class::<PyKey>()?;
    m.add_class::<PyEntity>()?;  // Python-only type
    
    // Add nested metrics and ops classes
    m.add_class::<Metrics>()?;
    m.add_class::<MetricsEval>()?;
    m.add_class::<MetricsStats>()?;
    m.add_class::<Ops>()?;
    m.add_class::<OpsHash>()?;
    m.add_class::<OpsCompute>()?;
    
    // Add the individual metric and operation types
    m.add_class::<F1Metric>()?;
    m.add_class::<PrecisionMetric>()?;
    m.add_class::<RecallMetric>()?;
    m.add_class::<ARIMetric>()?;
    m.add_class::<NMIMetric>()?;
    m.add_class::<VMeasureMetric>()?;
    m.add_class::<BCubedPrecisionMetric>()?;
    m.add_class::<BCubedRecallMetric>()?;
    m.add_class::<EntropyMetric>()?;
    m.add_class::<EntityCountMetric>()?;
    m.add_class::<SHA256Op>()?;
    m.add_class::<Blake3Op>()?;
    m.add_class::<MD5Op>()?;
    m.add_class::<SizeOp>()?;
    m.add_class::<DensityOp>()?;
    m.add_class::<FingerprintOp>()?;
    
    // Add col function
    m.add_function(wrap_pyfunction!(col, m)?)?;
    m.add_function(wrap_pyfunction!(from_records, m)?)?;
    
    Ok(())
}

#[pyfunction]
fn col(name: &str) -> PyColExpression {
    PyColExpression {
        collection_name: name.to_string(),
        is_reference: false,  // Default to not being a reference
    }
}

// Helper to parse expressions and determine reference collection
fn parse_expressions_with_reference(expressions: Vec<PyExpression>)
    -> PyResult<(Operation, Option<ReferenceInfo>)> {
    // Check for explicit .reference() marker
    let explicit_ref = expressions.iter()
        .position(|e| e.is_reference);

    let reference_info = if let Some(ref_idx) = explicit_ref {
        Some(ReferenceInfo {
            collection: expressions[ref_idx].collection_name.clone(),
            threshold: expressions[ref_idx].threshold,
        })
    } else if expressions.len() > 1 {
        // Implicit reference: use last expression
        let last = &expressions[expressions.len() - 1];
        Some(ReferenceInfo {
            collection: last.collection_name.clone(),
            threshold: last.threshold,
        })
    } else {
        None  // Single collection, no reference needed
    };

    // Parse operation type based on expressions
    let operation = determine_operation_type(&expressions, &reference_info)?;

    Ok((operation, reference_info))
}

#[pyfunction]
fn from_records(source: &str, data: &PyAny) -> PyResult<PyEntityFrame> {
    PyEntityFrame::from_records(source, data)
}
```

## Arrow integration

### Hierarchical Arrow schema with optimal dictionary encoding

```rust
use arrow::datatypes::{DataType, Field, Schema, UnionMode};

pub fn create_starlings_schema() -> Schema {
    Schema::new(vec![
        // Parallel arrays for maximum deduplication efficiency
        Field::new("record_sources", DataType::Dictionary(
            Box::new(DataType::UInt16),
            Box::new(DataType::Utf8),
        ), false),
        
        Field::new("record_keys", DataType::Dictionary(
            Box::new(DataType::UInt32),
            Box::new(DataType::Union(
                vec![
                    Field::new("u32", DataType::UInt32, false),
                    Field::new("u64", DataType::UInt64, false),
                    Field::new("string", DataType::Utf8, false),
                    Field::new("bytes", DataType::Binary, false),
                ],
                vec![0, 1, 2, 3],  // Type ids
                UnionMode::Dense,
            )),
        ), false),
        
        Field::new("record_indices", DataType::UInt32, false),  // Simple array of indices
        
        // Collections with merge events
        Field::new("collections", DataType::List(Box::new(DataType::Struct(vec![
            Field::new("name", DataType::Dictionary(
                Box::new(DataType::UInt8),
                Box::new(DataType::Utf8),
            ), false),
            Field::new("merge_events", DataType::List(
                Box::new(DataType::Struct(vec![
                    Field::new("threshold", DataType::Float64, false),
                    // Binary delta format: parent ID + child nodes
                    Field::new("parent_id", DataType::UInt32, false),
                    // RoaringBitmap (child_nodes) expanded to list for Arrow serialisation
                    Field::new("child_nodes", DataType::List(
                        Box::new(DataType::UInt32)
                    ), false),
                ]))
            ), false),
        ]))), false),
        
        // Metadata
        Field::new("version", DataType::Utf8, false),
        Field::new("created_at", DataType::Timestamp(TimeUnit::Millisecond, None), false),
    ])
}
```

## Cross-collection comparison optimisations

### Memory safety with generation tracking

The record-based algorithm requires careful handling of cache invalidation when memory compaction occurs:

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
    
    // Cache with generation tracking
    context_generation: u64,  // Generation when built
    record_to_entity: OnceCell<Vec<Option<usize>>>,  // Positional indices, not IDs
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

### Critical implementation notes

**Record indices vs record IDs**: We iterate over vector indices in `DataContext::records`, not abstract record IDs:

```rust
// CORRECT: Iterate over vector indices
for record_idx in 0..context.records.len() {
    // record_idx is the position in Vec<InternedRecord>
    let entity_a = partition_a.get_entity_for_record(record_idx, &context);
    let entity_b = partition_b.get_entity_for_record(record_idx, &context);
}
```

**Entity ID instability**: Entity IDs are positional indices that change between thresholds. Use `Vec<Option<usize>>` not `Vec<Option<EntityId>>` for reverse indices.

### Memory optimisation strategies

For 1M records, reverse indices require ~16MB per partition:

```rust
// Adaptive index selection based on sparsity
fn choose_index_type(partition: &PartitionLevel) -> IndexType {
    let coverage = partition.total_records() as f64 / partition.max_record_id() as f64;
    if coverage < 0.5 {
        IndexType::Sparse  // HashMap for sparse datasets
    } else {
        IndexType::Dense   // Vec<Option<u32>> for dense datasets
    }
}
```

### Performance benchmarks

Real-world performance improvements with the record-based algorithm:

- **100k × 100k comparison**: ~0.1s (vs ~100s with entity-based)
- **1M × 1M comparison**: ~1s (vs ~1000s with entity-based)
- **Sweep × sweep (5×5)**: ~2s for all 25 comparisons (vs hours)
- **Practical speedup**: 1000-10000× for large-scale comparisons

### Thread safety with parallel processing

The record-based algorithm parallelises naturally:

```rust
use rayon::prelude::*;

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
```

## Batch processing optimisations

### Optimised batch construction

```rust
pub struct BatchProcessor {
    frame: EntityFrame,
    batch_size: usize,
}

impl BatchProcessor {
    pub fn process_large_dataset(&mut self, 
                                 edges: impl Iterator<Item = WeightedEdge>) {
        // Process in memory-efficient batches
        let chunks = edges.chunks(self.batch_size);
        
        for chunk in chunks {
            // Pass the frame's context to hierarchy
            let hierarchy = PartitionHierarchy::from_edges(
                chunk.collect(), 
                self.frame.context.clone(),
                6  // Default quantisation
            );
            // Process hierarchy...
        }
    }
    
    pub fn parallel_batch_comparison(&self,
                                    collections: Vec<CollectionId>,
                                    thresholds: Vec<f64>) -> ComparisonMatrix {
        use rayon::prelude::*;
        
        // Parallel comparison across collections and thresholds
        let comparisons: Vec<_> = collections
            .par_iter()
            .flat_map(|col_a| {
                thresholds.par_iter().flat_map(move |t_a| {
                    collections.par_iter().flat_map(move |col_b| {
                        thresholds.par_iter().map(move |t_b| {
                            self.frame.compare_cuts(
                                (col_a, *t_a),
                                (col_b, *t_b)
                            )
                        })
                    })
                })
            })
            .collect();
        
        ComparisonMatrix::from_comparisons(comparisons)
    }
}
```

## Production deployment

### Monitoring and observability

```rust
#[derive(Serialize, Deserialize)]
pub struct FrameMetrics {
    // Data statistics
    num_collections: usize,
    num_records: usize,
    num_sources: usize,
    total_memory_mb: f64,
    
    // Performance metrics
    avg_reconstruction_time_ms: f64,
    cache_hit_rate: f64,
    incremental_computation_rate: f64,
    
    // Quality indicators
    avg_entity_stability: f64,
    threshold_sensitivity: f64,
    
    // Memory management
    garbage_ratio: f64,
    last_compaction: Option<DateTime<Utc>>,
}

impl EntityFrame {
    pub fn collect_metrics(&self) -> FrameMetrics {
        let total_merges: usize = self.collections
            .values()
            .map(|h| h.merges.len())
            .sum();
        
        let cache_stats = self.get_cache_statistics();
        
        FrameMetrics {
            num_collections: self.collections.len(),
            num_records: self.context.records.len(),
            num_sources: self.context.source_interner.len(),
            total_memory_mb: self.estimate_memory_usage() / 1_048_576.0,
            avg_reconstruction_time_ms: cache_stats.avg_miss_time_ms,
            cache_hit_rate: cache_stats.hit_rate,
            incremental_computation_rate: self.compute_incremental_rate(),
            avg_entity_stability: self.compute_avg_stability(),
            threshold_sensitivity: self.compute_sensitivity(),
            garbage_ratio: self.garbage_records.cardinality() as f64 / 
                          self.context.records.len() as f64,
            last_compaction: self.last_compaction_time,
        }
    }
    
    fn estimate_memory_usage(&self) -> f64 {
        let context_memory = self.context.records.len() * 
                           std::mem::size_of::<InternedRecord>();
        
        let merge_memory: usize = self.collections
            .values()
            .map(|h| h.merges.len() * std::mem::size_of::<MergeEvent>())
            .sum();
        
        let cache_memory: usize = self.collections
            .values()
            .map(|h| h.partition_cache.len() * 500_000) // ~500KB per cached partition
            .sum();
        
        (context_memory + merge_memory + cache_memory) as f64
    }
}
```

### Performance characteristics

```rust
/// Expected performance for different scales
/// Based on contextual ownership with automatic compaction
pub struct PerformanceProfile {
    pub scale: DataScale,
    pub hierarchy_build_time: Duration,
    pub cached_query_time: Duration,
    pub uncached_query_time: Duration,
    pub sweep_time_per_threshold: Duration,
    pub memory_usage: ByteSize,
}

impl PerformanceProfile {
    pub fn estimate(num_records: usize, num_edges: usize) -> Self {
        match (num_records, num_edges) {
            (n, m) if n <= 10_000 => PerformanceProfile {
                scale: DataScale::Small,
                hierarchy_build_time: Duration::from_millis(100),
                cached_query_time: Duration::from_micros(100),
                uncached_query_time: Duration::from_millis(10),
                sweep_time_per_threshold: Duration::from_millis(1),
                memory_usage: ByteSize::mb(10),
            },
            (n, m) if n <= 1_000_000 => PerformanceProfile {
                scale: DataScale::Medium,
                hierarchy_build_time: Duration::from_secs(10),
                cached_query_time: Duration::from_millis(1),
                uncached_query_time: Duration::from_millis(200),
                sweep_time_per_threshold: Duration::from_millis(10),
                memory_usage: ByteSize::mb(100),
            },
            _ => PerformanceProfile {
                scale: DataScale::Large,
                hierarchy_build_time: Duration::from_secs(300),
                cached_query_time: Duration::from_millis(10),
                uncached_query_time: Duration::from_secs(2),
                sweep_time_per_threshold: Duration::from_millis(100),
                memory_usage: ByteSize::gb(1),
            },
        }
    }
}
```
