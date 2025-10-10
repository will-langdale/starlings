use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyString, PyType};
use std::sync::Arc;

mod expressions;
use expressions::{parse_expression, parse_metric};
use starlings_core::expressions::{
    compute_single_metric, generate_sweep_thresholds, ExpressionType, MetricType,
};
use starlings_core::metrics::{CoreMetricType, MetricEngine, MetricResults};

use starlings_core::core::ensure_memory_safety;
use starlings_core::core::resource_monitor::SafetyError;
use starlings_core::debug_println;
use starlings_core::test_utils;
use starlings_core::{DataContext, EntityFrame, Key, PartitionHierarchy, PartitionLevel};

/// Convert threshold to cache key using fixed-point representation
/// This ensures consistent cache keys for floating point thresholds
#[inline]
fn threshold_to_cache_key(threshold: f64) -> u64 {
    (threshold * 1_000_000.0).round() as u64
}

/// Convert MetricType to CoreMetricType
#[allow(dead_code)]
fn convert_metric_type(metric: &MetricType) -> CoreMetricType {
    match metric {
        MetricType::F1 => CoreMetricType::F1,
        MetricType::Precision => CoreMetricType::Precision,
        MetricType::Recall => CoreMetricType::Recall,
        MetricType::ARI => CoreMetricType::ARI,
        MetricType::NMI => CoreMetricType::NMI,
        MetricType::VMeasure => CoreMetricType::VMeasure,
        MetricType::BCubedPrecision => CoreMetricType::BCubedPrecision,
        MetricType::BCubedRecall => CoreMetricType::BCubedRecall,
        MetricType::EntityCount => CoreMetricType::EntityCount,
        MetricType::Entropy => CoreMetricType::Entropy,
    }
}

/// Progress callback type for Rust-level progress reporting
type ProgressCallback = Arc<dyn Fn(f64, &str) + Send + Sync>;

/// Helper function to create progress callback wrapper
fn create_progress_wrapper(callback: Py<pyo3::PyAny>) -> ProgressCallback {
    Arc::new(move |progress: f64, message: &str| {
        Python::attach(|py| {
            if let Err(_e) = callback.call1(py, (progress, message)) {
                // Progress callback error - silently continue
            }
        });
    })
}

/// Helper function to map storage errors to Python exceptions
fn map_storage_error(error: String) -> PyErr {
    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(error)
}

/// Helper function to map safety errors to Python exceptions
fn map_safety_error(error: SafetyError) -> PyErr {
    match error {
        SafetyError::InsufficientMemory {
            required_mb,
            available_mb,
            limit_mb,
        } => PyErr::new::<pyo3::exceptions::PyMemoryError, _>(format!(
            "Operation requires {}MB but only {}MB available (limit: {}MB). Consider: 1) Smaller dataset, 2) Free memory, 3) Increase STARLINGS_MEMORY_LIMIT",
            required_mb, available_mb, limit_mb
        )),
    }
}

/// Generator for entity resolution edges that yields batches
#[pyclass]
pub struct EdgeGenerator {
    edges: Vec<(i64, i64, f64)>,
    batch_size: usize,
    current_index: usize,
}

#[pymethods]
impl EdgeGenerator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self) -> Option<Vec<(i64, i64, f64)>> {
        if self.current_index >= self.edges.len() {
            return None;
        }

        let end_index = (self.current_index + self.batch_size).min(self.edges.len());
        let batch = self.edges[self.current_index..end_index].to_vec();
        self.current_index = end_index;

        Some(batch)
    }
}

/// A partition of records into entities at a specific threshold.
#[pyclass(name = "Partition")]
#[derive(Clone)]
pub struct PyPartition {
    partition: Arc<PartitionLevel>,
}

#[pymethods]
impl PyPartition {
    /// Get the number of entities in this partition.
    fn __len__(&self) -> usize {
        self.partition.entities().len()
    }

    /// Get entities as list of lists of record indices.
    #[getter]
    fn entities(&self) -> Vec<Vec<u32>> {
        self.partition
            .entities()
            .iter()
            .map(|bitmap| bitmap.iter().collect())
            .collect()
    }

    /// Get the number of entities in this partition.
    #[getter]
    fn num_entities(&self) -> usize {
        self.partition.entities().len()
    }

    /// String representation for debugging.
    fn __repr__(&self) -> String {
        format!("Partition(entities={})", self.entities().len())
    }
}

/// Hierarchical partition structure that generates entities at any threshold.
#[pyclass(name = "Collection")]
pub struct PyCollection {
    hierarchy: PartitionHierarchy,
    is_view: bool,
}

impl PyCollection {
    /// Create a view collection (internal helper)
    pub(crate) fn new_view(hierarchy: PartitionHierarchy) -> Self {
        PyCollection {
            hierarchy,
            is_view: true,
        }
    }
}

#[pymethods]
impl PyCollection {
    /// Build collection from weighted edges.
    ///
    /// Creates a hierarchical partition structure from similarity edges between records.
    /// Records can be any hashable Python type (int, str, bytes) and are automatically
    /// converted to internal indices for efficient processing.
    ///
    /// Args:
    ///     edges (List[Tuple[Any, Any, float]]): List of (record_i, record_j, similarity) tuples.
    ///         Records can be any hashable type (int, str, bytes). Similarities should be
    ///         between 0.0 and 1.0.
    ///     source (str, optional): Source name for record context. Defaults to "default".
    ///
    /// Returns:
    ///     Collection: New Collection with hierarchy of merge events.
    ///
    /// Complexity:
    ///     O(m log m) where m = len(edges)
    ///
    /// Example:
    ///     ```python
    ///     # Basic usage with different key types
    ///     edges = [
    ///         ("cust_123", "cust_456", 0.95),
    ///         (123, 456, 0.85),
    ///         (b"hash1", b"hash2", 0.75)
    ///     ]
    ///     collection = Collection.from_edges(edges)
    ///     
    ///     # Get partition at threshold
    ///     partition = collection.at(0.8)
    ///     print(f"Entities: {len(partition.entities)}")
    ///     ```
    #[classmethod]
    #[pyo3(signature = (edges, *, source=None, progress_callback=None))]
    fn from_edges(
        _cls: &Bound<'_, PyType>,
        edges: Vec<(Py<PyAny>, Py<PyAny>, f64)>,
        source: Option<String>,
        progress_callback: Option<Py<PyAny>>,
        py: Python,
    ) -> PyResult<Self> {
        #[cfg(debug_assertions)]
        let start_time = std::time::Instant::now();

        let source_name = source.unwrap_or_else(|| "default".to_string());

        // Pre-flight safety check using global safety system
        let _num_entities = edges.len() / 5; // Rough estimate: 5 edges per entity on average
                                             // Ensure at least 1MB for small datasets to avoid integer division to 0
        let estimated_mb = ((edges.len() * 60) / (1024 * 1024)).max(1); // 60 bytes per edge estimate (with interning)

        // Global safety check replaces context.resource_monitor
        ensure_memory_safety(estimated_mb as u64).map_err(map_safety_error)?;

        // Pre-calculate capacity based on edge count (assume ~70% unique records)
        let estimated_records = (edges.len() * 14) / 10; // 1.4x edges for safety
        let context = DataContext::with_capacity(estimated_records);

        // Create progress callback wrapper for Rust use
        let progress_callback: Option<ProgressCallback> =
            progress_callback.map(create_progress_wrapper);

        // Report initial progress
        if let Some(ref callback) = progress_callback {
            let thread_count = rayon::current_num_threads();
            callback(
                0.0,
                &format!(
                    "Processing {} edges ({} threads)",
                    edges.len(),
                    thread_count
                ),
            );
        }

        // Efficiently convert all Python keys to Rust edges with optimised bulk processing
        #[cfg(debug_assertions)]
        let conversion_start = std::time::Instant::now();

        // Phase 1: Bulk Python object extraction with deduplication
        #[cfg(debug_assertions)]
        let phase1_start = std::time::Instant::now();

        use rustc_hash::FxHashMap;
        let mut key_to_id: FxHashMap<Key, u32> = FxHashMap::default();
        let mut extracted_edges = Vec::with_capacity(edges.len());

        // Pre-allocate hash map for key lookups
        key_to_id.reserve(estimated_records);

        for (key1_obj, key2_obj, threshold) in edges {
            // Convert Python objects to Rust Keys (bulk extraction with interning)
            let key1 = python_obj_to_key_fast(key1_obj, py, &context)?;
            let key2 = python_obj_to_key_fast(key2_obj, py, &context)?;

            extracted_edges.push((key1, key2, threshold));
        }

        #[cfg(debug_assertions)]
        let phase1_time = phase1_start.elapsed();

        // Report progress after Python conversion
        if let Some(ref callback) = progress_callback {
            callback(0.2, "Converted Python objects to Rust");
        }

        // Phase 2: Parallel batch key registration with rayon
        #[cfg(debug_assertions)]
        let phase2_start = std::time::Instant::now();

        use rayon::prelude::*;
        use std::sync::Mutex;

        // Collect unique keys in parallel
        let unique_keys: Vec<Key> = {
            let mut keys_set = FxHashMap::default();
            for (key1, key2, _) in &extracted_edges {
                keys_set.entry(key1.clone()).or_insert(());
                keys_set.entry(key2.clone()).or_insert(());
            }
            keys_set.into_keys().collect()
        };

        // Determine batch size based on memory status
        use starlings_core::core::safety::global_resource_monitor;
        let usage = global_resource_monitor().get_usage();
        let batch_size = if !usage.memory_under_limit {
            10_000 // Small batches when at limit
        } else if usage.memory_percent > 60.0 {
            50_000 // Medium batches when getting close
        } else {
            100_000 // Large batches when plenty of headroom
        };

        let key_to_id_mutex = Mutex::new(key_to_id);

        // Helper closure for batch processing
        let process_batch = |batch: &[Key]| {
            let ids = context.ensure_records_batch(&source_name, batch);
            let mut local_map = FxHashMap::default();
            for (key, id) in batch.iter().zip(ids.iter()) {
                local_map.insert(key.clone(), *id);
            }
            key_to_id_mutex.lock().unwrap().extend(local_map);
        };

        // Normal parallel processing
        unique_keys.par_chunks(batch_size).for_each(process_batch);

        if let Some(ref callback) = progress_callback {
            let msg = format!(
                "Parallel processing: batch size {}, {} threads active",
                batch_size,
                rayon::current_num_threads()
            );
            callback(0.35, &msg);
        }

        let key_to_id = key_to_id_mutex.into_inner().unwrap();

        #[cfg(debug_assertions)]
        let phase2_time = phase2_start.elapsed();

        // Report progress after key registration
        if let Some(ref callback) = progress_callback {
            callback(0.4, "Registered unique keys");
        }

        // Phase 3: Parallel edge ID mapping
        #[cfg(debug_assertions)]
        let phase3_start = std::time::Instant::now();

        // Use parallel iteration to map keys to IDs
        let rust_edges: Vec<(u32, u32, f64)> = extracted_edges
            .par_iter()
            .map(|(key1, key2, threshold)| {
                let id1 = key_to_id[key1];
                let id2 = key_to_id[key2];
                (id1, id2, *threshold)
            })
            .collect();

        #[cfg(debug_assertions)]
        let phase3_time = phase3_start.elapsed();

        #[cfg(debug_assertions)]
        let conversion_time = conversion_start.elapsed();

        // Report progress after ID mapping
        if let Some(ref callback) = progress_callback {
            callback(0.6, "Mapped keys to IDs");
        }

        #[cfg(debug_assertions)]
        let hierarchy_start = std::time::Instant::now();
        #[cfg(debug_assertions)]
        let edge_count = rust_edges.len();
        #[cfg(debug_assertions)]
        let record_count = context.len();
        let hierarchy = PartitionHierarchy::from_edges(
            rust_edges,
            Arc::new(context),
            6,
            progress_callback.clone(),
        )
        .map_err(map_storage_error)?;
        #[cfg(debug_assertions)]
        let hierarchy_time = hierarchy_start.elapsed();

        #[cfg(debug_assertions)]
        let total_time = start_time.elapsed();

        // Report final progress
        if let Some(ref callback) = progress_callback {
            callback(1.0, "Collection created successfully");
        }

        // Production-scale performance metrics (when STARLINGS_DEBUG=1 and large datasets)
        #[cfg(debug_assertions)]
        if edge_count >= 100_000 {
            debug_println!("🏭 Production-scale Collection.from_edges performance:");
            debug_println!(
                "   📊 Scale: {} edges, {} unique records",
                edge_count,
                record_count
            );
            debug_println!("   ⚡ Python->Rust conversion breakdown:");
            debug_println!("      Phase 1 (Python extraction): {:?}", phase1_time);
            debug_println!("      Phase 2 (Key interning): {:?}", phase2_time);
            debug_println!("      Phase 3 (ID mapping): {:?}", phase3_time);
            debug_println!("      Total conversion: {:?}", conversion_time);
            debug_println!("   🏗️  Hierarchy construction: {:?}", hierarchy_time);
            debug_println!("   📈 Total time: {:?}", total_time);
            debug_println!(
                "   🎯 Edges per second: {:.0}",
                edge_count as f64 / total_time.as_secs_f64()
            );
            debug_println!(
                "   📍 Time breakdown: {:.1}% conversion, {:.1}% hierarchy",
                (conversion_time.as_secs_f64() / total_time.as_secs_f64()) * 100.0,
                (hierarchy_time.as_secs_f64() / total_time.as_secs_f64()) * 100.0
            );
            if edge_count >= 1_000_000 {
                debug_println!(
                    "   🏆 1M edges <10s target: {}",
                    if total_time.as_secs_f64() < 10.0 {
                        "✅ ACHIEVED"
                    } else {
                        "❌ MISSED"
                    }
                );
            }
        }

        Ok(PyCollection {
            hierarchy,
            is_view: false,
        })
    }

    /// Get partition at specific threshold.
    ///
    /// Returns a Partition containing all entities that exist at the specified
    /// similarity threshold. The first call at a threshold reconstructs the partition
    /// from merge events (O(m)), while subsequent calls use cached results (O(1)).
    ///
    /// Args:
    ///     threshold (float): Threshold value between 0.0 and 1.0. Records with
    ///         similarity >= threshold will be merged into the same entity.
    ///
    /// Returns:
    ///     Partition: Partition object with entities at the specified threshold.
    ///
    /// Complexity:
    ///     First call at threshold: O(m) reconstruction
    ///     Subsequent calls: O(1) from cache
    ///
    /// Example:
    ///     ```python
    ///     collection = Collection.from_edges(edges)
    ///     
    ///     # Get partition at different thresholds
    ///     partition_low = collection.at(0.5)   # More, smaller entities
    ///     partition_high = collection.at(0.9)  # Fewer, larger entities
    ///     
    ///     print(f"At 0.5: {len(partition_low.entities)} entities")
    ///     print(f"At 0.9: {len(partition_high.entities)} entities")
    ///     ```
    fn at(&mut self, threshold: f64) -> PyResult<PyPartition> {
        let partition = self.hierarchy.at_threshold(threshold);
        Ok(PyPartition { partition })
    }

    /// Create a deep copy of this collection with independent context.
    ///
    /// Creates a new collection that is completely independent of the original,
    /// allowing modifications without affecting the original collection.
    fn copy(&self) -> PyResult<PyCollection> {
        let cloned_hierarchy = self.hierarchy.clone();
        Ok(PyCollection {
            hierarchy: cloned_hierarchy,
            is_view: false,
        })
    }

    /// Check if this collection is an immutable view from a frame.
    ///
    /// Returns:
    ///     bool: True if this is a view, False if it's an owned collection.
    fn is_view(&self) -> bool {
        self.is_view
    }

    /// String representation for debugging.
    fn __repr__(&self) -> String {
        "Collection".to_string()
    }
}

/// Generate entity resolution edges using the unified constructive algorithm.
///
/// Creates exactly n*5 edges that produce n entities at threshold 1.0 and n/2 entities
/// at threshold 0.0, following realistic entity resolution patterns.
///
/// Args:
///     n (int): Number of entities at threshold 1.0
///     num_thresholds (Optional[int]): If provided, snap to discrete thresholds;
///         if None, add jitter for PGO training
///     batch_size (int): Size of each batch yielded by the generator (default 100_000)
///
/// Returns:
///     EdgeGenerator: Generator that yields batches of (entity1, entity2, threshold) tuples
///
/// Example:
///     ```python
///     # Generate 1M entity dataset as a generator
///     edge_gen = generate_entity_resolution_edges(1_000_000)
///     
///     # Use with Collection.from_edges (handles generators automatically)
///     collection = Collection.from_edges(edge_gen)
///     ```
#[pyfunction]
#[pyo3(signature = (n, num_thresholds=None, batch_size=100_000))]
fn generate_entity_resolution_edges(
    n: usize,
    num_thresholds: Option<usize>,
    batch_size: usize,
    _py: Python<'_>,
) -> PyResult<EdgeGenerator> {
    // CRITICAL FIX: Add pre-flight safety check BEFORE generating edges
    // Ensure at least 1MB for small datasets to avoid integer division to 0
    let estimated_mb = ((n * 5 * 100) / (1024 * 1024)).max(1); // n entities * 5 edges * 100 bytes per edge
    ensure_memory_safety(estimated_mb as u64).map_err(map_safety_error)?;

    let edges = test_utils::generate_entity_resolution_edges(n, num_thresholds);

    let python_edges: Vec<(i64, i64, f64)> = edges
        .into_iter()
        .map(|(id1, id2, weight)| (id1 as i64, id2 as i64, weight))
        .collect();

    Ok(EdgeGenerator {
        edges: python_edges,
        batch_size,
        current_index: 0,
    })
}

/// Convert Python object to Rust Key (optimised for performance)
fn python_obj_to_key_fast(obj: Py<PyAny>, py: Python, context: &DataContext) -> PyResult<Key> {
    // Try integer types first (most common in large datasets)
    if let Ok(i) = obj.extract::<i64>(py) {
        if i < 0 {
            Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Integer key must be non-negative and fit in u64",
            ))
        } else if i <= u32::MAX as i64 {
            Ok(Key::U32(i as u32))
        } else {
            Ok(Key::U64(i as u64))
        }
    } else if let Ok(s) = obj.downcast_bound::<PyString>(py) {
        // CRITICAL: Intern the string instead of allocating
        let string_value = s.to_str()?;
        let interned_id = context.intern_string(string_value);
        Ok(Key::InternedString(interned_id))
    } else if let Ok(b) = obj.downcast_bound::<PyBytes>(py) {
        Ok(Key::Bytes(b.as_bytes().to_vec()))
    } else {
        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "Key must be str, bytes, or int",
        ))
    }
}

/// Generate all combinations of thresholds for cartesian product analysis
fn generate_threshold_combinations(
    expressions: &[ExpressionType],
) -> PyResult<Vec<Vec<(ExpressionType, f64)>>> {
    let mut threshold_sets = Vec::new();

    for expr in expressions {
        let thresholds = match expr {
            ExpressionType::Point { threshold, .. } => vec![*threshold],
            ExpressionType::Sweep {
                start, stop, step, ..
            } => generate_sweep_thresholds(*start, *stop, *step),
        };
        threshold_sets.push(thresholds);
    }

    // Generate cartesian product of all threshold combinations
    let mut combinations = vec![vec![]];

    for (expr_idx, thresholds) in threshold_sets.iter().enumerate() {
        let mut new_combinations = Vec::new();

        for combination in &combinations {
            for &threshold in thresholds {
                let mut new_combination = combination.clone();
                new_combination.push((expressions[expr_idx].clone(), threshold));
                new_combinations.push(new_combination);
            }
        }

        combinations = new_combinations;
    }

    Ok(combinations)
}

/// Get metric name as string
fn metric_name(metric: &MetricType) -> &'static str {
    match metric {
        MetricType::F1 => "f1",
        MetricType::Precision => "precision",
        MetricType::Recall => "recall",
        MetricType::ARI => "ari",
        MetricType::NMI => "nmi",
        MetricType::VMeasure => "v_measure",
        MetricType::BCubedPrecision => "bcubed_precision",
        MetricType::BCubedRecall => "bcubed_recall",
        MetricType::EntityCount => "entity_count",
        MetricType::Entropy => "entropy",
    }
}

/// Multi-collection container that enables hierarchies to share DataContext
#[pyclass(name = "EntityFrame")]
pub struct PyEntityFrame {
    frame: EntityFrame,
    engine: MetricEngine,
}

#[pymethods]
impl PyEntityFrame {
    /// Create a new empty EntityFrame
    #[new]
    fn new() -> Self {
        PyEntityFrame {
            frame: EntityFrame::new(),
            engine: MetricEngine::new(),
        }
    }

    /// Add a collection to the frame
    ///
    /// Args:
    ///     name (str): Name for the collection
    ///     collection (Collection): Collection to add
    ///
    /// Note: For now, this creates a new hierarchy from the collection's data.
    /// Future versions will support proper memory sharing.
    fn add_collection(&mut self, name: String, collection: &PyCollection) -> PyResult<()> {
        // Clone the hierarchy so we can add it to the frame
        let hierarchy_clone = collection.hierarchy.clone();
        self.frame
            .add_collection(name, hierarchy_clone)
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Failed to add collection: {}",
                    e
                ))
            })
    }

    /// Check if a collection exists
    ///
    /// Args:
    ///     name (str): Name of the collection
    ///
    /// Returns:
    ///     bool: True if collection exists
    fn has_collection(&self, name: &str) -> bool {
        self.frame.get_collection(name).is_some()
    }

    /// Get list of collection names
    ///
    /// Returns:
    ///     List[str]: Names of all collections in the frame
    fn collection_names(&self) -> Vec<String> {
        self.frame.collection_names()
    }

    /// Get number of collections
    ///
    /// Returns:
    ///     int: Number of collections in the frame
    fn __len__(&self) -> usize {
        self.frame.len()
    }

    /// Get a collection by name using dictionary-style access
    ///
    /// Args:
    ///     name (str): Name of the collection
    ///
    /// Returns:
    ///     Collection: The collection as a view (immutable)
    ///
    /// Raises:
    ///     KeyError: If collection doesn't exist
    fn __getitem__(&self, name: String) -> PyResult<PyCollection> {
        if let Some(hierarchy) = self.frame.get_collection(&name) {
            // Return as a view collection
            Ok(PyCollection::new_view(hierarchy.clone()))
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!(
                "Collection '{}' not found",
                name
            )))
        }
    }

    /// Check if a collection exists
    ///
    /// Args:
    ///     name (str): Name of the collection
    ///
    /// Returns:
    ///     bool: True if collection exists
    fn __contains__(&self, name: &str) -> bool {
        self.frame.get_collection(name).is_some()
    }

    /// Remove a collection from the frame
    ///
    /// Args:
    ///     name (str): Name of the collection to remove
    ///
    /// Returns:
    ///     bool: True if collection was removed
    fn remove_collection(&mut self, name: &str) -> bool {
        self.frame.remove_collection(name).is_some()
    }

    /// Universal analysis method using expressions.
    ///
    /// Always returns List[Dict[str, float]] where each dict represents one measurement.
    /// This uniform format works seamlessly with DataFrame libraries.
    ///
    /// Args:
    ///     expressions: Variable number of expression objects from sl.col()
    ///     metrics: List of metric functions to compute
    ///
    /// Returns:
    ///     List[Dict[str, float]]: Results where each dict contains:
    ///     - "{collection}_threshold" for all threshold values
    ///     - Direct metric names ("f1", "precision", "entity_count", etc.)
    ///
    /// Example:
    ///     ```python
    ///     # Point comparison
    ///     result = ef.analyse(
    ///         sl.col("splink").at(0.85),
    ///         sl.col("truth").at(1.0),
    ///         metrics=[sl.Metrics.eval.f1, sl.Metrics.eval.precision]
    ///     )
    ///     # Returns: [{"splink_threshold": 0.85, "truth_threshold": 1.0, "f1": 0.92, ...}]
    ///     ```
    #[pyo3(signature = (*expressions, metrics=None, progress_callback=None))]
    fn analyse(
        &mut self,
        expressions: &Bound<'_, pyo3::types::PyTuple>,
        metrics: Option<Vec<Bound<'_, PyAny>>>,
        progress_callback: Option<Py<PyAny>>,
        py: Python,
    ) -> PyResult<Vec<std::collections::HashMap<String, f64>>> {
        use std::collections::HashMap;

        // Parse expressions from Python
        let mut parsed_expressions = Vec::new();
        for expr in expressions.iter() {
            let parsed = parse_expression(&expr)?;
            parsed_expressions.push(parsed);
        }

        // Determine which collection is the reference for asymmetric metrics
        // This logs info about implicit reference if needed
        let _ = self.determine_reference(&parsed_expressions, py)?;

        // Parse metrics from Python, providing defaults if none specified
        let parsed_metrics = if let Some(metric_objs) = metrics {
            let mut metrics = Vec::new();
            for metric_obj in metric_objs {
                let parsed = parse_metric(&metric_obj)?;
                metrics.push(parsed);
            }
            metrics
        } else {
            // Provide default metrics based on number of expressions
            if parsed_expressions.len() >= 2 {
                // Multiple collections - use comparison metrics
                vec![MetricType::F1, MetricType::Precision, MetricType::Recall]
            } else {
                // Single collection - use statistics metrics
                vec![MetricType::EntityCount, MetricType::Entropy]
            }
        };

        // Report initial progress
        if let Some(ref callback) = progress_callback {
            callback.call1(py, (0.05, "Preparing analysis"))?;
        }

        // OPTIMISED PATH for single-collection sweeps
        if parsed_expressions.len() == 1 {
            if let ExpressionType::Sweep {
                collection,
                start,
                stop,
                step,
                ..
            } = &parsed_expressions[0]
            {
                debug_println!("🚀 Optimised path: single collection sweep");

                let hierarchy = self.frame.get_collection(collection).ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!(
                        "Collection '{}' not found",
                        collection
                    ))
                })?;

                // Thresholds are generated HIGH to LOW for Delta algorithm
                let thresholds = generate_sweep_thresholds(*start, *stop, *step);
                if thresholds.is_empty() {
                    return Ok(vec![]);
                }

                // 1. Build ONLY the first partition (highest threshold)
                let first_partition = hierarchy.at_threshold(thresholds[0]);

                // 2. Get merge events between all subsequent thresholds
                let mut merge_events_between = Vec::new();
                for i in 0..thresholds.len() - 1 {
                    let from = thresholds[i];
                    let to = thresholds[i + 1];
                    let merges = hierarchy
                        .get_merge_events_between(from, to)
                        .map_err(map_storage_error)?;
                    merge_events_between.push(merges);
                }

                // 3. Call the optimised sweep function in the metric engine
                let engine_metrics: Vec<CoreMetricType> =
                    parsed_metrics.iter().map(convert_metric_type).collect();
                let metric_results_vec = self.engine.compute_single_sweep_with_merges(
                    &first_partition,
                    &thresholds,
                    &merge_events_between,
                    &engine_metrics,
                    &hierarchy.context,
                );

                // 4. Format results
                let mut results = Vec::new();
                for (i, threshold) in thresholds.iter().enumerate() {
                    let mut result_map = HashMap::new();
                    result_map.insert(format!("{}_threshold", collection), *threshold);
                    for (metric_name, value) in &metric_results_vec[i] {
                        result_map.insert(metric_name.clone(), *value);
                    }
                    results.push(result_map);
                }

                return Ok(results);
            }
        }

        // Generate all combinations of thresholds for cartesian product
        let threshold_combinations = generate_threshold_combinations(&parsed_expressions)?;

        let mut results = Vec::new();

        // Cache for partitions to avoid redundant reconstruction
        let mut partition_cache: HashMap<(String, u64), Arc<PartitionLevel>> = HashMap::new();

        // Cache for metric results to avoid redundant computation
        let mut metrics_cache: HashMap<(String, u64, String, u64), MetricResults> = HashMap::new();

        // Extract collection names for potential future optimisation
        let _expr_collection_names: Vec<String> = parsed_expressions
            .iter()
            .map(|expr| match expr {
                ExpressionType::Point { collection, .. } => collection.clone(),
                ExpressionType::Sweep { collection, .. } => collection.clone(),
            })
            .collect();

        // Special optimisation for sweep × sweep comparisons
        // This path uses efficient batch partition building and the MetricEngine's sweep methods
        if parsed_expressions.len() == 2
            && matches!(
                (&parsed_expressions[0], &parsed_expressions[1]),
                (ExpressionType::Sweep { .. }, ExpressionType::Sweep { .. })
            )
        {
            // Using optimised sweep × sweep computation path

            // Extract sweep parameters
            let (col1, thresholds1) = match &parsed_expressions[0] {
                ExpressionType::Sweep {
                    collection,
                    start,
                    stop,
                    step,
                    ..
                } => (
                    collection.clone(),
                    generate_sweep_thresholds(*start, *stop, *step),
                ),
                _ => unreachable!("Sweep × sweep case should only have Sweep expressions"),
            };
            let (col2, thresholds2) = match &parsed_expressions[1] {
                ExpressionType::Sweep {
                    collection,
                    start,
                    stop,
                    step,
                    ..
                } => (
                    collection.clone(),
                    generate_sweep_thresholds(*start, *stop, *step),
                ),
                _ => unreachable!("Sweep × sweep case should only have Sweep expressions"),
            };

            // Report progress for partition building
            if let Some(ref callback) = progress_callback {
                callback.call1(py, (0.1, "Building partitions"))?;
            }

            // Build all partitions upfront
            let partitions1 = self.build_partitions_for_thresholds(&col1, &thresholds1)?;
            let partitions2 = self.build_partitions_for_thresholds(&col2, &thresholds2)?;

            if let Some(ref callback) = progress_callback {
                callback.call1(py, (0.4, "Partitions built"))?;
            }

            // Get shared context
            let context = if let Some(hierarchy) = self.frame.get_collection(&col1) {
                hierarchy.context.clone()
            } else {
                return Err(PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!(
                    "Collection '{}' not found",
                    col1
                )));
            };

            // Use MetricEngine for efficient computation
            if let Some(ref callback) = progress_callback {
                callback.call1(py, (0.5, "Computing metrics"))?;
            }

            // Convert expressions metrics to engine metrics
            let engine_metrics: Vec<CoreMetricType> = parsed_metrics
                .iter()
                .map(|m| match m {
                    MetricType::F1 => CoreMetricType::F1,
                    MetricType::Precision => CoreMetricType::Precision,
                    MetricType::Recall => CoreMetricType::Recall,
                    MetricType::ARI => CoreMetricType::ARI,
                    MetricType::NMI => CoreMetricType::NMI,
                    MetricType::VMeasure => CoreMetricType::VMeasure,
                    MetricType::BCubedPrecision => CoreMetricType::BCubedPrecision,
                    MetricType::BCubedRecall => CoreMetricType::BCubedRecall,
                    MetricType::EntityCount => CoreMetricType::EntityCount,
                    MetricType::Entropy => CoreMetricType::Entropy,
                })
                .collect();

            // Determine if this is a same-collection comparison
            let same_collection = col1 == col2;

            // Use MetricEngine to compute all metrics efficiently with Arc support
            let metric_results_vec = if partitions1.len() == 1 && partitions2.len() == 1 {
                // Single comparison - use compute_single_arc directly
                vec![self.engine.compute_single_arc(
                    &partitions1[0],
                    Some(&partitions2[0]),
                    &engine_metrics,
                    &context,
                    same_collection,
                )]
            } else {
                // Sweep comparison - use the new Arc-aware method
                self.engine.compute_sweep_arc(
                    &partitions1,
                    Some(&partitions2),
                    &engine_metrics,
                    &context,
                    same_collection,
                )
            };

            if let Some(ref callback) = progress_callback {
                callback.call1(py, (0.7, "Processing results"))?;
            }

            // Convert results to the expected format
            let total_combinations = thresholds1.len() * thresholds2.len();
            let mut completed = 0;
            let mut result_idx = 0;
            for threshold1 in thresholds1.iter() {
                for threshold2 in thresholds2.iter() {
                    let mut result = HashMap::new();
                    result.insert(format!("{}_threshold", col1), *threshold1);
                    result.insert(format!("{}_threshold", col2), *threshold2);

                    // Add all metrics from the engine results
                    for (metric_name, value) in &metric_results_vec[result_idx] {
                        result.insert(metric_name.clone(), *value);
                    }
                    result_idx += 1;

                    results.push(result);

                    // Update progress
                    completed += 1;
                    if let Some(ref callback) = progress_callback {
                        let progress = 0.7 + (0.3 * completed as f64 / total_combinations as f64);
                        callback.call1(
                            py,
                            (
                                progress,
                                format!(
                                    "Processing combination {}/{}",
                                    completed, total_combinations
                                ),
                            ),
                        )?;
                    }
                }
            }

            // Report completion
            if let Some(ref callback) = progress_callback {
                callback.call1(py, (1.0, "Analysis complete"))?;
            }

            return Ok(results);
        }

        // General path for all other cases (point comparisons, mixed sweep/point, etc.)
        let total_combinations = threshold_combinations.len();
        let mut combination_idx = 0;

        for combination in threshold_combinations {
            let mut result = HashMap::new();

            // Add threshold values to result
            for (expr, threshold) in &combination {
                let collection_name = match expr {
                    ExpressionType::Point { collection, .. } => collection,
                    ExpressionType::Sweep { collection, .. } => collection,
                };
                result.insert(format!("{}_threshold", collection_name), *threshold);
            }

            // Get partitions for this combination, using cache when possible
            let mut owned_partitions: Vec<Arc<PartitionLevel>> = Vec::new();
            let mut collection_names = Vec::new();

            // Build partitions efficiently using cache
            for (expr, threshold) in &combination {
                let collection_name = match expr {
                    ExpressionType::Point { collection, .. } => collection,
                    ExpressionType::Sweep { collection, .. } => collection,
                };
                collection_names.push(collection_name.clone());

                // Create cache key using fixed-point representation
                let threshold_key = threshold_to_cache_key(*threshold);
                let cache_key = (collection_name.clone(), threshold_key);

                let partition = if let Some(cached) = partition_cache.get(&cache_key) {
                    // Use cached partition
                    cached.clone()
                } else {
                    // Get hierarchy and build partition
                    let hierarchy =
                        self.frame.get_collection(collection_name).ok_or_else(|| {
                            PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!(
                                "Collection '{}' not found",
                                collection_name
                            ))
                        })?;

                    let partition = hierarchy.at_threshold(*threshold);
                    partition_cache.insert(cache_key, partition.clone());
                    partition
                };
                owned_partitions.push(partition);
            }

            // Compute comparison metrics efficiently (all at once to avoid cache issues)
            let comparison_metrics: Vec<&MetricType> = parsed_metrics
                .iter()
                .filter(|m| m.requires_comparison())
                .collect();

            let mut metric_values = HashMap::new();

            if !comparison_metrics.is_empty() {
                if owned_partitions.len() < 2 {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "Comparison metrics require at least 2 collections".to_string(),
                    ));
                }

                // Check contingency table cache
                let threshold_key1 = threshold_to_cache_key(combination[0].1);
                let threshold_key2 = threshold_to_cache_key(combination[1].1);
                let cont_cache_key = (
                    collection_names[0].clone(),
                    threshold_key1,
                    collection_names[1].clone(),
                    threshold_key2,
                );

                // Check cache first
                let cached_results = if let Some(cached) = metrics_cache.get(&cont_cache_key) {
                    cached.clone()
                } else {
                    // Get shared context
                    let context =
                        if let Some(hierarchy) = self.frame.get_collection(&collection_names[0]) {
                            hierarchy.context.clone()
                        } else {
                            return Err(PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!(
                                "Collection '{}' not found",
                                collection_names[0]
                            )));
                        };

                    // Convert all comparison metrics to engine metric types
                    let engine_metrics: Vec<CoreMetricType> = comparison_metrics
                        .iter()
                        .map(|m| match m {
                            MetricType::F1 => CoreMetricType::F1,
                            MetricType::Precision => CoreMetricType::Precision,
                            MetricType::Recall => CoreMetricType::Recall,
                            MetricType::ARI => CoreMetricType::ARI,
                            MetricType::NMI => CoreMetricType::NMI,
                            MetricType::VMeasure => CoreMetricType::VMeasure,
                            MetricType::BCubedPrecision => CoreMetricType::BCubedPrecision,
                            MetricType::BCubedRecall => CoreMetricType::BCubedRecall,
                            _ => CoreMetricType::F1, // Shouldn't happen
                        })
                        .collect();

                    // Compute all metrics at once using Arc method
                    let same_collection = collection_names[0] == collection_names[1];
                    let metric_results = self.engine.compute_single_arc(
                        &owned_partitions[0],
                        Some(&owned_partitions[1]),
                        &engine_metrics,
                        &context,
                        same_collection,
                    );

                    // Cache the results
                    metrics_cache.insert(cont_cache_key.clone(), metric_results.clone());

                    metric_results
                };

                // Extract values for comparison metrics
                for metric in &comparison_metrics {
                    let value = cached_results
                        .get(metric_name(metric))
                        .copied()
                        .unwrap_or(0.0);
                    metric_values.insert(metric_name(metric), value);
                }
            }

            // Now add all metric values to the result
            for metric in &parsed_metrics {
                let metric_value = if metric.requires_comparison() {
                    metric_values
                        .get(metric_name(metric))
                        .copied()
                        .unwrap_or(0.0)
                } else {
                    if owned_partitions.is_empty() {
                        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                            "No collections specified for metric computation".to_string(),
                        ));
                    }
                    // Use first partition for single-collection metrics
                    compute_single_metric(&owned_partitions[0], metric).map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string())
                    })?
                };

                result.insert(metric_name(metric).to_string(), metric_value);
            }

            results.push(result);

            // Update progress
            combination_idx += 1;
            if let Some(ref callback) = progress_callback {
                let progress = 0.1 + (0.9 * combination_idx as f64 / total_combinations as f64);
                callback.call1(
                    py,
                    (
                        progress,
                        format!(
                            "Processing combination {}/{}",
                            combination_idx, total_combinations
                        ),
                    ),
                )?;
            }

            // Progressive cache management based on collection sizes
            // Keep more cache for smaller collections, less for larger ones
            let max_entities = owned_partitions
                .iter()
                .map(|p| p.entities().len())
                .max()
                .unwrap_or(0);

            let (partition_limit, contingency_limit) = if max_entities > 100_000 {
                // Very large collections: minimal cache
                (10, 5)
            } else if max_entities > 10_000 {
                // Large collections: moderate cache
                (20, 10)
            } else if max_entities > 1_000 {
                // Medium collections: good cache
                (50, 25)
            } else {
                // Small collections: maximum cache
                (100, 50)
            };

            // Clear caches if they exceed adaptive limits
            if partition_cache.len() > partition_limit {
                // Simple strategy: clear half the cache when limit exceeded
                partition_cache.clear();
            }
            if metrics_cache.len() > contingency_limit {
                // Simple strategy: clear half the cache when limit exceeded
                metrics_cache.clear();
            }
        }

        // Report completion
        if let Some(ref callback) = progress_callback {
            callback.call1(py, (1.0, "Analysis complete"))?;
        }

        Ok(results)
    }

    /// String representation
    fn __repr__(&self) -> String {
        format!("EntityFrame(collections={})", self.frame.len())
    }
}

impl PyEntityFrame {
    /// Helper method to log info messages via Python's logging module
    fn log_info(&self, py: Python, message: &str) {
        use pyo3::types::PyModule;

        // Best effort logging - ignore failures silently
        if let Ok(logging) = PyModule::import(py, "logging") {
            if let Ok(logger) = logging.getattr("getLogger").and_then(|f| f.call0()) {
                let _ = logger.call_method1("info", (message,));
            }
        }
    }

    /// Helper function to determine which collection is the reference for metrics
    fn determine_reference(
        &self,
        expressions: &[ExpressionType],
        py: Python,
    ) -> PyResult<Option<String>> {
        // Check for explicit reference marking
        let explicit_refs: Vec<_> = expressions
            .iter()
            .filter_map(|expr| {
                let (collection, is_reference) = match expr {
                    ExpressionType::Point {
                        collection,
                        is_reference,
                        ..
                    } => (collection, is_reference),
                    ExpressionType::Sweep {
                        collection,
                        is_reference,
                        ..
                    } => (collection, is_reference),
                };

                if *is_reference {
                    Some(collection.clone())
                } else {
                    None
                }
            })
            .collect();

        // Check for multiple references (error condition)
        if explicit_refs.len() > 1 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Multiple collections marked as reference. Only one reference is allowed.",
            ));
        }

        // Return explicit reference if found
        if !explicit_refs.is_empty() {
            return Ok(Some(explicit_refs[0].clone()));
        }

        // If no explicit reference and we have multiple collections, use implicit (last expression)
        if expressions.len() >= 2 {
            let last_collection = match &expressions[expressions.len() - 1] {
                ExpressionType::Point { collection, .. }
                | ExpressionType::Sweep { collection, .. } => collection.clone(),
            };

            // Log that we're using implicit reference
            self.log_info(
                py,
                &format!(
                    "No explicit reference specified. Using '{}' (last expression) as implicit reference for asymmetric metrics.",
                    last_collection
                ),
            );

            return Ok(Some(last_collection));
        }

        // Single collection or no collections - no reference needed
        Ok(None)
    }

    /// Helper: Build partitions for a collection at multiple thresholds
    fn build_partitions_for_thresholds(
        &self,
        collection_name: &str,
        thresholds: &[f64],
    ) -> PyResult<Vec<Arc<PartitionLevel>>> {
        #[cfg(debug_assertions)]
        use starlings_core::debug_println;

        #[cfg(debug_assertions)]
        let start = std::time::Instant::now();

        let hierarchy = self.frame.get_collection(collection_name).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!(
                "Collection '{}' not found",
                collection_name
            ))
        })?;

        #[cfg(debug_assertions)]
        debug_println!(
            "      🔧 Building {} partitions incrementally for {}",
            thresholds.len(),
            collection_name
        );

        // Use incremental building for all partitions
        let partitions = hierarchy.build_partitions_incrementally(thresholds);

        #[cfg(debug_assertions)]
        {
            let total_time = start.elapsed();
            debug_println!(
                "      🔧 Total incremental partition building: {:?} ({} partitions)",
                total_time,
                partitions.len()
            );

            // Log some statistics about the partitions
            for (i, (threshold, partition)) in thresholds.iter().zip(&partitions).enumerate() {
                debug_println!(
                    "         Partition {} at {:.2}: {} entities, {} records",
                    i + 1,
                    threshold,
                    partition.entities().len(),
                    partition.total_records()
                );
            }
        }

        Ok(partitions)
    }
}

/// Python module definition
#[pymodule]
fn starlings(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Configure rayon thread pool to be a better system neighbour
    // Leave 2 cores free for system tasks (or 1 on small systems)
    let num_cpus = num_cpus::get();
    let thread_count = if num_cpus <= 4 {
        // For small systems (≤4 cores), leave 1 core free
        (num_cpus - 1).max(1)
    } else {
        // For larger systems, leave 2 cores free
        (num_cpus - 2).max(1)
    };

    // Initialise the global thread pool once at module import
    rayon::ThreadPoolBuilder::new()
        .num_threads(thread_count)
        .thread_name(|i| format!("starlings-{}", i))
        .build_global()
        .unwrap_or_else(|e| {
            // If we can't set the global pool (e.g., already set), just log and continue
            debug_println!(
                "Note: Could not configure thread pool ({}), using defaults",
                e
            );
        });

    // Log the configuration for transparency (only when STARLINGS_DEBUG=1)
    debug_println!(
        "🔧 Starlings: Using {} threads (of {} CPUs available)",
        thread_count,
        num_cpus
    );

    m.add_class::<PyCollection>()?;
    m.add_class::<PyPartition>()?;
    m.add_class::<PyEntityFrame>()?;
    m.add_class::<EdgeGenerator>()?;
    m.add_function(wrap_pyfunction!(generate_entity_resolution_edges, m)?)?;
    Ok(())
}
