use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyString, PyType};
use std::sync::Arc;

use starlings_core::core::ensure_memory_safety;
use starlings_core::core::resource_monitor::{AdaptiveLimits, ProcessingStrategy, SafetyError};
use starlings_core::debug_println;
use starlings_core::test_utils;
use starlings_core::{DataContext, Key, PartitionHierarchy, PartitionLevel};

/// Progress callback type for Rust-level progress reporting
type ProgressCallback = Arc<dyn Fn(f64, &str) + Send + Sync>;

/// Helper function to create progress callback wrapper
fn create_progress_wrapper(callback: Py<pyo3::PyAny>) -> ProgressCallback {
    Arc::new(move |progress: f64, message: &str| {
        Python::attach(|py| {
            if let Err(e) = callback.call1(py, (progress, message)) {
                eprintln!("Progress callback error: {}", e);
            }
        });
    })
}

/// Helper function to get strategy message with thread information
fn get_strategy_message(strategy: &ProcessingStrategy) -> String {
    let thread_count = rayon::current_num_threads();
    let base_msg = match strategy {
        ProcessingStrategy::InMemory { .. } => "In-memory processing",
        ProcessingStrategy::MemoryAware { should_spill, .. } => {
            if *should_spill {
                "Memory-aware processing with disk spilling"
            } else {
                "Memory-aware processing"
            }
        }
        ProcessingStrategy::Streaming { .. } => "Streaming with aggressive disk spilling",
        ProcessingStrategy::Insufficient { .. } => {
            unreachable!("Should have been caught earlier")
        }
    };
    format!("{} ({} threads)", base_msg, thread_count)
}

/// Helper function to map storage errors to Python exceptions
fn map_storage_error(error: String) -> PyErr {
    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(error)
}

/// Helper function to map safety errors to Python exceptions
fn map_safety_error(error: SafetyError) -> PyErr {
    match error {
        SafetyError::CircuitOpen(msg) => PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            format!("Circuit breaker tripped: {}", msg),
        ),
        SafetyError::InsufficientResources(msg) => {
            PyErr::new::<pyo3::exceptions::PyMemoryError, _>(format!(
                "Insufficient resources: {}",
                msg
            ))
        }
        SafetyError::OperationTooLarge(msg) => {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Operation too large: {}", msg))
        }
    }
}

/// Extract batch size from any processing strategy
fn extract_batch_size(strategy: &ProcessingStrategy) -> usize {
    match strategy {
        ProcessingStrategy::InMemory { batch_size, .. }
        | ProcessingStrategy::MemoryAware { batch_size, .. }
        | ProcessingStrategy::Streaming { batch_size, .. } => *batch_size,
        ProcessingStrategy::Insufficient { .. } => unreachable!(),
    }
}

/// Check if strategy requires streaming processing
fn requires_streaming(strategy: &ProcessingStrategy) -> bool {
    matches!(strategy, ProcessingStrategy::Streaming { .. })
}

/// Report resource warnings if present
fn report_resource_warnings(
    adaptive_limits: &AdaptiveLimits,
    progress_callback: &Option<ProgressCallback>,
) {
    if let Some(warning) = &adaptive_limits.memory_warning {
        if let Some(ref callback) = progress_callback {
            callback(0.25, warning);
        }
    }
    if let Some(warning) = &adaptive_limits.disk_warning {
        if let Some(ref callback) = progress_callback {
            callback(0.26, warning);
        }
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
    partition: PartitionLevel,
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
        let num_entities = edges.len() / 5; // Rough estimate: 5 edges per entity on average
        let estimated_mb = (edges.len() * 750) / (1024 * 1024); // 750 bytes per edge estimate

        // Global safety check replaces context.resource_monitor
        ensure_memory_safety(estimated_mb as u64).map_err(map_safety_error)?;

        // Pre-calculate capacity based on edge count (assume ~70% unique records)
        let estimated_records = (edges.len() * 14) / 10; // 1.4x edges for safety
        let context = DataContext::with_capacity(estimated_records);

        // Determine processing strategy based on dataset size and system resources
        use starlings_core::core::safety::global_resource_monitor;
        let strategy = global_resource_monitor().determine_processing_strategy(num_entities);

        // Handle insufficient resources case
        if let ProcessingStrategy::Insufficient {
            required_memory_mb,
            available_memory_mb,
            ..
        } = &strategy
        {
            return Err(PyErr::new::<pyo3::exceptions::PyMemoryError, _>(format!(
                "Insufficient system resources for {} entities. Need ~{}MB memory, have {}MB available. \
                 Try: 1) Smaller dataset, 2) Free system memory, 3) Set STARLINGS_SAFETY_LEVEL=performance",
                num_entities, required_memory_mb, available_memory_mb
            )));
        }

        // Create progress callback wrapper for Rust use
        let progress_callback: Option<ProgressCallback> =
            progress_callback.map(create_progress_wrapper);

        // Report initial progress with processing strategy info
        if let Some(ref callback) = progress_callback {
            let strategy_message = get_strategy_message(&strategy);
            callback(0.0, &strategy_message);
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
            // Convert Python objects to Rust Keys (bulk extraction)
            let key1 = python_obj_to_key_fast(key1_obj, py)?;
            let key2 = python_obj_to_key_fast(key2_obj, py)?;

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

        // Extract batch size and processing parameters from strategy
        let batch_size = extract_batch_size(&strategy);

        // Get adaptive limits for current conditions
        let adaptive_limits = global_resource_monitor().get_adaptive_limits(batch_size);

        // Show resource warnings if present
        report_resource_warnings(&adaptive_limits, &progress_callback);

        let key_to_id_mutex = Mutex::new(key_to_id);

        // Helper closure for batch processing with safety checks
        let process_batch = |batch: &[Key]| {
            // Check circuit breaker before each batch
            if global_resource_monitor().is_circuit_open() {
                eprintln!("⚠️  Circuit breaker open - aborting batch processing");
                return;
            }

            // Apply throttling if needed
            let throttle_delay = global_resource_monitor().throttle_if_needed();
            if throttle_delay.as_millis() > 0 {
                std::thread::sleep(throttle_delay);
            }

            let ids = context.ensure_records_batch(&source_name, batch);
            let mut local_map = FxHashMap::default();
            for (key, id) in batch.iter().zip(ids.iter()) {
                local_map.insert(key.clone(), *id);
            }
            key_to_id_mutex.lock().unwrap().extend(local_map);
        };

        // Process batches based on strategy and current system pressure
        let should_use_streaming = requires_streaming(&strategy)
            || adaptive_limits.should_throttle
            || adaptive_limits.should_spill_to_disk;

        if should_use_streaming {
            // Sequential streaming processing with memory management
            let delay_msg = if adaptive_limits.delay_between_batches_ms > 0 {
                format!(
                    " ({}ms delay between batches)",
                    adaptive_limits.delay_between_batches_ms
                )
            } else {
                String::new()
            };

            for batch in unique_keys.chunks(adaptive_limits.batch_size) {
                process_batch(batch);

                // Add delay if under resource pressure
                if adaptive_limits.delay_between_batches_ms > 0 {
                    std::thread::sleep(std::time::Duration::from_millis(
                        adaptive_limits.delay_between_batches_ms,
                    ));
                }
            }

            if let Some(ref callback) = progress_callback {
                let msg = format!(
                    "Streaming mode: batch size {}, resource throttling active{}",
                    adaptive_limits.batch_size, delay_msg
                );
                callback(0.35, &msg);
            }
        } else {
            // Normal parallel processing when resources are abundant
            unique_keys.par_chunks(batch_size).for_each(process_batch);

            if let Some(ref callback) = progress_callback {
                let msg = format!(
                    "Parallel processing: batch size {}, {} threads active",
                    batch_size,
                    rayon::current_num_threads()
                );
                callback(0.35, &msg);
            }
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

        Ok(PyCollection { hierarchy })
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
        Ok(PyPartition {
            partition: partition.clone(),
        })
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
    let estimated_mb = (n * 5 * 150) / (1024 * 1024); // n entities * 5 edges * 150 bytes per edge
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
fn python_obj_to_key_fast(obj: Py<PyAny>, py: Python) -> PyResult<Key> {
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
        Ok(Key::String(s.to_string()))
    } else if let Ok(b) = obj.downcast_bound::<PyBytes>(py) {
        Ok(Key::Bytes(b.as_bytes().to_vec()))
    } else {
        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "Key must be str, bytes, or int",
        ))
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
    m.add_class::<EdgeGenerator>()?;
    m.add_function(wrap_pyfunction!(generate_entity_resolution_edges, m)?)?;
    Ok(())
}
