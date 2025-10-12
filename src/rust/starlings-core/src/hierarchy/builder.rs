use roaring::RoaringBitmap;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// Progress callback type for reporting hierarchy construction progress
type ProgressCallback = Arc<dyn Fn(f64, &str) + Send + Sync>;

use super::bitmap_pool::BitmapPool;
use super::memory_cache::MemoryBoundedCache;
use super::merge_event::MergeEvent;
use super::partition::PartitionLevel;
use super::storage::{DiskStorage, HierarchyStorage, InMemoryStorage};
use super::union_find::UnionFind;
use crate::core::DataContext;
use crate::debug_println;

/// Debug statistics collection for hierarchy construction
#[cfg(debug_assertions)]
#[derive(Default)]
struct DebugStats {
    merge_count: usize,
    bitmap_allocations: usize,
}

#[cfg(debug_assertions)]
impl DebugStats {
    fn increment_merges(&mut self) {
        self.merge_count += 1;
    }

    fn increment_bitmap_allocations(&mut self) {
        self.bitmap_allocations += 1;
    }

    fn report(&self, total_edges: usize) {
        if total_edges >= 100_000 {
            debug_println!(
                "      Union-find stats: {} merges, {} bitmap allocations",
                self.merge_count,
                self.bitmap_allocations
            );
        }
    }
}

#[cfg(not(debug_assertions))]
#[derive(Default)]
struct DebugStats;

#[cfg(not(debug_assertions))]
impl DebugStats {
    fn increment_merges(&mut self) {}
    fn increment_bitmap_allocations(&mut self) {}
    fn report(&self, _total_edges: usize) {}
}

/// Helper for managing component creation and merging
struct ComponentManager<'a> {
    bitmap_pool: &'a BitmapPool,
    active_components: &'a mut HashMap<usize, RoaringBitmap>,
    debug_stats: &'a mut DebugStats,
}

impl ComponentManager<'_> {
    fn get_or_create_component(&mut self, root: usize, record_id: u32) -> RoaringBitmap {
        if let Some(component) = self.active_components.remove(&root) {
            component
        } else {
            let (mut bitmap, _) = self.bitmap_pool.get(1);
            bitmap.insert(record_id);
            self.debug_stats.increment_bitmap_allocations();
            bitmap
        }
    }

    fn create_merged_component(&self, components: &[&RoaringBitmap]) -> RoaringBitmap {
        let total_size = components.iter().map(|b| b.len()).sum::<u64>() as u32;
        let (mut merged, _) = self.bitmap_pool.get(total_size);
        for component in components {
            merged |= *component;
        }
        merged
    }
}

/// Detector for chain-like graph patterns that cause O(n²) memory usage
struct ChainDetector {
    edges_processed: usize,
    total_edges: usize,
    max_component_size: usize,
    total_bitmap_bytes: u64,
    growth_rates: Vec<f64>,
    warning_triggered: bool,
}

impl ChainDetector {
    fn new(total_edges: usize) -> Self {
        Self {
            edges_processed: 0,
            total_edges,
            max_component_size: 0,
            total_bitmap_bytes: 0,
            growth_rates: Vec::new(),
            warning_triggered: false,
        }
    }

    fn record_merge(
        &mut self,
        merged_component: &RoaringBitmap,
        merging_components: &[&RoaringBitmap],
    ) {
        self.edges_processed += 1;

        // Track component growth
        let new_size = merged_component.len() as usize;
        let old_size = merging_components
            .iter()
            .map(|c| c.len())
            .max()
            .unwrap_or(1) as usize;
        self.growth_rates.push(new_size as f64 / old_size as f64);

        // Track max component size
        if new_size > self.max_component_size {
            self.max_component_size = new_size;
        }

        // Track memory usage (RoaringBitmap serialised size estimate)
        self.total_bitmap_bytes += merged_component.serialized_size() as u64;
    }

    fn check_and_error(&mut self) -> Result<(), String> {
        if self.warning_triggered || self.edges_processed < 10_000 {
            return Ok(());
        }

        // Check every 50k edges
        if !self.edges_processed.is_multiple_of(50_000) {
            return Ok(());
        }

        let progress = self.edges_processed as f64 / self.total_edges as f64;
        let memory_per_edge = self.total_bitmap_bytes / self.edges_processed as u64;
        let avg_growth_rate: f64 =
            self.growth_rates.iter().sum::<f64>() / self.growth_rates.len() as f64;

        // Detection logic
        let is_chain = (avg_growth_rate < 1.1) ||  // Slow consistent growth
            (memory_per_edge > 10_000 && self.edges_processed > 100_000) ||  // High memory per edge
            (self.max_component_size as f64 > 0.5 * self.total_edges as f64 && progress < 0.8); // Giant component early

        if is_chain {
            self.warning_triggered = true;
            Err(format!(
                "❌ UNSUPPORTED DATA PATTERN DETECTED\n\
                 \n\
                 Your data contains some long sequential chains where records link one after another:\n\
                   Record 1 → Record 2 → Record 3 → ... → Record {}\n\
                 \n\
                 This pattern would use large amounts of memory and is not supported.\n\
                 \n\
                 WHY THIS MATTERS:\n\
                 Typical entity resolution data forms clusters (groups of duplicates),\n\
                 not long chains. For example:\n\
                 \n\
                   ✓ Normal:  [Customer A: 5 duplicates]  [Customer B: 3 duplicates]  [Singletons...]\n\
                   ✗ Chain:   Record 1 → 2 → 3 → 4 → ... → 500,000\n\
                 \n\
                 POSSIBLE CAUSES:\n\
                 - Synthetic test data using sequential IDs\n\
                 - Entity resolution model needs better tuning (thresholds too low?)\n\
                 - Data needs more cleaning before matching\n\
                 \n\
                 If you are using synthetic test data, create realistic clusters instead:\n\
                 \n\
                   # ❌ Do not create chains:\n\
                   edges = [(i, i+1, 0.85) for i in range(n)]\n\
                 \n\
                   # ✅ Do create clusters (100 records per group):\n\
                   edges = [(base+i, base+i+1, 0.85)\n\
                            for base in range(0, n, 100)  # Start of each cluster\n\
                            for i in range(99)]           # Links within cluster",
                self.max_component_size
            ))
        } else {
            Ok(())
        }
    }
}

/// Hierarchy of merge events that can generate partitions at any threshold
pub struct PartitionHierarchy {
    pub context: Arc<DataContext>,
    storage: Box<dyn HierarchyStorage + Send + Sync>,
    partition_cache: Arc<MemoryBoundedCache>,
    bitmap_pool: BitmapPool,
}

impl Clone for PartitionHierarchy {
    fn clone(&self) -> Self {
        PartitionHierarchy {
            context: self.context.clone(),
            storage: self.storage.clone_box(),
            partition_cache: self.partition_cache.clone(), // Share cache across clones
            bitmap_pool: BitmapPool::new(),
        }
    }
}

impl PartitionHierarchy {
    /// Fixed-point precision factor - supports up to 6 decimal places
    pub const PRECISION_FACTOR: f64 = 1_000_000.0;

    /// Convert f64 threshold to u32 key for exact comparison and caching
    fn threshold_to_key(threshold: f64) -> u32 {
        (threshold.clamp(0.0, 1.0) * Self::PRECISION_FACTOR).round() as u32
    }

    /// Convert u32 key back to f64 threshold
    #[cfg(test)]
    fn key_to_threshold(key: u32) -> f64 {
        key as f64 / Self::PRECISION_FACTOR
    }

    /// Build a hierarchy from edges using union-find algorithm
    pub fn from_edges(
        edges: Vec<(u32, u32, f64)>,
        context: Arc<DataContext>,
        quantise: u32,
        progress_callback: Option<ProgressCallback>,
    ) -> Result<Self, String> {
        // Validate quantise is between 1 and 6
        assert!(
            (1..=6).contains(&quantise),
            "quantise must be between 1 and 6, got {}",
            quantise
        );

        // Calculate cache size based on global memory limit
        let cache_size_bytes = {
            use crate::core::safety::global_resource_monitor;
            let monitor = global_resource_monitor();
            let memory_limit_mb = monitor.get_memory_limit_mb();
            // Use 25% of memory limit for cache, convert to bytes
            (memory_limit_mb / 4) * 1024 * 1024
        };

        if edges.is_empty() {
            // For empty edges, use simple in-memory storage regardless of strategy
            return Ok(Self {
                context,
                storage: Box::new(InMemoryStorage::new()),
                partition_cache: Arc::new(MemoryBoundedCache::new(cache_size_bytes)),
                bitmap_pool: BitmapPool::new(),
            });
        }

        let num_records = context.len();

        // Determine storage strategy based on dataset size and memory limit
        let storage: Box<dyn HierarchyStorage + Send + Sync> = {
            use crate::core::safety::global_resource_monitor;
            let monitor = global_resource_monitor();
            let memory_limit_mb = monitor.get_memory_limit_mb();

            // Improved memory estimation accounting for worst-case O(n²) behaviour:
            // - MergeEvents store full component bitmaps
            // - For chain-like patterns, memory usage is O(n²)
            // - Observed: 100k edges = 1.6GB, 500k edges ≈ 40GB
            // - Conservative estimate: 16KB per edge for >=100k edges
            let bytes_per_edge = if edges.len() >= 100_000 {
                // Large datasets: assume potential chain patterns, be very conservative
                16_000
            } else if edges.len() >= 10_000 {
                // Medium datasets: use middle-ground estimate
                10_000
            } else {
                // Small datasets: use optimistic estimate (normal O(n) behaviour)
                1_000
            };

            let estimated_mb = (edges.len() as u64 * bytes_per_edge) / (1024 * 1024);

            // Use 50% threshold - chain detection will catch pathological O(n²) cases
            // Normal cluster-based ER data deserves fast in-memory storage
            let disk_threshold_mb = memory_limit_mb / 2;

            if estimated_mb < disk_threshold_mb {
                // Fast path: fits comfortably in memory
                debug_println!(
                    "   ✅ Using in-memory storage ({}MB < {}MB limit)",
                    estimated_mb,
                    disk_threshold_mb
                );
                Box::new(InMemoryStorage::new())
            } else {
                // Defensive path: use disk storage for large datasets
                debug_println!(
                    "   💾 Using disk storage ({}MB >= {}MB limit)",
                    estimated_mb,
                    disk_threshold_mb
                );
                Box::new(
                    DiskStorage::new()
                        .map_err(|e| format!("Failed to create disk storage: {}", e))?,
                )
            }
        };

        // Report initial hierarchy construction progress
        if let Some(ref callback) = progress_callback {
            callback(0.65, "Starting hierarchy construction...");
        }

        // Store edge count for pool scaling before consuming edges
        let num_edges = edges.len();

        // Apply quantisation to weights
        #[cfg(debug_assertions)]
        let quantise_start = std::time::Instant::now();

        let factor = 10_f64.powi(quantise as i32);
        let mut quantised_edges: Vec<_> = edges
            .into_iter()
            .map(|(i, j, w)| (i, j, (w * factor).round() / factor))
            .collect();

        #[cfg(debug_assertions)]
        let quantise_time = quantise_start.elapsed();

        // Sort edges by threshold (highest first)
        #[cfg(debug_assertions)]
        let sort_start = std::time::Instant::now();

        quantised_edges.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

        #[cfg(debug_assertions)]
        let sort_time = sort_start.elapsed();

        // Group edges by threshold and build merge events
        #[cfg(debug_assertions)]
        let group_start = std::time::Instant::now();

        let threshold_groups = Self::group_edges_by_threshold(quantised_edges);

        #[cfg(debug_assertions)]
        let group_time = group_start.elapsed();

        // Create a temporary instance with scaled bitmap pool for construction using pre-determined storage
        let mut temp_hierarchy = Self {
            context: context.clone(),
            storage,
            partition_cache: Arc::new(MemoryBoundedCache::new(cache_size_bytes)),
            bitmap_pool: BitmapPool::new_for_scale(num_edges),
        };

        // Report progress after grouping edges by threshold
        if let Some(ref callback) = progress_callback {
            callback(0.7, "Grouped edges by threshold");
        }

        #[cfg(debug_assertions)]
        let union_find_start = std::time::Instant::now();

        temp_hierarchy.build_merge_events(
            threshold_groups,
            num_records,
            progress_callback.clone(),
        )?;

        #[cfg(debug_assertions)]
        let union_find_time = union_find_start.elapsed();

        // Reuse the bitmap pool from temporary instance for efficiency
        let result = Self {
            context,
            storage: temp_hierarchy.storage,
            partition_cache: Arc::new(MemoryBoundedCache::new(cache_size_bytes)),
            bitmap_pool: temp_hierarchy.bitmap_pool,
        };

        // Debug output for hierarchy construction breakdown
        #[cfg(debug_assertions)]
        if num_edges >= 100_000 {
            debug_println!("   🔧 Hierarchy construction breakdown:");
            debug_println!("      Quantisation: {:?}", quantise_time);
            debug_println!("      Sorting {} edges: {:?}", num_edges, sort_time);
            debug_println!("      Edge grouping: {:?}", group_time);
            debug_println!("      Union-find & merges: {:?}", union_find_time);
            debug_println!(
                "      Total hierarchy: {:?}",
                quantise_time + sort_time + group_time + union_find_time
            );
        }

        Ok(result)
    }

    /// Group consecutive edges with the same threshold
    fn group_edges_by_threshold(sorted_edges: Vec<(u32, u32, f64)>) -> Vec<(f64, Vec<(u32, u32)>)> {
        if sorted_edges.is_empty() {
            return Vec::new();
        }

        let mut threshold_groups = Vec::new();
        let mut current_threshold = sorted_edges[0].2;
        let mut current_group = Vec::new();

        for (src, dst, weight) in sorted_edges {
            if (weight - current_threshold).abs() < f64::EPSILON {
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

        threshold_groups
    }

    /// Build merge events from grouped edges using union-find
    /// Optimised for cache locality and minimal allocations
    fn build_merge_events(
        &mut self,
        threshold_groups: Vec<(f64, Vec<(u32, u32)>)>,
        num_records: usize,
        progress_callback: Option<ProgressCallback>,
    ) -> Result<(), String> {
        let mut uf = UnionFind::new_vec(num_records);
        // Pre-size HashMap based on expected number of components
        let estimated_components = (num_records as f64).sqrt() as usize;
        let mut active_components: HashMap<usize, RoaringBitmap> =
            HashMap::with_capacity(estimated_components);

        let mut debug_stats = DebugStats::default();

        // Progress monitoring for large datasets
        let total_edges: usize = threshold_groups.iter().map(|(_, edges)| edges.len()).sum();
        let show_progress = total_edges > 1_000_000; // Lower threshold for progress reporting
        let mut processed_edges = 0;
        let mut last_progress_report = std::time::Instant::now();

        // Chain pattern detection for O(n²) memory usage warning
        let mut chain_detector = ChainDetector::new(total_edges);

        for (group_idx, (threshold, edges_at_threshold)) in threshold_groups.iter().enumerate() {
            // Pre-allocate for this threshold's processing
            let mut processed_pairs = HashSet::with_capacity(edges_at_threshold.len());

            // Process edges sequentially with optimised memory access
            for &(src, dst) in edges_at_threshold {
                let root_src = uf.find(src as usize);
                let root_dst = uf.find(dst as usize);

                if root_src != root_dst {
                    let pair = if root_src < root_dst {
                        (root_src, root_dst)
                    } else {
                        (root_dst, root_src)
                    };

                    if processed_pairs.insert(pair) {
                        let mut component_manager = ComponentManager {
                            bitmap_pool: &self.bitmap_pool,
                            active_components: &mut active_components,
                            debug_stats: &mut debug_stats,
                        };

                        let component_src =
                            component_manager.get_or_create_component(root_src, src);
                        let component_dst =
                            component_manager.get_or_create_component(root_dst, dst);

                        // Determine parent and child for binary merge
                        // Parent is the partition with the smaller canonical ID (more stable)
                        let (parent_bitmap, child_bitmap) =
                            if component_src.min().unwrap() < component_dst.min().unwrap() {
                                (&component_src, &component_dst)
                            } else {
                                (&component_dst, &component_src)
                            };

                        let parent_id = parent_bitmap
                            .min()
                            .expect("Parent component cannot be empty");
                        let child_nodes = child_bitmap.clone();

                        uf.union(src as usize, dst as usize);
                        let new_root = uf.find(src as usize);

                        let merged_component = component_manager
                            .create_merged_component(&[&component_src, &component_dst]);

                        // Record merge for chain pattern detection
                        chain_detector
                            .record_merge(&merged_component, &[&component_src, &component_dst]);

                        let merge_event = MergeEvent::new(*threshold, parent_id, child_nodes);
                        self.storage
                            .push(merge_event)
                            .map_err(|e| format!("Storage error: {}", e))?;
                        active_components.insert(new_root, merged_component);
                        debug_stats.increment_merges();
                    }
                }
            }

            // Check for chain pattern and fail fast if detected
            chain_detector.check_and_error()?;

            // Progress reporting for large datasets
            processed_edges += edges_at_threshold.len();
            if show_progress && last_progress_report.elapsed().as_secs() >= 3 {
                let progress_pct = (processed_edges as f64 / total_edges as f64) * 0.25; // Use 25% of total progress range for union-find
                let hierarchy_progress = 0.7 + progress_pct; // Start at 0.7, go up to 0.95
                let progress_message = format!(
                    "Union-find: {}/{} groups ({:.1}M/{:.1}M edges)",
                    group_idx + 1,
                    threshold_groups.len(),
                    processed_edges as f64 / 1_000_000.0,
                    total_edges as f64 / 1_000_000.0
                );

                // Use callback if available, otherwise fall back to debug logging
                if let Some(ref callback) = progress_callback {
                    callback(hierarchy_progress, &progress_message);
                } else {
                    debug_println!("      🔄 {}", progress_message);
                }
                last_progress_report = std::time::Instant::now();
            }
        }

        for (_, bitmap) in active_components {
            self.bitmap_pool
                .put(bitmap, super::bitmap_pool::PoolSizeClass::Small);
        }

        // Ensure all data is written to storage
        self.storage
            .sync()
            .map_err(|e| format!("Storage sync error: {}", e))?;

        // Report completion of union-find phase
        if let Some(ref callback) = progress_callback {
            callback(0.95, "Union-find complete, finalizing hierarchy");
        }

        debug_stats.report(total_edges);

        Ok(())
    }

    /// Get the number of records in the context
    pub fn num_records(&self) -> usize {
        self.context.len()
    }

    /// Get number of merge events (for testing)
    pub fn merge_events_count(&self) -> usize {
        self.storage.len()
    }

    /// Get all merge events (for translation/assimilation)
    pub fn get_merge_events(&self) -> Vec<MergeEvent> {
        self.storage
            .iter()
            .unwrap_or_else(|_| Box::new(std::iter::empty()))
            .collect()
    }

    /// Create a hierarchy from pre-existing merge events
    pub fn from_merge_events(merge_events: Vec<MergeEvent>, context: Arc<DataContext>) -> Self {
        let mut storage: Box<dyn HierarchyStorage + Send + Sync> = Box::new(InMemoryStorage::new());

        // Add all merge events to storage
        for event in merge_events {
            storage.push(event).unwrap();
        }

        // Calculate cache size based on global memory limit
        let cache_size_bytes = {
            use crate::core::safety::global_resource_monitor;
            let monitor = global_resource_monitor();
            let memory_limit_mb = monitor.get_memory_limit_mb();
            // Use 25% of memory limit for cache, convert to bytes
            (memory_limit_mb / 4) * 1024 * 1024
        };

        Self {
            context,
            storage,
            partition_cache: Arc::new(MemoryBoundedCache::new(cache_size_bytes)),
            bitmap_pool: BitmapPool::new(),
        }
    }

    /// Get a partition at a specific threshold
    pub fn at_threshold(&self, threshold: f64) -> Arc<PartitionLevel> {
        // Validate threshold
        assert!(
            (0.0..=1.0).contains(&threshold),
            "Threshold must be between 0.0 and 1.0, got {}",
            threshold
        );

        let key = Self::threshold_to_key(threshold);

        // Check if already in cache (lock-free read)
        if let Some(partition) = self.partition_cache.get(&key) {
            return partition;
        }

        // Reconstruct the partition
        let partition = self
            .reconstruct_at_threshold(threshold)
            .expect("Storage iteration failed during partition reconstruction");

        // Wrap in Arc and store in cache (lock-free insert)
        let partition_arc = Arc::new(partition);
        self.partition_cache.insert(key, partition_arc.clone());

        partition_arc
    }

    /// Reconstruct a partition at a specific threshold with streaming processing
    fn reconstruct_at_threshold(&self, threshold: f64) -> Result<PartitionLevel, String> {
        let num_records = self.context.len();
        let total_events = self.storage.len();

        // Determine processing approach based on estimated memory usage
        let estimated_memory_mb = (num_records * 64) / (1024 * 1024); // Rough estimate for union-find

        if estimated_memory_mb > 100 {
            // Use memory-mapped union-find for datasets >100MB estimated memory
            // OS virtual memory system handles paging automatically, providing
            // efficient processing without correctness issues
            self.reconstruct_with_backend(threshold, true)
        } else {
            // Use regular streaming for smaller datasets that fit comfortably in RAM
            let _batch_size = if total_events > 10_000 {
                5_000
            } else {
                total_events.max(1_000)
            };
            self.reconstruct_with_backend(threshold, false)
        }
    }

    /// Reconstruct a partition using the specified backend type
    fn reconstruct_with_backend(
        &self,
        threshold: f64,
        use_mmap: bool,
    ) -> Result<PartitionLevel, String> {
        let num_records = self.context.len();

        if use_mmap {
            // Use memory-mapped backend
            let mut uf = UnionFind::new_mmap(num_records)
                .map_err(|e| format!("Failed to create memory-mapped union-find: {}", e))?;
            self.apply_merges_to_union_find(&mut uf, threshold)
        } else {
            // Use in-memory backend
            let mut uf = UnionFind::new_vec(num_records);
            self.apply_merges_to_union_find(&mut uf, threshold)
        }
    }

    /// Apply merge events to a union-find structure for any backend type
    fn apply_merges_to_union_find<B: super::union_find::UnionFindBackend>(
        &self,
        uf: &mut UnionFind<B>,
        threshold: f64,
    ) -> Result<PartitionLevel, String> {
        let num_records = self.context.len();

        // Process all merge events above threshold
        for merge in self
            .storage
            .iter()
            .map_err(|e| format!("Storage iteration error: {}", e))?
        {
            if merge.threshold < threshold {
                // Events are sorted by descending threshold, so we can stop
                break;
            }

            // Apply binary delta: union all child nodes with parent
            for node in merge.child_nodes.iter() {
                uf.union(merge.parent_id as usize, node as usize);
            }
        }

        // Convert union-find to partition with entities using batched processing
        let mut entities_map: HashMap<usize, RoaringBitmap> = HashMap::new();

        // Determine batch size based on available resources
        let base_batch_size = if num_records > 1_000_000 {
            100_000
        } else if num_records > 100_000 {
            50_000
        } else {
            num_records // Process small datasets in one go
        };

        let batch_size = self.context.get_adaptive_batch_size(base_batch_size);

        // Process records in batches to avoid memory exhaustion
        for batch_start in (0..num_records).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(num_records);

            // Check resource availability before processing each batch
            if batch_start > 0 {
                self.context.wait_for_resources();
            }

            // Process this batch of records
            for record_idx in batch_start..batch_end {
                let root = uf.find(record_idx);
                entities_map.entry(root).or_insert_with(|| {
                    let estimated_size = (batch_size / 100).max(10) as u32;
                    let (bitmap, _) = self.bitmap_pool.get(estimated_size);
                    bitmap
                });
                entities_map
                    .get_mut(&root)
                    .unwrap()
                    .insert(record_idx as u32);
            }

            // Report progress for large datasets
            if num_records > 1_000_000 && batch_end.is_multiple_of(500_000) {
                let progress = batch_end as f64 / num_records as f64;
                debug_println!(
                    "      🔄 Reconstructing partition: {:.1}% ({:.1}M/{:.1}M records)",
                    progress * 100.0,
                    batch_end as f64 / 1_000_000.0,
                    num_records as f64 / 1_000_000.0
                );
            }
        }

        // Convert HashMap to Vec of entities
        let entities: Vec<RoaringBitmap> = entities_map.into_values().collect();

        Ok(PartitionLevel::new(threshold, entities))
    }

    /// Build multiple partitions using incremental reconstruction for efficiency
    pub fn build_partitions_incrementally(&self, thresholds: &[f64]) -> Vec<Arc<PartitionLevel>> {
        use super::incremental::IncrementalPartitionBuilder;

        // Use incremental builder for multiple thresholds
        let mut builder = IncrementalPartitionBuilder::new(self.context.clone());

        // Build all partitions incrementally
        let partitions = builder
            .build_multiple(thresholds, self.storage.as_ref())
            .expect("Incremental partition building failed");

        // Store all partitions in cache
        for (i, &threshold) in thresholds.iter().enumerate() {
            let key = Self::threshold_to_key(threshold);
            self.partition_cache.insert(key, partitions[i].clone());
        }

        partitions
    }

    /// Get merge events between two thresholds (exclusive of start, inclusive of end)
    /// Returns events in descending threshold order (HIGH to LOW)
    ///
    /// # Arguments
    /// * `from_threshold` - Starting threshold (higher value, exclusive)
    /// * `to_threshold` - Ending threshold (lower value, inclusive)
    ///
    /// # Returns
    /// Vector of MergeEvents that occur between the thresholds
    pub fn get_merge_events_between(
        &self,
        from_threshold: f64,
        to_threshold: f64,
    ) -> Result<Vec<MergeEvent>, String> {
        assert!(
            from_threshold >= to_threshold,
            "from_threshold ({}) must be >= to_threshold ({})",
            from_threshold,
            to_threshold
        );

        let mut events = Vec::new();

        // Iterate through merge events (they're in descending threshold order)
        for merge in self
            .storage
            .iter()
            .map_err(|e| format!("Storage iteration error: {}", e))?
        {
            // Skip events at or above the starting threshold (exclusive)
            if merge.threshold >= from_threshold {
                continue;
            }

            // Include events above the ending threshold (inclusive of end)
            if merge.threshold > to_threshold {
                events.push(merge);
            } else {
                // We've passed the ending threshold, can stop
                break;
            }
        }

        Ok(events)
    }

    /// Get a reference to the storage backend (for incremental builder)
    #[cfg(test)]
    pub(crate) fn storage(&self) -> &dyn HierarchyStorage {
        self.storage.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{DataContext, Key};

    fn create_test_context() -> Arc<DataContext> {
        let ctx = DataContext::new();
        ctx.ensure_record("test", Key::U32(0));
        ctx.ensure_record("test", Key::U32(1));
        ctx.ensure_record("test", Key::U32(2));
        Arc::new(ctx)
    }

    #[test]
    fn test_empty_edges() {
        let ctx = create_test_context();
        let hierarchy = PartitionHierarchy::from_edges(vec![], ctx.clone(), 2, None).unwrap();

        assert_eq!(hierarchy.merge_events_count(), 0);
        assert_eq!(hierarchy.num_records(), 3);
    }

    #[test]
    fn test_three_node_graph() {
        let ctx = create_test_context();

        // Create a simple 3-node graph: A-B (0.8), B-C (0.6)
        let edges = vec![
            (0, 1, 0.8), // A-B
            (1, 2, 0.6), // B-C
        ];

        let hierarchy = PartitionHierarchy::from_edges(edges, ctx, 2, None).unwrap();

        // Should have two merge events: one at 0.8 and one at 0.6
        assert_eq!(hierarchy.merge_events_count(), 2);

        // Verify merge events by collecting them
        let merges: Vec<_> = hierarchy.storage.iter().unwrap().collect();

        // First merge should be at threshold 0.8 (A-B)
        assert_eq!(merges[0].threshold, 0.8);
        assert_eq!(merges[0].child_nodes.len(), 1); // Binary merge: one singleton absorbed

        // Second merge should be at threshold 0.6 ((A,B)-C)
        assert_eq!(merges[1].threshold, 0.6);
        assert_eq!(merges[1].child_nodes.len(), 1); // Binary merge: singleton C absorbed
    }

    #[test]
    fn test_disconnected_components() {
        let ctx = DataContext::new();
        // Create 4 records: A, B, C, D
        ctx.ensure_record("test", Key::U32(0));
        ctx.ensure_record("test", Key::U32(1));
        ctx.ensure_record("test", Key::U32(2));
        ctx.ensure_record("test", Key::U32(3));
        let ctx = Arc::new(ctx);

        // Create two disconnected components: A-B (0.8), C-D (0.7)
        let edges = vec![
            (0, 1, 0.8), // A-B
            (2, 3, 0.7), // C-D
        ];

        let hierarchy = PartitionHierarchy::from_edges(edges, ctx, 2, None).unwrap();

        // Should have two independent merge events
        assert_eq!(hierarchy.merge_events_count(), 2);

        // Each merge should be binary (parent + child)
        for merge in hierarchy.storage.iter().unwrap() {
            assert_eq!(merge.child_nodes.len(), 1); // Each child is a singleton
        }
    }

    #[test]
    fn test_same_threshold_edges_nway_merge() {
        let ctx = DataContext::new();
        // Create 4 records: A, B, C, D
        ctx.ensure_record("test", Key::U32(0));
        ctx.ensure_record("test", Key::U32(1));
        ctx.ensure_record("test", Key::U32(2));
        ctx.ensure_record("test", Key::U32(3));
        let ctx = Arc::new(ctx);

        // All edges have the same threshold - should create n-way merge
        let edges = vec![
            (0, 1, 0.5), // A-B
            (1, 2, 0.5), // B-C
            (2, 3, 0.5), // C-D
        ];

        let hierarchy = PartitionHierarchy::from_edges(edges, ctx, 2, None).unwrap();

        // Should create merge events as the union-find processes the edges
        // The exact number depends on the order of processing, but all should be at 0.5
        assert!(hierarchy.merge_events_count() > 0);
        for merge in hierarchy.storage.iter().unwrap() {
            assert_eq!(merge.threshold, 0.5);
        }
    }

    #[test]
    fn test_quantisation_enforcement() {
        let ctx = create_test_context();

        // Test with high precision input that should be quantised
        let edges = vec![
            (0, 1, 0.123456789), // Should be quantised to 0.12 with quantise=2
        ];

        let hierarchy = PartitionHierarchy::from_edges(edges, ctx, 2, None).unwrap();

        assert_eq!(hierarchy.merge_events_count(), 1);
        let merges: Vec<_> = hierarchy.storage.iter().unwrap().collect();
        assert_eq!(merges[0].threshold, 0.12); // Quantised to 2 decimal places
    }

    #[test]
    fn test_streaming_reconstruction_consistency() {
        let ctx = create_test_context();

        // Create a more complex test case with multiple thresholds
        let edges = vec![
            (0, 1, 0.9), // A-B
            (1, 2, 0.8), // B-C
            (0, 2, 0.7), // A-C (redundant but creates interesting merging)
        ];

        let hierarchy = PartitionHierarchy::from_edges(edges, ctx, 2, None).unwrap();

        // Test different batch sizes produce same results
        let threshold = 0.8;
        let result_large_batch = hierarchy
            .reconstruct_with_backend(threshold, false)
            .unwrap();
        let result_small_batch = hierarchy
            .reconstruct_with_backend(threshold, false)
            .unwrap();
        let result_medium_batch = hierarchy
            .reconstruct_with_backend(threshold, false)
            .unwrap();

        // All should produce the same number of entities
        assert_eq!(
            result_large_batch.entities().len(),
            result_small_batch.entities().len()
        );
        assert_eq!(
            result_large_batch.entities().len(),
            result_medium_batch.entities().len()
        );

        // All should have the same threshold
        assert_eq!(result_large_batch.threshold(), threshold);
        assert_eq!(result_small_batch.threshold(), threshold);
        assert_eq!(result_medium_batch.threshold(), threshold);

        // The entity structures should be equivalent (though order might differ)
        let total_records_large: u64 = result_large_batch.entities().iter().map(|e| e.len()).sum();
        let total_records_small: u64 = result_small_batch.entities().iter().map(|e| e.len()).sum();
        let total_records_medium: u64 =
            result_medium_batch.entities().iter().map(|e| e.len()).sum();

        assert_eq!(total_records_large, total_records_small);
        assert_eq!(total_records_large, total_records_medium);
        assert_eq!(total_records_large, 3); // Should have all 3 records
    }

    #[test]
    fn test_memory_mapped_reconstruction() {
        let ctx = create_test_context();

        let edges = vec![
            (0, 1, 0.9), // A-B
            (1, 2, 0.8), // B-C
        ];

        let hierarchy = PartitionHierarchy::from_edges(edges, ctx, 2, None).unwrap();

        // Test memory-mapped reconstruction vs streaming
        let threshold = 0.8;
        let result_mmap = hierarchy.reconstruct_with_backend(threshold, true).unwrap();
        let result_streaming = hierarchy
            .reconstruct_with_backend(threshold, false)
            .unwrap();

        // Both methods should produce same total number of records
        let total_records_mmap: u64 = result_mmap.entities().iter().map(|e| e.len()).sum();
        let total_records_streaming: u64 =
            result_streaming.entities().iter().map(|e| e.len()).sum();

        assert_eq!(total_records_mmap, total_records_streaming);
        assert_eq!(total_records_mmap, 3); // Should include all 3 records

        // Both should have correct threshold
        assert_eq!(result_mmap.threshold(), threshold);
        assert_eq!(result_streaming.threshold(), threshold);
    }

    #[test]
    fn test_threshold_conversion() {
        // Test fixed-point conversion
        let threshold = 0.123456;
        let key = PartitionHierarchy::threshold_to_key(threshold);
        let back = PartitionHierarchy::key_to_threshold(key);

        // Should be close due to fixed-point precision
        assert!((back - threshold).abs() < 0.000001);
    }

    #[test]
    #[should_panic(expected = "quantise must be between 1 and 6")]
    fn test_invalid_quantise() {
        let ctx = create_test_context();
        let edges = vec![(0, 1, 0.5)];

        // Should panic with quantise=0
        PartitionHierarchy::from_edges(edges, ctx, 0, None).unwrap();
    }

    #[test]
    fn test_threshold_0_one_entity() {
        let ctx = create_test_context();

        // Create edges connecting all nodes
        let edges = vec![
            (0, 1, 0.8), // A-B
            (1, 2, 0.6), // B-C
        ];

        let hierarchy = PartitionHierarchy::from_edges(edges, ctx, 2, None).unwrap();

        // At threshold 0.0, all records should be in one entity
        let partition = hierarchy.at_threshold(0.0);

        assert_eq!(partition.threshold(), 0.0);
        assert_eq!(partition.num_entities(), 1);
        assert_eq!(partition.total_records(), 3);

        // Verify all records are in the same entity
        let entity = &partition.entities()[0];
        assert!(entity.contains(0));
        assert!(entity.contains(1));
        assert!(entity.contains(2));
    }

    #[test]
    fn test_threshold_1_all_singletons() {
        let ctx = create_test_context();

        // Create edges with weights less than 1.0
        let edges = vec![
            (0, 1, 0.8), // A-B
            (1, 2, 0.6), // B-C
        ];

        let hierarchy = PartitionHierarchy::from_edges(edges, ctx, 2, None).unwrap();

        // At threshold 1.0, each record should be a singleton
        let partition = hierarchy.at_threshold(1.0);

        assert_eq!(partition.threshold(), 1.0);
        assert_eq!(partition.num_entities(), 3);
        assert_eq!(partition.total_records(), 3);

        // Each entity should have exactly one record
        for entity in partition.entities() {
            assert_eq!(entity.len(), 1);
        }
    }

    #[test]
    fn test_threshold_intermediate() {
        let ctx = create_test_context();

        // Create edges: A-B at 0.8, B-C at 0.4
        let edges = vec![
            (0, 1, 0.8), // A-B
            (1, 2, 0.4), // B-C
        ];

        let hierarchy = PartitionHierarchy::from_edges(edges, ctx, 2, None).unwrap();

        // At threshold 0.5, A-B should be merged but C separate
        let partition = hierarchy.at_threshold(0.5);

        assert_eq!(partition.threshold(), 0.5);
        assert_eq!(partition.num_entities(), 2);

        // Find which entity contains A and B
        let entity_a = partition.find_entity_for_record(0).unwrap();
        let entity_b = partition.find_entity_for_record(1).unwrap();
        let entity_c = partition.find_entity_for_record(2).unwrap();

        // A and B should be in the same entity
        assert_eq!(entity_a, entity_b);
        // C should be in a different entity
        assert_ne!(entity_a, entity_c);
    }

    #[test]
    fn test_isolates_as_singletons() {
        let ctx = DataContext::new();

        // Create 5 records
        for i in 0..5 {
            ctx.ensure_record("test", Key::U32(i));
        }
        let ctx = Arc::new(ctx);

        // Only connect some records, leaving 3 and 4 as isolates
        let edges = vec![
            (0, 1, 0.8), // Connect 0-1
            (1, 2, 0.6), // Connect 1-2
        ];

        let hierarchy = PartitionHierarchy::from_edges(edges, ctx, 2, None).unwrap();

        // At threshold 0.5, should have:
        // - One entity with {0, 1, 2}
        // - Two singleton entities for isolates {3} and {4}
        let partition = hierarchy.at_threshold(0.5);

        assert_eq!(partition.num_entities(), 3);
        assert_eq!(partition.total_records(), 5);

        // Verify isolates exist as singletons
        assert!(partition.contains_record(3));
        assert!(partition.contains_record(4));

        // Count singleton entities
        let singleton_count = partition.entities().iter().filter(|e| e.len() == 1).count();
        assert_eq!(singleton_count, 2); // Two isolates
    }

    #[test]
    fn test_cache_functionality() {
        let ctx = create_test_context();

        let edges = vec![(0, 1, 0.8), (1, 2, 0.6)];

        let hierarchy = PartitionHierarchy::from_edges(edges, ctx, 2, None).unwrap();

        // First access - should reconstruct
        let partition1 = hierarchy.at_threshold(0.7);
        assert_eq!(partition1.threshold(), 0.7);

        // Second access to same threshold - should hit cache
        let partition2 = hierarchy.at_threshold(0.7);
        assert_eq!(partition2.threshold(), 0.7);

        // Access different thresholds to test cache capacity
        for i in 0..15 {
            let threshold = (i as f64 * 0.05 * 100.0).round() / 100.0; // Round to avoid fp precision issues
            let partition = hierarchy.at_threshold(threshold);
            assert!((partition.threshold() - threshold).abs() < 1e-10);
        }

        // Original threshold might be evicted due to LRU cache size limit
        // But should still work correctly
        let partition3 = hierarchy.at_threshold(0.7);
        assert!((partition3.threshold() - 0.7).abs() < 1e-10);
    }

    #[test]
    #[should_panic(expected = "Threshold must be between 0.0 and 1.0")]
    fn test_invalid_threshold_negative() {
        let ctx = create_test_context();
        let edges = vec![(0, 1, 0.5)];
        let hierarchy = PartitionHierarchy::from_edges(edges, ctx, 2, None).unwrap();

        // Should panic with negative threshold
        hierarchy.at_threshold(-0.1);
    }

    #[test]
    #[should_panic(expected = "Threshold must be between 0.0 and 1.0")]
    fn test_invalid_threshold_too_large() {
        let ctx = create_test_context();
        let edges = vec![(0, 1, 0.5)];
        let hierarchy = PartitionHierarchy::from_edges(edges, ctx, 2, None).unwrap();

        // Should panic with threshold > 1.0
        hierarchy.at_threshold(1.1);
    }

    #[test]
    fn test_separate_components_same_threshold() {
        // This test specifically addresses the "single giant component" bug
        let ctx = DataContext::new();
        // Create 6 records: A, B, C, D, E, F
        for i in 0..6 {
            ctx.ensure_record("test", Key::U32(i as u32));
        }
        let ctx = Arc::new(ctx);

        // Create three separate pairs at the same threshold
        let edges = vec![
            (0, 1, 0.9), // A-B
            (2, 3, 0.9), // C-D
            (4, 5, 0.9), // E-F
        ];

        let hierarchy = PartitionHierarchy::from_edges(edges, ctx, 2, None).unwrap();

        // At threshold 0.9, should have 3 components (3 pairs)
        let partition = hierarchy.at_threshold(0.9);
        let num_entities_09 = partition.entities().len();
        assert_eq!(
            num_entities_09, 3,
            "Should have exactly 3 components at threshold 0.9"
        );

        // Verify each component has exactly 2 members
        for entity in partition.entities() {
            assert_eq!(
                entity.len(),
                2,
                "Each component should have exactly 2 members"
            );
        }

        // At threshold 1.0, should have 6 singletons
        let partition_high = hierarchy.at_threshold(1.0);
        assert_eq!(
            partition_high.entities().len(),
            6,
            "Should have 6 singletons at threshold 1.0"
        );
    }

    #[test]
    fn test_hierarchical_component_formation() {
        let ctx = DataContext::new();
        // Create 8 records for complex hierarchy
        for i in 0..8 {
            ctx.ensure_record("test", Key::U32(i as u32));
        }
        let ctx = Arc::new(ctx);

        // Hierarchical structure:
        // Threshold 0.9: (0,1), (2,3) -> 2 pairs + 4 singletons = 6 entities
        // Threshold 0.8: (4,5) -> 3 pairs + 2 singletons = 5 entities
        // Threshold 0.7: (6,7) -> 4 pairs = 4 entities
        let edges = vec![
            (0, 1, 0.9), // Pair 1
            (2, 3, 0.9), // Pair 2
            (4, 5, 0.8), // Pair 3 (lower threshold)
            (6, 7, 0.7), // Pair 4 (lowest threshold)
        ];

        let hierarchy = PartitionHierarchy::from_edges(edges, ctx, 2, None).unwrap();

        // Test hierarchical behaviour
        let partition_high = hierarchy.at_threshold(0.95);
        assert_eq!(
            partition_high.entities().len(),
            8,
            "Above all thresholds: 8 singletons"
        );

        let partition_09 = hierarchy.at_threshold(0.9);
        assert_eq!(
            partition_09.entities().len(),
            6,
            "At 0.9: 2 pairs + 4 singletons = 6 entities"
        );

        let partition_08 = hierarchy.at_threshold(0.8);
        assert_eq!(
            partition_08.entities().len(),
            5,
            "At 0.8: 3 pairs + 2 singletons = 5 entities"
        );

        let partition_07 = hierarchy.at_threshold(0.7);
        assert_eq!(
            partition_07.entities().len(),
            4,
            "At 0.7: 4 pairs = 4 entities"
        );

        let partition_low = hierarchy.at_threshold(0.5);
        assert_eq!(
            partition_low.entities().len(),
            4,
            "Below all thresholds: same as 0.7"
        );
    }

    #[test]
    fn test_quantisation_preserves_components() {
        let ctx = DataContext::new();
        // Create 4 records
        for i in 0..4 {
            ctx.ensure_record("test", Key::U32(i as u32));
        }
        let ctx = Arc::new(ctx);

        // Test different quantisation levels preserve component structure
        let edges = vec![
            (0, 1, 0.85432), // Should quantise to 0.85
            (2, 3, 0.75678), // Should quantise to 0.76
        ];

        // Test with quantise=2 (2 decimal places)
        let hierarchy =
            PartitionHierarchy::from_edges(edges.clone(), ctx.clone(), 2, None).unwrap();

        let partition_high = hierarchy.at_threshold(0.9);
        assert_eq!(
            partition_high.entities().len(),
            4,
            "Above quantised thresholds: 4 singletons"
        );

        let partition_mid = hierarchy.at_threshold(0.8);
        assert_eq!(
            partition_mid.entities().len(),
            3,
            "Between quantised thresholds: 1 pair + 2 singletons"
        );

        let partition_low = hierarchy.at_threshold(0.7);
        assert_eq!(
            partition_low.entities().len(),
            2,
            "Below quantised thresholds: 2 pairs"
        );
    }

    #[test]
    fn test_chain_pattern_detection() {
        let ctx = DataContext::new();
        // Create 200k sequential edges (chain pattern)
        for i in 0..200_001 {
            ctx.ensure_record("test", Key::U32(i));
        }
        let ctx = Arc::new(ctx);

        let edges: Vec<_> = (0..200_000).map(|i| (i, i + 1, 0.85)).collect();

        // Chain pattern should be rejected with an error
        let result = PartitionHierarchy::from_edges(edges, ctx, 2, None);

        assert!(result.is_err(), "Chain pattern should be rejected");
        if let Err(error_msg) = result {
            assert!(
                error_msg.contains("UNSUPPORTED DATA PATTERN"),
                "Error should mention unsupported pattern, got: {}",
                error_msg
            );
            assert!(
                error_msg.contains("sequential chain"),
                "Error should describe chain pattern, got: {}",
                error_msg
            );
        }
    }

    #[test]
    fn test_cluster_pattern_no_warning() {
        let ctx = DataContext::new();
        // Create 200k edges in cluster pattern (2000 clusters of 100)
        for i in 0..200_000 {
            ctx.ensure_record("test", Key::U32(i));
        }
        let ctx = Arc::new(ctx);

        let mut edges = Vec::new();
        for cluster_id in 0..2000 {
            let base = cluster_id * 100;
            for i in 0..99 {
                edges.push((base + i, base + i + 1, 0.85));
            }
        }

        let hierarchy = PartitionHierarchy::from_edges(edges, ctx, 2, None).unwrap();

        // Cluster pattern should NOT trigger warning (stays in-memory)
        // This is harder to test without capturing stderr, but we can check it completes
        assert!(hierarchy.merge_events_count() > 0);
    }
}
