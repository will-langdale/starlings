//! Incremental partition building for O(k) threshold sweep complexity
//!
//! This module provides efficient incremental partition reconstruction when moving
//! between thresholds. Instead of rebuilding from scratch (O(m) where m = total events),
//! it applies only the delta of merge events between thresholds (O(k) where k = events in range).

use super::merge_event::MergeEvent;
use super::partition::PartitionLevel;
use super::storage::HierarchyStorage;
use super::union_find::{UnionFind, UnionFindBackend, VecBackend};
use crate::core::DataContext;
use roaring::RoaringBitmap;
use std::collections::HashMap;
use std::sync::Arc;

/// Builder for incremental partition reconstruction across thresholds
pub struct IncrementalPartitionBuilder {
    /// The data context with all records
    context: Arc<DataContext>,
    /// Last threshold that was built
    last_threshold: Option<f64>,
    /// Union-find state at last threshold
    union_find: Option<UnionFind<VecBackend>>,
    /// Canonical ID (min record) for each entity root
    canonical_ids: HashMap<usize, u32>,
}

impl IncrementalPartitionBuilder {
    /// Create a new incremental builder
    pub fn new(context: Arc<DataContext>) -> Self {
        Self {
            context,
            last_threshold: None,
            union_find: None,
            canonical_ids: HashMap::new(),
        }
    }

    /// Build a partition at the given threshold, using incremental updates if possible
    pub fn build_at_threshold(
        &mut self,
        threshold: f64,
        storage: &dyn HierarchyStorage,
    ) -> Result<Arc<PartitionLevel>, String> {
        // Use incremental updates when moving to lower thresholds, otherwise rebuild
        match self.last_threshold {
            Some(last) if last >= threshold => self.apply_incremental_updates(threshold, storage),
            _ => self.build_from_scratch(threshold, storage),
        }
    }

    /// Build multiple partitions efficiently using incremental updates
    pub fn build_multiple(
        &mut self,
        thresholds: &[f64],
        storage: &dyn HierarchyStorage,
    ) -> Result<Vec<Arc<PartitionLevel>>, String> {
        // Sort thresholds in descending order for optimal incremental building
        let mut sorted_thresholds = thresholds.to_vec();
        sorted_thresholds.sort_by(|a, b| b.partial_cmp(a).unwrap());

        let mut results = Vec::with_capacity(thresholds.len());
        let mut threshold_to_partition: HashMap<u32, Arc<PartitionLevel>> = HashMap::new();

        // Build partitions in optimal order
        for &threshold in &sorted_thresholds {
            let partition = self.build_at_threshold(threshold, storage)?;
            let key = Self::threshold_to_key(threshold);
            threshold_to_partition.insert(key, partition);
        }

        // Return partitions in original order
        for &threshold in thresholds {
            let key = Self::threshold_to_key(threshold);
            results.push(
                threshold_to_partition
                    .get(&key)
                    .expect("Partition must have been built")
                    .clone(),
            );
        }

        Ok(results)
    }

    /// Build a partition from scratch at the given threshold
    fn build_from_scratch(
        &mut self,
        threshold: f64,
        storage: &dyn HierarchyStorage,
    ) -> Result<Arc<PartitionLevel>, String> {
        let num_records = self.context.len();
        let mut uf = UnionFind::new_vec(num_records);
        self.canonical_ids.clear();

        // Apply all merge events above the threshold
        for merge in storage
            .iter()
            .map_err(|e| format!("Storage iteration error: {}", e))?
        {
            if merge.threshold < threshold {
                break; // Events are sorted by descending threshold
            }

            self.apply_merge_event(&mut uf, &merge);
        }

        // Store state for next incremental update
        self.last_threshold = Some(threshold);
        self.union_find = Some(uf.clone());

        // Build partition from union-find
        Ok(Arc::new(self.build_partition_from_uf(&mut uf, threshold)))
    }

    /// Apply incremental updates from the last threshold to the new threshold
    fn apply_incremental_updates(
        &mut self,
        threshold: f64,
        storage: &dyn HierarchyStorage,
    ) -> Result<Arc<PartitionLevel>, String> {
        let last_threshold = self.last_threshold.expect("Must have last threshold");
        let mut uf = self
            .union_find
            .as_ref()
            .expect("Must have union-find")
            .clone();

        // Apply merge events between last_threshold and threshold
        for merge in storage
            .iter()
            .map_err(|e| format!("Storage iteration error: {}", e))?
        {
            // Skip events above last threshold (already applied)
            if merge.threshold >= last_threshold {
                continue;
            }

            // Stop at events below target threshold
            if merge.threshold < threshold {
                break;
            }

            // Apply this merge event
            self.apply_merge_event(&mut uf, &merge);
        }

        // Update state for next incremental update
        self.last_threshold = Some(threshold);
        self.union_find = Some(uf.clone());

        // Build partition from updated union-find
        Ok(Arc::new(self.build_partition_from_uf(&mut uf, threshold)))
    }

    /// Apply a single merge event to the union-find structure
    fn apply_merge_event<B: UnionFindBackend>(
        &mut self,
        uf: &mut UnionFind<B>,
        merge: &MergeEvent,
    ) {
        // Apply binary delta: union all child nodes with parent
        for node in merge.child_nodes.iter() {
            uf.union(merge.parent_id as usize, node as usize);
        }

        // Update canonical ID for the merged entity
        let root = uf.find(merge.parent_id as usize);
        // Parent ID is already the canonical ID, so keep it
        self.canonical_ids.insert(root, merge.parent_id);
    }

    /// Build a PartitionLevel from the current union-find state
    fn build_partition_from_uf<B: UnionFindBackend>(
        &self,
        uf: &mut UnionFind<B>,
        threshold: f64,
    ) -> PartitionLevel {
        let num_records = self.context.len();
        let mut root_to_entity: HashMap<usize, RoaringBitmap> = HashMap::new();

        // Group records by their root (entity)
        for record_id in 0..num_records {
            let root = uf.find(record_id);
            root_to_entity
                .entry(root)
                .or_default()
                .insert(record_id as u32);
        }

        // Convert to vector of entities
        let entities: Vec<RoaringBitmap> = root_to_entity.into_values().collect();

        PartitionLevel::new(threshold, entities)
    }

    /// Convert a threshold to a cache key
    fn threshold_to_key(threshold: f64) -> u32 {
        (threshold * 1_000_000.0).round() as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Key;
    use crate::hierarchy::PartitionHierarchy;

    fn create_test_hierarchy() -> (Arc<DataContext>, PartitionHierarchy) {
        let ctx = DataContext::new();

        // Create 6 records
        for i in 0..6 {
            ctx.ensure_record("test", Key::U32(i));
        }

        let ctx = Arc::new(ctx);

        // Create edges that merge at different thresholds
        let edges = vec![
            (0, 1, 0.9), // Merge 0-1 at 0.9
            (2, 3, 0.8), // Merge 2-3 at 0.8
            (0, 2, 0.7), // Merge (0-1) with (2-3) at 0.7
            (4, 5, 0.6), // Merge 4-5 at 0.6
        ];

        let hierarchy = PartitionHierarchy::from_edges(edges, ctx.clone(), 6, None).unwrap();
        (ctx, hierarchy)
    }

    #[test]
    fn test_incremental_builder_single() {
        let (ctx, hierarchy) = create_test_hierarchy();
        let storage = hierarchy.storage();

        let mut builder = IncrementalPartitionBuilder::new(ctx);

        // Build at threshold 0.75
        // Edges above 0.75: (0,1,0.9) and (2,3,0.8) are merged
        // Edges below 0.75: (0,2,0.7) and (4,5,0.6) are not merged
        // Result: {0,1}, {2,3}, {4}, {5} = 4 entities
        let partition = builder.build_at_threshold(0.75, storage).unwrap();
        assert_eq!(partition.num_entities(), 4);
    }

    #[test]
    fn test_incremental_builder_sweep() {
        let (ctx, hierarchy) = create_test_hierarchy();
        let storage = hierarchy.storage();

        let mut builder = IncrementalPartitionBuilder::new(ctx);

        // Build multiple thresholds
        let thresholds = vec![0.95, 0.85, 0.75, 0.65];
        let partitions = builder.build_multiple(&thresholds, storage).unwrap();

        assert_eq!(partitions.len(), 4);
        assert_eq!(partitions[0].num_entities(), 6); // 0.95: all singletons
        assert_eq!(partitions[1].num_entities(), 5); // 0.85: {0,1}, {2}, {3}, {4}, {5}
        assert_eq!(partitions[2].num_entities(), 4); // 0.75: {0,1}, {2,3}, {4}, {5}
        assert_eq!(partitions[3].num_entities(), 3); // 0.65: {0,1,2,3}, {4}, {5}
    }

    #[test]
    fn test_incremental_matches_full_rebuild() {
        let (ctx, hierarchy) = create_test_hierarchy();
        let storage = hierarchy.storage();

        let mut incremental_builder = IncrementalPartitionBuilder::new(ctx.clone());

        let thresholds = vec![0.95, 0.85, 0.75, 0.65, 0.55];

        // Build incrementally
        let incremental_results = incremental_builder
            .build_multiple(&thresholds, storage)
            .unwrap();

        // Compare with direct reconstruction
        for (i, &threshold) in thresholds.iter().enumerate() {
            let direct = hierarchy.at_threshold(threshold);
            let incremental = &incremental_results[i];

            assert_eq!(
                direct.num_entities(),
                incremental.num_entities(),
                "Mismatch at threshold {}",
                threshold
            );
            assert_eq!(
                direct.total_records(),
                incremental.total_records(),
                "Record count mismatch at threshold {}",
                threshold
            );
        }
    }
}
