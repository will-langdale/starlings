use quick_cache::sync::Cache;
use quick_cache::Weighter;
use std::sync::Arc;

use super::PartitionLevel;

/// Estimates memory usage of a PartitionLevel in bytes for the cache
#[derive(Clone, Debug, Default)]
pub struct PartitionWeighter;

impl Weighter<u32, Arc<PartitionLevel>> for PartitionWeighter {
    fn weight(&self, _key: &u32, val: &Arc<PartitionLevel>) -> u64 {
        // An item with a weight of 0 will not be chosen for eviction.
        // Ensure we always return at least 1.
        val.memory_usage_bytes().max(1)
    }
}

/// Memory-bounded cache for partition levels using a weight-based LRU policy.
pub struct MemoryBoundedCache {
    cache: Cache<u32, Arc<PartitionLevel>, PartitionWeighter>,
}

impl MemoryBoundedCache {
    /// Create a new cache with the specified memory limit in bytes.
    ///
    /// # Arguments
    /// * `memory_limit_bytes` - The maximum total weight (in bytes) the cache can hold.
    pub fn new(memory_limit_bytes: u64) -> Self {
        // Item count is not limited, only total weight.
        // We set a high item count limit that will likely never be reached.
        let item_limit = usize::MAX;

        let cache = Cache::with_weighter(item_limit, memory_limit_bytes, PartitionWeighter);

        Self { cache }
    }

    /// Get a partition from the cache.
    pub fn get(&self, key: &u32) -> Option<Arc<PartitionLevel>> {
        self.cache.get(key)
    }

    /// Insert a partition into the cache.
    pub fn insert(&self, key: u32, value: Arc<PartitionLevel>) {
        self.cache.insert(key, value);
    }

    /// Get the current memory usage in bytes.
    #[cfg(test)]
    pub fn memory_usage(&self) -> u64 {
        self.cache.weight()
    }

    /// Get the memory limit in bytes.
    #[cfg(test)]
    pub fn memory_limit(&self) -> u64 {
        self.cache.capacity()
    }

    /// Clear all entries from the cache.
    #[cfg(test)]
    pub fn clear(&self) {
        self.cache.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use roaring::RoaringBitmap;

    #[test]
    fn test_memory_bounded_cache_eviction() {
        // Create a small cache (limit to ~25KB) to test eviction.
        let cache = MemoryBoundedCache::new(25 * 1024);

        // Create test partitions. Each will be around 2KB.
        for i in 0..20 {
            let mut entities = Vec::new();
            for j in 0..10 {
                let mut bitmap = RoaringBitmap::new();
                // 100 * 4 bytes = 400 bytes per bitmap
                bitmap.insert_range(i * 1000 + j * 100..i * 1000 + j * 100 + 100);
                entities.push(bitmap);
            }
            let partition = Arc::new(PartitionLevel::new(i as f64, entities));
            cache.insert(i, partition);
        }

        // The cache's total weight should not exceed its capacity.
        assert!(cache.memory_usage() <= cache.memory_limit());
        // Since we inserted more than the capacity, some items must have been evicted.
        assert!(cache.cache.len() < 20);
    }

    #[test]
    fn test_partition_size_estimation_in_weigher() {
        let mut entities = Vec::new();
        for i in 0..10 {
            let mut bitmap = RoaringBitmap::new();
            bitmap.insert_range(i * 100..i * 100 + 50);
            entities.push(bitmap);
        }

        let partition = Arc::new(PartitionLevel::new(0.5, entities));
        let weighter = PartitionWeighter;

        let estimated_size = weighter.weight(&0, &partition);

        // 10 bitmaps * ~100 bytes/bitmap (for 50 contiguous integers) + base size
        let expected_min_size = 10 * 100;
        assert!(estimated_size > expected_min_size);
    }
}
