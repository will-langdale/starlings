/// Memory pool for efficient RoaringBitmap allocation and reuse
use roaring::RoaringBitmap;
use std::collections::VecDeque;
use std::sync::Mutex;

/// Size class for bitmap pool classification
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PoolSizeClass {
    Small,
    Large,
}

/// Memory pool for efficient RoaringBitmap allocation and reuse
#[derive(Debug)]
pub struct BitmapPool {
    small_pool: Mutex<VecDeque<RoaringBitmap>>,
    large_pool: Mutex<VecDeque<RoaringBitmap>>,
    max_small: usize,
    max_large: usize,
}

impl BitmapPool {
    const DEFAULT_SMALL_POOL_SIZE: usize = 1000;
    const DEFAULT_LARGE_POOL_SIZE: usize = 100;
    const SMALL_THRESHOLD: u32 = 1000;

    /// Create a new bitmap pool with default configuration
    pub fn new() -> Self {
        Self::with_capacity(Self::DEFAULT_SMALL_POOL_SIZE, Self::DEFAULT_LARGE_POOL_SIZE)
    }

    /// Create a bitmap pool scaled for expected dataset size
    pub fn new_for_scale(num_edges: usize) -> Self {
        let scale_factor = match num_edges {
            0..=10_000 => 1,
            10_001..=100_000 => 2,
            100_001..=1_000_000 => 5,
            _ => 10,
        };

        let small_pool_size = (Self::DEFAULT_SMALL_POOL_SIZE * scale_factor).min(10_000);
        let large_pool_size = (Self::DEFAULT_LARGE_POOL_SIZE * scale_factor).min(1_000);

        Self::with_capacity(small_pool_size, large_pool_size)
    }

    /// Create a bitmap pool with custom capacity limits
    pub fn with_capacity(max_small: usize, max_large: usize) -> Self {
        Self {
            small_pool: Mutex::new(VecDeque::with_capacity(max_small)),
            large_pool: Mutex::new(VecDeque::with_capacity(max_large)),
            max_small,
            max_large,
        }
    }

    /// Get a RoaringBitmap from the pool, creating new one if pool is empty
    pub fn get(&self, size_hint: u32) -> (RoaringBitmap, PoolSizeClass) {
        let size_class = if size_hint < Self::SMALL_THRESHOLD {
            PoolSizeClass::Small
        } else {
            PoolSizeClass::Large
        };

        let pool = match size_class {
            PoolSizeClass::Small => &self.small_pool,
            PoolSizeClass::Large => &self.large_pool,
        };

        if let Ok(mut guard) = pool.lock() {
            if let Some(mut bitmap) = guard.pop_front() {
                bitmap.clear();
                return (bitmap, size_class);
            }
        }

        (RoaringBitmap::new(), size_class)
    }

    /// Return a RoaringBitmap to the pool for reuse
    pub fn put(&self, mut bitmap: RoaringBitmap, size_class: PoolSizeClass) {
        bitmap.clear();

        let (pool, max_size) = match size_class {
            PoolSizeClass::Small => (&self.small_pool, self.max_small),
            PoolSizeClass::Large => (&self.large_pool, self.max_large),
        };

        if let Ok(mut guard) = pool.lock() {
            if guard.len() < max_size {
                guard.push_back(bitmap);
            }
        }
    }

    /// Get current pool statistics for monitoring
    pub fn stats(&self) -> BitmapPoolStats {
        let small_count = self.small_pool.lock().map_or(0, |guard| guard.len());
        let large_count = self.large_pool.lock().map_or(0, |guard| guard.len());

        BitmapPoolStats {
            small_available: small_count,
            large_available: large_count,
            total_available: small_count + large_count,
        }
    }

    /// Clear all pools to free memory
    pub fn clear(&self) {
        if let Ok(mut guard) = self.small_pool.lock() {
            guard.clear();
        }
        if let Ok(mut guard) = self.large_pool.lock() {
            guard.clear();
        }
    }
}

impl Default for BitmapPool {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for BitmapPool {
    fn clone(&self) -> Self {
        Self::with_capacity(self.max_small, self.max_large)
    }
}

/// Pool statistics for monitoring and debugging
#[derive(Debug, Clone)]
pub struct BitmapPoolStats {
    pub small_available: usize,
    pub large_available: usize,
    pub total_available: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bitmap_pool_basic() {
        let pool = BitmapPool::new();

        // Get a bitmap from empty pool
        let (mut bitmap1, size_class1) = pool.get(50);
        assert_eq!(bitmap1.len(), 0);
        assert_eq!(size_class1, PoolSizeClass::Small);

        // Add some data
        bitmap1.insert(1);
        bitmap1.insert(2);
        bitmap1.insert(3);
        assert_eq!(bitmap1.len(), 3);

        // Return to pool
        pool.put(bitmap1, size_class1);

        // Get another bitmap - should be reused and cleared
        let (bitmap2, _) = pool.get(50);
        assert_eq!(bitmap2.len(), 0);
    }

    #[test]
    fn test_pool_size_classification() {
        let pool = BitmapPool::new();

        // Test small pool
        let (small_bitmap, small_class) = pool.get(100);
        assert_eq!(small_class, PoolSizeClass::Small);
        pool.put(small_bitmap, small_class);

        // Test large pool
        let (large_bitmap, large_class) = pool.get(5000);
        assert_eq!(large_class, PoolSizeClass::Large);
        pool.put(large_bitmap, large_class);

        let stats = pool.stats();
        assert_eq!(stats.small_available, 1);
        assert_eq!(stats.large_available, 1);
        assert_eq!(stats.total_available, 2);
    }

    #[test]
    fn test_pool_capacity_limits() {
        let pool = BitmapPool::with_capacity(2, 1);

        // Create 3 separate bitmaps and populate them to make them distinct
        let (mut bitmap1, class1) = pool.get(10);
        bitmap1.insert(1);
        let (mut bitmap2, class2) = pool.get(10);
        bitmap2.insert(2);
        let (mut bitmap3, class3) = pool.get(10);
        bitmap3.insert(3);

        // Return them all to pool - only 2 should be kept
        pool.put(bitmap1, class1);
        pool.put(bitmap2, class2);
        pool.put(bitmap3, class3); // This should be dropped

        let stats = pool.stats();
        assert_eq!(stats.small_available, 2); // Only 2, third was dropped
    }

    #[test]
    fn test_pool_clear() {
        let pool = BitmapPool::new();

        // Add bitmaps to both pools
        let (bitmap1, class1) = pool.get(10);
        pool.put(bitmap1, class1); // small
        let (bitmap2, class2) = pool.get(5000);
        pool.put(bitmap2, class2); // large

        let stats_before = pool.stats();
        assert_eq!(stats_before.total_available, 2);

        pool.clear();

        let stats_after = pool.stats();
        assert_eq!(stats_after.total_available, 0);
    }

    #[test]
    fn test_concurrent_access() {
        use std::sync::Arc;
        use std::thread;

        let pool = Arc::new(BitmapPool::new());
        let mut handles = vec![];

        // Spawn multiple threads using the pool
        for i in 0..10 {
            let pool_clone = Arc::clone(&pool);
            let handle = thread::spawn(move || {
                let (mut bitmap, size_class) = pool_clone.get(i * 100);
                bitmap.insert(i);
                assert!(bitmap.contains(i));
                pool_clone.put(bitmap, size_class);
            });
            handles.push(handle);
        }

        // Wait for all threads
        for handle in handles {
            handle.join().unwrap();
        }

        // Pool should have bitmaps available
        let stats = pool.stats();
        assert!(stats.total_available > 0);
    }
}
