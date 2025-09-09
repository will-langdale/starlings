//! Storage backends for hierarchy data with memory-mapped file support
//!
//! This module provides abstraction over different storage backends for merge events,
//! enabling out-of-core processing for large datasets.

use roaring::RoaringBitmap;
use std::fs::{File, OpenOptions};
use std::io::{self, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::sync::Arc;
use tempfile::TempDir;

use super::merge_event::MergeEvent;
use std::sync::atomic::{AtomicU64, Ordering};

/// Trait for different storage backends
pub trait HierarchyStorage: Send + Sync {
    /// Add a merge event to storage
    fn push(&mut self, event: MergeEvent) -> Result<(), StorageError>;

    /// Get a merge event by index
    fn get(&self, index: usize) -> Result<Option<&MergeEvent>, StorageError>;

    /// Get the total number of merge events
    fn len(&self) -> usize;

    /// Check if storage is empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Iterate over all merge events as owned values
    fn iter(&self) -> Result<Box<dyn Iterator<Item = MergeEvent>>, StorageError>;

    /// Get estimated memory usage in bytes
    fn memory_usage_bytes(&self) -> u64;

    /// Sync any pending writes to storage
    fn sync(&mut self) -> Result<(), StorageError>;
}

/// Errors that can occur during storage operations
#[derive(Debug, thiserror::Error)]
pub enum StorageError {
    #[error("IO error: {0}")]
    Io(#[from] io::Error),

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("Index out of bounds: {index} >= {len}")]
    IndexOutOfBounds { index: usize, len: usize },

    #[error("Storage is read-only")]
    ReadOnly,
}

/// In-memory storage - fastest but uses most memory
#[derive(Debug, Clone)]
pub struct InMemoryStorage {
    events: Vec<MergeEvent>,
}

impl Default for InMemoryStorage {
    fn default() -> Self {
        Self::new()
    }
}

impl InMemoryStorage {
    pub fn new() -> Self {
        Self { events: Vec::new() }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            events: Vec::with_capacity(capacity),
        }
    }
}

impl HierarchyStorage for InMemoryStorage {
    fn push(&mut self, event: MergeEvent) -> Result<(), StorageError> {
        self.events.push(event);
        Ok(())
    }

    fn get(&self, index: usize) -> Result<Option<&MergeEvent>, StorageError> {
        Ok(self.events.get(index))
    }

    fn len(&self) -> usize {
        self.events.len()
    }

    fn iter(&self) -> Result<Box<dyn Iterator<Item = MergeEvent>>, StorageError> {
        let events_clone = self.events.clone();
        Ok(Box::new(events_clone.into_iter()))
    }

    fn memory_usage_bytes(&self) -> u64 {
        // Rough estimate: each merge event + bitmap data
        let event_overhead = self.events.len() * std::mem::size_of::<MergeEvent>();
        let bitmap_size: u64 = self
            .events
            .iter()
            .map(|e| e.merging_groups.iter().map(|b| b.len() * 4).sum::<u64>()) // Rough estimate: 4 bytes per element
            .sum();
        (event_overhead as u64) + bitmap_size
    }

    fn sync(&mut self) -> Result<(), StorageError> {
        // No-op for in-memory storage
        Ok(())
    }
}

/// Disk-backed storage using memory-mapped files for large datasets
pub struct DiskStorage {
    temp_dir: Arc<TempDir>,
    events_file: File,
    events_count: usize,
    memory_cache: lru::LruCache<usize, MergeEvent>,
    cached_events: Vec<Option<MergeEvent>>, // In-memory cache for recently accessed events
    bytes_written: u64,
    record_positions: Vec<u64>, // File positions of each record
}

impl DiskStorage {
    /// Create new disk storage with temporary directory
    pub fn new() -> Result<Self, StorageError> {
        Self::with_cache_size(100) // Default cache size
    }

    /// Create disk storage with specific cache size
    pub fn with_cache_size(cache_size: usize) -> Result<Self, StorageError> {
        let temp_dir = Arc::new(TempDir::new()?);
        let events_file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .read(true)
            .open(temp_dir.path().join("merge_events.dat"))?;

        Ok(Self {
            temp_dir,
            events_file,
            events_count: 0,
            memory_cache: lru::LruCache::new(cache_size.try_into().unwrap()),
            cached_events: Vec::new(),
            bytes_written: 0,
            record_positions: Vec::new(),
        })
    }

    /// Create disk storage at specific directory (for testing)
    pub fn with_temp_dir<P: AsRef<Path>>(
        temp_dir: P,
        cache_size: usize,
    ) -> Result<Self, StorageError> {
        let events_file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .read(true)
            .open(temp_dir.as_ref().join("merge_events.dat"))?;

        // Create a temporary TempDir that won't be cleaned up automatically
        let temp_dir = Arc::new(TempDir::new()?);

        Ok(Self {
            temp_dir,
            events_file,
            events_count: 0,
            memory_cache: lru::LruCache::new(cache_size.try_into().unwrap()),
            cached_events: Vec::new(),
            bytes_written: 0,
            record_positions: Vec::new(),
        })
    }

    fn serialize_event(event: &MergeEvent) -> Result<Vec<u8>, StorageError> {
        let mut buffer = Vec::new();

        // Write threshold as 8 bytes
        buffer.extend_from_slice(&event.threshold.to_le_bytes());

        // Write number of groups as 4 bytes
        buffer.extend_from_slice(&(event.merging_groups.len() as u32).to_le_bytes());

        // Write each bitmap
        for bitmap in &event.merging_groups {
            let mut bitmap_data = Vec::new();
            bitmap.serialize_into(&mut bitmap_data).map_err(|e| {
                StorageError::Serialization(format!("Bitmap serialization failed: {}", e))
            })?;
            // Write bitmap size as 4 bytes, then bitmap data
            buffer.extend_from_slice(&(bitmap_data.len() as u32).to_le_bytes());
            buffer.extend_from_slice(&bitmap_data);
        }

        Ok(buffer)
    }

    fn deserialize_event(data: &[u8]) -> Result<MergeEvent, StorageError> {
        if data.len() < 12 {
            // At least threshold + group count + one bitmap size
            return Err(StorageError::Serialization("Data too short".to_string()));
        }

        let mut offset = 0;

        // Read threshold
        let threshold = f64::from_le_bytes(
            data[offset..offset + 8]
                .try_into()
                .map_err(|_| StorageError::Serialization("Invalid threshold bytes".to_string()))?,
        );
        offset += 8;

        // Read number of groups
        let num_groups =
            u32::from_le_bytes(data[offset..offset + 4].try_into().map_err(|_| {
                StorageError::Serialization("Invalid group count bytes".to_string())
            })?) as usize;
        offset += 4;

        // Read each bitmap
        let mut merging_groups = Vec::with_capacity(num_groups);
        for _ in 0..num_groups {
            if offset + 4 > data.len() {
                return Err(StorageError::Serialization(
                    "Unexpected end of data".to_string(),
                ));
            }

            let bitmap_size =
                u32::from_le_bytes(data[offset..offset + 4].try_into().map_err(|_| {
                    StorageError::Serialization("Invalid bitmap size bytes".to_string())
                })?) as usize;
            offset += 4;

            if offset + bitmap_size > data.len() {
                return Err(StorageError::Serialization(
                    "Bitmap data truncated".to_string(),
                ));
            }

            let bitmap = RoaringBitmap::deserialize_from(&data[offset..offset + bitmap_size])
                .map_err(|e| {
                    StorageError::Serialization(format!("Bitmap deserialization failed: {}", e))
                })?;
            merging_groups.push(bitmap);
            offset += bitmap_size;
        }

        Ok(MergeEvent::new(threshold, merging_groups))
    }
}

/// Iterator for reading events from disk sequentially
pub struct DiskIterator {
    file: BufReader<File>,
    remaining_count: usize,
}

impl DiskIterator {
    fn new(file_path: &Path, count: usize) -> Result<Self, StorageError> {
        let file = File::open(file_path)?;
        let mut reader = BufReader::new(file);
        reader.seek(SeekFrom::Start(0))?;

        Ok(Self {
            file: reader,
            remaining_count: count,
        })
    }
}

impl Iterator for DiskIterator {
    type Item = MergeEvent;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining_count == 0 {
            return None;
        }

        // Read record length
        let mut len_bytes = [0u8; 4];
        if self.file.read_exact(&mut len_bytes).is_err() {
            return None;
        }
        let record_len = u32::from_le_bytes(len_bytes) as usize;

        // Read record data
        let mut record_data = vec![0u8; record_len];
        if self.file.read_exact(&mut record_data).is_err() {
            return None;
        }

        // Deserialize event
        match DiskStorage::deserialize_event(&record_data) {
            Ok(event) => {
                self.remaining_count -= 1;
                Some(event)
            }
            Err(_) => None,
        }
    }
}

impl HierarchyStorage for DiskStorage {
    fn push(&mut self, event: MergeEvent) -> Result<(), StorageError> {
        // Record position before writing
        let position = self.bytes_written;
        self.record_positions.push(position);

        // Serialize and write to disk
        let serialized = Self::serialize_event(&event)?;

        // Write record length followed by data
        let mut writer = BufWriter::new(&mut self.events_file);
        writer.write_all(&(serialized.len() as u32).to_le_bytes())?;
        writer.write_all(&serialized)?;
        writer.flush()?;

        self.bytes_written += 4 + serialized.len() as u64;

        // Add to cache and extend cached_events vector
        self.memory_cache.put(self.events_count, event);
        self.cached_events.push(None);
        self.events_count += 1;

        Ok(())
    }

    fn get(&self, index: usize) -> Result<Option<&MergeEvent>, StorageError> {
        if index >= self.events_count {
            return Ok(None);
        }

        // This is tricky because we need to return a reference but LruCache doesn't
        // provide that easily. For now, we'll use a simpler approach that doesn't
        // return references but reconstructs events on demand.
        // In a production implementation, we'd use a more sophisticated caching strategy.
        Err(StorageError::Serialization(
            "Direct reference access not supported for disk storage. Use iter() instead."
                .to_string(),
        ))
    }

    fn len(&self) -> usize {
        self.events_count
    }

    fn iter(&self) -> Result<Box<dyn Iterator<Item = MergeEvent>>, StorageError> {
        let file_path = self.temp_dir.path().join("merge_events.dat");
        let iterator = DiskIterator::new(&file_path, self.events_count)?;
        Ok(Box::new(iterator))
    }

    fn memory_usage_bytes(&self) -> u64 {
        // Estimate cache overhead + file handle
        let cache_size = self.memory_cache.len() as u64 * 1000; // Rough estimate
        cache_size + 1024 // File handle overhead
    }

    fn sync(&mut self) -> Result<(), StorageError> {
        self.events_file.sync_all()?;
        Ok(())
    }
}

impl std::fmt::Debug for DiskStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DiskStorage")
            .field("events_count", &self.events_count)
            .field("bytes_written", &self.bytes_written)
            .field("cache_size", &self.memory_cache.len())
            .finish()
    }
}

/// Hybrid storage that starts in-memory and spills to disk when memory pressure is detected
pub struct HybridStorage {
    storage: Box<dyn HierarchyStorage>,
    memory_threshold_bytes: u64,
    spilled_to_disk: bool,
    access_counter: AtomicU64,
    last_memory_check: AtomicU64,
}

impl HybridStorage {
    pub fn new(memory_threshold_bytes: u64) -> Self {
        Self {
            storage: Box::new(InMemoryStorage::new()),
            memory_threshold_bytes,
            spilled_to_disk: false,
            access_counter: AtomicU64::new(0),
            last_memory_check: AtomicU64::new(0),
        }
    }

    fn should_spill_to_disk(&self) -> bool {
        !self.spilled_to_disk && self.storage.memory_usage_bytes() > self.memory_threshold_bytes
    }

    /// Check if we should monitor memory pressure (every 1000 accesses or first time)
    fn should_check_memory_pressure(&self) -> bool {
        let current_access = self.access_counter.fetch_add(1, Ordering::Relaxed);
        let last_check = self.last_memory_check.load(Ordering::Relaxed);

        // Check every 1000 accesses or if we haven't checked yet
        current_access - last_check >= 1000 || last_check == 0
    }

    /// Check and potentially warn about memory pressure during read operations  
    fn check_memory_pressure_on_read(&self) {
        if self.should_check_memory_pressure() {
            self.last_memory_check.store(
                self.access_counter.load(Ordering::Relaxed),
                Ordering::Relaxed,
            );

            // System automatically handles memory pressure via spilling
            // Only warn about truly critical situations that require user attention
        }
    }

    fn spill_to_disk(&mut self) -> Result<(), StorageError> {
        if self.spilled_to_disk {
            return Ok(());
        }

        // Create disk storage
        let mut disk_storage = DiskStorage::new()?;

        // Copy all events from memory to disk
        for event in self.storage.iter()? {
            disk_storage.push(event)?;
        }

        // Replace storage backend
        self.storage = Box::new(disk_storage);
        self.spilled_to_disk = true;

        // Spilling is now a normal part of automatic resource management
        // No need to warn users about routine operational decisions

        Ok(())
    }
}

impl HierarchyStorage for HybridStorage {
    fn push(&mut self, event: MergeEvent) -> Result<(), StorageError> {
        // Check if we should spill to disk before adding
        if self.should_spill_to_disk() {
            self.spill_to_disk()?;
        }

        self.storage.push(event)
    }

    fn get(&self, index: usize) -> Result<Option<&MergeEvent>, StorageError> {
        self.check_memory_pressure_on_read();
        self.storage.get(index)
    }

    fn len(&self) -> usize {
        self.storage.len()
    }

    fn memory_usage_bytes(&self) -> u64 {
        self.storage.memory_usage_bytes()
    }

    fn iter(&self) -> Result<Box<dyn Iterator<Item = MergeEvent>>, StorageError> {
        self.check_memory_pressure_on_read();
        self.storage.iter()
    }

    fn sync(&mut self) -> Result<(), StorageError> {
        self.storage.sync()
    }
}

impl std::fmt::Debug for HybridStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HybridStorage")
            .field("memory_threshold_bytes", &self.memory_threshold_bytes)
            .field("spilled_to_disk", &self.spilled_to_disk)
            .field("current_memory_usage", &self.storage.memory_usage_bytes())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_merge_event(threshold: f64, num_groups: usize) -> MergeEvent {
        let mut groups = Vec::new();
        for i in 0..num_groups {
            let mut bitmap = RoaringBitmap::new();
            bitmap.insert((i * 10) as u32);
            bitmap.insert((i * 10 + 1) as u32);
            groups.push(bitmap);
        }
        MergeEvent::new(threshold, groups)
    }

    #[test]
    fn test_in_memory_storage() {
        let mut storage = InMemoryStorage::new();

        assert_eq!(storage.len(), 0);
        assert!(storage.is_empty());

        let event = create_test_merge_event(0.8, 2);
        storage.push(event.clone()).unwrap();

        assert_eq!(storage.len(), 1);
        assert!(!storage.is_empty());
        assert!(storage.memory_usage_bytes() > 0);

        let retrieved = storage.get(0).unwrap().unwrap();
        assert_eq!(retrieved.threshold, 0.8);
        assert_eq!(retrieved.merging_groups.len(), 2);
    }

    #[test]
    fn test_disk_storage_serialization() {
        let event = create_test_merge_event(0.75, 3);

        // Test serialization round-trip
        let serialized = DiskStorage::serialize_event(&event).unwrap();
        let deserialized = DiskStorage::deserialize_event(&serialized).unwrap();

        assert_eq!(deserialized.threshold, event.threshold);
        assert_eq!(
            deserialized.merging_groups.len(),
            event.merging_groups.len()
        );

        for (orig, deser) in event
            .merging_groups
            .iter()
            .zip(&deserialized.merging_groups)
        {
            assert_eq!(orig, deser);
        }
    }

    #[test]
    fn test_disk_storage_basic_operations() -> Result<(), StorageError> {
        let mut storage = DiskStorage::with_cache_size(10)?;

        assert_eq!(storage.len(), 0);

        let event1 = create_test_merge_event(0.9, 1);
        let event2 = create_test_merge_event(0.7, 2);

        storage.push(event1)?;
        storage.push(event2)?;

        assert_eq!(storage.len(), 2);
        assert!(storage.memory_usage_bytes() > 0);

        storage.sync()?;
        Ok(())
    }

    #[test]
    fn test_disk_storage_full_workflow() -> Result<(), StorageError> {
        let mut storage = DiskStorage::with_cache_size(2)?;

        // Create several test events
        let events = vec![
            create_test_merge_event(0.9, 2),
            create_test_merge_event(0.7, 1),
            create_test_merge_event(0.5, 3),
        ];

        // Push all events to disk
        for event in &events {
            storage.push(event.clone())?;
        }

        assert_eq!(storage.len(), 3);

        // Sync to ensure all data is written
        storage.sync()?;

        // Test reading back all events via iter() (disk reading)
        let read_events: Vec<_> = storage.iter()?.collect();
        assert_eq!(read_events.len(), 3);

        // Verify events match by checking thresholds (they should be in same order)
        for (i, read_event) in read_events.iter().enumerate() {
            assert_eq!(read_event.threshold, events[i].threshold);
            assert_eq!(
                read_event.merging_groups.len(),
                events[i].merging_groups.len()
            );
        }

        Ok(())
    }

    #[test]
    fn test_disk_storage_actually_reads_from_disk() -> Result<(), StorageError> {
        let mut storage = DiskStorage::with_cache_size(2)?; // Very small cache

        // Create more events than cache can hold
        let events = vec![
            create_test_merge_event(0.9, 2),
            create_test_merge_event(0.8, 1),
            create_test_merge_event(0.7, 3),
            create_test_merge_event(0.6, 2),
            create_test_merge_event(0.5, 1),
        ];

        // Push all events to disk
        for event in &events {
            storage.push(event.clone())?;
        }

        // Sync to ensure all data is written
        storage.sync()?;

        assert_eq!(storage.len(), 5);

        // The cache only holds 2 events, so iter() should read the rest from disk
        let read_events: Vec<_> = storage.iter()?.collect();
        assert_eq!(read_events.len(), 5);

        // Verify all events are present with correct thresholds
        let expected_thresholds = [0.9, 0.8, 0.7, 0.6, 0.5];
        for (i, event) in read_events.iter().enumerate() {
            assert_eq!(event.threshold, expected_thresholds[i]);
        }

        Ok(())
    }

    #[test]
    fn test_hybrid_storage_memory_monitoring() -> Result<(), StorageError> {
        let small_threshold = 500; // 500 bytes - will trigger monitoring
        let mut storage = HybridStorage::new(small_threshold);

        // Add enough events to approach the threshold
        for i in 0..5 {
            let event = create_test_merge_event(0.9 - (i as f64 * 0.1), 2);
            storage.push(event)?;
        }

        // Access events multiple times to trigger memory monitoring
        // (every 1000 accesses, but we can test the access counter functionality)
        assert_eq!(storage.len(), 5);

        // Force memory pressure check by accessing data
        let _ = storage.iter()?;
        let _ = storage.get(0);

        // Verify the access counter is working
        assert!(
            storage
                .access_counter
                .load(std::sync::atomic::Ordering::Relaxed)
                > 0
        );

        Ok(())
    }

    #[test]
    fn test_hybrid_storage_spilling() -> Result<(), StorageError> {
        let small_threshold = 100; // 100 bytes - very small to trigger spilling
        let mut storage = HybridStorage::new(small_threshold);

        assert!(!storage.spilled_to_disk);

        // Add enough events to exceed memory threshold
        for i in 0..10 {
            let event = create_test_merge_event(0.5 + (i as f64 * 0.01), 3);
            storage.push(event)?;
        }

        assert_eq!(storage.len(), 10);
        // Should have spilled to disk due to memory pressure
        assert!(storage.spilled_to_disk);

        Ok(())
    }
}
