use crate::core::key::Key;
use crate::core::record::InternedRecord;
use boxcar::Vec as BoxcarVec;
use dashmap::DashMap;
use lasso::{Capacity, Key as LassoKey, ThreadedRodeo};
use roaring::RoaringBitmap;
use rustc_hash::FxHasher;
use std::collections::HashMap;
use std::hash::BuildHasherDefault;
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;

type FxDashMap<K, V> = DashMap<K, V, BuildHasherDefault<FxHasher>>;

#[derive(Debug)]
pub struct DataContext {
    pub records: BoxcarVec<InternedRecord>,
    pub source_interner: Arc<ThreadedRodeo>, // For source names
    pub string_interner: Arc<ThreadedRodeo>, // For record keys
    pub identity_map: FxDashMap<InternedRecord, u32>,
    pub source_index: FxDashMap<u32, RoaringBitmap>,
    next_record_id: AtomicU32,
    /// Generation counter for cache invalidation
    /// Incremented on any structural change (compaction, record addition, etc.)
    generation: AtomicU64,
}

impl DataContext {
    /// Convert LassoKey to u32 safely, panicking only on systems with massive string interners
    #[inline]
    fn lasso_key_to_u32(key: lasso::Spur) -> u32 {
        LassoKey::into_usize(key)
            .try_into()
            .expect("String interner exceeded u32 capacity - consider using u64 identifiers")
    }

    #[must_use]
    pub fn new() -> Self {
        Self::with_capacity(0)
    }

    /// Get the current generation for cache validation
    pub fn generation(&self) -> u64 {
        self.generation.load(Ordering::Acquire)
    }

    /// Increment generation to invalidate caches
    pub fn increment_generation(&self) {
        self.generation.fetch_add(1, Ordering::Release);
    }

    /// Create DataContext with pre-allocated capacity for better performance
    #[must_use]
    pub fn with_capacity(estimated_records: usize) -> Self {
        let hasher = BuildHasherDefault::<FxHasher>::default();

        // CRITICAL FIX: Use Capacity::new(strings, bytes) instead of for_strings()
        // for_strings() over-allocates by 32x (31GB for 350k strings!)
        // We use explicit byte limits: ~100 bytes per string (2x safety margin over actual ~50 bytes)

        // String interner: conservative byte allocation
        let string_interner = if estimated_records > 0 {
            // Calculate conservative byte allocation: 100 bytes per string
            let string_bytes = NonZeroUsize::new((estimated_records * 100).max(4096)).expect(
                "Failed to create NonZeroUsize for string interner capacity (minimum 4096 bytes)",
            );

            Arc::new(ThreadedRodeo::with_capacity(Capacity::new(
                estimated_records,
                string_bytes,
            )))
        } else {
            // No pre-allocation for empty contexts
            Arc::new(ThreadedRodeo::new())
        };

        DataContext {
            records: BoxcarVec::new(),
            // Source interner: small, no pre-allocation needed
            source_interner: Arc::new(ThreadedRodeo::new()),
            string_interner,
            identity_map: DashMap::with_capacity_and_hasher(estimated_records, hasher.clone()),
            source_index: DashMap::with_hasher(hasher),
            next_record_id: AtomicU32::new(0),
            generation: AtomicU64::new(0),
        }
    }

    /// Batch ensure records for improved performance
    pub fn ensure_records_batch(&self, source: &str, keys: &[Key]) -> Vec<u32> {
        let source_id = Self::lasso_key_to_u32(self.source_interner.get_or_intern(source));
        let mut result = Vec::with_capacity(keys.len());

        for key in keys {
            let record = InternedRecord::new(source_id, key.clone());

            // Fast path: check if record already exists
            if let Some(existing_id) = self.identity_map.get(&record) {
                result.push(*existing_id);
                continue;
            }

            // Slow path: need to insert new record
            let record_id = self.next_record_id.fetch_add(1, Ordering::Relaxed);

            match self.identity_map.entry(record) {
                dashmap::mapref::entry::Entry::Occupied(entry) => {
                    // Another thread inserted it while we were working
                    result.push(*entry.get());
                }
                dashmap::mapref::entry::Entry::Vacant(entry) => {
                    let record = entry.key().clone(); // Only clone when inserting
                    entry.insert(record_id);

                    self.records.push(record);

                    self.source_index
                        .entry(source_id)
                        .or_default()
                        .insert(record_id);

                    result.push(record_id);
                }
            }
        }

        result
    }

    /// Thread-safe record interning with lock-free operations
    pub fn ensure_record(&self, source: &str, key: Key) -> u32 {
        let source_id = Self::lasso_key_to_u32(self.source_interner.get_or_intern(source));

        let record = InternedRecord::new(source_id, key);

        if let Some(existing_id) = self.identity_map.get(&record) {
            return *existing_id;
        }

        let record_id = self.next_record_id.fetch_add(1, Ordering::Relaxed);

        match self.identity_map.entry(record.clone()) {
            dashmap::mapref::entry::Entry::Occupied(entry) => *entry.get(),
            dashmap::mapref::entry::Entry::Vacant(entry) => {
                entry.insert(record_id);

                self.records.push(record);

                self.source_index
                    .entry(source_id)
                    .or_default()
                    .insert(record_id);

                record_id
            }
        }
    }

    /// Thread-safe record interning with attributes
    pub fn ensure_record_with_attributes(
        &self,
        source: &str,
        key: Key,
        attributes: HashMap<String, String>,
    ) -> u32 {
        let source_id = Self::lasso_key_to_u32(self.source_interner.get_or_intern(source));

        let mut interned_attrs = HashMap::new();
        for (k, v) in attributes {
            let key_id = Self::lasso_key_to_u32(self.source_interner.get_or_intern(k));
            let val_id = Self::lasso_key_to_u32(self.source_interner.get_or_intern(v));
            interned_attrs.insert(key_id, val_id);
        }

        let record = InternedRecord::with_attributes(source_id, key, interned_attrs);

        if let Some(existing_id) = self.identity_map.get(&record) {
            return *existing_id;
        }

        let record_id = self.next_record_id.fetch_add(1, Ordering::Relaxed);

        match self.identity_map.entry(record.clone()) {
            dashmap::mapref::entry::Entry::Occupied(entry) => *entry.get(),
            dashmap::mapref::entry::Entry::Vacant(entry) => {
                entry.insert(record_id);

                self.records.push(record);

                self.source_index
                    .entry(source_id)
                    .or_default()
                    .insert(record_id);

                record_id
            }
        }
    }

    pub fn get_record(&self, id: u32) -> Option<InternedRecord> {
        self.records.get(id as usize).map(|r| (*r).clone())
    }

    pub fn len(&self) -> usize {
        self.next_record_id.load(Ordering::Relaxed) as usize
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Check if it's safe to perform an operation with the given memory requirements
    ///
    /// # Errors
    /// Returns an error if the operation would exceed available system resources
    pub fn check_operation_safety(&self, estimated_records: usize) -> Result<(), String> {
        use crate::core::safety::ensure_memory_safety;
        // Estimate ~1MB per 1000 records as a rough approximation
        let estimated_mb = (estimated_records / 1000).max(1) as u64;
        ensure_memory_safety(estimated_mb).map_err(|e| e.to_string())
    }

    /// Check current memory pressure and return true if we should throttle
    pub fn should_throttle(&self) -> bool {
        use crate::core::safety::global_resource_monitor;
        let usage = global_resource_monitor().get_usage();
        !usage.memory_under_limit
    }

    /// Wait for resources if under pressure, with exponential backoff
    pub fn wait_for_resources(&self) {
        if self.should_throttle() {
            // Simple delay when near memory limit
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
    }

    /// Create a deep copy of this DataContext
    ///
    /// This is used when creating an owned copy of a collection with its own context.
    pub fn deep_copy(&self) -> Self {
        // Clone source interner - collect and sort by key to preserve ID order
        let new_source_interner = Arc::new(ThreadedRodeo::with_capacity(Capacity::for_strings(
            self.source_interner.len(),
        )));
        let mut source_entries: Vec<_> = self.source_interner.iter().collect();
        source_entries.sort_by_key(|(key, _)| LassoKey::into_usize(*key));
        for (_key, string) in source_entries {
            new_source_interner.get_or_intern(string);
        }

        // Clone string interner - collect and sort by key to preserve ID order
        let new_string_interner = Arc::new(ThreadedRodeo::with_capacity(Capacity::for_strings(
            self.string_interner.len(),
        )));
        let mut string_entries: Vec<_> = self.string_interner.iter().collect();
        string_entries.sort_by_key(|(key, _)| LassoKey::into_usize(*key));
        for (_key, string) in string_entries {
            new_string_interner.get_or_intern(string);
        }

        // Create new DataContext with cloned data
        let record_count = self.len();
        let mut new_context = DataContext::with_capacity(record_count);
        new_context.source_interner = new_source_interner;
        new_context.string_interner = new_string_interner;

        // Copy all records
        for (_idx, record) in self.records.iter() {
            new_context.records.push(record.clone());
        }

        // Rebuild identity map and source index
        for (idx, record) in new_context.records.iter() {
            let idx = idx as u32;
            new_context.identity_map.insert(record.clone(), idx);
            new_context
                .source_index
                .entry(record.source_id())
                .or_default()
                .insert(idx);
        }

        // Set the correct next_record_id
        new_context
            .next_record_id
            .store(self.len() as u32, Ordering::Relaxed);

        // Start with a fresh generation for the new context
        new_context.generation.store(0, Ordering::Relaxed);

        new_context
    }

    /// Get adaptive batch size for current resource conditions
    pub fn get_adaptive_batch_size(&self, default_size: usize) -> usize {
        use crate::core::safety::global_resource_monitor;
        let usage = global_resource_monitor().get_usage();

        // Simple adaptive sizing: reduce batch size if near limit
        // Always ensure at least 1 to avoid step_by(0) panic
        let size = if !usage.memory_under_limit {
            default_size / 4 // Quarter size when at limit
        } else if usage.memory_percent > 60.0 {
            default_size / 2 // Half size when getting close
        } else {
            default_size // Full size when plenty of headroom
        };

        size.max(1) // Never return 0
    }

    /// Intern a string and return its ID
    ///
    /// This is the ONLY place where strings are converted to InternedString keys.
    /// Thread-safe: ThreadedRodeo handles concurrent access.
    pub fn intern_string(&self, s: &str) -> u32 {
        let spur = self.string_interner.get_or_intern(s);
        Self::lasso_key_to_u32(spur)
    }

    /// Resolve an interned string ID back to its string value
    ///
    /// Used for debugging, display, and Python conversion.
    pub fn resolve_string(&self, id: u32) -> Option<String> {
        match LassoKey::try_from_usize(id as usize) {
            Some(spur) => match self.string_interner.try_resolve(&spur) {
                Some(s) => Some(s.to_string()),
                None => {
                    #[cfg(debug_assertions)]
                    eprintln!(
                        "⚠️  Failed to resolve string ID {}: Spur exists but not in interner",
                        id
                    );
                    None
                }
            },
            None => {
                #[cfg(debug_assertions)]
                eprintln!(
                    "⚠️  Failed to resolve string ID {}: Invalid Spur conversion",
                    id
                );
                None
            }
        }
    }

    pub fn get_source_name(&self, source_id: u32) -> Option<String> {
        let spur = LassoKey::try_from_usize(source_id as usize)?;
        self.source_interner
            .try_resolve(&spur)
            .map(|s| s.to_string())
    }

    pub fn get_records_by_source(&self, source_name: &str) -> Option<Vec<u32>> {
        let source_id = Self::lasso_key_to_u32(self.source_interner.get(source_name)?);
        self.source_index
            .get(&source_id)
            .map(|bitmap| bitmap.iter().collect())
    }

    /// Reserve space for the expected number of records (for performance)
    pub fn reserve(&self, _additional: usize) {}
}

impl Default for DataContext {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_record_deduplication() {
        let ctx = DataContext::new();

        let key1_id = ctx.intern_string("key1");
        let key2_id = ctx.intern_string("key2");

        let id1 = ctx.ensure_record("source1", Key::InternedString(key1_id));
        let id2 = ctx.ensure_record("source1", Key::InternedString(key1_id));
        let id3 = ctx.ensure_record("source1", Key::InternedString(key2_id));

        assert_eq!(id1, id2);
        assert_ne!(id1, id3);
        assert_eq!(ctx.len(), 2);
    }

    #[test]
    fn test_different_sources_different_records() {
        let ctx = DataContext::new();

        let id1 = ctx.ensure_record("source1", Key::U32(42));
        let id2 = ctx.ensure_record("source2", Key::U32(42));

        assert_ne!(id1, id2);
        assert_eq!(ctx.len(), 2);
    }

    #[test]
    fn test_source_index() {
        let ctx = DataContext::new();

        let a_id = ctx.intern_string("a");
        let b_id = ctx.intern_string("b");
        let c_id = ctx.intern_string("c");
        let d_id = ctx.intern_string("d");

        ctx.ensure_record("source1", Key::InternedString(a_id));
        ctx.ensure_record("source1", Key::InternedString(b_id));
        ctx.ensure_record("source2", Key::InternedString(c_id));
        ctx.ensure_record("source1", Key::InternedString(d_id));

        let source1_records = ctx.get_records_by_source("source1").unwrap();
        let source2_records = ctx.get_records_by_source("source2").unwrap();

        assert_eq!(source1_records.len(), 3);
        assert_eq!(source2_records.len(), 1);
    }

    #[test]
    fn test_record_with_attributes() {
        let ctx = DataContext::new();

        let mut attrs1 = HashMap::new();
        attrs1.insert("name".to_string(), "Alice".to_string());
        attrs1.insert("age".to_string(), "30".to_string());

        let mut attrs2 = HashMap::new();
        attrs2.insert("name".to_string(), "Alice".to_string());
        attrs2.insert("age".to_string(), "30".to_string());

        let mut attrs3 = HashMap::new();
        attrs3.insert("name".to_string(), "Bob".to_string());

        let id1 = ctx.ensure_record_with_attributes("people", Key::U32(1), attrs1);
        let id2 = ctx.ensure_record_with_attributes("people", Key::U32(1), attrs2);
        let id3 = ctx.ensure_record_with_attributes("people", Key::U32(1), attrs3);

        assert_eq!(id1, id2);
        assert_ne!(id1, id3);
    }

    #[test]
    fn test_index_stability() {
        let ctx = DataContext::new();

        let ids: Vec<u32> = (0..100)
            .map(|i| ctx.ensure_record("test", Key::U32(i)))
            .collect();

        for (i, &id) in ids.iter().enumerate() {
            assert_eq!(id, i as u32);
            let record = ctx.get_record(id).unwrap();
            assert_eq!(record.key, Key::U32(i as u32));
        }
    }

    #[test]
    fn test_get_source_name() {
        let ctx = DataContext::new();

        ctx.ensure_record("source_a", Key::U32(1));
        ctx.ensure_record("source_b", Key::U32(2));

        assert_eq!(ctx.get_source_name(0), Some("source_a".to_string()));
        assert_eq!(ctx.get_source_name(1), Some("source_b".to_string()));
        assert_eq!(ctx.get_source_name(999), None);
    }

    #[test]
    fn test_string_interning() {
        let ctx = DataContext::new();

        // Same string gets same ID
        let id1 = ctx.intern_string("test");
        let id2 = ctx.intern_string("test");
        assert_eq!(id1, id2);

        // Different strings get different IDs
        let id3 = ctx.intern_string("other");
        assert_ne!(id1, id3);

        // Can resolve back to string
        assert_eq!(ctx.resolve_string(id1), Some("test".to_string()));
        assert_eq!(ctx.resolve_string(id3), Some("other".to_string()));
    }

    #[test]
    fn test_record_with_interned_string() {
        let ctx = DataContext::new();
        let key_id = ctx.intern_string("customer_123");
        let key = Key::InternedString(key_id);

        let id1 = ctx.ensure_record("source1", key.clone());
        let id2 = ctx.ensure_record("source1", key);

        assert_eq!(id1, id2); // Same record
        assert_eq!(ctx.len(), 1);
    }

    #[test]
    fn test_deep_copy_preserves_all_strings() {
        let ctx = DataContext::new();

        // Intern a bunch of strings
        let strings = vec!["foo", "bar", "baz", "qux", "test_123", "another"];
        let mut ids = Vec::new();
        for s in &strings {
            let id = ctx.intern_string(s);
            ids.push(id);
        }

        // Deep copy
        let ctx2 = ctx.deep_copy();

        // Verify all strings resolve in both contexts
        for (&id, &expected) in ids.iter().zip(strings.iter()) {
            let resolved1 = ctx
                .resolve_string(id)
                .expect("Original context should resolve");
            let resolved2 = ctx2
                .resolve_string(id)
                .expect("Copied context should resolve");

            assert_eq!(resolved1, expected);
            assert_eq!(resolved2, expected);
            assert_eq!(resolved1, resolved2);
        }

        // Verify lengths match
        assert_eq!(ctx.string_interner.len(), ctx2.string_interner.len());
    }

    #[test]
    fn test_deep_copy_with_potential_gaps() {
        let ctx = DataContext::new();

        // Intern many strings to potentially create gaps
        for i in 0..1000 {
            ctx.intern_string(&format!("string_{}", i));
        }

        // Deep copy should not panic
        let ctx2 = ctx.deep_copy();

        // Lengths should match
        assert_eq!(ctx.string_interner.len(), ctx2.string_interner.len());
    }

    #[test]
    fn test_deep_copy_with_records() {
        let ctx = DataContext::new();

        // Create some records with string keys
        let key1_id = ctx.intern_string("record_0");
        let key2_id = ctx.intern_string("record_1");
        let key3_id = ctx.intern_string("record_2");

        let rec1 = ctx.ensure_record("source_a", Key::InternedString(key1_id));
        let rec2 = ctx.ensure_record("source_a", Key::InternedString(key2_id));
        let rec3 = ctx.ensure_record("source_b", Key::InternedString(key3_id));

        // Deep copy should preserve everything
        let ctx2 = ctx.deep_copy();

        // Verify all strings are preserved
        assert_eq!(ctx2.resolve_string(key1_id), Some("record_0".to_string()));
        assert_eq!(ctx2.resolve_string(key2_id), Some("record_1".to_string()));
        assert_eq!(ctx2.resolve_string(key3_id), Some("record_2".to_string()));

        // Verify source names are preserved
        assert_eq!(ctx2.get_source_name(0), Some("source_a".to_string()));
        assert_eq!(ctx2.get_source_name(1), Some("source_b".to_string()));

        // Verify records exist
        assert_eq!(ctx2.len(), 3);
        let copied_rec1 = ctx2.get_record(rec1).unwrap();
        let copied_rec2 = ctx2.get_record(rec2).unwrap();
        let copied_rec3 = ctx2.get_record(rec3).unwrap();

        assert_eq!(copied_rec1.key, Key::InternedString(key1_id));
        assert_eq!(copied_rec2.key, Key::InternedString(key2_id));
        assert_eq!(copied_rec3.key, Key::InternedString(key3_id));
    }
}
