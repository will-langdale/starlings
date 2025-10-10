use crate::core::key::Key;
use once_cell::sync::OnceCell;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

#[derive(Debug, Clone)]
pub struct InternedRecord {
    pub source_id: u32,
    pub key: Key,
    pub attributes: HashMap<u32, u32>,
    cached_hash: OnceCell<u64>,
}

impl PartialEq for InternedRecord {
    fn eq(&self, other: &Self) -> bool {
        self.source_id == other.source_id
            && self.key == other.key
            && self.attributes == other.attributes
    }
}

impl Eq for InternedRecord {}

impl Hash for InternedRecord {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let hash_value = self.cached_hash.get_or_init(|| {
            use rustc_hash::FxHasher;

            let mut hasher = FxHasher::default();

            self.source_id.hash(&mut hasher);
            self.key.hash(&mut hasher);

            // Order-independent hash for attributes using XOR and multiplication
            // This avoids the expensive sorting operation
            if !self.attributes.is_empty() {
                const GOLDEN_RATIO: u64 = 0x9e3779b97f4a7c15;
                let mut attr_hash: u64 = 0;

                for (k, v) in &self.attributes {
                    let mut pair_hasher = FxHasher::default();
                    k.hash(&mut pair_hasher);
                    v.hash(&mut pair_hasher);
                    let pair_hash = pair_hasher.finish();

                    attr_hash ^= pair_hash.wrapping_mul(GOLDEN_RATIO);
                }

                attr_hash.hash(&mut hasher);
            }

            hasher.finish()
        });

        hash_value.hash(state);
    }
}

impl InternedRecord {
    #[must_use]
    pub fn new(source_id: u32, key: Key) -> Self {
        InternedRecord {
            source_id,
            key,
            attributes: HashMap::new(),
            cached_hash: OnceCell::new(),
        }
    }

    #[must_use]
    pub fn with_attributes(source_id: u32, key: Key, attributes: HashMap<u32, u32>) -> Self {
        InternedRecord {
            source_id,
            key,
            attributes,
            cached_hash: OnceCell::new(),
        }
    }

    pub fn add_attribute(&mut self, key: u32, value: u32) {
        self.attributes.insert(key, value);
    }

    pub fn get_attribute(&self, key: u32) -> Option<&u32> {
        self.attributes.get(&key)
    }

    pub fn source_id(&self) -> u32 {
        self.source_id
    }

    pub fn key(&self) -> &Key {
        &self.key
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_record_creation() {
        let key = Key::InternedString(42); // Pretend 42 = "test_key"
        let record = InternedRecord::new(1, key.clone());

        assert_eq!(record.source_id, 1);
        assert_eq!(record.key, key);
        assert!(record.attributes.is_empty());
    }

    #[test]
    fn test_record_with_attributes() {
        let key = Key::U32(42);
        let mut attrs = HashMap::new();
        attrs.insert(1, 100);
        attrs.insert(2, 200);

        let record = InternedRecord::with_attributes(5, key.clone(), attrs.clone());

        assert_eq!(record.source_id, 5);
        assert_eq!(record.key, key);
        assert_eq!(record.attributes.len(), 2);
        assert_eq!(record.get_attribute(1), Some(&100));
        assert_eq!(record.get_attribute(2), Some(&200));
        assert_eq!(record.get_attribute(3), None);
    }

    #[test]
    fn test_record_add_attribute() {
        let key = Key::Bytes(vec![1, 2, 3]);
        let mut record = InternedRecord::new(10, key);

        record.add_attribute(1, 42);
        record.add_attribute(2, 84);

        assert_eq!(record.attributes.len(), 2);
        assert_eq!(record.get_attribute(1), Some(&42));
        assert_eq!(record.get_attribute(2), Some(&84));
    }

    #[test]
    fn test_record_equality() {
        let key1 = Key::U64(1000);
        let key2 = Key::U64(1000);

        let record1 = InternedRecord::new(1, key1);
        let record2 = InternedRecord::new(1, key2);

        assert_eq!(record1, record2);

        let mut record3 = record1.clone();
        record3.add_attribute(1, 42);

        assert_ne!(record1, record3);
    }
}
