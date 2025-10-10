use std::hash::{Hash, Hasher};

#[derive(Debug, Clone)]
pub enum Key {
    U32(u32),
    U64(u64),
    InternedString(u32), // ID into DataContext.string_interner
    Bytes(Vec<u8>),
}

impl PartialEq for Key {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Key::U32(a), Key::U32(b)) => a == b,
            (Key::U64(a), Key::U64(b)) => a == b,
            (Key::InternedString(a), Key::InternedString(b)) => a == b,
            (Key::Bytes(a), Key::Bytes(b)) => a == b,
            _ => false,
        }
    }
}

impl Eq for Key {}

impl Hash for Key {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            Key::U32(v) => v.hash(state),
            Key::U64(v) => v.hash(state),
            Key::InternedString(id) => id.hash(state),
            Key::Bytes(v) => v.hash(state),
        }
    }
}

impl Key {
    #[must_use]
    pub fn from_u32(v: u32) -> Self {
        Key::U32(v)
    }

    #[must_use]
    pub fn from_u64(v: u64) -> Self {
        Key::U64(v)
    }

    #[must_use]
    pub fn from_bytes(v: Vec<u8>) -> Self {
        Key::Bytes(v)
    }

    /// Create interned string key (requires DataContext to actually intern)
    #[must_use]
    pub fn from_interned_string(id: u32) -> Self {
        Key::InternedString(id)
    }

    /// Check if this key is an interned string
    #[must_use]
    pub fn is_interned_string(&self) -> bool {
        matches!(self, Key::InternedString(_))
    }

    /// Get the interner ID if this is an interned string
    #[must_use]
    pub fn as_interned_string(&self) -> Option<u32> {
        match self {
            Key::InternedString(id) => Some(*id),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_key_equality() {
        let k1 = Key::U32(42);
        let k2 = Key::U32(42);
        let k3 = Key::U32(43);
        assert_eq!(k1, k2);
        assert_ne!(k1, k3);

        // Interned strings with same ID are equal
        let s1 = Key::InternedString(42);
        let s2 = Key::InternedString(42);
        let s3 = Key::InternedString(43);
        assert_eq!(s1, s2);
        assert_ne!(s1, s3);

        let b1 = Key::Bytes(vec![1, 2, 3]);
        let b2 = Key::Bytes(vec![1, 2, 3]);
        let b3 = Key::Bytes(vec![1, 2, 4]);
        assert_eq!(b1, b2);
        assert_ne!(b1, b3);

        let u64_1 = Key::U64(1000000);
        let u64_2 = Key::U64(1000000);
        assert_eq!(u64_1, u64_2);
    }

    #[test]
    fn test_key_hash() {
        use std::collections::HashSet;

        let mut set = HashSet::new();
        set.insert(Key::U32(42));
        set.insert(Key::InternedString(100));
        set.insert(Key::Bytes(vec![1, 2, 3]));

        assert!(set.contains(&Key::U32(42)));
        assert!(set.contains(&Key::InternedString(100)));
        assert!(!set.contains(&Key::InternedString(101)));
        assert!(set.contains(&Key::Bytes(vec![1, 2, 3])));
        assert!(!set.contains(&Key::U32(43)));
    }

    #[test]
    fn test_different_types_not_equal() {
        let k1 = Key::U32(42);
        let k2 = Key::U64(42);
        let k3 = Key::InternedString(42);

        assert_ne!(k1, k2);
        assert_ne!(k1, k3);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_interned_string_helpers() {
        let key = Key::InternedString(123);
        assert!(key.is_interned_string());
        assert_eq!(key.as_interned_string(), Some(123));

        let key_u32 = Key::U32(42);
        assert!(!key_u32.is_interned_string());
        assert_eq!(key_u32.as_interned_string(), None);
    }
}
