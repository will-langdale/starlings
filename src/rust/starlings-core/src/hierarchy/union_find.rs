use memmap2;

/// Backend trait for union-find data storage
pub trait UnionFindBackend {
    fn get_parent(&self, i: usize) -> u32;
    fn set_parent(&mut self, i: usize, parent: u32);
    fn get_rank(&self, i: usize) -> u8;
    fn set_rank(&mut self, i: usize, rank: u8);
    fn size(&self) -> usize;
}

/// In-memory backend using Vec for small to medium datasets
#[derive(Debug, Clone)]
pub struct VecBackend {
    /// Packed parent and rank for better cache locality
    /// parent is u32, rank is u8 (sufficient for our use case)
    data: Vec<(u32, u8)>,
}

/// Memory-mapped backend for large datasets that exceed available RAM
#[derive(Debug)]
pub struct MmapBackend {
    mmap: memmap2::MmapMut,
    size: usize,
    _file: std::fs::File, // Keep file handle alive
}

/// Union-Find (Disjoint Set Union) data structure with pluggable backends.
/// Optimised for cache locality and performance with path halving and union by rank.
#[derive(Debug)]
pub struct UnionFind<B: UnionFindBackend> {
    backend: B,
}

impl VecBackend {
    fn new(size: usize) -> Self {
        let mut data = Vec::with_capacity(size);
        for i in 0..size {
            data.push((i as u32, 0));
        }
        Self { data }
    }
}

impl UnionFindBackend for VecBackend {
    fn get_parent(&self, i: usize) -> u32 {
        self.data[i].0
    }

    fn set_parent(&mut self, i: usize, parent: u32) {
        self.data[i].0 = parent;
    }

    fn get_rank(&self, i: usize) -> u8 {
        self.data[i].1
    }

    fn set_rank(&mut self, i: usize, rank: u8) {
        self.data[i].1 = rank;
    }

    fn size(&self) -> usize {
        self.data.len()
    }
}

impl MmapBackend {
    fn new(size: usize) -> Result<Self, Box<dyn std::error::Error>> {
        // Each element needs 5 bytes: 4 bytes parent (u32) + 1 byte rank (u8)
        let file_size = size * 5;

        // Create temporary file
        let file = tempfile::tempfile()?;
        file.set_len(file_size as u64)?;

        // Memory map the file
        let mut mmap = unsafe { memmap2::MmapMut::map_mut(&file)? };

        // Initialise each element as its own parent with rank 0
        for i in 0..size {
            let offset = i * 5;
            // Write parent (u32, little endian)
            mmap[offset..offset + 4].copy_from_slice(&(i as u32).to_le_bytes());
            // Write rank (u8)
            mmap[offset + 4] = 0;
        }

        Ok(Self {
            mmap,
            size,
            _file: file,
        })
    }
}

impl UnionFindBackend for MmapBackend {
    fn get_parent(&self, i: usize) -> u32 {
        debug_assert!(i < self.size);
        let offset = i * 5;
        u32::from_le_bytes([
            self.mmap[offset],
            self.mmap[offset + 1],
            self.mmap[offset + 2],
            self.mmap[offset + 3],
        ])
    }

    fn set_parent(&mut self, i: usize, parent: u32) {
        debug_assert!(i < self.size);
        let offset = i * 5;
        let bytes = parent.to_le_bytes();
        self.mmap[offset..offset + 4].copy_from_slice(&bytes);
    }

    fn get_rank(&self, i: usize) -> u8 {
        debug_assert!(i < self.size);
        let offset = i * 5 + 4;
        self.mmap[offset]
    }

    fn set_rank(&mut self, i: usize, rank: u8) {
        debug_assert!(i < self.size);
        let offset = i * 5 + 4;
        self.mmap[offset] = rank;
    }

    fn size(&self) -> usize {
        self.size
    }
}

impl UnionFind<VecBackend> {
    /// Create a new Union-Find structure with Vec backend (for small datasets)
    pub fn new_vec(size: usize) -> Self {
        Self {
            backend: VecBackend::new(size),
        }
    }
}

impl UnionFind<MmapBackend> {
    /// Create a new Union-Find structure with memory-mapped backend (for large datasets)
    pub fn new_mmap(size: usize) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self {
            backend: MmapBackend::new(size)?,
        })
    }
}

impl<B: UnionFindBackend + Clone> Clone for UnionFind<B> {
    fn clone(&self) -> Self {
        Self {
            backend: self.backend.clone(),
        }
    }
}

impl<B: UnionFindBackend> UnionFind<B> {
    /// Find the root of the set containing element x with path halving.
    /// Path halving provides better cache behaviour than full path compression.
    #[inline(always)]
    pub fn find(&mut self, mut x: usize) -> usize {
        debug_assert!(x < self.backend.size());

        // Path halving: make every node point to its grandparent
        while self.backend.get_parent(x) != x as u32 {
            let parent = self.backend.get_parent(x) as usize;
            let grandparent = self.backend.get_parent(parent);
            self.backend.set_parent(x, grandparent);
            x = grandparent as usize;
        }
        x
    }

    /// Union two sets containing elements x and y
    #[inline(always)]
    pub fn union(&mut self, x: usize, y: usize) -> bool {
        let root_x = self.find(x);
        let root_y = self.find(y);

        if root_x == root_y {
            return false;
        }

        let rank_x = self.backend.get_rank(root_x);
        let rank_y = self.backend.get_rank(root_y);

        // Union by rank - attach smaller tree under larger one
        match rank_x.cmp(&rank_y) {
            std::cmp::Ordering::Less => {
                self.backend.set_parent(root_x, root_y as u32);
            }
            std::cmp::Ordering::Greater => {
                self.backend.set_parent(root_y, root_x as u32);
            }
            std::cmp::Ordering::Equal => {
                self.backend.set_parent(root_y, root_x as u32);
                self.backend.set_rank(root_x, rank_x + 1);
            }
        }

        true
    }

    /// Check if two elements are in the same set
    #[inline]
    pub fn connected(&mut self, x: usize, y: usize) -> bool {
        self.find(x) == self.find(y)
    }

    /// Get the number of elements in the union-find structure
    pub fn len(&self) -> usize {
        self.backend.size()
    }

    /// Check if the union-find structure is empty
    pub fn is_empty(&self) -> bool {
        self.backend.size() == 0
    }

    /// Get all connected components as a vector of vectors
    pub fn get_all_components(&mut self) -> Vec<Vec<usize>> {
        let size = self.backend.size();
        // Pre-allocate HashMap with estimated capacity
        let mut components: std::collections::HashMap<usize, Vec<usize>> =
            std::collections::HashMap::with_capacity(size / 4);

        for element in 0..size {
            let root = self.find(element);
            components.entry(root).or_default().push(element);
        }

        components.into_values().collect()
    }
}

// Legacy type alias for backward compatibility
pub type MmapUnionFind = UnionFind<MmapBackend>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_union_find_basic() {
        let mut uf = UnionFind::new_vec(5);

        // Initially, each element is its own root
        for i in 0..5 {
            assert_eq!(uf.find(i), i);
        }

        // Union some elements
        assert!(uf.union(0, 1)); // Returns true - union performed
        assert!(!uf.union(0, 1)); // Returns false - already connected

        assert_eq!(uf.find(0), uf.find(1)); // Same root
        assert_ne!(uf.find(0), uf.find(2)); // Different roots
    }

    #[test]
    fn test_union_find_path_compression() {
        let mut uf = UnionFind::new_vec(10);

        // Create a chain: 0->1->2->3->4
        uf.union(0, 1);
        uf.union(1, 2);
        uf.union(2, 3);
        uf.union(3, 4);

        // All should have the same root
        let root = uf.find(0);
        for i in 0..5 {
            assert_eq!(uf.find(i), root);
        }

        // Path compression should have flattened the structure
        // (verified by the fact that subsequent finds are fast)
    }

    #[test]
    fn test_connected_components() {
        let mut uf = UnionFind::new_vec(6);

        // Create two components: {0,1,2} and {3,4}, with 5 isolated
        uf.union(0, 1);
        uf.union(1, 2);
        uf.union(3, 4);

        // Test connectivity
        assert!(uf.connected(0, 2));
        assert!(uf.connected(3, 4));
        assert!(!uf.connected(0, 3));
        assert!(!uf.connected(2, 5));

        // Test component collection
        let components = uf.get_all_components();
        assert_eq!(components.len(), 3); // Three components

        // Find the component sizes
        let mut sizes: Vec<usize> = components.iter().map(|c| c.len()).collect();
        sizes.sort();
        assert_eq!(sizes, vec![1, 2, 3]); // Sizes: 1 (isolated), 2, 3
    }

    #[test]
    fn test_mmap_union_find_basic() {
        let mut uf = UnionFind::new_mmap(5).unwrap();

        // Initially, each element is its own root
        for i in 0..5 {
            assert_eq!(uf.find(i), i);
        }

        // Union some elements
        assert!(uf.union(0, 1)); // Returns true - union performed
        assert!(!uf.union(0, 1)); // Returns false - already connected

        // Check that 0 and 1 are now connected
        assert_eq!(uf.find(0), uf.find(1));

        // Union more elements
        assert!(uf.union(2, 3));
        assert!(uf.union(1, 2)); // Connect the two components

        // Now 0, 1, 2, 3 should all be connected
        let root = uf.find(0);
        assert_eq!(uf.find(1), root);
        assert_eq!(uf.find(2), root);
        assert_eq!(uf.find(3), root);

        // Element 4 should still be separate
        assert_ne!(uf.find(4), root);
    }

    #[test]
    fn test_mmap_union_find_large() {
        // Test with a larger dataset to verify memory mapping works
        let size = 10_000;
        let mut uf = UnionFind::new_mmap(size).unwrap();

        // Connect elements in pairs: (0,1), (2,3), (4,5), etc.
        for i in (0..size - 1).step_by(2) {
            uf.union(i, i + 1);
        }

        // Verify pairs are connected
        for i in (0..size - 1).step_by(2) {
            assert_eq!(uf.find(i), uf.find(i + 1));
        }

        // Verify different pairs are not connected
        if size > 3 {
            assert_ne!(uf.find(0), uf.find(2));
        }
    }

    #[test]
    fn test_large_union_find() {
        const N: usize = 10000;
        let mut uf = UnionFind::new_vec(N);

        // Connect pairs: (0,1), (2,3), (4,5), ...
        for i in (0..N).step_by(2) {
            if i + 1 < N {
                uf.union(i, i + 1);
            }
        }

        // Verify connections
        for i in (0..N).step_by(2) {
            if i + 1 < N {
                assert!(uf.connected(i, i + 1));
                if i + 2 < N {
                    assert!(!uf.connected(i, i + 2));
                }
            }
        }

        // Should have approximately N/2 components
        let components = uf.get_all_components();
        assert_eq!(components.len(), N.div_ceil(2));
    }

    #[test]
    fn test_union_find_rank_optimisation() {
        let mut uf = UnionFind::new_vec(100);

        // Test basic union by rank: when trees have different ranks,
        // the root of the shorter tree should point to the root of the taller tree

        // Create two trees with different heights
        // Tree 1: Chain of length 3 (height ≈ 2)
        uf.union(0, 1);
        uf.union(1, 2);

        // Tree 2: Simple pair (height = 1)
        uf.union(10, 11);

        // Union the trees - all should end up in the same component
        uf.union(0, 10);

        // Test that all elements are connected (exact root doesn't matter)
        let root = uf.find(0);
        assert_eq!(uf.find(1), root);
        assert_eq!(uf.find(2), root);
        assert_eq!(uf.find(10), root);
        assert_eq!(uf.find(11), root);

        // Test connectivity
        assert!(uf.connected(0, 11));
        assert!(uf.connected(2, 10));
    }
}
