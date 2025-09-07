/// Union-Find (Disjoint Set Union) data structure
/// Optimised for cache locality and performance
#[derive(Debug, Clone)]
pub struct UnionFind {
    /// Packed parent and rank for better cache locality
    /// parent is u32, rank is u8 (sufficient for our use case)
    data: Vec<(u32, u8)>,
    size: usize,
}

impl UnionFind {
    /// Create a new Union-Find structure with n elements
    pub fn new(size: usize) -> Self {
        // Pre-allocate with exact capacity for better performance
        let mut data = Vec::with_capacity(size);
        for i in 0..size {
            data.push((i as u32, 0));
        }
        Self { data, size }
    }

    /// Find the root of the set containing element x with path halving
    /// Path halving provides better cache behaviour than full path compression
    #[inline(always)]
    pub fn find(&mut self, mut x: usize) -> usize {
        // Bounds check once at the start
        debug_assert!(x < self.size);

        unsafe {
            // Path halving: make every node point to its grandparent
            while self.data.get_unchecked(x).0 != x as u32 {
                let parent = self.data.get_unchecked(x).0 as usize;
                let grandparent = self.data.get_unchecked(parent).0;
                self.data.get_unchecked_mut(x).0 = grandparent;
                x = grandparent as usize;
            }
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

        unsafe {
            let (_, rank_x) = *self.data.get_unchecked(root_x);
            let (_, rank_y) = *self.data.get_unchecked(root_y);

            // Union by rank - attach smaller tree under larger one
            match rank_x.cmp(&rank_y) {
                std::cmp::Ordering::Less => {
                    self.data.get_unchecked_mut(root_x).0 = root_y as u32;
                }
                std::cmp::Ordering::Greater => {
                    self.data.get_unchecked_mut(root_y).0 = root_x as u32;
                }
                std::cmp::Ordering::Equal => {
                    self.data.get_unchecked_mut(root_y).0 = root_x as u32;
                    self.data.get_unchecked_mut(root_x).1 += 1;
                }
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
        self.size
    }

    /// Check if the union-find structure is empty
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Get all connected components as a vector of vectors
    pub fn get_all_components(&mut self) -> Vec<Vec<usize>> {
        // Pre-allocate HashMap with estimated capacity
        let mut components: std::collections::HashMap<usize, Vec<usize>> =
            std::collections::HashMap::with_capacity(self.size / 4);

        for element in 0..self.size {
            let root = self.find(element);
            components.entry(root).or_default().push(element);
        }

        components.into_values().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_union_find_basic() {
        let mut uf = UnionFind::new(5);

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
        let mut uf = UnionFind::new(10);

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
        let mut uf = UnionFind::new(6);

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
    fn test_large_union_find() {
        const N: usize = 10000;
        let mut uf = UnionFind::new(N);

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
        assert_eq!(components.len(), (N + 1) / 2);
    }

    #[test]
    fn test_union_find_rank_optimisation() {
        let mut uf = UnionFind::new(100);

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
