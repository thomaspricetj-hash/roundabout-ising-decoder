#[derive(Clone)]
pub struct LatticeGeometry {
    pub width: usize,
    pub height: usize,
}

impl LatticeGeometry {
    /// Construct geometry with strict invariants.
    #[inline(always)]
    pub fn new(width: usize, height: usize) -> Self {
        assert!(width > 0 && height > 0, "LatticeGeometry: dimensions must be non-zero");
        Self { width, height }
    }

    /// Total number of sites.
    #[inline(always)]
    pub fn num_sites(&self) -> usize {
        self.width * self.height
    }

    /// Convert (x, y) → index.
    #[inline(always)]
    pub fn to_index(&self, x: usize, y: usize) -> usize {
        debug_assert!(x < self.width && y < self.height);
        y * self.width + x
    }

    /// Convert index → (x, y).
    #[inline(always)]
    pub fn to_xy(&self, idx: usize) -> (usize, usize) {
        debug_assert!(idx < self.num_sites());
        (idx % self.width, idx / self.width)
    }

    /// Check bounds.
    #[inline(always)]
    pub fn in_bounds(&self, x: usize, y: usize) -> bool {
        x < self.width && y < self.height
    }

    /// Fast neighbor indices (no heap, no SmallVec).
    #[inline(always)]
    pub fn neighbors(&self, idx: usize) -> [Option<usize>; 4] {
        let (x, y) = self.to_xy(idx);

        let left  = if x > 0 { Some(idx - 1) } else { None };
        let right = if x + 1 < self.width { Some(idx + 1) } else { None };
        let up    = if y > 0 { Some(idx - self.width) } else { None };
        let down  = if y + 1 < self.height { Some(idx + self.width) } else { None };

        [left, right, up, down]
    }

    /// Classification helpers.
    #[inline(always)]
    pub fn is_edge(&self, idx: usize) -> bool {
        let (x, y) = self.to_xy(idx);
        x == 0 || x == self.width - 1 || y == 0 || y == self.height - 1
    }

    #[inline(always)]
    pub fn is_corner(&self, idx: usize) -> bool {
        let (x, y) = self.to_xy(idx);
        (x == 0 || x == self.width - 1) &&
        (y == 0 || y == self.height - 1)
    }

    #[inline(always)]
    pub fn is_interior(&self, idx: usize) -> bool {
        !self.is_edge(idx)
    }

    /// SIMD‑friendly iterator over all indices.
    #[inline(always)]
    pub fn iter_indices(&self) -> impl Iterator<Item = usize> {
        0..self.num_sites()
    }
}
