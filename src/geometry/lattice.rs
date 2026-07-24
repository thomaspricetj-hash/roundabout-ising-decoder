use crate::types::LatticeGeometry;

/// A tiny, fixed-size neighbor list optimized for hot loops.
/// Faster than SmallVec, no heap, no dependencies.
#[derive(Clone, Copy)]
pub struct NeighborList {
    pub items: [usize; 4],
    pub len: usize,
}

impl NeighborList {
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            items: [0; 4],
            len: 0,
        }
    }

    #[inline(always)]
    pub fn push(&mut self, value: usize) {
        debug_assert!(self.len < 4);
        self.items[self.len] = value;
        self.len += 1;
    }

    #[inline(always)]
    pub fn iter(&self) -> impl Iterator<Item = &usize> {
        self.items[..self.len].iter()
    }
}

pub trait LatticeGeometryExt {
    fn num_sites(&self) -> usize;

    fn to_index(&self, x: usize, y: usize) -> usize;
    fn to_xy(&self, idx: usize) -> (usize, usize);

    fn in_bounds(&self, x: usize, y: usize) -> bool;

    fn neighbors(&self, idx: usize) -> NeighborList;

    fn is_corner(&self, idx: usize) -> bool;
    fn is_edge(&self, idx: usize) -> bool;
    fn is_interior(&self, idx: usize) -> bool;
}

impl LatticeGeometryExt for LatticeGeometry {
    #[inline(always)]
    fn num_sites(&self) -> usize {
        self.width * self.height
    }

    #[inline(always)]
    fn to_index(&self, x: usize, y: usize) -> usize {
        debug_assert!(x < self.width && y < self.height);
        y * self.width + x
    }

    #[inline(always)]
    fn to_xy(&self, idx: usize) -> (usize, usize) {
        debug_assert!(idx < self.num_sites());
        (idx % self.width, idx / self.width)
    }

    #[inline(always)]
    fn in_bounds(&self, x: usize, y: usize) -> bool {
        x < self.width && y < self.height
    }

    #[inline(always)]
    fn neighbors(&self, idx: usize) -> NeighborList {
        let (x, y) = self.to_xy(idx);
        let mut out = NeighborList::new();

        if x > 0 {
            out.push(self.to_index(x - 1, y));
        }
        if x + 1 < self.width {
            out.push(self.to_index(x + 1, y));
        }
        if y > 0 {
            out.push(self.to_index(x, y - 1));
        }
        if y + 1 < self.height {
            out.push(self.to_index(x, y + 1));
        }

        out
    }

    #[inline(always)]
    fn is_corner(&self, idx: usize) -> bool {
        let (x, y) = self.to_xy(idx);
        (x == 0 || x == self.width - 1) &&
        (y == 0 || y == self.height - 1)
    }

    #[inline(always)]
    fn is_edge(&self, idx: usize) -> bool {
        let (x, y) = self.to_xy(idx);
        x == 0 || x == self.width - 1 || y == 0 || y == self.height - 1
    }

    #[inline(always)]
    fn is_interior(&self, idx: usize) -> bool {
        !self.is_edge(idx)
    }
}
