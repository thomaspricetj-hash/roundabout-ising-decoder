/// Correction object used throughout the decoder.
/// - ops: flat lattice operator buffer (0=I, 1=X, 2=Z, 3=Y)
/// - energy: computed by Ising solver or predictor
#[derive(Clone)]
pub struct Correction {
    pub ops: Vec<u8>,
    pub energy: f64,
}

impl Correction {
    /// Create an empty correction (all identity ops).
    #[inline(always)]
    pub fn empty(size: usize) -> Self {
        Self {
            ops: vec![0u8; size],
            energy: 0.0,
        }
    }

    /// Zero‑cost inline getter.
    #[inline(always)]
    pub fn get(&self, idx: usize) -> u8 {
        debug_assert!(idx < self.ops.len());
        self.ops[idx]
    }

    /// Zero‑cost inline setter.
    #[inline(always)]
    pub fn set(&mut self, idx: usize, op: u8) {
        debug_assert!(idx < self.ops.len());
        self.ops[idx] = op;
    }

    /// Safe getter.
    #[inline(always)]
    pub fn get_safe(&self, idx: usize) -> Option<u8> {
        self.ops.get(idx).copied()
    }

    /// Safe setter.
    #[inline(always)]
    pub fn set_safe(&mut self, idx: usize, op: u8) -> bool {
        if let Some(slot) = self.ops.get_mut(idx) {
            *slot = op;
            true
        } else {
            false
        }
    }

    /// Reset all ops to identity.
    #[inline(always)]
    pub fn clear(&mut self) {
        for op in &mut self.ops {
            *op = 0;
        }
        self.energy = 0.0;
    }

    /// Hamming weight (number of non‑identity ops).
    #[inline(always)]
    pub fn weight(&self) -> usize {
        self.ops.iter().filter(|&&op| op != 0).count()
    }

    /// Count specific operator type.
    #[inline(always)]
    pub fn count_op(&self, target: u8) -> usize {
        self.ops.iter().filter(|&&op| op == target).count()
    }

    /// Merge another correction into this one.
    /// Non‑zero ops in `other` overwrite zeros in `self`.
    #[inline(always)]
    pub fn merge_from(&mut self, other: &Correction) {
        debug_assert!(self.ops.len() == other.ops.len());
        for (a, b) in self.ops.iter_mut().zip(other.ops.iter()) {
            if *a == 0 && *b != 0 {
                *a = *b;
            }
        }
    }

    /// Combine two corrections into a new one.
    /// Non‑zero ops from either side are kept.
    pub fn combine(&self, other: &Correction) -> Correction {
        debug_assert!(self.ops.len() == other.ops.len());
        let mut ops = Vec::with_capacity(self.ops.len());
        for (a, b) in self.ops.iter().zip(other.ops.iter()) {
            ops.push(if *a != 0 { *a } else { *b });
        }
        Correction { ops, energy: 0.0 }
    }

    /// SIMD‑friendly iterator.
    #[inline(always)]
    pub fn iter(&self) -> impl Iterator<Item = &u8> {
        self.ops.iter()
    }

    /// Mutable SIMD‑friendly iterator.
    #[inline(always)]
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut u8> {
        self.ops.iter_mut()
    }

    /// Fast equality check (ignores energy).
    #[inline(always)]
    pub fn same_ops(&self, other: &Correction) -> bool {
        self.ops == other.ops
    }

    /// Replace all ops equal to `from` with `to`.
    #[inline(always)]
    pub fn replace(&mut self, from: u8, to: u8) {
        for op in &mut self.ops {
            if *op == from {
                *op = to;
            }
        }
    }

    /// Check operator type.
    #[inline(always)]
    pub fn is_identity(op: u8) -> bool { op == 0 }

    #[inline(always)]
    pub fn is_x(op: u8) -> bool { op == 1 }

    #[inline(always)]
    pub fn is_z(op: u8) -> bool { op == 2 }

    #[inline(always)]
    pub fn is_y(op: u8) -> bool { op == 3 }
}
