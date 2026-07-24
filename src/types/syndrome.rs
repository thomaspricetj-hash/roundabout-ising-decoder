#[derive(Clone)]
pub struct Syndrome {
    /// Stabilizer bits (0 or 1)
    pub bits: Vec<u8>,
}

impl Syndrome {
    /// Create an empty syndrome of given size.
    #[inline(always)]
    pub fn empty(size: usize) -> Self {
        Self {
            bits: vec![0u8; size],
        }
    }

    /// Zero‑cost inline getter.
    #[inline(always)]
    pub fn get(&self, idx: usize) -> u8 {
        debug_assert!(idx < self.bits.len());
        self.bits[idx]
    }

    /// Zero‑cost inline setter.
    #[inline(always)]
    pub fn set(&mut self, idx: usize, value: u8) {
        debug_assert!(idx < self.bits.len());
        debug_assert!(value <= 1);
        self.bits[idx] = value;
    }

    /// Safe getter.
    #[inline(always)]
    pub fn get_safe(&self, idx: usize) -> Option<u8> {
        self.bits.get(idx).copied()
    }

    /// Safe setter.
    #[inline(always)]
    pub fn set_safe(&mut self, idx: usize, value: u8) -> bool {
        if let Some(slot) = self.bits.get_mut(idx) {
            *slot = value;
            true
        } else {
            false
        }
    }

    /// Hamming weight (number of 1 bits).
    #[inline(always)]
    pub fn weight(&self) -> usize {
        self.bits.iter().filter(|&&b| b != 0).count()
    }

    /// Rolling hash used for semantic keys and pattern memory.
    #[inline(always)]
    pub fn rolling_hash(&self) -> u64 {
        let mut roll = 0u64;
        for (i, bit) in self.bits.iter().enumerate() {
            if *bit != 0 {
                roll ^= 1u64 << (i & 63);
            }
        }
        roll
    }

    /// Combined semantic key (bit count + rolling hash).
    #[inline(always)]
    pub fn semantic_key(&self) -> u64 {
        let count = self.weight() as u64;
        count ^ (self.rolling_hash() << 1)
    }

    /// Convert to boolean mask.
    #[inline(always)]
    pub fn to_mask(&self) -> Vec<bool> {
        self.bits.iter().map(|b| *b != 0).collect()
    }

    /// SIMD‑friendly iterator.
    #[inline(always)]
    pub fn iter(&self) -> impl Iterator<Item = &u8> {
        self.bits.iter()
    }

    /// Mutable SIMD‑friendly iterator.
    #[inline(always)]
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut u8> {
        self.bits.iter_mut()
    }

    /// Ensure all bits are valid (0 or 1).
    #[inline(always)]
    pub fn validate(&self) -> bool {
        self.bits.iter().all(|b| *b <= 1)
    }
}
