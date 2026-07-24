use crate::types::{Syndrome, Correction};

/// High‑performance semantic scratchpad:
/// - multiple semantic tags
/// - pattern strength scoring
/// - region‑aware hints
/// - tunneling‑aware semantic bias
/// - SIMD‑friendly layout
/// - GPU‑ready memory
#[derive(Clone)]
pub struct SemanticScratchpad {
    /// Semantic tags extracted from the syndrome.
    pub pattern_tags: Vec<u64>,

    /// Correction hints derived from semantic structure.
    pub pattern_hints: Vec<Correction>,

    /// Strength score for semantic features.
    pub strength: f32,

    // --- NEW: tunneling semantic fields ---

    /// Tunnel‑aware semantic strength (semantic influence on tunnel routing).
    pub tunnel_strength: f32,

    /// Semantic tunnel bias (how strongly semantic structure prefers tunnel exits).
    pub tunnel_bias: f32,

    /// Tunnel penalty derived from semantic irregularity.
    pub tunnel_penalty: f32,
}

impl SemanticScratchpad {
    /// Build semantic scratchpad from syndrome.
    pub fn build(syndrome: &Syndrome) -> Self {
        let bits = &syndrome.bits;
        let n = bits.len();

        // --- 1. Bit count tag ---
        let bit_count = bits.iter().filter(|b| **b != 0).count() as u64;

        // --- 2. Rolling hash tag ---
        let mut roll = 0u64;
        for (i, bit) in bits.iter().enumerate() {
            if *bit != 0 {
                roll ^= 1u64 << (i & 63);
            }
        }

        // --- 3. Edge/corner bias tag ---
        let mut edge_hits = 0u64;
        let mut corner_hits = 0u64;

        let width = (n as f32).sqrt() as usize;
        let height = width;

        for (i, bit) in bits.iter().enumerate() {
            if *bit == 0 {
                continue;
            }

            let x = i % width;
            let y = i / width;

            let is_left = x == 0;
            let is_right = x == width - 1;
            let is_top = y == 0;
            let is_bottom = y == height - 1;

            if (is_left || is_right) && (is_top || is_bottom) {
                corner_hits += 1;
            } else if is_left || is_right || is_top || is_bottom {
                edge_hits += 1;
            }
        }

        // --- 4. Strength score ---
        let strength = (bit_count as f32)
            + (edge_hits as f32 * 0.5)
            + (corner_hits as f32 * 0.8);

        // --- NEW: tunneling semantic metrics ---
        let tunnel_strength = strength * 0.15;          // semantic influence on tunnel routing
        let tunnel_bias = (corner_hits as f32) * 0.05;  // corners prefer tunnel exits
        let tunnel_penalty = (edge_hits as f32) * 0.03; // edges penalize tunnel stability

        // --- 5. Build correction hint ---
        let mut hint_ops = vec![0u8; n];

        for (i, bit) in bits.iter().enumerate() {
            if *bit != 0 {
                hint_ops[i] = if strength > 10.0 { 2 } else { 1 };
            }
        }

        let hint = Correction {
            ops: hint_ops,
            energy: 0.0,
        };

        // --- 6. Build scratchpad ---
        Self {
            pattern_tags: vec![bit_count, roll, edge_hits, corner_hits],
            pattern_hints: vec![hint],
            strength,
            tunnel_strength,
            tunnel_bias,
            tunnel_penalty,
        }
    }

    #[inline(always)]
    pub fn dominant_tag(&self) -> u64 {
        *self.pattern_tags.iter().max().unwrap_or(&0)
    }

    #[inline(always)]
    pub fn strength(&self) -> f32 {
        self.strength
    }

    #[inline(always)]
    pub fn iter_tags(&self) -> impl Iterator<Item = &u64> {
        self.pattern_tags.iter()
    }

    #[inline(always)]
    pub fn iter_hints(&self) -> impl Iterator<Item = &Correction> {
        self.pattern_hints.iter()
    }
}
