use crate::types::Correction;

/// High‑performance semantic index used for pattern memory,
/// predictor refinement, and cognitive routing. Provides fast
/// lookup, overwrite semantics, bounded capacity, and scoring.
#[derive(Clone)]
pub struct SemanticIndex {
    /// (pattern_key, correction, weight)
    /// Weight is used for decay and similarity scoring.
    pub pattern_templates: Vec<(u64, Correction, f32)>,

    /// Maximum number of stored patterns.
    pub capacity: usize,
}

impl SemanticIndex {
    /// Create an empty semantic index with a default capacity.
    pub fn empty() -> Self {
        Self {
            pattern_templates: Vec::new(),
            capacity: 256, // production-grade default
        }
    }

    /// Create with explicit capacity.
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            pattern_templates: Vec::with_capacity(cap),
            capacity: cap,
        }
    }

    /// Zero-cost inline: number of stored patterns.
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.pattern_templates.len()
    }

    /// Zero-cost inline: check if empty.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.pattern_templates.is_empty()
    }

    /// Insert or update a pattern.
    /// If the key exists, overwrite the correction and reset weight.
    /// If the key does not exist, insert a new entry.
    pub fn insert(&mut self, key: u64, corr: Correction) {
        // Overwrite existing
        if let Some(slot) = self.pattern_templates.iter_mut().find(|(k, _, _)| *k == key) {
            slot.1 = corr;
            slot.2 = 1.0; // reset weight
            return;
        }

        // Insert new
        self.pattern_templates.push((key, corr, 1.0));

        // Enforce capacity
        if self.pattern_templates.len() > self.capacity {
            self.pattern_templates.remove(0); // simple LRU eviction
        }
    }

    /// Retrieve a correction by key.
    #[inline(always)]
    pub fn get(&self, key: u64) -> Option<&Correction> {
        self.pattern_templates
            .iter()
            .find(|(k, _, _)| *k == key)
            .map(|(_, corr, _)| corr)
    }

    /// Retrieve a mutable correction by key.
    #[inline(always)]
    pub fn get_mut(&mut self, key: u64) -> Option<&mut Correction> {
        self.pattern_templates
            .iter_mut()
            .find(|(k, _, _)| *k == key)
            .map(|(_, corr, _)| corr)
    }

    /// Decay all pattern weights.
    /// Patterns with very low weight are removed.
    pub fn decay(&mut self, factor: f32) {
        for (_, _, w) in &mut self.pattern_templates {
            *w *= factor;
        }

        // Remove patterns with negligible weight
        self.pattern_templates.retain(|(_, _, w)| *w > 0.05);
    }

    /// Score similarity between a syndrome key and stored patterns.
    /// Returns the best match (if any).
    pub fn best_match(&self, key: u64) -> Option<&Correction> {
        let mut best_score = 0.0f32;
        let mut best_corr: Option<&Correction> = None;

        for (stored_key, corr, weight) in &self.pattern_templates {
            let score = Self::similarity(*stored_key, key) * *weight;
            if score > best_score {
                best_score = score;
                best_corr = Some(corr);
            }
        }

        best_corr
    }

    /// Simple similarity metric between two keys.
    /// Keys are XORed; fewer differing bits = higher similarity.
    #[inline(always)]
    fn similarity(a: u64, b: u64) -> f32 {
        let diff = (a ^ b).count_ones();
        let max_bits = 64.0;
        1.0 - (diff as f32 / max_bits)
    }

    /// SIMD-friendly iterator.
    #[inline(always)]
    pub fn iter(&self) -> impl Iterator<Item = &(u64, Correction, f32)> {
        self.pattern_templates.iter()
    }

    /// Mutable SIMD-friendly iterator.
    #[inline(always)]
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut (u64, Correction, f32)> {
        self.pattern_templates.iter_mut()
    }
}
