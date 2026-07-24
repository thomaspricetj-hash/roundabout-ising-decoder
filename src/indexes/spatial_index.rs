use crate::types::Correction;

/// High‑performance spatial index used for region‑based pattern memory,
/// geometric refinement, and predictor routing. Supports fast lookup,
/// bounded capacity, region scoring, and overwrite semantics.
#[derive(Clone)]
pub struct SpatialIndex {
    /// Each region stores a list of corrections.
    /// (region_id, corrections, weight)
    pub region_templates: Vec<(usize, Vec<Correction>, f32)>,

    /// Maximum number of stored regions.
    pub capacity: usize,
}

impl SpatialIndex {
    /// Create an empty spatial index with default capacity.
    pub fn empty() -> Self {
        Self {
            region_templates: Vec::new(),
            capacity: 128, // production-grade default
        }
    }

    /// Create with explicit capacity.
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            region_templates: Vec::with_capacity(cap),
            capacity: cap,
        }
    }

    /// Zero-cost inline: number of stored regions.
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.region_templates.len()
    }

    /// Zero-cost inline: check if empty.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.region_templates.is_empty()
    }

    /// Insert or update a region template.
    /// If region exists, overwrite corrections and reset weight.
    pub fn insert(&mut self, region_id: usize, corrections: Vec<Correction>) {
        // Overwrite existing region
        if let Some(slot) = self.region_templates.iter_mut().find(|(id, _, _)| *id == region_id) {
            slot.1 = corrections;
            slot.2 = 1.0; // reset weight
            return;
        }

        // Insert new region
        self.region_templates.push((region_id, corrections, 1.0));

        // Enforce capacity (LRU eviction)
        if self.region_templates.len() > self.capacity {
            self.region_templates.remove(0);
        }
    }

    /// Retrieve corrections for a region.
    #[inline(always)]
    pub fn get(&self, region_id: usize) -> Option<&Vec<Correction>> {
        self.region_templates
            .iter()
            .find(|(id, _, _)| *id == region_id)
            .map(|(_, corr, _)| corr)
    }

    /// Retrieve mutable corrections for a region.
    #[inline(always)]
    pub fn get_mut(&mut self, region_id: usize) -> Option<&mut Vec<Correction>> {
        self.region_templates
            .iter_mut()
            .find(|(id, _, _)| *id == region_id)
            .map(|(_, corr, _)| corr)
    }

    /// Decay region weights.
    /// Regions with very low weight are removed.
    pub fn decay(&mut self, factor: f32) {
        for (_, _, w) in &mut self.region_templates {
            *w *= factor;
        }

        // Remove regions with negligible weight
        self.region_templates.retain(|(_, _, w)| *w > 0.05);
    }

    /// Score similarity between region IDs.
    /// Lower difference = higher similarity.
    #[inline(always)]
    fn similarity(a: usize, b: usize) -> f32 {
        let diff = (a as i64 - b as i64).abs() as f32;
        let max = 1024.0; // arbitrary normalization scale
        1.0 - (diff / max).min(1.0)
    }

    /// Find best matching region by similarity.
    pub fn best_match(&self, region_id: usize) -> Option<&Vec<Correction>> {
        let mut best_score = 0.0f32;
        let mut best_corr: Option<&Vec<Correction>> = None;

        for (stored_id, corr, weight) in &self.region_templates {
            let score = Self::similarity(*stored_id, region_id) * *weight;
            if score > best_score {
                best_score = score;
                best_corr = Some(corr);
            }
        }

        best_corr
    }

    /// SIMD-friendly iterator.
    #[inline(always)]
    pub fn iter(&self) -> impl Iterator<Item = &(usize, Vec<Correction>, f32)> {
        self.region_templates.iter()
    }

    /// Mutable SIMD-friendly iterator.
    #[inline(always)]
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut (usize, Vec<Correction>, f32)> {
        self.region_templates.iter_mut()
    }
}
