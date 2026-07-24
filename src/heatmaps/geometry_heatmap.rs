use crate::types::LatticeGeometry;

/// High‑performance geometry heatmap used for routing, predictor refinement,
/// flow fields, and Ising biasing. Zero‑cost accessors and SIMD‑friendly layout.
#[derive(Clone)]
pub struct GeometryHeatmap {
    pub cells: Vec<f32>,
    pub width: usize,
    pub height: usize,
}

impl GeometryHeatmap {
    /// Create a uniform heatmap.
    pub fn uniform(geom: &LatticeGeometry, value: f32) -> Self {
        let n = geom.width * geom.height;
        let mut cells = Vec::with_capacity(n);
        cells.resize(n, value);

        Self {
            cells,
            width: geom.width,
            height: geom.height,
        }
    }

    /// Create from explicit cells.
    pub fn from_cells(geom: &LatticeGeometry, cells: Vec<f32>) -> Self {
        assert!(
            cells.len() == geom.width * geom.height,
            "GeometryHeatmap::from_cells: cell count mismatch"
        );
        Self {
            cells,
            width: geom.width,
            height: geom.height,
        }
    }

    /// Zero‑cost inline getter (unchecked).
    #[inline(always)]
    pub fn get(&self, idx: usize) -> f32 {
        debug_assert!(idx < self.cells.len());
        self.cells[idx]
    }

    /// Zero‑cost inline setter (unchecked).
    #[inline(always)]
    pub fn set(&mut self, idx: usize, value: f32) {
        debug_assert!(idx < self.cells.len());
        self.cells[idx] = value;
    }

    /// Safe getter with bounds check.
    #[inline(always)]
    pub fn get_safe(&self, idx: usize) -> Option<f32> {
        self.cells.get(idx).copied()
    }

    /// Safe setter with bounds check.
    #[inline(always)]
    pub fn set_safe(&mut self, idx: usize, value: f32) -> bool {
        if let Some(cell) = self.cells.get_mut(idx) {
            *cell = value;
            true
        } else {
            false
        }
    }

    /// SIMD‑friendly iterator.
    #[inline(always)]
    pub fn iter(&self) -> impl Iterator<Item = &f32> {
        self.cells.iter()
    }

    /// Mutable SIMD‑friendly iterator.
    #[inline(always)]
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut f32> {
        self.cells.iter_mut()
    }

    /// Normalize heatmap to [0, 1].
    pub fn normalize(&mut self) {
        let mut min = f32::MAX;
        let mut max = f32::MIN;

        for &v in &self.cells {
            if v < min { min = v; }
            if v > max { max = v; }
        }

        let range = max - min;
        if range <= 1e-12 {
            return;
        }

        for v in &mut self.cells {
            *v = (*v - min) / range;
        }
    }

    /// Scale all values by a constant.
    #[inline(always)]
    pub fn scale(&mut self, factor: f32) {
        for v in &mut self.cells {
            *v *= factor;
        }
    }

    /// Clamp all values to [min, max].
    #[inline(always)]
    pub fn clamp(&mut self, min: f32, max: f32) {
        for v in &mut self.cells {
            if *v < min { *v = min; }
            else if *v > max { *v = max; }
        }
    }

    /// Blend two heatmaps: result = a * self + b * other.
    pub fn blend(&mut self, other: &GeometryHeatmap, a: f32, b: f32) {
        debug_assert!(self.cells.len() == other.cells.len());
        for (v, o) in self.cells.iter_mut().zip(other.cells.iter()) {
            *v = a * *v + b * *o;
        }
    }

    /// Compute centroid weighted by heat values.
    pub fn weighted_centroid(&self) -> (f32, f32) {
        let mut sx = 0.0f32;
        let mut sy = 0.0f32;
        let mut total = 0.0f32;

        for (idx, &v) in self.cells.iter().enumerate() {
            let x = (idx % self.width) as f32;
            let y = (idx / self.width) as f32;

            sx += x * v;
            sy += y * v;
            total += v;
        }

        if total <= 1e-12 {
            return (0.0, 0.0);
        }

        (sx / total, sy / total)
    }

    /// Estimate gradient at a site (central difference).
    #[inline(always)]
    pub fn gradient(&self, idx: usize) -> (f32, f32) {
        let x = idx % self.width;
        let y = idx / self.width;

        let left = if x > 0 { self.get(idx - 1) } else { self.get(idx) };
        let right = if x + 1 < self.width { self.get(idx + 1) } else { self.get(idx) };

        let up = if y > 0 { self.get(idx - self.width) } else { self.get(idx) };
        let down = if y + 1 < self.height { self.get(idx + self.width) } else { self.get(idx) };

        let gx = 0.5 * (right - left);
        let gy = 0.5 * (down - up);

        (gx, gy)
    }

    /// Add another geometry field in-place: self += factor * other.
    #[inline(always)]
    pub fn add_scaled(&mut self, other: &GeometryHeatmap, factor: f32) {
        debug_assert!(self.cells.len() == other.cells.len());
        for (v, o) in self.cells.iter_mut().zip(other.cells.iter()) {
            *v += factor * *o;
        }
    }

    /// Apply a per-site mask: self[i] *= mask[i].
    #[inline(always)]
    pub fn apply_mask(&mut self, mask: &[f32]) {
        debug_assert!(self.cells.len() == mask.len());
        for (v, m) in self.cells.iter_mut().zip(mask.iter()) {
            *v *= *m;
        }
    }

    /// Create a geometry heatmap from a raw scalar field.
    pub fn from_scalar_field(geom: &LatticeGeometry, field: &[f32]) -> Self {
        assert!(
            field.len() == geom.width * geom.height,
            "GeometryHeatmap::from_scalar_field: length mismatch"
        );
        Self {
            cells: field.to_vec(),
            width: geom.width,
            height: geom.height,
        }
    }
}

