use crate::types::{Syndrome, LatticeGeometry};
use crate::spatial::cross_link_grid::CrossLinkGrid;

/// High‑performance syndrome heatmap used for predictor passes,
/// semantic refinement, geometric fusion, and Ising biasing.
/// Now includes optional cross‑link grid shaping.
#[derive(Clone)]
pub struct SyndromeHeatmap {
    pub cells: Vec<f32>,
    pub width: usize,
    pub height: usize,
}

impl SyndromeHeatmap {
    /// Build a heatmap directly from a syndrome bit vector.
    /// Bits are converted to floats (0.0 or 1.0).
    pub fn from_syndrome(syndrome: &Syndrome, geom: &LatticeGeometry) -> Self {
        let n = geom.width * geom.height;

        let mut cells = Vec::with_capacity(n);
        cells.resize(n, 0.0);

        // Fast bit → float conversion
        for (i, bit) in syndrome.bits.iter().enumerate() {
            if i >= n {
                break;
            }
            cells[i] = (*bit != 0) as i32 as f32;
        }

        Self {
            cells,
            width: geom.width,
            height: geom.height,
        }
    }

    // -------------------------------------------------------------------------
    // Cross‑layer shaping (cluster + semantic + door)
    // -------------------------------------------------------------------------
    pub fn apply_cross_links(
        &mut self,
        cross: &CrossLinkGrid,
        cluster_boost: f32,
        semantic_boost: f32,
        door_exit_boost: f32,
        door_entry_penalty: f32,
    ) {
        let n = self.cells.len();

        // --- cluster shaping ---
        for (cluster_idx, sites) in cross.cluster_to_sites.iter().enumerate() {
            let boost = cluster_boost * (cluster_idx as f32 + 1.0).sqrt();
            for &idx in sites {
                if idx < n {
                    self.cells[idx] += boost;
                }
            }
        }

        // --- semantic tag shaping ---
        for (tag_idx, sites) in cross.tag_to_sites.iter().enumerate() {
            let boost = semantic_boost * (tag_idx as f32 + 1.0);
            for &idx in sites {
                if idx < n {
                    self.cells[idx] += boost;
                }
            }
        }

        // --- door shaping ---
        for (door_idx, sites) in cross.door_to_sites.iter().enumerate() {
            for &idx in sites {
                if idx < n {
                    // exit sites get a boost
                    if let Some(did) = cross.door_for(idx) {
                        if did == door_idx {
                            self.cells[idx] += door_exit_boost;
                        }
                    }
                }
            }
        }

        // entry penalty (flat scan)
        for idx in 0..n {
            if let Some(door_id) = cross.door_for(idx) {
                // simple heuristic: entry sites are those not in door_to_sites
                if !cross.door_to_sites[door_id].contains(&idx) {
                    self.cells[idx] -= door_entry_penalty;
                }
            }
        }
    }

    // -------------------------------------------------------------------------
    // Existing logic (unchanged)
    // -------------------------------------------------------------------------

    #[inline(always)]
    pub fn get(&self, idx: usize) -> f32 {
        debug_assert!(idx < self.cells.len());
        self.cells[idx]
    }

    #[inline(always)]
    pub fn set(&mut self, idx: usize, value: f32) {
        debug_assert!(idx < self.cells.len());
        self.cells[idx] = value;
    }

    #[inline(always)]
    pub fn get_safe(&self, idx: usize) -> Option<f32> {
        self.cells.get(idx).copied()
    }

    #[inline(always)]
    pub fn set_safe(&mut self, idx: usize, value: f32) -> bool {
        if let Some(cell) = self.cells.get_mut(idx) {
            *cell = value;
            true
        } else {
            false
        }
    }

    #[inline(always)]
    pub fn iter(&self) -> impl Iterator<Item = &f32> {
        self.cells.iter()
    }

    #[inline(always)]
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut f32> {
        self.cells.iter_mut()
    }

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

    #[inline(always)]
    pub fn scale(&mut self, factor: f32) {
        for v in &mut self.cells {
            *v *= factor;
        }
    }

    #[inline(always)]
    pub fn clamp(&mut self, min: f32, max: f32) {
        for v in &mut self.cells {
            if *v < min { *v = min; }
            else if *v > max { *v = max; }
        }
    }

    pub fn weighted_centroid(&self) -> (f32, f32) {
        let mut sx = 0.0f32;
        let mut sy = 0.0f32;
        let mut total = 0.0f32;

        for (idx, &v) in self.cells.iter().enumerate() {
            if v <= 0.0 {
                continue;
            }

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
}
