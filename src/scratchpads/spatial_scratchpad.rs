use crate::heatmaps::{SyndromeHeatmap, GeometryHeatmap};

/// High‑performance spatial scratchpad:
/// - fast cluster extraction
/// - centroid + radius per cluster
/// - bounding boxes for fast rejection
/// - SIMD‑friendly iteration
/// - GPU‑ready memory layout
#[derive(Clone)]
pub struct SpatialScratchpad {
    /// Connected components of active syndrome sites.
    pub local_clusters: Vec<Vec<usize>>,

    /// Combined energy score per cluster.
    pub local_energy: Vec<f64>,

    /// Cached centroids for each cluster.
    pub centroids: Vec<(f32, f32)>,

    /// Cached radii for each cluster.
    pub radii: Vec<f32>,

    /// Cached bounding boxes for each cluster.
    pub bboxes: Vec<((usize, usize), (usize, usize))>,
}

impl SpatialScratchpad {
    /// Build spatial scratchpad from syndrome + geometry heatmaps.
    pub fn build(syn: &SyndromeHeatmap, geo: &GeometryHeatmap) -> Self {
        let clusters = Self::cluster_active_sites(syn);

        let mut local_energy = Vec::with_capacity(clusters.len());
        let mut centroids = Vec::with_capacity(clusters.len());
        let mut radii = Vec::with_capacity(clusters.len());
        let mut bboxes = Vec::with_capacity(clusters.len());

        let w = syn.width;
        let h = syn.height;

        for cluster in &clusters {
            // --- 1. Energy ---
            let energy: f64 = cluster
                .iter()
                .map(|idx| syn.cells[*idx] as f64 + geo.cells[*idx] as f64)
                .sum();
            local_energy.push(energy);

            // --- 2. Centroid ---
            let (cx, cy) = Self::compute_centroid(cluster, w);
            centroids.push((cx, cy));

            // --- 3. Radius ---
            let r = Self::compute_radius(cluster, w, cx, cy);
            radii.push(r);

            // --- 4. Bounding box ---
            let bbox = Self::compute_bbox(cluster, w, h);
            bboxes.push(bbox);
        }

        Self {
            local_clusters: clusters,
            local_energy,
            centroids,
            radii,
            bboxes,
        }
    }

    // -------------------------------------------------------------------------
    // Cluster extraction (DFS)
    // -------------------------------------------------------------------------
    fn cluster_active_sites(syn: &SyndromeHeatmap) -> Vec<Vec<usize>> {
        let w = syn.width;
        let h = syn.height;
        let mut visited = vec![false; w * h];
        let mut clusters = Vec::new();

        for idx in 0..(w * h) {
            if syn.cells[idx] <= 0.0 || visited[idx] {
                continue;
            }

            let mut stack = vec![idx];
            let mut cluster = Vec::new();
            visited[idx] = true;

            while let Some(cur) = stack.pop() {
                cluster.push(cur);

                let x = cur % w;
                let y = cur / w;

                // Branch-minimized neighbor iteration
                if x + 1 < w {
                    let n = cur + 1;
                    if syn.cells[n] > 0.0 && !visited[n] {
                        visited[n] = true;
                        stack.push(n);
                    }
                }
                if x > 0 {
                    let n = cur - 1;
                    if syn.cells[n] > 0.0 && !visited[n] {
                        visited[n] = true;
                        stack.push(n);
                    }
                }
                if y + 1 < h {
                    let n = cur + w;
                    if syn.cells[n] > 0.0 && !visited[n] {
                        visited[n] = true;
                        stack.push(n);
                    }
                }
                if y > 0 {
                    let n = cur - w;
                    if syn.cells[n] > 0.0 && !visited[n] {
                        visited[n] = true;
                        stack.push(n);
                    }
                }
            }

            if !cluster.is_empty() {
                clusters.push(cluster);
            }
        }

        clusters
    }

    // -------------------------------------------------------------------------
    // Centroid
    // -------------------------------------------------------------------------
    #[inline(always)]
    fn compute_centroid(cluster: &[usize], width: usize) -> (f32, f32) {
        let mut sx = 0.0f32;
        let mut sy = 0.0f32;

        for &idx in cluster {
            sx += (idx % width) as f32;
            sy += (idx / width) as f32;
        }

        let n = cluster.len() as f32;
        (sx / n, sy / n)
    }

    // -------------------------------------------------------------------------
    // Radius
    // -------------------------------------------------------------------------
    #[inline(always)]
    fn compute_radius(cluster: &[usize], width: usize, cx: f32, cy: f32) -> f32 {
        let mut r = 0.0f32;

        for &idx in cluster {
            let x = (idx % width) as f32;
            let y = (idx / width) as f32;

            let dx = x - cx;
            let dy = y - cy;
            let dist = (dx * dx + dy * dy).sqrt();

            if dist > r {
                r = dist;
            }
        }

        r
    }

    // -------------------------------------------------------------------------
    // Bounding box
    // -------------------------------------------------------------------------
    #[inline(always)]
    fn compute_bbox(
        cluster: &[usize],
        width: usize,
        height: usize,
    ) -> ((usize, usize), (usize, usize)) {
        let mut min_x = width;
        let mut min_y = height;
        let mut max_x = 0usize;
        let mut max_y = 0usize;

        for &idx in cluster {
            let x = idx % width;
            let y = idx / width;

            if x < min_x { min_x = x; }
            if y < min_y { min_y = y; }
            if x > max_x { max_x = x; }
            if y > max_y { max_y = y; }
        }

        ((min_x, min_y), (max_x, max_y))
    }
}
