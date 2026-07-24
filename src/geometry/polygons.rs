#[derive(Clone)]
pub struct PolygonRegion {
    pub id: usize,

    /// Sites belonging to this polygon region.
    /// Stored as a Vec because region sizes vary,
    /// but all hot-path operations use cached metadata.
    pub sites: Vec<usize>,

    /// Cached centroid (computed once).
    pub centroid: (f32, f32),

    /// Cached approximate radius.
    pub radius: f32,

    /// Cached bounding box for fast spatial rejection.
    pub bbox_min: (usize, usize),
    pub bbox_max: (usize, usize),
}

impl PolygonRegion {
    /// Construct a region and compute all cached metadata.
    pub fn new(id: usize, sites: Vec<usize>, geom: &crate::types::LatticeGeometry) -> Self {
        let (centroid, radius) = Self::compute_centroid_radius(&sites, geom);
        let (bbox_min, bbox_max) = Self::compute_bbox(&sites, geom);

        Self {
            id,
            sites,
            centroid,
            radius,
            bbox_min,
            bbox_max,
        }
    }

    /// Zero-cost inline: number of sites.
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.sites.len()
    }

    /// Zero-cost inline: check if empty.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.sites.is_empty()
    }

    /// Fast membership check using bounding box rejection first.
    #[inline(always)]
    pub fn contains(&self, idx: usize, geom: &crate::types::LatticeGeometry) -> bool {
        let (x, y) = (idx % geom.width, idx / geom.width);

        // Fast reject using bounding box
        if x < self.bbox_min.0 || x > self.bbox_max.0 ||
           y < self.bbox_min.1 || y > self.bbox_max.1 {
            return false;
        }

        // Fallback to exact membership
        self.sites.contains(&idx)
    }

    /// SIMD-friendly iterator over region sites.
    #[inline(always)]
    pub fn iter(&self) -> impl Iterator<Item = &usize> {
        self.sites.iter()
    }

    /// Compute centroid + radius (single pass).
    fn compute_centroid_radius(
        sites: &[usize],
        geom: &crate::types::LatticeGeometry,
    ) -> ((f32, f32), f32) {
        if sites.is_empty() {
            return ((0.0, 0.0), 0.0);
        }

        let mut sx = 0.0f32;
        let mut sy = 0.0f32;

        for &idx in sites {
            let x = (idx % geom.width) as f32;
            let y = (idx / geom.width) as f32;
            sx += x;
            sy += y;
        }

        let n = sites.len() as f32;
        let cx = sx / n;
        let cy = sy / n;

        // Approximate radius: max distance from centroid
        let mut r = 0.0f32;
        for &idx in sites {
            let x = (idx % geom.width) as f32;
            let y = (idx / geom.width) as f32;
            let dx = x - cx;
            let dy = y - cy;
            let dist = (dx * dx + dy * dy).sqrt();
            if dist > r {
                r = dist;
            }
        }

        ((cx, cy), r)
    }

    /// Compute bounding box for fast spatial rejection.
    fn compute_bbox(
        sites: &[usize],
        geom: &crate::types::LatticeGeometry,
    ) -> ((usize, usize), (usize, usize)) {
        if sites.is_empty() {
            return ((0, 0), (0, 0));
        }

        let mut min_x = usize::MAX;
        let mut min_y = usize::MAX;
        let mut max_x = 0usize;
        let mut max_y = 0usize;

        for &idx in sites {
            let x = idx % geom.width;
            let y = idx / geom.width;

            if x < min_x { min_x = x; }
            if y < min_y { min_y = y; }
            if x > max_x { max_x = x; }
            if y > max_y { max_y = y; }
        }

        ((min_x, min_y), (max_x, max_y))
    }
}
