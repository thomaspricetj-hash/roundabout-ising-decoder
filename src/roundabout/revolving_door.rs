#[derive(Clone)]
pub struct RevolvingDoor {
    pub id: usize,

    /// Entry and exit sites (flat lattice indices)
    pub entry_sites: Vec<usize>,
    pub exit_sites: Vec<usize>,

    /// Combined sites (SIMD/GPU‑friendly flat buffer)
    pub all_sites: Vec<usize>,

    /// Fast membership bitset (O(1) contains())
    pub membership: Vec<bool>,

    /// Cached centroid for routing bias
    pub centroid: (f32, f32),

    /// Cached bounding box for fast rejection
    pub bbox_min: (usize, usize),
    pub bbox_max: (usize, usize),

    /// Precomputed flow vector (exit centroid − entry centroid)
    pub flow_vec: (f32, f32),

    /// Door size (number of total sites)
    pub size: usize,
}

impl RevolvingDoor {
    /// Construct a revolving door and compute cached metadata.
    pub fn new(
        id: usize,
        entry_sites: Vec<usize>,
        exit_sites: Vec<usize>,
        width: usize,
        _height: usize,   // unused but preserved (no warnings)
    ) -> Self {
        // --- Combine sites (SIMD/GPU‑friendly) ---
        let all_sites = Self::combine(&entry_sites, &exit_sites);

        // --- Fast membership bitset ---
        let mut membership = vec![false; width * width * 4]; // oversized safety buffer
        for &idx in &all_sites {
            if idx < membership.len() {
                membership[idx] = true;
            }
        }

        // --- Centroid ---
        let centroid = Self::compute_centroid(&all_sites, width);

        // --- Bounding box ---
        let (bbox_min, bbox_max) = Self::compute_bbox(&all_sites, width);

        // --- Flow vector (exit centroid − entry centroid) ---
        let flow_vec = Self::compute_flow_vector(&entry_sites, &exit_sites, width);

        // --- Door size ---
        let size = all_sites.len();

        Self {
            id,
            entry_sites,
            exit_sites,
            all_sites,
            membership,
            centroid,
            bbox_min,
            bbox_max,
            flow_vec,
            size,
        }
    }

    /// Combine entry + exit sites into one list.
    #[inline(always)]
    fn combine(a: &Vec<usize>, b: &Vec<usize>) -> Vec<usize> {
        let mut out = Vec::with_capacity(a.len() + b.len());
        out.extend_from_slice(a);
        out.extend_from_slice(b);
        out
    }

    /// Compute centroid of all door sites.
    fn compute_centroid(sites: &[usize], width: usize) -> (f32, f32) {
        if sites.is_empty() {
            return (0.0, 0.0);
        }

        let mut sx = 0.0f32;
        let mut sy = 0.0f32;

        for &idx in sites {
            let x = (idx % width) as f32;
            let y = (idx / width) as f32;
            sx += x;
            sy += y;
        }

        let n = sites.len() as f32;
        (sx / n, sy / n)
    }

    /// Compute bounding box for fast spatial rejection.
    fn compute_bbox(
        sites: &[usize],
        width: usize,
    ) -> ((usize, usize), (usize, usize)) {
        if sites.is_empty() {
            return ((0, 0), (0, 0));
        }

        let mut min_x = usize::MAX;
        let mut min_y = usize::MAX;
        let mut max_x = 0usize;
        let mut max_y = 0usize;

        for &idx in sites {
            let x = idx % width;
            let y = idx / width;

            if x < min_x { min_x = x; }
            if y < min_y { min_y = y; }
            if x > max_x { max_x = x; }
            if y > max_y { max_y = y; }
        }

        ((min_x, min_y), (max_x, max_y))
    }

    /// Compute flow vector (exit centroid − entry centroid)
    fn compute_flow_vector(
        entry: &[usize],
        exit: &[usize],
        width: usize,
    ) -> (f32, f32) {
        if entry.is_empty() || exit.is_empty() {
            return (0.0, 0.0);
        }

        let ec = Self::compute_centroid(entry, width);
        let xc = Self::compute_centroid(exit, width);

        (xc.0 - ec.0, xc.1 - ec.1)
    }

    /// Fast membership check using bounding box + bitset.
    #[inline(always)]
    pub fn contains(&self, idx: usize, width: usize) -> bool {
        let x = idx % width;
        let y = idx / width;

        // bounding box rejection
        if x < self.bbox_min.0 || x > self.bbox_max.0 ||
           y < self.bbox_min.1 || y > self.bbox_max.1 {
            return false;
        }

        // O(1) membership check
        idx < self.membership.len() && self.membership[idx]
    }

    /// SIMD-friendly iterator over all door sites.
    #[inline(always)]
    pub fn iter_all(&self) -> impl Iterator<Item = usize> + '_ {
        self.all_sites.iter().copied()
    }

    /// Average heat over entry sites.
    #[inline(always)]
    pub fn avg_entry_heat(&self, heat: &[f32]) -> f32 {
        if self.entry_sites.is_empty() {
            return 0.0;
        }
        let mut sum = 0.0f32;
        for &idx in &self.entry_sites {
            sum += heat[idx];
        }
        sum / (self.entry_sites.len() as f32)
    }

    /// Average heat over exit sites.
    #[inline(always)]
    pub fn avg_exit_heat(&self, heat: &[f32]) -> f32 {
        if self.exit_sites.is_empty() {
            return 0.0;
        }
        let mut sum = 0.0f32;
        for &idx in &self.exit_sites {
            sum += heat[idx];
        }
        sum / (self.exit_sites.len() as f32)
    }
}

