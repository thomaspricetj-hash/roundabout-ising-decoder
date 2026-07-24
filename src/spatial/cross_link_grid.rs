use crate::types::LatticeGeometry;
use crate::scratchpads::{SpatialScratchpad, SemanticScratchpad};
use crate::roundabout::RevolvingDoor;

/// Cross-layer linking grid between spatial clusters, semantic tags, doors,
/// and now tunnel exits. Includes reverse-link maps and GPU/SIMD-friendly lookup.
#[derive(Clone)]
pub struct CrossLinkGrid {
    pub width: usize,
    pub height: usize,

    // --- flat lookup tables (fast per-site access) ---
    pub site_to_cluster: Vec<Option<usize>>,
    pub site_to_semantic_tag: Vec<Option<usize>>,
    pub site_to_door: Vec<Option<usize>>,
    pub site_to_tunnel: Vec<Option<usize>>,        // NEW

    // --- reverse-link maps (fast per-structure access) ---
    pub cluster_to_sites: Vec<Vec<usize>>,
    pub tag_to_sites: Vec<Vec<usize>>,
    pub door_to_sites: Vec<Vec<usize>>,
    pub tunnel_to_sites: Vec<Vec<usize>>,          // NEW
}

impl CrossLinkGrid {
    pub fn build(
        geom: &LatticeGeometry,
        spatial: &SpatialScratchpad,
        semantic: &SemanticScratchpad,
        doors: &[RevolvingDoor],
    ) -> Self {
        let n = geom.num_sites();

        // flat lookup tables
        let mut site_to_cluster = vec![None; n];
        let mut site_to_semantic_tag = vec![None; n];
        let mut site_to_door = vec![None; n];
        let mut site_to_tunnel = vec![None; n];     // NEW

        // reverse-link maps
        let mut cluster_to_sites = vec![Vec::new(); spatial.local_clusters.len()];
        let mut tag_to_sites = vec![Vec::new(); semantic.pattern_tags.len().max(1)];
        let mut door_to_sites = vec![Vec::new(); doors.len()];
        let mut tunnel_to_sites = vec![Vec::new(); doors.len()]; // NEW

        // --- cluster mapping ---
        for (cluster_idx, cluster) in spatial.local_clusters.iter().enumerate() {
            for &idx in cluster {
                site_to_cluster[idx] = Some(cluster_idx);
                cluster_to_sites[cluster_idx].push(idx);
            }
        }

        // --- semantic tag mapping ---
        let tag_idx = if semantic.pattern_tags.is_empty() { None } else { Some(0usize) };
        if let Some(ti) = tag_idx {
            for idx in 0..n {
                site_to_semantic_tag[idx] = Some(ti);
                tag_to_sites[ti].push(idx);
            }
        }

        // --- door + tunnel mapping ---
        for (door_idx, door) in doors.iter().enumerate() {
            // physical door sites
            for idx in door.iter_all() {
                site_to_door[idx] = Some(door.id);
                door_to_sites[door_idx].push(idx);
            }

            // NEW: tunnel exit sites
            for idx in &door.tunnel_exit_sites {
                if *idx < n {
                    site_to_tunnel[*idx] = Some(door.id);
                    tunnel_to_sites[door_idx].push(*idx);
                }
            }
        }

        Self {
            width: geom.width,
            height: geom.height,

            site_to_cluster,
            site_to_semantic_tag,
            site_to_door,
            site_to_tunnel,          // NEW

            cluster_to_sites,
            tag_to_sites,
            door_to_sites,
            tunnel_to_sites,         // NEW
        }
    }

    // --- flat lookup accessors ---
    #[inline(always)]
    pub fn cluster_for(&self, idx: usize) -> Option<usize> {
        self.site_to_cluster[idx]
    }

    #[inline(always)]
    pub fn semantic_tag_for(&self, idx: usize) -> Option<usize> {
        self.site_to_semantic_tag[idx]
    }

    #[inline(always)]
    pub fn door_for(&self, idx: usize) -> Option<usize> {
        self.site_to_door[idx]
    }

    #[inline(always)]
    pub fn tunnel_for(&self, idx: usize) -> Option<usize> {   // NEW
        self.site_to_tunnel[idx]
    }

    // --- reverse-link accessors ---
    #[inline(always)]
    pub fn sites_in_cluster(&self, cluster_idx: usize) -> &Vec<usize> {
        &self.cluster_to_sites[cluster_idx]
    }

    #[inline(always)]
    pub fn sites_with_tag(&self, tag_idx: usize) -> &Vec<usize> {
        &self.tag_to_sites[tag_idx]
    }

    #[inline(always)]
    pub fn sites_in_door(&self, door_idx: usize) -> &Vec<usize> {
        &self.door_to_sites[door_idx]
    }

    #[inline(always)]
    pub fn sites_in_tunnel(&self, door_idx: usize) -> &Vec<usize> { // NEW
        &self.tunnel_to_sites[door_idx]
    }
}

