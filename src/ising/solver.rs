use crate::types::{Syndrome, Correction};
use crate::ising::compute_energy;
use crate::scratchpads::{SpatialScratchpad, SemanticScratchpad};
use crate::roundabout::RevolvingDoor;
use crate::spatial::cross_link_grid::CrossLinkGrid;

/// High-performance Ising solver:
/// - branch-minimized candidate evaluation
/// - best-two tracking (faster than full sort)
/// - cross-layer energy shaping (clusters, doors, semantics)
#[derive(Clone)]
pub struct IsingSolver {
    pub width: usize,
    pub height: usize,
}

impl IsingSolver {
    #[inline(always)]
    pub fn new(width: usize, height: usize) -> Self {
        Self { width, height }
    }

    /// Minimize energy across candidate corrections.
    /// Includes:
    /// - base Ising energy
    /// - cluster cohesion energy
    /// - door alignment energy
    /// - semantic tag energy
    pub fn minimize(
        &self,
        _syndrome: &Syndrome,
        mut candidates: Vec<Correction>,
        spatial: &SpatialScratchpad,
        semantic: &SemanticScratchpad,
        doors: &Vec<RevolvingDoor>,
        cross: &CrossLinkGrid,
    ) -> Correction {
        let w = self.width;
        let h = self.height;

        if candidates.is_empty() {
            return Correction {
                ops: vec![0; w * h],
                energy: 0.0,
            };
        }

        let mut best: Option<Correction> = None;
        let mut second_best: Option<Correction> = None;

        for mut c in candidates.drain(..) {
            // Base Ising energy
            c.energy = compute_energy(&c, w, h);

            // Cross-layer shaping
            self.apply_cluster_energy(&mut c, spatial, cross);
            self.apply_door_energy(&mut c, doors, cross);
            self.apply_semantic_energy(&mut c, semantic, cross);

            match &best {
                None => best = Some(c),
                Some(b) => {
                    if c.energy < b.energy {
                        second_best = best.take();
                        best = Some(c);
                    } else {
                        match &second_best {
                            None => second_best = Some(c),
                            Some(sb) => {
                                if c.energy < sb.energy {
                                    second_best = Some(c);
                                }
                            }
                        }
                    }
                }
            }
        }

        best.unwrap()
    }

    // -------------------------------------------------------------------------
    // Cluster cohesion energy: reward ops inside strong clusters
    // -------------------------------------------------------------------------
    #[inline(always)]
    fn apply_cluster_energy(
        &self,
        corr: &mut Correction,
        spatial: &SpatialScratchpad,
        cross: &CrossLinkGrid,
    ) {
        for (idx, op) in corr.ops.iter().enumerate() {
            if *op == 0 {
                continue;
            }

            if let Some(cluster_idx) = cross.cluster_for(idx) {
                let c_energy = spatial.local_energy[cluster_idx];
                // reward staying inside high-energy clusters
                corr.energy -= (c_energy * 0.05) as f64;
            }
        }
    }

    // -------------------------------------------------------------------------
    // Door alignment energy: reward exit alignment, penalize entry congestion
    // -------------------------------------------------------------------------
    #[inline(always)]
    fn apply_door_energy(
        &self,
        corr: &mut Correction,
        doors: &Vec<RevolvingDoor>,
        cross: &CrossLinkGrid,
    ) {
        for (idx, op) in corr.ops.iter().enumerate() {
            if *op == 0 {
                continue;
            }

            if let Some(door_id) = cross.door_for(idx) {
                if let Some(door) = doors.iter().find(|d| d.id == door_id) {
                    if door.exit_sites.contains(&idx) {
                        // reward corrections on exits
                        corr.energy -= 0.2;
                    }
                    if door.entry_sites.contains(&idx) {
                        // slight penalty on entries
                        corr.energy += 0.1;
                    }
                }
            }
        }
    }

    // -------------------------------------------------------------------------
    // Semantic tag energy: reward ops in strong semantic regions
    // -------------------------------------------------------------------------
    #[inline(always)]
    fn apply_semantic_energy(
        &self,
        corr: &mut Correction,
        semantic: &SemanticScratchpad,
        cross: &CrossLinkGrid,
    ) {
        for (idx, op) in corr.ops.iter().enumerate() {
            if *op == 0 {
                continue;
            }

            if let Some(tag_idx) = cross.semantic_tag_for(idx) {
                if tag_idx < semantic.pattern_tags.len() {
                    let tag = semantic.pattern_tags[tag_idx];
                    corr.energy -= tag as f64 * 0.01;
                }
            }
        }
    }
}

