use crate::types::{Syndrome, Correction, LatticeGeometry};
use crate::heatmaps::{SyndromeHeatmap, GeometryHeatmap};
use crate::scratchpads::{SpatialScratchpad, SemanticScratchpad};
use crate::roundabout::{RoundaboutPreferences, RevolvingDoor};
use crate::ising::IsingSolver;
use crate::predictor::Predictor;
use crate::spatial::cross_link_grid::CrossLinkGrid;

pub struct RoundaboutIsingDecoder {
    geom: LatticeGeometry,
    roundabout: RoundaboutPreferences,
    doors: Vec<RevolvingDoor>,
    solver: IsingSolver,
    predictor: Predictor,
}

impl RoundaboutIsingDecoder {
    pub fn new(
        geom: LatticeGeometry,
        roundabout: RoundaboutPreferences,
        _spatial_index: crate::indexes::SpatialIndex,
        _semantic_index: crate::indexes::SemanticIndex,
        doors: Vec<RevolvingDoor>,
    ) -> Self {
        let solver = IsingSolver::new(geom.width, geom.height);
        Self {
            geom,
            roundabout,
            doors,
            solver,
            predictor: Predictor::new(),
        }
    }

    pub fn decode(&mut self, syndrome: &Syndrome) -> Correction {
        let syn_heat = SyndromeHeatmap::from_syndrome(syndrome, &self.geom);
        let geo_heat = GeometryHeatmap::uniform(&self.geom, 0.5);

        // --- NEW: tunnel-aware fused heat ---
        let fused_heat = self.fuse_heatmaps(&syn_heat, &geo_heat);

        let spatial_sp = SpatialScratchpad::build(&syn_heat, &geo_heat);
        let semantic_sp = SemanticScratchpad::build(syndrome);

        let cross = CrossLinkGrid::build(
            &self.geom,
            &spatial_sp,
            &semantic_sp,
            &self.doors,
        );

        // --- NEW: update tunnel metrics for each door ---
        self.update_tunnel_metrics(&fused_heat, &semantic_sp);

        let predicted = self.predictor.predict(
            syndrome,
            &spatial_sp,
            &semantic_sp,
            &fused_heat,
            &self.doors,
            &cross,
        );

        let mut candidates = self.initial_candidates(&spatial_sp, &semantic_sp, &fused_heat);
        candidates.push(predicted);

        let biased = self.apply_roundabout_bias(candidates, &fused_heat);

        self.solver.minimize(
            syndrome,
            biased,
            &spatial_sp,
            &semantic_sp,
            &self.doors,
            &cross,
        )
    }

    // -------------------------------------------------------------------------
    // NEW: tunnel-aware fused heat
    // -------------------------------------------------------------------------
    fn fuse_heatmaps(
        &self,
        syn: &SyndromeHeatmap,
        geo: &GeometryHeatmap,
    ) -> Vec<f32> {
        syn.cells
            .iter()
            .zip(geo.cells.iter())
            .map(|(s, g)| 0.7 * s + 0.3 * g)
            .collect()
    }

    // -------------------------------------------------------------------------
    // NEW: tunnel metric update
    // -------------------------------------------------------------------------
    fn update_tunnel_metrics(
        &mut self,
        fused_heat: &Vec<f32>,
        semantic: &SemanticScratchpad,
    ) {
        for door in &mut self.doors {
            // tunnel heat
            let t_heat = door.avg_tunnel_heat(fused_heat);
            door.tunnel_heat = t_heat;

            // tunnel score
            let score =
                t_heat * 0.4 +
                semantic.tunnel_strength * 0.3 +
                semantic.tunnel_bias * 0.2 -
                semantic.tunnel_penalty * 0.1;

            door.tunnel_score = score;

            // tunnel reliability (simple heuristic)
            let rel = (1.0 - semantic.tunnel_penalty).clamp(0.0, 1.0);
            door.tunnel_reliability = rel;
        }
    }

    // -------------------------------------------------------------------------
    // Initial candidates (unchanged)
    // -------------------------------------------------------------------------
    fn initial_candidates(
        &self,
        spatial_sp: &SpatialScratchpad,
        _semantic_sp: &SemanticScratchpad,
        fused_heat: &Vec<f32>,
    ) -> Vec<Correction> {
        let mut candidates = Vec::new();

        for cluster in &spatial_sp.local_clusters {
            let mut ops = vec![0u8; self.geom.width * self.geom.height];
            for &idx in cluster {
                if fused_heat[idx] > 0.0 {
                    ops[idx] = 1;
                }
            }
            candidates.push(Correction { ops, energy: 0.0 });
        }

        if candidates.is_empty() {
            candidates.push(Correction {
                ops: vec![0; self.geom.width * self.geom.height],
                energy: 0.0,
            });
        }

        candidates
    }

    // -------------------------------------------------------------------------
    // NEW: tunnel-aware roundabout bias
    // -------------------------------------------------------------------------
    fn apply_roundabout_bias(
        &self,
        mut candidates: Vec<Correction>,
        fused_heat: &Vec<f32>,
    ) -> Vec<Correction> {
        for door in &self.doors {
            let entry_avg = self.avg_heat(fused_heat, &door.entry_sites);
            let exit_avg = self.avg_heat(fused_heat, &door.exit_sites);

            let flow_bias = (exit_avg - entry_avg) * self.roundabout.exit_bias as f32;

            // --- NEW: tunnel bias contribution ---
            let tunnel_bias = door.tunnel_score * self.roundabout.tunnel_bias;

            for c in &mut candidates {
                for &idx in &door.exit_sites {
                    if c.ops[idx] != 0 {
                        c.energy += (flow_bias as f64 * 0.5)
                            + (tunnel_bias as f64 * 0.3);
                    }
                }
            }
        }

        for c in &mut candidates {
            c.energy += (self.roundabout.curvature_bias
                + self.roundabout.lateral_escape_bias) as f64
                * 0.01;
        }

        candidates
    }

    fn avg_heat(&self, fused_heat: &Vec<f32>, sites: &Vec<usize>) -> f32 {
        if sites.is_empty() {
            return 0.0;
        }
        let sum: f32 = sites.iter().map(|idx| fused_heat[*idx]).sum();
        sum / (sites.len() as f32)
    }
}


