use crate::types::{Syndrome, Correction, LatticeGeometry};
use crate::heatmaps::{SyndromeHeatmap, GeometryHeatmap};
use crate::scratchpads::{SpatialScratchpad, SemanticScratchpad};
use crate::indexes::{SpatialIndex, SemanticIndex};
use crate::roundabout::{RoundaboutPreferences, RevolvingDoor};
use crate::ising::IsingSolver;
use crate::predictor::Predictor;

pub struct RoundaboutIsingDecoder {
    geom: LatticeGeometry,
    roundabout: RoundaboutPreferences,
    spatial_index: SpatialIndex,
    semantic_index: SemanticIndex,
    doors: Vec<RevolvingDoor>,
    solver: IsingSolver,
    predictor: Predictor,
}

impl RoundaboutIsingDecoder {
    pub fn new(
        geom: LatticeGeometry,
        roundabout: RoundaboutPreferences,
        spatial_index: SpatialIndex,
        semantic_index: SemanticIndex,
        doors: Vec<RevolvingDoor>,
    ) -> Self {
        let solver = IsingSolver::new(geom.width, geom.height);
        Self {
            geom,
            roundabout,
            spatial_index,
            semantic_index,
            doors,
            solver,
            predictor: Predictor::new(),
        }
    }

    pub fn decode(&mut self, syndrome: &Syndrome) -> Correction {
        let syn_heat = SyndromeHeatmap::from_syndrome(syndrome, &self.geom);
        let geo_heat = GeometryHeatmap::uniform(&self.geom, 0.5);

        let fused_heat = self.fuse_heatmaps(&syn_heat, &geo_heat);

        let spatial_sp = SpatialScratchpad::build(&syn_heat, &geo_heat);
        let semantic_sp = SemanticScratchpad::build(syndrome);

        let predicted = self
            .predictor
            .predict(syndrome, &spatial_sp, &semantic_sp, &fused_heat, &self.doors);

        let mut candidates = self.initial_candidates(&spatial_sp, &semantic_sp, &fused_heat);
        candidates.push(predicted);

        let biased = self.apply_roundabout_bias(candidates, &fused_heat);

        self.solver.minimize(syndrome, biased)
    }

    fn fuse_heatmaps(
        &self,
        syn: &SyndromeHeatmap,
        geo: &GeometryHeatmap,
    ) -> Vec<f32> {
        syn.cells
            .iter()
            .zip(geo.cells.iter())
            .map(|(s, g)| {
                let base = 0.7 * s + 0.3 * g;
                base
            })
            .collect()
    }

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

    fn apply_roundabout_bias(
        &self,
        mut candidates: Vec<Correction>,
        fused_heat: &Vec<f32>,
    ) -> Vec<Correction> {
        for door in &self.doors {
            let entry_avg = self.avg_heat(fused_heat, &door.entry_sites);
            let exit_avg = self.avg_heat(fused_heat, &door.exit_sites);

            let flow_bias = (exit_avg - entry_avg) * self.roundabout.exit_bias as f32;

            for c in &mut candidates {
                for &idx in &door.exit_sites {
                    if c.ops[idx] != 0 {
                        c.energy += flow_bias as f64 * 0.5;
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

