use crate::types::{Syndrome, Correction};
use crate::scratchpads::{SpatialScratchpad, SemanticScratchpad};
use crate::roundabout::RevolvingDoor;
use crate::gpu::GpuBackend;
use crate::spatial::cross_link_grid::CrossLinkGrid;

/// High‑performance cognitive predictor:
/// - GPU‑accelerated fast path
/// - CPU fallback path
/// - semantic refinement
/// - door routing
/// - chain smoothing
/// - pattern memory with decay
/// - cross‑link grid integration
#[derive(Clone)]
pub struct Predictor {
    pub weight_syndrome: f32,
    pub weight_geometry: f32,
    pub weight_semantic: f32,
    pub weight_doors: f32,
    pub weight_smoothing: f32,

    pub pattern_memory: Vec<(u64, Correction, f32)>,
    pub gpu: Option<std::sync::Arc<dyn GpuBackend>>,
}

impl Predictor {
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            weight_syndrome: 1.0,
            weight_geometry: 0.7,
            weight_semantic: 0.5,
            weight_doors: 0.6,
            weight_smoothing: 0.4,
            pattern_memory: Vec::with_capacity(256),
            gpu: None,
        }
    }

    #[inline(always)]
    pub fn with_gpu(backend: std::sync::Arc<dyn GpuBackend>) -> Self {
        let mut p = Self::new();
        p.gpu = Some(backend);
        p
    }

    #[inline(always)]
    pub fn tune_weights(
        &mut self,
        ws: f32,
        wg: f32,
        wsem: f32,
        wdoor: f32,
        wsmooth: f32,
    ) {
        self.weight_syndrome = ws;
        self.weight_geometry = wg;
        self.weight_semantic = wsem;
        self.weight_doors = wdoor;
        self.weight_smoothing = wsmooth;
    }

    /// Main prediction pipeline.
    pub fn predict(
        &mut self,
        syndrome: &Syndrome,
        spatial: &SpatialScratchpad,
        semantic: &SemanticScratchpad,
        fused_heat: &Vec<f32>,
        doors: &Vec<RevolvingDoor>,
        cross: &CrossLinkGrid,        // <-- NEW
    ) -> Correction {
        // --- 1. Base prediction (GPU or CPU) ---
        let mut corr = if let Some(gpu) = &self.gpu {
            let ops = gpu.predictor_pass(&syndrome.bits, fused_heat, doors);
            Correction { ops, energy: 0.0 }
        } else {
            self.base_prediction(spatial, fused_heat, cross)
        };

        // --- 2. Semantic refinement ---
        self.semantic_refine(&mut corr, semantic, fused_heat, cross);

        // --- 3. Door routing + smoothing ---
        if let Some(gpu) = &self.gpu {
            gpu.door_routing(&mut corr.ops, fused_heat, doors);
            gpu.smooth_chains(&mut corr.ops);
        } else {
            self.apply_door_routing(&mut corr, fused_heat, doors, cross);
            self.smooth_chains(&mut corr);
        }

        // --- 4. Pattern memory ---
        self.apply_pattern_memory(syndrome, &mut corr);
        self.decay_memory();
        self.store_pattern(syndrome, &corr);

        corr
    }

    // -------------------------------------------------------------------------
    // CPU base prediction + cross‑link cluster bias
    // -------------------------------------------------------------------------
    #[inline(always)]
    fn base_prediction(
        &self,
        spatial: &SpatialScratchpad,
        fused_heat: &Vec<f32>,
        cross: &CrossLinkGrid,
    ) -> Correction {
        let mut ops = vec![0u8; fused_heat.len()];

        for cluster in &spatial.local_clusters {
            for &idx in cluster {
                let score = fused_heat[idx] * self.weight_syndrome;

                // cluster‑aware boosting
                if let Some(cluster_idx) = cross.cluster_for(idx) {
                    let c_energy = spatial.local_energy[cluster_idx];
                    if c_energy > 2.0 {
                        ops[idx] = 1;
                        continue;
                    }
                }

                if score > 0.3 {
                    ops[idx] = 1;
                }
            }
        }

        Correction { ops, energy: 0.0 }
    }

    // -------------------------------------------------------------------------
    // Semantic refinement + cross‑link semantic tag bias
    // -------------------------------------------------------------------------
    #[inline(always)]
    fn semantic_refine(
        &self,
        corr: &mut Correction,
        semantic: &SemanticScratchpad,
        fused_heat: &Vec<f32>,
        cross: &CrossLinkGrid,
    ) {
        for (idx, op) in corr.ops.iter_mut().enumerate() {
            if fused_heat[idx] > 0.6 && *op == 1 {
                if let Some(tag_idx) = cross.semantic_tag_for(idx) {
                    let tag = semantic.pattern_tags[tag_idx];
                    if tag > 3 {
                        *op = 2;
                    }
                }
            }
        }
    }

    // -------------------------------------------------------------------------
    // Door routing + cross‑link door bias
    // -------------------------------------------------------------------------
    #[inline(always)]
    fn apply_door_routing(
        &self,
        corr: &mut Correction,
        fused_heat: &Vec<f32>,
        doors: &Vec<RevolvingDoor>,
        cross: &CrossLinkGrid,
    ) {
        for door in doors {
            let entry_avg = self.avg_heat(fused_heat, &door.entry_sites);
            let exit_avg = self.avg_heat(fused_heat, &door.exit_sites);

            let favor_exit = exit_avg > entry_avg;

            for &idx in &door.exit_sites {
                let heat = fused_heat[idx];

                // door‑aware bias
                if let Some(door_id) = cross.door_for(idx) {
                    if door_id == door.id && heat > 0.35 {
                        corr.ops[idx] = 1;
                        continue;
                    }
                }

                if favor_exit {
                    if heat > 0.4 && corr.ops[idx] == 0 {
                        corr.ops[idx] = 1;
                    }
                } else {
                    if heat < 0.2 {
                        corr.ops[idx] = 0;
                    }
                }
            }
        }
    }

    // -------------------------------------------------------------------------
    // Chain smoothing
    // -------------------------------------------------------------------------
    #[inline(always)]
    fn smooth_chains(&self, corr: &mut Correction) {
        let ops = &mut corr.ops;
        let len = ops.len();
        if len < 3 {
            return;
        }

        for i in 1..(len - 1) {
            let left = ops[i - 1];
            let mid = ops[i];
            let right = ops[i + 1];

            if left == right && mid == 0 && left != 0 {
                ops[i] = left;
            }
        }
    }

    // -------------------------------------------------------------------------
    // Pattern memory application
    // -------------------------------------------------------------------------
    #[inline(always)]
    fn apply_pattern_memory(&self, syndrome: &Syndrome, corr: &mut Correction) {
        let key = self.pattern_key(syndrome);

        if let Some((_, stored, _)) =
            self.pattern_memory.iter().find(|(k, _, _)| *k == key)
        {
            for (i, op) in corr.ops.iter_mut().enumerate() {
                let s = stored.ops[i];
                if s != 0 && *op == 0 {
                    *op = s;
                }
            }
        }
    }

    // -------------------------------------------------------------------------
    // Pattern memory storage
    // -------------------------------------------------------------------------
    #[inline(always)]
    fn store_pattern(&mut self, syndrome: &Syndrome, corr: &Correction) {
        let key = self.pattern_key(syndrome);

        if let Some(slot) =
            self.pattern_memory.iter_mut().find(|(k, _, _)| *k == key)
        {
            slot.1 = corr.clone();
            slot.2 = 1.0;
            return;
        }

        self.pattern_memory.push((key, corr.clone(), 1.0));

        if self.pattern_memory.len() > 256 {
            self.pattern_memory.remove(0);
        }
    }

    // -------------------------------------------------------------------------
    // Pattern memory decay
    // -------------------------------------------------------------------------
    #[inline(always)]
    fn decay_memory(&mut self) {
        if let Some(gpu) = &self.gpu {
            let mut weights: Vec<f32> =
                self.pattern_memory.iter().map(|(_, _, w)| *w).collect();

            gpu.decay_pattern_memory(&mut weights);

            for (i, (_, _, w)) in self.pattern_memory.iter_mut().enumerate() {
                *w = weights[i];
            }
        } else {
            for (_, _, w) in &mut self.pattern_memory {
                *w *= 0.98;
            }
        }

        self.pattern_memory.retain(|(_, _, w)| *w > 0.1);
    }

    // -------------------------------------------------------------------------
    // Pattern key generator
    // -------------------------------------------------------------------------
    #[inline(always)]
    fn pattern_key(&self, syndrome: &Syndrome) -> u64 {
        let mut count = 0u64;
        let mut roll = 0u64;

        for (i, bit) in syndrome.bits.iter().enumerate() {
            if *bit != 0 {
                count += 1;
                roll ^= 1u64 << (i & 63);
            }
        }

        count ^ (roll << 1)
    }

    // -------------------------------------------------------------------------
    // Average heat utility
    // -------------------------------------------------------------------------
    #[inline(always)]
    fn avg_heat(&self, fused_heat: &Vec<f32>, sites: &Vec<usize>) -> f32 {
        if sites.is_empty() {
            return 0.0;
        }
        let mut sum = 0.0f32;
        for &idx in sites {
            sum += fused_heat[idx];
        }
        sum / (sites.len() as f32)
    }
}



