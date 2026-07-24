use crate::types::{Syndrome, Correction};
use crate::scratchpads::{SpatialScratchpad, SemanticScratchpad};
use crate::roundabout::RevolvingDoor;
use crate::gpu::GpuBackend;

#[derive(Clone)]
pub struct Predictor {
    pub weight_syndrome: f32,
    pub weight_geometry: f32,
    pub weight_semantic: f32,
    pub weight_doors: f32,
    pub weight_smoothing: f32,

    pub pattern_memory: Vec<(u64, Correction, f32)>, // f32 = decay weight
    pub gpu: Option<std::sync::Arc<dyn GpuBackend>>,
}

impl Predictor {
    pub fn new() -> Self {
        Self {
            weight_syndrome: 1.0,
            weight_geometry: 0.7,
            weight_semantic: 0.5,
            weight_doors: 0.6,
            weight_smoothing: 0.4,
            pattern_memory: Vec::new(),
            gpu: None,
        }
    }

    pub fn with_gpu(backend: std::sync::Arc<dyn GpuBackend>) -> Self {
        let mut p = Self::new();
        p.gpu = Some(backend);
        p
    }

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

    pub fn predict(
        &mut self,
        syndrome: &Syndrome,
        spatial: &SpatialScratchpad,
        semantic: &SemanticScratchpad,
        fused_heat: &Vec<f32>,
        doors: &Vec<RevolvingDoor>,
    ) -> Correction {
        let mut corr = if let Some(gpu) = &self.gpu {
            let ops = gpu.predictor_pass(&syndrome.bits, fused_heat, doors);
            Correction { ops, energy: 0.0 }
        } else {
            self.base_prediction(spatial, fused_heat)
        };

        self.semantic_refine(&mut corr, semantic, fused_heat);

        if let Some(gpu) = &self.gpu {
            gpu.door_routing(&mut corr.ops, fused_heat, doors);
            gpu.smooth_chains(&mut corr.ops);
        } else {
            self.apply_door_routing(&mut corr, fused_heat, doors);
            self.smooth_chains(&mut corr);
        }

        self.apply_pattern_memory(syndrome, &mut corr);
        self.decay_memory();
        self.store_pattern(syndrome, &corr);

        corr
    }

    fn base_prediction(
        &self,
        spatial: &SpatialScratchpad,
        fused_heat: &Vec<f32>,
    ) -> Correction {
        let mut ops = vec![0u8; fused_heat.len()];

        for cluster in &spatial.local_clusters {
            for &idx in cluster {
                let score = fused_heat[idx] * self.weight_syndrome;
                if score > 0.3 {
                    ops[idx] = 1;
                }
            }
        }

        Correction { ops, energy: 0.0 }
    }

    fn semantic_refine(
        &self,
        corr: &mut Correction,
        semantic: &SemanticScratchpad,
        fused_heat: &Vec<f32>,
    ) {
        if let Some(tag) = semantic.pattern_tags.first() {
            if *tag > 3 {
                for (idx, op) in corr.ops.iter_mut().enumerate() {
                    if fused_heat[idx] > 0.6 && *op == 1 {
                        *op = 2;
                    }
                }
            }
        }
    }

    fn apply_door_routing(
        &self,
        corr: &mut Correction,
        fused_heat: &Vec<f32>,
        doors: &Vec<RevolvingDoor>,
    ) {
        for door in doors {
            let entry_avg = self.avg_heat(fused_heat, &door.entry_sites);
            let exit_avg = self.avg_heat(fused_heat, &door.exit_sites);

            if exit_avg > entry_avg {
                for &idx in &door.exit_sites {
                    if fused_heat[idx] > 0.4 && corr.ops[idx] == 0 {
                        corr.ops[idx] = 1;
                    }
                }
            } else {
                for &idx in &door.exit_sites {
                    if fused_heat[idx] < 0.2 {
                        corr.ops[idx] = 0;
                    }
                }
            }
        }
    }

    fn smooth_chains(&self, corr: &mut Correction) {
        let len = corr.ops.len();
        if len < 3 {
            return;
        }

        for i in 1..(len - 1) {
            let left = corr.ops[i - 1];
            let mid = corr.ops[i];
            let right = corr.ops[i + 1];

            if left == right && mid == 0 && left != 0 {
                corr.ops[i] = left;
            }
        }
    }

    fn apply_pattern_memory(&self, syndrome: &Syndrome, corr: &mut Correction) {
        let key = self.pattern_key(syndrome);
        if let Some((_, stored, _w)) = self
            .pattern_memory
            .iter()
            .find(|(k, _, _)| *k == key)
        {
            for (i, op) in corr.ops.iter_mut().enumerate() {
                if stored.ops[i] != 0 && *op == 0 {
                    *op = stored.ops[i];
                }
            }
        }
    }

    fn store_pattern(&mut self, syndrome: &Syndrome, corr: &Correction) {
        let key = self.pattern_key(syndrome);
        if let Some(slot) = self
            .pattern_memory
            .iter_mut()
            .find(|(k, _, _)| *k == key)
        {
            slot.1 = corr.clone();
            slot.2 = 1.0;
        } else {
            self.pattern_memory.push((key, corr.clone(), 1.0));
            if self.pattern_memory.len() > 256 {
                self.pattern_memory.remove(0);
            }
        }
    }

    fn decay_memory(&mut self) {
        if let Some(gpu) = &self.gpu {
            let mut weights: Vec<f32> = self.pattern_memory.iter().map(|(_, _, w)| *w).collect();
            gpu.decay_pattern_memory(&mut weights);
            for (i, (_, _, w)) in self.pattern_memory.iter_mut().enumerate() {
                *w = weights[i];
            }
        } else {
            for (_, _, w) in &mut self.pattern_memory {
                *w *= 0.98;
            }
        }

        self.pattern_memory
            .retain(|(_, _, w)| *w > 0.1);
    }

    fn pattern_key(&self, syndrome: &Syndrome) -> u64 {
        let mut count = 0u64;
        let mut roll = 0u64;
        for (i, bit) in syndrome.bits.iter().enumerate() {
            if *bit != 0 {
                count += 1;
                roll ^= 1u64 << (i % 63);
            }
        }
        count ^ (roll << 1)
    }

    fn avg_heat(&self, fused_heat: &Vec<f32>, sites: &Vec<usize>) -> f32 {
        if sites.is_empty() {
            return 0.0;
        }
        let sum: f32 = sites.iter().map(|idx| fused_heat[*idx]).sum();
        sum / (sites.len() as f32)
    }
}

