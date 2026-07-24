use crate::roundabout::RevolvingDoor;

pub trait GpuBackend: Send + Sync {
    fn predictor_pass(
        &self,
        syndrome_bits: &[u8],
        fused_heat: &[f32],
        doors: &[RevolvingDoor],
    ) -> Vec<u8>;

    fn smooth_chains(
        &self,
        ops: &mut [u8],
    );

    fn door_routing(
        &self,
        ops: &mut [u8],
        fused_heat: &[f32],
        doors: &[RevolvingDoor],
    );

    fn decay_pattern_memory(
        &self,
        weights: &mut [f32],
    );
}
