use crate::types::{Syndrome, Correction};
use crate::ising::compute_energy;

#[derive(Debug, Clone)]
pub struct IsingSolver {
    pub width: usize,
    pub height: usize,
}

impl IsingSolver {
    pub fn new(width: usize, height: usize) -> Self {
        Self { width, height }
    }

    pub fn minimize(&self, _syndrome: &Syndrome, mut candidates: Vec<Correction>) -> Correction {
        for c in &mut candidates {
            c.energy = compute_energy(c, self.width, self.height);
        }

        candidates
            .into_iter()
            .min_by(|a, b| a.energy.partial_cmp(&b.energy).unwrap())
            .unwrap_or(Correction {
                ops: vec![0; self.width * self.height],
                energy: 0.0,
            })
    }
}
