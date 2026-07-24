use crate::types::{Syndrome, Correction};

#[derive(Debug, Clone)]
pub struct SemanticScratchpad {
    pub pattern_tags: Vec<u64>,
    pub pattern_hints: Vec<Correction>,
}

impl SemanticScratchpad {
    pub fn build(syndrome: &Syndrome) -> Self {
        // Simple hash: sum of bits as a "pattern tag"
        let tag = syndrome.bits.iter().map(|b| *b as u64).sum();
        let hint = Correction {
            ops: vec![0; syndrome.bits.len()],
            energy: 0.0,
        };
        Self {
            pattern_tags: vec![tag],
            pattern_hints: vec![hint],
        }
    }
}
