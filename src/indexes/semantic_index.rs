use crate::types::Correction;

#[derive(Debug, Clone)]
pub struct SemanticIndex {
    pub pattern_templates: Vec<(u64, Correction)>,
}

impl SemanticIndex {
    pub fn empty() -> Self {
        Self {
            pattern_templates: Vec::new(),
        }
    }
}
