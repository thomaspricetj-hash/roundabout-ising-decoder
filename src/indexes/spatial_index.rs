use crate::types::Correction;

#[derive(Debug, Clone)]
pub struct SpatialIndex {
    pub region_templates: Vec<Vec<Correction>>,
}

impl SpatialIndex {
    pub fn empty() -> Self {
        Self {
            region_templates: Vec::new(),
        }
    }
}
