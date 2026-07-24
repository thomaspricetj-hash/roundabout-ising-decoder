use crate::types::Syndrome;
use crate::types::LatticeGeometry;

#[derive(Debug, Clone)]
pub struct SyndromeHeatmap {
    pub cells: Vec<f32>,
    pub width: usize,
    pub height: usize,
}

impl SyndromeHeatmap {
    pub fn from_syndrome(syndrome: &Syndrome, geom: &LatticeGeometry) -> Self {
        let mut cells = vec![0.0; geom.width * geom.height];
        for (i, bit) in syndrome.bits.iter().enumerate() {
            if i < cells.len() {
                cells[i] = *bit as f32;
            }
        }
        Self {
            cells,
            width: geom.width,
            height: geom.height,
        }
    }
}
