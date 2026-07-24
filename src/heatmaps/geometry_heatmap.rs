use crate::types::LatticeGeometry;

#[derive(Debug, Clone)]
pub struct GeometryHeatmap {
    pub cells: Vec<f32>,
    pub width: usize,
    pub height: usize,
}

impl GeometryHeatmap {
    pub fn uniform(geom: &LatticeGeometry, value: f32) -> Self {
        let cells = vec![value; geom.width * geom.height];
        Self {
            cells,
            width: geom.width,
            height: geom.height,
        }
    }

    pub fn from_cells(geom: &LatticeGeometry, cells: Vec<f32>) -> Self {
        assert!(
            cells.len() == geom.width * geom.height,
            "GeometryHeatmap::from_cells: cell count mismatch"
        );
        Self {
            cells,
            width: geom.width,
            height: geom.height,
        }
    }

    pub fn get(&self, idx: usize) -> f32 {
        self.cells[idx]
    }

    pub fn set(&mut self, idx: usize, value: f32) {
        self.cells[idx] = value;
    }
}
