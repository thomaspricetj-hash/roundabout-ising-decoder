use crate::types::LatticeGeometry;

pub trait LatticeGeometryExt {
    fn num_sites(&self) -> usize;
}

impl LatticeGeometryExt for LatticeGeometry {
    fn num_sites(&self) -> usize {
        self.width * self.height
    }
}
