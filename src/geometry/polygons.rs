#[derive(Debug, Clone)]
pub struct PolygonRegion {
    pub id: usize,
    pub sites: Vec<usize>, // indices into lattice sites
}
