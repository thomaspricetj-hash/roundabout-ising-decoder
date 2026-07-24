#[derive(Debug, Clone)]
pub struct RevolvingDoor {
    pub id: usize,
    pub entry_sites: Vec<usize>,
    pub exit_sites: Vec<usize>,
}

impl RevolvingDoor {
    pub fn new(id: usize, entry_sites: Vec<usize>, exit_sites: Vec<usize>) -> Self {
        Self { id, entry_sites, exit_sites }
    }
}
