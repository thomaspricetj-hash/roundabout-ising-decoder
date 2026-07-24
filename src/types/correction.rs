#[derive(Debug, Clone)]
pub struct Correction {
    pub ops: Vec<u8>, // 0 = I, 1 = X, 2 = Z, 3 = Y
    pub energy: f64,
}
