#[derive(Debug, Clone)]
pub struct RoundaboutPreferences {
    pub exit_bias: f32,
    pub curvature_bias: f32,
    pub lateral_escape_bias: f32,
}

impl RoundaboutPreferences {
    pub fn default() -> Self {
        Self {
            exit_bias: 1.0,
            curvature_bias: 1.0,
            lateral_escape_bias: 1.0,
        }
    }
}
