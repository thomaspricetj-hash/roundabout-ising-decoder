// Core roundabout modules
pub mod preferences;
pub mod revolving_door;

// Re‑exports
pub use preferences::RoundaboutPreferences;
pub use revolving_door::RevolvingDoor;

// --- NEW: Tunneling types and helpers ---

/// Tunnel metrics used by both preferences and revolving doors.
#[derive(Debug, Clone)]
pub struct TunnelMetrics {
    /// Estimated latency of tunnel path.
    pub latency: f32,

    /// Estimated jitter of tunnel path.
    pub jitter: f32,

    /// Estimated packet/step loss rate.
    pub loss_rate: f32,

    /// Congestion level (0.0–1.0).
    pub congestion: f32,

    /// Stability score (0.0–1.0).
    pub stability: f32,
}

impl TunnelMetrics {
    #[inline(always)]
    pub fn default() -> Self {
        Self {
            latency: 0.0,
            jitter: 0.0,
            loss_rate: 0.0,
            congestion: 0.0,
            stability: 1.0,
        }
    }

    #[inline(always)]
    pub fn reliability_score(&self) -> f32 {
        // Lower latency, jitter, loss, congestion → higher reliability
        let penalty = 
            self.latency * 0.15 +
            self.jitter * 0.20 +
            self.loss_rate * 0.30 +
            self.congestion * 0.25;

        (1.0 - penalty).clamp(0.0, 1.0)
    }
}
