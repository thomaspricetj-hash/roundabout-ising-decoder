#[derive(Clone)]
pub struct RoundaboutPreferences {
    /// Bias toward exiting the roundabout.
    pub exit_bias: f32,

    /// Bias toward following curvature / flow direction.
    pub curvature_bias: f32,

    /// Bias toward lateral escape routes.
    pub lateral_escape_bias: f32,

    // --- NEW: Tunneling Preferences ---

    /// Learned bias toward selecting tunnel exits.
    pub tunnel_bias: f32,

    /// Reliability score for tunnel routing (0.0–1.0).
    pub tunnel_reliability: f32,

    /// Penalty applied when tunnel jitter or instability is detected.
    pub tunnel_stability_penalty: f32,

    /// Bias applied when congestion is detected on physical exits.
    pub tunnel_fallback_bias: f32,
}

impl RoundaboutPreferences {
    /// Default balanced profile.
    #[inline(always)]
    pub fn default() -> Self {
        Self {
            exit_bias: 1.0,
            curvature_bias: 1.0,
            lateral_escape_bias: 1.0,

            tunnel_bias: 0.0,
            tunnel_reliability: 1.0,
            tunnel_stability_penalty: 0.0,
            tunnel_fallback_bias: 0.0,
        }
    }

    /// Aggressive exit profile (forces early exit).
    #[inline(always)]
    pub fn aggressive_exit() -> Self {
        Self {
            exit_bias: 1.6,
            curvature_bias: 0.9,
            lateral_escape_bias: 0.7,

            tunnel_bias: 0.2,
            tunnel_reliability: 1.0,
            tunnel_stability_penalty: 0.0,
            tunnel_fallback_bias: 0.1,
        }
    }

    /// Conservative profile (stay circulating longer).
    #[inline(always)]
    pub fn conservative() -> Self {
        Self {
            exit_bias: 0.7,
            curvature_bias: 1.2,
            lateral_escape_bias: 1.1,

            tunnel_bias: 0.1,
            tunnel_reliability: 1.0,
            tunnel_stability_penalty: 0.0,
            tunnel_fallback_bias: 0.2,
        }
    }

    /// High‑curvature profile (follow flow strongly).
    #[inline(always)]
    pub fn high_curvature() -> Self {
        Self {
            exit_bias: 0.9,
            curvature_bias: 1.5,
            lateral_escape_bias: 0.8,

            tunnel_bias: 0.15,
            tunnel_reliability: 1.0,
            tunnel_stability_penalty: 0.0,
            tunnel_fallback_bias: 0.15,
        }
    }

    /// Normalize all biases so the strongest becomes 1.0.
    #[inline(always)]
    pub fn normalize(&mut self) {
        let max = self
            .exit_bias
            .max(self.curvature_bias)
            .max(self.lateral_escape_bias)
            .max(self.tunnel_bias);

        if max > 1e-6 {
            self.exit_bias /= max;
            self.curvature_bias /= max;
            self.lateral_escape_bias /= max;
            self.tunnel_bias /= max;
        }
    }

    /// Clamp all biases to a safe range.
    #[inline(always)]
    pub fn clamp(&mut self, min: f32, max: f32) {
        self.exit_bias = self.exit_bias.clamp(min, max);
        self.curvature_bias = self.curvature_bias.clamp(min, max);
        self.lateral_escape_bias = self.lateral_escape_bias.clamp(min, max);

        self.tunnel_bias = self.tunnel_bias.clamp(min, max);
        self.tunnel_reliability = self.tunnel_reliability.clamp(0.0, 1.0);
        self.tunnel_stability_penalty = self.tunnel_stability_penalty.clamp(0.0, max);
        self.tunnel_fallback_bias = self.tunnel_fallback_bias.clamp(min, max);
    }

    /// Blend two preference profiles.
    #[inline(always)]
    pub fn blend(&mut self, other: &RoundaboutPreferences, alpha: f32) {
        let a = alpha.clamp(0.0, 1.0);
        let b = 1.0 - a;

        self.exit_bias = a * self.exit_bias + b * other.exit_bias;
        self.curvature_bias = a * self.curvature_bias + b * other.curvature_bias;
        self.lateral_escape_bias = a * self.lateral_escape_bias + b * other.lateral_escape_bias;

        self.tunnel_bias = a * self.tunnel_bias + b * other.tunnel_bias;
        self.tunnel_reliability = a * self.tunnel_reliability + b * other.tunnel_reliability;
        self.tunnel_stability_penalty =
            a * self.tunnel_stability_penalty + b * other.tunnel_stability_penalty;
        self.tunnel_fallback_bias =
            a * self.tunnel_fallback_bias + b * other.tunnel_fallback_bias;
    }

    /// Zero‑cost inline: return as tuple for SIMD‑friendly operations.
    #[inline(always)]
    pub fn as_tuple(&self) -> (f32, f32, f32, f32) {
        (
            self.exit_bias,
            self.curvature_bias,
            self.lateral_escape_bias,
            self.tunnel_bias,
        )
    }
}
