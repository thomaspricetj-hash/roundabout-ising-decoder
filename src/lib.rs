pub mod geometry;
pub mod heatmaps;
pub mod scratchpads;
pub mod indexes;
pub mod roundabout;
pub mod ising;
pub mod decoder;
pub mod types;
pub mod predictor;
pub mod gpu;

// NEW: cross‑layer linking grid
pub mod spatial {
    pub mod cross_link_grid;
}

pub use decoder::RoundaboutIsingDecoder;
pub use types::{Syndrome, LatticeGeometry};
pub use roundabout::{RoundaboutPreferences, RevolvingDoor};
pub use indexes::{SpatialIndex, SemanticIndex};
pub use predictor::Predictor;

// NEW: re‑export CrossLinkGrid
pub use spatial::cross_link_grid::CrossLinkGrid;
