use crate::types::Correction;

/// High‑performance Ising-like energy function.
/// - local cost per non-identity op
/// - neighbor interaction penalty (smooth chains)
/// - edge/corner bias (optional)
/// Zero-cost inline helpers and branch-minimized hot loop.
#[inline(always)]
pub fn compute_energy(
    correction: &Correction,
    width: usize,
    height: usize,
) -> f64 {
    let ops = &correction.ops;
    let mut energy = 0.0f64;

    let w = width;
    let h = height;
    let n = ops.len();

    debug_assert!(n == w * h);

    // Tunable penalties
    const LOCAL_COST: f64 = 1.0;
    const NEIGHBOR_PENALTY: f64 = 0.2;
    const EDGE_PENALTY: f64 = 0.05;
    const CORNER_PENALTY: f64 = 0.1;

    for idx in 0..n {
        let op = ops[idx];
        if op == 0 {
            continue;
        }

        // Local cost
        energy += LOCAL_COST;

        // Compute (x, y)
        let x = idx % w;
        let y = idx / w;

        // Edge/corner bias (optional)
        let is_left   = x == 0;
        let is_right  = x == w - 1;
        let is_top    = y == 0;
        let is_bottom = y == h - 1;

        let is_edge =
            is_left || is_right || is_top || is_bottom;

        let is_corner =
            (is_left || is_right) &&
            (is_top || is_bottom);

        if is_corner {
            energy += CORNER_PENALTY;
        } else if is_edge {
            energy += EDGE_PENALTY;
        }

        // Neighbor interaction (branch-minimized)
        // Right neighbor
        if x + 1 < w {
            let nidx = idx + 1;
            if ops[nidx] != 0 {
                energy += NEIGHBOR_PENALTY;
            }
        }

        // Left neighbor
        if x > 0 {
            let nidx = idx - 1;
            if ops[nidx] != 0 {
                energy += NEIGHBOR_PENALTY;
            }
        }

        // Down neighbor
        if y + 1 < h {
            let nidx = idx + w;
            if ops[nidx] != 0 {
                energy += NEIGHBOR_PENALTY;
            }
        }

        // Up neighbor
        if y > 0 {
            let nidx = idx - w;
            if ops[nidx] != 0 {
                energy += NEIGHBOR_PENALTY;
            }
        }
    }

    energy
}
