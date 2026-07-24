use crate::types::Correction;

/// Simple Ising-like energy:
/// - cost per non-identity op
/// - neighbor interaction penalty (encourage smooth chains)
pub fn compute_energy(correction: &Correction, width: usize, height: usize) -> f64 {
    let mut energy = 0.0;

    // local cost
    for (i, op) in correction.ops.iter().enumerate() {
        if *op != 0 {
            energy += 1.0;
            let x = i % width;
            let y = i / width;

            let neighbors = [
                (x as isize + 1, y as isize),
                (x as isize - 1, y as isize),
                (x as isize, y as isize + 1),
                (x as isize, y as isize - 1),
            ];

            for (nx, ny) in neighbors {
                if nx < 0 || ny < 0 || nx >= width as isize || ny >= height as isize {
                    continue;
                }
                let nidx = (ny as usize) * width + (nx as usize);
                if correction.ops[nidx] != 0 {
                    energy += 0.2; // neighbor interaction
                }
            }
        }
    }

    energy
}
