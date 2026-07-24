use crate::heatmaps::{SyndromeHeatmap, GeometryHeatmap};

#[derive(Debug, Clone)]
pub struct SpatialScratchpad {
    pub local_clusters: Vec<Vec<usize>>,
    pub local_energy: Vec<f64>,
}

impl SpatialScratchpad {
    pub fn build(syn: &SyndromeHeatmap, geo: &GeometryHeatmap) -> Self {
        let clusters = Self::cluster_active_sites(syn);
        let mut local_energy = Vec::new();

        for cluster in &clusters {
            let energy: f64 = cluster
                .iter()
                .map(|idx| syn.cells[*idx] as f64 + geo.cells[*idx] as f64)
                .sum();
            local_energy.push(energy);
        }

        Self {
            local_clusters: clusters,
            local_energy,
        }
    }

    fn cluster_active_sites(syn: &SyndromeHeatmap) -> Vec<Vec<usize>> {
        let w = syn.width;
        let h = syn.height;
        let mut visited = vec![false; w * h];
        let mut clusters = Vec::new();

        for idx in 0..(w * h) {
            if syn.cells[idx] <= 0.0 || visited[idx] {
                continue;
            }

            let mut stack = vec![idx];
            let mut cluster = Vec::new();
            visited[idx] = true;

            while let Some(cur) = stack.pop() {
                cluster.push(cur);
                let x = cur % w;
                let y = cur / w;

                let neighbors = [
                    (x as isize + 1, y as isize),
                    (x as isize - 1, y as isize),
                    (x as isize, y as isize + 1),
                    (x as isize, y as isize - 1),
                ];

                for (nx, ny) in neighbors {
                    if nx < 0 || ny < 0 || nx >= w as isize || ny >= h as isize {
                        continue;
                    }
                    let nidx = (ny as usize) * w + (nx as usize);
                    if syn.cells[nidx] > 0.0 && !visited[nidx] {
                        visited[nidx] = true;
                        stack.push(nidx);
                    }
                }
            }

            if !cluster.is_empty() {
                clusters.push(cluster);
            }
        }

        clusters
    }
}
