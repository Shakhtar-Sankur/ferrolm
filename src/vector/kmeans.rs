//! k-means (Lloyd's algorithm), for the coarse quantizer and the product
//! quantizer's codebooks.

use crate::kernels;
use crate::pool::{Out, Pool};
use crate::rng::Rng;

/// The nearest of `centroids` (rows of `dim`) to `x`, by squared L2.
#[inline]
pub fn nearest(x: &[f32], centroids: &[f32], dim: usize) -> (usize, f32) {
    let mut best = (0, f32::INFINITY);
    for (c, cen) in centroids.chunks_exact(dim).enumerate() {
        let d = kernels::l2sq(x, cen);
        if d < best.1 {
            best = (c, d);
        }
    }
    best
}

/// Assigns every row of `data` to its nearest centroid, in parallel.
pub fn assign(pool: &Pool, data: &[f32], centroids: &[f32], dim: usize) -> Vec<u32> {
    let n = data.len() / dim;
    let mut out = vec![0u32; n];
    let o = Out::new(&mut out);
    const CHUNK: usize = 256;
    pool.run(n.div_ceil(CHUNK), &|t| {
        for i in t * CHUNK..((t + 1) * CHUNK).min(n) {
            let (c, _) = nearest(&data[i * dim..(i + 1) * dim], centroids, dim);
            // SAFETY: each task writes its own rows.
            unsafe { o.write(i, c as u32) };
        }
    });
    out
}

/// `k` centroids for `data` (rows of `dim`, a multiple of 8 when used with
/// the SIMD distance), from a random start, `iters` rounds.
pub fn train(pool: &Pool, data: &[f32], dim: usize, k: usize, iters: usize, seed: u64) -> Vec<f32> {
    let n = data.len() / dim;
    assert!(n >= k, "need at least {k} training vectors, got {n}");
    let mut rng = Rng::new(seed);
    // Distinct random rows as the starting centroids.
    let mut idx: Vec<usize> = (0..n).collect();
    for i in 0..k {
        let j = i + rng.below((n - i) as u64) as usize;
        idx.swap(i, j);
    }
    let mut cen: Vec<f32> = idx[..k]
        .iter()
        .flat_map(|&i| data[i * dim..(i + 1) * dim].iter().copied())
        .collect();
    for _ in 0..iters {
        let a = assign(pool, data, &cen, dim);
        let mut sums = vec![0f64; k * dim];
        let mut counts = vec![0usize; k];
        for (i, &c) in a.iter().enumerate() {
            counts[c as usize] += 1;
            for (s, &v) in sums[c as usize * dim..][..dim]
                .iter_mut()
                .zip(&data[i * dim..(i + 1) * dim])
            {
                *s += f64::from(v);
            }
        }
        for c in 0..k {
            if counts[c] > 0 {
                for d in 0..dim {
                    cen[c * dim + d] = (sums[c * dim + d] / counts[c] as f64) as f32;
                }
            }
        }
        // An empty cluster takes half of the largest one: a copy of its
        // centroid, nudged apart.
        for c in 0..k {
            if counts[c] == 0 {
                let big = (0..k).max_by_key(|&j| counts[j]).unwrap();
                for d in 0..dim {
                    let v = cen[big * dim + d];
                    let e = 1e-4 * (1.0 + v.abs());
                    cen[c * dim + d] = v + e;
                    cen[big * dim + d] = v - e;
                }
                counts[c] = counts[big] / 2;
                counts[big] -= counts[c];
            }
        }
    }
    cen
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_well_separated_clusters() {
        let pool = Pool::new(2);
        let mut rng = Rng::new(3);
        let centers = [[0f32; 8], [10.0; 8], [-10.0; 8]];
        let mut data = Vec::new();
        for i in 0..600 {
            for &c in &centers[i % 3] {
                data.push(c + rng.normal() as f32 * 0.1);
            }
        }
        let cen = train(&pool, &data, 8, 3, 10, 1);
        for c in &centers {
            let (_, d) = nearest(c, &cen, 8);
            assert!(d < 0.1, "no centroid near {c:?}");
        }
    }
}
