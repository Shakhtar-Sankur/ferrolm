//! Vector search: exact search, an HNSW graph index and an IVF-PQ
//! compressed index, the three designs behind FAISS, ScaNN and the vector
//! databases used for retrieval-augmented generation.
//!
//! Vectors are f32. Distances are squared Euclidean (`L2`) or negated inner
//! product (`InnerProduct`, cosine similarity when vectors are
//! normalised), so smaller is always nearer. Vectors are padded with zeros
//! to a multiple of 8 dimensions, which changes no distance.

pub mod flat;
pub mod hnsw;
pub mod io;
pub mod ivfpq;
pub mod kmeans;

use crate::kernels;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Metric {
    L2,
    InnerProduct,
}

impl Metric {
    #[inline]
    pub fn distance(self, a: &[f32], b: &[f32]) -> f32 {
        match self {
            Metric::L2 => kernels::l2sq(a, b),
            Metric::InnerProduct => -kernels::dot(a, b),
        }
    }

    pub fn code(self) -> u8 {
        match self {
            Metric::L2 => 0,
            Metric::InnerProduct => 1,
        }
    }

    pub fn from_code(c: u8) -> Option<Metric> {
        match c {
            0 => Some(Metric::L2),
            1 => Some(Metric::InnerProduct),
            _ => None,
        }
    }
}

/// A search result: an id and its distance to the query.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Neighbor {
    pub id: u32,
    pub distance: f32,
}

impl Eq for Neighbor {}

impl Ord for Neighbor {
    fn cmp(&self, o: &Self) -> Ordering {
        self.distance.total_cmp(&o.distance).then(self.id.cmp(&o.id))
    }
}

impl PartialOrd for Neighbor {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}

/// Keeps the `k` nearest of the neighbours pushed into it.
pub struct TopK {
    k: usize,
    heap: BinaryHeap<Neighbor>,
}

impl TopK {
    pub fn new(k: usize) -> TopK {
        TopK {
            k,
            heap: BinaryHeap::with_capacity(k + 1),
        }
    }

    /// The distance a candidate must beat to be kept.
    #[inline]
    pub fn bound(&self) -> f32 {
        if self.heap.len() < self.k {
            f32::INFINITY
        } else {
            self.heap.peek().unwrap().distance
        }
    }

    #[inline]
    pub fn push(&mut self, n: Neighbor) {
        if self.heap.len() < self.k {
            self.heap.push(n);
        } else if n < *self.heap.peek().unwrap() {
            self.heap.pop();
            self.heap.push(n);
        }
    }

    /// Nearest first.
    pub fn into_sorted(self) -> Vec<Neighbor> {
        self.heap.into_sorted_vec()
    }
}

/// Dimensions padded to a multiple of 8.
pub fn padded(dim: usize) -> usize {
    dim.div_ceil(8) * 8
}

/// Copies `n` vectors of `dim` into rows of `padded(dim)`.
pub fn pad(data: &[f32], dim: usize) -> Vec<f32> {
    let pd = padded(dim);
    if pd == dim {
        return data.to_vec();
    }
    let n = data.len() / dim;
    let mut out = vec![0f32; n * pd];
    for (src, dst) in data.chunks_exact(dim).zip(out.chunks_exact_mut(pd)) {
        dst[..dim].copy_from_slice(src);
    }
    out
}

/// Scales each vector to unit length (for cosine similarity).
pub fn normalize(data: &mut [f32], dim: usize) {
    for v in data.chunks_exact_mut(dim) {
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if n > 0.0 {
            for x in v.iter_mut() {
                *x /= n;
            }
        }
    }
}

/// Fraction of the true `k` nearest (the first `k` of each `truth` row)
/// that appear among the returned `k`.
pub fn recall(found: &[Vec<Neighbor>], truth: &[Vec<u32>], k: usize) -> f64 {
    let mut hit = 0usize;
    for (f, t) in found.iter().zip(truth) {
        let want: std::collections::HashSet<u32> = t.iter().take(k).copied().collect();
        hit += f.iter().take(k).filter(|n| want.contains(&n.id)).count();
    }
    hit as f64 / (found.len() * k) as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn topk_keeps_the_nearest_in_order() {
        let mut t = TopK::new(3);
        for (i, d) in [5.0f32, 1.0, 4.0, 3.0, 2.0, 9.0].iter().enumerate() {
            t.push(Neighbor {
                id: i as u32,
                distance: *d,
            });
        }
        let ids: Vec<u32> = t.into_sorted().iter().map(|n| n.id).collect();
        assert_eq!(ids, [1, 4, 3]);
        assert_eq!(Metric::L2.distance(&[0.0; 8], &[1.0; 8]), 8.0);
        assert_eq!(Metric::InnerProduct.distance(&[1.0; 8], &[2.0; 8]), -16.0);
    }
}
