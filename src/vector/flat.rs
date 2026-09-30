//! Exact search: every vector compared with the query. The ground truth
//! the approximate indexes are measured against.

use super::{Metric, Neighbor, TopK, pad, padded};
use crate::pool::{Out, Pool};

pub struct FlatIndex {
    pub metric: Metric,
    pub dim: usize,
    pd: usize,
    data: Vec<f32>,
}

impl FlatIndex {
    pub fn new(metric: Metric, dim: usize) -> FlatIndex {
        FlatIndex {
            metric,
            dim,
            pd: padded(dim),
            data: Vec::new(),
        }
    }

    pub fn add(&mut self, vectors: &[f32]) {
        self.data.extend(pad(vectors, self.dim));
    }

    pub fn len(&self) -> usize {
        self.data.len() / self.pd
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn vector(&self, id: u32) -> &[f32] {
        &self.data[id as usize * self.pd..][..self.pd]
    }

    pub fn search(&self, query: &[f32], k: usize) -> Vec<Neighbor> {
        let q = pad(query, self.dim);
        let mut top = TopK::new(k);
        for (i, v) in self.data.chunks_exact(self.pd).enumerate() {
            let d = self.metric.distance(&q, v);
            if d < top.bound() {
                top.push(Neighbor {
                    id: i as u32,
                    distance: d,
                });
            }
        }
        top.into_sorted()
    }

    /// Many queries at once, in parallel.
    pub fn search_batch(&self, pool: &Pool, queries: &[f32], k: usize) -> Vec<Vec<Neighbor>> {
        let n = queries.len() / self.dim;
        let mut out: Vec<Vec<Neighbor>> = vec![Vec::new(); n];
        let o = Out::new(&mut out);
        pool.run(n, &|i| {
            let r = self.search(&queries[i * self.dim..(i + 1) * self.dim], k);
            // SAFETY: each task writes its own slot.
            unsafe { *o.ptr(i) = r };
        });
        out
    }
}
