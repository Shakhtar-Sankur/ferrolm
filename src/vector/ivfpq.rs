//! IVF-PQ: an inverted file of product-quantized vectors (Jégou et al.,
//! 2011), the compressed index FAISS is known for.
//!
//! A coarse k-means splits the space into `nlist` cells; each vector is
//! stored in its cell as its residual from the cell's centroid, compressed
//! by a product quantizer: the residual is cut into `m` sub-vectors and
//! each is replaced by the index (one byte) of the nearest of 256
//! codewords. A query visits the `nprobe` nearest cells and scores each
//! stored vector with `m` table lookups. Optionally the best candidates
//! are re-ranked with exact distances (the raw vectors are then kept).

use super::io::{Reader, Writer};
use super::kmeans;
use super::{Metric, Neighbor, TopK, pad, padded};
use crate::kernels;
use crate::pool::{Out, Pool};

#[derive(Clone, Copy, Debug)]
pub struct IvfPqParams {
    pub nlist: usize,
    /// Sub-quantizers; the padded dimension must be a multiple of it.
    pub m: usize,
    pub iters: usize,
    /// Keep the raw vectors so results can be re-ranked exactly.
    pub keep_vectors: bool,
    pub seed: u64,
}

pub struct IvfPq {
    pub metric: Metric,
    pub dim: usize,
    pd: usize,
    nlist: usize,
    m: usize,
    dsub: usize,
    centroids: Vec<f32>,
    /// m × 256 × dsub.
    codebooks: Vec<f32>,
    ids: Vec<Vec<u32>>,
    codes: Vec<Vec<u8>>,
    vectors: Option<Vec<f32>>,
    count: usize,
}

const KSUB: usize = 256;

impl IvfPq {
    /// Trains the coarse quantizer and codebooks on `sample`.
    pub fn train(pool: &Pool, metric: Metric, dim: usize, sample: &[f32], p: IvfPqParams) -> IvfPq {
        let pd = padded(dim);
        assert_eq!(pd % p.m, 0, "padded dimension {pd} must be a multiple of m = {}", p.m);
        let x = pad(sample, dim);
        let n = x.len() / pd;
        let centroids = kmeans::train(pool, &x, pd, p.nlist, p.iters, p.seed);
        let assign = kmeans::assign(pool, &x, &centroids, pd);
        let dsub = pd / p.m;
        // Residuals, then one k-means per sub-space.
        let mut resid = x.clone();
        for (i, r) in resid.chunks_exact_mut(pd).enumerate() {
            let c = &centroids[assign[i] as usize * pd..][..pd];
            for (v, cv) in r.iter_mut().zip(c) {
                *v -= cv;
            }
        }
        let mut codebooks = vec![0f32; p.m * KSUB * dsub];
        for j in 0..p.m {
            let sub: Vec<f32> = resid
                .chunks_exact(pd)
                .flat_map(|r| r[j * dsub..(j + 1) * dsub].iter().copied())
                .collect();
            let cb = if dsub.is_multiple_of(8) {
                kmeans::train(pool, &sub, dsub, KSUB.min(n), p.iters, p.seed + 1 + j as u64)
            } else {
                // The SIMD distance needs multiples of 8; pad sub-vectors.
                let sp = pad(&sub, dsub);
                let c = kmeans::train(pool, &sp, padded(dsub), KSUB.min(n), p.iters, p.seed + 1 + j as u64);
                c.chunks_exact(padded(dsub))
                    .flat_map(|r| r[..dsub].iter().copied())
                    .collect()
            };
            codebooks[j * KSUB * dsub..][..cb.len()].copy_from_slice(&cb);
        }
        IvfPq {
            metric,
            dim,
            pd,
            nlist: p.nlist,
            m: p.m,
            dsub,
            centroids,
            codebooks,
            ids: vec![Vec::new(); p.nlist],
            codes: vec![Vec::new(); p.nlist],
            vectors: p.keep_vectors.then(Vec::new),
            count: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.count
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Bytes per stored vector in the compressed lists (codes and ids).
    pub fn bytes_per_vector(&self) -> usize {
        self.m + 4
    }

    fn sub_distance(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
    }

    fn encode(&self, resid: &[f32], out: &mut Vec<u8>) {
        for j in 0..self.m {
            let r = &resid[j * self.dsub..(j + 1) * self.dsub];
            let cb = &self.codebooks[j * KSUB * self.dsub..(j + 1) * KSUB * self.dsub];
            let mut best = (0u8, f32::INFINITY);
            for (c, w) in cb.chunks_exact(self.dsub).enumerate() {
                let d = Self::sub_distance(r, w);
                if d < best.1 {
                    best = (c as u8, d);
                }
            }
            out.push(best.0);
        }
    }

    /// Adds vectors; their ids continue from the current count.
    pub fn add(&mut self, pool: &Pool, vectors: &[f32]) {
        let x = pad(vectors, self.dim);
        let n = x.len() / self.pd;
        let assign = self.coarse_assign(pool, &x);
        // Encode in parallel, then file into the lists in order.
        let mut codes = vec![0u8; n * self.m];
        let o = Out::new(&mut codes);
        const CHUNK: usize = 512;
        pool.run(n.div_ceil(CHUNK), &|t| {
            let mut buf = Vec::with_capacity(self.m);
            let mut resid = vec![0f32; self.pd];
            for i in t * CHUNK..((t + 1) * CHUNK).min(n) {
                let c = &self.centroids[assign[i] as usize * self.pd..][..self.pd];
                for ((r, v), cv) in resid.iter_mut().zip(&x[i * self.pd..(i + 1) * self.pd]).zip(c) {
                    *r = v - cv;
                }
                buf.clear();
                self.encode(&resid, &mut buf);
                // SAFETY: each task writes its own rows.
                unsafe { o.slice(i * self.m, self.m).copy_from_slice(&buf) };
            }
        });
        for i in 0..n {
            let l = assign[i] as usize;
            self.ids[l].push((self.count + i) as u32);
            self.codes[l].extend_from_slice(&codes[i * self.m..(i + 1) * self.m]);
        }
        if let Some(v) = &mut self.vectors {
            v.extend_from_slice(&x);
        }
        self.count += n;
    }

    fn coarse_assign(&self, pool: &Pool, x: &[f32]) -> Vec<u32> {
        match self.metric {
            Metric::L2 => kmeans::assign(pool, x, &self.centroids, self.pd),
            Metric::InnerProduct => {
                let n = x.len() / self.pd;
                (0..n)
                    .map(|i| self.nearest_lists(&x[i * self.pd..(i + 1) * self.pd], 1)[0])
                    .collect()
            }
        }
    }

    /// The `nprobe` cells nearest to `q` (padded).
    fn nearest_lists(&self, q: &[f32], nprobe: usize) -> Vec<u32> {
        let mut top = TopK::new(nprobe);
        for (c, cen) in self.centroids.chunks_exact(self.pd).enumerate() {
            top.push(Neighbor {
                id: c as u32,
                distance: self.metric.distance(q, cen),
            });
        }
        top.into_sorted().into_iter().map(|n| n.id).collect()
    }

    /// The `k` nearest to `query`, visiting `nprobe` cells. With raw
    /// vectors kept, the best `k * refine` by compressed distance are
    /// re-ranked exactly (`refine` 0 skips it).
    pub fn search(&self, query: &[f32], k: usize, nprobe: usize, refine: usize) -> Vec<Neighbor> {
        let q = pad(query, self.dim);
        let lists = self.nearest_lists(&q, nprobe.min(self.nlist));
        let keep = if refine > 0 && self.vectors.is_some() {
            k * refine
        } else {
            k
        };
        let mut top = TopK::new(keep);
        let mut lut = vec![0f32; self.m * KSUB];
        // Inner product: the table does not depend on the cell.
        if self.metric == Metric::InnerProduct {
            self.fill_lut(&q, &mut lut);
        }
        let mut rq = vec![0f32; self.pd];
        for &l in &lists {
            let l = l as usize;
            let cen = &self.centroids[l * self.pd..(l + 1) * self.pd];
            let base = match self.metric {
                Metric::L2 => {
                    for ((r, a), b) in rq.iter_mut().zip(&q).zip(cen) {
                        *r = a - b;
                    }
                    self.fill_lut(&rq, &mut lut);
                    0.0
                }
                Metric::InnerProduct => -kernels::dot(&q, cen),
            };
            for (i, code) in self.codes[l].chunks_exact(self.m).enumerate() {
                let mut d = base;
                for (j, &c) in code.iter().enumerate() {
                    d += lut[j * KSUB + c as usize];
                }
                if d < top.bound() {
                    top.push(Neighbor {
                        id: self.ids[l][i],
                        distance: d,
                    });
                }
            }
        }
        let cand = top.into_sorted();
        match (&self.vectors, refine > 0) {
            (Some(v), true) => {
                let mut exact = TopK::new(k);
                for c in cand {
                    let d = self.metric.distance(&q, &v[c.id as usize * self.pd..][..self.pd]);
                    exact.push(Neighbor { id: c.id, distance: d });
                }
                exact.into_sorted()
            }
            _ => cand.into_iter().take(k).collect(),
        }
    }

    /// Distances from each sub-vector of `r` to every codeword.
    fn fill_lut(&self, r: &[f32], lut: &mut [f32]) {
        for j in 0..self.m {
            let rs = &r[j * self.dsub..(j + 1) * self.dsub];
            let cb = &self.codebooks[j * KSUB * self.dsub..(j + 1) * KSUB * self.dsub];
            for (c, w) in cb.chunks_exact(self.dsub).enumerate() {
                lut[j * KSUB + c] = match self.metric {
                    Metric::L2 => Self::sub_distance(rs, w),
                    Metric::InnerProduct => -rs.iter().zip(w).map(|(a, b)| a * b).sum::<f32>(),
                };
            }
        }
    }

    pub fn search_batch(
        &self,
        pool: &Pool,
        queries: &[f32],
        k: usize,
        nprobe: usize,
        refine: usize,
    ) -> Vec<Vec<Neighbor>> {
        let n = queries.len() / self.dim;
        let mut out: Vec<Vec<Neighbor>> = vec![Vec::new(); n];
        let o = Out::new(&mut out);
        pool.run(n, &|i| {
            let r = self.search(&queries[i * self.dim..(i + 1) * self.dim], k, nprobe, refine);
            // SAFETY: each task writes its own slot.
            unsafe { *o.ptr(i) = r };
        });
        out
    }

    pub fn save(&self) -> Vec<u8> {
        let mut w = Writer(b"FLIVFPQ1".to_vec());
        for v in [
            u32::from(self.metric.code()),
            self.dim as u32,
            self.nlist as u32,
            self.m as u32,
            self.count as u32,
        ] {
            w.u32(v);
        }
        w.f32s(&self.centroids);
        w.f32s(&self.codebooks);
        for l in 0..self.nlist {
            w.u32s(&self.ids[l]);
            w.bytes(&self.codes[l]);
        }
        w.f32s(self.vectors.as_deref().unwrap_or(&[]));
        w.u32(u32::from(self.vectors.is_some()));
        w.0
    }

    pub fn load(bytes: &[u8]) -> Result<IvfPq, String> {
        if !bytes.starts_with(b"FLIVFPQ1") {
            return Err("not an IVF-PQ index file".into());
        }
        let mut r = Reader::new(&bytes[8..]);
        let metric = Metric::from_code(r.u32()? as u8).ok_or("bad metric")?;
        let dim = r.u32()? as usize;
        let nlist = r.u32()? as usize;
        let m = r.u32()? as usize;
        let count = r.u32()? as usize;
        let centroids = r.f32s()?;
        let codebooks = r.f32s()?;
        let (mut ids, mut codes) = (Vec::with_capacity(nlist), Vec::with_capacity(nlist));
        for _ in 0..nlist {
            ids.push(r.u32s()?);
            codes.push(r.bytes()?);
        }
        let v = r.f32s()?;
        let has = r.u32()? == 1;
        let pd = padded(dim);
        Ok(IvfPq {
            metric,
            dim,
            pd,
            nlist,
            m,
            dsub: pd / m,
            centroids,
            codebooks,
            ids,
            codes,
            vectors: has.then_some(v),
            count,
        })
    }
}
