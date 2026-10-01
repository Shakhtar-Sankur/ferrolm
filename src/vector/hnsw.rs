//! HNSW, the hierarchical navigable small-world graph (Malkov and
//! Yashunin, 2018): the index behind most vector databases.
//!
//! Every vector is a node on layer 0; a geometrically shrinking random
//! subset also sits on layers 1, 2, ... A search walks greedily down from
//! the top layer's entry point, then runs a best-first search of width
//! `ef` on layer 0. Neighbours are chosen with the diversity heuristic: a
//! candidate is linked only if it is nearer the new node than any
//! neighbour already chosen, which keeps long-range links.
//!
//! Construction is parallel: each node's neighbour lists are guarded by
//! its own lock, as in hnswlib.

use super::io::{Reader, Writer};
use super::{Metric, Neighbor, pad, padded};
use crate::pool::Pool;
use crate::rng::Rng;
use std::cell::RefCell;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};

#[derive(Clone, Copy, Debug)]
pub struct HnswParams {
    /// Neighbours per node on upper layers (twice this on layer 0).
    pub m: usize,
    pub ef_construction: usize,
    pub seed: u64,
}

impl Default for HnswParams {
    fn default() -> Self {
        HnswParams {
            m: 16,
            ef_construction: 200,
            seed: 1,
        }
    }
}

pub struct Hnsw {
    pub metric: Metric,
    pub dim: usize,
    pd: usize,
    m: usize,
    m0: usize,
    data: Vec<f32>,
    levels: Vec<u8>,
    /// Layer 0: per node, a count then up to `m0` neighbour ids.
    links0: Links,
    /// Layers 1.. for the nodes that have them.
    upper: Vec<Mutex<Vec<Vec<u32>>>>,
    locks: Vec<Mutex<()>>,
    entry: Mutex<(u32, usize)>,
}

/// Layer-0 adjacency. Entries are written only under the node's lock;
/// relaxed atomics make the concurrent reads during construction sound
/// and cost nothing on x86.
struct Links(Vec<AtomicU32>);

impl Links {
    fn new(v: Vec<u32>) -> Links {
        Links(v.into_iter().map(AtomicU32::new).collect())
    }

    #[inline]
    fn get(&self, i: usize) -> u32 {
        self.0[i].load(Ordering::Relaxed)
    }

    #[inline]
    fn set(&self, i: usize, v: u32) {
        self.0[i].store(v, Ordering::Relaxed)
    }
}

thread_local! {
    static VISITED: RefCell<(Vec<u32>, u32)> = const { RefCell::new((Vec::new(), 0)) };
}

impl Hnsw {
    /// Builds an index over `vectors` (rows of `dim`).
    pub fn build(pool: &Pool, metric: Metric, dim: usize, vectors: &[f32], p: HnswParams) -> Hnsw {
        let pd = padded(dim);
        let data = pad(vectors, dim);
        let n = data.len() / pd;
        let (m, m0) = (p.m, 2 * p.m);
        let ml = 1.0 / (m as f64).ln();
        let mut rng = Rng::new(p.seed);
        let levels: Vec<u8> = (0..n)
            .map(|_| ((-(1.0 - rng.uniform()).ln() * ml) as usize).min(15) as u8)
            .collect();
        let upper = levels
            .iter()
            .map(|&l| Mutex::new(vec![Vec::new(); l as usize]))
            .collect();
        let h = Hnsw {
            metric,
            dim,
            pd,
            m,
            m0,
            data,
            links0: Links::new(vec![0; n * (m0 + 1)]),
            upper,
            locks: (0..n).map(|_| Mutex::new(())).collect(),
            entry: Mutex::new((0, levels.first().copied().unwrap_or(0) as usize)),
            levels,
        };
        if n <= 1 {
            return h;
        }
        // Node 0 is the first entry point; the rest go in parallel, handed
        // out in order so early nodes form the graph's backbone first.
        let next = AtomicUsize::new(1);
        let ef = p.ef_construction;
        pool.run(pool.threads(), &|_| {
            loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                if i >= n {
                    break;
                }
                h.insert(i as u32, ef);
            }
        });
        h
    }

    pub fn len(&self) -> usize {
        self.levels.len()
    }

    pub fn is_empty(&self) -> bool {
        self.levels.is_empty()
    }

    #[inline]
    fn vec(&self, id: u32) -> &[f32] {
        &self.data[id as usize * self.pd..][..self.pd]
    }

    #[inline]
    fn dist(&self, q: &[f32], id: u32) -> f32 {
        self.metric.distance(q, self.vec(id))
    }

    /// Neighbours of `id` on `layer`, copied (under the node's lock if
    /// `locked`).
    fn neighbors(&self, id: u32, layer: usize, locked: bool, out: &mut Vec<u32>) {
        out.clear();
        let _g = locked.then(|| self.locks[id as usize].lock().unwrap());
        if layer == 0 {
            let base = id as usize * (self.m0 + 1);
            let c = self.links0.get(base) as usize;
            out.extend((base + 1..base + 1 + c).map(|i| self.links0.get(i)));
        } else {
            let u = self.upper[id as usize].lock().unwrap();
            out.extend_from_slice(&u[layer - 1]);
        }
    }

    fn set_neighbors(&self, id: u32, layer: usize, list: &[u32]) {
        if layer == 0 {
            // The caller holds the node's lock.
            let base = id as usize * (self.m0 + 1);
            for (i, &x) in list.iter().enumerate() {
                self.links0.set(base + 1 + i, x);
            }
            self.links0.set(base, list.len() as u32);
        } else {
            self.upper[id as usize].lock().unwrap()[layer - 1] = list.to_vec();
        }
    }

    /// Best-first search of one layer from `entry`, keeping the `ef`
    /// nearest; nearest first.
    fn search_layer(&self, q: &[f32], entry: &[Neighbor], ef: usize, layer: usize, locked: bool) -> Vec<Neighbor> {
        VISITED.with(|v| {
            let mut v = v.borrow_mut();
            if v.0.len() < self.len() {
                v.0 = vec![0; self.len()];
                v.1 = 0;
            }
            v.1 = v.1.wrapping_add(1);
            if v.1 == 0 {
                v.0.fill(0);
                v.1 = 1;
            }
            let stamp = v.1;
            let mut cand: BinaryHeap<Reverse<Neighbor>> = BinaryHeap::new();
            let mut best: BinaryHeap<Neighbor> = BinaryHeap::new();
            for &e in entry {
                v.0[e.id as usize] = stamp;
                cand.push(Reverse(e));
                best.push(e);
            }
            while best.len() > ef {
                best.pop();
            }
            let mut nb = Vec::with_capacity(self.m0);
            while let Some(Reverse(c)) = cand.pop() {
                if best.len() >= ef && c.distance > best.peek().unwrap().distance {
                    break;
                }
                self.neighbors(c.id, layer, locked, &mut nb);
                for &x in &nb {
                    if v.0[x as usize] == stamp {
                        continue;
                    }
                    v.0[x as usize] = stamp;
                    let d = self.dist(q, x);
                    if best.len() < ef || d < best.peek().unwrap().distance {
                        let n = Neighbor { id: x, distance: d };
                        cand.push(Reverse(n));
                        best.push(n);
                        if best.len() > ef {
                            best.pop();
                        }
                    }
                }
            }
            best.into_sorted_vec()
        })
    }

    /// The diversity heuristic: from candidates sorted nearest first, keep
    /// those nearer the base than any already kept, up to `m`.
    fn select(&self, candidates: &[Neighbor], m: usize) -> Vec<u32> {
        let mut out: Vec<u32> = Vec::with_capacity(m);
        for c in candidates {
            if out.len() >= m {
                break;
            }
            let cv = self.vec(c.id);
            if out.iter().all(|&s| self.metric.distance(cv, self.vec(s)) > c.distance) {
                out.push(c.id);
            }
        }
        out
    }

    fn insert(&self, id: u32, ef: usize) {
        let q = self.vec(id).to_vec();
        let level = self.levels[id as usize] as usize;
        let (ep, top) = *self.entry.lock().unwrap();
        let mut cur = Neighbor {
            id: ep,
            distance: self.dist(&q, ep),
        };
        let mut nb = Vec::new();
        // Greedy descent through the layers above this node's own.
        for layer in (level + 1..=top).rev() {
            let mut changed = true;
            while changed {
                changed = false;
                self.neighbors(cur.id, layer, true, &mut nb);
                for &x in &nb {
                    let d = self.dist(&q, x);
                    if d < cur.distance {
                        cur = Neighbor { id: x, distance: d };
                        changed = true;
                    }
                }
            }
        }
        let mut entry = vec![cur];
        for layer in (0..=level.min(top)).rev() {
            let found = self.search_layer(&q, &entry, ef, layer, true);
            // As in FAISS: up to 2M links on layer 0, M above.
            let cap = if layer == 0 { self.m0 } else { self.m };
            let chosen = self.select(&found, cap);
            // Merge, never overwrite: in a parallel build another node may
            // already have linked itself here (having reached this node on
            // a layer above), and dropping that link could leave it with no
            // way in.
            self.add_links(id, layer, &chosen, cap);
            for &n in &chosen {
                self.add_links(n, layer, &[id], cap);
            }
            entry = found;
        }
        if level > top {
            let mut e = self.entry.lock().unwrap();
            if level > e.1 {
                *e = (id, level);
            }
        }
    }

    /// Adds `new` to `id`'s neighbours on `layer` under its lock, choosing
    /// again with the diversity heuristic if that makes more than `cap`.
    fn add_links(&self, id: u32, layer: usize, new: &[u32], cap: usize) {
        let _g = self.locks[id as usize].lock().unwrap();
        let mut list = Vec::new();
        self.neighbors(id, layer, false, &mut list);
        let before = list.len();
        for &x in new {
            if x != id && !list.contains(&x) {
                list.push(x);
            }
        }
        if list.len() == before {
            return;
        }
        if list.len() > cap {
            let base = self.vec(id);
            let mut c: Vec<Neighbor> = list
                .iter()
                .map(|&x| Neighbor {
                    id: x,
                    distance: self.metric.distance(base, self.vec(x)),
                })
                .collect();
            c.sort();
            list = self.select(&c, cap);
        }
        self.set_neighbors(id, layer, &list);
    }

    /// The `k` nearest to `query`, searching layer 0 with width `ef`.
    pub fn search(&self, query: &[f32], k: usize, ef: usize) -> Vec<Neighbor> {
        if self.is_empty() {
            return Vec::new();
        }
        let q = pad(query, self.dim);
        let (ep, top) = *self.entry.lock().unwrap();
        let mut cur = Neighbor {
            id: ep,
            distance: self.dist(&q, ep),
        };
        let mut nb = Vec::new();
        for layer in (1..=top).rev() {
            let mut changed = true;
            while changed {
                changed = false;
                self.neighbors(cur.id, layer, false, &mut nb);
                for &x in &nb {
                    let d = self.dist(&q, x);
                    if d < cur.distance {
                        cur = Neighbor { id: x, distance: d };
                        changed = true;
                    }
                }
            }
        }
        let mut r = self.search_layer(&q, &[cur], ef.max(k), 0, false);
        r.truncate(k);
        r
    }

    pub fn search_batch(&self, pool: &Pool, queries: &[f32], k: usize, ef: usize) -> Vec<Vec<Neighbor>> {
        let n = queries.len() / self.dim;
        let mut out: Vec<Vec<Neighbor>> = vec![Vec::new(); n];
        let o = crate::pool::Out::new(&mut out);
        pool.run(n, &|i| {
            let r = self.search(&queries[i * self.dim..(i + 1) * self.dim], k, ef);
            // SAFETY: each task writes its own slot.
            unsafe { *o.ptr(i) = r };
        });
        out
    }

    /// Nodes reachable on layer 0 from the entry point (a sanity check of
    /// construction: all of them, if the graph is connected).
    pub fn reachable(&self) -> usize {
        let (ep, _) = *self.entry.lock().unwrap();
        let mut seen = vec![false; self.len()];
        let mut stack = vec![ep];
        seen[ep as usize] = true;
        let mut nb = Vec::new();
        let mut count = 0;
        while let Some(x) = stack.pop() {
            count += 1;
            self.neighbors(x, 0, false, &mut nb);
            for &y in &nb {
                if !seen[y as usize] {
                    seen[y as usize] = true;
                    stack.push(y);
                }
            }
        }
        count
    }

    /// Mean layer-0 degree (a sanity check of construction).
    pub fn mean_degree(&self) -> f64 {
        let total: u64 = (0..self.len())
            .map(|i| u64::from(self.links0.get(i * (self.m0 + 1))))
            .sum();
        total as f64 / self.len().max(1) as f64
    }

    pub fn save(&self) -> Vec<u8> {
        let mut w = Writer(b"FLHNSW01".to_vec());
        w.u32(u32::from(self.metric.code()));
        w.u32(self.dim as u32);
        w.u32(self.m as u32);
        let (ep, top) = *self.entry.lock().unwrap();
        w.u32(ep);
        w.u32(top as u32);
        w.bytes(&self.levels);
        w.u32s(
            &self
                .links0
                .0
                .iter()
                .map(|a| a.load(Ordering::Relaxed))
                .collect::<Vec<u32>>(),
        );
        let mut flat = Vec::new();
        for u in &self.upper {
            for list in u.lock().unwrap().iter() {
                flat.push(list.len() as u32);
                flat.extend_from_slice(list);
            }
        }
        w.u32s(&flat);
        w.f32s(&self.data);
        w.0
    }

    pub fn load(bytes: &[u8]) -> Result<Hnsw, String> {
        if !bytes.starts_with(b"FLHNSW01") {
            return Err("not an HNSW index file".into());
        }
        let mut r = Reader::new(&bytes[8..]);
        let metric = Metric::from_code(r.u32()? as u8).ok_or("bad metric")?;
        let dim = r.u32()? as usize;
        let m = r.u32()? as usize;
        let ep = r.u32()?;
        let top = r.u32()? as usize;
        let levels = r.bytes()?;
        let links = r.u32s()?;
        let flat = r.u32s()?;
        let data = r.f32s()?;
        let mut upper = Vec::with_capacity(levels.len());
        let mut i = 0;
        for &l in &levels {
            let mut lists = Vec::with_capacity(l as usize);
            for _ in 0..l {
                let c = *flat.get(i).ok_or("truncated links")? as usize;
                lists.push(flat.get(i + 1..i + 1 + c).ok_or("truncated links")?.to_vec());
                i += 1 + c;
            }
            upper.push(Mutex::new(lists));
        }
        let n = levels.len();
        if links.len() != n * (2 * m + 1) || data.len() != n * padded(dim) {
            return Err("index file sizes do not match".into());
        }
        Ok(Hnsw {
            metric,
            dim,
            pd: padded(dim),
            m,
            m0: 2 * m,
            data,
            levels,
            links0: Links::new(links),
            upper,
            locks: (0..n).map(|_| Mutex::new(())).collect(),
            entry: Mutex::new((ep, top)),
        })
    }
}
