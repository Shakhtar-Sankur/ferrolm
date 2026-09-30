//! The paged key/value cache and its block manager.
//!
//! The cache is a pool of fixed-size blocks, `block_size` token slots each,
//! and every sequence owns a table of block ids, like pages of virtual
//! memory. A sequence grows one block at a time, so memory is committed as
//! tokens arrive instead of being reserved for the longest possible
//! sequence up front.
//!
//! Full blocks are also content-addressed: a block's key is a hash of its
//! tokens chained with the key of the block before it, so equal keys mean
//! equal prefixes. A new sequence that starts with a cached prefix (a
//! shared system prompt, say) reuses those blocks instead of recomputing
//! them. Blocks are reference counted; a cached block nobody uses stays
//! cached until its space is needed, and then the least recently used one
//! is evicted.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::hash::{DefaultHasher, Hash, Hasher};

/// Key and value storage for every layer, `f32`. Within a block, each kv
/// head's keys are contiguous (`[block][head][offset][dim]`), so attention
/// over one head reads memory sequentially.
pub struct KvCache {
    pub block_size: usize,
    pub blocks: usize,
    head_dim: usize,
    row: usize,
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
}

impl KvCache {
    pub fn new(layers: usize, kv_heads: usize, head_dim: usize, block_size: usize, blocks: usize) -> KvCache {
        let row = kv_heads * head_dim;
        let n = blocks * block_size * row;
        KvCache {
            block_size,
            blocks,
            head_dim,
            row,
            k: (0..layers).map(|_| vec![0.0; n]).collect(),
            v: (0..layers).map(|_| vec![0.0; n]).collect(),
        }
    }

    /// Stores the keys and values of all kv heads for one token, at slot
    /// `block * block_size + offset`.
    pub fn write(&mut self, layer: usize, slot: usize, k: &[f32], v: &[f32]) {
        let (bs, hd) = (self.block_size, self.head_dim);
        let (b, o) = (slot / bs, slot % bs);
        for h in 0..self.row / hd {
            let at = ((b * (self.row / hd) + h) * bs + o) * hd;
            self.k[layer][at..at + hd].copy_from_slice(&k[h * hd..(h + 1) * hd]);
            self.v[layer][at..at + hd].copy_from_slice(&v[h * hd..(h + 1) * hd]);
        }
    }

    /// All keys and values of one layer.
    pub fn layer(&self, layer: usize) -> (&[f32], &[f32]) {
        (&self.k[layer], &self.v[layer])
    }

    /// Floats per block, and the offset of a kv head within a block.
    pub fn block_stride(&self) -> usize {
        self.row * self.block_size
    }

    pub fn head_offset(&self, kv_head: usize) -> usize {
        kv_head * self.block_size * self.head_dim
    }
}

/// Allocation, reference counts and the prefix cache for a `KvCache`.
pub struct BlockManager {
    pub block_size: usize,
    refs: Vec<u32>,
    free: VecDeque<u32>,
    /// Content key and tokens of each cached block.
    key: Vec<Option<u64>>,
    tokens: Vec<Vec<u32>>,
    by_key: HashMap<u64, u32>,
    /// Cached blocks with no users, by the time they were last released.
    idle: BTreeMap<u64, u32>,
    idle_since: Vec<u64>,
    clock: u64,
    pub prefix_enabled: bool,
    pub stats: PrefixStats,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct PrefixStats {
    /// Full prompt blocks looked up, and those found cached.
    pub queried: u64,
    pub hit: u64,
    pub evicted: u64,
}

/// The key of the block before a sequence's first block.
pub const ROOT: u64 = 0;

pub fn chain(parent: u64, tokens: &[u32]) -> u64 {
    let mut h = DefaultHasher::new();
    parent.hash(&mut h);
    tokens.hash(&mut h);
    h.finish()
}

impl BlockManager {
    pub fn new(blocks: usize, block_size: usize, prefix_enabled: bool) -> BlockManager {
        BlockManager {
            block_size,
            refs: vec![0; blocks],
            free: (0..blocks as u32).collect(),
            key: vec![None; blocks],
            tokens: vec![Vec::new(); blocks],
            by_key: HashMap::new(),
            idle: BTreeMap::new(),
            idle_since: vec![0; blocks],
            clock: 0,
            prefix_enabled,
            stats: PrefixStats::default(),
        }
    }

    pub fn total(&self) -> usize {
        self.refs.len()
    }

    /// Blocks that can be allocated now (free or evictable).
    pub fn available(&self) -> usize {
        self.free.len() + self.idle.len()
    }

    /// Blocks in use by at least one sequence.
    pub fn used(&self) -> usize {
        self.total() - self.available()
    }

    pub fn allocate(&mut self) -> Option<u32> {
        let b = match self.free.pop_front() {
            Some(b) => b,
            None => {
                let (_, b) = self.idle.pop_first()?;
                self.forget(b);
                self.stats.evicted += 1;
                b
            }
        };
        self.refs[b as usize] = 1;
        Some(b)
    }

    fn forget(&mut self, b: u32) {
        if let Some(k) = self.key[b as usize].take() {
            self.by_key.remove(&k);
            self.tokens[b as usize].clear();
        }
    }

    pub fn release(&mut self, b: u32) {
        let i = b as usize;
        assert!(self.refs[i] > 0, "block {b} released more often than taken");
        self.refs[i] -= 1;
        if self.refs[i] == 0 {
            if self.key[i].is_some() {
                self.clock += 1;
                self.idle_since[i] = self.clock;
                self.idle.insert(self.clock, b);
            } else {
                self.free.push_back(b);
            }
        }
    }

    /// The cached block holding exactly `tokens` after the prefix `parent`,
    /// now shared with the caller.
    pub fn lookup(&mut self, parent: u64, tokens: &[u32]) -> Option<u32> {
        if !self.prefix_enabled {
            return None;
        }
        self.stats.queried += 1;
        let &b = self.by_key.get(&chain(parent, tokens))?;
        let i = b as usize;
        // Keys are 64-bit hashes; compare the tokens so that a collision
        // can never hand out the wrong cache contents.
        if self.tokens[i] != tokens {
            return None;
        }
        if self.refs[i] == 0 {
            self.idle.remove(&self.idle_since[i]);
        }
        self.refs[i] += 1;
        self.stats.hit += 1;
        Some(b)
    }

    /// Publishes a block the caller has just filled, so later sequences
    /// with the same prefix can share it. Returns the block's key.
    pub fn publish(&mut self, b: u32, parent: u64, tokens: &[u32]) -> u64 {
        let k = chain(parent, tokens);
        if self.prefix_enabled && self.key[b as usize].is_none() && !self.by_key.contains_key(&k) {
            self.by_key.insert(k, b);
            self.key[b as usize] = Some(k);
            self.tokens[b as usize] = tokens.to_vec();
        }
        k
    }

    pub fn refs(&self, b: u32) -> u32 {
        self.refs[b as usize]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocates_until_full_then_evicts_idle_cached_blocks_lru() {
        let mut m = BlockManager::new(3, 2, true);
        let a = m.allocate().unwrap();
        let b = m.allocate().unwrap();
        let c = m.allocate().unwrap();
        assert!(m.allocate().is_none());
        let ka = m.publish(a, ROOT, &[1, 2]);
        m.publish(b, ka, &[3, 4]);
        m.release(b);
        m.release(a);
        m.release(c);
        assert_eq!(m.available(), 3);
        // A sequence with the same first block shares it.
        assert_eq!(m.lookup(ROOT, &[1, 2]), Some(a));
        assert_eq!(m.lookup(ROOT, &[9, 9]), None);
        // The uncached free block goes first, then the least recently
        // released cached one (b); a is in use.
        assert_eq!(m.allocate(), Some(c));
        assert_eq!(m.allocate(), Some(b));
        assert_eq!(m.stats.evicted, 1);
        assert_eq!(m.lookup(ka, &[3, 4]), None, "evicted block must be forgotten");
        assert!(m.allocate().is_none());
        m.release(a);
        assert_eq!(m.lookup(ROOT, &[1, 2]), Some(a), "idle block stays cached");
        assert_eq!(m.refs(a), 1);
    }

    #[test]
    fn keys_depend_on_the_whole_prefix() {
        let k1 = chain(chain(ROOT, &[1, 2]), &[3, 4]);
        let k2 = chain(chain(ROOT, &[5, 6]), &[3, 4]);
        assert_ne!(k1, k2);
        let mut m = BlockManager::new(2, 2, false);
        let a = m.allocate().unwrap();
        m.publish(a, ROOT, &[1, 2]);
        m.release(a);
        assert_eq!(m.lookup(ROOT, &[1, 2]), None, "prefix caching disabled");
    }
}
