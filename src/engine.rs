//! The serving engine: a scheduler that runs many requests through the
//! model at once.
//!
//! Every step it builds one batch: first the running sequences (one token
//! each while decoding, or the next chunk of a long prompt), then new
//! requests from the queue, until the step's token budget or the sequence
//! limit is reached. Requests join and leave the batch at every step
//! (continuous batching), so a short reply never waits for a long one.
//!
//! Cache memory is allocated a block at a time as sequences grow. A new
//! request first looks for its prompt's leading blocks in the prefix
//! cache. When a growing sequence finds no free block, the most recently
//! admitted sequence is preempted: its blocks are released (full ones stay
//! cached) and it goes back to the front of the queue, to be recomputed
//! later from its prompt and the tokens it had already generated.
//!
//! With a draft model, decoding is speculative: the draft proposes `k`
//! tokens per sequence, the target model scores them all in one pass, and
//! the longest prefix the target agrees with is kept, plus one token of the
//! target's own. Under greedy decoding the output is exactly what the
//! target alone would produce; under sampling, the acceptance rule keeps
//! the target's distribution.

use crate::kv::{BlockManager, KvCache, ROOT};
use crate::model::{Chunk, Logits, Model};
use crate::pool::Pool;
use crate::rng::Rng;
use crate::sampler::{self, SamplingParams};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::sync::{Arc, Mutex};
use std::time::Instant;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Admission {
    /// Admit requests at every step.
    Continuous,
    /// Admit a new batch only when the previous one has finished
    /// (request-level batching, the baseline).
    Static,
}

#[derive(Clone, Debug)]
pub struct EngineConfig {
    pub block_size: usize,
    pub kv_blocks: usize,
    /// Tokens processed per step, across all sequences.
    pub max_batch_tokens: usize,
    pub max_seqs: usize,
    pub prefix_cache: bool,
    pub admission: Admission,
    /// Reserve this many token slots per sequence when it is admitted, as
    /// servers without paging do; `None` allocates blocks as tokens arrive.
    pub reserve_tokens: Option<usize>,
    /// Tokens the draft model proposes per step.
    pub spec_k: usize,
}

impl Default for EngineConfig {
    fn default() -> Self {
        EngineConfig {
            block_size: 16,
            kv_blocks: 1024,
            max_batch_tokens: 512,
            max_seqs: 64,
            prefix_cache: true,
            admission: Admission::Continuous,
            reserve_tokens: None,
            spec_k: 4,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Finish {
    /// `max_tokens` reached, or the context is full.
    Length,
    /// An end-of-sequence or stop token.
    Stop,
    Cancelled,
    Rejected(String),
}

#[derive(Clone, Debug, PartialEq)]
pub enum Event {
    Token(u32),
    Done(Finish),
}

pub struct Request {
    pub prompt: Vec<u32>,
    pub params: SamplingParams,
    pub events: Sender<Event>,
    pub cancel: Arc<AtomicBool>,
}

#[derive(Clone, Debug, Default)]
pub struct Stats {
    pub steps: u64,
    pub requests: u64,
    pub finished: u64,
    pub prompt_tokens: u64,
    pub generated_tokens: u64,
    /// Prompt tokens served from the prefix cache instead of computed.
    pub cached_tokens: u64,
    /// Tokens run through the target model.
    pub computed_tokens: u64,
    pub preemptions: u64,
    pub spec_proposed: u64,
    pub spec_accepted: u64,
    pub running: usize,
    pub waiting: usize,
    pub kv_used_blocks: usize,
    pub kv_total_blocks: usize,
    pub peak_running: usize,
    pub busy_secs: f64,
}

/// Submits requests to a running engine.
#[derive(Clone)]
pub struct Handle {
    tx: Sender<Request>,
    pub stats: Arc<Mutex<Stats>>,
}

impl Handle {
    pub fn submit(&self, prompt: Vec<u32>, params: SamplingParams) -> (Receiver<Event>, Arc<AtomicBool>) {
        let (tx, rx) = mpsc::channel();
        let cancel = Arc::new(AtomicBool::new(false));
        let req = Request { prompt, params, events: tx, cancel: Arc::clone(&cancel) };
        if let Err(e) = self.tx.send(req) {
            e.0.events.send(Event::Done(Finish::Rejected("engine stopped".into()))).ok();
        }
        (rx, cancel)
    }
}

struct Draft {
    model: Model,
    cache: KvCache,
    blocks: BlockManager,
}

struct Seq {
    req: Request,
    tokens: Vec<u32>,
    /// Tokens whose keys and values are in the cache.
    computed: usize,
    blocks: Vec<u32>,
    /// Prefix-cache keys of the full blocks, in order.
    keys: Vec<u64>,
    generated: usize,
    rng: Rng,
    draft_computed: usize,
    draft_blocks: Vec<u32>,
}

/// What one sequence does in a step.
struct Plan {
    seq: usize,
    /// Tokens run through the target model.
    n: usize,
    /// Draft proposals being verified.
    drafts: Vec<u32>,
    /// The draft's distributions for its proposals (when sampling).
    draft_dists: Vec<Vec<f32>>,
}

pub struct Engine {
    model: Model,
    draft: Option<Draft>,
    pool: Pool,
    cache: KvCache,
    blocks: BlockManager,
    cfg: EngineConfig,
    waiting: VecDeque<Seq>,
    running: Vec<Seq>,
    rx: Receiver<Request>,
    stats: Arc<Mutex<Stats>>,
    local: Stats,
    max_len: usize,
}

impl Engine {
    pub fn new(model: Model, draft: Option<Model>, pool: Pool, cfg: EngineConfig) -> (Engine, Handle) {
        let c = &model.cfg;
        let cache = KvCache::new(c.layers, c.kv_heads, c.head_dim, cfg.block_size, cfg.kv_blocks);
        let blocks = BlockManager::new(cfg.kv_blocks, cfg.block_size, cfg.prefix_cache);
        let draft = draft.map(|m| {
            assert_eq!(m.cfg.vocab, c.vocab, "draft and target must share a vocabulary");
            let d = &m.cfg;
            // As many token slots as the target, plus room for proposals.
            let n = cfg.kv_blocks + cfg.max_seqs;
            Draft {
                cache: KvCache::new(d.layers, d.kv_heads, d.head_dim, cfg.block_size, n),
                blocks: BlockManager::new(n, cfg.block_size, false),
                model: m,
            }
        });
        let max_len = c.max_position.min(cfg.kv_blocks * cfg.block_size);
        let (tx, rx) = mpsc::channel();
        let stats = Arc::new(Mutex::new(Stats { kv_total_blocks: cfg.kv_blocks, ..Stats::default() }));
        let e = Engine {
            model,
            draft,
            pool,
            cache,
            blocks,
            cfg,
            waiting: VecDeque::new(),
            running: Vec::new(),
            rx,
            stats: Arc::clone(&stats),
            local: Stats::default(),
            max_len,
        };
        (e, Handle { tx, stats })
    }

    pub fn model(&self) -> &Model {
        &self.model
    }

    /// Serves requests until every `Handle` is dropped.
    pub fn run(&mut self) {
        loop {
            if self.running.is_empty() && self.waiting.is_empty() {
                match self.rx.recv() {
                    Ok(r) => self.enqueue(r),
                    Err(_) => return,
                }
            }
            if !self.drain() && self.running.is_empty() && self.waiting.is_empty() {
                return;
            }
            self.step();
        }
    }

    /// Serves until no request is queued or running.
    pub fn run_until_idle(&mut self) {
        loop {
            self.drain();
            if self.running.is_empty() && self.waiting.is_empty() {
                return;
            }
            self.step();
        }
    }

    /// Queues new requests; false once all handles are gone.
    fn drain(&mut self) -> bool {
        loop {
            match self.rx.try_recv() {
                Ok(r) => self.enqueue(r),
                Err(TryRecvError::Empty) => return true,
                Err(TryRecvError::Disconnected) => return false,
            }
        }
    }

    fn enqueue(&mut self, req: Request) {
        self.local.requests += 1;
        let n = req.prompt.len();
        if n == 0 || n >= self.max_len {
            let why = format!("prompt of {n} tokens; this server fits 1 to {}", self.max_len - 1);
            req.events.send(Event::Done(Finish::Rejected(why))).ok();
            self.local.finished += 1;
            return;
        }
        if let Some(bad) = req.prompt.iter().find(|&&t| t as usize >= self.model.cfg.vocab) {
            req.events.send(Event::Done(Finish::Rejected(format!("token {bad} is outside the vocabulary")))).ok();
            self.local.finished += 1;
            return;
        }
        self.local.prompt_tokens += n as u64;
        let seed = req.params.seed;
        self.waiting.push_back(Seq {
            tokens: req.prompt.clone(),
            computed: 0,
            blocks: Vec::new(),
            keys: Vec::new(),
            generated: 0,
            rng: Rng::new(seed),
            draft_computed: 0,
            draft_blocks: Vec::new(),
            req,
        });
    }

    /// Releases a sequence's cache blocks.
    fn release(&mut self, s: &mut Seq) {
        for b in s.blocks.drain(..) {
            self.blocks.release(b);
        }
        s.keys.clear();
        s.computed = 0;
        if let Some(d) = &mut self.draft {
            for b in s.draft_blocks.drain(..) {
                d.blocks.release(b);
            }
        }
        s.draft_computed = 0;
    }

    fn finish(&mut self, mut s: Seq, why: Finish) {
        self.release(&mut s);
        s.req.events.send(Event::Done(why)).ok();
        self.local.finished += 1;
    }

    fn preempt(&mut self, mut s: Seq) {
        self.release(&mut s);
        self.local.preemptions += 1;
        self.waiting.push_front(s);
    }

    /// Grows a running sequence's block tables to hold `target` and
    /// `draft` tokens; false if the cache is full.
    fn grow(&mut self, i: usize, target: usize, draft: usize) -> bool {
        let bs = self.cfg.block_size;
        let s = &mut self.running[i];
        while s.blocks.len() * bs < target {
            match self.blocks.allocate() {
                Some(b) => s.blocks.push(b),
                None => return false,
            }
        }
        if let Some(d) = &mut self.draft {
            while s.draft_blocks.len() * bs < draft {
                match d.blocks.allocate() {
                    Some(b) => s.draft_blocks.push(b),
                    None => return false,
                }
            }
        }
        true
    }

    fn reap_cancelled(&mut self) {
        let cancelled = |s: &Seq| s.req.cancel.load(Ordering::Relaxed);
        let mut i = 0;
        while i < self.running.len() {
            if cancelled(&self.running[i]) {
                let s = self.running.remove(i);
                self.finish(s, Finish::Cancelled);
            } else {
                i += 1;
            }
        }
        let (gone, keep): (Vec<Seq>, Vec<Seq>) = self.waiting.drain(..).partition(|s| cancelled(s));
        self.waiting = keep.into();
        for s in gone {
            self.finish(s, Finish::Cancelled);
        }
    }

    /// Runs one batch.
    pub fn step(&mut self) {
        let start = Instant::now();
        self.reap_cancelled();
        let plans = self.schedule();
        if !plans.is_empty() {
            self.execute(plans);
        }
        self.local.steps += 1;
        self.local.busy_secs += start.elapsed().as_secs_f64();
        self.local.peak_running = self.local.peak_running.max(self.running.len());
        let mut st = self.stats.lock().unwrap();
        *st = Stats {
            running: self.running.len(),
            waiting: self.waiting.len(),
            kv_used_blocks: self.blocks.used(),
            kv_total_blocks: self.blocks.total(),
            cached_tokens: self.local.cached_tokens,
            ..self.local.clone()
        };
    }

    fn schedule(&mut self) -> Vec<Plan> {
        let bs = self.cfg.block_size;
        let k = if self.draft.is_some() { self.cfg.spec_k } else { 0 };
        let mut budget = self.cfg.max_batch_tokens;
        let mut plans = Vec::new();

        // Running sequences, oldest first; newest are preempted first.
        let mut i = 0;
        while i < self.running.len() && budget > 0 {
            let s = &self.running[i];
            let left = s.tokens.len() - s.computed;
            let decoding = left == 1;
            // Speculate only when there is room for every proposal.
            let spec = decoding && k > 0 && s.tokens.len() + k < self.max_len && budget > k;
            let n = if spec { k + 1 } else { left.min(budget) };
            let target = s.computed + n;
            let draft = if spec { s.tokens.len() + k - 1 } else { 0 };
            if self.grow(i, target.max(self.reserve_for(i)), draft) {
                plans.push(Plan { seq: i, n, drafts: Vec::new(), draft_dists: Vec::new() });
                budget -= n;
                i += 1;
                continue;
            }
            let victim = self.running.pop().unwrap();
            self.preempt(victim);
            // If the victim was this sequence, stop here.
        }

        // New requests, first come first served.
        let static_open = self.running.is_empty();
        while let Some(s) = self.waiting.front_mut() {
            if self.running.len() >= self.cfg.max_seqs {
                break;
            }
            match self.cfg.admission {
                Admission::Continuous if budget == 0 => break,
                Admission::Static if !static_open => break,
                _ => {}
            }
            // Leading full blocks already in the prefix cache (never the
            // block holding the last prompt token, whose logits we need).
            if s.blocks.is_empty() {
                let mut parent = ROOT;
                for chunk in s.tokens[..s.tokens.len() - 1].chunks_exact(bs) {
                    match self.blocks.lookup(parent, chunk) {
                        Some(b) => {
                            parent = crate::kv::chain(parent, chunk);
                            s.blocks.push(b);
                            s.keys.push(parent);
                        }
                        None => break,
                    }
                }
                s.computed = s.blocks.len() * bs;
            }
            let n = (s.tokens.len() - s.computed).min(budget);
            let want = match self.cfg.reserve_tokens {
                Some(r) => r.max(s.computed + n).min(self.max_len),
                None => s.computed + n,
            };
            let need = want.div_ceil(bs).saturating_sub(s.blocks.len());
            // Keep a little headroom so running sequences can grow.
            let headroom = if self.running.is_empty() { 0 } else { (self.blocks.total() / 100).max(1) };
            if self.blocks.available() < need + headroom {
                let mut s = self.waiting.pop_front().unwrap();
                self.release(&mut s);
                self.waiting.push_front(s);
                break;
            }
            let s = self.waiting.pop_front().unwrap();
            self.local.cached_tokens += s.computed as u64;
            self.running.push(s);
            let i = self.running.len() - 1;
            let ok = self.grow(i, want, 0);
            debug_assert!(ok);
            if n > 0 {
                plans.push(Plan { seq: i, n, drafts: Vec::new(), draft_dists: Vec::new() });
                budget -= n;
            }
        }
        plans
    }

    /// Token slots reserved for a running sequence under `reserve_tokens`.
    fn reserve_for(&self, i: usize) -> usize {
        self.cfg.reserve_tokens.map_or(0, |r| r.min(self.max_len).max(self.running[i].tokens.len()))
    }

    fn execute(&mut self, mut plans: Vec<Plan>) {
        let bs = self.cfg.block_size;
        let spec: Vec<usize> = (0..plans.len())
            .filter(|&p| {
                let s = &self.running[plans[p].seq];
                self.draft.is_some() && s.tokens.len() - s.computed == 1 && plans[p].n > 1
            })
            .collect();
        if !spec.is_empty() {
            self.propose(&mut plans, &spec);
        }

        // One target pass over every plan.
        let inputs: Vec<Vec<u32>> = plans
            .iter()
            .map(|p| {
                let s = &self.running[p.seq];
                let mut v = s.tokens[s.computed..s.computed + p.n - p.drafts.len()].to_vec();
                v.extend_from_slice(&p.drafts);
                v
            })
            .collect();
        let chunks: Vec<Chunk> = plans
            .iter()
            .zip(&inputs)
            .map(|(p, t)| {
                let s = &self.running[p.seq];
                let done = s.computed + t.len() - p.drafts.len() == s.tokens.len();
                Chunk {
                    tokens: t,
                    pos: s.computed,
                    blocks: &s.blocks,
                    logits: if !p.drafts.is_empty() {
                        Logits::All
                    } else if done {
                        Logits::Last
                    } else {
                        Logits::None
                    },
                }
            })
            .collect();
        let want: Vec<Logits> = chunks.iter().map(|c| c.logits).collect();
        let logits = self.model.forward(&self.pool, &mut self.cache, &chunks);
        drop(chunks);
        let vocab = self.model.cfg.vocab;

        let mut row = 0;
        let mut finished = Vec::new();
        for (p, w) in plans.iter().zip(&want) {
            self.local.computed_tokens += p.n as u64;
            let s = &mut self.running[p.seq];
            let mut new = Vec::new();
            match w {
                Logits::None => s.computed += p.n,
                Logits::Last => {
                    s.computed += p.n;
                    new.push(sampler::sample(&logits[row * vocab..(row + 1) * vocab], &s.req.params, &mut s.rng));
                    row += 1;
                }
                Logits::All => {
                    // Verify proposals d1..dk against target rows 0..k.
                    let rows = &logits[row * vocab..(row + p.n) * vocab];
                    row += p.n;
                    let base = s.tokens.len();
                    let mut accepted = 0;
                    for (j, &d) in p.drafts.iter().enumerate() {
                        let l = &rows[j * vocab..(j + 1) * vocab];
                        let params = &s.req.params;
                        let take = if params.greedy() {
                            let t = sampler::argmax(l);
                            if t == d { None } else { Some(t) }
                        } else {
                            let pd = dense(&sampler::distribution(l, params), vocab);
                            let qd = &p.draft_dists[j];
                            let (px, qx) = (pd[d as usize], qd[d as usize]);
                            if qx > 0.0 && s.rng.uniform() < f64::from((px / qx).min(1.0)) {
                                None
                            } else {
                                // Resample from the part of p that q under-covers.
                                let resid: Vec<(u32, f32)> = pd
                                    .iter()
                                    .zip(qd)
                                    .enumerate()
                                    .filter_map(|(t, (&a, &b))| (a > b).then_some((t as u32, a - b)))
                                    .collect();
                                Some(if resid.is_empty() { sampler::draw(&dense_pairs(&pd), &mut s.rng) } else { sampler::draw(&resid, &mut s.rng) })
                            }
                        };
                        match take {
                            None => {
                                new.push(d);
                                accepted += 1;
                            }
                            Some(t) => {
                                new.push(t);
                                break;
                            }
                        }
                    }
                    if accepted == p.drafts.len() {
                        let l = &rows[p.drafts.len() * vocab..];
                        new.push(sampler::sample(l, &s.req.params, &mut s.rng));
                    }
                    self.local.spec_proposed += p.drafts.len() as u64;
                    self.local.spec_accepted += accepted as u64;
                    // Both caches are valid through the last accepted proposal.
                    s.computed = base + accepted;
                    s.draft_computed = s.draft_computed.min(base + accepted);
                }
            }
            for t in new {
                let s = &mut self.running[p.seq];
                let params = &s.req.params;
                let eos = !params.ignore_eos && self.model.cfg.eos.contains(&t);
                if eos || params.stop_ids.contains(&t) {
                    finished.push((p.seq, Finish::Stop));
                    break;
                }
                s.tokens.push(t);
                s.generated += 1;
                self.local.generated_tokens += 1;
                if s.req.events.send(Event::Token(t)).is_err() {
                    finished.push((p.seq, Finish::Cancelled));
                    break;
                }
                if s.generated >= params.max_tokens || s.tokens.len() >= self.max_len {
                    finished.push((p.seq, Finish::Length));
                    break;
                }
            }
            // Publish newly completed blocks to the prefix cache.
            let s = &mut self.running[p.seq];
            while s.keys.len() < s.computed.min(s.tokens.len()) / bs {
                let i = s.keys.len();
                let parent = s.keys.last().copied().unwrap_or(ROOT);
                let key = self.blocks.publish(s.blocks[i], parent, &s.tokens[i * bs..(i + 1) * bs]);
                s.keys.push(key);
            }
        }
        finished.sort_by_key(|f| std::cmp::Reverse(f.0));
        for (i, why) in finished {
            let s = self.running.remove(i);
            self.finish(s, why);
        }
    }

    /// Runs the draft model `k` times over the speculating sequences and
    /// records its proposals in their plans.
    fn propose(&mut self, plans: &mut [Plan], spec: &[usize]) {
        let k = self.cfg.spec_k;
        let d = self.draft.as_mut().unwrap();
        let vocab = d.model.cfg.vocab;
        for round in 0..k {
            let inputs: Vec<Vec<u32>> = spec
                .iter()
                .map(|&p| {
                    let s = &self.running[plans[p].seq];
                    if round == 0 {
                        s.tokens[s.draft_computed..].to_vec()
                    } else {
                        vec![*plans[p].drafts.last().unwrap()]
                    }
                })
                .collect();
            let chunks: Vec<Chunk> = spec
                .iter()
                .zip(&inputs)
                .map(|(&p, t)| {
                    let s = &self.running[plans[p].seq];
                    let pos = if round == 0 { s.draft_computed } else { s.tokens.len() + round - 1 };
                    Chunk { tokens: t, pos, blocks: &s.draft_blocks, logits: Logits::Last }
                })
                .collect();
            let logits = d.model.forward(&self.pool, &mut d.cache, &chunks);
            drop(chunks);
            for (r, &p) in spec.iter().enumerate() {
                let l = &logits[r * vocab..(r + 1) * vocab];
                let s = &mut self.running[plans[p].seq];
                if round == 0 {
                    s.draft_computed = s.tokens.len();
                }
                let params = &s.req.params;
                let t = if params.greedy() {
                    sampler::argmax(l)
                } else {
                    let dist = sampler::distribution(l, params);
                    let t = sampler::draw(&dist, &mut s.rng);
                    plans[p].draft_dists.push(dense(&dist, vocab));
                    t
                };
                plans[p].drafts.push(t);
            }
        }
        // The draft cache now holds every proposal but the last.
        for &p in spec {
            let s = &mut self.running[plans[p].seq];
            s.draft_computed = s.tokens.len() + k - 1;
        }
    }
}

fn dense(pairs: &[(u32, f32)], vocab: usize) -> Vec<f32> {
    let mut v = vec![0f32; vocab];
    for &(t, p) in pairs {
        v[t as usize] = p;
    }
    v
}

fn dense_pairs(v: &[f32]) -> Vec<(u32, f32)> {
    v.iter().enumerate().filter(|e| *e.1 > 0.0).map(|(t, &p)| (t as u32, p)).collect()
}

/// Collects a request's events into its generated tokens and finish reason.
pub fn collect(rx: &Receiver<Event>) -> (Vec<u32>, Finish) {
    let mut out = Vec::new();
    loop {
        match rx.recv() {
            Ok(Event::Token(t)) => out.push(t),
            Ok(Event::Done(f)) => return (out, f),
            Err(_) => return (out, Finish::Cancelled),
        }
    }
}
