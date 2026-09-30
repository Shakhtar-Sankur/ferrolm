//! The transformer forward pass (Llama, Mistral and Qwen2 families) over a
//! batch of sequence chunks whose keys and values live in a paged cache.
//!
//! One call runs any mix of prompt chunks and single decode tokens from
//! different sequences: their rows are stacked for every matrix multiply,
//! so the weights are read once per step however many sequences are in the
//! batch. Attention reads each sequence's own blocks. Every per-row
//! computation reduces in a fixed order, so a token's logits are bitwise
//! the same whatever else is in the batch and however its prompt was split
//! into chunks.

use crate::config::{Config, RopeScaling};
use crate::kernels::{self, Matrix};
use crate::kv::KvCache;
use crate::pool::{Out, Pool};
use crate::rng::Rng;
use crate::safetensors::{Checkpoint, f32_to_bf16};
use std::cell::RefCell;
use std::path::Path;

thread_local! {
    static SCRATCH: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

pub struct Layer {
    attn_norm: Vec<f32>,
    /// q, k and v projections stacked.
    wqkv: Matrix,
    bqkv: Option<Vec<f32>>,
    wo: Matrix,
    mlp_norm: Vec<f32>,
    /// gate and up projections stacked.
    w_gate_up: Matrix,
    w_down: Matrix,
}

pub struct Model {
    pub cfg: Config,
    embed: Matrix,
    layers: Vec<Layer>,
    norm: Vec<f32>,
    lm_head: Option<Matrix>,
    inv_freq: Vec<f32>,
}

/// Which logits a chunk needs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Logits {
    None,
    Last,
    All,
}

/// `tokens` at positions `pos..pos + tokens.len()` of a sequence whose
/// cache blocks are `blocks`; earlier positions must already be cached.
#[derive(Clone, Debug)]
pub struct Chunk<'a> {
    pub tokens: &'a [u32],
    pub pos: usize,
    pub blocks: &'a [u32],
    pub logits: Logits,
}

impl Model {
    pub fn load(dir: &Path) -> Result<Model, String> {
        let cfg = Config::load(dir)?;
        let ck = Checkpoint::open(dir)?;
        // Row-major weights, stacked (q/k/v, gate/up) before packing.
        let stack = |names: &[String]| -> Result<Matrix, String> {
            let (mut rows, mut cols, mut data) = (0, 0, Vec::new());
            for name in names {
                let info = ck.info(name)?;
                let [r, c] = info.shape[..] else {
                    return Err(format!("{name}: expected a matrix, got {:?}", info.shape));
                };
                if cols != 0 && c != cols {
                    return Err(format!("{name}: {c} columns, expected {cols}"));
                }
                rows += r;
                cols = c;
                data.extend(ck.bf16(name)?);
            }
            Ok(Matrix::new(rows, cols, data))
        };
        let mat = |name: &str| stack(&[name.to_string()]);
        let vec = |name: &str| ck.f32(name);
        let mut layers = Vec::with_capacity(cfg.layers);
        for i in 0..cfg.layers {
            let p = format!("model.layers.{i}");
            let a = format!("{p}.self_attn");
            let bqkv = if cfg.qkv_bias {
                let mut b = vec(&format!("{a}.q_proj.bias"))?;
                b.extend(vec(&format!("{a}.k_proj.bias"))?);
                b.extend(vec(&format!("{a}.v_proj.bias"))?);
                Some(b)
            } else {
                None
            };
            layers.push(Layer {
                attn_norm: vec(&format!("{p}.input_layernorm.weight"))?,
                wqkv: stack(&["q", "k", "v"].map(|x| format!("{a}.{x}_proj.weight")))?,
                bqkv,
                wo: mat(&format!("{a}.o_proj.weight"))?,
                mlp_norm: vec(&format!("{p}.post_attention_layernorm.weight"))?,
                w_gate_up: stack(&["gate", "up"].map(|x| format!("{p}.mlp.{x}_proj.weight")))?,
                w_down: mat(&format!("{p}.mlp.down_proj.weight"))?,
            });
        }
        let lm_head = if cfg.tie_embeddings || !ck.tensors.contains_key("lm_head.weight") {
            None
        } else {
            Some(mat("lm_head.weight")?)
        };
        let m = Model {
            embed: mat("model.embed_tokens.weight")?,
            norm: vec("model.norm.weight")?,
            inv_freq: inv_freq(&cfg),
            layers,
            lm_head,
            cfg,
        };
        m.check_shapes()?;
        Ok(m)
    }

    /// A model with random weights, for tests and kernel benchmarks.
    pub fn random(cfg: Config, seed: u64) -> Model {
        let mut rng = Rng::new(seed);
        let mut mat = |rows: usize, cols: usize, scale: f64| {
            let data = (0..rows * cols)
                .map(|_| f32_to_bf16((rng.normal() * scale) as f32))
                .collect();
            Matrix::new(rows, cols, data)
        };
        let (h, hd) = (cfg.hidden, cfg.head_dim);
        let qkv = (cfg.heads + 2 * cfg.kv_heads) * hd;
        let s = 1.0 / (h as f64).sqrt();
        let layers = (0..cfg.layers)
            .map(|_| Layer {
                attn_norm: vec![1.0; h],
                wqkv: mat(qkv, h, s * 2.0),
                bqkv: cfg.qkv_bias.then(|| vec![0.01; qkv]),
                wo: mat(h, cfg.heads * hd, s),
                mlp_norm: vec![1.0; h],
                w_gate_up: mat(2 * cfg.intermediate, h, s),
                w_down: mat(h, cfg.intermediate, 1.0 / (cfg.intermediate as f64).sqrt()),
            })
            .collect();
        let embed = mat(cfg.vocab, h, 1.0);
        let lm_head = (!cfg.tie_embeddings).then(|| mat(cfg.vocab, h, s * 4.0));
        Model {
            norm: vec![1.0; h],
            inv_freq: inv_freq(&cfg),
            embed,
            layers,
            lm_head,
            cfg,
        }
    }

    fn check_shapes(&self) -> Result<(), String> {
        let c = &self.cfg;
        let qkv = (c.heads + 2 * c.kv_heads) * c.head_dim;
        let ok = self.embed.rows == c.vocab
            && self.embed.cols == c.hidden
            && self.layers.iter().all(|l| {
                l.wqkv.rows == qkv
                    && l.wqkv.cols == c.hidden
                    && l.wo.rows == c.hidden
                    && l.wo.cols == c.heads * c.head_dim
                    && l.w_gate_up.rows == 2 * c.intermediate
                    && l.w_down.cols == c.intermediate
            });
        if ok { Ok(()) } else { Err("weight shapes do not match config.json".into()) }
    }

    fn lm_head(&self) -> &Matrix {
        self.lm_head.as_ref().unwrap_or(&self.embed)
    }

    /// Runs `chunks` through the model, writing their keys and values into
    /// `cache`, and returns the requested logits rows, in chunk order.
    pub fn forward(&self, pool: &Pool, cache: &mut KvCache, chunks: &[Chunk]) -> Vec<f32> {
        let c = &self.cfg;
        let (h, hd, nh, nkv) = (c.hidden, c.head_dim, c.heads, c.kv_heads);
        let q_dim = nh * hd;
        let qkv_dim = (nh + 2 * nkv) * hd;
        let t: usize = chunks.iter().map(|ch| ch.tokens.len()).sum();
        if t == 0 {
            return Vec::new();
        }
        // Where each row comes from.
        let mut pos = Vec::with_capacity(t);
        let mut slot = Vec::with_capacity(t);
        let mut owner = Vec::with_capacity(t);
        let bs = cache.block_size;
        for (ci, ch) in chunks.iter().enumerate() {
            assert!(
                ch.blocks.len() * bs >= ch.pos + ch.tokens.len(),
                "chunk overruns its cache blocks"
            );
            for i in 0..ch.tokens.len() {
                let p = ch.pos + i;
                pos.push(p);
                slot.push(ch.blocks[p / bs] as usize * bs + p % bs);
                owner.push(ci);
            }
        }
        let mut x = vec![0f32; t * h];
        for (r, &tok) in chunks.iter().flat_map(|ch| ch.tokens).enumerate() {
            self.embed.row_f32(tok as usize, &mut x[r * h..(r + 1) * h]);
        }
        let (cos, sin) = self.rope_tables(&pos);

        let mut xn = vec![0f32; t * h];
        let mut qkv = vec![0f32; t * qkv_dim];
        let mut att = vec![0f32; t * q_dim];
        let mut proj = vec![0f32; t * h];
        let mut gate_up = vec![0f32; t * 2 * c.intermediate];
        let mut act = vec![0f32; t * c.intermediate];

        for (li, l) in self.layers.iter().enumerate() {
            par_rows(pool, t, &x, &mut xn, |a, b| kernels::rms_norm(a, &l.attn_norm, c.rms_eps, b));
            kernels::matmul(pool, &xn, t, &l.wqkv, l.bqkv.as_deref(), &mut qkv);
            for r in 0..t {
                let row = &mut qkv[r * qkv_dim..(r + 1) * qkv_dim];
                let (cr, sr) = (&cos[r * hd / 2..(r + 1) * hd / 2], &sin[r * hd / 2..(r + 1) * hd / 2]);
                for head in row[..(nh + nkv) * hd].chunks_exact_mut(hd) {
                    rope(head, cr, sr);
                }
                cache.write(li, slot[r], &row[q_dim..q_dim + nkv * hd], &row[q_dim + nkv * hd..]);
            }
            self.attention(pool, cache, li, chunks, &pos, &owner, &qkv, &mut att);
            kernels::matmul(pool, &att, t, &l.wo, None, &mut proj);
            kernels::add(&mut x, &proj);
            par_rows(pool, t, &x, &mut xn, |a, b| kernels::rms_norm(a, &l.mlp_norm, c.rms_eps, b));
            kernels::matmul(pool, &xn, t, &l.w_gate_up, None, &mut gate_up);
            let inter = c.intermediate;
            par_rows(pool, t, &gate_up, &mut act, |a, b| kernels::silu_mul(a, inter, b));
            kernels::matmul(pool, &act, t, &l.w_down, None, &mut proj);
            kernels::add(&mut x, &proj);
        }

        // Only the rows whose logits were asked for go through the head.
        let mut rows = Vec::new();
        let mut r = 0;
        for ch in chunks {
            let n = ch.tokens.len();
            match ch.logits {
                Logits::None => {}
                Logits::Last => rows.push(r + n - 1),
                Logits::All => rows.extend(r..r + n),
            }
            r += n;
        }
        let mut hsel = vec![0f32; rows.len() * h];
        for (i, &r) in rows.iter().enumerate() {
            kernels::rms_norm(&x[r * h..(r + 1) * h], &self.norm, c.rms_eps, &mut hsel[i * h..(i + 1) * h]);
        }
        let mut logits = vec![0f32; rows.len() * c.vocab];
        kernels::matmul(pool, &hsel, rows.len(), self.lm_head(), None, &mut logits);
        logits
    }

    #[allow(clippy::too_many_arguments)]
    fn attention(
        &self,
        pool: &Pool,
        cache: &KvCache,
        layer: usize,
        chunks: &[Chunk],
        pos: &[usize],
        owner: &[usize],
        qkv: &[f32],
        att: &mut [f32],
    ) {
        let c = &self.cfg;
        let (hd, nh, nkv) = (c.head_dim, c.heads, c.kv_heads);
        let qkv_dim = (nh + 2 * nkv) * hd;
        let group = nh / nkv;
        let scale = 1.0 / (hd as f32).sqrt();
        let bs = cache.block_size;
        let t = pos.len();
        let out = Out::new(att);
        let (k, v) = cache.layer(layer);
        let block_stride = cache.block_stride();
        pool.run(t * nkv, &|task| {
            let (r, g) = (task / nkv, task % nkv);
            let view = kernels::KvView {
                k,
                v,
                blocks: chunks[owner[r]].blocks,
                bs,
                hd,
                block_stride,
                head_off: cache.head_offset(g),
            };
            let q = &qkv[r * qkv_dim + g * group * hd..][..group * hd];
            // SAFETY: each task writes its own (row, kv group) slice.
            let o = unsafe { out.slice(r * nh * hd + g * group * hd, group * hd) };
            SCRATCH.with(|s| kernels::attend(q, hd, pos[r] + 1, scale, &view, &mut s.borrow_mut(), o));
        });
    }

    /// cos and sin of each row's rotation angles, `head_dim / 2` per row,
    /// computed in f32 as Hugging Face does.
    fn rope_tables(&self, pos: &[usize]) -> (Vec<f32>, Vec<f32>) {
        let half = self.cfg.head_dim / 2;
        let mut cos = Vec::with_capacity(pos.len() * half);
        let mut sin = Vec::with_capacity(pos.len() * half);
        for &p in pos {
            for &f in &self.inv_freq {
                let a = p as f32 * f;
                cos.push(a.cos());
                sin.push(a.sin());
            }
        }
        (cos, sin)
    }
}

/// Rotates one head's vector in place (the "rotate half" convention).
fn rope(x: &mut [f32], cos: &[f32], sin: &[f32]) {
    let half = x.len() / 2;
    for i in 0..half {
        let (a, b) = (x[i], x[i + half]);
        x[i] = a * cos[i] + -b * sin[i];
        x[i + half] = b * cos[i] + a * sin[i];
    }
}

fn inv_freq(c: &Config) -> Vec<f32> {
    let dim = c.head_dim;
    let base: Vec<f32> = (0..dim / 2)
        .map(|i| 1.0 / c.rope_theta.powf((2 * i) as f32 / dim as f32))
        .collect();
    match c.rope_scaling {
        RopeScaling::None => base,
        RopeScaling::Llama3 {
            factor,
            low_freq_factor,
            high_freq_factor,
            original_max_position,
        } => {
            let old = original_max_position as f32;
            let low_wavelen = old / low_freq_factor;
            let high_wavelen = old / high_freq_factor;
            base.into_iter()
                .map(|f| {
                    let wavelen = 2.0 * std::f32::consts::PI / f;
                    if wavelen < high_wavelen {
                        f
                    } else if wavelen > low_wavelen {
                        f / factor
                    } else {
                        let smooth = (old / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor);
                        (1.0 - smooth) * f / factor + smooth * f
                    }
                })
                .collect()
        }
    }
}

/// Applies `f` to each `width`-wide row of `x`, writing the row of `y`, in
/// parallel for large batches.
fn par_rows(pool: &Pool, rows: usize, x: &[f32], y: &mut [f32], f: impl Fn(&[f32], &mut [f32]) + Sync) {
    const PER_TASK: usize = 4;
    let (xw, yw) = (x.len() / rows, y.len() / rows);
    if rows <= PER_TASK {
        f(x, y);
        return;
    }
    let out = Out::new(y);
    pool.run(rows.div_ceil(PER_TASK), &|t| {
        let r0 = t * PER_TASK;
        let r1 = (r0 + PER_TASK).min(rows);
        // SAFETY: tasks cover disjoint row ranges.
        let yr = unsafe { out.slice(r0 * yw, (r1 - r0) * yw) };
        f(&x[r0 * xw..r1 * xw], yr);
    });
}
