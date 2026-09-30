//! Activation-aware weight quantization (after AWQ, Lin et al. 2023).
//!
//! A few input channels of each projection carry much larger activations
//! than the rest, and rounding error on their weights costs the most. So
//! before quantizing, each weight column c is multiplied by a scale
//! s_c = (mean |x_c|)^α and the input is divided by s_c. The product is
//! unchanged, but the important columns now use more of the quantization
//! range. The division is folded into whatever produces the input: the
//! RMSNorm weights for the q/k/v and gate/up projections, the up
//! projection's rows for the down projection, and (when every query head
//! has its own kv head) the v projection's rows for the output projection.
//! α is chosen per projection by measuring the output error on calibration
//! activations; α = 0 is plain quantization, so the search never does
//! worse on that sample.

use crate::kernels::{Matrix, matmul};
use crate::kv::KvCache;
use crate::model::{Capture, Chunk, Logits, Model};
use crate::pool::Pool;
use crate::quant::{QMatrix, Quant};
use std::sync::atomic::Ordering;

/// What calibration found for one projection: the chosen α and how much
/// it reduced the output error compared with plain quantization.
#[derive(Clone, Debug)]
pub struct Choice {
    pub layer: usize,
    pub which: &'static str,
    pub alpha: f32,
    pub error_plain: f64,
    pub error_awq: f64,
}

const NAMES: [&str; 4] = ["qkv", "o", "gate_up", "down"];

impl Model {
    /// Runs `seqs` through the model and returns a sample of up to
    /// `max_rows` input rows for each projection of each layer.
    pub fn calibrate(&self, pool: &Pool, seqs: &[Vec<u32>], max_rows: usize) -> Vec<[Vec<Vec<f32>>; 4]> {
        *self.capture.lock().unwrap() = Some(Capture::new(max_rows));
        self.capturing.store(true, Ordering::Relaxed);
        let c = &self.cfg;
        for s in seqs {
            let bs = 16;
            let blocks: Vec<u32> = (0..s.len().div_ceil(bs) as u32).collect();
            let mut cache = KvCache::new(c.layers, c.kv_heads, c.head_dim, bs, blocks.len());
            self.forward(
                pool,
                &mut cache,
                &[Chunk {
                    tokens: s,
                    pos: 0,
                    blocks: &blocks,
                    logits: Logits::None,
                }],
            );
        }
        self.capturing.store(false, Ordering::Relaxed);
        self.capture
            .lock()
            .unwrap()
            .take()
            .map(|c| c.layers)
            .unwrap_or_default()
    }

    /// Quantizes the transformer layers with activation-aware scaling,
    /// using activations from `calibrate`.
    pub fn quantize_awq(&mut self, kind: Quant, calib: &[[Vec<Vec<f32>>; 4]]) -> Vec<Choice> {
        let pool = Pool::with_all_cores();
        let (heads, kv_heads) = (self.cfg.heads, self.cfg.kv_heads);
        let (q_dim, kv_dim) = (heads * self.cfg.head_dim, kv_heads * self.cfg.head_dim);
        let inter = self.cfg.intermediate;
        let mut choices = Vec::new();
        for (li, l) in self.layers.iter_mut().enumerate() {
            let cal = &calib[li];
            let mut qkv = l.wqkv.to_f32();
            let mut gate_up = l.w_gate_up.to_f32();
            let hidden = l.attn_norm.len();

            // Output projection: its input comes from v, so the scale can
            // be folded into v's rows only when heads map one to one.
            let o = l.wo.to_f32();
            if heads == kv_heads {
                let (s, ch) = search(&pool, kind, &o, l.wo.rows, q_dim, &cal[1], li, NAMES[1]);
                for (c, &sc) in s.iter().enumerate() {
                    let row = q_dim + kv_dim + c;
                    for v in &mut qkv[row * hidden..(row + 1) * hidden] {
                        *v /= sc;
                    }
                }
                l.wo = apply(kind, &o, l.wo.rows, q_dim, &s, &cal[1]);
                choices.push(ch);
            } else {
                l.wo = apply(kind, &o, l.wo.rows, q_dim, &vec![1.0; q_dim], &cal[1]);
            }

            // Down projection: fold into the up projection's rows.
            let down = l.w_down.to_f32();
            let (s, ch) = search(&pool, kind, &down, l.w_down.rows, inter, &cal[3], li, NAMES[3]);
            for (c, &sc) in s.iter().enumerate() {
                for v in &mut gate_up[(inter + c) * hidden..(inter + c + 1) * hidden] {
                    *v /= sc;
                }
            }
            l.w_down = apply(kind, &down, l.w_down.rows, inter, &s, &cal[3]);
            choices.push(ch);

            // Gate/up and q/k/v: fold into the RMSNorm weights.
            let (s, ch) = search(&pool, kind, &gate_up, l.w_gate_up.rows, hidden, &cal[2], li, NAMES[2]);
            for (g, &sc) in l.mlp_norm.iter_mut().zip(&s) {
                *g /= sc;
            }
            l.w_gate_up = apply(kind, &gate_up, l.w_gate_up.rows, hidden, &s, &cal[2]);
            choices.push(ch);

            let (s, ch) = search(&pool, kind, &qkv, l.wqkv.rows, hidden, &cal[0], li, NAMES[0]);
            for (g, &sc) in l.attn_norm.iter_mut().zip(&s) {
                *g /= sc;
            }
            l.wqkv = apply(kind, &qkv, l.wqkv.rows, hidden, &s, &cal[0]);
            choices.push(ch);
        }
        choices
    }
}

/// Mean |x| and mean x² per input channel.
fn stats(rows: &[Vec<f32>], k: usize) -> (Vec<f32>, Vec<f32>) {
    let (mut a, mut q) = (vec![0f64; k], vec![0f64; k]);
    for r in rows {
        for c in 0..k {
            a[c] += f64::from(r[c].abs());
            q[c] += f64::from(r[c]) * f64::from(r[c]);
        }
    }
    let n = rows.len().max(1) as f64;
    (
        a.iter().map(|v| (v / n) as f32).collect(),
        q.iter().map(|v| (v / n) as f32).collect(),
    )
}

/// Scales for α: (mean |x|)^α, normalised so their geometric middle is 1.
fn scales(abs_mean: &[f32], alpha: f32) -> Vec<f32> {
    let s: Vec<f32> = abs_mean.iter().map(|&a| a.max(1e-5).powf(alpha)).collect();
    let (mx, mn) = s.iter().fold((0f32, f32::MAX), |(a, b), &v| (a.max(v), b.min(v)));
    let norm = (mx * mn).sqrt().max(1e-8);
    s.iter().map(|v| (v / norm).max(1e-4)).collect()
}

/// Quantizes `w` (rows × k, row-major) with its columns multiplied by `s`;
/// rounding error is weighted by the scaled inputs' mean square.
fn apply(kind: Quant, w: &[f32], rows: usize, k: usize, s: &[f32], cal: &[Vec<f32>]) -> Matrix {
    let mut ws = w.to_vec();
    for row in ws.chunks_exact_mut(k) {
        for (v, &sc) in row.iter_mut().zip(s) {
            *v *= sc;
        }
    }
    let (_, sq) = stats(cal, k);
    let imp: Vec<f32> = sq.iter().zip(s).map(|(q, sc)| q / (sc * sc) + 1e-8).collect();
    Matrix::from_quantized(QMatrix::quantize(kind, rows, k, &ws, Some(&imp)))
}

/// The α (and scales) that minimise the projection's output error on the
/// calibration rows.
#[allow(clippy::too_many_arguments)]
fn search(
    pool: &Pool,
    kind: Quant,
    w: &[f32],
    rows: usize,
    k: usize,
    cal: &[Vec<f32>],
    layer: usize,
    which: &'static str,
) -> (Vec<f32>, Choice) {
    let (abs_mean, _) = stats(cal, k);
    let m = cal.len();
    let x: Vec<f32> = cal.iter().flatten().copied().collect();
    // The reference output, with the unquantized weights.
    let reference = Matrix::from_f32(rows, k, w);
    let mut y_ref = vec![0f32; m * rows];
    matmul(pool, &x, m, &reference, None, &mut y_ref);
    let mut best: Option<(f64, f32, Vec<f32>)> = None;
    let mut plain = 0.0;
    for step in 0..=10 {
        let alpha = step as f32 / 10.0;
        let s = scales(&abs_mean, alpha);
        let q = apply(kind, w, rows, k, &s, cal);
        let xs: Vec<f32> = x
            .chunks_exact(k)
            .flat_map(|r| r.iter().zip(&s).map(|(v, sc)| v / sc))
            .collect();
        let mut y = vec![0f32; m * rows];
        matmul(pool, &xs, m, &q, None, &mut y);
        let err: f64 = y.iter().zip(&y_ref).map(|(a, b)| f64::from(a - b).powi(2)).sum::<f64>() / (m * rows) as f64;
        if step == 0 {
            plain = err;
        }
        if best.as_ref().is_none_or(|b| err < b.0) {
            best = Some((err, alpha, s));
        }
    }
    let (err, alpha, s) = best.unwrap();
    (
        s,
        Choice {
            layer,
            which,
            alpha,
            error_plain: plain,
            error_awq: err,
        },
    )
}
