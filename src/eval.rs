//! Perplexity: how well a model predicts held-out text, the standard check
//! that quantization kept a model's quality. The text is cut into
//! non-overlapping windows of `ctx` tokens; each window is scored on
//! predicting its tokens 2..ctx from the ones before.

use crate::kv::KvCache;
use crate::model::{Chunk, Logits, Model};
use crate::pool::Pool;

#[derive(Clone, Debug)]
pub struct Perplexity {
    pub ppl: f64,
    pub mean_nll: f64,
    pub tokens: usize,
    pub windows: usize,
}

pub fn perplexity(model: &Model, pool: &Pool, tokens: &[u32], ctx: usize, windows: usize) -> Perplexity {
    let c = &model.cfg;
    let bs = 16;
    let blocks: Vec<u32> = (0..ctx.div_ceil(bs) as u32).collect();
    let mut cache = KvCache::new(c.layers, c.kv_heads, c.head_dim, bs, blocks.len());
    let (mut nll, mut n, mut used) = (0f64, 0usize, 0usize);
    for w in tokens.chunks_exact(ctx).take(windows) {
        let logits = model.forward(
            pool,
            &mut cache,
            &[Chunk {
                tokens: w,
                pos: 0,
                blocks: &blocks,
                logits: Logits::All,
            }],
        );
        for (i, row) in logits.chunks_exact(c.vocab).enumerate().take(ctx - 1) {
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let sum: f64 = row.iter().map(|&v| f64::from(v - max).exp()).sum();
            let target = f64::from(row[w[i + 1] as usize] - max);
            nll += sum.ln() - target;
            n += 1;
        }
        used += 1;
    }
    let mean = nll / n.max(1) as f64;
    Perplexity {
        ppl: mean.exp(),
        mean_nll: mean,
        tokens: n,
        windows: used,
    }
}

/// `count` calibration sequences of `len` tokens, evenly spaced through
/// `tokens`.
pub fn sample_sequences(tokens: &[u32], count: usize, len: usize) -> Vec<Vec<u32>> {
    let span = tokens.len().saturating_sub(len);
    (0..count)
        .map(|i| tokens[span * i / count.max(1)..][..len].to_vec())
        .collect()
}
