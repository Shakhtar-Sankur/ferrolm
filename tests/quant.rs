//! Quantized models against the bf16 model they came from.

use ferrolm::kv::KvCache;
use ferrolm::model::{Chunk, Logits, Model};
use ferrolm::pool::Pool;
use ferrolm::quant::Quant;
use ferrolm::rng::Rng;
use std::path::Path;

fn model() -> Model {
    Model::load(&Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/llama-gqa")).unwrap()
}

fn logits(m: &Model, pool: &Pool, tokens: &[u32]) -> Vec<f32> {
    let c = &m.cfg;
    let blocks: Vec<u32> = (0..tokens.len().div_ceil(16) as u32).collect();
    let mut cache = KvCache::new(c.layers, c.kv_heads, c.head_dim, 16, blocks.len());
    m.forward(
        pool,
        &mut cache,
        &[Chunk {
            tokens,
            pos: 0,
            blocks: &blocks,
            logits: Logits::All,
        }],
    )
}

fn rel_error(a: &[f32], b: &[f32]) -> f64 {
    let num: f64 = a.iter().zip(b).map(|(x, y)| f64::from(x - y).powi(2)).sum();
    let den: f64 = b.iter().map(|y| f64::from(*y).powi(2)).sum();
    (num / den).sqrt()
}

fn tokens(n: usize, seed: u64) -> Vec<u32> {
    let mut rng = Rng::new(seed);
    (0..n).map(|_| rng.between(3, 511) as u32).collect()
}

#[test]
fn int8_is_close_to_bf16_and_int4_further() {
    let pool = Pool::new(4);
    let t = tokens(96, 1);
    let reference = logits(&model(), &pool, &t);
    let mut errs = Vec::new();
    for kind in [Quant::Int8, Quant::Int4] {
        let mut m = model();
        m.quantize(kind, None);
        assert_eq!(m.quant(), kind);
        errs.push(rel_error(&logits(&m, &pool, &t), &reference));
    }
    println!("relative logit error: int8 {:.4}, int4 {:.4}", errs[0], errs[1]);
    // The fixture's weights are random, which quantizes far worse than
    // trained weights (on SmolLM2-360M int8 leaves perplexity unchanged).
    assert!(errs[0] < 0.06, "int8 error {}", errs[0]);
    assert!(errs[1] > 4.0 * errs[0], "int4 error {}", errs[1]);
}

#[test]
fn awq_folds_scales_without_changing_the_function_and_reduces_error() {
    let pool = Pool::new(4);
    let calib: Vec<Vec<u32>> = (0..8).map(|i| tokens(64, 10 + i)).collect();
    let held_out = tokens(96, 99);
    let base = model();
    let reference = logits(&base, &pool, &held_out);
    let acts = base.calibrate(&pool, &calib, 256);
    assert_eq!(acts.len(), base.cfg.layers);
    assert!(acts.iter().all(|l| l.iter().all(|rows| rows.len() == 256)));

    let mut plain = model();
    plain.quantize(Quant::Int4, None);
    let mut awq = model();
    let choices = awq.quantize_awq(Quant::Int4, &acts);
    // Per projection, the chosen scaling is never worse than none on the
    // calibration sample (α = 0 is one of the candidates).
    assert!(choices.iter().all(|c| c.error_awq <= c.error_plain));
    let e_plain = rel_error(&logits(&plain, &pool, &held_out), &reference);
    let e_awq = rel_error(&logits(&awq, &pool, &held_out), &reference);
    println!("held-out relative logit error: int4 {e_plain:.4}, int4+AWQ {e_awq:.4}");
    assert!(e_awq < e_plain, "AWQ did not help on held-out text");
}
