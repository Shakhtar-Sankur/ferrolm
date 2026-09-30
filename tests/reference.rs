//! ferrolm against Hugging Face transformers on the fixture models (see
//! scripts/make_fixtures.py): the same logits, and the same greedy text.

use ferrolm::json::{self, Json};
use ferrolm::kv::KvCache;
use ferrolm::model::{Chunk, Logits, Model};
use ferrolm::pool::Pool;
use std::path::Path;

fn load(dir: &Path, reference: &str) -> (Model, Json) {
    let model = Model::load(dir).unwrap();
    let r = json::parse(&std::fs::read_to_string(dir.join(reference)).unwrap()).unwrap();
    (model, r)
}

/// Downloaded models to check too (comma-separated directories prepared
/// by scripts/reference_real.py).
pub fn real_models() -> Vec<std::path::PathBuf> {
    std::env::var("FERROLM_REAL_MODELS")
        .map(|v| v.split(',').filter(|s| !s.is_empty()).map(Into::into).collect())
        .unwrap_or_default()
}

fn ids(v: &Json) -> Vec<u32> {
    v.as_arr().iter().map(|x| x.as_usize().unwrap() as u32).collect()
}

fn cache_for(m: &Model, tokens: usize) -> (KvCache, Vec<u32>) {
    let bs = 16;
    let blocks = tokens.div_ceil(bs);
    let c = &m.cfg;
    // Deliberately scattered block ids, as a real block table would be.
    let table = (0..blocks as u32).map(|i| (i * 7) % blocks as u32).collect::<Vec<_>>();
    let mut t = table.clone();
    t.sort();
    t.dedup();
    assert_eq!(t.len(), blocks, "block ids must be distinct");
    (KvCache::new(c.layers, c.kv_heads, c.head_dim, bs, blocks), table)
}

fn argmax(v: &[f32]) -> u32 {
    let mut best = 0;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best as u32
}

fn check(name: &str) {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(name);
    check_dir(&dir, "reference.json", 1e-3);
}

/// Logits within `tol` of transformers' (relative to the largest logit),
/// and the same greedy continuation.
fn check_dir(dir: &Path, reference: &str, tol: f64) {
    let name = dir.display().to_string();
    let pool = Pool::new(4);
    let (m, r) = load(dir, reference);
    let prompt = ids(r.get("prompt").unwrap());
    let greedy = ids(r.get("greedy").unwrap());
    let vocab = m.cfg.vocab;
    let (mut cache, table) = cache_for(&m, prompt.len() + greedy.len() + 1);

    let logits = m.forward(
        &pool,
        &mut cache,
        &[Chunk {
            tokens: &prompt,
            pos: 0,
            blocks: &table,
            logits: Logits::All,
        }],
    );
    let rows = ids(r.get("logits_rows").unwrap());
    let (mut worst, mut scale) = (0f64, 1f64);
    for (row, expect) in rows.iter().zip(r.get("logits").unwrap().as_arr()) {
        let got = &logits[*row as usize * vocab..][..vocab];
        for (g, e) in got.iter().zip(expect.as_arr()) {
            let e = e.as_f64().unwrap();
            worst = worst.max((f64::from(*g) - e).abs());
            scale = scale.max(e.abs());
        }
    }
    println!("{name}: max |logit - transformers| = {worst:.2e} (largest logit {scale:.1})");
    assert!(
        worst < tol * scale,
        "{name}: logits differ from transformers by {worst}"
    );

    // Greedy decoding, one token per step, must reproduce transformers'
    // generate() exactly.
    let mut next = argmax(&logits[(prompt.len() - 1) * vocab..]);
    let mut out = vec![next];
    let mut pos = prompt.len();
    while out.len() < greedy.len() {
        let l = m.forward(
            &pool,
            &mut cache,
            &[Chunk {
                tokens: &[next],
                pos,
                blocks: &table,
                logits: Logits::Last,
            }],
        );
        next = argmax(&l);
        out.push(next);
        pos += 1;
    }
    assert_eq!(out, greedy, "{name}: greedy continuation differs from transformers");
}

#[test]
fn llama_with_grouped_query_attention_matches_transformers() {
    check("llama-gqa");
}

#[test]
fn llama3_with_tied_embeddings_and_rope_scaling_matches_transformers() {
    check("llama3-tied");
}

#[test]
fn qwen2_with_qkv_bias_matches_transformers() {
    check("qwen2");
}

#[test]
fn downloaded_models_match_transformers() {
    for dir in real_models() {
        check_dir(&dir, "ferrolm-reference.json", 1e-3);
    }
}
