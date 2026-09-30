//! Forward-pass speed on a random model with SmolLM2-360M's shape.
use ferrolm::config::{Config, RopeScaling};
use ferrolm::kv::KvCache;
use ferrolm::model::{Chunk, Logits, Model};
use ferrolm::pool::Pool;
use std::time::Instant;

fn main() {
    let cfg = Config {
        arch: "llama".into(), vocab: 49152, hidden: 960, intermediate: 2560, layers: 32, heads: 15,
        kv_heads: 5, head_dim: 64, rms_eps: 1e-5, rope_theta: 100000.0, rope_scaling: RopeScaling::None,
        max_position: 8192, tie_embeddings: true, qkv_bias: false, eos: vec![0], bos: None,
    };
    let m = Model::random(cfg.clone(), 1);
    let pool = Pool::with_all_cores();
    println!("backend {} threads {}", ferrolm::kernels::backend(), pool.threads());
    let bs = 16;
    let mut cache = KvCache::new(cfg.layers, cfg.kv_heads, cfg.head_dim, bs, 2048);
    // prefill 256 tokens
    let prompt: Vec<u32> = (0..256).map(|i| (i * 31 % 49152) as u32).collect();
    let tables: Vec<Vec<u32>> = (0..64).map(|s| (s * 32..s * 32 + 32).map(|b| b as u32).collect()).collect();
    let t = Instant::now();
    m.forward(&pool, &mut cache, &[Chunk { tokens: &prompt, pos: 0, blocks: &tables[0], logits: Logits::Last }]);
    let dt = t.elapsed().as_secs_f64();
    println!("prefill 256: {:.2}s = {:.0} tok/s", dt, 256.0 / dt);
    for batch in [1usize, 4, 16, 32, 64] {
        let toks = vec![5u32; batch];
        let chunks: Vec<Chunk> = (0..batch).map(|s| Chunk { tokens: &toks[s..s + 1], pos: 100, blocks: &tables[s], logits: Logits::Last }).collect();
        m.forward(&pool, &mut cache, &chunks);
        let n = 5;
        let t = Instant::now();
        for _ in 0..n { m.forward(&pool, &mut cache, &chunks); }
        let dt = t.elapsed().as_secs_f64() / n as f64;
        println!("decode batch {batch:3}: {:.1} ms/step = {:.0} tok/s", dt * 1e3, batch as f64 / dt);
    }
}
