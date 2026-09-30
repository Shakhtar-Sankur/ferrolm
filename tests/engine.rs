//! Serving must not change what a request generates. Under greedy decoding,
//! every request's tokens must be identical whether it runs alone or in a
//! batch, whatever its prompt is split into, whether its prefix came from
//! the cache, whether it was preempted and recomputed, and whether it was
//! decoded speculatively. Sampling with a draft model must keep the target
//! model's distribution.

use ferrolm::engine::{Admission, Engine, EngineConfig, Finish, Handle, collect};
use ferrolm::model::Model;
use ferrolm::pool::Pool;
use ferrolm::rng::Rng;
use ferrolm::sampler::SamplingParams;
use std::path::Path;

fn model() -> Model {
    Model::load(&Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/llama-gqa")).unwrap()
}

fn prompts(n: usize, seed: u64) -> Vec<Vec<u32>> {
    let mut rng = Rng::new(seed);
    (0..n)
        .map(|_| {
            let len = rng.between(1, 150) as usize;
            (0..len).map(|_| rng.between(3, 511) as u32).collect()
        })
        .collect()
}

fn greedy(max_tokens: usize) -> SamplingParams {
    SamplingParams {
        max_tokens,
        ignore_eos: true,
        ..Default::default()
    }
}

/// Submits everything, serves until idle, and returns each request's
/// tokens along with the engine's statistics.
fn serve(
    cfg: EngineConfig,
    draft: Option<Model>,
    reqs: &[(Vec<u32>, SamplingParams)],
) -> (Vec<(Vec<u32>, Finish)>, ferrolm::engine::Stats) {
    let (mut e, h): (Engine, Handle) = Engine::new(model(), draft, Pool::new(4), cfg);
    let rx: Vec<_> = reqs.iter().map(|(p, s)| h.submit(p.clone(), s.clone()).0).collect();
    e.run_until_idle();
    let out = rx.iter().map(collect).collect();
    let stats = h.stats.lock().unwrap().clone();
    (out, stats)
}

fn alone(reqs: &[(Vec<u32>, SamplingParams)]) -> Vec<Vec<u32>> {
    reqs.iter()
        .map(|r| {
            let cfg = EngineConfig {
                max_seqs: 1,
                prefix_cache: false,
                ..Default::default()
            };
            serve(cfg, None, std::slice::from_ref(r)).0.remove(0).0
        })
        .collect()
}

#[test]
fn batching_and_chunked_prefill_do_not_change_outputs() {
    let reqs: Vec<_> = prompts(24, 1).into_iter().map(|p| (p, greedy(24))).collect();
    let want = alone(&reqs);
    // A small token budget splits prompts into chunks and mixes prefill
    // with decoding in the same step.
    for (budget, admission) in [
        (37, Admission::Continuous),
        (512, Admission::Continuous),
        (64, Admission::Static),
    ] {
        let cfg = EngineConfig {
            max_batch_tokens: budget,
            admission,
            prefix_cache: false,
            ..Default::default()
        };
        let (got, st) = serve(cfg, None, &reqs);
        for (i, (g, f)) in got.iter().enumerate() {
            assert_eq!(f, &Finish::Length);
            assert_eq!(g, &want[i], "request {i}, budget {budget}, {admission:?}");
        }
        assert!(st.peak_running > 1, "requests never overlapped");
    }
}

#[test]
fn prefix_cache_hits_do_not_change_outputs() {
    let mut rng = Rng::new(9);
    let system: Vec<u32> = (0..100).map(|_| rng.between(3, 511) as u32).collect();
    let reqs: Vec<_> = prompts(16, 2)
        .into_iter()
        .map(|p| ([system.clone(), p].concat(), greedy(16)))
        .collect();
    let want = alone(&reqs);
    let cfg = EngineConfig {
        max_seqs: 4,
        ..Default::default()
    };
    let (got, st) = serve(cfg, None, &reqs);
    for (i, (g, _)) in got.iter().enumerate() {
        assert_eq!(g, &want[i], "request {i}");
    }
    // The first four requests are admitted together, before the shared
    // prefix is cached; each later one reuses its 6 full blocks.
    assert!(
        st.cached_tokens >= 12 * 96,
        "only {} tokens from the cache",
        st.cached_tokens
    );
}

#[test]
fn preemption_and_recompute_do_not_change_outputs() {
    let reqs: Vec<_> = prompts(20, 3).into_iter().map(|p| (p, greedy(40))).collect();
    let want = alone(&reqs);
    // 40 blocks of 16 tokens cannot hold 20 growing sequences.
    let cfg = EngineConfig {
        kv_blocks: 40,
        max_batch_tokens: 128,
        ..Default::default()
    };
    let (got, st) = serve(cfg, None, &reqs);
    for (i, (g, _)) in got.iter().enumerate() {
        assert_eq!(g, &want[i], "request {i}");
    }
    assert!(st.preemptions > 0, "the test never preempted");
}

#[test]
fn speculative_greedy_decoding_matches_the_target_exactly() {
    let reqs: Vec<_> = prompts(12, 4).into_iter().map(|p| (p, greedy(30))).collect();
    let want = alone(&reqs);
    // A related draft (the target's first layer) and a perfect one (the
    // target itself), with different proposal lengths.
    for (layers, k) in [(1, 4), (3, 3), (1, 1)] {
        let cfg = EngineConfig {
            spec_k: k,
            kv_blocks: 200,
            ..Default::default()
        };
        let (got, st) = serve(cfg, Some(model().truncated(layers)), &reqs);
        for (i, (g, _)) in got.iter().enumerate() {
            assert_eq!(g, &want[i], "request {i}, draft of {layers} layers, k={k}");
        }
        let rate = st.spec_accepted as f64 / st.spec_proposed as f64;
        println!(
            "draft layers {layers}, k={k}: {:.0}% of proposals accepted",
            rate * 100.0
        );
        assert!(st.spec_proposed > 0);
        if layers == 3 {
            assert_eq!(st.spec_accepted, st.spec_proposed, "the target as its own draft");
        }
    }
}

/// The exact probability of each two-token continuation of `prompt`
/// under the target model with these sampling settings.
fn exact_two_token_distribution(prompt: &[u32], params: &SamplingParams) -> Vec<(Vec<u32>, f64)> {
    use ferrolm::kv::KvCache;
    use ferrolm::model::{Chunk, Logits};
    let m = model();
    let pool = Pool::new(4);
    let blocks: Vec<u32> = (0..16).collect();
    let next = |tokens: &[u32]| {
        let c = &m.cfg;
        let mut cache = KvCache::new(c.layers, c.kv_heads, c.head_dim, 16, 16);
        let l = m.forward(
            &pool,
            &mut cache,
            &[Chunk {
                tokens,
                pos: 0,
                blocks: &blocks,
                logits: Logits::Last,
            }],
        );
        ferrolm::sampler::distribution(&l, params)
    };
    let mut out = Vec::new();
    for (t1, p1) in next(prompt) {
        let mut ext = prompt.to_vec();
        ext.push(t1);
        for (t2, p2) in next(&ext) {
            out.push((vec![t1, t2], f64::from(p1) * f64::from(p2)));
        }
    }
    out
}

#[test]
fn speculative_sampling_keeps_the_target_distribution() {
    // Sample the first two tokens after a fixed prompt many times, with and
    // without a draft model, and compare each histogram with the exact
    // distribution by a chi-square test.
    let prompt = prompts(1, 5).remove(0);
    let params = |seed| SamplingParams {
        temperature: 1.0,
        top_k: 4,
        seed,
        max_tokens: 2,
        ignore_eos: true,
        ..Default::default()
    };
    let exact = exact_two_token_distribution(&prompt, &params(0));
    let n = 4000;
    let reqs: Vec<_> = (0..n).map(|i| (prompt.clone(), params(i as u64 + 1))).collect();
    let chi2 = |out: &[(Vec<u32>, Finish)]| {
        let mut stat = 0.0;
        for (seq, p) in &exact {
            let observed = out.iter().filter(|(t, _)| t == seq).count() as f64;
            let expected = p * n as f64;
            stat += (observed - expected).powi(2) / expected;
        }
        assert!(
            out.iter().all(|(t, _)| exact.iter().any(|(s, _)| s == t)),
            "impossible continuation sampled"
        );
        stat
    };
    let cfg = EngineConfig {
        spec_k: 3,
        kv_blocks: 600,
        ..Default::default()
    };
    let plain = chi2(&serve(cfg.clone(), None, &reqs).0);
    let (spec, st) = serve(cfg, Some(model().truncated(1)), &reqs);
    let spec = chi2(&spec);
    // 16 outcomes: 15 degrees of freedom; 37.7 is the 0.1% critical value.
    println!(
        "chi-square vs exact: plain {plain:.1}, speculative {spec:.1} (df 15); {}/{} proposals accepted",
        st.spec_accepted, st.spec_proposed
    );
    assert_eq!(exact.len(), 16);
    assert!(st.spec_accepted > 0 && st.spec_accepted < st.spec_proposed);
    assert!(plain < 37.7, "plain sampling is off: {plain}");
    assert!(spec < 37.7, "speculative sampling changed the distribution: {spec}");
}

#[test]
fn reserving_whole_sequences_admits_fewer_at_once() {
    let reqs: Vec<_> = prompts(16, 6).into_iter().map(|p| (p, greedy(8))).collect();
    let paged = EngineConfig {
        kv_blocks: 100,
        ..Default::default()
    };
    let reserved = EngineConfig {
        reserve_tokens: Some(512),
        ..paged.clone()
    };
    let (a, sa) = serve(paged, None, &reqs);
    let (b, sb) = serve(reserved, None, &reqs);
    assert_eq!(a, b);
    // 100 blocks reserve three 512-token sequences; paging fits many more.
    assert_eq!(sb.peak_running, 3);
    assert!(sa.peak_running > 6, "paged peak {}", sa.peak_running);
}

#[test]
fn rejects_what_cannot_be_served_and_honours_cancellation() {
    let cfg = EngineConfig {
        kv_blocks: 8,
        ..Default::default()
    };
    let (mut e, h) = Engine::new(model(), None, Pool::new(2), cfg);
    let (too_long, _) = h.submit(vec![5; 200], greedy(4));
    let (empty, _) = h.submit(vec![], greedy(4));
    let (bad, _) = h.submit(vec![600], greedy(4));
    let (cancelled, c) = h.submit(vec![5; 10], greedy(1000));
    c.store(true, std::sync::atomic::Ordering::Relaxed);
    let (ok, _) = h.submit(vec![5; 10], greedy(1000));
    e.run_until_idle();
    assert!(matches!(collect(&too_long).1, Finish::Rejected(_)));
    assert!(matches!(collect(&empty).1, Finish::Rejected(_)));
    assert!(matches!(collect(&bad).1, Finish::Rejected(_)));
    assert_eq!(collect(&cancelled).1, Finish::Cancelled);
    // The context is full at 8 blocks of 16 tokens.
    let (t, f) = collect(&ok);
    assert_eq!((t.len(), f), (118, Finish::Length));
}
