//! The approximate indexes against exact search on clustered random data.

use ferrolm::pool::Pool;
use ferrolm::rng::Rng;
use ferrolm::vector::flat::FlatIndex;
use ferrolm::vector::hnsw::{Hnsw, HnswParams};
use ferrolm::vector::ivfpq::{IvfPq, IvfPqParams};
use ferrolm::vector::{Metric, normalize, recall};

/// `n` vectors around 50 random centres, as real embeddings cluster.
fn data(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    // The same 50 centres for every call; `seed` picks the points.
    let mut crng = Rng::new(12345 + dim as u64);
    let centres: Vec<f32> = (0..50 * dim).map(|_| crng.normal() as f32 * 3.0).collect();
    let mut rng = Rng::new(seed);
    (0..n)
        .flat_map(|_| {
            let c = rng.below(50) as usize;
            (0..dim)
                .map(|d| centres[c * dim + d] + rng.normal() as f32)
                .collect::<Vec<_>>()
        })
        .collect()
}

fn truth(pool: &Pool, metric: Metric, dim: usize, base: &[f32], queries: &[f32], k: usize) -> Vec<Vec<u32>> {
    let mut flat = FlatIndex::new(metric, dim);
    flat.add(base);
    flat.search_batch(pool, queries, k)
        .iter()
        .map(|r| r.iter().map(|n| n.id).collect())
        .collect()
}

#[test]
fn hnsw_finds_the_nearest_neighbours_and_round_trips() {
    let pool = Pool::new(4);
    for (metric, dim) in [(Metric::L2, 40), (Metric::InnerProduct, 64)] {
        let mut base = data(20_000, dim, 1);
        let mut queries = data(200, dim, 2);
        if metric == Metric::InnerProduct {
            normalize(&mut base, dim);
            normalize(&mut queries, dim);
        }
        let t = truth(&pool, metric, dim, &base, &queries, 10);
        let h = Hnsw::build(
            &pool,
            metric,
            dim,
            &base,
            HnswParams {
                m: 16,
                ef_construction: 100,
                seed: 1,
            },
        );
        assert!(h.mean_degree() > 8.0, "graph too sparse: {}", h.mean_degree());
        let r_low = recall(&h.search_batch(&pool, &queries, 10, 16), &t, 10);
        let r_high = recall(&h.search_batch(&pool, &queries, 10, 128), &t, 10);
        println!("{metric:?}: HNSW recall@10 {r_low:.3} (ef 16), {r_high:.3} (ef 128)");
        assert!(r_high >= 0.98, "{metric:?}: recall {r_high}");
        assert!(r_high >= r_low);
        let loaded = Hnsw::load(&h.save()).unwrap();
        assert_eq!(
            loaded.search_batch(&pool, &queries, 10, 64),
            h.search_batch(&pool, &queries, 10, 64)
        );
    }
}

#[test]
fn ivfpq_recall_grows_with_probes_and_reranking() {
    let pool = Pool::new(4);
    let dim = 32;
    let base = data(20_000, dim, 3);
    let queries = data(200, dim, 4);
    let t = truth(&pool, Metric::L2, dim, &base, &queries, 10);
    let p = IvfPqParams {
        nlist: 64,
        m: 8,
        iters: 10,
        keep_vectors: true,
        seed: 1,
    };
    let mut ix = IvfPq::train(&pool, Metric::L2, dim, &base[..10_000 * dim], p);
    ix.add(&pool, &base);
    assert_eq!(ix.len(), 20_000);
    let r1 = recall(&ix.search_batch(&pool, &queries, 10, 1, 0), &t, 10);
    let r16 = recall(&ix.search_batch(&pool, &queries, 10, 16, 0), &t, 10);
    let r16r = recall(&ix.search_batch(&pool, &queries, 10, 16, 10), &t, 10);
    println!("IVF-PQ recall@10: nprobe 1 {r1:.3}, nprobe 16 {r16:.3}, nprobe 16 + rerank {r16r:.3}");
    assert!(r16 > r1 && r16r > r16 && r16r > 0.9, "{r1} {r16} {r16r}");
    let loaded = IvfPq::load(&ix.save()).unwrap();
    assert_eq!(
        loaded.search_batch(&pool, &queries, 10, 8, 5),
        ix.search_batch(&pool, &queries, 10, 8, 5)
    );
}
