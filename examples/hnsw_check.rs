//! HNSW against exact search on anisotropic unit vectors (like text
//! embeddings, which share a common direction): reachability and recall.
use ferrolm::pool::Pool;
use ferrolm::rng::Rng;
use ferrolm::vector::flat::FlatIndex;
use ferrolm::vector::hnsw::{Hnsw, HnswParams};
use ferrolm::vector::{Metric, normalize};

fn main() {
    let (n, nq, d) = (5183, 300, 384);
    let mut rng = Rng::new(7);
    let common: Vec<f32> = (0..d).map(|_| rng.normal() as f32).collect();
    let centers: Vec<Vec<f32>> = (0..50).map(|_| (0..d).map(|_| rng.normal() as f32).collect()).collect();
    let sample = |rng: &mut Rng| -> Vec<f32> {
        let c = &centers[rng.below(50) as usize];
        (0..d)
            .map(|i| 3.0 * common[i] + c[i] + 0.7 * rng.normal() as f32)
            .collect::<Vec<f32>>()
    };
    let mut base: Vec<f32> = (0..n).flat_map(|_| sample(&mut rng)).collect();
    let mut qs: Vec<f32> = (0..nq).flat_map(|_| sample(&mut rng)).collect();
    normalize(&mut base, d);
    normalize(&mut qs, d);
    let pool = Pool::new(4);
    let mut flat = FlatIndex::new(Metric::InnerProduct, d);
    flat.add(&base);
    let truth = flat.search_batch(&pool, &qs, 10);
    for threads in [1, 4] {
        let p = Pool::new(threads);
        let h = Hnsw::build(
            &p,
            Metric::InnerProduct,
            d,
            &base,
            HnswParams {
                m: 16,
                ef_construction: 200,
                seed: 1,
            },
        );
        print!(
            "threads {threads}: reachable {}/{n}, degree {:.1};",
            h.reachable(),
            h.mean_degree()
        );
        for ef in [16, 64, 128, 512] {
            let got = h.search_batch(&pool, &qs, 10, ef);
            let mut hit = 0;
            for (g, t) in got.iter().zip(&truth) {
                hit += g.iter().filter(|x| t.iter().any(|y| y.id == x.id)).count();
            }
            print!(" ef {ef}: {:.4}", hit as f64 / (10 * nq) as f64);
        }
        println!();
    }
}
