use ferrolm::kernels::{Matrix, matmul};
use ferrolm::pool::Pool;
use ferrolm::quant::Quant;
use std::time::Instant;
fn main() {
    let pool = Pool::with_all_cores();
    let (n, k) = (5120usize, 960usize);
    let w = Matrix::new(n, k, (0..n * k).map(|i| ((i * 7919) % 1000) as u16 | 0x3c00).collect());
    for kind in [Quant::Bf16, Quant::Int8, Quant::Int4] {
        let q = w.quantized(kind, None);
        for m in [1usize, 16, 256] {
            let x: Vec<f32> = (0..m * k).map(|i| (i % 13) as f32 * 0.01).collect();
            let mut y = vec![0f32; m * n];
            matmul(&pool, &x, m, &q, None, &mut y);
            let reps = (4e9 / (2.0 * (m * k * n) as f64)).max(20.0) as usize;
            let t = Instant::now();
            for _ in 0..reps {
                matmul(&pool, &x, m, &q, None, &mut y);
            }
            let dt = t.elapsed().as_secs_f64() / reps as f64;
            println!(
                "{} {:>4} m={m:3}: {:7.3} ms  {:5.0} GFLOPS  {:4.1} MB",
                ferrolm::kernels::backend(),
                kind.name(),
                dt * 1e3,
                2.0 * (m * k * n) as f64 / dt / 1e9,
                q.bytes() as f64 / 1e6
            );
        }
    }
}
