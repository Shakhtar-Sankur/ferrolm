use ferrolm::kernels::{Matrix, matmul};
use ferrolm::pool::Pool;
use std::time::Instant;
fn main() {
    let pool = Pool::with_all_cores();
    for &(m, k, n) in &[
        (256usize, 960usize, 2560usize),
        (256, 2560, 960),
        (64, 960, 5120),
        (16, 960, 5120),
        (1, 960, 5120),
    ] {
        let w = Matrix::new(n, k, (0..n * k).map(|i| ((i * 7919) % 1000) as u16 | 0x3c00).collect());
        let x: Vec<f32> = (0..m * k).map(|i| (i % 13) as f32 * 0.01).collect();
        let mut y = vec![0f32; m * n];
        matmul(&pool, &x, m, &w, None, &mut y);
        let reps = (2e10 / (2.0 * (m * k * n) as f64)).max(3.0) as usize;
        let t = Instant::now();
        for _ in 0..reps {
            matmul(&pool, &x, m, &w, None, &mut y);
        }
        let dt = t.elapsed().as_secs_f64() / reps as f64;
        println!(
            "{} m={m:4} k={k} n={n}: {:.2} ms, {:.0} GFLOPS, {:.1} GB/s weights",
            ferrolm::kernels::backend(),
            dt * 1e3,
            2.0 * (m * k * n) as f64 / dt / 1e9,
            (n * k * 2) as f64 / dt / 1e9
        );
    }
}
