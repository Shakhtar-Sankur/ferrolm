use ferrolm::kernels::{attend, KvView};
use std::time::Instant;
fn main() {
    let (hd, group, bs) = (64usize, 3usize, 16usize);
    let row = 5 * hd;
    for &n in &[101usize, 1000] {
        let blocks: Vec<u32> = (0..(n.div_ceil(bs)) as u32).map(|b| b * 3).collect();
        let slots = 3 * blocks.len() * bs + bs;
        let k: Vec<f32> = (0..slots * row).map(|i| ((i % 17) as f32) * 0.01).collect();
        let v = k.clone();
        let q: Vec<f32> = (0..group * hd).map(|i| (i % 5) as f32 * 0.1).collect();
        let view = KvView { k: &k, v: &v, blocks: &blocks, bs, hd, block_stride: row * bs, head_off: hd * bs };
        let mut s = Vec::new();
        let mut out = vec![0f32; group * hd];
        let reps = 20000;
        let t = Instant::now();
        for _ in 0..reps { attend(&q, hd, n, 0.125, &view, &mut s, &mut out); }
        let dt = t.elapsed().as_secs_f64() / reps as f64;
        println!("n={n}: {:.2} us per task", dt * 1e6);
    }
}
