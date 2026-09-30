//! Turning logits into the next token: greedy, or temperature sampling
//! with top-k and nucleus (top-p) filtering from a per-request seeded
//! generator.

use crate::rng::Rng;

#[derive(Clone, Debug)]
pub struct SamplingParams {
    /// 0 means greedy.
    pub temperature: f32,
    pub top_p: f32,
    /// 0 means no limit.
    pub top_k: usize,
    pub seed: u64,
    pub max_tokens: usize,
    /// Token ids that end the reply (in addition to the model's EOS).
    pub stop_ids: Vec<u32>,
    pub ignore_eos: bool,
}

impl Default for SamplingParams {
    fn default() -> Self {
        SamplingParams {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            seed: 0,
            max_tokens: 64,
            stop_ids: Vec::new(),
            ignore_eos: false,
        }
    }
}

impl SamplingParams {
    pub fn greedy(&self) -> bool {
        self.temperature <= 0.0
    }
}

/// The index of the largest value (the first, on ties).
pub fn argmax(v: &[f32]) -> u32 {
    let mut best = 0;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best as u32
}

/// The distribution sampling draws from, as (token, probability) pairs
/// with probability > 0, after temperature, top-k and top-p.
pub fn distribution(logits: &[f32], p: &SamplingParams) -> Vec<(u32, f32)> {
    let t = p.temperature.max(1e-5);
    let mut v: Vec<(u32, f32)> = logits.iter().enumerate().map(|(i, &l)| (i as u32, l / t)).collect();
    v.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    if p.top_k > 0 {
        v.truncate(p.top_k);
    }
    let max = v[0].1;
    let mut sum = 0f64;
    for e in &mut v {
        e.1 = (e.1 - max).exp();
        sum += f64::from(e.1);
    }
    for e in &mut v {
        e.1 = (f64::from(e.1) / sum) as f32;
    }
    if p.top_p < 1.0 {
        let mut acc = 0f32;
        let keep = v
            .iter()
            .position(|e| {
                acc += e.1;
                acc >= p.top_p
            })
            .map_or(v.len(), |i| i + 1);
        v.truncate(keep);
        let s: f32 = v.iter().map(|e| e.1).sum();
        for e in &mut v {
            e.1 /= s;
        }
    }
    v
}

/// Draws from `(token, probability)` pairs.
pub fn draw(dist: &[(u32, f32)], rng: &mut Rng) -> u32 {
    let mut u = rng.uniform() as f32 * dist.iter().map(|e| e.1).sum::<f32>();
    for &(t, pr) in dist {
        if u < pr {
            return t;
        }
        u -= pr;
    }
    dist.last().map_or(0, |e| e.0)
}

pub fn sample(logits: &[f32], p: &SamplingParams, rng: &mut Rng) -> u32 {
    if p.greedy() {
        argmax(logits)
    } else {
        draw(&distribution(logits, p), rng)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filters_and_samples_in_proportion() {
        let logits = [2.0f32, 1.0, 0.0, -1.0];
        let p = SamplingParams {
            temperature: 1.0,
            top_k: 3,
            top_p: 0.95,
            ..Default::default()
        };
        let d = distribution(&logits, &p);
        assert_eq!(d.iter().map(|e| e.0).collect::<Vec<_>>(), [0, 1, 2]);
        let mut rng = Rng::new(3);
        let mut counts = [0usize; 4];
        for _ in 0..20000 {
            counts[draw(&d, &mut rng) as usize] += 1;
        }
        for (i, &(_, pr)) in d.iter().enumerate() {
            let f = counts[i] as f32 / 20000.0;
            assert!((f - pr).abs() < 0.015, "token {i}: {f} vs {pr}");
        }
        assert_eq!(counts[3], 0);
        assert_eq!(argmax(&[1.0, 3.0, 3.0]), 1);
    }
}
