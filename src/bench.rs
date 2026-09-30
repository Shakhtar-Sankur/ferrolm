//! A load generator for the engine: requests with random prompts arrive
//! all at once or as a Poisson process, and every token's arrival time is
//! recorded, giving throughput, time to first token (TTFT), time per output
//! token (TPOT) and end-to-end latency.

use crate::engine::{Engine, Event, Finish, Handle, Stats};
use crate::rng::Rng;
use crate::sampler::SamplingParams;
use std::time::{Duration, Instant};

#[derive(Clone, Debug)]
pub struct Workload {
    pub requests: usize,
    /// Requests per second (Poisson arrivals); `None` sends all at once.
    pub rate: Option<f64>,
    pub prompt_len: (usize, usize),
    pub gen_len: (usize, usize),
    /// Tokens every prompt starts with (a shared system prompt).
    pub shared_prefix: usize,
    pub seed: u64,
    pub temperature: f32,
    /// Real prompts, used in turn instead of random tokens; replies then
    /// end at the model's end-of-turn token or `gen_len`.
    pub prompts: Vec<Vec<u32>>,
    /// Tokens that end a reply (the chat template's end-of-turn).
    pub stop_ids: Vec<u32>,
}

#[derive(Clone, Debug)]
struct Record {
    arrival: f64,
    first: Option<f64>,
    done: f64,
    tokens: usize,
    prompt: usize,
    finish: Finish,
}

#[derive(Clone, Debug, Default)]
pub struct Summary {
    pub requests: usize,
    pub duration: f64,
    pub prompt_tokens: usize,
    pub output_tokens: usize,
    pub output_tps: f64,
    pub total_tps: f64,
    pub req_per_s: f64,
    pub ttft_p50: f64,
    pub ttft_p90: f64,
    pub ttft_p99: f64,
    pub tpot_p50: f64,
    pub tpot_p99: f64,
    pub e2e_p50: f64,
    pub e2e_p99: f64,
    pub failed: usize,
    pub engine: Stats,
}

fn pct(v: &mut [f64], p: f64) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(f64::total_cmp);
    let i = ((p / 100.0) * (v.len() - 1) as f64).round() as usize;
    v[i]
}

/// Runs the workload against `engine` and returns the measurements.
pub fn run(mut engine: Engine, handle: Handle, w: &Workload) -> Summary {
    let vocab = engine.model().cfg.vocab as u64;
    let mut rng = Rng::new(w.seed);
    let tok = |rng: &mut Rng| rng.between(100.min(vocab - 1), vocab - 1) as u32;
    let prefix: Vec<u32> = (0..w.shared_prefix).map(|_| tok(&mut rng)).collect();
    let mut t = 0.0;
    let plan: Vec<(f64, Vec<u32>, usize)> = (0..w.requests)
        .map(|i| {
            if let Some(r) = w.rate {
                t += rng.exponential(1.0 / r);
            }
            let n = rng.between(w.prompt_len.0 as u64, w.prompt_len.1 as u64) as usize;
            let mut p = prefix.clone();
            if w.prompts.is_empty() {
                p.extend((0..n).map(|_| tok(&mut rng)));
            } else {
                p.extend_from_slice(&w.prompts[i % w.prompts.len()]);
            }
            let g = rng.between(w.gen_len.0 as u64, w.gen_len.1 as u64) as usize;
            (t, p, g)
        })
        .collect();

    let engine_thread = std::thread::spawn(move || engine.run());
    let start = Instant::now();
    let collectors: Vec<_> = plan
        .into_iter()
        .enumerate()
        .map(|(i, (at, prompt, gen_len))| {
            let wait = Duration::from_secs_f64(at).saturating_sub(start.elapsed());
            std::thread::sleep(wait);
            let params = SamplingParams {
                temperature: w.temperature,
                seed: w.seed.wrapping_add(i as u64 + 1),
                max_tokens: gen_len,
                ignore_eos: w.prompts.is_empty(),
                stop_ids: if w.prompts.is_empty() {
                    Vec::new()
                } else {
                    w.stop_ids.clone()
                },
                ..Default::default()
            };
            let arrival = start.elapsed().as_secs_f64();
            let prompt_len = prompt.len();
            let (rx, _cancel) = handle.submit(prompt, params);
            std::thread::spawn(move || {
                let mut r = Record {
                    arrival,
                    first: None,
                    done: 0.0,
                    tokens: 0,
                    prompt: prompt_len,
                    finish: Finish::Cancelled,
                };
                loop {
                    match rx.recv() {
                        Ok(Event::Token(_)) => {
                            r.tokens += 1;
                            r.first.get_or_insert(start.elapsed().as_secs_f64());
                        }
                        Ok(Event::Done(f)) => {
                            r.finish = f;
                            break;
                        }
                        Err(_) => break,
                    }
                }
                r.done = start.elapsed().as_secs_f64();
                r
            })
        })
        .collect();
    let records: Vec<Record> = collectors.into_iter().map(|c| c.join().unwrap()).collect();
    let engine_stats = handle.stats.lock().unwrap().clone();
    drop(handle);
    engine_thread.join().ok();
    summarize(&records, engine_stats)
}

fn summarize(rs: &[Record], engine: Stats) -> Summary {
    let ok: Vec<&Record> = rs
        .iter()
        .filter(|r| matches!(r.finish, Finish::Length | Finish::Stop))
        .collect();
    let first_arrival = rs.iter().map(|r| r.arrival).fold(f64::INFINITY, f64::min);
    let duration = rs.iter().map(|r| r.done).fold(0.0, f64::max) - first_arrival;
    let output: usize = ok.iter().map(|r| r.tokens).sum();
    let prompt: usize = ok.iter().map(|r| r.prompt).sum();
    let mut ttft: Vec<f64> = ok.iter().filter_map(|r| r.first.map(|f| f - r.arrival)).collect();
    let mut tpot: Vec<f64> = ok
        .iter()
        .filter(|r| r.tokens > 1)
        .map(|r| (r.done - r.first.unwrap()) / (r.tokens - 1) as f64)
        .collect();
    let mut e2e: Vec<f64> = ok.iter().map(|r| r.done - r.arrival).collect();
    Summary {
        requests: rs.len(),
        duration,
        prompt_tokens: prompt,
        output_tokens: output,
        output_tps: output as f64 / duration,
        total_tps: (output + prompt) as f64 / duration,
        req_per_s: ok.len() as f64 / duration,
        ttft_p50: pct(&mut ttft, 50.0),
        ttft_p90: pct(&mut ttft, 90.0),
        ttft_p99: pct(&mut ttft, 99.0),
        tpot_p50: pct(&mut tpot, 50.0),
        tpot_p99: pct(&mut tpot, 99.0),
        e2e_p50: pct(&mut e2e, 50.0),
        e2e_p99: pct(&mut e2e, 99.0),
        failed: rs.len() - ok.len(),
        engine,
    }
}

impl Summary {
    pub fn print(&self, label: &str) {
        let e = &self.engine;
        println!("== {label}");
        println!(
            "  {} requests in {:.1} s: {:.1} output tok/s, {:.1} total tok/s, {:.2} req/s{}",
            self.requests,
            self.duration,
            self.output_tps,
            self.total_tps,
            self.req_per_s,
            if self.failed > 0 {
                format!(", {} FAILED", self.failed)
            } else {
                String::new()
            }
        );
        println!(
            "  TTFT p50 {:.2} s  p90 {:.2} s  p99 {:.2} s | TPOT p50 {:.0} ms  p99 {:.0} ms | E2E p50 {:.1} s  p99 {:.1} s",
            self.ttft_p50,
            self.ttft_p90,
            self.ttft_p99,
            self.tpot_p50 * 1e3,
            self.tpot_p99 * 1e3,
            self.e2e_p50,
            self.e2e_p99
        );
        let mut extra = format!(
            "  engine: {} steps, peak batch {} sequences, {} preemptions, {} prompt tokens from prefix cache",
            e.steps, e.peak_running, e.preemptions, e.cached_tokens
        );
        if e.spec_proposed > 0 {
            extra.push_str(&format!(
                ", {:.0}% of {} draft tokens accepted",
                100.0 * e.spec_accepted as f64 / e.spec_proposed as f64,
                e.spec_proposed
            ));
        }
        println!("{extra}");
    }

    pub fn to_json(&self, label: &str) -> String {
        let e = &self.engine;
        format!(
            concat!(
                r#"{{"label":{},"requests":{},"duration_s":{:.3},"prompt_tokens":{},"output_tokens":{},"#,
                r#""output_tok_s":{:.2},"total_tok_s":{:.2},"req_s":{:.3},"ttft_p50_s":{:.4},"ttft_p90_s":{:.4},"#,
                r#""ttft_p99_s":{:.4},"tpot_p50_s":{:.4},"tpot_p99_s":{:.4},"e2e_p50_s":{:.3},"e2e_p99_s":{:.3},"#,
                r#""failed":{},"steps":{},"peak_running":{},"preemptions":{},"cached_tokens":{},"#,
                r#""spec_proposed":{},"spec_accepted":{}}}"#
            ),
            crate::json::quote(label),
            self.requests,
            self.duration,
            self.prompt_tokens,
            self.output_tokens,
            self.output_tps,
            self.total_tps,
            self.req_per_s,
            self.ttft_p50,
            self.ttft_p90,
            self.ttft_p99,
            self.tpot_p50,
            self.tpot_p99,
            self.e2e_p50,
            self.e2e_p99,
            self.failed,
            e.steps,
            e.peak_running,
            e.preemptions,
            e.cached_tokens,
            e.spec_proposed,
            e.spec_accepted
        )
    }
}
