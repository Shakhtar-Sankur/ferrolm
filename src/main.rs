use ferrolm::bench::{self, Workload};
use ferrolm::config::{Config, RopeScaling};
use ferrolm::engine::{Admission, Engine, EngineConfig, Event};
use ferrolm::model::Model;
use ferrolm::pool::Pool;
use ferrolm::quant::Quant;
use ferrolm::sampler::SamplingParams;
use ferrolm::server::{Detokenizer, Server};
use ferrolm::tokenizer::Tokenizer;
use std::io::Write;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

const USAGE: &str = "ferrolm: an LLM inference server in Rust

USAGE:
  ferrolm serve    --model DIR [--draft DIR] [--host 127.0.0.1] [--port 8000] [engine options]
  ferrolm generate --model DIR [--draft DIR] --prompt TEXT [--chat] [--max-tokens 128]
                   [--temperature 0] [--top-p 1] [--top-k 0] [--seed 1] [engine options]
  ferrolm bench    --model DIR|random:SHAPE [--draft DIR|random:SHAPE] [--requests 64]
                   [--rate REQ_PER_S] [--prompt-len 64..256] [--gen-len 32..128]
                   [--shared-prefix 0] [--prompt-file FILE] [--temperature 0] [--seed 1] [--label NAME]
                   [--json FILE] [engine options]
  ferrolm perplexity --model DIR --data FILE [--quant int8|int4] [--awq --calib FILE]
                   [--ctx 512] [--windows 40] [--json FILE]
  ferrolm info     --model DIR|random:SHAPE

ENGINE OPTIONS:
  --quant bf16|int8|int4 weight format for the transformer layers (default bf16)
  --draft-quant FORMAT   the same, for the draft model
  --kv-mem 1G            memory for the KV cache (K, M, G suffixes)
  --block-size 16        tokens per cache block
  --max-batch-tokens 512 tokens processed per step
  --max-seqs 64          sequences in a batch
  --admission continuous|static
  --reserve TOKENS       reserve this many token slots per sequence up front
  --no-prefix-cache      disable prefix caching
  --spec-k 4             tokens the draft proposes per step
  --threads N            compute threads (default: all cores)

random:SHAPE runs a model with random weights and the shape of smollm2-135m,
smollm2-360m or smollm2-1.7b, for measuring speed without downloading weights.
";

struct Args(Vec<String>);

impl Args {
    fn get(&self, name: &str) -> Option<&str> {
        let i = self.0.iter().position(|a| a == name)?;
        self.0.get(i + 1).map(String::as_str)
    }

    fn has(&self, name: &str) -> bool {
        self.0.iter().any(|a| a == name)
    }

    fn num<T: std::str::FromStr>(&self, name: &str, default: T) -> T {
        self.get(name).map_or(default, |v| {
            v.parse().unwrap_or_else(|_| die(&format!("{name}: bad value {v:?}")))
        })
    }

    fn range(&self, name: &str, default: (usize, usize)) -> (usize, usize) {
        match self.get(name) {
            None => default,
            Some(v) => {
                let (a, b) = v.split_once("..").unwrap_or((v, v));
                let p = |x: &str| x.parse().unwrap_or_else(|_| die(&format!("{name}: bad range {v:?}")));
                (p(a), p(b))
            }
        }
    }
}

fn die(msg: &str) -> ! {
    eprintln!("ferrolm: {msg}");
    std::process::exit(2)
}

fn bytes(s: &str) -> usize {
    let (n, mul) = match s.chars().last() {
        Some('K' | 'k') => (&s[..s.len() - 1], 1 << 10),
        Some('M' | 'm') => (&s[..s.len() - 1], 1 << 20),
        Some('G' | 'g') => (&s[..s.len() - 1], 1 << 30),
        _ => (s, 1),
    };
    (n.parse::<f64>().unwrap_or_else(|_| die(&format!("bad size {s:?}"))) * mul as f64) as usize
}

fn shape(name: &str) -> Config {
    let (hidden, inter, layers, heads, kv, theta) = match name {
        "smollm2-135m" => (576, 1536, 30, 9, 3, 100000.0),
        "smollm2-360m" => (960, 2560, 32, 15, 5, 100000.0),
        "smollm2-1.7b" => (2048, 8192, 24, 32, 32, 130000.0),
        _ => die(&format!(
            "unknown shape {name:?}; try smollm2-135m, smollm2-360m or smollm2-1.7b"
        )),
    };
    Config {
        arch: "llama".into(),
        vocab: 49152,
        hidden,
        intermediate: inter,
        layers,
        heads,
        kv_heads: kv,
        head_dim: hidden / heads,
        rms_eps: 1e-5,
        rope_theta: theta,
        rope_scaling: RopeScaling::None,
        max_position: 8192,
        tie_embeddings: true,
        qkv_bias: false,
        eos: vec![0],
        bos: None,
    }
}

fn load(spec: &str, quant: Quant) -> Model {
    let t = Instant::now();
    let mut m = match spec.strip_prefix("random:") {
        Some(s) => Model::random(shape(s), 1),
        None => Model::load(Path::new(spec)).unwrap_or_else(|e| die(&format!("{spec}: {e}"))),
    };
    m.quantize(quant, None);
    eprintln!(
        "ferrolm: loaded {spec} ({}, {:.0}M parameters, {} layers) in {:.1} s",
        m.cfg.arch,
        m.cfg.params() as f64 / 1e6,
        m.cfg.layers,
        t.elapsed().as_secs_f64()
    );
    eprintln!(
        "ferrolm: weights {} ({:.0} MB)",
        quant.name(),
        m.weight_bytes() as f64 / 1e6
    );
    m
}

fn quant_arg(a: &Args, name: &str) -> Quant {
    a.get(name).map_or(Quant::Bf16, |q| {
        Quant::parse(q).unwrap_or_else(|| die(&format!("{name}: expected bf16, int8 or int4")))
    })
}

fn engine_config(a: &Args, m: &Model) -> EngineConfig {
    let block_size = a.num("--block-size", 16usize);
    let mem = bytes(a.get("--kv-mem").unwrap_or("1G"));
    let kv_blocks = (mem / (m.cfg.kv_bytes_per_token() * block_size)).max(1);
    EngineConfig {
        block_size,
        kv_blocks,
        max_batch_tokens: a.num("--max-batch-tokens", 512),
        max_seqs: a.num("--max-seqs", 64),
        prefix_cache: !a.has("--no-prefix-cache"),
        admission: match a.get("--admission").unwrap_or("continuous") {
            "continuous" => Admission::Continuous,
            "static" => Admission::Static,
            other => die(&format!("unknown admission {other:?}")),
        },
        reserve_tokens: a
            .get("--reserve")
            .map(|v| v.parse().unwrap_or_else(|_| die("bad --reserve"))),
        spec_k: a.num("--spec-k", 4),
    }
}

fn build(a: &Args) -> (Engine, ferrolm::engine::Handle, EngineConfig) {
    let model = load(
        a.get("--model").unwrap_or_else(|| die("--model is required")),
        quant_arg(a, "--quant"),
    );
    let draft = a.get("--draft").map(|d| load(d, quant_arg(a, "--draft-quant")));
    let cfg = engine_config(a, &model);
    let pool = match a.get("--threads") {
        Some(n) => Pool::new(n.parse().unwrap_or_else(|_| die("bad --threads"))),
        None => Pool::with_all_cores(),
    };
    eprintln!(
        "ferrolm: {} kernels, {} threads, KV cache {} blocks x {} tokens ({:.0} MB)",
        ferrolm::kernels::backend(),
        pool.threads(),
        cfg.kv_blocks,
        cfg.block_size,
        (cfg.kv_blocks * cfg.block_size * model.cfg.kv_bytes_per_token()) as f64 / 1e6
    );
    let (e, h) = Engine::new(model, draft, pool, cfg.clone());
    (e, h, cfg)
}

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let Some(cmd) = argv.first().cloned() else {
        print!("{USAGE}");
        return;
    };
    let a = Args(argv);
    match cmd.as_str() {
        "serve" => {
            let dir = a.get("--model").unwrap_or_else(|| die("--model is required"));
            let tok = Tokenizer::load(Path::new(dir)).unwrap_or_else(|e| die(&format!("tokenizer: {e}")));
            let (mut engine, handle, _) = build(&a);
            std::thread::spawn(move || engine.run());
            let name = Path::new(dir)
                .file_name()
                .map_or(dir.to_string(), |n| n.to_string_lossy().into_owned());
            let addr = format!(
                "{}:{}",
                a.get("--host").unwrap_or("127.0.0.1"),
                a.num("--port", 8000u16)
            );
            let server = Arc::new(Server::new(handle, Arc::new(tok), name));
            server.listen(&addr).unwrap_or_else(|e| die(&format!("{addr}: {e}")));
        }
        "generate" => {
            let dir = a.get("--model").unwrap_or_else(|| die("--model is required"));
            let tok = Tokenizer::load(Path::new(dir)).unwrap_or_else(|e| die(&format!("tokenizer: {e}")));
            let text = a.get("--prompt").unwrap_or_else(|| die("--prompt is required"));
            let prompt = if a.has("--chat") {
                tok.encode(&tok.chat_prompt(&[("user".into(), text.into())]), true)
            } else {
                tok.encode(text, true)
            };
            let (mut engine, handle, _) = build(&a);
            let params = SamplingParams {
                temperature: a.num("--temperature", 0.0),
                top_p: a.num("--top-p", 1.0),
                top_k: a.num("--top-k", 0),
                seed: a.num("--seed", 1),
                max_tokens: a.num("--max-tokens", 128),
                stop_ids: tok.stop_ids.clone(),
                ignore_eos: false,
            };
            let n_prompt = prompt.len();
            let (rx, _) = handle.submit(prompt, params);
            let stats = Arc::clone(&handle.stats);
            drop(handle);
            let start = Instant::now();
            let worker = std::thread::spawn(move || engine.run());
            let mut detok = Detokenizer::new(&tok, Vec::new());
            let (mut first, mut n) = (None, 0);
            let mut out = std::io::stdout();
            for ev in rx {
                match ev {
                    Event::Token(t) => {
                        n += 1;
                        first.get_or_insert(start.elapsed().as_secs_f64());
                        print!("{}", detok.push(t));
                        out.flush().ok();
                    }
                    Event::Done(f) => {
                        println!("{}", detok.flush());
                        let total = start.elapsed().as_secs_f64();
                        let ttft = first.unwrap_or(total);
                        let s = stats.lock().unwrap().clone();
                        let mut line = format!(
                            "[{n_prompt} prompt tokens, {n} generated ({f:?}); first token {:.2} s, then {:.1} tok/s",
                            ttft,
                            (n.max(1) - 1) as f64 / (total - ttft).max(1e-9)
                        );
                        if s.spec_proposed > 0 {
                            line.push_str(&format!(
                                "; {:.0}% of draft tokens accepted",
                                100.0 * s.spec_accepted as f64 / s.spec_proposed as f64
                            ));
                        }
                        eprintln!("{line}]");
                        break;
                    }
                }
            }
            worker.join().ok();
        }
        "bench" => {
            let (engine, handle, _) = build(&a);
            let w = Workload {
                requests: a.num("--requests", 64),
                rate: a.get("--rate").map(|r| r.parse().unwrap_or_else(|_| die("bad --rate"))),
                prompt_len: a.range("--prompt-len", (64, 256)),
                gen_len: a.range("--gen-len", (32, 128)),
                shared_prefix: a.num("--shared-prefix", 0),
                seed: a.num("--seed", 1),
                temperature: a.num("--temperature", 0.0),
                prompts: Vec::new(),
                stop_ids: Vec::new(),
            };
            let mut w = w;
            if let Some(f) = a.get("--prompt-file") {
                // One prompt per line, as a user turn in the chat template.
                let dir = a.get("--model").unwrap();
                let tok = Tokenizer::load(Path::new(dir)).unwrap_or_else(|e| die(&format!("tokenizer: {e}")));
                let text = std::fs::read_to_string(f).unwrap_or_else(|e| die(&format!("{f}: {e}")));
                w.prompts = text
                    .lines()
                    .filter(|l| !l.trim().is_empty())
                    .map(|l| tok.encode(&tok.chat_prompt(&[("user".into(), l.trim().into())]), true))
                    .collect();
                w.stop_ids = tok.stop_ids.clone();
            }
            let label = a.get("--label").unwrap_or("bench").to_string();
            let s = bench::run(engine, handle, &w);
            s.print(&label);
            if let Some(f) = a.get("--json") {
                let mut file = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(f)
                    .unwrap_or_else(|e| die(&format!("{f}: {e}")));
                writeln!(file, "{}", s.to_json(&label)).ok();
            }
        }
        "perplexity" => {
            // Quality of the (optionally quantized) model on held-out text.
            let dir = a.get("--model").unwrap_or_else(|| die("--model is required"));
            let data = a.get("--data").unwrap_or_else(|| die("--data FILE is required"));
            let quant = quant_arg(&a, "--quant");
            let ctx = a.num("--ctx", 512usize);
            let windows = a.num("--windows", 40usize);
            let tok = Tokenizer::load(Path::new(dir)).unwrap_or_else(|e| die(&format!("tokenizer: {e}")));
            let text = std::fs::read_to_string(data).unwrap_or_else(|e| die(&format!("{data}: {e}")));
            let tokens = tok.encode(&text, false);
            let pool = Pool::with_all_cores();
            let mut model = load(dir, Quant::Bf16);
            let t = Instant::now();
            let mut label = quant.name().to_string();
            if quant != Quant::Bf16 && a.has("--awq") {
                let calib_file = a.get("--calib").unwrap_or_else(|| die("--awq needs --calib FILE"));
                let calib = std::fs::read_to_string(calib_file).unwrap_or_else(|e| die(&format!("{calib_file}: {e}")));
                let seqs = ferrolm::eval::sample_sequences(&tok.encode(&calib, false), a.num("--calib-seqs", 32), 512);
                let acts = model.calibrate(&pool, &seqs, a.num("--calib-rows", 256));
                let choices = model.quantize_awq(quant, &acts);
                let gain: f64 = choices
                    .iter()
                    .map(|c| c.error_plain / c.error_awq.max(1e-30))
                    .sum::<f64>()
                    / choices.len().max(1) as f64;
                eprintln!(
                    "ferrolm: AWQ on {} projections, mean output-error reduction {:.2}x",
                    choices.len(),
                    gain
                );
                label.push_str("+awq");
            } else {
                model.quantize(quant, None);
            }
            let prep = t.elapsed().as_secs_f64();
            let t = Instant::now();
            let p = ferrolm::eval::perplexity(&model, &pool, &tokens, ctx, windows);
            let secs = t.elapsed().as_secs_f64();
            println!(
                "{label}: perplexity {:.3} over {} tokens ({} windows of {ctx}); weights {:.0} MB; quantize {:.0} s, eval {:.0} s",
                p.ppl,
                p.tokens,
                p.windows,
                model.weight_bytes() as f64 / 1e6,
                prep,
                secs
            );
            if let Some(f) = a.get("--json") {
                let mut file = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(f)
                    .unwrap_or_else(|e| die(&format!("{f}: {e}")));
                writeln!(
                    file,
                    r#"{{"model":{},"format":{},"ppl":{:.4},"tokens":{},"weight_mb":{:.1}}}"#,
                    ferrolm::json::quote(dir),
                    ferrolm::json::quote(&label),
                    p.ppl,
                    p.tokens,
                    model.weight_bytes() as f64 / 1e6
                )
                .ok();
            }
        }
        "ann" => {
            // Vector search on a standard benchmark (e.g. SIFT1M .fvecs).
            use ferrolm::vector::{
                Metric,
                hnsw::{Hnsw, HnswParams},
                io,
                ivfpq::{IvfPq, IvfPqParams},
                recall,
            };
            let path =
                |k: &str| Path::new(a.get(k).unwrap_or_else(|| die(&format!("{k} FILE is required")))).to_path_buf();
            let n = a
                .get("--n")
                .map(|v| v.parse::<usize>().unwrap_or_else(|_| die("bad --n")));
            let (base, dim) = io::read_fvecs(&path("--base"), n).unwrap_or_else(|e| die(&e));
            let (queries, qdim) = io::read_fvecs(
                &path("--query"),
                a.get("--queries").map(|v| v.parse().unwrap_or(10_000)),
            )
            .unwrap_or_else(|e| die(&e));
            assert_eq!(dim, qdim, "query dimension");
            let truth = io::read_ivecs(&path("--truth"), Some(queries.len() / dim)).unwrap_or_else(|e| die(&e));
            let pool = match a.get("--threads") {
                Some(t) => Pool::new(t.parse().unwrap_or_else(|_| die("bad --threads"))),
                None => Pool::with_all_cores(),
            };
            let one = Pool::new(1);
            let list = |k: &str, d: &str| -> Vec<usize> {
                a.get(k)
                    .unwrap_or(d)
                    .split(',')
                    .map(|v| v.parse().unwrap_or_else(|_| die(&format!("bad {k}"))))
                    .collect()
            };
            let nq = queries.len() / dim;
            let label = a.get("--label").unwrap_or("ferrolm").to_string();
            let json = a.get("--json").map(String::from);
            let emit = |index: &str, param: String, r: f64, qps1: f64, qpsn: f64, build: f64| {
                println!(
                    "{label} {index} {param}: recall@10 {r:.4}  {qps1:.0} QPS (1 thread)  {qpsn:.0} QPS ({} threads)  build {build:.0} s",
                    pool.threads()
                );
                if let Some(f) = &json {
                    let mut file = std::fs::OpenOptions::new()
                        .create(true)
                        .append(true)
                        .open(f)
                        .unwrap_or_else(|e| die(&format!("{f}: {e}")));
                    writeln!(file, r#"{{"engine":{},"index":"{index}","param":"{param}","recall10":{r:.5},"qps_1":{qps1:.1},"qps_n":{qpsn:.1},"threads":{},"build_s":{build:.1},"n":{}}}"#, ferrolm::json::quote(&label), pool.threads(), base.len() / dim).ok();
                }
            };
            // Single-threaded QPS on a subset (it is slow); multi-threaded on all.
            let sub = nq.min(a.num("--single-thread-queries", 2000usize));
            type Search<'a> = &'a dyn Fn(&Pool, &[f32]) -> Vec<Vec<ferrolm::vector::Neighbor>>;
            let timed = |f: Search| -> (f64, f64, f64) {
                let t = Instant::now();
                let r = f(&pool, &queries);
                let qpsn = nq as f64 / t.elapsed().as_secs_f64();
                let t = Instant::now();
                f(&one, &queries[..sub * dim]);
                let qps1 = sub as f64 / t.elapsed().as_secs_f64();
                (recall(&r, &truth, 10), qps1, qpsn)
            };
            match a.get("--index").unwrap_or("hnsw") {
                "hnsw" => {
                    let p = HnswParams {
                        m: a.num("--m", 16),
                        ef_construction: a.num("--ef-construction", 200),
                        seed: 1,
                    };
                    let t = Instant::now();
                    let h = Hnsw::build(&pool, Metric::L2, dim, &base, p);
                    let build = t.elapsed().as_secs_f64();
                    eprintln!(
                        "ferrolm: HNSW over {} vectors built in {build:.0} s, mean degree {:.1}",
                        h.len(),
                        h.mean_degree()
                    );
                    for ef in list("--ef", "16,32,64,128,256") {
                        let (r, q1, qn) = timed(&|p, q| h.search_batch(p, q, 10, ef));
                        emit(
                            "hnsw",
                            format!("M={} efC={} ef={ef}", p.m, p.ef_construction),
                            r,
                            q1,
                            qn,
                            build,
                        );
                    }
                }
                "ivfpq" => {
                    let (learn, _) = io::read_fvecs(&path("--learn"), None).unwrap_or_else(|e| die(&e));
                    let p = IvfPqParams {
                        nlist: a.num("--nlist", 1024),
                        m: a.num("--pq-m", 16),
                        iters: a.num("--iters", 20),
                        keep_vectors: true,
                        seed: 1,
                    };
                    let t = Instant::now();
                    let mut ix = IvfPq::train(&pool, Metric::L2, dim, &learn, p);
                    ix.add(&pool, &base);
                    let build = t.elapsed().as_secs_f64();
                    eprintln!(
                        "ferrolm: IVF-PQ over {} vectors built in {build:.0} s, {} bytes per vector",
                        ix.len(),
                        ix.bytes_per_vector()
                    );
                    for refine in list("--refine", "0,10") {
                        for nprobe in list("--nprobe", "1,4,16,64") {
                            let (r, q1, qn) = timed(&|p, q| ix.search_batch(p, q, 10, nprobe, refine));
                            emit(
                                "ivfpq",
                                format!("nlist={} m={} nprobe={nprobe} refine={refine}", p.nlist, p.m),
                                r,
                                q1,
                                qn,
                                build,
                            );
                        }
                    }
                }
                other => die(&format!("unknown index {other:?}")),
            }
        }
        "info" => {
            let m = load(
                a.get("--model").unwrap_or_else(|| die("--model is required")),
                quant_arg(&a, "--quant"),
            );
            println!("{:#?}", m.cfg);
            println!("kernels: {}", ferrolm::kernels::backend());
            println!("KV cache per token: {} bytes", m.cfg.kv_bytes_per_token());
        }
        "help" | "--help" | "-h" => print!("{USAGE}"),
        other => die(&format!("unknown command {other:?}\n\n{USAGE}")),
    }
}
