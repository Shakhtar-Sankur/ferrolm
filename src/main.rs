use ferrolm::bench::{self, Workload};
use ferrolm::config::{Config, RopeScaling};
use ferrolm::encoder::Encoder;
use ferrolm::engine::{Admission, Engine, EngineConfig, Event};
use ferrolm::json::Json;
use ferrolm::model::Model;
use ferrolm::pool::Pool;
use ferrolm::quant::Quant;
use ferrolm::rag::{self, IndexKind, Retriever};
use ferrolm::sampler::SamplingParams;
use ferrolm::server::{Detokenizer, Embedder, Server};
use ferrolm::tokenizer::Tokenizer;
use std::collections::HashMap;
use std::io::Write;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

const USAGE: &str = "ferrolm: an LLM inference server in Rust

USAGE:
  ferrolm serve    --model DIR [--draft DIR] [--host 127.0.0.1] [--port 8000] [engine options]
                   [--embedding-model DIR [--embedding-threads 2]]
  ferrolm generate --model DIR [--draft DIR] --prompt TEXT [--chat] [--max-tokens 128]
                   [--temperature 0] [--top-p 1] [--top-k 0] [--seed 1] [engine options]
  ferrolm bench    --model DIR|random:SHAPE [--draft DIR|random:SHAPE] [--requests 64]
                   [--rate REQ_PER_S] [--prompt-len 64..256] [--gen-len 32..128]
                   [--shared-prefix 0] [--prompt-file FILE] [--temperature 0] [--seed 1] [--label NAME]
                   [--json FILE] [engine options]
  ferrolm perplexity --model DIR --data FILE [--quant int8|int4] [--awq --calib FILE]
                   [--ctx 512] [--windows 40] [--json FILE]
  ferrolm ann      --index hnsw|ivfpq --base F --query F --truth F [--learn F] [--json FILE]
                   (SIFT1M-format vector search benchmark; see bench/ann.sh)
  ferrolm retrieval --encoder DIR --beir DIR [--qrels FILE] [--index flat|hnsw] [--ef 128]
                   [--query-prefix TEXT] [--max-tokens 0] [--overlap 0] [--json FILE]
  ferrolm qa       --model DIR --encoder DIR --data SQUAD.jsonl [--n 200] [--k 3]
                   [--modes closed,rag,oracle] [--index flat|hnsw] [--query-prefix TEXT]
                   [--json FILE] [engine options]
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

fn read_jsonl(path: &Path) -> Vec<Json> {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| die(&format!("{}: {e}", path.display())));
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| ferrolm::json::parse(l).unwrap_or_else(|e| die(&format!("{}: {e}", path.display()))))
        .collect()
}

fn jstr(v: &Json, key: &str) -> String {
    v.get(key).and_then(Json::as_str).unwrap_or("").to_string()
}

/// BEIR corpus: ids, and "title text" per document.
fn read_corpus(path: &Path) -> (Vec<String>, Vec<String>) {
    read_jsonl(path)
        .iter()
        .map(|d| {
            (
                jstr(d, "_id"),
                format!("{} {}", jstr(d, "title"), jstr(d, "text")).trim().to_string(),
            )
        })
        .unzip()
}

fn index_kind(a: &Args) -> IndexKind {
    match a.get("--index").unwrap_or("hnsw") {
        "flat" => IndexKind::Flat,
        "hnsw" => IndexKind::Hnsw {
            m: a.num("--m", 16),
            ef_construction: a.num("--ef-construction", 200),
            ef: a.num("--ef", 128),
        },
        other => die(&format!("unknown index {other:?} (flat or hnsw)")),
    }
}

fn append_line(path: &str, line: &str) {
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .unwrap_or_else(|e| die(&format!("{path}: {e}")));
    writeln!(f, "{line}").ok();
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
            let mut server = Server::new(handle, Arc::new(tok), name);
            if let Some(e) = a.get("--embedding-model") {
                let enc = Encoder::load(Path::new(e)).unwrap_or_else(|err| die(&format!("{e}: {err}")));
                let name = Path::new(e)
                    .file_name()
                    .map_or(e.to_string(), |n| n.to_string_lossy().into_owned());
                server.embedder = Some(Embedder::new(name, enc, a.num("--embedding-threads", 2)));
            }
            let server = Arc::new(server);
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
        "retrieval" => {
            // BEIR-format evaluation: corpus.jsonl, queries.jsonl, qrels TSV.
            let enc_dir = a.get("--encoder").unwrap_or_else(|| die("--encoder is required"));
            let beir = Path::new(a.get("--beir").unwrap_or_else(|| die("--beir DIR is required")));
            let qrels_path = a.get("--qrels").map_or_else(|| beir.join("qrels-test.tsv"), Into::into);
            let pool = Pool::new(a.num("--threads", ferrolm::pool::cores()));
            let (ids, docs) = read_corpus(&beir.join("corpus.jsonl"));
            let queries: HashMap<String, String> = read_jsonl(&beir.join("queries.jsonl"))
                .iter()
                .map(|q| (jstr(q, "_id"), jstr(q, "text")))
                .collect();
            let mut qrels: HashMap<String, HashMap<String, u32>> = HashMap::new();
            let qtext =
                std::fs::read_to_string(&qrels_path).unwrap_or_else(|e| die(&format!("{}: {e}", qrels_path.display())));
            for line in qtext.lines().skip(1) {
                let f: Vec<&str> = line.split('\t').collect();
                if f.len() == 3 {
                    qrels
                        .entry(f[0].into())
                        .or_default()
                        .insert(f[1].into(), f[2].parse().unwrap_or(0));
                }
            }
            let mut qids: Vec<&String> = qrels.keys().filter(|q| queries.contains_key(*q)).collect();
            qids.sort();
            let enc = Encoder::load(Path::new(enc_dir)).unwrap_or_else(|e| die(&format!("{enc_dir}: {e}")));
            let kind = index_kind(&a);
            let prefix = a.get("--query-prefix").unwrap_or("");
            let t = Instant::now();
            let r = Retriever::build(
                &pool,
                enc,
                &docs,
                a.num("--max-tokens", 0),
                a.num("--overlap", 0),
                kind,
                prefix,
            );
            let build = t.elapsed().as_secs_f64();
            eprintln!(
                "ferrolm: {} documents, {} passages, {} tokens embedded in {:.0} s ({:.0} tokens/s); index built in {:.1} s",
                docs.len(),
                r.passages.len(),
                r.embed_tokens,
                r.embed_seconds,
                r.embed_tokens as f64 / r.embed_seconds,
                build - r.embed_seconds
            );
            let texts: Vec<&str> = qids.iter().map(|q| queries[*q].as_str()).collect();
            let t = Instant::now();
            let qv = r.embed_queries(&pool, &texts);
            let hits = r.search_docs(&pool, &qv, 101);
            let qsec = t.elapsed().as_secs_f64();
            let (mut n10, mut r100) = (0.0, 0.0);
            for (q, h) in qids.iter().zip(&hits) {
                // As BEIR does, never count the query's own id as a hit.
                let ranked: Vec<&str> = h
                    .iter()
                    .map(|&(d, _)| ids[d as usize].as_str())
                    .filter(|d| d != q)
                    .collect();
                n10 += rag::ndcg(&ranked, &qrels[*q], 10);
                r100 += rag::recall_at(&ranked, &qrels[*q], 100);
            }
            let nq = qids.len() as f64;
            let (n10, r100) = (n10 / nq, r100 / nq);
            let label = a.get("--label").unwrap_or("ferrolm").to_string();
            println!(
                "{label}: {} queries, nDCG@10 {n10:.4}, recall@100 {r100:.4}; {:.1} ms per query (embed + search)",
                qids.len(),
                1000.0 * qsec / nq
            );
            if let Some(f) = a.get("--json") {
                append_line(
                    f,
                    &format!(
                        r#"{{"label":{},"dataset":{},"queries":{},"ndcg10":{n10:.5},"recall100":{r100:.5},"docs":{},"passages":{},"embed_tokens_per_s":{:.0},"ms_per_query":{:.2}}}"#,
                        ferrolm::json::quote(&label),
                        ferrolm::json::quote(&beir.display().to_string()),
                        qids.len(),
                        docs.len(),
                        r.passages.len(),
                        r.embed_tokens as f64 / r.embed_seconds,
                        1000.0 * qsec / nq
                    ),
                );
            }
        }
        "qa" => {
            // Question answering with and without retrieval, on SQuAD-format
            // JSONL ({question, context, answers}): closed book, retrieved
            // passages from every distinct context, or the gold context.
            let enc_dir = a.get("--encoder").unwrap_or_else(|| die("--encoder is required"));
            let data = read_jsonl(Path::new(
                a.get("--data").unwrap_or_else(|| die("--data FILE is required")),
            ));
            let mut contexts: Vec<String> = Vec::new();
            let mut ctx_id: HashMap<String, u32> = HashMap::new();
            for d in &data {
                let c = jstr(d, "context");
                if !ctx_id.contains_key(&c) {
                    ctx_id.insert(c.clone(), contexts.len() as u32);
                    contexts.push(c);
                }
            }
            // A fixed random sample of questions.
            let n = a.num("--n", 200usize).min(data.len());
            let mut rng = ferrolm::rng::Rng::new(a.num("--seed", 1));
            let mut order: Vec<usize> = (0..data.len()).collect();
            for i in 0..n {
                let j = i + rng.below((data.len() - i) as u64) as usize;
                order.swap(i, j);
            }
            let sample: Vec<&Json> = order[..n].iter().map(|&i| &data[i]).collect();
            let k = a.num("--k", 3usize);
            let pool = Pool::new(a.num("--threads", ferrolm::pool::cores()));
            let enc = Encoder::load(Path::new(enc_dir)).unwrap_or_else(|e| die(&format!("{enc_dir}: {e}")));
            let prefix = a.get("--query-prefix").unwrap_or("");
            let r = Retriever::build(
                &pool,
                enc,
                &contexts,
                a.num("--max-tokens", 0),
                a.num("--overlap", 0),
                index_kind(&a),
                prefix,
            );
            let questions: Vec<String> = sample.iter().map(|d| jstr(d, "question")).collect();
            let qrefs: Vec<&str> = questions.iter().map(String::as_str).collect();
            let hits = r.search(&pool, &qrefs, k);
            drop(pool);
            let gold: Vec<u32> = sample.iter().map(|d| ctx_id[&jstr(d, "context")]).collect();
            let found = hits
                .iter()
                .zip(&gold)
                .filter(|(h, g)| h.iter().any(|&(p, _)| r.passages[p as usize].doc == **g))
                .count();
            eprintln!(
                "ferrolm: {} contexts indexed; gold context in the top {k} for {found} of {n} questions",
                contexts.len()
            );
            let dir = a.get("--model").unwrap_or_else(|| die("--model is required"));
            let tok = Tokenizer::load(Path::new(dir)).unwrap_or_else(|e| die(&format!("tokenizer: {e}")));
            let (mut engine, handle, _) = build(&a);
            let worker = std::thread::spawn(move || engine.run());
            let label = a.get("--label").unwrap_or("ferrolm").to_string();
            let mut samples = Vec::new();
            for mode in a.get("--modes").unwrap_or("closed,rag,oracle").split(',') {
                let prompts: Vec<String> = (0..n)
                    .map(|i| {
                        let q = &questions[i];
                        let passages: Vec<&str> = match mode {
                            "closed" => Vec::new(),
                            "rag" => hits[i]
                                .iter()
                                .map(|&(p, _)| r.passages[p as usize].text.as_str())
                                .collect(),
                            "oracle" => vec![contexts[gold[i] as usize].as_str()],
                            m => die(&format!("unknown mode {m:?}")),
                        };
                        tok.chat_prompt(&[("user".into(), rag::prompt(q, &passages))])
                    })
                    .collect();
                let t = Instant::now();
                let mut rxs = Vec::new();
                let mut prompt_tokens = 0;
                for p in &prompts {
                    let ids = tok.encode(p, true);
                    prompt_tokens += ids.len();
                    let params = SamplingParams {
                        temperature: 0.0,
                        top_p: 1.0,
                        top_k: 0,
                        seed: 1,
                        max_tokens: a.num("--max-new-tokens", 32),
                        stop_ids: tok.stop_ids.clone(),
                        ignore_eos: false,
                    };
                    rxs.push(handle.submit(ids, params).0);
                }
                let (mut em, mut f1) = (0.0, 0.0);
                for (i, rx) in rxs.into_iter().enumerate() {
                    let mut toks = Vec::new();
                    for ev in rx {
                        match ev {
                            Event::Token(t) => toks.push(t),
                            Event::Done(_) => break,
                        }
                    }
                    let text = tok.decode(&toks, true);
                    let answer = text.trim().lines().next().unwrap_or("").trim().to_string();
                    let golds: Vec<String> = sample[i]
                        .get("answers")
                        .map(|v| v.as_arr().iter().filter_map(|x| x.as_str().map(String::from)).collect())
                        .unwrap_or_default();
                    let (e, f) = rag::squad_scores(&answer, &golds);
                    em += e;
                    f1 += f;
                    if i < 5 {
                        samples.push(format!(
                            "[{mode}] {} -> {answer:?} (gold {:?})",
                            questions[i],
                            golds.first()
                        ));
                    }
                }
                let secs = t.elapsed().as_secs_f64();
                let (em, f1) = (100.0 * em / n as f64, 100.0 * f1 / n as f64);
                println!(
                    "{label} {mode}: exact match {em:.1}, F1 {f1:.1} on {n} questions ({:.0} prompt tokens each, {secs:.0} s)",
                    prompt_tokens as f64 / n as f64
                );
                if let Some(f) = a.get("--json") {
                    append_line(
                        f,
                        &format!(
                            r#"{{"label":{},"mode":"{mode}","n":{n},"k":{k},"em":{em:.2},"f1":{f1:.2},"retrieval_hits":{found},"prompt_tokens":{:.1},"seconds":{secs:.1}}}"#,
                            ferrolm::json::quote(&label),
                            prompt_tokens as f64 / n as f64
                        ),
                    );
                }
            }
            for s in samples {
                eprintln!("{s}");
            }
            drop(handle);
            worker.join().ok();
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
