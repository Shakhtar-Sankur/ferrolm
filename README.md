# ferrolm

An LLM inference server written from scratch in Rust, with no dependencies.
It loads Hugging Face checkpoints (SmolLM2, Llama, Qwen2) and serves an
OpenAI-compatible API, with the techniques production servers such as vLLM
use: a **paged KV cache**, **continuous batching**, **prefix caching** and
**speculative decoding**, on hand-written **AVX-512 / AVX2** kernels.

Its outputs are checked against Hugging Face transformers, and every
serving optimisation is tested to leave a request's output bit for bit
unchanged.

On a 4-vCPU machine, with real SmolLM2 weights:

| Technique | Measured effect |
|---|---|
| Continuous batching vs one request at a time | **3.4×** throughput (115 vs 34 tokens/s) |
| Continuous vs static batching, at 0.5 requests/s | first token **18× sooner** (0.54 s vs 9.55 s, median) |
| Paged vs reserved KV cache, same 256 MB | **20 sequences at once instead of 1**, 2.4× throughput |
| Prefix caching, shared 512-token system prompt | first token **46× sooner** (0.33 s vs 15.3 s, median) |
| Speculative decoding, SmolLM2-1.7B + 135M draft | **1.47×** single-stream speed, identical text |

## Results

All runs: `bench/run.sh`, raw numbers in `bench/results/`. Machine: 4 vCPUs
of an Intel Xeon (Cascade Lake, AVX-512), 15 GB RAM. Models in bf16. The
load generator sends prompts of random tokens with fixed reply lengths
(except for speculative decoding, which needs real text), and records when
every token arrives.

### Continuous batching

![Throughput of one-at-a-time, static and continuous batching](docs/throughput.svg)

The same 64 requests (prompts of 32 to 256 tokens, replies of 16 to 256),
queued at once, on SmolLM2-360M:

| Scheduling | Output tok/s | First token, median | Finished, median |
|---|---|---|---|
| One request at a time | 34.1 | 134.1 s | 137.9 s |
| Static batching (up to 32 per batch) | 101.5 | 26.1 s | 57.7 s |
| Continuous batching (up to 32 running) | **115.3** | **15.4 s** | 60.2 s |

Batching wins because decoding one token reads every weight once; a batch
reads them once for all its sequences. Continuous batching adds more when
requests arrive over time, because static batching makes new requests wait
for the whole current batch to finish:

![Time to first token under load](docs/latency.svg)

| Arrivals/s | Static: first token p50 / p90 | Continuous: p50 / p90 | Static tok/s | Continuous tok/s |
|---|---|---|---|---|
| 0.5 | 9.55 s / 15.8 s | **0.54 s / 0.89 s** | 74.9 | 82.5 |
| 1 | 15.1 s / 28.6 s | **0.75 s / 1.54 s** | 89.3 | 109.4 |
| 2 | 30.3 s / 32.7 s | **3.17 s / 18.6 s** | 93.2 | 110.3 |
| 3 | 13.6 s / 44.6 s | **5.61 s / 24.1 s** | 94.6 | 112.5 |
| 4 | 13.9 s / 46.9 s | **7.12 s / 26.0 s** | 99.2 | 118.6 |

This CPU saturates at about 0.7 requests/s for this workload, so from
2 requests/s on both queue up; continuous batching still answers first and
serves more.

### Paged KV cache

With 256 MB for keys and values (204 blocks, 3,264 tokens of SmolLM2-360M), reserving
a 2,048-token window per sequence, as a server without paging must, fits
one sequence at a time. Allocating 16-token blocks as sequences grow fits
many:

| KV cache, 256 MB | Most sequences at once | Output tok/s |
|---|---|---|
| Reserve 2,048 tokens per sequence | 1 | 34.6 |
| Paged, 16-token blocks | **20** | **83.8** (2.4×) |

When the paged cache filled up, the scheduler preempted and later
recomputed sequences 42 times; every request still completed.

### Prefix caching

32 requests, one per second, all starting with the same 512-token system
prompt:

| | First token p50 / p90 | Finished, median | Prompt tokens from cache |
|---|---|---|---|
| No prefix cache | 15.3 s / 32.3 s | 48.0 s | 0 |
| Prefix cache | **0.33 s / 0.63 s** | **2.9 s** | 15,872 |

Without the cache every request recomputes the system prompt and the
server falls behind; with it, each request after the first computes only
its own few tokens.

### Speculative decoding

![Speculative decoding speed by proposal length](docs/speculative.svg)

SmolLM2-1.7B answering six real instructions (the first six in
`bench/prompts.txt`, up to 128 tokens each) one at a time, with
SmolLM2-135M proposing `k` tokens per step. Under greedy decoding the text is identical with and without the
draft (tested), so the speedup is free. The best `k` here is 3: **13.8 vs
9.4 tokens/s**. Longer proposals are accepted less often and waste draft
work.

Speculation pays only when the target is memory-bound. With 8 sequences
batched, the target is already compute-bound and the draft's extra work
makes it slower: 26.9 tokens/s with a draft (k = 4) vs 30.4 without.

### Kernels

A single decode step of SmolLM2-360M (one sequence) takes about 26 ms and
of SmolLM2-1.7B about 103 ms, which is streaming the bf16 weights at
roughly 30 GB/s. The matrix multiply reaches 310 to 340 GFLOPS on
prefill-sized batches with AVX-512 on the 4 vCPUs (`examples/gemm.rs`).

## Correctness

A fast server that returns different text is a broken server, so every
optimisation here is checked against a reference or an invariant, and CI
runs the checks on every push.

| Check | Against | Result |
|---|---|---|
| Logits for a 50-token prompt | Hugging Face transformers (float32), on SmolLM2-135M, SmolLM2-360M and Qwen2.5-0.5B | max difference 6e-5 to 3.5e-4 (largest logit 22 to 31) |
| 32 greedy tokens | transformers `generate()` | identical, all three models |
| Three fixture architectures (Llama with GQA; Llama 3 with tied embeddings and RoPE scaling; Qwen2 with q/k/v biases) | transformers, random weights | max difference < 1.4e-5; 40 greedy tokens identical |
| Tokenizer ids | Hugging Face `tokenizers`: SmolLM2-, Llama 3- and Qwen2-style tokenizers on 2,026 multilingual and random-Unicode strings each, plus the real tokenizers on 1,026 each | identical |
| Chat prompts | transformers `apply_chat_template` | identical strings |
| Batching, chunked prefill, static vs continuous admission | the same request run alone | bit-identical tokens |
| Prefix-cache hits, preemption and recompute | the same request run alone | bit-identical tokens |
| Speculative decoding, greedy (a one-layer draft, and the target as its own draft; k = 1, 3, 4) | the target model alone | bit-identical tokens |
| Speculative sampling | the exact two-token distribution of the target | chi-square 19.9 on 15 degrees of freedom (limit 37.7) |
| SIMD kernels | the portable scalar path | bit-identical results |

The "bit-identical" rows rest on one design rule, below.

## Design

**Batch-invariant kernels.** Every output of the matrix multiply is a single
chain of fused multiply-adds over the inner dimension, in order, from zero.
Tiling only decides which chains run side by side in registers, never the
order inside a chain. Attention and the norms reduce in fixed orders too. So
a token's logits do not depend on what else is in the batch, how its prompt
was chunked, the thread count or the instruction set: AVX-512, AVX2 and the
portable path agree bit for bit. That turns "does batching change the
output?" from a tolerance question into an equality test, and it is also
why greedy speculative decoding reproduces the target's text exactly.

**Kernels.** Weights stay in bf16 and are packed at load time into panels of
32 output rows, so the kernel streams them sequentially and widens bf16 to
f32 with two shuffles. Register tiles go up to 8 rows × 32 columns
(AVX-512) or 3 × 32 (AVX2), with the inner dimension blocked for the L1
cache. Attention is fused per kv head, so grouped-query heads share each
key/value load. A persistent thread pool with spinning workers keeps the
hundreds of small jobs in a decode step cheap.

**Paged KV cache.** The cache is a pool of 16-token blocks and each sequence
owns a block table, so memory is committed as tokens arrive instead of
reserved for the longest possible sequence. Within a block each kv head's
keys are contiguous.

**Prefix caching.** Full blocks are content-addressed by a hash chained over
the whole prefix. A new request reuses cached leading blocks (checked
token by token, so a hash collision cannot serve wrong data), blocks are
reference counted, and idle cached blocks are evicted least recently used.

**Scheduler.** Each step packs running sequences' decode tokens and chunks of
new prompts into one batch under a token budget, so long prompts never stall
decoding and requests join and leave at every step. When the cache is full,
the most recently admitted sequence is preempted and later recomputed.
Static (request-level) admission and whole-sequence reservation are kept as
switches, for the baselines.

**Speculative decoding.** A small draft model proposes k tokens per sequence;
the target scores all of them in one batched pass and keeps the longest
agreeing prefix plus one token of its own. Under sampling, proposals are
accepted with probability min(1, p/q) and rejections resample from the
residual, which preserves the target distribution.

**Everything else from scratch.** safetensors loading, a byte-level BPE
tokenizer with hand-written pre-tokenizer matchers and Unicode NFC, JSON,
HTTP/1.1 with server-sent events, Prometheus metrics. No crates.

## Usage

```sh
cargo build --release
scripts/fetch_model.sh HuggingFaceTB/SmolLM2-360M-Instruct models/smollm2-360m

# OpenAI-compatible server
./target/release/ferrolm serve --model models/smollm2-360m --port 8000
curl localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"What is the capital of France?"}],"stream":true}'

# One prompt from the command line, optionally with a draft model
./target/release/ferrolm generate --model models/smollm2-1.7b --draft models/smollm2-135m \
  --chat --prompt "Explain recursion." --max-tokens 200

# Load test
./target/release/ferrolm bench --model models/smollm2-360m --requests 64 --rate 2
```

Endpoints: `POST /v1/completions`, `POST /v1/chat/completions` (streaming,
`stop`, `seed`, `temperature`, `top_p`, `top_k`), `GET /v1/models`,
`GET /health`, `GET /metrics`. Engine options (`--kv-mem`,
`--max-batch-tokens`, `--max-seqs`, `--spec-k`, ...) are listed by
`ferrolm help`.

Checkpoints: Llama-architecture models and Qwen2/2.5 in bf16, f16 or f32
safetensors, with byte-level BPE tokenizers. Tested on real weights:
SmolLM2-135M/360M/1.7B-Instruct and Qwen2.5-0.5B-Instruct. Llama 3 features
(tied embeddings, llama3 RoPE scaling, its tokenizer pattern and chat
format) are covered by fixture tests; Mistral-style configs without sliding
windows load but are untested.

## Limitations

- CPU only; weights are bf16 with f32 activations and an f32 KV cache. No
  quantization, so memory bandwidth bounds single-stream decoding.
- Tokenizers: byte-level BPE only (not SentencePiece), with the GPT-2,
  Llama 3 and Qwen2 split patterns. Chat templates: ChatML and Llama 3 are
  rendered natively; other Jinja templates fall back to plain text.
- One completion per request (`n = 1`), no tool calls or logprobs, one node.
- The benchmarks come from one 4-vCPU machine; absolute numbers depend on
  the CPU and its memory bandwidth, the ratios less so.

## Layout

| Path | What |
|---|---|
| `src/kernels.rs` | packed bf16 matrix multiply (AVX-512, AVX2, portable), attention, exp |
| `src/model.rs` | the transformer forward pass over batched chunks |
| `src/kv.rs` | paged KV cache, block manager, prefix cache |
| `src/engine.rs` | scheduler, preemption, speculative decoding |
| `src/tokenizer.rs`, `src/unicode.rs` | BPE tokenizer, pre-tokenizers, NFC |
| `src/server.rs` | HTTP server, streaming, detokenization |
| `src/bench.rs` | load generator |
| `tests/` | reference and invariance tests |
| `scripts/` | fixture and reference generators, model download, charts |
| `bench/run.sh` | reproduces every benchmark in this README |

## License

Apache-2.0
