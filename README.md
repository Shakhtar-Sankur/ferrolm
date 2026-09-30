# ferrolm

An LLM inference server written from scratch in Rust, with no dependencies:
a paged KV cache, continuous batching, prefix caching and speculative
decoding, on hand-written AVX-512/AVX2 kernels. It loads Hugging Face
checkpoints (Llama, SmolLM2, Qwen2) and serves an OpenAI-compatible API.

Work in progress; benchmarks and the full write-up are coming.
