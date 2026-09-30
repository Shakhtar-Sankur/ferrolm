//! ferrolm: an LLM inference server written from scratch in Rust.

pub mod awq;
pub mod bench;
pub mod config;
pub mod encoder;
pub mod engine;
pub mod eval;
pub mod json;
pub mod kernels;
pub mod kv;
pub mod model;
pub mod pool;
pub mod quant;
pub mod rng;
pub mod safetensors;
pub mod sampler;
pub mod server;
pub mod tokenizer;
pub mod unicode;
pub mod vector;
pub mod wordpiece;
