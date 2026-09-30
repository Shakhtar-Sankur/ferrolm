//! The model hyperparameters, read from a Hugging Face `config.json`.
//! Supported families: Llama (Llama 2/3, SmolLM2, TinyLlama-style
//! checkpoints), Mistral without sliding windows, and Qwen2.

use crate::json::{self, Json};
use std::path::Path;

#[derive(Clone, Debug, PartialEq)]
pub enum RopeScaling {
    None,
    /// Llama 3.1's frequency-dependent scaling.
    Llama3 {
        factor: f32,
        low_freq_factor: f32,
        high_freq_factor: f32,
        original_max_position: usize,
    },
}

#[derive(Clone, Debug)]
pub struct Config {
    pub arch: String,
    pub vocab: usize,
    pub hidden: usize,
    pub intermediate: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub rms_eps: f32,
    pub rope_theta: f32,
    pub rope_scaling: RopeScaling,
    pub max_position: usize,
    pub tie_embeddings: bool,
    /// Biases on the q, k and v projections (Qwen2, or `attention_bias`).
    pub qkv_bias: bool,
    pub eos: Vec<u32>,
    pub bos: Option<u32>,
}

impl Config {
    pub fn load(dir: &Path) -> Result<Config, String> {
        let path = dir.join("config.json");
        let text = std::fs::read_to_string(&path).map_err(|e| format!("{path:?}: {e}"))?;
        Config::parse(&json::parse(&text)?)
    }

    pub fn parse(v: &Json) -> Result<Config, String> {
        let num = |k: &str| v.get(k).and_then(Json::as_f64);
        let int = |k: &str| {
            v.get(k)
                .and_then(Json::as_usize)
                .ok_or_else(|| format!("config.json: missing {k}"))
        };
        let arch = v
            .get("model_type")
            .and_then(Json::as_str)
            .unwrap_or("llama")
            .to_string();
        if !["llama", "mistral", "qwen2"].contains(&arch.as_str()) {
            return Err(format!("unsupported model_type {arch:?}"));
        }
        if arch != "llama" && v.get("use_sliding_window").and_then(Json::as_bool) == Some(true) {
            return Err("sliding-window attention is not supported".into());
        }
        let hidden = int("hidden_size")?;
        let heads = int("num_attention_heads")?;
        let kv_heads = v
            .get("num_key_value_heads")
            .and_then(Json::as_usize)
            .unwrap_or(heads);
        let head_dim = v
            .get("head_dim")
            .and_then(Json::as_usize)
            .unwrap_or(hidden / heads);
        if heads % kv_heads != 0 || head_dim % 16 != 0 || hidden % 16 != 0 {
            return Err(format!(
                "unsupported shape: {heads} heads, {kv_heads} kv heads, head_dim {head_dim}, hidden {hidden}"
            ));
        }
        // transformers 4.x writes `rope_theta` and `rope_scaling`; 5.x folds
        // both into `rope_parameters`.
        let rope = v.get("rope_parameters").filter(|r| r.as_obj().is_some());
        let rope_theta = rope
            .and_then(|r| r.get("rope_theta"))
            .or_else(|| v.get("rope_theta"))
            .and_then(Json::as_f64)
            .unwrap_or(10000.0) as f32;
        let rope_scaling = match rope.or_else(|| v.get("rope_scaling")) {
            None | Some(Json::Null) => RopeScaling::None,
            Some(s) => {
                let kind = s
                    .get("rope_type")
                    .or_else(|| s.get("type"))
                    .and_then(Json::as_str)
                    .unwrap_or("");
                let f = |k: &str| s.get(k).and_then(Json::as_f64).map(|x| x as f32);
                match kind {
                    "llama3" => RopeScaling::Llama3 {
                        factor: f("factor").ok_or("llama3 rope_scaling without factor")?,
                        low_freq_factor: f("low_freq_factor").unwrap_or(1.0),
                        high_freq_factor: f("high_freq_factor").unwrap_or(4.0),
                        original_max_position: s
                            .get("original_max_position_embeddings")
                            .and_then(Json::as_usize)
                            .unwrap_or(8192),
                    },
                    "default" => RopeScaling::None,
                    other => return Err(format!("unsupported rope_scaling {other:?}")),
                }
            }
        };
        let ids = |k: &str| -> Vec<u32> {
            match v.get(k) {
                Some(Json::Num(n)) => vec![*n as u32],
                Some(Json::Arr(a)) => a.iter().filter_map(Json::as_usize).map(|n| n as u32).collect(),
                _ => Vec::new(),
            }
        };
        Ok(Config {
            vocab: int("vocab_size")?,
            hidden,
            intermediate: int("intermediate_size")?,
            layers: int("num_hidden_layers")?,
            heads,
            kv_heads,
            head_dim,
            rms_eps: num("rms_norm_eps").unwrap_or(1e-6) as f32,
            rope_theta,
            rope_scaling,
            max_position: v
                .get("max_position_embeddings")
                .and_then(Json::as_usize)
                .unwrap_or(4096),
            tie_embeddings: v
                .get("tie_word_embeddings")
                .and_then(Json::as_bool)
                .unwrap_or(false),
            qkv_bias: arch == "qwen2"
                || v.get("attention_bias").and_then(Json::as_bool) == Some(true),
            eos: ids("eos_token_id"),
            bos: ids("bos_token_id").first().copied(),
            arch,
        })
    }

    /// Parameters, counting a tied embedding once.
    pub fn params(&self) -> usize {
        let q = self.heads * self.head_dim;
        let kv = self.kv_heads * self.head_dim;
        let attn = self.hidden * (q + 2 * kv) + q * self.hidden;
        let mlp = 3 * self.hidden * self.intermediate;
        let emb = self.vocab * self.hidden * if self.tie_embeddings { 1 } else { 2 };
        self.layers * (attn + mlp + 2 * self.hidden) + emb + self.hidden
    }

    /// Bytes of K and V cache per token.
    pub fn kv_bytes_per_token(&self) -> usize {
        2 * self.layers * self.kv_heads * self.head_dim * 4
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_rope_parameters_from_transformers_5_configs() {
        let v = json::parse(
            r#"{"model_type":"qwen2","vocab_size":8,"hidden_size":64,"intermediate_size":64,
            "num_hidden_layers":1,"num_attention_heads":4,"num_key_value_heads":2,
            "rope_parameters":{"rope_theta":1000000.0,"rope_type":"default"}}"#,
        )
        .unwrap();
        let c = Config::parse(&v).unwrap();
        assert_eq!((c.rope_theta, c.rope_scaling.clone(), c.qkv_bias), (1e6, RopeScaling::None, true));
    }

    #[test]
    fn parses_a_llama3_config() {
        let v = json::parse(
            r#"{"model_type":"llama","vocab_size":128256,"hidden_size":2048,"intermediate_size":8192,
            "num_hidden_layers":16,"num_attention_heads":32,"num_key_value_heads":8,"rms_norm_eps":1e-5,
            "rope_theta":500000.0,"max_position_embeddings":131072,"tie_word_embeddings":true,
            "rope_scaling":{"factor":32.0,"high_freq_factor":4.0,"low_freq_factor":1.0,
            "original_max_position_embeddings":8192,"rope_type":"llama3"},
            "bos_token_id":128000,"eos_token_id":[128001,128008,128009]}"#,
        )
        .unwrap();
        let c = Config::parse(&v).unwrap();
        assert_eq!((c.head_dim, c.kv_heads, c.eos.len()), (64, 8, 3));
        assert!(matches!(c.rope_scaling, RopeScaling::Llama3 { factor: 32.0, .. }));
        // Llama 3.2 1B has 1.24B parameters.
        assert_eq!(c.params() / 10_000_000, 123);
    }
}
