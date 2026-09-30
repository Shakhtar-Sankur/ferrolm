//! BERT-style text encoders for embeddings (bge, MiniLM, E5): token,
//! position and segment embeddings with LayerNorm, then post-norm
//! transformer layers with bidirectional attention and GELU, pooled to one
//! L2-normalised vector per text.
//!
//! Many texts are encoded in one pass: their tokens are stacked for every
//! matrix multiply and attention stays within each text, so there is no
//! padding and, as in the decoder, a text's vector does not depend on what
//! it was batched with.

use crate::json::{self, Json};
use crate::kernels::{self, Matrix};
use crate::pool::{Out, Pool};
use crate::safetensors::Checkpoint;
use crate::wordpiece::WordPiece;
use std::path::Path;

struct Layer {
    wqkv: Matrix,
    bqkv: Vec<f32>,
    wo: Matrix,
    bo: Vec<f32>,
    ln1: (Vec<f32>, Vec<f32>),
    w1: Matrix,
    b1: Vec<f32>,
    w2: Matrix,
    b2: Vec<f32>,
    ln2: (Vec<f32>, Vec<f32>),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Pooling {
    Cls,
    Mean,
}

pub struct Encoder {
    pub dim: usize,
    heads: usize,
    inter: usize,
    eps: f32,
    pub max_len: usize,
    word: Vec<f32>,
    position: Vec<f32>,
    token_type: Vec<f32>,
    ln_emb: (Vec<f32>, Vec<f32>),
    layers: Vec<Layer>,
    pub pooling: Pooling,
    pub tokenizer: WordPiece,
}

impl Encoder {
    pub fn load(dir: &Path) -> Result<Encoder, String> {
        let cfg_text = std::fs::read_to_string(dir.join("config.json")).map_err(|e| format!("config.json: {e}"))?;
        let cfg = json::parse(&cfg_text)?;
        if cfg.get("model_type").and_then(Json::as_str) != Some("bert") {
            return Err("only BERT encoders are supported".into());
        }
        let num = |k: &str| {
            cfg.get(k)
                .and_then(Json::as_usize)
                .ok_or_else(|| format!("config.json: missing {k}"))
        };
        let dim = num("hidden_size")?;
        let heads = num("num_attention_heads")?;
        let inter = num("intermediate_size")?;
        let layers_n = num("num_hidden_layers")?;
        let eps = cfg.get("layer_norm_eps").and_then(Json::as_f64).unwrap_or(1e-12) as f32;
        let ck = Checkpoint::open(dir)?;
        // Checkpoints saved from BertModel have no prefix; others use "bert.".
        let pre = if ck.tensors.contains_key("embeddings.word_embeddings.weight") {
            ""
        } else {
            "bert."
        };
        let v = |n: &str| ck.f32(&format!("{pre}{n}"));
        let mat =
            |n: &str, rows: usize, cols: usize| -> Result<Matrix, String> { Ok(Matrix::from_f32(rows, cols, &v(n)?)) };
        let mut layers = Vec::with_capacity(layers_n);
        for i in 0..layers_n {
            let p = format!("encoder.layer.{i}");
            let a = format!("{p}.attention");
            let mut qkv = v(&format!("{a}.self.query.weight"))?;
            qkv.extend(v(&format!("{a}.self.key.weight"))?);
            qkv.extend(v(&format!("{a}.self.value.weight"))?);
            let mut bqkv = v(&format!("{a}.self.query.bias"))?;
            bqkv.extend(v(&format!("{a}.self.key.bias"))?);
            bqkv.extend(v(&format!("{a}.self.value.bias"))?);
            layers.push(Layer {
                wqkv: Matrix::from_f32(3 * dim, dim, &qkv),
                bqkv,
                wo: mat(&format!("{a}.output.dense.weight"), dim, dim)?,
                bo: v(&format!("{a}.output.dense.bias"))?,
                ln1: (
                    v(&format!("{a}.output.LayerNorm.weight"))?,
                    v(&format!("{a}.output.LayerNorm.bias"))?,
                ),
                w1: mat(&format!("{p}.intermediate.dense.weight"), inter, dim)?,
                b1: v(&format!("{p}.intermediate.dense.bias"))?,
                w2: mat(&format!("{p}.output.dense.weight"), dim, inter)?,
                b2: v(&format!("{p}.output.dense.bias"))?,
                ln2: (
                    v(&format!("{p}.output.LayerNorm.weight"))?,
                    v(&format!("{p}.output.LayerNorm.bias"))?,
                ),
            });
        }
        let pooling = match std::fs::read_to_string(dir.join("1_Pooling/config.json")) {
            Ok(t) => {
                let p = json::parse(&t)?;
                if p.get("pooling_mode_cls_token").and_then(Json::as_bool) == Some(true) {
                    Pooling::Cls
                } else {
                    Pooling::Mean
                }
            }
            Err(_) => Pooling::Cls,
        };
        let position = v("embeddings.position_embeddings.weight")?;
        Ok(Encoder {
            dim,
            heads,
            inter,
            eps,
            max_len: position.len() / dim,
            word: v("embeddings.word_embeddings.weight")?,
            position,
            token_type: v("embeddings.token_type_embeddings.weight")?,
            ln_emb: (v("embeddings.LayerNorm.weight")?, v("embeddings.LayerNorm.bias")?),
            layers,
            pooling,
            tokenizer: WordPiece::load(dir)?,
        })
    }

    /// Token ids for each text (with `[CLS]`/`[SEP]`, truncated).
    pub fn tokenize(&self, texts: &[&str]) -> Vec<Vec<u32>> {
        texts.iter().map(|t| self.tokenizer.encode(t, self.max_len)).collect()
    }

    /// One unit-length embedding per text.
    pub fn embed(&self, pool: &Pool, texts: &[&str]) -> Vec<Vec<f32>> {
        self.embed_tokens(pool, &self.tokenize(texts))
    }

    pub fn embed_tokens(&self, pool: &Pool, seqs: &[Vec<u32>]) -> Vec<Vec<f32>> {
        let d = self.dim;
        let t: usize = seqs.iter().map(Vec::len).sum();
        if t == 0 {
            return vec![Vec::new(); seqs.len()];
        }
        let mut x = vec![0f32; t * d];
        let mut starts = Vec::with_capacity(seqs.len());
        let mut r = 0;
        for s in seqs {
            starts.push(r);
            for (p, &tok) in s.iter().enumerate() {
                let row = &mut x[r * d..(r + 1) * d];
                let w = &self.word[tok as usize * d..(tok as usize + 1) * d];
                let pe = &self.position[p * d..(p + 1) * d];
                for i in 0..d {
                    row[i] = w[i] + pe[i] + self.token_type[i];
                }
                r += 1;
            }
        }
        layer_norm_rows(&mut x, &self.ln_emb.0, &self.ln_emb.1, self.eps);
        let mut qkv = vec![0f32; t * 3 * d];
        let mut att = vec![0f32; t * d];
        let mut proj = vec![0f32; t * d];
        let mut h1 = vec![0f32; t * self.inter];
        for l in &self.layers {
            kernels::matmul(pool, &x, t, &l.wqkv, Some(&l.bqkv), &mut qkv);
            self.attention(pool, seqs, &starts, &qkv, &mut att);
            kernels::matmul(pool, &att, t, &l.wo, Some(&l.bo), &mut proj);
            kernels::add(&mut x, &proj);
            layer_norm_rows(&mut x, &l.ln1.0, &l.ln1.1, self.eps);
            kernels::matmul(pool, &x, t, &l.w1, Some(&l.b1), &mut h1);
            for v in h1.iter_mut() {
                *v = gelu(*v);
            }
            kernels::matmul(pool, &h1, t, &l.w2, Some(&l.b2), &mut proj);
            kernels::add(&mut x, &proj);
            layer_norm_rows(&mut x, &l.ln2.0, &l.ln2.1, self.eps);
        }
        seqs.iter()
            .zip(&starts)
            .map(|(s, &st)| {
                let mut e = match self.pooling {
                    Pooling::Cls => x[st * d..(st + 1) * d].to_vec(),
                    Pooling::Mean => {
                        let mut m = vec![0f32; d];
                        for row in x[st * d..(st + s.len()) * d].chunks_exact(d) {
                            kernels::add(&mut m, row);
                        }
                        m.iter().map(|v| v / s.len() as f32).collect()
                    }
                };
                let n = e.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
                for v in &mut e {
                    *v /= n;
                }
                e
            })
            .collect()
    }

    /// Bidirectional attention within each sequence.
    fn attention(&self, pool: &Pool, seqs: &[Vec<u32>], starts: &[usize], qkv: &[f32], att: &mut [f32]) {
        let d = self.dim;
        let hd = d / self.heads;
        let scale = 1.0 / (hd as f32).sqrt();
        let rows: Vec<(usize, usize, usize)> = seqs
            .iter()
            .zip(starts)
            .flat_map(|(s, &st)| (0..s.len()).map(move |i| (st + i, st, s.len())))
            .collect();
        let out = Out::new(att);
        pool.run(rows.len() * self.heads, &|task| {
            let (r, st, len) = rows[task / self.heads];
            let h = task % self.heads;
            let q = &qkv[r * 3 * d + h * hd..][..hd];
            let mut s = Vec::with_capacity(len);
            for j in 0..len {
                let k = &qkv[(st + j) * 3 * d + d + h * hd..][..hd];
                s.push(q.iter().zip(k).map(|(a, b)| a * b).sum::<f32>() * scale);
            }
            let mx = s.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0f32;
            for v in &mut s {
                *v = (*v - mx).exp();
                sum += *v;
            }
            // SAFETY: each task writes its own (row, head) slice.
            let o = unsafe { out.slice(r * d + h * hd, hd) };
            o.fill(0.0);
            for (j, &p) in s.iter().enumerate() {
                let v = &qkv[(st + j) * 3 * d + 2 * d + h * hd..][..hd];
                for (a, &b) in o.iter_mut().zip(v) {
                    *a += p / sum * b;
                }
            }
        });
    }
}

fn layer_norm_rows(x: &mut [f32], g: &[f32], b: &[f32], eps: f32) {
    let d = g.len();
    for row in x.chunks_exact_mut(d) {
        let mean = row.iter().map(|&v| f64::from(v)).sum::<f64>() / d as f64;
        let var = row.iter().map(|&v| (f64::from(v) - mean).powi(2)).sum::<f64>() / d as f64;
        let inv = 1.0 / (var + f64::from(eps)).sqrt();
        for ((v, &gg), &bb) in row.iter_mut().zip(g).zip(b) {
            *v = ((f64::from(*v) - mean) * inv) as f32 * gg + bb;
        }
    }
}

/// GELU with the exact error function, as BERT uses.
fn gelu(x: f32) -> f32 {
    let x = f64::from(x);
    (0.5 * x * (1.0 + erf(x / std::f64::consts::SQRT_2))) as f32
}

/// erf to about 1e-13: its Taylor series for |x| < 2.5, and a continued
/// fraction for erfc beyond, where that converges quickly.
fn erf(x: f64) -> f64 {
    if x.abs() < 2.5 {
        let x2 = x * x;
        let mut term = x;
        let mut sum = x;
        for n in 1..120 {
            term *= -x2 / n as f64;
            sum += term / (2 * n + 1) as f64;
        }
        return sum * 2.0 / std::f64::consts::PI.sqrt();
    }
    let sign = x.signum();
    sign * (1.0 - erfc_pos(x.abs()))
}

/// erfc for x >= 2.5, by a continued fraction (Lentz's method).
fn erfc_pos(x: f64) -> f64 {
    if x > 27.0 {
        return 0.0;
    }
    // erfc(x) = exp(-x²)/sqrt(pi) * 1/(x + 1/2/(x + 1/(x + 3/2/(x + ...))))
    let tiny = 1e-300;
    let mut f = x;
    let mut c = x;
    let mut d = 0.0;
    for n in 1..300 {
        let a = n as f64 / 2.0;
        d = x + a * d;
        if d.abs() < tiny {
            d = tiny;
        }
        c = x + a / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = 1.0 / d;
        let delta = c * d;
        f *= delta;
        if (delta - 1.0).abs() < 1e-16 {
            break;
        }
    }
    (-x * x).exp() / std::f64::consts::PI.sqrt() / f
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn erf_matches_known_values() {
        for (x, want) in [
            (0.0, 0.0),
            (0.3, 0.328_626_759_459_127_4),
            (0.5, 0.520_499_877_813_046_5),
            (1.0, 0.842_700_792_949_714_9),
            (2.0, 0.995_322_265_018_952_7),
            (-1.5, -0.966_105_146_475_310_7),
            (4.0, 0.999_999_984_582_742_1),
        ] {
            assert!((erf(x) - want).abs() < 1e-12, "erf({x}) = {} want {want}", erf(x));
        }
    }
}
