//! Retrieval-augmented generation: split documents into passages, embed
//! them with a BERT encoder, index the vectors (exactly or with HNSW),
//! retrieve the passages nearest a question and build a grounded prompt.
//! Also the metrics used to evaluate it: nDCG@k and recall@k for
//! retrieval (as BEIR computes them), exact match and F1 for answers (as
//! the SQuAD script computes them).

use crate::encoder::Encoder;
use crate::pool::Pool;
use crate::vector::flat::FlatIndex;
use crate::vector::hnsw::{Hnsw, HnswParams};
use crate::vector::{Metric, Neighbor};
use std::collections::HashMap;

/// Texts embedded per forward pass while indexing.
const BATCH: usize = 32;

/// A piece of a document small enough to embed whole.
#[derive(Clone, Debug)]
pub struct Passage {
    pub doc: u32,
    pub text: String,
}

#[derive(Clone, Copy, Debug)]
pub enum IndexKind {
    Flat,
    Hnsw {
        m: usize,
        ef_construction: usize,
        ef: usize,
    },
}

pub struct Retriever {
    pub encoder: Encoder,
    /// Prepended to queries (bge models are trained with an instruction).
    pub query_prefix: String,
    pub passages: Vec<Passage>,
    /// Exact search over all passage vectors, always kept (it is also the
    /// ground truth when measuring HNSW).
    flat: FlatIndex,
    /// An HNSW graph and its search width, when chosen.
    hnsw: Option<(Hnsw, usize)>,
    /// Seconds spent embedding the passages, and their token count.
    pub embed_seconds: f64,
    pub embed_tokens: usize,
}

/// Splits `text` into passages of at most `max_tokens` encoder tokens
/// (counting `[CLS]` and `[SEP]`) on word boundaries, consecutive passages
/// sharing about `overlap` tokens. `max_tokens == 0` keeps the text whole.
pub fn chunk(encoder: &Encoder, text: &str, max_tokens: usize, overlap: usize) -> Vec<String> {
    let words: Vec<&str> = text.split_whitespace().collect();
    if max_tokens == 0 || words.is_empty() {
        return vec![text.to_string()];
    }
    let budget = max_tokens.saturating_sub(2).max(1);
    let cost: Vec<usize> = words
        .iter()
        .map(|w| encoder.tokenizer.encode(w, usize::MAX).len().saturating_sub(2).max(1))
        .collect();
    let mut out = Vec::new();
    let mut start = 0;
    while start < words.len() {
        let (mut end, mut used) = (start, 0);
        while end < words.len() && (end == start || used + cost[end] <= budget) {
            used += cost[end];
            end += 1;
        }
        out.push(words[start..end].join(" "));
        if end == words.len() {
            break;
        }
        // Step back over about `overlap` tokens, always moving forward.
        let mut back = end;
        let mut shared = 0;
        while back > start + 1 && shared + cost[back - 1] <= overlap {
            back -= 1;
            shared += cost[back];
        }
        start = back;
    }
    out
}

impl Retriever {
    /// Chunks, embeds and indexes `docs`.
    pub fn build(
        pool: &Pool,
        encoder: Encoder,
        docs: &[String],
        max_tokens: usize,
        overlap: usize,
        kind: IndexKind,
        query_prefix: &str,
    ) -> Retriever {
        let mut passages = Vec::new();
        for (d, text) in docs.iter().enumerate() {
            for p in chunk(&encoder, text, max_tokens, overlap) {
                passages.push(Passage { doc: d as u32, text: p });
            }
        }
        let texts: Vec<&str> = passages.iter().map(|p| p.text.as_str()).collect();
        let seqs = encoder.tokenize(&texts);
        let t = std::time::Instant::now();
        let vecs = encoder.embed_many(pool, &seqs, BATCH);
        let embed_seconds = t.elapsed().as_secs_f64();
        let dim = encoder.dim;
        let flat: Vec<f32> = vecs.concat();
        let mut exact = FlatIndex::new(Metric::InnerProduct, dim);
        exact.add(&flat);
        let hnsw = match kind {
            IndexKind::Flat => None,
            IndexKind::Hnsw { m, ef_construction, ef } => {
                let p = HnswParams {
                    m,
                    ef_construction,
                    seed: 1,
                };
                Some((Hnsw::build(pool, Metric::InnerProduct, dim, &flat, p), ef))
            }
        };
        Retriever {
            encoder,
            query_prefix: query_prefix.to_string(),
            passages,
            flat: exact,
            hnsw,
            embed_seconds,
            embed_tokens: seqs.iter().map(Vec::len).sum(),
        }
    }

    /// Unit-length query vectors, with the query prefix.
    pub fn embed_queries(&self, pool: &Pool, queries: &[&str]) -> Vec<f32> {
        let texts: Vec<String> = queries.iter().map(|q| format!("{}{q}", self.query_prefix)).collect();
        let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
        self.encoder
            .embed_many(pool, &self.encoder.tokenize(&refs), BATCH)
            .concat()
    }

    /// Sets the HNSW search width (no effect on exact search).
    pub fn set_ef(&mut self, ef: usize) {
        if let Some((_, e)) = &mut self.hnsw {
            *e = ef;
        }
    }

    /// The `k` nearest passages to each query vector: (passage, cosine),
    /// from the index, or by exact search.
    pub fn search_vectors(&self, pool: &Pool, q: &[f32], k: usize, exact: bool) -> Vec<Vec<(u32, f32)>> {
        let found: Vec<Vec<Neighbor>> = match &self.hnsw {
            Some((h, ef)) if !exact => h.search_batch(pool, q, k, (*ef).max(k)),
            _ => self.flat.search_batch(pool, q, k),
        };
        found
            .into_iter()
            .map(|v| v.into_iter().map(|n| (n.id, -n.distance)).collect())
            .collect()
    }

    /// The `k` nearest passages to each query.
    pub fn search(&self, pool: &Pool, queries: &[&str], k: usize) -> Vec<Vec<(u32, f32)>> {
        let q = self.embed_queries(pool, queries);
        self.search_vectors(pool, &q, k, false)
    }

    /// The `k` best documents for each query vector, scoring a document by
    /// its best passage.
    pub fn search_docs(&self, pool: &Pool, q: &[f32], k: usize, exact: bool) -> Vec<Vec<(u32, f32)>> {
        let per_doc = self.passages.len().div_ceil(self.docs().max(1)).max(1);
        self.search_vectors(pool, q, k * per_doc.min(8), exact)
            .into_iter()
            .map(|hits| {
                let mut seen = HashMap::new();
                let mut out: Vec<(u32, f32)> = Vec::new();
                for (p, s) in hits {
                    let d = self.passages[p as usize].doc;
                    if seen.insert(d, ()).is_none() {
                        out.push((d, s));
                    }
                }
                out.truncate(k);
                out
            })
            .collect()
    }

    fn docs(&self) -> usize {
        self.passages.last().map_or(0, |p| p.doc as usize + 1)
    }
}

/// A prompt that asks for an answer grounded in `passages` (none: closed
/// book).
pub fn prompt(question: &str, passages: &[&str]) -> String {
    if passages.is_empty() {
        return format!("Answer the question with a short phrase, not a sentence.\n\nQuestion: {question}");
    }
    let mut s = String::from(
        "Answer the question using the passages below. Reply with a short phrase copied from them, not a sentence.\n\n",
    );
    for (i, p) in passages.iter().enumerate() {
        s.push_str(&format!("[{}] {p}\n\n", i + 1));
    }
    s.push_str(&format!("Question: {question}"));
    s
}

/// Chat messages asking `question`, after one worked example that shows the
/// expected form of answer (a short span, as in SQuAD). The example is
/// invented, not drawn from any evaluation set.
pub fn messages(question: &str, passages: &[&str]) -> Vec<(String, String)> {
    const DEMO_PASSAGE: &str = "The Eiffel Tower was completed in 1889 as the entrance arch to the World's Fair in Paris. \
        At 330 metres it was the tallest man-made structure in the world until 1930.";
    let demo = if passages.is_empty() {
        prompt("In which city is the Eiffel Tower?", &[])
    } else {
        prompt("When was the Eiffel Tower completed?", &[DEMO_PASSAGE])
    };
    let answer = if passages.is_empty() { "Paris" } else { "1889" };
    vec![
        ("user".into(), demo),
        ("assistant".into(), answer.into()),
        ("user".into(), prompt(question, passages)),
    ]
}

/// nDCG@k of a ranking against graded relevance judgements, as BEIR
/// computes it with pytrec_eval: gain = relevance, discount log2(rank + 1),
/// ideal ranking from all judged-relevant documents.
pub fn ndcg(ranked: &[&str], rel: &HashMap<String, u32>, k: usize) -> f64 {
    let dcg: f64 = ranked
        .iter()
        .take(k)
        .enumerate()
        .map(|(i, d)| f64::from(*rel.get(*d).unwrap_or(&0)) / (i as f64 + 2.0).log2())
        .sum();
    let mut ideal: Vec<u32> = rel.values().copied().filter(|&r| r > 0).collect();
    ideal.sort_unstable_by(|a, b| b.cmp(a));
    let idcg: f64 = ideal
        .iter()
        .take(k)
        .enumerate()
        .map(|(i, &r)| f64::from(r) / (i as f64 + 2.0).log2())
        .sum();
    if idcg == 0.0 { 0.0 } else { dcg / idcg }
}

/// Fraction of the relevant documents found in the first `k`.
pub fn recall_at(ranked: &[&str], rel: &HashMap<String, u32>, k: usize) -> f64 {
    let total = rel.values().filter(|&&r| r > 0).count();
    if total == 0 {
        return 0.0;
    }
    let hit = ranked
        .iter()
        .take(k)
        .filter(|d| rel.get(**d).is_some_and(|&r| r > 0))
        .count();
    hit as f64 / total as f64
}

/// SQuAD answer normalisation: lower case, no punctuation, no articles,
/// single spaces.
pub fn normalize_answer(s: &str) -> String {
    let lower: String = s.to_lowercase().chars().filter(|c| !c.is_ascii_punctuation()).collect();
    lower
        .split_whitespace()
        .filter(|w| !matches!(*w, "a" | "an" | "the"))
        .collect::<Vec<_>>()
        .join(" ")
}

/// SQuAD exact match and token F1 of `pred` against the best of `golds`.
pub fn squad_scores(pred: &str, golds: &[String]) -> (f64, f64) {
    let p = normalize_answer(pred);
    let mut em = 0f64;
    let mut f1 = 0f64;
    for g in golds {
        let g = normalize_answer(g);
        if p == g {
            em = 1.0;
        }
        let pt: Vec<&str> = p.split_whitespace().collect();
        let gt: Vec<&str> = g.split_whitespace().collect();
        let mut counts: HashMap<&str, i32> = HashMap::new();
        for t in &gt {
            *counts.entry(t).or_default() += 1;
        }
        let mut common = 0;
        for t in &pt {
            if let Some(c) = counts.get_mut(t)
                && *c > 0
            {
                *c -= 1;
                common += 1;
            }
        }
        if common > 0 {
            let prec = common as f64 / pt.len() as f64;
            let rec = common as f64 / gt.len() as f64;
            f1 = f1.max(2.0 * prec * rec / (prec + rec));
        }
    }
    (em, f1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ndcg_and_recall() {
        let rel: HashMap<String, u32> = [("a".to_string(), 1), ("b".to_string(), 1)].into();
        assert!((ndcg(&["a", "b", "c"], &rel, 10) - 1.0).abs() < 1e-12);
        let want = (1.0 / 3f64.log2()) / (1.0 + 1.0 / 3f64.log2());
        assert!((ndcg(&["c", "a", "x"], &rel, 10) - want).abs() < 1e-12);
        assert_eq!(ndcg(&["c", "d"], &rel, 10), 0.0);
        assert_eq!(recall_at(&["c", "a", "b"], &rel, 2), 0.5);
    }

    #[test]
    fn squad_metrics_match_the_official_script() {
        let g = vec!["the Denver Broncos".to_string(), "Broncos".to_string()];
        assert_eq!(squad_scores("Denver Broncos.", &g), (1.0, 1.0));
        let (em, f1) = squad_scores("the Broncos of Denver", &g);
        assert_eq!(em, 0.0);
        // best: "denver broncos" vs "broncos of denver": 2 common, p=2/3, r=1.
        assert!((f1 - 0.8).abs() < 1e-12);
        assert_eq!(squad_scores("Carolina", &g), (0.0, 0.0));
    }
}
