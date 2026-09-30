//! BERT's WordPiece tokenization, compatible with Hugging Face
//! `tokenizers` (`BertNormalizer`, `BertPreTokenizer`, `WordPiece`, and a
//! `[CLS] … [SEP]` template), for embedding models such as bge and
//! MiniLM.

use crate::json::{self, Json};
use crate::unicode::{bert_control, bert_mark, bert_punctuation, bert_space, nfd};
use std::collections::HashMap;
use std::path::Path;

pub struct WordPiece {
    vocab: HashMap<String, u32>,
    lowercase: bool,
    strip_accents: bool,
    chinese_chars: bool,
    clean_text: bool,
    prefix: String,
    max_chars: usize,
    unk: u32,
    cls: u32,
    sep: u32,
}

impl WordPiece {
    pub fn load(dir: &Path) -> Result<WordPiece, String> {
        let path = dir.join("tokenizer.json");
        let text = std::fs::read_to_string(&path).map_err(|e| format!("{path:?}: {e}"))?;
        WordPiece::from_json(&json::parse(&text)?)
    }

    pub fn from_json(v: &Json) -> Result<WordPiece, String> {
        let model = v.get("model").ok_or("tokenizer.json without model")?;
        if model.get("type").and_then(Json::as_str) != Some("WordPiece") {
            return Err("not a WordPiece tokenizer".into());
        }
        let vocab: HashMap<String, u32> = model
            .get("vocab")
            .and_then(Json::as_obj)
            .ok_or("WordPiece without vocab")?
            .iter()
            .filter_map(|(k, id)| id.as_usize().map(|i| (k.clone(), i as u32)))
            .collect();
        let norm = v.get("normalizer");
        let flag = |k: &str| norm.and_then(|n| n.get(k)).and_then(Json::as_bool);
        let lowercase = flag("lowercase").unwrap_or(true);
        let id = |t: &str| vocab.get(t).copied().ok_or_else(|| format!("vocabulary has no {t}"));
        let unk_tok = model.get("unk_token").and_then(Json::as_str).unwrap_or("[UNK]");
        Ok(WordPiece {
            lowercase,
            strip_accents: flag("strip_accents").unwrap_or(lowercase),
            chinese_chars: flag("handle_chinese_chars").unwrap_or(true),
            clean_text: flag("clean_text").unwrap_or(true),
            prefix: model
                .get("continuing_subword_prefix")
                .and_then(Json::as_str)
                .unwrap_or("##")
                .to_string(),
            max_chars: model
                .get("max_input_chars_per_word")
                .and_then(Json::as_usize)
                .unwrap_or(100),
            unk: id(unk_tok)?,
            cls: id("[CLS]")?,
            sep: id("[SEP]")?,
            vocab,
        })
    }

    fn normalize(&self, text: &str) -> String {
        let mut s = String::with_capacity(text.len());
        for c in text.chars() {
            if self.clean_text {
                if bert_control(c) {
                    continue;
                }
                if bert_space(c) {
                    s.push(' ');
                    continue;
                }
            }
            if self.chinese_chars && is_cjk(c) {
                s.push(' ');
                s.push(c);
                s.push(' ');
            } else {
                s.push(c);
            }
        }
        if self.strip_accents {
            s = nfd(&s).chars().filter(|&c| !bert_mark(c)).collect();
        }
        if self.lowercase {
            s = s.to_lowercase();
        }
        s
    }

    /// Token ids of `text`, as `[CLS] … [SEP]`, at most `max_len` in all.
    pub fn encode(&self, text: &str, max_len: usize) -> Vec<u32> {
        let norm = self.normalize(text);
        let mut out = vec![self.cls];
        // Pre-tokenize: split on whitespace, isolate punctuation.
        let mut word = String::new();
        let mut words = Vec::new();
        for c in norm.chars() {
            if c.is_whitespace() {
                if !word.is_empty() {
                    words.push(std::mem::take(&mut word));
                }
            } else if bert_punctuation(c) {
                if !word.is_empty() {
                    words.push(std::mem::take(&mut word));
                }
                words.push(c.to_string());
            } else {
                word.push(c);
            }
        }
        if !word.is_empty() {
            words.push(word);
        }
        for w in &words {
            self.word(w, &mut out);
        }
        out.truncate(max_len.saturating_sub(1).max(1));
        out.push(self.sep);
        out
    }

    /// Greedy longest-match-first pieces of one word.
    fn word(&self, w: &str, out: &mut Vec<u32>) {
        let chars: Vec<char> = w.chars().collect();
        if chars.len() > self.max_chars {
            out.push(self.unk);
            return;
        }
        let mut pieces = Vec::new();
        let mut start = 0;
        while start < chars.len() {
            let mut end = chars.len();
            let mut found = None;
            while start < end {
                let mut sub: String = chars[start..end].iter().collect();
                if start > 0 {
                    sub.insert_str(0, &self.prefix);
                }
                if let Some(&id) = self.vocab.get(&sub) {
                    found = Some(id);
                    break;
                }
                end -= 1;
            }
            match found {
                Some(id) => {
                    pieces.push(id);
                    start = end;
                }
                None => {
                    out.push(self.unk);
                    return;
                }
            }
        }
        out.extend(pieces);
    }
}

fn is_cjk(c: char) -> bool {
    matches!(c as u32,
        0x4E00..=0x9FFF | 0x3400..=0x4DBF | 0x20000..=0x2A6DF | 0x2A700..=0x2B73F
        | 0x2B740..=0x2B81F | 0x2B820..=0x2CEAF | 0xF900..=0xFAFF | 0x2F800..=0x2FA1F)
}
