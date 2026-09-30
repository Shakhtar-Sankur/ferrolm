//! Byte-level BPE tokenization compatible with Hugging Face `tokenizers`
//! for the `tokenizer.json` files of GPT-2-style models: SmolLM2, Llama 3
//! and Qwen2. Supported pieces: added (special) tokens, the `ByteLevel`,
//! `Digits` and `Split` pre-tokenizers with the GPT-2, Llama 3 and Qwen2
//! split patterns, BPE with `ignore_merges`, and `TemplateProcessing`
//! prefixes. Anything else is reported as unsupported rather than
//! tokenized differently.

use crate::json::{self, Json};
use crate::unicode::{is_letter, is_number};
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::Mutex;

#[derive(Clone, Copy, Debug, PartialEq)]
enum Pattern {
    /// GPT-2's pattern (also used by `ByteLevel` with `use_regex`).
    Gpt2,
    /// Llama 3's pattern; Qwen2's differs only in taking one digit at a time.
    Llama3 { max_digits: usize },
}

#[derive(Clone, Debug, PartialEq)]
enum Pre {
    Digits { individual: bool },
    Split(Pattern),
    ByteLevel { add_prefix_space: bool, regex: bool },
}

#[derive(Clone, Debug)]
struct Added {
    id: u32,
    content: String,
}

/// How chat messages become a prompt.
#[derive(Clone, Debug, PartialEq)]
pub enum ChatFormat {
    /// `<|im_start|>role\ncontent<|im_end|>\n`, with the template's default
    /// system prompt when the conversation has none.
    ChatMl { default_system: Option<String> },
    /// `<|start_header_id|>role<|end_header_id|>\n\ncontent<|eot_id|>`.
    Llama3,
    /// No template: contents joined with blank lines.
    Plain,
}

pub struct Tokenizer {
    token_bytes: Vec<Vec<u8>>,
    by_bytes: HashMap<Vec<u8>, u32>,
    merges: HashMap<(u32, u32), (u32, u32)>,
    byte_ids: [u32; 256],
    /// Longest first, so the longest added token wins at a position.
    added: Vec<Added>,
    special: HashSet<u32>,
    pre: Vec<Pre>,
    ignore_merges: bool,
    prefix: Vec<u32>,
    pub chat: ChatFormat,
    /// Tokens that end a generated reply, beyond the model's own EOS.
    pub stop_ids: Vec<u32>,
    cache: Mutex<HashMap<Vec<u8>, Vec<u32>>>,
}

/// GPT-2's reversible map from bytes to printable characters.
fn byte_chars() -> [char; 256] {
    let mut map = ['\0'; 256];
    let mut n = 0u32;
    for b in 0..256u32 {
        let printable = (0x21..=0x7E).contains(&b) || (0xA1..=0xAC).contains(&b) || (0xAE..=0xFF).contains(&b);
        map[b as usize] = if printable {
            char::from_u32(b).unwrap()
        } else {
            n += 1;
            char::from_u32(255 + n).unwrap()
        };
    }
    map
}

impl Tokenizer {
    /// Loads `tokenizer.json` (and, if present, `tokenizer_config.json` for
    /// the chat template) from a model directory.
    pub fn load(dir: &Path) -> Result<Tokenizer, String> {
        let path = dir.join("tokenizer.json");
        let text = std::fs::read_to_string(&path).map_err(|e| format!("{path:?}: {e}"))?;
        let mut t = Tokenizer::from_json(&json::parse(&text)?)?;
        if let Ok(cfg) = std::fs::read_to_string(dir.join("tokenizer_config.json")) {
            let cfg = json::parse(&cfg)?;
            let template = cfg.get("chat_template").and_then(Json::as_str).unwrap_or("");
            t.chat = chat_format(template);
        }
        let stop: &[&str] = match t.chat {
            ChatFormat::ChatMl { .. } => &["<|im_end|>"],
            ChatFormat::Llama3 => &["<|eot_id|>", "<|eom_id|>"],
            ChatFormat::Plain => &[],
        };
        t.stop_ids = stop.iter().filter_map(|s| t.token_id(s)).collect();
        Ok(t)
    }

    pub fn from_json(v: &Json) -> Result<Tokenizer, String> {
        if !matches!(v.get("normalizer"), None | Some(Json::Null)) {
            return Err("tokenizer normalizers are not supported".into());
        }
        let model = v.get("model").ok_or("tokenizer.json without model")?;
        if model.get("type").and_then(Json::as_str) != Some("BPE") {
            return Err("only BPE tokenizers are supported".into());
        }
        let chars = byte_chars();
        let char_byte: HashMap<char, u8> = chars.iter().enumerate().map(|(b, &c)| (c, b as u8)).collect();
        let to_bytes = |s: &str| -> Vec<u8> {
            if s.chars().all(|c| char_byte.contains_key(&c)) {
                s.chars().map(|c| char_byte[&c]).collect()
            } else {
                s.as_bytes().to_vec()
            }
        };
        let vocab = model.get("vocab").and_then(Json::as_obj).ok_or("BPE without vocab")?;
        let size = vocab.values().filter_map(Json::as_usize).max().map_or(0, |m| m + 1);
        let mut token_bytes = vec![Vec::new(); size];
        let mut by_bytes = HashMap::with_capacity(vocab.len());
        for (s, id) in vocab {
            let id = id.as_usize().ok_or("bad vocab id")?;
            let b = to_bytes(s);
            by_bytes.insert(b.clone(), id as u32);
            token_bytes[id] = b;
        }
        let mut byte_ids = [0u32; 256];
        for (b, &c) in chars.iter().enumerate() {
            byte_ids[b] = *by_bytes
                .get(&to_bytes(&c.to_string()))
                .ok_or_else(|| format!("vocabulary has no token for byte {b:#x}"))?;
        }
        let mut merges = HashMap::new();
        for (rank, m) in model.get("merges").map(Json::as_arr).unwrap_or(&[]).iter().enumerate() {
            let (a, b) = match m {
                Json::Str(s) => s.split_once(' ').ok_or("bad merge")?,
                Json::Arr(p) if p.len() == 2 => (
                    p[0].as_str().ok_or("bad merge")?,
                    p[1].as_str().ok_or("bad merge")?,
                ),
                _ => return Err("bad merge".into()),
            };
            let (ba, bb) = (to_bytes(a), to_bytes(b));
            let joined = [ba.as_slice(), bb.as_slice()].concat();
            let (Some(&ia), Some(&ib), Some(&ij)) = (by_bytes.get(&ba), by_bytes.get(&bb), by_bytes.get(&joined)) else {
                return Err(format!("merge {a:?} {b:?} refers to unknown tokens"));
            };
            merges.entry((ia, ib)).or_insert((rank as u32, ij));
        }
        let mut added = Vec::new();
        let mut special = HashSet::new();
        for a in v.get("added_tokens").map(Json::as_arr).unwrap_or(&[]) {
            let id = a.get("id").and_then(Json::as_usize).ok_or("added token without id")? as u32;
            let content = a.get("content").and_then(Json::as_str).ok_or("added token without content")?.to_string();
            let is_special = a.get("special").and_then(Json::as_bool).unwrap_or(false);
            if is_special {
                special.insert(id);
            }
            if token_bytes.len() <= id as usize {
                token_bytes.resize(id as usize + 1, Vec::new());
            }
            if token_bytes[id as usize].is_empty() {
                token_bytes[id as usize] = content.as_bytes().to_vec();
            }
            added.push(Added { id, content });
        }
        added.sort_by(|a, b| b.content.len().cmp(&a.content.len()));
        let pre = match v.get("pre_tokenizer") {
            None | Some(Json::Null) => Vec::new(),
            Some(p) => pre_steps(p)?,
        };
        let prefix = template_prefix(v.get("post_processor"), &by_bytes, &added);
        Ok(Tokenizer {
            token_bytes,
            by_bytes,
            merges,
            byte_ids,
            added,
            special,
            pre,
            ignore_merges: model.get("ignore_merges").and_then(Json::as_bool).unwrap_or(false),
            prefix,
            chat: ChatFormat::Plain,
            stop_ids: Vec::new(),
            cache: Mutex::new(HashMap::new()),
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.token_bytes.len()
    }

    pub fn token_id(&self, s: &str) -> Option<u32> {
        self.added
            .iter()
            .find(|a| a.content == s)
            .map(|a| a.id)
            .or_else(|| self.by_bytes.get(s.as_bytes()).copied())
    }

    pub fn is_special(&self, id: u32) -> bool {
        self.special.contains(&id)
    }

    /// Token ids for `text`; `add_special` adds the post-processor's prefix
    /// (a BOS token, for Llama 3).
    pub fn encode(&self, text: &str, add_special: bool) -> Vec<u32> {
        let mut out = if add_special { self.prefix.clone() } else { Vec::new() };
        let mut rest = text;
        while !rest.is_empty() {
            // The earliest added token, longest at that position.
            let hit = self
                .added
                .iter()
                .filter_map(|a| rest.find(a.content.as_str()).map(|i| (i, a)))
                .min_by_key(|(i, a)| (*i, std::cmp::Reverse(a.content.len())));
            match hit {
                Some((i, a)) => {
                    self.encode_plain(&rest[..i], &mut out);
                    out.push(a.id);
                    rest = &rest[i + a.content.len()..];
                }
                None => {
                    self.encode_plain(rest, &mut out);
                    break;
                }
            }
        }
        out
    }

    fn encode_plain(&self, text: &str, out: &mut Vec<u32>) {
        if text.is_empty() {
            return;
        }
        let mut pieces = vec![text.to_string()];
        for step in &self.pre {
            pieces = pieces.iter().flat_map(|p| split(step, p)).collect();
        }
        for p in pieces {
            self.bpe(p.as_bytes(), out);
        }
    }

    fn bpe(&self, word: &[u8], out: &mut Vec<u32>) {
        if self.ignore_merges
            && let Some(&id) = self.by_bytes.get(word)
        {
            out.push(id);
            return;
        }
        if let Some(ids) = self.cache.lock().unwrap().get(word) {
            out.extend_from_slice(ids);
            return;
        }
        let mut sym: Vec<u32> = word.iter().map(|&b| self.byte_ids[b as usize]).collect();
        loop {
            // The lowest-ranked adjacent pair, leftmost on ties.
            let best = sym
                .windows(2)
                .enumerate()
                .filter_map(|(i, w)| self.merges.get(&(w[0], w[1])).map(|&(rank, id)| (rank, i, id)))
                .min();
            let Some((_, i, id)) = best else { break };
            sym[i] = id;
            sym.remove(i + 1);
        }
        let mut cache = self.cache.lock().unwrap();
        if cache.len() > 100_000 {
            cache.clear();
        }
        cache.insert(word.to_vec(), sym.clone());
        out.extend(sym);
    }

    /// The bytes of `ids`, skipping special tokens if asked.
    pub fn decode_bytes(&self, ids: &[u32], skip_special: bool) -> Vec<u8> {
        let mut out = Vec::new();
        for &id in ids {
            if skip_special && self.special.contains(&id) {
                continue;
            }
            if let Some(b) = self.token_bytes.get(id as usize) {
                out.extend_from_slice(b);
            }
        }
        out
    }

    pub fn decode(&self, ids: &[u32], skip_special: bool) -> String {
        String::from_utf8_lossy(&self.decode_bytes(ids, skip_special)).into_owned()
    }

    /// Renders chat messages (role, content) as a prompt that ends where
    /// the assistant's reply begins.
    pub fn chat_prompt(&self, messages: &[(String, String)]) -> String {
        let mut s = String::new();
        match &self.chat {
            ChatFormat::ChatMl { default_system } => {
                if let Some(sys) = default_system
                    && messages.first().is_none_or(|m| m.0 != "system")
                {
                    s.push_str(&format!("<|im_start|>system\n{sys}<|im_end|>\n"));
                }
                for (role, content) in messages {
                    s.push_str(&format!("<|im_start|>{role}\n{content}<|im_end|>\n"));
                }
                s.push_str("<|im_start|>assistant\n");
            }
            ChatFormat::Llama3 => {
                if self.prefix.is_empty() {
                    s.push_str("<|begin_of_text|>");
                }
                for (role, content) in messages {
                    s.push_str(&format!("<|start_header_id|>{role}<|end_header_id|>\n\n{}<|eot_id|>", content.trim()));
                }
                s.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
            }
            ChatFormat::Plain => {
                for (_, content) in messages {
                    s.push_str(content);
                    s.push_str("\n\n");
                }
            }
        }
        s
    }
}

fn chat_format(template: &str) -> ChatFormat {
    if template.contains("<|im_start|>") {
        // A default system prompt appears as a literal system turn.
        let marker = "<|im_start|>system\\n";
        let default_system = template.match_indices(marker).find_map(|(i, _)| {
            let rest = &template[i + marker.len()..];
            let end = rest.find("<|im_end|>")?;
            let body = &rest[..end];
            (!body.contains('{') && !body.contains('\'')).then(|| body.to_string())
        });
        ChatFormat::ChatMl { default_system }
    } else if template.contains("<|start_header_id|>") {
        ChatFormat::Llama3
    } else {
        ChatFormat::Plain
    }
}

fn pre_steps(p: &Json) -> Result<Vec<Pre>, String> {
    let kind = p.get("type").and_then(Json::as_str).unwrap_or("");
    Ok(match kind {
        "Sequence" => {
            let mut v = Vec::new();
            for s in p.get("pretokenizers").map(Json::as_arr).unwrap_or(&[]) {
                v.extend(pre_steps(s)?);
            }
            v
        }
        "ByteLevel" => vec![Pre::ByteLevel {
            add_prefix_space: p.get("add_prefix_space").and_then(Json::as_bool).unwrap_or(false),
            regex: p.get("use_regex").and_then(Json::as_bool).unwrap_or(true),
        }],
        "Digits" => vec![Pre::Digits {
            individual: p.get("individual_digits").and_then(Json::as_bool).unwrap_or(false),
        }],
        "Split" => {
            let re = p
                .get("pattern")
                .and_then(|x| x.get("Regex"))
                .and_then(Json::as_str)
                .ok_or("Split pre-tokenizer without a regex")?;
            let behavior = p.get("behavior").and_then(Json::as_str).unwrap_or("");
            if behavior != "Isolated" || p.get("invert").and_then(Json::as_bool) == Some(true) {
                return Err(format!("unsupported Split behavior {behavior:?}"));
            }
            vec![Pre::Split(pattern_for(re)?)]
        }
        other => return Err(format!("unsupported pre-tokenizer {other:?}")),
    })
}

const GPT2_RE: &str = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";
const LLAMA3_RE: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
const QWEN2_RE: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

fn pattern_for(re: &str) -> Result<Pattern, String> {
    match re {
        GPT2_RE => Ok(Pattern::Gpt2),
        LLAMA3_RE => Ok(Pattern::Llama3 { max_digits: 3 }),
        QWEN2_RE => Ok(Pattern::Llama3 { max_digits: 1 }),
        _ => Err(format!("unsupported split pattern {re:?}")),
    }
}

fn template_prefix(pp: Option<&Json>, by_bytes: &HashMap<Vec<u8>, u32>, added: &[Added]) -> Vec<u32> {
    let find = |s: &str| {
        added
            .iter()
            .find(|a| a.content == s)
            .map(|a| a.id)
            .or_else(|| by_bytes.get(s.as_bytes()).copied())
    };
    let steps: Vec<&Json> = match pp {
        Some(p) if p.get("type").and_then(Json::as_str) == Some("Sequence") => {
            p.get("processors").map(Json::as_arr).unwrap_or(&[]).iter().collect()
        }
        Some(p) => vec![p],
        None => Vec::new(),
    };
    let mut prefix = Vec::new();
    for p in steps {
        if p.get("type").and_then(Json::as_str) != Some("TemplateProcessing") {
            continue;
        }
        for piece in p.get("single").map(Json::as_arr).unwrap_or(&[]) {
            match piece.get("SpecialToken").and_then(|t| t.get("id")).and_then(Json::as_str) {
                Some(tok) => prefix.extend(find(tok)),
                None => break,
            }
        }
    }
    prefix
}

fn split(step: &Pre, s: &str) -> Vec<String> {
    match step {
        Pre::Digits { individual } => {
            let mut out: Vec<String> = Vec::new();
            let mut prev_digit = false;
            for c in s.chars() {
                let d = c.is_numeric();
                let join = !out.is_empty() && d == prev_digit && !(d && *individual);
                if join {
                    out.last_mut().unwrap().push(c);
                } else {
                    out.push(c.to_string());
                }
                prev_digit = d;
            }
            out
        }
        Pre::Split(p) => by_pattern(*p, s),
        Pre::ByteLevel { add_prefix_space, regex } => {
            let s = if *add_prefix_space && !s.starts_with(' ') { format!(" {s}") } else { s.to_string() };
            if *regex { by_pattern(Pattern::Gpt2, &s) } else { vec![s] }
        }
    }
}

/// Splits `s` into the matches of the pattern (which cover every character).
fn by_pattern(p: Pattern, s: &str) -> Vec<String> {
    let c: Vec<char> = s.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < c.len() {
        let j = match p {
            Pattern::Gpt2 => gpt2_match(&c, i),
            Pattern::Llama3 { max_digits } => llama3_match(&c, i, max_digits),
        };
        let j = j.max(i + 1);
        out.push(c[i..j].iter().collect());
        i = j;
    }
    out
}

fn is_space(c: char) -> bool {
    c.is_whitespace()
}

fn is_other(c: char) -> bool {
    !is_space(c) && !is_letter(c) && !is_number(c)
}

fn run(c: &[char], mut i: usize, f: impl Fn(char) -> bool) -> usize {
    while i < c.len() && f(c[i]) {
        i += 1;
    }
    i
}

fn contraction(c: &[char], i: usize, fold: bool) -> Option<usize> {
    if c[i] != '\'' {
        return None;
    }
    for suffix in ["s", "t", "re", "ve", "m", "ll", "d"] {
        let n = suffix.len();
        if i + 1 + n <= c.len()
            && c[i + 1..i + 1 + n]
                .iter()
                .zip(suffix.chars())
                .all(|(&a, b)| a == b || (fold && a.to_lowercase().eq(std::iter::once(b))))
        {
            return Some(i + 1 + n);
        }
    }
    None
}

/// `\s+(?!\S)|\s+` at a whitespace character.
fn trailing_space(c: &[char], i: usize) -> usize {
    let j = run(c, i, is_space);
    if j == c.len() || j - 1 == i { j } else { j - 1 }
}

fn gpt2_match(c: &[char], i: usize) -> usize {
    if let Some(j) = contraction(c, i, false) {
        return j;
    }
    let k = if c[i] == ' ' { i + 1 } else { i };
    if k < c.len() {
        for class in [is_letter as fn(char) -> bool, is_number, is_other] {
            if class(c[k]) {
                return run(c, k, class);
            }
        }
    }
    if is_space(c[i]) {
        return trailing_space(c, i);
    }
    i + 1
}

fn llama3_match(c: &[char], i: usize, max_digits: usize) -> usize {
    let n = c.len();
    if let Some(j) = contraction(c, i, true) {
        return j;
    }
    // [^\r\n\p{L}\p{N}]?\p{L}+
    let ch = c[i];
    if ch != '\r' && ch != '\n' && !is_letter(ch) && !is_number(ch) && i + 1 < n && is_letter(c[i + 1]) {
        return run(c, i + 1, is_letter);
    }
    if is_letter(ch) {
        return run(c, i, is_letter);
    }
    // \p{N}{1,max}
    if is_number(ch) {
        let mut j = i;
        while j < n && j - i < max_digits && is_number(c[j]) {
            j += 1;
        }
        return j;
    }
    // ' ?[^\s\p{L}\p{N}]+[\r\n]*'
    let k = if ch == ' ' { i + 1 } else { i };
    if k < n && is_other(c[k]) {
        let j = run(c, k, is_other);
        return run(c, j, |x| x == '\r' || x == '\n');
    }
    if is_space(ch) {
        // \s*[\r\n]+ : up to the last newline in the whitespace run.
        let j = run(c, i, is_space);
        if let Some(last) = (i..j).rev().find(|&k| c[k] == '\r' || c[k] == '\n') {
            return last + 1;
        }
        return trailing_space(c, i);
    }
    i + 1
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pieces(p: Pattern, s: &str) -> Vec<String> {
        by_pattern(p, s)
    }

    #[test]
    fn gpt2_pattern_splits_like_the_regex() {
        assert_eq!(
            pieces(Pattern::Gpt2, "Hello world's  2024!!  \n x"),
            ["Hello", " world", "'s", " ", " 2024", "!!", "  \n", " x"]
        );
        assert_eq!(pieces(Pattern::Gpt2, "a   "), ["a", "   "]);
    }

    #[test]
    fn llama3_pattern_splits_like_the_regex() {
        assert_eq!(
            pieces(Pattern::Llama3 { max_digits: 3 }, "I'LL pay 12345 now.\n\n  ok"),
            ["I", "'LL", " pay", " ", "123", "45", " now", ".\n\n", " ", " ok"]
        );
        assert_eq!(pieces(Pattern::Llama3 { max_digits: 3 }, "x  \n\n y"), ["x", "  \n\n", " y"]);
    }
}
