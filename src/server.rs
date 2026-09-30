//! An OpenAI-compatible HTTP server over the engine: `/v1/completions` and
//! `/v1/chat/completions` (with server-sent-event streaming),
//! `/v1/embeddings` (when an encoder is loaded), `/v1/models`, `/health`
//! and Prometheus `/metrics`. One thread per connection; the
//! engine runs on its own thread and batches whatever the connections
//! submit.

use crate::encoder::Encoder;
use crate::engine::{Event, Finish, Handle};
use crate::json::{self, Json, quote};
use crate::pool::Pool;
use crate::sampler::SamplingParams;
use crate::tokenizer::Tokenizer;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::AtomicU32;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

pub struct Server {
    pub handle: Handle,
    pub tokenizer: Arc<Tokenizer>,
    pub model_name: String,
    pub started: Instant,
    pub embedder: Option<Embedder>,
    ids: AtomicU64,
}

/// An embedding model served at `/v1/embeddings`, with its own threads.
pub struct Embedder {
    pub name: String,
    pub encoder: Encoder,
    pool: Mutex<Pool>,
    requests: AtomicU64,
    inputs: AtomicU64,
    tokens: AtomicU64,
    busy: AtomicU32,
}

/// Most texts in one embeddings request.
const MAX_INPUTS: usize = 2048;
/// Texts encoded in one forward pass, to bound activation memory.
const EMBED_BATCH: usize = 32;

impl Embedder {
    pub fn new(name: String, encoder: Encoder, threads: usize) -> Embedder {
        Embedder {
            name,
            encoder,
            pool: Mutex::new(Pool::new(threads)),
            requests: AtomicU64::new(0),
            inputs: AtomicU64::new(0),
            tokens: AtomicU64::new(0),
            busy: AtomicU32::new(0),
        }
    }

    /// The response body for an OpenAI embeddings request, or an error
    /// message. `input` is a string or an array of strings;
    /// `encoding_format` is "float" (default) or "base64" (little-endian f32).
    pub fn respond(&self, req: &Json) -> Result<String, String> {
        let texts: Vec<&str> = match req.get("input") {
            Some(Json::Str(s)) => vec![s.as_str()],
            Some(Json::Arr(a)) if !a.is_empty() => a
                .iter()
                .map(|v| v.as_str().ok_or("input must be a string or an array of strings"))
                .collect::<Result<_, _>>()?,
            Some(Json::Arr(_)) => return Err("input must not be empty".into()),
            _ => return Err("input must be a string or an array of strings".into()),
        };
        if texts.len() > MAX_INPUTS {
            return Err(format!("at most {MAX_INPUTS} inputs per request"));
        }
        let base64 = match req.get("encoding_format").and_then(Json::as_str) {
            None | Some("float") => false,
            Some("base64") => true,
            Some(f) => return Err(format!("unsupported encoding_format {f:?}")),
        };
        let seqs = self.encoder.tokenize(&texts);
        let tokens: usize = seqs.iter().map(Vec::len).sum();
        let vecs = self.embed(&seqs);
        self.requests.fetch_add(1, Ordering::Relaxed);
        self.inputs.fetch_add(texts.len() as u64, Ordering::Relaxed);
        self.tokens.fetch_add(tokens as u64, Ordering::Relaxed);
        let mut data = String::new();
        for (i, v) in vecs.iter().enumerate() {
            if i > 0 {
                data.push(',');
            }
            let emb = if base64 {
                let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
                format!("\"{}\"", base64_encode(&bytes))
            } else {
                let mut e = String::from("[");
                for (j, x) in v.iter().enumerate() {
                    if j > 0 {
                        e.push(',');
                    }
                    e.push_str(&x.to_string());
                }
                e.push(']');
                e
            };
            data.push_str(&format!(r#"{{"object":"embedding","index":{i},"embedding":{emb}}}"#));
        }
        Ok(format!(
            r#"{{"object":"list","data":[{data}],"model":{},"usage":{{"prompt_tokens":{tokens},"total_tokens":{tokens}}}}}"#,
            quote(&self.name)
        ))
    }

    /// Embeddings for tokenized texts, a bounded batch at a time.
    pub fn embed(&self, seqs: &[Vec<u32>]) -> Vec<Vec<f32>> {
        self.busy.fetch_add(1, Ordering::Relaxed);
        let pool = self.pool.lock().unwrap_or_else(|e| e.into_inner());
        let out = self.encoder.embed_many(&pool, seqs, EMBED_BATCH);
        drop(pool);
        self.busy.fetch_sub(1, Ordering::Relaxed);
        out
    }
}

fn base64_encode(b: &[u8]) -> String {
    const A: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut s = String::with_capacity(b.len().div_ceil(3) * 4);
    for c in b.chunks(3) {
        let n =
            (u32::from(c[0]) << 16) | (u32::from(*c.get(1).unwrap_or(&0)) << 8) | u32::from(*c.get(2).unwrap_or(&0));
        for k in 0..4 {
            if k <= c.len() {
                s.push(A[(n >> (18 - 6 * k) & 63) as usize] as char);
            } else {
                s.push('=');
            }
        }
    }
    s
}

impl Server {
    pub fn new(handle: Handle, tokenizer: Arc<Tokenizer>, model_name: String) -> Server {
        Server {
            handle,
            tokenizer,
            model_name,
            started: Instant::now(),
            embedder: None,
            ids: AtomicU64::new(1),
        }
    }

    pub fn listen(self: Arc<Self>, addr: &str) -> std::io::Result<()> {
        let listener = TcpListener::bind(addr)?;
        eprintln!("ferrolm: serving {} on http://{addr}", self.model_name);
        for conn in listener.incoming() {
            let Ok(conn) = conn else { continue };
            let s = Arc::clone(&self);
            std::thread::spawn(move || {
                if let Err(e) = s.connection(conn) {
                    eprintln!("ferrolm: connection error: {e}");
                }
            });
        }
        Ok(())
    }

    fn connection(&self, mut conn: TcpStream) -> std::io::Result<()> {
        conn.set_nodelay(true).ok();
        let mut reader = BufReader::new(conn.try_clone()?);
        let mut line = String::new();
        reader.read_line(&mut line)?;
        let mut parts = line.split_whitespace();
        let (method, path) = (
            parts.next().unwrap_or("").to_string(),
            parts.next().unwrap_or("").to_string(),
        );
        let mut length = 0usize;
        loop {
            let mut h = String::new();
            if reader.read_line(&mut h)? == 0 || h == "\r\n" || h == "\n" {
                break;
            }
            if let Some((k, v)) = h.split_once(':')
                && k.trim().eq_ignore_ascii_case("content-length")
            {
                length = v.trim().parse().unwrap_or(0);
            }
        }
        if length > 8 << 20 {
            return respond(
                &mut conn,
                413,
                "application/json",
                &error_json("request body too large"),
            );
        }
        let mut body = vec![0u8; length];
        reader.read_exact(&mut body)?;
        let path = path.split('?').next().unwrap_or("");
        match (method.as_str(), path) {
            ("GET", "/health") => respond(&mut conn, 200, "text/plain", "ok\n"),
            ("GET", "/v1/models") => {
                let model = |name: &str| {
                    format!(
                        r#"{{"id":{},"object":"model","created":{},"owned_by":"ferrolm"}}"#,
                        quote(name),
                        unix_now()
                    )
                };
                let mut data = model(&self.model_name);
                if let Some(e) = &self.embedder {
                    data = format!("{data},{}", model(&e.name));
                }
                let body = format!(r#"{{"object":"list","data":[{data}]}}"#);
                respond(&mut conn, 200, "application/json", &body)
            }
            ("GET", "/metrics") => respond(&mut conn, 200, "text/plain; version=0.0.4", &self.metrics()),
            ("POST", "/v1/completions") => self.complete(&mut conn, &body, false),
            ("POST", "/v1/chat/completions") => self.complete(&mut conn, &body, true),
            ("POST", "/v1/embeddings") => {
                let Some(e) = &self.embedder else {
                    return respond(
                        &mut conn,
                        404,
                        "application/json",
                        &error_json("no embedding model loaded (start with --embedding-model)"),
                    );
                };
                let out = std::str::from_utf8(&body)
                    .map_err(|_| "body is not UTF-8".to_string())
                    .and_then(|b| json::parse(b).map_err(|e| format!("invalid JSON: {e}")))
                    .and_then(|v| e.respond(&v));
                match out {
                    Ok(b) => respond(&mut conn, 200, "application/json", &b),
                    Err(m) => respond(&mut conn, 400, "application/json", &error_json(&m)),
                }
            }
            _ => respond(&mut conn, 404, "application/json", &error_json("no such endpoint")),
        }
    }

    fn complete(&self, conn: &mut TcpStream, body: &[u8], chat: bool) -> std::io::Result<()> {
        let req = match std::str::from_utf8(body)
            .map_err(|_| "body is not UTF-8".to_string())
            .and_then(json::parse)
        {
            Ok(v) => v,
            Err(e) => {
                return respond(
                    conn,
                    400,
                    "application/json",
                    &error_json(&format!("invalid JSON: {e}")),
                );
            }
        };
        let parsed = self.parse_request(&req, chat);
        let (prompt, params, stops, stream) = match parsed {
            Ok(p) => p,
            Err(e) => return respond(conn, 400, "application/json", &error_json(&e)),
        };
        let prompt_tokens = prompt.len();
        let id = format!(
            "{}-{}",
            if chat { "chatcmpl" } else { "cmpl" },
            self.ids.fetch_add(1, Ordering::Relaxed)
        );
        let (rx, cancel) = self.handle.submit(prompt, params);
        let mut text = Detokenizer::new(&self.tokenizer, stops);
        let mut completion_tokens = 0;
        let object = if chat {
            "chat.completion.chunk"
        } else {
            "text_completion"
        };
        let chunk = |delta: &str, finish: Option<&str>, first: bool| -> String {
            let finish = finish.map_or("null".to_string(), quote);
            let choice = if chat {
                let role = if first { r#""role":"assistant","# } else { "" };
                format!(
                    r#"{{"index":0,"delta":{{{role}"content":{}}},"finish_reason":{finish}}}"#,
                    quote(delta)
                )
            } else {
                format!(r#"{{"index":0,"text":{},"finish_reason":{finish}}}"#, quote(delta))
            };
            format!(
                r#"{{"id":{},"object":"{object}","created":{},"model":{},"choices":[{choice}]}}"#,
                quote(&id),
                unix_now(),
                quote(&self.model_name)
            )
        };
        if stream {
            conn.write_all(
                b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n",
            )?;
        }
        let mut first = true;
        let finish = loop {
            let ev = rx.recv().unwrap_or(Event::Done(Finish::Cancelled));
            let (delta, done) = match ev {
                Event::Token(t) => {
                    completion_tokens += 1;
                    let d = text.push(t);
                    if text.stopped {
                        cancel.store(true, Ordering::Relaxed);
                        (d, Some(Finish::Stop))
                    } else {
                        (d, None)
                    }
                }
                Event::Done(f) => (text.flush(), Some(f)),
            };
            if let Some(Finish::Rejected(why)) = &done {
                if stream {
                    write_sse(conn, &error_json(why))?;
                    conn.write_all(b"data: [DONE]\n\n")?;
                    return Ok(());
                }
                return respond(conn, 400, "application/json", &error_json(why));
            }
            if stream && !delta.is_empty() {
                if write_sse(conn, &chunk(&delta, None, first)).is_err() {
                    // The client went away: stop generating for it.
                    cancel.store(true, Ordering::Relaxed);
                    return Ok(());
                }
                first = false;
            }
            if let Some(f) = done {
                break f;
            }
        };
        let reason = if finish == Finish::Length { "length" } else { "stop" };
        let usage = format!(
            r#"{{"prompt_tokens":{prompt_tokens},"completion_tokens":{completion_tokens},"total_tokens":{}}}"#,
            prompt_tokens + completion_tokens
        );
        if stream {
            write_sse(conn, &chunk("", Some(reason), first))?;
            conn.write_all(b"data: [DONE]\n\n")?;
            return conn.flush();
        }
        let out = text.text();
        let choice = if chat {
            format!(
                r#"{{"index":0,"message":{{"role":"assistant","content":{}}},"finish_reason":"{reason}"}}"#,
                quote(&out)
            )
        } else {
            format!(r#"{{"index":0,"text":{},"finish_reason":"{reason}"}}"#, quote(&out))
        };
        let body = format!(
            r#"{{"id":{},"object":"{}","created":{},"model":{},"choices":[{choice}],"usage":{usage}}}"#,
            quote(&id),
            if chat { "chat.completion" } else { "text_completion" },
            unix_now(),
            quote(&self.model_name)
        );
        respond(conn, 200, "application/json", &body)
    }

    #[allow(clippy::type_complexity)]
    fn parse_request(&self, v: &Json, chat: bool) -> Result<(Vec<u32>, SamplingParams, Vec<String>, bool), String> {
        let tok = &self.tokenizer;
        let prompt = if chat {
            let mut msgs = Vec::new();
            for m in v.get("messages").ok_or("missing messages")?.as_arr() {
                let role = m.get("role").and_then(Json::as_str).ok_or("message without role")?;
                let content = match m.get("content") {
                    Some(Json::Str(s)) => s.clone(),
                    // Content parts: keep the text ones.
                    Some(Json::Arr(parts)) => parts
                        .iter()
                        .filter_map(|p| p.get("text").and_then(Json::as_str))
                        .collect::<Vec<_>>()
                        .join(""),
                    _ => return Err("message without content".into()),
                };
                msgs.push((role.to_string(), content));
            }
            if msgs.is_empty() {
                return Err("messages is empty".into());
            }
            tok.encode(&tok.chat_prompt(&msgs), true)
        } else {
            match v.get("prompt") {
                Some(Json::Str(s)) => tok.encode(s, true),
                Some(Json::Arr(a)) if a.iter().all(|x| x.as_usize().is_some()) => {
                    a.iter().map(|x| x.as_usize().unwrap() as u32).collect()
                }
                Some(Json::Arr(a)) if a.len() == 1 => tok.encode(a[0].as_str().ok_or("bad prompt")?, true),
                _ => return Err("prompt must be a string or an array of token ids".into()),
            }
        };
        if v.get("n").and_then(Json::as_usize).is_some_and(|n| n != 1) {
            return Err("only n = 1 is supported".into());
        }
        let num = |k: &str| v.get(k).and_then(Json::as_f64);
        let max_tokens = v
            .get("max_completion_tokens")
            .or_else(|| v.get("max_tokens"))
            .and_then(Json::as_usize)
            .unwrap_or(if chat { 512 } else { 16 });
        let params = SamplingParams {
            temperature: num("temperature").unwrap_or(1.0) as f32,
            top_p: num("top_p").unwrap_or(1.0) as f32,
            top_k: v.get("top_k").and_then(Json::as_usize).unwrap_or(0),
            seed: v
                .get("seed")
                .and_then(Json::as_usize)
                .map_or_else(rand_seed, |s| s as u64),
            max_tokens: max_tokens.max(1),
            stop_ids: tok.stop_ids.clone(),
            ignore_eos: v.get("ignore_eos").and_then(Json::as_bool).unwrap_or(false),
        };
        if !(0.0..=100.0).contains(&params.temperature) || !(0.0..=1.0).contains(&params.top_p) || params.top_p == 0.0 {
            return Err("temperature must be in [0, 100] and top_p in (0, 1]".into());
        }
        let stops = match v.get("stop") {
            Some(Json::Str(s)) => vec![s.clone()],
            Some(Json::Arr(a)) => a.iter().filter_map(Json::as_str).map(String::from).collect(),
            _ => Vec::new(),
        };
        let stream = v.get("stream").and_then(Json::as_bool).unwrap_or(false);
        Ok((
            prompt,
            params,
            stops.into_iter().filter(|s| !s.is_empty()).collect(),
            stream,
        ))
    }

    fn metrics(&self) -> String {
        let s = self.handle.stats.lock().unwrap().clone();
        let mut m = String::new();
        let mut put = |name: &str, kind: &str, help: &str, v: String| {
            m.push_str(&format!(
                "# HELP ferrolm_{name} {help}\n# TYPE ferrolm_{name} {kind}\nferrolm_{name} {v}\n"
            ));
        };
        put(
            "requests_total",
            "counter",
            "Requests received.",
            s.requests.to_string(),
        );
        put(
            "requests_finished_total",
            "counter",
            "Requests finished, cancelled or rejected.",
            s.finished.to_string(),
        );
        put(
            "prompt_tokens_total",
            "counter",
            "Prompt tokens received.",
            s.prompt_tokens.to_string(),
        );
        put(
            "generation_tokens_total",
            "counter",
            "Tokens generated.",
            s.generated_tokens.to_string(),
        );
        put(
            "prefix_cache_tokens_total",
            "counter",
            "Prompt tokens served from the prefix cache.",
            s.cached_tokens.to_string(),
        );
        put(
            "preemptions_total",
            "counter",
            "Sequences preempted for lack of cache blocks.",
            s.preemptions.to_string(),
        );
        put(
            "spec_proposed_tokens_total",
            "counter",
            "Draft tokens proposed.",
            s.spec_proposed.to_string(),
        );
        put(
            "spec_accepted_tokens_total",
            "counter",
            "Draft tokens accepted.",
            s.spec_accepted.to_string(),
        );
        put("steps_total", "counter", "Engine steps.", s.steps.to_string());
        put(
            "busy_seconds_total",
            "counter",
            "Time the engine spent in steps.",
            format!("{:.3}", s.busy_secs),
        );
        put(
            "running_sequences",
            "gauge",
            "Sequences in the running batch.",
            s.running.to_string(),
        );
        put(
            "waiting_sequences",
            "gauge",
            "Sequences waiting for admission.",
            s.waiting.to_string(),
        );
        put(
            "kv_cache_blocks_used",
            "gauge",
            "Cache blocks in use.",
            s.kv_used_blocks.to_string(),
        );
        put(
            "kv_cache_blocks_total",
            "gauge",
            "Cache blocks.",
            s.kv_total_blocks.to_string(),
        );
        put(
            "uptime_seconds",
            "gauge",
            "Seconds since start.",
            format!("{:.0}", self.started.elapsed().as_secs_f64()),
        );
        if let Some(e) = &self.embedder {
            let n = |a: &AtomicU64| a.load(Ordering::Relaxed).to_string();
            put(
                "embedding_requests_total",
                "counter",
                "Embedding requests served.",
                n(&e.requests),
            );
            put("embedding_inputs_total", "counter", "Texts embedded.", n(&e.inputs));
            put("embedding_tokens_total", "counter", "Tokens embedded.", n(&e.tokens));
            put(
                "embedding_requests_running",
                "gauge",
                "Embedding requests running or waiting.",
                e.busy.load(Ordering::Relaxed).to_string(),
            );
        }
        m
    }
}

/// Turns generated tokens into text incrementally: emits only complete
/// UTF-8 characters, and holds back text that could still become a stop
/// string.
pub struct Detokenizer<'a> {
    tok: &'a Tokenizer,
    bytes: Vec<u8>,
    /// Bytes already returned.
    sent: usize,
    stops: Vec<String>,
    pub stopped: bool,
}

impl<'a> Detokenizer<'a> {
    pub fn new(tok: &'a Tokenizer, stops: Vec<String>) -> Detokenizer<'a> {
        Detokenizer {
            tok,
            bytes: Vec::new(),
            sent: 0,
            stops,
            stopped: false,
        }
    }

    /// Adds a token; returns the text that is now safe to send.
    pub fn push(&mut self, t: u32) -> String {
        if self.stopped {
            return String::new();
        }
        self.bytes.extend(self.tok.decode_bytes(&[t], true));
        let valid = match std::str::from_utf8(&self.bytes) {
            Ok(s) => s.len(),
            Err(e) => e.valid_up_to(),
        };
        let text = std::str::from_utf8(&self.bytes[..valid]).unwrap();
        // Stop at the earliest stop string that appears.
        if let Some(cut) = self.stops.iter().filter_map(|s| text.find(s.as_str())).min() {
            self.bytes.truncate(cut);
            self.stopped = true;
            return self.take(cut);
        }
        // Keep back a possible stop-string prefix at the end.
        let hold = self
            .stops
            .iter()
            .flat_map(|s| (1..s.len()).filter(|&k| s.is_char_boundary(k) && text.ends_with(&s[..k])))
            .max()
            .unwrap_or(0);
        self.take(valid - hold)
    }

    fn take(&mut self, upto: usize) -> String {
        let upto = upto.max(self.sent);
        let s = String::from_utf8_lossy(&self.bytes[self.sent..upto]).into_owned();
        self.sent = upto;
        s
    }

    /// Whatever is left at the end.
    pub fn flush(&mut self) -> String {
        let n = self.bytes.len();
        self.take(n)
    }

    pub fn text(&self) -> String {
        String::from_utf8_lossy(&self.bytes).into_owned()
    }
}

fn write_sse(conn: &mut TcpStream, data: &str) -> std::io::Result<()> {
    conn.write_all(format!("data: {data}\n\n").as_bytes())?;
    conn.flush()
}

fn respond(conn: &mut TcpStream, status: u16, ctype: &str, body: &str) -> std::io::Result<()> {
    let reason = match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        413 => "Payload Too Large",
        _ => "Error",
    };
    let head = format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    conn.write_all(head.as_bytes())?;
    conn.write_all(body.as_bytes())?;
    conn.flush()
}

fn error_json(msg: &str) -> String {
    format!(
        r#"{{"error":{{"message":{},"type":"invalid_request_error"}}}}"#,
        quote(msg)
    )
}

fn unix_now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

fn rand_seed() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(1, |d| d.as_nanos() as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tok() -> Tokenizer {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tokenizers/smollm2");
        Tokenizer::load(&dir).unwrap()
    }

    #[test]
    fn detokenizer_emits_whole_characters_and_cuts_at_stop_strings() {
        let t = tok();
        let ids = t.encode("héllo 😀 world. STOP here", false);
        let mut d = Detokenizer::new(&t, vec!["STOP".into()]);
        let mut out = String::new();
        for &i in &ids {
            out.push_str(&d.push(i));
            if d.stopped {
                break;
            }
        }
        out.push_str(&d.flush());
        assert_eq!(out, "héllo 😀 world. ");
        assert!(d.stopped);
        // Byte-by-byte tokens of an emoji produce it in one piece.
        let mut d = Detokenizer::new(&t, vec![]);
        let pieces: Vec<String> = t.encode("😀", false).iter().map(|&i| d.push(i)).collect();
        assert_eq!(pieces.concat(), "😀");
        assert!(pieces.iter().all(|p| p.is_empty() || p == "😀"));
    }
}
