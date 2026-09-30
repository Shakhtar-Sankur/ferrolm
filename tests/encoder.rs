//! The BERT encoder and WordPiece tokenizer against Hugging Face (see
//! scripts/make_encoder_fixtures.py): identical token ids, and embeddings
//! with cosine similarity to transformers' above 0.999. Runs on the
//! committed tiny fixture, and on downloaded models listed in
//! FERROLM_ENCODERS.

use ferrolm::encoder::Encoder;
use ferrolm::json::{self, Json};
use ferrolm::pool::Pool;
use std::path::{Path, PathBuf};

fn check(dir: &Path) {
    let name = dir.display().to_string();
    let enc = Encoder::load(dir).unwrap();
    let r = json::parse(&std::fs::read_to_string(dir.join("ferrolm-encoder.json")).unwrap()).unwrap();
    let mut bad = 0;
    let cases = r.get("cases").unwrap().as_arr();
    for c in cases {
        let text = c.get("text").and_then(Json::as_str).unwrap();
        let want: Vec<u32> = c
            .get("ids")
            .unwrap()
            .as_arr()
            .iter()
            .map(|x| x.as_usize().unwrap() as u32)
            .collect();
        let got = enc.tokenizer.encode(text, usize::MAX);
        if got != want {
            bad += 1;
            if bad <= 3 {
                eprintln!("{name}: {text:?}\n  got  {got:?}\n  want {want:?}");
            }
        }
    }
    assert_eq!(bad, 0, "{name}: {bad} of {} tokenizations differ", cases.len());

    let pool = Pool::new(4);
    let sentences: Vec<&str> = r
        .get("sentences")
        .unwrap()
        .as_arr()
        .iter()
        .map(|s| s.as_str().unwrap())
        .collect();
    let got = enc.embed(&pool, &sentences);
    let mut worst = 1f64;
    for (g, w) in got.iter().zip(r.get("embeddings").unwrap().as_arr()) {
        let w: Vec<f64> = w.as_arr().iter().map(|v| v.as_f64().unwrap()).collect();
        let cos: f64 = g.iter().zip(&w).map(|(a, b)| f64::from(*a) * b).sum();
        worst = worst.min(cos);
    }
    println!(
        "{name}: {} tokenizer cases identical; embedding cosine vs transformers >= {worst:.6}",
        cases.len()
    );
    assert!(worst > 0.999, "{name}: cosine {worst}");
    // Batching does not change a text's vector.
    let one = enc.embed(&pool, &sentences[2..3]);
    assert_eq!(one[0], got[2]);
}

#[test]
fn tiny_bert_matches_transformers() {
    check(&Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/bert-tiny"));
}

#[test]
fn downloaded_encoders_match_transformers() {
    let dirs: Vec<PathBuf> = std::env::var("FERROLM_ENCODERS")
        .map(|v| v.split(',').filter(|s| !s.is_empty()).map(Into::into).collect())
        .unwrap_or_default();
    for d in dirs {
        check(&d);
    }
}

#[test]
fn embeddings_endpoint_returns_the_encoders_vectors() {
    use ferrolm::server::Embedder;
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/bert-tiny");
    let enc = Encoder::load(&dir).unwrap();
    let texts = [
        "a short one",
        "a much longer sentence about retrieval and embeddings",
        "",
    ];
    let want = enc.embed(&Pool::new(2), &texts);
    let e = Embedder::new("bert-tiny".into(), Encoder::load(&dir).unwrap(), 2);

    let req =
        json::parse(r#"{"input":["a short one","a much longer sentence about retrieval and embeddings",""]}"#).unwrap();
    let out = json::parse(&e.respond(&req).unwrap()).unwrap();
    let data = out.get("data").unwrap().as_arr();
    assert_eq!(data.len(), 3);
    for (i, d) in data.iter().enumerate() {
        assert_eq!(d.get("index").and_then(Json::as_usize), Some(i));
        let got: Vec<f32> = d
            .get("embedding")
            .unwrap()
            .as_arr()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        assert_eq!(got, want[i], "input {i}");
    }
    let tokens: usize = enc.tokenize(&texts).iter().map(Vec::len).sum();
    assert_eq!(
        out.get("usage").unwrap().get("prompt_tokens").and_then(Json::as_usize),
        Some(tokens)
    );

    // base64: little-endian f32, the same values.
    let req = json::parse(r#"{"input":"a short one","encoding_format":"base64"}"#).unwrap();
    let out = json::parse(&e.respond(&req).unwrap()).unwrap();
    let b64 = out.get("data").unwrap().as_arr()[0]
        .get("embedding")
        .and_then(Json::as_str)
        .unwrap()
        .to_string();
    let val = |c: u8| {
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
            .bytes()
            .position(|x| x == c)
            .unwrap() as u32
    };
    let mut bytes = Vec::new();
    for q in b64.as_bytes().chunks(4) {
        let pad = q.iter().filter(|&&c| c == b'=').count();
        let n = q
            .iter()
            .filter(|&&c| c != b'=')
            .enumerate()
            .fold(0u32, |n, (k, &c)| n | val(c) << (18 - 6 * k));
        bytes.extend(&n.to_be_bytes()[1..4 - pad]);
    }
    let got: Vec<f32> = bytes
        .chunks(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(got, want[0]);

    for bad in [
        r#"{"input":[]}"#,
        r#"{"input":[1,2]}"#,
        r#"{}"#,
        r#"{"input":"x","encoding_format":"int8"}"#,
    ] {
        assert!(e.respond(&json::parse(bad).unwrap()).is_err(), "{bad}");
    }
}
