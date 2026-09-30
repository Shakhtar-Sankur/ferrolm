//! ferrolm's tokenizer against Hugging Face `tokenizers` (see
//! scripts/make_tokenizer_fixtures.py): identical ids for a multilingual
//! sample set and 2,000 random Unicode strings per tokenizer (with
//! combining marks and Hangul jamo, for NFC), and exact round trips
//! through decoding.

use ferrolm::json::{self, Json};
use ferrolm::tokenizer::Tokenizer;
use std::path::Path;

fn check(kind: &str) {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/tokenizers")
        .join(kind);
    let cases = json::parse(&std::fs::read_to_string(dir.join("cases.json")).unwrap()).unwrap();
    check_cases(&dir, kind, &cases, kind == "qwen2");
}

fn check_cases(dir: &Path, kind: &str, cases: &Json, normalizes: bool) {
    let tok = Tokenizer::load(dir).unwrap();
    let mut failures = 0;
    for c in cases.as_arr() {
        let text = c.get("text").and_then(Json::as_str).unwrap();
        let want: Vec<u32> = c
            .get("ids")
            .unwrap()
            .as_arr()
            .iter()
            .map(|x| x.as_usize().unwrap() as u32)
            .collect();
        let got = tok.encode(text, true);
        if got != want {
            failures += 1;
            if failures <= 5 {
                eprintln!("{kind}: {text:?}\n  got  {got:?}\n  want {want:?}");
            }
            continue;
        }
        // Decoding (without the post-processor's prefix) gives back the
        // text exactly, in NFC form if the tokenizer normalizes.
        let want = if normalizes {
            ferrolm::unicode::nfc(text).into_owned()
        } else {
            text.to_string()
        };
        assert_eq!(tok.decode(&tok.encode(text, false), false), want, "{kind}: round trip");
    }
    assert_eq!(
        failures,
        0,
        "{kind}: {failures} of {} cases differ",
        cases.as_arr().len()
    );
}

#[test]
fn smollm2_style_tokenizer_matches_hugging_face() {
    check("smollm2");
}

#[test]
fn llama3_style_tokenizer_matches_hugging_face() {
    check("llama3");
}

#[test]
fn qwen2_style_tokenizer_matches_hugging_face() {
    check("qwen2");
}

/// Real tokenizers (see scripts/reference_real.py and
/// FERROLM_REAL_MODELS): ids, and chat prompts rendered exactly as
/// transformers' `apply_chat_template` renders them.
#[test]
fn downloaded_tokenizers_and_chat_templates_match_hugging_face() {
    let dirs: Vec<std::path::PathBuf> = std::env::var("FERROLM_REAL_MODELS")
        .map(|v| v.split(',').filter(|s| !s.is_empty()).map(Into::into).collect())
        .unwrap_or_default();
    for dir in dirs {
        let name = dir.display().to_string();
        let r = json::parse(&std::fs::read_to_string(dir.join("ferrolm-tokenizer.json")).unwrap()).unwrap();
        let spec = json::parse(&std::fs::read_to_string(dir.join("tokenizer.json")).unwrap()).unwrap();
        let normalizes = spec
            .get("normalizer")
            .and_then(|n| n.get("type"))
            .and_then(Json::as_str)
            == Some("NFC");
        check_cases(&dir, &name, r.get("cases").unwrap(), normalizes);
        let tok = Tokenizer::load(&dir).unwrap();
        for chat in r.get("chats").unwrap().as_arr() {
            let msgs: Vec<(String, String)> = chat
                .get("messages")
                .unwrap()
                .as_arr()
                .iter()
                .map(|m| {
                    let f = |k| m.get(k).and_then(Json::as_str).unwrap().to_string();
                    (f("role"), f("content"))
                })
                .collect();
            let want = chat.get("prompt").and_then(Json::as_str).unwrap();
            assert_eq!(tok.chat_prompt(&msgs), want, "{name}: chat template");
        }
        println!("{name}: tokenizer and chat template match");
    }
}
