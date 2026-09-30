//! ferrolm's tokenizer against Hugging Face `tokenizers` (see
//! scripts/make_tokenizer_fixtures.py): identical ids for a multilingual
//! sample set and 2,000 random Unicode strings per tokenizer, and exact
//! round trips through decoding.

use ferrolm::json::{self, Json};
use ferrolm::tokenizer::Tokenizer;
use std::path::Path;

fn check(kind: &str) {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tokenizers").join(kind);
    let tok = Tokenizer::load(&dir).unwrap();
    let cases = json::parse(&std::fs::read_to_string(dir.join("cases.json")).unwrap()).unwrap();
    let mut failures = 0;
    for c in cases.as_arr() {
        let text = c.get("text").and_then(Json::as_str).unwrap();
        let want: Vec<u32> = c.get("ids").unwrap().as_arr().iter().map(|x| x.as_usize().unwrap() as u32).collect();
        let got = tok.encode(text, true);
        if got != want {
            failures += 1;
            if failures <= 5 {
                eprintln!("{kind}: {text:?}\n  got  {got:?}\n  want {want:?}");
            }
            continue;
        }
        // Decoding (without the post-processor's prefix) gives back the
        // text exactly.
        assert_eq!(tok.decode(&tok.encode(text, false), false), text, "{kind}: round trip");
    }
    assert_eq!(failures, 0, "{kind}: {failures} of {} cases differ", cases.as_arr().len());
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
