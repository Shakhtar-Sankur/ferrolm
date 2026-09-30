"""Trains small byte-level BPE tokenizers with Hugging Face `tokenizers`,
configured like SmolLM2's (Digits + ByteLevel), Llama 3's and Qwen2's
(Split pattern + ByteLevel), and records the library's own encodings of a
multilingual test set plus random Unicode strings, so the Rust tokenizer
can be checked against it exactly.

Usage: python scripts/make_tokenizer_fixtures.py tests/fixtures/tokenizers
"""

import glob
import json
import os
import random
import sys

from tokenizers import Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers, processors, trainers
import unicodedata

LLAMA3 = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
QWEN2 = LLAMA3.replace(r"\p{N}{1,3}", r"\p{N}")

SAMPLES = [
    "Hello, world! It's a test.",
    "I'LL SEE YOU'RE there, won't we? They'd've",
    "  leading spaces and trailing   ",
    "tabs\tand\nnewlines\r\n\r\nmixed \n \n",
    "Numbers: 1234567890, 3.14159, 1e-10, ٣٤٥ ①②③ Ⅻ",
    "def f(x):\n    return x**2  # comment\n\n\nclass A:\n\tpass",
    "नमस्ते दुनिया, यह हिंदी है।",
    "আমার নাম সংকুর, আমি কলকাতায় থাকি।",
    "你好，世界！这是一个测试。",
    "こんにちは世界、カタカナとひらがな。",
    "Привет, мир! Ёжик.",
    "مرحبا بالعالم",
    "emoji 😀👍🏽 family 👨‍👩‍👧 flags 🇮🇳",
    "Zero\u200bwidth and\u00a0nbsp and\u3000ideographic space",
    "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n",
    "<|begin_of_text|>text<|eot_id|>",
    "",
    " ",
    "\n",
    "a" * 300,
    "email@example.com https://example.com/path?q=1&r=2#frag",
    "Ünïcödé àçcénts façade naïve coöperate",
    "ΑΒΓ αβγ ſs ǅ ﬁ",
    "decomposed e\u0301 a\u0308 o\u0302\u0323 \u1100\u1161\u11a8 \u212b \u2126 q\u0307\u0323",
    "Tiếng Việt có dấu: Trường Sa, Hoàng Sa",
    unicodedata.normalize("NFD", "Ünïcödé, 한국어, Tiếng Việt, ﬁ"),
]


def corpus():
    files = glob.glob(os.path.join(os.path.dirname(os.__file__), "*.py"))[:200]
    for f in files:
        with open(f, encoding="utf-8", errors="ignore") as fh:
            yield fh.read()
    for _ in range(50):
        yield from SAMPLES


def random_strings(n, seed):
    rng = random.Random(seed)
    pools = [
        (0x20, 0x7E), (0x0A, 0x0D), (0xA0, 0x24F), (0x370, 0x3FF), (0x400, 0x4FF),
        (0x590, 0x6FF), (0x900, 0x9FF), (0x980, 0x9FF), (0xE00, 0xE7F), (0x2000, 0x206F),
        (0x2150, 0x218F), (0x2460, 0x24FF), (0x3000, 0x30FF), (0x4E00, 0x4E80),
        (0xAC00, 0xAC80), (0xFF00, 0xFFEF), (0x1F300, 0x1F64F), (0x1D400, 0x1D4FF),
    ]
    out = []
    for _ in range(n):
        s = []
        for _ in range(rng.randint(0, 40)):
            a, b = rng.choice(pools)
            cp = rng.randint(a, b)
            if 0xD800 <= cp <= 0xDFFF:
                continue
            s.append(chr(cp))
            if rng.random() < 0.15:
                s.append(chr(rng.randint(0x300, 0x36F)))
            if rng.random() < 0.05:
                s.append(chr(rng.randint(0x1100, 0x11FF)))
            if rng.random() < 0.2:
                s.append(rng.choice([" ", "  ", "\n", "'s", "'LL", "123", "\t", " \n "]))
        out.append("".join(s))
    return out


def build(kind):
    tok = Tokenizer(models.BPE(ignore_merges=(kind == "llama3")))
    if kind == "qwen2":
        tok.normalizer = normalizers.NFC()
    if kind == "smollm2":
        tok.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Digits(individual_digits=True),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True),
        ])
    else:
        tok.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Split(Regex(LLAMA3 if kind == "llama3" else QWEN2), behavior="isolated"),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ])
    tok.decoder = decoders.ByteLevel()
    specials = {
        "smollm2": ["<|endoftext|>", "<|im_start|>", "<|im_end|>"],
        "llama3": ["<|begin_of_text|>", "<|end_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"],
        "qwen2": ["<|endoftext|>", "<|im_start|>", "<|im_end|>"],
    }[kind]
    trainer = trainers.BpeTrainer(
        vocab_size=3000, special_tokens=specials, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train_from_iterator(corpus(), trainer)
    if kind == "llama3":
        tok.post_processor = processors.TemplateProcessing(
            single="<|begin_of_text|> $A", special_tokens=[("<|begin_of_text|>", tok.token_to_id("<|begin_of_text|>"))]
        )
    return tok


def main(out):
    os.makedirs(out, exist_ok=True)
    for kind in ["smollm2", "llama3", "qwen2"]:
        tok = build(kind)
        d = os.path.join(out, kind)
        os.makedirs(d, exist_ok=True)
        tok.save(os.path.join(d, "tokenizer.json"))
        texts = SAMPLES + random_strings(2000, seed=hash(kind) % 1000)
        cases = [{"text": t, "ids": tok.encode(t, add_special_tokens=True).ids} for t in texts]
        with open(os.path.join(d, "cases.json"), "w") as f:
            json.dump(cases, f, ensure_ascii=False)
        print(kind, tok.get_vocab_size(), "tokens,", len(cases), "cases")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "tests/fixtures/tokenizers")
