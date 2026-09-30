"""Reference data for ferrolm's BERT encoder and WordPiece tokenizer.

  python scripts/make_encoder_fixtures.py fixture tests/fixtures/bert-tiny
      builds a tiny random-weight BERT with a freshly trained WordPiece
      tokenizer (so CI needs no downloads)
  python scripts/make_encoder_fixtures.py real models/bge-small
      records references for a downloaded embedding model

Either way it writes ferrolm-encoder.json: `tokenizers`' ids for test
strings, and transformers' pooled, normalised embeddings for sentences.
"""

import json
import os
import sys

import torch
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, processors, trainers
from transformers import BertConfig, BertModel

sys.path.insert(0, os.path.dirname(__file__))
from make_tokenizer_fixtures import SAMPLES, corpus, random_strings  # noqa: E402

SENTENCES = [
    "The capital of France is Paris.",
    "Paris is the largest city in France and its capital.",
    "Photosynthesis converts light energy into chemical energy in plants.",
    "A hash table maps keys to values using a hash function.",
    "Rust's ownership rules prevent data races at compile time.",
    "The quick brown fox jumps over the lazy dog!",
    "Ünïcödé text, CJK 你好世界 and emoji 😀 all tokenize.",
    "short",
]


def pooled(model, tok, texts, cls):
    out = []
    for t in texts:
        ids = tok.encode(t).ids[:512]
        with torch.no_grad():
            h = model(torch.tensor([ids])).last_hidden_state[0]
        e = h[0] if cls else h.mean(0)
        out.append(torch.nn.functional.normalize(e, dim=0).tolist())
    return out


def main(mode, d):
    os.makedirs(d, exist_ok=True)
    torch.manual_seed(0)
    if mode == "fixture":
        tok = Tokenizer(models.WordPiece(unk_token="[UNK]"))
        tok.normalizer = normalizers.BertNormalizer(lowercase=True)
        tok.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
        tok.decoder = decoders.WordPiece()
        trainer = trainers.WordPieceTrainer(vocab_size=2000, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
                                            show_progress=False)
        tok.train_from_iterator(corpus(), trainer)
        tok.post_processor = processors.TemplateProcessing(
            single="[CLS] $A [SEP]", special_tokens=[("[CLS]", tok.token_to_id("[CLS]")), ("[SEP]", tok.token_to_id("[SEP]"))])
        tok.save(os.path.join(d, "tokenizer.json"))
        cfg = BertConfig(vocab_size=tok.get_vocab_size(), hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
                         intermediate_size=128, max_position_embeddings=128, attn_implementation="eager")
        model = BertModel(cfg, add_pooling_layer=False).eval()
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n.endswith("LayerNorm.weight"):
                    p.copy_(1.0 + 0.1 * torch.randn_like(p))
                elif n.endswith("bias"):
                    p.copy_(0.1 * torch.randn_like(p))
                else:
                    p.copy_(torch.randn_like(p) * 0.08)
        from safetensors.torch import save_file
        save_file({k: v.contiguous() for k, v in model.state_dict().items()}, os.path.join(d, "model.safetensors"))
        c = cfg.to_dict()
        c["model_type"] = "bert"
        json.dump(c, open(os.path.join(d, "config.json"), "w"), indent=1)
        os.makedirs(os.path.join(d, "1_Pooling"), exist_ok=True)
        json.dump({"pooling_mode_cls_token": True, "pooling_mode_mean_tokens": False},
                  open(os.path.join(d, "1_Pooling", "config.json"), "w"))
        cls = True
    else:
        tok = Tokenizer.from_file(os.path.join(d, "tokenizer.json"))
        model = BertModel.from_pretrained(d, attn_implementation="eager").eval()
        pc = os.path.join(d, "1_Pooling", "config.json")
        cls = json.load(open(pc)).get("pooling_mode_cls_token", True) if os.path.exists(pc) else True
    texts = SAMPLES + random_strings(500, seed=11)
    tok.no_truncation()
    cases = [{"text": t, "ids": tok.encode(t).ids} for t in texts]
    ref = {"cases": cases, "sentences": SENTENCES, "embeddings": pooled(model, tok, SENTENCES, cls)}
    json.dump(ref, open(os.path.join(d, "ferrolm-encoder.json"), "w"), ensure_ascii=False)
    print(d, len(cases), "tokenizer cases,", len(SENTENCES), "embeddings")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
