"""Records Hugging Face's reference outputs for a downloaded model, next to
its weights, for tests/reference.rs and tests/tokenizer.rs to compare
against (set FERROLM_REAL_MODELS=dir1,dir2,... when running the tests):

  ferrolm-reference.json  transformers' float32 logits for a text prompt and
                          its greedy continuation
  ferrolm-tokenizer.json  `tokenizers` encodings of the test strings, and the
                          chat template rendered by transformers

Usage: python scripts/reference_real.py MODEL_DIR [MODEL_DIR ...]
"""

import json
import os
import sys

import torch
from tokenizers import Tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

sys.path.insert(0, os.path.dirname(__file__))
from make_tokenizer_fixtures import SAMPLES, random_strings  # noqa: E402

PROMPT = (
    "The history of computing is a story of abstraction. Early machines were programmed by rewiring "
    "circuits; later, stored programs let the same hardware run many tasks. Compilers, operating systems "
    "and networks each hid a layer of detail. Today, large language models"
)
CHAT = [
    {"role": "system", "content": "You are a concise assistant."},
    {"role": "user", "content": "Name three prime numbers."},
    {"role": "assistant", "content": "2, 3 and 5."},
    {"role": "user", "content": "And the next one?"},
]


def main(dirs):
    torch.manual_seed(0)
    for d in dirs:
        tok = Tokenizer.from_file(os.path.join(d, "tokenizer.json"))
        auto = AutoTokenizer.from_pretrained(d)
        texts = SAMPLES + random_strings(1000, seed=7)
        cases = [{"text": t, "ids": tok.encode(t, add_special_tokens=True).ids} for t in texts]
        chats = []
        for conv in (CHAT, CHAT[1:2]):
            chats.append({
                "messages": conv,
                "prompt": auto.apply_chat_template(conv, tokenize=False, add_generation_prompt=True),
            })
        with open(os.path.join(d, "ferrolm-tokenizer.json"), "w") as f:
            json.dump({"cases": cases, "chats": chats}, f, ensure_ascii=False)

        model = AutoModelForCausalLM.from_pretrained(d, torch_dtype=torch.float32, attn_implementation="eager").eval()
        ids = tok.encode(PROMPT, add_special_tokens=True).ids
        prompt = torch.tensor([ids])
        rows = [0, 1, len(ids) // 2, len(ids) - 1]
        with torch.no_grad():
            logits = model(prompt).logits[0]
            # Plain greedy decoding: the model's generation_config.json may
            # add a repetition penalty or sampling defaults (Qwen2.5 does).
            plain = GenerationConfig(max_new_tokens=32, do_sample=False, repetition_penalty=1.0,
                                     eos_token_id=None, pad_token_id=0)
            gen = model.generate(prompt, generation_config=plain)[0, len(ids):]
        ref = {
            "prompt": ids,
            "logits_rows": rows,
            "logits": [logits[i].tolist() for i in rows],
            "greedy": gen.tolist(),
            "transformers": __import__("transformers").__version__,
            "torch": torch.__version__,
        }
        with open(os.path.join(d, "ferrolm-reference.json"), "w") as f:
            json.dump(ref, f)
        print(d, len(ids), "prompt tokens; greedy:", repr(tok.decode(gen.tolist())))


if __name__ == "__main__":
    main(sys.argv[1:])
