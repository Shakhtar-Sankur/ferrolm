"""Builds small random-weight models with Hugging Face transformers and
records transformers' own logits and greedy continuations for them, so the
Rust tests can check ferrolm against the reference implementation without
downloading anything.

Usage: python scripts/make_fixtures.py tests/fixtures
"""

import json
import os
import sys

import torch
from safetensors.torch import save_file
from transformers import LlamaConfig, LlamaForCausalLM, Qwen2Config, Qwen2ForCausalLM

VARIANTS = {
    # Llama with grouped-query attention and untied embeddings.
    "llama-gqa": (LlamaConfig, LlamaForCausalLM, dict(
        vocab_size=512, hidden_size=128, intermediate_size=352, num_hidden_layers=3,
        num_attention_heads=8, num_key_value_heads=2, max_position_embeddings=512,
        rms_norm_eps=1e-5, rope_theta=10000.0, tie_word_embeddings=False,
        bos_token_id=1, eos_token_id=2)),
    # Llama 3 style: tied embeddings and llama3 RoPE scaling.
    "llama3-tied": (LlamaConfig, LlamaForCausalLM, dict(
        vocab_size=384, hidden_size=96, intermediate_size=256, num_hidden_layers=2,
        num_attention_heads=6, num_key_value_heads=3, head_dim=16, max_position_embeddings=2048,
        rms_norm_eps=1e-5, rope_theta=500000.0, tie_word_embeddings=True,
        rope_scaling=dict(rope_type="llama3", factor=8.0, low_freq_factor=1.0,
                          high_freq_factor=4.0, original_max_position_embeddings=64),
        bos_token_id=1, eos_token_id=[2, 3])),
    # Qwen2: biases on q, k and v.
    "qwen2": (Qwen2Config, Qwen2ForCausalLM, dict(
        vocab_size=448, hidden_size=64, intermediate_size=160, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=1024,
        rms_norm_eps=1e-6, rope_theta=1000000.0, tie_word_embeddings=True,
        bos_token_id=1, eos_token_id=2)),
}


def main(out_dir):
    torch.manual_seed(0)
    for name, (cfg_cls, model_cls, kw) in VARIANTS.items():
        d = os.path.join(out_dir, name)
        os.makedirs(d, exist_ok=True)
        cfg = cfg_cls(**kw, attn_implementation="eager")
        model = model_cls(cfg).eval()
        # Random init is too tame to exercise the attention; widen it, then
        # round every weight to bf16 so both sides use identical values.
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n.endswith("norm.weight"):
                    p.copy_(1.0 + 0.1 * torch.randn_like(p))
                elif n.endswith("bias"):
                    p.copy_(0.5 * torch.randn_like(p))
                else:
                    p.copy_(torch.randn_like(p) * (2.0 / p.shape[-1]) ** 0.5)
                p.copy_(p.to(torch.bfloat16).to(torch.float32))
        state = {k: v.to(torch.bfloat16).contiguous() for k, v in model.state_dict().items()}
        if cfg.tie_word_embeddings:
            state.pop("lm_head.weight", None)
        save_file(state, os.path.join(d, "model.safetensors"), metadata={"format": "pt"})
        c = cfg.to_dict()
        c["model_type"] = cfg.model_type
        with open(os.path.join(d, "config.json"), "w") as f:
            json.dump(c, f, indent=1, sort_keys=True)

        g = torch.Generator().manual_seed(1)
        prompt = torch.randint(4, cfg.vocab_size, (1, 150), generator=g)
        with torch.no_grad():
            logits = model(prompt).logits[0]
            gen = model.generate(prompt, max_new_tokens=40, do_sample=False,
                                 eos_token_id=None, pad_token_id=0)[0, prompt.shape[1]:]
        ref = {
            "prompt": prompt[0].tolist(),
            "logits_rows": [0, 1, 17, 63, 64, 100, 149],
            "logits": [logits[i].tolist() for i in [0, 1, 17, 63, 64, 100, 149]],
            "greedy": gen.tolist(),
            "transformers": __import__("transformers").__version__,
            "torch": torch.__version__,
        }
        with open(os.path.join(d, "reference.json"), "w") as f:
            json.dump(ref, f)
        print(name, sum(p.numel() for p in model.parameters()), "params")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "tests/fixtures")
