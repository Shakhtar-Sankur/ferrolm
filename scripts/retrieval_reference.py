"""The same BEIR retrieval evaluation as `ferrolm retrieval`, computed with
Hugging Face transformers (float32, CLS pooling, L2-normalised, exact
search), for checking ferrolm's nDCG@10 and recall@100.

Usage: python scripts/retrieval_reference.py MODEL_DIR BEIR_DIR [QUERY_PREFIX]
"""

import json
import math
import os
import sys
import time

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


def jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def embed(tok, model, texts, batch=32):
    out = []
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    vecs = [None] * len(texts)
    with torch.no_grad():
        for s in range(0, len(order), batch):
            idx = order[s:s + batch]
            enc = tok([texts[i] for i in idx], padding=True, truncation=True, max_length=512, return_tensors="pt")
            h = model(**enc).last_hidden_state[:, 0]
            h = torch.nn.functional.normalize(h, dim=-1)
            for i, v in zip(idx, h.numpy()):
                vecs[i] = v
    return np.stack(vecs)


def ndcg(ranked, rel, k=10):
    dcg = sum(rel.get(d, 0) / math.log2(i + 2) for i, d in enumerate(ranked[:k]))
    ideal = sorted((r for r in rel.values() if r > 0), reverse=True)
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal[:k]))
    return dcg / idcg if idcg else 0.0


def main(model_dir, beir, prefix=""):
    torch.set_num_threads(os.cpu_count())
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModel.from_pretrained(model_dir).eval()
    corpus = jsonl(os.path.join(beir, "corpus.jsonl"))
    ids = [d["_id"] for d in corpus]
    docs = [(d["title"] + " " + d["text"]).strip() for d in corpus]
    queries = {q["_id"]: q["text"] for q in jsonl(os.path.join(beir, "queries.jsonl"))}
    qrels = {}
    with open(os.path.join(beir, "qrels-test.tsv")) as f:
        next(f)
        for line in f:
            q, d, r = line.rstrip("\n").split("\t")
            qrels.setdefault(q, {})[d] = int(r)
    qids = sorted(q for q in qrels if q in queries)
    t = time.time()
    D = embed(tok, model, docs)
    secs = time.time() - t
    ntok = sum(len(x) for x in tok(docs, truncation=True, max_length=512)["input_ids"])
    Q = embed(tok, model, [prefix + queries[q] for q in qids])
    scores = Q @ D.T
    n10 = r100 = 0.0
    for qi, q in enumerate(qids):
        top = np.argsort(-scores[qi], kind="stable")[:101]
        ranked = [ids[j] for j in top if ids[j] != q][:100]
        rel = qrels[q]
        n10 += ndcg(ranked, rel)
        r100 += sum(1 for d in ranked if rel.get(d, 0) > 0) / sum(1 for r in rel.values() if r > 0)
    print(json.dumps({"label": "transformers reference", "prefix": prefix, "queries": len(qids),
                      "ndcg10": round(n10 / len(qids), 5), "recall100": round(r100 / len(qids), 5),
                      "embed_tokens_per_s": round(ntok / secs), "threads": torch.get_num_threads()}))


if __name__ == "__main__":
    main(*sys.argv[1:])
