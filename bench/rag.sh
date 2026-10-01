#!/bin/sh
# Retrieval and RAG evaluation. Needs data/scifact (BEIR SciFact: corpus.jsonl,
# queries.jsonl, qrels-test.tsv), data/squad/dev.jsonl (SQuAD v1.1 dev as
# JSONL: question, context, answers), models/bge-small (BAAI/bge-small-en-v1.5)
# and models/smollm2-1.7b (SmolLM2-1.7B-Instruct).
set -eu
rm -f bench/results/retrieval.jsonl bench/results/qa.jsonl
PREFIX="Represent this sentence for searching relevant passages: "
F=./target/release/ferrolm
cargo build --release -q
# Embeds the corpus once; scores exact search and HNSW at several ef.
$F retrieval --encoder models/bge-small --beir data/scifact --index hnsw \
  --query-prefix "$PREFIX" --label bge-small --json bench/results/retrieval.jsonl
# The same evaluation with transformers, for comparison.
${PYTHON:-python3} scripts/retrieval_reference.py models/bge-small data/scifact "$PREFIX" \
  > bench/results/retrieval_reference.jsonl
$F qa --model models/smollm2-1.7b --encoder models/bge-small --data data/squad/dev.jsonl \
  --n ${QA_N:-200} --k 3 --index flat --query-prefix "$PREFIX" --kv-mem 2G \
  --label "SmolLM2-1.7B + bge-small" --json bench/results/qa.jsonl
