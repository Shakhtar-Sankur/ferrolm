#!/bin/sh
# Retrieval and RAG evaluation. Needs data/scifact (BEIR SciFact: corpus.jsonl,
# queries.jsonl, qrels-test.tsv), data/squad/dev.jsonl (SQuAD v1.1 dev as
# JSONL: question, context, answers), models/bge-small (BAAI/bge-small-en-v1.5)
# and models/smollm2-1.7b (SmolLM2-1.7B-Instruct).
set -eu
PREFIX="Represent this sentence for searching relevant passages: "
F=./target/release/ferrolm
cargo build --release -q
for index in flat hnsw; do
  $F retrieval --encoder models/bge-small --beir data/scifact --index $index \
    --query-prefix "$PREFIX" --label "bge-small, $index" --json bench/results/retrieval.jsonl
done
$F qa --model models/smollm2-1.7b --encoder models/bge-small --data data/squad/dev.jsonl \
  --n ${QA_N:-200} --k 3 --index hnsw --query-prefix "$PREFIX" --kv-mem 2G \
  --label "SmolLM2-1.7B + bge-small" --json bench/results/qa.jsonl
