#!/bin/sh
# SIFT1M: ferrolm's HNSW and IVF-PQ against FAISS with the same parameters.
# Needs data/sift/sift_{base,query,learn}.fvecs and sift_groundtruth.ivecs
# (ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz) and `pip install faiss-cpu`.
# Run on an otherwise idle machine: QPS is measured.
set -eu
PY=${PYTHON:-python3}
OUT=${1:-bench/results/ann.jsonl}
S=data/sift
cargo build --release -q
: > "$OUT"
for index in hnsw ivfpq; do
  ./target/release/ferrolm ann --index $index --base $S/sift_base.fvecs --query $S/sift_query.fvecs \
    --truth $S/sift_groundtruth.ivecs --learn $S/sift_learn.fvecs --json "$OUT"
  $PY scripts/ann_faiss.py $S $index "$OUT"
done
