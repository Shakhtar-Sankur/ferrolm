#!/bin/sh
# Weight quantization: WikiText-2 perplexity and single-stream decode speed
# for bf16, int8, int4 and int4 with activation-aware scaling.
# Needs data/wikitext2-{test,train}.txt and models/smollm2-{360m,1.7b}.
set -eu
F=./target/release/ferrolm
cargo build --release -q
OUT=bench/results/quant.jsonl
: > $OUT
M=models/smollm2-360m
for q in bf16 int8 int4; do
  $F perplexity --model $M --data data/wikitext2-test.txt --quant $q --windows 20 --json $OUT
done
$F perplexity --model $M --data data/wikitext2-test.txt --quant int4 --awq --calib data/wikitext2-train.txt --windows 20 --json $OUT
# Decode speed: one request, 128 greedy tokens.
for m in models/smollm2-360m models/smollm2-1.7b; do
  for q in bf16 int8 int4; do
    printf '%s %s: ' "$m" "$q"
    $F generate --model $m --quant $q --chat --prompt "Explain how a hash map works." --max-tokens 128 2>&1 >/dev/null | tail -1
  done
done | tee bench/results/quant_speed.txt
