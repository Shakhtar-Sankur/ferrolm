#!/bin/sh
# Reproduces the README's benchmarks. Needs the models from
# scripts/fetch_model.sh in models/. Results go to bench/results/*.jsonl.
set -e
B=./target/release/ferrolm
M=models/smollm2-360m
OUT=bench/results
mkdir -p $OUT
rm -f $OUT/*.jsonl
cargo build --release -q

# 1. Scheduling: the same 64 requests of mixed lengths, all queued at once.
W="--requests 64 --prompt-len 32..256 --gen-len 16..256 --seed 1"
$B bench --model $M $W --max-seqs 1 --label sequential --json $OUT/policies.jsonl
$B bench --model $M $W --max-seqs 32 --admission static --label static --json $OUT/policies.jsonl
$B bench --model $M $W --max-seqs 32 --label continuous --json $OUT/policies.jsonl

# 2. Latency under load: Poisson arrivals at increasing rates.
for rate in 0.5 1 2 3 4; do
  for adm in static continuous; do
    $B bench --model $M --requests 48 --prompt-len 32..256 --gen-len 16..256 --seed 2 --rate $rate \
      --max-seqs 32 --admission $adm --label "$adm@$rate" --json $OUT/load.jsonl
  done
done

# 3. Memory: 256 MB of KV cache, paged vs reserving a 2048-token window per sequence.
$B bench --model $M $W --kv-mem 256M --reserve 2048 --label reserved-2048 --json $OUT/memory.jsonl
$B bench --model $M $W --kv-mem 256M --label paged --json $OUT/memory.jsonl

# 4. Prefix caching: a 512-token system prompt shared by every request.
P="--requests 32 --shared-prefix 512 --prompt-len 16..64 --gen-len 32 --rate 1 --seed 3"
$B bench --model $M $P --no-prefix-cache --label no-prefix-cache --json $OUT/prefix.jsonl
$B bench --model $M $P --label prefix-cache --json $OUT/prefix.jsonl

# 5. Speculative decoding on real prompts: SmolLM2-1.7B with SmolLM2-135M as the draft.
S="--model models/smollm2-1.7b --prompt-file bench/prompts.txt --requests 6 --gen-len 128 --kv-mem 1G"
$B bench $S --max-seqs 1 --label "1.7b k=0" --json $OUT/spec.jsonl
for k in 2 3 4 5 6; do
  $B bench $S --max-seqs 1 --draft models/smollm2-135m --spec-k $k --label "1.7b k=$k" --json $OUT/spec.jsonl
done
S="--model models/smollm2-1.7b --prompt-file bench/prompts.txt --requests 16 --gen-len 128 --kv-mem 1G --max-seqs 8"
$B bench $S --label "1.7b batch8 k=0" --json $OUT/spec.jsonl
$B bench $S --draft models/smollm2-135m --spec-k 4 --label "1.7b batch8 k=4" --json $OUT/spec.jsonl
