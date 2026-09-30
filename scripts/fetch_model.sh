#!/bin/sh
# Downloads a model from Hugging Face: fetch_model.sh ORG/NAME DIR
set -e
mkdir -p "$2"
for f in config.json generation_config.json tokenizer.json tokenizer_config.json model.safetensors; do
  curl -sSfL --retry 4 -o "$2/$f" "https://huggingface.co/$1/resolve/main/$f" || echo "missing $1/$f"
done
