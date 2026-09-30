#!/bin/sh
# Downloads a model from Hugging Face: fetch_model.sh ORG/NAME DIR
set -e
mkdir -p "$2"
mkdir -p "$2/1_Pooling"
for f in config.json generation_config.json tokenizer.json tokenizer_config.json model.safetensors modules.json 1_Pooling/config.json; do
  curl -sSfL --retry 4 -o "$2/$f" "https://huggingface.co/$1/resolve/main/$f" || echo "missing $1/$f"
done
