#!/bin/bash
set -eu
cd "$(dirname "$0")"
[[ "$(uname -s)" == Darwin && "$(uname -m)" == arm64 ]] || { echo "Requires an Apple silicon Mac."; exit 1; }
[[ -x .venv/bin/python ]] || { echo "Run ./setup.sh first."; exit 1; }
mkdir -p .runtime data
if [[ ! -x .runtime/ollama ]]; then
  curl -fL --retry 2 https://github.com/ollama/ollama/releases/download/v0.40.17/ollama-darwin.tgz -o .runtime/ollama.tgz
  tar -xzf .runtime/ollama.tgz -C .runtime
fi
source ./service.sh
.runtime/ollama pull "${VISION_MODEL:-qwen3-vl:4b-instruct}"
echo "Local annotation model ready. Choose Dataset Annotation in the app."
