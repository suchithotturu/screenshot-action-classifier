#!/bin/bash
export OLLAMA_HOST=127.0.0.1:11434 OLLAMA_MODELS="$PWD/.models" OLLAMA_NO_CLOUD=1 OLLAMA_NUM_PARALLEL=1
mkdir -p data
if ! curl --noproxy '*' -fsS --max-time 2 "http://$OLLAMA_HOST/api/version" >/dev/null 2>&1; then
  nohup .runtime/ollama serve >data/ollama.log 2>&1 </dev/null &
  for i in {1..50}; do
    curl --noproxy '*' -fsS "http://$OLLAMA_HOST/api/version" >/dev/null 2>&1 && break
    sleep .2
  done
fi
curl --noproxy '*' -fsS "http://$OLLAMA_HOST/api/version" >/dev/null || { echo "Model service failed. See data/ollama.log"; exit 1; }
