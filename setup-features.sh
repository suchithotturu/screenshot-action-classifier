#!/bin/bash
set -eu
cd "$(dirname "$0")"
mkdir -p .models/clip .runtime
clang -fobjc-arc -O2 -framework Foundation -framework Vision -framework CoreGraphics classifier/vision.m -o .runtime/astra-ocr
sha=583fd1110a514667812fee7d684952aaf82a99b959760c8d7dca7e0ab9839299
checkpoint=.models/clip/vision.onnx
if [[ ! -f "$checkpoint" ]] || [[ "$(shasum -a 256 "$checkpoint" | cut -d' ' -f1)" != "$sha" ]]; then
  curl -fL --retry 2 https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/6ef1ebc/onnx/vision_model_quantized.onnx -o "$checkpoint.part"
  [[ "$(shasum -a 256 "$checkpoint.part" | cut -d' ' -f1)" == "$sha" ]] || { echo "CLIP checksum mismatch"; exit 1; }
  mv "$checkpoint.part" "$checkpoint"
fi
