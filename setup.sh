#!/bin/bash
set -eu
cd "$(dirname "$0")"
mkdir -p .runtime data
[[ "$(uname -s)" == Darwin && "$(uname -m)" == arm64 ]] || { echo "This version requires an Apple silicon Mac (Apple Vision OCR)."; exit 1; }
xcode-select -p >/dev/null 2>&1 || { echo "Run xcode-select --install, finish installation, then retry."; exit 1; }
[[ $# -eq 0 || "$1" == --annotation ]] || { echo "Usage: ./setup.sh [--annotation]"; exit 1; }
uvbin="$(command -v uv || true)"
[[ -x "$uvbin" ]] || uvbin="$HOME/Developer/tools/bin/uv"
[[ -x "$uvbin" ]] || { echo "Install uv from docs.astral.sh/uv and retry."; exit 1; }
export UV_CACHE_DIR="$PWD/.uv-cache" UV_PYTHON_INSTALL_DIR="$PWD/.python"
[[ -x .venv/bin/python ]] || "$uvbin" venv --python 3.12.16
"$uvbin" pip sync --python .venv/bin/python requirements.lock
./setup-features.sh
(cd models && shasum -a 256 -c classifier.sha256)
.venv/bin/python -c 'from classifier.learning import load_model; from classifier.schema import NAMES; m=load_model(); assert set(m["tasks"]["content_type"]["model"].classes_)==set(NAMES); print("Eight-category classifier ready.")'
if [[ "${1:-}" == --annotation ]]; then
  ./setup-annotation.sh
fi
echo "Start the app: .venv/bin/python -m classifier app --mode classify"
