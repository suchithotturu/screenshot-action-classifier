#!/bin/bash
set -eu
cd "$(dirname "$0")"
[[ -x .venv/bin/python ]] || { echo "Run ./setup.sh first."; exit 1; }
url=http://127.0.0.1:8514
if curl --noproxy '*' -fsS "$url/_stcore/health" >/dev/null 2>&1; then open "$url"; exit; fi
(for i in {1..60}; do
  if curl --noproxy '*' -fsS "$url/_stcore/health" >/dev/null 2>&1; then open "$url"; break; fi
  sleep .5
done) &
exec .venv/bin/python -m streamlit run app.py
