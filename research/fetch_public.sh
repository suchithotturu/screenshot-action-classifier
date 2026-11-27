#!/bin/bash
set -eu
cd "$(dirname "$0")/.."
mkdir -p data/raw
fetch() {
  local target="data/raw/$1" digest="$2" url="$3"
  if [[ -f "$target" ]] && [[ "$(shasum -a 256 "$target" | cut -d' ' -f1)" == "$digest" ]]; then return; fi
  curl -fL --retry 2 "$url" -o "$target.part"
  [[ "$(shasum -a 256 "$target.part" | cut -d' ' -f1)" == "$digest" ]] || { echo "Source archive changed: $1"; exit 1; }
  mv "$target.part" "$target"
}
fetch masc.zip bd2ac80a0a18a36f7aae796bf6b20b96fc502d4c5eddb031deee7f7a071badb7 https://www.kaggle.com/api/v1/datasets/download/alihmed/masc-dataset
fetch flowers.tgz 4c54ace7911aaffe13a365c34f650e71dd5bf1be0a58b464e5a7183e3e595d9c https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
fetch enrico-screens.zip 303c3fced6012f54606c32bcc3404a68e7aa2f9b046a298a749b4ba3f68e802d https://userinterfaces.aalto.fi/enrico/resources/screenshots.zip
fetch enrico-hierarchies.zip 53ead2025053643a077a0a2cf146ceecaf1ed74da83373fb6f22ed565051787b https://userinterfaces.aalto.fi/enrico/resources/hierarchies.zip
fetch design_topics.csv a3c7d74e00c1f9020cdeba9620f10c65edc3c91b74d545cf5b058e902283db54 https://raw.githubusercontent.com/luileito/enrico/master/design_topics.csv
fetch issues.csv e61078922b3bae2719fce4fa95518ffaabb123a531a6cef1eff96bec6eea31e5 https://raw.githubusercontent.com/luileito/enrico/master/issues.csv
fetch cord-train.parquet da3994eee1bf9bd3c57f0d53a72c3a6812c8696c5ba26245987949ddf73483cc https://huggingface.co/datasets/naver-clova-ix/cord-v2/resolve/main/data/train-00000-of-00004-b4aaeceff1d90ecb.parquet
