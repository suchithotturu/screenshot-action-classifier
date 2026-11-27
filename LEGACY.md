# Screenshot inbox

A local CS 513-B prototype for finding useful saved screens in a mixed phone export. The model reads pixels and proposes an image type, category and supporting text. The app derives cautious actions, preserves receipts as references and keeps ambiguous images available for review.

## Open and analyze

Double-click **Launch.command** in this folder. It starts or reuses Ollama, verifies that the local vision model is installed, and opens [Screenshot inbox](http://127.0.0.1:8513). Keep the launcher terminal open while using the app. On a narrow screen, expand the sidebar using the arrow at the upper left.

1. Click **Choose folder…** or paste the full path to an exported image folder. Click **Analyze**. The initial path opens the synthetic demo.
2. Filter by Actions, References, Needs review, Photos or Error. Use **Details and correction** to edit a suggestion, include a photo by assigning an actionable category and action, or mark an item reviewed, completed or dismissed.
3. **Cancel after current image** saves the in-flight result. Open a saved analysis and use **Resume / retry failed** to continue. An unchanged corrupt file will fail again. After replacing a source or changing settings, start a new analysis.
4. Open **Export this analysis** for JSON, CSV or an offline HTML report. Each export covers the whole selected run. It includes relative filenames, visible content, model provenance and corrections. HTML also includes thumbnails.

## Local commands

Run from this directory on an Apple silicon Mac. Setup requires uv, internet access and several GB of disk space. The prepared installation already includes Python, libraries, Ollama and model weights.

```sh
./setup.sh                        # provision or repair
./Launch.command                  # normal launch
source ./service.sh               # model only, for CLI use
.venv/bin/python -m classifier.batch --folder "./demo"
.venv/bin/python -m pytest -q
.venv/bin/python smoke.py          # actual image inference
.venv/bin/python verify_local.py   # analyze demo once first to populate cache
```

The app listens on 127.0.0.1:8513. Ollama listens on 127.0.0.1:11434. Launch uses local weights and assets. Setup is the download step. Close Streamlit with Ctrl-C in its terminal. The reusable local model service can outlive the app. Ollama unloads idle weights.

## How it works

The scanner recursively sorts source entries, skips links, checks input size and decoded dimensions, reads the first frame, applies EXIF orientation and prepares RGB JPEG pixels in memory. PNG, JPEG, WebP, HEIC and HEIF use Pillow and pillow-heif. Exact duplicate bytes share a SHA-256 identifier and keep distinct relative paths. Different image bytes are distinct. The model receives one image, the classification prompt and JSON schema. Filenames are excluded from inference. A schema failure gets one retry. The default maximum side is 1200 pixels, with one worker, 4096 context tokens, 650 generated tokens and a 150-second request timeout.

The image-type decision is visual and can be wrong, particularly for cropped screenshots and photos of screens. Quotes and entities are model transcriptions. Review them against the original before using them. A complete supported date must match the extracted text before normalization. The app recognizes ISO and English month-name dates. A deadline requires a deadline role. Display uses the current date in America/New_York. Completion and dismissal are user decisions.

SQLite in data/library.db holds run manifests, predictions and separate human edits. Thumbnails and logs also live in data/. Files under .models/, .runtime/ and .venv/ are local installation assets. A cache hit requires matching content and analysis configuration, including the model digest, prompt hash, preprocessing and policy versions. Edits follow the content hash across runs. To test a different local vision model or resolution, set VISION_MODEL or IMAGE_SIDE before launching. DISPLAY_TIMEZONE accepts an IANA timezone. The prompt lives in prompt.txt. The taxonomy, schema, policy and decoding settings live in classifier/model.py. Restart the app after configuration changes.

## Demo, sources and course work

The demo contains seven rendered phone screens, one blank crop, a duplicate, an invalid JPEG, an unsupported text file and a public NASA astronaut portrait. Rebuild the rendered fixtures with `.venv/bin/python fixtures.py`. These fixtures exercise the workflow. See [the verification log](VERIFICATION.md) for observed results, and [the sample report](samples/demo.html) for cards, evidence and provenance.

The portrait comes from the [scikit-image sample collection](https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.astronaut). Runtime and model references include the [Ollama model card](https://ollama.com/library/qwen3-vl:4b-instruct) and [structured-output API](https://docs.ollama.com/capabilities/structured-outputs).

### Evaluation plan

The class guidelines specify a team of 3–4. Team arrangements and the final presentation remain course administration and reporting work. The prototype uses pretrained model-assisted labeling. Model outputs are pseudo-labels. Human judgments of visible utility and optional owner-reported intentions are separate observations.

Use research/annotations.csv for consented real examples. Before tuning, group exact copies, crops and related screens, then reserve validation and test groups. Annotators should label image type and one primary utility category independently of model predictions. Resolve disagreements and preserve counts. Record the OCR engine/version and elapsed extraction time. Use OCR text derived from pixels alone. Keep owner statements, model explanations and generated names out of the features.

The optional comparison script takes precomputed OCR, real human labels and an actual JSON run export. It fits character TF-IDF on training screenshots only and compares Naive Bayes, linear SVM and the frozen VLM on identical held-out IDs. Use validation to choose settings and evaluate the untouched test partition once. Missing VLM results and rejected screenshots count as abstentions. Report class support, macro F1 across all six classes, per-class precision and recall, confusion matrices, abstentions and latency. Measure screenshot detection separately on the full mixed set. Add OCR latency to baseline prediction time for an end-to-end comparison. No real annotations have been supplied.

```sh
uv run --script research/evaluate.py research/annotations.csv samples/demo.json --partition validation
```
