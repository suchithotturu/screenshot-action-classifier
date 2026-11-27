# Screenshot Intelligence · CS 513

A local screenshot and image organizer with an eight-category supervised classifier, a dataset annotation workbench and measured comparisons of logistic regression, linear SVM, k-NN and Complement Naive Bayes.

**Runs on Apple silicon macOS and 64-bit Intel/AMD Windows.** Mac uses Apple Vision OCR; Windows uses local RapidOCR with a separately fitted model. Both trained models are included. Classification needs no API key, Qwen installation, training database or dataset download.

## Windows quick start

Install Git and [uv](https://docs.astral.sh/uv/getting-started/installation/), then open PowerShell:

```powershell
git clone https://github.com/suchithotturu/screenshot-action-classifier.git
cd screenshot-action-classifier
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup.ps1
.\.venv\Scripts\python.exe -m classifier app --mode classify
```

Open **http://127.0.0.1:8514**. After setup, you can double-click **Launch-Windows.cmd** instead. Setup installs Python and the pinned Windows dependencies, uses the small packaged RapidOCR models, downloads the checksum-pinned CLIP encoder and checks the included Windows classifier. No GPU, Apple tools or Qwen are required. Windows on ARM is not yet supported by these pinned dependencies.

If ONNX Runtime reports a missing DLL, install Microsoft's current [Visual C++ x64 Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist) and rerun setup. The `-ExecutionPolicy Bypass` option applies to that setup process; it does not change your saved PowerShell policy. See [WINDOWS.md](WINDOWS.md) for annotation, training and verification commands.

## Mac quick start

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and Apple's Command Line Tools (`xcode-select --install`) first. Then:

```sh
git clone https://github.com/suchithotturu/screenshot-action-classifier.git
cd screenshot-action-classifier
./setup.sh
.venv/bin/python -m classifier app --mode classify
```

Open **http://127.0.0.1:8514**. Alternatively, double-click `Launch.command` after setup. Expand **Choose an image folder**, enter an absolute path to your screenshots and click **Classify folder**. The included `demo` directory contains synthetic examples, a public NASA portrait, a duplicate and deliberately invalid files to exercise error handling. It is a workflow demo, not an accuracy benchmark.

Setup creates an isolated Python 3.12.16 environment, installs `requirements.lock`, compiles the Vision OCR helper, downloads the checksum-pinned CLIP encoder (about 89 MB), and verifies the included classifier. Initial setup needs internet access. Classification works locally afterwards. Setup does **not** download or call Qwen.

If `uv` is not found, add it to your shell's PATH and reopen Terminal. If Apple developer tools are missing, finish `xcode-select --install` before rerunning setup. Port 8514 must be free; use `.venv/bin/python -m classifier app --port 8515` if necessary, then open that port.

## App flow

1. **Dataset Annotation:** import images, obtain optional local Qwen suggestions, inspect/correct labels, save review state and export the dataset. Source labels and machine pseudo-labels retain distinct provenance. A suggestion does not count as human approval.
2. **Training:** extract automatic pixel features, split related images into separate groups, fit transforms only on training examples, compare classifiers on validation data and evaluate the saved winner on holdout data. Training requires an explicit command.
3. **Classify Images:** choose a fresh folder. The saved model predicts content from the same automatic features. Confident learned predictions take priority; local OCR rules assist uncertain cases, with remaining uncertainty displayed as Miscellaneous. Screenshot detection is advisory. Processing errors remain separate. Results are saved locally and can be filtered or downloaded.

The categories are Messages & conversations, Social & entertainment, Places & recommendations, Products & offers, Reading & information, Receipts & confirmations, App screens and Miscellaneous. Classification runs do not automatically enter training.

## Features and models

The Mac content model is **logistic regression** on 3,527 features: 3,000 OCR TF-IDF word/phrase features, 512 frozen CLIP image features and 15 engineered text/image measurements. The Windows/RapidOCR model is trained and selected separately using the same public partitions; its measurements are in [reports/rapidocr/results.json](reports/rapidocr/results.json). The separate screenshot detector uses CLIP features. Names, paths, source metadata and Qwen descriptions never enter the classifier features. Qwen is used only for annotation. CLIP is a pretrained deep-learning feature extractor, not a neural network trained from scratch for this project.

| Algorithm | Best feature condition | Validation macro F1 |
| --- | --- | ---: |
| Logistic regression | OCR + numeric + CLIP | 0.762 |
| Linear SVM | OCR + numeric + CLIP | 0.743 |
| k-NN (5 neighbors, cosine distance) | OCR TF-IDF | 0.643 |
| Complement Naive Bayes | OCR TF-IDF | 0.526 |

The enlarged study contains 1,280 eligible images: 911 training, 189 validation and 180 public holdout records, with zero tracked group overlaps. Source-only content testing achieved **89.7% accuracy on 155 images**, with six supported categories. Shopping and entertainment have no independent test labels: a separate 25-image machine-label check achieved **68% teacher agreement**. Validation also includes pseudo-labels, so the comparison is exploratory. Flower/photo and receipt/source differences can inflate apparent performance; broad phone-library accuracy is not established.

See the [research report](reports/REPORT.md), [complete measurements](reports/results.json), [offline presentation](reports/presentation.html), and [model card](models/MODEL_CARD.md).

## Optional local Qwen annotation

Classification is ready after the basic setup. To enable assisted dataset creation as well:

```sh
./setup-annotation.sh
# Or, on a fresh clone: ./setup.sh --annotation
.venv/bin/python -m classifier app --mode annotation
```

This downloads Ollama and Qwen3-VL 4B Instruct locally; allow several additional gigabytes of disk space. The service binds to loopback. Manual annotation is available without Qwen. Start the service again later with `bash service.sh`. Owner purpose and usefulness remain unknown unless the image owner confirms them. See [LABELING.md](LABELING.md).

## Training your own model

The repository contains the trained artifact and aggregate reports, not the original training database or raw images. For a new study, annotate your own imported images and run:

```sh
.venv/bin/python -m classifier features
.venv/bin/python -m classifier train
.venv/bin/python -m classifier evaluate
.venv/bin/python -m classifier export
```

For the public-data acquisition workflow:

```sh
bash research/fetch_public.sh
.venv/bin/python -m research.prepare_public
.venv/bin/python -m research.import_public
./setup-annotation.sh
.venv/bin/python -m research.expand_taxonomy
.venv/bin/python -m research.train_eight
.venv/bin/python -m classifier evaluate
.venv/bin/python -m research.demo_check --cold
```

Public downloads total about 1.4 GiB before extraction. This creates a new grouped experiment; fresh Qwen suggestions may differ from the published study. `train_eight` blocks Qwen and Python network connections during training, preserves a database backup and publishes only when the fitted classifier contains all eight categories. Do not run it concurrently with annotation.

## Verification and storage

```sh
.venv/bin/python -m pytest -q
.venv/bin/python -m classifier predict ./demo
```

The suite covers annotations, exports/restores, grouping, train-only feature fitting, content routing and model portability. The public holdout demonstration script requires the locally rebuilt research database. A separate fresh-clone check is recorded in [SHARING.md](SHARING.md).

Images, OCR, annotation databases, feature caches, exports and classification history stay under the ignored `data/` directory. Model/runtime downloads stay under ignored `.models/` and `.runtime/`. The GitHub repository contains code, synthetic fixtures, the fitted classifier and aggregate research artifacts. It contains no personal screenshots or private database. See [third-party notices](THIRD_PARTY.md) for dataset and model attribution. The original Qwen-first inbox remains available as **Legacy inbox**.
