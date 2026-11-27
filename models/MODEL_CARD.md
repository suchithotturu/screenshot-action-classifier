# Eight-category classifier

Training ID: `310009966ab1d1e6`. Integrity: `classifier.sha256`.

The joblib artifact contains a fitted content logistic-regression head, a separate screenshot logistic-regression head, their preprocessing transforms, ordered feature names, thresholds and runtime versions. It contains no training images, personal screenshots, OCR caches or annotation database. Content uses 3,000 TF-IDF features, 15 standardized measurements and 512 normalized CLIP visual features. The detector uses CLIP alone. Only load the trusted artifact distributed with this repository; joblib uses Python pickle serialization.

The study used 1,280 public images from MASC, Enrico, TensorFlow Flowers and CORD. See [third-party notices](../THIRD_PARTY.md). Training used 762 source labels and 149 explicit Qwen pseudo-labels. Validation used 152 source labels and 37 pseudo-labels. No owner-purpose classifier is available. No personal-folder images entered training.

The source-only content test achieved 139/155 correct across six supported categories. Shopping and entertainment lack independent test labels. The separate 25-row machine-label check achieved 17/25 teacher agreement. Broad real-world accuracy is not established. CORD receipt photos differ from payment-screen screenshots, ordinary flower photos supply narrow Miscellaneous examples, and app/source styles may be shortcuts. Miscellaneous also receives uncertain cases through deployment policy. Confident incorrect predictions remain possible.

Training environment: Python 3.12.16, scikit-learn 1.7.2, Apple Vision revision 3 English OCR on macOS 15.7.9, ONNX Runtime 1.23.2, Pillow 13.0.0, quantized CLIP ViT-B/32. Setup pins the encoder checksum and Python dependencies. macOS version differences are allowed while retaining the full OS value in feature-cache provenance; OCR results can vary between OS releases. Changes to the encoder, preprocessing or feature/runtime versions still require retraining.

All model inputs are automatically extracted from image pixels. Source names, file paths, owner IDs, Qwen descriptions and annotation text are excluded. See [results](../reports/results.json) for measurements and limitations.

## Windows / RapidOCR artifact

The separate `rapidocr/classifier.joblib` artifact has training ID `5b01904228d376bd`; its integrity file is `rapidocr/classifier.sha256`. Windows selects it automatically. It was fitted on features from RapidOCR-ONNXRuntime 1.4.4 with checksum-verified bundled PP-OCRv4 models, using exactly the same 911/189/180 public group partitions as the Mac study. No new Qwen annotations or personal images were used.

Logistic regression with OCR/numeric/CLIP fusion won provisional validation at macro F1 0.788. Source-only holdout content accuracy was 140/155 (90.3%) across six supported classes; the separate 25-row machine-label check achieved 15/25 teacher agreement. The same source and pseudo-label caveats apply. Measurements came from the portable backend running on macOS; native Windows runtime performance and installation remain to be verified on a Windows machine. See [portable results](../reports/rapidocr/results.json).

## Current character-feature update

The prepared models now include 1,500 additional train-fitted character TF-IDF features, for 5,027 content inputs. The Mac artifact is `3cae9ac061854287`; the portable/Windows artifact is `642c50cc3e8cbd86`. Both remain logistic regression trained only on the original public corpus. Their source-only test accuracies are 141/155 and 143/155, respectively. Character features won public validation over baseline, layout-only and combined conditions; test scores did not select them. The spatial module is implemented and measured, but is not in these winning prepared heads.

A separate 1,600-screen synthetic corpus and mixed-training experiment are documented in [the feature study](../reports/feature-study/REPORT.md). Synthetic-only and mixed artifacts are experimental and are not shipped as the default. Historical figures above describe their original artifacts. Current model integrity files and results.json match the new active artifacts.

## Engineered visual experiment

HOG (3,780), HSV histograms (24) and spatial edge density (16) were tested as additions and as image-only controls. No addition won validation, so the active prepared artifacts above are unchanged. Alternative visual artifacts are fitted locally by `research.visual_study`; their feature version is checked independently of the existing OCR/CLIP contract. See [the measurements](../reports/visual-study/REPORT.md).
