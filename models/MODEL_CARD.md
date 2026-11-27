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
