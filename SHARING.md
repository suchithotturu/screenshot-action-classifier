# Sharing and fresh-clone verification

The repository includes the fitted classifier so a friend can run inference immediately after setup. It excludes the original database, personal screenshots, raw research downloads, generated exports, private history and runtime caches.

Supported target: Apple silicon macOS with Apple Command Line Tools and uv. Windows and Linux require replacing Apple Vision OCR, rebuilding the feature contract and retraining; they are not supported by these instructions.

The prepared Mac previously passed an empty-cache ten-image check with Qwen stopped and Python networking blocked, no training-image overlap, and identical predictions after reload. This does not substitute for a fresh checkout check.
## Publication check · November 27, 2026

A clean local Git clone was created in a new temporary directory on the prepared Apple silicon Mac. It had no existing virtual environment, training database, feature cache, OCR helper, CLIP weights or Qwen installation. `./setup.sh` completed successfully: Python/dependencies were provisioned, the OCR helper compiled, CLIP downloaded and passed its SHA-256 check, and the included eight-category model loaded successfully.

`.venv/bin/python -m research.verify_shared_install` then ran with the Qwen client and Python socket/DNS access blocked. All 10 valid demo inputs processed successfully; the deliberately broken JPEG produced the expected error and the text file was skipped. Training records remained zero before and after classification. The saved model retained training ID `310009966ab1d1e6` and all eight classes. This used actual Apple Vision OCR and ONNX CLIP, not mocked feature extraction.

The clean checkout's regression suite passed **37 tests**. A second physical Mac, different macOS release and optional fresh Qwen download were not exercised. The portability regression checks that a macOS release difference is accepted while a preprocessing change is rejected.

To repeat the inference check after setup:

```sh
.venv/bin/python -m research.verify_shared_install
.venv/bin/python -m pytest -q
```
