# Sharing and fresh-clone verification

The repository includes the fitted classifier so a friend can run inference immediately after setup. It excludes the original database, personal screenshots, raw research downloads, generated exports, private history and runtime caches.

Targets: Apple silicon macOS with Apple Command Line Tools and uv; 64-bit Intel/AMD Windows with uv. Windows uses RapidOCR and a separate fitted artifact. The original Mac verification below predates the Windows support. Native Windows verification is distinguished from portable-backend checks.

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

## Windows portability update · November 27, 2026

Windows now selects a pinned RapidOCR backend with bundled, checksum-verified OCR models and its own fitted eight-category artifact. The Mac artifact stays separate. The portable model was trained on the existing public study's frozen partitions, with Qwen and Python networking blocked; partitions were asserted unchanged. The complete x64 Windows dependency set resolved with binary wheels available.

On the prepared Mac, the RapidOCR backend ran the real OCR and CLIP pipeline against an empty data/cache directory. All ten valid demo files processed, the corrupt JPEG and unsupported text file produced their expected outcomes, and predictions matched after reloading. The training library contained zero records before and after. Qwen and Python socket/DNS access were blocked. The portable source-only test matrices replayed identically. The regression suite passed 43 tests with Apple Vision and 43 with RapidOCR. The optional PowerShell parser check on macOS stalled and was stopped without a result; it is not counted as a successful check.

Windows setup, launch and optional annotation scripts are included. These checks exercise the portable Python implementation on macOS, not a native Windows installation. Native PowerShell execution, Windows DLL loading and optional Windows Ollama setup remain to be checked on a Windows machine. Installation commands are in [WINDOWS.md](WINDOWS.md).

A fresh local Git clone of the portable update was also tested on the Mac. It started with no original database, feature cache, Mac OCR helper or model downloads. A new virtual environment was populated from `requirements-windows.lock` using host-compatible wheels; `classifier.setup_runtime` downloaded and verified CLIP and the matched RapidOCR artifact. The cold-cache offline demo passed with identical results after reload, and all 43 tests passed. This checks checkout completeness and the portable Python pipeline; Windows binary loading and PowerShell execution remain unverified.
