# Run on Windows

Target: Windows 10/11, 64-bit Intel/AMD (x64). No GPU or Qwen installation is needed for classification. Windows on ARM is not covered by the pinned binary dependencies.

## First run

Install [Git for Windows](https://git-scm.com/download/win) and [uv](https://docs.astral.sh/uv/getting-started/installation/). Reopen PowerShell after installing them. You do not need to install Python separately; setup provisions an isolated Python 3.12 environment.

```powershell
git clone https://github.com/suchithotturu/screenshot-action-classifier.git
cd screenshot-action-classifier
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup.ps1
.\.venv\Scripts\python.exe -m classifier app --mode classify
```

Open **http://127.0.0.1:8514**. After the initial setup, double-click **Launch-Windows.cmd** to start the app. Expand **Choose an image folder**, enter an absolute path such as `C:\Users\YourName\Pictures\Screenshots` and click **Classify folder**. The `demo` folder is included for a quick workflow check.

Setup downloads the pinned Python dependencies and CLIP image encoder. RapidOCR's small OCR models are packaged with its Python wheel and checked by SHA-256. Both OCR and CLIP run locally on CPU. A separately fitted eight-category RapidOCR classifier is included under `models/rapidocr/`; the Mac/Apple Vision classifier remains separate. Windows automatically selects RapidOCR and its matching trained model. The app does not send screenshots to a cloud service.

The Windows model uses the same 15 engineered measurements, train-only OCR TF-IDF and frozen CLIP image embeddings as the Mac experiment, with its own fitted vocabulary, weights, confidence cutoff and measured model selection. Changing OCR engines requires the matching artifact; loading an incompatible feature contract is rejected. See [the portable results](reports/rapidocr/results.json).

## Common setup issues

- **`uv` or `git` not recognized:** finish installing it, then reopen PowerShell.
- **Missing ONNX Runtime DLL:** install Microsoft's [current Visual C++ x64 Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist), then rerun setup. ONNX Runtime documents the Windows runtime prerequisite in its [installation guide](https://onnxruntime.ai/docs/install/).
- **PowerShell scripts disabled:** use the exact `powershell -NoProfile -ExecutionPolicy Bypass -File .\setup.ps1` command above. It changes policy only for that process.
- **Port already occupied:** use `.\.venv\Scripts\python.exe -m classifier app --port 8515`, then open `http://127.0.0.1:8515`.
- **Repository already cloned:** stop the app, run `git pull`, rerun setup, then launch again.

## Optional annotation with local Qwen

For classification alone, skip this section. To generate annotation suggestions, first install [Ollama for Windows](https://ollama.com/download/windows), then:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup-annotation.ps1
.\.venv\Scripts\python.exe -m classifier app --mode annotation
```

This downloads Qwen3-VL 4B Instruct locally and needs additional disk space and memory. Annotation may be slower on CPU. The helper starts or reuses the local loopback Ollama service. The model tag is local, and the application calls only `127.0.0.1:11434` for annotations. Neither training nor classification calls Qwen.

## Train on an annotated Windows library

Import and label data in Dataset Annotation, then:

```powershell
.\.venv\Scripts\python.exe -m classifier features
.\.venv\Scripts\python.exe -m classifier train
.\.venv\Scripts\python.exe -m classifier evaluate
.\.venv\Scripts\python.exe -m classifier export
```

These commands select the RapidOCR feature pipeline and artifact directory automatically. The checkout does not include the original training database. The public acquisition scripts use Bash; rebuilding that corpus is separate from running the supplied classifier. The prepared public partitions can also be used with `research.train_portable` after the corpus has been rebuilt.

## Verify an installation

```powershell
.\.venv\Scripts\python.exe -m research.verify_shared_install
.\.venv\Scripts\python.exe -m pytest -q
```

The first command blocks Python network connections and the Qwen client while classifying the demo with real OCR and CLIP. Ten valid files should process successfully, the deliberately corrupt JPEG should produce an error and the text file should be skipped. This is a workflow check, not a labeled accuracy benchmark. Tests that create symlinks may require Windows Developer Mode or elevated privileges.

The portable OCR, matched trained model and offline inference are exercised on the prepared Mac with `CLASSIFIER_OCR=rapidocr`. Native Windows installation and PowerShell execution require verification on a Windows machine; they are not claimed as completed here. See [SHARING.md](SHARING.md) for the observed checks.
