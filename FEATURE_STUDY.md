# Character and layout features: reproducible study

The classifier now supports word/numeric/CLIP fusion plus optional character TF-IDF, OCR spatial features, or both. These modules use no new large model and no Qwen calls. Character TF-IDF uses 3–5-character sequences within word boundaries, up to 1,500 train-fitted columns. Layout uses 22 train-standardized measurements from automatic OCR line boxes. Existing model artifacts retain their original feature shapes.

## Synthetic screenshots

`research.synthetic_screens` creates **1,600 fictional phone screens**, 200 for each of the eight categories. Screens include chat threads/inboxes, social feeds, map/place cards, shopping listings, articles, receipts/confirmations, settings and miscellaneous wallpaper/illustration captures. Colors, dark/light themes, text, amounts, dimensions and configuration vary. All content is code-rendered; there are no personal screenshots, downloaded media or image-generation/LLM calls.

The split is fixed before fitting: **960 training, 320 validation, 320 test** images. Category/configuration groups and exact hashes do not cross splits. However, configurations share rendering code and phrase banks. These screens are useful for exercising feature extraction and controlled comparisons. They do not establish accuracy on real phone screenshots. Labels describe the generator's intended category, rather than independent human review.

[Balanced preview](reports/feature-study/synthetic-preview.jpg)

## Run from a fresh clone: synthetic data only

Complete the normal app setup first. On Windows:

```powershell
.\.venv\Scripts\python.exe -m research.synthetic_screens --per-category 200
.\.venv\Scripts\python.exe -m research.feature_study --synthetic-only
```

On Mac, install the optional portable OCR package and select it for this experiment:

```sh
uv pip install --python .venv/bin/python -r requirements-portable.in
.venv/bin/python -m research.synthetic_screens --per-category 200
CLASSIFIER_OCR=rapidocr .venv/bin/python -m research.feature_study --synthetic-only
```

This uses the same RapidOCR pipeline as Windows. OCR models are bundled in the pinned package; the shared CLIP encoder comes from normal setup. Feature extraction runs locally on CPU and is cached. Allow several minutes for the initial 1,600-image run. Later runs can reuse the cache. The original training database is not required for `--synthetic-only`.

## Prepared public/mixed comparison

On the prepared installation, `research.feature_study` additionally compares training on the public corpus, synthetic corpus, and public-plus-synthetic corpus. Public and Mixed candidates use the same public validation rows for selection. A separate source-only test, machine-label agreement check and synthetic test are reported. The public experiment remains exploratory and includes pseudo-labels in validation.

```sh
CLASSIFIER_OCR=rapidocr .venv/bin/python -m research.feature_study
# Existing cached Vision features can support a public-only Mac comparison:
.venv/bin/python -m research.feature_study --public-only
```

Both logistic regression and linear SVM are compared across Baseline, Characters, Layout and Characters + layout, with fixed settings and identical partitions within each study. The baseline includes OCR word TF-IDF, 15 engineered measurements and CLIP. Test results do not choose the model. Public data must first be rebuilt to run those two commands on a new clone.

Results: [portable study](reports/feature-study/rapidocr/REPORT.md), [Mac public study](reports/feature-study/vision/REPORT.md). Complete matrices, per-class metrics and fit times are in their adjacent `results.json` files.

## Use an experiment in classification

Locally fitted study artifacts appear in the app's **Model** selector inside **Choose an image folder**. The prepared classifier remains the default. You can also use a matching artifact explicitly:

```powershell
.\.venv\Scripts\python.exe -m classifier predict "C:\path\to\screenshots" --model models/experiments/feature-study/rapidocr/synthetic/classifier.joblib
```

On Mac, set `CLASSIFIER_OCR=rapidocr` when using a RapidOCR study artifact. An artifact from the wrong OCR pipeline is rejected. Synthetic-only and mixed models remain experiments. `research.publish_feature_study` can promote only the public-data winner when public validation improves; it backs up the prepared model and reports, checks saved test replay, and never selects a model using test accuracy.

Synthetic images and experiment artifacts stay in ignored local directories. The generator, study code, aggregate measurements and preview are shared. A new clone can regenerate its own corpus and study artifacts.
