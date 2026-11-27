# Character and OCR-layout feature study

Backend: **vision**. Seed 513. Character features use train-fitted character 3–5-gram TF-IDF (up to 1,500 columns); layout adds 22 train-standardized spatial measurements from automatic OCR boxes. Baseline is word TF-IDF + 15 numeric features + frozen CLIP. No Qwen or network calls were allowed during extraction, training or evaluation.

## Planned comparison

All four feature conditions use identical partitions and fixed classifier settings. When present, Public and Mixed models are selected on the same 189-row public validation set; Synthetic models use synthetic validation. Thresholds use the tenth percentile of correct validation margins. Test measurements do not select models. The prepared public source test includes 155 rows across six supported categories; 25 machine-labeled rows are measured separately in results.json. The source-test column below averages over supported classes only. Synthetic test covers all eight classes. A synthetic-only run does not evaluate public data.

| Training corpus | Features | Algorithm | Columns | Validation macro F1 | Public source test supported F1 | Synthetic test macro F1 |
|---|---|---|---:|---:|---:|---:|
| Public | Baseline | Logistic regression | 3527 | 0.762 | 0.879 | — |
| Public | Baseline | Linear SVM | 3527 | 0.743 | 0.885 | — |
| Public | Characters | Logistic regression | 5027 | 0.819 | 0.890 | — |
| Public | Characters | Linear SVM | 5027 | 0.782 | 0.926 | — |
| Public | Layout | Logistic regression | 3549 | 0.746 | 0.826 | — |
| Public | Layout | Linear SVM | 3549 | 0.801 | 0.884 | — |
| Public | Characters + layout | Logistic regression | 5049 | 0.765 | 0.835 | — |
| Public | Characters + layout | Linear SVM | 5049 | 0.814 | 0.903 | — |

## Selected models

- **Public:** Logistic regression / Characters; validation macro F1 0.819. Local artifact: `models/experiments/feature-study/vision/public/classifier.joblib`. Reload predictions identical.

## Limits

- Synthetic labels describe generator intent, not independent human review.
- Held-out synthetic configurations share drawing code and phrase banks.
- Public validation includes Qwen pseudo-labels; source-only and teacher test rows are separate.
- Existing public tests have been inspected previously; all results are exploratory.

Complete matrices, per-class results, teacher agreement and timings are in [results.json](results.json). Production content rules are excluded from these classifier metrics.
