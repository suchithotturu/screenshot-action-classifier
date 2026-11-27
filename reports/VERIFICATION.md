# Character and layout feature update

The active Mac and portable heads are now `3cae9ac061854287` and `642c50cc3e8cbd86`, respectively. Both use 5,027 features with character TF-IDF additions. Spatial OCR features are implemented and measured but did not win selection. Public-only validation improved from 0.762 to 0.819 (Vision) and 0.787 to 0.821 (RapidOCR). The unchanged source-only holdout accuracies were 141/155 and 143/155; there are only six independently source-supported categories. Saved matrices replay identically. Both backends pass 47 tests.

The separate 1,600-screen synthetic study uses 960/320/320 partitions and generator-intent labels. The chosen synthetic-only baseline scored 318/320 on synthetic test but 45/155 on the public source test. Adding synthetic training did not win public validation. Synthetic and mixed models therefore remain separate experiments. [Measured results and limits](feature-study/REPORT.md)

The entries below are historical acceptance records.

---

# Eight-category deployment verification

The saved model `310009966ab1d1e6` contains all eight content classes. The enlarged study uses 1,280 eligible images: 911 training, 189 validation and 180 public holdout records, with zero combined-group overlaps. The library additionally preserves 9 excluded fixtures and 219 excluded, unannotated acquisition candidates. All 1,508 records validate against the current schema; the portable export was regenerated. Public images alone entered training.

Local Qwen annotated the additional public UI images. Content training used 762 source labels and 149 explicit machine pseudo-labels. Validation used 152 source labels and 37 pseudo-labels. Logistic regression with OCR/CLIP fusion won at validation macro F1 0.762. SVM, k-NN, Complement Naive Bayes and a dummy baseline are measured in the [report](REPORT.md). Source-only testing achieved 139/155 correct across six supported categories; the separate 25-image machine-label test achieved 17/25 teacher agreement. There are no independently human-verified shopping or entertainment test labels, and these exploratory results do not establish eight-category accuracy.

Qwen was stopped before training and classification. Training blocked its client and Python socket/DNS access. The ten-image cold-cache demo completed successfully in 6.29 seconds, with identical results after reload and zero training-image overlap. The personal 63-image folder was rerun using the new model with zero training-image overlap: 37 supervised decisions, 5 OCR-assisted decisions and 21 miscellaneous fallbacks. All eight routes occur in that folder. These counts measure coverage, without labeled accuracy claims. Stored source-only test matrices replay identically. All **36 regression tests passed**.

Evidence: [eight-category artifact check](eight-category-check.json), [offline check](offline-check.json), [personal-folder routing check](routing-check.json), [measured results](results.json). The classifier app runs on loopback port 8514. The entries below are historical verification records.

---

# Content-first routing correction

The classifier now evaluates image content regardless of screenshot-detector confidence. Valid images with unclear content receive Miscellaneous. Actual decode or feature failures retain Processing error. Strong automatic OCR patterns supplement the four-class trained head for message composers, inboxes, sales, confirmations, maps and menus. Exported predictions distinguish supervised-model decisions, OCR rules and fallback assignments. The personal folder remains outside training.

The 63-image regression previously contained 54 Needs review results. It now contains 11 conversations, 5 product offers, 9 places, 18 reading items, 4 app screens, 2 confirmations and 14 Miscellaneous. These are routing counts without independently verified accuracy labels. Source images and prior runs remain available. [Aggregate evidence](routing-check.json)

All **34 tests passed**, including screenshot-detector independence, OCR cue precedence, ambiguous-content fallback, and saved-run identity during filtering. Five sales cards were verified in the Streamlit view with inference disabled. A history selector now includes a unique run identifier to prevent collisions between runs created in the same minute.

The fresh-feature-cache demonstration passed with Qwen and Python network connections blocked and identical decisions after model reload. The stored supervised-head test matrices also replay unchanged. Earlier head accuracy figures describe those heads separately from the new OCR rules.

---

# Current prepared dataset and model verification

The working annotation database was reset and rebuilt at the user's request. It contains **966 automatically labeled images**. The prior database, reviewed labels, fitted model and research reports are preserved in the backup directory recorded in [rebuild.json](rebuild.json). The original legacy inbox database remains available separately.

The public experiment uses 957 images. Nine demonstration fixtures have example labels and remain excluded from research training and metrics. Public source labels supply targets, with seven saved Qwen content corrections marked pseudo and restricted to training. Validation and test targets keep source provenance. Owner purpose and usefulness remain unknown. New Qwen calls for the rebuild and classification checks were zero.

The updated experiment compares logistic regression, linear SVM, k-NN and Complement Naive Bayes. The **14-test regression suite passed**. Reloading the saved artifact reproduced the public test matrices. The content head correctly classified **80 of 87** public test screenshots, with macro F1 **0.9267**. This is an exploratory comparison using previously inspected source labels.

The strict seven-image test completed from an **empty feature cache** in 3.74 seconds on the prepared Mac. The reload with a warm cache took approximately 0.005 seconds. A Python audit hook blocked socket connections and DNS, and the Qwen client was replaced with a failing stub. All seven images processed successfully with zero training-image hash overlap and identical decisions after reload. This resolves the previously unverified cold-cache run below. Evidence is in [offline-check.json](offline-check.json).

The UI and validated 966-row export were checked with Qwen calls disabled. Dataset Annotation displays the auto-labeled state and makes manual review optional. Classify Images uses the newly saved model. The existing source-style and flower-negative confounding limitations still apply.

---

# Earlier acceptance record · November 26, 2026

The prepared deliverable runs locally on an Apple silicon Mac. The original three tests passed before implementation. The final regression suite passed all **11 tests**. Live checks used the real Vision framework, ONNX encoder and bundled Ollama/Qwen model. The review test edited an isolated fixture copy. The public study retains its source-provided labels.

| Specification check | Result and evidence |
| --- | --- |
| Prototype baseline and preservation | Passed. Original Streamlit served a healthy loopback response. Legacy inbox opened in Streamlit AppTest. SQL set differences against the backup were empty in both directions for 5 runs, 59 items, 38 cached predictions and 3 corrections. |
| Import, corrupt files and duplicate grouping | Passed. The demo yielded 9 canonical images, one exact duplicate, one unsupported text file and one corrupt JPEG. Repeat import reused records. A JPEG-reencoded test image shared its PNG original’s group and partition. The public study combined two exact MASC captures with separate Rico IDs. |
| Real Qwen and invalid-output fallback | Passed. Qwen3-VL 4B Instruct recorded 10 real MASC proposals and one shirt-fixture proposal. Its shirt suggestion identified a product and quoted the visible price. Invalid structured responses were injected to exercise bounded retries, failure recording and subsequent recovery. |
| Drafts, approval, undo, skip and restart | Passed. Streamlit AppTest edited a prefilled content label, reopened the editor, then approved the displayed batch. A separate live-proposal test changed shirt capture status from True to Unknown. The new SQLite connection and JSONL retained the correction and the original proposal. Transaction rollback rejected a stale revision without committing other batch edits. |
| Reviewer roles and blind mode | Passed in automated tests. A mismatched owner could not confirm purpose. A blind draft returned empty labels and blocked assisted disclosure. Photos cleared screenshot-only targets. True, False and Unknown were exercised. Independent blind *research* review awaits human participants. |
| Validated portable snapshot | Passed. All 966 JSONL rows validated. A complete restore into an empty temporary library preserved every ID, label object and saved proposal. The restored library had 957 split rows, matching history/proposal/frozen-target counts and SQLite integrity `ok`. |
| Real, cached and empty features | Passed. OCR read the shirt fixture and price; a uniformly blank image returned valid empty text. A 3,100-pixel-tall image retained both separated occurrences of the test phrase. Fresh and repeated normalized 512-value CLIP vectors had maximum absolute difference 0. |
| Split and feature leakage audit | Passed for tracked groups. Zero combined-group overlaps. Training-only vocabulary and feature-name tests held with a validation-only token and forbidden metadata. Grouping uses source metadata solely to prevent leakage. Residual undetected crops and source confounding are documented. |
| Fitted heads and comparative evaluation | Passed. Two substantive algorithms, three feature conditions and a dummy baseline were run. Repeated training retained training ID `90d3a9484ecaf6ea`. Loading the artifact reproduced both test confusion matrices and accepted counts. Purpose and personal testing had no eligible owner data. |
| Offline unseen-folder prediction | Passed. Seven demonstration images had zero training-hash overlap. The reloaded artifact returned identical labels, margins, routes and actions. The Python audit hook blocked socket connections and DNS, and the Qwen API was replaced with a failing stub. The first demonstration extracted missing features for the crop and portrait. The final replay used the existing cache. |
| Abstentions, missing heads and safe actions | Passed. The mixed run displayed reading, places, a photo and four review cases. The blank crop and portrait entered review. Regression tests covered missing model files, missing detectors and low margins. The UI displayed unavailable purpose, OCR evidence, score cutoffs and editable suggestions. |
| UI, academic materials and run commands | Passed for the prepared Mac. The in-app browser displayed the live classifier and saved mixed-folder results. AppTest exercised annotation and legacy views. CLI ingest, features via the shared extractor, training, evaluation, export, restore and prediction were exercised. The HTML presentation embeds its two figures for offline viewing. The original paper/model/dataset references and complete metrics are in the report. |

## Separate evidence and scope

- [Live feature and annotation checks](acceptance.json)
- [Offline reload and routing check](offline-check.json)
- [All measured comparisons](results.json)
- [Review and agreement counts](review-summary.json)
- [Sanitized public records](public-samples.jsonl)

The app ships with 966 records pending explicit human approval. Source-labeled public training examples remain eligible without an additional human approval. The seeded public batch contains real machine proposals. No one was impersonated as a human reviewer for the research evaluation.

A supplementary attempt to repeat the entire seven-image test with a brand-new feature cache was stopped before execution when automatic approval review reached its usage limit. That specific run is unverified. The earlier fresh extraction checks and seven-image offline replay above completed. The optional stricter command is `.venv/bin/python -m research.demo_check --cold` in a normal local terminal. The Python network guard covers the application process. There was no machine-wide network capture.

The prepared installation and current archive checksums were tested. A second clean Mac installation and human usability study were outside the observed checks.
