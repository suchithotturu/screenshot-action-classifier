# Implementation and verification log

Verified on November 23, 2026, on an Apple M4 Max with 36 GB of memory, macOS 15.7.9 and approximately 843 GiB available at setup. This project started in an empty workspace. The user supplied no phone-image folder.

## Real model and batch

Installed Python 3.12.16 and the packages in requirements.lock into the project. Ollama 0.40.17 served qwen3-vl:4b-instruct, Q4_K_M, digest 84f7ed8042fa7cdbd902f7dcb7d0370e032095047cc3244179f263b77b5bc318. The first pixel test read “Cotton camp shirt” and “$48”, classified the astronaut portrait as an ordinary photo and repeated the shirt classification. Requests took 4.88, 3.55 and 1.61 seconds. During the mixed-folder checks, Ollama reported 3,719,676,558 bytes of GPU residency and macOS memory_pressure -Q reported 76% system-wide memory free.

An early test left the shirt action empty and categorized the monthly gym membership as shopping. The final policy derives the action from its category and literal supporting text. The revised prompt explicitly distinguishes a gym signup from fitness equipment. These development fixtures influenced the prompt and cannot estimate generalization.

The final uncached run (3) enumerated 12 files. Nine unique images produced validated records in 3.22–3.74 seconds each, including preprocessing. The model identified seven screenshots, one uncertain blank crop and one ordinary portrait. Five cards offered actions. The gym offer and pottery workshop were registration suggestions. The receipt was a reference, the meme was OTHER and the crop required review. December 12, 2026 was retained as the workshop event date with no due date. One exact duplicate retained its path, one text file was skipped and one deliberately corrupt JPEG failed without aborting the batch. No model-call failure was reclassified as OTHER.

## Persistence, reports and interface

Launched Launch.command from a minimal shell environment, verified model readiness and opened the local page. Browser checks exercised Analyze, progress, cancel, resume, the reference filter, details and saving the receipt as reviewed. Cancellation completed one in-flight image. Resuming the same run preserved it. Reloading reopened the saved analysis. The final demo receipt review appears separately from the model prediction in all three sample exports.

Run 4 re-read the same folder and reused all nine valid results, preserving the duplicate, skip and corrupt-file error. JSON and CSV were read back and checked for 12 records. The HTML parser opened the report, which contains embedded thumbnails. The project root was absent from the shareable files. The screenshot in samples/ shows the running inbox.

A further uncached image request and the full cached run passed with a Python audit hook restricting TCP connections to 127.0.0.1:11434. The service ran with cloud features disabled. This test covered the Python client. The machine remained online, and traffic from the Ollama and browser processes was not independently firewalled.

## Isolated tests and limits

The pytest suite uses a temporary database and injected model responses for deterministic failure, correction and retry checks. It covers schema rejection, bounded retry, event versus deadline, ambiguous years, exclusion of reference/photo actions, unsupported URLs, symlink skipping, corrupt images, cancellation, duplicate and cache recovery, prompt invalidation, correction precedence, completed and dismissed states after reopening, embedded thumbnails, HTML escaping and spreadsheet formula protection.

Small text, partial dates and ambiguous images still require human review. Model-transcribed text can contain OCR errors even on legible screens. The native chooser and HEIC decoding were implemented, while the live demo exercised PNG, JPEG failure and WebP. Main model passes use a single bounded resolution. No private photos were opened. Independent human-label evaluation and baseline training are pending.
