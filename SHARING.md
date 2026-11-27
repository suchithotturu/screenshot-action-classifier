# Sharing and fresh-clone verification

The repository includes the fitted classifier so a friend can run inference immediately after setup. It excludes the original database, personal screenshots, raw research downloads, generated exports, private history and runtime caches.

Supported target: Apple silicon macOS with Apple Command Line Tools and uv. Windows and Linux require replacing Apple Vision OCR, rebuilding the feature contract and retraining; they are not supported by these instructions.

The prepared Mac previously passed an empty-cache ten-image check with Qwen stopped and Python networking blocked, no training-image overlap, and identical predictions after reload. This does not substitute for a fresh checkout check. The publication check below will record the actual checkout and setup outcome.
