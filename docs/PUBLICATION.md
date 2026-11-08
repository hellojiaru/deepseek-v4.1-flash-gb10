# Publication checks — 2026-11-09

This independent repository publishes a reviewed source snapshot of the C4 baseline, its paired frontend, build tools, documentation and optional Creative v1 asset. The default launcher serves the official original model without steering. The unrelated Altar/GLM configuration, private fleet, credentials, operational recovery programs and private DSH conversations are excluded.

## Fresh standalone build

The staged source was archived with Git, hash-checked after transfer and extracted into a new directory on a Linux/aarch64 GB10. The build output directory did not exist. The job used two CPU cores, a 6 GiB memory limit and a 2,400-second upper bound; CUDA devices were hidden. Existing downloaded Rust dependency source caches were reused, not prior object libraries or executables.

The complete build and controls passed in about 146 seconds:

- Native CUDA engine, native24/steering/vision probe executables and the release Rust frontend compiled successfully.
- DSW conversion, verification/loading and Engram packing tools compiled successfully.
- Five host tests and all 50 frontend tests passed, including normally ignored transport/tokenizer tests using the actual official1017 tokenizer.
- Seven launcher/asset tests passed. These verify the original default, explicit opt-in and byte-for-byte readable JSON → bundled binary reproduction.
- The CPU speculative-sampling control completed 3,359,678 checks with no failures.

Toolchain: CUDA 13.0.88, GCC 13.3.0, CMake 3.28.3 and Rust 1.99.1. Existing unused-variable and Rust deprecation warnings remain visible. The full source/input hashes, artifact hashes and sanitized receipt are in [PREPUBLICATION_20261109.json](integration/PREPUBLICATION_20261109.json).

The serving ranks, paired frontend, vision worker and thermal processes retained their PID, creation identity, arguments and executable hashes. No full model was loaded by this build, no serving process was restarted, and no new GPU performance result is implied. Earlier actual model tests are separately dated in [BENCHMARKS.md](BENCHMARKS.md). Build paths/toolchains may affect executable hashes; bit-reproducible whole binaries are not claimed.

## Privacy and licensing

Gitleaks 8.30.1 scanned the publication tree. Its three default-rule findings were manually verified as two SHA256 digests and a Python model-token-ID variable, not credentials. The checked-in configuration narrowly describes these exact false positives without excluding any whole file; scanning with it reports zero findings. Publication also checks the reachable public Git history, commit email identity, private paths/addresses, ignored files and relative documentation links. Scanning alone is not a proof of every security property.

Example network addresses use the reserved 198.18.0.0/15 benchmark range. Immutable golden/tokenizer fixtures inherited from the already public upstream retain their reference bytes and attribution; they are not private user histories. Real agent benchmark reports contain aggregate/sanitized results, not the underlying private conversations.

The independent public history starts with the reviewed snapshot; previous private integration branches/tags stay local. Upstream authorship and fixed source references are retained in [NOTICE](../NOTICE), [source provenance](upstream/ROOT_README.md) and [BASELINE.json](integration/BASELINE.json). Full MIT, Apache-2.0, NumPy/PCG, Unicode and Python license texts are included where applicable. The recorded external license copies' hashes were checked. Model weights and the tokenizer remain separately obtained by users under their own license.

The optional [Creative v1 asset](../assets/steering/creative-v1/README.md) includes its MIT license, manifest and readable values. Private calibration prompts/captures are not distributed. Its historical extraction revision is incomplete; the asset card distinguishes reproducible packing from unavailable complete refitting. It is never enabled by the default original-model entry point.

## Ongoing public checks

GitHub Actions runs the launcher/asset controls, Rust frontend tests, CPU engine helpers and synthetic CPU vision tests without a model-serving endpoint. These are CPU regressions, not four-node GPU acceptance. See the [Actions page](https://github.com/hellojiaru/deepseek-v4.1-flash-gb10/actions) for the exact public commit's current result. A release should be tagged only after those workflows pass.
