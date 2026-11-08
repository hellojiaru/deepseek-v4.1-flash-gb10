# Public-source preparation validation

Validated on 2026-10-21, without inference, GPU execution, remote operations,
or changes to a running service.

- The adopted upstream revision is
  `a92b8f44046f4435c3c29c07fbc39e6e4b157080`. Its engine subtree has no changes
  relative to `15a344b918b49eb28f7ba86c5c5738d802ccba60`; the intervening frontend
  work is therefore not replaced by this engine migration.
- Runtime implementation content matches the previously deployed engine source.
  Public preparation adds provenance comments in `common/spec_sampling.h`,
  `model/sample.cu`, and `dspark/dspark.cu`, and normalizes source line endings
  to LF. These are provenance/line-ending changes, not algorithm changes;
  this is not a claim of byte-identical source or binaries. Build/test integration
  is separate from that comparison.
- Private deployment manifests, serialized requests, user images, credentials,
  binaries, object files, and maintenance scripts were not imported.
- The standalone host project configured and compiled with MSVC 19.44,
  C++20, and installed CUDA 12.4 headers. It does not link or initialize the CUDA
  runtime. All four registered tests passed:
  `vision_prefix_test`, `vision_features_test`, `probability_host_test`, and
  `distribution_oracle` (CPython 3.12).
- The independent Python oracle passed seven test methods, including all
  3,969 pairs of the small rational distributions. Regenerating both checked-in
  synthetic fixture files reproduced their contents.
- A separate Ubuntu/WSL host check compiled and ran `runtime_metrics_test`,
  `vision_prefix_test`, and `vision_features_test` with g++/C++20. All passed;
  the metrics test included 100,000 updates, atomic publication, and private
  file permissions. This used no CUDA runtime or GPU.
- After public line-ending normalization, the Python distribution oracle again
  passed all seven methods and the synthetic vision CPU suite passed all eleven
  tests. These image tests exercise preprocessing and bounded cache contracts
  with a fake CPU model, not the full vision encoder's semantic correctness.
- Rust validation is handled by the public CPU workflow; no local Rust result is
  claimed in this record before that build/test run completes.
- `probabilistic_sample_probe.cu` retains SHA-256
  `d5de5f8c6cc9267be7ed34e48c4b388eaba21c74fcd02057373c7690b5a01298`,
  matching the separately exercised four-rank probe source. It was neither
  rebuilt nor run during this public-source preparation.

This is an offline portability and source-migration check. It does not replace
a fresh full CUDA 13 / SM121a build, Linux file-helper execution, four-rank GPU
execution, or end-to-end model and task-quality validation. Previously measured
deployment performance must not be relabeled as a benchmark of a newly built
public checkout.
