# Upstream source provenance

This standalone DeepSeek project is derived from [blaesus/dsgb10](https://github.com/blaesus/dsgb10). Its MIT copyright and all third-party notices are retained.

- Imported public source reference: [`d793c869f40bf638b06c4c1738af0d29473f5ed5`](https://github.com/hellojiaru/dsgx10/tree/d793c869f40bf638b06c4c1738af0d29473f5ed5).
- Native24 C4 integration source: `16a67bc9ad72d1079889cb83804a2179a595e9c5`, with per-file origin and qualified hashes in [BASELINE.json](../integration/BASELINE.json).
- The original upstream design, kernel, RDMA and frontend authorship remains upstream's. This project integrates and maintains a paired build, concurrency/verification changes, completion handling, probabilistic speculation, vision and optional steering.
- The unrelated GLM/Altar configuration is not part of this release.

The independent repository starts with a reviewed source snapshot, not the private integration history. Historical upstream DeepSeek notes remain under `ds41f-4xgx10-switchless/docs`; dates and historical measurements there are not this release's current benchmark configuration. Network examples were replaced with reserved benchmark addresses. Public upstream golden fixtures retain their exact reference content/token identities; they are not users' private DSH histories.

Use the root [README](../../README.md), [build guide](../BUILD_AND_RUN.md) and [benchmarks](../BENCHMARKS.md) for the current supported setup.
