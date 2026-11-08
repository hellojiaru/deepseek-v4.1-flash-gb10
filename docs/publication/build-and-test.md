# Build and test the additions

Commands below start at the repository root unless stated otherwise. They do
not download model weights, configure a fleet, start a service, or reserve GPUs.
Read the existing [configuration README](../../ds41f-4xgx10-switchless/README.md)
for the underlying weight conversion and native ring requirements.

## Host-only checks

```sh
cd ds41f-4xgx10-switchless
python3 engine/tests/distribution_oracle.py
cmake -S engine/tests -B build-host -DCMAKE_BUILD_TYPE=Release
cmake --build build-host --config Release -j2
ctest --test-dir build-host -C Release --output-on-failure
(cd frontend && cargo test --release --locked -j2)
python3 -m unittest discover -s vision/tests -p 'test_*.py' -v
```

The probability oracle needs Python's standard library. Vision CPU tests need
the packages in `vision/requirements.txt`; install a suitable PyTorch build in
an isolated environment. These tests use generated images, not model weights or
private conversations. Test selection and optional CUDA-header/file tests are
documented in [engine/tests/README.md](../../ds41f-4xgx10-switchless/engine/tests/README.md).
Some frontend tests requiring a real tokenizer or larger temporary files are
explicitly ignored by default; others return early when neither
`DSGB10_TEST_TOKENIZER` nor their default tokenizer file is available. A normal
`cargo test` result does not establish that those integration cases executed.

The Linux file-budget test uses the same restricted spool path as the runtime.
Run it only in a disposable container with a private `/tmp`, not against a
running frontend's spool. It is opt-in with `-DDSGB10_ENABLE_FILE_TESTS=ON`.

## Engine and frontend build

The existing build targets CUDA 13 and **SM121a** (`compute_121a`/`sm_121a`). A
generic Blackwell description is not sufficient compatibility evidence. The
upstream environment uses CMake 3.28+, a C++20 compiler, CUDA headers/toolchain,
libibverbs development files, and Rust/Cargo. Keep the Rust lockfile; GPU and
driver versions must support the target architecture.

```sh
cd ds41f-4xgx10-switchless
cmake -S engine -B engine/build -DCMAKE_BUILD_TYPE=Release
cmake --build engine/build -j2
(cd frontend && cargo build --release --locked -j2)
```

These commands compile the code. They do not establish four-node connectivity,
successful checkpoint loading, correct answers, or performance. Supply your own
fleet configuration and paths; do not reuse another installation's addresses,
service units, sockets, or operational scripts.

## Vision worker configuration

The checkpoint used in the development validation was
[`deepseek-ai/DeepSeek-V4.1-Flash-1017` at `1bae0e1088993b9b82de08f59de5be5a34e42170`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/tree/1bae0e1088993b9b82de08f59de5be5a34e42170).
Obtain it separately under its applicable terms. `--revision` is a caller-supplied
identity string, not an automatic download or verification of remote contents.
The worker also hashes its local source and loaded vision data for cache identity.
Public source line endings are normalized to LF; this can change the identity
relative to the historical deployment even without a code change. Do not pin or
reuse the historical encoder/cache identity in a new deployment.

After reserving resources, the worker entry point is:

```sh
python3 vision/encoder.py \
  --checkpoint /path/to/checkpoint \
  --revision 1bae0e1088993b9b82de08f59de5be5a34e42170 \
  --socket /path/to/private/vision.sock --device cuda
```

This command **loads weights and uses the GPU**. Do not run it as a build test
or beside an active workload without a resource plan. Bind the frontend using
`DSGB10_VISION_SOCKET` and `DSGB10_VISION_REVISION` with the same socket and
revision, and explicitly select `--encoding 1017 --vision-prefix-policy tail-exact`.
`v4.1-1017` is also accepted as an encoding name. The public frontend defaults
remain temperature 1 and top-p 0.95; use explicit `--temperature 1 --top-p 1` to
match the probability-path measurements. The engine's `--spec-probabilistic`
requires `--spec` and excludes unsupported debug/ratio-gate combinations.

Use filesystem permissions to keep Unix sockets and the fixed feature spool
`/tmp/dsgb10-vision-features` private. The vision worker is not an Internet-facing
service, but HTTP(S) image inputs can cause outbound fetches; it is **not a
network-isolated image decoder**. Apply your deployment's input and network
policy before exposing image URLs to untrusted clients.

## Explicit four-rank sampling probe

Follow [the probe instructions](../../ds41f-4xgx10-switchless/engine/tests/README.md)
to build `probabilistic_sample_probe` with `-DDSGB10_BUILD_GPU_PROBE=ON`. It
requires four exclusively reserved compatible ranks, their native transport,
a user-supplied fleet configuration, and a fresh bootstrap port. It needs no
model weights. Coordinate all ranks and retain an external timeout and cleanup
mechanism. It is deliberately not part of default CTest.

The probe runs the real sharded verifier with independently generated proposals,
CUDA Graph reuse, slot changes, and fallback controls. It does not run the
complete drafter/model or assess application quality. Full-model evaluation
still needs an authorized workload, cold/warm controls, model/encoding pins,
and separately measured prefill and decode.

## Public HTTP smoke fixture

The standard-library script generates a green PNG and synthetic text locally.
Its default mode only reports request hashes and sizes; it performs no HTTP call:

```sh
cd ds41f-4xgx10-switchless
python3 tools/public_smoke.py --model your-test-model
python3 -m unittest discover -s tools -p 'test_public_smoke.py' -v
```

After reserving a test server, opt in explicitly:

```sh
python3 tools/public_smoke.py --execute \
  --base-url http://127.0.0.1:18890/v1 --model your-test-model \
  --text-blocks 32 --output smoke-result.json
```

The optional bearer key is read from `OPENAI_API_KEY` (or the environment name
selected by `--api-key-env`); it is not included in the result. The three calls
are text, an image-containing tool-result continuation, and an identical image
request. The text request and continuation share the same tools and first user
message. Thinking is explicitly disabled for this short smoke. The script checks
the text answer (`READY`), both image answers, and `finish_reason=stop`; it captures
usage and complete HTTP latency, and hashes rather than saves answer text.
Image content blocks in a `tool` role message are an extension supported by this
fork, **not a format guaranteed by every OpenAI-compatible service**. The
`chat_template_kwargs.thinking` setting is likewise server-specific. It does not measure
streaming first-token latency, native decode rate, a fixed token count, or actual
tool execution. The synthetic transcript is constructed by the fixture. This
smoke is not the private performance benchmark, the 58-image capacity test, or
a complete cache replacement/reordering matrix. Use actual usage/native counters
when building a larger authorized benchmark.

## Runtime budgets to retain

The included image path has separate bounds: a 256 MiB request body, 16 MiB
compressed bytes and 40 million pixels per decoded image, at most 1,024 visual
rows per image for the supported checkpoint, encoder batches of at most eight
images, a 512 MiB active feature buffer, and a 12 GiB feature spool. These are
different resources; none is a promise of that much free GPU or host memory.
The frontend feature cache accounts up to 1 GiB of logical file bytes. Historical
images still consume context and disk, even when their features are cached.

Existing feature generations count against the spool budget. Do not delete
files while a frontend or engine may own them. A killed process is not proof
that all readers have gone away; reclaim stale generations only after quiescing
the owning deployment and verifying ownership. No general deployment or cleanup
automation is supplied in this publication.
