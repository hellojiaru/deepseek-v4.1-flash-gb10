# DeepSeek GB10 Runtime

**A fast local DeepSeek-V4.1-Flash server for long-running agents on four GB10 machines.**

[中文说明](README.zh-CN.md) · [Setup](docs/BUILD_AND_RUN.md) · [Benchmarks](docs/BENCHMARKS.md) · [Compared with other engines](docs/COMPARISON.md)

Run the official 1017 checkpoint at its native mixed precision, with an OpenAI-compatible API, image input, tool calls and reusable conversation history. The default setup serves the original model without activation steering or extra weight quantization.

| Everyday agent generation | Reading a long, uncached input | Eight independent contexts |
|---|---|---|
| **About 110 tokens/s** typical long-response speed | **About 9,600 tokens/s** at 262K input | **8 × 1M tokens**, filled and generated from |

Agent speed describes longer responses in real tasks. Input-reading speed includes frontend overhead. The measurements, complete averages and conditions are explained below.

## What this is good at

- **Working through large projects.** A 262,145-token source-code input reached its first answer in 27.28 seconds with no prefix-cache hit. Subsequent tool turns can reuse the existing history.
- **Responsive single-user agents.** Real workloads include multi-step programming, visual applications, file work and engine research. Short controlled code generation reaches about 142 tokens/s.
- **Several agents sharing one model.** Start with four active requests, or choose eight. C8 measured 212 tokens/s aggregate on code and 181 on prose, while retaining approximately the same single-request speed.
- **Keeping the whole workflow local.** Streaming, reasoning output, function calls, image input and prompt-cache accounting are available through one endpoint. An external agent client handles the actual tools; the model server does not require DSH.

This is a specialized Linux/CUDA engine for **4 × GB10, 128 GB unified memory per machine**, connected by the supported ConnectX-7 RDMA ring. It is not a general-purpose replacement for every vLLM or SGLang deployment.

## Performance

Measured with DeepSeek-V4.1-Flash-1017 on four GB10 nodes. C1 means one request; C4 and C8 mean four and eight active requests. Concurrent numbers below are the total output of the group, not each user's speed.

| Workload | Code | Prose |
|---|---:|---:|
| One request, default C4 configuration | **142.2 tokens/s** | **94.1 tokens/s** |
| Two concurrent requests, total | **148.8 tokens/s** | **127.6 tokens/s** |
| Four concurrent requests, total | **173.4 tokens/s** | **152.6 tokens/s** |
| Eight concurrent requests, total | **212.2 tokens/s** | **180.8 tokens/s** |
| One request, C8 configuration | 142.3 tokens/s | 93.9 tokens/s |

These controlled tests use temperature 0, reasoning disabled, two repetitions (three for C2), 2,048/1,024 output tokens for single-request code/prose and 512 per concurrent request. Different prompts and speculative-decoding acceptance change the result. The C8 setting improves group throughput; it does not make each of eight users as fast as a user alone.

| Cold source-code input | First content | Effective input rate |
|---|---:|---:|
| 65,537 tokens | 7.06 s | 9,281 tokens/s |
| 262,145 tokens | 27.28 s | 9,611 tokens/s |

Each length was measured three times, with zero cached input tokens. Rates include frontend processing, tokenization and communication, rather than just a GPU kernel. See [measurement details and raw summaries](docs/BENCHMARKS.md).

### Real agent work

An audit of ordinary single-task sessions from October 18–31 covers **eight tasks, 1,589 timed model responses and 2.55 million output tokens**. Among 1,056 responses of at least 256 tokens, the median was **108.7 tokens/s**, and 232 reached 130 tokens/s or more. Across all timed responses, total output divided by generation time was **103.4 tokens/s**; whole-task averages ranged from **95.9 to 122.2 tokens/s**. The audit separates these tasks from CORAL and simultaneous four-agent tests, retains slower responses, and explains the difference between generation time and total project time.

A later C4 production instance ran for **123.6 hours** before planned maintenance. Its separate workload audit covers 5,905 responses and 3.97 million output tokens, including repeated tool turns, history reuse and compaction. Process uptime includes idle time. This is useful operating history, not a five-day full-load stress test.

[Ordinary agent results](docs/benchmarks/ordinary-agents-20261109.md) · [Later service history](docs/benchmarks/real-workloads-20261108.md)

## Context capacity

| Configuration | Active requests | Context per request | Combined active context |
|---|---:|---:|---:|
| Default | 4 | 1,048,576 tokens | 4,194,304 tokens |
| Optional C8 | 8 | 1,048,576 tokens | **8,388,608 tokens** |

The C8 capacity run actually sent **1,048,320 input tokens plus 256 generated tokens per request**. All eight completed, and all 24 embedded position markers were recalled. The most constrained node still had 12.6 GiB of available host memory, with no new swap use, OOM kill or disk-cache spill.

**8M is the total of eight 1M windows; a single request is still limited to 1M.** Input, image expansion, tool history and generated output share that window.

Eight completely cold 1M inputs took about 14.6 minutes as a group. Prefills currently run sequentially, so the last request waited about 862 seconds for its first content. Cached agent continuations behave differently. One full-capacity run is complete; long-duration full-window endurance and comprehensive 1M answer quality remain open work.

[Full 8M capacity test](docs/benchmarks/capacity-8m-20261108.md)

## Get started with the original model

You need Linux/aarch64, CUDA 13, CMake 3.28 or newer, a C++20 compiler, libibverbs, Rust/Cargo, Python, the official checkpoint and a correctly configured four-node ring. Image support additionally needs the vision worker's PyTorch dependencies. Prepare the rank weight files and Engram data using [model setup](docs/MODEL_SETUP.md).

Build from the repository root on the target platform, then distribute the build and source entry points to the four nodes:

```sh
python3 tools/build.py --build-dir /opt/deepseek-build --jobs 2
```

Copy the [example fleet](ds41f-4xgx10-switchless/fleet/homelab.conf) to your own configuration and set the real host, RDMA and model paths. Start ranks 1, 2 and 3 on their respective nodes, then rank 0. For example, on the rank-1 node:

```sh
python3 tools/serve_original.py rank \
  --rank 1 --fleet /opt/deepseek/fleet.conf \
  --build-dir /opt/deepseek-build --runtime-dir /opt/deepseek-runtime \
  --spill-dir /opt/deepseek-spill --disk-tag my-deepseek
```

The default is four slots. Add `--profile c8-reference` to each rank for eight slots after checking available memory. A running model must be drained before starting another full instance.

On rank 0, start the vision worker and paired API frontend as described in the [full startup guide](docs/BUILD_AND_RUN.md). The frontend reads its API key from a file. `serve_original.py` rejects steering options, and the launcher removes inherited steering environment variables: users do not need our optional modes or vector assets.

Connect an OpenAI-compatible agent client, or make a basic request:

```python
from pathlib import Path
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:18893/v1",
    api_key=Path("/opt/deepseek/api-key").read_text().strip(),
)
reply = client.chat.completions.create(
    model="deepseek-v4.1-flash-dsgb10-official-1017",
    messages=[{"role": "user", "content": "Help me review this project."}],
    max_tokens=4096,
)
print(reply.choices[0].message.content)
```

Install `openai` for this example. The service itself does not depend on that SDK. Generation and reasoning options are documented in [API capabilities](docs/API.md).

## Choosing this engine

Choose it when long-context input speed, a fast personal coding agent and a small number of concurrent agents matter most, and you have the supported four-node hardware.

Other projects have stronger high-concurrency scheduling, larger shared KV pools, grammar-constrained output or a wider deployment ecosystem. We report our measurements without calling unmatched public numbers a head-to-head win. See the [current comparison and missing capabilities](docs/COMPARISON.md).

Important limits:

- A large cold prefill can delay other requests and pause ongoing generation.
- JSON and tool calls are supported, but strict JSON-schema/regex decoding and grammar-enforced tool arguments are not implemented.
- Long-image histories and full 1M task quality need broader evaluation.
- Live KV reuse does not survive a process restart. New quantization formats, W4A4 and scale compaction are not part of this baseline.

## Optional: change model behavior with activation steering

The default above uses the original checkpoint behavior. Users who want a different response style or refusal tendency can explicitly enable **activation steering**: a small set of directions adjusts internal activations while the original weight files stay unchanged.

With an explicitly supplied asset, the API offers two choices from the same resident model:

- **Normal**: the original path, with steering disabled.
- **Creative**: the supplied directions applied to activations. The name is an alias, not a higher temperature or a guarantee of better creative writing.

Requests can select either path without loading another model. Execution switches at safe mode boundaries, and cache namespaces keep the two histories separate. The repository includes an **optional experimental [Creative v1 asset](assets/steering/creative-v1/README.md)**, with both a small binary and readable numeric JSON. It is never loaded by the default launcher. Its legacy calibration provenance and limitations are documented alongside it. Users can also generate, inspect and enable their own local directions. Different assets can produce different behavior and should be evaluated on the user's tasks.

See [optional steering setup](docs/STEERING.md) for asset format, explicit launch flags, cache identity and current limits. The current build uses a fixed strength of 1.5 for the enabled path; replacing directions requires a new asset identity and restart, not a live per-request strength slider.

## Credits and license

Built on [blaesus/dsgb10](https://github.com/blaesus/dsgb10), retaining its native C++/CUDA engine, RDMA architecture and attribution. This project integrates the paired frontend, native24 work, completion handling, probabilistic sampling and vision support into a separately buildable source tree.

See [LICENSE](LICENSE), [NOTICE](NOTICE) and [third-party licenses](docs/integration/LICENSES.md). DeepInfer-derived sampling code retains Apache-2.0 terms. Model weights and optional external assets have their own licenses.

[Build and run](docs/BUILD_AND_RUN.md) · [API](docs/API.md) · [Benchmarks](docs/BENCHMARKS.md) · [Maintenance](docs/MAINTENANCE.md)
