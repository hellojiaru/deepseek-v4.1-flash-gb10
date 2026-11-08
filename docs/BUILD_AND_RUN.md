# Build and run

The default setup uses the official checkpoint without activation steering. The launcher starts one component at a time; it does not change your network, stop another service or reserve the RDMA ring for you.

## 1. Prepare the machines

Supported target: four Linux/aarch64 GB10 nodes, each with 128 GB unified memory, CUDA 13 and the supported dual-link ConnectX-7 ring. Install CMake >= 3.28, a C++20 compiler, libibverbs development headers, Rust/Cargo and Python3. Hardware topology and fleet configuration must match the actual links.

The language runtime uses C++/CUDA and ibverbs; the HTTP frontend uses Rust. The native vision worker uses PyTorch, NumPy, Pillow and safetensors; its requirements are in `ds41f-4xgx10-switchless/vision/requirements.txt`. Install a PyTorch build supporting your GB10/CUDA environment; the tested deployment used 2.13.0+cu130. Keep these dependencies in your own environment rather than changing an existing serving environment in place.

## 2. Build cleanly

From the repository root:

```sh
python3 tools/build.py --build-dir /opt/deepseek-build --jobs 2
```

The build directory must not already exist. Cargo dependencies are locked; first use needs access to the crate sources. Use `--offline` only if they are already cached. The build compiles the runtime and paired frontend from source, records commands and SHA values in `BUILD_RESULT.json`, and does not load weights or run GPU probes.

With the actual tokenizer available, include the full frontend tests:

```sh
DSGB10_TEST_TOKENIZER=/models/DeepSeek-V4.1-Flash-1017/tokenizer.json \
  python3 tools/build.py --build-dir /opt/deepseek-build-full \
  --jobs 2 --full-frontend-tests
```

Host-only helpers can be checked with `--host-only`. Missing-data early returns in some historical tests are not full model acceptance. New launcher/asset controls run with:

```sh
python3 -m unittest discover -s tests -p 'test_*.py' -v
```

## 3. Prepare weights and fleet

Follow [model setup](MODEL_SETUP.md) for the fixed official1017 revision, DSW conversion, byte verification and Engram packing. Distribute each rank's data, the built engine, the paired frontend on rank 0, the root `tools/` and `profiles/`, and the vision source/dependencies where needed. Keep paths consistent with each node's local filesystem.

Copy `ds41f-4xgx10-switchless/fleet/homelab.conf` to `/opt/deepseek/fleet.conf` on each node and replace its example hosts, RDMA addresses/devices and data paths. The example is a template, not an auto-discovery result.

## 4. Start the original model

Drain any existing full model using your own service manager. Confirm per-node memory/disk headroom and exclusive use of the ring. Start ranks 1, 2, 3 on their respective machines, then rank 0. Change `--rank` in this example for each node:

```sh
python3 tools/serve_original.py rank \
  --rank 1 --fleet /opt/deepseek/fleet.conf \
  --build-dir /opt/deepseek-build --runtime-dir /opt/deepseek-runtime \
  --spill-dir /opt/deepseek-spill --disk-tag my-deepseek
```

Each node uses its own local runtime/spill directory. Four ranks form one model instance. Wait for each rank's ready record before starting the HTTP frontend.

On rank 0, in the environment with working vision dependencies:

```sh
python3 tools/serve_original.py vision \
  --runtime-dir /opt/deepseek-runtime \
  --checkpoint /models/DeepSeek-V4.1-Flash-1017 \
  --vision-socket /opt/deepseek-runtime/vision.sock
```

Create a key file once, without printing its value:

```sh
umask 077
python3 -c 'from pathlib import Path; import secrets; Path("/opt/deepseek/api-key").open("x").write(secrets.token_urlsafe(32)+"\n")'
```

Start the paired frontend on rank 0:

```sh
python3 tools/serve_original.py frontend \
  --build-dir /opt/deepseek-build --runtime-dir /opt/deepseek-runtime \
  --listen 127.0.0.1:18893 \
  --key-file /opt/deepseek/api-key \
  --tokenizer /models/DeepSeek-V4.1-Flash-1017/tokenizer.json \
  --vision-socket /opt/deepseek-runtime/vision.sock \
  --vision-revision 1bae0e1088993b9b82de08f59de5be5a34e42170
```

For a trusted LAN endpoint, bind to that machine's specific LAN address and apply your normal access controls. The frontend refuses wildcard binds. `--model-name` changes the exposed API model name.

`serve_original.py` rejects steering options. `run.py` also clears inherited steering settings unless the operator supplies the explicit opt-in flags. An ordinary launch therefore needs no vector asset, no mode suffix and no dual-mode setup.

## 5. Choose four or eight active requests

`profiles/c4.json` is the default: four slots, 1,048,576 tokens each, native24 verification, slice64 and probability-based speculation. Use `--profile c8-reference` on every rank for eight slots with the same per-request limit. Keep the frontend/profile pair aligned.

The eight-slot configuration has completed a real 8×1M fill with generation, with no new swap/OOM kill/disk spill; [capacity report](benchmarks/capacity-8m-20261108.md). Test available headroom on your nodes before adopting it. The separate session-history pool is not a second fixed active-token window. Full-window long-duration endurance remains unqualified.

## Optional modes, API and operations

[Activation steering](STEERING.md) explains how to opt in with the bundled experimental asset or your own directions. [API reference](API.md) covers reasoning, tools, images, cache and unsupported structured-output constraints. [Benchmarks](BENCHMARKS.md) separates throughput, real agent history and latency limits.

These examples are foreground processes. For persistent service use, configure your own supervisor and explicit dependencies. Stop only your instance, let accepted work finish, and restore its paired configuration if rolling back. A process restart loses live KV state. The [maintenance guide](MAINTENANCE.md) contains source/protocol and operational checks; historical fleet scripts are retained for provenance and should not be replayed against another deployment.
