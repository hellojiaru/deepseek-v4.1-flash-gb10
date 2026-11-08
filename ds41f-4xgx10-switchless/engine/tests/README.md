# Public engine regression tests

All included fixtures are synthetic probability distributions or generated
token/image-span metadata. No user messages, screenshots, weights, credentials,
host addresses, or production deployment scripts are needed.

## Offline CPU tests

The independent exact-law oracle needs only Python 3:

```sh
python3 engine/tests/distribution_oracle.py
```

It checks 3,969 pairs of small rational distributions, nine named distributions,
first-rejection/bonus behavior, and a counterexample for proposal-dependent
look-ahead gating. It proves the finite test distributions' algebraic law, not
the accuracy of every floating-point or GPU execution.

Build host helpers separately from the CUDA engine, from this project directory:

```sh
cmake -S engine/tests -B build-host -DCMAKE_BUILD_TYPE=Release
cmake --build build-host --config Release
ctest --test-dir build-host -C Release --output-on-failure
```

`vision_prefix_test` and `vision_features_test` require only a C++20 compiler.
`probability_host_test` additionally uses installed CUDA headers (not a CUDA
compiler or GPU); it exhaustively checks 8,388,608 representable RNG midpoints,
proposal metadata, residual arithmetic, and batch first-rejection behavior.
The standalone configuration skips that target if headers are unavailable.
On Linux, `runtime_metrics_test` checks the host-only epoch/counter writer.

The Linux-only `vision_budget_test` is disabled by default. It exercises the
actual restricted spool-file helper, cancellation, invalid files, 58 synthetic
image spans totaling 53,416 rows, and chunk/mask boundaries. Run it only inside a
disposable container with a private `/tmp`, with
`-DDSGB10_ENABLE_FILE_TESTS=ON`. It never starts an inference service, but its
real file checks use `/tmp/dsgb10-vision-features`; do not share that directory
with a running frontend.

Regenerate the purely synthetic GPU fixtures in a temporary directory using
`python3 engine/tests/distribution_oracle.py --freeze TEMP_DIRECTORY`, then
compare `distribution_fixtures.h` and `distribution-fixtures.json` before
intentionally updating their checked-in versions.

## Explicit GPU/communication probe

The probe is **not** part of CTest. Build it only in the normal Linux engine
project with the upstream CUDA 13 / SM121a requirements:

```sh
cmake -S engine -B build -DDSGB10_BUILD_GPU_PROBE=ON
cmake --build build --target probabilistic_sample_probe -j2
```

It requires four exclusively reserved compatible ranks, the existing ring
transport dependencies, a user-supplied fleet configuration, and a fresh test
port. It uses no model weights. On each rank, launch the same binary with its
rank number, for example:

```sh
timeout --kill-after=5s 150s ./build/tests/probabilistic_sample_probe \
  --fleet /path/to/test-fleet.conf --rank 0 --port 29680 \
  --trials 4096 --deadline 120
```

Use ranks 0 through 3 and the same test port; coordinate and reap all processes
using your own test infrastructure. The local wall deadline does not prove
cleanup under every remote peer failure, so retain the external timeout. Do not
run this against an active production fleet or its socket/port. No launcher,
SSH automation, service restart, or automatic GPU discovery is provided here.

The probe exercises the actual sharded `ops::sample`, independent CPU-generated
proposals, exact-law expected distributions, CUDA Graph reuse, multiple slots,
stale/unsupported metadata fallback, and deterministic cases. It does not run
the complete DSpark drafter, a complete model, image encoding, or task quality.
Full-model acceptance, end-to-end performance, and image correctness require
separate application-level tests.
