# Multimodal agent fork

This fork extends the `ds41f-4xgx10-switchless` configuration. The other engine
configuration and the upstream performance history are retained, not revalidated
by this work. The integration base is upstream commit
[`a92b8f44046f4435c3c29c07fbc39e6e4b157080`](https://github.com/blaesus/dsgx10/commit/a92b8f44046f4435c3c29c07fbc39e6e4b157080).

The additions are:

- Native image encoding, image-aware prefix identity, and separate prefix
  policies (`legacy`, `exact`, and `tail-exact`). Image identity, position, model
  mode, encoder identity, and policy participate in cache compatibility.
- Protocol 5 file-backed image features. The frontend retains immutable BF16
  feature files; rank 0 reads the image spans needed by each prefill chunk.
  Historical images remain in the request. This avoids requiring all historical
  features to reside in the active GPU feature buffer at once.
- Optional probabilistic DSpark. Stochastic proposals retain their actual
  distribution; verification uses probability-ratio acceptance and residual
  resampling. The default remains off; enable with `--spec-probabilistic`.
- Host and explicit GPU tests, synthetic fixtures, and runtime counters that
  distinguish proposed, accepted, and actually adopted draft tokens.

See [build and test](build-and-test.md), [measurements and limits](measurements.md),
[source and license provenance](provenance.md), and the
[publication privacy audit](privacy-audit.md).

## Compatibility and limits

The engine remains specific to four compatible GB10/SM121a ranks and the
existing native ring transport. This is not a general DeepInfer deployment,
SM100 port, NCCL backend, or distributed vision service. The frontend and rank 0
must share the feature-file filesystem and Unix user. Use a disk-backed spool;
putting the spool on tmpfs converts its disk budget into host-memory pressure.

The matching engine and frontend must be installed together: protocol 5 is not
wire-compatible with older engine/frontend combinations. Treat their CLI,
checkpoint, tokenizer, encoding version, and prefix policy as one configuration.
The `--encoding 1017` option predates this fork and is retained from upstream.

The native vision worker is a Python runtime dependency, in addition to the
upstream C++/CUDA engine and Rust frontend. The earlier upstream statement that
Python is only used offline does not describe this fork's image-serving path.

Image feature caching does not eliminate image tokens, the HTTP request body, or
all preprocessing. A cache hit is not a guarantee that a changed client request
still has the same token prefix. Client-side history editing or image offload is
outside this repository. The file protocol is a private local transport, not a
public file-upload or arbitrary-file-read API.

This work does **not** claim to fix every long-context visual reasoning error.
Some incorrect answers observed before these changes remain. There is no claim
of unrestricted image history, persistent KV cache across restarts, validated
1M-token/384K-output endurance, or universal 100+ token/s generation. Unreleased
numerical experiments and an unfinished full-model reference implementation are
not part of this fork.

## Contribution boundaries

Useful review units are: (1) the sampler's open-interval RNG fix and host test;
(2) probabilistic DSpark, lifecycle handling, and distribution tests; (3) vision
encoding and protocol 5; and (4) image-aware prefix policy and cache regressions.
These are proposed review boundaries, not a claim that upstream has accepted
them. Engine/frontend protocol changes need paired review even when the code is
split into separate commits.

Do not submit local deployment scripts, provider credentials, real user
conversations, screenshots, request/response captures, or machine/account
inventories with a contribution. Use generated inputs or inputs you have
permission to redistribute. The existing upstream benchmark material has its
own attribution in the root `NOTICE`; it is distinct from the private workloads
used during development of this fork.
