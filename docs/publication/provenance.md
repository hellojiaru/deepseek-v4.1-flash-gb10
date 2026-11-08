# Source and license provenance

The upstream dsgx10 code remains under its [MIT license](../../LICENSE),
Copyright (c) 2026 Andy Shu. Existing third-party notices remain in the root
[NOTICE](../../NOTICE). Do not remove upstream authorship or present the full
engine as a new implementation by this fork.

## Integration base

The public fork is based on
[`blaesus/dsgx10` revision `a92b8f44046f4435c3c29c07fbc39e6e4b157080`](https://github.com/blaesus/dsgx10/tree/a92b8f44046f4435c3c29c07fbc39e6e4b157080).
The implementation was ported from a locally tested branch, then integrated
against this newer upstream tree. It is not a byte-for-byte publication of a
deployment directory. Private workloads, service scripts, logs, model weights,
unreleased numerical changes, and unfinished reference-model experiments are
excluded.

## DeepInfer-informed probabilistic sampling

The probability branch was reimplemented in dsgx10's existing C++/CUDA sampler
and native ring execution framework after reading DeepInfer. It is **not a
clean-room implementation** and is not a wholesale import of DeepInfer's
Python/TileLang kernels. The relevant fixed reference is
[`deepseek-ai/DeepInfer` revision `8b16149741a4935e1dec8cea27853f273b374c5b`](https://github.com/deepseek-ai/DeepInfer/tree/8b16149741a4935e1dec8cea27853f273b374c5b),
licensed under [Apache-2.0](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/LICENSE).
A copy with unchanged license text and LF-normalized line endings is included at
[`LICENSES/Apache-2.0-DeepInfer.txt`](../../LICENSES/Apache-2.0-DeepInfer.txt).

| Reference consulted | Corresponding additions |
|---|---|
| [`models/mtp/dflash.py`, stochastic draft generation](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/deepinfer/model_executor/models/mtp/dflash.py#L279-L344) | `engine/dspark/dspark.cu`, proposal distribution capture and random proposals |
| [`tilelang_ops/cuda/fused_sample.py`, acceptance and residual sampling](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/deepinfer/model_executor/layers/tilelang_ops/cuda/fused_sample.py#L375-L443) | `engine/model/sample.cu`, `engine/common/spec_sampling.h`, probability-ratio acceptance and residual resampling |

Paths in this table are relative to `ds41f-4xgx10-switchless`. Existing MIT
portions of these files retain their original license. The DeepInfer-derived
probabilistic additions are distributed under Apache-2.0 with source notices;
this does not relicense the entire repository or remove the original MIT grant.
The referenced DeepInfer revision has no root `NOTICE` file; this project's
attribution is not represented as an upstream NOTICE.

Material adaptations include sharded proposal normalization/verification using
the native ring, per-slot proposal metadata and lifecycle invalidation, CUDA
Graph integration, independent random domains, and audit counters. The residual
uses stable log-difference arithmetic rather than copying DeepInfer's residual
penalty constant. Unsupported sampling/gate configurations preserve the earlier
path. Independent synthetic tests and the exact rational oracle accompany the
port; no claim is made that the complete DeepInfer runtime was ported.

When redistributing these additions, retain the MIT notices, include the Apache
license, retain relevant attribution, and identify subsequent modifications.
The full license texts govern; this file is a source map, not a substitute for
their terms or a grant for model weights.

## Native vision reference

The bundled DeepSeek reference is MIT, Copyright (c) 2023 DeepSeek, with its
license at [`ref/LICENSE-deepseek`](../../ds41f-4xgx10-switchless/ref/LICENSE-deepseek).
For this fork:

- `vision/reference_image_processor.py` has the same source content as this
  base's `ref/inference/image_processor.py`, with line endings normalized to LF.
- `vision/reference_vision.py` is adapted from this base's
  `ref/inference/vision.py`: SDPA gets an explicit batch dimension, and the
  precomputed cosine/sine tensors are moved to the input device. This does not
  replace CPU RoPE arithmetic with a claimed equivalent GPU computation.
- `vision/encoder.py` supplies bounded encoding/cache/service integration; the
  model and processor implementations retain the above source provenance.

The public base's sources can be inspected at
[`ref/inference` at the pinned upstream revision](https://github.com/blaesus/dsgx10/tree/a92b8f44046f4435c3c29c07fbc39e6e4b157080/ds41f-4xgx10-switchless/ref/inference).
The tested model checkpoint is
[`DeepSeek-V4.1-Flash-1017`, revision `1bae0e1088993b9b82de08f59de5be5a34e42170`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/tree/1bae0e1088993b9b82de08f59de5be5a34e42170).
A checkpoint pin is **not** evidence that every bundled reference source was
independently fetched and compared at that checkpoint revision. The local
relationship, historical runtime hashes, and public LF-normalized source hashes are recorded in
[`vision/PROVENANCE.json`](../../ds41f-4xgx10-switchless/vision/PROVENANCE.json).
The worker's encoder identity hashes source bytes, so line-ending normalization
changes that identity even when Python source content is unchanged. Do not reuse
an old encoder/cache identity for a rebuilt public copy; let the worker derive
the identity from its actual source and checkpoint data.

PyTorch, NumPy, Pillow, safetensors, CUDA/driver/verbs components, and Rust crates
retain their own licenses. This repository does not redistribute model weights
or promise that every external dependency uses the root MIT license.
