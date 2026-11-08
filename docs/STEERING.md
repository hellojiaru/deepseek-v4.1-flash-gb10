# Optional activation steering

[中文详细说明](STEERING.zh-CN.md)

## What this repository distributes

This project publishes the **method, projection/loading source, data format and local packing tool**, plus an explicitly optional [Creative v1 asset](../assets/steering/creative-v1/README.md). The bundle contains a binary, readable numeric JSON, a hash manifest and its MIT license. It is not downloaded or loaded automatically. Users can instead derive and inspect their own asset locally.

The ordinary setup is `tools/serve_original.py`: the original checkpoint, no vector loaded, no extra model copy. It rejects steering options and clears inherited steering environment variables. Optional steering is separate from that default.

The readable JSON reproduces the bundled binary exactly. The legacy calibration prompts and activation captures are not included, so the original fitting run cannot be fully replayed from this repository. A hash checks identity; it does not establish the meaning, quality or licensing of an asset.

## What the two modes actually do

Activation steering adjusts an internal attention-output vector, not the stored model weights. For a unit direction `r`, output `y` and strength `lambda`, the implemented operation is:

```text
y_new = y - lambda * dot(y, r) * r
```

The current engine exposes two states:

| API mode | Strength | Effect |
|---|---:|---|
| Normal | 0 | The projection kernel returns before reading or writing activations. |
| Creative | 1.5 | The supplied direction is applied to attention outputs. |

Writing `y` as components parallel and perpendicular to `r` gives `y_new = y_perpendicular + (1-lambda)*y_parallel`. At 1.5, the parallel component is multiplied by -0.5. This is not a temperature preset, an automatic creativity improvement or a guarantee of a particular refusal rate.

Both prefill and decode use the direction. The current decode hook is after the local attention `wo_b` projection and before its TP sum/mHC fusion; prefill projects after the four-rank output sum. Linear projection commutes with a sum in exact arithmetic, but BF16 rounding order can differ. A direction fitted at an unrelated hidden-state hook must not be assumed equivalent.

The model's weight files remain unchanged. The DSpark draft path is not assigned a separate fitted vector by this mechanism. Mode changes happen at safe execution boundaries; the same physical pool is shared, and histories are cached in separate mode namespaces.

## A direction-fitting workflow

The historical calibration approach used normalized differences from matched prompt pairs and an **uncentered first principal direction** over middle layers. The following describes that approach so users can reproduce and assess a local experiment. The bundled legacy direction is one example; this workflow is not a quality guarantee for that asset or a new fit.

1. **Define the behavior.** For example, distinguish an unnecessary refusal of an otherwise answerable request from an appropriate substantive response. Label the actual responses rather than equating length with success. Keep factual correctness and requested behavior as separate checks.
2. **Create matched pairs.** Use self-authored or appropriately licensed prompts with similar task, language, length and structure. Include several domains and harmless controls. Avoid treating a topic difference as proof of a behavior direction. Keep a separately held-out set by topic and wording.
3. **Capture the original model.** With steering disabled, collect the selected internal activation from each pair at the same hook, token-position/pooling rule and numerical configuration. Record the exact checkpoint, tokenizer/template, hook, pooling, dtype, rank interpretation and source hashes. Ordinary Chat Completions returns text, not these internal activations: this step needs an instrumented local reference/capture implementation. This repository does not currently provide a one-command full-model capture pipeline.
4. **Form normalized differences.** For pair `p`, layer `l`, subtract the two activation summaries: `d[p,l] = h[A,p,l] - h[B,p,l]`. Discard zero/invalid differences, then normalize each valid pair-layer difference to unit norm.
5. **Fit a shared direction.** The historical approach pooled layers 10 through 29 inclusive, without subtracting their mean, and took the first right-singular vector. Normalize the result and copy that shared direction to the 40 layer entries. The file format also permits different directions per layer; that would be a separately evaluated experiment.
6. **Inspect and record.** Save a readable JSON plus checkpoint/source/sample identities, independent prompt count and fitting choices. Layer rows are not additional independent prompts. Principal-component energy alone does not demonstrate a behavioral improvement.
7. **Evaluate before adopting.** Compare original and steered responses on held-out tasks, then check ordinary accuracy, code execution, tools, images, thinking, repeated histories and mode switching. Retain regressions as well as successes. A reduction in refusals does not prove better factual accuracy or overall creative quality.

Illustrative offline fitting pseudocode, assuming `H` already contains consistently captured activation summaries with shape `[pairs, 2, 40, 5120]`:

```python
import numpy as np

H = np.asarray(H, dtype=np.float64)
D = (H[:, 0, 10:30, :] - H[:, 1, 10:30, :]).reshape(-1, 5120)
norms = np.linalg.norm(D, axis=1)
valid = np.isfinite(D).all(axis=1) & np.isfinite(norms) & (norms > 1e-12)
D = D[valid] / norms[valid, None]
if len(D) == 0:
    raise ValueError("no valid pair-layer differences")
_, singular_values, Vt = np.linalg.svd(D, full_matrices=False)  # no centering
r = Vt[0] / np.linalg.norm(Vt[0])
asset = {"hidden_size": 5120,
         "directions": {str(layer): r.tolist() for layer in range(40)}}
```

This code illustrates fitting, not activation capture or a prequalified calibration set. The sign of `r` does not change the implemented projection. Rescaling `r` is rejected by the loader's unit-norm check. To change behavior with the current interface, fit a genuinely different direction. Arbitrary strength would require an engine/interface change and new qualification; it is not an existing request parameter.

## Readable local asset and runtime format

A local JSON must contain all layers `0` through `39`, each with 5,120 finite numeric components and squared norm within `2e-4` of 1. The following shows the structure, not usable vector values:

```json
{
  "hidden_size": 5120,
  "directions": {
    "0": ["5120 numeric components"],
    "1": ["5120 numeric components"],
    "39": ["5120 numeric components"]
  }
}
```

Pack a locally obtained asset using its verified SHA:

```sh
python3 tools/pack_creative.py /abs/my-directions.json /abs/my-directions.bin \
  --source-sha256 <sha256-of-json> \
  --checkpoint-revision 1bae0e1088993b9b82de08f59de5be5a34e42170
```

The tool validates shape, finiteness and norm, refuses to overwrite output, and produces a DSCR0001 file plus a manifest. It does not train or download a direction.

| Byte offset | Content |
|---:|---|
| 0–7 | ASCII `DSCR0001` |
| 8–11 | Little-endian uint32 layer count: 40 |
| 12–15 | Little-endian uint32 dimension: 5120 |
| 16 onward | Layer-major little-endian float32 values: 40×5120 |

Total size is 819,216 bytes. This is raw numeric data; the loader does not deserialize executable Python/pickle objects. The complete JSON-to-binary conversion is in [pack_creative.py](../tools/pack_creative.py), and loading/validation is in [model.cu](../ds41f-4xgx10-switchless/engine/model/model.cu). A user can inspect their JSON, recompute its hash and reproduce the binary locally. The optional bundled example has both representations and a byte-for-byte conversion check in CI.

## Explicitly enable the bundled asset or a local asset

For the bundled example, copy `assets/steering/creative-v1/directions.bin` to an absolute path on **each** node. Read `suggested_scope` in its `manifest.json` and use that exact scope on all ranks and the frontend. The [asset card](../assets/steering/creative-v1/README.md) explains its provenance, verification and limitations. Use the same rank, frontend, tokenizer and vision arguments from the [startup guide](BUILD_AND_RUN.md), with `tools/run.py` and the additional flags below.

Use `tools/run.py` instead of `serve_original.py`. On **every rank**, add:

```sh
--enable-steering --creative-file /abs/my-directions.bin \
  --creative-scope <suggested_scope-from-manifest>
```

On the paired **frontend**, add:

```sh
--enable-steering --creative-scope <the-same-scope>
```

The engine checks shape, finite values, norms and agreement across all four ranks. The frontend then lists `<model-name>-normal` and `<model-name>-creative`. Choose the desired name per request. Switching these two states does not reload the full model; replacing a direction requires a new checkpoint/asset scope and restart of the paired instance.

The current enabled strength is fixed at 1.5. Live vector replacement and per-request arbitrary strength are not implemented. Retest mode cache isolation and ordinary capabilities after any change; a cache identity is not evidence that a new asset preserves quality.

## Source map and evidence boundaries

- [creative.cu](../ds41f-4xgx10-switchless/engine/model/creative.cu): the projection and Normal early return.
- [model.cu](../ds41f-4xgx10-switchless/engine/model/model.cu): decode hook, asset validation and mode state.
- [prefill.cu](../ds41f-4xgx10-switchless/engine/model/prefill.cu): prefill hook.
- [run.py](../tools/run.py): explicit opt-in and environment handling.
- [Original-launch acceptance](benchmarks/original-launch-20261109.md): original model versus the existing Normal path, output equivalence in the tested 24 cases and capability checks.

The bundled direction is an experimental historical asset, not a new behavioral benchmark or an endorsement for every task. Its binary identity is preserved; private prompt labels and capture records are excluded. The exact extraction checkpoint revision was not retained in the available legacy record. Do not describe it as freshly fitted on 1017 or infer a measured quality gain from successful loading. Keep externally obtained assets' licenses and provenance separate.
