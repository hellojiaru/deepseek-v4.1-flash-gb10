# Measurements and what they establish

These are historical validation results from the deployed implementation from
which the additions were ported. They are **not fresh measurements of this
rebased public checkout**. The original input and operational evidence contain
private material and are not distributed. Public tests reproduce algorithmic
and protocol checks, not the exact private workload's throughput.

The older performance tables in the root/configuration README belong to
upstream. They use different revisions, prompts, generation lengths, and
sampling settings. They must not be combined with the following table into a
single benchmark or a claim that this fork always generates at 100+ token/s.

## Probabilistic DSpark: same-binary off/on/off

Four GB10 ranks ran the same candidate binary with the probability path disabled,
enabled, then disabled again. The fixed private game-development input had
18,306 prompt tokens. Each run generated up to 2,048 output tokens at
temperature 1, top-p 1, top-k 0. Normal mode used three fixed seeds; Creative
used one. Each phase also included six short tool/image continuation requests.
The phases spanned two bounded maintenance windows; they were not one uninterrupted
experiment. Per-case inputs, sampling settings, cached-token counts, and output
counts were matched across phases.

Decode throughput is `(output_tokens - 1) / native_decode_seconds`. Combined
rates use total tokens divided by total time, not the mean of per-request rates.
Improvement uses the mean decode time of the two disabled phases.

| Mode | Off before | On | Off after | Improvement |
|---|---:|---:|---:|---:|
| Normal, three seeds combined | 70.71 token/s | 93.24 token/s | 70.68 token/s | 31.88% |
| Creative, one seed | 74.82 token/s | 91.11 token/s | 74.85 token/s | 21.75% |

Actually adopted draft tokens / verified draft tokens changed from
3,064/8,246 (37.16%) to 3,745/5,795 (64.62%) in Normal, and from 1,090/2,595
(42.00%) to 1,242/1,950 (63.69%) in Creative. Verification steps fell from
3,074 to 2,393 and 956 to 804 respectively. This supports a benefit from more
effective speculation; it does not establish that an individual kernel became
faster. Raw accept flags include rows discarded after the first rejection and
must not be reported as adopted draft tokens.

Disabled-before/after outputs and speculation counts matched; time drift was
below 0.1%. Enabled outputs can differ, even at the same seed. Every long sample
hit its output cap during reasoning, so this was not a completed game or task
quality evaluation. It tested serial serving, not concurrent-session throughput.
No temperature reduction or changed cache-hit target was used to obtain the gain.

The enabled phase reported 7,800 probability-verification rows and zero
unexpected fallback, skipped, or zero-residual events. The old deterministic
proposal plus target-sample equality path is not being described as an invalid
target distribution. The new path changes proposal/acceptance efficiency.

## Distribution and execution checks

The standard-library Python oracle covers 3,969 pairs of rational distributions,
first rejection, bonus tokens, and a look-ahead gating counterexample. A separate
four-rank GPU probe exercised the actual verifier over nine distributions with
4,096 trials each, plus Graph input changes, slot reordering, metadata failures,
and unsupported settings. Replicated rank outputs are not four independent
samples. The microprobe supplies independent proposals; full-model runs are
needed to exercise the drafter's proposal lifecycle.

The supported probability path is temperature > 0, top-p = 1, top-k = 0, with
a supported causal gate. Other combinations retain the earlier path. Floating
point and finite random precision remain; these tests are not a proof of exact
arbitrary tiny probabilities or all concurrency schedules.

## File-backed historical image features

A capacity/control test used generated images: 49 at 1,944 x 1,215 pixels, five
at 780 x 1,688, and four at 1,200 x 750. The 58 images produced 53,416 BF16 rows
of width 5,120: 546,979,840 logical feature bytes (521.64 MiB), larger than the
old all-history 512 MiB request limit. Adding a 59th image increased the logical
total to 556,830,720 bytes.

Twelve requests across Normal/Creative covered cold and hot history, an appended
image, first-image replacement, a cold reference, and a short text request with
a large requested output budget. Layout/count/byte checks passed, and every
expected color answer in that generated fixture passed. Peak active feature
storage was 98,508,800 bytes (93.95 MiB), **not total model/GPU memory**. Hot
requests required no additional feature-file reads or image encoding. A separate
16-case before/after gateway matrix introduced no new failures, while two known
incorrect answers in the baseline remained.

This establishes bounded feature transport and limited cache/answer regressions.
It does not establish that every long visual reasoning task is correct, or that
a 58-image synthetic fixture reproduces a user's original image history. Encoder
cache warmth and KV warmth are separate; do not compare one mode's encoder-cold
time to another mode's encoder-warm time as a model-speed comparison.

## A public local comparison recipe

Use the checked-in synthetic probability fixtures for the verifier checks. For
an application benchmark, create redistributable inputs and record their hashes,
actual token counts, image geometry/visual rows, model revision, encoding,
mode, temperature, top-p/top-k, seeds, and generation cap. Run the same binary
off/on/off with identical inputs and an explicit cold/warm cache policy. Record
HTTP first-token latency separately from native prefill/decode time; record
adopted/verified drafts separately from raw acceptance flags. Review complete
answers as well as speed and length. Preserve failures rather than quietly
excluding them. This recipe does not reproduce the private-input numbers above.
