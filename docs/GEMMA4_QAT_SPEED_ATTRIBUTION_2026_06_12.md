# Gemma 4 QAT Speed Attribution - 2026-06-12

## Scope

This note tracks the immediate question: whether poor Gemma 4 MXFP4/JANG_4M
speed versus llama.cpp GGUF is caused by Osaurus app overhead or by
vmlx-swift/runtime behavior.

Focus models for this checkpoint:

- `OsaurusAI--gemma-4-E2B-it-qat-MXFP4`
- `OsaurusAI--gemma-4-E2B-it-qat-JANG_4M`
- Matching llama.cpp GGUF baseline:
  `/Users/eric/models/unsloth-gemma4-qat-gguf/gemma-4-E2B-it-qat-GGUF/gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf`

## Current Verdict

Status: `PARTIAL / RAW SPEED ROWS EXIST ON THE DRAFT VMLX SPEED BRANCH`.

The current evidence points at vmlx-swift, not Osaurus, for both the earlier
missing raw speed result and the remaining decode gap versus GGUF. The valid raw
vmlx-swift entrypoint is `RunBench` with `BENCH_PERF=1`.

The old local failure below was a real loader blocker on this checkout. A draft
vmlx speed branch now has local artifacts proving raw release `RunBench` decode
rows at commit `320db4da`. Those rows show E2B MXFP4/JANG_4M are slower than
GGUF, but not catastrophically slow:

- E2B MXFP4 deterministic: `144.7 tok/s`.
- E2B JANG_4M deterministic: `130.7 tok/s`.
- GGUF E2B Q4_K_XL decode: `173.677288 tok/s`.

Interpretation: for E2B, vMLX MXFP4 is about 16.7% slower than GGUF and
JANG_4M is about 24.7% slower than GGUF on the current raw deterministic row.
The current branch in this checkout is not that draft speed branch, so do not
claim this worktree itself is fixed until the relevant load fixes are ported and
`RunBench` is rerun here.

## Proven Data

### llama.cpp GGUF baseline

Artifact:

- `/tmp/vmlx-gemma4-e2b-compare-20260612T155643Z/gguf-llama-bench.json`

Result:

- Prompt eval: `7384.720274 tok/s`
- Decode: `173.677288 tok/s`
- Model: `gemma4 E2B Q4_0`
- Backend: llama.cpp build `d2462f8f7`, Metal backend on Apple M5 Max
- Test shape: `n_prompt=512` and `n_gen=128`

### vmlx-swift raw RunBench failure

Artifact:

- `/tmp/vmlx-gemma4-e2b-compare-20260612T155643Z/runbench-perf-20260612T160512Z/mxfp4.runbench.err`

Failure:

```text
MLXNN.UpdateError.unhandledKeys(
  path: ["language_model", "model", "per_layer_model_projection"],
  modules: ["Gemma4", "G4LanguageModel", "TextModel", "G4ScaledLinear"],
  keys: ["scales"]
)
```

Observed immediately before abort:

```text
[Load] JANG shape walk produced 277 per-layer quant override(s) over default (bits=4, gs=32)
[Load] JANG shape walk produced 277 per-layer quant override(s) over default (bits=4, gs=32)
```

That means the model is identified as quantized, but the unified Gemma 4
projection module does not accept or use the quantization sidecar for
`per_layer_model_projection`.

## Invalid Comparisons To Avoid

Do not use Osaurus tiny tool/API turns as raw decode speed evidence. The prior
Osaurus rows were useful for correctness, tool calling, cache, and UI behavior,
but they are not a fair speed comparison against `llama-bench` because they
included app/server overhead, request setup, tool routing, short outputs, and
sometimes VL/cache proof paths.

Do not use deprecated `mlxpress` rows for this question. Current package speed
proof is `RunBench`, especially `BENCH_PERF=1`.

## Attribution Test

Use this decision table:

| Raw vmlx-swift RunBench | Osaurus API long generation | Attribution |
| --- | --- | --- |
| Cannot load or crashes before decode | Any result | vmlx-swift loader/runtime blocker |
| Loads but raw tok/s is already far below GGUF | Any result | vmlx-swift runtime/kernel/cache path |
| Raw tok/s is close to GGUF, Osaurus is far slower | Slow only in Osaurus | Osaurus app/API/policy/cache overhead |
| Both raw and Osaurus are close to GGUF | No speed bug for that row | Compare correctness/cache next |

Current row lands in the first case: raw vmlx-swift cannot load and decode.

The draft speed branch rows land in the second case: raw vmlx-swift loads and
decodes, but remains measurably slower than the GGUF baseline. That shifts the
active question from "why no number?" to "which vMLX runtime path causes the
remaining 17-25% E2B gap?"

## Verified Raw Speed Matrix From Draft Branch

Artifact roots verified locally:

- `/tmp/vmlx-gemma4-qat-speed-standard-20260612T163135Z/SUMMARY.txt`
- `/tmp/vmlx-gemma4-e4b-standard-20260612T163814Z/SUMMARY.txt`
- `/tmp/vmlx-gemma4-12b-standard-20260612T164459Z/SUMMARY.txt`
- `/tmp/vmlx-gemma4-26b-a4b-standard-20260612T164936Z/SUMMARY.txt`
- `/tmp/vmlx-gemma4-31b-standard-20260612T165017Z/SUMMARY.txt`

All rows use release `RunBench`, `BENCH_PERF=1`, `BENCH_PERF_PATH=batch`,
single request, `BENCH_MAX_TOKENS=128`, `BENCH_PERF_WARMUP=1`, and
`BENCH_PERF_RUNS=3`.

| Model | Quant | Deterministic | Bundle Defaults | Caveat |
| --- | --- | ---: | ---: | --- |
| E2B | MXFP4 | `144.7 tok/s` | `135.3 tok/s` | clean speed row |
| E2B | JANG_4M | `130.7 tok/s` | `126.0 tok/s` | clean speed row |
| E4B | MXFP4 | `92.0 tok/s` | `88.7 tok/s` | clean speed row |
| E4B | JANG_4M | `81.5 tok/s` | `79.1 tok/s` | clean speed row |
| 12B | MXFP4 | `48.9 tok/s` | `48.3 tok/s` | deterministic reports `loop=YES` |
| 12B | JANG_4M | `39.9 tok/s` | `39.1 tok/s` | clean speed row |
| 26B A4B | MXFP4 | `95.4 tok/s` | `91.5 tok/s` | bundle-default reports `unclosedReasoning=YES` |
| 26B A4B | JANG_4M | `82.4 tok/s` | `78.1 tok/s` | clean speed row |
| 31B | MXFP4 | `22.3 tok/s` | `20.9 tok/s` | clean speed row |
| 31B | JANG_4M | `16.1 tok/s` | `15.8 tok/s` | bundle-default reports `unclosedReasoning=YES` |

These are raw text speed rows only. They do not prove Osaurus app behavior,
tool calling, VL/audio behavior, or prefix/L2 cache TTFT.

## Useful Swift MLX PRs / Repos

Upstream `ml-explore/mlx-swift-lm` has the closest current Gemma 4 work:

- PR #309, merged 2026-06-03: `per_layer_model_projection` was changed from a
  custom scaled module to a plain `Linear`, with the scale applied after the
  projection. This is the clean fix shape because the existing loader can then
  replace it with `QuantizedLinear`.
- PR #342, open: QAT E-series checkpoints omit `k_proj`, `v_proj`, and `k_norm`
  on KV-shared layers. The fix makes those modules optional and drops redundant
  PTQ tensors in sanitize.
- PR #249, merged 2026-05-22: fused Gemma4 decode fragments (`residual+rmsNorm`
  and `gelu*mul`) with `compile(shapeless:)`, measured `51.2 -> 63.4 tok/s`
  on E2B 4-bit, a `+23.8%` decode improvement.
- PR #337, merged 2026-06-11: `Gemma4.prepare` now honors `windowSize`; this is
  a TTFT/prefill-memory fix, not a decode-token/s fix.
- PR #333, merged 2026-06-10: no-cache forwards now still pass shared KV to
  later layers; this matters for eval/embedding parity, not the normal cached
  decode loop.

Local references:

- `/Users/eric/vmlx-swift-lm/scripts/perf-sweep.sh` has the older generic
  Gemma4 `RunBench` sweep pattern.
- Draft branch `vmlx-origin/pr-44:scripts/run-gemma4-qat-speed-standard.sh` has
  the current QAT-specific speed script with deterministic and bundle-default
  rows.
- `/Users/eric/vmlx-swift-lm/BENCHMARK-GEMMA-4-26B.md` gives a direct runtime
  comparison shape for 26B 4-bit: vmlx-swift-lm averaged `88.4 tok/s` decode
  and `1091 tok/s` prefill over turns 2-5, with explicit cache-method notes.

## Current Root-Cause Hypothesis

The immediate hard failure is in the MLXVLM Gemma 4 unified language path:
`G4ScaledLinear` for `language_model.model.per_layer_model_projection` lacks
verified support for quantized sidecar keys such as `scales` and, for affine
variants, possibly `biases`.

Once that loader issue is fixed and verified, the next likely performance
bottlenecks to check are:

- Whether Gemma 4 MXFP4/JANG_4M projections use fused Metal quantized matmul or
  fall back to slower generic `quantizedMM`/sidecar handling.
- Whether JANG_4M layers are streaming/dequantizing more than llama.cpp GGUF's
  Metal Q4 path.
- Whether TurboQuant KV or rotating KV cache encode/decode is accidentally on
  the critical path for the benchmark shape.
- Whether BatchEngine single-stream decode adds overhead relative to the direct
  iterator path.

These are hypotheses until `RunBench BENCH_PERF=1` reaches decode and prints a
valid tok/s line.

## Required Next Proof

The next valid proof must be raw vmlx-swift first, then Osaurus only if raw
vmlx-swift is healthy.

Raw vmlx-swift commands:

```bash
swift build -c release --product RunBench --jobs 2

BENCH_PERF=1 \
BENCH_MODEL=/Users/eric/models/OsaurusAI--gemma-4-E2B-it-qat-MXFP4 \
BENCH_PERF_VARIANT=e2b-mxfp4 \
BENCH_MAX_TOKENS=128 \
BENCH_PERF_RUNS=3 \
BENCH_PERF_WARMUP=1 \
.build/arm64-apple-macosx/release/RunBench

BENCH_PERF=1 \
BENCH_MODEL=/Users/eric/models/OsaurusAI--gemma-4-E2B-it-qat-JANG_4M \
BENCH_PERF_VARIANT=e2b-jang4m \
BENCH_MAX_TOKENS=128 \
BENCH_PERF_RUNS=3 \
BENCH_PERF_WARMUP=1 \
.build/arm64-apple-macosx/release/RunBench
```

Required output:

- A successful load.
- A `PERF ... tokps_median=...` or equivalent decode tok/s line.
- No weird/control-character output if the run emits visible text.
- RAM/process state captured separately from the speed line.

Only after this raw proof should Osaurus API be used to measure additional app
overhead.

## Process Hygiene

As of this note, leftover `swift-build`/`swift-frontend`/`swift-driver`,
`llama-cli`, and Osaurus runtime processes were stopped so there is no ongoing
background CPU/RAM burn from this triage lane.
