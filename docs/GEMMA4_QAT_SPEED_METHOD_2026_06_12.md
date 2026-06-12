# Gemma 4 QAT Speed Method - 2026-06-12

## Priority

Current release priority for Gemma 4 QAT MXFP4/JANG_4M is:

1. Speed: raw vmlx-swift release speed must be pushed toward llama.cpp GGUF
   speed before broad release claims.
2. Tool calling support: once raw speed/load is healthy, prove Osaurus
   tool-call and agent-loop behavior on the same pinned engine.

Do not promote a row from tool correctness alone if the raw vMLX engine speed
path is broken or unmeasured.

## Current Release Gate Status

Status: `PARTIAL`.

The current vMLX branch has Gemma 4 QAT sidecar loading fixes for MXFP4 and
JANG_4M, and the local E2B text rows load and generate coherently through
release `RunBench`. This is not yet a full Osaurus release checkpoint.

Do not mark the Osaurus PR merge-ready until all of these are true on the
pinned engine:

- Raw vMLX speed is understood and either improved or explicitly accepted
  against a matching llama.cpp GGUF baseline.
- Osaurus app/API loads MXFP4 and JANG_4M without RAM surprises on the target
  device class.
- Multi-turn chat is coherent with no loops, no protocol markers, no strange
  characters, and token/s recorded.
- Tool calling works through the Osaurus agent loop with exact tool names and
  JSON arguments.
- Prefix cache, disk L2 cache, and paged-cache-disabled behavior are proven
  from active runtime telemetry.
- VL/audio rows use real media payloads where the model supports them.

Known app-facing proof exists for a 12B JANG_4M checkpoint, but that proof is
not the full MXFP4/JANG_4M Gemma matrix and does not close the raw speed gate.

## Standard Token/S Definition

All raw vMLX Gemma 4 QAT speed claims must use `RunBench BENCH_PERF=1` from a
release build.

The reported decode token/s is:

```text
tokps = genTokens / genSec
```

where `genTokens` and `genSec` come from the generation completion info emitted
by the vMLX generation path. The promoted row is `tokps_median` over measured
runs after warmup. Do not mix this with wall-clock HTTP request time, UI TTFT,
small tool-call turns, or Osaurus agent-loop timing.

Minimum standard shape:

- Build: `swift build -c release --product RunBench`.
- Path: `BENCH_PERF_PATH=batch`.
- Batch: single request, no multibatch.
- Prompt: `Write one long paragraph describing ocean waves. Be verbose and detailed.`
- Tokens: `BENCH_MAX_TOKENS=128`.
- Warmup: `BENCH_PERF_WARMUP=1`.
- Runs: `BENCH_PERF_RUNS=3`.
- Promote: median decode tok/s, best tok/s, TTFT, prompt tok/s, RSS, physical
  footprint, model path, vMLX commit, and artifact root.

## Two Required Speed Rows

Every model needs two separate rows because sampler settings change the
runtime path and output shape.

### Row A: Deterministic Engine Comparison

Purpose: compare vMLX decode hot path against llama.cpp with as few sampler
variables as possible.

Required settings:

```bash
BENCH_PERF=1
BENCH_PERF_TEMP=0
BENCH_PERF_TOP_P=1
BENCH_PERF_TOP_K=0
BENCH_PERF_MIN_P=0
BENCH_PERF_USE_GENERATION_CONFIG unset
```

This row answers whether vMLX kernels/loading/cache are near GGUF speed.

### Row B: Bundle Defaults

Purpose: measure the actual user-facing bundle defaults.

Required settings:

```bash
BENCH_PERF=1
BENCH_PERF_USE_GENERATION_CONFIG=1
BENCH_PERF_SEED=1234
```

Explicit `BENCH_PERF_TEMP`, `BENCH_PERF_TOP_P`, `BENCH_PERF_TOP_K`,
`BENCH_PERF_MIN_P`, and repetition overrides must be absent unless the test is
explicitly labeled as an override diagnostic.

## Standard Script

Use:

```bash
scripts/run-gemma4-qat-speed-standard.sh
```

With no arguments, it runs E2B MXFP4 and E2B JANG_4M. To run the full matrix,
pass explicit local model directories under `/Users/eric/models`.

The script writes:

- `METADATA.txt`
- `SUMMARY.txt`
- one stdout/stderr pair for every model and mode

## llama.cpp Comparison

Use llama.cpp `llama-bench` for the GGUF baseline and compare only decode rows
with `n_gen=128` against vMLX deterministic Row A. Keep prompt eval separate.

Current known E2B GGUF baseline artifact:

- `/tmp/vmlx-gemma4-e2b-compare-20260612T155643Z/gguf-llama-bench.json`

Current known GGUF numbers:

- Prompt eval: `7384.720274 tok/s`
- Decode: `173.677288 tok/s`

Current E2B QAT comparison against that decode target:

| Row | vMLX tok/s | GGUF tok/s | Gap |
|---|---:|---:|---:|
| E2B MXFP4 deterministic, current-main clean PR branch | `122.1` | `173.677` | `29.7%` slower |
| E2B JANG_4M deterministic, current-main clean PR branch | `115.0` | `173.677` | `33.8%` slower |
| E2B MXFP4 deterministic, older PR base diagnostic | `143.2` | `173.677` | `17.5%` slower |
| E2B JANG_4M deterministic, older PR base diagnostic | `132.5` | `173.677` | `23.7%` slower |

Do not compare only MXFP4 against JANG_4M. Every speed row with a documented
GGUF peer must report the vMLX-vs-GGUF gap.

There is no current local 12B/31B GGUF peer artifact in this doc. Do not infer
their GGUF gap from E2B; first create matching llama.cpp `n_gen=128` baselines
for the same local bundle/quant tier.

## SwitchGLU Speed Work

Gemma 4 MoE uses `SwitchGLU` for routed experts. JANG/MXFP bundles can carry
`gate_up_proj` and `down_proj` tensor layouts that must be remapped into the
runtime `switch_glu` module contract before speed tuning.

SwitchGLU speed work must follow these rules:

- First prove the exact weight remap/load path, including `gate_up_proj`,
  `down_proj`, `scales`, and `biases` handling.
- Run the standard deterministic speed row before and after any SwitchGLU
  change.
- Run the standard bundle-default speed row before and after any SwitchGLU
  change.
- Do not enable whole-SwitchGLU compile or a TurboQuant replacement by default
  unless output quality, no-loop behavior, and token identity/semantic parity
  pass on the standard rows.
- Any SwitchGLU diagnostic must be labeled as diagnostic until it passes the
  same coherency, leak, and speed gates as the default path.

## Current E2B Release Numbers

Artifact:

- `/tmp/vmlx-gemma4-qatsidecar-clean-postrevert-e2b-20260612T182246Z`

Rows:

- E2B MXFP4 deterministic release: `tokps_median=122.1`,
  `tokps_best=122.3`, `TTFT ~113-120 ms`, `prompt_tps ~689-727`.
- E2B MXFP4 bundle defaults: `tokps_median=118.6`,
  `tokps_best=118.8`, `temp=1.00`, `topP=0.95`, `topK=64`.
- E2B JANG_4M deterministic release: `tokps_median=115.0`,
  `tokps_best=118.2`, `TTFT ~100-104 ms`, `prompt_tps ~699-729`.
- E2B JANG_4M bundle defaults: `tokps_median=110.3`,
  `tokps_best=110.7`, `temp=1.00`, `topP=0.95`, `topK=64`.

Interpretation versus GGUF E2B decode `173.677288 tok/s`:

- MXFP4 is about 29.7 percent slower than GGUF on this raw deterministic row.
- JANG_4M is about 33.8 percent slower than GGUF on this raw deterministic row.

This is no longer a load failure after the sidecar patch, but speed remains an
active vMLX optimization target before broad Osaurus release claims.

## Current QAT Text Speed Matrix

Artifacts below use the standard release `RunBench BENCH_PERF=1` method from
this document. These rows prove text load/decode speed only. They do not prove
Osaurus app tool-loop execution, SSD prefix cache TTFT, VL/audio behavior, or
lower-spec physical footprint gates.

| Model | Quant | Deterministic | Bundle Defaults | Artifact |
|---|---|---:|---:|---|
| E2B | MXFP4 | `122.1 tok/s` | `118.6 tok/s` | `/tmp/vmlx-gemma4-qatsidecar-clean-postrevert-e2b-20260612T182246Z/SUMMARY.txt` |
| E2B | JANG_4M | `115.0 tok/s` | `110.3 tok/s` | `/tmp/vmlx-gemma4-qatsidecar-clean-postrevert-e2b-20260612T182246Z/SUMMARY.txt` |
| E4B | MXFP4 | `93.8 tok/s` | `90.4 tok/s` | `/tmp/vmlx-gemma4-upstream-ple-linear-router-e4b-26b-20260612T172809Z/SUMMARY.txt` |
| E4B | JANG_4M | `82.6 tok/s` | `80.4 tok/s` | `/tmp/vmlx-gemma4-upstream-ple-linear-router-e4b-26b-20260612T172809Z/SUMMARY.txt` |
| 12B | MXFP4 | `39.8 tok/s`, `loop=YES` | `37.8 tok/s` | `/tmp/vmlx-gemma4-upstream-ple-linear-router-12b-31b-20260612T172930Z/SUMMARY.txt` |
| 12B | JANG_4M | `32.7 tok/s` | `32.6 tok/s` | `/tmp/vmlx-gemma4-upstream-ple-linear-router-12b-31b-20260612T172930Z/SUMMARY.txt` |
| 26B A4B | MXFP4 | `97.2 tok/s` | `91.9 tok/s`, `unclosedReasoning=YES` | `/tmp/vmlx-gemma4-upstream-ple-linear-router-e4b-26b-20260612T172809Z/SUMMARY.txt` |
| 26B A4B | JANG_4M | `84.2 tok/s` | `81.1 tok/s` | `/tmp/vmlx-gemma4-upstream-ple-linear-router-e4b-26b-20260612T172809Z/SUMMARY.txt` |
| 31B | MXFP4 | `18.4 tok/s` | `18.8 tok/s` | `/tmp/vmlx-gemma4-upstream-ple-linear-router-12b-31b-20260612T172930Z/SUMMARY.txt` |
| 31B | JANG_4M | `17.2 tok/s` | `16.6 tok/s`, `unclosedReasoning=YES` | `/tmp/vmlx-gemma4-upstream-ple-linear-router-12b-31b-20260612T172930Z/SUMMARY.txt` |

Behavior caveats from the same artifacts:

- 12B MXFP4 deterministic is speed-measured but not a clean behavior pass
  because the row reports `loop=YES`.
- 26B A4B MXFP4 bundle-default is speed-measured but not a clean behavior pass
  because it reports `unclosedReasoning=YES` with hidden reasoning output.
- 31B JANG_4M bundle-default is speed-measured but not a clean behavior pass
  because it reports `unclosedReasoning=YES` with hidden reasoning output.

## Current Speed Diagnostics

### Attribution Summary

Fresh artifacts:

- Full standard matrix:
  `/tmp/vmlx-gemma4-qatsidecar-clean-postrevert-e2b-20260612T182246Z`,
  `/tmp/vmlx-gemma4-upstream-ple-linear-router-20260612T172733Z`,
  `/tmp/vmlx-gemma4-upstream-ple-linear-router-e4b-26b-20260612T172809Z`,
  `/tmp/vmlx-gemma4-upstream-ple-linear-router-12b-31b-20260612T172930Z`
- Older PR-base control:
  `/tmp/vmlx-gemma4-qatsidecar-fix-oldbase-e2b-20260612T181842Z`
- Current-main diagnostics:
  `/tmp/vmlx-gemma4-qatsidecar-clean-fused-e2b-20260612T181201Z`,
  `/tmp/vmlx-gemma4-qatsidecar-clean-scaledple-e2b-20260612T181748Z`,
  `/tmp/vmlx-gemma4-current-512-mxfp4-20260612T181941Z`,
  `/tmp/vmlx-gemma4-oldbase-512-mxfp4-20260612T181941Z`
- Graph snapshots:
  `/tmp/vmlx-gemma4-graph-stats-20260612T173342Z`
- Compiled decode diagnostic:
  `/tmp/vmlx-gemma4-compiled-decode-check-20260612T173514Z`
- TurboQuant KV diagnostic:
  `/tmp/vmlx-gemma4-tq-kv-speed-check-20260612T173612Z`

Current conclusion:

- This speed gap is not Osaurus UI/API timing. The numbers above are direct
  `RunBench` release rows against local model folders.
- This is not TurboQuant KV. The standard rows report `kvMode=none`, and the
  TurboQuant KV diagnostic was slower for short single-batch E2B.
- This is not primarily BatchEngine overhead. Current-main E2B MXFP4 batch and
  direct iterator rows both measured `122.1 tok/s`.
- This is not primarily VLM-first factory overhead. Forced LLM text controls
  were same or slower than the VLM inline Gemma4 path.
- This is not solved by only restoring the older custom scaled PLE projection
  path on current main. That A/B measured E2B MXFP4 `120.5 tok/s` and JANG_4M
  `113.2 tok/s`, effectively unchanged.
- This is not solved by the fused Gemma4 RMSNorm/GELU micro-fragments alone on
  current main. That A/B measured E2B MXFP4 `121.2 tok/s` and JANG_4M
  `112.6 tok/s`, effectively unchanged.
- The upstream PLE design is correct source architecture: use normal `Linear`
  for `per_layer_model_projection` and multiply by
  `pow(hidden_size, -0.5)` after projection so loader quantization owns MXFP/JANG
  sidecars. This cleans the model path but does not close the GGUF gap.
- The current-main runtime is materially slower than the older PR-base control:
  E2B MXFP4 `122.1` vs `143.2 tok/s`, and E2B JANG_4M `115.0` vs
  `132.5 tok/s`. The next speed investigation should bisect current-main
  runtime changes around eval/decode/cache/model container behavior, not
  remeasure Osaurus.
- Dense 12B/31B are the bad speed class. E2B/E4B/26B A4B are usable but still
  below the E2B GGUF baseline; 12B/31B dense rows are far slower and need
  focused runtime/kernel work before release promotion.

Representative graph rows:

| Row | tok/s | graphNodes | asType | Notes |
|---|---:|---:|---:|---|
| E2B MXFP4 | `149.2` | `2615` | `285` | Clean short graph row. |
| 26B A4B MXFP4 | `97.9` | `3485` | `272` | Uses fixed router parity path. |
| 12B MXFP4 | `50.8` | `3404` | `242` | Dense/unified text path. |
| 31B MXFP4 | `23.8` | `3710` | `242` | Large dense text path. |

This does not look like simple graph-node explosion. The next speed target is
dense Gemma4 quantized matmul/attention scaling plus compiled decode promotion,
not more Osaurus request-path measurement.

### Compiled batch decode diagnostic

Artifact:

- `/tmp/vmlx-gemma4-compiled-decode-check-20260612T173514Z`

| Row | Standard | Compiled batch | Result |
|---|---:|---:|---|
| E2B MXFP4 | `146.7 tok/s` | `147.8 tok/s` | Noise/small gain. |
| 12B MXFP4 | `39.8 tok/s` | `50.4 tok/s` | Real gain, but still `loop=YES`. |
| 31B MXFP4 | `18.4 tok/s` | `23.4 tok/s` | Real gain, still far below desired parity. |

Compiled decode is a real dense-path lever. It is not sufficient alone and
cannot be enabled as a release fix for 12B until the loop behavior is resolved.

### TurboQuant KV diagnostic

Artifact:

- `/tmp/vmlx-gemma4-tq-kv-speed-check-20260612T173612Z`

| Row | Standard | TurboQuant KV | Result |
|---|---:|---:|---|
| E2B MXFP4 | `146.7 tok/s` | `89.7 tok/s` | Slower, higher footprint. |
| 12B MXFP4 | `39.8 tok/s` | `32.5 tok/s` | Slower, still `loop=YES`. |
| 31B MXFP4 | `18.4 tok/s` | `16.5 tok/s` | Slower, higher footprint. |

TurboQuant KV can still be a RAM/cache policy feature for long context, but
for single-batch short-context decode it is not the GGUF-parity speed fix.
Do not use TurboQuant KV speed rows to claim raw decode improvement unless a
future long-context benchmark proves it.

### VLM text SDPA cleanup

Artifact:

- `/tmp/vmlx-gemma4-vlm-text-sdpa-fix-20260612T165942Z`

Result: not a meaningful speed fix.

| Row | Before | After | Notes |
|---|---:|---:|---|
| E2B MXFP4 deterministic | `144.7 tok/s` | `145.0 tok/s` | Still `graphNodes=2615`, `asType=285`. |
| E2B JANG_4M deterministic | `130.7 tok/s` | `134.2 tok/s` | Small gain/noise; still `graphNodes=2820`, `asType=285`. |
| 26B A4B MXFP4 deterministic | `95.4 tok/s` | `95.2 tok/s` | No gain; still `graphNodes=3635`, `asType=272`. |
| 26B A4B JANG_4M deterministic | `82.4 tok/s` | `81.9 tok/s` | No gain; still `graphNodes=3635`, `asType=272`. |

Interpretation: the VLM inline Gemma4 text path had a stale fp16 attention
upcast that differed from `Gemma4Text.swift`, but current QAT rows do not get
their llama.cpp gap from this path.

### SwitchGLU fused gate-up cap

Artifact:

- `/tmp/vmlx-gemma4-switchglu-fused-cap-20260612T170138Z`

Result: fused gate-up is already engaged by default for 26B A4B. Disabling it
adds about 90 graph nodes and lowers footprint by several GB, but is slower.
Removing the cache cap with `VMLX_FUSED_GATE_UP_CACHE_LIMIT_MB=-1` does not
improve speed.

| Row | Default fused | Fused disabled | Fused unlimited |
|---|---:|---:|---:|
| 26B A4B MXFP4 deterministic | `95.4 tok/s` | `90.9 tok/s` | `93.9 tok/s` |
| 26B A4B JANG_4M deterministic | `82.4 tok/s` | `80.2 tok/s` | `80.9 tok/s` |

Interpretation: the default fused affine `SwitchGLU` path is worth keeping for
speed, but it is not the missing GGUF-parity fix. The remaining gap is still in
runtime/kernel/path work: affine `gatherQuantizedMM`, graph/asType cleanup,
compiled decode eligibility, and cache/runtime policy.

### VLM router parity

Artifact:

- `/tmp/vmlx-gemma4-vlm-router-parity-20260612T171031Z`

The VLM inline Gemma4 router now mirrors the text/Python router contract:
RMSNorm uses the scaled router weight, top-k is selected from raw logits, and
softmax is applied only over the selected top-k logits. The previous VLM path
softmaxed over every expert, then gathered and renormalized.

Measured result:

| Row | Before | After | Graph |
|---|---:|---:|---|
| 26B A4B MXFP4 deterministic | `93.3-95.4 tok/s` | `96.7 tok/s` | `3635 -> 3485` nodes |
| 26B A4B JANG_4M deterministic | `82.4-84.6 tok/s` | `81.9 tok/s` | `3635 -> 3485` nodes |

Keep the router parity fix because it removes incorrect extra work and matches
the model contract. Do not describe it as the full speed fix.

### PLE dense dequantization rejection

Artifact:

- `/tmp/vmlx-gemma4-ple-dense-projection-20260612T171946Z`

Diagnostic: dequantize `per_layer_model_projection` to dense bf16 during
sanitize.

Result: rejected. It was slower on every tested PLE row:

| Row | Standard | Dense PLE diagnostic |
|---|---:|---:|
| E2B MXFP4 | about `144-147 tok/s` | `140.3 tok/s` |
| E2B JANG_4M | about `135 tok/s` | `129.7 tok/s` |
| E4B MXFP4 | about `92-94 tok/s` | `89.2 tok/s` |
| E4B JANG_4M | about `81-83 tok/s` | `78.5 tok/s` |

Do not repeat this as a speed optimization. The correct source architecture is
normal `Linear` plus post-projection scale, leaving quantized sidecars in the
normal loader path.

Loader blockers fixed to produce this matrix:

- E4B PLE `per_layer_projection` shape inference now uses
  `hidden_size_per_layer_input` instead of ambiguous `(bits, group_size)`
  shape guesses.
- E2B/E4B PLE `per_layer_model_projection` now uses a normal `Linear` plus a
  post-projection scale, matching upstream mlx-swift-lm PR #309 and allowing the
  standard quantized loader path to own MXFP/JANG sidecars.
- 12B/31B unified model types route to the existing Gemma4 implementation via
  `gemma4_unified` / `gemma4_unified_text` aliases.
- Unified text speed loads skip unsupported `vision_embedder.*`; unified VL
  remains unproven until that vision namespace is implemented.
- 26B A4B MoE expert tensors split fused
  `experts.gate_up_proj.{weight,scales,biases}` into
  `experts.switch_glu.{gate_proj,up_proj}` and remap
  `experts.down_proj.*` into `experts.switch_glu.down_proj.*`.
