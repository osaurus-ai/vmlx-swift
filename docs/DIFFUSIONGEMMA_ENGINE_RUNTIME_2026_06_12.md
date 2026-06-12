# DiffusionGemma Block-Diffusion Runtime — 2026-06-12

Native `diffusion_gemma` generation engine for
`google/diffusiongemma-26B-A4B-it` (30-layer Gemma4-style MoE, 128 experts
top-8, 26B total / ~4B active). Companion to the quant-prep lane in
`docs/DIFFUSIONGEMMA_ENGINE_QUANT_PREP_2026_06_12.md` (PR #45).

## Engine

DiffusionGemma is NOT autoregressive. Generation runs an outer
block-autoregressive loop over fixed 256-token canvases; each canvas is
produced by an inner denoising loop:

1. **Encoder forward** (causal, cache-writing) over committed tokens —
   the prompt on prefill, then each finalized canvas.
2. Random canvas init (uniform over vocab), self-conditioning reset.
3. **Denoising steps** (`max_denoising_steps` 48, counts down):
   decoder forward (bidirectional over the canvas, reading — never
   writing — the encoder cache, with self-conditioning soft embeddings),
   linear temperature schedule (t 0.8 → 0.4), categorical denoiser
   sampling, entropy-bound acceptance (entropy_bound 0.1,
   arXiv 2505.24857), renoise of rejected positions, stable+confident
   adaptive stopping (stability 1, confidence 0.005).
4. Finalize the argmax canvas; truncate the emitted stream after EOS
   ([1, 106, 50]).

All diffusion sampling parameters come from the bundle's
`generation_config.json`. User `maxTokens` only caps output length —
no synthetic temperature/top-p overrides anywhere in the lane.

Components:

- `Libraries/MLXLLM/Models/DiffusionGemma.swift` — model family
  (dual-mode attention, parallel dense-MLP+MoE layers, self-conditioning,
  encoder layer scalars, sanitize for the MXFP bundles).
- `Libraries/MLXLMCommon/Diffusion/BlockDiffusion.swift` — protocol +
  sampler primitives (unit-tested against the HF reference math).
- `Libraries/MLXLMCommon/Diffusion/BlockDiffusionTokenIterator.swift` —
  the loop, prefix/disk cache integration, instrumentation.
- Dispatch: `generate()`, `ChatSession`, and `BatchEngine`
  (exclusive solo path, same mechanism as native MTP) route on
  `model is BlockDiffusionModel`. The model's `prepare()` throws, so a
  silent AR route is impossible — `Tests/MLXLMTests/DiffusionGemmaTests`
  asserts this (15 tests).

## Cache topology

- Encoder cache: `RotatingKVCache(window=1024, keep=0)` on the 25
  sliding layers, `KVCacheSimple` on the 5 full-attention layers —
  identical to Gemma4. KV is plain fp16; TurboQuant KV is not enabled
  for this lane (rotating-cache TQ does not exist upstream for any
  family yet).
- Decoder reads the encoder cache via new read-only accessors
  (`RotatingKVCache.temporallyOrderedKV()`, `KVCacheSimple.readKV()`);
  decoder forwards never mutate cache offsets (test-asserted).
- Prefix cache: rotating layers cannot round-trip through paged KV
  blocks, so the iterator marks the coordinator paged-incompatible and
  prompt boundaries are served by the disk (L2/SSD) tier — the same
  contract the AR iterators use for Gemma4.
- End-of-turn contract: the emitted reply (minus trailing EOS) is
  committed to the encoder cache, so multi-turn live-cache sessions
  mirror AR cache state exactly.

## Verification rows (M5 Max, release build, 2026-06-12)

### Multi-turn coherency (BENCH_COHERENT, ChatSession, 3 turns × 2 loops)

MXFP4 (15 GB bundle, load 1.26 s):

| Turn | Reply | Steps | Decode tok/s | TTFT |
|---|---|---|---|---|
| 1 "My favorite color is blue." | "Blue is a very popular and calming color!" | 8 | 4.3 | 3.74 s (cold) |
| 2 "What is my favorite color?" | "Your favorite color is blue." | 3 | 13.1 | 0.60 s |
| 3 "Is that a warm or cool color?" | "Blue is considered a cool color." | 3 | 14.8 | 0.61 s |

MXFP8 (26 GB bundle, load 1.29 s): same answers; steps 2–6; decode
7.4–16.2 tok/s; warm TTFT 0.52–0.77 s.

Steady decode rate: ~5.6 denoising forwards/s (MXFP4), ~4.5 (MXFP8).
Emitted tok/s = forwards/s × tokens-accepted-per-forward — short replies
under-fill the 256-token canvas, so the long-form row below is the
representative throughput number.

### Long-form throughput (BENCH_SOLO_LONGFORM, ~740-token essay)

| | MXFP4 | MXFP8 |
|---|---|---|
| Wall tok/s | 36.8 | 42.3 |
| Tokens per forward | 7.49 | 10.94 |
| Denoise steps per canvas | 32.7 | 22.7 |
| Canvases | 3 | 3 |
| TTFT (first full canvas) | 5.9 s | 5.8 s |

MXFP8 is FASTER than MXFP4 long-form despite slower per-forward compute:
higher-precision logits converge in fewer denoising steps (68 vs 98
forwards for the same essay). Output was a coherent, factually grounded
multi-paragraph essay on both quants.

Per-step profile (instrumented): the 256-wide decoder forward is 91% of
denoise time (178 ms/forward MXFP4, 254 ms MXFP8); the sampler pipeline
(softmax/entropy/sort over the 262k vocab) is 9%. The engine is
forward-bound — throughput is governed by the model's convergence rate.

### Speed/quality control (`GenerateParameters.diffusionMaxDenoisingSteps`)

Per-request override of the bundle's denoising budget — the hook for an
app-level Quality ⟷ Speed slider. Swept on MXFP4, same essay prompt:

| max_denoising_steps | Wall tok/s | TTFT | Coherency |
|---|---|---|---|
| 48 (bundle default) | 36.9 | 6.7 s | clean |
| 24 | 57.9 | 4.2 s | clean |
| 16 | 73.8 | 3.4 s | clean (essay + 3-turn recall verified) |
| 8 | 140.5 | 1.8 s | BREAKS (word-salad spans) |

Raising `entropy_bound` (0.1 → 0.4 at 48 steps) does not help: 37.9
tok/s — the step budget is the lever. Recommended app slider range
16–48, default 48 (quality). The override is clamped to ≥ 1 and ignored
by autoregressive models.

### Tool calls (BENCH_BATCH_TOOLCALL, BatchEngine solo path)

Both quants: structured `get_weather({"location":"Tokyo"})` extracted
via the Gemma4 parser (`<|tool_call>call:name{...}<tool_call|>`), zero
raw markers leaked to `.chunk`. MXFP4 25.6 decode tok/s (4 steps),
MXFP8 40.5 decode tok/s (2 steps). Reasoning stamp resolves to
`harmony` (`<|channel>thought` envelope; template-verified).

### Prefix + disk (SSD) cache

- `BENCH_SOLO_DISK_RESTORE` (new row, `BatchEngine.generate` solo path):
  session 1 stores the 138-token prompt boundary; a FRESH coordinator on
  the same disk dir hits the disk tier (`matched=138/138`); session 2
  restores fully — `prefixCacheRestoredTokens=138, prefillSec=0.000,
  encoderForwards=0` — and answers identically.
- `BENCH_BATCH_CACHE_HIT` passes (turn-2 extension restored from the
  stored boundary).
- Warm ChatSession turns prefill in 0.07–0.13 s vs 0.7–1.4 s cold.

### Memory

Peak process RSS during the 3-turn coherent run (release build):
MXFP4 12.7 GB, MXFP8 23.8 GB.

### Isolation

All shared-surface changes are additive (read-only cache accessors,
optional `GenerationConfigFile` fields, protocol-gated dispatch
branches, one registry entry). `DiffusionGemmaTests` 15/15 green; the
cross-suite parallel-Metal crash seen when running many suites in one
process reproduces identically on clean main @ 710eb0d7 (pre-existing
test-infra flake, not introduced by this lane).

## Boundaries

- **Text only.** The vision tower ships in the bundles (fp16) but is
  not wired; image/VL is not claimed. Video has no `video_token_id`;
  audio is absent from the bundle config.
- Block diffusion runs as an exclusive solo path in `BatchEngine`;
  batched multi-slot canvas scheduling is future work.
- TurboQuant KV and compiled-decode fast paths are not applicable yet.
