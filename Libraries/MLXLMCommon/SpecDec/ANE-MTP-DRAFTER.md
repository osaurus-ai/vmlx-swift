# ANE MTP drafter — design and measured basis

Status: experimental branch `exp/ane-mtp-draft`. Nothing here is wired into
a shipping path. Every number below was measured on this machine unless it
says otherwise.

## The idea

Native MTP spends each round on D sequential draft-head forwards followed by
one wide trunk verify. The draft chain is tiny (one decoder layer + lm_head)
but latency-bound: dozens of small kernels queued on the same GPU that then
has to run the verify. The Apple Neural Engine is a separate engine with its
own DMA path, a ~90 µs dispatch floor, and no dependency on the Metal queue.
Moving the draft head there does two things:

1. the draft chain runs off the GPU command stream, and
2. it can run *concurrently* with the trunk verify (pipelining), so on the
   accept path the draft cost leaves the round entirely.

"Acceptance" stays a comparison of trunk argmax against the drafted ids; the
ANE's contribution to it is producing the ids (and the head hidden/KV rows)
without a GPU round-trip.

## Measured physics (M5 Max, macOS 26.4, private AppleNeuralEngine API)

Probe: `tools/ane-draft-probe/aneprobe.m` (int8 per-row weights via
`constexpr_affine_dequantize`, fp16 planes, IOSurface I/O).

| fact | number |
|---|---|
| eval floor (1024×1024 int8, 32 rows) | 0.09 ms |
| 5120→5120 int8 @32 rows | 0.25 ms → ~104 GB/s effective |
| 5120→17408 int8 @32 rows | 0.67 ms → ~132 GB/s |
| 5120→5120 @256 rows | 1.13 ms → compute-bound, ~12 TOPS |
| rows 1 / 8 (either layout) | **eval fails** — fp16 plane pitch must be a multiple of 64 B → 32-row tile |
| LUT 4-bit (`constexpr_lut_to_dense`) | compiles; **same time as int8** — expanded at compile time, no bandwidth win |
| blockwise 4-bit (`constexpr_blockwise_shift_scale`) | only per-row scales compile; gs64/gs256 refused by ANECCompile |
| ANE eval with GPU at 52 TFLOPS (GEMM) | 0.673 ms vs 0.678 alone; GPU unchanged |
| ANE eval with GPU blit at 525 GB/s | 0.692 ms (+2%); GPU falls to 439 GB/s (−16%); fabric total 568 GB/s |
| ANE eval loop vs MLX 4-bit decode loop (M=4, 16×8192² gs64, 283 GB/s) | GPU pass 2.137 → 2.217 ms (**+3.8%**); ANE 0.681 → 0.677 ms (unchanged) — `ANEGPUContentionProbeTests` |

So: the ANE streams ~130 G params/s regardless of weight width, is ~9× slower
per parameter than the GPU on 4-bit weights, and is genuinely concurrent. It
is worth using only as *free* capacity that overlaps GPU work, never as a
faster serial engine — except where the GPU step is overhead-bound, which the
draft head is.

## Full head step on the ANE

`tools/ane-draft-probe/mtp_head_mil.py` emits one Qwen3.5-family MTP step
(fc → RMSNorm → q/k/v + head-norm + RoPE → GQA attention over a W-position
KV window (input planes) → o → SwiftGLU MLP → norm → lm_head → per-chunk
argmax) as a Core ML program; `coremlc` renders the canonical `model.mil`,
`tools/ane-draft-probe/computeplan.swift` proves every compute op is ANE-placed, and
the probe times it through the private bridge.

| head | window | draft vocab | params | step |
|---|---|---|---|---|
| Qwen3.8-27B | 1024 | 248320 | 1.67 B | 12.46 ms |
| Qwen3.8-27B | 1024 | 32768 | 561 M | 4.57 ms |
| Flash-Next (MoE ≈ dense top-10 bytes) | 1024 | 248320 | 732 M | 5.94 ms |
| Flash-Next | 1024 | 32768 | 180 M | 1.72 ms |
| Flash-Next | 4096 | 32768 | 180 M | 3.23 ms |
| Ornith-9B | 512 | 32768 | 361 M | 2.81 ms |

Prior GPU evidence for the same Flash-Next head (NativeMTP stats lines,
2026-08-30 … 09-04 campaigns): **2.3–2.9 ms per draft step** dispatch at D3.

Two consequences:

- The lm_head is the cost (77% of the 27B head, >85% of Flash-Next). A
  **pruned draft vocabulary** is the lever: 32k of 248k rows makes the
  Flash-Next step 1.7 ms. A draft that falls outside the pruned vocab is just
  a draft the verifier rejects; it never affects correctness.
- `reduce_argmax` / `cast` are CPU-only under Core ML, so the argmax is done
  arithmetically on the ANE per lm_head chunk (max, then
  `clip(1 + 1024·(x − max))·ramp` → index; indices ≤ 16384 exact in fp16) and
  the host picks the winning chunk. Ties within 1e-3 resolve to the larger
  index — below fp16 logit resolution, so effectively a real tie.

## Architecture

```
GPU (MLX)                          CPU                         ANE (private API)
──────────                         ───                         ─────────────────
verify forward ──hidden[n]──▶ copy fp16 into a_hidden plane
                              embed row (int8 CPU table) ─▶ b_embed plane
                                                             eval step ──▶ (h_out, k_new, v_new, chunk max/idx)
                              pick chunk → draft id; append k/v into window ring
                              loop D times (row 0 of the tile)
◀── drafts as MLXArray ids ── DraftBatch
verify forward (unchanged) ─▶ accept / reject ─▶ aligned commit: ≤ D+1 (trunk-hidden, token) pairs
                                                             ONE tile eval (rows = pairs, causal in-tile mask)
                                                             writes committed k/v rows; window cursor = committed
```

Components (each independently testable):

1. **`ANEBridge`** (C/ObjC target) — the only code that speaks to
   `_ANEInMemoryModel`: create (compile-or-cached load) from MIL text +
   weight blob, IOSurface planes, N inputs / M outputs, per-procedure
   requests, unload. Compile cache under `~/Library/Caches/vmlx/ane/`
   keyed by content hash (MIL + weights). Staging must stay in `$TMPDIR`
   (aned cannot read the home directory). Inputs bind to MIL parameters in
   **alphabetical** name order.
2. **`ANEMILEmitter`** (Swift) — builds the head program text + blob from the
   loaded model's MTP weights: dequantize each Linear to fp16, requantize
   int8 per-row (parity target: cosine ≥ 0.9999 vs the MLX head on the same
   input — measured on synthetic 0.999998–1.0), norm weights folded in, RoPE
   as input planes, draft vocab = first `V_draft` ids (Qwen ids are roughly
   merge-rank ordered) with the id map kept for the verifier.
3. **`ANEHeadWindow`** — the head's K/V as ring planes of W positions with a
   valid mask; cursor = committed length; drafting appends speculative rows
   past the cursor, rejection rewinds the cursor (the `trimHeadChain`
   equivalent), aligned commit writes the confirmed rows.
4. **`ANEMTPDrafter`** — `makeDrafts`-shaped API returning a `DraftBatch`
   from CPU ids; plus `commit(pairs:)` and `prime(prompt:)`.
5. **Iterator seam** — `NativeMTPTokenIterator.makeDrafts` gains a drafter
   choice (`VMLX_ANE_MTP=1` / `.nativeMTP(..., drafter: .ane)`); verify,
   sampling, trunk cache handling untouched. Greedy first; sampled drafting
   (needs draft probabilities) later.

## Pipelining (phase 2)

With the drafter off the GPU, the chain can continue during the verify:
after handing D drafts to the GPU, the ANE keeps stepping from its own hidden
(d_{D+2}, d_{D+3}, …). On full accept the trunk's bonus token is compared to
the head's own prediction at that boundary; when equal, the continued chain
*is* the next round's draft and no draft latency is exposed. The 32-row tile
allows the boundary to be forked: row k continues from the k-th best head
candidate, so a bonus token inside the head's top-32 still has a ready chain.
Measure before building: the gain is bounded by the draft fraction of the
round (~15–20% on Flash-Next at D3) minus the GPU bandwidth tax while both
engines stream (≤16% of GPU BW during the ANE's active window).

## Prior art reviewed (2026-09-10)

**ANEMLL** (`Anemll/Anemll`, 1.6k★; issues #33/#35/#58, PR #50): whole small
models (1–8B) on the ANE via Core ML — LUT4 FFN + LUT6 lm_head, static
context 512–1024, chunked lm_head, `MLState` KV, one function per shape.
Their own numbers say it: ANE sits at 10–20% utilization at decode because
it is memory-bound (#35), and an M1 4B does 11 t/s on ANE vs 23 t/s MLX
(#33). No speculative decoding anywhere in the repo. The org has since
moved big-MoE work to Metal (`ds4-qwen`, `ds4-ssd`, `anemll-flash-llama.cpp`).
Useful pieces: `fp16_preflight` (residual-stream overflow check — Gemma3
needs α-scaling, Qwen3 peaks ~15k of 65k), `ane_profiler.py` (compute-plan
+ per-op placement, same approach as `computeplan.swift`), and the Qwen3.5
DeltaNet port's ANE constraints (per-layer `MLState`, no batch prefill for
the recurrence, transposed state layout, fp32 recurrence step — meaning a
Flash-Next/GDN *trunk* on the ANE is a non-starter; the MTP head we target
is attention + MLP only).

**Core AI** (`apple/coreai-models`, `coreai-torch`, `coreai-optimization`;
macOS 27 / Xcode 27): the public successor to the private path —
`SpecializationOptions(preferredComputeUnitKind: .neuralEngine)`, I/O bound
to `MTLBuffer`/`IOSurface` via `NDArray.MutableRawView`, "chunked static →
Neural Engine" LLM engines, and a GPU-pipelined engine
(`CoreAIPipelinedEngine`: non-blocking encode, GPU-side sampling, depth-3
buffer rotation). Apple's Neural Engine authoring rules match this note's
measurements exactly (64-byte last-axis alignment, BC1S layout, conv-as-
linear, fp16 only, static shapes, "keep the whole graph resident").
Not available on Eric's Macs (26.3.2 / 26.4) today; the bridge is written so
the emitter and drafter can retarget Core AI when the OS moves.

**Hybrid ANE+GPU trunk decode ("fused decode")**: the fabric numbers rule it
out for bandwidth-bound decode of 4-bit weights. The GPU reads 0.5 B/param;
the ANE must read ≥1 B/param (4-bit expands at compile). On a shared fabric
that delivered 568 GB/s combined in the blit test, moving any slice of the
trunk to the ANE lowers total params/s (GPU alone ≈ 1.05 T params/s; GPU +
ANE ≈ 0.88 T + 0.13 T = 1.01 T at best, before the ANE's 32-row tile
computes 32× the needed rows). mlx-serve reached the same place empirically:
ANE offload helps *prefill* (compute-bound) on M1–M4 and is off on M5. What
does compose with the GPU is overhead-bound work — the draft head, samplers,
small controllers — which is this design.

## What this is not

- Not a decode engine: fp16-only, ~130 G params/s, static shapes.
- Not an exact drafter at a pruned vocab — it is a drafter; the verifier is
  the truth. Acceptance is the metric, measured against the GPU head on the
  same prompt.
- Not Flash-Next-complete yet: the MoE MTP layer needs router + top-k on the
  CPU and either a per-expert procedure bank (≈ 10 evals × (90 µs + 40 µs))
  or CPU/AMX experts (49 MB int8 per step). Dense heads (Qwen3.5/3.8 27B,
  Ornith-9B) are the first target.

## Verification plan

1. Bridge unit test: compile + eval a 1024² int8 linear, parity vs CPU.
2. Emitter parity: real MTP weights → ANE head vs `mtp.preNormHidden` +
   `projectToLogits` on the same (hidden, token); cosine on `h_out`, top-1
   agreement over ≥ 256 positions of a real prompt.
3. Acceptance A/B on one bundle, fixed prompt set, greedy: GPU head vs ANE
   head at V_draft ∈ {32k, 64k, full}; report accepted/drafted per depth.
4. Wall: tok/s at fixed D, same prompt, ≥ 3 runs median, both arms, on the
   runtime machine; then phase 2.
