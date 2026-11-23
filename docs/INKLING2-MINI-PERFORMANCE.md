# Inkling-2-Mini (`inkling2_loop`) decode performance

How the Swift runtime decodes Inkling-2-Mini, where the time per token goes, what was fused, how
the bundled dSpark drafter is driven, and what remains. Numbers are M5 Max (40-core GPU) with the
published checkpoint (`thinkingmachines/Inkling-2-Mini`: BF16 dense + unembed, NVFP4 experts on
layers 0-19, MXFP8 experts on layers 20-23), greedy unless stated.

## 1. What one token costs

The text model has 24 layers; layers 6-17 run three passes (the loop), so a token executes
**48 layer visits**, each with its own KV cache and short-conv state (48 + 48 cache slots).

| per token | bytes | notes |
|---|---:|---|
| dense attention-input / output projections, 48 visits | ~2.6 GB | BF16, re-read every pass (54 MB per layer does not stay in cache) |
| routed experts, top-8 of 128, 48 visits | ~1.7 GB | NVFP4 (MXFP8 on 4 layers) |
| unembed 201024 x 2304 | 0.93 GB | BF16 |
| loop message gates, router, KV reads | < 0.3 GB | |

So a token streams ~5.5 GB of weights. A dependent chain of MLX batch-1 GEMVs with the same
shapes runs at ~500 GB/s on the large projections but much lower on the small ones (measured
per op, distinct weights per visit so nothing stays cache resident):

| op (per visit) | time | effective |
|---|---:|---:|
| `[q\|r\|gate\|sink\|k\|v]` 8600 x 2304 BF16 | 79 us | 499 GB/s |
| `k` or `v` alone 1056 x 2304 BF16 | 18 us | 270 GB/s |
| `o_proj` 2304 x 3072 BF16 | 33 us | 429 GB/s |
| router 128 x 2304 fp32 | 11 us | 107 GB/s |
| experts top-8 gate/up + down, NVFP4 | 101 us | 356 GB/s |
| unembed (once per token) | 1.79 ms | 517 GB/s |

Summed over a token this is ~13 ms of weight streaming before any of the small kernels (depth
attention, short convs, clips, attention, residual, router selection), each of which costs a
few microseconds of launch and barrier on top of its work. **Exact BF16 decode on an M5 Max is
therefore bounded at roughly 60-65 tok/s; the rest of the speed has to come from speculation or
from fewer weight bytes (a quantized dense / unembed), which changes the model's numbers.**

## 2. The single-token path (`Inkling2Decode.swift`)

The general forward (also the prefill path and the reference) is ~110 ops per layer visit. The
decode path is a handful of GEMVs / gathers joined by fused Metal kernels:

| kernel | replaces |
|---|---|
| `inkling2_dec_depth` | block attention-residual over the depth stack **+ the layer's input RMSNorm(s)** (1 or 3 queries) |
| `inkling2_gate_down` | both unweighted RMSNorms of a loop message gate's input, the rounding to the weight dtype, and the rank-64 down projection |
| `inkling2_gate_mix` | forget / gain projections (fp32 arithmetic on the BF16 weights, 4 columns per thread) + the `tanh` mix |
| `inkling2_gate_norm` | the gated row's weighted RMSNorm (q / k / v input norms or the post-attention norm) in the model dtype + its fp32 copy |
| `inkling2_dec_qkv` | k / v short convs (static or dynamic), q / r / k clips, v norm, log scaling, the relative-position bias row (split over 128-key slices per head) and the conv-state update |
| `inkling2_dec_mid` | post-attention conv + residual + the MLP's depth attention **+ post-attention RMSNorm** |
| `inkling2_moe_gate_up_act` | NVFP4 routed gate/up for the 8 picks (a gate row and its interleaved up row in one simdgroup, one 16-value scale group per lane step) **+ the softcapped SiLU-gate activation** with the per-expert scale |
| `inkling2_moe_down` | NVFP4 routed down projection (8 lanes per output row, so the 1152-wide rows split evenly) |
| `inkling2_dec_final` | expert combine + post-MLP conv + residual + the packed conv-state write |

Attention itself is MLX's fused SDPA with the per-token sink; the gate on its output is one small
kernel. Every load-time constant a kernel takes (norm weights in fp32, depth queries, offsets,
clip thresholds) is built once per layer, and the per-token scalars (visible keys, log-scale
factor) are one array per token shared by all 48 visits instead of new arrays per call.

### Fused loop message gates

In loop passes 2 and 3 every loop layer gates its q, k, v and MLP inputs with the previous pass's
messages: `y = (1 - tanh(W_f d)) * h + tanh(W_g d) * m` with `d = down([rms(h), rms(m)])`, then the
layer norm. Previously that was five dispatches per gate group (input-norm kernel, batched down
GEMV, cast, fp32 GEMV against an fp32 copy of the forget/gain weights, output kernel). Now it is
three (`gate_down`, `gate_mix`, `gate_norm`), the forget / gain weights are read in their BF16
storage (half the bytes; bf16 -> fp32 is exact, so the arithmetic is unchanged), and the reads are
vectorized. Rounding points follow the reference: the gate input is rounded to the down weight's
dtype, the down output to that dtype, the mix stays fp32, the norm output is rounded once.
`Tests/MLXLMTests/Inkling2FusedKernelTests.swift` checks all three against the reference written
with plain MLX ops (decode, verify-row and MLP-gate shapes).

A first version computed the whole forget/gain projection plus the norm in one threadgroup per
row; it was 1.2 ms per token *slower* (each row's 590 KB of gate weights streamed through a single
GPU core). Splitting the column-parallel mix from the row norm fixed that.

### Routed experts

MLX's `gatherQuantizedMM` for one row against 8 NVFP4 experts ran at ~400 GB/s (gate/up) and
~285 GB/s (down) where a dense NVFP4 GEMV of the same bytes reaches ~525 GB/s. The batch-1 kernels
above read the checkpoint's packed codes and e4m3 block scales directly (16-entry e2m1 table in
threadgroup memory, vectorized x loads) and match MLX's gather to ~1e-7 relative:

| per visit | MLX gather | fused kernels |
|---|---:|---:|
| gate/up (+ activation) | 59 us (+ activation kernel) | 51 us |
| down | 42 us | 33 us |

The MXFP8 layers (20-23) keep MLX's gather. `VMLX_INKLING2_MLX_EXPERTS=1` forces MLX's gather
everywhere (A/B switch).

### Measured (bench, pipelined like `TokenIterator`)

| | 64-token context | 2k context | 8k context |
|---|---:|---:|---:|
| before | 17.7 ms (56 tok/s) | 18.9 ms (53 tok/s) | 19.9 ms (50 tok/s) |
| fused gates + bias split | 16.0 ms (62.6 tok/s) | 16.2-16.8 ms (60 tok/s) | 17.1 ms (58.4 tok/s) |
| + NVFP4 expert kernels | 15.3 ms (65.5 tok/s) | 15.6-16.0 ms (62.4 tok/s) | 16.5 ms (60.5 tok/s) |

At 2k context the calling thread spends ~2.2 ms building the graph and ~5 ms encoding per token;
the GPU takes ~15.6 ms, so on a performance core the step is GPU bound. Context dependence dropped
from +1.2 ms at 2k to +0.5 ms once the relative-bias rows were spread over the GPU.

Parity against the reference implementation stays within its run-to-run band: short prompt
teacher-forced top-1 31/31 (KL median ~1e-6), 2.3k-token prompt 30-31/31 (KL median ~8.5e-4
against a same-run chunking control of ~2.5e-3), fast-vs-general path 0 argmax mismatches.

Tried and dropped (measured slower or not worth it): a router GEMV kernel on the BF16 router
weight (MLX's fp32 GEMV is faster), one M = 3 matmul for the three attention-input rows (-0.1 ms
for ~1 GB of duplicated weights), a single-threadgroup-per-row gate kernel (+1.2 ms), and MLX
command-buffer sizing (`MLX_MAX_OPS_PER_BUFFER`: no effect).

## 3. The multi-row step (`Inkling2Verify.swift`)

Speculative verify and short chunks (2-16 rows) take a fused multi-row path with the same kernels
generalized to T rows (short convs walk the rows in order inside the kernels; the per-row sink is
folded into the additive mask). Verify cost at 2k context:

| rows | before | now |
|---:|---:|---:|
| 1 | 21.9 ms | 18.9 ms |
| 4 | 35.9 ms | 30.6 ms |
| 8 | 50.2 ms | 42.0 ms |

(synchronous, including the logits of every row.) Each extra row costs ~2.9 ms, almost exactly
the expert bytes of one more token: each row routes to its own top-8 of 128 and adjacent rows
rarely share experts. Dense weights are read once for all rows.

## 4. dSpark (bundled drafter)

The bundle ships a drafter (`mtp.safetensors`): three DFlash-style layers over captured target
features plus a Markov `rnn_hidden` head that walks the block. It runs inside the generic
`TokenIterator` dSpark runtime:

* **Context capture.** Every target forward over a dSpark cache captures four layer-visit inputs,
  appends the drafter's context KV for those rows and keeps the last row's features; the draft
  state lives in the request's cache array, so plain decode steps, verify commits and prefix-cache
  restores keep it in sync.
* **Draft.** One block forward of the drafter, one GEMM of the block's rows against the unembed,
  then the Markov walk (per step: gated state update, a rank-R vocabulary bias, softcap, argmax or
  sampled with the request's noise) entirely on the GPU, one evaluation per round.
* **Verify + commit.** The target verifies all drafted rows in one multi-row forward with conv-input
  recording; commit trims KV rows and rewinds conv states to the accepted prefix in place.
* **Adaptive governor.** The shared runtime measures milliseconds per emitted token for speculative
  rounds and for AR steps and keeps an acceptance EMA; it chooses AR vs speculation and the verify
  width from those measurements. There is no fixed draft depth: a bundle's block size only bounds
  the drafter's own block.

Cost model on prose (acceptance ~0.4 per drafted token): a round is ~4.5 ms draft + ~30 ms verify
of ~4 rows and emits ~2.4 tokens, about the same as 2.4 AR steps (15.4 ms each), so the governor
runs mostly AR and dSpark lands at parity (bench: 66.2 vs 65.8 tok/s greedy, 63.7 vs 65.3
sampled). Because every verified row pays its own experts, the break-even acceptance on this MoE
is high; on predictable text (code edits, quoting, structured output, tool arguments) acceptance
is 0.65-0.8 and speculation is a clear win (in-app: +52% on a copy task).

Draft cost per round: the drafter's three dense layers (~470 MB of weights, ~1 ms), one GEMM of
the block's rows against the unembed (0.93 GB, ~1.8 ms) and the Markov walk, whose per-step
vocabulary-wide fold projection (201024 x 256) reads ~100 MB per drafted token.

## 5. Tool calls and caching

A tool-call turn stores a post-answer prefix-cache entry at prompt + generated + the stop token, so
the tool-result continuation restores at exactly that boundary and only prefills the tool result.
(Changing the reasoning-effort level re-prefills: the template renders the effort line before the
history.)

The 40 sliding-window visits (layers other than 5, 11, 17, 23; the loop layers once per pass) keep only their window:
`TemporalWindowKVCache` holds the last `window - 1 + 64` rows in temporal order under a logical position, and stored
prefix entries carry exactly those rows, so a stored entry grows by 32 KiB per token instead of 192 KiB (plus a fixed
~170 MiB of windows). M5 Ultra, JANGH2, app-route agentic cache loop at 7.7k-11.2k tokens against the full-history
control (`VMLX_INKLING_WINDOWED_KV=0`): identical greedy text and reuse, SSD 4.21 vs 15.47 GB, largest entry 532 MB vs
2.10 GB, peak MLX memory 12.4 vs 16.0 GiB. In a dev-built Osaurus on an M5 Max, four chat turns at ~2-4.6k tokens left
1.8 GB of entries (240-358 MB each) where the full-history runtime had left 6.2 GB (350-830 MB each).

## 6. Running inside an app

`VMLX_DECODE_CPU_TRACE=1` logs, every 128 decode steps, the wall time and the decode thread's CPU
time per step plus its QoS class. In the bench a step costs ~7.4 ms of CPU on a performance core
against ~15.6 ms of GPU (GPU bound). Forced onto efficiency cores (`taskpolicy -b`) the same step
costs ~32 ms of CPU and decode becomes CPU bound at ~32 ms per token, so where a host schedules
the decode thread matters as much as the kernels.

Measured in a dev-built Osaurus on the same M5 Max (published checkpoint, the app's own chat
template and system prompt, ~2.5k-token prompt with tools, sampling at the bundle's temperature
1.0, dSpark on):

* Prose answers (~1.2-1.6k tokens): 44-48 tok/s before these changes, **50-53 tok/s** after.
* The app's decode thread runs at user-initiated QoS and is not CPU bound (CPU 9-12 ms per step
  against 18-19 ms wall), but it spends 30-60% more CPU per step than the same engine route in a
  bench, and its steps are ~2 ms slower (18.3 vs 16.1-16.5 ms per AR step at the same context).
  Ruled out with controlled A/B runs: the batch-engine / prefix-cache restore route (the bench
  reproduces it at bench speed), the app being frontmost or its window visible, other generations
  overlapping the measured answer, one busy CPU core elsewhere, and sampling parameters. The rest
  of that gap is not yet explained.
* **Sustained load throttles this laptop hard.** After ~30 minutes of back-to-back generation the
  same requests ran at half speed (Inkling 50 -> 24 tok/s; the on-device foundation model's own
  background jobs 45 -> 15 tok/s) with no thermal warning reported by `pmset`; four idle minutes
  restore full speed. Any speed comparison has to interleave its arms in the same thermal state.

## 7. Not done yet## 7. Not done yet

* The NVFP4 down kernel is at ~360 GB/s and `o_proj` at ~430 GB/s; both have some headroom.
* A pruned draft vocabulary for the Markov walk (the walk's full-vocabulary work is most of the
  ~5 ms draft).
* Quantized dense / unembed weights: 8-bit dense measured ~2.6 ms per token less GPU time
  (attention-input GEMV 79 -> 50 us, unembed 1.79 -> 1.04 ms); a model-quality trade-off, not done.

## 8. JANGH bundles (`OsaurusAI/Inkling-2-Mini-JANGH2`)

The JANGH2 bundle keeps the model's control path as released and changes three weight families: routed experts
are JANGH codebook banks (2 / 3 bits, Hadamard-32 rows, fp16 row scales; 3-bit on the looped layers' down
projections and six gate/up stacks), the attention-input / output projections, embeddings and unembed are affine
8-bit (group 64), and the sinks, message gates, convs, router and norms stay as released. A token streams ~2.95 GB
instead of ~5.5 GB.

### What runs

| piece | path |
|---|---|
| routed experts, decode | `inkling2_jh_gate_up_act`: the x row rotated to H32 in threadgroup memory, the router's top-k selection (same arithmetic and tie order as `routerSelect`), gate and up codebook dots per row, `softcap(g) * sigmoid(g) * softcap(u)`, bf16 round, H32 output rotation; `inkling2_jh_down`: per-pick down rows. The final combine applies the route weights. |
| routed experts, prefill / long verify | `JANGHRoutedExpertLayer` with `JANGTActivation.inkling` (generic JANGH decode kernels and expert-sorted NAX tiles) |
| q / r / gate / k / v | one affine GEMV over `[q|r|gate|k|v]` on one-row visits; per-row GEMVs on three-row (full / message-gated) visits; the plain sink and dynamic-tap projections as transposed views |
| o_proj, embed, unembed | MLX affine kernels |

### Measured (M5 Ultra, 64-token context unless stated)

| step | pipelined decode |
|---|---:|
| generic JANGH decode route | 74.9 tok/s |
| + batch-1 JANGH expert kernels (per-pick down) | 92.6 |
| + no-op Reshape / per-call Transpose removed from the step (2537 -> 1543 graph nodes) | 96.2 |
| + router selection inside the gate/up kernel, one GEMV for one-row visits | 105.0 |
| (native checkpoint on the same machine) | 81.0 |

App route (`TokenIterator` + batch engine + prefix cache, 2.3k-token prompt, M5 Ultra): 90 tok/s before the node
reductions (10.9 ms wall / 6.4 ms decode-thread CPU per step), 98.9 tok/s after (9.97 ms wall / 6.1 ms CPU). The
decode thread's CPU (~4.6 us per graph node) is now ~60% of the step: below ~6 ms of GPU per token it becomes the
limit, so further speed needs fewer graph nodes as much as faster kernels. dSpark at 2.7k context: verify 1 / 4 / 8
rows = 11.1 / 20.4 / 26.7 ms; greedy code 129.5 vs 102.4 tok/s AR, prose at parity (the governor stays AR).

In a dev-built Osaurus on an M5 Max (the app's own prompt and tools, the bundle's sampling, driven through the GUI):
prose answers 71.9-72.7 tok/s, code turns 66.4-67.6 tok/s, tool-result continuations at 3-7k context 58.5-68.0 tok/s.

### Per-token budget (M5 Ultra, dependent chains over the 48 layer visits)

| component | ms / token |
|---|---:|
| attention-input GEMVs (q/r/gate/k/v + sink) | ~1.6 |
| o_proj | 0.45 |
| router GEMV | 0.57 |
| routed experts (gate/up 1.1 + down 0.6) | 1.75 |
| message gates (24 visits x q/k/v + mlp) | 1.3 |
| depth / qkv / KV append + SDPA (1k window) / sigmoid / mid / final | 0.3 / 0.3 / 0.85 / 0.1 / 0.6 / 0.36 |
| unembed | 0.7 |

Bandwidth ceilings measured on the same machine: streaming read 1.1 TB/s, BF16 GEMV 850 GB/s, MLX affine 8-bit
GEMV 700 GB/s. The weight bytes alone cost ~3.5-4 ms per token; the rest is the dependent chain of 10-12 kernels
per layer visit, each of which now runs within 1.3-2x of its own bandwidth or latency floor.

Tried and dropped: a split-key decode attention with the mask, sink and gate folded into the combine (slower than
MLX's fused SDPA at 64 and 2k context), padding the rotated x in threadgroup memory against bank conflicts and a
fused gate+up dot loop (no change).

Also measured and dropped (M5 Ultra, 3 interleaved runs, bit-identical outputs): computing the gate inputs' RMS scales
once (in the depth / mid kernels and once per pass for the message rows) so the gate down projection becomes a wider
plain GEMV (102.8-106.1 tok/s against 107.1: the extra reduction in the single-threadgroup kernels costs more than
the per-group norm it removes), and a router GEMV with one output per threadgroup reading the BF16 weight in MLX's
exact accumulation order (106.4 against 107.1).
