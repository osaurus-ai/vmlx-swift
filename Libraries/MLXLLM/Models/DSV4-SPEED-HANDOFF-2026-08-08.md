# DSV4 speed handoff — 2026-08-08

This is the authoritative continuation point for the DSV4-Flash speed work in
`/Users/eric/vmlx-swift` and the consuming Osaurus checkout at
`/Users/eric/osaurus-main-live`.

## Stop state

- The user explicitly stopped implementation and requested a Claude handoff.
- No RunBench, Swift build, xcodebuild, or Osaurus process was left running.
- Do not resume a 1K-token, 500K-context, GUI, or full correctness campaign
  yet. First make short decode clear the mandatory speed floor.
- The last retained code point includes deferred pool advancement and fused
  HC-post. A compiled HC-pre region was subsequently built and measured, but
  it regressed decode by about 50% and was removed.
- A replacement explicit decode-only HC-pre Metal kernel was then added. Its
  release build was interrupted at the user's handoff request and exited 130;
  it is unbuilt, unbenchmarked, and must not be treated as a speed win.

## Hard requirements

1. **33 tok/s is the mandatory floor; 35+ tok/s remains the target.** Measure
   decode excluding TTFT in 128-token windows. Short generation, long
   generation, the final window, and long-context decode must all hold the
   floor. An aggregate average cannot hide a late collapse.
2. Keep prompt processing and decode separate. Target about **400 pp/s** for
   prefill, but do not spend time on long prefill campaigns until short decode
   reaches at least 33 tok/s.
3. Eventually prove near-500K context with quantized pooled cache, about 6-8 GB
   effective pool payload, measured `phys_footprint`, cache topology/counters,
   and per-window decode. This is deferred, not waived.
4. Time final-token-to-stream-completion separately. The reported end-of-output
   hang is still open.
5. Preserve model-native generation config and DSML/tool rendering. No forced
   sampler, reasoning-tag, stop-token, or prompt-template behavior is allowed
   to make a speed/coherence row look good.
6. SSD cache proof must cover stable prompt boundaries, prefix/suffix reuse,
   real disk hits, and the DSV4 companion/pool state. Do not call an entry a
   full cache hit when telemetry says `effectiveKVLayers=0`, `blocks=0`, and
   `payload=false`.

## Repository truth

### vmlx-swift

- Checkout: `/Users/eric/vmlx-swift`
- Branch: `work/dsv4-fixes`
- HEAD: `e56571e4b295276b7b9e8310ca6e5e70910e8949`
- Remote branch: `vmlx-origin/work/dsv4-fixes` at the same SHA
- Base: `vmlx-origin/main` at
  `5052be1ad6fdecdbc0da111abf8db5744894d17a`
- Branch relation: ahead 2, behind 0
- Existing commits:
  - `096b44ef` — fused qmm lm_head, shared RoPE tables, Metal live-buffer guard
  - `e56571e4` — loader-limit/memory-plan follow-up and DSV4 fastpath pin

Dirty files at handoff:

```text
 M Libraries/MLXLLM/Models/DSV4-SPEED-CAMPAIGN.md
 M Libraries/MLXLLM/Models/DeepseekV4.swift
 M Libraries/MLXLLM/Models/DeepseekV4Compressor.swift
 M Libraries/MLXLLM/Models/DeepseekV4MathHelpers.swift
 M Libraries/MLXLMCommon/Cache/TQDiskSerializer.swift
 M RunBench/Bench.swift
 M Source/Cmlx/mlx
 M Tests/MLXLMTests/DeepseekV4PoolQuantizationTests.swift
?? Libraries/MLXLLM/Models/DSV4-SPEED-HANDOFF-2026-08-08.md
```

`Source/Cmlx/mlx` is pre-existing user-owned dirt. It points from
`a828cb47...` to local `9dabb6c4...-dirty`. Do not restore it, stage it, or
include it in this change.

### Osaurus

- Checkout: `/Users/eric/osaurus-main-live`
- Branch: `main`
- HEAD/upstream: `f6384c3ced5885f35ef55a23acfe31986f2e8a4c`
  (`osaurus/main`, ahead 0, behind 0)
- Six dirty repin files currently point to `e56571e4`:
  - `Packages/OsaurusCore/Package.swift`
  - `Packages/OsaurusCore/Package.resolved`
  - `osaurus.xcworkspace/xcshareddata/swiftpm/Package.resolved`
  - `App/osaurus.xcodeproj/project.xcworkspace/xcshareddata/swiftpm/Package.resolved`
  - `Packages/OsaurusCore/Tests/Service/RuntimePolicySourceTests.swift`
  - `Packages/OsaurusCore/Tests/Service/ImageGenerationBridgeContractTests.swift`

That pin is now stale for this speed lane because the new work is uncommitted.
Do not open the final Osaurus PR with `e56571e4`.

## What was changed in the dirty vmlx-swift tree

### 1. Honest decode-window instrumentation — built and used

`RunBench/Bench.swift` accepts `BENCH_DECODE_WINDOW_TOKENS` and prints the
rate and total context for each decode window. This exposed the late-answer
collapse that the aggregate number hid.

### 2. Long-context pool storage — product built, focused tests blocked

`DeepseekV4Compressor.swift` and `TQDiskSerializer.swift` now:

- keep a production 64 MiB BF16 hot tier per pool branch, overridable with
  `DSV4_POOL_BF16_MAX_BYTES`;
- accept an explicit hot-byte limit for focused tests;
- quantize old rows in bulk instead of leaving hundreds of 64-row pieces;
- losslessly concatenate q8 codes/scales/biases into at most 16K-row slabs;
- compact imported/restored segments with a binary cascade;
- make serializer restore accept 16K-row q8 slabs.

The tests in `DeepseekV4PoolQuantizationTests.swift` were updated for the
explicit 2 MiB test hot tier and compact slab geometry. Before the later HC
changes, `swift build -c release --product RunBench` passed. The focused test
command was blocked before reaching these tests by the existing unrelated
`MLXPressPolicyTests` error `no such module 'Testing'`.

Memory warning: 64 MiB is a per-branch hot ceiling, not a proven process
footprint. Measure Activity Monitor `phys_footprint` before accepting it.

### 3. Deferred decode pool advancement — built and benchmarked

`DeepseekV4Cache` now buffers single-token compressor/indexer inputs and only
runs the projection when a compression boundary completes. The default is on;
`DSV4_DEFER_POOL=0` is the A/B opt-out.

State/export/copy paths flush pending rows, trim drops invalid pending rows,
and the flush closure weakly references the cache to avoid a cycle. This
removed repeated per-token compressor work on ratio-4/ratio-128 layers while
preserving the first ten greedy tokens in the measured runs.

This is the largest verified improvement in the dirty tree:

- aggregate 1024-token decode: 20.6 -> 22.3 tok/s (+8.3%);
- final 128-token window: 17.4 -> 20.0 tok/s (+14.9%);
- first 128-token window: 22.6 -> 23.4 tok/s.

It reduces the late-output slope but does not meet the mandatory speed floor.
Full-token parity and lifecycle tests are still required before landing.

### 4. Fused HC-post Metal contraction — built and short-benchmarked

`DeepseekV4Math.hcPost` uses a decode-only Metal kernel for the official fp32
source-HC accumulation and final cast. Multi-token prefill keeps the batched
MLX fallback. DSV4 executes HC-post twice in each of 43 layers, so this removes
86 broadcast/matmul graph constructions per generated token.

The 257-token short gate improved from roughly 23.4/23.1 to 23.8/23.4 tok/s,
23.6 aggregate. The first ten greedy tokens remained identical. Useful, but
only about +0.4 tok/s.

### 5. Shared compiled HC-pre region — **REJECTED AND REMOVED**

The weights-as-input shared HC-pre region compiled successfully, but the exact
257-token A/B regressed from 23.7/23.3 tok/s windows (23.5 aggregate) with the
region off to 12.4/11.0 tok/s (11.7 aggregate) with it on. TTFT regressed from
6.786s to 13.323s. The first ten greedy tokens matched. This reproduces the
historical DSV4 method-compile failure and is not a route to 33+ tok/s.

Artifacts: `/tmp/dsv4-hcpre-off-257.log` and
`/tmp/dsv4-hcpre-on-257.log`.

### 6. Explicit HC-pre Metal kernel — **UNBUILT AND UNVERIFIED**

`DeepseekV4Math.hcPreDecode` and `deepseek_v4_hc_pre_decode` were added as a
replacement strategy after the MLX compile region failed. The kernel assigns
one 256-thread threadgroup to each single-token HC row and intends to fuse the
16K-wide fp32 RMS reduction, 24-output projection, Sinkhorn transform, and
four-way residual mix. Multi-token prefill falls back to the existing MLX
path. `DSV4_HC_PRE_METAL=0` is the intended A/B opt-out.

The build was manually interrupted at the user's handoff request before the
Metal source compiled or linked. The next agent must first inspect the kernel,
then run the release build. If it builds, compare `DSV4_HC_PRE_METAL=0` versus
default with the exact 257-token/two-window command. Reject it on any crash,
token mismatch, or throughput regression.

## Measured evidence

All rows used the real bundle
`/Users/eric/models/DeepSeek-V4-Flash-0731-JANG`, greedy temperature 0,
2,000 synthetic prompt tokens, 512-token prefill steps, and no coordinator.

| Code point | Generated | 128-token windows (tok/s) | Aggregate | TTFT | Artifact |
|---|---:|---|---:|---:|---|
| Before dirty-tree pool work | 1025 | 22.6, 22.3, 22.4, 22.1, 20.5, 19.8, 18.7, 17.4 | 20.6 | 6.589 s | `/tmp/dsv4-window-2k-current.log` |
| Deferred pool | 1025 | 23.4, 23.1, 23.1, 23.1, 23.0, 22.2, 21.0, 20.0 | 22.3 | 6.749 s | `/tmp/dsv4-window-2k-deferred.log` |
| Deferred pool + HC-post | 257 | 23.8, 23.4 | 23.6 | 6.827 s | `/tmp/dsv4-window-2k-hcpost.log` |

The repeated first ten tokens were:

```text
[71, 439, 368, 393, 1230, 1230, 1230, 1230, 1230, 1230]
```

That is only a quick parity signal, not complete output parity.

The stage profiler artifact is `/tmp/dsv4-stage-2200-current.log`. It inserts
evaluation barriers and therefore is not a throughput row. It showed MoE and
attention dominating, with substantial repeated HC/norm/host overhead.

Earlier live Osaurus evidence:

- turn 1: `TTFT 23.21s - 15.3 tok/s - 1,887 tokens`
  (`/tmp/raptorproof/t1_texts.txt`);
- turn 2 cache reuse: `TTFT 1.51s - 28.0 tok/s - 233 tokens`
  (`/tmp/raptorproof/t2_texts.txt`);
- trace `/tmp/dsv4port/trace.log` records disk boundary hits, but the store
  rows also report `effectiveKVLayers=0 blocks=0 payload=false
  companion=false`. Report this exactly; it is not proof that the quantized
  DSV4 pool payload was restored.

## Main Python-to-Swift performance differences

Authoritative reference:
`/Users/eric/jang/jang-tools/jang_tools/dsv4/mlx_model.py`.

### Decode

Python treats single-token decode as host/graph-bound and uses:

- `DeepseekV4Attention._decode_pre_region`: q/kv projections, norms, RoPE,
  and FP8 QAT in one compiled region;
- `DeepseekV4Attention._decode_out_region`: inverse RoPE and grouped output
  projections in one region;
- `MoE._decode_moe_region`: gate, routed/shared experts, reductions, and add
  in one compiled region;
- one shared `_get_hc_pre_compiled` region;
- the fused `_dsv4_hc_post_decode` Metal kernel;
- `DeepseekV4DecoderLayer._decode_tail_region`: attention output projection,
  HC-post, FFN HC-pre, norm, MoE, and final HC-post fused together;
- deferred compressor/indexer advancement at compression boundaries.

Swift now has deferred pooling and HC-post, plus an unverified explicit HC-pre
Metal kernel.
It still builds most attention and MoE/tail graphs separately each token. That
is the main remaining path to 33+ tok/s. Do not re-enable the existing generic
whole-model compile mode: it measured only 13.9-14.5 tok/s.

### Prefill

Python keeps the large batch path separate from decode and uses:

- 512-token **attention-only** subchunks inside a larger outer prefill chunk,
  preserving large-batch MoE throughput;
- heads-16 indexed-attention Metal for selected compressed rows;
- layerwise materialization only beyond 24,576 tokens or when subchunking is
  unavailable, avoiding a measured 25-30% moderate-context penalty;
- tiled quantized-pool views beyond 16,384 pool rows;
- 64 MiB BF16 hot pool storage and compact 16K q8 slabs;
- lazy q8 slab restore/delta-anchor compaction.

Swift's `prefillStepSize=512` chunks the whole model and is not equivalent to
Python's attention-only subchunking. Port the separation rather than blindly
shrinking the outer chunk. Reference commits are recorded in
`DSV4-SPEED-CAMPAIGN.md`.

## Exact continuation sequence

1. Re-read this handoff and `DSV4-SPEED-CAMPAIGN.md`. Preserve
   `Source/Cmlx/mlx` and the six Osaurus files.
2. Keep the rejected MLX-compiled HC-pre region removed. Inspect and build the
   separate explicit `deepseek_v4_hc_pre_decode` Metal kernel. The interrupted
   build provides no evidence that its Metal source is valid.
3. If the kernel builds, run the exact 257-token A/B with
   `DSV4_HC_PRE_METAL=0` as control. Retain it only if tokens match and speed
   materially improves.
4. Port the Python attention pre region, then attention output/tail region.
   Use the same 257-token A/B after each change. The next useful threshold is
   33 tok/s, not another small percentage gain.
5. Once short decode is at least 33 tok/s, run a 1,025-token windowed row to
   verify no late collapse. Only then proceed to prefill tuning, live Osaurus,
   completion latency, SSD companion-state proof, and near-500K validation.
6. Add focused tests for deferred pool flush/export/copy/trim behavior, HC-post
   parity, and compiled-region parity. Resolve or route around the unrelated
   `no such module 'Testing'` test-target issue without weakening tests.

## Landing and PR order

vmlx-swift and Osaurus are separate Git repositories, so their changes cannot
literally be one GitHub PR. The intended release shape is:

1. Finish, test, document, commit, and land the vmlx-swift changes first.
   The final vmlx SHA must exist on `vmlx-origin` and be on/merged to main.
2. In `/Users/eric/osaurus-main-live`, create the integration branch from the
   current `osaurus/main` after checking for upstream movement.
3. Replace stale `e56571e4` with the final landed vmlx SHA in all four pin
   surfaces and both tripwire tests listed above. Regenerate resolution only
   as needed; ensure all four resolved/manifest revisions agree.
4. Build the real Release Osaurus app against that exact checkout and complete
   the live GUI proof. Record TTFT, per-window tok/s, prompt processing,
   completion latency, cache topology/counters, and coherent multi-turn/tool
   output.
5. Open **one Osaurus integration PR** containing the complete final repin and
   its tests. Do not split the six Osaurus files across PRs and do not point it
   at an unmerged vmlx branch SHA.

No commits, pushes, merges, or PRs were performed during this handoff.
