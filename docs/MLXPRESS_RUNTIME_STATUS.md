# MLXPress Runtime Status

Last updated: 2026-05-13

This repo is now the combined `vmlx-swift` integration target for the Osaurus
MLX runtime stack. It contains the base MLX Swift runtime plus the imported
LM/VLM/cache/distributed libraries and MLXPress CLI/library targets. Older
JangPress references in source are compatibility internals only; new runtime
status and user-facing docs should use MLXPress.

## Current Library Baseline

- Top-level branch: `vmlx-0.31.3`.
- Root `Package.swift` exposes local products for `MLX`, `MLXLMCommon`,
  `MLXLLM`, `MLXVLM`, `MLXEmbedders`, distributed runtime targets, `MLXPress`,
  `RunBench`, `mlxpress`, and `mlxpress-selfcheck`.
- `Source/Cmlx/mlx`: `885e5d82f3e23bd4d7b7e55d11e5536c4b5b6378`.
  - Based on Osaurus MLX `7086ba37b1250ba2622a66b181de33e135af6484`.
  - Preserves the local vMLX custom-kernel output-shape patch.
  - Adapts the retained-buffer backport to this branch's `Device` stream API.
  - Adds safetensors mmap expert-advice hooks.
- `Source/Cmlx/mlx-c`: `ddef9122f990014d2b1d78ecccb07cd44c848301`.
  - Adds C ABI wrappers for the mmap advice hooks.
- `.gitmodules` now points `Source/Cmlx/mlx` at `https://github.com/osaurus-ai/mlx`.
- The root package no longer needs sibling `../vmlx-swift-lm` or `../mlxpress`
  checkouts for local MLXPress builds. Those sibling repos remain useful as
  upstream workspaces until this combined tree is committed/pushed.

## Implemented Runtime Surface

- `MLX_SAFETENSORS_MMAP=1` or `VMLINUX_MMAP_SAFETENSORS=1` enables mmap-backed
  safetensors loading for file-path loads.
- Loaded tensor arrays share the mmap-backed shard buffer instead of copying the
  tensor payload into separate allocations.
- Routed expert regions are registered by `(layer, expert)` when tensor names
  match known MoE/JANGQ/JANGTQ patterns.
- ZAYA split stacked JANGTQ tensors are now registered when names match
  `layers.N.zaya_block.experts.switch_mlp.{gate,up,down}_proj.{tq_packed,tq_norms}`.
- Public C++ API:
  - `safetensors_mmap_advise_routed(int32_t advice, int32_t cold_pct)`
  - `safetensors_mmap_advise_experts(int32_t advice, const int32_t* layers, const int32_t* experts, int64_t count)`
- Public C ABI:
  - `mlx_safetensors_mmap_advise_routed`
  - `mlx_safetensors_mmap_advise_experts`
  - `mlx_safetensors_mmap_advise_layer`
- The native macOS purgeable-memory primitive is restored as
  `JangPressMachCache` with public MLXPress aliases
  (`MLXPressMachCache`, `MLXPressMachConfig`, `MLXPressMachStats`). It
  allocates routed expert tiles in `VM_FLAGS_PURGABLE` anonymous regions,
  supports per-expert tensor components, exposes
  `acquire(layer:experts:)`, component acquire, `release(layer:experts:)`, and
  `releaseColdTiles(compressPercent:)`, and maps the user compression percent
  to a hot/cold tile split. Focused synthetic tests prove registration,
  acquire/release volatile accounting, hot pins, unknown-expert errors,
  component acquire/release, no-copy MLXArray views over tile bytes, and
  cold-percent release. This is the required WKdm/native-compression primitive.
- Active expert streaming has an opt-in Mach bridge:
  `MLXPRESS_STREAMING_MACH_ACTIVE_TENSORS=1` /
  `JANGPRESS_STREAMING_MACH_ACTIVE_TENSORS=1`. When enabled, unstacked and
  stacked per-expert JANGTQ tensors are registered in `MLXPressMachCache`,
  exposed to MLX through `mlx_array_new_data_managed_payload`, and released
  cold after evaluated chunks by `MLXPRESS_MACH_COMPRESS_PCT` /
  `JANGPRESS_MACH_COMPRESS_PCT` / numeric `MLXPRESS` / numeric `JANGPRESS`
  (default 70). This is a diagnostic bridge toward the historical resident
  JangPress method; a MiniMax/Hy3/Kimi row still has to prove Activity Monitor
  footprint, decode token/s, file/page pressure, and coherent multi-turn output
  before it can claim the old JangPress speed/RAM methodology.
- Offset-addressed active windows now have a separate opt-in Mach bridge:
  `MLXPRESS_STREAMING_MACH_OFFSET_SPANS=1` /
  `JANGPRESS_STREAMING_MACH_OFFSET_SPANS=1`. This path only attaches to exact
  one-expert active offset windows, registers the bytes in anonymous
  purgeable Mach storage keyed by `(layer, expert, projection, suffix)`, and
  profiles `tensor.mach_offset_read`, `tensor.mach_offset_register`,
  `tensor.mach_offset_hit`, and `tensor.mach_offset_array`. Coalesced
  multi-expert windows still fall back to mmap/cache so Kimi rows can separate
  anonymous resident storage from file-backed LRU behavior. This is created
  code, not readiness proof; it still needs a Kimi row with prompt/decode
  token/s, Activity Monitor RAM, read/page pressure, and coherence.
- Mach offset registration now has a direct-fill path: the cached safetensors
  descriptor is read straight into the purgeable VM region through `pread`,
  instead of allocating a temporary `Data` buffer and copying into Mach storage.
  `MLXPRESS_STREAMING_MACH_OFFSET_SPAN_BUDGET_MB` /
  `JANGPRESS_STREAMING_MACH_OFFSET_SPAN_BUDGET_MB` add an opt-in cap for exact
  offset-window Mach tiles; `tensor.mach_offset_evict` and
  `tensor.mach_offset_budget_skip` report budget behavior. This is the next
  Kimi diagnostic change after rejecting unbounded Mach retention; it still
  requires a real token/s + RAM + coherency row.
- Runtime success is always judged on the same three gates together: decode
  token/s must beat the comparable non-MLXPress/prior baseline, Activity
  Monitor RAM must stay below the family gate, and visible plus reasoning
  output must stay coherent without looping. Rows missing any one of those are
  diagnostic only.
- Offset-addressed streaming has a bounded offset-span residency diagnostic:
  `MLXPRESS_STREAMING_OFFSET_SPAN_CACHE_MB` /
  `JANGPRESS_STREAMING_OFFSET_SPAN_CACHE_MB`. It now caches any offset span by
  `(file, offset, byteCount, dtype)`, including route-specific active expert
  windows. It still does not write permanent stacked overlays and it does not
  build temporary active expert banks. The purpose is to test the
  resident-compute hypothesis from the old JangPress notes: keep reusable
  offset buffers warm under a budget and measure whether decode speed rises
  without losing Activity Monitor footprint, effective read-pressure, or
  coherence gates.
  MiniMax-Small row
  `docs/local/model-validation/20260513T0913Z-minimax-small-offset-full-span-cache-128gb-noprofile-2turn/`
  proves the low-footprint/read-pressure side: coherent two-turn output, cache
  stack on, explicit reads at 0 MB/generated token, effective read pressure
  826.25 MB/generated token, and 2.54 GB / 6.9% peak Activity Monitor
  footprint. It is rejected as the JangPress speed target because decode was
  only 0.823 then 0.816 tok/s. The matching profiled row showed 98.85%
  offset-span cache hit rate and no evictions, so the remaining MiniMax speed
  gap is not span remapping or profiling overhead; it is the active-streaming
  offset JANGTQ execution structure versus resident compute semantics.
- C++ whole-shard mmap GPU correctness is now covered by
  `SaveTests/testMmapSafetensorsLoadCanFeedGPUComputation`: an aligned
  safetensors fixture is loaded with `MLX_SAFETENSORS_MMAP=1`, the mmap
  registry reports layer advice bytes, and a Metal reduction over the loaded
  tensor returns the expected nonzero value. This validates the production C++
  path for aligned tensors; the separate Swift `MmapSafetensorsLoader` remains
  diagnostics-only for CPU/header inspection.
- C++ tensor-span mmap correctness is now covered by
  `SaveTests/testMmapSafetensorsTensorBufferModeDoesNotTrackWholeShard`: a
  sparse safetensors fixture is loaded with
  `MLX_SAFETENSORS_MMAP_TENSOR_BUFFERS=1`, the tracked mmap buffer counter is
  nonzero and less than one quarter of the sparse file size, and a GPU reduction
  over both tensors returns the expected value. This proves tensor-buffer mode
  is no longer silently ignored and does not need a Metal buffer over the whole
  shard.
- Important blocker: GPU-correct tensor-span mmap is still not a complete
  low-footprint MiniMax fix. The new diagnostic row
  `docs/local/model-validation/20260513T083103Z-minimax-small-jpreg-load-footprint-tensor-span-mmap/`
  loaded MiniMax-Small with `MLXPRESS_MMAP_TENSOR_BUFFERS=1`: post-load RSS was
  0.6 GB, Activity Monitor footprint was still 35.0 GB, and live mmap-tracked
  Metal buffers were only 1.3 GB. The remaining resident-memory problem is
  MiniMax JANGTQ load-time restacking into full `switch_mlp` banks, not C++
  inability to map tensor spans. The Swift load wrapper therefore leaves tensor
  mmap buffers as explicit diagnostic opt-in through
  `MLXPRESS_MMAP_TENSOR_BUFFERS` / `JANGPRESS_MMAP_TENSOR_BUFFERS` instead of
  enabling them for every MLXPress mmap load.
- MLXPress active-expert streaming can skip per-expert JANGTQ tensors during
  generic weight load before `model.update`, but this is now treated as an
  explicit fallback/diagnostic path. The default MLXPress method is
  compression-first canonical mmap residency: keep routed weights loaded as
  mmap-backed tensors, advise inactive routed pages cold, and let macOS
  reclaim/compress them instead of rereading active slices from SSD forever.
- `MLXPRESS_STREAMING_PROFILE=1` now records opt-in timing counters for the
  streaming expert path: router readback, stacked slice reads, `MLXArray(Data)`
  construction, active expert stacking, gate/up and down evals, Kimi router,
  shared expert build, and layer eval. `MLXPRESS_STREAMING_PROFILE_EVERY`
  controls periodic summaries and `MLXPRESS_STREAMING_PROFILE_TOP` controls
  how many rows are printed; `mlxpress` dumps a per-turn summary when the
  profile is enabled. Profile rows now include total bytes, bytes per call, and
  effective MB/s so Kimi speed work can distinguish memory bandwidth from
  orchestration/kernel sequencing.
- `mlxpress --metrics-jsonl PATH` writes durable JSON Lines records for each
  run: `run_start`, per-turn token/s and coherency, streaming-profile snapshots
  when profiling is enabled, post-decode/peak memory, Activity Monitor gate
  verdicts, MLX allocator active/cache/peak bytes, and system pressure
  snapshots with page-in/page-out and swap-in/swap-out deltas. When a file-read
  gate is configured it writes `file_read_pressure` and
  `effective_read_pressure` records. The effective row uses the larger of
  instrumented explicit reads and system page-in deltas, so mmap paths cannot
  look ready just because they bypass the `pread` counters. Use this on every
  serious model row so token/s, routed profile counters, Activity Monitor
  footprint, MLX allocator accounting, file-read pressure, and swap/page
  pressure are captured without hand-copying stderr.
- `mlxpress --file-read-gate-mb-per-token N` enables streaming profiling and
  fails if cumulative active tensor file reads exceed `N` MB per generated
  token. `--file-read-gate-report-only` records the verdict without aborting.
  The same threshold now also checks effective read pressure from page-ins.
  This is the runtime gate that prevents a low-RAM row from being mislabeled
  ready while it is still rereading too many safetensor bytes per token or
  faulting through whole mmap'd tensors.
- Active-expert reads now use bounded cache residency when
  `MLXPRESS_STREAMING_EXPERT_CACHE_MB` is nonzero. For stacked tensors the
  cache tracks per-expert slice `Data`; for unstacked tensors it tracks
  per-expert `MLXArray` tensors. Both paths are keyed by
  `(layer, expert, projection, suffix)`, emit `active_slice_residency` JSONL
  rows, and make the file-read gate count actual safetensor bytes read rather
  than bytes assembled from resident entries. Default remains 0 MB for the
  lowest-RAM path.
- Active-bank residency has an opt-in diagnostic knob,
  `MLXPRESS_STREAMING_BANK_CACHE_MB` / `JANGPRESS_STREAMING_BANK_CACHE_MB`.
  It caches fully assembled active banks keyed by exact
  `(layer, projection, suffix, sortedExperts)` and reports split bank fields in
  `active_slice_residency` JSONL rows. MiniMax proof shows this is not a
  production speed fix: exact top-k sets barely repeat, so the knob must stay
  default-off.
- `MLXPRESS_STREAMING_SOURCE_ORDER_READS=1` is an opt-in diagnostic that reads
  stacked active slices in safetensor offset order and coalesces adjacent expert
  ranges. It is not default-enabled because the Kimi proof row was slower and
  did not reduce MB/generated-token.
- `scripts/summarize-mlxpress-metrics.py` reads one or more metrics JSONL files
  and emits stable comparison fields: prompt token/s, decode token/s, coherency,
  peak Activity Monitor percent, Activity Monitor gate pass, peak MLX
  active/cache/peak allocator MB, and top streaming profile rows with
  milliseconds and MB/s. It also reports
  `profile_read_mb` and `profile_read_mb_per_gen_token` from active tensor
  read counters, so Kimi rows can separate model-file streaming pressure from
  process RAM footprint or OS swap; new metrics rows also summarize
  `pagein_mb`, `swapin_mb`, and `swapout_mb`. Active residency rows add
  combined `cache_hit_rate` / `cache_byte_hit_rate` / `cache_resident_mb` plus
  split `tensor_resident_mb`, `slice_resident_mb`, and `bank_resident_mb`
  fields, so MiniMax/Hy3 unstacked tensor-cache rows do not look empty, Kimi
  stacked-slice rows remain visible, and exact-bank experiments cannot hide
  memory growth. Use it before/after every Kimi scheduler or direct-kernel
  experiment, and keep the same fields for every other model family.
- `MLXPRESS_GENERATION_PROFILE=1` /
  `JANGPRESS_GENERATION_PROFILE=1` records coarse prompt/decode stage timing
  to stderr at generation end. It times prompt preparation/eval, compiled or
  regular model forward, KV quantization, async eval submission, sampling,
  decode step construction, and token sync. Use it for local bottleneck rows,
  but keep a paired no-profile token/s row before blaming profiler overhead.
- Routed JANGTQ bundles now default to compression-first canonical mmap
  residency under MLXPress. Active-expert streaming is explicit opt-in via
  `--active-expert-streaming on` / `MLXPRESS_STREAMING_EXPERTS=1` for fallback
  diagnostics. The permanent prestacked safetensors overlay is explicit opt-in
  only via `MLXPRESS_PRESTACK=1` / `JANGPRESS_PRESTACK=1`.
- Non-stacked routed bundles now also have two load-time resident-compute
  diagnostics. `MLXPRESS_LOADTIME_MACH_STACKS=1` materializes projection stacks
  into `MLXPressMachCache` and exposes managed array views; MiniMax proved this
  is too coarse because Activity Monitor rises to full routed size after GPU
  touch. `--ephemeral-prestack on` is now the typed MLXPress/CLI wrapper for
  the current MiniMax diagnostic: it sets the explicit prestack knobs, writes a
  unique temporary stacked overlay, maps it through the mmap loader, and now
  removes the overlay immediately after load once mappings are established
  (with the old process-exit cleanup as a fallback). That path is still
  explicit opt-in, but it is the first MiniMax row to combine coherent
  multi-turn output, cache stack, low Activity Monitor footprint, no permanent
  overlay, and usable decode speed. Fresh CLI
  proof:
  `docs/local/model-validation/20260513T174200Z-minimax-small-cli-ephemeral-prestack-nocold-strict-2turn/`
  passed strict no-length-stop coherency with 2.63 GB / 7.093% peak Activity
  Monitor footprint, 0 MB explicit reads, 780.76 MB/generated-token effective
  read pressure, and 13.162 aggregate tok/s.
- The core MLXPress throughput premise is now documented and guarded:
  decompression/refault can cost time, but the per-token effective working set
  should be far smaller than full uncompressed routed weights or repeated SSD
  slice reads. Low Activity Monitor RAM plus high `file_read_pressure` or
  `effective_read_pressure` is a failed compression-path proof.
- JANGTQ Hadamard now includes the Python-side <=1024-block SIMD-shuffle path
  in Swift: `JANGTQKernelLibrary.hadamardShuffleLE1024` uses
  `simd_shuffle_xor` for the first five butterfly stages, and
  `JANGTQKernels.hadamardRotate` dispatches it whenever all power-of-two
  blocks are <=1024. This covers MiniMax's 1536-dim intermediate rotation
  (`1024 + 512`) without the heavier all-stage threadgroup-memory path. This is
  a generic JANGTQ speed prerequisite only; real model decode token/s still has
  to be measured in a coherent MLXPress row.
- Streaming JANGTQ reduce now defaults to a 4-token prefill chunk
  (`MLXPRESS_STREAMING_REDUCE_TOKEN_CHUNK_SIZE=4`,
  `MLXPRESS_STREAMING_FAST_REDUCE_MAX_TOKENS=4`). The Kimi stacked profile row
  kept peak Activity Monitor footprint under 2 GB while reducing short-prompt
  active expert chunk count from 1140 to 180.
- Active-expert streaming now respects caller-provided
  `MLXPRESS_STREAMING_EVAL_EACH_LAYER` overrides. MLXPress still defaults the
  layer materialization boundary on for low-RAM safety when the variable is
  unset, but diagnostic rows can turn it off without the session wrapper
  silently forcing it back to `1`. A bounded scheduler knob,
  `MLXPRESS_STREAMING_EVAL_LAYER_STRIDE` /
  `JANGPRESS_STREAMING_EVAL_LAYER_STRIDE`, now evaluates every Nth routed layer
  while keeping the all-off mode available. This is meant to test the middle
  ground between the slow per-layer eval path and the high-residency no-eval
  path.
- When the explicit streaming fallback is enabled, stacked routed tensors use
  active bank reads
  (`MLXPRESS_STREAMING_BANK_LOAD=1` / `JANGPRESS_STREAMING_BANK_LOAD=1`).
  When a layer has stacked `switch_mlp` tensors, the streaming path reads the
  selected expert byte ranges into one contiguous temporary bank per
  projection/suffix, then constructs one `MLXArray` for the active bank instead
  of many per-expert arrays plus `MLX.stacked`. This is a low-RAM cleanup and
  profiling aid, not a complete Kimi speed fix.
- Active-expert streaming expert-slice reads now use a size-aware Darwin read
  cache policy. Fit-in-RAM routed JANGTQ byte sets use normal OS-cached `pread`
  so MiniMax/Hy3-class rows can warm in RAM instead of rereading from SSD.
  Oversized routed sets above the default 70% physical-memory threshold keep
  `F_NOCACHE` to avoid polluting the OS file cache; override with
  `MLXPRESS_STREAMING_F_NOCACHE=0/1` or
  `MLXPRESS_STREAMING_F_NOCACHE_THRESHOLD_GB`.
- Load-time per-expert stacking has a diagnostic materialization switch:
  `MLXPRESS_LOADTIME_MATERIALIZE_STACKS=0` /
  `JANGPRESS_LOADTIME_MATERIALIZE_STACKS=0`. Default stays materialized because
  lazy MLX stack graphs can retain every per-expert input and may defer the same
  full-bank allocation to decode. Use this only to diagnose non-stacked JANGTQ
  resident mmap rows.
- A diagnostic mmap layer advisor is exported as
  `mlx_safetensors_mmap_advise_layer` and surfaced through
  `MLXPressMmapColdSweep`. It is opt-in via
  `MLXPRESS_MMAP_LAYER_COLD_SWEEP=1`; it is not default-enabled because the
  Kimi short-answer probe showed no peak reduction.
- Hy3 JANGTQ_K now preserves mixed projection bits: gate/up use the 2-bit
  `codebook.4096.2` table and down uses the 4-bit `codebook.1536.4` table.
- MLXPress CLI coherency telemetry now records loop checks for visible and
  reasoning channels, optional minimum visible/generated-token gates, and
  optional max-token stop failure. Direct CLI runs exit nonzero when strict
  coherency gates are requested and fail.
- `mlxpress --inspect --json` now emits `bundle.architecture` plus
  `readiness`: config-derived attention kinds, RoPE/MRoPE signals, cache
  storage/encode requirements, companion-state kinds, media-cache identity,
  canonical load method, per-gate created/proven/partial/blocked states,
  required proofs, and current blockers. This is the runtime-visible checklist
  agents must consult before calling a model family MLXPress-ready.
- `bundle.architecture` now also separates `matmulKinds`,
  `positionVectorKinds`, `cacheEncodingKinds`, and `hybridSplitKinds`, so
  Hadamard/TurboQuant matmul, 2D/3D MRoPE vectors, media-salted cache encoding,
  and path-dependent hybrid state can be checked without parsing prose.
- Inspect now reads RoPE values from direct config keys and nested
  `rope_parameters` / `rope_scaling`, and exposes MLA partial-RoPE attention,
  Qwen VL `[3,batch,seq]` position IDs, image/video `THW` grids, MRoPE deltas,
  cache-policy salt, disk token-hash salt, and path-dependent companion-state
  hit/reject rules.
- Inspect treats a non-empty `vision_config` as a VLM/media-cache contract even
  when the nested model type is named generically, for example
  `model_type=qwen3_5`, unless bundle metadata explicitly says
  `has_vision=false`. Those bundles now surface Qwen 2D/3D MRoPE vector gates
  and block on real media-cache proof instead of looking text-only.
- `StreamOrDevice.stream(_:)` now preserves the caller-provided stream instead
  of falling back to the default stream. That matters for future isolated
  rederive diagnostics, router readback isolation, and cache warm-pass
  diagnostics where a named stream must actually be used.
- The readiness checklist now tracks attention architecture, Hadamard/TurboQuant
  matmul, RoPE/MRoPE position handling, cache block storage/encode, hybrid
  companion-state split, VL/vector media cache identity, parser autodetect,
  per-turn RAM/speed artifacts, cold/warm deviation proof, and async
  rederive/warm-pass behavior as first-class gates.
- `docs/MLXPRESS_ACTIVE_EXPERT_SCHEDULER_PLAN.md` now tracks the concrete
  helpers needed to make low-RAM active expert streaming fast: active expert
  trace, budgeted slice residency, routing-aware prefetch, direct
  stacked-offset JANGTQ dispatch, family policies, and pressure watcher gates.
- The architecture ledger now carries a per-family build matrix for Kimi,
  MiniMax, Hy3, Qwen3.6, Qwen VL, ZAYA, Ling/Bailing, and DSV4. Each row
  records the attention/position contract, decode/matmul contract, cache-stack
  contract, parser/tool contract, and async/warm/deviation contract to check
  before adding more runtime proof claims.
- ZAYA VL inspect rows now block on missing real media cache proof and CCA
  companion-state proof:
  `docs/local/model-validation/20260513T005500Z-zaya-vl-readiness-inspect/RESULT.md`.

## Verification

Passing:

- `swift build --jobs 2 --product mlxpress` from this root repo.
- `swift build --jobs 2 --product mlxpress-selfcheck` from this root repo.
- `scripts/prepare-mlx-metal.sh` after deleting
  `.build/arm64-apple-macosx/debug/{default,mlx}.metallib`; it compiles
  `Source/Cmlx/mlx-generated/metal/*.metal` with the installed Xcode Metal
  toolchain and writes both colocated runtime filenames.
- `.build/debug/mlxpress-selfcheck`
- `.build/debug/mlxpress --runtime-check --json`
- `.build/debug/mlxpress ~/models/JANGQ/Hy3-preview-JANGTQ_K --inspect --json`
- `.build/debug/mlxpress` strict coherency exit smoke:
  thinking-on MiniMax CRACK K reasoning-only row returned exit 3 with
  `visible=fail`, `reasoning-loop=pass`, `length-stop=fail`; thinking-off
  checklist row returned exit 0 with `loop=pass`, `min-generated=pass`, and
  `length-stop=pass`.
- `swift build --target Cmlx`
- `swift build --target MLX`
- `swift build -c release --target Cmlx`
- `swift build -c release --product Example1`
- `nm -gU .build/arm64-apple-macosx/release/Example1 | rg 'mlx_safetensors_mmap_advise'`
- `git diff --check` in the top repo and both nested Cmlx submodules
- `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer swift test --filter StreamTests/testExplicitStreamOrDevicePreservesStream`
  passes. This verifies `StreamOrDevice.stream(_:)` preserves the caller stream;
  the MLX test helper brackets the assertion with the prepared debug metallib
  directory so the C++ fallback `default.metallib` lookup succeeds under
  SwiftPM's XCTest runner.
- `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer swift test --filter MLXPressLowRamPolicySourceTests`
  passes. `MLXPressPolicyTests` intentionally registers only the focused source
  coverage file, not the whole imported `MLXLMTests` tree.
- `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer swift test --filter SaveTests/testMmapSafetensorsLoadCanFeedGPUComputation`
  passes. This proves the current C++ whole-shard safetensors mmap loader can
  feed a GPU/Metal reduction correctly for aligned tensors under
  `MLX_SAFETENSORS_MMAP=1`, even when
  `MLX_SAFETENSORS_MMAP_START_COLD=1` and
  `MLX_SAFETENSORS_MMAP_COLD_PCT=100`.
- `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer swift test --filter SaveTests/testMmapSafetensorsTensorBufferModeDoesNotTrackWholeShard`
  passes. This proves opt-in tensor-span mmap creates live mmap-backed Metal
  buffers without tracking an entire sparse shard, and those tensors feed GPU
  computation correctly.
- `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer swift test --filter JANGTQHadamardShuffleTests/testMiniMaxSizedShufflePathMatchesCPUReference`
  passes. This registered focused test runs the new 1536-dim SIMD-shuffle
  Hadamard path on GPU and compares the output against a CPU reference.

## Prompt And Decode Order

For `mlxpress` text generation the current order is:

1. CLI parses turns/options and inspects `MLXPressBundleFacts`.
2. `MLXPressSession.load` applies decode memory defaults, disables permanent
   prestack, keeps active-expert streaming off unless explicitly requested, and
   calls `loadModelContainer` with mmap safetensors + MLXPress compression
   policy so routed tensors are canonical resident weights.
3. `Load.swift` loads dense/bookend/shared/attention/routed tensors through
   mmap-backed safetensors. It skips streamable routed JANGTQ tensors only when
   the explicit active-streaming fallback is enabled. Missing
   `jangtq_runtime.safetensors` falls back to deterministic signs/codebooks from
   seed and bit metadata.
4. Cache stack is attached after load: paged KV, disk L2, TurboQuant KV policy,
   and model/cache salt.
5. `MLXPressSession.generate` forces base autoregressive decode by setting
   `DraftStrategy.none`, prepares the prompt through the tokenizer/chat
   template, and calls `ModelContainer.generate`.
6. `TokenIterator` optionally fetches a cache hit, pre-fills the remaining
   prompt in `prefillStepSize` windows, then loops one accepted token at a time.
7. Kimi routed layers execute the resident-weight path:
   `DeepseekV3JANGTQDecoderLayer -> DeepseekV3Attention -> DeepseekV3JANGTQMoE
   -> MoEGate -> TurboQuantSwitchGLU`. Explicit active-streaming fallback swaps
   the MoE reducer to `StreamingTurboQuantSwitchGLU.reduced`.
8. In the default path, JANGTQ kernels consume the loaded routed weights and
   canonical mmap advice decides which routed pages stay warm/cold. If the
   explicit active-streaming fallback is enabled, the streaming reducer reads
   only the active stacked expert slices for the layer/top-k set, builds
   temporary banks, runs Hadamard + fused gate/up + down JANGTQ gathers, then
   releases those banks. That fallback proves low footprint but is not the
   core MLXPress speed method.

Blocked:

- `swift test ...` without `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer`
  still selects `/Library/Developer/CommandLineTools`; that toolchain lacks the
  macOS XCTest module, so XCTest-backed targets fail before filtered tests run.

## Boundaries

- This repo alone is not a coherence claim for any model family.
- Detailed per-family implementation notes live in
  `docs/MLXPRESS_ATTENTION_ARCHITECTURE_LEDGER.md`; update that ledger as
  each model attention/cache architecture is probed or fixed.
- Current stacked Kimi K2.6 JANGTQ_K status in this combined repo:
  - Bundle: `~/models/JANGQ/Kimi-K2.6-JANGTQ_K`,
    `model_type=kimi_k25`, 61 layers, 60 routed layers, 384 experts,
    top-k 8, stacked `switch_mlp.{gate,up,down}_proj` JANGTQ tensors,
    328.11 GB safetensors bytes.
  - Low-RAM load path is proven for stacked tensors. Active-expert streaming
    indexes 60 stacked routed layers, skips 360 stacked routed tensors during
    generic weight load, reads only the requested expert slice by safetensor
    byte offset, uses the Darwin size-aware cached-`pread` policy
    (`F_NOCACHE` by default for this oversized Kimi bundle), and does not write
    a permanent prestacked overlay.
  - Runtime sidecar is now present for this bundle:
    `~/models/JANGQ/Kimi-K2.6-JANGTQ_K/jangtq_runtime.safetensors`.
    Header inspection shows `codebook.7168.2`, `signs.7168.42`,
    `codebook.2048.4`, and `signs.2048.42`, matching Kimi JANGTQ_K gate/up
    2-bit and down 4-bit metadata. The deterministic missing-sidecar fallback
    remains available for other bundles, but the current Kimi K row is no
    longer using that fallback. The bundle required `tokenizer.json`; the local
    copy now has one.
  - Default low-RAM exact two-turn proof:
    `docs/local/model-validation/20260513T024727Z/` passed with cache stack
    enabled, `active-expert tensor cache budget=0 MB`, `red color` /
    `blue fruit`, post-load 144.9 MB, post-decode 1.80 GB, peak 3.26 GB
    / 1.0% of model bytes, avg decode telemetry 0.28 tok/s.
  - Longer no-thinking multi-turn proof:
    `docs/local/model-validation/20260513T024821Z/` passed with no-loop and
    minimum-generation checks, output
    `Rain falls softly on the green grass.` /
    `The moon glows brightly in the night sky.`, post-load 145.8 MB,
    post-decode 2.81 GB, peak 4.68 GB / 1.4%, avg decode telemetry
    0.39 tok/s.
  - Kimi speed is still blocked. The current path is correct for low RAM and
    coherence, but decode remains far below MiniMax/Hy3-class throughput
    because Kimi stacked JANGTQ_K still performs per-token active routed MoE
    work across 60 layers x top-k 8. The next speed work is a fused/direct
    stacked-offset JANGTQ dispatch path that avoids rebuilding active expert
    arrays around each token.
  - Profiling evidence:
    `docs/local/model-validation/20260513T032126Z-kimi-k-profile-1tok/`
    returned coherent `red`, post-load 147.1 MB, peak 1.73 GB / 0.5%, but
    decode was 0.11 tok/s for one generated token. With 4-token streaming
    reduce chunks it executed 180 active expert chunks, read 20.19 GB of active
    stacked slices, and materialized 8,640 active tensors.
  - Sustained short decode profile:
    `docs/local/model-validation/20260513T033000Z-kimi-k-profile-6tok/`
    returned coherent `The green grass swayed gently`, peak 2.39 GB / 0.7%,
    and 0.35 tok/s for six generated tokens. It still read 130.55 GB of active
    stacked slices and materialized 55,866 active tensors, so the bottleneck is
    repeated decode-layer active expert construction, not the mmap load.
  - Active bank-read profile:
    `docs/local/model-validation/20260513T033557Z-kimi-k-bankload-profile-1tok/`
    returned coherent `red`, peak 1.76 GB / 0.5%, and 0.12 tok/s for one
    generated token. The profile shows `tensor.stacked_bank_read` and
    `tensor.stacked_bank_array`, confirming the rebuilt CLI was using the bank
    path rather than stale per-expert `tensor.stacked_read` counters.
  - Sustained bank-read speed row:
    `docs/local/model-validation/20260513T033702Z-kimi-k-bankload-profile-6tok/`
    generated a non-looping visible phrase at 0.39 tok/s with peak 3.15 GB
    / 1.0%. It cut the active tensor construction counter to 3,960 bank arrays
    but still read 127.76 GB of active stacked slices, so bank reads are useful
    cleanup but do not solve the underlying 60-layer x top-k routed decode
    cost.
  - Metrics JSONL and bandwidth row:
    `docs/local/model-validation/20260513T034200Z-kimi-k-metrics-jsonl/`
    proves `--metrics-jsonl` on the real Kimi path. It wrote six JSONL records
    with prompt/decode token/s, coherency, memory, and activity-gate results.
    The streaming profile showed `tensor.stacked_bank_read` at 20.19 GB over
    2.90 s, about 6,968 MB/s, and bank-array construction at about
    19,643 MB/s. That is far below available unified-memory bandwidth, so the
    next fix should target routed decode scheduling, direct stacked-offset
    kernels, or graph/eval sequencing rather than assuming file reads alone
    are saturating the machine.
  - Streaming-profile JSONL row:
    `docs/local/model-validation/20260513T034856Z-kimi-k-streaming-profile-jsonl/`
    extends that artifact format with a machine-readable `streaming_profile`
    record. It captured 20 profile rows, including `tensor.stacked_bank_read`
    at 20.19 GB / 6,933 MB/s, `tensor.stacked_bank_array` at 19,408 MB/s,
    and `router.indices_readback` at 1.97 s. Future scheduler/kernel rows
    should compare against this JSONL shape.
  - No-eval boundary diagnostic after env fix:
    `docs/local/model-validation/20260513T035500Z-kimi-k-no-each-layer-eval-after-envfix/`
    proves `MLXPRESS_STREAMING_EVAL_EACH_LAYER=0` is now honored by MLXPress
    instead of being overwritten during active-streaming setup. The row
    returned coherent `red`, prompt 2.04 tok/s, decode 0.12 tok/s, peak
    1.86 GB / 0.6%, and a streaming profile without `gateup.eval`,
    `down.eval`, `router.scores_eval`, `reduce.score_sum_eval`, or
    `kimi.layer_eval` top rows. Profile-internal time dropped versus the JSONL
    baseline, but one-token decode throughput did not meaningfully improve, so
    this is a scheduler diagnostic only.
  - File-streaming diagnosis:
    `docs/local/model-validation/20260513T041000Z-kimi-k-file-streaming-diagnosis/`
    records the current conclusion: Kimi's sub-1 tok/s path is primarily
    intentional safetensor expert-slice streaming, not a full-model process RAM
    or proven swap failure. The one-token profile reads 20.19 GB of active
    stacked experts while process peak stays under 2 GB. Preventing this means
    reducing repeated read bytes per token with direct stacked-offset kernels,
    routing-aware active-slice reuse, and better scheduler/eval sequencing, not
    loading the whole model or forcing OS-cache for an oversized Kimi bundle.
  - System-pressure proof row:
    `docs/local/model-validation/20260513T041500Z-kimi-k-system-pressure-profile/`
    used the rebuilt CLI with `system_pressure` JSONL records. It returned
    coherent `red`, prompt 1.90 tok/s, decode 0.119 tok/s, peak 1.77 GB /
    0.5%, `profile_read_mb_per_gen_token=20190.94`, `pagein_mb=24790.98`,
    `swapin_mb=0.06`, and `swapout_mb=0.00`. This confirms the tiny Kimi row
    is SSD/file-backed model streaming, not swap-out thrash.
  - File-read gate report:
    `docs/local/model-validation/20260513T042500Z-kimi-k-file-read-gate-report/`
    proves the new gate catches the issue. With a report-only
    1,000 MB/generated-token gate, the coherent `red` row stayed low RAM
    (1.79 GB / 0.5%) and had 0 MB swap-out, but emitted
    `file_read_pressure.passed=false` because active expert reads were
    20,190.94 MB per generated token.
  - Active expert trace:
    `docs/local/model-validation/20260513T043500Z-kimi-k-active-expert-trace/`
    proves the trace surface on Kimi. The coherent `red` row stayed low RAM
    (1.76 GB / 0.5%) and recorded 180 trace calls, 1,440 routed slots,
    1,440 unique expert touches, 309 consecutive reuse touches, and reuse rate
    0.215 while still failing the file-read gate at 20,190.94 MB/generated
    token.
  - Active slice residency diagnostics:
    `docs/local/model-validation/20260513T043530Z-kimi-k-active-slice-residency-512mb/`
    proves a 512 MB budget is too small for Kimi's reuse distance: hit rate
    0.000, file reads unchanged, peak 2.47 GB / 0.8%, coherent `red`.
    `docs/local/model-validation/20260513T043713Z-kimi-k-active-slice-residency-8192mb/`
    proves the residency path is real: hit rate 0.215 and file reads drop to
    15,858.30 MB/generated-token. It still does not improve decode
    (0.117 tok/s) and raises peak footprint to 11.35 GB / 3.5%, so this is a
    diagnostic knob, not Kimi readiness.
  - Sidecar-present effective-gate rows:
    `docs/local/model-validation/20260513T045222Z-kimi-k-sidecar-bankload-effective-gate/`
    and
    `docs/local/model-validation/20260513T045340Z-kimi-k-sidecar-bankload-effective-gate-warm/`
    prove the newly added sidecar is valid and loaded. Both rows returned
    coherent `red` and passed the Activity Monitor gate. The cold row saw
    98,234.52 MB/generated-token explicit reads because it processed more
    prompt work; the warm row is the apples-to-apples baseline at 0.117 tok/s
    and 20,190.94 MB/generated-token. Sidecar presence fixes metadata
    correctness but does not fix Kimi speed.
  - Direct stacked mmap effective-gate row:
    `docs/local/model-validation/20260513T045423Z-kimi-k-direct-stacked-mmap-effective-gate/`
    proves the first opt-in direct mmap prototype is not the production fix.
    It returned coherent `red` and kept peak footprint at 1.67 GB / 0.5%, but
    decode fell to 0.022 tok/s and `effective_read_pressure` failed at
    969,853.44 MB/generated-token due page-ins. This is why future direct work
    must be an offset-descriptor/kernel path, not feeding whole stacked mmap
    tensors into the current eval kernels.
  - Source-order bank read diagnostic:
    `docs/local/model-validation/20260513T050957Z-kimi-k-bank-read-plan-effective-gate/`
    enabled `MLXPRESS_STREAMING_SOURCE_ORDER_READS=1`. It stayed coherent and
    low-RAM, but decode dropped to 0.099 tok/s, bank read bandwidth fell to
    3,961 MB/s, effective read pressure stayed high, and system swap-out rose.
    The follow-up default row
    `docs/local/model-validation/20260513T051221Z-kimi-k-bank-read-plan-default-restored/`
    confirms this diagnostic path is off by default. Do not treat source-order
    reads as the Kimi speed fix.
  - Build/profile nuance: after changing streaming code, explicitly run
    `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer swift build --jobs 2 --product mlxpress`
    before profiling. The combined build command that also names
    `mlxpress-selfcheck` can leave the CLI binary stale in incremental output,
    which makes profile counters misleading.
  - MiniMax is faster because its hot path is smaller and simpler: hidden
    3072 vs Kimi 7168, KV heads 8 vs Kimi 64, simpler dense RoPE attention,
    optional compiled router, no Kimi-style MLA partial/no-PE attention cost,
    and no dense shared-expert MLP added after every routed MoE block. Kimi also
    lacks the MLA absorb decode branch and a direct stacked-offset JANGTQ
    kernel/scheduler that can avoid per-token slice array rebuilds.
  - Active slice construction now uses `MLXArray(Data, dtype:)` directly
    instead of first copying into typed Swift arrays. Follow-up exact row
    `docs/local/model-validation/20260513T030410Z/` stayed coherent and
    low-RAM with peak 3.37 GB / 1.0% and avg decode 0.30 tok/s. This removes
    one avoidable copy, but does not change the main speed diagnosis.
  - Current MiniMax-Small split after the Hadamard SIMD-shuffle and streaming
    read-cache fixes:
    - `docs/local/model-validation/20260513T061054Z-minimax-small-streaming-oscache-hadamard/`
      used explicit active-expert streaming with the new auto OS-cache policy
      (`streamable=32.81GB`, threshold 89.60GB). Output was coherent and
      low-RAM, peak 2.53 GB / 6.8%, but decode averaged only 1.82 tok/s and
      read-pressure failed at 5,913 MB/generated-token. This proves
      `F_NOCACHE` was not the only MiniMax slowdown; per-token active bank
      construction is still too much movement.
    - `docs/local/model-validation/20260513T061253Z-minimax-small-resident-speed-hadamard/`
      used resident routed tensors with active streaming off and
      `--activity-gate-report-only`. Output was coherent, explicit file reads
      were zero, and effective read pressure passed at 522 MB/generated-token,
      but peak footprint was 35.37 GB / 95.5% and decode averaged 3.72 tok/s
      (turn 2: 5.89 tok/s). This is faster than streaming but still nowhere
      near the Python 44 tok/s ceiling and fails the low-RAM MLXPress gate.
    - Fresh allocator-telemetry rows:
      `docs/local/model-validation/20260513T063458Z-minimax-small-streaming-telemetry/`
      and
      `docs/local/model-validation/20260513T063643Z-minimax-small-resident-telemetry/`
      show the split precisely. Streaming is low-footprint
      (2.53 GB / 6.8%) but has 254,261 MB explicit active-slice reads and
      1.83 tok/s decode; MLX active/peak accounting is ~39.16 GB. Resident
      mode has zero explicit reads and coherent output, but Activity Monitor
      still reports 35.37 GB / 95.5%, while MLX active/peak is ~71.95 GB.
      `MLX_SAFETENSORS_MMAP_DEBUG=1` confirms all 39 model shards mmap-load;
      the extra ~32.8 GB is consistent with MiniMax non-stacked per-expert
      tensors being materialized into `switch_mlp` banks by
      `loadTimeMaterializedStacked`, not with a C++ mmap fallback for model
      shards.
    - The lazy-stack diagnostic row
      `docs/local/model-validation/20260513T064821Z-minimax-small-resident-lazy-stack/`
      ran with `MLXPRESS_LOADTIME_MATERIALIZE_STACKS=0` and stayed coherent,
      but did not reduce the resident footprint: decode averaged 2.67 tok/s,
      peak Activity Monitor remained 35.4 GB / 95.6%, and MLX active/peak
      stayed ~71.95 GB. Skipping the explicit load-time eval only defers the
      same stacked-bank allocation/accounting; it is not a memory fix.
    - The follow-up resident-expert load-only rows remove that stacked-bank
      variable and narrow the blocker further. `MLXPRESS_RESIDENT_EXPERTS=1`
      now registers MiniMax per-expert tensors into the streaming expert store
      and avoids materializing full `switch_mlp` banks, but the load-only row
      `docs/local/model-validation/20260513T090900Z-minimax-small-resident-experts-load-footprint/`
      still reported 35.2 GB post-load Activity Monitor footprint with
      34.6 GB of live mmap-tracked Metal buffers. `MADV_PAGEOUT` row
      `docs/local/model-validation/20260513T092000Z-minimax-small-resident-pageout-load-footprint/`
      and `msync(MS_INVALIDATE | MS_ASYNC)` row
      `docs/local/model-validation/20260513T092038Z-minimax-small-resident-force-invalidate-load-footprint/`
      also stayed at 35.2 GB. A focused small safetensors GPU test passes
      after force invalidation, so the issue is not immediate GPU corruption;
      it is that creating live Metal buffers for the whole routed set defeats
      the Activity Monitor low-RAM gate.
    - Short generation row
      `docs/local/model-validation/20260513T093159Z-minimax-small-resident-experts-short-generation/`
      confirms the no-full-bank resident path can still produce sane output:
      `Green` / `Blue`, both turns passed coherency/no-loop checks, prompt
      throughput was 4.32 then 9.61 tok/s, and decode throughput was 0.25 then
      0.98 tok/s for one generated token per turn. It still failed the 30%
      Activity Monitor gate with peak 35.25 GB / 95.2% and MLX peak 70.64 GB,
      so this is a correctness diagnostic only, not a readiness row.
    - The active-streaming mmap tensor rows replace `pread -> Data -> MLXArray`
      active-slice loads with the new C++ file-region mmap array path. Short
      row
      `docs/local/model-validation/20260513T095534Z-minimax-small-streaming-mmap-active-tensors-short-generation/`
      produced coherent `Green` / `Blue`, passed the 30% Activity Monitor gate
      at 2.44 GB / 6.6%, and recorded zero explicit active-slice reads, but
      one-token decode was still only 0.25 then 0.96 tok/s.
    - Longer row
      `docs/local/model-validation/20260513T095717Z-minimax-small-streaming-mmap-active-tensors-longer-generation/`
      is the current best low-RAM MiniMax diagnostic: coherent multi-turn
      apples/rivers output, cache stack on, peak Activity Monitor 2.66 GB /
      7.2%, zero explicit active-slice reads, and decode 1.63 then 2.39 tok/s
      for 16/21 generated tokens. `tensor.mmap_array` handled 222,426 MB of
      logical active tensor bytes at about 16.4 GB/s; remaining speed blockers
      are `reduce.call_chunk`, `gateup.total`, `gateup.eval`, `down.eval`, and
      `router.indices_readback`.
    - Multi-file offset kernels are now wired for MiniMax-style expert-major
      tensors. `JANGTQStackedOffsetDescriptor` classifies true stacked tensors
      as `stacked-contiguous`, MiniMax same-file layouts as
      `expert-major-single-file-offsets`, and split-shard layouts as multiple
      `expert-major-multi-file-offsets` descriptors with `UInt32.max`
      sentinels for missing experts. Focused tests cover descriptor layouts
      plus split-shard sentinel summing for fused gate/up and down gather.
    - Release row
      `docs/local/model-validation/20260513T110922Z-minimax-small-offset-kernels-multifile-release-longer-generation/`
      proves that the opt-in offset path is live on MiniMax-Small: coherent
      apples/rivers multi-turn, cache stack on, zero explicit active-slice
      reads, effective read pressure 559.95 MB/generated-token, and peak
      Activity Monitor 2.70 GB / 7.3%. It removes the old `stack.*` and
      `router.indices_readback` profile rows, but decode is still only
      1.07 tok/s. Remaining top rows are `reduce.call_chunk`, `gateup.total`,
      `gateup.eval`, `down.eval`, `router.scores_eval`, and
      `reduce.score_sum_eval`.
    - No-layer-eval diagnostic
      `docs/local/model-validation/20260513T111455Z-minimax-small-offset-kernels-multifile-release-no-layer-eval/`
      stayed coherent and low-RAM, improved prompt throughput to 11.93 tok/s,
      and passed read/Activity gates, but decode only improved to 1.16 tok/s.
      Its MLX allocator peak rose to 429.77 GB while Activity Monitor stayed
      near 2.73 GB, so eval-boundary tuning is a diagnostic lever, not the
      MiniMax speed fix.
    - The matching 8 GB tensor-cache + mmap-active row
      `docs/local/model-validation/20260513T095912Z-minimax-small-streaming-mmap-active-tensors-8gb-cache-longer-generation/`
      stayed coherent and very low-RAM (2.44 GB / 6.6%) with a 60.1% tensor
      hit rate, but decode regressed to 1.12 then 1.03 tok/s. Bounded tensor
      residency reduces mmap-array misses but does not remove the CPU router
      readback or active-bank/reduce path, so it is rejected as the MiniMax
      speed fix.
    - The 8 GB active expert tensor-cache row
      `docs/local/model-validation/20260513T070926Z-minimax-small-streaming-8gb-tensor-cache/`
      stayed coherent and passed the 30% Activity Monitor gate
      (10.45 GB / 28.2%). It reduced explicit reads from 254,261 MB to
      29,668 MB and passed the 1,000 MB/generated-token file/effective read
      gates with `cache_hit_rate=0.624`, `tensor_resident_mb=8191`, and
      `tensor_evictions=38047`, but decode stayed at 1.17 tok/s. This proves
      bounded residency can control read pressure, but current per-token
      router readback and active bank construction still dominate speed.
    - The matching no-layer-eval row
      `docs/local/model-validation/20260513T071450Z-minimax-small-streaming-8gb-no-layer-eval/`
      kept the same gates and cache hit rate but only improved decode to
      1.21 tok/s. Do not chase eval-boundary tuning as the main MiniMax fix.
    - The exact active-bank cache row
      `docs/local/model-validation/20260513T073632Z-minimax-small-streaming-8gb-tensor-8gb-bank-cache-glu/`
      ran the same 8 GB tensor cache plus an 8 GB bank cache after wiring the
      bank cache into `StreamingTurboQuantSwitchGLU`. Output stayed coherent
      and the relaxed 59% Activity Monitor gate passed (18.44 GB / 49.8%),
      but decode averaged 1.09 tok/s. Bank reuse was only 6 hits against
      17,478 misses, with 15,663 bank evictions. This is worse than the
      tensor-cache-only row and proves exact top-k bank caching is not the
      MiniMax speed path.
    - Historical JangPress notes in
      `../vmlx-swift-lm/Libraries/MLXLMCommon/Cache/JANGPRESS-PER-MODEL-RESULTS.md`
      are now the correct methodology target, with caveats: MiniMax Small
      measured 46.66 engine tok/s OFF, 47.12 tok/s with pct=0 armed, and
      43.74 tok/s with soft pct=70; RSS post-decode stayed about 5.4 GB and
      partial forceRelease left 9.4 GB resident after reclaiming 22.9 GB. That
      was the resident compute path plus JangPress cold-page reclamation, not
      current per-token active-expert streaming. In practical target terms,
      the old pct=70 MiniMax row is the "roughly 40 tok/s while well below a
      relaxed 59% original-size footprint" methodology to recover. The old row
      also had a chunk buffer output bug and recorded RSS rather than Activity
      Monitor-style `phys_footprint`, so it is a speed/RSS mechanism target,
      not a current coherence or low-Activity-Monitor proof.
    - The load-time Mach-stack probe
      `docs/local/model-validation/20260513T164210Z-minimax-small-loadtime-mach-stacks-resident-one-turn/`
      tested resident `TurboQuantSwitchGLU` compute with whole-projection
      Mach-backed stacks and active streaming off. It produced coherent output
      and improved decode to 7.99 tok/s, but failed the low-footprint gate
      after GPU touch: peak Activity Monitor was 35.34 GB / 95.4%, post-decode
      was 34.92 GB / 94.3%, and effective read pressure was 1,897.49
      MB/generated-token. Whole-projection anonymous purgeable stacks are too
      coarse and do not reproduce the old JangPress reclaim method.
    - The explicit mmap-prestack probe
      `docs/local/model-validation/20260513T165112Z-minimax-small-explicit-prestack-mmap-one-turn/`
      restacked 372 routed tensors (32.8 GB payload) into a row-local
      safetensors overlay, loaded it through `MLX_SAFETENSORS_MMAP=1`, and kept
      active streaming off. It produced coherent output at 12.27 tok/s with
      cache stack and TurboQuant KV on, peak Activity Monitor at 2.57 GB /
      6.9%, and zero explicit active reads. It still failed effective read
      pressure at 4,595.48 MB/generated-token and the overlay was only a
      diagnostic artifact.
    - The warm explicit mmap-prestack row
      `docs/local/model-validation/20260513T165237Z-minimax-small-explicit-prestack-mmap-warm-2turn/`
      reused that overlay for two turns before deleting it. It passed coherent
      red-apple and blue-sky output with cache stack on, peak Activity Monitor
      2.62 GB / 7.1%, explicit reads at zero, turn decode 12.75 and 14.57
      tok/s, and aggregate decode 13.66 tok/s. Effective read pressure still
      failed at 1,995.36 MB/generated-token, so it is not readiness.
    - The no-profile ephemeral mmap-prestack row
      `docs/local/model-validation/20260513T170143Z-minimax-small-ephemeral-prestack-mmap-noprofile-2turn/`
      reran the same resident-compute shape with `MLXPRESS_PRESTACK_EPHEMERAL=1`
      and no `MLXPRESS_GENERATION_PROFILE`. The temporary 32.8 GB overlay was
      removed at exit, coherence passed on both turns, peak Activity Monitor
      stayed 2.62 GB / 7.1%, explicit reads stayed zero, and decode was 12.60 /
      14.01 tok/s (aggregate 13.36). This proves per-token profiling was not
      the missing 39-47 tok/s factor. The current best MiniMax path is usable
      but still blocked by 1,997.41 MB/generated-token effective read pressure
      and speed below the historical resident JangPress target.
    - The no-cold ephemeral mmap-prestack row
      `docs/local/model-validation/20260513T170902Z-minimax-small-ephemeral-prestack-mmap-nocold-2turn/`
      removed `MLX_SAFETENSORS_MMAP_START_COLD=1` / force cold advice while
      keeping the same ephemeral overlay, resident `TurboQuantSwitchGLU`, cache
      stack, disk L2, TurboQuant KV, and active streaming off. It passed
      coherent two-turn output, kept peak Activity Monitor at 2.62 GB / 7.1%,
      kept explicit reads at zero, passed effective read pressure at 662.31
      MB/generated-token, and improved aggregate decode to 14.22 tok/s. This is
      now the strongest MiniMax diagnostic. It proves forced cold-start advice
      was counterproductive for this resident path, but it is still not
      production-ready because it uses an explicit temporary stacked overlay and
      remains far below the old 39-47 tok/s JangPress target.
    - KV-mode split on the no-cold resident path:
      `docs/local/model-validation/20260513T171226Z-minimax-small-ephemeral-prestack-mmap-nocold-kvnone-2turn/`
      kept cache stack/disk L2 on but used `--kv-cache none`. The same short
      two-turn prompts stayed coherent and low-footprint, with peak Activity
      Monitor 2.66 GB / 7.2%, effective read pressure 647.82
      MB/generated-token, and decode 24.56 / 24.44 tok/s. This proves
      TurboQuant KV is a large short-row latency cost on this path.
    - Sustained-output split:
      `docs/local/model-validation/20260513T171708Z-minimax-small-ephemeral-prestack-mmap-nocold-tqkv-long/`
      generated a coherent 129-token paragraph with TurboQuant KV at 19.70
      tok/s, peak Activity Monitor 2.59 GB / 7.0%, and effective read pressure
      236.15 MB/generated-token. The matching KV-none row
      `docs/local/model-validation/20260513T171915Z-minimax-small-ephemeral-prestack-mmap-nocold-kvnone-long/`
      generated 118 coherent tokens at 21.61 tok/s, peak 2.51 GB / 6.8%, and
      effective read pressure 258.01 MB/generated-token. Therefore TurboQuant
      KV is a major short-turn cost, but on sustained rows it explains about a
      10% gap, not the whole distance to the historical 39-47 tok/s target.
    - The profiled sustained TurboQuant-KV row
      `docs/local/model-validation/20260513T172505Z-minimax-small-ephemeral-prestack-mmap-nocold-tqkv-long-profile/`
      produced the same coherent 129-token paragraph at 19.07 tok/s. Its
      generation profile shows `decode.async_eval_submit` dominating with
      6,161.4 ms total / 47.396 ms average, while `decode.kv_quantize` was
      352.6 ms total / 2.733 ms average and `decode.model_forward` graph build
      was 289.7 ms total / 2.211 ms average. The remaining speed gap is
      therefore resident graph/eval behavior under `TurboQuantSwitchGLU` and
      attention, not TurboQuant KV alone.
    - Compiled MiniMax decode before cache promotion was rejected. Diagnostic row
      `docs/local/model-validation/20260513T175200Z-minimax-small-cli-ephemeral-prestack-compiled-kvnone-profile-strict-2turn/`
      used `--compiled-decode on`, `--allow-minimax-compiled-decode`,
      `--compiled-max-cache-length 512`, `--kv-cache none`, and the same
      no-cold `--ephemeral-prestack on` resident path. The compiled closure did
      engage (`decode.compiled_forward` averaged 0.906 ms and 0.976 ms), peak
      Activity Monitor passed at 2.58 GB / 6.943%, explicit reads were zero,
      and decode reached 24.56 / 24.64 tok/s. The row is a failure because both
      turns looped and length-stopped: red apple repeated as `A red apple that
      is...`, and sky repeated as `The sky is a short sentence...`. This proves
      a faster compiled MiniMax row is not acceptable until compiled/uncompiled
      parity is fixed.
    - The follow-up cache-promotion patch fixed that single-sequence compiled
      loop. Row
      `docs/local/model-validation/20260513T180400Z-minimax-small-cli-ephemeral-prestack-compiled-kvnone-promotedcache-strict-2turn/`
      promotes `KVCacheSimple` to `CompilableKVCache` before compiling, keeps
      the same no-cold `--ephemeral-prestack on` path and `--kv-cache none`,
      and passes strict two-turn coherency. Both turns ended with `stop=stop`;
      outputs were `A red apple is an apple that is red in color.` and the
      Rayleigh-scattering sky answer. Decode reached 26.29 / 26.26 tok/s, peak
      Activity Monitor was 2.64 GB / 7.109%, explicit reads were zero, and
      effective read pressure passed at 845.69 MB/generated-token. This is now
      the best strict low-footprint MiniMax speed diagnostic, but it is still
      not production readiness because it disables TurboQuant KV, depends on
      the temporary overlay, and remains below the old 39-47 tok/s target.
      The longer guard row
      `docs/local/model-validation/20260513T180900Z-minimax-small-cli-ephemeral-prestack-compiled-kvnone-promotedcache-longer-2turn/`
      generated 107 coherent visible tokens over two turns with no loops and
      both turns `stop=stop`: 24.15 / 25.62 tok/s, peak Activity Monitor
      2.79 GB / 7.510%, zero explicit reads, and 284.56 MB/generated-token
      effective read pressure. Treat that as the stronger no-loop evidence,
      and the 26.27 tok/s short row as the short-turn speed datapoint.
    - The same single-sequence compiled-cache promotion now covers TurboQuant
      KV, so the cache-stack diagnostic no longer has to disable KV
      compression. Strict row
      `docs/local/model-validation/20260513T181400Z-minimax-small-cli-ephemeral-prestack-compiled-tqkv-promotedcache-strict-2turn/`
      passed with `default-kv=turboQuant(k=3,v=3)`, both turns `stop=stop`,
      26.95 / 24.11 tok/s, peak Activity Monitor 2.53 GB / 6.818%, zero
      explicit reads, and 845.65 MB/generated-token effective read pressure.
      Longer guard row
      `docs/local/model-validation/20260513T181800Z-minimax-small-cli-ephemeral-prestack-compiled-tqkv-promotedcache-longer-2turn/`
      generated 169 coherent visible tokens over two turns with no loops:
      22.03 / 23.13 tok/s, peak Activity Monitor 2.63 GB / 7.089%, zero
      explicit reads, and 180.10 MB/generated-token effective read pressure.
      This row is now superseded for speed diagnosis. The root cause was not
      compiled TurboQuant KV; `--activity-gate 59` was also shrinking
      `MLX.Memory.memoryLimit` to a gate-derived cap, which throttled compiled
      decode even though Activity Monitor footprint was already low.
    - Activity Monitor gating is now measurement/enforcement only and no
      longer rewrites the MLX memory budget. The strict cache-stack follow-up
      `docs/local/model-validation/20260513T192000Z-minimax-small-mlxpress-cache-stack-compiled-tqkv-activitygate-unthrottled-2turn-stop/`
      passed with `default-kv=turboQuant(k=3,v=3)`, disk L2, compiled decode,
      `--fail-on-length-stop`, and `--activity-gate 59`. Both turns ended with
      `stop=stop`, visible text was coherent with no loops, decode reached
      49.09 / 48.66 tok/s, peak Activity Monitor stayed at 2.65 GB / 7.159%,
      explicit reads were zero, and effective read pressure was 0.09
      MB/generated-token. This closes the old MiniMax speed/RAM target for the
      explicit ephemeral-prestack diagnostic path.
    - The cleanup-after-load follow-up
      `docs/local/model-validation/20260513T193500Z-minimax-small-mlxpress-ephemeral-cleanup-after-load-compiled-tqkv-2turn/`
      proves the temporary overlay can be removed immediately after the mmap
      loader establishes mappings. The row logs both overlay creation and
      removal before decode, then stays coherent for two turns at
      47.54 / 47.19 tok/s with `default-kv=turboQuant(k=3,v=3)`, peak Activity
      Monitor 2.64 GB / 7.131%, zero explicit reads, and
      585.69 MB/generated-token effective read pressure.
    - MiniMax is still not final production readiness because this fastest
      path still writes a temporary 32.8 GB prestacked overlay during load,
      even though it now deletes it immediately after mapping. The remaining
      production work is to replace that load-time overlay with the no-overlay
      resident mmap/offset kernel path while preserving the same speed,
      Activity Monitor footprint, cache stack, and coherency behavior.
    - The Mach active tensor diagnostic
      `docs/local/model-validation/20260513T120747Z-minimax-small-mach-active-tensors-noprofile-longer-generation/`
      wired real active JANGTQ tensor bytes through `MLXPressMachCache` with
      `MLXPRESS_STREAMING_MACH_ACTIVE_TENSORS=1`. It passed coherence and the
      Activity Monitor gate (10.85 GB / 29.3%), removed explicit active tensor
      file reads, and held effective read pressure to 898.74 MB/generated
      token. It is rejected as the MiniMax speed fix because decode was only
      0.106 then 0.085 tok/s and `reduce.call_chunk` dominated 802.8 s across
      5,208 calls. The problem is now the active-streaming execution structure,
      not explicit SSD streaming.
    - The offset-kernel eval-stride diagnostic
      `docs/local/model-validation/20260513T124000Z-minimax-small-offset-kernels-eval-stride4-short/`
      tested `MLXPRESS_STREAMING_EVAL_LAYER_STRIDE=4`. It kept peak Activity
      Monitor footprint low at 2.52 GB / 6.8% and explicit reads at zero, but
      failed the overall coherency gate because turn 1 length-stopped and
      aggregate decode was only 0.909 tok/s. The skipped layer work moved into
      `router.scores_eval` (63.2 s total), so stride tuning is rejected as the
      current MiniMax speed fix.
    - The scored-down offset kernel row
      `docs/local/model-validation/20260513T131500Z-minimax-small-offset-scored-down-longer-generation/`
      enabled `MLXPRESS_STREAMING_SCORED_DOWN_KERNELS=1`, fused router-score
      reduction into the offset down-proj kernel, and passed coherent two-turn
      output with cache stack on. It kept peak Activity Monitor footprint at
      2.53 GB / 6.8%, explicit reads at zero, and effective read pressure at
      623.56 MB/generated token. Decode improved to 1.150 tok/s aggregate
      versus the prior passing offset row's 1.071 tok/s. This is a valid
      incremental kernel win, not the final resident-speed fix.
    - The scored-down no-eval row
      `docs/local/model-validation/20260513T132500Z-minimax-small-offset-scored-down-no-eval-longer-generation/`
      passed coherent two-turn output and low Activity Monitor footprint
      (2.64 GB / 7.1%) with zero explicit reads and 1.250 tok/s aggregate
      decode. It is rejected as a production default because MLX allocator
      peak rose to 1.57 TB. Use it as the current offset-graph upper bound and
      evidence that the next scheduler must be allocator-aware.
    - Active-expert streaming now has an allocator-pressure materialization
      scheduler for that exact middle ground. `MLXPRESS_STREAMING_EVAL_MLX_PEAK_MB`
      / `JANGPRESS_STREAMING_EVAL_MLX_PEAK_MB` and
      `MLXPRESS_STREAMING_EVAL_MLX_ACTIVE_MB` /
      `JANGPRESS_STREAMING_EVAL_MLX_ACTIVE_MB` let an
      `MLXPRESS_STREAMING_EVAL_EACH_LAYER=0` row keep the no-eval decode
      shape until MLX allocator peak or active memory crosses the configured
      limit. The scheduler records `scheduler.allocator_eval` profile rows,
      reuses the same materialization path as forced per-layer eval, clears
      the MLX cache, resets peak accounting by default, and releases Mach cold
      tiles when the down-proj boundary materializes. This is not a speed proof
      until a new MiniMax/Hy3/Kimi row records coherent multi-turn output,
      token/s, Activity Monitor footprint, MLX peak, and effective read
      pressure with the knob enabled.
    - The first allocator-pressure row
      `docs/local/model-validation/20260513T134510Z-minimax-small-offset-scored-down-allocator-peak128/`
      enabled `MLXPRESS_STREAMING_EVAL_MLX_PEAK_MB=131072` with scored-down
      offset kernels and `MLXPRESS_STREAMING_EVAL_EACH_LAYER=0`. It passed
      coherent two-turn output, zero explicit reads, 844.42 MB/generated-token
      effective read pressure, 0 MB swap-out, and 2.53 GB / 6.8% peak Activity
      Monitor footprint. MLX peak stayed bounded at 128.69 GB and
      `scheduler.allocator_eval` fired 73 times. Decode was 1.241 tok/s
      aggregate, so this proves bounded scheduler behavior but rejects
      allocator-pressure eval as the old JangPress speed recovery by itself.
    - Active-shard filtering is now available as a diagnostic through
      `MLXPRESS_STREAMING_OFFSET_ACTIVE_SHARD_FILTER=1` /
      `JANGPRESS_STREAMING_OFFSET_ACTIVE_SHARD_FILTER=1`. It reads routed
      expert ids for offset-dispatch chunks, computes one shared active file
      set across paired gate/up/norm descriptors, and skips offset shard files
      that cannot contain active experts. The first attempted row
      `docs/local/model-validation/20260513T135500Z-minimax-small-offset-active-shard-filter-peak128/`
      failed before generation because independent projection filtering broke
      the fused gate/up same-file invariant; the corrected row
      `docs/local/model-validation/20260513T135912Z-minimax-small-offset-active-shard-filter-peak128/`
      passed coherent two-turn output, zero explicit reads, 843.77
      MB/generated-token effective read pressure, 0 MB swap-out, and 2.44 GB /
      6.6% peak Activity Monitor footprint. It is rejected as a speed fix:
      decode fell to 1.134 tok/s because `router.indices_readback` dominated
      33.95 s. It is useful evidence that CPU readback is worse than launching
      the inactive offset shard partials unless future GPU-side shard selection
      removes the synchronization.
  - Thinking-on is not production-passing yet. Probe
    `docs/local/model-validation/20260513T025104Z/` kept low RAM and produced
    coherent reasoning preview, but length-stopped at 48 generated tokens and
    leaked reasoning-style text into visible output instead of answering
    exactly `4`. Treat Kimi reasoning mode as parser/template blocked.
- Earlier sibling `mlxpress` package evidence has focused cache-stack-on smokes for:
  - MiniMax M2.7 Small JANGTQ: `ready` / `done`, peak 2.62 GB.
  - MiniMax M2.7 JANGTQ_K: `ready` / `done`, peak 2.04 GB.
  - Hy3 preview JANGTQ: `ready` / `done`, peak 4.66 GB.
  - ZAYA JANGTQ_K: visible output `2` with thinking disabled.
- Reasoning-on cache-stack short deterministic proofs now pass in `mlxpress` for:
  - `MiniMax-M2.7-JANGTQ_K-CRACK`: full MLXPress, disk L2, TurboQuant KV,
    multi-turn `4` / `6`, peak 2.16 GB, prompt 122.28 / 126.10 tok/s. Each
    turn had expected visible output, stop token, and `unclosed-reasoning=false`,
    but the generated turns were only 10 and 6 tokens, so this is not a
    long no-loop proof.
  - `Hy3-preview-JANGTQ`: full MLXPress, disk L2, TurboQuant KV, multi-turn
    `4` / `6`, peak 4.46 GB, prompt 51.84 / 106.99 tok/s.
- MiniMax CRACK K long no-loop status in the combined repo:
  - Thinking off: pass. Cold/warm MLXPress disk-L2 row
    `docs/local/deviation/20260512T214248Z-minimax-crack-k-long-loop-thinking-off/RESULT.md`
    produced identical two-turn visible checklists, `loop=pass`,
    `min-generated=pass`, `length-stop=pass`; cold prompt avg 95.75 tok/s,
    warm prompt avg 388.13 tok/s, decode 6.18 / 5.61 tok/s, peak 2.07 /
    2.34 GB.
  - Thinking on: not production-passing for final-answer closure yet. Long
    probes produced coherent, non-looping reasoning text
    (`reasoning-loop=pass`) but no visible final answer before max tokens, so
    strict validation returns exit 3. Treat this as runtime/channel sanity plus
    prompt/template closure work, not a green production row.
- Cold/warm deviation harness rows now pass in `mlxpress` with isolated L2 disk
  directories and token/s recorded:
  - `MiniMax-M2.7-Small-JANGTQ`: cache-off / cold / warm output all `ready`,
    prompt 12.75 / 56.55 / 92.63 tok/s, decode 26.38 / 0.26 / 0.27 tok/s.
  - `MiniMax-M2.7-JANGTQ_K-CRACK`: `4` / `4`, prompt 8.56 -> 120.99 tok/s,
    decode 1.88 -> 1.73 tok/s.
  - `Hy3-preview-JANGTQ` with `reasoning_effort=low`: `4` / `4`, prompt
    5.39 -> 52.16 tok/s, decode 1.21 -> 1.18 tok/s.
  - `Hy3-preview-JANGTQ` with `reasoning_effort=no_think`: single-turn
    `4` / `4` and multi-turn `4/6` / `4/6`; multi-turn prompt
    36.00 -> 59.75 tok/s, decode 0.60 -> 0.60 tok/s.
  - `Hy3-preview-JANGTQ` with `reasoning_effort=high`: single-turn `4` / `4`
    and multi-turn `4/6` / `4/6`; multi-turn prompt 69.89 -> 119.74 tok/s,
    decode 1.17 -> 1.17 tok/s.
- Hy3 preview JANGTQ_K is partially unblocked in the combined repo but not
  production-passing yet:
  - Historical active-streaming fallback skips 91,008 per-expert tensors.
    Rerun Hy3 K on the compression-first canonical mmap path before promoting
    this as an MLXPress success row.
  - Post-load footprint is now 4.72 GB / 4.7% of 101.52 GB safetensors, down
    from the previous no-prestack 93.01 GB footprint.
  - Non-streaming prestack is not acceptable for this bundle: the first
    `Hy3-preview-JANGTQ_K` prestacked overlay wrote 89.1 GB of routed tensors
    and post-load Activity Monitor footprint reached 93.5 GB / 92.1% before
    decode. Do not treat that path as MLXPress-compatible for Hy3 K.
  - Strict 30% peak gate still fails during decode: best hard-gate run reached
    30.83 GB / 30.4% before generation completed.
  - Report-only run decoded but was not coherent: prompt 0.52 tok/s, decode
    0.37 tok/s, output `2 + 2` for `2+2?`, expected `4`, peak 74.49 GB.
  - Offset-dispatch shard-alignment is now guarded. The first Hy3 K offset row
    failed before generation because gate/up shard groups were not aligned for
    the fused offset path. The corrected row
    `docs/local/model-validation/20260513T141501Z-hy3-k-offset-alignment-fallback-no-thinking-short/`
    records `offset_dispatch_unaligned_shards` instead of crashing, returns
    coherent `4` / `6` with cache stack on, and keeps peak Activity Monitor
    footprint at 4.44 GB / 4.4%. It is still rejected as usable MLXPress:
    decode is only 0.106 / 0.102 tok/s, MLX peak is 129.90 GB, and effective
    read pressure fails at 217,923 MB/generated token from page-ins. Hy3 K
    needs GPU-side shard-aware offset selection or a resident-compute path, not
    CPU-side shard filtering or the current fallback page-in churn.
  - Active-shard filtering is also rejected for Hy3 K. Row
    `docs/local/model-validation/20260513T142627Z-hy3-k-offset-active-shard-filter-no-thinking-short/`
    stayed coherent (`4` / `6`) and low-footprint (4.38 GB / 4.3%), improved
    aggregate decode only from 0.103 to 0.118 tok/s, and reduced effective read
    pressure only from 217,923 to 163,286 MB/generated token. The profile moved
    the hot cost into `router.indices_readback` at 51.63 s over 474 calls.
    This confirms the shard filter must be GPU-side if it is used at all.
  - Active-window offset-span narrowing improves the rejected CPU-side filter
    but does not make it ready. Row
    `docs/local/model-validation/20260513T143837Z-hy3-k-offset-active-window-filter-no-thinking-short/`
    narrowed each offset mmap span to the active expert byte window, stayed
    coherent (`4` / `6`) with cache stack, disk L2, and TurboQuant KV on, kept
    peak Activity Monitor footprint at 4.38 GB / 4.3%, and improved aggregate
    decode from 0.118 to 0.558 tok/s. It skipped 519,023 MB of offset-span
    bytes and cut page-ins from 326,573 MB to 90,082 MB, but effective read
    pressure still failed at 45,041 MB/generated token and
    `router.indices_readback` still took 16.31 s over 474 calls.
  - The post-MiniMax activity-gate fix rerun
    `docs/local/model-validation/20260513T201500Z-hy3-k-segmented-groups-unthrottled-no-thinking-short/`
    confirms Hy3 K is not blocked by the old gate-derived MLX memory limit.
    It stayed coherent (`4` / `6`) with cache stack and TurboQuant KV on, held
    peak Activity Monitor footprint to 4.41 GB / 4.336%, and had zero explicit
    active tensor reads, but aggregate decode only moved from 0.729 to
    0.792 tok/s and effective read pressure stayed failed at
    14,952.85 MB/generated-token. The top profile rows remain
    `router.indices_readback`, `reduce.call_chunk_scored`, and
    `tensor.offset_span_mmap_array`; Hy3 K needs GPU-side active-window
    selection or resident compute semantics, not more activity-gate tuning.
  - Active offset-window coalescing is now implemented as an explicit
    speed/RAM tradeoff knob for the fallback path. `JANGTQActiveOffsetWindow`
    and `JANGTQStackedOffsetDescriptor.activeExpertByteWindows(...)` keep
    one active expert per window by default; coalescing is opt-in with
    `MLXPRESS_STREAMING_OFFSET_ACTIVE_WINDOW_COALESCE_MB` /
    `JANGPRESS_STREAMING_OFFSET_ACTIVE_WINDOW_COALESCE_MB`. Row
    `docs/local/model-validation/20260513T191100Z-hy3-k-offset-window-coalesce16-no-thinking-short/`
    proves the knob is real but not solved: coherence and Activity Monitor
    gates held, decode improved to 1.004 tok/s, but effective read pressure
    worsened to 29,345.11 MB/generated-token. A rejected current-code row
    with the former adjacent-by-default behavior,
    `docs/local/model-validation/20260513T192000Z-hy3-k-offset-window-coalesce0-current-no-thinking-short/`,
    reproduced the same 29,355.32 MB/generated-token pressure even with the
    knob at 0 MB. That proved adjacent merge was too broad for Hy3, so default
    coalescing is disabled. The corrected default row
    `docs/local/model-validation/20260513T193000Z-hy3-k-offset-window-default-no-coalesce-current-no-thinking-short/`
    restored the segmented shape: coherent `4` / `6`, 0.781 tok/s decode,
    4.320% peak Activity Monitor footprint, 22,803.84 MB mapped spans, and
    14,948.93 MB/generated-token effective read pressure. Smaller opt-in gap
    sweeps or a GPU-side active-window path are still needed.
  - Flexible unaligned-shard offset grouping is now created for Hy3-style
    split layouts, but it is not a speed fix by itself. Row
    `docs/local/model-validation/20260513T150028Z-hy3-k-flexible-offset-groups-no-thinking-short/`
    grouped gate/up/down spans by shared expert coverage instead of identical
    file URL sets, recorded `offset_dispatch_flexible_shard_groups=6`, stayed
    coherent (`4` / `6`), and kept peak Activity Monitor footprint at
    4.41 GB / 4.3%. It is rejected because decode stayed at 0.112 tok/s,
    page-ins were 332,258 MB, `tensor.offset_span_mmap_array` mapped
    1,347,166 MB of spans, and effective read pressure failed at
    166,129 MB/generated token. This confirms that unaligned Hy3 K shards can
    stay on offset kernels, but broad full-span mmap is still the wrong
    performance shape without active-window narrowing or resident compute.
  - Segmented active-window spans are now correctness-fixed for Hy3-style
    unaligned shard sets. The first two segmented attempts
    `docs/local/model-validation/20260513T151616Z-hy3-k-segmented-active-window-no-thinking-short/`
    and
    `docs/local/model-validation/20260513T151837Z-hy3-k-single-expert-active-window-no-thinking-short/`
    are rejected because grouping by file set only overwrote multiple
    single-expert spans and produced empty visible output. The fixed row
    `docs/local/model-validation/20260513T152335Z-hy3-k-segmented-groups-fixed-no-thinking-short/`
    adds expert coverage to the group key, stays coherent (`4` / `6`) with
    cache stack, disk L2, and TurboQuant KV on, keeps peak Activity Monitor
    footprint at 4.39 GB / 4.3%, and improves aggregate decode to 0.729 tok/s.
    It maps only 22,804 MB of offset spans and cuts page-ins to 29,889 MB,
    down from 90,082 MB in the prior active-window row and 332,258 MB in the
    broad-span row. It is still rejected for readiness because effective read
    pressure remains 14,944 MB/generated token and `router.indices_readback`
    still costs 5.43 s over 474 calls.
  - Therefore Hy3 JANGTQ_K is load/coherency-fixed on the short no-thinking
    offset rows, but speed/effective-read-pressure-blocked.
- Reasoning-on generation is still not confirmed for ZAYA. MiniMax CRACK K
  also needs an explicit prompt instruction to close `</think>` for short
  deterministic validation.
- `Kimi-K2.6-Small-JANGTQ` has historical active-streaming fallback evidence,
  but the compression-first canonical mmap path still needs fresh low-RAM,
  low-file-pressure generation proof before promotion:
  - Inspect: routed JANGTQ, `model_type=kimi_k25`, 211 routed experts, top-k 8,
    142.58 GB effective safetensors bytes.
  - Inspect now exposes `matmulKinds` as JANGTQ Hadamard rotation,
    randomized Hadamard power-of-two blocks, runtime sidecar signs/codebooks,
    TurboQuant codebook gather over `tq_packed`/`tq_norms`, active-expert
    slice matmul, and fused gate/up SwiGLU. `attentionKinds` includes
    `mla-partial-rope-attention`; `positionVectorKinds` includes the scalar
    token offset plus `partial-rope-head-vector(qk=64)`.
  - Historical active-streaming fallback rows indexed 60 routed layers /
    211 experts and skipped 75,960 per-expert tensors during weight load.
    Those rows are now fallback evidence, not the default success path.
  - Historical default-policy proof before the compression-first correction:
    `docs/local/model-validation/20260512T222112Z-kimi-small-default-active/RESULT.md`
    shows active streaming selected without `--active-expert-streaming on`,
    `stackedLayers=0`, no permanent prestack path, and post-load gate pass.
    Rerun this as a compression-first mmap row before promoting Kimi Small.
  - Post-load Activity Monitor footprint was 1.23 GB / 0.9% with MLXPress,
    disk L2, and TurboQuant KV enabled.
  - After no-cache active-slice reads, the one-token probe
    `docs/local/model-validation/20260512T222606Z-kimi-small-default-active-nocache-read/RESULT.md`
    passed the 30% gate with peak footprint 1.57 GB / 1.1%, but it emitted no
    visible output before the cap and is not a coherence row.
  - The real no-thinking short-answer row
    `docs/local/model-validation/20260512T222803Z-kimi-small-no-thinking-short-answer/RESULT.md`
    still failed the hard 30% gate before stdout, with peak footprint
    43.24 GB / 30.3%.
  - The diagnostic layer cold-sweep row
    `docs/local/model-validation/20260512T223823Z-kimi-small-no-thinking-short-answer-layer-sweep/RESULT.md`
    also failed at 43.24 GB / 30.3%, so layer-level mmap advice is not the
    missing Kimi memory fix.
  - A report-only short-answer row
    `docs/local/model-validation/20260512T224306Z-kimi-small-no-thinking-short-answer-report-only/RESULT.md`
    ran past the gate, peaked at 43.43 GB / 30.5%, emitted no stdout, emitted
    no token/s telemetry, and ended with exit 137.
  - A more aggressive cold-fraction row with `--mlxpress 80` still failed at
    43.25 GB / 30.3%, so the miss is not fixed by the default 70% cold
    fraction being too low.
  - A 512 MiB malloc-trace row
    `docs/local/model-validation/20260512T231130Z-kimi-small-malloc-trace-512m/`
    found repeated 0.59 GiB `concatenate_gpu` allocations through
    `StreamingTurboQuantSwitchGLU.callChunk` during DeepSeek/Kimi JANGTQ
    decode. A source change now precomputes routed expert ids once per MoE
    call, but the strict short-answer row
    `docs/local/model-validation/20260512T231800Z-kimi-small-precomputed-indices-short-answer/`
    still failed at 43.24 GB / 30.3%.
  - A smaller prefill window row
    `docs/local/model-validation/20260512T232430Z-kimi-small-prefill8-short-answer/`
    also failed at 43.25 GB / 30.3%.
  - The Kimi standalone `chat_template.jinja` uses `thinking`, while the
    tokenizer config bridges `enable_thinking` to `thinking`. The CLI now sends
    both aliases for every row.
  - Base active-streaming short no-thinking proof after that fix:
    `docs/local/model-validation/20260513T000500Z-kimi-small-base-thinking-alias-4tok/RESULT.md`
    produced visible `4`, prompt 25 tok at 0.40 tok/s, decode 1 tok at
    0.41 tok/s, `reasoning-chars=0`, and 1.56 GB / 1.1% peak footprint.
  - MLXPress/cache-stack short no-thinking proof:
    `docs/local/model-validation/20260513T000900Z-kimi-small-mlxpress-cache-thinking-alias-4tok/RESULT.md`
    produced visible `4`, prompt 25 tok at 0.43 tok/s, decode 1 tok at
    0.10 tok/s, `reasoning-chars=0`, 130.08 GB routed bytes under management,
    paged KV + disk L2 + TurboQuant KV, and 1.86 GB / 1.3% peak footprint.
  - Longer MLXPress/cache-stack no-thinking rows remain blocked:
    `docs/local/model-validation/20260513T002000Z-kimi-small-mlxpress-cache-thinking-alias-longer/RESULT.md`
    and
    `docs/local/model-validation/20260513T003100Z-kimi-small-mlxpress-cache-single-expert-expanded-longer/RESULT.md`
    both emitted only `+` before failing at 43.14 GB / 30.3%.
  - Avoiding `MLX.stacked` for one-expert streaming chunks did not lower that
    peak. The follow-up 512 MiB malloc trace
    `docs/local/model-validation/20260513T004200Z-kimi-small-mlxpress-cache-alias-malloc-trace-512m/RESULT.md`
    enabled the trace hook but emitted no >512 MiB allocation records before
    the gate failed, so the next Kimi memory pass should inspect many smaller
    Metal allocations, command-buffer lifetime, RoPE/KV update residency, and
    file/cache residency around active-streaming TQ matmul.
  - Kimi is now partial, not base-decode blocked. It still needs longer
    decode-memory reduction plus thinking, tool-call, multi-turn, and
    cold/warm cache-hit rows with token/s and strict no-loop checks.
  - A 128 MiB forced-longer malloc trace
    `docs/local/model-validation/20260514T000210Z-kimi-small-forced-longer-malloc-trace-128m/RESULT.md`
    reproduced the failure with repeated `&` output, no telemetry row, peak
    42.78 GB / 30.0%, 296 large allocation records, and `concatenate_gpu`
    through `StreamingTurboQuantSwitchGLU.reduced` at router index readback.
  - The router-readback materialization barrier row
    `docs/local/model-validation/20260514T001355Z-kimi-small-forced-longer-router-barrier-trace-128m/RESULT.md`
    still failed: repeated `&` output, peak 43.2 GB / 30.3%, 294 large
    allocation records, same `StreamingTurboQuantSwitchGLU.reduced` stack.
  - A stronger diagnostic barrier that evaluated the original `x` tensor first
    was worse and was reverted:
    `docs/local/model-validation/20260514T001948Z-kimi-small-forced-longer-original-x-barrier-trace-128m/RESULT.md`
    peaked at 43.74 GB / 30.7% with 298 large allocations.
  - Moving the router-index CPU readback inside the reduce chunk was also worse
    and was reverted:
    `docs/local/model-validation/20260514T002812Z-kimi-small-forced-longer-chunk-local-index-trace-128m/RESULT.md`
    peaked at 43.76 GB / 30.7% with 299 large allocations.
  - Current conclusion: Kimi longer rows are blocked at both gates. The next
    memory fix should change the router-index readback/precomputed-routing
    strategy or streaming-MoE prefill graph residency, not add more eager
    materialization or sliced-readback variants.
  - Current streaming fallback fix: `DeepseekV3JANGTQMoE` chunks the routed MoE
    input before `MoEGate`, so explicit Kimi/DeepSeek active-expert streaming
    runs router evaluation, router-index CPU readback, active expert slice
    loading, TurboQuant gather, and shared-expert MLP on small token chunks. The
    chunk size defaults to 16 and honors `MLXPRESS_STREAMING_MOE_TOKEN_CHUNK_SIZE`
    and the legacy `VMLX_JANGTQ_PREFILL_STEP`; chunk sync defaults on and
    honors `MLXPRESS_STREAMING_MOE_PREFILL_SYNC` / `VMLX_JANGTQ_PREFILL_SYNC`.
  - Validation after that pre-router chunking:
    `docs/local/model-validation/20260513T012521Z/` kept post-load low at
    1.23 GB / 0.9%, but decode still failed the 30% gate at 43.2 GB / 30.3%.
    The partial stdout was `uuualphaalp`, with no completed token/s row. That
    means the remaining Kimi blocker is per-token decoder graph/Metal
    residency and coherency during routed decode, not just full-prompt routing.
  - Current source fix: streaming JANGTQ decoder layers now materialize and
    clear each layer output by default through `MLXPRESS_STREAMING_EVAL_DECODER_LAYER`
    / `MLXPRESS_STREAMING_EVAL_EACH_LAYER`, unless the caller explicitly sets
    the env override to `0`. The after-env-fix no-eval diagnostic proves the
    knob is respected, but it only covered a one-token coherent row. Longer
    strict Kimi rows are still needed before changing the default boundary
    policy.
  - Current stacked-tensor fix: stacked `switch_mlp.{gate,up,down}_proj`
    fallback reads only the requested expert slice from safetensors by offset
    instead of reading the full stacked tensor into an MLX array and then
    slicing. This keeps the explicit fallback bounded, but the target path is
    still compression-first canonical mmap residency.
- `Kimi-K2.6-JANGTQ_K` is the current target gate, and it is still not ready:
  - Inspect: routed JANGTQ VL wrapper over Kimi K2.x text, 384 routed experts,
    top-k 8, 328.11 GB model bytes, MLA partial-RoPE attention with
    `qk_rope_head_dim=64`, 60 routed layers, dense shared experts, and media
    cache blockers for the VL path. Text rows do not enter the VL projector.
  - Baseline offset active-window row
    `docs/local/model-validation/20260514T010000Z-kimi-k-offset-active-window-1tok/`
    produced coherent `red`, kept peak Activity Monitor footprint to 1.44 GB /
    0.439%, and eliminated explicit reads, but failed effective read pressure
    at 73,493.88 MB/generated token from page-ins.
  - Active-window offset-span cache is now wired for route-specific windows.
    Row
    `docs/local/model-validation/20260513T195055Z-kimi-k-offset-active-window-cache8gb-1tok/`
    produced coherent `red`, kept peak footprint to 1.79 GB / 0.545%, and cut
    effective read pressure to 28,077.70 MB/generated token. It is rejected
    because decode was only 0.398 tok/s and the page-in gate still failed.
  - The larger compression-thesis row
    `docs/local/model-validation/20260513T195406Z-kimi-k-offset-active-window-cache32gb-short/`
    produced a coherent 30-token sentence with no loop, cache stack and
    TurboQuant KV on, and peak Activity Monitor footprint only 2.41 GB /
    0.735%. It is rejected because decode was 0.321 tok/s and effective read
    pressure still failed at 4,780.81 MB/generated token.
  - The 32 GB row is the key Kimi diagnosis: even a short coherent generation
    touched most of the routed expert set (`active_expert_trace`: 25,440
    routed slots, 21,196 unique expert touches, 0.263 reuse rate), so a
    file-backed active-window LRU still evicted 60,895 spans and page-in churn
    stayed high. Bigger file-backed LRU tuning is not the production method.
    Kimi needs anonymous/purgeable resident storage or a persistent
    offset-addressed kernel path that avoids file-backed refaults, plus the
    MLA absorb-decode follow-up for the attention side.
  - New next diagnostic: `MLXPRESS_STREAMING_MACH_OFFSET_SPANS=1` /
    `JANGPRESS_STREAMING_MACH_OFFSET_SPANS=1` registers exact one-expert
    active offset windows into Mach purgeable anonymous storage. This should
    be tested with offset kernels, active shard filtering, scored-down kernels,
    cache stack/TurboQuant KV, active expert trace, and metrics JSONL. Passing
    still requires coherent output plus prompt/decode token/s, low Activity
    Monitor footprint, and improved effective read pressure; first-turn
    `tensor.mach_offset_read` bytes are expected because the anonymous tiles
    must be populated from the real safetensors.
  - First Mach offset-span row
    `docs/local/model-validation/20260513T202041Z-kimi-k-mach-offset-spans-1tok/`
    proves wiring, not readiness. It produced coherent `red`, held peak
    Activity Monitor footprint to 6.02 GB / 1.834%, registered 6,432 Mach
    offset tensors, and cut effective read pressure versus the no-cache offset
    row to 24,819.72 MB/generated-token. It is rejected because decode was
    only 0.368 tok/s, slower than the 0.955 tok/s no-cache offset baseline,
    and still failed the 1,000 MB/generated-token effective-read gate.
    The run exposed and fixed a measurement gap: `tensor.mach_offset_read` is
    now included in explicit file-read pressure and summary read counters.
  - Same-process warm test
    `docs/local/model-validation/20260513T202514Z-kimi-k-mach-offset-spans-2turn/`
    is also rejected. It produced coherent `red` / `blue` and kept peak
    Activity Monitor to 20.88 GB / 6.364%, but aggregate decode fell to
    0.209 tok/s, turn 2 decoded at only 0.144 tok/s, resident Mach offset bytes
    grew to 60.73 GB, explicit file-read pressure was 30,363.52 MB/generated
    token, and effective read pressure was 30,871.54 MB/generated-token. The
    classifier fix is confirmed (`rows=tensor.mach_offset_read`), and the
    warm-hit row shows that unbounded Mach offset residency is not enough for
    Kimi without a scheduler/eviction/prefetch policy or lower-level persistent
    offset kernels.
  - Source follow-up after the rejected warm row: Mach tile registration now
    supports direct `pread` into the purgeable region through
    `JangPressMachCache.registerFilled`, exact Mach offset spans can be capped
    with `MLXPRESS_STREAMING_MACH_OFFSET_SPAN_BUDGET_MB` /
    `JANGPRESS_STREAMING_MACH_OFFSET_SPAN_BUDGET_MB`, and full routed-span
    residency can be prewarmed with
    `MLXPRESS_STREAMING_MACH_FULL_OFFSET_SPANS=1` plus
    `MLXPRESS_STREAMING_MACH_PREWARM_FULL_OFFSET_SPANS=1`. The direct-fill fix
    also chunks large Darwin `pread` calls so full Kimi descriptors no longer
    fail with `errno=22` on multi-GB routed tensors.
  - Direct-fill validation:
    `docs/local/model-validation/20260513T212055Z-kimi-k-fullspan-prewarm-1layer-after-chunked-pread/`
    prewarmed all six full routed spans for one layer (5.26 GB) after the
    chunked-read fix. It produced coherent `red`, but decode was only
    0.431 tok/s and the row remained a narrow wiring smoke.
  - Eight-layer full-span validation:
    `docs/local/model-validation/20260513T212755Z-kimi-k-fullspan-prewarm-8layer-after-coldstate-refault/`
    prewarmed 48 spans / 42.06 GB and fixed the earlier Mach acquire/refault
    failure by using the high-priority volatile cold state and chunked refault.
    It produced coherent `red`, but decode was only 0.303 tok/s and 52
    non-prewarmed layers still kept the row partial.
  - The `MLXPRESS_MACH_RELEASE_POLICY=pageout` diagnostic
    `docs/local/model-validation/20260513T213128Z-kimi-k-fullspan-prewarm-8layer-pageout/`
    is rejected. It was slower at 0.136 tok/s and raised footprint, so the
    default remains compression-first volatile/LIFO release.
  - Partial no-readback rows proved why full coverage was needed:
    `docs/local/model-validation/20260513T213224Z-kimi-k-fullspan-8layer-no-readback-no-eval/`
    and
    `docs/local/model-validation/20260513T213522Z-kimi-k-fullspan-8layer-no-readback-eval/`
    had zero explicit `pread` reads, but failed effective read pressure at
    roughly 880-901 GB/generated-token because non-prewarmed layers still used
    file-backed mmap spans.
  - Full 60-layer eager-prewarm diagnostic
    `docs/local/model-validation/20260513T214811Z-kimi-k-fullspan-60layer-no-readback-eval-generation/`
    is rejected. It completed full routed-span prewarm for 360 spans /
    315.48 GB in 454.36 s and passed post-load Activity Monitor footprint
    with only 416.2 MB increase, but the load recorded 341.40 GB page-ins and
    78.57 GB swapouts. During generation it crossed the hard peak footprint
    gate at 98.65 GB / 30.1% of model bytes before any token completed; the
    run was interrupted with exit status 130 to protect the machine, leaving
    `turns=0`, `gen_tokens=0`, and `decode_tps=0.000` in `summary.txt`.
    This proves full eager Mach prewarm is not the Kimi production path.
- The pre-existing local deletions of `xcode/default.profraw` and
  `Source/Cmlx/mlx-c/examples/arrays.safetensors` were left untouched.

## Next Work

1. Restore resident compute semantics without full routed-bank residency:
   the no-cold ephemeral mmap-prestack row is the current best MiniMax
   diagnostic and is now exposed as `--ephemeral-prestack on`, with coherent
   two-turn output, no permanent overlay after mmap load, peak Activity
   Monitor 2.62 GB / 7.1%, effective read pressure 662.31 MB/generated-token,
   and 14.22 aggregate tok/s. It remains below the old 39-47 tok/s JangPress
   target and still needs a temporary stacked overlay during load. On a longer
   coherent paragraph, the same path with
   TurboQuant KV reaches 19.70 tok/s while passing Activity and effective-read
   gates; disabling TurboQuant KV reaches 21.61 tok/s, so KV encoding is a
   short-turn latency issue but only about a 10% sustained gap. The new strict
   CLI flag row
   `docs/local/model-validation/20260513T174200Z-minimax-small-cli-ephemeral-prestack-nocold-strict-2turn/`
   passed `--fail-on-length-stop` with normal stops, but stayed at 13.162
   aggregate tok/s, so it proves wiring and gates rather than the target speed.
   The pre-fix compiled MiniMax override row
   `docs/local/model-validation/20260513T175200Z-minimax-small-cli-ephemeral-prestack-compiled-kvnone-profile-strict-2turn/`
   reached 24.60 tok/s with `--kv-cache none`, but it visibly looped both
   turns and length-stopped. The follow-up cache-promotion row
   `docs/local/model-validation/20260513T180400Z-minimax-small-cli-ephemeral-prestack-compiled-kvnone-promotedcache-strict-2turn/`
   fixes that specific compiled-cache bug and passes strict two-turn coherency
   at 26.27 tok/s with peak Activity Monitor 2.64 GB / 7.109%. It is the
   current best KV-none short speed diagnostic, and the longer guard row
   `docs/local/model-validation/20260513T180900Z-minimax-small-cli-ephemeral-prestack-compiled-kvnone-promotedcache-longer-2turn/`
   holds coherence for 107 generated tokens at about 24.9 tok/s. The
   TurboQuant-KV compiled row
   `docs/local/model-validation/20260513T181800Z-minimax-small-cli-ephemeral-prestack-compiled-tqkv-promotedcache-longer-2turn/`
   is the stronger cache-stack baseline: 169 coherent tokens, 22.03 / 23.13
   tok/s, peak Activity Monitor 2.63 GB / 7.089%, zero explicit reads, and
   180.10 MB/generated-token effective read pressure. It is still not a
   production pass because it relies on the ephemeral overlay and does not
   restore the old 39-47 tok/s JangPress target.
   The Mach active
   tensor row proved coherent low-RAM output and zero explicit active tensor
   reads, but decode remained 0.094 tok/s aggregate because the Swift
   active-streaming path still pays thousands of `reduce.call_chunk` calls and
   router/temporary-bank work. The next fix must keep no-cold mmap-backed
   resident `TurboQuantSwitchGLU` semantics while removing normal-path overlay
   creation, or move routed expert addressing into a persistent/offset-addressed
   kernel that avoids per-token active-bank rebuilds. A new fallback-side
   coalescing knob,
   `MLXPRESS_STREAMING_OFFSET_ACTIVE_WINDOW_COALESCE_MB`, can now reduce
   active offset-window fragmentation for Hy3/Kimi experiments, but it is
   opt-in only. The default path keeps one expert per window after the Hy3
   adjacent-merge regression row. Any coalescing change must be judged only by
   fresh token/s, Activity Monitor, and effective-read rows.
2. Fix the resident mmap/Metal-buffer accounting problem: current C++
   tensor-span mmap feeds GPU correctly, and MiniMax no longer has to build
   full `switch_mlp` banks in resident-expert mode, but MiniMax-Small still
   reports ~95% of model bytes in Activity Monitor when all routed tensors have
   live mmap-backed Metal buffers. `DONTNEED`, `PAGEOUT`, and forced
   `msync(MS_INVALIDATE)` advice do not lower that footprint. The remaining
   resident fix must avoid live Metal buffers for the whole routed set while
   also avoiding the slow per-token active-bank reconstruction fallback.
3. Rerun Hy3 K on the compression-first canonical mmap path with active
   streaming off, and separately design a GPU-side shard-aware offset path for
   its unaligned gate/up/down shard groups. The current fallback row is
   coherent and low-footprint but fails speed and effective read pressure.
4. Validate the Kimi pre-gate MoE chunking and stacked-slice reader with longer
   no-thinking / thinking / tool-call
   multi-turn cold-warm rows with token/s and no-loop checks.
5. Keep speculative/MTP decode disabled for base MLXPress unless explicitly
   enabled by a later topology-specific implementation.
6. Validate cache-stack-enabled coherent output per model family and attention
   architecture, starting with the locally downloaded JANGQ/JANGTQ families.
7. Use `scripts/compare-cache-deviation.sh` for cache-off/cold/warm
   deviation rows wherever a full baseline is safe, and record `--skip-off`
   explicitly when only cold/warm MLXPress comparison is safe. Activity Monitor
   gate failures now kill MLXPress-on rows by default; pass
   `--activity-gate-report-only` only for deliberate diagnostics.
8. Document each family as implemented, gated, or blocked with the exact command
   and model path used for proof. Keep
   `docs/MLXPRESS_ATTENTION_ARCHITECTURE_LEDGER.md` current with the functions,
   variables, parser modes, cache modes, and blockers found while building.
