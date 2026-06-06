# MLXPress Model Readiness Checklist

Last updated: 2026-05-13

This is the checklist system agents must use before saying a model family
works with MLXPress. The source of truth is both:

- Runtime inspect output:
  `.build/debug/mlxpress <model-dir> --inspect --json`
- This document plus `docs/MLXPRESS_ATTENTION_ARCHITECTURE_LEDGER.md`
- Active expert speed plan:
  `docs/MLXPRESS_ACTIVE_EXPERT_SCHEDULER_PLAN.md`

## States

- `proven`: implemented and backed by current runtime artifacts.
- `created`: code surface exists, but the model-family row still needs proof.
- `partial`: some proof exists, but a required mode, parser, cache state, or
  coherence row is still incomplete.
- `blocked`: a hard gate is failing.
- `missing`: no current implementation or proof exists.
- `not_applicable`: not required for that exact model row.

## Universal Gates

Every model, regardless of size, must use the same low-RAM loading discipline:

- One canonical MLXPress load method per row: mmap safetensors, bounded MLX
  allocator/cache policy, compression-first routed-weight residency, and no
  permanent prestack overlay unless explicitly requested. Active-expert
  streaming is a fallback/diagnostic path, not the default success path.
  `--ephemeral-prestack on` is an explicit temporary-overlay diagnostic for
  MiniMax-class unstacked bundles; it removes the temporary overlay after mmap
  load and keeps a process-exit cleanup fallback, so it satisfies the
  no-permanent-file rule but does not replace the final no-overlay resident
  kernel path.
- The historical JangPress speed/RAM methodology is resident compute plus
  macOS cold-page reclaim/compression. A coherent active-streaming row near
  1 tok/s with tiny Activity Monitor footprint is still blocked if it does not
  recover usable decode speed, low effective read pressure, and multi-turn
  coherency together.
- The three non-negotiable MLXPress gates are speed, RAM, and coherency:
  decode token/s must be better than the comparable non-MLXPress or prior
  baseline for that model path, Activity Monitor footprint must stay below the
  family gate, and visible plus reasoning output must be coherent with no
  loop. Passing only one or two of these is a diagnostic, not readiness.
- Native Mach purgeable routed-expert storage may be used only as a measured
  compression-first path: `MLXPressMachCache` tiles must be populated from real
  model bytes, acquired for the selected top-k, released by the configured
  compression percentage, and validated with Activity Monitor footprint,
  prompt/decode token/s, read/page pressure, and coherent multi-turn output.
  The primitive existing by itself is a `created` state, not readiness proof.
  The opt-in bridge `MLXPRESS_STREAMING_MACH_ACTIVE_TENSORS=1` is diagnostic
  until one family row proves that its managed/no-copy Mach tiles improve speed
  while preserving all hard gates.
  The offset-window bridge `MLXPRESS_STREAMING_MACH_OFFSET_SPANS=1` /
  `JANGPRESS_STREAMING_MACH_OFFSET_SPANS=1` is also diagnostic: it may only be
  credited when exact active expert offset windows show `tensor.mach_offset_*`
  rows, improved decode token/s, lower effective read pressure, and coherent
  multi-turn output under the Activity Monitor gate.
- Activity Monitor physical footprint is a hard gate. A row that expands to
  full model size in RAM is failed even if it generates coherent text.
- Token/s must be recorded for every completed generation row. Missing token/s
  means the row is blocked or diagnostic.
- File-read pressure must be recorded for streaming JANGTQ rows. A row that
  stays low in Activity Monitor but reads tens of GB per generated token is
  still blocked for speed, even if it is coherent.
- If `MLXPRESS_STREAMING_OFFSET_ACTIVE_WINDOW_COALESCE_MB` is used, record it
  in the row command and treat it as a speed/RAM tradeoff: it can reduce
  per-token offset windows and kernel groups, but it may map extra inactive
  expert bytes. Passing requires token/s, Activity Monitor, and effective-read
  gates together.
- Coherency must be checked for visible output and reasoning output. No hidden
  reasoning-only pass, no loop, and no max-token fake pass.
- Multi-turn proof must run in one loaded session.
- Cache-stack proof must include paged KV, disk L2, TurboQuant KV where
  enabled, cache hit/miss evidence, and path-dependent companion state where
  the architecture requires it.
- VL/video proof must use real media payloads and media-keyed cache hits.
- Every completed row must leave a durable artifact with stdout/stderr,
  prompt/decode token/s, Activity Monitor post-load/post-decode/peak gates,
  cache-hit tier, and no-loop/coherency verdicts.
- Prefer `--metrics-jsonl PATH` for every serious row. It records per-turn
  prompt token/s, decode token/s, coherency fields, streaming-profile snapshots
  when `MLXPRESS_STREAMING_PROFILE=1`, post-decode/peak memory, and Activity
  Monitor gate verdicts as JSON Lines, so later comparisons do not depend on
  pasted terminal output. New rows also record system pressure deltas
  (`pageins`, `pageouts`, `swapins`, `swapouts`) around the run so a slow row
  can prove whether it caused OS swap/page pressure or only model-file
  streaming.
- Summarize every pair of before/after speed rows with
  `scripts/summarize-mlxpress-metrics.py <baseline>/metrics.jsonl <candidate>/metrics.jsonl`
  and keep the output or JSON next to the artifact. This is the standard way to
  compare decode token/s, peak Activity Monitor percent, coherency, and top
  routed-profile stages. For streaming JANGTQ rows, also inspect
  `profile_read_mb` and `profile_read_mb_per_gen_token`; these are the
  model-file read-pressure counters that tell whether the row is still
  re-reading too much from safetensors even when Activity Monitor RAM is low.
  For all families, inspect `pagein_mb`, `swapin_mb`, and `swapout_mb` in new
  summaries before concluding that SSD pressure is coming from OS swap. When
  active residency is enabled, also inspect combined `cache_hit_rate`,
  `cache_byte_hit_rate`, and `cache_resident_mb`, plus split
  `tensor_resident_mb`, `slice_resident_mb`, and `bank_resident_mb`.
  MiniMax/Hy3 unstacked rows use the tensor cache; Kimi stacked rows use the
  slice cache. Exact active-bank caching is diagnostic only unless the row has
  real `bank_hits` and improves decode token/s under the Activity Monitor
  gate. A row that needs a huge residency budget but does not improve decode
  speed is still blocked.
- Use `--file-read-gate-mb-per-token N` when validating a candidate Kimi or
  routed-JANGTQ speed fix. Use `--file-read-gate-report-only` for diagnostics.
  A production proof should pass both the explicit file-read gate and the
  `effective_read_pressure` gate, which catches mmap page-ins that bypass
  `pread` counters.
- Parser autodetect must be verified through model capability stamps,
  chat/JANG config stamps, or model-type inference. No family gets a pass from
  caller-side hard-coded parser selection alone.
- Async router warm advice, SSM rederive, and companion-state warm restore must
  be treated as correctness surfaces, not performance-only knobs.

## Architecture Axes

Every family row must explicitly account for these before it can move from
`partial` to `proven`:

- Attention architecture: dense/full, sliding/rotating, routed MoE, hybrid
  SSM/linear attention, CCA, compressor/indexer, or VLM encoder + text
  decoder.
- Hadamard/TurboQuant matmul: JANGTQ rows must prove `hadamardRotate` plus
  `gatherTQ` / fused gate-up paths without full expert-stack residency. The
  row must name runtime sidecar signs/codebooks and the power-of-two Hadamard
  block decomposition used for non-power-of-two dimensions. Rows with all
  Hadamard blocks <=1024 should exercise the Swift SIMD-shuffle path; MiniMax
  1536-dim routed activations decompose as `1024 + 512`.
- RoPE/MRoPE: text RoPE offsets, sliding-window offsets, Qwen 2D/3D MRoPE
  image/video grid IDs, MRoPE deltas, partial-RoPE head vectors, and any
  dual-RoPE topologies must be cache-hit safe. Inspect must read both direct
  config fields and nested `rope_parameters` / `rope_scaling` dictionaries.
- Cache block storage/encode: paged `CacheBlock` chain hashes, disk L2
  safetensors payloads, `TQDiskSerializer` layer-kind tags, TurboQuant KV
  compressed state, rotating/sliding metadata, prompt-boundary raw-KV snapshots,
  cache-policy salt, and model/media salt must match the actual cache type.
- Hybrid split: Mamba/Arrays/CCA/DSV4 compressor-indexer companion state must
  be serialized/restored with the KV hit, or the hit must be rejected.
- VL/vector cache identity: image/video/audio processor settings, grid shapes,
  media content identity, and reasoning/template state must be part of the
  cache scope so equal text with different media cannot alias.

## Runtime Checklist Surface

`MLXPressModelReadinessChecklist` is emitted by `mlxpress --inspect --json`.
The same inspect payload also emits `bundle.architecture`, a machine-readable
summary derived from `config.json`/nested configs. It reports:

- `family`
- `attentionArchitecture`
- `loadMethod`
- `overallState`
- per-gate `items`
- `requiredProofs`
- `blockers`
- `bundle.architecture.attentionKinds`
- `bundle.architecture.matmulKinds`
- `bundle.architecture.positionEncodings`
- `bundle.architecture.positionVectorKinds`
- `bundle.architecture.cacheStorageKinds`
- `bundle.architecture.cacheEncodingKinds`
- `bundle.architecture.companionStateKinds`
- `bundle.architecture.hybridSplitKinds`
- `bundle.architecture.mediaCacheKinds`

Current universal gate names include:

- `attention-architecture-classified`
- `hadamard-tq-matmul-contract`
- `rope-mrope-position-contract`
- `cache-block-storage-encode`
- `hybrid-companion-state-split`
- `vl-vector-media-cache-proof`
- `parser-autodetect-stack`
- `per-turn-ram-speed-artifact`
- `cold-warm-deviation-proof`
- `synchronous-prompt-boundary-rederive`

The important low-level strings to look for are not cosmetic. For example:
`mla-partial-rope-attention`, `randomized-hadamard-pow2-blocks`,
`qwen-vl-position-ids[3,batch,seq]`, `qwen-vl-mrope-delta-vector`,
`cache-policy-salt(kvMode,kvBits,kvGroup,maxKV,promptBoundaryRawKV)`, and
`path-dependent-hit-reject-or-serialize-companion-state` each mark a real
runtime correctness axis.

## Prompt-To-Artifact Matrix

Every model-family proof row should map the prompt to these concrete artifacts:

| Requirement | Artifact / field |
| --- | --- |
| Load is low RAM | Post-load Activity Monitor gate with model bytes and ratio |
| Decode is low RAM | Post-decode and peak Activity Monitor gates |
| Speed is measured | Prompt token/s and decode token/s for every completed turn |
| Coherent output | Persisted stdout plus visible/reasoning no-loop verdict |
| Enough output | Generated-token and visible-character minimums, no max-token fake pass |
| Multi-turn | One process/session with repeated `--turn`, not separate loads |
| Cache stack | Paged/disk/TurboQuant KV settings, cold/warm hit tier, isolated disk dir |
| Deviation | Cache-off/cold/warm comparison or explicit `--skip-off` rationale |
| Parser | Reasoning/tool parser source and no-thinking/thinking/tool transcript |
| Hybrid state | Serialized/restored/rejected companion-state evidence |
| VL media | Media bytes/shape/dtype/processor scope, grid vectors, salt hit/miss |
| JANGTQ low RAM | Canonical mmap routed weights, cold-page advice, low Activity Monitor footprint, low file/page pressure, and no permanent prestack overlay |

Use it before running heavy generation. If `overallState` is `blocked`, fix the
named blocker before trying to call the model MLXPress-ready.

## Current Family Rows

### Kimi K2.x

Current inspect state: `partial`.

Required extra proof:

- Stacked Kimi JANGTQ_K now has low-RAM no-thinking multi-turn coherence
  proof, but decode speed is still too low for production.
- Thinking-on, tool-call, and cold/warm cache-hit rows must complete with
  token/s and no-loop checks before this family can be called ready.
- Cold/warm cache-hit rows must prove disk L2/TurboQuant KV reuse without
  deviation or path-dependent state bugs.

Known evidence:

- Low-RAM active-streaming load is proven for
  `~/models/JANGQ/Kimi-K2.6-Small-JANGTQ`.
- Low-RAM stacked active-streaming load and no-thinking multi-turn generation
  are proven for `~/models/JANGQ/Kimi-K2.6-JANGTQ_K`.
  The bundle has 60 stacked routed layers, 384 experts, top-k 8, and
  328.11 GB safetensors bytes. The loader skips 360 stacked routed tensors,
  reads active expert slices by safetensor byte offset, uses cached Darwin
  `pread` file descriptors with the size-aware read-cache policy (`F_NOCACHE`
  by default for this oversized Kimi bundle), and does not write any permanent
  prestacked tensors.
- Default exact two-turn stacked Kimi row:
  `docs/local/model-validation/20260513T024727Z/` passed with cache stack
  enabled, active expert tensor cache budget 0 MB, output `red color` /
  `blue fruit`, post-load 144.9 MB, post-decode 1.80 GB, peak 3.26 GB
  / 1.0%, and avg decode telemetry 0.28 tok/s.
- Longer stacked Kimi no-thinking row:
  `docs/local/model-validation/20260513T024821Z/` passed with no-loop,
  minimum-visible, and minimum-generation checks; output
  `Rain falls softly on the green grass.` /
  `The moon glows brightly in the night sky.`, post-load 145.8 MB,
  post-decode 2.81 GB, peak 4.68 GB / 1.4%, and avg decode telemetry
  0.39 tok/s.
- Kimi stacked speed remains blocked. A top-k batched decode-sized reduce path
  and cached safetensor file descriptors improve prompt prefill, but decode is
  still far below MiniMax/Hy3-class throughput because each token still
  rebuilds active routed expert arrays across 60 MoE layers x top-k 8.
- `MLXPRESS_STREAMING_PROFILE=1` is now the required first diagnostic for Kimi
  speed work. Profile rows must report active expert chunk count,
  `tensor.stacked_read` / `tensor.stacked_bank_read`,
  `tensor.stacked_array` / `tensor.stacked_bank_array`, gate/up eval, down
  eval, router readback, effective MB/s / bytes-per-call for byte-moving
  stages, peak Activity Monitor footprint, and token/s.
- Four-token streaming reduce chunks are the current low-RAM prefill default:
  `MLXPRESS_STREAMING_REDUCE_TOKEN_CHUNK_SIZE=4` and
  `MLXPRESS_STREAMING_FAST_REDUCE_MAX_TOKENS=4`. Profile row
  `docs/local/model-validation/20260513T032126Z-kimi-k-profile-1tok/`
  kept peak footprint at 1.73 GB / 0.5% and reduced active expert chunks from
  1140 to 180 for a 17-token prompt, but one-token decode still measured only
  0.11 tok/s.
- Sustained short profile
  `docs/local/model-validation/20260513T033000Z-kimi-k-profile-6tok/`
  produced coherent text (`The green grass swayed gently`) with peak
  2.39 GB / 0.7% and 0.35 tok/s. It read 130.55 GB of active stacked expert
  slices and materialized 55,866 active tensors for six visible tokens. Treat
  repeated active-slice read/array/stack construction as the current speed root
  cause unless a newer profile disproves it.
- Active bank loading is now the default stacked path through
  `MLXPRESS_STREAMING_BANK_LOAD=1` / `JANGPRESS_STREAMING_BANK_LOAD=1`.
  Profile rows
  `docs/local/model-validation/20260513T033557Z-kimi-k-bankload-profile-1tok/`
  and
  `docs/local/model-validation/20260513T033702Z-kimi-k-bankload-profile-6tok/`
  confirm the path is live. It reduces the active array construction counter
  from per-expert tensors to bank tensors, but the sustained row is still only
  0.39 tok/s and still reads 127.76 GB of active stacked slices. This does not
  move Kimi from `partial`.
- Metrics JSONL row
  `docs/local/model-validation/20260513T034200Z-kimi-k-metrics-jsonl/`
  proves durable per-turn metrics on Kimi and adds bandwidth fields to the
  streaming profile. The one-token row measured `tensor.stacked_bank_read` at
  about 6,968 MB/s and `tensor.stacked_bank_array` at about 19,643 MB/s while
  decode stayed 0.12 tok/s. Treat that as evidence that the current blocker is
  not raw memory bandwidth saturation alone.
- Streaming-profile JSONL row
  `docs/local/model-validation/20260513T034856Z-kimi-k-streaming-profile-jsonl/`
  proves the routed profile rows are now stored directly in `metrics.jsonl`.
  The `streaming_profile` record captured 20 rows, including active bank read
  bandwidth, bank-array construction bandwidth, router readback time, and
  per-stage milliseconds.
- No-eval boundary diagnostic
  `docs/local/model-validation/20260513T035500Z-kimi-k-no-each-layer-eval-after-envfix/`
  proves a real wiring bug was fixed: MLXPress active-streaming setup now
  leaves explicit `MLXPRESS_STREAMING_EVAL_EACH_LAYER=0` caller overrides
  alone instead of forcing the env var back to `1`. The row produced coherent
  `red`, stayed low RAM at 1.86 GB / 0.6% peak, and removed eval-boundary
  rows from the top streaming profile. It did not materially improve
  one-token decode speed, so treat it as a diagnostic scheduler row, not a
  readiness upgrade.
- Full routed-span Mach residency remains blocked for Kimi. Direct Mach
  registration now chunks large `pread` calls and can prewarm complete routed
  spans, but the full 60-layer row
  `docs/local/model-validation/20260513T214811Z-kimi-k-fullspan-60layer-no-readback-eval-generation/`
  failed production gates: 454.36 s prewarm for 360 spans / 315.48 GB,
  341.40 GB page-ins and 78.57 GB swapouts during load, peak generation
  footprint 98.65 GB / 30.1%, and no completed token before interruption
  (`turns=0`, `gen_tokens=0`, `decode_tps=0.000`). Do not treat low post-load
  footprint from full eager prewarm as readiness.
- Bounded eval stride is now available as
  `MLXPRESS_STREAMING_EVAL_LAYER_STRIDE` /
  `JANGPRESS_STREAMING_EVAL_LAYER_STRIDE`. Use it in comparison rows to record
  whether evaluating every Nth routed layer improves decode token/s without
  violating Activity Monitor footprint or effective-read gates. MiniMax stride
  4 row
  `docs/local/model-validation/20260513T124000Z-minimax-small-offset-kernels-eval-stride4-short/`
  stayed low-RAM but failed readiness: aggregate decode was 0.909 tok/s, turn 1
  length-stopped, `router.scores_eval` absorbed 63.2 s, and MLX allocator peak
  rose to 114.57 GB.
- Scored-down offset kernels are available behind
  `MLXPRESS_STREAMING_SCORED_DOWN_KERNELS=1` /
  `JANGPRESS_STREAMING_SCORED_DOWN_KERNELS=1`. The passing MiniMax row
  `docs/local/model-validation/20260513T131500Z-minimax-small-offset-scored-down-longer-generation/`
  stayed coherent with cache stack on, peak Activity Monitor 2.53 GB / 6.8%,
  zero explicit reads, and 1.150 tok/s aggregate decode. Treat this as an
  incremental kernel win only; readiness still needs resident-class speed.
- Scored-down no-eval row
  `docs/local/model-validation/20260513T132500Z-minimax-small-offset-scored-down-no-eval-longer-generation/`
  is coherent and reaches 1.250 tok/s with 2.64 GB / 7.1% peak Activity Monitor
  footprint, but it is not a production-ready mode because MLX allocator peak
  reaches 1.57 TB. Any future scheduler must gate both Activity Monitor
  footprint and MLX allocator/graph residency.
- File-streaming diagnosis
  `docs/local/model-validation/20260513T041000Z-kimi-k-file-streaming-diagnosis/`
  documents why the current Kimi speed issue looks like repeated safetensor
  slice reads rather than a proven swap failure. The one-token rows read
  20.19 GB of active stacked experts while staying under 2 GB peak process
  footprint. Readiness requires reducing that read pressure without raising
  Activity Monitor footprint or breaking coherency.
- System-pressure proof row
  `docs/local/model-validation/20260513T041500Z-kimi-k-system-pressure-profile/`
  confirms that diagnosis with fresh metrics: coherent `red`, peak 1.77 GB /
  0.5%, 20.19 GB active expert reads per generated token, 24.79 GB page-ins,
  0.06 MB swap-in, and 0 MB swap-out. This is not enough for readiness, but it
  proves which pressure source the next Kimi speed fix must attack.
- File-read gate report
  `docs/local/model-validation/20260513T042500Z-kimi-k-file-read-gate-report/`
  proves the gate is live. The coherent one-token Kimi row passed the
  Activity Monitor gate but failed a report-only 1,000 MB/generated-token file
  read gate at 20,190.94 MB/generated-token. Treat this as the current concrete
  speed blocker to beat.
- Active expert trace
  `docs/local/model-validation/20260513T043500Z-kimi-k-active-expert-trace/`
  records 180 trace calls, 1,440 routed slots, 1,440 unique expert touches,
  309 consecutive reuse touches, and reuse rate 0.215 on a coherent low-RAM
  `red` row. This proves the next scheduler should be driven by measured reuse,
  not assumed hot experts.
- Active slice residency diagnostics:
  `docs/local/model-validation/20260513T043530Z-kimi-k-active-slice-residency-512mb/`
  gets zero hits with a 512 MB budget, while
  `docs/local/model-validation/20260513T043713Z-kimi-k-active-slice-residency-8192mb/`
  gets 21.5% byte hits and lowers file reads to 15,858.30 MB/generated-token.
  Decode remains about 0.12 tok/s and the 8 GB row peaks at 11.35 GB, so
  residency alone is not a Kimi fix.
- Route-specific offset-span cache is now live, but it is not a Kimi readiness
  fix. Row
  `docs/local/model-validation/20260513T195055Z-kimi-k-offset-active-window-cache8gb-1tok/`
  produced coherent `red`, kept peak Activity Monitor footprint to 1.79 GB /
  0.545%, and cut effective read pressure versus the no-cache offset row, but
  still failed at 28,077.70 MB/generated token and decoded at 0.398 tok/s.
  Row
  `docs/local/model-validation/20260513T195406Z-kimi-k-offset-active-window-cache32gb-short/`
  produced a coherent 30-token sentence and held peak footprint to 2.41 GB /
  0.735%, but decoded at 0.321 tok/s and failed effective read pressure at
  4,780.81 MB/generated token. The trace touched most routed experts in the
  short generation: 25,440 routed slots, 21,196 unique expert touches, 0.263
  reuse rate, and 60,895 cache evictions. Do not treat bigger file-backed LRU
  budgets as the Kimi production path; Kimi needs anonymous/purgeable resident
  storage or persistent offset-addressed kernels, plus MLA absorb-decode.
- Mach offset-span residency is now created but unproven for Kimi. Enable it
  with `MLXPRESS_STREAMING_MACH_OFFSET_SPANS=1` /
  `JANGPRESS_STREAMING_MACH_OFFSET_SPANS=1`. It only registers exact
  one-expert active offset windows into anonymous purgeable Mach storage and
  profiles `tensor.mach_offset_read`, `tensor.mach_offset_register`,
  `tensor.mach_offset_hit`, and `tensor.mach_offset_array`; full spans and
  coalesced windows still fall back to mmap/cache. Do not mark Kimi improved
  unless a row shows coherent output, cache stack/TurboQuant KV on, prompt and
  decode token/s, low Activity Monitor footprint, and lower effective read
  pressure than the 32 GB file-backed active-window row.
- Follow-up source work after rejecting unbounded Mach retention: the Mach
  offset path now fills purgeable storage directly from safetensors via
  `JangPressMachCache.registerFilled` and exposes
  `MLXPRESS_STREAMING_MACH_OFFSET_SPAN_BUDGET_MB` /
  `JANGPRESS_STREAMING_MACH_OFFSET_SPAN_BUDGET_MB`. The next readiness row must
  include the direct-fill path, record `tensor.mach_offset_evict` /
  `tensor.mach_offset_budget_skip` if a cap is used, and prove the three hard
  gates together: better decode token/s, low Activity Monitor RAM, and coherent
  no-loop output.
- First Mach offset-span Kimi row
  `docs/local/model-validation/20260513T202041Z-kimi-k-mach-offset-spans-1tok/`
  is rejected as readiness: coherent `red`, 6.02 GB / 1.834% peak Activity
  Monitor, and 25.6% byte hit rate are good, but decode was only 0.368 tok/s
  and effective read pressure still failed at 24,819.72 MB/generated-token.
  It also found a measurement bug: `tensor.mach_offset_read` is now counted by
  the explicit file-read classifier and metrics summarizer.
- Same-process warm Mach offset-span row
  `docs/local/model-validation/20260513T202514Z-kimi-k-mach-offset-spans-2turn/`
  is rejected: coherent `red` / `blue` and 20.88 GB / 6.364% peak Activity
  Monitor, but aggregate decode only 0.209 tok/s, turn 2 only 0.144 tok/s,
  resident Mach offset bytes 60.73 GB, explicit read pressure
  30,363.52 MB/generated-token, and effective read pressure
  30,871.54 MB/generated-token. This confirms that unbounded Mach offset tile
  retention is not the Kimi production path.
- Sidecar-present rows
  `docs/local/model-validation/20260513T045222Z-kimi-k-sidecar-bankload-effective-gate/`
  and
  `docs/local/model-validation/20260513T045340Z-kimi-k-sidecar-bankload-effective-gate-warm/`
  confirm the user-added `jangtq_runtime.safetensors` is valid for Kimi
  JANGTQ_K (`7168/2-bit` gate-up and `2048/4-bit` down metadata) and is no
  longer falling back to deterministic runtime generation. The warm row is
  coherent and low-RAM but still only 0.117 tok/s and still fails the 1,000
  MB/generated-token gate at 20,190.94 MB/generated-token.
- Direct stacked mmap row
  `docs/local/model-validation/20260513T045423Z-kimi-k-direct-stacked-mmap-effective-gate/`
  confirms that whole stacked tensor mmap dispatch is not the speed fix:
  visible output is coherent and peak footprint is only 0.5%, but decode falls
  to 0.022 tok/s and the effective read gate fails at 969,853.44
  MB/generated-token from page-ins.
- If profile rows still show only `tensor.stacked_read` after changing the
  streaming code, rebuild the CLI explicitly with
  `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer swift build --jobs 2 --product mlxpress`
  before debugging the runtime path. A stale incremental CLI binary can look
  like a failed optimization.
- Direct `MLXArray(Data, dtype:)` active-slice construction is implemented.
  Row `docs/local/model-validation/20260513T030410Z/` stayed coherent and
  low-RAM with peak 3.37 GB / 1.0% and avg decode 0.30 tok/s, so the removed
  copy is useful cleanup but not the main speed fix.
- `MLXPRESS_STREAMING_EXPERT_CACHE_MB` exists as an explicit bounded in-memory
  active-slice cache knob, but the default is 0 MB. An 8 GB cache experiment
  `docs/local/model-validation/20260513T024555Z/` did not improve avg decode
  speed and raised peak footprint to 11.22 GB, so it remains a fallback
  diagnostic/tuning knob. The default low-RAM path is compression-first
  canonical mmap residency.
- Kimi thinking-on is blocked, not proven. Probe
  `docs/local/model-validation/20260513T025104Z/` stayed low RAM and produced
  coherent reasoning preview, but length-stopped and leaked reasoning-style
  text into visible output instead of answering exactly `4`.
- Base active-streaming fallback short no-thinking decode now passes after the
  CLI sends both `thinking=false` and `enable_thinking=false`:
  `docs/local/model-validation/20260513T000500Z-kimi-small-base-thinking-alias-4tok/RESULT.md`.
- MLXPress/cache-stack short no-thinking decode also passes with visible `4`,
  token/s telemetry, 130.08 GB routed bytes under management, paged KV, disk
  L2, TurboQuant KV, and 1.86 GB / 1.3% peak footprint:
  `docs/local/model-validation/20260513T000900Z-kimi-small-mlxpress-cache-thinking-alias-4tok/RESULT.md`.
- `bundle.architecture.matmulKinds` now marks Kimi JANGTQ as
  `jangtq-hadamard-rotation`, `randomized-hadamard-pow2-blocks`,
  `turboquant-codebook-gather(tq_packed+tq_norms)`,
  `jangtq-runtime-signs-and-codebooks`, `routed-active-expert-slice-matmul`,
  and `fused-gate-up-swiglu-tq`.
- `bundle.architecture.positionVectorKinds` now marks Kimi partial RoPE as a
  `partial-rope-head-vector(qk=64)` row rather than treating it like a scalar
  offset-only cache. It also marks Kimi attention as
  `mla-partial-rope-attention`.
- Strict decode still fails the 30% Activity Monitor gate before stdout at
  roughly 43.2 GB / 30.3% of 142.58 GB safetensors bytes.
- `--mlxpress 80`, smaller `--prefill-step-size 8`, and diagnostic layer cold
  sweep did not reduce the peak.
- 512 MiB malloc tracing shows repeated `concatenate_gpu` allocations through
  `StreamingTurboQuantSwitchGLU.callChunk` during DeepSeek/Kimi JANGTQ decode.
- 128 MiB forced-longer tracing now narrows the current stack to
  `StreamingTurboQuantSwitchGLU.reduced` at router-index CPU readback. The row
  emits repeated `&`, so Kimi longer decode is blocked by both memory and
  coherency:
  `docs/local/model-validation/20260514T000210Z-kimi-small-forced-longer-malloc-trace-128m/RESULT.md`.
- The router-readback materialization barrier did not clear the failure, and
  evaluating the original residual tensor first was worse and reverted:
  `docs/local/model-validation/20260514T001355Z-kimi-small-forced-longer-router-barrier-trace-128m/RESULT.md`,
  `docs/local/model-validation/20260514T001948Z-kimi-small-forced-longer-original-x-barrier-trace-128m/RESULT.md`.
- Moving router-index readback inside the reduce chunk was also worse and
  reverted:
  `docs/local/model-validation/20260514T002812Z-kimi-small-forced-longer-chunk-local-index-trace-128m/RESULT.md`.
  Treat this as an architectural streaming-MoE/prefill issue before attempting
  another low-level materialization tweak.
- Source fix now chunks `DeepseekV3JANGTQMoE` before routing, so the Kimi
  router, router-index CPU readback, active expert loads, TurboQuant gather,
  and shared-expert MLP execute on small token windows. The runtime honors
  `MLXPRESS_STREAMING_MOE_TOKEN_CHUNK_SIZE` and legacy
  `VMLX_JANGTQ_PREFILL_STEP`; sync defaults on through
  `MLXPRESS_STREAMING_MOE_PREFILL_SYNC` / `VMLX_JANGTQ_PREFILL_SYNC`.
- Validation row `docs/local/model-validation/20260513T012521Z/` proves that
  pre-router chunking alone does not clear Kimi longer decode: post-load stays
  low at 1.23 GB / 0.9%, but peak decode hits 43.2 GB / 30.3%, partial stdout
  is `uuualphaalp`, and no token/s row completes.
- Streaming JANGTQ decoder layers now materialize and clear layer output by
  default through `MLXPRESS_STREAMING_EVAL_DECODER_LAYER` /
  `MLXPRESS_STREAMING_EVAL_EACH_LAYER`, unless the caller explicitly sets the
  env override to `0`. The after-env-fix no-eval diagnostic proves the knob is
  respected, but it only covered a one-token coherent row. Longer strict Kimi
  rows are still needed before changing the default boundary policy.
- Stacked Kimi JANGTQ2K must also use the low-RAM stacked-slice path: the
  `switch_mlp` stacked fallback reads one expert slice by safetensor byte
  offset. A full stacked tensor read followed by `stacked[expertIdx]` is not an
  acceptable MLXPress path.

### MiniMax M2

Current inspect state: `partial`.

Known evidence:

- Thinking-off rows can be coherent with MLXPress, disk L2, TurboQuant KV, and
  token/s recorded, but the current combined-repo MiniMax-Small split is still
  not production-passing: active-streaming OS-cache is low-RAM but slow and
  fails read pressure; resident routed tensors pass read pressure but fail the
  Activity Monitor RAM gate.
- Older JangPress MiniMax notes are a target, not current proof. They showed
  roughly 40-47 tok/s and low RSS after routed-page release, but they did not
  measure Activity Monitor `phys_footprint` and one row had an output chunk
  bug. Current combined-repo rows must re-prove the same methodology with
  physical footprint, coherent multi-turn output, and token/s recorded.
- Load-time Mach-backed whole-projection stacks are rejected for MiniMax
  readiness. Row
  `docs/local/model-validation/20260513T164210Z-minimax-small-loadtime-mach-stacks-resident-one-turn/`
  produced coherent text and reached 7.99 tok/s, but peak Activity Monitor was
  35.34 GB / 95.4% and post-decode was 34.92 GB / 94.3%. The granularity is
  too coarse to count as the compression-first method.
- Forced-cold ephemeral mmap-prestack was useful but not the best policy. Row
  `docs/local/model-validation/20260513T170143Z-minimax-small-ephemeral-prestack-mmap-noprofile-2turn/`
  used `MLXPRESS_PRESTACK_EPHEMERAL=1`, active streaming off, cache stack,
  disk L2, TurboQuant KV, and no generation profiler. It passed coherent
  two-turn output, removed the temporary 32.8 GB overlay at exit, kept peak
  Activity Monitor at 2.62 GB / 7.1%, and decoded at 12.60 / 14.01 tok/s. It
  failed effective read pressure at 1,997.41 MB/generated-token.
- No-cold ephemeral mmap-prestack is the current strongest MiniMax diagnostic,
  but not a production pass. Row
  `docs/local/model-validation/20260513T170902Z-minimax-small-ephemeral-prestack-mmap-nocold-2turn/`
  leaves the mmap-prestack overlay under normal OS cache policy instead of
  forcing pages cold at load. It passed coherent two-turn output, removed the
  temporary 32.8 GB overlay at exit, kept peak Activity Monitor at 2.62 GB /
  7.1%, passed effective read pressure at 662.31 MB/generated-token, and
  decoded at 13.89 / 14.49 tok/s. The same path is now exposed as
  `--ephemeral-prestack on`. It still fails readiness because it depends on a
  temporary stacked overlay and aggregate speed is 14.22 tok/s, below the
  historical resident JangPress 39-47 tok/s target. Use this path as the next
  resident-compute development baseline, not as a green proof.
- The typed CLI flag for that path is now wired and strict-gated. Row
  `docs/local/model-validation/20260513T174200Z-minimax-small-cli-ephemeral-prestack-nocold-strict-2turn/`
  used `--ephemeral-prestack on` with cache stack, disk L2, TurboQuant KV,
  `--fail-on-length-stop`, min visible/generated gates, and no active expert
  streaming. It passed coherent two-turn output, both turns ended with
  `stop=stop`, peak Activity Monitor was 2.63 GB / 7.093%, explicit reads were
  0 MB/generated-token, effective read pressure passed at
  780.76 MB/generated-token, and aggregate decode was 13.162 tok/s. This proves
  the option wiring, not production speed.
- Sustained no-cold mmap-prestack rows show the cache-stack path is closer than
  short rows suggest, but still blocked. Row
  `docs/local/model-validation/20260513T171708Z-minimax-small-ephemeral-prestack-mmap-nocold-tqkv-long/`
  produced a coherent 129-token paragraph with TurboQuant KV at 19.70 tok/s,
  peak Activity Monitor 2.59 GB / 7.0%, and effective read pressure
  236.15 MB/generated-token. The matching KV-none diagnostic
  `docs/local/model-validation/20260513T171915Z-minimax-small-ephemeral-prestack-mmap-nocold-kvnone-long/`
  reached 21.61 tok/s. TurboQuant KV is therefore a large short-row latency
  cost but only about a 10% sustained decode gap. MiniMax remains `partial`
  because the path still uses an explicit temporary overlay and is below the
  old 39-47 tok/s target.
- Profiled sustained row
  `docs/local/model-validation/20260513T172505Z-minimax-small-ephemeral-prestack-mmap-nocold-tqkv-long-profile/`
  shows the next blocker is not TurboQuant KV alone:
  `decode.async_eval_submit` took 6,161.4 ms total / 47.396 ms average, while
  `decode.kv_quantize` took 352.6 ms total / 2.733 ms average. Readiness still
  needs a lower evaluated resident graph cost or a compiled/persistent kernel
  path that preserves low Activity Monitor footprint.
- Pre-fix compiled MiniMax decode is explicitly not a valid speed fix. Row
  `docs/local/model-validation/20260513T175200Z-minimax-small-cli-ephemeral-prestack-compiled-kvnone-profile-strict-2turn/`
  used `--compiled-decode on`, `--allow-minimax-compiled-decode`,
  `--compiled-max-cache-length 512`, `--kv-cache none`, and
  `--ephemeral-prestack on`. It proved the compiled closure engaged and reached
  24.56 / 24.64 tok/s with peak Activity Monitor 2.58 GB / 6.943%, explicit
  reads at zero, and effective read pressure 158.58 MB/generated-token. It is
  rejected because both turns looped and length-stopped. Keep MiniMax compiled
  decode denied by default until parity is fixed.
- The single-sequence compiled-cache parity bug is now patched, but still
  diagnostic. Row
  `docs/local/model-validation/20260513T180400Z-minimax-small-cli-ephemeral-prestack-compiled-kvnone-promotedcache-strict-2turn/`
  promotes `KVCacheSimple` to `CompilableKVCache` before compiling. The same
  MiniMax path now passes strict two-turn coherency with `--kv-cache none`:
  26.29 / 26.26 tok/s, `stop=stop` on both turns, peak Activity Monitor
  2.64 GB / 7.109%, explicit reads at zero, and effective read pressure
  845.69 MB/generated-token. Longer guard row
  `docs/local/model-validation/20260513T180900Z-minimax-small-cli-ephemeral-prestack-compiled-kvnone-promotedcache-longer-2turn/`
  generated 107 coherent no-loop visible tokens at 24.15 / 25.62 tok/s with
  peak Activity Monitor 2.79 GB / 7.510%, zero explicit reads, and effective
  read pressure 284.56 MB/generated-token. This is not production readiness
  because TurboQuant KV/cache-stack parity, no-overlay resident storage, and
  the 39-47 tok/s resident JangPress target remain open.
- Single-sequence compiled TurboQuant KV is now wired and validated on MiniMax,
  making the cache-stack path much stronger. Strict row
  `docs/local/model-validation/20260513T181400Z-minimax-small-cli-ephemeral-prestack-compiled-tqkv-promotedcache-strict-2turn/`
  passes with `default-kv=turboQuant(k=3,v=3)`, `stop=stop` on both turns,
  26.95 / 24.11 tok/s, peak Activity Monitor 2.53 GB / 6.818%, zero explicit
  reads, and effective read pressure 845.65 MB/generated-token. Longer guard
  row
  `docs/local/model-validation/20260513T181800Z-minimax-small-cli-ephemeral-prestack-compiled-tqkv-promotedcache-longer-2turn/`
  generated 169 coherent no-loop visible tokens at 22.03 / 23.13 tok/s with
  peak Activity Monitor 2.63 GB / 7.089%, zero explicit reads, and effective
  read pressure 180.10 MB/generated-token. This row is superseded for speed:
  the CLI was using the Activity Monitor gate as a hidden
  `MLX.Memory.memoryLimit` throttle.
- Activity Monitor gating is now decoupled from the MLX memory budget. Strict
  cache-stack row
  `docs/local/model-validation/20260513T192000Z-minimax-small-mlxpress-cache-stack-compiled-tqkv-activitygate-unthrottled-2turn-stop/`
  passed with TurboQuant KV, disk L2, compiled decode, `--fail-on-length-stop`,
  and `--activity-gate 59`: 49.09 / 48.66 tok/s, both turns `stop=stop`,
  no-loop coherent visible text, peak Activity Monitor 2.65 GB / 7.159%, zero
  explicit reads, and effective read pressure 0.09 MB/generated-token. MiniMax
  is still partial only because this proof depends on the temporary ephemeral
  prestack overlay rather than the final no-permanent-file resident mmap/offset
  kernel path.
- The cleanup-after-load row
  `docs/local/model-validation/20260513T193500Z-minimax-small-mlxpress-ephemeral-cleanup-after-load-compiled-tqkv-2turn/`
  confirms the overlay is removed after mmap load before decode and the model
  remains coherent: 47.54 / 47.19 tok/s, peak Activity Monitor 2.64 GB /
  7.131%, zero explicit reads, and 585.69 MB/generated-token effective read
  pressure. This satisfies no-permanent-file behavior for the diagnostic path;
  it is still partial because the overlay is still written during load.
- Tensor-span mmap is now GPU-correct but not sufficient for MiniMax readiness.
  `docs/local/model-validation/20260513T083103Z-minimax-small-jpreg-load-footprint-tensor-span-mmap/`
  loaded with opt-in tensor-buffer mmap and reported 0.6 GB RSS, 35.0 GB
  Activity Monitor footprint, and only 1.3 GB of live mmap-tracked Metal
  buffers. That first row identified load-time `switch_mlp` bank creation as a
  blocker.
- `MLXPRESS_RESIDENT_EXPERTS=1` now removes the MiniMax full-bank variable by
  registering per-expert tensors into `JANGTQStreamingExpertStore`; the focused
  resident-expert test verifies sanitize does not materialize `switch_mlp`
  banks. Load-only rows
  `docs/local/model-validation/20260513T090900Z-minimax-small-resident-experts-load-footprint/`,
  `docs/local/model-validation/20260513T092000Z-minimax-small-resident-pageout-load-footprint/`,
  and
  `docs/local/model-validation/20260513T092038Z-minimax-small-resident-force-invalidate-load-footprint/`
  still report 35.2 GB Activity Monitor footprint and 34.6 GB live
  mmap-tracked Metal buffers. `MADV_PAGEOUT` and forced
  `msync(MS_INVALIDATE)` do not make the full routed Metal-buffer set pass the
  low-RAM gate. MiniMax readiness therefore requires a resident/offset
  active-expert dispatch that avoids both full-bank materialization and live
  Metal buffers for the whole routed set.
- Short generation row
  `docs/local/model-validation/20260513T093159Z-minimax-small-resident-experts-short-generation/`
  produced coherent `Green` / `Blue` with token/s recorded, proving the
  no-full-bank resident path is semantically viable. It still failed the
  Activity Monitor gate at 35.25 GB / 95.2% peak and only decoded one-token
  turns at 0.25 then 0.98 tok/s, so it is not a readiness row.
- Active-streaming mmap-active tensors are now wired for both unstacked and
  stacked active expert slices under `MLXPRESS_STREAMING_MMAP_ACTIVE_TENSORS=1`.
  MiniMax longer row
  `docs/local/model-validation/20260513T095717Z-minimax-small-streaming-mmap-active-tensors-longer-generation/`
  passed coherent multi-turn output, cache stack, zero explicit active-slice
  reads, and low Activity Monitor peak (2.66 GB / 7.2%), but decoded only
  1.63 then 2.39 tok/s. This is the current best low-RAM fallback proof, not a
  production-speed proof.
- Multi-file offset kernels are now created and tested for true stacked
  tensors, MiniMax-style expert-major single-file offsets, and split-shard
  multi-file offsets with sentinel zeroing. Release row
  `docs/local/model-validation/20260513T110922Z-minimax-small-offset-kernels-multifile-release-longer-generation/`
  passed coherent multi-turn output, cache stack, zero explicit active-slice
  reads, effective-read gate, and 2.70 GB / 7.3% Activity Monitor peak. It is
  still not a readiness row because decode averaged only 1.07 tok/s.
- No-layer-eval offset diagnostic
  `docs/local/model-validation/20260513T111455Z-minimax-small-offset-kernels-multifile-release-no-layer-eval/`
  stayed coherent and low-RAM, but decode only improved to 1.16 tok/s and MLX
  allocator peak rose to 429.77 GB. Do not treat eval-boundary removal as the
  production speed path without additional graph-residency work.
- The 8 GB tensor-cache variant
  `docs/local/model-validation/20260513T095912Z-minimax-small-streaming-mmap-active-tensors-8gb-cache-longer-generation/`
  also stayed coherent and low-RAM but regressed to 1.12 then 1.03 tok/s.
  Bounded active tensor residency is therefore rejected as the MiniMax speed
  fix; readiness still requires resident `TurboQuantSwitchGLU` semantics via a
  direct offset-descriptor kernel or non-permanent page-backed stitched banks.
- `MLXPressMachCache` is restored and synthetically tested as the native
  purgeable-memory primitive for that resident path. It now supports
  per-component JANGTQ tensor tiles, no-copy MLXArray views, and an opt-in
  active streaming bridge via `MLXPRESS_STREAMING_MACH_ACTIVE_TENSORS=1`. It
  produced a real MiniMax-Small row with coherent two-turn output, cache stack
  on, peak Activity Monitor footprint 10.85 GB / 29.3%, zero explicit active
  tensor reads, and effective read pressure 898.74 MB/generated token. It is
  rejected as readiness because decode was only 0.106 then 0.085 tok/s and
  `reduce.call_chunk` dominated the profile. MiniMax still needs resident
  `TurboQuantSwitchGLU` semantics without full routed-bank Metal residency or a
  lower-level offset/persistent kernel that avoids per-token active-bank rebuilds.
- Full offset-span residency is also rejected as the MiniMax speed fix.
  `docs/local/model-validation/20260513T0913Z-minimax-small-offset-full-span-cache-128gb-noprofile-2turn/`
  used a 128 GB logical offset-span cache, scored-down offset kernels, cache
  stack, disk L2, and TurboQuant KV. It stayed coherent, kept explicit reads at
  zero, passed effective read pressure at 826.25 MB/generated token, and held
  peak Activity Monitor footprint to 2.54 GB / 6.9%. Decode was still only
  0.823 then 0.816 tok/s. The profiled sibling row had 98.85% offset-span cache
  hit rate and no evictions by turn 2, so the remaining gap is not CPU span
  remapping or profiling overhead. MiniMax readiness still requires resident
  `TurboQuantSwitchGLU` compute semantics without full routed Metal-buffer
  residency, or a persistent/offset-addressed kernel that avoids the current
  active-streaming execution structure.
- Thinking-on rows can produce coherent reasoning, but production readiness
  still needs final visible answer closure without max-token failure.

### Hy3 / Hunyuan

Current state: mixed.

- Uniform Hy3 has coherent MLXPress rows.
- Hy3 JANGTQ_K active-streaming load is low-RAM, but decode peak/coherency are
  blocked. Non-streaming prestack is not an acceptable MLXPress path.
- Hy3 JANGTQ_K offset compatibility is improving but not ready:
  flexible unaligned-shard grouping keeps split layouts on offset kernels, and
  segmented active windows are now coherent after adding expert coverage to the
  group key. Row
  `docs/local/model-validation/20260513T152335Z-hy3-k-segmented-groups-fixed-no-thinking-short/`
  passes short `4` / `6` coherency with cache stack on and low Activity Monitor
  footprint, but still fails effective read pressure at
  14,944 MB/generated-token and only decodes at 0.729 tok/s.
- The activity-gate unthrottled rerun
  `docs/local/model-validation/20260513T201500Z-hy3-k-segmented-groups-unthrottled-no-thinking-short/`
  confirms that Hy3 K's current blocker is not the old `--activity-gate`
  memory-limit coupling. It remains coherent and low-footprint, but decode is
  only 0.792 tok/s and effective read pressure remains about
  14,953 MB/generated-token.
- Active-window coalescing now exists as a fallback tuning surface. Row
  `docs/local/model-validation/20260513T191100Z-hy3-k-offset-window-coalesce16-no-thinking-short/`
  with `MLXPRESS_STREAMING_OFFSET_ACTIVE_WINDOW_COALESCE_MB=16` stays coherent
  and low-footprint, and improves decode to 1.004 tok/s, but fails effective
  read pressure at 29,345 MB/generated-token. This proves the tradeoff is
  measurable and that 16 MB is too broad; it is not readiness.
- A rejected current-code row
  `docs/local/model-validation/20260513T192000Z-hy3-k-offset-window-coalesce0-current-no-thinking-short/`
  showed the former adjacent-by-default merge was also too broad: 0.962 tok/s,
  58,710.64 MB page-ins, and 29,355 MB/generated-token effective read pressure.
  Default Hy3 offset windows must stay one-window-per-expert; coalescing is
  opt-in only.
- The corrected default row
  `docs/local/model-validation/20260513T193000Z-hy3-k-offset-window-default-no-coalesce-current-no-thinking-short/`
  confirms the regression fix: coherent `4` / `6`, 0.781 tok/s decode,
  4.320% peak Activity Monitor footprint, 22,803.84 MB mapped spans, and
  14,949 MB/generated-token effective read pressure. It is still not readiness.

### Qwen3.6 MoE

Current inspect state should be treated as `created` or `partial` until rerun.

- JANGTQ model plumbing exists.
- Full cache-stack proof and any path-dependent cache audit still need current
  combined-repo artifacts.
- Qwen3.6 configs can carry a real `vision_config` while still using generic
  `qwen3_5` model-type strings. Treat those as multimodal rows unless JANG
  metadata explicitly says `has_vision=false`: inspect must expose
  `vl-encoder-plus-text-attention`, Qwen `[3,batch,seq]` position IDs,
  image/video `THW`, MRoPE deltas, and media-salted cache identity, then block
  until real media cold/warm proof exists.

### Qwen VL

Current inspect state: `blocked` until real media rows run.

- 2D/3D MRoPE image/video grid position IDs must be verified across cold/warm
  cache hits.
- Inspect must expose Qwen VL `[3,batch,seq]` position IDs, image `THW`, video
  `THW`, and the MRoPE delta vector before running the heavy media row.
- Same text with different image/video payloads must use distinct media-salted
  cache keys or miss.
- Cache rows must record media processor settings, grid shape, token count,
  token/s, cache hit/miss tier, and Activity Monitor footprint.

### ZAYA

Current inspect state should be treated as `partial` or `missing` for VL rows.

- CCA cache state is path-dependent and must be serialized or rejected on disk
  hits.
- VL rows require real image/video payloads and media-keyed cache proof.

### Ling / Bailing Hybrid

Current inspect state should be treated as `partial` or `missing`.

- Hybrid companion-state behavior must be audited before full disk-hit reuse.
- Current combined-repo low-RAM multi-turn proof still needs rerun.

### DeepSeek V4 Flash

Treat as a special topology.

- Compressor/indexer state must be cache-keyed and deviation-tested.
- Do not generalize DSV4 cache behavior to Kimi, MiniMax, Hy3, or ZAYA.
