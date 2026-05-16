# vMLX Swift Production Engine Gate

This document translates the Python-side Round 2 and Round 3 release gates into
production-grade gates for the full `vmlx-swift` engine.

The production target is the consolidated Swift engine that can replace the
split stack of `vmlx-swift-lm`, Jinja, swift-transformers, model-family glue,
cache/runtime code, and downstream Osaurus-facing integration points.

The important rule is the same: prove behavior with live artifacts. A source read
or absence check is not enough. Every row that can execute in this Swift package
must leave an artifact under:

```text
docs/local/swift-release-gates/<round>/<row>/
```

`docs/local` is intentionally local-only so large logs, metrics, media fixtures,
and model outputs do not leak into public commits. Summary reports can be
committed only after removing private paths, secrets, and model-specific local
details.

## Boundary

`vmlx-swift` is the engine/runtime package. It directly owns:

- Swift package graph integrity.
- Model loading and generation through package APIs, `RunBench`, focused tests,
  and optional CLI probes.
- Sampling defaults, chat-template kwargs, reasoning policy, tool parsing,
  tokenizer/template integration, cache stack behavior, active expert routing,
  TurboQuant/JANGTQ execution, SSM and media cache behavior, distributed planning,
  and runtime status ABI.
- Build and test readiness for the vendored Swift stack: Jinja, Hub,
  Tokenizers, Generation, Models, MLXLMCommon, MLXLLM, MLXVLM, MLXEmbedders,
  low-level MLX/Cmlx integration, RunBench, distributed worker tools, and any
  thin optional CLI wrappers.

It does not directly own:

- Electron tray, buttons, visual states, i18n, packaged app launch, or app
  restart behavior.
- OpenAI/Ollama/Anthropic HTTP endpoints unless a standalone Swift server is
  added.
- `/health`, `/admin/cache-stats`, `/admin/deep-sleep`, or `/admin/wake` unless a
  Swift daemon surface is added.
- Image generation routes unless image generation models are added to this
  package.

Those rows become downstream Osaurus gates after Osaurus repins to this package.

## Required Swift Artifact Tools

Use these existing surfaces before adding new ones:

- `swift build --target <target>` for package-level readiness.
- `swift test --filter <suite>` for focused tokenizer, template, policy, cache,
  parser, model-family, and generation behavior.
- `swift test` over the full package once the package graph is clean enough to
  make that meaningful.
- Package-level APIs in MLXLMCommon, MLXLLM, MLXVLM, MLXEmbedders, Hub,
  Tokenizers, Jinja, Generation, Models, and distributed targets.
- `.build/debug/RunBench ...` for model generation, speed, cache, tool, VL,
  audio, and long-context probes.
- Distributed tools such as `TPRankWorker` and peer smoke tools for distributed
  mode.
- Thin CLI probes only when they exercise the same underlying engine APIs and
  leave useful metrics.
- Existing scripts for model validation, cache deviation, model inventory,
  matrix runs, and metrics summarization.

Add only the missing diagnostic knobs that turn a claim into a behavioral
artifact:

- `RunBench` or package-test output for resolved generation config.
- Engine/cache stats JSON for prefix, paged, disk L2, SSM companion, media, and
  TurboQuant KV counters.
- Engine environment/config snapshot after runtime setup, if env-derived config
  remains part of model-family activation.
- `RunBench --reasoning-matrix-json`.
- `RunBench --tool-matrix-json`.
- `RunBench --vl-multiturn-json`.
- `RunBench --audio-multiturn-json`.
- `RunBench --speed-matrix-json --compare-baseline <json>`.

## Current Focused Proof Added 2026-05-14

These rows are now covered in this checkout with artifacts under
`docs/local/swift-release-gates/dsv4-fixes/`:

- DSV4 fallback tools: pre-fix failure plus post-fix JANGTQ-K and JANGTQ2
  template-smoke passes for top-level OpenAI tools and Osaurus-sized schemas.
- DSV4 standalone template parity: `DSV4Minimal.jinja` now renders OpenAI
  tool schemas between the final no-system user/developer turn and assistant
  prefill, and a focused test reads the standalone file directly so it cannot
  silently lag behind `ChatTemplateFallbacks.dsv4Minimal`.
- DSV4 reasoning policy: `reasoning_effort=max` and low/medium/high pass
  through without hidden aliasing; the legacy raw-max env no longer gates max.
- DSV4 fallback max preface: focused template test proves max reaches rendered
  prompt text.
- Cache topology: focused tests prove dense paged prefix, media salt isolation,
  hybrid SSM companion requirement, disk prefix restore, DSV4 CSA/HSA/SWA disk
  restore, ZAYA CCA v2 disk restore, and path-dependent/sliding detection.
- VL shape guards: focused tests prove finite-positive extent handling across
  Qwen/ZAYA/GLM/Gemma/LFM/Smol VL processors and preserve the 2D-vs-3D disk
  restore guard.
- JANGTQ Hadamard/matmul dispatch: focused tests prove MiniMax-sized Hadamard,
  offset gather, scored gather, fused gate/up, split-shard sentinels, and
  decode-token slots. The wired rank proof also covers Hadamard rank-3/rank-4
  shape preservation, dense JANGTQ matmul rank-2/rank-3 input restoration, and
  TurboQuant KV Hadamard rank-4 inverse round-trip.
- ZAYA-VL JANGTQ_K schema: focused decoder test proves nested
  `mxtq_bits.routed_expert` dictionaries preserve gate/up and down bit widths
  instead of failing on non-`Int` metadata.
- ZAYA-VL JANGTQ_K runtime module bits: live
  `BENCH_ZAYA_MOE_BITS=1` reflects the real
  `ZAYA1-VL-8B-JANGTQ_K` model as 40 `TurboQuantSwitchGLU` modules with
  `gateUp=2`, `down=4`, `seed=42`. The metadata contract gate now handles
  ZAYA1-VL's 40 CCA+MoE blocks and vision-LoRA local expert sidecars instead
  of applying text-only ZAYA assumptions.
- ZAYA-VL sidecar/tools: focused test and real-bundle template smokes prove
  JANGQ ZAYA-VL shims preserve vision placeholders while adding the ZAYA XML
  tool schema. The shim writes the corrected template into
  `tokenizer_config.json`, `chat_template.json`, and `chat_template.jinja`
  because `swift-transformers` prefers sidecar template files over
  `tokenizer_config.json`.
- Kimi template compatibility: focused and real-bundle template smoke prove
  `tojson(separators=(',', ':'))` renders through Swift Jinja, and tokenizer
  context mirrors `enable_thinking` to `thinking` only when `thinking` is not
  explicitly supplied.
- DSV4 live rows: JANGTQ-K three-turn chat coherence passes with visible output,
  `.stop`, tok/s, no raw reasoning leakage, and correct `sapphire-42` recall.
  DSV4 paged-incompatible growing-cache row restores through disk with salted
  hits and nil-salt misses.
- DSV4 reasoning probe: the old "7 + 5" arithmetic row was a bad ambiguous gate;
  the explicit `Q: What is 7 + 5? ... A:` prompt now passes reasoning off/on/max
  on JANGTQ-K. This is documented as a prompt-gate correction, not a sampling
  fallback or hidden model guard.
- ZAYA CCA growing-turn cache: live `RunBench` diagnostic proves
  `BatchEngine` and direct coordinator probes now use the same generation
  parameter salted cache key. Deterministic phrase recall passes on JANGTQ_K and
  JANGTQ4; the older generic "previous answer" prompt remains a semantic
  weakness and is not hidden.
- RunBench executable caveat: after editing executable sources, use
  `swift build --product RunBench`, not target-only builds, before trusting a
  live binary artifact.

## Current Focused Proof Added 2026-05-15

- MTP bundle status plumbing: loader and model configuration now expose
  preserved/disabled/error MTP status without claiming speculative accept/reject
  decode is implemented. The local Qwen3.6 JANG_4M MTP/VL bundle probe passed
  strict status checks and the focused MTP suite.
- Osaurus single-package surface: `VMLX` remains the intended umbrella product
  for the next Osaurus repin, but this is a package-graph prerequisite only. It
  does not replace live multi-turn model/cache/API proof.
- Moving sibling boundary: `../vmlx-swift-lm` is behind `origin/main` and dirty
  with parallel local edits. Treat it as evidence to compare, not a source to
  copy wholesale.

## Current Focused Proof Added 2026-05-16

- Scope boundary: MTP is parked for this non-MTP production pass. The current
  work is DSV4 Flash, Nemotron Omni, ZAYA1-VL, cache/template correctness, and
  live coherent multi-turn behavior.
- DSV4 non-MTP live rows under
  `docs/local/live-model-matrix/20260516Tdsv4-nonmtp/` prove JANGTQ-K/JANGTQ2
  config/template detection, JANGTQ-K three-turn recall, reasoning
  off/on/max, and explicit `repetition_penalty=1.0` without hidden sampler
  floors. The 5.5k semantic recall row returns the requested facts, but the full
  B7 long-context/vector drift gate is still open.
- Nemotron Omni video generation now follows the real source contract: the
  processor carries the post-EVS keep count through `LMInput.ProcessedVideo`,
  and the model applies RADIO video embedding, class-token strip, pixel shuffle,
  MLP projection, EVS, and post-EVS placeholder alignment before LM prompt
  splice. The focused unit row asserts 16 groups x 256 tokens at pruning 0.7
  becomes 1228 video tokens.
- Nemotron Omni strict pre-fix proof:
  `docs/local/live-model-matrix/20260516Tomni-nonmtp/Nemotron-Omni-Nano-JANGTQ4-CRACK_omni_strict.out`
  fails the video LMInput row with a repeated filler loop. Post-fix proof:
  `.../Nemotron-Omni-Nano-JANGTQ4-CRACK_omni_evs_v2.out` passes 13/13 text,
  image, video, audio, reasoning toggle, mixed media, media salt, and hybrid SSM
  warm-pass rows.
- Nemotron Omni BatchEngine now has stricter false-green checks for empty
  visible output, raw reasoning tags, repeated filler loops, repeated bigram
  loops, and image-missing denials. With `maxBatchSize=2` forcing scheduler
  traversal, text B=1, text B=2, and audio B=1 pass, but image B=1 with
  explicit `enable_thinking=false` still returns a missing-image denial in
  `.../Nemotron-Omni-Nano-JANGTQ4-CRACK_omni_batch_forced_evs_v3.out`.
  This is a real blocker, not a row to mask by forcing reasoning on.
- Nemotron Omni no-thinking media-tail fix: the direct tail probe
  `.../Nemotron-Omni-Nano-JANGTQ4-CRACK_omni_tail_probe.out` shows compact
  `<think></think>` and `<think></think>\n` fail to ground the same prepared
  image tensor, while `<think>\n</think>\n\n` passes. The processor now
  canonicalizes only the already-closed media no-thinking tail to that newline
  form, leaving reasoning disabled and leaving text-only prompts alone.
- Nemotron Omni post-tail-fix proof:
  `.../Nemotron-Omni-Nano-JANGTQ4-CRACK_omni_batch_nothink_tail_fix.out`
  passes 18/18. Direct image with `enable_thinking=false` is grounded at
  `96.6 tok/s`, and BatchEngine image B=1 with `enable_thinking=false` is
  grounded at `47.8 tok/s`; text/audio BatchEngine rows still pass.
- Build proof after the non-MTP runtime/harness changes:
  `swift build -c release --product RunBench --jobs 2` passes. Focused
  `swift test` is still blocked before the target test by the unrelated
  `Tests/MLXPressPolicyTests/MLXPressLowRamPolicySourceTests.swift:4:8:
  no such module 'Testing'` compile error; latest artifact:
  `docs/local/live-model-matrix/20260516Tomni-nonmtp/NemotronHOmniSmokeTests_tail_fix.err`.

## Fix Gates

| Row | Python-side intent | Swift equivalent | Required proof |
| --- | --- | --- | --- |
| F1 | Reconcile stale test expectation after label rename. | Reconcile any stale runtime-status, JANGTQ kernel, reasoning parser, tokenizer, template, model-family, or cache-state tests after consolidation. | Failing focused `swift test` log, patch, passing rerun log, and test count delta. |
| F2 | Classify anonymous live gate scripts. | Classify every untracked gate script, local fixture, and model matrix under `scripts/`, `tools/`, `Tests/`, and `docs/local`. | `git status --short` before/after plus a short classification file. |
| F3 | Typecheck after removed declarations. | Compile all exported package products after removing remote packages and vendoring sources. | `swift package describe`, builds for Jinja, Hub, Tokenizers, Generation, Models, MLXLMCommon, MLXLLM, MLXVLM, MLXEmbedders, RunBench, and distributed tools; include a full package graph failure list if full build is blocked. |

## Round 2 Swift Behavioral Gates

| Row | Swift row | Engine surfaces | Artifact |
| --- | --- | --- | --- |
| B1 | Bundle sampling defaults. | `GenerationConfigFile`, `GenerationConfig`, `GenerateParameters`, `ModelConfiguration`, chat template kwargs. | Three local bundles with different `generation_config.json`; no sampling fields in request; resolved generation config JSON proves bundle values beat generic fallback. |
| B2 | Environment/config isolation at engine start. | Model-family detection, loader config, cache config, JANGTQ/TurboQuant config, distributed config. | Poison parent shell with JANGTQ, DSV4, dense-lane, cache env vars; launch an engine probe through package API or RunBench; capture sanitized runtime config snapshot. |
| B3 | Greedy and repetition-penalty behavior after guard removal. | `GenerateParameters`, reasoning/template policy, direct package generation, RunBench generation harness. | DSV4, MiniMax, and Ling prompts at greedy and rep=1.0; first 400 chars, stop reason, loop detector, think close status, token/s. |
| B4 | Reasoning on/off/effort matrix. | `DeepseekV4ReasoningPolicy`, reasoning parser, `UserInput.additionalContext`, template kwargs, `--reasoning-effort`. | Matrix for DSV4, MiniMax, Qwen, GLM, Mistral, Kimi, GPT-OSS, Gemma where local; `max` must pass through instead of silent downgrade. |
| B5 | SSM/hybrid async re-derive. | `SSMReDerive`, `SSMStateCache`, `SSMCompanionDiskStore`, `CacheCoordinator`, hybrid model input preparation. | Three-turn GatedDeltaNet/Qwen3.5-style run with prefix mismatch and partial overlap; companion hits/misses and coherent turn 3 output. |
| B6 | VL multi-turn cache behavior. | `UserInput.images/videos/audios`, `MediaSalt`, VLM processors, placeholder suffix cache, `MLXVLM`. | Image+text, text-only same session, different image; media salt nil for text-only turn, resume path hit, grounded answer per image. |
| B7 | DSV4 long-context regression and vector drift. | DSV4 cache, compressor, paged/disk state, finalizer budget, vector probe equivalents. | Long-context run on canonical DSV4 bundle; no 3k to 4k loop; paged+dsv4 cache detail; disk blocks; terminal restored; vector probe drift reported honestly. |
| B8 | Tool-calling per family. | `ToolCallProcessor`, `ToolCallFormat`, model-family parsers, chat template tool injection. | One tool request and one tool-result follow-up per parser family supported in Swift; schema valid, no plaintext leak, replay order preserved. |
| B9 | Simultaneous cache layer probe. | Prefix cache, paged cache, disk L2, SSM companion cache, media cache, metrics JSONL. | One or more model runs that collectively increment every legitimate counter; unsupported layers marked N-A with model reason. |
| B10 | Live HTTP probes. | Not direct package scope. Osaurus API gate after repin, or future Swift daemon. | For `vmlx-swift`, replace with package API, RunBench, focused tests, and runtime status JSON. |
| B11 | Decode and prefill speed matrix. | `RunBench`, metrics JSONL, metrics summarizer. | DSV4, MiniMax, Qwen, Zaya, GLM, Kimi, Mistral, Gemma rows where local; report wall tok/s, decode-window tok/s, PP tok/s, baseline delta. |
| B12 | JANGTQ MPP/NAX dispatch verification. | Swift JANGTQ/TurboQuant dispatch, active expert streaming, kernel status ABI. | DSV4 and MiniMax JANGTQ runs with accelerated and non-accelerated modes; runtime status records kernel path and token/s delta. |
| B13 | Deep sleep and wake. | Not direct package scope unless a daemon lifecycle API is added. | For now, Osaurus process lifecycle gate. Swift package equivalent is load, unload/deinit, reload same process without stale cache corruption. |
| B14 | Tray and listeners. | Not engine scope. | Osaurus app gate after repin. Swift equivalent is process lifecycle events emitted by any future runtime observer. |
| B15 | Image generation. | Not current text/VLM engine scope. | N-A unless image generation support is added. |
| B16 | Distributed mode. | `MLXDistributedCore`, `MLXDistributedTransport`, `MLXDistributedJACCL`, `MLXDistributedTP`, `TPRankWorker`, `DistributedModePlanner`. | `TPRankWorker` build, no-peer distributed plan returns inactive cleanly, no crash. |
| B17 | i18n removed-key sweep. | Not engine scope. | Osaurus UI gate after repin. |
| B18 | Repo-local app smoke. | Not engine scope. | Osaurus packaged app gate after repin. Swift equivalent is local CLI smoke for DSV4 auto-detect env and no poisoned inherited env. |

## Round 3 Inverse Gates

Every engine feature needs an explicit OFF proof and an ON proof.

| Row | Swift inverse | Required proof |
| --- | --- | --- |
| I1 | Prefix cache OFF/ON. | Same prompt twice with cache disabled: no hit counter increment. Re-enable: hit counter increments. |
| I2 | Paged cache OFF/ON. | Health or metrics state shows paged disabled; multi-turn still coherent; re-enable shows allocated/shared block activity. |
| I3 | Disk L2 OFF/ON. | No cache files written when disabled; cache files written when enabled; no private model paths in committed report. |
| I4 | KV cache quantization OFF/ON. | `--kv-cache none` takes non-quantized attention path on long context; quantized mode reports expected branch and remains coherent. |
| I5 | SSM companion cache OFF/ON. | Hybrid model falls back to clean prefill when off; companion hits appear when on. |
| I6 | TurboQuant OFF/ON. | Dense or non-TQ path still decodes coherently; TQ path restores speed/kernel status. |
| I7 | JANGTQ acceleration OFF/ON. | Off mode uses baseline codebook path; auto mode uses accelerated profitable shape; token/s delta recorded. |
| I8 | Streaming OFF/ON. | CLI/package stream and non-stream modes both produce complete output and correct stop reason where supported. |
| I9 | Tools OFF/ON. | Omitted or empty tools never invokes parser; enabled tools produce schema-valid calls. |
| I10 | Reasoning OFF/ON. | `enable_thinking=false` and omitted effort produce no hidden think block; enabled modes close think block when model supports it. |
| I11 | Image generation OFF. | N-A in package until image generation is added; downstream API should return a clean unsupported response. |
| I12 | Distributed OFF/ON. | No distributed flag reports inactive and no peer loop; distributed flag parses and no-peer coordinator exits cleanly. |
| I13 | Default sampling OFF. | With no default flags, resolution order is bundle metadata, then documented fallback; artifact captures resolved kwargs. |

## Round 3 Regression Gates

| Row | Swift regression | Required proof |
| --- | --- | --- |
| R1 | Decode tok/s. | Compare current matrix against last accepted `RunBench` or metrics baseline; flag >5 percent drops with attribution. |
| R2 | TTFT. | Measure first-token latency at 1k, 16k, and 64k for DSV4, MiniMax, Qwen, and Kimi where local. |
| R3 | RAM watermark. | Capture `phys_footprint` or equivalent process footprint during 64k DSV4 prefill; do not substitute MLX memory-limit throttling. |
| R4 | Cache hit ratio. | Re-run a prior multi-turn cache conversation and compare prefix/paged/disk/SSM hit ratios. |
| R5 | Quality. | Fixed 10-prompt coherence set across reasoning families; visible answer, no loop, think closure, normal stop. |
| R6 | Crash/hang. | Kill generation mid-stream or cancel the task; verify no zombie process, stale Metal context, or stuck in-flight request. |
| R7 | Memory leak. | 50-turn MiniMax loop with footprint sampled every 5 turns; flag monotonic growth beyond the agreed gate. |

## Round 3 Edge Conditions

| Row | Swift condition | Required proof |
| --- | --- | --- |
| C1 | Empty prompt. | CLI/API rejects or no-ops cleanly with no decode task. |
| C2 | Single-message session. | One prompt, one answer, cache/session object can be replayed by test harness if session persistence exists. |
| C3 | Context-limit history. | Scheduler truncates or rejects according to model limit; no crash and warning recorded. |
| C4 | Simultaneous sessions. | Two concurrent generation tasks for different models or configs do not cross streams or cache state. |
| C5 | Mid-stream cancel. | Cancellation stops decode and metrics show no stuck in-flight work. |
| C6 | Mid-stream model switch. | Current generation cancels before new model load; no stale cache reuse. |
| C7 | Engine interruption. | Process or task interruption surfaces a clean error and allows a later run. |
| C8 | Model load failure. | Nonexistent path fails within bounded time with actionable error. |
| C9 | Out of memory. | Oversized model fails cleanly; no silent precision fallback that changes requested model semantics. |
| C10 | Invalid tool choice. | Unknown tool name returns structured error, not parser crash. |
| C11 | Tool-result interleaving. | tool call, tool result, assistant reply order preserved with no raw JSON leak. |
| C12 | Chat-template kwargs precedence. | Explicit `chat_template_kwargs` override generic request flags according to adapter contract. |

## Status and Capability Gates

The Python prompt uses `/health` and `/admin/cache-stats`. The Swift package
equivalent is a stable runtime status JSON plus metrics JSONL. If a standalone
daemon is added, mirror the same fields through `/health`.

Required Swift status fields:

- `continuous_batching`.
- `cache_summary.prefix`.
- `cache_summary.paged`.
- `cache_summary.disk_l2`.
- `cache_summary.ssm_companion`.
- `cache_summary.media`.
- `architecture.hybrid_ssm`.
- `architecture.sliding_window`.
- `architecture.turbo_quant`.
- `architecture.vision`.
- `architecture.audio`.
- `kernel_type`.
- `jangtq_acceleration`.
- `model_loaded`.
- `model_family`.
- `kv_cache_quantization`.
- `sampling_defaults`.
- `supports_tools`.
- `supports_vision`.
- `supports_reasoning`.
- `reasoning_efforts`.
- `compat_warnings`.

Status rows S1, S2, S6, and S7 are Osaurus UI rows. S3, S4, and S5 become Swift
runtime status ABI rows.

## Button and Visual Gates

Buttons and visuals are not in `vmlx-swift`. They belong in Osaurus after it
repins to this engine. The Swift-equivalent contribution is to expose reliable
state transitions so the app can render correctly:

- idle.
- loading.
- ready.
- generating.
- cancelling.
- sleeping or unloaded if the host implements sleep.
- errored.
- model capability changes.
- token count and token/s updates.
- reasoning trace boundaries.
- tool-call boundaries.
- media attachment boundaries.

Osaurus must then prove send, stop, retry, edit, copy, model picker, settings,
session lifecycle, sleep/wake, model refresh, tool toggle, reasoning effort,
image gallery, server tab, empty states, light/dark mode, resize, long message
rendering, code blocks, math, tool cards, image attachments, toasts, streaming,
reasoning trace UI, errors, locales, and about panel.

## Behavior Gates

| Row | Swift responsibility | Osaurus responsibility |
| --- | --- | --- |
| X1 | Preserve or reject draft/session state only if Swift owns sessions. | Draft autosave, close/reopen behavior. |
| X2 | Auto-detect model family and capabilities from bundle files. | Show detected capabilities in picker. |
| X3 | Provide deterministic reload/load APIs. | Last-active session and model auto-load. |
| X4 | None. | Update check. |
| X5 | Safe shared runtime state if multiple clients attach. | Multi-window behavior. |
| X6 | Generation continues if host process remains alive. | Background notification and minimized app state. |
| X7 | None. | Notifications permission flow. |

## Minimum Swift Gate Subset

Before claiming this engine is production-ready for Osaurus, the minimum
behavioral subset is:

1. Package graph and vendor proof: no remote Jinja, swift-transformers,
   HuggingFace, EventSource, or yyjson package dependency remains.
2. Build proof for Jinja, Hub, Tokenizers, Generation, Models, MLXLMCommon,
   MLXLLM, MLXVLM, MLXEmbedders, RunBench, TPRankWorker, and distributed tools.
3. Runtime status proof from package APIs, focused tests, and RunBench metrics.
4. Sampling-default proof across three real bundles.
5. Reasoning on/off/effort proof across every local reasoning family.
6. Greedy and rep=1.0 proof for DSV4, MiniMax, and Ling.
7. Cache inverse proof for prefix, paged, disk L2, KV quantization, SSM companion,
   and media cache where applicable.
8. SSM async re-derive proof on a hybrid model.
9. VL multi-turn proof on every local VL family.
10. Tool-call parser proof for every supported family.
11. DSV4 long-context and vector drift proof.
12. Speed and PP matrix with prior baseline comparison.
13. JANGTQ/TurboQuant acceleration on/off proof and low-RAM active expert proof.
14. Distributed inactive/no-peer proof.
15. Osaurus repin proof: the downstream app must pass HTTP, UI, tray, i18n,
    deep-sleep/wake, image, and packaged-app gates against this package.

## Next Feature Tracks After Cleanup

These are the biggest follow-on features after the engine cleanup and release
gates are closed. They should not silently alter the baseline release behavior.
Each needs its own opt-in gate, artifact set, and rollback path.

### M5 Max PP Compatibility Speedup

Goal: improve prompt-processing throughput on M5 Max while preserving the same
model outputs, cache semantics, low-RAM routed behavior, and downstream Osaurus
API compatibility.

Required gates:

- Run the B11/R1/R2 speed matrix before and after the change on the same machine.
- Report PP tok/s, decode-window tok/s, wall tok/s, TTFT, and `phys_footprint`
  separately at 1k, 4k, 16k, and 64k prompts.
- Cover at least DSV4, MiniMax JANGTQ, Qwen3.6, Qwen3.5 hybrid, Kimi, Mistral,
  Gemma, and one VL family where local.
- Prove cache compatibility: prefix, paged, disk L2, SSM companion, media salt,
  and TurboQuant KV behavior must remain identical or have an explicit migration.
- Prove no coherency regression with the fixed 10-prompt set and a multi-turn
  cache stack run.
- Keep Activity Monitor footprint under the existing family gates. Do not count
  `MLX.Memory.memoryLimit` throttling as a valid low-RAM result.
- If the speedup depends on a new Metal path, kernel dispatch, or command-buffer
  strategy, status JSON must expose the selected path so Osaurus can display and
  debug it.

Acceptance rule: faster PP is not enough. The row passes only if PP improves
without output drift, cache corruption, hidden memory growth, or decode-speed
regression beyond the agreed threshold.

### MTP Model Activation

Goal: support real MTP-capable models as an explicit activation path while
keeping plain autoregressive decode as the baseline until MTP proves itself per
family.

Current package surface added on 2026-05-15:

- `MTPBundleInspector`, `MTPBundleStatus`, and `MTPRuntimeMode` provide no-load
  detection from config metadata, JANG runtime metadata, safetensors indexes, and
  safetensors headers.
- `ModelConfiguration` / `ResolvedModelConfiguration` carry `mtpStatus` so
  Osaurus can expose truthful capabilities after model load.
- `preserved_enabled` is status only: it means the artifact has MTP tensors
  preserved, not that speculative MTP decode may auto-launch.
- Detailed Osaurus, cache, and VL wiring is in
  `docs/VMLX_SWIFT_MTP_OSAURUS_WIRING_2026_05_15.md`.

Required gates:

- MTP must be off by default unless the model bundle explicitly declares a valid
  MTP head/path and the request or launch config enables it.
- The runtime must expose `mtp_available`, `mtp_enabled`, draft depth, accepted
  token count, rejected token count, acceptance rate, fallback count, and speed
  delta in metrics/status.
- D2/D3 MTP requires a draft step that returns hidden state as well as logits.
  A logits-only `mtp_forward` is acceptable for metadata or D1 diagnostics only;
  it is not a production MTP activation path.
- D3 must verify `[primary, d1, d2, d3]` in one target forward and commit an
  accepted prefix length of `0...3` plus the verifier bonus. A D1 loop that takes
  roughly 128 verify cycles for 256 output tokens is not the target; a D3 row
  should move toward the 50-70 verify-cycle range before speed work is credited.
- Hybrid/SSM MTP must carry the same variable-prefix rollback/commit semantics:
  only accepted base tokens and accepted companion state may survive rejection.
- Partial acceptance is not all-or-nothing. If D3 accepts `d1` and `d2` but
  rejects `d3`, the cache state after `[primary, d1, d2]` is the only valid
  committed state. The implementation may use capture/commit for speed or
  rollback+repair as a measured correctness-first stepping stone.
- Every native-MTP speed row must include AR baseline tok/s, MTP tok/s, requested
  and effective depth, verify calls, output tokens, accepted/drafted by depth,
  average committed tokens per verify call, bonus tokens, correction/rejection
  count, target verify time, draft time, residual sampling time, cache mode,
  verify kernel mode, draft-head mode, and output tail review.
- Unsupported models must fall back to normal decode with a clear status reason,
  not a crash and not silent speculative behavior.
- Run ON/OFF inverse rows per family: MTP off equals baseline output behavior;
  MTP on produces coherent output, normal stop, no loop, and no hidden reasoning
  artifact.
- Prove multi-turn cache correctness with MTP enabled: prefix, paged, disk L2,
  SSM companion, media cache, and TurboQuant KV where applicable.
- Prove reasoning compatibility: `enable_thinking=false`, `enable_thinking=true`,
  and every supported `reasoning_effort` must keep the same visible/reasoning
  separation as baseline decode.
- Prove tool compatibility: tool calls must remain schema-valid, and tool result
  interleaving must not be reordered by accepted draft tokens.
- Prove cancellation and mid-stream failure: rejected drafts or cancelled streams
  must not leave stale cache state.
- Compare speed and quality against baseline for each MTP-enabled family. Record
  where MTP is slower, lower quality, or unsafe and keep it disabled there.

Acceptance rule: MTP is production-ready only for a named model family after that
family passes its own ON/OFF, multi-turn, cache, reasoning, tool, speed, and
coherency gates. There is no global "MTP works" claim.

## Production-Ready Definition

A model family is not ready because it loads. It is ready only when the current
Swift engine proves all applicable rows:

- low Activity Monitor physical footprint for routed/compressed rows.
- token/s recorded for generation.
- coherent visible answer.
- no looping.
- reasoning closes or is intentionally absent.
- multi-turn works.
- cache topology proves the relevant hits.
- media rows use real media payloads.
- tool rows preserve schema and replay order.
- speed regressions are within gate or explicitly explained.
- unsupported surfaces are marked N-A with the exact reason, not hidden.
