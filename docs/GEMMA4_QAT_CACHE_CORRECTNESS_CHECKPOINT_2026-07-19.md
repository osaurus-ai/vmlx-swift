# Gemma 4 QAT cache correctness checkpoint — 2026-07-19

Status: **PARTIAL — the exact current merged vMLX tree is reconfirmed in an
isolated Release Osaurus build with the local Gemma 4 12B JANG_4M bundle for
native cache, explicit TurboQuant 4/4, SSD-only restart/partial restore, long
rotating-window recall, visible prefill progress, and the safe rejection of an
explicit paged-RAM request. The current 12B Activity Monitor Memory row is
9.48 GB and remains a failed low-footprint gate. A controlled built-in web-tool
continuation is now live-proven with native cache and explicit TQ 4/4, but the
Thinking on/off tool-loop propagation is only partial: both UI states completed
one controlled tool continuation, but the thinking-on Gemma row emitted no
visible reasoning block. The admin generation-settings ownership defect is now
source-fixed and live-proven in a rebuilt isolated Release app: the automatic
one-token chat-prefill warm-up no longer overwrites the last visible request.
The full parser/tool-error matrix and the wider API/stop matrix remain open.
Historical MXFP8 evidence below is retained but was not rerun on
2026-07-20.**

This checkpoint is intentionally limited to locally installed Gemma 4 MXFP8
and JANG_4M bundles through the real Osaurus/vMLX Swift runtime. MXFP4 is not a
substitute test artifact and is not part of this checkpoint.

## Source contract

- Ordinary Osaurus single-batch loads keep paged RAM KV off by default.
- Engine-selected/native cache mode keeps TurboQuant KV off. TurboQuant is an
  explicit user opt-in with explicit key/value bit widths.
- Block-disk L2 is on by default and is independent of paged RAM cache. An
  explicit block-disk Off must survive memory-safety policy resolution.
- The tested Gemma 4 12B bundles declare 48 attention layers: 8 full-attention
  `KVCacheSimple` layers and 40 `RotatingKVCache` sliding-attention layers.
- TurboQuant converts only eligible `KVCacheSimple` layers. It must preserve
  all rotating SWA layers as rotating caches.
- The shared conversion hook is type-selective, not architecture-name driven:
  it converts `KVCacheSimple` only and leaves rotating, DeepseekV4, Mamba,
  Arrays/CCA, and composite cache entries native. For hybrid families, an
  explicit TQ request therefore applies only to actual full-attention KV
  layers; companion state must be restored in its matching native type or the
  prefix hit must be rejected and synchronously rederived.
- `TurboQuantCacheTransitionSnapshot` records the real before/after cache
  classes at the conversion point. A configured TurboQuant mode or a non-zero
  compression-event counter is not accepted as layer-level proof.
- Prompt-boundary L2 entries may intentionally remain typed raw KV plus typed
  rotating state even when live decode uses TurboQuant. Telemetry must report
  the live codec and stored codec separately; neither implies the other.
- Gemma mixed SWA/full attention has no recurrent SSM/GLA companion state.
  Hybrid SSM/GDN/CCA async-rederive gates therefore remain separate family
  rows and must not be inferred from Gemma evidence.

## Current evidence

| Row | Current evidence | Status |
|---|---|---|
| Stale `<end_of_turn>` stop | vMLX `604d24e4`; focused Gemma 3/4 regression 3/3; live MXFP8 and JANG_4M both emitted a literal marker and continued through `FINAL-OMEGA` | VERIFIED-LIVE |
| Default paged policy | Exact Release binary at `bbc0b20d`: Settings visibly showed paged off and `/admin/cache-stats` reported requested/effective paged false. Explicit paged On on Gemma truthfully remained effective false with `is_paged_incompatible=true` | VERIFIED-LIVE |
| MXFP8, TurboQuant off | Current build visibly emitted `MXFP8-NATIVE-COLD-7412` at 31.9 tok/s and, after restoring defaults, `MXFP8-DEFAULT-RESTORED-9021` at 32.0 tok/s. Endpoint: native fp16, 8 KV + 40 rotating, transition null, paged off, SSD on | VERIFIED-LIVE |
| MXFP8, TurboQuant 4/4 | Current build visibly emitted eight exact `MXFP8-TQ-*` lines at 38.0 tok/s. Transition: 8 KV to 8 TQ; all 40 rotating layers preserved | VERIFIED-LIVE |
| MXFP8, TQ restart/partial SSD restore | After quitting/relaunching only the isolated app, changed `MXFP8-RESTORE-*` lines completed at 36.7 tok/s. Endpoint: disk hits 2, misses 13, stores 3; paged hits/misses 0; transition re-established 8-to-8/40 | VERIFIED-LIVE |
| JANG_4M, TurboQuant off | Current build visibly emitted `NATIVE-COLD-6724` at 36.8 tok/s and partial-reuse `NATIVE-PARTIAL-9381` at 41.7 tok/s; endpoint native 8 KV + 40 rotating, transition null, paged off, SSD active | VERIFIED-LIVE |
| JANG_4M, TurboQuant 4/4 | Current build visibly emitted eight exact CACHE lines at 55.2 tok/s; process-restart RESTORE lines at 52.8 tok/s; exact 8 KV to 8 TQ conversion with all 40 rotating layers preserved | VERIFIED-LIVE |
| TurboQuant transition telemetry | `TurboQuantCacheTransitionTelemetryTests` 3/3 on current source plus current-build endpoint transitions for both 12B formats | VERIFIED-SOURCE + VERIFIED-LIVE |
| Explicit block-disk Off | Settings visibly saved SSD Off; `DISK-OFF-CURRENT-2846` completed at 41.4 tok/s; endpoint reported block disk false and all disk/paged counters zero | VERIFIED-SOURCE + VERIFIED-LIVE |
| Engine-selected TurboQuant default | Settings visibly restored Engine Selected; current MXFP8 endpoint reported `effective_kv_mode=fp16`, transition null, paged off, SSD on | VERIFIED-SOURCE + VERIFIED-LIVE |
| RAM safety refusal/override | Strict custom 10% visibly refused the 31B JANG_4M at a 12.8 GiB budget before load. No Automatic Limits then loaded the same model and visibly emitted `RAM-OVERRIDE-3179` at 12.2 tok/s. Endpoint reported `automatic_memory_limits_disabled=true` and allowed the estimated 30.9 GiB request. Safe Auto was restored and saved | VERIFIED-LIVE for control behavior |
| Activity Monitor footprint | Exact proof executable inspected. 31B JANG_4M: main Memory 28.37 GB; inspector Real Memory 19.30 GB, Private 996 MB. Bundle is 25G on disk (25,926,564 KiB). Main Memory exceeded full bundle size, so this does not satisfy the low-footprint gate | FAILED-LIVE |

## 2026-07-20 current merged-tree reconfirmation

This run used only
`/Users/eric/models/OsaurusAI/OsaurusAI--gemma-4-12B-it-qat-JANG_4M`.
No MXFP4 bundle was loaded or used as a substitute.

- vMLX remote-main merge commit: `364eab42`.
- Osaurus current main: `9b0331fd4`; its package pin is vMLX `4b431c6a`.
- `364eab42` and `4b431c6a` resolve to the identical vMLX tree
  `8f52a9fb0f72694fc7f03f06b017a93c12924886`; their tree diff is empty.
- Isolated Release app:
  `/private/tmp/osaurus-gemma4-closeout-baseline-release-derived/Build/Products/Release/osaurus.app`,
  bundle id `com.dinoki.osaurus.gemma4closeoutbaseline`, ad-hoc-signed executable
  SHA-256 `7253244d2e06e6ef92c9d53a23e6fe36284d27a1ac54f72173d4044ede160158`.
- The app used an isolated test root and keychain-free UI. The production app
  and its preferences were not used for these rows.

| Row | Visible app evidence | Runtime/source evidence | Status |
|---|---|---|---|
| Fresh settings | Server -> Settings -> Cache visibly showed Prefix On, GPU Cache Off, Disk Cache On, Codec Engine Selected, and SSM rederive On; Thinking was visibly Off in chat | Fresh isolated preferences; no production `UserDefaults` reuse | VERIFIED-LIVE |
| Native long-window cold prefill | An 8,635-token ledger visibly showed `Prefill 0/8635` and `8192/8635`, then emitted `START=amber-17; MIDDLE=cobalt-88; END=jade-42`; TTFT 7.39 s, 38.1 tok/s, 25 tokens | Topology 48 layers = 8 native full KV + 40 rotating SWA; TQ layer count 0; paged counters 0 | VERIFIED-LIVE |
| Native partial prefix | Same live conversation emitted `R037=river-37-quartz; R219=river-219-quartz`; TTFT 0.94 s, 41.3 tok/s, 28 tokens | Partial-prefix lookup used disk while paged remained disabled | VERIFIED-LIVE |
| Native fresh-process L2 | After quitting and relaunching only the isolated app, it emitted `R005=river-5-quartz; R200=river-200-quartz`; TTFT 0.92 s, 41.6 tok/s, 27 tokens | Fresh-process counters reached disk hits 4 / stores 5 / misses 8; paged hits/misses stayed 0 | VERIFIED-LIVE |
| Explicit TQ 4/4 validation | Selecting TurboQuant without widths visibly produced a blocking validation message; entering 4/4 saved and unloaded the model | Settings require explicit key and value widths; codec remains opt-in | VERIFIED-LIVE |
| TQ 4/4 cold/warm | Cold reload emitted `R011=river-11-quartz; R188=river-188-quartz` at TTFT 9.06 s, 14.9 tok/s; warm partial reuse emitted `R042=river-42-quartz; R177=river-177-quartz` at TTFT 1.33 s, 28.8 tok/s | Exactly 8 full-KV layers converted to TQ; all 40 rotating layers stayed native | VERIFIED-LIVE |
| TQ 4/4 fresh-process L2 | After full isolated-app restart, it emitted `R073=river-73-quartz; R231=river-231-quartz`; TTFT 1.44 s, 27.4 tok/s, 28 tokens | Disk hits 3 / misses 8 / stores 4; before topology 8 KV + 40 rotating, after 8 TQ + 40 rotating; paged 0 | VERIFIED-LIVE |
| Explicit paged request on mixed SWA | With GPU Cache visibly enabled in Settings, chat emitted `R099=river-99-quartz; R204=river-204-quartz`; TTFT 2.37 s, 29.3 tok/s, 28 tokens | Configured paged true, effective paged false, `is_paged_incompatible=true`, paged hits/misses 0, disk hits 2; TQ 8 + rotating 40 | VERIFIED-SOURCE + VERIFIED-LIVE |
| Defaults restored | The same isolated UI visibly saved GPU Cache Off and Codec Engine Selected; Prefix and Disk remained On. A cold reload then emitted `NATIVE-DEFAULTS-RESTORED-2041`; TTFT 14.91 s, 29.3 tok/s, 19 tokens | Endpoint returned fp16, transition null, 8 KV + 40 rotating, configured/effective paged false, disk hits 1, and MLXPress disabled | VERIFIED-LIVE |
| Native built-in tool continuation | With Thinking visibly Off and Codec Engine Selected, the model called built-in web search exactly once for `Osaurus GitHub repository`, then emitted exactly `TOOL-CONTINUED-NATIVE`; TTFT 1.68 s, 39.0 tok/s, 11 tokens | The UI rendered one search tool card followed by one assistant content turn; no protocol marker, raw tool JSON, loop, or empty post-tool answer leaked | VERIFIED-LIVE |
| TQ 4/4 built-in tool continuation | Settings visibly saved TurboQuant 4/4 and unloaded the model. After the cold reload, with Thinking still visibly Off, the model called built-in web search exactly once for `vMLX Swift GitHub repository`, then emitted exactly `TOOL-CONTINUED-TQ44`; TTFT 2.18 s, 14.2 tok/s, 12 tokens | `/admin/cache-stats` reported `effective_kv_mode=turbo(4,4)`, an exact 8 native-KV to 8 TQ transition, 40 rotating layers preserved, paged disabled/incompatible, disk hits 2 / misses 8 / stores 3, and MLXPress disabled | VERIFIED-SOURCE + VERIFIED-LIVE |
| Thinking-on built-in tool continuation | The model popover visibly reported Thinking On. Gemma called built-in web search exactly once for `Gemma 4 model family`, then emitted exactly `THINKING-ON-TOOL-CONTINUED`; TTFT 2.08 s, 38.5 tok/s, 14 tokens. The popover was then visibly restored to Thinking Off | `ChatTurnGenerationControls` freezes the explicit UI choice and applies it to every loop request and cap finalizer; `ChatEngine` maps it into `disableThinking`; `MLXBatchAdapter.additionalContext` maps that to `enable_thinking`. No reasoning block was visible, so semantic reasoning emission is not claimed | PARTIAL-SOURCE + VERIFIED-LIVE continuation |
| Last effective generation telemetry | The baseline app incorrectly reported the chat-prefill warm-up's `max_tokens=1`, `temperature=0`. In rebuilt Release executable SHA-256 `fc49aba748ef4c0388ddfbcbc2458663efd0b0e6b190a7de1cd77a72fa18a76b`, the exact JANG_4M UI turn visibly emitted `USER-VISIBLE-TELEMETRY-7319` at TTFT 1.27 s, 38.8 tok/s, 16 tokens. A second UI turn called built-in web search exactly once and emitted `TOOL-TELEMETRY-8842` at TTFT 2.02 s, 33.1 tok/s, 14 tokens. Logs then recorded a sentinel-only warm-up, while `/admin/cache-stats` still reported the visible request's bundle-owned `max_tokens=16384`, `temperature=1`, `top_k=64`, and `top_p=0.95` | Source trace found both pending and submitted settings recorders accepted `generation.warmupPrefill`; the Osaurus patch now excludes those housekeeping requests at both record sites and adds a focused classification test. The exact app used isolated bundle ID `com.dinoki.osaurus.gemma4telemetryproof` and fresh preferences | VERIFIED-SOURCE + VERIFIED-LIVE |
| Unavailable calculator request | The same UI prompt requested a calculator call, but Agent -> Abilities -> Tools visibly reported `0 of 0 assigned` and `No tools available`. Auto-discovery instead called time, built-in web search, and capability search before emitting the correct arithmetic result | This row cannot diagnose a calculator JSON schema because no calculator schema was assigned. It is retained as an Osaurus tool-availability/fallback UX finding, not classified as a Gemma parser failure | FAILED-LIVE / OUT OF CHECKPOINT |
| 12B physical footprint | Activity Monitor visibly showed the exact PID 23207 at 9.48 GB. `footprint` measured 9,712 MB current / 12 GB peak, including 8,634 MB dirty IOAccelerator memory. The model's weight files total 10,135,442,741 bytes | This is close to full weight residency and does not meet the repository's strict low-physical-footprint criterion | FAILED-LIVE |

The explicit TurboQuant mode reduced steady warm decode from about 41 tok/s to
27-29 tok/s in this run. That is recorded as an opt-in tradeoff, not used to
change the default or to hide a correctness issue.

### Current RAM root-cause classification

This is not a growing KV-cache or MLXPress leak:

- After the settings save unloaded the model, the same process fell to a
  1,015 MB physical footprint. After the default native reload it returned to
  9,712 MB, isolating the increase to model residency.
- Runtime telemetry reported MLXPress `enabled=false`, backend `none`, cold
  fraction null, routed bytes 0, and tiles 0. The load decision reported mmap
  requested and the ordinary Safe Auto 70% memory budget.
- This exact 12B bundle is dense: the decoded Gemma configuration has no routed
  experts. Osaurus/vMLX source allows experimental MLXPress only for explicitly
  opted-in routed bundles, so enabling the user toggle cannot create inactive
  expert pages for this model.
- The safetensors mmap loader wraps mapped shard memory in no-copy Metal
  buffers. Gemma's dense decode touches every decoder layer; the live VM map
  consequently attributed 8.4 GB to resident dirty IOAccelerator buffers while
  only 63 MB remained categorized as resident mapped-file pages.

Therefore the current evidence points to dense weight residency, not a cache
ownership bug that can be fixed with a cache guard. Meeting the strict
below-full-bundle footprint criterion would require a separately designed and
live-proven dense-weight streaming/offload path, with its decode-speed cost
measured. No such behavior is being silently enabled in this correctness patch.

## Required live closure matrix

Every row must use a fresh Release development build with an isolated bundle
identifier and preferences root. Visible UI behavior and matching runtime
telemetry are both required.

| Gate | Required evidence | Status |
|---|---|---|
| Current-build MXFP8 TQ transition | 48 total / 8 KV / 40 rotating before and 48 / 8 TQ / 40 rotating after, plus coherent visible native/TQ/restart output | VERIFIED-LIVE |
| Current-build JANG_4M TQ transition | Same current-build exact transition and coherent visible native/TQ/restart output | VERIFIED-LIVE |
| Paged default/effective policy | Default visibly off; explicit request on paged-incompatible Gemma truthfully remained effective off with zero paged hits/misses | VERIFIED-LIVE |
| SSD L2 with paged off | Both 12B formats restored partial prefixes from disk after process restart while paged counters remained zero | VERIFIED-LIVE |
| Explicit SSD L2 off | Visible SSD-Off save plus endpoint false/zero-counter proof | VERIFIED-LIVE |
| Fresh-process L2 restore | Both 12B formats showed post-restart disk hits and coherent changed-prefix continuations | VERIFIED-LIVE |
| Raw-prefill fallback | Fresh isolated storage had no prior cache entry. The new 8,635-token ledger visibly progressed from `Prefill 0/8635` through `8192/8635`, returned all three exact sentinels, and later telemetry showed disk misses while paged remained off | VERIFIED-LIVE |
| TurboQuant after L2 restore | Post-restart MXFP8 and JANG_4M rows re-established 8 TQ layers while preserving 40 rotating layers | VERIFIED-LIVE |
| TurboQuant off control | Current restored-default MXFP8 endpoint reported fp16, transition null, and zero TQ topology | VERIFIED-LIVE |
| Long rotating-window sentinel | Current JANG_4M run crossed the 1,024-token SWA window with an 8,635-token ledger, reproduced exact early/middle/tail sentinels, then performed coherent partial and post-restart L2 continuations | VERIFIED-LIVE |
| Ten-turn coherence | Twelve visible generated turns covered native cold/partial/restart, TQ cold/partial/restart, explicit paged request, restored defaults, an unavailable-tool fallback, and native/TQ/thinking-on web-tool continuations. Every turn exposed TTFT/tokens and recorded tok/s; all ended with visible content, with no protocol-marker leakage or loop | VERIFIED-LIVE |
| Tool/parser continuation | Native and explicit TQ 4/4 each completed one built-in web-search call plus an exact post-tool content turn with Thinking off. Required-tool enforcement across an assigned non-search schema, auto/no-tool, tool-error recovery, and a further post-tool user turn remain untested | PARTIAL |
| Reasoning/template state | Thinking Off and On were visibly toggled in the real model popover, source trace carries the explicit choice through every chat tool-loop reconstruction, and both modes produced exact post-tool content. The thinking-on Gemma row showed no reasoning block, so template/rendered-prompt or reasoning-channel proof is still missing; final UI state is Off | PARTIAL |
| Generation telemetry ownership | `last_effective_generation` must describe the last visible user/API request, not a discarded warm-up | Baseline failed; two-site source fix plus rebuilt Release UI and post-warm-up endpoint proof are current | VERIFIED-SOURCE + VERIFIED-LIVE |
| API parity | Chat Completions stream/non-stream, Responses stream/non-stream, Anthropic, and Ollama reconstruct the same visible content and finish reason | OPEN |
| Stop/cancel cleanup | Cancel during prefill and decode; next warm turn neither restores a poisoned partial boundary nor leaks stale output | OPEN |
| Memory settings next-load semantics | Strict custom 10% refusal and No Automatic Limits success are live-proven on the same 31B bundle; Safe Auto restored. Performance/Balanced modes remain open | PARTIAL |
| Low-RAM physical footprint | Activity Monitor main Memory must remain below full bundle size during load/generation | FAILED-LIVE on current 12B: 9.48 GB versus 10.14 GB weights, with 8.4 GB resident dirty IOAccelerator; historical 31B row also failed |
| Delegation/admission | Local text subagent, Computer Use/AppleScript, image generation/edit, and concurrent delegation receive the same memory admission decision before model eviction | OPEN; Osaurus-level row |

## Non-Gemma rows retained for the wider campaign

- Hybrid SSM/GDN/GLA families (Qwen 3.5/3.5 VL, Ornith, Bonsai, Nemotron and
  applicable MiniMax variants) require typed companion-state hit/miss/rederive
  counters, selective TQ conversion of full-attention KV only, native
  SSM/GDN/CCA companion-state preservation, partial-hit rollback/rederive
  proof, media-salt proof for VL, and coherent post-hit continuation with
  TurboQuant off/on. Detached async rederive is not accepted as a production
  substitute for prompt-boundary synchronization.
- LFM and MiniMax M2.7 also remain wider-campaign rows for explicit
  TurboQuant off/on, partial SSD-only restore, paged eviction where their
  topology supports it, multi-turn coherence, TTFT/tok/s, and physical
  footprint. Gemma evidence does not close those rows.
- DSV4 Flash and OpenPangu must remain hard-excluded from TurboQuant KV even
  under explicit user opt-in. DSV4/ZAYA and MiniMax-M3 native composite caches
  remain explicit exceptions until their typed native codecs are separately
  source- and live-proven. Their evidence must not be generalized from Gemma.
- JANGTQ/MXTQ weight-format correctness is a separate issue from TurboQuant KV
  cache encoding and must remain a separate matrix.
- AppleScript 8B JANG_6M import/discovery, Computer Use app switching, success
  finalization, unexpected tool fallback, and spawned-agent plugin inheritance
  remain Osaurus UI/runtime rows. They are not closed by this cache checkpoint.

No row becomes release-ready from source inspection, a configured setting, or
an aggregate counter alone.

## Current-source test note

With the Xcode 26 toolchain selected, the current patch's focused run passed
8/8 assertions: 3 transition telemetry rows, the MiMo selective-conversion
source contract, both paged-incompatible coordinator rows, Engine Selected
keeping TQ off, and explicit SSD-Off preservation through memory-safety
resolution. The
older `automaticRuntimeCachePolicyCoversDownloadedArchitectureFamilies`
matrix still contains a separate stale Hunyuan reasoning expectation
(`think_xml` while the current source returns `hy_v3`). That unrelated row is
not changed or counted as Gemma cache proof in this checkpoint.

The 2026-07-20 patch adds
`gemmaMixedTurboQuantWrappedRotatingResumesFromDisk`. Its focused current-source
run passed 1/1 in 4.586 seconds. The test crosses a rotating window, persists a
mixed compressed-TQ/native-rotating snapshot with paged requested but marked
incompatible, starts a fresh coordinator, restores only from typed SSD L2,
compares rotating metadata and tensors, appends the same suffix to both source
and restored rings, and compares their temporally ordered KV exactly. This is a
focused regression result only; the full Swift suite has not yet been rerun.

Two wider test invocations are not counted as passes. The command-line-tools
Swift 6.3.1 graph stopped while compiling the unrelated `MLXPressPolicyTests`
target because that toolchain could not import `Testing`. Retrying with Xcode
26.6 / Swift 6.3.3 built the package but the selected test executable stopped
before assertions because it could not find MLX's default Metal library. The
exact focused cache regression above remains the only new current-source test
result until the package test graph is run with both Swift Testing and the MLX
Metal library available.

The companion Osaurus telemetry patch was built in the Xcode workspace's Debug
test graph and then executed from that exact build with the enumerated Swift
Testing identifier
`OsaurusCoreTests/MLXBatchAdapterTests/lastEffectiveGenerationTelemetry_excludesChatPrefillWarmups()`.
It passed 1/1 in 0.000 seconds. The first invocation omitted the identifier's
trailing parentheses and therefore selected no leaf test; that invocation is
not counted.
