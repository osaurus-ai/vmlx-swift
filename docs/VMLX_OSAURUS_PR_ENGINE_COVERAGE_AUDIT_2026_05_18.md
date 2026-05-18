# vMLX Swift / Osaurus PR Engine Coverage Audit - 2026-05-18

This is the current crosswalk from recent user-authored Osaurus PRs and pinned
runtime libraries to `vmlx-swift` engine proof. It is intentionally stricter
than a package build: a row is not switch-ready unless the exact runtime surface
has model-aware cache proof, multi-turn coherency, visible stop behavior, and no
hidden sampler/parser guard.

Fresh inspection commands used in this pass:

```sh
gh pr list -R osaurus-ai/osaurus --author @me --state all --search "created:>=2026-04-24" --json number,title,state,url,headRefOid,updatedAt,mergedAt,closedAt,isDraft,mergeStateStatus --limit 100
gh pr list --repo osaurus-ai/osaurus --state all --author @me --limit 60 --json number,title,state,isDraft,author,headRefName,baseRefName,updatedAt,createdAt,mergedAt,url
gh pr view -R osaurus-ai/osaurus <pr> --json number,title,state,headRefOid,mergeCommit,commits,files
git -C /Users/eric/osaurus-staging show HEAD:osaurus.xcworkspace/xcshareddata/swiftpm/Package.resolved
gh api 'repos/osaurus-ai/{repo}/commits?since=2026-04-24T00:00:00Z&until=2026-05-18T23:59:59Z&per_page=100'
gh api repos/osaurus-ai/{repo}/compare/{pin}...main
```

MTP tuning behavior commits covered by this refresh:

```text
3889499 fix(mtp): use qwen tuning file for auto depth
6af1096 fix(mtp): require tuning for qwen auto launch
d228fdd fix(mtp): expose tuning-gated status snapshot
1a166ad test(mtp): record missing qwen tuning evidence
```

2026-05-18 continuation refresh:

- Single-package Osaurus switch status, current checkpoint:
  package/wiring readiness is high but still draft-gated; full runtime
  production readiness remains partial until the live model matrix closes the
  family-specific rows below. Current rough split is package wiring ~90%,
  overall Osaurus production readiness ~60-65%.
- Consolidated package graph proof now exists on the vmlx-swift PR lane:
  `swift package describe --type json`, focused target builds for Hub,
  HuggingFace, and Tokenizers, focused runtime tests, and release
  `RunBench` build all pass after the vendored module prefix and yyjson
  dependency cleanup.
- OsaurusCore has a draft switch PR wired to the consolidated `vmlx-swift`
  revision. `OsaurusCore` builds against the single package, and focused
  runtime/template/batch adapter tests pass, but the PR must stay draft until
  the live runtime rows below and downstream app/server probes are complete.
- New non-Kimi inventory/metadata/MTP policy proof:
  `docs/local/live-model-matrix/20260518T054941Z_non_kimi_inventory_pr1/`
  found 30 non-Kimi local bundles, with 5 tensor-proven MTP candidates and no
  MTP inferred from CRACK names. Metadata/template rerun with `--allow-huge`
  passed 60/60 rows under
  `docs/local/live-model-matrix/20260518T055141Z_non_kimi_metadata_allowhuge_pr1/`.
  Focused MTP policy profile recorded 11 pass rows and 19
  `n-a:no-mtp-tensors` rows under
  `docs/local/live-model-matrix/20260518T055344Z_non_kimi_mtp_policy_pr1/`.
- Representative MTP census proof:
  `/Users/eric/models/JANGQ/Qwen3.6-27B-MXFP4-MTP` is tensor-proven,
  `vmlx_mtp_tuning.json` driven, and auto-launches with tuned depth 2.
  `/Users/eric/models/JANGQ/Qwen3.6-35B-A3B-JANG_2K-MTP` is tensor-proven but
  tuning-blocked, so auto-launch remains off. The Qwen and MiniMax CRACK
  bundles inspected in the same census had no usable MTP tensors and stay off.
- ZAYA1-8B-JANGTQ4 release-path checkpoint:
  `docs/local/live-model-matrix/20260518T061037Z_zaya_jangtq4_turnmatrix_pr1_post_applicability/`
  passes config/template, production tiered cache off/on, batch single/chat,
  growing chat cache, disk restore, B=2 concurrent, and per-slot sampler rows.
  Production rows decode around 66-67 tok/s with ~5.1 GiB RSS and coherent
  reasoning on/off behavior. ZAYA CCA is a path-dependent hybrid cache topology,
  so pure paged-prefix cache-hit and live TurboQuant KV B=2 probes are now
  explicit `n-a` rows; the correct proof path is disk/SSM restore plus JANGTQ
  expert-kernel config/runtime evidence, not a sampler guard.
- Harness fix from the ZAYA checkpoint:
  `RunBench` now checks `cacheRequiresDiskBackedCoordinatorRestore` before
  running pure paged-prefix cache-hit and live TurboQuant KV B=2 rows. This
  prevents path-dependent CCA/SSM/sliding-window models from being mislabeled
  failed by a non-applicable probe. Focused regression:
  `RunBenchApplicabilityFocusedTests` passes 2/2.

- Live GitHub refresh at `2026-05-17 19:57 PDT` still returns 15
  user-authored Osaurus PRs in the 2026-04-24+ runtime-pin window. The
  crosswalk rows below cover each returned PR: #931, #932, #943, #944, #946,
  #953, #967, #990, #993, #998, #1037, #1057, #1066, #1073, and #1110.
- GitHub still reports Osaurus PR #1110 as open, non-draft, all checks green,
  and `mergeStateStatus=DIRTY`; do not treat it as merged switch state.
- Current `osaurus-staging` branch is `feat/dsv4-vmlx-pin` at
  `b0a96dd4 Wire native DSV4 tokenizer bridge`, with local uncommitted Osaurus
  edits present. Those local edits are not part of the pinned public PR state.
- Live compare refresh of the four pinned runtime repos found no topology change
  from the prior audit: `Jinja` is identical to default `main`,
  `vmlx-swift-lm` pin `2cc64dd` is two commits ahead of its default `main`,
  `swift-transformers` pin `087a66b` is still fork-diverged from tokenizer
  speed commits on default `main`, and `mlx-swift` pin `0a56f904` is still an
  Osaurus fork lane with stream/wired-limit work rather than upstream parity.
- Fresh `vmlx-swift` evidence added after the first audit:
  `20260518T_gemma4_e2b_refresh_no_fake_guards/` and
  `20260518T_ling_jangtq2_no_guard_refresh/`. Both prove no hidden sampler
  guard behavior; failures, where present, were harness/product-budget issues,
  not decode fixes.
- VLM JANG weight loading now matches the LLM factory and passes the real
  `quantizationContainer?.quantization` value into `loadWeights` instead of the
  deprecated `baseConfig.quantization` alias. This keeps MXFP4/MXFP8 group-size
  inference source-backed for VL/Omni bundles; it is not a sampler, parser, or
  EOS workaround. `swift build -c release --product RunBench` passed after the
  change.
- Release-mode Swift Testing initially failed before focused MLXLM tests because
  `MLXTests/WiredMemoryTests.swift` referenced DEBUG-only wired-memory event
  helpers. The tests are now explicitly DEBUG-gated with a release skip, keeping
  the production `WiredMemoryManager` event surface unchanged. Focused release
  tests now pass for `vlmJangLoadUsesQuantizationContainer` and
  `nilServerSamplingFieldsDoNotAddFakeGuards`.
- DSV4 reasoning policy no longer lets the deprecated
  `VMLINUX_DSV4_FORCE_DIRECT_RAIL` environment key silently override an explicit
  `reasoning_effort=max` request. The first red test proved the old behavior
  rewrote the request to `enable_thinking=false`; the fixed release test suite
  now passes 6/6 for `DeepseekV4ReasoningPolicyTests`.
- Qwen native-MTP auto-launch is now driven by the bundle-local
  `vmlx_mtp_tuning.json`. `MTPBundleInspector` reads the file into
  `MTPBundleStatus`, and `NativeMTPAutoDecodePolicy` returns a depth only when
  the tuning row is validated, output-equivalent, unblocked, and tensor-proven.
  The old hardcoded Qwen profile/depth rules are removed; local 27B MXFP4 proves
  `best_depth=2` is honored, and local 35B JANG_2K proves a blocked tuning row
  keeps auto-launch off.

2026-05-17 20:25 PDT live refresh:

- `gh pr list --repo osaurus-ai/osaurus --state all --limit 20` shows the
  newest Osaurus PRs are mostly app/plugin/UI work: #1145, #1144, #1141,
  #1140, #1139, #1138, #1137, #1136, #1135, #1134, #1132, #1131, #1130,
  #1128, #1127, #1126, #1125, #1124, and #1123 are merged; #1133 remains open
  draft for plugin host multimodal contracts. None of those change the
  vMLX runtime pin window recorded below.
- `gh pr view 1110` still reports PR #1110 open/non-draft with green checks but
  `mergeStateStatus=DIRTY`; it has no public PR comments or reviews in the
  queried metadata. Its commits remain the DSV4 runtime chain ending at
  `b0a96dd4 Wire native DSV4 tokenizer bridge`.
- `osaurus-staging` still resolves `mlx-swift 0a56f904`, `Jinja 58d21aa`,
  `swift-transformers 087a66b`, and `vmlx-swift-lm 2cc64dd` in the workspace
  `Package.resolved`. Local Osaurus edits are present but are not pinned public
  PR state.
- MTP display/helper semantics were tightened after the initial tuning-file
  patch: `MTPBundleStatus.canAutoLaunchMTP`, `speculativeDecodeEnabled`, and
  `VMLXServerRuntimeSettings.effectiveMTPLaunchMode(for:)` now require usable
  `vmlx_mtp_tuning.json` metadata, not just tensor evidence. Tensor-proven Qwen
  bundles missing tuning report off/blocked and `statusLine` says tuning is
  required.

2026-05-17 20:44 PDT live refresh:

- `gh pr list --repo osaurus-ai/osaurus --state all --limit 12` shows one newer
  merged README-only PR, #1146, plus the same app/plugin/UI/coordinator/doc
  merges (#1145, #1144, #1141, #1140, #1139, #1138, #1137, #1136, #1135,
  #1134). Open PR #1133 remains a draft plugin-host multimodal contract. These
  do not change the vMLX runtime pin window.
- `vmlx-swift` head now exposes `MTPBundleStatus.snapshot` for Osaurus status
  JSON, including computed gates that raw `Codable` fields do not carry:
  `has_usable_native_mtp_tuning`, `can_auto_launch`, and
  `requires_native_mtp_tuning_before_auto_launch`.
- If metadata or tensor names indicate MTP compatibility and the bundle-local
  `vmlx_mtp_tuning.json` file is absent, `MTPBundleInspector` records
  `tuning_file_missing=vmlx_mtp_tuning.json` in `configEvidence`. This proves the
  runtime looked for the same kind of bundle-local sidecar as
  `generation_config.json` and failed closed instead of falling back to a name
  or profile rule.
- Focused verification for this refresh:
  `MTPRuntimeFocusedTests|VMLXServerRuntimeSettingsTests` passes 65/65 with the
  Xcode framework path, including the new factory source guard that both LLM and
  VLM factories inspect MTP, resolve native activation before weight loading,
  pass `loadPreservedMTP: loadNativeMTP`, preserve `generationDefaults:
  generationConfig`, and carry `mtpStatus` into `ModelConfiguration`.

2026-05-17 21:09 PDT `vmlx-swift-lm` parity refresh:

- The current reference repo state is still dirty and concurrent-agent active:
  `/Users/eric/vmlx-swift-lm` is `main...origin/main [behind 5]` with local
  edits across factories, DSV4, Hy3, ZAYA, cache, BatchEngine, templates, and
  tests. Treat that worktree as reference material only; do not overwrite it or
  copy unreviewed local edits.
- Upstream reference commits checked in this pass:
  `4546a5d fix(dsv4): render DSML tools in fallback template`,
  `e1280c3 fixed nested ternary operator error during build`,
  `6561a72 fix(dsv4): preserve overlap compressor state across decode`,
  `f728718 fix(dsv4): mask HSA top-k scores causally`, and
  `4365651 fix: decode nested ZAYA JANGTQ bits`.
- Current `vmlx-swift` has focused parity tests for the runtime-relevant pieces:
  DSV4 DSML tools in the fallback and standalone templates, no-system tool
  rendering, DSV4 native encoder system/user separation, Jinja
  `tojson(separators=...)`, ratio-4 overlap compressor preservation across
  decode calls, causal masking before HSA top-k, nested ZAYA/ZAYA1-VL
  JANGTQ_K gate/up/down bit decoding, Qwen/ZAYA/GLM/Gemma/LFM/Smol VL extent
  guards, Qwen3.6 VLM native-MTP MRoPE continuation, Qwen3.6 VLM sparse-MoE MTP
  sidecars, and Gemma3/Gemma4 masked-scatter error propagation.
- Focused verification for this parity pass:
  `DeepseekV4IndexerCausalTopKTests|DeepseekV4ChatTemplateFallbackFocusedTests|ZayaConfigDecodeFocusedTests|VLShapeGuardFocusedTests`
  passes 31/31 with the Xcode framework path. This is source/test parity for
  those fixes only; it is not a replacement for the live DSV4, ZAYA, Gemma, VL,
  cache, speed, and low-footprint gates listed below.

2026-05-17 21:10 PDT live refresh:

- Open Osaurus PR #1133 remains the relevant post-runtime watch item for the
  package switch: it is draft/behind and its author comments explicitly frame
  multimodal plugin contracts as spec-first because not every model supports
  every modality. The `vmlx-swift` switch PR therefore needs explicit
  per-model capability/status JSON for text, vision, audio, video, tools,
  reasoning, native MTP, and cache topology, plus unsupported-modality error
  shape and logging/redaction boundaries.
- The single-package `VMLX` umbrella surface now has focused test coverage for
  the Osaurus-facing runtime types it must expose: `GenerationConfigFile`,
  `JangCapabilities`, parser resolution/tool/reasoning parser types, and the
  `MTPBundleStatus.snapshot` JSON fields that tell Osaurus whether
  bundle-local `vmlx_mtp_tuning.json` permits native MTP auto-launch.
- Follow-up implementation adds the status surface #1133 needs without changing
  decode behavior: `JangCapabilities` now parses explicit
  `supports_text` / `supports_vision` / `supports_video` / `supports_audio`
  booleans, and `ModelRuntimeCapabilitySnapshot` emits a single Codable
  support matrix (`supported` / `unsupported` / `unknown`) with parser stamps,
  cache type, `generation_config.json` defaults, and native-MTP tuning status.
  Focused umbrella tests cover the `VMLX` re-export, JSON keys, media support
  parsing, native-MTP support, and served-name preservation for
  `ResolvedModelConfiguration`.

2026-05-17 21:12 PDT live PR/comment refresh:

- Current recent Osaurus PR state:
  #1146, #1145, #1144, #1141, #1140, #1139, #1138, #1137, #1136, #1135,
  #1134, #1132, #1131, #1130, #1128, #1127, #1126, #1125, #1124, #1123,
  #1122, #1120, #1119, #1117, #1116, and #1115 are merged. #1133 remains
  open draft/behind, #1118 remains open/behind, and #1110 remains open/dirty.
- #1132 adds the multimodal plugin IO-lane spec. #1133's comments still make
  the follow-up explicit: keep the multimodal contract spec-first until the
  support matrix says which model/provider families accept image/audio/video,
  which unsupported-modality errors plugins see, and where redaction/logging
  boundaries sit. The `vmlx-swift` package switch must therefore expose
  capability/status JSON rather than letting Osaurus infer modality from model
  names.
- #1120 shrinks first-turn prompt tool surface and has a direct reviewer concern
  about KV-cache invalidation if contexts are modified. Engine-side contract:
  vmlx hashes the already-rendered token stream plus model/media/cache-policy
  salts, so tool-schema prompt edits can only reuse the shared prefix and must
  re-prefill the modified suffix. Fresh focused coverage added:
  `promptToolSurfaceEditsNeverReturnFullPromptHit` in
  `CacheCoordinatorTopologyFocusedTests`; the focused suite now passes 26/26.
- Real local Qwen tuning-file proof was refreshed with
  `VMLX_MTP_REAL_BUNDLE=/Users/eric/models/JANGQ/Qwen3.6-27B-MXFP4-MTP`
  and `VMLX_MTP_REAL_BUNDLE_EXPECTS_VL=1`; the optional real-bundle
  `MTPRuntimeFocusedTests/optionalRealLocalMTPBundleInspection` row passed and
  proved current code sees tensor evidence, VL tensors, usable
  `vmlx_mtp_tuning.json`, speculative launch, and `loadConfiguration.nativeMTP`.
- #1119 adds Osaurus model idle residency policy. `vmlx-swift` already exposes
  server runtime power settings and cache coordinator release/disable surfaces,
  but the switch PR still needs a live deep-sleep/wake proof against the actual
  Osaurus server process before claiming production lifecycle readiness.
- #1118 is PocketTTS language selection and remains open/behind. It is mostly
  output-speech UI/config work, not a vmlx inference-engine dependency, but it
  touches resolver pins; the switch PR must re-run pin-integrity checks after
  rebasing any open voice/runtime PRs.

2026-05-17 21:23 PDT support-matrix validator refresh:

- The #1133 unsupported-modality/error-shape gap now has a concrete
  `vmlx-swift` API, not just descriptive JSON. `ModelRuntimeCapabilityRequest`
  summarizes requested `text`, `vision`, `video`, `audio`, `tools`,
  `reasoning`, and `native_mtp` lanes without retaining prompt text, paths,
  image bytes, or audio samples. `ModelRuntimeCapabilitySnapshot.validate`
  returns deterministic `unsupported_modality` and `unknown_modality_support`
  issue rows with redacted log fields. This lets Osaurus fail closed before
  routing multimodal plugin requests to a model/provider that has not proven the
  requested lane.
- Focused verification for this refresh:
  `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcrun swift test
  --filter 'VMLXUmbrellaProductTests' --jobs 2 -Xswiftc -F -Xswiftc
  /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/Library/Frameworks`
  passes 7/7. The suite covers VMLX re-export of the validator types,
  unsupported-lane JSON shape, unknown-lane fail-closed/default behavior,
  `.allowUnknown`, and UserInput-derived request summaries that do not leak
  prompt content.
- Follow-up #1119 settings validation tightened the server power contract:
  positive light/deep sleep timers are now required when set, and deep sleep
  ordering is checked only after both timers are valid. This prevents Osaurus
  from silently accepting negative or zero idle-residency values before the live
  deep-sleep/wake gate is run. Focused verification:
  `VMLXServerRuntimeSettingsTests|RuntimeMoETopKOverrideFocusedTests`
  passes 19/19 with the Xcode framework path.
- Follow-up native-MTP settings validation made the cache-boundary policy
  non-optional: `keepDraftCacheSeparate=false` and
  `acceptedTokensOnlyEnterBaseCache=false` are now validation errors. This keeps
  Qwen MTP sidecar/private draft state out of the committed prefix/paged/SSM
  cache unless the verifier accepted the tokens.
- Follow-up request-time server validation layers the support matrix with
  server toggles. `VMLXServerRuntimeSettings.validateRequest` now returns
  `server_modality_disabled` for `multimodal.vlmMode = force_off`, disabled
  video/audio lanes, or `mtp.mode = off`, while preserving the same redacted
  issue JSON shape. Focused verification:
  `VMLXServerRuntimeSettingsTests|RuntimeMoETopKOverrideFocusedTests`
  passes 23/23 with the Xcode framework path.
- Follow-up parser settings validation rejects unknown
  `toolParserOverride` / `reasoningParserOverride` strings while allowing known
  aliases and explicit no-op values (`auto`, `none`, `off`, `disabled`). This
  keeps Osaurus parser pickers from passing stale UI labels into the engine.
- Follow-up request-summary redaction coverage now verifies
  `ModelRuntimeCapabilityRequest(input:)` records text, image, video, audio,
  tools, reasoning, and native-MTP lanes from `UserInput` without serializing
  prompt content or tool names. Focused verification:
  `VMLXUmbrellaProductTests` passes 7/7 with the Xcode framework path.
- Follow-up media-cache settings validation now rejects
  `multimodal.requireMediaSaltForCache=false` whenever prefix, paged KV,
  block-L2, or legacy disk cache reuse is enabled. This keeps image/video/audio
  requests from sharing cache keys by text alone.
- Follow-up native-MTP activation hardening now applies the same tuning gate to
  the low-level direct factory path: explicit `LoadConfiguration.nativeMTP=true`
  / `VMLX_NATIVE_MTP=1` activation requires complete tensor evidence and usable
  bundle-local `vmlx_mtp_tuning.json`; tensor evidence alone throws
  `requestedWithoutUsableTuning` instead of preserving MTP sidecar weights.

## Current Switch Verdict

Not ready to say "single `vmlx-swift` dependency is production-clear for all
Osaurus models." Large parts are proven, but there are still explicit open
promotion blockers:

- DSV4: coherent post-fix chat exists, but long-context/vector drift, API
  route matrix, speed matrix, and low-footprint production gates remain open.
- ZAYA1-VL JANGTQ_K: still fails the production math row and cold structured VL
  cache budget. This cannot be hidden with top-k, repetition penalty, or looser
  validators.
- MiniMax large CRACK: cache-chat and strict TQ B=2 now pass after the real TQ
  cache-codec tail fix, but low-footprint active-routed proof is still open.
- Hy3 JANGTQ_K: old active-streaming evidence exists, but it needs a current
  non-Kimi all-model rerun before Osaurus promotion.
- GPT-OSS / GLM5 / Mistral4 / Pixtral: parser/unit coverage exists, but there
  are no local live decode rows in this pass.
- Omni live voice: core text/image/audio/video rows pass. A focused 2026-05-18
  cache-on repeat gate fixed the iterator bench cache-store evidence gap, but
  repeated live audio still remains a semantic quality/termination gate.
- Qwen high-resolution video: bounded media resize rows pass; raw 1080p video
  is not production-clear because the pre-fix row peaked at 164.2 GiB physical
  footprint.

## Osaurus PR Crosswalk

The main crosswalk below focuses on the active 2026-04-24 and newer runtime
pin window. Earlier April PRs still matter as lineage inputs, especially #917
structured tool calls/thinking defaults, #878 Qwen 3.6/JANGTQ, #867
template-driven reasoning detection, #863 runtime pin and lifecycle fixes, #799
Gemma4 hybrid KV, #795 VLM classification/media persistence, and tool/document
surface fixes in #827/#791/#779. Those older PRs are not counted as
switch-ready by age or merge state; they are covered only when the corresponding
row below has current `vmlx-swift` live proof.

| PR | State | Runtime payload | Current `vmlx-swift` coverage | Remaining requirement |
| --- | --- | --- | --- | --- |
| #931 `fix(ci): bump vmlx-swift-lm pin to 5b84387` | merged | Early resolver pin movement. | Captured in `docs/VMLX_OSAURUS_PR_PIN_LINEAGE_2026_05_17.md`; later pins supersede this. | No standalone engine blocker; use as lineage only. |
| #932 `feat: honor per-model generation_config.json sampling defaults` | merged | Bundle `generation_config.json` defaults must flow into local generation. | Current ledger rows report bundle defaults per family: e.g. MiniMax `temp=1.000 topP=0.950 topK=40 rep=nil`, Qwen `topK=20`, ZAYA `temp=0.600 topP=1.000 topK=0`, Laguna `temp=0.700 topP=0.900`. | Keep every new live row printing resolved defaults. Do not add hidden fallback penalties or top-k clamps when a model loops. |
| #943 `feat: jang_config.json chat metadata + LFM false-thinking-block fix` | closed/unmerged | Early branch for chat metadata and false-thinking-block handling. | Superseded by #944 and later parser/no-hidden-reasoning rows. | Lineage only. Do not count #943 as shipped resolver state. |
| #944 `feat: jang_config.json chat metadata + vmlx bump` | merged | `jang_config.json` chat metadata, DSV4/Kimi/LFM routing, reasoning capability metadata. | DSV4 template/metadata rows and the non-Kimi config/template sweep are current. Kimi is deliberately excluded by user direction. | LFM is not live-cleared by this matrix; if Osaurus exposes it, add a live multi-turn/cache row. |
| #946 `feat(model-picker): Performance filter` | merged | UI filtering based on model performance/fit. | Engine docs now record speed/RSS caveats by family, especially DSV4, MiniMax, ZAYA, Qwen, Gemma4, and Omni. | Osaurus UI must consume these as explicit capability/performance metadata, not infer from name or size alone. |
| #953 `fix(preflight): detect mislabeled JANGTQ bundles + vmlx fa77575 auto-correct` | merged | Mislabeled JANGTQ detection, sidecar/family preflight, streaming/event mapping. | Current ledger keeps tensor/sidecar evidence separate from model names. ZAYA/MiniMax/Qwen CRACK rows explicitly state non-MTP unless tensor evidence exists. | Add switch-PR resolver tests that reject name-only MTP/JANGTQ claims. |
| #967 `feat: Nemotron-3 Hybrid + storage fix + multimodal API` | merged | Nemotron hybrid, multimodal content parts, storage, early resolver skew. | Fresh Omni rows cover JANGTQ/JANGTQ4/MXFP4 core text/image/audio/video, media salt, hybrid SSM, and BatchEngine rows. | Repeated cache-on audio quality remains partial; resolver skew means the Osaurus switch PR must pin one path only. |
| #990 `feat(api): OpenAI input_audio + video_url content parts` | closed/unmerged | API-surface context for audio/video content parts. | Do not treat as shipped resolver state. Its content is effectively covered later by #967/#1073 live voice/multimodal rows. | No direct switch blocker, but API route probes remain package-wide open. |
| #993 `fix(preflight): reject JANGTQ Mistral 3 / Laguna before vmlx loads` | merged | Preflight/fail-fast behavior and converged Jinja identity. | Laguna is live-proven; Mistral3/Laguna parser and JANGTQ Hadamard fixes are represented in parser/cache refresh and ledger notes. | If Osaurus re-enables Mistral3 JANGTQ, require a current live row; do not rely on preflight-only proof. |
| #998 `fix(quality): revert default KV mode .turboQuant(4,4) -> .none` | merged | Important no-fake-default precedent: global TQ caused degenerate repetition, real fix was to stop forcing it. | `vmlx-swift` now uses explicit TQ rows only. MiniMax strict TQ B=2 was fixed by preserving exact prompt tail in the TQ codec, not by forcing sampler policy. | Keep TQ off unless explicitly selected or model-compatible; never use global TQ as quality default. |
| #1037 `Ling/ZAYA hardening + BatchEngine lifecycle` | merged | Ling/Bailing, ZAYA, BatchEngine lifecycle, topology-aware cache. | Ling JANGTQ2/MXFP4 pass; ZAYA text JANGTQ4/JANGTQ_K pass; ZAYA1-VL JANGTQ4 passes. Cache proof is topology-specific: disk/SSM/CCA, not generic prefix hit. | ZAYA1-VL JANGTQ_K remains partial; Hy3 K needs current rerun. |
| #1057 `MiniMax speed fix` | merged | MiniMax speed/lifecycle, typed load config, VLM detection, tokenizer/Jinja compatibility. | Large MiniMax JANGTQ_K/JANG cache-off infer rows pass; production-shaped chat-cache row passes; strict TQ B=2 now passes after `6560879`. | Low-footprint active-routed MiniMax proof is still open. Shape-inferred 6-bit metadata repair in JANG_K should be corrected in bundle or explicitly accepted. |
| #1066 `pin DSV4 vmlx update` | merged | DSV4 tokenizer/cache/runtime pin, local tokenizer fallback. | DSV4 separator fix and template kwargs rows pass; DSV4 live cache OFF/ON chat is coherent. | DSV4 remains partial until long-context/vector/API/speed/low-footprint gates pass. |
| #1073 `Nemotron Omni live voice input path` | merged | Live voice, Parakeet/RADIO, media-cache token-aware restore, DSV4 pool/compressor fixes. | Omni JANGTQ/JANGTQ4/MXFP4 core matrices pass; current docs track Parakeet chunk concat caveat. Focused 2026-05-18 repeat-audio gate now proves block-L2 and `ssm_companion` writes for BatchEngine and manual TokenIterator paths after the bench store fix. DSV4 pool/compressor lineage is recorded. | Repeated cache-on audio semantic quality/termination and package-wide HTTP route proof remain open. |
| #1110 `Harden DSV4 reasoning gates and runtime proof` | open, dirty | Native DSV4 chat encoder/tokenizer bridge, live DSV4 proof, runtime pin check. Current Osaurus head pins `vmlx-swift-lm 2cc64dd`. | `vmlx-swift` has DSV4 prompt-boundary fix and partial live proof, but it does not yet close the full #1110 bar. | Do not treat #1110 as merged release state; switch PR must resolve dirty state and rerun DSV4 release gates. |
| #1118 `Add PocketTTS language selection` | open, behind | TTS language UI/config and resolver-pin churn. | No direct vmlx inference-engine change; keep Omni/Parakeet input-audio evidence separate from output TTS. | Re-run pin-integrity checks after any rebase/merge before the package switch. |
| #1119 `Add model idle residency policy` | merged | Server idle residency, unload/sleep policy, runtime lifecycle hooks. | `VMLXServerRuntimeSettings.power` documents light/deep sleep settings and cache release/disable APIs exist. | Needs live Osaurus server deep-sleep/wake proof with loaded models before lifecycle readiness is claimed. |
| #1120 `Shrink first-turn prompt tool surface` | merged | Prompt/tool-surface TTFT shrink, prefix-hash/eval concern, tool schema prompt composition. | vmlx cache tiers are rendered-token keyed and salted by model/media/KV policy/reasoning scope. Fresh focused test `promptToolSurfaceEditsNeverReturnFullPromptHit` passes as part of 26/26 `CacheCoordinatorTopologyFocusedTests`. | Osaurus should pass the rendered prompt/token stream through vmlx; do not add app-layer cache reuse based on logical conversation IDs. |
| #1132 `Specify multimodal plugin IO lanes` | merged | Spec for plugin image/audio/video IO lanes. | `ModelRuntimeCapabilitySnapshot` and explicit media support booleans expose the engine-side support matrix Osaurus needs. | Complete live per-family capability matrix and unsupported-modality error shape before exposing broad plugin multimodal routing. |
| #1133 `Pin plugin host multimodal request contracts` | open draft, behind | Contract tests for plugin-host multimodal requests; comments say spec-first/not ready. | vmlx now exports per-model support JSON, native-MTP status, parser stamps, generation defaults, and cache type for Osaurus to consume. | Keep draft until the model/provider support matrix, fallback/error shape, and redaction/logging boundaries are settled. |

## Pinned Dependency Window

Current open Osaurus PR head (#1110) resolves:

| Package | Revision | Commit fact | Pin topology | `vmlx-swift` requirement |
| --- | --- | --- | --- | --- |
| `osaurus-ai/mlx-swift` | `0a56f904` | `2026-05-01 deps(mlx): advance submodule to 96aa27a5 (mx::malloc tracer for Bug 2)` | Diverged from default `main`: pin carries Osaurus stream/default-stream, wired-limit, evalLock removal, custom-kernel lifetime, and malloc-tracer work; default main also has unrelated doc/API changes not in the pin. | Compare behavior through local Cmlx/MLX checkout; package identity alone is not enough. Large-allocation tracing and stream behavior remain perf/debug surfaces for long-prompt and M5 speed gates. |
| `osaurus-ai/Jinja` | `58d21aa` | `2026-05-01 fix(parser): for-loop iterable accepts binary expressions` | Identical to default `main` at refresh time. | Vendored Jinja fallback tests must keep binary iterable and `tojson(separators:)` behavior. |
| `osaurus-ai/swift-transformers` | `087a66b` | `2026-05-11 fix(tokenizer): skip unused placeholders in delimiter regex` | Diverged from default `main`: pin carries `deps: use osaurus Jinja` plus unused-placeholder delimiter skip; default `main` carries later tokenizer speed work (MetaspaceDecoder, byte-level regex/table, Bert regex, Unigram O(N)). | Tokenizers must skip `<unusedN>` placeholders and preserve wrapper-token paths for MiniMax, DSV4, Qwen, and Omni. Later speed commits are performance-watch items unless the switch PR repins or vendors them. |
| `osaurus-ai/vmlx-swift-lm` | `2cc64dd` | `2026-05-16 Wire native DSV4 chat encoder` | Pin is two commits ahead of default `main`: `c90898fb test(tooling): keep MiniMax stream open across chunks` and `2cc64dd Wire native DSV4 chat encoder`. | DSV4 native chat encoder/tokenizer bridge is part of the switch-readiness target, not yet fully released in `vmlx-swift` by a complete DSV4 gate. MiniMax streaming/open-chunk behavior must stay covered by the no-hidden-guard and chat-cache rows. |

Recent dependency scan, 2026-05-04 through 2026-05-18:

- `vmlx-swift-lm` contains the bulk of recent runtime fixes: DSV4 SWA/CSA/HSA
  correctness, DSV4 paged-incompatible disk restore, MiniMax template and
  streaming fixes, Ling/Bailing hybrid cache handling, ZAYA CCA cache and
  JANGTQ_K bit decoding, Omni live audio/RADIO/Parakeet/media-cache work, and
  DSV4 native chat encoder/tokenizer bridge. `vmlx-swift` cannot be called a
  complete replacement until the local ledger maps each of those families to
  real multi-turn/cache/media proofs or an explicit blocker.
- `swift-transformers` default-branch tokenizer speed work is not in #1110's
  pinned runtime. It should be tracked as Osaurus-switch performance risk, not
  cited as current proof.
- `mlx-swift` pin is intentionally an Osaurus fork lane, not default upstream
  `main`; stream/default-stream and malloc-tracer behavior are part of the
  low-level performance/debug contract.

## Dependency Fixes Mapped To Engine Surfaces

| Upstream fix family | Required engine surface | Current proof | Status |
| --- | --- | --- | --- |
| Jinja parser and compact tool JSON | DSV4/Kimi/Gemma4/ZAYA/Laguna tool templates render without broken syntax or bloated separators. | `docs/local/production-readiness/20260517T2200_jinja_pin_parity/` and parser/cache refresh rows. | Covered for non-Kimi; Kimi excluded by instruction. |
| Swift-transformers unused placeholder skip | Added-token delimiter regex must not include thousands of unused placeholders and must preserve special wrapper tokens. | Vendored tokenizer static check plus MiniMax/DSV4/Qwen/Omni live template rows. | Covered for pinned behavior; later speed commits not yet part of Osaurus pin. |
| Generation defaults | Bundle defaults are source of truth before explicit request override. | Ledger rows print temp/topP/topK/minP/rep per family. | Covered for tested rows; require same telemetry for new rows. |
| Hybrid SSM / CCA / SWA cache | Cache proof must be topology-specific, not generic prefix-hit. | Qwen/Ling/ZAYA/Gemma4 rows record disk L2, SSM companion, CCA, SWA incompatibility, and media salt where applicable. Fresh Gemma E2B refresh records disk hits/stores plus VL media-salt restore; fresh Ling no-guard refresh confirms Bailing template/decode stress without fake sampler fixes. | Covered for listed PASS rows; DSV4 long-context and ZAYA1-VL K remain open. |
| TurboQuant KV | Explicit TQ mode must preserve coherency and prove actual compression. | `20260518T_minimax_m27_jangtqk_tq_tail_fix_exact/` proves actual TQ transitions and exact outputs after tail preservation. | Fixed for MiniMax strict row; keep family-by-family gates. |
| VL/media salt | Image/video/audio state must be isolated across turns and cache hits. | Qwen, ZAYA1-VL, Gemma4, and Omni rows prove same/different media behavior where implemented. | Raw Qwen high-res video and repeated Omni cache-on audio remain open. |
| Reasoning on/off | No fake close; reasoning off must affect template/runtime where supported, and visible output must remain coherent. | Gemma4 reasoning matrix, MiniMax rows, DSV4 reasoning kwargs, Ling/Bailing aliases. Fresh Gemma E2B no-guard red/green pair proves the harness now accepts coherent stellar equivalents instead of forcing decode behavior; fresh Ling row proves the Russian stress prompt with `temp=0.7` stops normally. | Covered for tested families; package-wide model matrix still open for absent local bundles. |
| MTP autodetect | Only real tensor evidence plus usable tuning may enable MTP; model names and stale metadata are insufficient. Qwen auto-depth must come from bundle-local `vmlx_mtp_tuning.json`, not profile/name rules. | Non-Kimi MTP census and Qwen MTP settings docs; CRACK rows explicitly stay MTP off. Focused tests cover tuned D2, validated D3, missing tuning, blocked tuning rows, valid tuning without MTP tensors, `MTPBundleStatus.snapshot`, missing-tuning evidence, and LLM/VLM factory wiring into `ModelConfiguration`. | Correct fail-closed policy covered; full MTP speed target remains separate/open. |

## Production-Quality Checklist Still Required

Before the Osaurus switch PR can honestly say `vmlx-swift` replaces the split
libraries, every exposed model family needs:

1. Config/template row with resolved `generation_config.json` defaults.
2. Multi-turn live row with visible coherent output, normal stop, no loops, no
   gibberish, no hidden reasoning-only fake pass.
3. Reasoning on/off/effort row where the family supports reasoning.
4. Tool parser row where the family supports tools.
5. Cache-on/cache-off row with topology-specific stats:
   prefix/paged/L2 disk/TurboQuant/SSM/CCA/SWA/media salt as applicable.
6. Continuous batching row for single-slot and B=2 behavior.
7. VL/video/audio row with real media payloads where the family supports media.
8. Speed and RAM row, including tok/s and physical-footprint caveat.
9. Failure rows left visible with artifact paths; no sampler guard, fake EOS,
   fake `</think>`, name-based MTP, or forced repetition penalty.

The active ledger files for those rows remain:

```text
docs/VMLX_SWIFT_MODEL_CAPABILITY_LEDGER.md
docs/VMLX_ACTIVE_MODEL_PRODUCTION_SCOPE_2026_05_17.md
docs/VMLX_OSAURUS_PR_PIN_LINEAGE_2026_05_17.md
```
