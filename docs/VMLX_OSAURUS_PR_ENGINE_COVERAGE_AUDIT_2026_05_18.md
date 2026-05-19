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
- Laguna current-head refresh added
  `20260518T_current_laguna_xs_turnmatrix/` and
  `20260518T_current_laguna_xs_turnmatrix_after_compile_gate/`. The first
  artifact exposed a real optional compiled-decode parity failure: compile-on
  3-turn chat looped while compile-off stayed coherent. The fix keeps Laguna
  off compiled decode in both `TokenIterator` and `BatchEngine` until parity is
  proven; the post-fix artifact passes release rows with bundle defaults
  (`temp=0.700 topP=0.900 topK=0 rep=nil`), about 33 tok/s production decode,
  disk L2 `hits=1,misses=23,stores=21`, and no sampler/repetition/reasoning
  guard.
- MiniMax JANG_K evidence was refreshed under
  `20260518T_minimax_m27_jangk_crack_turnmatrix_after_quant_diag_fix/`.
  The row upgrades the large JANG_K CRACK bundle from infer-only to
  production-shaped chat/cache proof: cache OFF/ON, BatchEngine chat, disk
  restore, B=2, per-slot sampler, and TurboQuant-KV B=2 pass with bundle
  defaults and MTP off. The JANG loader diagnostic now distinguishes expected
  explicit mixed-bit per-layer metadata from real config drift, so this bundle
  no longer reports a false `config-metadata mismatch patched in-memory`.
  Focused current proof
  `20260518T_minimax_jangk_growing_chat_after_harness_fix/` re-runs the
  chat-template growing-cache row with bundle defaults, MTP off, disk hit
  `47/83`, and stop-bounded `vmlx-cache-green` recall. The live matrix now
  marks the raw token-prefix `batch_cache_hit` row as N-A for MiniMax because
  it is a structural raw-prompt diagnostic, not production chat proof.
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
- Fresh Qwen MTP census refresh was run from this checkout under
  `docs/internal/live-gates/20260518T_qwen_mtp_census_refresh/`. The four MXFP
  MTP bundles all require real `mtp.*` tensor evidence plus bundle-local
  `vmlx_mtp_tuning.json` before `canAutoLaunchMTP=true`, and all report VL
  tensor evidence. Current tuning rows are: 27B MXFP4 `best_depth=2` at
  45.712 tok/s, 27B MXFP8 `best_depth=3` at 28.936 tok/s, 35B MXFP4
  `best_depth=3` at 131.187 tok/s, and 35B MXFP8 `best_depth=3` at
  101.605 tok/s. This is a metadata/tuning census, not fresh decode proof; the
  27B MXFP8 speed remains below the desired 35 tok/s class and needs runtime
  optimization rather than a fake activation guard.
- Fresh live process rows after explicit user approval to load models:
  `20260518T_dsv4_fresh_no_fake_rep_coherence/` passes DSV4 JANGTQ-K chat,
  reasoning off/on/max, and 5.5k-token semantic recall with
  `BENCH_DSV4_REPETITION_PENALTY=1.0` and
  `BENCH_DSV4_MAX_REPETITION_PENALTY=1.0`, so the row does not rely on a hidden
  repetition guard. `20260518T_dsv4_fresh_growing_chat_cache/` proves DSV4
  post-answer disk restore: turn 2 hits disk at `25/43`, prompt time drops
  `4.329s -> 0.215s`, and generic paged counters remain zero because
  `pagedIncompatible=true` is the correct DSV4 topology.
- Add the Python-side final-renderer DSV4 checklist to the Osaurus switch gate:
  the server settings renderer must surface native DSV4 cache copy / SWA+CSA+HSA
  topology, keep the paged block-size control fixed/disabled for DSV4 with the
  expected 256 display row when active metadata reports it, disable generic KV
  q4/q8 and JIT controls, show DSV4 pool quant state, show generation defaults
  from bundle metadata, and ensure CLI preview omits invalid flags
  `--kv-cache-quantization`, `--enable-jit`, `--is-mllm`, and
  `--speculative-model`. This is a UI/settings mapping gate, not an engine
  sampler guard.
- Fresh Ling, MiniMax, and Hy3 rows were re-run from this checkout:
  `20260518T_ling_jangtq2_fresh_prod_cache/` passes 7/7 with bundle defaults
  (`temp=0.600 topP=1.000 topK=0 rep=nil`) and disk/SSM stats
  `disk{hits=1,misses=23,stores=21}` plus `ssm{hits=1}`.
  `20260518T_minimax_small_jangtq_fresh_prod_cache/` passes 7/7 with
  MiniMax reasoning ON routed to `.reasoning`, reasoning OFF producing zero
  reasoning deltas, about 48-50 tok/s, and a paged hit; the bundle still logs
  shape-inferred mixed JANG metadata repair and that remains visible.
  `20260518T_hy3_jangtq_fresh_prod_cache/` passes 7/7 with bundle defaults
  (`temp=0.900 topP=1.000 topK=-1 rep=nil`), but cold S1 TTFT is about
  61.8s before the cache hit drops to 183ms, so performance remains a watch item.
- Fresh ZAYA text strict rerun is an honest regression/capability boundary, not
  a sampler issue. `20260518T_zaya_jangtq4_fresh_prod_cache/`,
  `20260518T_zaya_jangtq4_seed0_compare/`, and the paired greedy row under
  `20260518T_zaya_jangtq4_math_rootcause/` all fail the current stricter
  reasoning-off math row: direct-mode first token is `2` for `2 + 2` even with
  `temp=0 topP=1 topK=0 rep=nil`. Top-k probes show both JANGTQ4 and MXFP4 rank
  token `2` above token `4` on the same rendered thinking-off prompt, so this is
  a ZAYA direct-mode/template boundary rather than cache contamination or a
  JANGTQ-only decode bug. The same no-hidden-guard artifact shows thinking-on
  story generation remains inside reasoning at 256 and 768 tokens. Do not
  promote ZAYA text as fully reasoning/direct-mode production-clear from stale
  blue-sky rows, and do not hide this with repetition, temperature, forced
  thinking closure, or top-k policy.
- Osaurus PR #1147 now carries a dedicated live UI/API matrix document,
  `docs/VMLX_SWIFT_OSAURUS_LIVE_MATRIX_2026_05_18.md`, and policy tests that
  require it to cover the remaining switch gates. The matrix explicitly keeps
  Qwen-VL, Gemma VLM/Gemma3n, ZAYA-VL, Nemotron Omni audio/video, DSV4,
  MiniMax, Ling, Hy3, MTP tuning, saved reasoning settings, generation
  defaults, tool/reasoning parser leakage, `/v1/chat/completions`,
  `/v1/responses`, `/v1/messages`, Ollama routes, prefix/paged/block-L2/SSM
  cache stats, media salt, TTFT, tok/s, and memory proof as required live
  Osaurus app/API rows. This is intentionally not a green production claim:
  several rows are still pending live Osaurus UI/API evidence even where vmlx
  has source tests or direct `RunBench` artifacts.
- Fresh Gemma3n E2B production-path probe was run from this checkout under
  `docs/internal/live-gates/20260518T_gemma3n_e2b_prod/`. The harness prompt
  for the UTF-8 row was tightened to require exact literal `café` and `你好`
  strings instead of relying on an ambiguous "word" instruction. The model is
  coherent and fast on the math/reasoning-on/off/cache rows (about 120 tok/s,
  ~2.7 GiB RSS, no reasoning marker leakage, and disk L2 hits/stores when the
  coordinator is enabled), but it is not production-clear: the UTF literal row
  fails at bundle defaults and under greedy diagnostics by translating or
  drifting into unrelated Chinese text. Keep this as a real Gemma3n live red
  row until tokenizer/template/decode behavior is root-caused; do not hide it
  with sampler clamps or app-side output repair.
- Fresh DSV4 DSML tool-call refresh was run from this checkout under
  `docs/internal/live-gates/20260518T_dsv4_dsml_toolcall_refresh/`. The row
  loads `DeepseekV4JANGTQModel`, reports `Tool format: dsml` and
  `Reasoning stamp: think_xml`, emits one structured
  `get_weather({"location":"Tokyo"})` event through `BatchEngine.generate`,
  stops normally after 43 generated tokens, and leaks no raw DSML or thinking
  markers into `.chunk`. This is parser/template evidence only; DSV4 remains
  partial until long-context/vector/API/speed/low-footprint rows close.

2026-05-18 13:48 PDT Osaurus PR #1147 matrix expansion:

- Osaurus PR #1147 head `67a24031` adds a per-family UI/API execution matrix to
  `docs/VMLX_SWIFT_OSAURUS_LIVE_MATRIX_2026_05_18.md`. The matrix now names
  exact real-user rows for DSV4, Qwen VL / Qwen3.6 MTP VL, Gemma4/Gemma VLM,
  ZAYA/ZAYA-VL, Nemotron Omni / Parakeet / RADIO, MiniMax, Ling/Hy3 hybrid SSM,
  and GLM/GPT-OSS/Mistral parser families.
- The added rows make the remaining switch proof explicit: chat UI defaults,
  server settings visuals, `/v1/chat/completions`, `/v1/responses`,
  `/v1/messages`, Ollama routes, saved-setting isolation, native `top_k` and
  `chat_template_kwargs`, tool/coding context injection, media salt, video
  frame rows, Parakeet/RADIO facts, ZayaCCACache/path-dependent media state,
  prefix/paged/L2/SSM/cache inverses, physical footprint, TTFT, token/s, and no
  parser/tag leakage.
- The same commit adds a "Settings Carryover and Cache-Key Failure Modes"
  section requiring explicit inverse rows for Qwen thinking carryover, DSV4
  `max` carryover, VLM-to-text media carryover, cache OFF/ON restoration,
  tool/coding context carryover, and generation defaults from
  `generation_config.json` / `jang_config.json`. This is still a live-test
  checklist, not a production-clear claim.
- Osaurus focused source policy verification after the change:
  `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer swift test
  --package-path Packages/OsaurusCore --filter RuntimePolicySourceTests --jobs 2`
  passes 28/28.

2026-05-18 13:56 PDT Osaurus PR #1147 completion audit checkpoint:

- Osaurus PR #1147 head `cb24ea29` adds
  `docs/internal/live-gates/20260518T_pr1147_completion_audit.md`. The audit
  restates the switch objective as concrete deliverables and maps Qwen VL /
  Qwen3.6 MTP VL, Gemma VL, ZAYA-VL, Nemotron Omni, DSV4, MiniMax, Ling/Hy3,
  generation defaults, saved-setting carryover, and old-library removal to
  required artifact folders.
- The audit includes a local `/Users/eric/models` census and records that
  Qwen3.6 MTP activation must come from real `vmlx_mtp_tuning.json` rows:
  27B MXFP4 selects D2, 27B JANG_4M / 27B MXFP8 / 35B MXFP4 / 35B MXFP8 select
  D3, and 35B JANG_2K is blocked and must not auto-enable.
- The audit's current outcome is explicitly `Not complete.` This engine repo
  should not treat Osaurus PR #1147 as production-clear until the named live
  app/API folders contain visible output, cache stats, TTFT, tok/s, RSS /
  physical footprint, media sequence proof, parser/tool proof, and carryover
  inverse proof.
- Osaurus focused source policy verification after the audit change:
  `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer swift test
  --package-path Packages/OsaurusCore --filter RuntimePolicySourceTests --jobs 2`
  passes 28/28.

2026-05-18 14:05 PDT Osaurus PR #1147 bundle census gate:

- Osaurus PR #1147 head `2705ab08` adds
  `scripts/pr1147_collect_bundle_census.py` and the generated artifact folder
  `docs/internal/live-gates/pr1147/bundle-census/`. The helper reads
  `config.json`, `generation_config.json`, `jang_config.json`,
  processor configs, `model.safetensors.index.json`, and
  `vmlx_mtp_tuning.json` without loading model weights.
- The generated census covers 29 local model rows across DSV4, Qwen3.6
  MTP/VL and CRACK, Gemma, ZAYA/ZAYA-VL, Nemotron Omni, MiniMax, Ling, and Hy3.
  It records generation defaults including native `top_k`, VLM processor/token
  evidence, safetensors MTP tensor counts, and the conservative MTP
  `auto_enable` reason.
- Important MTP results: Qwen3.6 27B JANG_4M has 31 MTP tensors and D3;
  27B MXFP4 has 23 MTP tensors and D2; 27B MXFP8 has 23 MTP tensors and D3;
  35B MXFP4 has 42 MTP tensors and D3; 35B MXFP8 has 31 MTP tensors and D3.
  Qwen3.6 35B JANG_2K has MTP tensor evidence but tuning blocks native MTP, and
  Qwen3.6 CRACK rows have no MTP tensor evidence, so they stay off.
- This is file-level proof only. It does not prove Osaurus UI defaults, HTTP
  route behavior, cache hits, media turn coherency, parser separation, or
  memory footprint.
- Verification for the checkpoint: `python3 -m py_compile
  scripts/pr1147_collect_bundle_census.py`, JSON validation of
  `bundle_census.json`, `git diff --check`, and focused
  `RuntimePolicySourceTests` 28/28.

2026-05-18 14:09 PDT Osaurus PR #1147 HTTP route probe scaffold:

- Osaurus PR #1147 head `902d810a` adds
  `scripts/pr1147_http_route_probe.py`. It captures `/health`, `/v1/models`,
  `/models`, `/tags`, `/mcp/health`, and `/admin/cache-stats` by default and
  can opt into generation rows for `/v1/chat/completions`, `/v1/responses`,
  `/v1/messages`, `/api/chat`, and `/api/generate` when a model is explicitly
  supplied with `--run-generation`.
- The helper writes request bodies, full response bodies, status/content-type,
  body byte counts, and excerpts to `http_route_probe.json` plus a route
  summary. It intentionally refuses generation rows without `--model`.
- This is still a scaffold unless run against a live Osaurus server and paired
  with cache stats, visible output review, parser/no-leak review, timing, and
  memory artifacts for the same model row.
- Verification for the checkpoint: `python3 -m py_compile
  scripts/pr1147_http_route_probe.py scripts/pr1147_collect_bundle_census.py`,
  `git diff --check`, and focused `RuntimePolicySourceTests` 28/28.

2026-05-18 14:14 PDT Osaurus PR #1147 Keychain-safe launch gate:

- Osaurus PR #1147 head `e3b047d7` adds
  `docs/internal/live-gates/20260518T_pr1147_keychain_safe_launch.md`.
  The live route probe attempt showed that direct binary launch with a fake
  `HOME` is invalid for app-level gates because Osaurus needs the real macOS
  Keychain context for the database encryption key.
- The PR now records that live app/API gates must use a Keychain-safe launch:
  normal LaunchServices with the real user keychain context, or an explicitly
  created/unlocked temporary test keychain with the original keychain search
  list restored afterward.
- Route probe artifacts from a broken fake-HOME launch must not count as
  startup, settings, cache, model, or UI proof.
- Verification for the checkpoint: focused `RuntimePolicySourceTests` 28/28
  after pinning the launch note.

2026-05-18 14:19 PDT Osaurus PR #1147 live execution manifest:

- Osaurus PR #1147 head `d8e2e233` adds
  `docs/internal/live-gates/20260518T_pr1147_live_user_api_execution_manifest.md`.
  The manifest is an execution checklist, not a pass report. It names the
  source anchors that live rows must exercise: `ModelRuntime.swift`,
  `MLXBatchAdapter.swift`, `GenerationEventMapper.swift`,
  `LocalGenerationDefaults.swift`, `ModelMediaCapabilities.swift`,
  `ModelFamilyNames.swift`, `HTTPHandler.swift`, and the single consolidated
  `vmlx-swift` SwiftPM graph.
- The manifest defines the exact artifact folder shape under
  `docs/internal/live-gates/pr1147/<model-slug>/`: UI model picker, chat
  settings, server settings/CLI preview, health snapshots, cache stats,
  process memory, chat UI turns, route outputs, tool/reasoning parser review,
  media sequence, carryover inverse, and row summary.
- It expands the remaining live rows for DSV4, Qwen VL/Qwen3.6 MTP VL,
  Gemma4/Gemma VLM, Gemma3n text, ZAYA/ZAYA-VL, Nemotron Omni/Parakeet/RADIO,
  MiniMax, Ling/Hy3 hybrid SSM, and other local parser families. Each row names
  required UI defaults, app turn sequence, API sequence, cache/memory proof,
  parser/no-leak checks, and inverse OFF/ON states.
- It explicitly keeps MTP activation gated by real `mtp.*` tensors plus
  validated `vmlx_mtp_tuning.json`, keeps sampler/default checks tied to
  `jang_config.json` and `generation_config.json` including native `top_k`, and
  requires saved-setting carryover proof for reasoning, DSV4 `max`, media,
  cache mode, and tool/coding context.
- Osaurus focused source policy verification after the manifest change:
  `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer swift test
  --package-path Packages/OsaurusCore --filter RuntimePolicySourceTests --jobs 2`
  passes 28/28.

2026-05-18 14:24 PDT Osaurus PR #1147 keychain-safe launch helper:

- Osaurus PR #1147 head `82c763eb` adds
  `scripts/pr1147_keychain_safe_app_launch.sh`. The helper launches the debug
  `.app` through LaunchServices, refuses fake `HOME`, sets `OSU_MODELS_DIR`
  through `launchctl`, and restores the previous launchctl environment after
  the LaunchServices request is accepted.
- Local dry-run proof: the helper prints the expected real-home launch plan for
  `build/XcodeDerivedData-codex-live-pr1147/Build/Products/Debug/osaurus.app`
  and exits 64 when forced through `HOME=/tmp/osaurus-fake-home`, preventing
  the Keychain error path from being used as live evidence.
- Verification for the checkpoint: `bash -n
  scripts/pr1147_keychain_safe_app_launch.sh`, real-home `--dry-run`,
  fake-home refusal check, `git diff --check`, and focused
  `RuntimePolicySourceTests` 28/28. Local `shellcheck` was not installed; the
  GitHub `shellcheck` job is the authoritative shellcheck signal for this row.

2026-05-18 14:34 PDT Osaurus PR #1147 cache-stats admin route fix:

- Osaurus PR #1147 head `857ecad3` adds a read-only `/admin/cache-stats`
  endpoint in `HTTPHandler` backed by `ModelRuntime.cachedModelSummaries()` and
  vmlx `CacheCoordinator.snapshotStats()` when models are loaded. The route
  returns a cold snapshot without loading any model.
- Live metadata probe before the fix:
  `docs/internal/live-gates/pr1147/http-route-probe-metadata-20260518T1425/`
  shows `/admin/cache-stats` returned HTTP 404 with body `Not Found`.
- The rebuilt debug app was launched through the keychain-safe LaunchServices
  helper with real `HOME` and `/Users/eric/models`, bound `127.0.0.1:4242`,
  and post-fix probe
  `docs/internal/live-gates/pr1147/http-route-probe-metadata-20260518Tpost-cache-stats/`
  shows `/admin/cache-stats` returning HTTP 200 with empty `models` and zero
  aggregate counters for prefix, paged, block-L2, and SSM companion fields.
- Verification for the checkpoint: `git diff --check`,
  `RuntimePolicySourceTests` 28/28, and
  `MCPHTTPHandlerTests/admin_cache_stats_returns_empty_snapshot_without_loading_model`
  1/1. This is metadata/admin-route proof only; it is not model-specific cache
  hit, L2 write, SSM rederive, coherency, speed, or memory proof.

2026-05-18 14:48 PDT Osaurus PR #1147 component edge-case matrix:

- Osaurus PR #1147 head `ad27a35f` adds
  `docs/internal/live-gates/20260518T_pr1147_component_edge_case_matrix.md`.
  The matrix breaks the remaining production-clear requirement into concrete
  source-to-artifact wiring for `ModelOptions`, `ModelMediaCapabilities`,
  `LocalGenerationDefaults`, `MLXBatchAdapter`, `GenerationEventMapper`,
  `ModelRuntime`, `HTTPHandler`, and the SwiftPM old-library sweep.
- The matrix names per-family live proof for DSV4, Qwen3.6 MTP/VL and
  non-MTP controls, Gemma4 VLM, Gemma3n text, ZAYA/ZAYA-VL, Nemotron Omni /
  Parakeet / RADIO, MiniMax, Ling/Hy3 hybrid SSM, and local GLM/GPT-OSS/Mistral
  parser families. Kimi remains intentionally excluded from this PR budget.
- The new gate explicitly requires UI clicked/default state proof, route parity
  across chat/responses/messages/Ollama where supported, real media sequences,
  saved-setting carryover inverses, native `top_k` and bundle defaults,
  coding/tool context injection checks, prefix/paged/block-L2/SSM/media cache
  hit proof, TTFT/tok/s/RSS/physical footprint, no parser leakage, no name-only
  MTP, and no hidden sampler/EOS/repetition/forced reasoning-close repair.
- Verification for the checkpoint: focused Osaurus
  `RuntimePolicySourceTests` passes 28/28 after adding source-policy coverage
  for the new matrix. This remains checklist/source-policy evidence, not live
  model production proof.

2026-05-18 14:44 PDT Osaurus PR #1147 route artifact helper hardening:

- Osaurus PR #1147 head `3fb88d40` updates
  `scripts/pr1147_http_route_probe.py` so generation route artifacts no longer
  overwrite stream and non-stream outputs for the same path. Each route row now
  carries a label and unique request/body filenames.
- In generation mode the helper captures before/after `/health`,
  `/admin/cache-stats`, and process-memory snapshots around each route, plus
  before/after the full generation set. This makes API artifacts diagnosable
  for route status, cache counter movement, and RSS context before a human
  reviews output tails and parser leakage.
- Verification for the checkpoint: `python3 -m py_compile
  scripts/pr1147_http_route_probe.py`, `--help`, `git diff --check`, and
  focused Osaurus `RuntimePolicySourceTests` 28/28. This remains harness
  readiness, not live model production proof.

2026-05-18 14:51 PDT Osaurus PR #1147 VLM live sequence probe:

- Osaurus PR #1147 head `e27e5b55` adds
  `scripts/pr1147_live_sequence_probe.py` and
  `scripts/tests/test_pr1147_live_sequence_probe.py`. The probe drives the
  VLM/Omni artifact shape the PR gate now requires: image+text, text-only,
  different-image, repeat-image, video, and audio turns when media inputs are
  supplied.
- The helper builds OpenAI Chat Completions media parts (`image_url`,
  `video_url`, `input_audio`) and Open Responses text/image input items,
  records raw request/response bodies, per-turn `/health` and
  `/admin/cache-stats` snapshots, process-memory JSON, and extracted output
  tails for later human review.
- Verification for the checkpoint: Python unit tests for media-boundary turn
  planning and chat/responses request shapes pass 2/2; `python3 -m py_compile`
  passes for the live sequence and route probes; focused Osaurus
  `RuntimePolicySourceTests` passes 28/28. This is still harness readiness and
  not a live model production-clear row.

2026-05-18 15:02 PDT Osaurus PR #1147 ZAYA-VL first live row:

- Osaurus PR #1147 head `0e965e85` hardens the live sequence probe and records
  the first real ZAYA-VL app/API artifact set under
  `docs/internal/live-gates/pr1147/zaya1-vl-8b-mxfp4/`.
- The pre-fix artifact
  `vlm-sequence-20260518T1458/` shows the probe helper itself was wrong for
  Responses follow-up history: it carried Chat Completions media shape into
  Responses, so later Responses rows returned HTTP 400.
- The helper now keeps route-native user history: Chat uses `text` /
  `image_url`, while Responses uses `input_text` / `input_image`. It also uses
  a macOS-compatible `ps` parser for process memory snapshots.
- The post-fix artifact
  `vlm-sequence-20260518T1504/` completed 10 Chat/Responses rows with red
  image, text-only follow-up, blue image, red repeat, and video. It is
  explicitly FAIL/PARTIAL, not a production pass. Chat grounds the first red
  image and text-only follow-up, but the blue-image turn still describes red;
  Responses returns generic "media" explanations instead of grounded image
  answers; ZAYA1-VL video returns HTTP 500 `ZAYA1-VL video input is not
  implemented`; and the run did not collect cache-hit proof under a residency
  mode that keeps the loaded model visible after non-streaming requests.
- Osaurus PR #1147 head `ee8be5f6` clarifies the cache-stat nuance: source
  trace shows `ServerConfiguration.default.modelIdleResidencyPolicy` is
  `.immediately`, and `ModelRuntime.generateEventStream` schedules idle unload
  when the generation lease releases. Empty post-request `/health.loaded` and
  `/admin/cache-stats.models` can therefore be a settings/probe limitation,
  not proof that cache did or did not hit. Real cache rows must set
  non-immediate residency through the app/settings path or capture stream-time
  snapshots before the generation lease releases.
- Verification for the checkpoint: `python3 -m unittest
  scripts/tests/test_pr1147_live_sequence_probe.py` passes 4/4,
  `python3 -m py_compile scripts/pr1147_live_sequence_probe.py
  scripts/pr1147_http_route_probe.py` passes, `git diff --check` passes, and
  focused Osaurus `RuntimePolicySourceTests` passes 28/28.
- Engine consequence: this confirms the live Osaurus UI/API gate is correctly
  exposing real failures now. `vmlx-swift` should not treat ZAYA-VL as
  production-clear until media switch/cache-state, Responses media handling,
  unsupported-video capability reporting, and cache proof under the correct
  residency conditions are root-caused and fixed without sampler, prompt, or
  output guards.

2026-05-18 15:14 PDT Osaurus PR #1147 Responses media fix:

- Osaurus PR #1147 head `287f5f42` fixes one source-side cause from the ZAYA-VL
  red artifact: `/v1/responses` converted input messages with
  `OpenResponsesMessageContent.plainText`, which keeps `input_text` but drops
  `input_image` before the request reaches `MLXBatchAdapter`.
- The PR now converts Responses `input_text` and `input_image` parts into
  Chat Completions `MessageContentPart` values, so `ChatMessage.imageUrls`
  carries the image URL into the same multimodal mapping path used by
  `/v1/chat/completions`.
- Regression test:
  `ChatEngineTests.openResponsesRequest_preservesInputImageIntoChatRequest`
  decodes a Responses payload with `input_text` plus `input_image` and proves
  the converted chat request preserves both plain text and `imageUrls`.
- Verification for the checkpoint: the focused regression test passes 1/1,
  `ChatEngineTests` passes 12/12, `MultimodalContentPartTests` passes 13/13
  with the two existing MLXArray fixture skips, `RuntimePolicySourceTests`
  passes 28/28, and `git diff --check` passes.
- Boundary: this fixes the source route-conversion bug only. The ZAYA-VL row is
  still not production-clear until a fresh keychain-safe live rerun shows
  grounded Responses image output, resolves the Chat blue-image stale-red
  answer, handles or hides unsupported ZAYA video, and collects cache proof
  under non-immediate residency or stream-time snapshots.

2026-05-18 15:35 PDT Osaurus PR #1147 ZAYA-VL image-only capability fix:

- Osaurus PR #1147 head `c70a5dcb` marks ZAYA1-VL as `.imageOnly` from
  `ModelMediaCapabilities.from(modelId:)` and keeps `model_type=zaya1_vl`
  directory detection image-only. This matches the current vmlx-swift boundary:
  ZAYA1-VL image/text is implemented, but video still throws
  `ZAYA1-VL video input is not implemented` until a real ZAYA video processor
  exists.
- Focused Osaurus tests now pin the behavior across picker, bundle directory
  detection, drag-drop gating, end-to-end composer accept sets, and MC/DC
  capability coverage. ZAYA1-VL image is accepted; video and audio are rejected.
- Verification for the checkpoint: `CapabilityFromModelIdTests` 5/5,
  `CapabilityFromDirectoryTests` 5/5, `DragDropAcceptMatrixTests` 6/6,
  `EndToEndComposerAcceptSetTests` 1/1 with 33 cases,
  `ModelMediaCapabilitiesMCDCTests` 32/32, `RuntimePolicySourceTests` 28/28,
  and `git diff --check`.
- Boundary: this prevents Osaurus UI/composer from advertising fake ZAYA video.
  It does not make the live ZAYA-VL different-image grounding row pass, does
  not prove Responses image grounding after the route-conversion fix, and does
  not prove cache hits until a keychain-safe rerun uses non-immediate residency
  or stream-time cache snapshots.

2026-05-18 15:44 PDT Osaurus PR #1147 forced-behavior audit gate:

- Osaurus PR #1147 head `1bc461cc` adds a dedicated forced-behavior audit gate
  to the live matrix, component edge-case matrix, completion audit, and source
  policy tests. The gate requires source/live search for forced sampler
  defaults, forced repetition penalties, reasoning rail rewrites, forced
  `</think>` close tokens, token/logit shaping, and parser output repair.
- For every hit, the artifact must record why the behavior was originally
  added, prove whether it still fires, and replace it with a real template,
  decode, tokenizer, cache, or model-family fix. The only allowed output
  defaults are bundle metadata (`generation_config.json` / `jang_config.json`)
  or explicit user/API kwargs.
- Verification for the checkpoint: focused Osaurus `RuntimePolicySourceTests`
  passes 28/28 and `git diff --check` passes.

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

2026-05-18 03:22 PDT Nemotron Omni consolidation refresh:

- This pass explains why the package switch is taking longer than a mechanical
  package merge: Osaurus currently gets correctness from the interaction of
  pinned packages, app/session policy, Python-side processors, and runtime
  glue. Moving that into one Swift package makes each hidden boundary an engine
  contract. The current live example is Nemotron Omni media: the source Python
  path uses CLIP/RADIO image preprocessing with bicubic interpolation,
  `align_corners=false`, and antialiasing, then applies video EVS after the
  full placeholder run has been spliced into `inputs_embeds` and `input_ids`.
- Focused Swift tests now cover source-style video frame labels, aspect-
  preserving video target sizing, compact no-thinking media tail,
  source-compatible EVS keep indices, pre-encoded Parakeet embedding reuse, and
  post-prepare media cache keys. The focused commands pass:
  `MediaCachePlaceholderTests` 3/3 and `NemotronHOmniPreEncodedAudioTests`
  13/13:
  `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcrun swift test
  --filter <test-suite> --jobs 2`.
- Release `RunBench` rebuilt after the source-style media changes. Live JANGTQ
  artifact
  `docs/local/live-model-matrix/20260518T_omni_bicubic_antialias_jangtq/`
  exits 0 and reports `20 passed, 0 failed | load 1.37s`. The live sampling
  line is sourced from the bundle generation config:
  `temp=0.600 topP=0.950 topK=0 minP=0.000 rep=1.000 seed=0`.
- The 20/20 live result is a structural/runtime pass, not a blanket semantic
  production sign-off. Image reasoning-on grounds the orange-to-light-blue
  gradient; no-thinking image/video outputs are still weakly grounded in several
  rows, and short budgets can leave visible content truncated. These are live
  model/runtime quality caveats that still need a longer media-quality gate.
- Cache-boundary fix: `TokenIterator`, `BatchEngine`, and `NativeMTPTokenIterator`
  now honor `LMOutput.effectivePromptTokens` for post-EVS cache storage and
  logit-processor prompt state. Inputs whose cache key is only known after
  model prepare skip pre-prepare cache fetch instead of restoring under the
  pre-pruned token stream. History-boundary aliases are also skipped when their
  saved counts are in the pre-pruned coordinate space. This is correctness
  routing, not a sampler/parser workaround.
- Remaining cache gap at this point in the audit was video EVS pre-prepare
  effective-key resolution. See the 06:35 PDT update below: the in-memory
  alias contract is now implemented and focused-tested, but a live repeated
  video cache-hit row is still required before calling full video cache
  production-clear.

2026-05-18 04:25 PDT Nemotron Omni live-behavior correction:

- A later live reread showed the earlier 20/20 Omni matrix was too permissive:
  the no-thinking image row had been repaired, but the harness still counted
  reasoning-only/max-token output as acceptable in one chat-history path. The
  bench now treats a chat-history turn that hits `max_tokens` before a normal
  stop as FAIL and does not fall back to reasoning text as visible content.
- Processor fix: media placeholders are now attached to the chat message that
  actually carried media instead of always being prepended to the last user
  message. This matters for VL multi-turn history where turn 1 has an image
  and turn 2 is text-only. The change is prompt construction only; it is not a
  sampler, EOS, repetition, or forced reasoning-close guard.
- Bench fix: Omni text/image multi-turn rows now render real chat history
  instead of reusing a populated raw KV/Mamba cache with unrelated fresh
  prompts. That old harness behavior was a cache-contract violation and could
  induce loops. It was removed rather than hidden with generation parameters.
- Live artifacts:
  - `docs/local/live-model-matrix/20260518T110730Z_omni_nothink_instruction_jangtq/`
    proved the source-style no-thinking media instruction changes the JANGTQ
    image no-thinking answer from the earlier monochrome/curved-line failure
    to a grounded gradient answer, but still failed image multi-turn under the
    old raw-cache harness.
  - `docs/local/live-model-matrix/20260518T112412Z_omni_strict_chat_nofallback_256_jangtq/`
    is the honest red row: strict chat-history VL reasoning-on at 256 tokens
    fails because turn 2 stays inside reasoning and hits max tokens with no
    visible answer.
  - `docs/local/live-model-matrix/20260518T112503Z_omni_strict_chat_nofallback_512_jangtq/`
    is the strict green direct path: 15/15 pass, row 4 closes `</think>` at
    493 generated tokens and visible content is `Orange and blue`.
  - `docs/local/live-model-matrix/20260518T112555Z_omni_batch_512_jangtq/`
    is the BatchEngine path: 19/19 pass, including B1 text B=1, B2 text B=2,
    B3 image B=1, and B4 audio B=1. B3/B4 throughput is low
    (`11.2`/`11.5 tok/s`) and needs performance work, but coherence and routing
    are now live-proven for the tested JANGTQ Omni bundle.
- Verification after the patch:
  `swift build -c release --product RunBench --jobs 2` passes. The focused
  unit suite also passes when invoked through the Xcode toolchain:
  `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcrun swift test
  --filter NemotronHOmniPreEncodedAudioTests --jobs 2` runs 14/14 tests green.
  The system `swift test` path remains misleading here because it compiles an
  unrelated debug target that imports `Testing` before test selection.

2026-05-18 06:20 PDT Nemotron Omni strict media refresh:

- Root cause update: the remaining JANGTQ Omni image failures were not sampler,
  EOS, repetition, or cache issues. The open-thinking media template and
  assistant-only media tail hallucinated over placeholder/prompt text, while the
  same image tensor path grounded correctly with the bundle's closed-thinking
  direct-answer media contract. Text-only reasoning remains live and tested.
- Processor contract is now explicit: Nemotron Omni media turns render the
  direct-answer media template (`enable_thinking=false` for template rendering)
  and carry the source-style direct-answer instruction. This is a model-family
  media capability boundary, not a hidden sampling guard; Osaurus should not
  expose media reasoning for this Omni path until a real grounded media-thinking
  row exists.
- The strict image validator no longer accepts fake text-only image answers or
  negated gradient claims. It now rejects known hallucinations such as white
  background/text-prompt descriptions and requires the warm/blue synthetic
  fixture to be grounded.
- Focused verification:
  `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcrun swift test
  --filter NemotronHOmniPreEncodedAudioTests --jobs 2` passes 16/16.
  `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcrun swift build
  -c release --product RunBench --jobs 2` passes.
- Live artifact:
  `docs/local/live-model-matrix/20260518T_omni_jangtq_media_direct_contract_prompt_postfix/omni_jangtq.log`
  passes 19/19 using bundle defaults
  (`temp=0.600 topP=0.950 topK=0 minP=0.000 rep=1.000 seed=20260517`).
 Covered rows include text single-turn, text multi-turn, image single-turn,
  image reasoning-off direct, image multi-turn, video encoder, Parakeet audio
  encoder, video LMInput, audio LMInput, text reasoning OFF, text ON/OFF/ON
  reasoning toggle, mixed image+audio, media-salt isolation, hybrid SSM
  warm-pass parity, and BatchEngine text/image/audio rows.

2026-05-18 06:35 PDT post-prepare media cache alias refresh:

- Root cause: post-EVS media prompts were stored under
  `LMOutput.effectivePromptTokens`, but future requests only have the raw
  pre-EVS token stream before `prepare`. The previous safe behavior skipped
  pre-prepare fetch forever. That avoided false hits but also prevented
  repeated video prompts from using the existing prefix/paged/L2 cache entry.
- Fix: `CacheCoordinator` now records and resolves media-salted raw-to-effective
  prompt aliases. Exact repeats resolve to the stored effective token sequence;
  growing prompts use the longest recorded raw prefix and append the raw suffix.
  The alias refuses nil or mismatched media/cache salt, so same-text/different-
  media, different reasoning scope, and different KV policy still miss.
- Wiring: `BatchEngine`, `TokenIterator`, and `NativeMTPTokenIterator` now
  resolve the alias before fetch and record it when `prepare` returns effective
  prompt tokens. This is cache-key routing only; it does not change sampler,
  stop-token, repetition, or reasoning behavior.
- Focused verification:
  `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcrun swift test
  --filter 'MediaCachePlaceholderTests|CacheCoordinatorTopologyFocusedTests'
  --jobs 2` passes 31 tests across 6 suites. The new behavior test proves
  exact and growing raw prompts resolve to effective tokens, different media
  salts miss, nil salt misses, and the resolved effective prefix fetches the
  stored paged cache entry.
- Full focused policy gate:
  `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcrun swift test
  --filter MLXLMCommonFocusedTests --jobs 2` passes 242 Swift Testing tests
  across 28 suites plus 22 selected XCTest focused rows.

2026-05-18 06:48 PDT live repeated-video alias proof:

- Bench coverage: `OmniBench` now has env-gated row
  `BENCH_OMNI_VIDEO_CACHE_ALIAS=1`, `5d. video repeated cache alias`. The row
  prepares the same video twice through the real `NemotronHOmniProcessor`, uses
  `MLXLMCommon.generate(..., cacheCoordinator:)`, proves the first generation
  recorded a media-salted raw-to-effective alias, probes the resolved effective
  key, and then runs the replay through the same production iterator path.
- Live strict artifact:
  `docs/local/live-model-matrix/20260518T134746Z_omni_jangtq_strict_192_video_cache_alias/omni_jangtq_strict_192_video_cache_alias.log`
  exits 0 and reports `20 passed, 0 failed | load 1.37s` on
  `/Users/eric/models/dealign.ai/Nemotron-Omni-Nano-JANGTQ-CRACK` with bundle
  defaults plus fixed seed (`temp=0.600 topP=0.950 topK=0 minP=0.000
  rep=1.000 seed=20260517`).
- The new row reports `raw=4028`, `effective=1382`, media salt
  `473907c829cc`, direct probe `disk matched=1382/1382 remaining=0`, replay
  hit counter `1->2`, and coherent visible video output on both cold and
  replay runs. This clears the live repeated-video cache-hit gap for the
  tested JANGTQ Omni bundle.

2026-05-18 07:40 PDT rebuilt strict Omni confirmation:

- Release `RunBench` was rebuilt after the current source timestamp:
  `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcrun swift build
  -c release --product RunBench --jobs 2` completed successfully.
- Fresh artifact:
  `docs/local/live-model-matrix/20260518T_current_omni_jangtq_strict_after_rebuild/omni_jangtq_strict.log`
  exits 0 and reports `20 passed, 0 failed | load 1.92s` on
  `/Users/eric/models/dealign.ai/Nemotron-Omni-Nano-JANGTQ-CRACK` with
  `BENCH_OMNI=1`, `BENCH_OMNI_BATCH=1`,
  `BENCH_OMNI_VIDEO_CACHE_ALIAS=1`, `BENCH_MAX_TOKENS=192`, and
  `BENCH_OMNI_RANDOM_SEED=20260517`.
- The live sampling line again comes from `generation_config.json`:
  `temp=0.600 topP=0.950 topK=0 minP=0.000 rep=1.000`; no sampler,
  repetition, EOS, or forced reasoning-close guard was added.
- The repeated-video row again reports `raw=4028`, `effective=1382`,
  `disk matched=1382/1382 remaining=0`, and replay hits `1->2`.

2026-05-18 08:02 PDT Gemma 4 E4B gap closure:

- The previously unlisted Osaurus-local E4B bundle
  `/Users/eric/osaurus_models/finished/gemma-4-e4b-it-4bit` now has current
  text/cache, reasoning, and image-cache proof under
  `docs/local/live-model-matrix/20260518T_current_gemma4_e4b_prod_text_cache/`.
- `prod_default_cache.log` passes 7/7 through the production BatchEngine row
  with real bundle defaults
  `temp=1.000 topP=0.950 topK=64 minP=0.000 rep=nil`, Harmony parser, S2 TTFT
  `73ms -> 29ms`, about `118-129 tok/s`, peak RSS `4727 MiB`, and L2 stats
  `disk{hits=1,misses=17,stores=14,maxBytes=4294967296}`.
- `reasoning_turn_matrix.log` passes one loaded transcript with reasoning
  OFF/ON/OFF/ON plus efforts `low,medium,high,max`. The ON recall turn routes
  `527` reasoning chars, all turns stop normally, and cache stats report
  `disk{hits=1,misses=24,stores=16,...}`.
- `vl_chat_cache.log` proves structured chat image cache behavior: same image
  HITs `disk 301/301` and drops TTFT `168ms -> 26ms`, a different image MISSes,
  and the follow-up remains grounded in `red, white, and blue`.
- Boundary: this is Gemma4 image/text proof only. Audio/video-specific Gemma4
  plugin behavior remains separate until a dedicated media lane is run.

2026-05-18 11:25 PDT fresh process rows after explicit model-load approval:

- Release `RunBench` was rebuilt on the current checkout with
  `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcrun swift build
  -c release --product RunBench --jobs 2`. The build completed; warnings were
  pre-existing bench cleanup warnings and not release blockers for these rows.
- Fresh DSV4 DSML row:
  `docs/local/live-model-matrix/20260518T_fresh_user_allowed_process_rows/dsv4_dsml_toolcall_fresh.log`
  loads `DeepSeek-V4-Flash-JANGTQ-K`, reports `Tool format: dsml`,
  `Reasoning stamp: think_xml`, emits one structured
  `get_weather({"location":"Tokyo"})` tool call, stops normally after 43
  generated tokens, and leaks no raw DSML or reasoning markers into `.chunk`.
- Fresh Gemma3n E2B text/cache row:
  `docs/local/live-model-matrix/20260518T_fresh_user_allowed_process_rows/gemma3n_e2b_prod_default_cache_fresh.log`
  passes 7/7 with bundle defaults from `generation_config.json`
  (`temp=0.600 topP=0.950 topK=64 minP=0.000 rep=nil`), no reasoning rail,
  about 122 tok/s, peak RSS 2772 MiB, and L2 disk stats
  `hits=1,misses=21,stores=21`. The row remains text-only; Gemma3n image/audio
  towers are still intentionally not claimed.
- Fresh Gemma4 E2B bundle-default row:
  `docs/local/live-model-matrix/20260518T_fresh_user_allowed_process_rows/gemma4_e2b_prod_default_cache_fresh.log`
  is an honest red artifact at 6/7. It does not loop or crash, and cache stats
  record `disk{hits=1,misses=17,stores=14}`, but at bundle defaults
  (`temp=1.000 topP=0.950 topK=64 rep=nil`) the UTF-8 prompt produced coherent
  Chinese text containing `你好` and a Chinese cafe word while omitting the
  literal token `cafe`/`café` requested by the harness.
- Fresh Gemma4 E2B explicit-greedy row:
  `docs/local/live-model-matrix/20260518T_fresh_user_allowed_process_rows/gemma4_e2b_prod_greedy_cache_fresh.log`
  passes 7/7 with no repetition penalty (`temp=0.000 topP=1.000 topK=0
  rep=nil`), about 184-205 tok/s, peak RSS 3173 MiB, and the same L2 disk
  topology (`hits=1,misses=17,stores=14`). This is a runtime/cache coherence
  proof and a validator/default-sampling caveat, not permission to add a hidden
  sampler guard in Osaurus or vMLX.

2026-05-18 07:00 PDT clean consumer package repair:

- Osaurus PR #1147 pin testing exposed a clean-resolve package failure at
  `vmlx-swift@d2c6356`: `Package.swift` referenced `Libraries/vMLXFluxVideo`
  and sibling Flux targets that were present only as untracked concurrent-agent
  work in this checkout. A clean Osaurus consumer therefore failed before
  exercising the consolidated inference package.
- Fix: remove the untracked Flux products, targets, probe, tests, and umbrella
  re-export from the production manifest until that work lands as tracked
  source. This does not touch or delete the other agent's local Flux files; it
  restores clean package resolution for the shipped inference engine.
- Verification: `swift package describe --type json` succeeds after the
  manifest repair. The Osaurus switch PR pin can move to the next pushed
  `vmlx-swift` revision instead of inheriting the missing-target failure.

2026-05-18 04:45 PDT build/coverage/live refresh:

- Consolidated package build gate: `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
  xcrun swift build --target VMLX --jobs 2` passes. Artifact log:
  `docs/local/build-logs/vmlx_target_build_20260518T113748Z.log`; that saved
  umbrella-target build has no `warning:` or `error:` lines after explicitly
  excluding documentation/template reference files from SwiftPM source targets.
- Broad test-target compile gate: `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
  xcrun swift build --build-tests --jobs 2` passes.
- Full active focused runtime-policy gate:
  `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcrun swift test
  --filter MLXLMCommonFocusedTests --jobs 2` passes. Current count after the
  post-prepare media cache alias refresh: 242 Swift Testing tests across 28 suites plus
  22 selected XCTest focused rows.
- Source-coverage fix: `BatchEngineGrowingChatCacheSourceTests` was still in
  inactive `Tests/MLXLMTests` and filtered runs executed zero tests. The guard
  is now moved into active `MLXLMCommonFocusedTests`; the exact filter
  `BatchEngineGrowingChatCacheSourceTests` runs 8/8 tests green and covers
  post-answer cache boundaries, disk restore materialization, full-hit trim,
  SSM/ZAYA/rotating-cache guardrails, and absence of hidden reasoning close
  forcing.
- DSV4 cleanup: `DeepseekV4.swift` now uses current `quantizedMM`, keeps the
  prefill local mask immutable, and `DeepseekV4MathHelpers.yarnInvFreq` no
  longer has unreachable code. Focused DSV4/cache source gate passes 29/29:
  `DeepseekV4IndexerCausalTopKTests|DeepseekV4ReasoningPolicyTests|
  DeepseekV4ChatTemplateFallbackFocusedTests|BatchEngineGrowingChatCacheSourceTests`.
- Current live Omni direct artifact:
  `docs/local/live-model-matrix/20260518T114355Z_omni_strict_chat_nofallback_512_jangtq_current/`
  exits 0 and reports `15 passed, 0 failed | load 1.46s`. The row includes
  image no-thinking direct (`A gradient background transitioning from orange to
  yellow.`), image multi-turn (`Orange and blue` on turn 2), audio media-salt
  isolation, reasoning ON/OFF/ON, video, mixed image+audio, and hybrid
  SSM warm-pass with `["KVCacheSimple", "MambaCache"]`.
- Current live Omni BatchEngine artifact:
  `docs/local/live-model-matrix/20260518T114440Z_omni_batch_512_jangtq_current/`
  exits 0 and reports `19 passed, 0 failed | load 1.38s`, including B1 text
  B=1, B2 concurrent text B=2, B3 image B=1, and B4 audio B=1. B3/B4 remain
  slower (`11.6`/`11.3 tok/s`) but they no longer fail coherence/routing in
  this current JANGTQ row.
- Release `RunBench` still builds (`swift build -c release --product RunBench
  --jobs 2`), but the executable build emits bench/source warnings in
  `RunBench`, `Source/MLX`, and `Source/MLXNN`. These are not live-row
  blockers, but they are still cleanup work before claiming warning-free
  package production polish.

2026-05-18 05:33 PDT Gemma3n E2B correction:

- The local Gemma3n E2B MLX bundle
  `/Users/eric/models/mlx-community/gemma-3n-E2B-it-4bit` exposed why this
  package switch is not a mechanical merge. The initial Swift text path failed
  at load because full VLM checkpoint keys arrived as `language_model.model.*`
  with `vision_tower.*` and `audio_tower.*` sidecars, while the text model
  expected canonical `language_model.*` keys. This is now handled by
  `Gemma3nTextModel.sanitize(weights:)`, which canonicalizes the text prefixes
  and drops non-text towers for the text-only runtime path.
- A second real runtime bug was prompt RoPE positioning: Gemma3n attention was
  applying query RoPE after cache update, so prompt queries could see the
  advanced cache offset. `Gemma3nAttention` now captures the rotary offset
  before the key/value update and applies that same captured offset to keys and
  queries. This is a cache-position fix, not a sampler or EOS workaround.
- A third runtime mismatch was Gemma3n conditional-generation embedding scale.
  Source `mlx_vlm` uses unscaled `inputs_embeds` for VLM-style prompt prefill
  but calls the language model directly for generated decode tokens, where
  token embeddings are scaled by `sqrt(hidden_size)`. Swift now detects the
  conditional-generation config and keeps prompt-prefill embeddings unscaled
  while restoring scale for single cached decode tokens. This fixed the
  token-2 drift that caused looping/word-puzzle behavior after the first token.
- The production harness also had weak validators. The old S3 validator
  returned PASS for `accepted non-blue`, and the S5 row claimed verbatim UTF-8
  success while accepting any non-empty text. Python reference on the same
  bundle also fails the old sky/planet/verbatim prompts, so the gate now uses
  reference-satisfiable prompts and validates actual expected content (`4`,
  `Mars`, and `café`/`你好`). The older `BENCH_OFFICIAL` path had the same
  non-blue and fake-verbatim weakness; it is tightened too. Focused tests guard
  against reintroducing those fake passes anywhere in `RunBench`.
- Verification:
  `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcrun swift test
  --filter Gemma3nTextSanitizeFocusedTests --jobs 2` passes 8/8, and
  `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcrun swift build
  -c release --product RunBench --jobs 2` passes.
- Live artifacts:
  - `docs/local/live-model-matrix/20260518T123300Z_gemma3n_e2b_prod_greedy_strict_promptfix_192/`
    passes 7/7 with explicit greedy/no repetition penalty, about `130 tok/s`,
    peak RSS about `2723 MiB`, and no reasoning leakage. S5 satisfies the
    UTF-8 inclusion predicate but length-stops at 192 tokens, so do not treat
    it as exact-verbatim proof.
  - `docs/local/live-model-matrix/20260518T123320Z_gemma3n_e2b_prod_bundle_defaults_strict_promptfix_192/`
    passes 7/7 using real bundle defaults
    `temp=0.600 topP=0.950 topK=64 minP=0.000 rep=nil`.
  - `docs/local/live-model-matrix/20260518T123340Z_gemma3n_e2b_prod_cachecoord_strict_promptfix_192/`
    passes 7/7 with L2 coordinator enabled. S2 TTFT drops from `61ms` to
    `24ms`; cache stats report `pagedIncompatible=true`,
    `disk{hits=1,misses=21,stores=21,maxBytes=4294967296}`, and no SSM
    rederive because this text path is non-hybrid.
- 2026-05-18 07:58 PDT current rerun:
  `docs/local/live-model-matrix/20260518T_current_gemma3n_e2b_prod_default_vs_greedy/`
  re-runs the same row after the latest `RunBench` rebuild. Fresh-cache
  `default_fresh_cache.log` passes 7/7 with real bundle defaults
  `temp=0.600 topP=0.950 topK=64 rep=nil`, about `123-125 tok/s`, peak RSS
  `2771 MiB`, S2 TTFT `65ms -> 23ms`, and L2 stats
  `disk{hits=1,misses=21,stores=21,maxBytes=4294967296}`. Fresh-cache
  `greedy_fresh_cache.log` also passes 7/7 with explicit
  `temp=0.000 topP=1.000 topK=0 rep=nil`, about `129-131 tok/s`, peak RSS
  `2753 MiB`, and L2 stats `disk{hits=1,misses=21,stores=20,...}`. Both rows
  wrote real `.safetensors` cache blocks under fresh `/tmp` cache roots.
- Boundary: this clears the current Gemma3n text-only loading/decode/cache row.
  It does not claim Gemma3n vision or audio towers are wired in Swift; those
  towers are intentionally dropped in the text sanitizer until the VLM/audio
  path has its own media processor and cache proof.

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
- Follow-up prefix-toggle hardening now makes
  `VMLXServerRuntimeSettings.cacheCoordinatorConfig(...)` honor
  `cache.prefix.enabled=false` by disabling both paged prompt reuse and
  block/legacy disk L2 in the concrete `CacheCoordinatorConfig`. This closes a
  real Osaurus panel edge case where stale paged/block toggles could keep prompt
  cache reuse active after the user turned Prefix Cache off. The fix leaves
  live KV codec selection alone, so it is not a hidden sampler, repetition, or
  TurboQuant quality workaround. Focused verification:
  `VMLXServerRuntimeSettingsTests|RuntimeMoETopKOverrideFocusedTests` passes
  25/25 with the Xcode framework path.
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
- Omni live voice: text, image, audio, video, mixed-media, media-salt, hybrid
  SSM, and BatchEngine rows pass in the latest strict JANGTQ matrices, with
  direct-answer media mode separated from text reasoning. The post-prepare
  video EVS alias contract is now implemented, focused-tested, and live-proven
  by a repeated-video cache-hit row. Repeated cache-on audio semantic
  quality/termination still remains partial.
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
| #1066 `pin DSV4 vmlx update` | merged | DSV4 tokenizer/cache/runtime pin, local tokenizer fallback. | DSV4 separator fix and template kwargs rows pass; DSV4 live cache OFF/ON chat is coherent. 2026-05-18 follow-up: the DSV4 fallback now renders assistant `tool_calls` back as DSML and `role=tool` replies as `<tool_result>...`; `ToolCall` also preserves explicit ids so Osaurus can correlate follow-up tool results. | DSV4 remains partial until long-context/vector/API/speed/low-footprint gates pass. |
| #1073 `Nemotron Omni live voice input path` | merged | Live voice, Parakeet/RADIO, media-cache token-aware restore, DSV4 pool/compressor fixes. | Omni JANGTQ/JANGTQ4/MXFP4 core matrices pass; the latest strict JANGTQ media-direct row is 19/19 with bundle sampling defaults and proves text reasoning remains separate from direct-answer media mode. Focused 2026-05-18 repeat-audio gates prove block-L2 and `ssm_companion` writes for BatchEngine and manual TokenIterator paths after the bench store fix; the current-head role-metadata fix removes the false JANGTQ4 Omni `config-metadata mismatch patched in-memory` warning. Seeded cache-on/cache-off repeat artifacts match visible output across all 12 audio rows while cache ON writes 433 MB and cache OFF writes 0 B, so the remaining short-audio semantic variance is not a prefix/L2/SSM restore bug. The post-prepare EVS alias contract now resolves raw media prompts to effective cache keys in `BatchEngine`, `TokenIterator`, and `NativeMTPTokenIterator`; the strict JANGTQ `BENCH_OMNI_VIDEO_CACHE_ALIAS=1` artifact proves raw 4028-token video prompts resolve to 1382 effective tokens and replay via a disk hit. DSV4 pool/compressor lineage is recorded. | Broader Omni media-thinking research, short stochastic audio-description quality, and package-wide HTTP route proof remain open. |
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
| Jinja parser and compact tool JSON | DSV4/Kimi/Gemma4/ZAYA/Laguna tool templates render without broken syntax or bloated separators. | `docs/local/production-readiness/20260517T2200_jinja_pin_parity/` and parser/cache refresh rows. Focused DSV4 red/green coverage now includes top-level tool schemas, assistant DSML tool history, `<tool_result>` follow-up turns, and explicit tool-call id preservation. | Covered for non-Kimi; Kimi excluded by instruction. |
| Swift-transformers unused placeholder skip | Added-token delimiter regex must not include thousands of unused placeholders and must preserve special wrapper tokens. | Vendored tokenizer static check plus MiniMax/DSV4/Qwen/Omni live template rows. | Covered for pinned behavior; later speed commits not yet part of Osaurus pin. |
| Generation defaults | Bundle defaults are source of truth before explicit request override. | Ledger rows print temp/topP/topK/minP/rep per family. | Covered for tested rows; require same telemetry for new rows. |
| Hybrid SSM / CCA / SWA cache | Cache proof must be topology-specific, not generic prefix-hit. | Qwen/Ling/ZAYA/Gemma4 rows record disk L2, SSM companion, CCA, SWA incompatibility, and media salt where applicable. Fresh Gemma E2B refresh records disk hits/stores plus VL media-salt restore; fresh Ling no-guard refresh confirms Bailing template/decode stress without fake sampler fixes. | Covered for listed PASS rows; DSV4 long-context and ZAYA1-VL K remain open. |
| TurboQuant KV | Explicit TQ mode must preserve coherency and prove actual compression. | `20260518T_minimax_m27_jangtqk_tq_tail_fix_exact/` proves actual TQ transitions and exact outputs after tail preservation. | Fixed for MiniMax strict row; keep family-by-family gates. |
| VL/media salt | Image/video/audio state must be isolated across turns and cache hits. | Qwen, ZAYA1-VL, Gemma4, and Omni rows prove same/different media behavior where implemented. Omni video EVS now stores under post-prepare effective tokens and the strict JANGTQ repeated-video alias row proves pre-prepare raw prompts resolve to the effective cache key and replay via disk hit. | Raw Qwen high-res video and repeated Omni cache-on audio semantics remain open. |
| Reasoning on/off | No fake close; reasoning off must affect template/runtime where supported, and visible output must remain coherent. | Gemma4 reasoning matrix, MiniMax rows, DSV4 reasoning kwargs, Ling/Bailing aliases. Fresh Gemma E2B no-guard red/green pair proves the harness now accepts coherent stellar equivalents instead of forcing decode behavior; fresh Ling row proves the Russian stress prompt with `temp=0.7` stops normally. | Covered for tested families; package-wide model matrix still open for absent local bundles. |
| MTP autodetect | Only real tensor evidence plus usable tuning may enable MTP; model names and stale metadata are insufficient. Qwen auto-depth must come from bundle-local `vmlx_mtp_tuning.json`, not profile/name rules. | Non-Kimi MTP census and Qwen MTP settings docs; CRACK rows explicitly stay MTP off. Focused tests cover tuned D2, validated D3, missing tuning, blocked tuning rows, valid tuning without MTP tensors, `MTPBundleStatus.snapshot`, missing-tuning evidence, and LLM/VLM factory wiring into `ModelConfiguration`. | Correct fail-closed policy covered; full MTP speed target remains separate/open. |

2026-05-18 15:53 PDT forced-behavior source audit mirror:

- Osaurus PR #1147 head `713aa6b7` adds
  `docs/internal/live-gates/pr1147/forced-behavior-audit-20260518T1545/REPORT.md`.
  This source-level audit is explicitly not live model proof. It names each
  output-shaping candidate, why it appears to exist, whether it is background,
  request-driven, template-bridge, or red, and what live artifact is required
  before release.
- The audit rows cover background `no_think` calls in preflight/greetings,
  explicit OpenAI `frequency_penalty` to repetition-penalty mapping,
  DSV4/Hy3 reasoning-template bridges, Ling `enable_thinking=false` plus
  reasoning-to-content merge, MiniMax no-thinking template fallback,
  family UI defaults for Qwen/Nemotron/ZAYA/Laguna/Hy3/Ling/Venice, metadata
  fallback sampler resolution, and template/JANG-config reasoning detection.
- The important red row is Ling: Osaurus currently force-sets
  `enable_thinking=false` and maps Ling `.reasoning` deltas to visible content.
  That cannot count as model correctness until live vmlx evidence proves the
  actual Ling template/parser/decode path is coherent without UI-side repair.
- Focused Osaurus verification after the audit:
  `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer swift test
  --package-path Packages/OsaurusCore --filter RuntimePolicySourceTests --jobs 2`
  passes 28/28.

2026-05-18 16:07 PDT Ling forced-behavior source fix:

- Osaurus PR #1147 head `a398a9cd` replaces the Ling fake-guard path with a
  real profile/template contract. Ling now defaults `disableThinking=true`,
  sends `enable_thinking=false` by default, honors explicit opt-in through
  `disableThinking=false` or positive reasoning request, and keeps `.reasoning`
  deltas on the reasoning rail instead of converting them to visible content.
- vmlx runtime proof lives in
  `docs/internal/live-gates/20260518T_ling_jangtq2_forced_behavior_refresh/`.
  The no-guard run loaded
  `/Users/eric/models/dealign.ai/Ling-2.6-flash-JANGTQ2-CRACK` as
  `BailingHybridModel`, then passed greedy no-repetition, thinking-on
  `rep=1.0`, and Russian `temp=0.7` stress rows at about 37-38 tok/s with
  no loop, BOS repetition, reasoning leak, marker leak, or forced close repair.
- Osaurus verification:
  `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer swift test
  --package-path Packages/OsaurusCore --filter
  'LocalReasoningCapabilityTests|GenerationEventMapperTests|MLXBatchAdapterTests|ModelProfileRegistryTests|RuntimePolicySourceTests'
  --jobs 2` passes 96/96.
- Boundary: this closes the Ling force-off/reasoning-merge source hit. It does
  not production-clear Ling or Hy3 in Osaurus until app/API long-output,
  cache/SSM, memory, route, and saved-setting carryover artifacts are attached.

2026-05-18 16:12 PDT Hy3 no-guard and native reasoning refresh:

- vmlx runtime proof lives in
  `docs/internal/live-gates/20260518T_hy3_jangtq_no_guard_refresh/`.
  `BENCH_NO_GUARD_SAMPLING=1` loaded
  `/Users/eric/models/JANGQ/Hy3-preview-JANGTQ` as `Hy3Model`, then passed
  greedy no-repetition and `rep=1.0` rows with no loop, BOS repetition, marker
  leak, or empty visible output.
- The no-guard artifact also exposed an important contract boundary: generic
  `enable_thinking=true` is not Hy3's native reasoning control. The rendered
  prompt tail stayed at `reasoning_effort:no_think`, so Osaurus must keep Hy3
  on the native `reasoning_effort` mapping rather than treating it like Qwen.
- `BENCH_REASONING_TURN_MATRIX=1` then proved the live multi-turn path:
  saved/recalled `copper-lantern`, answered sky color and math, native
  `reasoning_effort=low/high` routed reasoning deltas without marker leaks, and
  every row stopped normally. Cache stats recorded paged misses plus disk L2
  hit/store counters. This is still vmlx proof only; Osaurus app/API route,
  UI, cache, memory, tool-result, and saved-setting artifacts remain open.

2026-05-18 16:23 PDT Osaurus PR #1147 no-fake-guard release contract:

- Osaurus PR #1147 head `dc2684e0` adds a pinned
  `No-Fake-Guard Release Contract` to
  `docs/internal/live-gates/20260518T_pr1147_completion_audit.md` and locks it
  with `RuntimePolicySourceTests`.
- The contract mirrors the engine-side release rule: VL, MTP, JANG, JANGTQ,
  MXFP, MLX, dense, MoE, hybrid SSM, sliding-window, DSV4-native cache, ZAYA
  CCA, and omni/media rows must be coherent through the real runtime path. A
  model cannot pass because Osaurus or vmlx hid the symptom with a family clamp,
  parser repair, forced stop token, or output rewrite.
- Every fix after a red model run now requires a before/after live proof pair:
  the pre-fix failure artifact, a root-cause note naming the real template,
  tokenizer/BOS/EOS, native `top_k`, generation metadata, attention
  architecture, cache restore, SSM rederive, media preprocessing, MTP
  verification/commit, parser, or scheduler issue, and a post-fix artifact with
  coherent visible output, normal stop, no loop, no marker leak, cache stats,
  TTFT, tok/s, RSS, and physical-footprint context.
- MTP remains fail-closed: auto-launch requires real `mtp.*` tensor evidence
  plus an unblocked `vmlx_mtp_tuning.json`; CRACK or display-name-only rows
  stay MTP-disabled with the reason recorded. Do not force D3 when tuning says
  D2, and do not enable a blocked tuning row.
- Reasoning must be proven by family: Qwen/QwQ/MiniMax-style
  `enable_thinking`, DSV4 `instruct`/high/`reasoning_effort=max`, Hy3 native
  `reasoning_effort`, Ling default-off plus explicit opt-in, and unsupported
  families with hidden/ignored controls and no stale cache-key component.
- VLM/omni rows must send real media and prove processor/runtime agreement.
  Image+text, text-only follow-up with media salt nil/absent, different-media,
  repeated-media, unsupported-media inverse, and cache-salt behavior are
  required before any media family can pass.
- Cache and batching rows must prove single-batch and feasible multi-batch
  behavior with prefix, paged, block-L2 disk, TurboQuant KV encode/decode, SSM
  companion, DSV4 native cache/pool, ZAYA CCA, media cache, sleep/wake, and L2
  max-GB enforcement as topology-valid ON/OFF or N-A rows.
- Verification for the Osaurus gate update:
  `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer swift test
  --package-path Packages/OsaurusCore --filter RuntimePolicySourceTests
  --jobs 2` passes 28/28, and `git diff --check` passes.

## Osaurus PR #1147 Function-Level Gate

The Osaurus package-switch PR now carries an F1-F12 live checklist in
`docs/VMLX_SWIFT_OSAURUS_LIVE_MATRIX_2026_05_18.md`. Treat those rows as the
engine-facing contract for any new `vmlx-swift` runtime proof:

| Row | Engine evidence `vmlx-swift` must emit or preserve |
| --- | --- |
| F1 Model detection and metadata | Bundle path, family, parser, JANG/JANGTQ sidecars, VLM/audio/video support, MTP tensor count, `vmlx_mtp_tuning.json`, `generation_config.json`, `top_k`, and `jang_config.json`. |
| F2 UI defaults and saved settings | Capability/default snapshots with DSV4 `instruct`/`max`, Qwen/ZAYA/Nemotron/Ling no-thinking defaults, Gemma Harmony support, cache topology, and per-model scoping keys so Osaurus does not carry stale settings across families. |
| F3 Request construction | Stable request telemetry showing omitted sampler fields resolved from metadata, explicit sampler fields preserved, native `top_k` applied, tools injected only where supported, and media/content parts preserved. |
| F4 VL/video/audio preprocessing | Processor stamps and media facts: Qwen3VLProcessor/MRoPE, Gemma VLM path, ZAYA CCA, Nemotron Parakeet/RADIO, image size, video frame count, audio/pre-encode facts, media token count, media salt, and unsupported-media errors. |
| F5 Media cache boundaries | Media-salt nil/absent on text-only turns, different-media misses, repeated-media hits, restart/unload restore, and no cross-model/session media-state reuse. |
| F6 Cache stack and memory | Prefix, paged, block L2, SSM companion, DSV4 native cache, ZAYA CCA, media cache, TurboQuant KV status, L2 max-GB, TTFT, tok/s, RSS, physical footprint, and disk bytes. |
| F7 Cache inverses | ON/OFF behavior for prefix, paged, block L2, SSM companion, media cache, TurboQuant KV, reasoning, tools, streaming, VLM force-off, sleep/wake, and diagnostic flags without changing sampler defaults. |
| F8 Scheduler and lifecycle | Single-user local-chat shape, same-model continuous batching, different-model isolation, cancel/stop drain, sleep/wake restoration, no zombie Swift engine, and no stale listener/orphaned Metal context. |
| F9 Parser/channel separation | Reasoning, tools, and visible content split by family; no leaked `<think>`, DSML, Harmony, Gemma4, Qwen XML, MiniMax XML, GLM/Hunyuan, Nemotron, JSON tool schema, or tool-result markers in visible chunks. |
| F10 Old-library sweep | Consolidated modules only: `MLX`, `MLXLLM`, `MLXVLM`, `MLXLMCommon`, `VMLINUXTokenizers`, and `VMLINUXJinja`; no active dependency on old `vmlx-swift-lm`, standalone `mlx-swift`, standalone `swift-transformers`, or standalone `Jinja`. |
| F11 No fake runtime guards | Red rows stay red until root-caused. No forced repetition penalties, hidden sampler floors, forced reasoning close tags, parser repairs, fake cache fallbacks, name-only MTP, permanent overlays, or length-cap-only success. |
| F12 Forced behavior audit | Source, prompt dumps, settings previews, and live outputs must search for forced sampler defaults, forced repetition penalties, forced reasoning rail rewrites, forced `</think>` close tokens, token/logit shaping, parser repair, and template fallback behavior. Every hit must be background-only, explicit-user/API-driven, or fixed at the real template/tokenizer/decode/cache root cause before the model row can pass. |

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

2026-05-18 17:47 PDT completion-audit refresh:

- Live GitHub state for Osaurus PR #1147:
  `https://github.com/osaurus-ai/osaurus/pull/1147` is open and draft at head
  `9b0e8f19f250abbfc3f8aa400c20e31ea2a5a4fa`; `mergeStateStatus=BLOCKED`.
  CI is not a final green release signal at this snapshot: `test-core` is still
  in progress, while `test-cli`, `swiftlint`, `shellcheck`, and release drafter
  are green.
- The local PR worktree
  `/Users/eric/.config/superpowers/worktrees/osaurus/codex-vmlx-swift-package-switch`
  is clean at the same head. The current `vmlx-swift` checkout is not clean
  because another agent's Flux/native image work is present under
  `Libraries/vMLXFlux*`, `Tests/vMLXFluxTests/`, `tools/vMLXFluxProbe/`, and
  `docs/VMLX_FLUX_NATIVE_STATUS_2026_05_15.md`; do not mix that work into this
  engine/readiness row.
- The Osaurus-side tracker now names the current switch state explicitly in
  `docs/internal/live-gates/20260518T_pr1147_model_function_compatibility_tracker.md`:
  the PR is source-wired and harnessed, but not production-ready. Existing live
  red/partial rows remain red for Gemma3n UTF/string behavior, ZAYA text UTF /
  direct-mode behavior, ZAYA-VL media carryover/Responses grounding, and missing
  broad UI/API/cache/memory/saved-setting evidence across the remaining model
  families.
- This `vmlx-swift` audit mirrors that status. Do not convert a direct
  `RunBench` pass, source test, metadata census, or Osaurus route scaffold into
  a production claim until the matching Osaurus chat-app/API artifact folder has
  visible coherent output, normal stop, no loop, no marker leak, resolved bundle
  defaults, cache stats, TTFT, tok/s, RSS plus physical-footprint context, and
  an explicit no-fake-guard review.

## PR #1147 Next Live Rows

These are the concrete rows that still need app/API execution before Osaurus can
switch entirely to this package. Each row must include the raw request, output
body or stream frames, `health`, `/admin/cache-stats`, process memory, resolved
generation defaults, parser/channel review, and a short human output review.

| Row | Current blocker or proof gap | Required next artifact |
| --- | --- | --- |
| DSV4 Flash settings/runtime | vmlx has DSML, prompt-boundary, reasoning, and DSV4 cache proof; Osaurus still needs the real server-settings visual/API row. | `pr1147/deepseek-v4-flash-*/` with UI model picker, server-settings CLI preview, `reasoning_effort=max`, DSML tools off/on/result, long/growing chat, DSV4 native cache stats, block size 256 fixed/disabled, generic q4/q8/JIT omitted, pool quant visible, TTFT/tok/s/footprint. |
| Qwen3.6 MTP VL | Census and vmlx rows prove tensor/tuning-gated MTP; Osaurus has not yet proven UI/API MTP ON/OFF with media. | `pr1147/qwen3.6-*-mxfp*-mtp/` with real `mtp.*` count, `vmlx_mtp_tuning.json` depth, MTP off baseline, MTP on depth/speed/acceptance, image T1, text-only T2 with nil media salt, different image/video, repeated-media hit, reasoning on/off, Qwen tools, cache stats, footprint. |
| Qwen non-MTP controls | CRACK rows must remain MTP disabled from real weight evidence, not names. | `pr1147/qwen3.6-*-crack/` with status reason `no mtp tensor evidence`, no MTP UI activation, text/VL/media cache proof where supported, and no stale MTP cache-key component. |
| Gemma4 / Gemma VLM | vmlx has Gemma4 text/VL/Harmony/tool proof; Osaurus route/UI proof remains open. | `pr1147/gemma4-*/` with image+text, text-only follow-up, image switch/repeat, Harmony analysis/final separation, Gemma tool call/result, stream/non-stream Chat and Responses, cache topology stats, TTFT/tok/s/footprint, no Harmony/Gemma marker leak. |
| Gemma3n E2B text | Osaurus live row remains fail/partial: math and sky are coherent, UTF/string row is red, and the latest post-scrubber row lacks cache proof due immediate idle unload. | Rerun with non-immediate residency or stream-time snapshots, exact UTF/string row, Chat and Responses stream/non-stream, cache counters and physical footprint. Root cause must be template/tokenizer/decode/route, not sampler clamp or output repair. |
| ZAYA text | Osaurus MXFP4 text row proves cache movement but Responses UTF is red; vmlx JANGTQ4 direct-mode math remains a strict-prompt blocker. | `pr1147/zaya1-8b-*/` with math, follow-up, UTF, reasoning/coding-context isolation, SSM companion and block-L2 counters, and source trace for UTF/direct-mode failures. No top-k/repetition/stop-token guard. |
| ZAYA-VL | Osaurus artifacts expose partial/red media carryover; ZAYA1-VL video is correctly unsupported. | Fresh image-only row with red image T1, text-only T2, blue image T3, repeated image, Chat and Responses grounding, CCA/media salt hit/miss, unsupported-video UI/API rejection, speed/footprint, no stale prior-image state. |
| Nemotron Omni / Parakeet / RADIO | Text and one WAV Chat smoke exist; streaming, repeat media, RADIO/video, and sleep/wake are not closed. | `pr1147/nemotron-omni-*/` with audio raw and pre-encoded, repeated audio cache, image/video where supported, Parakeet/RADIO facts, streaming terminal frames, sleep/wake, cache stats, TTFT/tok/s/footprint, no reasoning-only short-budget pass. |
| MiniMax | vmlx has reasoning/cache/TQ proof; Osaurus UI/API tool/reasoning rows are open. | `pr1147/minimax-m2.7-*/` with reasoning-only behavior, `enable_thinking` off/on, MiniMax XML/JSON parser, tools off/on/result, cache stack, TurboQuant KV inverse, no MTP from CRACK/name, tok/s/footprint. |
| Ling / Hy3 hybrid SSM | vmlx no-guard proof exists; Osaurus app/API/cache rows remain partial/open. | `pr1147/ling-*` and `pr1147/hy3-*` with family-specific reasoning defaults, explicit opt-in/native effort, long prompt, overlap/mismatch async rederive, SSM companion hit/miss/store, no KV-only unsafe hit, parser/no-leak review, TTFT/footprint. |
| GLM / GPT-OSS / Mistral parser families | Parser source tests are not model production proof. | Only run when local bundles exist; require base-architecture parser detection, reasoning/tool rows, route parity, marker no-leak, and topology-specific cache stats. |
| UI saved settings and visuals | Source policies exist; user-facing controls and carryover still need proof. | Cross-family app artifact: Qwen thinking -> Ling/no-thinking, DSV4 `max` -> Qwen/Gemma/ZAYA/Nemotron, VLM media -> text-only, cache OFF/ON restore, tool/coding context switch, send/stop/retry/edit/copy, thinking panel, tool card, media preview, unsupported-media error, token/s display, sleep/wake state. |
| Old library and zombie-code sweep | Current sweep is partial because late fixes can reintroduce imports or CLI flags. | Final post-live sweep of Osaurus and `vmlx-swift`: no active inference path through old `vmlx-swift-lm`, standalone `Jinja`, standalone `swift-transformers`, invalid DSV4 CLI flags, app-side parser repair, or hidden sampler guard. |

## Forced-Behavior Source Search Terms

Every source or live-output audit should at minimum search these concepts before
marking a row clear. A hit is allowed only when it is explicit user/API policy,
bundle metadata, a topology safety gate, or a diagnostic artifact. Any hit that
changes output to hide incoherence is a release blocker.

```text
forced temperature/top_p/top_k/min_p
default repetition penalty / family repetition floor
forced EOS / hidden stop sequence / length-cap success
forced </think> / reasoning close repair / reasoning-to-visible conversion
parser scrub / parser repair / XML or DSML output cleanup
token or logit shaping outside explicit sampler/request policy
name-based MTP / name-based VLM / metadata-only MTP auto enable
KV-only hybrid SSM cache hit / media salt reuse / cross-session CCA reuse
DSV4 generic paged/KV/JIT flag use
global TurboQuant KV quality default
saved reasoning/tool/media/cache setting crossing incompatible families
```

2026-05-18 17:58 PDT vmlx source-harness forced-behavior fix:

- A source scan found a harness-level false-green pattern in `RunBench`: generic
  batch turns and DSV4 coherence rows used
  `text.isEmpty ? reasoning : text` when constructing "visible" output. That
  could let a reasoning-only response count as visible user content in a
  release artifact.
- The fix keeps reasoning and visible text separate in the benches. Preview logs
  may still print the reasoning byte count for diagnostics, but pass/fail
  predicates now require actual `.chunk` text unless the row is explicitly a
  reasoning-channel inspection. UTF, math, rapid-turn, DSV4 recall, and fallback
  tool-content checks no longer accept reasoning-only text as visible output.
- Focused guard:
  `NoHiddenReasoningCloseBiasFocusedTests.runBenchGatesDoNotCountReasoningOnlyOutputAsVisible`
  searches the active `RunBench` production harness files so this fallback is
  not reintroduced in normal, stability, VL, or Omni rows. The same pass also
  removed the reasoning-as-output fallback from `StabilityBench` rows S1, S2,
  S3, S4, S5, S6, S7, S8, S9, S10, S11, and S12. JangPress is not part of the
  current active Osaurus switch-readiness gate.
