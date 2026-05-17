# vMLX Swift Completion Audit - 2026-05-16

This audit turns `goal.md` into a concrete completion checklist for the
consolidated `vmlx-swift` engine. It is intentionally conservative: a row is
only `live-proven`, `unit-tested`, or `static-proven` when there is a real local
artifact or source reference. Everything else stays `open`.

Current pushed branch state:

- Branch: `vmlx-0.31.3`
- Latest pushed checkpoints are tracked in git history; this audit is
  artifact-first and should not be treated as authoritative from the header hash
  alone.
- 2026-05-17 local update: the latest six-variant Qwen3.6 matrix is documented
  in `docs/VMLX_QWEN36_MTP_MATRIX_2026_05_17.md`. It supersedes earlier
  optimistic MTP speed wording in this audit: current Swift D3 text rows are
  coherent for all six variants and cache repeat rows hit disk+SSM, but D3 is
  slower than AR in that matrix and VL+MTP still fails 35B JANG_2K under strict
  no-length-cap criteria. Later MXFP-only production and VL reruns with
  `BENCH_MAX_TOKENS=384` pass 27B/35B MXFP4/MXFP8 text gates 7/7 and clear the
  35B MXFP4 VL row, so the earlier MXFP visible-answer failures were
  short-budget failures for those rows.
- 2026-05-17 later local update:
  `docs/local/production-readiness/20260517T174743Z_qwen_mtp_chunk_policy_finalize/`
  found and fixed a no-diagnostics `chunk_commit` verifier-state bug. Phase/GDN
  diagnostics had accidentally materialized recurrent prefix snapshots; without
  those diagnostics, 35B MXFP4 D3 stored lazy GDN state and degenerated until
  length stop. `MambaCache.recordPrefixCommitState(...)` now materializes the
  snapshot before storing it.
- 2026-05-17 focused Ling/Hy3/Gemma4 update:
  `docs/local/production-readiness/20260517T1252_ling_hy3_gemma4_runtime_contracts/`
  passes `MLXLMCommonFocusedTests` with 167 tests in 26 suites. The new active
  proof covers actual Ling/Bailing hybrid cache allocation, actual
  `Gemma4TextModel.newCache` SWA/global topology, Hy3 nextn exclusion from base
  decode cache plus q/k/v sanitizer fusion, and lower-only MoE top-k override
  cache scoping. This is focused contract proof; it is not a replacement for
  live multi-turn model rows.
- 2026-05-17 Hy3 mixed-qkv follow-up:
  `docs/local/production-readiness/20260517T1300_hy3_mixed_qkv_runtime_contracts/`
  passes `MLXLMCommonFocusedTests` with 168 tests in 26 suites after replacing
  the Hy3 mixed packed-width q/k/v sanitizer `fatalError` with a real
  dequantize-then-fuse fallback. If quant metadata is missing, the source keys
  remain so load verification fails instead of silently running random qkv
  weights.
- 2026-05-17 MTP launch-settings follow-up:
  `docs/local/production-readiness/20260517T1305_mtp_settings_profile_validation/`
  passes `VMLINUXServerRuntimeSettingsTests` with 15 Swift Testing rows. The
  server settings surface now has a full-evidence validation path so preserved
  Qwen MTP can be shown as a D3 candidate without auto-launch, while a force-on
  launch against blocked profiles such as Qwen3.6 JANG_2K is rejected before an
  Osaurus session starts.
- 2026-05-17 Harmony prompt-tail follow-up:
  `docs/local/production-readiness/20260517T1315_harmony_prompt_tail_parser/`
  passes the `HarmonyParserFocusedTests` suite with 6/6 rows. The added guard
  exercises `ReasoningParser.forPrompt(...)`, which is the production
  BatchEngine/Evaluate path, so GPT-OSS/Gemma4-style Harmony channel markers
  stay stripped even when prompt-tail detection is in the loop.
- 2026-05-17 no-hidden-reasoning regression sweep:
  `docs/local/production-readiness/20260517T1320_no_hidden_reasoning_regression/`
  passes 42 rows across Harmony, Hy3, Ling/Bailing, Laguna, Mistral/Ministral,
  Gemma4 VLM source contracts, Mistral3 JANGTQ dispatch, and no-hidden-close-bias
  checks.
- 2026-05-17 Gemma4 SWA compile-policy follow-up:
  `docs/local/production-readiness/20260517T1325_gemma4_swa_compile_policy/`
  passes 5/5 focused rows. Default Gemma4 mixed SWA/full-attention cache stays
  `.heterogeneous` and uncompiled; an explicit bounded all-rotating cache is
  `.rotating` and compile-eligible.
- 2026-05-17 server-settings validation follow-up:
  `docs/local/production-readiness/20260517T1335_server_settings_validation/`
  passes `VMLINUXServerRuntimeSettingsTests` with 16/16 rows. The server
  settings contract now rejects invalid network, concurrency, prefix-cache,
  generation, sleep, TurboQuant KV, and MTP values instead of clamping.
- 2026-05-17 active cache-policy salt follow-up:
  `docs/local/production-readiness/20260517T1348_cache_policy_salt_active/`
  passes `CacheCoordinatorTopologyFocusedTests` with 24/24 rows. The active
  focused target now pins semantic reasoning scope, reasoning effort, KV codec,
  and max-KV policy into cache salts and coordinator hash tiers, preventing
  incompatible paged/disk/SSM cache reuse across mode changes.
- 2026-05-17 parser fallback follow-up:
  `docs/local/production-readiness/20260517T1405_parser_fallback_matrix/`
  contains a red/green pair for `DirectCapabilityParserAliasFocusedTests`. The
  pre-fix red log proved Ling/Bailing and Qwen3.6/Qwen3-VL model-type fallbacks
  returned nil tool formats; the post-fix green log passes 8/8 after adding
  real fallback routing to GLM/deepseek arg-key tools and Qwen XML-function
  tools. The broader `NoHiddenReasoningCloseBiasFocusedTests_green.log` passes
  44/44 after the change. This does not override JANG capability stamps.
- 2026-05-17 Qwen-VL capability-alias follow-up:
  `docs/local/production-readiness/20260517T1415_qwen_vl_capability_aliases/`
  contains a red/green pair proving direct capability stamps `qwen3_vl`,
  `qwen3_5_vl`, and `qwen3_6_vl` previously bypassed both Qwen thinking and
  XML tool routing. The post-fix direct alias suite passes 9/9, and the broader
  `NoHiddenReasoningCloseBiasFocusedTests_green.log` passes 45/45. This is a
  parser capability fix only; it does not enable MTP or VL paths by name.
- 2026-05-17 Harmony fragmentation/leak follow-up:
  `docs/local/production-readiness/20260517T1435_harmony_fragment_leak/`
  contains a red/green pair for the Gemma4/GPT-OSS Harmony parser. The red log
  proved stray `<|message|>` and `<|channel|>` markers could leak into visible
  free text outside a well-formed channel. The final green log passes
  `HarmonyParserFocusedTests` 8/8 and the broader no-hidden/parser sweep 47/47,
  including one-character token fragmentation and start-role buffering so
  `<|start|>assistant` does not leak `assistant` when split across chunks.
- 2026-05-17 Gemma4/SWA direct compile follow-up:
  `docs/local/production-readiness/20260517T1450_gemma4_rotating_compile_direct/`
  contains a red/green pair for `Gemma4CacheTopologyFocusedTests`. The red log
  proved direct `TokenIterator.setupCompiledDecode` still promoted only
  `KVCacheSimple`, leaving all-rotating Gemma4/SWA caches uncompiled outside
  BatchEngine. The green logs pass the Gemma4 suite 6/6 and the broader cache
  topology suite 25/25 after adding real `CompilableRotatingKVCache` promotion.
- 2026-05-17 Ling/Bailing capability-alias follow-up:
  `docs/local/production-readiness/20260517T1505_ling_bailing_capability_aliases/`
  contains a red/green pair for direct `bailing*` and `ling*` capability
  stamps. The pre-fix red log proved those direct stamps could bypass
  think-XML reasoning and GLM/deepseek-style tool parsing; the green direct
  suite passes 10/10 and the broader no-hidden/parser sweep passes 48/48.
- 2026-05-17 versioned Harmony/GLM alias follow-up:
  `docs/local/production-readiness/20260517T1347_harmony_glm_alias_normalization/`
  passes the broader no-hidden/parser sweep with 49/49 rows after normalizing
  hyphen/underscore/dot aliases. This pins `gemma-4`/`gemma_4` to Gemma4
  Harmony + Gemma4 tools, `glm-5.1-flash`/`glm_5_1_flash` to think-XML +
  GLM tools, and `gpt-oss`/`gptoss` to Harmony reasoning + GLM-style tool
  parsing instead of returning nil and leaking controls to visible output.
- 2026-05-17 Hy3 product-alias facade follow-up:
  `docs/local/production-readiness/20260517T1355_hy3_product_alias_parser/`
  contains the red/green evidence for product/version parser aliases through
  the public `ParserResolution` facade. The red run proved stamped
  `hy3-preview` capabilities returned nil for both reasoning and Hunyuan tools
  even though `model_type` fallback worked. The green focused parser run passes
  51/51 rows and the aggregate `MLXLMCommonFocusedTests` run passes 186/186
  after resolving `hy3*` / `hy_v3*` / `hunyuan*` capability stamps directly.
  The same pass updates public comments so `deepseek` remains GLM/DeepSeek-V3
  arg-key tools, while `deepseek_v4` remains DSML.
- Latest pushed runtime checkpoint for the Qwen text-SSM/private-MTP cache
  fix: `3146fac` (`fix(mtp): repair qwen ssm reject cache`)
- The MTP/cache work for task-local load-time native MTP activation, MXFP8
  metadata/norm handling, native-MTP cache telemetry, and CacheCoordinator
  restore/store wiring is now represented by pushed docs and focused artifacts
  under `docs/local/production-readiness/`.
- Prior non-MTP checkpoints in this pass: `50df533`, `0deb14b`,
  `6e435d7`, `7962647`, `9a56de1`, and `ed04161`.
- Previous MTP runtime checkpoint: `0fdb164`
  (`feat(runtime): add exact native mtp sampling`)
- Current worktree is not clean because another agent is working on Flux-native
  Swift files. The dirty Flux files are excluded from this audit's commit scope.
- Current focus: DSV4 Flash, Nemotron Omni, ZAYA1-VL, cache/template
  correctness, live coherent multi-turn rows, and explicit-only Qwen3.6 native
  MTP readiness. Native MTP is not auto-launch eligible yet.

## Objective Restated as Deliverables

The package is complete only when all of these are true:

1. `vmlx-swift` is the single Swift package surface Osaurus can pin instead of
   separately pinning `vmlx-swift-lm`, `mlx-swift`, `swift-transformers`, and
   Jinja.
2. All runtime surfaces have live or unit-test evidence: model loading, chat
   templates, generation config, reasoning modes, tool parsers, VL/audio/media,
   cache topology, continuous batching, TurboQuant/JANGTQ, MTP, and distributed
   stubs.
3. Every local model family under `~/models` has a status row with load,
   template, generation, multi-turn, cache, batching, speed, and result.
4. Cache behavior is topology-specific: no generic paged/prefix hit is accepted
   for DSV4 CSA/HSA/SWA, ZAYA CCA, Qwen hybrid SSM, VL/media, audio, or other
   path-dependent state.
5. No fake runtime guard is used to hide bad behavior. No hidden repetition
   penalty, no forced think close, no forced stop, no hidden reasoning downgrade,
   and no all-or-nothing MTP acceptance.
6. Osaurus wiring is mapped and later repinned only after the package evidence
   is strong enough.
7. All claims are pushed, committed, and documented with exact local artifact
   paths for raw logs that cannot be committed.

## Prompt-to-Artifact Checklist

| Requirement from `goal.md` | Current evidence | Status |
| --- | --- | --- |
| Check repo/process state before long tests. | `pgrep` checks before/after recent RunBench/test work; no `RunBench`/Swift build/test processes running after latest push. | live-proven |
| Do not overwrite unrelated work. | Current dirty files are Flux-owned (`Package.swift`, `Libraries/VMLX/VMLX.swift`, `Libraries/vMLXFlux*`, `Tests/vMLXFluxTests`, `tools/vMLXFluxProbe`) and were left unstaged. | live-proven |
| Package graph should consolidate MLX/Jinja/Transformers-style code. | `Package.swift` at pushed HEAD exposes products `MLX`, `Jinja`, `Tokenizers`, `MLXLMCommon`, `MLXLLM`, `MLXVLM`, `MLXPress`, and `VMLX`; `Libraries/VMLX/VMLX.swift` re-exports the stable Osaurus import surface. | static-proven |
| Osaurus current pin/wiring map. | `docs/VMLX_OSAURUS_CONSOLIDATION_AUDIT_2026_05_15.md`; current OsaurusCore pins are `mlx-swift=0a56f904`, `vmlx-swift-lm=2cc64dd`, `Jinja=58d21aa`, `swift-transformers=087a66b`. | static-proven |
| Compare `../vmlx-swift-lm`. | `docs/VMLX_SWIFT_LM_DEEP_COMPARISON_2026_05_15.md` plus updated consolidation audit. Sibling is dirty and behind origin; do not copy blindly. | static-proven |
| Local model inventory. | `docs/VMLX_LIVE_MODEL_MATRIX_2026_05_15.md`; local `find ~/models -maxdepth 4 -name config.json` shows DSV4, Hy3, Kimi, Laguna, MiniMax, ZAYA/ZAYA-VL, Qwen3.5/3.6, Gemma4, Ling, and Nemotron Omni families. | static-proven |
| Live infer matrix for installed models. | `docs/local/live-model-matrix/20260515Tinfer-under20/REPORT.md` and `status.tsv`. Many under-20GB rows passed; several rows failed or were skipped. | partial |
| MTP activation must use real tensor evidence, not names. | `9b1b254`; `docs/VMLX_SWIFT_MTP_OSAURUS_WIRING_2026_05_15.md`; local fail-closed artifact `docs/local/native-mtp-qwen36-20260515/jang4m-crack-native-mtp-denied-postrevert.log`. Fresh settings proof `docs/local/production-readiness/20260517T1305_mtp_settings_profile_validation/VMLXServerRuntimeSettingsTests_profile_validation.log` passes 15/15 and proves preserved-only Qwen MTP can be exposed as a D3 candidate without auto-launch, while profile-blocked force-on launch reports a real settings error before runtime. | live-proven |
| MTP depth-3 production acceleration. | Current local Swift rows prove recursive D3 draft, accepted-prefix commit, coherent visible output, and no loop/leak for Qwen3.6 27B JANG_4M, 27B MXFP4, 27B MXFP8, 35B MXFP4, and 35B MXFP8. The fresh MXFP production reverify passes 27B/35B MXFP4/MXFP8 7/7 with `BENCH_MAX_TOKENS=384`, explicit `chunk_commit`, bundle defaults, disk L2 hit, and SSM companion hit. The latest focused artifact `docs/local/production-readiness/20260517T174743Z_qwen_mtp_chunk_policy_finalize/MTPRuntimeFocusedTests_after_prefix_snapshot_materialize.log` passes 42/42 and pins tensor-gated activation, preserved-only denial, D3 hidden-state draft/verify, accepted-prefix SSM offsets, partial-reject repair, verifier env override ordering, materialized recurrent prefix snapshots, BatchEngine exclusive native-MTP lane, quant shape-walk, and norm-convention behavior. Native MTP remains explicit/tensor-gated; auto-launch still needs Osaurus/server policy gates. | live-proven for explicit Qwen3.6 correctness rows; speed partial |
| Qwen3.6 MTP coherency and token/s. | Historical target artifacts include `qwen36_27b_jang4m_mtp_d3_count_python_prompt_normfix_regression.log` (`47.7 tok/s`), `qwen36_27b_mxfp4_mtp_d3_count_python_prompt_normfix.log` (`50.5 tok/s`), `qwen36_27b_mxfp4_mtp_d3_count_python_prompt_postaudit_count.log` (`49.6 tok/s` after task-local activation), `qwen36_27b_mxfp8_mtp_d3_count_python_prompt_normfix.log` (`29.5 tok/s`), and `qwen36_35b_mxfp8_mtp_d3_count_python_prompt.log` (`130.6 tok/s`). Fresh no-diagnostics snapshot-fix rows are more conservative: 35B MXFP4 exact `1..50` at `84.7 tok/s`, and 27B MXFP4 exact `1..50` at `26.1 tok/s`, both with `unclosedReasoning=NO`, `loop=NO`, `leaks=none`. | live-proven for coherency; current speed partial |
| Qwen3.6 native-MTP speed target. | Historical rows met the 27B MXFP4 and JANG_4M target, but the current no-diagnostics Swift rows do not reproduce that speed. 27B MXFP4 is correct at `26.1 tok/s`, and 35B MXFP4 is correct at `84.7 tok/s` versus a fresh AR row at `91.0 tok/s`. Treat the speed target as open until the current verifier path reaches the Python/older Swift target without diagnostics. | partial |
| Qwen3.6 text hybrid-SSM accepted-prefix cache commit and private MTP-cache reject semantics. | 2026-05-17 fixes align the text Qwen3.5/Qwen3.6 `GatedDeltaNet` with the VLM path: regular SSM forwards now advance `MambaCache.offset`, recorded partial-prefix snapshots store `baseOffset + prefixLength`, and partial rejects recreate the private MTP draft cache instead of trimming stale rejected state. The latest fix materializes prefix recurrent snapshots before saving a verifier commit point, closing the no-diagnostics `chunk_commit` failure in `docs/local/production-readiness/20260517T174743Z_qwen_mtp_chunk_policy_finalize/`. Focused proofs: `docs/local/production-readiness/20260517Tqwen36-mtp-ssm-offset/mtp_runtime_focused_after_private_mtp_refresh.log` passes 17/17, and `MTPRuntimeFocusedTests_after_prefix_snapshot_materialize.log` passes 42/42. | unit-tested and live-proven for current 35B MXFP4 no-diagnostics row |
| Qwen3.6 MXFP8 norm convention. | 2026-05-17 fix propagates `norm_convention` from safetensor metadata or `jang_config.json` into Qwen3.5/Qwen3.6 text and VLM sanitizers. Current Swift rows proved the real bug: repaired MXFP4/MXFP8 language backbones are already MLX-ready, while preserved MTP norms can still be raw and need independent MTP norm shifting. Pre-fix MXFP8 D3 accepted zero drafts at `7.0 tok/s`; post-fix accepts depths `1/2/3` and reaches `29.5 tok/s`. | live-proven |
| Qwen3.6 native-MTP cache stack. | Native MTP now accepts a `CacheCoordinator` on `Evaluate.generate` and `BatchEngine.generate` solo paths. Qwen3.6 hybrid/Mamba topology correctly flips `pagedIncompatible=true`; paged counters stay zero by design, while L2 disk and SSM companion counters increment on repeated exact prompts and current MXFP growing-chat rows. Artifacts: `qwen36_27b_mxfp4_mtp_d3_hybrid_disk_cache_repeated_prompt.log`, `qwen36_27b_mxfp4_mtp_d3_hybrid_disk_cache_postaudit.log`, `qwen36_27b_mxfp8_mtp_d3_hybrid_disk_cache_repeated_prompt.log`, `qwen36_35b_mxfp8_mtp_d3_hybrid_disk_cache_repeated_prompt.log`, and `docs/local/qwen36-mtp-current/20260517T124945Z-35b-mxfp4-vl-mtp-budget384/vl_chat_cache_mtp_d3_budget384.log` for 35B MXFP4 VL same-media disk restore. | live-proven for MXFP exact store/fetch and growing-cache rows; package-wide partial |
| Cache-policy and reasoning-scope isolation. | `docs/local/production-readiness/20260517T1348_cache_policy_salt_active/CacheCoordinatorTopologyFocusedTests.log` passes 24/24 and proves the active focused target includes dynamic reasoning scope across DiskCache, CacheBlock, and SSMStateCache hashes; semantic-only `cacheScopeSalt` extraction; text-only policy salts; and distinct salts for float KV, affine KV, TurboQuant KV, and `maxKVSize` rotating policy. | unit-tested |
| Qwen3.6 native-MTP growing-chat partial cache reuse. | Current post-fix MXFP rows under `docs/local/qwen36-mtp-current/20260517T131050Z-mxfp-growing-chat-mtp-d3-exact-postfix/` plus `20260517T131024Z-35b-mxfp4-growing-chat-mtp-d3-exact-postfix/` prove D3 native MTP with bundle defaults, coherent two-turn output, canonical history-boundary disk hit `18/53`, SSM companion hit increments, and lower turn-2 prompt time. The pre-fix 35B MXFP4 chunk exact-pq row `20260517T125723Z-mxfp-growing-chat-mtp-d3-stats/35b-mxfp4/...` failed with repeated garbage; the fix routes non-greedy hybrid SSM to `sequential_repair` while keeping greedy `chunk_commit` available. | live-proven for current MXFP rows; package-wide partial |
| Qwen3.5 hybrid BatchEngine mixed KV codec isolation. | Pre-fix release turnmatrix `docs/local/live-model-matrix/20260517T161305Z_release_turnmatrix_qwen35_35b_4bit/` failed only `batch_tq_b2`: plain slot output drifted when the neighbor used TurboQuant KV. The repair is real cache topology handling: `BatchArraysCache.splitBack()` propagates the model-mutated recurrent offset back into per-slot Mamba caches, and `BatchEngine` splits simultaneously active decode slots by cache/codec compatibility instead of forcing plain KV and TurboQuant KV into one forward. Focused proof: `docs/local/live-model-matrix/20260517T164102Z_batch_arrays_cache_offset_focused/BatchArraysCacheFocusedTests.log` passes 1/1, and `docs/local/live-model-matrix/20260517T163920Z_qwen35_tq_b2_after_compat_split/qwen35_batch_tq_b2.out` passes with slot 0 identical to the B=2 plain/plain reference and `compatibilitySplits=191`. Full post-fix proof: `docs/local/live-model-matrix/20260517T164940Z_release_turnmatrix_qwen35_35b_4bit_after_compat_split/REPORT.md` passes all runnable rows, including `batch_tq_b2`, VL structured cache, media-salt isolation, and mixed text/image/video; only generic `batch_cache_hit` is `N-A` by topology/harness semantics. Video is coherent but slow (`TTFT 136312ms`). | live-proven |
| DSV4 native encoder, CSA/HSA/SWA, long context, vector drift. | New non-MTP DSV4 JANGTQ-K rows under `docs/local/live-model-matrix/20260516Tdsv4-nonmtp/` prove config/template, three-turn recall, reasoning off/on/max with rep=1.0, a 5.5k-token semantic recall row, and DSV4 paged-incompatible salted disk-cache restore. A 16k-token long-context row currently fails with Metal OOM, and the ds4 official vector fixture is not present locally. | partial/failing |
| Prefix cache OFF/ON and cache hit proof. | Existing matrix/harness describes rows; not complete for every topology and model family. | open |
| Paged cache OFF/ON. | Existing focused tests and some model rows exist, but no package-wide matrix artifact proves all relevant architectures. | open |
| Disk L2 OFF/ON and fresh-session restore. | Existing docs and some rows exist; package-wide, per-topology proof remains incomplete. | open |
| Qwen-style stateless full-history cache boundary. | `docs/local/live-model-matrix/20260516Tguard-removal/Qwen3.6-27B-JANG_4M-CRACK_growing_chat_cache_history_boundary_final.out` and `.../Qwen3.6-27B-MXFP4-CRACK_growing_chat_cache_probe.out` prove real disk hits at canonical history boundaries after rendered turn-2 history diverges from the turn-1 generation prompt. | live-proven for Qwen3.6 JANG_4M and MXFP4 CRACK non-MTP |
| SSM companion cache and async rederive. | Qwen/hybrid rows are required by docs; not exhaustively live-proven for all relevant local models. | open |
| VL media salt, same-image hit, changed-image miss. | `docs/local/live-model-matrix/20260516Tzaya-vl-think-template-fix/ZAYA1-VL-8B-JANGTQ4_vl_chat_cache.out`: same-media replay HIT, different-media MISS, coherent blue/orange follow-up. `docs/local/live-model-matrix/20260516Tguard-removal/Qwen3.6-27B-JANG_4M-CRACK_vl_chat_cache_final.out` and `.../Qwen3.6-27B-MXFP4-CRACK_vl_chat_cache_probe.out`: Qwen3VLProcessor same-media disk HIT `84/84`, different-media MISS, coherent follow-up. | live-proven for ZAYA1-VL JANGTQ4 plus Qwen3.6 JANG_4M/MXFP4 CRACK |
| Nemotron Omni audio/Parakeet/RADIO. | Video generation now carries the processor's post-EVS keep count and applies real EVS before prompt splice. `docs/local/live-model-matrix/20260516Tomni-nonmtp/Nemotron-Omni-Nano-JANGTQ4-CRACK_omni_evs_v2.out` passes 13/13 TokenIterator rows; strict pre-fix artifact `..._omni_strict.out` failed the video row. The second fix canonicalizes the closed no-thinking media tail to `<think>\n</think>\n\n`; tail probe `..._omni_tail_probe.out` proves compact tail fails and spaced tail grounds the same image, and `..._omni_batch_nothink_tail_fix.out` passes 18/18 including direct and BatchEngine image with `enable_thinking=false`. Fresh wrapper fix: `docs/local/live-model-matrix/20260517T170614Z_omni_live_voice_fresh_recheck/processor_audio_wrapper_red.log` proves the stale Swift audio wrapper tokenized literal `<sound>` text and missed source wrapper IDs `28`/`29`; `processor_audio_wrapper_green.log` passes after switching to `<so_start>`/`<so_end>`, and `NemotronHOmniPreEncodedAudioTests_after_audio_wrapper_fix.log` passes 9/9. Live after-fix evidence: `omni_audio_latency_jangtq4_both_paths_cache_off_32_after_audio_wrapper_fix.jsonl` reloads JANGTQ4, uses bundle defaults, pre-encodes Parakeet to `63 x 2688` in 46.7 ms, and streams raw PCM plus pre-encoded audio through BatchEngine and TokenIterator at 65.4-76.1 tok/s with no literal sound-marker leak; `omni_audio_latency_jangtq4_both_paths_cache_off_32_repeats3_after_audio_wrapper_fix.jsonl` repeats 12 cache-off rows with no marker leak and 70.6-76.6 tok/s on iterator rows; `omni_runbench_jangtq4_48_after_audio_wrapper_fix.log` passes integrated `BENCH_OMNI=1` + `BENCH_OMNI_BATCH=1` 18/18 at `maxTokens=48`. Caveat: short BatchEngine/pre-encoded audio rows are still semantically weak, independently encoded Parakeet chunks remain not concat-safe, and cache-on repeated live audio remains partial until the cache-hit output-quality/root-cause gate is fixed. | live-proven for Omni JANGTQ4 core; cache-on repeated audio partial |
| Reasoning on/off/effort matrix. | Focused DSV4 pass-through exists. MiniMax Small now has live thinking ON/OFF alternation with `.reasoning` deltas ON and zero reasoning OFF. Ling/Bailing template has no active thinking rail in this bundle and returns visible content with no marker leak; active focused tests now also pin Bailing/Ling `enable_thinking` directive insertion/replacement. Gemma 4/GPT-OSS Harmony parser contracts now have active focused proof for Gemma 4 `<|channel>...<channel|>` and GPT-OSS `<|channel|>analysis/final<|message|>...` splitting, including `gpt_oss* -> harmony` model-type stamping. Fresh prompt-tail proof `docs/local/production-readiness/20260517T1315_harmony_prompt_tail_parser/HarmonyParserFocusedTests.log` passes 6/6 and verifies the production `ReasoningParser.forPrompt(...)` path strips GPT-OSS/Gemma4 Harmony control markers before `.chunk` emission. The fresh fragmentation/leak proof `docs/local/production-readiness/20260517T1435_harmony_fragment_leak/` passes the Harmony suite 8/8 and broader no-hidden sweep 47/47 after adding one-character chunking and stray control-token scrub coverage. The broader no-hidden-reasoning sweep `docs/local/production-readiness/20260517T1320_no_hidden_reasoning_regression/NoHiddenReasoningCloseBiasFocusedTests.log` passes 42/42 across Harmony, Hy3, Ling/Bailing, Laguna, Mistral/Ministral, Gemma4 VLM source contracts, Mistral3 dispatch, and no forced close-bias checks. The alias-normalization sweep `docs/local/production-readiness/20260517T1347_harmony_glm_alias_normalization/NoHiddenReasoningCloseBiasFocusedTests.log` passes 49/49 and adds hyphen/underscore/dot coverage for `gemma-4`, `glm-5.1-flash`, and `gpt-oss` aliases. Direct capability aliases with suffixes now also resolve instead of bypassing the parser: `gemma4_27b` / `gpt_oss_20b` / `gpt_oss_120b` route to Harmony, `glm4_moe_lite` / `glm5_air` / `deepseek_v4_flash` / `laguna_glm_thinking_v5` route to think-XML, direct Qwen-VL capability stamps `qwen3_vl` / `qwen3_5_vl` / `qwen3_6_vl` route to Qwen thinking, and explicit `mistral4*` capability stamps route `[THINK]...[/THINK]` while the `mistral4` model-type fallback stays no-reasoning unless the bundle stamps that parser. Active Hy3 focused tests pin Hunyuan/think_xml aliases plus open/closed prompt-tail reasoning separation, and now also prove preserved nextn layers are not inserted into the base decode cache. Active Laguna tests pin thinking-on/off prompt-tail routing and mixed-shape RoPE decode. Active Mistral3/Ministral3 tests pin no-reasoning fallback so literal `<think>` text is not hidden by a fake parser. The live Gemma 4 no-leak smoke did not elicit reasoning deltas, so long-budget Gemma 4 thinking remains partial. Full model-family reasoning-effort matrix is not complete. | partial |
| Tool parser matrix by family. | DSV4 and selected templates have focused proof. Gemma 4 has focused proof that Harmony reasoning followed by a Gemma 4 tool-call envelope produces one structured tool call and no visible marker leak. Hy3 now has focused Hunyuan parser proof for multiple scalar-argument calls and a reasoning-then-tool-call stream with no visible marker leak. GLM/DeepSeek/Laguna capability aliases with suffixes (`glm4_moe_lite`, `glm5_air`, `deepseek_v3`, `laguna_*`) now route through the GLM arg_key/arg_value parser instead of returning nil, and `glm5*` model-type fallback also infers the GLM parser so GLM-5.1-style bundles do not require a JANG stamp. The alias-normalization follow-up also pins `gemma_4`/`gemma-4` to Gemma4 tools, `glm_5_1_flash`/`glm-5.1-flash` to GLM tools, and GPT-OSS aliases (`gpt_oss`, `gpt-oss`, `gptoss`) to GLM-style tools while their reasoning remains Harmony. Fresh red/green proof in `docs/local/production-readiness/20260517T1405_parser_fallback_matrix/` adds Ling/Bailing model-type fallback to the same GLM/deepseek arg-key parser and Qwen3.6/Qwen3-VL model-type fallback to XML-function tools. The follow-up `docs/local/production-readiness/20260517T1415_qwen_vl_capability_aliases/` proves direct Qwen-VL capability stamps now route to XML-function tools instead of bypassing the parser. DSV4 capability aliases now resolve before that generic DeepSeek branch: `deepseek_v4`, `deepseek_v4_flash`, and `deepseekv4` route to DSML, while `deepseek` and `deepseek_v3` remain GLM-style. Mistral3/Ministral3, Mistral 4, and Pixtral model-type/capability aliases now route to the Mistral tool parser (`mistral3`, `ministral3`, `mistral4`, `mistral4_large`, `mistral_small_4`, `pixtral*`). Focused factory-dispatch tests now also prove Mistral3 `mxtq` routes to `Mistral3TextJANGTQModel` while `mxfp4`/missing format stay vanilla. Full dsml/deepseek/gemma4/jang/zaya/llama/qwen/mistral live matrix remains open, and Kimi is intentionally outside the current active sweep. | partial |
| Generation config defaults apply. | `docs/local/live-model-matrix/20260516Tguard-removal/Ling-2.6-flash-JANGTQ2-CRACK_prod_bundle_defaults_coord.out` proves the Ling folder has no sampling defaults and resolves through fallback to `temp=0.600 topP=1.000 topK=0 minP=0.000 rep=nil`. `.../MiniMax-M2.7-Small-JANGTQ_prod_bundle_defaults_coord.out` proves MiniMax's `generation_config.json` applies `temp=1.000 topP=0.950 topK=40 rep=nil`. `.../Gemma-4-26B-A4B-it-JANG_4M-CRACK_prod_bundle_defaults_coord.out` proves Gemma 4's folder config applies `temp=1.000 topP=0.950 topK=64 rep=nil`. `docs/local/production-readiness/20260517T165508Z_qwen_mtp_settings_recheck/VMLXServerRuntimeSettingsTests.log` passes 12/12 and pins bundle generation config before server overrides, nil server fields preserving engine/bundle defaults, top-k reaching speculative sampler probabilities, no hidden sampler guards, and invalid sampling values reporting instead of clamping. Package-wide live override matrix remains incomplete. | partial |
| Ling/Bailing release turnmatrix. | `docs/local/live-model-matrix/20260517T170008Z_release_turnmatrix_ling_jangtq2/REPORT.md` passes all runnable rows for `Ling-2.6-flash-JANGTQ2-CRACK`: config/template, MTP metadata, production defaults cache OFF/ON, BatchEngine single/chat/disk restore/concurrent/per-slot/TurboQuant B=2. The larger `Ling-2.6-flash-MXFP4-CRACK` row at `docs/local/live-model-matrix/20260517T180538Z_release_turnmatrix_ling_mxfp4_current/REPORT.md` also passes config/template/MTP metadata, production cache OFF/ON 7/7, disk+SSM cache stats, disk restore, B=2 concurrent, per-slot sampler, and TurboQuant B=2 plain-slot isolation. Both resolve bundle/default sampling to `temp=0.600 topP=1.000 topK=0 minP=0.000 rep=nil`; active focused Bailing/Ling directive tests now pass in `NoHiddenReasoningCloseBiasFocusedTests`. Fresh focused cache proof instantiates `BailingHybridModel` and proves linear layers allocate `ArraysCache`, global MLA layers allocate KV/RotatingKV caches, trailing partial layer groups are global, and disk-backed coordinator restore remains required. MXFP4 is slower at about 9.7-10.1 tok/s and carries a high resident footprint, so speed/footprint should be watched separately. | live-proven for JANGTQ2 and MXFP4 text rows |
| Hy3 JANGTQ/JANGTQ_K release turnmatrix. | `docs/local/live-model-matrix/20260517T180931Z_release_turnmatrix_hy3_jangtq_current/REPORT.md` passes all runnable rows for `Hy3-preview-JANGTQ`: config/template/MTP metadata, production defaults cache OFF/ON 7/7, paged/disk cache stats, disk restore, B=2 concurrent, per-slot sampler, and TurboQuant B=2 isolation. Bundle sampling reaches runtime as `temp=0.900 topP=1.000 topK=-1 minP=0.000 rep=nil`. `Hy3-preview-JANGTQ_K` initially failed/killed under eager load in `docs/local/live-model-matrix/20260517T182455Z_release_turnmatrix_hy3_jangtqk_current/`; the active-streaming probe without a bound model dir failed with `missing active JANGTQ gate/up tensors for layer 1`; explicit model-dir streaming then passed 7/7 at low footprint, and the post-fix no-env proof `docs/local/live-model-matrix/20260517T184132Z_hy3_jangtqk_streaming_autodir_after_fix/` passes 7/7 after `loadWeights` binds the loaded model directory. Active focused Hy3 parser/no-leak tests now pass in `NoHiddenReasoningCloseBiasFocusedTests`; the same focused suite now pins preserved nextn exclusion from base decode cache, normal q/k/v sanitizer fusion while dropping nextn tensors, and mixed-bit q/k/v dequantize-then-fuse fallback instead of process abort. `RuntimeMoETopKOverrideFocusedTests` also pins explicit top-k override as lower-only and cache-key-scoped, not a hidden runtime mutation. | live-proven for JANGTQ; JANGTQ_K correctness/low-footprint partial, speed blocked |
| Gemma 4 text turnmatrix. | `docs/local/live-model-matrix/20260517T160608Z_release_turnmatrix_gemma4_26b/REPORT.md` passes config/template, `BENCH_PROD` cache OFF/ON 7/7, BatchEngine single/chat, disk restore, B=2 concurrent, B=2 per-slot sampler, and TurboQuant-KV B=2 isolation. Cache ON stats show `pagedIncompatible=true` with zero paged counters by design and disk L2 `hits=1`, `stores=14`, `maxBytes=4294967296`; the generic paged prefix-hit row is correctly N-A. Fresh Harmony parser artifact `docs/local/live-model-matrix/20260517T_harmony_parser_fix_current/` adds focused Gemma 4 + GPT-OSS channel proof and a live Gemma 4 no-marker-leak smoke. Active focused Gemma4 SWA/cache coverage now passes cache-topology tests, including actual `Gemma4TextModel.newCache` mixed/no-maxKV and all-rotating/maxKVSize allocation, plus 4 BatchKVCache rotating-slot mask tests in `CacheCoordinatorTopologyFocusedTests`. Fresh compile-policy proof `docs/local/production-readiness/20260517T1325_gemma4_swa_compile_policy/Gemma4CacheTopologyFocusedTests.log` passes 5/5 and pins the speed/correctness boundary: default heterogeneous SWA/full-attention cache is not compile-eligible, while explicit bounded all-rotating cache is. Active Gemma4 VLM contracts also pin explicit audio rejection, direct `<|image|>` token lookup, `rmsNormNoScale` parity/dtype preservation, and Gemma3/Gemma4 maskedScatter recoverable errors instead of process aborts; this is source/math coverage, not a live VL production row. | live-proven for text path; broader reasoning/tool/VL partial |
| Single-batch and continuous batching. | Omni BatchEngine harness now forces `maxBatchSize=2` for B=1 rows so it exercises the scheduler path instead of the solo fast path. Text B=1, text B=2, image B=1, and audio B=1 pass after the no-thinking media-tail fix in `docs/local/live-model-matrix/20260516Tomni-nonmtp/Nemotron-Omni-Nano-JANGTQ4-CRACK_omni_batch_nothink_tail_fix.out`. Qwen3.5 35B post-fix turnmatrix passes BatchEngine single/chat/disk restore/B=2/per-slot/TurboQuant B=2, with mixed plain/TurboQuant decode split by cache compatibility and preserving the plain-slot output exactly. Ling and MiniMax Small have coordinated `BENCH_PROD` rows with same-prompt TTFT checks. Full per-family B>1 batching remains incomplete. | live-proven for Omni JANGTQ4 and Qwen3.5 35B; package-wide partial |
| TurboQuant/JANGTQ encode/decode and acceleration toggles. | Focused JANGTQ/Hadamard/matmul proof exists. Current live rows now include Ling JANGTQ2/MXFP4 TurboQuant-KV B=2 isolation, Hy3 JANGTQ TurboQuant B=2 isolation, and Hy3 JANGTQ_K active-expert streaming correctness after binding the loaded model directory. JANGTQ_K speed is still blocked at about 1.4 tok/s, and full acceleration toggle coverage remains open. | partial |
| Distributed mode. | Targets exist (`MLXDistributed*`, `TPRankWorker`), but no no-peer distributed clean artifact is recorded for this audit. | open |
| Full `swift test`. | Full package test remains open, but the local `Testing`/`XCTest` import blocker was narrowed to the shell toolchain/framework search path, not source. The passing local invocation is `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcrun swift test ... -Xswiftc -F -Xswiftc /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/Library/Frameworks`. Focused artifacts: `docs/local/live-model-matrix/20260516Tnonmtp-tests/NemotronHOmniPreEncodedAudioTests_expanded_xcode.out`, the MLXPress policy run from the same invocation, and `docs/local/production-readiness/20260517Tswift-mtp-current/mtp_runtime_focused_postaudit.log`, which passes 22/22 `MTPRuntimeFocusedTests`. Fresh current-checkout focused Omni command `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcrun swift test --filter NemotronHOmniPreEncodedAudioTests --jobs 2 -Xswiftc -F -Xswiftc /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/Library/Frameworks` passes 8/8 rows in `docs/local/live-model-matrix/20260517T_omni_current_recheck/NemotronHOmniPreEncodedAudioTests.log`: live audio buffer, pre-encoded Parakeet embedding, video EVS token count, RADIO pixel shuffle, Parakeet relative shift, projector remaps, Parakeet weight transpose, and latency-bench generation-default plumbing. Fresh `MLXLMCommonFocusedTests` passes 168/168 Swift Testing rows plus the selected XCTest rows under `docs/local/production-readiness/20260517T1300_hy3_mixed_qkv_runtime_contracts/MLXLMCommonFocusedTests_after_hy3_mixed_qkv.log`. Fresh server-settings profile validation passes 15/15 under `docs/local/production-readiness/20260517T1305_mtp_settings_profile_validation/VMLXServerRuntimeSettingsTests_profile_validation.log`; the later server-settings validation run passes 16/16 under `docs/local/production-readiness/20260517T1335_server_settings_validation/VMLXServerRuntimeSettingsTests.log`. Fresh active cache topology passes 24/24 under `docs/local/production-readiness/20260517T1348_cache_policy_salt_active/CacheCoordinatorTopologyFocusedTests.log`, including the cache-policy salt rows that were previously only present in inactive `Tests/MLXLMTests`. `Tests/MLXLMTests/NemotronHOmniSmokeTests.swift` is currently not wired into an active package test target, so the earlier filtered row that ran 0 tests is not counted. | partial |
| Release build. | `swift build -c release --product RunBench --jobs 2` passed after the non-MTP Omni tail fix; artifact: `docs/local/live-model-matrix/20260516Tnonmtp-tests/RunBench_release_build.out`. Fresh current-checkout release build also passed before the Hy3 JANGTQ_K post-fix proof, using `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcrun swift build -c release --product RunBench`. | live-proven |
| Osaurus single-package repin. | Not done. Osaurus still pins split runtime stack. | open |

## Current Model Matrix Snapshot

Local raw matrix: `docs/local/live-model-matrix/20260515Tinfer-under20/`.

Notable pass rows:

- Laguna XS.2 JANGTQ config/template/prod defaults.
- ZAYA text JANGTQ4, JANGTQ_K, and MXFP4 config/template/prod defaults.
- ZAYA-VL config/template and VL batch chat rows.
- Qwen3.5 35B A3B config/template/prod defaults/VL batch chat.
- Gemma4 JANG_4M config/template/prod defaults.
- Nemotron Omni JANGTQ/JANGTQ4 config/template/prod defaults/Omni rows.
- Qwen3.6 27B JANG_4M CRACK and JANG_4M MTP config/template/prod defaults/VL batch chat.
- Qwen3.6 27B MXFP4 CRACK config/template/prod defaults/VL batch chat.
- Qwen3.6 35B A3B JANGTQ config/template/prod defaults/VL batch chat.

Known failing rows from that snapshot:

- `ZAYA1-VL-8B-JANGTQ4_.infer_prod_defaults_cache_off` -> `fail:133`
- `ZAYA1-VL-8B-JANGTQ_K_.infer_prod_defaults_cache_off` -> `fail:133`
- `ZAYA1-VL-8B-MXFP4_.infer_prod_defaults_cache_off` -> `fail:133`
- `Qwen3.6-27B-MXFP4-MTP_.infer_prod_defaults_cache_off` -> `fail:133`
- `Qwen3.6-27B-MXFP4-MTP_.infer_vl_batch_chat` -> `fail:133`

2026-05-16 non-MTP ZAYA1-VL follow-up:

- Root cause for JANGTQ4/MXFP4 text failures was not sampling and not a model
  quality issue: the loader shim manufactured a Qwen-style `<think>` assistant
  prefill even though ZAYA1-VL bundles stamp `think_in_template=false` and their
  shipped VL sidecar generation prompt is plain `<|im_start|>assistant\n`.
- Fix scope: ZAYA1-VL metadata shim still materializes vision placeholders and
  Zyphra XML tools, but no longer emits `<think>` or `enable_thinking` rails.
  The production harness now probes the actual rendered prompt before requiring
  `.reasoning` deltas; non-toggleable templates must still produce visible,
  coherent output with no marker leak.
- 2026-05-16 shape/template audit found and fixed a second ZAYA1-VL template
  leak: the sidecar-free metadata shim's fallback did not prefill thinking, but
  still rendered historical assistant `reasoning_content` as `<think>...</think>`.
  That is not allowed when `think_in_template=false`. The fallback now omits the
  reasoning-content block entirely. Pre-fix failing artifact:
  `docs/local/live-model-matrix/20260516Tshape-template-audit/focused_shape_template_tests.out`.
  Post-fix proof:
  `docs/local/live-model-matrix/20260516Tshape-template-audit/focused_shape_template_tests_after_zaya_fix.out`.
- `ZAYA1-VL-8B-JANGTQ4` now passes `BENCH_TEMPLATE_SMOKE=1`,
  `BENCH_PROD=1`, and `BENCH_VL_CHAT_CACHE=1`:
  `docs/local/live-model-matrix/20260516Tzaya-vl-think-template-fix/ZAYA1-VL-8B-JANGTQ4_template_smoke.out`,
  `.../ZAYA1-VL-8B-JANGTQ4_prod_postharness.out`, and
  `.../ZAYA1-VL-8B-JANGTQ4_vl_chat_cache.out`.
- `ZAYA1-VL-8B-MXFP4` now passes `BENCH_PROD=1` at
  `docs/local/live-model-matrix/20260516Tzaya-vl-think-template-fix/ZAYA1-VL-8B-MXFP4_prod.out`.
- `ZAYA1-VL-8B-JANGTQ_K` remains blocked: bundle-default sampling and explicit
  greedy both fail the same `7+8-11` row (`7`/`8` stochastic, `6` greedy) while
  other short rows pass. Artifacts:
  `docs/local/live-model-matrix/20260516Tzaya-vl-think-template-fix/ZAYA1-VL-8B-JANGTQ_K_prod.out`
  and `.../ZAYA1-VL-8B-JANGTQ_K_prod_greedy.out`. Do not call this variant
  production-ready until the quant/runtime cause is found.
- Follow-up on `ZAYA1-VL-8B-JANGTQ_K` narrows the failure away from cache,
  streaming, parser, EOS, and sampler policy. The real prompt/logit probe at
  `docs/local/live-model-matrix/20260516Tzaya-vl-jangtqk-debug/ZAYA1-VL-8B-JANGTQ_K_topk_math.out`
  uses the same rendered prompt and token IDs as JANGTQ4/MXFP4, but the K
  bundle ranks `6`, `7`, `8`, then the correct `4` on the first assistant
  token. JANGTQ4 and MXFP4 rank `4` first in
  `.../ZAYA1-VL-8B-JANGTQ4_topk_math.out` and
  `.../ZAYA1-VL-8B-MXFP4_topk_math.out`.
- Runtime metadata is not the remaining K bug: the diagnostic at
  `.../ZAYA1-VL-8B-JANGTQ_K_moe_bits_v2.out` loads the real model and reflects
  all 40 `TurboQuantSwitchGLU` modules as `gateUp=2`, `down=4`, `seed=42`.
  The `BENCH_ZAYA_CONTRACT` gate was corrected to understand ZAYA1-VL's
  40-layer CCA+MoE topology and vision-LoRA `local_experts`; K, JANGTQ4, and
  MXFP4 now pass contract at `.../*_contract_v2.out`.
- The remaining K blocker is also not explained by the Swift Metal mixed-bit
  JANGTQ kernels. `BENCH_ZAYA_TQ_KERNEL_PROBE=1` loads actual
  `ZAYA1-VL-8B-JANGTQ_K` layer tensors plus the real sidecar and compares
  gate/up 2-bit, fused SwiGLU, and down 4-bit kernels against a CPU dequant
  reference. Single-layer proof:
  `docs/local/live-model-matrix/20260516Tzaya-vl-jangtqk-debug/ZAYA1-VL-8B-JANGTQ_K_tq_kernel_probe.out`.
  Cross-layer proof:
  `.../ZAYA1-VL-8B-JANGTQ_K_tq_kernel_probe_layers.out` covers layers
  `0,1,10,20,39` and experts `0,7,15`; all max diffs are about `1e-5`.
  Treat the math-row failure as a real K artifact/runtime-quality blocker until
  a packer-side dequant comparison or a rebuilt K bundle proves otherwise. Do
  not add a sampler/template guard for it.

2026-05-16 non-MTP DSV4 follow-up:

- Config/template gates pass for the local DSV4 Flash JANGTQ-K and JANGTQ2
  bundles:
  `docs/local/live-model-matrix/20260516Tdsv4-nonmtp/DeepSeek-V4-Flash-JANGTQ-K_config_smoke.out`,
  `.../DeepSeek-V4-Flash-JANGTQ2_config_smoke.out`,
  `.../DeepSeek-V4-Flash-JANGTQ-K_template_kwargs.out`, and
  `.../DeepSeek-V4-Flash-JANGTQ2_template_kwargs.out`.
- JANGTQ-K three-turn chat is coherent with thinking disabled and recalls the
  injected `sapphire-42` token:
  `.../DeepSeek-V4-Flash-JANGTQ-K_coherence_chat.out`.
- Reasoning off/on/max with explicit `repetition_penalty=1.0` answers `12`,
  does not need a hidden sampler floor, and does not leak raw reasoning tags:
  `.../DeepSeek-V4-Flash-JANGTQ-K_coherence_reasoning_rep1.out`.
- A 5.5k-token semantic recall row returns `CERULEAN RIVER and OSLO`, but it is
  slow (`~0.07 tok/s`) and is not a substitute for the full DSV4 B7
  long-context/vector drift gate:
  `.../DeepSeek-V4-Flash-JANGTQ-K_coherence_long.out`.
- Current DSV4 cache-topology proof:
  `.../DeepSeek-V4-Flash-JANGTQ-K_growing_chat_cache_current.out` and
  `.err`. It reports `pagedIncompatible=true`, salted prompt/post-answer disk
  hits with `diskArrays=yes`, nil-salt misses, and a coherent turn-2 recall.
  Prompt prefill time drops from `15.423s` to `0.307s`.
- Current DSV4 long-context blocker:
  `.../DeepSeek-V4-Flash-JANGTQ-K_coherence_long_16k_current.out` prepares a
  `16318` token prompt, then `.err` fails with Metal
  `kIOGPUCommandBufferCallbackErrorOutOfMemory`. Do not claim DSV4 long-context
  production readiness until the memory path is fixed and rerun.
- The Python ds4 official vector fixture expected by
  `tests/cross_matrix/dsv4_vector_probe.py` was not present at
  `/tmp/ds4-read/tests/test-vectors/official.vec` on this machine, so the
  vector-drift row remains explicitly blocked rather than inferred.

2026-05-16 non-MTP Ling/MiniMax guard-removal follow-up:

- A new diagnostic harness, `BENCH_NO_GUARD_SAMPLING=1`, was added to
  `RunBench/Bench.swift`. It is test-only and does not modify production
  defaults. It prints the rendered prompt tail and effective request sampling
  values, then fails on empty visible output, repeated-output loops, repeated
  BOS tokens, visible reasoning-marker leaks, or unclosed reasoning when
  thinking is explicitly enabled.
- Ling/Bailing explicit no-guard evidence:
  `docs/local/live-model-matrix/20260516Tguard-removal/Ling-2.6-flash-JANGTQ2-CRACK_no_guard_sampling.out`.
  The three live calls use `rep=nil` for greedy "say hi", explicit `rep=1.000`
  for the temp-0.6 star story, and `temp=0.700 topP=1.000 topK=0 minP=0.000
  rep=nil` for the Russian Three.js stress prompt. All stop normally with no
  loops, no BOS repetition, no raw reasoning marker leaks, and measured decode
  around `35-37 tok/s`.
- Ling caveat: the Russian Three.js stress row is on-task and non-looping, but
  it contains one Chinese token (`点位`) inside an otherwise Russian response.
  That is recorded as a quality caveat, not hidden by a top-p/repetition
  fallback. The bundle's `generation_config.json` has no sampling defaults, so
  the resolved production fallback is genuinely `temp=0.600 topP=1.000 topK=0
  minP=0.000 rep=nil`.
- Ling coordinated bundle-default evidence:
  `docs/local/live-model-matrix/20260516Tguard-removal/Ling-2.6-flash-JANGTQ2-CRACK_prod_bundle_defaults_coord.out`
  passes 7/7 with an explicit cache coordinator and the same resolved fallback
  values. Same-prompt TTFT drops from `284ms` to `193ms`.
- MiniMax Small explicit no-guard evidence:
  `docs/local/live-model-matrix/20260516Tguard-removal/MiniMax-M2.7-Small-JANGTQ_no_guard_sampling.out`.
  Greedy/no-rep "say hi" returns visible `Hi!`; the temp-0.6, explicit
  `rep=1.000`, thinking-on story returns visible content about a star, `.reasoning`
  is separated, no markers leak, and decode is about `46-47 tok/s`.
- MiniMax Small coordinated bundle-default evidence:
  `docs/local/live-model-matrix/20260516Tguard-removal/MiniMax-M2.7-Small-JANGTQ_prod_bundle_defaults_coord.out`
  passes 7/7 and proves its folder `generation_config.json` values apply:
  `temp=1.000 topP=0.950 topK=40 minP=0.000 rep=nil`. Same-prompt TTFT drops
  from `385ms` to `67ms`, and reasoning ON/OFF alternation is correct.

2026-05-16 non-MTP Gemma 4 follow-up:

- Gemma 4 coordinated bundle-default evidence:
  `docs/local/live-model-matrix/20260516Tguard-removal/Gemma-4-26B-A4B-it-JANG_4M-CRACK_prod_bundle_defaults_coord.out`
  passes 7/7 with an explicit cache coordinator. The resolved sampling line
  proves the folder `generation_config.json` is used:
  `temp=1.000 topP=0.950 topK=64 minP=0.000 rep=nil`. Same-prompt TTFT drops
  from `458ms` to `63ms`; Harmony reasoning is separated from visible content,
  and thinking-off rows emit zero reasoning.
- Gemma 4 explicit no-guard evidence:
  `docs/local/live-model-matrix/20260516Tguard-removal/Gemma-4-26B-A4B-it-JANG_4M-CRACK_no_guard_sampling_768_unclosed_loopheuristic.out`
  and `.err`. Greedy/no-rep "say hi" is coherent. The thinking-on star-story
  stress row fails honestly: at `maxTokens=768`, `temp=0.600 topP=1.000 topK=0
  minP=0.000 rep=1.000`, it emits reasoning only, hits length, and produces no
  visible answer. This is not hidden by a forced close or repetition floor.
- Gemma 4 thinking-off control evidence:
  `docs/local/live-model-matrix/20260516Tguard-removal/Gemma-4-26B-A4B-it-JANG_4M-CRACK_story_bundle_defaults_think_off_probe.out`.
  The same story prompt with `enable_thinking=false` uses the template's closed
  empty Harmony thought channel, returns a coherent visible two-sentence story,
  stops normally after `31` tokens, and reports `unclosedReasoning=false`.
  This isolates the failure to the open thinking-channel path rather than base
  decode, tokenizer, or generation_config merge.
- The failure exposed a terminal-info bug in the shared solo generation loop:
  public `BatchEngine.generate`/solo-fast-path info was snapshotting after
  parser flush and could report `unclosedReasoning=NO` even when the parser
  was still inside Harmony reasoning. `Libraries/MLXLMCommon/Evaluate.swift`
  now snapshots `handler.unclosedReasoning` before `handler.onGenerationEnd`.
  Focused proof:
  `docs/local/live-model-matrix/20260516Tguard-removal/NoHiddenReasoningCloseBiasFocusedTests_unclosed_info.out`.
- The diagnostic loop heuristic now also catches long repeated non-whitespace
  scalar runs, so numeric/token-salad tails are not missed merely because they
  are not repeated word spans.

2026-05-16 non-MTP Qwen3.6 cache/template-boundary follow-up:

- Scope: `/Users/eric/models/dealign.ai/Qwen3.6-27B-JANG_4M-CRACK` and
  `/Users/eric/models/dealign.ai/Qwen3.6-27B-MXFP4-CRACK` are non-MTP rows.
  This evidence does not claim MTP activation; MTP remains gated by real
  `mtp.*` tensor/config evidence and the explicit native-MTP path.
- Pre-fix diagnostic:
  `docs/local/live-model-matrix/20260516Tguard-removal/Qwen3.6-27B-JANG_4M-CRACK_growing_chat_cache_diagnostic.out`
  and `.err`. Turn 1 stored salted prompt/post-answer disk entries, but the
  real turn-2 full-history prompt diverged before the raw generation-prompt
  boundary: `prompt=1863/1867 postAnswer=1863/1873`. The stored turn-1 prompt
  ended in the active-generation no-thinking rail
  `<|im_start|>assistant\n<think>\n\n</think>\n\n`; the next request rendered
  the same assistant message as history with visible content instead. Reusing
  raw turn-1 KV under the turn-2 token key would have been unsafe.
- Fix: tokenizers can now expose the same chat template with
  `addGenerationPrompt=false`. `LLMUserInputProcessor` records a
  `cachePrefixTokenCounts` boundary only when that no-generation render is a
  strict prefix of the actual generation prompt. `BatchEngine` and
  `TokenIterator` store a real cache snapshot for that boundary by trimming
  compatible prompt snapshots or by correctness-first re-deriving the boundary
  with the original media fields, cache salt, and cache parameters. Hybrid-pool
  re-derive uses the existing DSV4 full-window prefill rule.
- Final live proof:
  `docs/local/live-model-matrix/20260516Tguard-removal/Qwen3.6-27B-JANG_4M-CRACK_growing_chat_cache_history_boundary_final.out`
  and `.err`. The row records `Cache history-boundary counts: [1860]`; turn 2
  probes `HIT tier=disk matched=1860/1897 remaining=37 diskArrays=yes`, answers
  `qwen36-cache-green` coherently, stops normally, and drops prompt prefill from
  `2.360s` to `0.157s` (`ratio=0.07`). Nil-salt probes miss, so the hit is not
  a salt collision.
- Paired MXFP4 proof:
  `docs/local/live-model-matrix/20260516Tguard-removal/Qwen3.6-27B-MXFP4-CRACK_growing_chat_cache_probe.out`
  and `.err`. MXFP4 records `Cache history-boundary counts: [1863]`; turn 2
  probes `HIT tier=disk matched=1863/1903 remaining=40 diskArrays=yes`, answers
  `qwen36-mxfp-cache-green`, stops normally, and drops prompt prefill from
  `2.329s` to `0.165s` (`ratio=0.07`). This proves the fix is not a
  JANG_4M-only artifact.
- Focused proof:
  `docs/local/live-model-matrix/20260516Tguard-removal/NoHiddenReasoningCloseBiasFocusedTests_history_boundary_final.out`
  passes 3/3 source guards for no hidden reasoning close bias, terminal
  unclosed-reasoning telemetry, and the real history-boundary cache mechanism.
  Release build proof:
  `docs/local/live-model-matrix/20260516Tguard-removal/RunBench_release_build_history_boundary_final.out`.
- Osaurus implication: stateless full-history chat requests need safe
  history-boundary cache entries for templates that add assistant
  generation-control rails only on the active turn. This fix stores real KV/SSM
  state for the prefix the next request actually contains; it is not a forced
  hit, repetition guard, template monkeypatch, or fake cache reuse.
- Qwen3.6 VL cache proof:
  `docs/local/live-model-matrix/20260516Tguard-removal/Qwen3.6-27B-JANG_4M-CRACK_vl_chat_cache_final.out`
  plus
  `docs/local/live-model-matrix/20260516Tguard-removal/Qwen3.6-27B-MXFP4-CRACK_vl_chat_cache_probe.out`.
  Both models load with `Qwen3VLProcessor`; the structured chat rows attach
  real generated gradient images, get grounded cold answers, prove
  same-media disk restore `HIT disk 84/84`, proves a changed image misses, and
  answers the text-only follow-up coherently with `Red and blue.` This covers
  Qwen VL chat-template/media-salt/MRoPE path at the real engine level for both
  non-MTP 27B artifacts.
- Still open: broader VL/media cache rows still need separate video/audio
  proofs where supported, plus per-family MRoPE 2D/3D vector and
  Hadamard/JANGTQ matmul coverage beyond this Qwen3.6 JANG_4M row.

2026-05-16 non-MTP Nemotron Omni follow-up:

- The strict pre-fix artifact
  `docs/local/live-model-matrix/20260516Tomni-nonmtp/Nemotron-Omni-Nano-JANGTQ4-CRACK_omni_strict.out`
  failed `5b. video LMInput end-to-end` with a repeated filler loop. That was a
  real runtime bug, not model quality.
- The fix carries the processor's post-EVS video token count through
  `LMInput.ProcessedVideo`, then applies RADIO video embedding, pixel shuffle,
  MLP projection, and EVS before LM prompt splice. The prompt placeholder count
  now matches the post-EVS embedding count, so video placeholders and embeddings
  stay one-to-one even if the EVS keep count changes.
- Post-fix TokenIterator evidence:
  `docs/local/live-model-matrix/20260516Tomni-nonmtp/Nemotron-Omni-Nano-JANGTQ4-CRACK_omni_evs_v2.out`
  passes 13/13 text, image, video, audio, reasoning toggle, mixed media,
  media-salt isolation, and hybrid SSM warm-pass rows.
- BatchEngine evidence:
  `docs/local/live-model-matrix/20260516Tomni-nonmtp/Nemotron-Omni-Nano-JANGTQ4-CRACK_omni_batch_forced_evs_v3.out`
  passes text B=1, text B=2, and audio B=1 through a forced scheduler path, but
  fails image B=1 because the model says the image is blank or missing when
  `enable_thinking=false`. This is now correctly caught as a failure instead of
  counted as a false pass. Do not hide it by forcing reasoning on.
- The direct tail probe
  `docs/local/live-model-matrix/20260516Tomni-nonmtp/Nemotron-Omni-Nano-JANGTQ4-CRACK_omni_tail_probe.out`
  proves the actual media no-thinking contract: compact `<think></think>` and
  `<think></think>\n` both fail with the missing-image denial, while the closed
  newline-delimited tail `<think>\n</think>\n\n` grounds the same prepared image
  tensor. This is a prompt-rendering correction, not a hidden reasoning override.
- Post-tail-fix BatchEngine evidence:
  `docs/local/live-model-matrix/20260516Tomni-nonmtp/Nemotron-Omni-Nano-JANGTQ4-CRACK_omni_batch_nothink_tail_fix.out`
  passes 18/18. The direct `3b. image reasoning OFF direct` row returns a
  grounded gradient description at `96.6 tok/s`; `B3. BatchEngine image B=1`
  returns the same grounded class of answer at `47.8 tok/s`.

Skipped rows because of the 20GB cutoff include DSV4 Flash, Hy3, Kimi,
MiniMax, Ling, and other large bundles. They are not production passes.

## Current MTP Boundary

Live-proven:

- MTP activation is explicit and per-load through `LoadConfiguration.nativeMTP`;
  `VMLINUX_NATIVE_MTP=1` remains a compatibility override for direct factory
  callers only.
- Supported Qwen model types must expose real MTP tensor evidence.
- CRACK config metadata without tensors fails closed.
- Qwen3.6 MTP proof targets are
  `/Users/eric/models/JANGQ/Qwen3.6-27B-JANG_4M-MTP` and
  `/Users/eric/models/JANGQ/Qwen3.6-27B-MXFP4-MTP`.
  The 35B MoE/VL follow-up target copied from `erics-m5-max2.local` is
  `/Users/eric/models/JANGQ/Qwen3.6-35B-A3B-MXFP4-MTP`.
- Qwen3.6 JANG_4M-MTP and MXFP4-MTP D1 smokes are coherent with telemetry.
- Qwen3.6 D3 prefix-commit smokes are coherent on both proof targets:
  `docs/local/native-mtp-qwen36-20260516-d3-prefix-commit/jang4m-mtp-d3-prefix-commit-no-checkpoint-256.log`
  and
  `docs/local/native-mtp-qwen36-20260516-d3-prefix-commit/mxfp4-mtp-d3-prefix-commit-no-checkpoint-256.log`.
  Both rows report `rollbackRepair=0`.
- MTP phase telemetry now reports target verify, MTP draft, sampling, and cache
  commit time. The 2026-05-16 phase rows show target verify dominates, so the
  next speed pass must focus on compiled/tuned small-M verifier execution.
- `BatchEngine.generate` now honors `DraftStrategy.nativeMTP(depth:)` through
  the exclusive solo native-MTP lane instead of silently falling through to
  ordinary AR batching. Focused test proof:
  `MTPRuntimeFocusedTests` passes 17/17, including active native-MTP dispatch,
  missing-head fail-closed dispatch, private MTP-cache refresh on partial
  reject, and `BatchEngine.submit` rejection for raw batched native-MTP. Real
  bundle proof:
  `docs/local/live-model-matrix/20260516Tbatch-mtp-dispatch/Qwen3.6-27B-JANG_4M-MTP_batch_native_mtp_d3.out`
  and `.err` show path=`batch`, coherent text, `33.3 tok/s`, `loop=NO`,
  `leaks=none`, and `[NativeMTP] depth=3 ... prefixCommit=13`.
- VL/media and JANGTQ tensor-shape guard coverage was rerun after this dispatch
  fix: `VLShapeGuardFocusedTests`, `MediaCachePlaceholderTests`, and
  `JANGTQHadamardShuffleTests` passed 19/19. This covers finite 2D extent
  validation, text-only media-cache suffix policy, 2D/3D JANGTQ matmul inputs,
  3D/4D Hadamard shape preservation, and TurboQuant-KV Hadamard rank-four
  shape preservation.
- Qwen3.6 VLM native-MTP verifier now routes through the same MRoPE
  continuation-state resolver as normal Qwen3VL decode instead of using raw
  text-only cache offsets after media prefill. This is a real position-ID fix,
  not a sampler/template guard. Focused artifact:
  `docs/local/live-model-matrix/20260516Tqwen35-vlm-mrope-mtp/mlxlmcommon_focused_after_qwen35_vlm_moe_sidecar.out`
  passes 81/81 tests across 13 suites, including the new
  `Qwen3.6 VLM native MTP verifier reuses MRoPE continuation state` and
  `Qwen3.6 VLM native MTP decoder uses sparse MoE for MoE sidecars` rows plus
  Hadamard/matmul/media-salt/cache-topology/no-hidden-guard rows. No-load
  tensor-key evidence for the real proof targets is in
  `docs/local/live-model-matrix/20260516Tqwen35-vlm-mrope-mtp/qwen36_mtp_vl_tensor_census.json`:
  JANG_4M-MTP has 31 `mtp.*` entries and 333 `vision_tower.*` entries;
  27B MXFP4-MTP has 23 MTP entries and 333 vision entries; 35B A3B MXFP4-MTP
  has 42 MTP entries, 333 vision entries, `model_type=qwen3_5_moe`, and
  `text_model_type=qwen3_5_moe_text`. The optional real-bundle inspector rows
  also pass for all three paths in the same artifact directory.
- Qwen3.6 35B A3B MXFP4-MTP was transferred from
  `erics-m5-max2.local:/Volumes/eric/models/JANGQ/Qwen3.6-35B-A3B-MXFP4-MTP`
  to `/Users/eric/models/JANGQ/Qwen3.6-35B-A3B-MXFP4-MTP`. A dry-run
  `rsync --delete --itemize-changes` reported no differences after transfer.
  The artifact has 37 local regular files, 22G on disk, and
  `runtime.total_weight_bytes=23115460648`.
- Native MTP activation now recognizes `qwen3_5_moe` and
  `qwen3_5_moe_text`, but still requires `VMLINUX_NATIVE_MTP=1` and real MTP
  tensor evidence. The VLM native-MTP decoder now instantiates
  `SparseMoeBlock` for MoE MTP sidecars instead of a dense MLP, so the 35B
  `mtp.layers.0.mlp.switch_mlp.*` / `shared_expert.*` layout matches the Swift
  module tree. Focused artifact:
  `docs/local/live-model-matrix/20260516Tqwen35-vlm-mrope-mtp/mlxlmcommon_focused_after_qwen35_vlm_moe_sidecar.out`
  passes 81/81 tests across 13 suites. The 35B row is not a live load/generate
  pass yet; it is a tensor-evidence, activation, and module-layout readiness
  pass before the heavy live smoke.
- Qwen3.6 text hybrid-SSM partial-prefix commit was aligned with the VLM path
  on 2026-05-17. The text `Qwen35GatedDeltaNet` now advances
  `MambaCache.offset` during regular forwards and records D3 verifier
  accepted-prefix snapshots at `baseOffset + prefixLength`. That prevents a
  partial MTP rejection from restoring recurrent state with the pre-verify SSM
  offset. This is cache-boundary correctness, not a sampler/top-k/repetition
  guard. Focused proof:
  `docs/local/production-readiness/20260517Tqwen36-mtp-ssm-offset/mtp_runtime_focused_after_private_mtp_refresh.log`
  passes 17/17 `MTPRuntimeFocusedTests`, including the new text-SSM offset row.
  The private MTP draft cache also now refreshes on partial reject instead of
  trimming the old cache; telemetry reports this as `mtpCacheRefresh`. That
  prevents stale rejected draft KV/state from surviving into the next draft
  round.
  Matching live JANG_4M text rows on the same prompt are recorded at
  `docs/local/production-readiness/20260517Tqwen36-mtp-ssm-offset/qwen36_27b_jang4m_ar_text_live.log`
  and
  `docs/local/production-readiness/20260517Tqwen36-mtp-ssm-offset/qwen36_27b_jang4m_mtp_d3_private_cache_refresh_live.log`.
  Both are coherent with no loops or marker leaks; AR is `20.0 tok/s` and D3
  native MTP is `18.0 tok/s` with `prefixCommit=50`, `rollbackRepair=0`,
  `mtpCacheRefresh=46`, and `avgCommittedPerVerify=1.92`. This confirms the
  cache-boundary fixes did not create a coherency regression, but it also
  confirms this Swift path is still below the 45 tok/s production MTP threshold.
- Qwen3.6 MXFP8 norm handling now uses metadata when present instead of
  relying on value/conv-layout heuristics. `loadWeights` passes
  `norm_convention` from safetensor metadata or `jang_config.json` into model
  sanitize. Qwen3.5/Qwen3.6 text and VLM sanitizers shift base and MTP norms
  only for explicit `qwen3_5_language_mlx_plus_one`, while an explicit
  non-plus-one convention prevents conv1d layout sanitization from also
  shifting norms. Focused proof:
  `docs/local/production-readiness/20260517Tqwen36-mtp-ssm-offset/mtp_runtime_focused_after_norm_convention_v2.log`
  passes 19/19 `MTPRuntimeFocusedTests`; broader proof:
  `docs/local/production-readiness/20260517Tqwen36-mtp-ssm-offset/mlxlmcommon_focused_after_norm_convention.log`
  passes 85/85. This is a metadata-contract fix for the Python MXFP8 incident,
  not a model-behavior guard.
- The broader active focused target was rerun after the ZAYA template fix:
  `docs/local/live-model-matrix/20260516Tshape-template-audit/mlxlmcommon_focused_target_after_zaya_fix.out`
  passes 78/78 tests across 13 suites. Covered surfaces include CacheCoordinator
  topology, DSV4 paged-incompatible CSA/HSA disk restore, hybrid SSM companion
  requirements, ZAYA CCA disk payloads, media-salt isolation, DSV4 reasoning
  policy pass-through/no hidden max downgrade, VMLX server sampling settings,
  Qwen/VL finite extent guards, Omni Parakeet/RADIO/EVS shape rows, native-MTP
  fail-closed dispatch, JANGTQ active expert descriptors, and JANGTQ
  Hadamard/matmul rank handling.
- `Tests/MLXLMTests` is still not an active package test target in the current
  `Package.swift`; a direct filter against those names ran zero tests at
  `docs/local/live-model-matrix/20260516Tshape-template-audit/mlxlmtests_shape_template_tests.out`.
  Do not count those source files as release evidence until they are wired,
  migrated into `MLXLMCommonFocusedTests`, or intentionally deleted.

Not yet complete:

- Production-speed depth-3 acceleration at or above the current 45 tok/s
  threshold.
- Compiled/tuned small-M verifier.
- True multi-slot native-MTP scheduling with paged KV/block-L2/SSM companion
  cache. Current `BatchEngine.generate` native-MTP support is an exclusive solo
  lane; `BatchEngine.submit` intentionally fails closed.
- MTP with VL multi-turn/media salt/cache proof.
- 35B A3B MXFP4-MTP live load/generate/coherency/token-s proof.
- Auto-launch eligibility for Osaurus.

## Immediate Next Gates

1. Continue diagnosing the remaining `fail:133` rows without MTP scope:
   `ZAYA1-VL-8B-JANGTQ_K` is still a real coherence failure on one math row,
   but the Swift mixed-bit JANGTQ kernel path now has actual-tensor CPU parity
   proof. The next useful check is packer-side/reference dequant comparison or a
   rebuilt K artifact, not a runtime sampler workaround. The JANGTQ4/MXFP4
   ZAYA1-VL text-template failures are fixed and live-proven.
2. Finish migrating any still-useful lightweight rows from
   `Tests/MLXLMTests/NemotronHOmniSmokeTests.swift` into an active test target.
   The critical EVS/remap/shape rows are now covered by
   `NemotronHOmniPreEncodedAudioTests`; the remaining unwired file should either
   be deleted, wired, or reduced to non-duplicated coverage.
3. Extend the Omni-style no-thinking media-tail gate to any other local
   multimodal reasoning model that shows the same compact-tail failure. Do not
   assume it globally; prove it per family with a direct media tensor row.
4. The current focused target passes 160 tests in 24 suites in
   `docs/local/production-readiness/20260517T_laguna_mistral_gemma4_active_contracts/MLXLMCommonFocusedTests_all.log`,
   including parser/no-leak, Laguna template/RoPE, Mistral3/Ministral3
   no-reasoning, Mistral3 JANGTQ factory dispatch, Gemma4 VLM guard/norm/error
   contracts, cache topology, server settings, MTP runtime,
   JANGTQ/Hadamard/matmul, VL shape/MRoPE, ZAYA config, Omni Parakeet/RADIO, and
   MiniMax resident-expert contracts. A full-package clean-worktree pass still
   waits on the concurrent Flux `Package.swift` work to be committed or isolated.
5. Keep native MTP explicit-only until the speed and cache-composition gates
   close. The current BatchEngine fix prevents false positives; it does not
   make native MTP auto-launch eligible.
6. Fix DSV4 16k+ long-context memory behavior, then rerun B7-style
   long-context and vector drift. Current pushed engine passes the DSV4 disk
   restore/cache topology row but fails the 16k long-context row with Metal OOM;
   the ds4 official vector fixture is also absent locally.
7. Expand the active model matrix beyond the 20GB cutoff for MiniMax, Ling,
   Hy3, Gemma 4, ZAYA/ZAYA-VL, Qwen, and Nemotron one family at a time, with
   process checks before and after each run. DSV4 and Kimi are intentionally
   deferred in the current sweep.

## Mergeability Call

Current state: **not release-ready and not complete**.

The package has useful pushed progress, especially consolidated product surface
work, MTP tensor-gated activation, and many local model smokes. It still has
uncovered or failing requirements in the model matrix, cache topology matrix,
full test graph, Osaurus repin, DSV4 long-context proof, VL text-path failures,
and production MTP D3 acceleration.
