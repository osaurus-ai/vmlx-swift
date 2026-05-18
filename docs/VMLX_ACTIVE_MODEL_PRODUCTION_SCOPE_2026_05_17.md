# vMLX Active Model Production Scope - 2026-05-17

This note records the current active scope after the latest explicit user
direction: exclude Kimi, but include DSV4 Flash and every other local
`~/models` family in the systematic production matrix. It is not a
production-ready claim. It is the checklist for the remaining model families
that still need live multi-turn/cache/VL/Omni proof before Osaurus moves fully
onto `vmlx-swift`.

## Current Exclusions

Do not include these in the active production matrix until the user reopens
them:

- Kimi / Kimi-K2.x

DSV4 Flash is back in scope. Generic `JANGTQ_K` rows are not globally excluded
anymore under the "all models except Kimi" instruction, but old rows that were
marked historical remain historical until re-run under the current non-Kimi
matrix. Do not promote them from old evidence.

The harness supports this directly:

```sh
scripts/vmlx-live-model-matrix.sh \
  --profile inventory \
  --exclude-regex 'Kimi'
```

Fresh inventory artifact:

```text
docs/local/live-model-matrix/20260517T_non_kimi_inventory_dsv4_included/
docs/local/live-model-matrix/20260517T_scope_exclude_kimi_dsv4_inventory/
docs/local/live-model-matrix/20260517T235436Z_non_kimi_inventory_mtp_auto_refresh/
```

No-load MTP census artifact:

```text
docs/local/live-model-matrix/20260517T_scope_exclude_kimi_dsv4_mtp_census/
docs/local/live-model-matrix/20260518T000022Z_non_kimi_mtp_auto_policy_postfix/
```

No-load config/template metadata artifact:

```text
docs/local/live-model-matrix/20260517T_scope_exclude_kimi_dsv4_metadata/
```

Fresh focused MTP/settings artifact:

```text
docs/local/production-readiness/20260517T160343Z_qwen_mtp_settings_current/
docs/local/production-readiness/20260517T165508Z_qwen_mtp_settings_recheck/
docs/local/production-readiness/20260517T1305_mtp_settings_profile_validation/
docs/local/production-readiness/20260517T1252_ling_hy3_gemma4_runtime_contracts/
docs/local/production-readiness/20260517T1300_hy3_mixed_qkv_runtime_contracts/
docs/local/production-readiness/20260517T1315_harmony_prompt_tail_parser/
docs/local/production-readiness/20260517T1320_no_hidden_reasoning_regression/
docs/local/production-readiness/20260517T1325_gemma4_swa_compile_policy/
docs/local/production-readiness/20260517T1335_server_settings_validation/
docs/local/production-readiness/20260517T1348_cache_policy_salt_active/
docs/local/production-readiness/20260517T1405_parser_fallback_matrix/
docs/local/production-readiness/20260517T1415_qwen_vl_capability_aliases/
docs/local/production-readiness/20260517T1435_harmony_fragment_leak/
docs/local/production-readiness/20260517T1450_gemma4_rotating_compile_direct/
docs/local/production-readiness/20260517T1505_ling_bailing_capability_aliases/
```

Fresh live Gemma4 schema/VL artifacts:

```text
docs/local/live-model-matrix/20260517T210417Z_gemma4_vl_chat_cache/
docs/local/live-model-matrix/20260517T212204Z_gemma4_batch_toolcall_real_schema/
docs/local/live-model-matrix/20260517T_reasoning_turn_matrix_harness/
docs/local/live-model-matrix/20260517T_gemma4_e2b_osaurus_loop_report/
```

Fresh live DSV4 artifacts:

```text
docs/local/live-model-matrix/20260517T_dsv4_current_non_kimi_scope/
```

Fresh live Qwen artifacts:

```text
docs/local/live-model-matrix/20260517T_qwen36_27b_mxfp4_crack_current_non_kimi_turnmatrix/
docs/local/live-model-matrix/20260517T_qwen36_27b_mxfp4_crack_video_resize_postfix_turnmatrix/
```

Fresh live ZAYA artifacts:

```text
docs/local/live-model-matrix/20260517T_zaya_jangtq4_current_non_kimi_turnmatrix/
docs/local/live-model-matrix/20260517T_zaya_vl_jangtq4_current_non_kimi_turnmatrix/
```

The current non-Kimi inventory contains 30 local `~/models` bundles:

- 12 text bundles
- 15 VL bundles
- 3 Omni bundles

The older Kimi+DSV4+K-excluded inventory contains 28 local bundles and remains
in this note only as historical context:

- 12 text bundles
- 13 VL bundles
- 3 Omni bundles

Fresh Osaurus PR/pin lineage artifact:

```text
docs/VMLX_OSAURUS_PR_PIN_LINEAGE_2026_05_17.md
```

## Extended Family Status After PR/Pin Review

This section is the current working status for the non-Kimi families the user
explicitly called out. A `PASS` row still does not mean package-wide production
complete; it means the named surface has a current artifact. Rows marked
`PARTIAL` or `OPEN` must not be hidden by sampler guards, repetition penalties,
forced thinking closure, or name-based MTP activation.

| Family | Current evidence | Status | Production concern |
| --- | --- | --- | --- |
| DSV4 Flash | `20260517T_dsv4_current_non_kimi_scope/` contains the pre-fix failed prompt-boundary row, red/green focused separator tests, post-fix JANGTQ2 cache OFF/ON chat rows, a post-fix JANGTQ-K chat row, a DSV4 topology probe, and a JANGTQ2 `BENCH_PROD` row. The root cause was real prompt construction: system text glued directly to `<User>`, producing `sappberry/sappium` drift. The fix adds a system-to-user separator in the standalone DSV4 Jinja, compiled fallback, and Swift encoder. Bundle-default production sampling is visible as `temp=1.000 topP=1.000 topK=0 rep=nil`; no repetition or temperature guard was added. | PARTIAL | Coherent multi-turn and reasoning on/off rows now pass, but this is not low-footprint production-cleared: the JANGTQ2 production row reports about 61.5 GiB peak RSS, DSV4 is `pagedIncompatible=true`, paged prefix hits are `N-A`, and long-context/vector/speed/API gates remain open. Disk L2 stats are recorded in `PROD_CACHE_STATS`; generic paged-cache claims must not be made for this topology. |
| Ling / Bailing hybrid | `20260517T170008Z_release_turnmatrix_ling_jangtq2/` and `20260517T180538Z_release_turnmatrix_ling_mxfp4_current/` pass config/template, production cache off/on, BatchEngine single/chat/disk/B=2/per-slot/TurboQuant rows. Fresh `20260517T2148_nonexcluded_parser_cache_refresh/` rechecks Bailing/Ling aliases and actual hybrid cache topology. | PASS for text JANGTQ2 and MXFP4 | Generic paged prefix hit is `N-A` by topology; disk-backed restore and SSM companion state are the real cache proof. The harness MTP metadata row is not native-MTP activation. |
| Hy3 / Hunyuan | `20260517T180931Z_release_turnmatrix_hy3_jangtq_current/` passes JANGTQ release rows. `20260517T184132Z_hy3_jangtqk_streaming_autodir_after_fix/` remains historical K-only evidence until re-run under the current all-non-Kimi matrix. Fresh `20260517T2148_nonexcluded_parser_cache_refresh/` rechecks Hy3/Hunyuan parser/no-leak, mixed qkv sanitizer, and nextn exclusion. | PASS for JANGTQ; JANGTQ_K needs current re-run before promotion | Preserved nextn layers stay excluded from base decode cache. The K row is not allowed to ride on old evidence if Osaurus exposes it. |
| Gemma 4 / Harmony | `20260517T160608Z_release_turnmatrix_gemma4_26b/` passes text rows. `20260517T210417Z_gemma4_vl_chat_cache/` proves image cache salt and grounded follow-up. `20260517T212204Z_gemma4_batch_toolcall_real_schema/` proves real `UserInput.tools` schema -> structured `get_weather` tool call. Fresh `20260517T2148_nonexcluded_parser_cache_refresh/` rechecks Harmony fragmentation/no-leak plus Gemma4 SWA/full-attention cache compile policy. Fresh `20260517T2150_gemma4_harmony_long_reasoning/` proves long-budget single-turn thinking on/off with 1420 reasoning chars ON and zero reasoning chars OFF. Fresh `20260517T_reasoning_turn_matrix_harness/` proves one loaded BatchEngine multi-turn ON/OFF/ON with prior assistant `reasoningContent` carried forward and effort `low/medium/high/max` closing with visible output on 26B. Fresh `20260517T_gemma4_e2b_osaurus_loop_report/` proves the local Osaurus `gemma-4-e2b-it-4bit` bundle passes cache ON/OFF, BatchEngine chat, TurboQuant B=2 isolation, VL chat/cache, VL batch chat, and the same reasoning turn matrix when the budget is raised; the retained 256-token row fails by length before visible output and is a real product-setting caveat. | PASS for text, VL image cache, tool schema, Harmony reasoning on/off, E2B cache/VL, and multi-turn reasoning matrix | Default heterogeneous SWA/full-attention cache is correctly uncompiled; explicit all-rotating bounded cache is compile-eligible. Do not hide low-budget thinking failures with forced reasoning closure. API route matrix coverage is still package-wide, not a Gemma4 parser leak gap. |
| GPT-OSS / GLM5 / Mistral4 / Pixtral parsers | `20260517T2148_nonexcluded_parser_cache_refresh/` covers GPT-OSS Harmony, GLM5 aliases, Mistral4/Pixtral aliases, and marker leak prevention in the current tree. | UNIT ONLY / OPEN for runtime | No local GPT-OSS, GLM5, Mistral4, or Pixtral bundle has a live decode row in this pass. Parser coverage is not model production readiness. |
| Laguna XS.2 | `20260517T_release_turnmatrix_laguna_xs_after_b2_fix/` passes release rows. Production decode is about `31 tok/s`, bundle defaults are `temp=0.700 topP=0.900 topK=0 rep=nil`, disk L2 hits, and TurboQuant B=2 isolation passes. | PASS for text | Paged prefix hit is `N-A` by topology; disk-backed restore is the accepted cache proof. |
| ZAYA text | Fresh current `20260517T_zaya_jangtq4_current_non_kimi_turnmatrix/` passes JANGTQ4 config/template, production defaults cache OFF/ON, BatchEngine single/chat/disk/B=2/per-slot/TurboQuant rows; generic prefix cache hit is correctly `N-A`. Bundle defaults apply as `temp=0.600 topP=1.000 topK=0 rep=nil`, cache-on speed is about `64.7-66.3 tok/s`, peak RSS is about `5.1 GiB`, and `PROD_CACHE_STATS` records disk L2 plus SSM companion state. Earlier `20260517T_release_turnmatrix_zaya_scope/` also passes JANGTQ4 and MXFP4 text rows; JANGTQ_K evidence remains historical until re-run under the current all-non-Kimi matrix. | PASS for current JANGTQ4 and prior MXFP4; JANGTQ_K needs current re-run before promotion | Keep the 50+ tok/s watch active. Do not treat the ZAYA CCA path as generic paged cache; it is path-dependent and disk/SSM-backed. |
| ZAYA1-VL | Fresh current `20260517T_zaya_vl_jangtq4_current_non_kimi_turnmatrix/` passes JANGTQ4 config/template, production defaults cache OFF/ON, BatchEngine text rows, disk restore, B=2 concurrent/per-slot/TurboQuant rows, VL batch chat, structured VL chat cache, and media-salt isolation. The VL row grounds the image with compile OFF/ON, answers the follow-up color as `blue`, hits disk restore for same-media prompts (`97/97`), and correctly misses when the image changes. Earlier `20260517T_release_turnmatrix_zaya_scope/` and targeted rows also prove MXFP4 image/text/cache surfaces. | PASS for current JANGTQ4 and prior MXFP4 implemented image/text/cache; JANGTQ_K needs current re-run before promotion | The K variant keeps a historical math/top-k diagnostic and cannot be promoted from old evidence. Video is `N-A` because the processor does not implement video input. Do not relabel the disk-backed media-salt cache proof as generic paged-prefix reuse. |
| Nemotron Omni / Parakeet / RADIO | JANGTQ4 core artifacts under `20260517T170614Z_omni_live_voice_fresh_recheck/` and related Omni rows prove wrapper-token parity, pre-encoded Parakeet shape, raw PCM/pre-encoded paths, and 65-76 tok/s cache-off streaming rows without literal sound-marker leaks. Fresh `20260517T214045Z_nemotron_jangtq_omni_recheck/` passes the JANGTQ 48-token Omni matrix 18/18 with text, image, video/audio LMInput, media salt, hybrid SSM, and BatchEngine rows. Fresh `20260517T2215_nemotron_mxfp4_omni_recheck/` passes the MXFP4 48-token Omni matrix 18/18 with the same core surfaces. | PASS for JANGTQ, JANGTQ4, and MXFP4 core; PARTIAL for repeated cache-on audio | Independently encoded Parakeet chunks are not concat-safe. Live voice must retain PCM and submit full-snapshot pre-encoded audio or raw PCM at endpoint. Repeated cache-on live audio still needs a focused quality/root-cause gate before full live-voice production promotion. |
| MiniMax M2.7 | Small JANGTQ rows prove generation config (`temp=1.000 topP=0.950 topK=40 rep=nil`), greedy/no-guard behavior, coherent thinking on/off alternation, and no loops/leaks at 38-47 tok/s depending on row. | PARTIAL | CRACK models are not MTP models unless tensor evidence proves otherwise. Larger JANG/JANGTQ_K CRACK proof still needs real low-footprint active routed evidence, and no hidden sampler guard is allowed to compensate for any failure. |
| Qwen3.6 27B MXFP4 non-MTP | Fresh `20260517T_qwen36_27b_mxfp4_crack_video_resize_postfix_turnmatrix/` passes config/template, bundle-default production cache OFF/ON, BatchEngine text rows, disk restore, B=2 concurrent/per-slot/TurboQuant rows, VL batch chat, structured VL cache, media-salt isolation, and mixed text/image/video. Bundle defaults apply as `temp=1.000 topP=0.950 topK=20 rep=nil`; MTP depth is `off`; reasoning ON/OFF flips work at a 2048-token production budget; same-media image cache hits disk `99/99`; video uses explicit `BENCH_VL_VIDEO_RESIZE=224` and logs `video pixels shape: [560, 1536]`. The preserved pre-fix artifact `20260517T_qwen36_27b_mxfp4_crack_current_non_kimi_turnmatrix/` shows the unbounded 1080p video turn failed after peaking at 164.2 GiB physical footprint in MLX/Metal allocation. | PASS for implemented bounded text/image/video/cache surfaces; high-res video scaling remains PARTIAL | Do not auto-enable MTP on this CRACK bundle. Do not treat raw high-resolution video as production-clear; Osaurus wiring needs an explicit media resize/token budget setting rather than relying on the bundle's very large video preprocessor limit. |
| Qwen3.5 35B non-MTP | `20260517T164940Z_release_turnmatrix_qwen35_35b_4bit_after_compat_split/` passes config/template, production cache off/on, BatchEngine, TQ B=2 compatibility split, VL cache, media salt, and mixed text/image/video. | PASS with throughput watch | High-resolution video can be very slow; Qwen hybrid SSM cache restore must stay topology-specific. |

## No-Fake-Guard Invariants For The Remaining Rows

- `generation_config.json` values are the default source before any explicit
  server/user override. Top-k must remain the bundle value when present
  (`20`, `40`, `64`, or family-specific values), not a global hidden override.
- `rep=nil` or the bundle value must stay visible in telemetry. Do not inject a
  repetition penalty because a row loops; fix the runtime/cache/template cause.
- `enable_thinking=false` can be a template/request input, but the runtime must
  not fake-close a model's reasoning output. Parser rows are allowed to route
  well-formed hidden reasoning away from visible text; they are not allowed to
  make incoherent output look coherent.
- Native MTP is tensor-evidence gated. Metadata rows and model names are not
  sufficient; CRACK or metadata-only bundles stay fail-closed. Supported Qwen
  bundles with real MTP tensors now auto-resolve native D3 through the settings
  bridge.
- Generic paged/prefix cache hits are not accepted for path-dependent families.
  ZAYA CCA, Ling/Bailing linear attention, Qwen hybrid SSM, Gemma4 SWA, Omni
  media, and VL media salts require their family-specific cache proof.

## Native-MTP Speed Rows and Current Regression Watch

These are prompt-specific Qwen3.6 count-prompt rows. They prove the native-MTP
loop and cache behavior for those artifacts only; they do not prove global
production readiness.

Prompt:

```text
Count from 1 to 50 in order, separated by commas.
```

Passing rows produced exact visible `1..50`, stopped normally, and did not use
hidden sampling guards, forced repetition penalties, or forced reasoning closure.
Rows marked historical are useful targets, not current release proof.

| Bundle | Historical best MTP tok/s | Fresh current Swift row | Current decision |
|---|---:|---|---|
| Qwen3.6 27B JANG_4M | 48.9 D2 | not part of the current MXFP-only verifier recheck | Tensor-proven supported Qwen target; auto-resolves native D3, but re-run speed/coherency before release claim. |
| Qwen3.6 27B MXFP4 | 50.5 D3 | 26.1 tok/s D3 `chunk_commit`, exact `1..50`, no diagnostics | Correctness fixed; 45 tok/s target remains open in current Swift. |
| Qwen3.6 27B MXFP8 | 31.7 D2 | not re-run after the prefix-snapshot fix in this artifact | Correctness/speed recheck still required. |
| Qwen3.6 35B JANG_2K | n-a | not in current scope | Excluded from the current MXFP-only MTP focus. |
| Qwen3.6 35B MXFP4 | 171.4 D3 | 84.7 tok/s D3 `chunk_commit`, exact `1..50`, no diagnostics; fresh AR was 91.0 tok/s | Correctness fixed; speed-positive claim remains open for this row. |
| Qwen3.6 35B MXFP8 | 129.9 D3 | not re-run after the prefix-snapshot fix in this artifact | Re-run before release claim. |

The latest sampler sweeps did not justify hidden overrides:

- 27B MXFP8 D3 `top_p=0.95` remained best among `1.00/0.95/0.90/0.85`.
- 27B MXFP8/27B MXFP4 `min_p` did not improve speed or acceptance.
- `top_k=20` from `generation_config.json` was better than forcing `top_k=0`
  on the 27B D3 rows.

## Qwen3.6 MTP Production Reverify

Fresh 2026-05-17 MXFP artifacts:

```text
docs/local/qwen36-mtp-current/20260517T124139Z-27b-mxfp4-prod-budget384/
docs/local/qwen36-mtp-current/20260517T124237Z-27b-mxfp8-prod-budget384/
docs/local/qwen36-mtp-current/20260517T124323Z-35b-mxfp4-prod-budget384/
docs/local/qwen36-mtp-current/20260517T124351Z-35b-mxfp8-prod-budget384/
docs/local/qwen36-mtp-current/20260517T124945Z-35b-mxfp4-vl-mtp-budget384/
```

All four rows pass `BENCH_PROD=1` 7/7 with D3 native MTP,
`VMLINUX_NATIVE_MTP_HYBRID_VERIFY=chunk_commit`, cache coordinator, hybrid SSM
state, and `BENCH_MAX_TOKENS=384`. The gate uses bundle defaults
`temp=1.000 topP=0.950 topK=20 minP=0.000 rep=nil`; there is no hidden
temperature clamp, repetition penalty, or forced reasoning close.

This resolves the earlier short-budget visible-answer failures for the MXFP
variants. The 35B MXFP4 VL+MTP row also passes with the larger budget: cold
red/blue image, same-media disk hit, different-media miss, and text-only
follow-up are coherent. The current default policy is tensor-gated auto-launch
for supported Qwen MXFP/JANG_4M bundles only; native MTP remains non-batched
until the server scheduling gates are proven, and blocked profiles such as
35B JANG_2K stay off.

Current hybrid-SSM verifier policy update: stochastic exact-pq native MTP does
not use the fast chunk verifier unless an explicit verifier env requests it. A
35B MXFP4 growing-chat row failed under bundle defaults when forced through
`chunk_commit`; D1 reproduced it, while sequential repair passed. Post-fix rows under
`docs/local/qwen36-mtp-current/20260517T131050Z-mxfp-growing-chat-mtp-d3-exact-postfix/`
and
`docs/local/qwen36-mtp-current/20260517T131024Z-35b-mxfp4-growing-chat-mtp-d3-exact-postfix/`
prove all four MXFP variants now run bundle-default D3 exact-pq with
`verifierMode=sequential_repair`, coherent two-turn output, disk-prefix hits,
and SSM hits. Greedy rows still use `chunk_commit` where proven.

Fresh current verifier-root-cause artifact:

```text
docs/local/production-readiness/20260517T174743Z_qwen_mtp_chunk_policy_finalize/
```

This artifact found a real no-diagnostics `chunk_commit` bug, not a sampling
problem. With `VMLINUX_NATIVE_MTP_PHASE_DIAG` and GDN diagnostics disabled,
35B MXFP4 D3 `chunk_commit` originally stored lazy recurrent prefix state and
degenerated into garbage until length stop (`acceptedByDepth=0:382`,
`avgAcceptP=0.000`, `33.6 tok/s`). The fix materializes Mamba/GDN prefix
snapshots in `MambaCache.recordPrefixCommitState(...)` before storing a
verifier commit point. After the fix, the same no-diagnostics row returns exact
`1..50`, stops normally, reaches `84.7 tok/s`, and reports
`acceptedByDepth=2:3,3:45`, `avgCommittedPerVerify=3.94`, `avgAcceptP=0.979`.

The same artifact proves 27B MXFP4 no-diagnostics D3 `chunk_commit` correctness
at `26.1 tok/s` with exact `1..50`. That is not enough for the requested
45 tok/s 27B target; it is a correctness baseline for the next speed pass.

Current focused gate at 2026-05-17 09:03 PDT:

```text
docs/local/production-readiness/20260517T160343Z_qwen_mtp_settings_current/
docs/local/production-readiness/20260517T165508Z_qwen_mtp_settings_recheck/
docs/local/production-readiness/20260517T174743Z_qwen_mtp_chunk_policy_finalize/
```

- `MTPRuntimeFocusedTests_after_prefix_snapshot_materialize.log`: 42/42 pass.
  Coverage includes cached verifier
  masks carrying cache offsets, tensor-proven preserved MTP detection with
  auto-enable, metadata-only bundles without tensor evidence, explicit
  tensor-gated Qwen3.5 MoE activation, task-local activation and env override
  behavior, JANG metadata parsing, tensor/runtime-evidence-gated auto policy,
  recursive D3 hidden-state draft/verify contract, Qwen3.5 SSM accepted-prefix
  offsets, partial-reject lazy repair, private draft-cache refresh, greedy
  chunk verifier telemetry, explicit verifier env override ordering before the
  stochastic Mamba fallback, materialized prefix recurrent snapshots before
  verifier commit, BatchEngine native-MTP exclusive lane, and rejection of
  native MTP through batched `submit`.
- The same focused run also pins shape-walk quantization for MXFP4,
  JANG_2K, stock MLX affine embeddings, Qwen3.6 linear attention value dim,
  ZAYA CCA output width, JANG shared-expert gate width, Qwen3.5 norm convention
  propagation, and the rule that MTP sidecar tensors do not force backbone norm
  shifts.
- `VMLINUXServerRuntimeSettingsTests.log`: 12/12 pass. Coverage includes
  bundle generation config before server overrides, nil server sampling fields
  preserving engine/bundle defaults, top-k reaching speculative sampler
  probabilities, no hidden sampler guards, invalid sampling/sleep values
  reported instead of clamped, concrete prefix/paged/L2/SSM cache coordinator
  settings, paged-vs-legacy disk conflict rejection, TurboQuant KV bit-width
  validation, tensor-proven Qwen MTP auto-launch, load-time native-MTP sidecar
  selection, metadata-only force-on rejection, and policy/draft-limit launch
  resolution.
- Fresh 09:55 PDT recheck after the BatchEngine mixed-codec patch keeps both
  gates green on the current checkout: `MTPRuntimeFocusedTests.log` passes
  40/40 and `VMLINUXServerRuntimeSettingsTests.log` passes 12/12. The rerun
  specifically re-confirms tensor-evidence-only MTP activation, task-local/env
  override behavior, D3 hidden-state draft/verify, SSM accepted-prefix offsets,
  private draft-cache refresh, BatchEngine exclusive native-MTP lane, bundle
  generation defaults, top-k reaching sampler probabilities, and invalid
  settings reporting instead of clamping.
- Fresh 10:54 PDT recheck after the prefix-snapshot materialization fix keeps
  server settings green: `VMLINUXServerRuntimeSettingsTests_after_prefix_snapshot_materialize.log`
  passes 12/12, and the release `RunBench` product builds successfully in
  `build_RunBench_release_after_prefix_snapshot_materialize.log`.
- Fresh real-MTP auto-launch policy recheck under
  `docs/local/production-readiness/20260517T_real_mtp_auto_launch_policy/`
  proves Osaurus launch-time validation can use full `config.json` plus optional
  `jang_config.json` evidence. Supported tensor-proven Qwen MXFP bundles resolve
  `LoadConfiguration.nativeMTP=true` plus native D3 draft strategy; a force-on
  request against a blocked profile such as Qwen3.6 JANG_2K still produces a
  settings error before launch.
- Fresh 13:35 PDT settings-validation recheck passes
  `VMLINUXServerRuntimeSettingsTests.log` 16/16. It adds fail-loud validation
  for network host/port/rate-limit/timeout, concurrency batch sizes, and prefix
  cache sizing/TTL, so Osaurus can reject impossible panel settings instead of
  clamping them silently or launching an invalid server.
- Fresh 13:48 PDT active cache-policy recheck passes
  `CacheCoordinatorTopologyFocusedTests.log` 24/24. It moves the cache-scope
  and KV-policy salt checks into the active focused target, proving reasoning
  scope, reasoning effort, KV codec, and max-KV policy change the coordinator
  cache key instead of sharing paged/disk/SSM state across incompatible modes.
  This is focused cache-key proof, not a live multi-turn substitute.
- Fresh 14:05 PDT parser-fallback recheck has red/green artifacts for the
  active direct parser alias suite. The red log proves `bailing_hybrid`,
  `bailing_moe*`, `qwen3_6*`, and `qwen3_vl` model-type fallbacks previously
  returned nil tool formats. The green log passes 8/8 after routing
  Ling/Bailing to the GLM/deepseek arg-key parser and Qwen3.6/Qwen3-VL to the
  XML-function parser. JANG capability stamps still remain the highest-priority
  source when present. The broader no-hidden/parser sweep also passes 44/44 in
  `NoHiddenReasoningCloseBiasFocusedTests_green.log`.
- Fresh 14:15 PDT Qwen-VL capability-alias recheck has a red/green pair for
  direct parser capability stamps. The red log proves `qwen3_vl`,
  `qwen3_5_vl`, and `qwen3_6_vl` capability names could bypass both the
  reasoning parser and XML tool parser when the bundle exposed the capability
  name directly instead of relying on model-type fallback. The green log passes
  9/9 in `DirectCapabilityParserAliasFocusedTests_green.log`, and the broader
  no-hidden/parser sweep passes 45/45 after routing those direct capability
  stamps to Qwen thinking and XML-function tools.
- Fresh 14:35 PDT Harmony fragmentation/leak recheck has red/green artifacts
  for Gemma4/GPT-OSS style channel parsing. The red log proved stray
  `<|message|>` and `<|channel|>` control tokens could leak into visible free
  text outside a well-formed Harmony channel. The first fix also caught a
  start-role streaming regression (`assistant` leaking when `<|start|>` arrived
  before the role suffix), so the final green run keeps `<|start|>assistant`
  buffered until channel resolution or terminal scrub. `HarmonyParserFocusedTests`
  passes 8/8 and the broader no-hidden/parser sweep passes 47/47.
- Fresh 14:50 PDT Gemma4/SWA direct-compile recheck has a red/green artifact
  for the direct `TokenIterator` compiled path. The red log proved
  `setupCompiledDecode` still promoted only plain `KVCacheSimple` caches, so an
  all-rotating cache produced by Gemma4 with explicit `maxKVSize` could not use
  `CompilableRotatingKVCache` outside BatchEngine. The green log passes 6/6 in
  `Gemma4CacheTopologyFocusedTests_green.log` after adding the real rotating
  promotion; the broader cache topology sweep passes 25/25 across Gemma4,
  Ling/Bailing hybrid cache, ZAYA CCA, DSV4 disk-pool restore, media salts, and
  BatchKV rotating masks.
- Fresh 15:05 PDT Ling/Bailing capability-alias recheck has a red/green pair
  for direct capability stamps. The red log proved `bailing*` and `ling*`
  capability names could bypass both the think-XML reasoning parser and the
  GLM/deepseek arg-key tool parser even though model-type fallback was already
  covered. The green direct-alias suite passes 10/10 and the broader
  no-hidden/parser sweep passes 48/48 after routing those direct stamps to the
  same real Ling/Bailing parser contracts.

## Qwen3.5 35B 4-bit Loader Repair - 2026-05-17

Fresh live artifact after cleanup:

```text
docs/local/live-model-matrix/20260517T_qwen35_after_cleanup_infer/
```

This fixes the current `Qwen3.5-35B-A3B-4bit` release-gate blocker without
changing sampling policy. The failing row was a real loader/runtime shape bug:
the shape-walk quantization inference picked the preferred `(bits=8,
group_size=32)` candidate for stock MLX affine embedding tensors before
honoring the declared `group_size=64`. That unpacked the text embedding path to
1024 hidden units, then the first Qwen3.5 RMSNorm trapped because its weight is
2048-wide.

Code-level repair:

- `JangLoader.inferBitWidthAndGroupSize(...)` now honors a known
  `group_size` first when the packed/scales shape makes a valid bit width.
- Already-quantized embedding checkpoint tensors now load as
  `QuantizedEmbedding(weight:scales:biases:groupSize:bits:mode:)`, instead of
  quantizing a placeholder embedding and relying on a later parameter update.
- The text-only Qwen3.5 model registers nested modules with `@ModuleInfo`, so
  package-level parameter updates can reach the real text stack.

Current proof:

- focused tests pass for the stock MLX affine embedding shape case and the
  quantized embedding checkpoint initializer;
- release `RunBench` builds;
- `REPORT.md` passes all four rows: config, template, production defaults with
  cache off, and VL BatchEngine chat;
- production text uses bundle defaults (`temp=1.000`, `topP=0.950`, `topK=20`,
  `minP=0.000`, `rep=nil`) and passes 7/7 reasoning on/off rows with visible
  coherent answers at about 90-101 tok/s;
- VL BatchEngine chat loads `Qwen35MoE` with `Qwen3VLProcessor`; compile OFF and
  ON both ground the red/blue gradient image and answer the follow-up color.

Earlier failing diagnostics for this exact root cause are preserved under:

```text
docs/local/live-model-matrix/20260517T_qwen35_embedding_fix/
```

## Qwen3.5 35B BatchEngine TurboQuant B=2 Isolation - 2026-05-17

Fresh focused artifacts:

```text
docs/local/live-model-matrix/20260517T161305Z_release_turnmatrix_qwen35_35b_4bit/
docs/local/live-model-matrix/20260517T163920Z_qwen35_tq_b2_after_compat_split/
docs/local/live-model-matrix/20260517T164102Z_batch_arrays_cache_offset_focused/
docs/local/live-model-matrix/20260517T164940Z_release_turnmatrix_qwen35_35b_4bit_after_compat_split/
```

The full Qwen3.5 35B 4-bit release turnmatrix was green except
`batch_tq_b2`. That row exposed real BatchEngine/cache corruption: slot 0 plain
KV drifted when slot 1 decoded with TurboQuant KV in the same model forward.
This was not a model-quality issue and not a sampler issue.

Root causes:

- `BatchArraysCache.splitBack()` copied updated recurrent state arrays back to
  each per-slot `MambaCache`, but did not propagate the wrapper `offset` that
  Qwen35/GatedDeltaNet mutates during recurrent decode. B>1 hybrid SSM decode
  could therefore carry fresh state arrays with stale logical positions.
- Mixed plain KV and TurboQuant KV slots were admitted together and then forced
  into one decode forward even though their live cache codec signatures are not
  compatible.

Fix:

- `BatchArraysCache.splitBack()` now pushes the model-mutated decode-step
  advance back into every wrapped per-slot Mamba cache and refreshes the wrapper
  `offsetArray`.
- `BatchEngine` now groups active decode slots by cache topology and live KV
  codec before batching. Homogeneous plain/plain and TurboQuant/TurboQuant rows
  still batch normally. Mixed incompatible groups remain concurrently admitted
  but decode in separate compatible forwards in the same scheduler iteration.
  This is cache-topology routing, not a hidden generation guard.

Proof:

- `BatchArraysCacheFocusedTests.log` passes 1/1 and pins offset propagation for
  the model-mutated wrapper path.
- `qwen35_batch_tq_b2.out` passes the focused live row. Slot 0 plain output
  beside a TurboQuant neighbor is identical to the B=2 plain/plain reference,
  and diagnostics report `compatibilitySplits=191`.
- Full post-fix release turnmatrix now passes. `REPORT.md` records config,
  template, production defaults cache OFF/ON, BatchEngine single/chat/disk
  restore/concurrent/per-slot/TurboQuant B=2, VL batch chat, VL structured
  cache, media-salt isolation, and mixed text/image/video as `pass`. The
  generic `batch_cache_hit` row is `N-A` by topology/harness semantics.
- The mixed text/image/video row is coherent, including reasoning ON text,
  text-cache replay, image grounding, and video grounding. It is not a speed
  win: the video turn reports `TTFT 136312ms` and total `138.20s`, so high-res
  video throughput remains a performance watch even though the row is
  functionally correct.

## Qwen3.6 35B JANGTQ VLM Routed-Expert Repair - 2026-05-17

Fresh artifacts:

```text
docs/local/live-model-matrix/20260517T_qwen35_jangtq_vl_fix/
docs/local/live-model-matrix/20260517T_qwen35_jangtq_vl_matrix_after_fix/
docs/local/live-model-matrix/20260517T_qwen35_jangtq_turnmatrix_after_vl_fix/
docs/local/live-model-matrix/20260517T_qwen35_qwen3vl_video_config_fix/
```

Root cause:

- `Qwen3.6-35B-A3B-JANGTQ-CRACK` advertised `Qwen3VLProcessor` and vision
  tensors, but the VLM MoE path bound the text stack as affine `SwitchGLU`.
  Loader binding then rejected `switch_mlp.*.{tq_packed,tq_norms,tq_bits}` and
  fell back to the text-only `Qwen35JANGTQModel`, silently dropping the image.
- The fix is real routed-expert support in the VLM `Qwen35MoE` path:
  `text_config` receives the resolved JANGTQ metadata, VLM sparse MoE layers use
  `TurboQuantSwitchGLU` or `StreamingTurboQuantSwitchGLU`, and metadata-aware
  sanitize stacks per-expert JANGTQ tensors while dropping `tq_bits` sidecar
  keys that are not module parameters.

Current proof:

- pre-fix `BENCH_VL_BATCH_CHAT=1` loads `Qwen35JANGTQModel` with
  `LLMUserInputProcessor` and fails image grounding;
- post-fix `BENCH_VL_BATCH_CHAT=1` loads `Qwen35MoE` with `Qwen3VLProcessor`;
  compile OFF and ON both ground the red/blue gradient image and answer the
  follow-up text turn as `Red`;
- focused `vl` matrix passes `vl_batch_chat` and `vl_media_salt`;
- media-salt row proves same-image disk-backed restore HIT and different-image
  MISS with identical token counts, so image cache isolation is not a false
  positive;
- broader turn matrix passes config, template, production defaults with cache
  OFF/ON, BatchEngine single/chat/disk-restore/concurrent/per-slot/TurboQuant
  rows, VL batch chat, VL chat cache, and media-salt isolation. The generic
  batch cache-hit row remains `N-A` by topology/harness semantics.
- Qwen3VL video processor config is now wired through the real
  `video_preprocessor_config.json` contract. The focused video smoke row loads
  `Qwen35MoE` with `Qwen3VLProcessor`, attaches `LMInput.video` with pixels
  shape `[560, 1536]` on the resized 1080p fixture, and returns coherent
  visible content with `enable_thinking=false`.

Open boundary:

- `vl_mixed_text_image_video` completed T1 text reasoning-on, T2 repeated text
  cache, and T3 image with thinking off, then stayed in the T4 high-resolution
  video prefill/forward path for more than seven minutes on the 1080p fixture.
  The config repair does not fake-clamp the video budget: this bundle's
  `video_preprocessor_config.json` declares a large `longest_edge=25165824`, so
  the true high-resolution video row still needs a throughput/scaling gate. Do
  not count high-res video as production-clear for this bundle yet.

## Qwen3.6 27B MXFP4 Non-MTP Video-Budget Gate - 2026-05-17

Fresh artifacts:

```text
docs/local/live-model-matrix/20260517T_qwen36_27b_mxfp4_crack_current_non_kimi_turnmatrix/
docs/local/live-model-matrix/20260517T_qwen36_27b_mxfp4_crack_video_resize_postfix_turnmatrix/
```

Current proof:

- The post-fix turnmatrix passes config/template, production defaults with cache
  OFF/ON, BatchEngine single/chat/disk-restore/concurrent/per-slot/TurboQuant
  B=2, VL batch chat, structured VL chat cache, media-salt isolation, and mixed
  text/image/video.
- Bundle generation defaults reach runtime as
  `temp=1.000 topP=0.950 topK=20 minP=0.000 rep=nil`; MTP depth is `off`, as
  expected for this CRACK bundle.
- Reasoning ON/OFF closes with visible answers at a 2048-token production
  budget, and OFF rows emit no reasoning deltas.
- The structured VL cache row proves same-media disk restore HIT `99/99` and
  changed-media MISS; the text-only follow-up remains grounded in the image.
- The mixed video row now passes with explicit `BENCH_VL_VIDEO_RESIZE=224` and
  logs `video pixels shape: [560, 1536]`.

Open boundary:

- The preserved pre-fix artifact records the unbounded 1080p video turn failing
  after peaking at 164.2 GiB physical footprint inside MLX/Metal allocation.
  This is not a sampler guard issue and not a model-coherency failure. Osaurus
  needs a real Qwen video media resize/token-budget setting before raw
  high-resolution video can be exposed as production-clear.

## Active Non-Excluded Family Matrix

| Family | Local bundles | Engine surfaces to prove | Current MTP policy |
|---|---|---|---|
| Qwen3.6/Qwen3.5 text+VL MXFP/JANG/JANGTQ | Qwen3.6 MTP, Qwen3.6 CRACK, Qwen3.5 A3B 4-bit | Qwen chat template, `generation_config.json`, GatedDelta/hybrid SSM cache, disk L2 + SSM companion, VL media salt, explicit video resize/token budget, text-only continuation after media, reasoning on/off | Supported Qwen bundles with real MTP tensor evidence auto-resolve native D3; CRACK/no-tensor bundles stay off; JANG_2K remains blocked. |
| ZAYA text/VL JANGTQ/MXFP | JANGQ and Osaurus ZAYA1 text/VL | Zaya CCA cache, JANGTQ/MXFP decode, VL adapters, media salt, Hadamard/matmul shape coverage, multi-turn cache hit | No native MTP. |
| MiniMax M2.7 JANG/JANGTQ | Small JANGTQ and CRACK JANG; JANGTQ_K needs current re-run | MiniMax template, reasoning on/off, JANGTQ streaming experts, low-footprint active routed decode, prefix/paged/disk/TurboQuant KV, multi-turn coherence | Local MiniMax CRACK rows are non-MTP unless real MTP tensor evidence appears. Larger K/JANG CRACK rows need fresh current evidence before Osaurus promotion. |
| Ling/Bailing hybrid | Ling JANGTQ2 and MXFP4 CRACK | Bailing thinking template, hybrid cache/SSM rederive, nextn metadata handling, JANGTQ/MXFP decode, multi-turn coherence | Extra nextn-layer evidence exists, but Swift native-MTP auto-launch remains off. |
| Hy3 | HYV3 JANGTQ; JANGTQ_K needs current re-run | Hy3 template kwargs, native runtime registration, compiled decode guard, nextn metadata handling, routed/JANGTQ decode, cache topology | Extra nextn-layer evidence exists, but Swift native-MTP auto-launch remains off. JANGTQ is live-proven; K variants need fresh current evidence before Osaurus promotion. |
| Gemma 4 | Gemma-4 JANG_4M CRACK | Gemma4 template fallback/tools, sliding-window cache topology, RMSNorm no-scale parity, reasoning parser, multi-turn coherence | No native MTP. |
| Nemotron Omni | JANGTQ/JANGTQ4/MXFP4 Omni Nano | Parakeet audio encoder, RADIO vision, Omni text/image/audio/video ingest, BatchEngine stress, cache/media state, text output coherence | No native MTP. |
| Laguna | Laguna XS JANGTQ | Laguna/Mistral-style template and RoPE params, JANGTQ decode, prefix/paged/disk cache, multi-turn coherence | No native MTP. |

The no-load MTP census now distinguishes `mtp_tensors` from `mtp_auto`.
Tensor-proven supported Qwen MTP resolves native D3 through the settings bridge;
Hy3/Ling/DSV4 nextn-style rows can report tensor evidence while `mtp_auto=no`,
and JANG_2K remains blocked until its own runtime policy is proven.

The no-load metadata/template sweep passed 56/56 rows across the 28
non-excluded bundles. Warnings remain visible and must not be collapsed into a
production pass; current warnings include BOS/EOS overlap on some Qwen/Laguna
bundles, MiniMax tokenizer/config BOS mismatch, and ZAYA1 JANGTQ_K tokenizer EOS
not present in the effective EOS set.

## ZAYA Cache and Release-Speed Checkpoint

2026-05-17 ZAYA cache-on failures were traced to real tensor-shape issues, not
sampling behavior:

- ZAYA text CCA `sub.o_proj` JANG shape inference used full hidden width instead
  of the CCA output width. The loader now infers the real 1024-wide input for
  the 2048-wide artifacts.
- BatchEngine and TokenIterator history-boundary rederive now feed remaining
  tokens as batch-first `[1, T]` tensors, matching normal prefill/decode. The
  previous 1D rederive path could reach ZAYA CCA with a 2D activation and trap
  in `transposed(0,2,1)`.

Post-fix release artifacts:

```text
docs/local/live-model-matrix/20260517T_zaya_speed_regression/
```

Release speed rows, compared to the older handoff floor:

| Bundle | Older documented row | 2026-05-17 release row | Result |
|---|---:|---:|---|
| ZAYA1-8B-JANGTQ4 | 54.7 tok/s, `8831/1365` graph/asType | 66.5 tok/s median, 66.6 best; graph `8831/1365` | PASS |
| ZAYA1-8B-JANGTQ_K | no older speed floor recorded in the handoff table | 65.7 tok/s median, 65.8 best | Historical until re-run under current non-Kimi matrix |
| ZAYA1-8B-MXFP4 | 66.8 tok/s, `8791/1285` graph/asType | 66.2 tok/s median, 66.7 best; graph `8791/1285` | PASS |

The earlier ~16 tok/s measurements came from `.build/debug/RunBench`; those
remain useful for correctness while debugging, but they are not production
speed gates. Speed claims must use `.build/release/RunBench` or an equivalent
release-built server binary.

ZAYA1-8B-JANGTQ4 also passed the release `BENCH_PROD` cache-on multi-turn row:

- 7/7 rows passed with reasoning on/off flips, visible answers, no loop/leak,
  and normal stop reasons;
- `generation_config.json` defaults were applied: `temp=0.600`, `topP=1.000`,
  `topK=0`, `rep=nil`;
- cache stats showed `disk{hits=1,stores=21}` and `ssm{hits=1,reDerives=0}`;
- ZAYA remains `pagedIncompatible=true` by topology, so generic paged-prefix
  hits must not be advertised for this family.

## Nemotron Omni Live Voice Reverify

Fresh 2026-05-17 artifacts:

```text
docs/local/live-model-matrix/20260517T_omni_reverify/
docs/local/live-model-matrix/20260517T_omni_current_recheck/
docs/local/live-model-matrix/20260517T155603Z_omni_live_voice_current_verify/
docs/local/live-model-matrix/20260517T164618Z_omni_live_voice_current_recheck/
docs/local/live-model-matrix/20260517T164640Z_omni_live_audio_streaming_jangtq4_current/
docs/local/live-model-matrix/20260517T164702Z_omni_integrated_jangtq4_current/
docs/local/live-model-matrix/20260517T170614Z_omni_live_voice_fresh_recheck/
```

Current result on `Nemotron-Omni-Nano-JANGTQ4-CRACK`:

- `NemotronHOmniPreEncodedAudioTests`: 8/8 focused tests pass. Coverage includes
  retained live audio buffer snapshots, pre-encoded Parakeet embedding
  preservation, RADIO pixel shuffle, Parakeet relative shift, projector remaps,
  Parakeet source weight transposes, video EVS placeholder count, and no hidden
  greedy sampling override in the latency bench.
- Release build for `OmniAudioChunkStabilityBench` passed; `OmniAudioLatencyBench`
  was already up to date in `.build/release`.
- `omni_audio_latency_jangtq4_both_paths.log`: real audio fixture loads and runs
  through both BatchEngine streaming and TokenIterator streaming. The bench uses
  bundle defaults (`temp=0.600 topP=0.950 topK=0 minP=0.000 rep=1.000`), not a
  hardcoded sampler guard. Parakeet pre-encode is 63 tokens in 43.4 ms; all
  eight raw/pre-encoded repeated turns produce coherent audio-grounded text.
- `omni_audio_chunk_stability_jangtq4.log`: full retained-audio Parakeet encode
  is 63 tokens in 48.9 ms and all 10 prefix comparisons remain unstable at the
  default tolerance. This confirms the live voice contract: accumulate PCM,
  refresh a full retained-audio pre-encode while the user speaks, and submit the
  latest exact pre-encoded snapshot at endpoint. Do not concatenate independently
  encoded Parakeet chunks.
- `omni_runbench_jangtq4_48.log`: integrated `BENCH_OMNI=1` passes 14/14 rows
  with `maxTokens=48`, including text, multi-turn, image, video, audio,
  reasoning on/off, mixed image+audio, media-salt isolation, and hybrid SSM
  warm-pass. Load: 1.87 s. Decode rows: 85.4-105.5 tok/s.
- Current-checkout recheck:
  - `NemotronHOmniPreEncodedAudioTests.log`: 8/8 passes.
  - `omni_audio_latency_jangtq4_current_32.log`: release-built live audio bench
    reloads JANGTQ4, uses bundle defaults (`temp=0.600 topP=0.950 topK=0
    minP=0.000 rep=1.000`), pre-encodes Parakeet to 63 x 2688 in 46.8 ms, and
    streams both BatchEngine and TokenIterator paths with raw PCM and
    pre-encoded embeddings. First deltas are 157-224 ms; decode rates are
    62.5-74.1 tok/s. The 16-token smoke in the same folder proves wiring but is
    too short to judge every stochastic audio answer.
  - `omni_audio_chunk_stability_jangtq4_current.log`: 10/10 prefix comparisons
    are not concat-safe at the default tolerance, so the retained-full-snapshot
    live-voice contract remains required.
  - `omni_runbench_jangtq4_48_current.log`: integrated current `BENCH_OMNI=1`
    passes 14/14 at `maxTokens=48`, with load 1.95 s and decode rows
    90.4-109.9 tok/s.
- Fresh 08:56 PDT current-verify recheck:
  - `NemotronHOmniPreEncodedAudioTests.log`: Xcode-backed focused test command
    passes 8/8, including retained live audio snapshots, pre-encoded Parakeet
    preservation, RADIO pixel shuffle, Parakeet relative shift, EVS placeholder
    count, projector remaps, source weight transposes, and generation-default
    plumbing.
  - `build_omni_audio_latency.log`, `build_omni_audio_chunk_stability.log`,
    and `build_runbench.log`: all three release products rebuilt.
  - `omni_audio_latency_jangtq4_both_paths_32.log`: JANGTQ4 loads, uses bundle
    defaults (`temp=0.600 topP=0.950 topK=0 minP=0.000 rep=1.000`), pre-encodes
    Parakeet to 63 x 2688 in 50.1 ms, and streams raw PCM plus pre-encoded
    audio through both BatchEngine and TokenIterator. First deltas are
    203.5-219.3 ms raw BatchEngine, 176.0-188.7 ms pre-encoded BatchEngine,
    184.6-188.5 ms raw TokenIterator, and 157.1-157.7 ms pre-encoded
    TokenIterator. Decode rates are 62.3-73.1 tok/s.
  - `cache_artifacts_listing.txt`: the audio bench wrote `cache_index.db`,
    safetensors block entries, and `ssm_companion` state under the raw and
    pre-encoded cache dirs.
  - `omni_audio_chunk_stability_jangtq4.log`: 10/10 prefix comparisons remain
    not concat-safe, so retained PCM plus full-snapshot pre-encode remains the
    required live voice contract.
  - `omni_runbench_jangtq4_48.log`: integrated `BENCH_OMNI=1`
    `BENCH_OMNI_BATCH=1` passes 18/18 at `maxTokens=48`. Load is 1.79 s,
    direct decode rows are 88.4-110.3 tok/s, and BatchEngine rows are
    37.6-70.8 tok/s.
- Fresh 09:46 PDT current recheck after the BatchEngine mixed-codec scheduler
  patch:
  - `NemotronHOmniPreEncodedAudioTests.log`: Xcode-backed focused test command
    passes 8/8 on the current checkout. Covered rows are retained live audio
    snapshots, caller-supplied Parakeet embeddings, RADIO pixel shuffle,
    Parakeet relative shift, EVS placeholder count, projector remaps, source
    weight transposes, and generation-default plumbing.
  - `omni_audio_latency_jangtq4_both_paths_cache_off_32.jsonl`: release-built
    JANGTQ4 live-audio bench loads in 1.74 s, uses bundle defaults
    (`temp=0.600 topP=0.950 topK=0 minP=0.000 rep=1.000`), pre-encodes
    Parakeet to `63 x 2688` in 46.4 ms, and streams raw PCM plus pre-encoded
    audio through both BatchEngine and TokenIterator. First deltas are
    221.2 ms raw BatchEngine, 170.3 ms pre-encoded BatchEngine, 183.3 ms raw
    TokenIterator, and 152.3 ms pre-encoded TokenIterator. Decode rates are
    66.0-75.6 tok/s.
  - `omni_runbench_jangtq4_48.log`: integrated `BENCH_OMNI=1`
    `BENCH_OMNI_BATCH=1` passes 18/18 at `maxTokens=48` with bundle defaults.
    Load is 1.74 s, direct decode rows are 93.1-112.5 tok/s, and BatchEngine
    rows are 45.0-69.5 tok/s. Covered rows include text, image, video encoder,
    video LMInput, audio encoder, audio LMInput, reasoning on/off, mixed
    image+audio, media-salt isolation, hybrid SSM warm-pass, and BatchEngine
    text/image/audio.

## ZAYA Harness and VL Follow-Up - 2026-05-17

Targeted rerun artifacts:

```text
docs/local/live-model-matrix/20260517T_release_targeted_rerun_after_harness_fixes/
```

Harness fixes from this pass:

- `BENCH_VL_BATCH_CHAT` no longer assumes the synthetic CoreImage gradient has a
  specific top/bottom orientation. The row still requires visible image color
  grounding; it now asks for one visible color and accepts the actual red/blue
  colors in the generated image.
- ZAYA video rows now report `not applicable` when the processor explicitly
  throws `ZAYA1-VL video input is not implemented`. This is a family capability
  boundary, not a model coherency failure.
- `BENCH_BATCH_TQ_B2` now uses a shape-matched B=2 plain/plain baseline for
  plain-slot isolation. The old B=1 solo baseline remains a diagnostic, but it
  is not a valid cross-slot corruption oracle for families that diverge only
  because B=2 batching changes low-level numeric tie breaks deep in an open
  decode.

Post-harness targeted evidence:

| Row | Result | Finding |
|---|---|---|
| `ZAYA1-8B-MXFP4.batch_tq_b2` | PASS | Plain slot beside TurboQuant matched B=2 plain/plain exactly; old B=1 solo comparison still drifts at token 110 and is logged as diagnostic only. |
| `ZAYA1-VL-8B-JANGTQ4.vl_batch_chat` | PASS | Compile OFF and ON both ground the image and answer the follow-up color as `blue`. |
| `ZAYA1-VL-8B-JANGTQ4.vl_mixed_text_image_video` | N-A for video | Text and image turns pass; video turn is explicitly not implemented for this processor. |
| `ZAYA1-VL-8B-MXFP4.vl_mixed_text_image_video` | N-A for video | Text and image turns pass; video turn is explicitly not implemented for this processor. |

Historical K-only diagnostic retained for traceability:

- `ZAYA1-VL-8B-JANGTQ_K` is still not production-clear if that lane is reopened.
  The release matrix
  produced `8` for the `7+8-11` smoke where the expected visible answer is `4`,
  and the structured VL cache row exhausted the 192-token budget on the cold
  image turn. Do not hide this with sampling clamps or looser validators; this
  needs runtime/bundle root-cause work before that artifact is green.
- Follow-up top-k evidence shows the math failure is present before decoding
  policy: on the exact chat-rendered math prompt, `ZAYA1-VL-8B-JANGTQ4` ranks
  token `4` first, while `ZAYA1-VL-8B-JANGTQ_K` ranks `6`, `7`, `8`, then `4`.
  The JANGTQ_K layer-1 actual tensor kernel probe passed for experts 0/7/15
  with tiny max diffs, so the next investigation is broader artifact/runtime
  parity across layers or conversion, not a sampling fallback.

## Laguna XS Release Matrix - 2026-05-17

Clean release artifact:

```text
docs/local/live-model-matrix/20260517T_release_turnmatrix_laguna_xs_after_b2_fix/
```

Laguna is now green for the current text turnmatrix:

- config/template smoke: PASS;
- `BENCH_PROD` cache OFF and cache ON: 7/7 each, coherent visible output,
  normal stops, reasoning on/off routed correctly, bundle defaults applied
  (`temp=0.700`, `topP=0.900`, `topK=0`, `rep=nil`, `seed=0`);
- release decode telemetry: about 31 tok/s on the production rows;
- disk restore row: PASS, with the disk cache directory populated;
- generic paged prefix hit row: N-A because Laguna is paged-incompatible and
  uses disk-backed restore;
- B=2 concurrent, per-slot sampler, and TurboQuant-KV B=2: PASS with
  `activeCountHighWatermarkForDiagnostics >= 2`.
- active focused Laguna contracts now run in
  `NoHiddenReasoningCloseBiasFocusedTests`: parser aliases route to
  think-XML reasoning plus GLM tools, thinking-off prompt tails start visible
  content instead of hidden reasoning, thinking-on prompt tails route only
  pre-`</think>` bytes into reasoning, assistant-history reasoning/content is
  preserved, and mixed-shape `rope_parameters` drops only top-level scalar
  entries.

Harness fix from this row:

- The B=2 proof now records an internal BatchEngine active-slot high-water
  mark. External polling can miss short-lived overlap while model forwards
  monopolize the actor executor, so the release gate now drains streams while
  observing both live `activeCount` and the engine's high-water mark.

## Ling/Bailing JANGTQ2 Release Matrix - 2026-05-17

Fresh release artifact:

```text
docs/local/live-model-matrix/20260517T170008Z_release_turnmatrix_ling_jangtq2/
```

`Ling-2.6-flash-JANGTQ2-CRACK` is green for the current text turnmatrix:

- config/template smoke: PASS;
- MTP metadata gate: PASS, while native MTP stays inactive/explicit unless a
  verified runtime profile exists;
- `BENCH_PROD` cache OFF and cache ON: 7/7 each, coherent visible output,
  normal stop reasons, no reasoning marker leak, and no hidden sampler guard;
- bundle-default sampling resolved to `temp=0.600`, `topP=1.000`, `topK=0`,
  `minP=0.000`, `rep=nil`, `seed=0`;
- release decode telemetry is about 37-39 tok/s on production rows;
- cache ON records `disk{hits=1,stores=21}` and `ssm{hits=1,reDerives=0}`;
- generic paged prefix hit row is `N-A` because Ling/Bailing is
  paged-incompatible and uses disk-backed restore;
- BatchEngine single/chat/disk-restore/B=2/per-slot/TurboQuant B=2 all pass.
  The TurboQuant B=2 row preserves the plain slot exactly against the B=2
  plain/plain reference and records `compatibilitySplits=9`.

The larger `Ling-2.6-flash-MXFP4-CRACK` bundle now also has a current release
turnmatrix:

```text
docs/local/live-model-matrix/20260517T180538Z_release_turnmatrix_ling_mxfp4_current/
```

MXFP4 Ling status:

- config/template/MTP metadata rows: PASS, with `modelType=bailing_hybrid`,
  `dispatch=bailing_hybrid`, 32 layers, `weightFormat=mxfp4`, and tokenizer
  BOS/EOS coverage;
- `BENCH_PROD` cache OFF and cache ON: 7/7 each, visible coherent answers,
  normal stops, no loop/leak, no active thinking rail, and no hidden sampler
  guard;
- bundle/default fallback sampling resolved to `temp=0.600`, `topP=1.000`,
  `topK=0`, `minP=0.000`, `rep=nil`, `seed=0`;
- release decode telemetry is about 9.7-10.1 tok/s on the short production
  rows, with cache OFF peak RSS `46690MiB` and cache ON peak RSS `46384MiB`;
- cache ON records the expected hybrid topology:
  `hybrid=true`, `pagedIncompatible=true`, disk L2 `hits=1`, `stores=21`,
  `maxBytes=4294967296`, and SSM companion `hits=1`, `reDerives=0`;
- BatchEngine single/chat/disk-restore/B=2/per-slot/TurboQuant B=2 all pass,
  and the TurboQuant B=2 row preserves the plain slot exactly against the B=2
  plain/plain reference with `compatibilitySplits=9`.
- active focused Bailing/Ling template coverage now runs in
  `NoHiddenReasoningCloseBiasFocusedTests`: `enable_thinking=true` inserts
  `detailed thinking on`, `enable_thinking=false` inserts or replaces
  `detailed thinking off`, and non-Bailing or missing-toggle paths remain
  unchanged.
- fresh active focused Bailing/Ling cache coverage now runs in
  `CacheCoordinatorTopologyFocusedTests`: the actual `BailingHybridModel`
  allocates `ArraysCache` for linear-attention layers and KV/RotatingKV caches
  for global MLA layers, keeps trailing partial layer groups global, and remains
  marked for disk-backed coordinator restore rather than accepting a generic
  prefix hit.

Boundary: Ling/Bailing is now live-proven for the current text turnmatrix on
both JANGTQ2 and MXFP4. Native MTP stays fail-closed unless the family gets a
separate verified runtime profile; the current rows do not auto-activate
`nextn`/MTP metadata by name.

## Hy3 JANGTQ Release Matrix - 2026-05-17

Fresh release artifact:

```text
docs/local/live-model-matrix/20260517T180931Z_release_turnmatrix_hy3_jangtq_current/
```

`Hy3-preview-JANGTQ` is green for the current text turnmatrix:

- config/template/MTP metadata rows: PASS, with `modelType=hy_v3`,
  `dispatch=hy_v3`, 80 layers, `weightFormat=mxtq`, JANGTQ sidecar tensors,
  routed expert bits `2`, and tokenizer BOS/EOS coverage;
- `BENCH_PROD` cache OFF and cache ON: 7/7 each, visible coherent output,
  normal stops, no active thinking rail, and no hidden sampler guard;
- generation config is applied as declared for this bundle:
  `temp=0.900`, `topP=1.000`, `topK=-1`, `minP=0.000`, `rep=nil`, `seed=0`.
  The negative top-k is recorded as the bundle value reaching the runtime, not
  silently rewritten in the live row;
- release decode telemetry is about 23-24 tok/s after load; cold first prompt
  time is high, but same-prompt cache falls to hundreds of milliseconds;
- cache ON is non-hybrid/paged-compatible for this row and records paged
  `hits=1`, disk L2 `stores=21`, and `maxBytes=4294967296`;
- disk restore, B=2 concurrent, per-slot sampler, and TurboQuant-KV B=2 all
  pass. The TurboQuant B=2 row preserves the plain slot exactly against the B=2
  plain/plain reference with `compatibilitySplits=10`.

Hy3 JANGTQ_K is not promoted for the active pass until it is re-run under the
current non-Kimi matrix. The current historical failure
mode is still retained here because it was narrowed and documented:

```text
docs/local/live-model-matrix/20260517T182455Z_release_turnmatrix_hy3_jangtqk_current/
docs/local/live-model-matrix/20260517T183024Z_hy3_jangtqk_streaming_probe/
docs/local/live-model-matrix/20260517T183237Z_hy3_jangtqk_streaming_modeldir_probe/
docs/local/live-model-matrix/20260517T184132Z_hy3_jangtqk_streaming_autodir_after_fix/
```

- The eager JANGTQ_K production row was killed before load completion. The
  model is 102GB on disk and uses nested routed expert bits
  `gate_proj=2`, `up_proj=2`, `down_proj=4`; this is a real low-RAM loading
  problem, not a sampling problem.
- With active expert streaming enabled but no model directory bound, load
  skipped `91008` per-expert tensors and then failed at runtime with
  `missing active JANGTQ gate/up tensors for layer 1`.
- Binding the real model directory proved the active-streaming correctness path:
  load completed in `6.03s`, peak RSS was `6383MiB`, active expert tensors were
  indexed as `layers=79 experts=192 stackedLayers=0`, and the production
  content matrix passed 7/7 with bundle sampling unchanged.
- The engine now binds the loaded model directory inside `loadWeights` when
  active streaming is enabled, so the same row passes without requiring a
  process-global `MLXPRESS_MODEL_DIR`: load `3.84s`, peak RSS `6168MiB`, same
  active expert index, and 7/7 coherent content rows.
- active focused Hy3 parser/no-leak coverage now runs in
  `NoHiddenReasoningCloseBiasFocusedTests`: Hy3 aliases resolve to Hunyuan
  tool calls plus `think_xml` reasoning, Hunyuan parses multiple scalar-argument
  calls, reasoning-before-tool-call streaming does not leak markers, and
  prompt-tail open/closed `<think>` states route reasoning/content correctly.
- the same focused suite now pins active Hy3 nextn handling instead of relying
  on metadata absence: `Hy3Model.newCache(parameters:nil)` allocates only the
  base decode layers, `loraLayers` excludes preserved nextn/MTP layers, and the
  sanitizer fuses real q/k/v projection tensors while dropping preserved nextn
  tensors from the base runtime load path.
- Hy3 mixed-bit q/k/v projection sanitize no longer aborts the process. If the
  three projections use different packed widths but carry quant metadata, the
  engine dequantizes them to dense tensors and fuses the real `qkv_proj.weight`;
  if metadata is insufficient, source keys remain intact so load verification
  fails instead of running a random qkv projection.
- `RuntimeMoETopKOverrideFocusedTests` pins the override as lower-only and
  cache-key-scoped: an explicit override may reduce Hy3 routed experts for an
  experiment, but it will not raise another model's trained top-k and it does
  not silently mutate the bundle default.

Boundary: Hy3 JANGTQ_K is not production-clear for the current pass without a
fresh current re-run. It is correctness/low-footprint proven through active expert
streaming but still speed-blocked at about 1.4 tok/s on the short production
row. The remaining work is active expert residency/stacked bank speed, not a
fake repetition penalty, temperature floor, or template guard. Also, the
active-expert store remains process-global; true simultaneous multi-model
streaming needs a per-loaded-model store or module-captured index before
Osaurus should expose multiple active JANGTQ_K streaming models.

## Gemma 4 Text Release Matrix - 2026-05-17

Clean release artifact:

```text
docs/local/live-model-matrix/20260517T160608Z_release_turnmatrix_gemma4_26b/
```

`Gemma-4-26B-A4B-it-JANG_4M-CRACK` is now green for the current text
turnmatrix:

- config smoke: PASS, with `modelType=gemma4`, `dispatch=gemma4_text`,
  30 layers, sliding-window topology, `tokenizerEOSCovered=true`, and
  `bosInEOS=false`;
- template smoke: PASS for plain, thinking false/true, `reasoning_effort=max`,
  tools, large tool context, multi-turn off, and reasoning-history rendering;
- `BENCH_PROD` cache OFF and cache ON: 7/7 each, coherent visible output,
  normal stops, no loop/leak, and bundle defaults applied through the engine;
- footprint stayed stable across cache modes: `peakRSS=13140MiB` cache OFF and
  `peakRSS=13234MiB` cache ON;
- cache ON stats showed this Gemma topology is `pagedIncompatible=true`; paged
  counters correctly stayed zero, while disk L2 recorded `hits=1`, `misses=16`,
  `stores=14`, and `maxBytes=4294967296`;
- BatchEngine single, chat, disk restore, B=2 concurrent, B=2 per-slot sampler,
  and TurboQuant-KV B=2 isolation: PASS;
- active focused SWA/cache coverage is now wired under
  `CacheCoordinatorTopologyFocusedTests`: 4 Gemma 4 topology tests prove the
  actual `Gemma4TextModel.newCache` allocation matches the mixed
  RotatingKVCache+KVCacheSimple no-`maxKVSize` path, the all-rotating
  `maxKVSize` path stays `.rotating`/compile-eligible, and full-attention
  rotating caches keep the attention-sink shape; 4 BatchKVCache rotating-slot
  tests prove post-wrap masks use the capped effective key length instead of
  `offset + n`;
- direct TokenIterator compiled decode now promotes all-rotating Gemma4/SWA
  caches to `CompilableRotatingKVCache` when the caller explicitly supplies
  `maxKVSize`; the default mixed SWA/full-attention topology remains
  `.heterogeneous` and uncompiled;
- the generic prefix-extension paged cache-hit row is N-A because this model is
  routed through the disk-backed paged-incompatible cache path.

This clears the current Gemma 4 text multi-turn/cache/batching row. The
image/text VL cache row and live tool-call schema row are now covered by the
fresh artifacts below. Long-budget Harmony reasoning remains a separate open
row.

Fresh Gemma 4 VL structured chat-cache artifact:

```text
docs/local/live-model-matrix/20260517T210417Z_gemma4_vl_chat_cache/
```

`BENCH_VL_CHAT_CACHE=1` passes on the current release `RunBench` binary:

- model loads as `Gemma4` with `Gemma4Processor`;
- cache coordinator selects the disk-backed path-dependent restore topology;
- image A cold turn generates grounded visible output:
  "orange at the top to blue at the bottom";
- same image replay hits the coordinator with `HIT disk 308/308` and TTFT drops
  from `609ms` to `32ms`;
- different image probe misses correctly;
- text-only follow-up remains grounded in the earlier image colors and leaks no
  raw `<think>`, `<image>`, or media markers;
- `/usr/bin/time -l` records peak memory footprint `30617874576` bytes.

Fresh Gemma 4 live tool-call schema artifact:

```text
docs/local/live-model-matrix/20260517T212204Z_gemma4_batch_toolcall_real_schema/
```

`BENCH_BATCH_TOOLCALL=1` now passes on the current release `RunBench` binary
after the harness was tightened to send a real `get_weather` schema through
`UserInput.tools` and to fail empty-output/no-tool rows:

- model loads as `Gemma4`;
- `Tool format: gemma4` and `Reasoning stamp: harmony` are active;
- stream emits `toolCalls: 1`, `stop`, `genTokens=14`;
- structured call is `get_weather({"location":"Tokyo"})`;
- visible chunks and reasoning chars are both zero, so no raw Harmony or
  tool-call markers leak to `.chunk`;
- `/usr/bin/time -l` records peak memory footprint `25524235616` bytes.

Additional Harmony parser follow-up:

```text
docs/local/live-model-matrix/20260517T_harmony_parser_fix_current/
docs/local/production-readiness/20260517T_parser_cache_nonexcluded_current/
docs/local/production-readiness/20260517T_laguna_mistral_gemma4_active_contracts/
```

- The shared Harmony reasoning parser now covers both Gemma 4
  `<|channel>...<channel|>` envelopes and GPT-OSS/Harmony
  `<|channel|>analysis|final<|message|>...<|end|>/<|return|>` channels.
- Focused tests in `NoHiddenReasoningCloseBiasFocusedTests` prove Gemma 4
  reasoning/content separation, GPT-OSS analysis/final splitting, and Gemma 4
  reasoning followed by a structured tool call without leaking control markers.
- The focused Harmony suite also now has red/green proof for one-character
  stream fragmentation and stray control-token scrubbing: direct
  `<|message|>`/`<|channel|>` markers are removed from visible free text, while
  malformed non-marker text remains visible and no close tag is synthesized.
- The same active focused suite also pins `gpt_oss*` model types to the Harmony
  parser and the GLM boundary: bare `glm4` remains non-reasoning while
  `glm4_moe*`/`glm5*` route through think-XML, and suffixed GLM/DeepSeek/Laguna
  capability names route to the GLM tool parser.
- A direct capability-alias follow-up closes the suffix leak path that model-type
  heuristics did not cover: `gemma4_27b`, `gpt_oss_20b`, and `gpt_oss_120b`
  route to Harmony; `glm4_moe_lite`, `glm5_air`, `deepseek_v4_flash`, and
  `laguna_glm_thinking_v5` route to think-XML; explicit `mistral4*` reasoning
  capability stamps route `[THINK]...[/THINK]` while the `mistral4` model-type
  fallback remains no-reasoning unless a bundle stamps that parser. Mistral 4
  and Pixtral tool aliases route to the Mistral tool parser. GLM 5 model-type
  fallback now also infers the GLM tool parser (`glm5`, `glm5_air`,
  `glm5_1_flash`) so reasoning and tool parsing stay aligned without relying
  on a JANG capability stamp.
- 2026-05-17 current-checkout parser/cache recheck fixed one real parser-order
  bug: `ToolCallFormat.fromCapabilityName("deepseek_v4" | "deepseek_v4_flash" |
  "deepseekv4")` now resolves to DSML before the generic DeepSeek/GLM prefix,
  while `deepseek` and `deepseek_v3` remain GLM-style.
- 2026-05-17 Hy3 product-alias facade recheck fixed a second parser source gap:
  stamped `hy3-preview` product capabilities now resolve directly to think-XML
  reasoning plus Hunyuan tools through `ParserResolution`, not only through the
  `model_type` heuristic. The focused parser suite passes 51/51 and the full
  active focused target passes 186/186 in
  `docs/local/production-readiness/20260517T1355_hy3_product_alias_parser/`.
- The same current-checkout recheck activated old inactive Mistral/Laguna/Gemma4
  contracts: `mistral3`/`mistral3_text`/`ministral3` remain no-reasoning and
  use the Mistral tool parser, Laguna template/RoPE contracts are active, and
  Gemma4 VLM source guards reject unsupported `LMInput.audio` explicitly while
  resolving `<|image|>` through `convertTokenToId` instead of `encode().last`.
  Additional VL focused contracts now pin Gemma4 VLM `rmsNormNoScale` parity and
  dtype preservation, plus Gemma3/Gemma4 `maskedScatter` recoverable
  `VLMError.processing` paths instead of process-aborting `fatalError`.
  These are parser/source/math contracts, not a live Gemma4 VL production row.
- Mistral3/Ministral factory-dispatch is now active too: a tiny synthetic config
  proves `weight_format=mxtq` and `MXTQ` route to `Mistral3TextJANGTQModel`,
  `mxtq_bits=4` changes the packed dense width, and `mxfp4` or missing
  `weight_format` stays on the vanilla `Mistral3TextModel` path.
- The fresh aggregate active focused target passes 168 tests in 26 suites in
  `docs/local/production-readiness/20260517T1300_hy3_mixed_qkv_runtime_contracts/MLXLMCommonFocusedTests_after_hy3_mixed_qkv.log`.
- A post-toolcall current-checkout recheck with
  `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcrun swift test --filter MLXLMCommonFocusedTests`
  passes 187 Swift Testing rows in 26 suites plus the selected XCTest focused
  rows. This includes the new Gemma4 live-probe source guard and keeps the
  Harmony, Ling/Bailing, Hy3, Gemma4 SWA/VLM, MTP metadata/SSM cache, Omni,
  JANGTQ, and server-settings no-hidden-guard contracts green.
- A post-fix live Gemma 4 `BENCH_HARMONY_CHECK` row passes: marker strings are
  absent from `.chunk`, the output is coherent visible README guidance, and
  generation stops through the normal path.

Boundary: the Gemma 4 live Harmony smoke did not elicit reasoning deltas on
that prompt, and there is no local GPT-OSS bundle in `~/models` for a live
GPT-OSS decode row. The GPT-OSS claim is therefore parser-contract proof only,
not a model-runtime production pass. Long-budget Gemma 4 thinking remains a
separate open row.

## Nemotron Omni JANGTQ Release Matrix - 2026-05-17

Clean post-failgate artifact:

```text
docs/local/live-model-matrix/20260517T_release_turnmatrix_nemotron_omni_jangtq_after_omni_failgate_v2/
```

The text/cache/batch side is healthy, but the Omni media row is not
production-clear:

- config/template smoke: PASS;
- `BENCH_PROD` cache OFF and cache ON: PASS with visible coherent answers,
  reasoning on/off routed correctly, and no hidden sampler guard;
- release text throughput is about 104-110 tok/s on direct text rows;
- disk restore row: PASS, with `cache_index.db`, safetensors entries, and
  `ssm_companion` present in the cache directory;
- BatchEngine text B=1 and B=2: PASS;
- Omni aggregate row: FAIL by design, because 5 of 18 subrows fail with
  repeated bigram loops on image/audio LMInput paths:
  image single-turn, image reasoning-off direct, image multi-turn,
  audio LMInput end-to-end, and BatchEngine image B=1.

Harness fix from this row:

- `OmniBench` now exits nonzero when any printed subrow fails. The previous
  artifact reported `.omni | pass` despite a summary of `13 passed, 5 failed`;
  the fresh artifact reports `.omni | fail:1` and keeps the failed media rows
  visible for root-cause work. This is a real blocked media-runtime row, not a
  sampling-policy issue.

## Nemotron Omni Live Voice Consolidation - 2026-05-17

The live voice benches from the standalone Swift LM package are now part of
this package:

- `OmniAudioLatencyBench`
- `OmniAudioChunkStabilityBench`

Build verification:

```sh
swift build -c release --product OmniAudioLatencyBench
swift build -c release --product OmniAudioChunkStabilityBench
```

Fresh local JANGTQ live voice artifacts:

```text
docs/local/live-model-matrix/20260517T_omni_audio_latency_jangtq.jsonl
docs/local/live-model-matrix/20260517T_omni_audio_latency_jangtq_prompt48.jsonl
docs/local/live-model-matrix/20260517T_omni_audio_chunk_stability_jangtq.jsonl
docs/local/live-model-matrix/20260517T_omni_live_voice_current/omni_audio_latency_both_paths_genconfig_precise.log
docs/local/live-model-matrix/20260517T_omni_live_voice_current/omni_audio_latency_both_paths_explicit_prompt.log
docs/local/live-model-matrix/20260517T_omni_live_voice_current/omni_audio_chunk_stability.log
docs/local/live-model-matrix/20260517T_omni_live_voice_current/omni_full_bench_48_rebuilt_runbench.log
```

Latency bench findings:

- the fixture decodes to 80,620 samples at 16 kHz, about 5.04 seconds of audio;
- Parakeet pre-encoding produced 63 audio tokens with hidden width 2688 in
  about 44-48 ms;
- the current release-built latency bench resolves sampling from the bundle's
  `generation_config.json`, not a hardcoded greedy fallback:
  `temperature=0.600`, `top_p=0.950`, `top_k=0`, `min_p=0.000`,
  `repetition_penalty=1.000`;
- raw and pre-encoded BatchEngine paths both stream; pre-encoded first-delta
  is about 172-186 ms on the explicit/current prompts, versus about 236 ms for
  the current raw-audio default prompt;
- raw and pre-encoded TokenIterator paths also stream; pre-encoded first-delta
  is about 156-163 ms, versus about 186-195 ms for raw audio;
- the cache directory contains safetensors entries, `cache_index.db`, and
  `ssm_companion`, proving the disk/SSM cache side is being exercised.

Coherency boundary:

- with the explicit prompt `What do you hear in the audio? Answer in one
  concise sentence.`, all four raw/pre-encoded BatchEngine/TokenIterator rows
  correctly identify the fixture as a single sharp high-pitched electronic
  beep, notification, or alert;
- at 48 tokens the answer repeats the concise sentence twice, so this is
  coherent audio grounding but not a clean long-budget termination pass;
- at 192 tokens, `OmniBench` still records repeated-bigram failures on several
  media rows. That remains an engine/runtime stop or continuation issue to
  root-cause, not a reason to clamp sampler settings.

Chunk stability findings:

- independent Parakeet chunk embeddings are not concat-safe;
- every prefix/full comparison required rollback at the default tolerance;
- live voice should retain PCM and either pre-encode the full current snapshot
  or pass raw PCM for the model turn. Do not concatenate independently encoded
  chunk embeddings into the model context.

OmniBench generation defaults correction:

- the older 192-token Omni aggregate forced greedy `temperature=0.0`, which
  bypassed the bundle `generation_config.json`;
- `OmniBench` now resolves sampling from the model's generation defaults by
  default (`temp=0.600`, `topP=0.950`, `rep=1.000` for the current JANGTQ
  bundle) and only uses greedy when `BENCH_OMNI_GREEDY=1` is explicitly set;
- failure diagnostics now print the repeated phrase and output excerpt instead
  of only reporting `repeated bigram loop`.

Fresh generation-config artifact:

```text
docs/local/live-model-matrix/20260517T_omni_generation_config_fix/omni.out
```

Result at `BENCH_MAX_TOKENS=192`, `BENCH_OMNI_RANDOM_SEED=20260517`,
`BENCH_OMNI_BATCH=1`:

- 12/18 rows pass;
- text-only, text multi-turn, audio encoder, audio LMInput, reasoning OFF,
  reasoning toggle, mixed image+audio, media-salt isolation, hybrid SSM parity,
  BatchEngine text B=1, and BatchEngine text B=2 pass;
- remaining failures are image/video long-budget continuation rows and one
  BatchEngine audio row, with explicit repeated-phrase diagnostics;
- this improves the evidence path but still does not make Omni media
  production-clear at long budgets.

Current 48-token full Omni matrix after rebuilding `RunBench`:

- 17/18 rows pass on
  `Nemotron-Omni-Nano-JANGTQ-CRACK` with bundle generation defaults;
- passing rows include text single-turn, text multi-turn, image single-turn,
  image reasoning-off direct, video encoder, audio encoder, video LMInput,
  audio LMInput, reasoning OFF, reasoning toggle, mixed image+audio,
  media-salt isolation, hybrid SSM parity, BatchEngine text B=1/B=2,
  BatchEngine image B=1, and BatchEngine audio B=1;
- the remaining failure is `image multi-turn x2` with default thinking enabled,
  where the output loops on decoded image-placeholder text (`br br`). This is
  not an audio/Parakeet failure and must remain visible as a non-audio Omni
  media/runtime blocker. Do not hide it with sampler clamps.

Fresh live voice recheck at 2026-05-17 08:10 PDT:

```text
docs/local/live-model-matrix/20260517T_omni_live_voice_recheck_now/
```

- Release builds passed for `OmniAudioLatencyBench`,
  `OmniAudioChunkStabilityBench`, and `RunBench`.
- `swift test --filter NemotronHOmniPreEncodedAudioTests` is blocked before
  the focused Omni tests execute because this local CLI toolchain cannot import
  Swift `Testing`; the failing log is preserved as a test-runner/toolchain
  issue, not as runtime evidence.
- `Nemotron-Omni-Nano-JANGTQ4-CRACK` full Omni `RunBench` passed 18/18 rows at
  48 tokens using bundle generation defaults. Load was 1.92 s; direct decode
  rows were about 92-113 tok/s and BatchEngine rows about 38-68 tok/s.
- `Nemotron-Omni-Nano-MXFP4-CRACK` full Omni `RunBench` now also passes 18/18
  rows at 48 tokens in
  `docs/local/live-model-matrix/20260517T2215_nemotron_mxfp4_omni_recheck/nemotron_mxfp4_omni.log`.
  Load is 2.17 s with `NemotronHOmni` and `NemotronHOmniProcessor`; bundle
  defaults resolve to `temp=0.600 topP=0.950 topK=0 minP=0.000 rep=1.000`.
  Direct rows cover text single-turn, three-turn text, image, video/audio
  encoder, video/audio LMInput, reasoning OFF, ON/OFF/ON toggle, mixed
  image+audio, media-salt isolation, and hybrid SSM warm-pass parity at about
  127-137 tok/s. BatchEngine text/image/audio rows pass at about 50-82 tok/s.
- JANGTQ4 live audio path streamed both raw PCM and pre-encoded Parakeet
  embeddings through BatchEngine and TokenIterator. Parakeet pre-encode was
  46.2 ms for 63 audio tokens, and first-delta latency improved from raw
  192-227 ms to pre-encoded 160-179 ms depending on path.
- The other local Omni bundles also passed the live-audio smoke:
  `Nemotron-Omni-Nano-JANGTQ-CRACK` pre-encoded in 43.9 ms and streamed all
  four path/mode rows at 32 tokens; `Nemotron-Omni-Nano-MXFP4-CRACK`
  pre-encoded in 48.1 ms and streamed all four path/mode rows at 32 tokens.
- Cache proof exists for the audio benches: the emitted cache directories
  contain `cache_index.db`, safetensors block entries, and `ssm_companion`
  directories; see `cache_artifacts_listing.txt`.
- Chunk stability remains negative by design: 10/10 prefix comparisons were
  unstable at default tolerance, so production live voice must retain PCM and
  refresh the full current pre-encode, or submit raw PCM. Concatenating
  independently encoded Parakeet chunks would be wrong.
- Coherency remains partial at longer audio budgets: answers are audio-grounded,
  but some 48-token rows repeat concise sentences or continue to the token cap.
  This is an honest runtime/termination boundary, not a reason to add hidden
  sampling or forced-stop guards.

Fresh current-verify live voice recheck at 2026-05-17 08:56 PDT:

```text
docs/local/live-model-matrix/20260517T155603Z_omni_live_voice_current_verify/
```

- Xcode-backed `NemotronHOmniPreEncodedAudioTests` passed 8/8. The plain CLI
  `swift test` toolchain issue remains a command-selection problem; the current
  passing command includes the Xcode framework search path.
- Release builds passed for `OmniAudioLatencyBench`,
  `OmniAudioChunkStabilityBench`, and `RunBench`.
- JANGTQ4 live audio streamed raw PCM and pre-encoded Parakeet embeddings
  through BatchEngine and TokenIterator using bundle defaults. Parakeet
  pre-encode was 50.1 ms for 63 x 2688 embeddings. First-delta latency was
  203.5-219.3 ms raw BatchEngine, 176.0-188.7 ms pre-encoded BatchEngine,
  184.6-188.5 ms raw TokenIterator, and 157.1-157.7 ms pre-encoded
  TokenIterator.
- Cache proof exists for the audio bench: emitted cache dirs contain
  `cache_index.db`, safetensors block entries, and `ssm_companion` artifacts.
- Chunk stability remains intentionally negative: 10/10 prefix comparisons were
  unstable at default tolerance, so concatenating independently encoded
  Parakeet chunks remains invalid.
- Integrated `BENCH_OMNI=1` + `BENCH_OMNI_BATCH=1` passed 18/18 at
  `maxTokens=48`, covering text, image, video, audio, reasoning on/off, mixed
  image+audio, media-salt isolation, hybrid SSM warm-pass, and BatchEngine
  text/image/audio rows.

Fresh current-checkout live voice recheck at 2026-05-17 09:31 PDT:

```text
docs/local/live-model-matrix/20260517T163112Z_omni_live_voice_reverify_current/
```

- Xcode-backed `NemotronHOmniPreEncodedAudioTests` compiled and passed 8/8:
  retained live audio buffer snapshots, caller-supplied Parakeet embedding
  preservation, EVS placeholder count, RADIO pixel shuffle, Parakeet relative
  shift, projector remaps, source weight transposes, and bundle-default
  sampling plumbing.
- `OmniAudioLatencyBench` and `OmniAudioChunkStabilityBench` rebuilt in
  release mode from this checkout.
- JANGTQ4 live audio at 32 tokens loaded in 3.06 s, decoded the 5.04 s fixture,
  applied bundle defaults (`temp=0.600 topP=0.950 topK=0 minP=0.000
  rep=1.000`), and pre-encoded Parakeet to `63 x 2688` in 59.8 ms. Raw PCM and
  pre-encoded audio both streamed through BatchEngine and TokenIterator.
  First-delta / tok/s rows:
  - BatchEngine raw: 221.4 ms, 64.2 tok/s;
  - BatchEngine pre-encoded: 178.7 ms, 72.0 tok/s;
  - TokenIterator raw: 182.2 ms, 71.1 tok/s;
  - TokenIterator pre-encoded: 153.1 ms, 75.9 tok/s.
- Repeated JANGTQ4 audio turns with cache OFF were clean across 12/12
  raw/pre-encoded BatchEngine/TokenIterator rows: grounded audio text, no media
  marker leak, and 66.0-75.7 tok/s. Repeated turns with disk cache ON mostly
  remained grounded but exposed a real output-quality edge: one sampled
  TokenIterator pre-encoded cache-reuse row emitted sound marker text and a
  few sampled rows were weak/non-grounded. Treat cache-on repeated live audio as
  PARTIAL until the cache-hit quality gate is tightened and the root cause is
  isolated. Do not hide this with sampler clamps, forced stop tokens, or
  post-hoc text cleanup.
- `omni_audio_chunk_stability_jangtq4.log`: 10/10 prefix comparisons remain
  not concat-safe at default tolerance. The live voice contract is still
  retained PCM plus full-snapshot pre-encode, or raw PCM at endpoint.
- `omni_runbench_jangtq4_48.log`: integrated `BENCH_OMNI=1`
  `BENCH_OMNI_BATCH=1` passed 18/18 at `maxTokens=48`. Load was 1.76 s, direct
  decode rows were 95.2-113.6 tok/s, and BatchEngine rows were 45.2-71.0
  tok/s. This proves the core JANGTQ4 Omni path across text, image, video,
  audio, mixed media, media salt, hybrid SSM warm-pass, and BatchEngine rows.
- Short 16-token raw/pre-encoded smoke rows for the sibling JANGTQ and MXFP4
  bundles loaded and streamed all four path/mode combinations, but the visible
  answers were too weak to count as coherency proof. Use JANGTQ4 as the current
  live-voice production candidate; do not promote JANGTQ or MXFP4 live-audio
  rows without longer grounded repeat gates.

Fresh source-wrapper fix and recheck at 2026-05-17 10:06 PDT:

```text
docs/local/live-model-matrix/20260517T170614Z_omni_live_voice_fresh_recheck/
```

- Root cause fixed: Swift was emitting audio media placeholders as literal
  `<sound>` and `</sound>` text around `<so_embedding>` slots. The bundled
  Nemotron processor uses `<so_start>` and `<so_end>` wrapper tokens. The
  processor now emits source-compatible wrapper tokens; this is a real
  tokenization/parity fix, not a sampler guard or text cleanup.
- `processor_audio_wrapper_red.log`: the focused regression failed before the
  fix because token `95690` from the literal sound marker was present and
  wrapper token IDs `28`/`29` were missing.
- `processor_audio_wrapper_green.log` passes after the fix, and
  `NemotronHOmniPreEncodedAudioTests_after_audio_wrapper_fix.log` passes the
  full focused Omni/Parakeet suite at 9/9.
- `omni_audio_latency_jangtq4_both_paths_cache_off_32_after_audio_wrapper_fix.jsonl`
  loads JANGTQ4 in 1.91 s, uses bundle defaults from `generation_config.json`,
  pre-encodes the 5.04 s fixture to `63 x 2688` in 46.7 ms, and streams
  raw PCM plus pre-encoded audio through BatchEngine and TokenIterator at
  65.4-76.1 tok/s. No literal sound-marker leak appears.
- `omni_audio_latency_jangtq4_both_paths_cache_off_32_repeats3_after_audio_wrapper_fix.jsonl`
  repeats all four path/mode rows three times with cache OFF. No marker leak
  appears in 12/12 rows. Iterator raw/pre-encoded rows are mostly grounded
  around note/chime/beep descriptions at 70.6-76.6 tok/s. BatchEngine
  pre-encoded turns 2 and 3 remain weak, so this is not a full audio-quality
  promotion for every short stochastic row.
- `omni_runbench_jangtq4_48_after_audio_wrapper_fix.log` passes the integrated
  `BENCH_OMNI=1` `BENCH_OMNI_BATCH=1` matrix: 18 passed, 0 failed at
  `maxTokens=48`. Direct audio grounds the fixture as a sharp electronic tone;
  mixed image+audio mentions a distinct synthetic/electronic sound. BatchEngine
  audio B=1 is still structurally passing but semantically weak.
- `omni_audio_chunk_stability_jangtq4_after_audio_wrapper_fix.jsonl` remains
  intentionally negative: `chunk_concat_safe_default=false` and 10 unstable
  comparisons. Retain PCM and pre-encode the full current snapshot, or send raw
  PCM at endpoint. Do not concatenate independently encoded Parakeet chunks.

## Required Proof Per Active Bundle

For each non-excluded bundle, the production row must include:

- no-load inventory with architecture, model type, quant format, MTP tensor
  evidence, VL/Omni profile, and `generation_config.json` sampling defaults;
- live config/template smoke;
- multi-turn visible coherent answer with token/s and normal stop reason;
- reasoning on/off proof for reasoning-capable families, with no visible
  reasoning leak when off and no forced close when on;
- cache OFF and cache ON rows;
- cache stats for the applicable topology: prefix, paged KV, block disk L2,
  TurboQuant KV, SSM companion, Zaya CCA, or media salt;
- BatchEngine B=1 and real B=2 overlap where the family supports text batching;
- VL rows for VL bundles: image turn, text-only continuation, different-image
  miss, same-image hit, and media-salt isolation;
- Omni rows for Nemotron: text, image, audio, video where fixtures are present;
- MTP rows only when tensor evidence exists and the family has a verified
  runtime path.

## Non-Negotiables

- Do not infer MTP from a model name.
- Do not enable native MTP automatically from metadata alone.
- Do not hide loops with repetition-penalty or temperature clamps.
- Do not patch reasoning output by inserting close tags or moving hidden
  content into visible content.
- Do not treat load success, cache metadata, or a length-capped answer as a
  production pass.
- If a model is incoherent, mark the row failed and root-cause the runtime,
  cache, template, or decode path.
