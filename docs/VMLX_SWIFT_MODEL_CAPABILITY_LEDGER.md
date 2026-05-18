# vMLX Swift Model Capability Ledger

Snapshot date: 2026-05-14.

This ledger is about the full `vmlx-swift` engine, not a wrapper. A model row is
not production-ready unless it has a live multi-turn coherence artifact. Config
and template checks only prove that the bundle is readable and the prompt path
renders.

Scope update, 2026-05-17: Kimi remains excluded by user direction, but DSV4
Flash and the rest of the non-Kimi local `~/models` inventory are back in the
active Osaurus switch-readiness matrix. Rows that were previously marked
historical or de-scoped remain historical until they are re-run under the
current non-Kimi matrix; do not promote them from stale evidence.

## Status Key

- `PASS`: exercised live or by a focused gate and met the row contract.
- `PARTIAL`: some required behavior works, but another required behavior failed
  or is not proven.
- `FAIL`: the row hit a concrete engine/runtime error.
- `TODO`: local bundle exists, but live behavior has not been run yet.
- `N-A`: not applicable for that model family.

## Artifact Index

Local-only artifacts live under `docs/local/swift-release-gates/model-ledger/`.
Focused fix artifacts live under `docs/local/swift-release-gates/dsv4-fixes/`.

- Config smoke summary: `config-smoke/summary.tsv`.
- Template/kwargs smoke summary: `template-smoke/summary.tsv`.
- DSV4 tools fallback pre/post:
  `dsv4-fixes/pre_dsv4_template_tools.log`,
  `dsv4-fixes/post_clean_dsv4_template_tools_jangtq_k.log`,
  `dsv4-fixes/post_clean_dsv4_template_tools_jangtq2.log`.
- Focused Swift tests:
  `dsv4-fixes/post_dsv4_template_focused_test.log`,
  `dsv4-fixes/post_dsv4_reasoning_policy_test.log`,
  `dsv4-fixes/focused_cache_topology_test.log`,
  `dsv4-fixes/focused_vl_shape_guard_test.log`,
  `dsv4-fixes/focused_jangtq_hadamard_test.log`,
  `dsv4-fixes/focused_jangtq_hadamard_rank234_test.log`,
  `dsv4-fixes/focused_mlxlmcommon_final_test.log`,
  `dsv4-fixes/pre_zaya_vl_jangtqk_nested_bits_test.log`,
  `dsv4-fixes/post_zaya_vl_jangtqk_nested_bits_test.log`,
  `dsv4-fixes/pre_kimi_jinja_tojson_separators_test.log`,
  `dsv4-fixes/post_kimi_jinja_tojson_separators_test.log`,
  `dsv4-fixes/pre_kimi_enable_thinking_alias_gate.log`,
  `dsv4-fixes/post_kimi_enable_thinking_alias_gate.log`,
  `dsv4-fixes/focused_after_zaya_kimi_template_fixes.log`,
  `dsv4-fixes/focused_zaya_vl_tool_template_after_sidecar_regression_test.log`,
  `dsv4-fixes/focused_mlxlmcommon_after_dsv4_zaya_zayavl_fixes.log`.
- Live DSV4:
  `dsv4-fixes/post_dsv4_jangtqk_template_kwargs.log`,
  `dsv4-fixes/post_dsv4_jangtq2_template_kwargs.log`,
  `dsv4-fixes/live_dsv4_jangtqk_chat_multiturn_coherence_tokps.log`,
  `dsv4-fixes/live_dsv4_jangtqk_growing_chat_cache.log`,
  `dsv4-fixes/live_dsv4_jangtqk_reasoning_modes_tokps.log`,
  `dsv4-fixes/live_dsv4_jangtq2_reasoning_modes_tokps.log`,
  `dsv4-fixes/live_dsv4_jangtqk_fim_vs_chat_probe.log`,
  `dsv4-fixes/live_dsv4_jangtqk_fim_vs_chat_arithmetic_probe_rerun.log`,
  `dsv4-fixes/live_dsv4_jangtqk_reasoning_modes_tokps_after_prompt_gate.log`,
  `dsv4-fixes/live_dsv4_jangtqk_reasoning_modes_tokps_after_max_prompt_gate.log`.
- Live ZAYA text: `live/ZAYA1-8B-JANGTQ_K/jpreg.log`,
  `live/ZAYA1-8B-JANGTQ_K/batch_chat.log`,
  `dsv4-fixes/live_zaya_jangtq_growing_chat_cache_diagnostic.log`,
  `dsv4-fixes/live_zaya_jangtqk_native_growing_chat_cache.log`,
  `dsv4-fixes/live_zaya_jangtq4_native_growing_chat_cache.log`,
  `dsv4-fixes/live_zaya_jangtqk_native_growing_chat_cache_semantic_gate.log`,
  `dsv4-fixes/live_zaya_jangtqk_native_growing_chat_cache_recall_phrase.log`,
  `dsv4-fixes/live_zaya_jangtq4_native_growing_chat_cache_recall_phrase.log`.
- Live Qwen3.6 text/hybrid: `live/Qwen3.6-27B-MXFP4-CRACK/jpreg.log`.
- Live MiniMax text/JANGTQ: `live/MiniMax-M2.7-Small-JANGTQ/jpreg.log`;
  current large CRACK cache gate:
  `live-model-matrix/20260518T_minimax_m27_jangtqk_crack_turnmatrix_strict_tq_gate/`;
  final focused post-macro chat cache/TQ proof:
  `live-model-matrix/20260518T_minimax_m27_jangtqk_growing_chat_cache_fail_loud_macro/`,
  `live-model-matrix/20260518T_minimax_m27_jangtqk_tq_b2_strict_fail_loud_macro/`,
  `live-model-matrix/20260518T_minimax_m27_jangtqk_tq_tail_fix_exact/`.
- Live ZAYA-VL MXFP4:
  `live/ZAYA1-VL-8B-MXFP4/vl_batch_chat.log`,
  `live/ZAYA1-VL-8B-MXFP4/vl_chat_cache.log`.
- Live ZAYA-VL JANGTQ_K:
  `live/ZAYA1-VL-8B-JANGTQ_K/vl_chat_cache.log`.
- ZAYA-VL sidecar/tools template fix:
  `dsv4-fixes/debug_zaya_vl_jangtqk_template_smoke_direct_render.log`,
  `dsv4-fixes/post_zaya_vl_jangtqk_tool_template_smoke_after_sidecar_fix.log`,
  `dsv4-fixes/post_zaya_vl_jangtq4_tool_template_smoke_after_sidecar_fix.log`.

## Live Multi-Turn Rows First

| Model | Family / Swift model | Live result | What worked | What did not pass yet |
| --- | --- | --- | --- | --- |
| `JANGQ/DeepSeek-V4-Flash-JANGTQ2` | `deepseek_v4` / `DeepseekV4JANGTQModel` | `PARTIAL` | Current non-Kimi DSV4 artifact proves the prompt-boundary root cause and fix. Pre-fix, the system string glued to `<User>` and live chat drifted into `sappberry-42`; post-fix the standalone Jinja, compiled fallback, and Swift encoder all insert a newline separator, with 13/13 focused tests passing. Live cache OFF and cache ON 3-turn chat is coherent with `rep=1.0`, no raw `<think>` leakage, normal `.stop`, and visible tok/s. `BENCH_PROD` passes 7/7 using bundle defaults (`temp=1.000 topP=1.000 topK=0 rep=nil`) and shows reasoning on/off routing through `.reasoning` vs visible chunks. | Not low-footprint production-cleared: production row reports about 61.5 GiB peak RSS. DSV4 is `pagedIncompatible=true`; generic paged prefix hit is explicitly `N-A`, while disk L2 stats show `hits=1,misses=19,stores=14`. Long-context/vector drift, API routes, sleep/wake, and speed matrix remain open. |
| `JANGQ/DeepSeek-V4-Flash-JANGTQ-K` | `deepseek_v4` / `DeepseekV4JANGTQModel` | `PARTIAL` | Current post-fix 3-turn chat row passes on the second DSV4 bundle with the corrected system/User separator, visible `sapphire-42` recall, coherent follow-up, no raw reasoning leakage, `.stop`, and tok/s. Template kwargs for thinking off/on/max pass. | Needs the same bundle-default production/cache-stat/speed/long-context/API matrix as JANGTQ2 before promotion. The old pre-fix exact gate failed with `sappium-42`, so do not promote stale DSV4 evidence. |
| `JANGQ/ZAYA1-8B-JANGTQ4` | `zaya` / `ZayaModel` | `PASS` | Fresh current turnmatrix passes config/template, production defaults cache OFF/ON, BatchEngine single/chat/disk restore/concurrent/per-slot/TurboQuant B=2. Bundle defaults apply as `temp=0.600 topP=1.000 topK=0 rep=nil`; reasoning ON/OFF flips produce visible answers, cache-on speed is about `64.7-66.3 tok/s`, peak RSS is about `5.1 GiB`, disk L2 has hits/stores, and SSM companion hits are recorded. | Generic paged prefix hit is `N-A` by topology because ZAYA CCA is path-dependent and disk/SSM-backed. |
| `JANGQ/ZAYA1-8B-JANGTQ_K` | `zaya` / `ZayaModel` | `PASS` | Fresh current turnmatrix `20260518T001613Z_zaya_jangtqk_current_turnmatrix/` passes config/template, production defaults cache OFF/ON, BatchEngine single/chat/disk restore/concurrent/per-slot/TurboQuant B=2. Bundle defaults apply as `temp=0.600 topP=1.000 topK=0 rep=nil`; reasoning ON/OFF flips produce visible answers with zero reasoning on OFF rows; cache-on speed is about `61-63 tok/s`, peak RSS is about `3.8 GiB`, disk L2 has hits/stores, and SSM companion hits are recorded. | Generic paged prefix hit is `N-A` by topology because ZAYA CCA is path-dependent and disk/SSM-backed. Config still warns that tokenizer EOS `106` is not in effective EOS `[1]`; keep this visible and do not paper over it with a sampler guard. |
| `dealign.ai/Qwen3.6-27B-MXFP4-CRACK` | `qwen3_5` / `Qwen35` | `PASS / HIGH-RES VIDEO WATCH` | Fresh current turnmatrix `20260517T_qwen36_27b_mxfp4_crack_video_resize_postfix_turnmatrix/` passes config/template, production defaults cache OFF/ON, BatchEngine single/chat/disk restore/concurrent/per-slot/TurboQuant B=2, VL batch chat, structured VL chat cache, media-salt isolation, and mixed text/image/video. Bundle defaults apply as `temp=1.000 topP=0.950 topK=20 minP=0.000 rep=nil`; MTP depth is `off`; reasoning ON/OFF closes with visible answers at a 2048-token budget; same-media image cache hits disk `99/99`; video passes with explicit `BENCH_VL_VIDEO_RESIZE=224` and `video pixels shape: [560, 1536]`. | Raw 1080p video is not production-cleared. The preserved pre-fix artifact peaked at 164.2 GiB physical footprint in MLX/Metal allocation before termination. Osaurus needs an explicit Qwen video media resize/token-budget setting; do not auto-enable MTP on this CRACK bundle. |
| `dealign.ai/Qwen3.6-27B-JANG_4M-CRACK` | `qwen3_5` / `Qwen35` | `PASS / HIGH-RES VIDEO WATCH` | Fresh current turnmatrix `20260518T000445Z_qwen36_27b_jang4m_crack_turnmatrix/` passes config/template, production defaults cache OFF/ON, BatchEngine single/chat/disk restore/concurrent/per-slot/TurboQuant B=2, VL batch chat, structured VL chat cache, media-salt isolation, and mixed text/image/video. Bundle defaults apply as `temp=1.000 topP=0.950 topK=20 minP=0.000 rep=nil`; MTP depth is `off`; production decode is about `28 tok/s`; peak RSS is about `14.4 GiB` cache-off and `15.5 GiB` cache-on; same-media image cache hits disk `99/99`; bounded video passes with `BENCH_VL_VIDEO_RESIZE=224` and `video pixels shape: [560, 1536]`. | Raw high-resolution video remains unproven, and this CRACK bundle must not auto-enable native MTP without tensor evidence. Generic paged cache hit is `N-A` by Qwen hybrid topology; disk L2 and SSM companion stats are the cache proof. |
| `dealign.ai/Qwen3.6-35B-A3B-JANGTQ-CRACK` | `qwen3_5_moe` / `Qwen35MoE` | `PASS / HIGH-RES VIDEO WATCH` | Fresh current turnmatrix `20260518T_qwen36_35b_jangtq_crack_turnmatrix/` passes config/template, production defaults cache OFF/ON, BatchEngine single/chat/disk restore/concurrent/per-slot/TurboQuant B=2, VL batch chat, structured VL chat cache, media-salt isolation, and mixed text/image/video. Bundle defaults apply as `temp=1.000 topP=0.950 topK=20 minP=0.000 rep=nil`; MTP depth is `off`; production decode is about `84-89 tok/s`; peak RSS is about `11.8 GiB`; same-media cache hits disk, disk L2 writes real `.safetensors` blocks, and mixed video passes with `BENCH_VL_VIDEO_RESIZE=224` and `video pixels shape: [560, 1536]`. | Raw high-resolution video remains unproven, and this CRACK bundle must not auto-enable native MTP without tensor evidence. Generic paged cache hit is `N-A` by Qwen hybrid topology; disk L2 and SSM companion stats are the cache proof. |
| `JANGQ/MiniMax-M2.7-Small-JANGTQ` | `minimax_m2` / `MiniMaxJANGTQModel` | `PARTIAL` | Loads in 9.7s; 3-turn chat is coherent; no loop; TQ disk round-trip passes; decode around 30.6 tok/s; tracked mmap buffers about 37 GB. | Thinking-on probe produced 483 chars reasoning and no visible answer. Activity Monitor-style footprint reaches about 38.2 GB, so this is not a low-RAM active-streaming pass. |
| `JANGQ/ZAYA1-VL-8B-JANGTQ4` | `zaya1_vl` / `Zaya1VL` | `PASS` | Fresh current turnmatrix passes config/template, production defaults cache OFF/ON, BatchEngine text rows, disk restore, B=2 concurrent/per-slot/TurboQuant B=2, VL batch chat, structured VL chat cache, and media-salt isolation. Bundle defaults apply as `temp=0.600 topP=1.000 topK=0 rep=nil`; cache-on speed is about `59.4-63.6 tok/s` on short turns, peak RSS is about `6.8 GiB`, same-media disk restore hits `97/97`, different-media probe misses correctly, and compile OFF/ON VL two-turn chat both ground the image and answer the color follow-up as `blue`. | Generic paged prefix hit is `N-A` by topology. Video remains `N-A` because `ZAYA1-VL video input is not implemented`; this is a family capability gap, not a failed implemented row. |
| `Osaurus/ZAYA1-VL-8B-MXFP4` | `zaya1_vl` / `Zaya1VL` | `PASS` | Release turnmatrix passes config/template, production defaults cache OFF/ON, BatchEngine rows, VL batch chat, structured chat cache, and media-salt isolation. Video is reported `N-A` because ZAYA1-VL processor does not implement video input. | None for implemented image/text/cache surfaces; video remains a family capability gap, not a failed row. |
| `JANGQ/ZAYA1-VL-8B-JANGTQ_K` | `zaya1_vl` / `Zaya1VL` | `PARTIAL / CURRENT BLOCKER` | Fresh current turnmatrix `20260518T_zaya_vl_jangtqk_current_turnmatrix/` passes config/template, BatchEngine single/chat/disk restore/concurrent/per-slot/TurboQuant B=2, VL batch chat, media-salt isolation, and text/image mixed row with video reported `N-A`. Focused decoder test preserves nested `mxtq_bits.routed_expert` widths as gate/up=2 and down=4, and kernel probes match CPU dequant on sampled layer/expert rows. Fresh root-cause artifact `20260518T_zaya_vl_jangtqk_rootcause_topk/` proves the failure is first-token logits, not decode policy: K ranks `6,7,8,4`, while JANGTQ4 ranks `4` first on the identical rendered prompt. The same artifact reflects 40 real mixed-bit modules and shows K/JANGTQ4 share 5,315 tensor keys, with sampled regular/down tensors byte-identical; the systematic bit-plan delta is gate/up 2-bit versus 4-bit. | Production defaults cache OFF and ON still fail the same math row: S1/S2 return `8` for `7+8-11`, and the structured VL cache cold-image row exhausts the larger 512-token budget. Current evidence points at a ZAYA1-VL K-profile fidelity gap, not sampler/cache/parser/EOS. Do not promote this K lane or hide it with temperature, top-k, repetition, forced-stop, or reasoning-closure guards. |
| `dealign.ai/Ling-2.6-flash-MXFP4-CRACK` | `bailing_hybrid` / `BailingHybridModel` | `PASS` | Current release turnmatrix passes config/template/MTP metadata, production defaults cache OFF/ON, BatchEngine single/chat/disk restore/concurrent/per-slot/TurboQuant B=2. Bundle defaults apply with `rep=nil`; disk L2 and SSM companion hits are recorded. | Generic paged prefix hit is `N-A` by topology because Ling/Bailing uses disk-backed restore. |
| `JANGQ/Hy3-preview-JANGTQ` | `hy_v3` / `Hy3Model` | `PASS` | Current release turnmatrix passes config/template/MTP metadata, production defaults cache OFF/ON, paged cache hit, disk restore, B=2 concurrent, per-slot sampler, and TurboQuant B=2. Bundle defaults apply as `temp=0.900 topP=1.000 topK=-1 minP=0.000 rep=nil`. | Cold first prompt is slow; JANGTQ_K needs a current all-non-Kimi matrix re-run before Osaurus promotion. |
| `JANGQ/Hy3-preview-JANGTQ_K` | `hy_v3` / `Hy3Model` | `PARTIAL / NEEDS CURRENT RE-RUN` | Eager load was killed, but active expert streaming now passes the short production matrix without a process-global model-dir override after `loadWeights` binds the loaded model directory. It skips 91,008 per-expert tensors, indexes 79 layers x 192 experts, and passes 7/7 at about 6.2 GiB RSS. | Needs a current all-non-Kimi matrix re-run before Osaurus promotion. Speed remains blocked at about 1.4 tok/s, and multi-model active streaming still needs a per-loaded-model store before Osaurus exposes simultaneous JANGTQ_K sessions. |
| `dealign.ai/MiniMax-M2.7-JANGTQ_K-CRACK` | `minimax_m2` / `MiniMaxJANGTQModel` | `PARTIAL / CACHE-CHAT + TQ B=2 PASS` | Fresh strict turnmatrix `20260518T_minimax_m27_jangtqk_crack_turnmatrix_strict_tq_gate/` passes config/template, production defaults cache OFF/ON, BatchEngine single/chat, disk restore, B=2 concurrent, per-slot sampler, and the new production-shaped growing-chat cache row. Bundle defaults apply as `temp=1.000 topP=0.950 topK=40 rep=nil`; MTP depth is `off`; reasoning OFF rows carry zero reasoning. The final post-macro focused cache proof `20260518T_minimax_m27_jangtqk_growing_chat_cache_fail_loud_macro/` records `Cache history-boundary counts: [47]`, disk hit `matched=47/83`, turn 2 `finish=stop`, text `vmlx-cache-green`, and prompt-time ratio `0.08`. The focused TQ tail proof `20260518T_minimax_m27_jangtqk_tq_tail_fix_exact/` shows strict TurboQuant(4,4) B=2 stops normally for plain+TQ and TQ+TQ with exact outputs and actual compression counters (`1` then `2`). | Raw token-prefix `batch_cache_hit` fails by design for this MiniMax template because the raw Q/A prompt length-stops, so it is not counted as production chat proof. The preserved pre-fix TQ artifact `20260518T_minimax_m27_jangtqk_tq_b2_strict_fail_loud_macro/` showed length-stop drift when live TQ compressed the active prompt tail. The real fix is codec-side exact tail preservation and delayed middle-span compression, not a sampler/repetition guard. Low-footprint active-routed proof remains open. |
| `dealign.ai/MiniMax-M2.7-JANG_K-CRACK` | `minimax_m2` / `MiniMaxModel` | `PARTIAL / INFER PASS` | Fresh `20260518T001257Z_minimax_m27_jangk_crack_infer/` passes config/template plus cache-off `BENCH_PROD` 7/7. Bundle defaults apply as `temp=1.000 topP=0.950 topK=40 rep=nil`; MTP depth is `off`; reasoning ON/OFF alternation is coherent; decode is about `45-50 tok/s`; peak RSS is about `41.0 GiB`; loader logs an in-memory shape-inferred 6-bit metadata repair. | Needs full cache/multibatch/disk matrix and an explicit decision on whether the shape-inferred metadata repair is acceptable as production metadata handling or should be corrected in the bundle. |
| `dealign.ai/Gemma-4-26B-A4B-it-JANG_4M-CRACK` | `gemma4` / `Gemma4` | `PASS` | Text release turnmatrix passes config/template, cache OFF/ON `BENCH_PROD` 7/7, BatchEngine single/chat/disk restore/concurrent/per-slot/TurboQuant B=2. Structured VL chat-cache row passes: image A cold, same-image replay disk hit `308/308`, different-image miss, and text-only follow-up stays grounded. Live tool-call schema row passes through `UserInput.tools` with `get_weather({"location":"Tokyo"})`, `toolCalls=1`, and no raw marker leak. Long-budget single-turn Harmony reasoning on/off passes with 1420 reasoning chars ON and zero reasoning chars OFF. Fresh `20260517T_reasoning_turn_matrix_harness/` proves one loaded BatchEngine multi-turn ON/OFF/ON with prior assistant `reasoningContent` carried forward, and effort `low/medium/high/max` closes with visible output. | No active Gemma4 text/VL/tool/reasoning blocker from this row. GPT-OSS remains parser-contract only because no local GPT-OSS bundle is present. |
| `/Users/eric/osaurus_models/finished/gemma-4-e2b-it-4bit` | `gemma4` / `Gemma4` | `PASS` | Osaurus-local E2B bundle passes template smoke, `BENCH_PROD` cache OFF/ON, BatchEngine chat, TurboQuant B=2 isolation, VL chat/cache, VL batch chat, and `BENCH_REASONING_TURN_MATRIX=1` at realistic budgets. Fresh current refresh `20260518T_gemma4_e2b_refresh_no_fake_guards/` re-proves cache OFF 7/7, cache ON 7/7, 1536-token reasoning turn matrix, and structured VL chat cache. Bundle defaults apply as `temp=1.000 topP=0.950 topK=64 rep=nil`; cache-on rows use disk-backed restore because Gemma4 heterogeneous SWA/full-attention cache is paged-incompatible in this topology. The cache-on refresh records `disk{hits=1,misses=20,stores=14}`, while the reasoning matrix records `disk{hits=8,misses=0,stores=16}` and the VL row hits same-media disk restore `301/301`. | The retained 256-token thinking-on row failed by length before visible output; this is a real server/UI budget setting caveat, not a reason to force-close reasoning or inject sampler guards. This bundle is outside `~/models` and was tested because Osaurus logs reported E2B looping. |

Fresh parser/cache contract refresh:
`docs/local/production-readiness/20260517T2148_nonexcluded_parser_cache_refresh/`
passes 77 Swift Testing rows in 14 suites under the full Xcode toolchain. This
keeps the Ling/Bailing, Hy3/Hunyuan, Gemma4/Harmony, Mistral4/Pixtral, GLM5.1,
GPT-OSS, media-salt, hybrid companion, disk fallback, and no-forced-close
contracts current against the checked-out tree.

## Local Bundle Inventory

All listed bundles have `config.json`. Config smoke passed for every row in this
table. Template smoke status is from `BENCH_TEMPLATE_SMOKE=1`; it does not load
weights.

| Bundle | Size | model_type | Sidecar | Chat template | Template status | Live status |
| --- | ---: | --- | --- | --- | --- | --- |
| `JANGQ/DeepSeek-V4-Flash-JANGTQ-K` | 80G | `deepseek_v4` | yes | fallback/native DSV4 | `PASS` - top-level tools, Osaurus-sized tools, thinking on/off, and max preface render | `PARTIAL` |
| `JANGQ/DeepSeek-V4-Flash-JANGTQ2` | 74G | `deepseek_v4` | yes | fallback/native DSV4 | `PASS` - top-level tools, Osaurus-sized tools, thinking on/off, and max preface render | `PARTIAL` |
| `JANGQ/Hy3-preview-JANGTQ` | 79G | `hy_v3` | yes | file | `PASS` | `PASS` |
| `JANGQ/Hy3-preview-JANGTQ_K` | 102G | `hy_v3` | yes | file | `PASS` | `PARTIAL` |
| `JANGQ/Kimi-K2.6-JANGTQ_K` | 328G | `kimi_k25` | yes | file | `PASS` - `tojson(separators=(',', ':'))` and `enable_thinking`/`thinking` alias render through tools and thinking-off rows | `TODO` |
| `JANGQ/Kimi-K2.6-Small-JANGTQ` | 143G | `kimi_k25` | yes | file | `PASS` - `tojson(separators=(',', ':'))` and `enable_thinking`/`thinking` alias render through tools and thinking-off rows | `TODO` |
| `JANGQ/Laguna-XS.2-JANGTQ` | 9.4G | `laguna` | yes | file | `PASS` | `PASS` |
| `JANGQ/MiniMax-M2.7-Small-JANGTQ` | 37G | `minimax_m2` | yes | file | `PASS` | `PARTIAL` |
| `JANGQ/ZAYA1-8B-JANGTQ4` | 4.6G | `zaya` | yes | file | `PASS` | `PASS` |
| `JANGQ/ZAYA1-8B-JANGTQ_K` | 3.4G | `zaya` | yes | file | `PASS` | `PASS` |
| `JANGQ/ZAYA1-VL-8B-JANGTQ4` | 6.3G | `zaya1_vl` | yes | tokenizer/sidecar shim | `PASS` - vision placeholders plus ZAYA XML tools and Osaurus-sized schemas render | `PASS` |
| `JANGQ/ZAYA1-VL-8B-JANGTQ_K` | 5.0G | `zaya1_vl` | yes | tokenizer/sidecar shim | `PASS` - vision placeholders plus ZAYA XML tools and Osaurus-sized schemas render | `PARTIAL` |
| `Osaurus/ZAYA1-8B-MXFP4` | 5.5G | `zaya` | no | file | `PASS` | `PASS` |
| `Osaurus/ZAYA1-VL-8B-MXFP4` | 7.1G | `zaya1_vl` | no | tokenizer | `PASS` | `PASS` |
| `Tencent/Hy3-preview` | 557G | `hy_v3` | no | file | `PASS` | `TODO` |
| `dealign.ai/Gemma-4-26B-A4B-it-JANG_4M-CRACK` | 15G | `gemma4` | no | file | `PASS` | `PARTIAL` |
| `dealign.ai/Ling-2.6-flash-JANGTQ2-CRACK` | 29G | `bailing_hybrid` | yes | file | `PASS` | `PASS` |
| `dealign.ai/Ling-2.6-flash-MXFP4-CRACK` | 63G | `bailing_hybrid` | no | file | `PASS` | `PASS` |
| `dealign.ai/MiniMax-M2.7-JANGTQ_K-CRACK` | 74G | `minimax_m2` | yes | file | `PASS` | `PARTIAL` |
| `dealign.ai/MiniMax-M2.7-JANG_K-CRACK` | 80G | `minimax_m2` | no | file | `PASS` | `PARTIAL` |
| `dealign.ai/Nemotron-Omni-Nano-JANGTQ-CRACK` | 12G | `nemotron_h` | yes | file | `PASS` | `PASS` |
| `dealign.ai/Nemotron-Omni-Nano-JANGTQ4-CRACK` | 19G | `nemotron_h` | yes | file | `PASS` | `PASS` |
| `dealign.ai/Nemotron-Omni-Nano-MXFP4-CRACK` | 21G | `nemotron_h` | no | file | `PASS` | `PASS` |
| `dealign.ai/Qwen3.6-27B-JANG_4M-CRACK` | 16G | `qwen3_5` | no | file | `PASS` | `PASS` |
| `dealign.ai/Qwen3.6-27B-MXFP4-CRACK` | 14G | `qwen3_5` | no | file | `PASS` | `PARTIAL` |
| `dealign.ai/Qwen3.6-35B-A3B-JANGTQ-CRACK` | 11G | `qwen3_5_moe` | yes | file | `PASS` | `PASS` |
| `Qwen3.5-35B-A3B-4bit` | 19G | `qwen3_5_moe` | no | file | `PASS` | `PASS` |

## Engine Function Coverage By Family

### DeepSeek V4 Flash

- Local bundles: `DeepSeek-V4-Flash-JANGTQ-K`, `DeepSeek-V4-Flash-JANGTQ2`.
- Swift dispatch: `deepseek_v4` through `DeepseekV4ForCausalLM`.
- Cache topology: DSV4-specific cache path with rotating/sliding layers,
  long-context mode, paged/disk coordinator compatibility, and DSV4 finalizer
  controls. This still needs live long-context proof in this checkout.
- JANGTQ/TurboQuant: sidecar present; routed experts should use TurboQuant
  codebook path when the bundle and runtime select it.
- Template/reasoning kwargs: plain, `enable_thinking=false`,
  `enable_thinking=true`, `reasoning_effort=max`, and multi-turn history render.
- Current template status: fixed in this checkout. The fallback now renders the
  DSML tool schema block for both normal and Osaurus-sized top-level OpenAI
  tools, and `reasoning_effort=max` passes through to the DSV4 max preface
  without an environment-gated downgrade. The current non-Kimi pass also fixes
  the DSV4 system/User boundary in all three prompt paths; pre-fix logs show
  `system<User>` glue causing `sappberry/sappium` drift, while post-fix logs
  show `system\n<User>`.
- Current live proof: JANGTQ2 passes cache OFF/ON 3-turn chat and bundle-default
  `BENCH_PROD` 7/7; JANGTQ-K passes post-fix 3-turn chat. Reasoning on/off
  routes correctly, visible output is coherent, stop reason is `.stop`, no raw
  reasoning leaks, and tok/s is emitted. DSV4 cache stats are topology-specific:
  `pagedIncompatible=true`; generic paged prefix hits are `N-A`, while L2 disk
  stats are recorded when disk cache is enabled.
- Current live nuance: the older arithmetic gate used an ambiguous "7 + 5"
  wording and failed; the explicit `Q: What is 7 + 5? ... A:` gate now passes
  for reasoning off/on/max on JANGTQ-K. This is a gate prompt correction, not a
  hidden sampling clamp.
- Required live gates still open: long-context regression, vector drift,
  full speed matrix, API routes, low-footprint mode, and broader reasoning
  matrix.

### Qwen3.6 / Qwen3.5 Hybrid

- Local bundles: Qwen3.6 27B JANG/MXFP4, Qwen3.6 35B MoE JANGTQ, Qwen3.5 35B
  4bit.
- Swift dispatch: `qwen3_5` / `qwen3_5_moe` through Qwen35 text/VLM-compatible
  path.
- Cache topology: hybrid linear-attention layers use `MambaCache`; attention
  layers use `KVCacheSimple` or `RotatingKVCache`; BatchEngine has SSM warm-pass
  and companion-state support.
- Template/reasoning: template smoke passes; reasoning parser routes Qwen-style
  `<think>` output.
- Live proof: Qwen3.6 27B MXFP4 non-MTP now has a fresh current turnmatrix under
  `docs/local/live-model-matrix/20260517T_qwen36_27b_mxfp4_crack_video_resize_postfix_turnmatrix/`.
  It passes config/template, production cache OFF/ON using bundle defaults,
  BatchEngine single/chat/disk/concurrent/per-slot/TurboQuant B=2 rows, VL batch
  chat, structured VL cache, media-salt isolation, and mixed text/image/video.
  Reasoning ON/OFF closes with visible answers at a 2048-token production
  budget, same-media image cache hits disk `99/99`, and MTP remains `off`
  because this CRACK bundle has no native-MTP tensor claim. The preserved
  pre-fix artifact
  `docs/local/live-model-matrix/20260517T_qwen36_27b_mxfp4_crack_current_non_kimi_turnmatrix/`
  records the raw 1080p video failure: the row peaked at 164.2 GiB physical
  footprint inside MLX/Metal allocation before it was terminated. The current
  matrix therefore uses explicit `BENCH_VL_VIDEO_RESIZE=224` and logs
  `video pixels shape: [560, 1536]`; raw high-resolution video remains a
  throughput/resource gate, not a sampler/model-coherency failure.
- `Qwen3.6-27B-JANG_4M-CRACK` now has a fresh current turnmatrix under
  `docs/local/live-model-matrix/20260518T000445Z_qwen36_27b_jang4m_crack_turnmatrix/`.
  It passes config/template, production cache OFF/ON, BatchEngine
  single/chat/disk/concurrent/per-slot/TurboQuant B=2 rows, VL batch chat,
  structured VL cache, media-salt isolation, and mixed text/image/video.
  Bundle defaults apply as `temp=1.000 topP=0.950 topK=20 rep=nil`; MTP remains
  `off`; production decode is about `28 tok/s`; peak RSS is about `14.4 GiB`
  cache-off and `15.5 GiB` cache-on; same-media image restore hits disk
  `99/99`; `PROD_CACHE_STATS` records `pagedIncompatible=true`, disk L2
  hits/stores, and SSM companion hit. The bounded video row uses
  `BENCH_VL_VIDEO_RESIZE=224` and logs `video pixels shape: [560, 1536]`.
- `Qwen3.6-35B-A3B-JANGTQ-CRACK` now loads the VLM path as `Qwen35MoE` with
  `Qwen3VLProcessor`; image turns ground the red/blue gradient, text-only
  follow-up works, same-image media-salt restore hits, and different-image
  media-salt restore misses. Qwen3VL video processing now reads
  `video_preprocessor_config.json` and uses frame-count-aware video resize math;
  the fresh current turnmatrix
  `docs/local/live-model-matrix/20260518T_qwen36_35b_jangtq_crack_turnmatrix/`
  passes config/template, production cache OFF/ON, BatchEngine
  single/chat/disk/concurrent/per-slot/TurboQuant B=2 rows, VL batch chat,
  structured VL cache, media-salt isolation, and mixed text/image/video.
  Bundle defaults apply as `temp=1.000 topP=0.950 topK=20 rep=nil`; MTP remains
  `off`; production decode is about `84-89 tok/s`; peak RSS is about `11.8 GiB`;
  `PROD_CACHE_STATS` records `pagedIncompatible=true`, disk L2 hits/stores, and
  SSM companion hit. The bounded 1080p video row attaches `LMInput.video`, logs
  `BENCH_VL_VIDEO_RESIZE=224` and `video pixels shape: [560, 1536]`, and returns
  coherent visible content with thinking disabled.
- Qwen3.5 35B 4-bit now has loader, template, production defaults, VL, and
  BatchEngine mixed-codec proof. The pre-fix release turnmatrix artifact
  `docs/local/live-model-matrix/20260517T161305Z_release_turnmatrix_qwen35_35b_4bit/`
  was green except `batch_tq_b2`. The post-fix full turnmatrix
  `docs/local/live-model-matrix/20260517T164940Z_release_turnmatrix_qwen35_35b_4bit_after_compat_split/REPORT.md`
  passes all runnable rows: config, template, cache OFF/ON production defaults,
  BatchEngine single/chat/disk restore/concurrent/per-slot/TurboQuant B=2, VL
  batch chat, VL structured cache, media-salt isolation, and mixed
  text/image/video. Generic `batch_cache_hit` is `N-A` by topology/harness
  semantics. The high-res video turn is functionally coherent but slow
  (`TTFT 136312ms`), so throughput remains a watch item.
- Current issues: high-resolution Qwen3VL video is not production-cleared by the
  bounded 224px matrix. Osaurus wiring needs an explicit media resize/token
  budget setting for Qwen video, and any unbounded high-res row must be treated
  as a resource/scaling gate. Broader Qwen async rederive and API/sleep-wake
  coverage remain open across the family.

### MiniMax M2.7

- Local bundles: small JANGTQ plus larger CRACK JANG/JANGTQ_K rows.
- Swift dispatch: `minimax_m2`; JANGTQ bundles route to `MiniMaxJANGTQModel`.
- Cache topology: standard KV layers, BatchEngine cache coordinator, TQ disk
  serializer, and compiled decode path where cache types allow it.
- JANGTQ/TurboQuant: sidecar present; routed expert path uses
  `TurboQuantSwitchGLU`/codebook-style JANGTQ tensors.
- Template/reasoning/tools: template smoke passes; tool parser is MiniMax M2
  invoke/parameter format.
- Live proof: small JANGTQ has coherent 3-turn chat and TQ disk round-trip.
  Fresh large CRACK infer artifacts prove both JANGTQ_K and JANG_K pass
  cache-off production defaults with coherent reasoning ON/OFF flips, MTP off,
  no hidden repetition penalty, and about `45-50 tok/s`. The current JANGTQ_K
  cache artifact
  `docs/local/live-model-matrix/20260518T_minimax_m27_jangtqk_crack_turnmatrix_strict_tq_gate/`
  proves the production-shaped chat-cache path too: the MiniMax controllable
  template path now honors the corrected `enable_thinking=false` fallback at
  history-boundary render time, the coordinator hits the canonical history
  prefix (`47/83` tokens), disk L2 is used, and the warm turn stops cleanly with
  `vmlx-cache-green`. The focused TQ artifact
  `docs/local/live-model-matrix/20260518T_minimax_m27_jangtqk_tq_tail_fix_exact/`
  proves strict TurboQuant(4,4) B=2 with real compression enabled
  (`tqCompressionsA=1`, `tqCompressionsB=2`): the plain+TQ and TQ+TQ rows both
  stop normally and return the exact requested five-item answers.
- Current issues: the old raw token-prefix cache diagnostic is not a valid
  production chat proof for MiniMax; it fails because the raw Q/A prompt
  length-stops. The pre-fix strict `BENCH_BATCH_TQ_B2` artifact showed a real
  runtime/cache-codec incompatibility: live TQ compressed the active prompt tail
  and MiniMax missed the intended stop boundary. The fix preserves sink and
  recent prompt/decode tail tokens exactly, and only compresses the older middle
  span once it exists. No repetition penalty, temperature, top-k, or forced-stop
  policy was added. The JANGTQ_K row still peaks around `55.9 GiB`, and the
  JANG_K row logs an in-memory shape-inferred 6-bit metadata repair, so these
  are not full Osaurus promotion rows yet.

### ZAYA Text

- Local bundles: JANGTQ4, JANGTQ_K, MXFP4. JANGTQ_K now has a current
  all-non-Kimi turnmatrix artifact.
- Swift dispatch: `zaya` through `ZayaModel`.
- Cache topology: ZAYA CCA alternates `ZayaCCACache` with normal KV cache;
  BatchEngine has `BatchZayaCCACache`; disk serializer supports `zayaCCA`.
- JANGTQ/TurboQuant: sidecar present on JANGQ bundles; release rows prove
  TurboQuant-KV B=2 isolation for active JANGTQ4/MXFP4 rows, and the current
  JANGTQ_K turnmatrix proves the same implemented B=2 isolation row.
- Template/reasoning/tools: template smoke passes; tool parser is ZAYA XML;
  reasoning parser is ZAYA/think-XML style.
- Live proof:
  `docs/local/live-model-matrix/20260517T_zaya_jangtq4_current_non_kimi_turnmatrix/`
  freshly passes the JANGTQ4 text turnmatrix under the current all-non-Kimi
  scope: config/template, production defaults cache OFF/ON, BatchEngine
  single/chat, disk restore, B=2 concurrent, B=2 per-slot sampler, and
  TurboQuant-KV B=2. Bundle defaults apply (`temp=0.600`, `topP=1.000`,
  `topK=0`, `rep=nil`), reasoning ON/OFF flips produce visible answers, cache-on
  speed is about 64.7-66.3 tok/s, peak RSS is about 5.1 GiB, disk L2 and SSM
  companion hits are recorded, and generic paged cache is `N-A`.
  `docs/local/live-model-matrix/20260517T_release_turnmatrix_zaya_scope/`
  also passes the prior active text release turnmatrix for JANGTQ4 and MXFP4;
  `docs/local/live-model-matrix/20260518T001613Z_zaya_jangtqk_current_turnmatrix/`
  freshly passes the JANGTQ_K text turnmatrix under the current scope:
  config/template, production defaults cache OFF/ON, BatchEngine single/chat,
  disk restore, B=2 concurrent, B=2 per-slot sampler, and TurboQuant-KV B=2.
  Bundle defaults apply (`temp=0.600`, `topP=1.000`, `topK=0`, `rep=nil`),
  reasoning ON/OFF flips produce visible answers, cache-on speed is about
  61-63 tok/s, peak RSS is about 3.8 GiB, disk L2 and SSM hits are recorded,
  and generic paged cache is `N-A`.
- Current issue: generic paged prefix hit is `N-A` because ZAYA is
  paged-incompatible and uses disk-backed restore. Do not advertise generic
  paged hits for this topology.

### ZAYA VL

- Local bundles: JANGTQ4, JANGTQ_K, MXFP4. JANGTQ_K now has a current
  all-non-Kimi turnmatrix artifact, but remains a blocker.
- Swift dispatch: `zaya1_vl` through `Zaya1VL`.
- Cache topology: text side uses ZAYA CCA state; media path uses media salt and
  VLM processor state; BatchEngine has VL chat/cache matrix harnesses.
- Template/reasoning/tools: JANGQ ZAYA-VL sidecar bundles now pass template
  smoke for vision placeholders, thinking on/off, multi-turn history, top-level
  tools, and Osaurus-sized ZAYA XML schemas. The fix rewrites
  `tokenizer_config.json`, `chat_template.json`, and `chat_template.jinja` in the
  tokenizer shim because `swift-transformers` otherwise prefers the original
  sidecar and silently drops tools.
- Live proof:
  `docs/local/live-model-matrix/20260517T_zaya_vl_jangtq4_current_non_kimi_turnmatrix/`
  freshly passes the current JANGTQ4 VL turnmatrix under the current non-Kimi
  scope: config/template, production defaults cache OFF/ON, BatchEngine text
  rows, disk restore, B=2 concurrent/per-slot/TurboQuant B=2, VL batch chat,
  structured VL chat cache, and media-salt isolation. The VL chat row grounds
  the image with compile OFF and ON, answers the follow-up color as `blue`,
  hits disk restore for the same media (`97/97`), and correctly misses when the
  image changes. `docs/local/live-model-matrix/20260517T_release_turnmatrix_zaya_scope/`
  plus targeted reruns under
  `docs/local/live-model-matrix/20260517T_release_targeted_rerun_after_harness_fixes/`
  also pass the implemented ZAYA-VL image/text/cache surfaces for JANGTQ4 and MXFP4:
  production defaults cache OFF/ON, BatchEngine rows, VL batch chat, structured
  cache, and media-salt isolation. Video rows are explicitly `N-A` because this
  processor throws `ZAYA1-VL video input is not implemented`.
- Current K issue: `docs/local/live-model-matrix/20260518T_zaya_vl_jangtqk_current_turnmatrix/`
  improved the K lane but did not clear it. Config/template, BatchEngine
  single/chat/disk restore/B=2 concurrent/per-slot/TurboQuant, VL batch chat,
  media-salt isolation, and text/image mixed rows pass. Production defaults
  cache OFF and ON still fail S1/S2 by returning `8` for `7+8-11`, and the
  structured VL cache row exhausts a larger 512-token cold-image budget. Prior
  top-k evidence ranks the wrong first token before decoding, while sampled
  mixed-bit kernel probes pass, so this remains artifact/runtime root-cause
  work rather than a sampler, parser, cache, or forced-stop fix.
- Root-cause boundary refresh:
  `docs/local/live-model-matrix/20260518T_zaya_vl_jangtqk_rootcause_topk/`
  reruns the first-token top-k check and adds a tensor-boundary probe. K and
  JANGTQ4 render the same prompt IDs; K ranks `6,7,8,4`, while JANGTQ4 ranks
  `4` first. K reflects 40 `TurboQuantSwitchGLU` modules with `gateUp=2`,
  `down=4`, `seed=42`. K and JANGTQ4 share 5,315 tensor keys, sampled regular
  affine tensors are byte-identical, and sampled `down_proj` TQ tensors are
  byte-identical; the systematic bit-plan delta is K gate/up 2-bit versus
  JANGTQ4 gate/up 4-bit. Treat this as a K-profile fidelity blocker unless a
  future gate-safe mixed runtime profile is built and live-proven.

### Ling / Bailing Hybrid

- Local bundles: Ling JANGTQ2 CRACK and Ling MXFP4 CRACK.
- Swift dispatch: `bailing_hybrid` through `BailingHybridModel`.
- Cache topology: global/MLA layers use KV or rotating KV; linear-attention
  layers use `ArraysCache`; BatchEngine has `BatchArraysCache`.
- Template/reasoning/tools: template smoke passes; Bailing/Ling template context
  maps thinking controls; tool parser maps to GLM/deepseek-style format.
- Current status: `docs/local/live-model-matrix/20260517T170008Z_release_turnmatrix_ling_jangtq2/REPORT.md`
  and `docs/local/live-model-matrix/20260517T180538Z_release_turnmatrix_ling_mxfp4_current/REPORT.md`
  pass all runnable rows for the JANGTQ2 CRACK and MXFP4 CRACK bundles:
  config/template, MTP metadata, production defaults cache OFF/ON, BatchEngine
  single/chat, disk restore, B=2 concurrent, B=2 per-slot sampler, and
  TurboQuant-KV B=2. Bundle defaults resolve to `temp=0.600`, `topP=1.000`,
  `topK=0`, `minP=0.000`, `rep=nil`; JANGTQ2 release decode telemetry is about
  37-39 tok/s and MXFP4 is about 9.7-10.1 tok/s. Cache ON records disk L2 and
  SSM hits with `pagedIncompatible=true`; generic paged prefix hit is `N-A`.
- Boundary: Ling/Bailing is current-text-matrix green for JANGTQ2 and MXFP4.
  Native MTP remains fail-closed until a family-specific verified runtime
  profile exists.

### Gemma 4

- Local bundle: `Gemma-4-26B-A4B-it-JANG_4M-CRACK`.
- Swift dispatch: `gemma4` through Gemma 4 text/VLM-compatible path.
- Cache topology: full-attention layers use simple or max-sized rotating KV;
  sliding layers use `RotatingKVCache`.
- Template/reasoning/tools: template smoke passes; Gemma 4 harmony/channel
  reasoning parser and Gemma 4 tool parser are wired.
- Current status: `docs/local/live-model-matrix/20260517T160608Z_release_turnmatrix_gemma4_26b/`
  passes the text release turnmatrix: config/template, cache OFF/ON
  `BENCH_PROD` 7/7, BatchEngine single/chat, disk restore, B=2 concurrent,
  B=2 per-slot sampler, and TurboQuant-KV B=2 isolation. Cache ON is
  `pagedIncompatible=true`, so paged counters remain zero by design while disk
  L2 records a real hit/store row. The live structured VL cache row
  `docs/local/live-model-matrix/20260517T210417Z_gemma4_vl_chat_cache/gemma4_vl_chat_cache.out`
  passes with image A cold, same-image replay disk hit `308/308`, different
  image miss, and a grounded text-only follow-up; peak footprint in that row is
  about 30.6 GB. The live tool-call schema row
  `docs/local/live-model-matrix/20260517T212204Z_gemma4_batch_toolcall_real_schema/gemma4_batch_toolcall.out`
  passes with `Tool format: gemma4`, `Reasoning stamp: harmony`, one structured
  `get_weather({"location":"Tokyo"})` call, `stop`, `genTokens=14`, zero visible
  chunks, and no raw marker leak. The long-budget Harmony reasoning row
  `docs/local/live-model-matrix/20260517T2150_gemma4_harmony_long_reasoning/`
  passes the explicit thinking-on path with `1420` reasoning chars, `668`
  visible-content chars, normal EOS, and `77.9 tok/s`; the inverse
  `enable_thinking=false` row reports `0` reasoning chars, `547` visible-content
  chars, normal EOS, and `69.7 tok/s`. Remaining open row is a full multi-turn
  reasoning/API matrix, not single-turn Harmony leakage.

### Nemotron Omni / Nemotron H

- Local bundles: JANGTQ, JANGTQ4, MXFP4 CRACK.
- Swift dispatch: text config is `nemotron_h`; VLM/Omni route uses Nemotron H
  Omni processor when multimodal config requires it.
- Cache topology: hybrid SSM/cache state applies; Omni path also has image,
  video, audio, and pre-encoded audio handling.
- Template/reasoning/tools: template smoke passes; reasoning parser maps to
  Nemotron H; tool parser maps to Nemotron format.
- Current status: JANGTQ, JANGTQ4, and MXFP4 core Omni paths are live-proven. The
  fresh JANGTQ artifact
  `docs/local/live-model-matrix/20260517T214045Z_nemotron_jangtq_omni_recheck/omni_jangtq_48.log`
  passes 18/18 at 48 tokens with bundle generation defaults, `NemotronHOmni`
  plus `NemotronHOmniProcessor`, text-only, three-turn text, image, video
  encoder, audio encoder, video LMInput, audio LMInput, reasoning OFF,
  ON/OFF/ON reasoning toggle, mixed image+audio, media-salt isolation, hybrid
  SSM warm-pass parity, and BatchEngine text/image/audio rows. Direct rows
  report about 91-112 tok/s and BatchEngine rows about 47-86 tok/s. JANGTQ4
  core Omni path is live-proven in
  `docs/local/live-model-matrix/20260517T164618Z_omni_live_voice_current_recheck/`,
  `docs/local/live-model-matrix/20260517T164640Z_omni_live_audio_streaming_jangtq4_current/`,
  `docs/local/live-model-matrix/20260517T164702Z_omni_integrated_jangtq4_current/`,
  and `docs/local/live-model-matrix/20260517T170614Z_omni_live_voice_fresh_recheck/`.
  MXFP4 core Omni path is live-proven in
  `docs/local/live-model-matrix/20260517T2215_nemotron_mxfp4_omni_recheck/nemotron_mxfp4_omni.log`,
  which passes 18/18 at 48 tokens with bundle generation defaults, text-only,
  three-turn text, image, video/audio encoder, video/audio LMInput, reasoning
  OFF, ON/OFF/ON reasoning toggle, mixed image+audio, media-salt isolation,
  hybrid SSM warm-pass parity, and BatchEngine text/image/audio rows. Direct
  rows report about 127-137 tok/s and BatchEngine rows about 50-82 tok/s.
  The latest recheck fixed the audio wrapper parity bug: Swift now emits the
  bundled processor's `<so_start>`/`<so_end>` tokens around `<so_embedding>`
  slots instead of literal `<sound>` text. Focused pre-encoded audio tests pass
  9/9; release `BENCH_OMNI=1` `BENCH_OMNI_BATCH=1` passes 18/18 at 48 tokens;
  raw PCM and pre-encoded Parakeet stream through BatchEngine and TokenIterator
  with bundle defaults at 65.4-76.6 tok/s in cache-off rows.
  Chunked Parakeet embeddings are not concat-safe, so live voice must retain
  PCM and submit a full-snapshot pre-encode or raw PCM at endpoint.
- Open: cache-off repeated rows no longer leak literal sound markers after the
  wrapper fix, but some short BatchEngine/pre-encoded rows are still weak.
  Repeated audio with disk cache ON remains a separate output-quality edge from
  the earlier gate. Cache-on live audio stays PARTIAL until a focused root-cause
  gate is added.

### Kimi K2.6

- Local bundles: full K and small JANGTQ.
- Swift dispatch: `kimi_k25`; factory dispatches through the DeepSeek V3/Kimi
  family path for JANGTQ.
- Cache topology: standard attention cache with JANGTQ routed experts; likely
  rotating/full attention behavior follows Kimi config. Needs live inspection.
- Template/reasoning/tools: plain, thinking, reasoning max, history, and tool
  rows now render. Swift Jinja accepts Kimi's compact
  `tojson(separators=(',', ':'))`, and tokenizer context mirrors
  `enable_thinking` to `thinking` when the caller did not set `thinking`.
- Current status: config and template smoke pass, no live multi-turn row.

### Hy3

- Local bundles: JANGTQ, JANGTQ_K, full Tencent BF16. JANGTQ_K needs a current
  all-non-Kimi re-run before Osaurus promotion.
- Swift dispatch: `hy_v3` through `Hy3Model`.
- Cache topology: standard generation cache plus JANGTQ sidecar on JANGQ rows;
  JANGTQ is paged-compatible in the current row, while JANGTQ_K requires active
  expert streaming to avoid full eager materialization.
- Template/reasoning/tools: template smoke passes; reasoning/parser maps to Hy3
  / Hunyuan-style behavior.
- Current status: `docs/local/live-model-matrix/20260517T180931Z_release_turnmatrix_hy3_jangtq_current/REPORT.md`
  passes all runnable rows for `Hy3-preview-JANGTQ`: config/template/MTP
  metadata, production defaults cache OFF/ON, paged cache hit, disk restore,
  B=2 concurrent, per-slot sampler, and TurboQuant-KV B=2 isolation. The
  `Hy3-preview-JANGTQ_K` eager row under
  `docs/local/live-model-matrix/20260517T182455Z_release_turnmatrix_hy3_jangtqk_current/`
  was killed before load completion, but
  `docs/local/live-model-matrix/20260517T184132Z_hy3_jangtqk_streaming_autodir_after_fix/`
  passes 7/7 through active expert streaming at about 6.2 GiB RSS after
  `loadWeights` binds the loaded model directory. This K evidence is historical;
  if reopened, JANGTQ_K remains speed-blocked at about 1.4 tok/s.

### Laguna XS.2

- Local bundle: `Laguna-XS.2-JANGTQ`.
- Swift dispatch: `laguna` through `LagunaModel`.
- Cache topology: sliding layers use `RotatingKVCache`; full layers can use
  simple or full-sized rotating KV; comments indicate this was made
  compile-friendly.
- JANGTQ/TurboQuant: sidecar present; routed experts use JANGTQ/TurboQuant
  switch layers.
- Template/reasoning/tools: template smoke passes; tool parser maps to GLM-style
  parser; reasoning parser maps to Laguna.
- Current status: `docs/local/live-model-matrix/20260517T_release_turnmatrix_laguna_xs_after_b2_fix/REPORT.md`
  passes the text release turnmatrix: config/template, production defaults with
  cache OFF/ON, BatchEngine single/chat, disk restore, B=2 concurrent, B=2
  per-slot sampler, and TurboQuant-KV B=2 isolation. Bundle defaults apply
  (`temp=0.700`, `topP=0.900`, `topK=0`, `rep=nil`) and decode telemetry is
  about 31 tok/s on production rows. Generic paged prefix hit is `N-A` because
  Laguna is paged-incompatible and uses disk-backed restore.

## Immediate Engine Gaps Found

1. Current K-lane caveat: `ZAYA1-VL-8B-JANGTQ_K` remains a real blocker.
   The current turnmatrix now proves many implemented text/image/cache rows,
   but production defaults still fail the math row and structured VL cache row,
   and top-k evidence shows the wrong first token before decoding policy. It is
   not an active Osaurus switch blocker unless that specific K lane is exposed.
2. ZAYA-VL video remains `N-A` for JANGTQ4/MXFP4 because the processor does not
   implement video input. Text/image/cache surfaces are live-proven for those
   two bundles.
3. DSV4 template/tools, chat coherence, paged-incompatible disk cache, and
   explicit reasoning off/on/max arithmetic are fixed at current gate level, but
   DSV4 still needs long-context, vector-drift, broader matrix, and speed gates.
4. Qwen3.6 and MiniMax produce coherent multi-turn visible answers with
   thinking off/default chat, but some thinking-on probes spend the current
   budget in reasoning and need higher-budget closure proof.
5. MiniMax small JANGTQ does not satisfy low-footprint expectations in the live
   row: RSS stays low, but `phys_footprint` reaches full model scale.

## Columns Still Needed Per Model

Every remaining `TODO` row needs this exact live record:

- model path and bundle size.
- Swift model class and processor class.
- live multi-turn output excerpts, not only pass/fail.
- token/s and prompt-processing tok/s.
- `phys_footprint` and RSS.
- cache family from `newCache`: `KVCacheSimple`, `RotatingKVCache`,
  `MambaCache`, `ArraysCache`, `ZayaCCACache`, `CacheList`, `DeepseekV4`,
  `TurboQuantKVCache`.
- prefix/paged/disk cache hit or miss counters where the harness exposes them.
- TurboQuant KV encode/decode or TQ disk round-trip where applicable.
- JANGTQ sidecar and routed expert dispatch status.
- reasoning off/on/effort behavior with visible/reasoning separation.
- tool parser behavior with no raw marker leaks.
- chat-template kwargs: `enable_thinking`, `reasoning_effort`, tool schemas,
  multi-turn history, and family-specific kwargs.
- media salt and VL/audio/video grounding for multimodal models.
- async re-derive / SSM companion state for hybrid models.
- compile on/off behavior only if the cache family is compile-compatible.
