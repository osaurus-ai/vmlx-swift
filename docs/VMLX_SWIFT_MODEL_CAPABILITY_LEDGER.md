# vMLX Swift Model Capability Ledger

Snapshot date: 2026-05-14.

This ledger is about the full `vmlx-swift` engine, not a wrapper. A model row is
not production-ready unless it has a live multi-turn coherence artifact. Config
and template checks only prove that the bundle is readable and the prompt path
renders.

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
- Live MiniMax text/JANGTQ: `live/MiniMax-M2.7-Small-JANGTQ/jpreg.log`.
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
| `JANGQ/DeepSeek-V4-Flash-JANGTQ-K` | `deepseek_v4` / `DeepseekV4JANGTQModel` | `PARTIAL` | Live 3-turn chat with `enable_thinking=false` is coherent: saves `sapphire-42`, recalls it, answers the follow-up; no raw `<think>` leakage; stop reason is `.stop`; tok/s is emitted. DSV4 paged-incompatible cache restores through disk with salted hit and nil-salt miss. Explicit arithmetic prompts now pass for reasoning off/on/max; `reasoning_effort=max` reaches the model instead of being downgraded. | Long-context/vector drift, broader reasoning matrix, greedy/rep behavior on other DSV4 bundles, and full speed matrix still open. |
| `JANGQ/ZAYA1-8B-JANGTQ_K` | `zaya` / `ZayaModel` | `PASS` | Release turnmatrix passes config/template, production defaults cache OFF/ON, BatchEngine single/chat/disk restore/concurrent/per-slot/TurboQuant B=2. Bundle defaults apply, reasoning ON/OFF flips produce visible answers, disk L2 and SSM hits are recorded, and release decode is about 64-66 tok/s. | Older weak previous-answer diagnostic is superseded by the deterministic phrase/turnmatrix rows; generic paged prefix hit remains `N-A` by topology. |
| `dealign.ai/Qwen3.6-27B-MXFP4-CRACK` | `qwen3_5` / `Qwen35` | `PARTIAL` | Loads in 1.7s; 3-turn chat is coherent; no loop; SSM warm second-turn row recorded; avg prompt around 360 tok/s; decode around 21.7 tok/s. | Thinking-on probe produced 377 chars of reasoning and no visible answer within budget. Footprint rises to about 16.2 GB, expected for MXFP4 but not a low-footprint routed row. |
| `JANGQ/MiniMax-M2.7-Small-JANGTQ` | `minimax_m2` / `MiniMaxJANGTQModel` | `PARTIAL` | Loads in 9.7s; 3-turn chat is coherent; no loop; TQ disk round-trip passes; decode around 30.6 tok/s; tracked mmap buffers about 37 GB. | Thinking-on probe produced 483 chars reasoning and no visible answer. Activity Monitor-style footprint reaches about 38.2 GB, so this is not a low-RAM active-streaming pass. |
| `Osaurus/ZAYA1-VL-8B-MXFP4` | `zaya1_vl` / `Zaya1VL` | `PASS` | Release turnmatrix passes config/template, production defaults cache OFF/ON, BatchEngine rows, VL batch chat, structured chat cache, and media-salt isolation. Video is reported `N-A` because ZAYA1-VL processor does not implement video input. | None for implemented image/text/cache surfaces; video remains a family capability gap, not a failed row. |
| `JANGQ/ZAYA1-VL-8B-JANGTQ_K` | `zaya1_vl` / `Zaya1VL` | `PARTIAL` | Config/template, batch single/chat/disk restore/per-slot/TurboQuant B=2, and media-salt rows pass. Focused decoder test preserves nested `mxtq_bits.routed_expert` widths as gate/up=2 and down=4, and kernel probes match CPU dequant on sampled layer/expert rows. | Real coherence blocker remains: math/top-k evidence ranks the wrong first token before decoding, `prod_defaults` fails `7+8-11`, and structured VL cache exhausts the 192-token budget. |

## Local Bundle Inventory

All listed bundles have `config.json`. Config smoke passed for every row in this
table. Template smoke status is from `BENCH_TEMPLATE_SMOKE=1`; it does not load
weights.

| Bundle | Size | model_type | Sidecar | Chat template | Template status | Live status |
| --- | ---: | --- | --- | --- | --- | --- |
| `JANGQ/DeepSeek-V4-Flash-JANGTQ-K` | 80G | `deepseek_v4` | yes | fallback/native DSV4 | `PASS` - top-level tools, Osaurus-sized tools, thinking on/off, and max preface render | `PARTIAL` |
| `JANGQ/DeepSeek-V4-Flash-JANGTQ2` | 74G | `deepseek_v4` | yes | fallback/native DSV4 | `PASS` - top-level tools, Osaurus-sized tools, thinking on/off, and max preface render | `PARTIAL` |
| `JANGQ/Hy3-preview-JANGTQ` | 79G | `hy_v3` | yes | file | `PASS` | `TODO` |
| `JANGQ/Hy3-preview-JANGTQ_K` | 102G | `hy_v3` | yes | file | `PASS` | `TODO` |
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
| `dealign.ai/Ling-2.6-flash-MXFP4-CRACK` | 63G | `bailing_hybrid` | no | file | `PASS` | `TODO` |
| `dealign.ai/MiniMax-M2.7-JANGTQ_K-CRACK` | 74G | `minimax_m2` | yes | file | `PASS` | `TODO` |
| `dealign.ai/Nemotron-Omni-Nano-JANGTQ-CRACK` | 12G | `nemotron_h` | yes | file | `PASS` | `TODO` |
| `dealign.ai/Nemotron-Omni-Nano-JANGTQ4-CRACK` | 19G | `nemotron_h` | yes | file | `PASS` | `TODO` |
| `dealign.ai/Nemotron-Omni-Nano-MXFP4-CRACK` | 21G | `nemotron_h` | no | file | `PASS` | `TODO` |
| `dealign.ai/Qwen3.6-27B-JANG_4M-CRACK` | 16G | `qwen3_5` | no | file | `PASS` | `TODO` |
| `dealign.ai/Qwen3.6-27B-MXFP4-CRACK` | 14G | `qwen3_5` | no | file | `PASS` | `PARTIAL` |
| `dealign.ai/Qwen3.6-35B-A3B-JANGTQ-CRACK` | 11G | `qwen3_5_moe` | yes | file | `PASS` | `PARTIAL` |
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
  without an environment-gated downgrade.
- Current live proof: JANGTQ-K chat coherence passes with visible output,
  `.stop`, no raw reasoning leakage, and tok/s; DSV4 paged-incompatible cache
  restores through disk with salted hits and nil-salt misses.
- Current live nuance: the older arithmetic gate used an ambiguous "7 + 5"
  wording and failed; the explicit `Q: What is 7 + 5? ... A:` gate now passes
  for reasoning off/on/max on JANGTQ-K. This is a gate prompt correction, not a
  hidden sampling clamp.
- Required live gates still open: long-context regression, vector drift,
  full speed matrix, and broader reasoning matrix.

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
- Live proof: Qwen3.6 MXFP4 has coherent 3-turn chat and SSM warm second-turn.
  `Qwen3.6-35B-A3B-JANGTQ-CRACK` now loads the VLM path as `Qwen35MoE` with
  `Qwen3VLProcessor`; image turns ground the red/blue gradient, text-only
  follow-up works, same-image media-salt restore hits, and different-image
  media-salt restore misses. Qwen3VL video processing now reads
  `video_preprocessor_config.json` and uses frame-count-aware video resize math;
  the resized 1080p video smoke attaches `LMInput.video` and returns coherent
  visible content with thinking disabled.
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
- Current issues: thinking-on can emit reasoning without visible answer within
  small budgets on some rows, so higher-budget closure proof is still required.
  The 35B JANGTQ mixed text/image/video row is also blocked on the high-res
  Qwen3VL video turn after more than seven minutes in prefill/forward. The
  video config path is fixed, but the full high-resolution row remains a
  throughput/scaling gate because the bundle declares a large video pixel
  budget.

### MiniMax M2.7

- Local bundles: small JANGTQ and CRACK JANGTQ_K.
- Swift dispatch: `minimax_m2`; JANGTQ bundles route to `MiniMaxJANGTQModel`.
- Cache topology: standard KV layers, BatchEngine cache coordinator, TQ disk
  serializer, and compiled decode path where cache types allow it.
- JANGTQ/TurboQuant: sidecar present; routed expert path uses
  `TurboQuantSwitchGLU`/codebook-style JANGTQ tensors.
- Template/reasoning/tools: template smoke passes; tool parser is MiniMax M2
  invoke/parameter format.
- Live proof: small JANGTQ has coherent 3-turn chat and TQ disk round-trip.
- Current issues: thinking-on no visible answer within budget; low-RAM active
  streaming is not proven because footprint reached full model scale.

### ZAYA Text

- Local bundles: JANGTQ4, JANGTQ_K, MXFP4.
- Swift dispatch: `zaya` through `ZayaModel`.
- Cache topology: ZAYA CCA alternates `ZayaCCACache` with normal KV cache;
  BatchEngine has `BatchZayaCCACache`; disk serializer supports `zayaCCA`.
- JANGTQ/TurboQuant: sidecar present on JANGQ bundles; release rows prove
  TurboQuant-KV B=2 isolation for JANGTQ4, JANGTQ_K, and MXFP4.
- Template/reasoning/tools: template smoke passes; tool parser is ZAYA XML;
  reasoning parser is ZAYA/think-XML style.
- Live proof: `docs/local/live-model-matrix/20260517T_release_turnmatrix_zaya_scope/`
  passes the text release turnmatrix for JANGTQ4, JANGTQ_K, and MXFP4:
  config/template, production defaults cache OFF/ON, BatchEngine single/chat,
  disk restore, B=2 concurrent, B=2 per-slot sampler, and TurboQuant-KV B=2.
  Bundle defaults apply (`temp=0.600`, `topP=1.000`, `topK=0`, `rep=nil`),
  reasoning ON/OFF flips produce visible answers, disk L2 and SSM hits are
  recorded, and release speed rows are about 64-66 tok/s.
- Current issue: generic paged prefix hit is `N-A` because ZAYA is
  paged-incompatible and uses disk-backed restore. Do not advertise generic
  paged hits for this topology.

### ZAYA VL

- Local bundles: JANGTQ4, JANGTQ_K, MXFP4.
- Swift dispatch: `zaya1_vl` through `Zaya1VL`.
- Cache topology: text side uses ZAYA CCA state; media path uses media salt and
  VLM processor state; BatchEngine has VL chat/cache matrix harnesses.
- Template/reasoning/tools: JANGQ ZAYA-VL sidecar bundles now pass template
  smoke for vision placeholders, thinking on/off, multi-turn history, top-level
  tools, and Osaurus-sized ZAYA XML schemas. The fix rewrites
  `tokenizer_config.json`, `chat_template.json`, and `chat_template.jinja` in the
  tokenizer shim because `swift-transformers` otherwise prefers the original
  sidecar and silently drops tools.
- Live proof: `docs/local/live-model-matrix/20260517T_release_turnmatrix_zaya_scope/`
  plus targeted reruns under
  `docs/local/live-model-matrix/20260517T_release_targeted_rerun_after_harness_fixes/`
  pass the implemented ZAYA-VL image/text/cache surfaces for JANGTQ4 and MXFP4:
  production defaults cache OFF/ON, BatchEngine rows, VL batch chat, structured
  cache, and media-salt isolation. Video rows are explicitly `N-A` because this
  processor throws `ZAYA1-VL video input is not implemented`.
- Current issue: `ZAYA1-VL-8B-JANGTQ_K` remains a real blocker. Its math/top-k
  evidence ranks the wrong first token before decoding, `prod_defaults` fails
  `7+8-11`, and the structured VL cache row exhausts the 192-token cold-image
  budget. Kernel probes pass on sampled layers, so this is not closed by a
  sampler/template guard.

### Ling / Bailing Hybrid

- Local bundles: Ling JANGTQ2 CRACK and Ling MXFP4 CRACK.
- Swift dispatch: `bailing_hybrid` through `BailingHybridModel`.
- Cache topology: global/MLA layers use KV or rotating KV; linear-attention
  layers use `ArraysCache`; BatchEngine has `BatchArraysCache`.
- Template/reasoning/tools: template smoke passes; Bailing/Ling template context
  maps thinking controls; tool parser maps to GLM/deepseek-style format.
- Current status: `docs/local/live-model-matrix/20260517T170008Z_release_turnmatrix_ling_jangtq2/REPORT.md`
  passes all runnable rows for the JANGTQ2 CRACK bundle: config/template, MTP
  metadata, production defaults cache OFF/ON, BatchEngine single/chat, disk
  restore, B=2 concurrent, B=2 per-slot sampler, and TurboQuant-KV B=2. Bundle
  defaults resolve to `temp=0.600`, `topP=1.000`, `topK=0`, `minP=0.000`,
  `rep=nil`; release decode telemetry is about 37-39 tok/s. Cache ON records
  disk L2 and SSM hits with `pagedIncompatible=true`; generic paged prefix hit
  is `N-A`.
- Boundary: `Ling-2.6-flash-MXFP4-CRACK` still needs its own live turnmatrix.

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
  L2 records a real hit/store row. Remaining open rows are long-budget harmony
  reasoning, live tool-call schema, and any VL-specific proof.

### Nemotron Omni / Nemotron H

- Local bundles: JANGTQ, JANGTQ4, MXFP4 CRACK.
- Swift dispatch: text config is `nemotron_h`; VLM/Omni route uses Nemotron H
  Omni processor when multimodal config requires it.
- Cache topology: hybrid SSM/cache state applies; Omni path also has image,
  video, audio, and pre-encoded audio handling.
- Template/reasoning/tools: template smoke passes; reasoning parser maps to
  Nemotron H; tool parser maps to Nemotron format.
- Current status: JANGTQ4 core Omni path is live-proven in
  `docs/local/live-model-matrix/20260517T164618Z_omni_live_voice_current_recheck/`,
  `docs/local/live-model-matrix/20260517T164640Z_omni_live_audio_streaming_jangtq4_current/`,
  `docs/local/live-model-matrix/20260517T164702Z_omni_integrated_jangtq4_current/`,
  and `docs/local/live-model-matrix/20260517T170614Z_omni_live_voice_fresh_recheck/`.
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
  gate is added. The JANGTQ and MXFP4 sibling bundles have only short streaming
  smoke rows in the latest artifact, not production coherency proof.

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

- Local bundles: JANGTQ, JANGTQ_K, full Tencent BF16.
- Swift dispatch: `hy_v3` through `Hy3Model`.
- Cache topology: standard generation cache plus JANGTQ sidecar on JANGQ rows.
  Needs live cache-family inspection.
- Template/reasoning/tools: template smoke passes; reasoning/parser maps to Hy3
  / Hunyuan-style behavior.
- Current status: no live row in this snapshot.

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

1. `ZAYA1-VL-8B-JANGTQ_K` remains a real production blocker: the release matrix
   still fails the math production row and structured VL cache row, and top-k
   evidence shows the wrong first token before decoding policy.
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
