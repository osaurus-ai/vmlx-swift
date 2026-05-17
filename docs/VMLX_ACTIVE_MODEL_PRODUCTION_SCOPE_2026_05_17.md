# vMLX Active Model Production Scope - 2026-05-17

This note records the current active scope after the explicit user direction to
exclude Kimi and DSV4 Flash for now. It is not a production-ready claim. It is
the checklist for the remaining model families that still need live
multi-turn/cache/VL/Omni proof before Osaurus moves fully onto `vmlx-swift`.

## Current Exclusions

Do not include these in the active production matrix until the user reopens
them:

- Kimi / Kimi-K2.x
- DeepSeek-V4 / DSV4 Flash

The harness supports this directly:

```sh
scripts/vmlx-live-model-matrix.sh \
  --profile inventory \
  --exclude-regex 'Kimi|DeepSeek-V4|DSV4'
```

Fresh inventory artifact:

```text
docs/local/live-model-matrix/20260517T_scope_exclude_kimi_dsv4_inventory/
```

No-load MTP census artifact:

```text
docs/local/live-model-matrix/20260517T_scope_exclude_kimi_dsv4_mtp_census/
```

No-load config/template metadata artifact:

```text
docs/local/live-model-matrix/20260517T_scope_exclude_kimi_dsv4_metadata/
```

Fresh focused MTP/settings artifact:

```text
docs/local/production-readiness/20260517T160343Z_qwen_mtp_settings_current/
```

That inventory contains 28 non-excluded local bundles:

- 12 text bundles
- 13 VL bundles
- 3 Omni bundles

## Best Native-MTP Speed Rows So Far

These are prompt-specific Qwen3.6 count-prompt rows. They prove the native-MTP
loop and cache behavior for those artifacts only; they do not prove global
production readiness.

Prompt:

```text
Count from 1 to 50 in order, separated by commas.
```

All rows below produced exact visible `1..50`, stopped normally, and did not use
hidden sampling guards, forced repetition penalties, or forced reasoning closure.

| Bundle | Without MTP AR tok/s | Best MTP tok/s | Best depth | Cache row | Current decision |
|---|---:|---:|---:|---|---|
| Qwen3.6 27B JANG_4M | 27.4 | 48.9 | D2 | disk L2 + SSM hit | MTP explicit only; D2 is current speed row. |
| Qwen3.6 27B MXFP4 | 31.8 | 50.5 | D3 | disk L2 + SSM hit | D3 reaches the 45 tok/s target in cache-warm rows. |
| Qwen3.6 27B MXFP8 | 17.3 | 31.7 | D2 | disk L2 + SSM hit | D2 wins; D3 is coherent but slower on this prompt. |
| Qwen3.6 35B JANG_2K | 120.1 | n-a | n-a | n-a | Chunk MTP is blocked by correctness failures; use AR. |
| Qwen3.6 35B MXFP4 | 105.3 | 171.4 | D3 | disk L2 + SSM hit | D3 is the current speed row. |
| Qwen3.6 35B MXFP8 | 79.1 | 129.9 | D3 | disk L2 + SSM hit | D3 is the current speed row. |

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
follow-up are coherent. It does not change the default policy: native MTP stays
explicit, tensor-gated, and non-batched until the remaining 35B JANG_2K VL and
server scheduling gates are proven.

Current hybrid-SSM verifier policy update: stochastic exact-pq native MTP does
not use the fast chunk verifier. A 35B MXFP4 growing-chat row failed under
bundle defaults when forced through `chunk_commit`; D1 reproduced it, while
sequential repair passed. Post-fix rows under
`docs/local/qwen36-mtp-current/20260517T131050Z-mxfp-growing-chat-mtp-d3-exact-postfix/`
and
`docs/local/qwen36-mtp-current/20260517T131024Z-35b-mxfp4-growing-chat-mtp-d3-exact-postfix/`
prove all four MXFP variants now run bundle-default D3 exact-pq with
`verifierMode=sequential_repair`, coherent two-turn output, disk-prefix hits,
and SSM hits. Greedy rows still use `chunk_commit` where proven.

Current focused gate at 2026-05-17 09:03 PDT:

```text
docs/local/production-readiness/20260517T160343Z_qwen_mtp_settings_current/
```

- `MTPRuntimeFocusedTests.log`: 40/40 pass. Coverage includes cached verifier
  masks carrying cache offsets, preserved-only MTP detection without
  auto-enable, metadata-only bundles without tensor evidence, explicit
  tensor-gated Qwen3.5 MoE activation, task-local activation and env override
  behavior, JANG metadata parsing, tensor/runtime-evidence-gated auto policy,
  recursive D3 hidden-state draft/verify contract, Qwen3.5 SSM accepted-prefix
  offsets, partial-reject lazy repair, private draft-cache refresh, greedy
  chunk verifier telemetry, BatchEngine native-MTP exclusive lane, and rejection
  of native MTP through batched `submit`.
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
  validation, preserved-only MTP auto-launch denial, force-on requiring verified
  accept/reject runtime, and policy/draft-limit launch resolution.

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

## Active Non-Excluded Family Matrix

| Family | Local bundles | Engine surfaces to prove | Current MTP policy |
|---|---|---|---|
| Qwen3.6/Qwen3.5 text+VL MXFP/JANG/JANGTQ | Qwen3.6 MTP, Qwen3.6 CRACK, Qwen3.5 A3B 4-bit | Qwen chat template, `generation_config.json`, GatedDelta/hybrid SSM cache, disk L2 + SSM companion, VL media salt, text-only continuation after media, reasoning on/off | Only Qwen bundles with real MTP tensor evidence may be manually requested. No auto-launch yet. |
| ZAYA text/VL JANGTQ/MXFP | JANGQ and Osaurus ZAYA1 text/VL | Zaya CCA cache, JANGTQ/MXFP decode, VL adapters, media salt, Hadamard/matmul shape coverage, multi-turn cache hit | No native MTP. |
| MiniMax M2.7 JANG/JANGTQ | Small JANGTQ and CRACK JANG/JANGTQ_K | MiniMax template, reasoning on/off, JANGTQ streaming experts, low-footprint active routed decode, prefix/paged/disk/TurboQuant KV, multi-turn coherence | Local MiniMax CRACK rows are non-MTP unless real MTP tensor evidence appears. |
| Ling/Bailing hybrid | Ling JANGTQ2 and MXFP4 CRACK | Bailing thinking template, hybrid cache/SSM rederive, nextn metadata handling, JANGTQ/MXFP decode, multi-turn coherence | Extra nextn-layer evidence exists, but Swift native-MTP auto-launch remains off. |
| Hy3 | HYV3 JANGTQ/JANGTQ_K | Hy3 template kwargs, native runtime registration, compiled decode guard, nextn metadata handling, routed/JANGTQ decode, cache topology | Extra nextn-layer evidence exists, but Swift native-MTP auto-launch remains off. |
| Gemma 4 | Gemma-4 JANG_4M CRACK | Gemma4 template fallback/tools, sliding-window cache topology, RMSNorm no-scale parity, reasoning parser, multi-turn coherence | No native MTP. |
| Nemotron Omni | JANGTQ/JANGTQ4/MXFP4 Omni Nano | Parakeet audio encoder, RADIO vision, Omni text/image/audio/video ingest, BatchEngine stress, cache/media state, text output coherence | No native MTP. |
| Laguna | Laguna XS JANGTQ | Laguna/Mistral-style template and RoPE params, JANGTQ decode, prefix/paged/disk cache, multi-turn coherence | No native MTP. |

The no-load MTP census confirms every `mtp=yes` row currently reports
`canAutoLaunch=false`. That includes Qwen3.6, Hy3, and Ling/Bailing bundles.
This is the right fail-closed Osaurus behavior until the relevant family has a
verified accept/reject runtime and live cache/VL/reasoning proof.

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
| ZAYA1-8B-JANGTQ_K | no older speed floor recorded in the handoff table | 65.7 tok/s median, 65.8 best | PASS against 50 tok/s floor |
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

Remaining non-false-positive blocker:

- `ZAYA1-VL-8B-JANGTQ_K` is still not production-clear. The release matrix
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

Harness fix from this row:

- The B=2 proof now records an internal BatchEngine active-slot high-water
  mark. External polling can miss short-lived overlap while model forwards
  monopolize the actor executor, so the release gate now drains streams while
  observing both live `activeCount` and the engine's high-water mark.

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
- the generic prefix-extension paged cache-hit row is N-A because this model is
  routed through the disk-backed paged-incompatible cache path.

This clears the current Gemma 4 text multi-turn/cache/batching row. It does not
yet close separate long-budget harmony reasoning, live tool-call schema, or VL
rows.

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
