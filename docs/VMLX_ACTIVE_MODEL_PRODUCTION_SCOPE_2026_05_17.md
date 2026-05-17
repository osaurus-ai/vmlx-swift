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
