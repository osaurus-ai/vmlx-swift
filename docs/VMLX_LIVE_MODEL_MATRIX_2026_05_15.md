# vMLX Live Model Matrix - 2026-05-15

This is the live validation workflow for the consolidated `vmlx-swift` engine.
It is model-load proof, not a source-read checklist.

## Harness

Use:

```sh
scripts/vmlx-live-model-matrix.sh --profile inventory
scripts/vmlx-live-model-matrix.sh --profile infer --max-size-gb 20
scripts/vmlx-live-model-matrix.sh --profile all --max-size-gb 20
scripts/vmlx-live-model-matrix.sh --profile turnmatrix --max-size-gb 6
scripts/vmlx-live-model-matrix.sh --profile batch --model ~/models/<bundle>
scripts/vmlx-live-model-matrix.sh --profile all --allow-huge
```

Artifacts are written under:

```text
docs/local/live-model-matrix/<timestamp>/
```

`docs/local` stays uncommitted because it contains local paths, raw model
outputs, cache directories, and machine-specific timing.

## Local Inventory Snapshot

Current local bundles discovered under `~/models` include:

- ZAYA text and ZAYA-VL JANGTQ/MXFP variants.
- DSV4 Flash JANGTQ-K and JANGTQ2.
- Hy3 JANGTQ and native Tencent Hy3.
- Kimi-K2.6 JANGTQ small and full.
- MiniMax M2.7 JANGTQ, JANG_2L, and CRACK variants.
- Qwen3.5/Qwen3.6 text, MoE, MXFP, JANG, and MTP variants.
- Ling/Bailing flash JANGTQ2 and MXFP.
- Gemma 4 JANG_4M.
- Nemotron Omni Nano JANGTQ/JANGTQ4/MXFP.

The inventory file records size, architecture, model type, profile, whether
MTP tensor evidence is present, and the local bundle's `generation_config.json`
sampling fields: `max_new_tokens`, `temperature`, `top_p`, `top_k`, `min_p`,
`repetition_penalty`, and `do_sample`. MTP is never inferred from the directory
or model name. Metadata can say that MTP is expected, but a bundle is marked
`mtp=yes` only when the actual weight map or safetensors headers contain MTP
tensors.

## Per-Family Rows

| Profile | Harness rows |
| --- | --- |
| `metadata` | `BENCH_CONFIG_SMOKE=1`, `BENCH_TEMPLATE_SMOKE=1` |
| `infer` | Metadata plus `BENCH_PROD=1` with no cache coordinator. VL bundles also run `BENCH_VL_BATCH_CHAT=1`; Omni bundles also run `BENCH_OMNI=1`. This is the plain "does the model load, template, reason on/off, and answer coherently?" gate before cache/JangPress/MLXPress work. |
| `text` | `BENCH_PROD=1`, `BENCH_PROD_COORD=1` |
| `batch` | `BENCH_BATCH=1`, `BENCH_BATCH_CHAT=1`, `BENCH_BATCH_CACHE_HIT=1`, `BENCH_BATCH_DISK_RESTORE=1`, `BENCH_BATCH_CONCURRENT=1`, `BENCH_BATCH_PERSLOT_SAMPLER=1`, `BENCH_BATCH_TQ_B2=1` |
| `vl` | `BENCH_VL_BATCH_CHAT=1`, `BENCH_VL_BATCH_MEDIASALT=1` |
| `omni` | `BENCH_OMNI=1`, `BENCH_OMNI_BATCH=1` |
| `mtp` | `MTPRuntimeFocusedTests` with `VMLX_MTP_REAL_BUNDLE`; non-MTP tensor bundles are recorded as `n-a:no-mtp-tensors` |
| `turnmatrix` | Metadata and MTP metadata when applicable, then detected-family live rows. Text rows run `BENCH_PROD` twice with bundle/default sampling: tiered cache OFF and tiered cache ON. Batch rows cover B=1, chat, prefix/paged hit, L2 restore, B=2 overlap, per-slot sampler, and TurboQuant KV B=2. VL rows add text-only VL-off turns, BatchEngine media turns, structured chat cache, media-salt isolation, and text/image/video mixed turns when the video fixture exists. Omni rows add the full text/image/audio/video hybrid probe. |

The `all` profile runs metadata first, then the detected family live profile,
and MTP metadata tests for bundles with MTP tensor evidence.

`turnmatrix` uses two token budgets by default:

- `VMLX_MATRIX_PROD_MAX_TOKENS=2048` for `BENCH_PROD` reasoning/cache ON/OFF
  rows, because short reasoning budgets can create false failures.
- `VMLX_MATRIX_PROD_SEED=0` for stochastic samplers, so a gate rerun tests
  the same distribution path instead of a different time-seeded sample.
- `VMLX_MATRIX_MAX_TOKENS=192` for batch, TurboQuant KV, and media rows so
  the full matrix remains practical.

Raise either for specific long-context gates, for example:

```sh
VMLX_MATRIX_PROD_MAX_TOKENS=4096 VMLX_MATRIX_MAX_TOKENS=384 scripts/vmlx-live-model-matrix.sh \
  --profile turnmatrix --model ~/models/JANGQ/DeepSeek-V4-Flash-JANGTQ-K
```

The cache-OFF row disables the tiered cache coordinator. It does not disable
the per-request KV cache required for normal autoregressive decoding.

`BENCH_PROD` uses `generation_config.json` / engine defaults for sampling by
default and prints the resolved sampler, including any request seed. Explicit
greedy is available only when requested with `BENCH_PROD_GREEDY=1` or
`VMLX_MATRIX_INCLUDE_GREEDY=1 scripts/vmlx-live-model-matrix.sh --profile turnmatrix ...`.
Greedy failures are recorded as requested-sampling consequences, not hidden by
family floors or repetition-penalty clamps.

## Production Pass Criteria

A model is not production-ready unless the artifact proves:

- real model load happened on this MacBook;
- multi-turn visible output is coherent and not looping;
- reasoning-capable rows with `enable_thinking=true` emit `.reasoning`
  deltas and still deliver the final answer in visible `.chunk` output;
- rows with `enable_thinking=false` emit zero `.reasoning` deltas and no
  reasoning envelope markers in visible output;
- token/s or prompt/decode telemetry is present;
- stop reason is normal or explicitly explained;
- cache topology is shown for the model family;
- cache OFF and cache ON rows both produce coherent multi-turn output;
- single-batch and multi-batch rows prove actual active slot overlap, not
  serialized reads from a fake concurrent harness;
- TurboQuant KV rows prove B=2 mixed plain/TQ and all-TQ slots complete
  coherently without cross-slot drift;
- new-session cache rows prove disk L2 restore with a fresh coordinator;
- VL/video/audio rows use real media payloads and media-salt behavior;
- VL models pass both text-only turns and media turns; text-only is the VL-off
  payload row, not a separate fake model mode;
- MTP rows prove preserved metadata and keep speculative decode disabled until
  accept/reject runtime exists;
- physical footprint is low for JANGTQ/active-routed models.

Skipped rows are blocked, not passes. Report-only memory gates are diagnostics,
not production readiness.

## Osaurus Server Panel Dependency

The server panel should read the same settings and status concepts documented in
`docs/VMLX_SERVER_PANEL_ENGINE_CONTRACT_2026_05_15.md`. UI toggles should not
invent behavior that this package cannot prove live.
