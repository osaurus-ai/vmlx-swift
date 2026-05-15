# vMLX Live Model Matrix - 2026-05-15

This is the live validation workflow for the consolidated `vmlx-swift` engine.
It is model-load proof, not a source-read checklist.

## Harness

Use:

```sh
scripts/vmlx-live-model-matrix.sh --profile inventory
scripts/vmlx-live-model-matrix.sh --profile all --max-size-gb 20
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
MTP evidence is present, and the local bundle's `generation_config.json`
sampling fields: `max_new_tokens`, `temperature`, `top_p`, `top_k`, `min_p`,
`repetition_penalty`, and `do_sample`.

## Per-Family Rows

| Profile | Harness rows |
| --- | --- |
| `metadata` | `BENCH_CONFIG_SMOKE=1`, `BENCH_TEMPLATE_SMOKE=1` |
| `text` | `BENCH_PROD=1`, `BENCH_PROD_COORD=1` |
| `batch` | `BENCH_BATCH=1`, `BENCH_BATCH_CHAT=1`, `BENCH_BATCH_CACHE_HIT=1`, `BENCH_BATCH_DISK_RESTORE=1`, `BENCH_BATCH_CONCURRENT=1`, `BENCH_BATCH_PERSLOT_SAMPLER=1`, `BENCH_BATCH_TQ_B2=1` |
| `vl` | `BENCH_VL_BATCH_CHAT=1`, `BENCH_VL_BATCH_MEDIASALT=1` |
| `omni` | `BENCH_OMNI=1`, `BENCH_OMNI_BATCH=1` |
| `mtp` | `MTPRuntimeFocusedTests` with `VMLX_MTP_REAL_BUNDLE` |

The `all` profile runs metadata first, then the detected family live profile,
and MTP metadata tests for bundles with MTP evidence.

## Production Pass Criteria

A model is not production-ready unless the artifact proves:

- real model load happened on this MacBook;
- multi-turn visible output is coherent and not looping;
- token/s or prompt/decode telemetry is present;
- stop reason is normal or explicitly explained;
- cache topology is shown for the model family;
- single-batch and multi-batch rows prove actual active slot overlap, not
  serialized reads from a fake concurrent harness;
- TurboQuant KV rows prove B=2 mixed plain/TQ and all-TQ slots complete
  coherently without cross-slot drift;
- new-session cache rows prove disk L2 restore with a fresh coordinator;
- VL/video/audio rows use real media payloads and media-salt behavior;
- MTP rows prove preserved metadata and keep speculative decode disabled until
  accept/reject runtime exists;
- physical footprint is low for JANGTQ/active-routed models.

Skipped rows are blocked, not passes. Report-only memory gates are diagnostics,
not production readiness.

## Osaurus Server Panel Dependency

The server panel should read the same settings and status concepts documented in
`docs/VMLX_SERVER_PANEL_ENGINE_CONTRACT_2026_05_15.md`. UI toggles should not
invent behavior that this package cannot prove live.
