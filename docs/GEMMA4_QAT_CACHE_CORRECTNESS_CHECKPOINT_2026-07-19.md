# Gemma 4 QAT cache correctness checkpoint — 2026-07-19

Status: **PARTIAL — live mixed-cache TurboQuant transition, paged-L1 eviction,
and fresh-process L2 restore still require the telemetry build below.**

This checkpoint is intentionally limited to locally installed Gemma 4 MXFP8
and JANG_4M bundles through the real Osaurus/vMLX Swift runtime. MXFP4 is not a
substitute test artifact and is not part of this checkpoint.

## Source contract

- Ordinary Osaurus single-batch loads keep paged RAM KV off by default.
- The tested Gemma 4 12B bundles declare 48 attention layers: 8 full-attention
  `KVCacheSimple` layers and 40 `RotatingKVCache` sliding-attention layers.
- TurboQuant converts only eligible `KVCacheSimple` layers. It must preserve
  all rotating SWA layers as rotating caches.
- `TurboQuantCacheTransitionSnapshot` records the real before/after cache
  classes at the conversion point. A configured TurboQuant mode or a non-zero
  compression-event counter is not accepted as layer-level proof.
- Prompt-boundary L2 entries may intentionally remain typed raw KV plus typed
  rotating state even when live decode uses TurboQuant. Telemetry must report
  the live codec and stored codec separately; neither implies the other.
- Gemma mixed SWA/full attention has no recurrent SSM/GLA companion state.
  Hybrid SSM/GDN/CCA async-rederive gates therefore remain separate family
  rows and must not be inferred from Gemma evidence.

## Current evidence

| Row | Current evidence | Status |
|---|---|---|
| Stale `<end_of_turn>` stop | vMLX `604d24e4`; focused Gemma 3/4 regression 3/3; live MXFP8 and JANG_4M both emitted a literal marker and continued through `FINAL-OMEGA` | VERIFIED-LIVE |
| Default paged policy | Real isolated Release app Settings showed GPU paged KV off; `/admin/cache-stats` reported `paged.enabled=false` | VERIFIED-LIVE |
| MXFP8, TurboQuant off | Two visible UI turns, 31.0/29.3 tok/s; native codec; 8 KV + 40 rotating topology | VERIFIED-LIVE |
| MXFP8, TurboQuant 4/4 | Two visible UI turns, 15.4/28.6 tok/s; compression events and L2 counters increased | PARTIAL: exact per-layer transition requires telemetry rebuild |
| JANG_4M, TurboQuant off | Two visible UI turns, both 40.8 tok/s; native codec; 8 KV + 40 rotating topology | VERIFIED-LIVE |
| JANG_4M, TurboQuant 4/4 | Two visible UI turns, 35.3/35.8 tok/s; compression events and L2 counters increased | PARTIAL: exact per-layer transition requires telemetry rebuild |
| TurboQuant transition telemetry | `TurboQuantCacheTransitionTelemetryTests` 3/3: synthetic mixed Gemma topology reports 8 converted KV and 40 preserved rotating layers and round-trips through Codable | VERIFIED-SOURCE; live endpoint pending |
| RAM safety refusal/override | Real Settings UI: Strict 10% refused MXFP8 at 12.8 GB budget; No Automatic Limits then loaded and generated `LOADED`; restored Safe Auto afterwards | VERIFIED-LIVE |
| Activity Monitor footprint | Real Activity Monitor showed Osaurus at 5.13 GB with green memory pressure after the override row | VERIFIED-LIVE for that row only |

## Required live closure matrix

Every row must use a fresh Release development build with an isolated bundle
identifier and preferences root. Visible UI behavior and matching runtime
telemetry are both required.

| Gate | Required evidence | Status |
|---|---|---|
| Exact MXFP8 TQ transition | `/admin/cache-stats` before: 48 total / 8 KV / 40 rotating; after: 48 total / 8 TQ / 40 rotating; coherent visible continuation | OPEN |
| Exact JANG_4M TQ transition | Same exact before/after layer proof and coherent visible continuation | OPEN |
| Paged L1 explicit opt-in | User turns paged cache on in Settings; reload shows effective on; second matching prompt increases paged hits and returns coherent tail sentinel | OPEN |
| Paged L1 capacity/eviction | Constrain the visible paged-cache capacity, create more blocks than fit, observe allocated/free/eviction counters and bounded Activity Monitor footprint | OPEN |
| L1-before-L2 ordering | With a matching block still resident, paged-hit counter increases without a disk-hit increase for the same fetch | OPEN |
| L2 fallback after eviction | Evict the matching paged block while retaining its disk entry; next matching prompt increases disk hit and restores typed 8-full/40-rotating state | OPEN |
| Fresh-process L2 restore | Quit only the isolated proof app, relaunch it, load the same model, and show a disk hit plus a coherent continuation dependent on cached context | OPEN |
| Raw-prefill fallback | Use a cache-salted or genuinely new prefix absent from both tiers; paged/disk miss counters increase, real prefill progress is visible, output remains coherent | OPEN |
| TurboQuant after L2 restore | With TQ enabled, restore the typed raw prompt boundary from L2, then prove the resumed live decode transitions 8 KV layers to TQ while preserving 40 rotating layers | OPEN |
| TurboQuant off control | Repeat the same restore with Native selected; live transition remains null and TQ layer count remains zero | OPEN |
| Long rotating-window sentinel | Cross the 1,024-token SWA window, reuse prefix/L2, and reproduce exact early/middle/tail facts without a loop or truncated tail | OPEN |
| Ten-turn coherence | Ten visible turns with cache reuse, measured TTFT/tok/s on every generated turn, no marker leakage, no hidden-only answer, no looping | OPEN |
| Tool/parser continuation | Required tool, auto tool, no tool, tool-result continuation, tool error recovery, and post-tool text turn with TQ off/on | OPEN |
| Reasoning/template state | UI Thinking off/on/auto maps to the emitted reasoning/content channels and model-owned generation config; no prompt or sampler masking | OPEN |
| API parity | Chat Completions stream/non-stream, Responses stream/non-stream, Anthropic, and Ollama reconstruct the same visible content and finish reason | OPEN |
| Stop/cancel cleanup | Cancel during prefill and decode; next warm turn neither restores a poisoned partial boundary nor leaks stale output | OPEN |
| Memory settings next-load semantics | Safe Auto, Strict, Custom, and No Automatic Limits are visibly changed, saved, and proven on the next real model load; refusal occurs before eviction/load | PARTIAL: Strict/No-Limits proven; remaining modes open |
| Delegation/admission | Local text subagent, Computer Use/AppleScript, image generation/edit, and concurrent delegation receive the same memory admission decision before model eviction | OPEN; Osaurus-level row |

## Non-Gemma rows retained for the wider campaign

- Hybrid SSM/GDN/GLA families (Qwen 3.5/3.5 VL, Ornith, Bonsai, Nemotron and
  applicable MiniMax variants) require typed companion-state hit/miss/rederive
  counters, partial-hit rollback/rederive proof, media-salt proof for VL, and
  coherent post-hit continuation with TurboQuant off/on.
- DSV4/ZAYA and MiniMax-M3 native composite caches remain explicit exceptions
  to generic TurboQuant KV until their typed native codecs are separately
  live-proven. Their evidence must not be generalized from Gemma.
- JANGTQ/MXTQ weight-format correctness is a separate issue from TurboQuant KV
  cache encoding and must remain a separate matrix.
- AppleScript 8B JANG_6M import/discovery, Computer Use app switching, success
  finalization, unexpected tool fallback, and spawned-agent plugin inheritance
  remain Osaurus UI/runtime rows. They are not closed by this cache checkpoint.

No row becomes release-ready from source inspection, a configured setting, or
an aggregate counter alone.
