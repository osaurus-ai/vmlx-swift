# Gemma 4 QAT cache correctness checkpoint — 2026-07-19

Status: **PARTIAL — exact mixed-cache TurboQuant and fresh-process L2 restore
are live-proven on vMLX `718522bc`, but the current cache-default/explicit-Off
patch still requires a rebuilt isolated Release app and repeat UI proof.**

This checkpoint is intentionally limited to locally installed Gemma 4 MXFP8
and JANG_4M bundles through the real Osaurus/vMLX Swift runtime. MXFP4 is not a
substitute test artifact and is not part of this checkpoint.

## Source contract

- Ordinary Osaurus single-batch loads keep paged RAM KV off by default.
- Engine-selected/native cache mode keeps TurboQuant KV off. TurboQuant is an
  explicit user opt-in with explicit key/value bit widths.
- Block-disk L2 is on by default and is independent of paged RAM cache. An
  explicit block-disk Off must survive memory-safety policy resolution.
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
| Default paged policy | Isolated Release app at exact vMLX `718522bc`: Settings showed paged KV off and `/admin/cache-stats` reported `paged.enabled=false` | VERIFIED-LIVE on prior pin; current patch rebuild pending |
| MXFP8, TurboQuant off | Fresh single-root first turn `COBALT-LIGHTHOUSE-7314`: 32.0 tok/s; cold-restart exact continuation: 31.7 tok/s; disk hit restored boundary 1,769; native 8 KV + 40 rotating topology; transition null | VERIFIED-LIVE on prior pin |
| MXFP8, TurboQuant 4/4 | Fresh single-root first turn `COPPER-HARBOR-6158`: 28.1 tok/s; cold-restart exact continuation: 26.1 tok/s; disk hit restored boundary 1,771; transition 8 KV to 8 TQ with all 40 rotating layers preserved | VERIFIED-LIVE on prior pin |
| JANG_4M, TurboQuant off | Fresh single-root first turn `RIVER-CLOCK-7319`: 39.1 tok/s; follow-up 41.4 tok/s; native 8 KV + 40 rotating topology; paged off | VERIFIED-LIVE on prior pin |
| JANG_4M, TurboQuant 4/4 | Fresh single-root first turn `SAPPHIRE-FORGE-9062`: 34.3 tok/s; cold-restart exact continuation: 32.5 tok/s; partial/full disk hits; transition 8 KV to 8 TQ with all 40 rotating layers preserved | VERIFIED-LIVE on prior pin |
| TurboQuant transition telemetry | `TurboQuantCacheTransitionTelemetryTests` 3/3 on the current source: completion retention, Codable round-trip, and exact mixed Gemma 8-KV-to-8-TQ/40-rotating transition | VERIFIED-SOURCE; current-patch live repeat pending |
| Explicit block-disk Off | Current source preserves `blockDisk.enabled=false` through memory-safety resolution; focused regression 1/1 | VERIFIED-SOURCE; live Settings/save/restart/counter proof pending |
| Engine-selected TurboQuant default | Current source resolves engine-selected to native `.none`; focused default/codec regressions 2/2 | VERIFIED-SOURCE; live Settings and endpoint proof pending |
| RAM safety refusal/override | Real Settings UI: Strict 10% refused MXFP8 at 12.8 GB budget; No Automatic Limits then loaded and generated `LOADED`; restored Safe Auto afterwards | VERIFIED-LIVE |
| Activity Monitor footprint | Exact prior-pin proof process/path inspected in Activity Monitor: main Memory 1.99 GB; inspector Real Memory 1.37 GB, Private 1.03 GB | VERIFIED-LIVE on prior pin; current-patch repeat pending |

## Required live closure matrix

Every row must use a fresh Release development build with an isolated bundle
identifier and preferences root. Visible UI behavior and matching runtime
telemetry are both required.

| Gate | Required evidence | Status |
|---|---|---|
| Current-build MXFP8 TQ transition | Rebuild after the default/Off patch; repeat 48 total / 8 KV / 40 rotating before and 48 / 8 TQ / 40 rotating after, plus coherent visible continuation | OPEN |
| Current-build JANG_4M TQ transition | Same current-build exact transition and coherent visible continuation | OPEN |
| Paged default/effective policy | Defaults visibly off. Gemma's rotating topology is paged-incompatible, so an explicit request must truthfully report effective paged off rather than claim nonexistent paged hits | OPEN on current build |
| SSD L2 with paged off | With paged off and SSD L2 on, cold restart restores the longest valid full or partial prefix from disk and produces a coherent continuation | VERIFIED-LIVE on prior pin; current-build repeat open |
| Explicit SSD L2 off | Turn SSD Cache (L2) off in Settings, save/restart/load, and show block-disk and legacy disk both false, zero disk hits/stores, and no new cache artifacts in the isolated root | OPEN |
| Fresh-process L2 restore | Turn SSD L2 back on, quit only the isolated proof app, relaunch it, and show a disk hit plus a coherent continuation dependent on cached context | VERIFIED-LIVE on prior pin; current-build repeat open |
| Raw-prefill fallback | Use a cache-salted or genuinely new prefix absent from both tiers; paged/disk miss counters increase, real prefill progress is visible, output remains coherent | OPEN |
| TurboQuant after L2 restore | With TQ enabled, restore the typed raw prompt boundary from L2, then prove the resumed live decode transitions 8 KV layers to TQ while preserving 40 rotating layers | VERIFIED-LIVE on prior pin; current-build repeat open |
| TurboQuant off control | Repeat the same restore with Native selected; live transition remains null and TQ layer count remains zero | VERIFIED-LIVE on prior pin; current-build repeat open |
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
- LFM and MiniMax M2.7 also remain wider-campaign rows for explicit
  TurboQuant off/on, partial SSD-only restore, paged eviction where their
  topology supports it, multi-turn coherence, TTFT/tok/s, and physical
  footprint. Gemma evidence does not close those rows.
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

## Current-source test note

The current patch's four directly changed cache-policy assertions pass when
run individually, and the transition/stale-EOS suites pass 3/3 each. The older
`automaticRuntimeCachePolicyCoversDownloadedArchitectureFamilies` matrix still
contains a separate stale Hunyuan reasoning expectation (`think_xml` while the
current source returns `hy_v3`). That unrelated row is not changed or counted
as Gemma cache proof in this checkpoint.
