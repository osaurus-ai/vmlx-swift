# Nemotron Omni cache, RAM, and agent matrix — 2026-07-19

Scope: the exact local bundle
`/Users/eric/models/dealign.ai/Nemotron-Omni-Nano-JANGTQ4-CRACK`, vMLX Swift,
and the real Osaurus Release app. JANGTQ model weights and TurboQuant KV cache
encoding are separate features throughout this matrix.

This document is the live checklist. A row is `VERIFIED-LIVE` only when the
current-source isolated Release app has visible output plus telemetry or a
persisted artifact that proves the claimed tier. Source inspection and unit
tests alone are not live proof. The `nemotron4634proof` rows below are baseline
evidence for merged main before this cache patch. Current-patch live evidence
comes from the isolated app and storage root named below.

Current exact live lane:

- vMLX Swift `0975201e745a1774fda1e78d1bc99b5bd1c668c6`;
- Osaurus `1244a8a94ff8b5b3b00a375fe138964f5589a809`;
- Release app
  `/tmp/osaurus-nemotron-cache157-dd-9662b79/Build/Products/Release/osaurus.app`;
- bundle id `com.dinoki.osaurus.nemotroncache157proof`;
- isolated root `/tmp/osaurus-nemotron-cache157-runtime-ae910a19-clean2`;
- exact non-MXFP4 model
  `/Users/eric/models/dealign.ai/Nemotron-Omni-Nano-JANGTQ4-CRACK`.

Current clean-branch source gates:

- `CacheCoordinatorTopologyFocusedTests`: 39/39 passed; log
  `/tmp/vmlx_nemotron_clean_cache_topology_stateonly_20260719.log`.
- `CacheCoordinatorPagedIncompatibleHybridTests`: 12/12 passed; log
  `/tmp/vmlx_nemotron_clean_incompatible_failclosed_20260719.log`.
- `BatchEngineGrowingChatCacheSourceTests`: 16/16 passed; log
  `/tmp/vmlx_nemotron_clean_source_final_20260719.log`.
- `git diff --check`: clean after the focused source fixes. These results do
  not upgrade any `LIVE-OPEN` row.

## Required cache topology

- Default: prefix cache ON, paged RAM cache OFF, disk L2 ON, TurboQuant KV OFF.
- TurboQuant opt-in: encode only the six attention-KV layers. Preserve all 23
  Mamba companion layers in their native state and restore them at the exact
  matched boundary.
- Paged opt-in: try paged RAM blocks first. A usable hybrid hit requires both
  attention KV blocks and a complete native SSM companion snapshot at the same
  boundary.
- Paged miss/eviction: fall through to the longest valid disk-L2 boundary.
  Missing or incomplete companion state must reject the warm path and safely
  prefill; it must never restore attention KV alone.
- Paged OFF: disk L2 must still find the longest stored prompt/history boundary,
  restore it after unload/restart, and prefill only the unmatched suffix.
- Media: cache keys must include media bytes and effective reasoning/KV policy;
  same media may reuse, different media must miss.

## Current matrix

| Row | Required evidence | Status | Current evidence / blocker |
|---|---|---|---|
| Default cache settings | Real UI shows prefix ON, paged OFF, disk L2 ON, TQ OFF; fresh restart preserves values | VERIFIED-LIVE | Current Settings UI showed prefix ON, paged OFF, disk L2 ON, `Engine Selected`, and SSM re-derive ON. Final live-server state after restoring defaults reported `paged_kv_enabled=false`, `live_kv_codec=engine_selected`, disk and prefix enabled, and no loaded models. |
| Selective TQ topology | Real load + generation shows 6 TQ-KV and 23 native Mamba layers; TQ counters advance only on attention slots | VERIFIED-LIVE / QUALITY-PARTIAL | The real Settings UI selected TurboQuant and required explicit 4/4 bits. Telemetry reported `effective_kv_mode=turbo(4,4)`, `converted_turbo_quant_kv_layer_count=6`, six TQ-KV layers after conversion, 23 native Mamba layers, and three compressions. Text and the simple chronological-color video stayed coherent; the prior quantitative-video degradation remains an open quality caveat. |
| Default disk-L2 restart | Quit/restart with paged OFF; no media reattachment; coherent recall and disk+SSM hits | BASELINE-LIVE / CURRENT-PARTIAL | Baseline no-attachment image follow-up after restart: TTFT 1.17 s, 102.6 tok/s, disk L2 hits 4, SSM hits 4. Current patch disk-only partial-prefix unit row passes; UI rerun pending. |
| Paged ON is honored | UI toggle ON; effective stats show paged enabled and a real paged hit | VERIFIED-LIVE | The user enabled paged cache in Settings, first with two blocks and then 64. A cold `CACHE PRIME` turn took 2.06 s TTFT at 105.6 tok/s; its warm continuation returned `CACHE PAGED` at 0.53 s and 101.1 tok/s. Telemetry reported 97 paged/prefix hits, four misses, 64 total blocks, and zero evictions in the 64-block row. |
| Paged partial-prefix hit | Growing/diverging prompt hits full blocks, restores six attention layers at N, restores complete SSM state at N, and prefills suffix | VERIFIED-LIVE | The warm conversation visibly processed the unmatched tail (`517/537` during the run), remained coherent, and finished with 97 paged hits plus matching hybrid-companion activity. Focused tests independently verify the exact restored boundary and 6/23 layer ordering. |
| Paged eviction to disk fallback | Force paged eviction; paged miss/eviction counter rises; same prefix then hits disk L2 with coherent output | VERIFIED-LIVE | With block size 64 and max blocks 2 selected through Settings, the coherent mixed-media follow-up returned the audio code and both image colors at 2.16 s and 108.6 tok/s. Telemetry reported 152 evictions, two disk-L2 hits, and two SSM-companion hits with zero companion misses/rederives for that restore. |
| Paged miss with missing SSM | KV block exists but companion is absent/incomplete; paged block is released; disk is tried; otherwise clean prefill | SOURCE-TESTED / LIVE-OPEN | Focused missing and partial companion rows reject the paged hit; current exact-model UI row pending. |
| Unknown/state-only topology fail-closed | An unrecognized or recurrent-only cache never publishes a token-only paged hit; typed disk/prefill remains reachable | SOURCE-TESTED / LIVE-OPEN | Explicit unknown-cache allowlist test fails closed; state-only Mamba row allocates zero paged blocks and restores the valid disk+companion boundary. Current UI proof is pending. |
| Disk-only partial-prefix reuse | Paged OFF; growing or reasoning-divergent turn hits the longest stored L2 boundary and prefills only suffix | SOURCE-TESTED / LIVE-OPEN | Fresh-coordinator focused row restores the longest 8-token disk+companion prefix and leaves a 3-token suffix. UI restart proof pending. |
| TQ disk restart | TQ ON; restart; disk payload restores encoded attention KV and native Mamba state; output coherent | VERIFIED-LIVE / QUALITY-PARTIAL | Saving TQ 4/4 unloaded the model; the next same-chat turn reloaded and answered `TQ HYBRID OK`. A warm turn answered `TQ WARM OK` at 0.58 s and 73.0 tok/s. Post-media telemetry reported three disk hits, four SSM hits, two safe SSM rederives, 108 paged hits, and the exact 6-TQ/23-Mamba transition. The cold conversion row was 4.9 tok/s and prior detailed-video quality remains open. |
| TQ OFF restoration | Engine-selected/FP16 restart and coherent media/text follow-up | VERIFIED-LIVE | The current app began in engine-selected FP16 and completed image/audio/video/mixed rows. At the end the user restored `Engine Selected` and paged OFF in Settings; the server reported those exact effective settings with the model unloaded. |
| Same/different media isolation | Same media reuses; different image/audio/video bytes miss; post-media text remains coherent | PARTIAL | Baseline same-media restart/follow-up and mixed media passed. Explicit different-media miss counter and current patch rerun remain open. |
| Reasoning OFF/ON/transition | Real toolbar toggles; clean visible content; no hidden-only turn, marker leak, loop, or forced prompt/sampler | PARTIAL | Baseline fresh OFF and fresh ON chats passed. One OFF-to-ON transition repeated a correct line three times; current patch still needs adversarial rerun. |
| Tools | No-tool acknowledgement, auto tool, required tool, valid args, tool-result continuation, failed-tool honesty, loop stop | PARTIAL / DEFECT-REPRODUCED | With the default tools-enabled assistant, the AIFF transcript began correctly as `Amber Lighthouse 7`, then made irrelevant web searches and fabricated retrieval guidance. A real custom agent with Tools OFF and Memory OFF returned `AmberLighthouse7` directly at 0.39 s and 103.4 tok/s. Media transport is proven; automatic tool choice and exact transcript formatting remain separate open behavior. |
| Content/delta streaming | Visible reasoning/content deltas, terminal finish, no truncation or raw protocol markers across text/media/tools | OPEN | Basic baseline live UI output passed; structured stream/tool matrix is not complete. |
| RAM footprint | Activity Monitor / `phys_footprint` during load, prefill, decode, cache store, restart | VERIFIED-LIVE / DELEGATION-OPEN | Activity Monitor visibly reported Osaurus at 17.84 GB after TQ 4/4, paged reuse, and video decode. Runtime RSS was about 1.9 GiB, illustrating why Activity Monitor footprint remains the owning user-facing gate. Concurrent delegation remains untested. |
| RAM warning override | Known-size oversized model triggers warning; user changes the real setting and load proceeds/blocks according to that value | OPEN | External bundle has unknown catalog size, so the warning never appeared in the baseline run. |
| Spawn/delegation RAM preflight | Real settings enable delegation; selected Nemotron target resolves; insufficient-RAM notification is truthful; accepted run completes | OPEN | No current live proof. |
| Model swap/unload | Unload drops volatile paged/SSM state but preserves disk L2; reload restores without stale cross-model state | PARTIAL | Baseline restart persistence passed; repeated model-swap and isolation rows remain open. |

## Current multimodal rows

- AIFF routing: the real file picker accepted `nemotron-audio.aiff`; the
  controlled Tools-OFF/Memory-OFF agent returned `AmberLighthouse7` at 0.39 s
  TTFT and 103.4 tok/s. The missing spaces are a model ASR/instruction issue,
  not a dropped attachment. The Osaurus database stored the user attachment as
  `type=audio`, `format=aiff`.
- Video, FP16 default: `red-green-blue.mp4` returned `Red, Green, Blue` at
  7.74 s and 103.8 tok/s.
- Image plus audio, FP16 default: the first answer returned only the audio. A
  no-attachment follow-up correctly returned the red background and blue
  center square at 0.57 s and 104.5 tok/s; the next post-reload turn combined
  the code and both colors correctly at 2.16 s and 108.6 tok/s. This proves
  both media streams persisted, while one-shot multi-instruction compliance is
  still partial.
- Video, TQ 4/4 opt-in: the same simple chronological-color fixture returned
  `Red, Green, Blue` at 6.99 s and 60.3 tok/s. This is a simple-path pass, not a
  reversal of the already documented fine-detail video-quality caveat.
- Disk storage under the isolated root reached 6.78 GiB after the cumulative
  matrix. Because that root contains all prior rows, this is a total, not a
  clean per-turn compression delta.

## Immediate implementation and proof order

1. Keep “needs typed disk serialization” separate from “cannot use paged
   attention KV plus companion state.” DSV4, ZAYA CCA, rotating/SWA, affine,
   and other unproven cache types remain paged-incompatible; only Mamba/Arrays
   hybrid plus plain/TurboQuant attention slots use the new route.
2. Run focused tests for hybrid paged partial hits, layer ordering, incomplete
   companion rejection, paged eviction followed by disk hit, TQ restoration,
   media salt isolation, and disk-only longest-boundary reuse.
3. Rebuild an isolated Release Osaurus app pinned to the exact engine commit.
4. Drive default, paged ON, tiny-pool eviction, paged OFF, TQ ON/OFF, restart,
   reasoning, tool, RAM-warning, model-swap, and delegation rows through the UI.
5. Record `phys_footprint`, TTFT, token/s, cache tier/counters, effective cache
   topology, visible answer/reasoning, and exact artifact path for every row.

## Non-goals and safety boundaries

- Do not enable TurboQuant or paged cache by default.
- Do not apply generic TQ to DSV4 Flash, OpenPangu, ZAYA CCA, DSV4 composite
  pools, Mamba/SSM/GDN companion state, or rotating/SWA state.
- Do not add forced thinking tags, prompt coercion, sampler changes, token
  biases, synthetic closers, or length caps to make a row look coherent.
- A coherent response without tier telemetry is not cache proof; a cache hit
  without visible coherent multi-turn output is not correctness proof.
