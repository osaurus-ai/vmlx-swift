# vMLX Server Panel Engine Contract - 2026-05-15

This records the engine-side contract for the Osaurus server-oriented panel.
The source of truth is `VMLXServerRuntimeSettings` in `MLXLMCommon`; the UI
should persist and send that shape instead of inventing panel-only flags.

## Core Rule

Every knob is either an explicit user override or `nil` / `auto` / `engineSelected`.
The engine must not turn a failing model row into a pass by silently adding
sampling clamps, hidden EOS guards, forced reasoning closure, or cache-disabled
fallbacks.

## Panel Mapping

| Panel section | Engine type | Notes |
| --- | --- | --- |
| Server Settings | `VMLXServerNetworkSettings` | Host, port, optional API key, served name, rate limit, timeout, log level, CORS. |
| Concurrent Processing | `VMLXServerConcurrencySettings` | Continuous batching stays on by default because it is the gateway to prefix reuse, paged KV, block disk L2, and architecture-specific restore. |
| Prefix Cache | `VMLXPrefixCacheSettings` | Memory limit can be explicit MB, percent, or engine default. TTL `nil` means no expiration. |
| Paged KV Cache | `VMLXPagedKVCacheSettings` | Mutually exclusive with legacy prompt disk cache. Persistent paged cache uses block disk L2. |
| KV Cache Quantization | `VMLXKVCacheCodec`, `turboQuantKeyBits`, `turboQuantValueBits`, and `VMLXStoredKVCacheCodec` | Live codec and stored codec are separate so TurboQuant KV can be live while disk storage remains engine-selected. TurboQuant live KV requires explicit key/value bit widths. |
| Disk Cache | `VMLXDiskCacheSettings` | Legacy prompt disk cache only. The validator errors if this runs with paged KV. |
| Block Disk Cache L2 | `VMLXBlockDiskCacheSettings` | Persistent cache layer for paged KV and architecture-specific cache blocks. |
| Power Management | `VMLXServerPowerSettings` | JIT load, wake-on-request, light sleep, deep sleep. Deep sleep must be later than light sleep. |
| Performance & Generation | `VMLXServerGenerationDefaults` | `nil` means bundle metadata first, then documented engine fallback. Do not write hidden default sampling guards here. |
| Tool Integration | `VMLXServerToolSettings` | MCP config, auto tool choice, parser overrides, reasoning parser override, optional custom template. |
| Multimodal Support | `VMLXServerMultimodalSettings` | Auto/force off/force on VLM mode plus video/audio toggles and media-salt cache requirement. |
| Speculative Decoding / MTP | `VMLXServerMTPSettings` | Auto does not launch MTP unless `MTPBundleStatus.canAutoLaunchMTP` is true. Force-on errors until accept/reject decode is verified. |

## Gateway Runtime States

The server panel should expose each loaded model as a gateway session with at
least these states:

- `stopped`: no process/model resident.
- `loading`: loader or JIT load in progress.
- `ready`: model loaded and idle.
- `generating`: at least one request in flight.
- `lightSleeping`: allocator/cache pressure reduced, fast wake expected.
- `deepSleeping`: model weights unloaded or equivalent RAM release happened.
- `errored`: last load/generation failed with a structured error.

Sleep/wake is a lifecycle operation. It must not mutate sampling defaults,
parser selection, cache topology, or MTP mode.

## Sampling Defaults

Server settings are not a place to hide model-family behavior changes.
`VMLXServerRuntimeSettings.resolvedGenerateParameters(generationConfig:fallback:)`
is the server-side merge point:

1. Start from the caller fallback.
2. Apply the bundle's `generation_config.json`.
3. Apply only explicit server/UI overrides.

`nil temperature`, `nil topP`, `nil topK`, `nil minP`, and
`nil repetitionPenalty` mean "use the bundle generation config, then the
engine fallback if the bundle is silent." A requested greedy decode stays
greedy. A requested `repetitionPenalty = 1.0` is a no-op, not a signal to
install a family floor. If a family loops at a requested sampling point,
record the runtime failure and fix template/cache/position/state handling.

## Cache Topology Requirements

For each model row the panel should show the topology actually used:

- prefix cache enabled/disabled plus hits/misses;
- paged KV enabled/disabled plus block counts;
- block disk L2 enabled/disabled plus bytes written/read;
- TurboQuant KV live codec, if enabled;
- SSM companion cache hits/misses for hybrid SSM or linear-attention models;
- media salt behavior for VL/video/audio models;
- DSV4 CSA/HSA/SWA cache detail for DSV4-family models;
- active routed expert mode and physical footprint for JANGTQ/MoE models.

If a layer is not legitimate for a model, the row should say `N-A` with the
reason. Do not show a fake zero counter as a pass.

`VMLXServerRuntimeSettings.cacheCoordinatorConfig(...)` is the panel-to-engine
bridge. It maps paged cache, block-disk/legacy disk selection, disk max GB,
disk directory, SSM re-derive, model-key isolation, default max KV size, and
explicit TurboQuant KV bit widths into `CacheCoordinatorConfig`. If
`liveKVCodec == .turboQuant` but the bit widths are missing, validation errors
and the builder does not silently invent 3/3 or any other hidden codec.

## Batch Scheduler Requirements

Single-batch and multi-batch are different claims:

- B=1 proves the regular BatchEngine path can prefill, decode, stop, store,
  and stream coherently for one request.
- B>1 must prove live scheduler overlap with `activeCount >= 2`, per-slot
  sampler isolation, per-slot KV policy isolation, and no cross-slot token
  drift versus a deterministic B=1 reference.
- TurboQuant KV must be covered both as mixed plain/TQ slots and all-TQ slots.
- Disk L2 restore must be covered with a fresh coordinator/session, not only
  with a warm in-process paged cache.
- Hybrid SSM/linear-attention/CCA rows must carry the companion state or disk
  format-v2 state along with the KV arrays; a KV-only hit is a false positive.

## Multimodal/VL Requirements

VL models add cache and shape invariants beyond text:

- image/video/audio payloads must affect `mediaSalt`; text-only follow-up turns
  may have nil salt, but media turns with identical token IDs and different
  tensors must not alias;
- Qwen-style VL models need 3-axis MRoPE position IDs and image/video THW grid
  metadata to stay aligned with the text stream;
- restore paths must respect 2D vs 3D/4D tensor ranks before reading sequence
  dimensions, otherwise they must fall back to fresh prefill with an explicit
  artifacted miss;
- JANGTQ VL matmul/Hadamard paths must preserve rank-2, rank-3, and rank-4
  shapes through dense, routed, and vision-tower layers;
- MTP+VL is not considered live until draft text, media embeddings, media salt,
  and accepted-token base cache updates are all proven together.

## MTP Contract

`MTPBundleStatus.mode == preserved_enabled` means the bundle preserved MTP
metadata/tensors. It does not mean speculative decode is live. The engine may
only auto-launch MTP when status says a verified accept/reject runtime exists.

MTP cache rules:

- draft cache is separate from base cache;
- rejected draft tokens never enter base KV/paged/disk state;
- only accepted tokens are appended to base cache;
- VL bundles with MTP must keep media embeddings, media salt, and draft text
  state separate.

## Live Proof Dependency

The panel should not claim a model/config is production-ready until the
corresponding artifact exists under `docs/local/live-model-matrix/` or an
equivalent Osaurus release gate. Required proof is live multi-turn output,
token/s telemetry, low physical footprint where relevant, and cache topology
hits for the model family.
