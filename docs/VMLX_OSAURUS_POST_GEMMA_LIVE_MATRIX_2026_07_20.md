# vMLX Swift / Osaurus post-Gemma live model matrix — 2026-07-20

## Proof contract

This ledger is for the real Release-config Osaurus UI. Source inspection,
focused tests, CLI harnesses, API calls, and old artifacts may diagnose a row,
but they do not close it. A closed user-facing row requires all of:

- the exact model bundle and quant/runtime identified;
- the effective app settings shown in the UI;
- visible load and response behavior in the app;
- persisted turn/tool data checked when streaming or parsing is in doubt;
- TTFT, token/s, response-token count, and Activity Monitor `phys_footprint`;
- the applicable prefix, paged-RAM, SSD L2, SSM/linear-state, and TurboQuant
  counters before and after the request;
- coherent multi-turn behavior, not load-only or one short completion;
- real image/video/audio payloads for advertised media capabilities.

`VERIFIED-LIVE` is reserved for rows meeting that contract. `PARTIAL` names
the evidence that exists and the missing gate. `BROKEN` names a reproduced
failure. `OPEN` means no current-source live UI proof yet.

## Default policy that every model row must re-check

- Prefix cache: on.
- Paged RAM cache: off. It may be enabled explicitly for a separate UI row.
- SSD block cache: on and independently usable while paged RAM is off.
- Live TurboQuant KV encoding: off (`engine_selected` resolves to native KV in
  Osaurus). It may be enabled explicitly for a separate UI row and must be
  restored to off afterward.
- TurboQuant must apply only to compatible KV attention state. Hybrid
  SSM/GDN/linear-attention, rotating SWA, recurrent, CCA/DSA/MLA, and other
  companion state must retain the architecture's native typed state and its
  proven prompt-boundary restore/re-derive contract.
- DSV4 Flash and OpenPangu must not receive generic TurboQuant KV merely from a
  name or model-family guess.
- No forced thinking tags, prompt coercion, sampler clamps, EOS bias, hidden
  repetition penalties, synthetic defaults, or output-length masking may be
  used to make a model appear coherent.

## Current Release UI evidence

Proof app:

- bundle id: `com.dinoki.osaurus.gemma4alignmentproof20260720`
- app: `/private/tmp/osaurus-gemma4-alignment-release-derived-20260720/Build/Products/Release/osaurus.app`
- isolated root: `/private/tmp/osaurus-gemma4-alignment-proof-root-20260720-1414`
- Osaurus source tree: merged `main` at `59334020f54950f16247a2b60474de7b11fbb54b`
- vMLX Swift pin: `f2b184841e98d969e46dec83109f27cd7bb57357`
- production Osaurus preferences/keychain were not used.

### New-chat SSD checkpoint reproduction

A later isolated Release app used bundle id
`com.dinoki.osaurus.applescriptemergency20260720`, SHA-256
`114fbe282e9e2872abe88ca8c991da6ebc1b7c9e19f0ef5029bd94c54511fd9b`,
root `/private/tmp/osaurus-applescript-emergency-live-root-20260720`, and the
same exact vMLX pin. The real UI showed Prefix On, paged RAM Off, Disk Cache
On, Engine Selected/native, and SSM rederive On with exact `Ornith 1.0 9B
MXFP8` and Thinking Off.

SSD persistence itself worked: a fresh process and new chat restored a
2,201-token boundary with 48 recurrent companion states. The reported
from-zero behavior was also reproduced. A later new-chat request logged
`HIT disk boundary=2234 remaining=0 ssm=48 fmtV=2 ... skipExactDisk=true`,
then the UI visibly advanced raw prefill from zero in 512-token chunks. Source
root: `CacheCoordinator.fetch` omits N in its preferred probes but re-admits N
from `candidateTokenCounts`; the hybrid GDN full-hit guard then lacks an N-1
seed and rolls back to full prefill. The candidate excludes N from every probe
when `skipExactDiskBoundary` is set, selecting the longest safe partial
boundary instead.

The same trace showed repeated synchronous rewrites of 125-143 MB KV files
and 26 MB recurrent sidecars plus eviction churn at the 10 GB cap. The
candidate adds current-process-validated no-rewrite paths for both payload
types. KV files require matching size, mtime, token count, and SQLite size.
SSM pairs require matching tensor and sidecar fingerprints plus completeness,
state-count, boundary, model-key, and linked-KV metadata. Changed, missing,
corrupt, unindexed, or unvalidated inherited files still take the healing
write. Skipped SSM stores touch only filesystem metadata so the paired quota's
recency remains aligned without evaluating or serializing recurrent tensors.

Current Xcode 26.6 / Swift 6.3.3 Metal-backed source tests:

- seven `MLXLMTests.diskCache*` tests passed in 1.051 seconds;
- all ten `SSMStateCacheTests` passed in 0.039 seconds;
- all 29 `CacheCoordinatorTopologyFocusedTests` passed in 9.548 seconds,
  including Gemma mixed rotating/TQ, Nemotron Mamba/TQ, DSV4 typed disk,
  ZAYA CCA, paged eviction, paged-off partial SSD restore, and Ornith GDN
  snapshot-detachment contracts;
- the new exact indexed-boundary exclusion passed independently in 0.038
  seconds before the suite run.

These changes still have source and focused-test candidate evidence only. The
row stays **PARTIAL** until a rebuilt Release app visibly proves suffix-only
prefill, coherent output, both `disk-store SKIP` and `ssm-store SKIP`
telemetry, and stable payload bytes.

### Stable system/tool prefix across unrelated new chats

The same isolated Release app subsequently reproduced a second, independent
SSD miss after the exact-boundary rollback was removed. With Prefix On, paged
RAM Off, Disk Cache On, native KV, SSM rederive On, Ornith MXFP8, and Thinking
Off visible in the UI, an unrelated new chat showed `Prefill 1536/2420` and
logged `MISS all tiers tokens=2420 skipExactDisk=true`. Prior turns had stored
their complete prompt/history boundaries, but none had stored the leading
system instructions plus tool schemas without the previous user message.
Content-addressed lookup therefore had no prefix shared by the new session and
correctly prefetched from zero.

The current candidate derives two separate exact chat-template boundaries:

- canonical full history without the assistant generation rail; and
- the leading system/developer instructions plus the same tool schemas, also
  without the generation rail.

Both are accepted only when their token IDs are a byte-for-byte prefix of the
active prompt; templates that reorder or conditionally rewrite the content fail
closed. The stable boundary is passed explicitly to the disk probe so it cannot
fall out of the index's 128-largest-length scan. Dense caches trim to it.
Disk-backed rotating Gemma SWA and Qwen/Ornith/Bonsai GDN/SSM topologies force
an architecture-native boundary rederive rather than flattening typed state.
Gemma 4 is wired explicitly through its custom `Gemma4Processor` for text-only
requests; Qwen3VL is likewise wired only on its text branch. Their image/audio
placeholder paths remain excluded until separate media-salt restore proof.
After an entry has actually been written or deserialized in the current
process, matching file fingerprint plus SQLite metadata suppresses redundant
warm-turn rederive/write; inherited filenames alone are not trusted.

Current edited-tree evidence:

- 5/5 canonical-template boundary tests pass: system+tools, tools-only,
  non-prefix fail-closed behavior, no bare-BOS persistence when neither stable
  instruction nor tool material exists, and distinct stable token prefixes
  after a template configuration change;
- 17/17 BatchEngine/TokenIterator/Native-MTP cache wiring tests pass;
- the Metal-backed current-process validation test passes, including fresh
  process false-before-fetch, true-after-deserialize, and false-after external
  fingerprint mutation;
- the hybrid validation test proves a current-process-validated KV file alone
  cannot suppress healing when the matching complete recurrent sidecar is
  absent, then accepts the same boundary after the companion pair is stored;
- the bounded-index regression stores more than 128 larger candidate lengths,
  proves the stable boundary is absent from the normal scan, and restores it
  only through the processor-proven preferred-boundary path;
- 42/42 cache-topology tests, 10/10 recurrent companion tests, and 8/8 disk
  cache tests pass after the validation-pair change; the Gemma processor source
  row additionally proves normalized tools and both text-only boundary fields
  reach `LMInput`.

Status remains **PARTIAL**. Required live closure is an isolated Release app on
the exact candidate pin showing cold store, unrelated-new-chat partial L2 hit,
full app restart partial L2 hit, truthful visible suffix-only prefill/TTFT, and
coherent output on Ornith/Qwen hybrid plus Gemma 4 JANG_4M rotating-SWA. Qwen
3.5 VL media reuse remains a separate media-salt/placeholder row; the text path
shares the stable-boundary implementation, but no media claim is made here.

| Family / bundle | Architecture and runtime | Current result | Current evidence | Missing before closure |
|---|---|---|---|---|
| Qwen 3.5 dense — `dealign.ai/Qwen3.6-27B-JANG_4M-CRACK` | 64 layers: 48 linear-attention + 16 full-attention; JANG_4M | PARTIAL | Thinking visibly off. Bundle-default row returned the requested two lines at TTFT 1.78 s, 22.5 tok/s, 22 tokens. Explicit greedy Settings row returned the requested two lines at TTFT 1.87 s, 22.5 tok/s, 24 tokens. No tool calls or reasoning deltas. | Multi-turn, required/automatic/no-tool/tool-error rows; real image/video; SSD cold/full/partial/restart; paged-on hot/eviction; explicit TQ on/off; Activity Monitor per row. |
| Qwen 3.5 dense — `dealign.ai/Qwen3.6-27B-MXFP8-CRACK-MTP` | Same hybrid topology; MXFP8; MTP tensors preserved | BROKEN for agent tools; PARTIAL for plain chat | Live Activity showed Native MTP **Not active**. With bundle defaults and Thinking off it made three unsolicited valid `todo` calls, then persisted only `# Qw` (3 tokens) at TTFT 0.82 s and 15.6 tok/s. A second sampled run echoed the instruction. With explicit greedy settings (`temperature=0`, `top-p=1`, `top-k=0`, `min-p=0`, penalty 1) and Tools on, it again made three unsolicited `todo` calls plus `share_artifact`, then produced the requested two lines at TTFT 2.55 s, 15.1 tok/s, 24 tokens. With the agent's Tools switch visibly off, the same greedy model produced exactly the requested two lines at TTFT 2.48 s, 15.7 tok/s, 27 tokens. SQLite contains the same tool calls and final content shown in the Tools-on runs, so the first truncation is not only a SwiftUI content-delta display loss. | Capture raw emitted token/stop reason after tool continuation; test the bundle's `[248046,248044]` effective EOS against its `<|endoftext|>` emission without mutating the bundle; compare another MXFP8 Qwen artifact; audit tool prompt/description pressure and continuation history. Do not attribute this to MTP or plain content streaming. |
| Qwen 3.5/3.6 MoE MXFP8/JANG/JANGTQ | 30 linear-attention + 10 full-attention for local 35B A3B bundles | OPEN | Metadata inventory only. | Full live matrix; do not inherit dense proof. |
| Qwen 3.5 VL | Vision/video path plus hybrid text cache | OPEN | The selected local dense bundles advertise Vision in the picker; no current payload row. | Real image, same-image reuse, different-image salt, video, post-media text/tool continuation, restart/L2 restore. |
| Ornith / Bonsai Qwen aliases | Qwen hybrid backbone with family-specific templates/tools | PARTIAL-LIVE on Ornith; Bonsai remains OPEN in this matrix | Ornith fresh-process/new-chat SSD hits carried 48 recurrent states, but an exact indexed candidate bypassed the skip-exact contract and forced visible full prefill. Two AppleScript parent turns stayed coherent with Thinking visibly Off. | Rebuild with the cache candidate and prove suffix-only partial restore, no duplicate write, same-chat/new-chat/restart coherence; then separate Bonsai, paged/TQ, and media rows. |
| LFM2.5 8B A1B MXFP8 — `dealign.ai/LFM2.5-8B-A1B-MXFP8-CRACK` | `lfm2_moe`; 24-layer Conv1d + full-attention hybrid, ~1B-active MoE; MXFP8 | **BROKEN** for automatic tools and cross-session prompt/cache isolation; PARTIAL for one cold plain-chat row | The Release UI loaded the exact local bundle. Its model picker exposed no Thinking switch because the bundled template contains `<think>` replay markers but no `enable_thinking` kwarg. A no-tool prompt produced the exact two requested lines at TTFT 0.90 s and 179.1 tok/s, but also 442 persisted reasoning characters (162 total generated tokens). An explicit `todo` request spent 9.4 s / 1,611 tokens at 170.7 tok/s with 6,251 persisted reasoning characters, emitted `{"markdown":"Create task LFM25-TOOL-7319"}` as visible content, and executed no tool. Runtime logs recorded 1,583 reasoning deltas, 19 tool-progress/sentinel hints, and `capturedTools=0`; SQLite confirms `tool_calls` is NULL, so this is not only a SwiftUI display issue. A fresh-chat repeat of the original plain prompt then answered with `LFM25-TOOL-DONE-7319`, copied from the intervening tool chat, instead of the requested `LFM25-MXFP8-DEFAULT-7318`; the persisted reasoning likewise states that it saw the wrong prior string while the persisted user turn contains the correct new prompt. During that bad row the UI showed a full `0→3104`-token prefill, and Live Activity stayed at prefix `0/0`, L2 `0/8` while stores rose `16→23`, and SSM `0/0/0`: no hit was reported, so this currently points to in-process hybrid/session state or prompt-materialization contamination rather than a proven SSD restore. Source trace confirms `.lfm2` format inference and the `LFM2ToolMinimal` fallback were selected by `model_type=lfm2_moe`; the bundle itself advertises native tagged Python-call syntax but has no generation-time thinking-off branch. Activity Monitor showed 2.27 GB `phys_footprint` for the isolated proof app after these rows (production Osaurus remained a separate 942.2 MB process). | Trace the exact rendered-token boundary from `ChatSessionWarmup` through `BatchEngine.stepPrefill`; prove whether the wrong suffix originates in warmup/session cache, recurrent companion state, or input-token mutation. Capture the exact raw tagged buffer that produced 19 progress hints but no parsed call. Test the JANG_2L control only after isolating this contamination, then real post-tool continuation, multi-turn, SSD partial restore, explicit paged/TQ, and companion-state correctness. Do not invent an `enable_thinking` toggle the bundle does not implement. |
| MiniMax M2.7 JANG_K | Full-KV MoE | OPEN | Local 80 GB bundle is visible in the picker. Old harness evidence is diagnostic only. | Load/RAM override, coherent multi-turn, full-KV TQ on/off, prefix/paged/L2 partial/eviction/restart, usable speed. |
| MiniMax M2.7 Small JANGTQ | Routed JANGTQ runtime; distinct from TQ KV encoding | OPEN | Local 37 GB bundle is visible in the picker. | Low-footprint active-streaming runtime, output coherency/speed, then independent TQ-KV on/off and cache stack. |
| DSV4 Flash JANG | Native DSV4 hybrid cache topology; local bundle ~97 GB | OPEN | Bundle is visible in the picker. No current live load/decode proof. | Confirm supported artifact/profile first, then RAM override, long coherent multi-turn, prefix/SSD restore, typed cache behavior, and measured speed. The requested near-20 tok/s target is UNVERIFIED. Generic TQ must remain inapplicable unless source and live topology prove otherwise. |
| Nemotron Omni Nano JANGTQ4 | Hybrid/recurrent plus advertised multimodal | OPEN | Bundle is visible in the picker. | Image/video/audio payloads, text/tool continuation, typed companion cache, SSD partial/restart, RAM/speed. |
| ZAYA / AppleScript 16B JANG_4M and curated ZAYA 8B JANG_6M | Tool-specialized agent models; the installed curated 8B bundle is `model_type=zaya`, while the locally named 16B bundle is actually `model_type=gemma4` | PARTIAL-LIVE for the supported dedicated 16B route; 8B remains BROKEN/OPEN | In the current isolated Release UI, exact parent Ornith MXFP8 Thinking Off plus exact dedicated `JANGQ-AI/AppleScript-16B-A4B-JANG_4M` changed visible unsaved `Hello World`→`Hello from OracHQ` once, then in a new chat changed `Hello from OracHQ`→`Hello again` once. Each ended with grounded parent success at 49.9 tok/s (TTFT 1.45 s / 1.41 s), no duplicate mutation, and no Save workflow; the second used one mutation plus one read-only verification. The literal-field validation was in this binary and the former generated-script-in-`content` failure did not recur. | Run exact JANGQ 8B cold/warm controls after the hybrid SSD fix; requested Save, larger-document substring, already-correct text, denial/cancel/error recovery, acknowledgement-only, reasoning on/off, spawn/delegation, footprint, and cache policy rows. Dedicated helpers remain excluded from generic Computer Use's different `agent_action` schema. |

## Required cache sequence per compatible model

1. Confirm the default UI state: paged off, TurboQuant off, prefix on, SSD L2 on.
2. Cold prompt A; record TTFT/tok/s/tokens/footprint and all cache counters.
3. Exact warm prompt A; require a truthful prefix or SSD hit and coherent output.
4. Partial-prefix prompt A+B; require the longest safe block match, not an unsafe
   full hit; record restored token/block counts and architecture companion state.
5. Evict/unload/restart, then repeat A+B with paged still off; require direct SSD
   reuse when a compatible disk block exists, otherwise a truthful cold prefill.
6. Enable paged RAM in the UI; repeat exact and partial prompts; prove RAM hot-tier
   hits, bounded eviction, and SSD fallback after the RAM block is absent.
7. Restore paged off.
8. For compatible models only, explicitly select TurboQuant in the UI and repeat
   cold/exact/partial/restart rows. Record actual compressed layer count, bit
   widths, cache bytes/GB growth, and companion-state restore/re-derive counters.
9. Restore TurboQuant to off/native and repeat one coherence row.
10. Change the memory-safety setting through the UI, save, and prove that a model
    rejected under the conservative setting can load under the explicit user
    override without silently changing unrelated engine limits.

## Immediate investigation order

1. Qwen dense MXFP8 tool/EOS failure versus JANG_4M control.
2. Qwen 35B A3B MoE MXFP8/JANG or JANGTQ control.
3. Qwen VL image/video and hybrid cache-salt/restore rows.
4. LFM2.5 MXFP8 companion-state/cache/tool rows.
5. MiniMax M2.7 full-KV JANG_K, then MiniMax Small JANGTQ runtime.
6. DSV4 Flash supported-artifact/load/speed/cache row.
7. Nemotron Omni multimodal and AppleScript/ZAYA agent completion rows.

No row above is release-ready merely because it appears in the model picker or
loads. The current broad campaign status is **PARTIAL / OPEN**, with the Qwen
27B MXFP8 agent behavior explicitly **BROKEN**.
