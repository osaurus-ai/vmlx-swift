# ZAYA selective TurboQuant cache checkpoint — 2026-07-19

Status: **PARTIAL — exact-model Swift runtime proof passes; no live Osaurus UI proof yet.**

## Scope

- Bundle: `/Users/eric/models/OsaurusAI/Osaurus-AppleScript-8B-JANG_6M`
- Architecture: ZAYA, 80 decoder entries: 40 CCA-attention layers and 40
  MoE-only entries.
- Cache policy under test: paged RAM default off, SSD L2 on, TurboQuant KV
  default off and user-opt-in only.

## Current-source root cause

Before this branch, `ZayaModel.newCache` used `ZayaCCACache` for the 40 real
attention layers and `KVCacheSimple` as unused placeholders for the 40 MoE
layers. Generic TQ conversion only promoted top-level `KVCacheSimple`, so an
enabled TQ policy compressed/counted the unused placeholders while the real KV
inside every `ZayaCCACache` stayed native. Telemetry could therefore claim a
TQ transition without compressing a single ZAYA attention layer.

## Required invariant

- TQ encodes only the internal attention KV in each `ZayaCCACache`.
- `conv_state` and `prev_hs` remain native fp32.
- SSD L2 always stores both native CCA arrays atomically with attention KV.
  ZAYA `turbo(4,4)` and higher may store typed encoded attention KV; sub-4-bit
  live TQ uses a raw prompt-boundary disk record because exact-model `3/3`
  encoded-disk A/B drifted while raw-disk `3/3` did not.
- MoE-only decoder entries use an explicit state-free placeholder and never
  count as KV or TQ layers.
- ZAYA remains paged-incompatible until token-addressable paged blocks can
  restore an exact matching CCA prompt boundary. Explicit paged opt-in must be
  reported as ineffective, not silently treated as a paged hit.

## Proof matrix

| Gate | State | Evidence / remaining work |
|---|---|---|
| Selective CCA KV promotion | PASS-RUNTIME | Exact JANG_6M runs reported 40 native CCA layers plus 40 real attention-TQ layers; the 40 MoE placeholders were not counted. |
| Native fp32 CCA preservation | PASS-FOCUSED | Current source passed 22 selected ZAYA cache checks, including native CCA state, TQ promotion, disk atomicity, cross-layer boundary rejection, and the four-bit disk floor. Log: `/tmp/vmlx_zaya_selective_tq_unit_final2_20260719.log`. |
| TQ-native SSD round trip | PASS-RUNTIME-4BIT | Current source at `4/4` stored 40 `zayaCCATQ` plus 40 native CCA pairs, no raw ZAYA KV, hit `297/340`, and produced the same `CERULEAN-47` answer warm/cold (13.28 vs 10.43 tok/s). Log: `/tmp/vmlx_zaya_jang6m_tq44_current_live_20260719.log`. |
| Sub-4-bit disk safety | PASS-RUNTIME-3BIT | Current source at `3/3` still transitioned the live 40 attention layers to TQ, but stored 40 raw ZAYA KV plus native CCA pairs; `297/340` hit and warm/cold both returned `CERULEAN-47` (10.32 tok/s each). Log: `/tmp/vmlx_zaya_jang6m_tq33_current_live_20260719.log`. |
| Accurate topology telemetry | PASS-FOCUSED | Explicit MoE placeholder and nested ZAYA TQ counting passed the focused source/runtime checks. |
| Paged-off SSD full hit | NOT-APPLICABLE-EXACT | ZAYA disk restore intentionally rejects exact path-dependent prompt boundaries; a growing partial boundary is required. UI must report paged ineffective rather than a false hit. |
| SSD partial-prefix hit | PASS-RUNTIME | Both current-source real-bundle rows hit `297/340`, then prefilled 43 tokens. Four retained records occupied 53 MB for `3/3` raw-disk policy and 23 MB for `4/4` typed-TQ disk policy. |
| Restart SSD restore | OPEN | Requires isolated Release Osaurus restart. |
| Thinking off coherence | PASS-RUNTIME / UI OPEN | Both live rows used `enable_thinking=false`, emitted visible answers with no reasoning/tool markers, and stopped normally. Visible app confirmation remains open. |
| Thinking on coherence | OPEN | Requires real UI turns and visible channel inspection. |
| AppleScript direct tools | OPEN | TextEdit, Calculator, Safari, success finalization, and non-request feedback. |
| Spawn/delegation tools | OPEN | Configure AppleScript delegate in Settings and exercise inherited tool scope. |
| RAM admission override | OPEN | Prove strict refusal, user toggle override, load, unload, and Safe Auto restore. |
| Activity Monitor + tok/s | OPEN | Record physical footprint, TTFT, prefill, cold/warm decode rate. |

No merge-readiness claim is permitted until the source tests pass and the
isolated Release Osaurus app is operated through its visible UI for the live
rows above.

## Failure that determined the disk policy

The first exact `turbo(3,3)` encoded-disk row restored the expected structural
payload (`40 zayaCCATQ`, `40 skip`, all 40 native CCA pairs) and recorded two
disk hits, but its warm answer became incoherent and lost `CERULEAN-47`; the
cache-off comparison answered correctly. Log:
`/tmp/vmlx_zaya_jang6m_selective_tq_live_20260719.log`.

Two controlled follow-ups separated codec loss from companion-state restore:

- raw KV plus the same typed native CCA restore was coherent;
- live `turbo(3,3)` plus a raw SSD boundary was coherent;
- encoded SSD `turbo(4,4)` was coherent for the same partial restore.

The implementation therefore does not silently change a user's selected live
bit width. It keeps sub-4-bit ZAYA disk boundaries lossless and does not persist
a sub-4-bit TQ-native post-answer boundary. A raw stripped boundary remains
available for correct partial reuse.

## Required follow-on family matrix

ZAYA proof does not prove any other hybrid topology. Each row needs current
source inspection plus an isolated Release Osaurus UI run with cold/warm,
partial-prefix, restart, cache-size, TTFT/tok/s, and coherence evidence.

| Family/topology | Required TQ ownership | Current state |
|---|---|---|
| Qwen 3.5/3.6 and Ornith GatedDeltaNet | TQ only full-attention KV; GDN/Arrays recurrent state native with exact-boundary companion restore or async rederive | NOT CURRENT-LIVE-VERIFIED |
| Nemotron hybrid/Mamba | TQ only attention KV; Mamba convolution/SSM state native and boundary-aligned | NOT CURRENT-LIVE-VERIFIED |
| LFM/Jamba/Falcon-H1 and other SSM hybrids | TQ only attention KV; typed recurrent companion state native | NOT CURRENT-LIVE-VERIFIED |
| Gemma 4 rotating SWA | TQ only eligible full-attention KV; rotating/SWA ring state and geometry remain native | NOT CURRENT-LIVE-VERIFIED |
| MiniMax and ordinary full-attention models | TQ may own all real KV layers; no fake placeholder layers may be counted | NOT CURRENT-LIVE-VERIFIED |
| DSV4 hybrid pool and OpenPangu special cache | Generic TQ must remain ineligible; native typed cache/restore only | NEGATIVE UI PROOF OPEN |
| VL/video variants of the rows above | Same topology rules plus real-media cache salt, companion state, restart, and post-media text/tool turns | NOT CURRENT-LIVE-VERIFIED |
