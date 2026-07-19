# Nemotron Omni Multimodal Correctness Checkpoint — 2026-07-19

Status: **PARTIAL — the patched engine passes the rebuilt isolated Release
Osaurus default-cache multimodal matrix; video Thinking-on and 4/4
TurboQuant-KV video accuracy remain open.**

Scope is the locally available non-MXFP4 bundle:

`/Users/eric/models/dealign.ai/Nemotron-Omni-Nano-JANGTQ4-CRACK`

## Current-source root causes

1. The vision projector did not match the bundle's authoritative
   `modeling.py`. The bundle uses `RMSNorm(eps=1e-5) -> Linear ->
   SquaredReLU -> Linear`; Swift used `LayerNorm -> Linear -> GELU -> Linear`.
   The indexed bundle weights contain `mlp1.0.weight` and no norm bias.
2. RADIO ViT blocks used MLXNN's `LayerNorm` default epsilon rather than the
   RADIO/timm `1e-6` contract.
3. The processor contained a media-only synthetic system instruction and
   silently forced `enable_thinking=false`. Those guards masked the projection
   defect and contradicted the caller's visible Thinking setting. They are
   removed; explicit on/off and an omitted control now reach the bundle chat
   template unchanged.
4. `NemotronHOmni.prepare` chunked text prefill but forwarded the entire
   multimodal embedding sequence to the Mamba/attention stack in one call. A
   real 512 px image expands this bundle's prompt beyond 4K positions, so the
   Mamba sequence-quadratic intermediates produced a 76 GiB live-app physical
   footprint high-water mark. Multimodal embeddings are now materialized once
   and sent through the same bounded 512-position prefill used by text.
5. Hybrid cross-turn prefix capture rejected every media-bearing `LMInput`.
   The first image turn therefore stored only full prompt/post-answer keys,
   neither of which is a prefix of the next rendered chat turn. Image/audio
   inputs may now capture the generation-suffix-stripped boundary only when
   every media placeholder is wholly before that boundary. Media tensors stay
   on the split head so the re-derived Mamba state includes the media. Video
   EVS remains excluded because its stable key is only available after
   post-prepare pruning.

## Current direct evidence

| Row | Status | Evidence |
|---|---|---|
| Vision tensor parity | PASS | `/tmp/nemotron_projector_fix_tensor_parity_20260719.log`: exact-bundle Swift projector mean `0.000916`, std `0.189765`, min `-2.8788`, max `9.472`, closely matching the independent PyTorch projector mean `0.000696`, std `0.186325`, min `-2.78125`, max `9.5`. |
| Smoke regression | PASS | `/tmp/nemotron_projector_fix_smoke_tests_xcode2_20260719.log`: 14/14 `NemotronHOmniSmokeTests`, including the deterministic RMSNorm + SquaredReLU contract test. |
| Prompt-contract regression | PASS | `/tmp/nemotron_preencoded_audio_tests_xcode2_20260719.log`: 19/19 focused tests; explicit Thinking on/off and bundle-default behavior remain distinct and no hidden direct-answer instruction is injected. |
| Exact JANGTQ4 full matrix | PARTIAL | `/tmp/nemotron_jangtq4_projector_fix_full_matrix_20260719.log`: 19/20 rows passed across text, image, video, audio, mixed media, multi-turn, media salt, cache, and BatchEngine. Video with Thinking on hit the 512-token test ceiling. |
| Exact JANGTQ4 1024-token matrix | PARTIAL | `/tmp/nemotron_jangtq4_projector_fix_native_1024_20260719.log`: 14/15; video with Thinking on remained the sole length-stop row. Three isolated Swift seeds reproduced the long-reasoning row in `/tmp/nemotron_swift_video_thinking_seed_{1,2,3}_20260719.log`. |
| Independent reference | PASS with topology caveat | `/tmp/nemotron_python_vmlx_jangtq4_reference_20260719/SUMMARY.json` passed 13/13 image/video/audio/multi-turn rows. `/tmp/nemotron_python_reference_thinking_on_smpte_video_20260719.log` stopped normally, but that dispatcher sampled four images rather than exercising Swift's native 32-frame temporal-video tower, so it does not prove a Swift decode defect. |
| Media-boundary source regression | PASS | `/tmp/nemotron_media_hybrid_strip_tests_swift_20260719.log`: 6/6 focused tests. The media tensor is present only on the prefix head, the suffix is media-free, placeholders after the boundary fail closed, and text/dense/no-cache controls retain their prior behavior. |
| Patched Release exact-bundle matrix | PARTIAL | `/tmp/nemotron_jangtq4_patched_omni_192_20260719.log`: 19/20 with bundle sampling defaults and fixed seed. Text, three-turn text, image, image follow-up, audio, mixed media, media-salt isolation, hybrid SSM, BatchEngine image/audio, video Thinking off, and repeated-video disk alias passed. Video Thinking on remained the sole 192-token length stop. |
| Patched direct physical footprint | PASS for direct diagnostic only | `/tmp/nemotron_jangtq4_patched_footprint_192_20260719.log`: `phys_footprint_peak` remained 29 GiB through the 20-row exact-bundle matrix, down from the 76 GiB high-water mark observed in the pre-patch live app. This is not the app acceptance gate. |

## Pre-patch live-app reproduction

The isolated Release Osaurus build at vMLX `6fb10658` was operated through the
actual UI under bundle identifier `com.dinoki.osaurus.nemotron20260719proof`.
The exact JANGTQ4 model was selected from the user-configured
`/Users/eric/models` storage path with Thinking visibly off. A two-region image
was correctly described as yellow over blue at 115.1 tok/s, and the text-only
follow-up correctly recalled those colors at 55.4 tok/s. However, the first
turn reached 76 GiB physical footprint and the follow-up visibly re-prefilled
`0/4729`. Its isolated L2 database contained only the full 4433/4577 and
4729/4758 boundaries, confirming that media-prefix reuse had not occurred.

That run is reproduction evidence for the two additional root causes above;
it is not verification of the current patch. A newly pinned and rebuilt app
must show both lower physical footprint and a real partial-prefix/L2 restore.

The 32-frame Swift default is retained. The bundle-side Nemotron video helper
and the retained Python JANG tool both define `target_frames=32`; reducing the
frame count merely to shorten reasoning would be an unproven behavior change.

## Osaurus boundary already traced

Osaurus constructs real image, video, and audio content parts in
`ChatView.buildUserChatMessage`, lowers video data URLs and audio PCM/container
payloads in `ModelRuntime`, and builds `MLXLMCommon.UserInput(chat:)` in
`MLXBatchAdapter`. The companion Osaurus change recognizes
`config_omni.json` as a VLM sidecar so this exact bundle is selectable through
the multimodal UI instead of being filtered as text-only.

## Rebuilt Osaurus live proof

Osaurus pinned this engine at
`4634af5151ffd71262d180e32962939dd8b2263f`, built an ad-hoc-signed Release app
at `/tmp/osaurus-nemotron-proof-dd-4634/Build/Products/Release/osaurus.app`, and
ran it under the isolated bundle `com.dinoki.osaurus.nemotron4634proof` and
root `/tmp/osaurus-nemotron-ui-proof-20260719-4634`. The exact JANGTQ4 bundle
was selected from `/Users/eric/models` through the real UI; no MXFP4 bundle was
loaded.

With Thinking off, prefix on, paged RAM off, SSD L2 on, SSM re-derive on, and
live KV codec `Engine Selected`, the UI produced:

- correct image at 6.46s TTFT / 103.4 tok/s and correct no-attachment recall at
  0.95s / 103.2 tok/s;
- correct post-restart image recall at 1.17s / 102.6 tok/s, with fresh-process
  telemetry showing four disk-L2 hits and four SSM-companion hits;
- correct audio transcript at 1.05s / 101.3 tok/s;
- correct mixed image/audio result at 5.53s / 101.4 tok/s;
- SMPTE/timecode video recognition at 12.20s / 109.3 tok/s, with a remaining
  fine-detail miss on the final frame number;
- a correct fresh Thinking-on mixed-media answer at 5.56s / 101.8 tok/s and a
  correct no-attachment follow-up at 0.95s / 101.9 tok/s.

`vmmap -summary` measured a 17.6-17.9 GiB physical footprint and 20.3 GiB peak
during the current live matrix, down from the pre-patch 76 GiB high-water mark.
The isolated chat evidence is retained in
`/tmp/osaurus-nemotron-ui-proof-20260719-4634/chat-history/history.sqlite`.

The default fresh-load telemetry was six FP16 KV layers plus 23 Mamba layers,
disk-backed restore and SSM companion state required, zero TurboQuant-KV
layers, and paged cache off. JANGTQ4 describes the weights and was not reported
as TurboQuant KV.

## Explicit TurboQuant-KV live row

The real Server -> Settings -> Cache UI was changed to TurboQuant with explicit
4-bit key and value widths, saved, and the app restarted. Telemetry showed a
selective transition from six KV + 23 Mamba layers to six TurboQuant-KV + 23
Mamba layers, three compressions, four disk hits, four SSM companion hits, and
zero paged hits. Mixed image/audio stayed correct at 6.71s / 21.3 tok/s and its
cached follow-up stayed correct at 0.97s / 68.3 tok/s.

The video row is not accepted for quality. On the same prompt and fixture, the
4/4 cache run coherently recognized SMPTE bars but misread a roughly five-second
clip as five minutes / 24 frames. After restoring `Engine Selected`, saving,
and restarting, the FP16-cache run tracked the counter to about 140 at 11.91s
TTFT / 110.6 tok/s. Fresh telemetry then confirmed effective `fp16`, zero
TurboQuant-KV layers, and paged cache off.

## Remaining release limits

- Video with Thinking enabled still length-stops in the deterministic direct
  matrix. Do not force Thinking off or change sampling to manufacture a pass.
- One off-to-on Thinking transition follow-up repeated a correct sentence
  three times; a fresh Thinking-on chat did not reproduce it.
- One transcript-only audio request selected a valid but unsolicited
  `share_artifact` tool. That is a model/tool-choice caveat, not a media
  transport or schema-parser failure.
- 4/4 TurboQuant-KV is structurally selective and cache-compatible for this
  hybrid topology, but the live video-detail A/B is a quality failure.
