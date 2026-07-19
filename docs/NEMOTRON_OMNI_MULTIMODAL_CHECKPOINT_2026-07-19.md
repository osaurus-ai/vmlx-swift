# Nemotron Omni Multimodal Correctness Checkpoint — 2026-07-19

Status: **PARTIAL — patched engine is not yet accepted in the rebuilt Osaurus UI.**

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

## Remaining release proof

- Commit and push the bounded media-prefill and safe hybrid-media-boundary
  change, then pin the exact revision in the isolated Osaurus checkout.
- Run Osaurus multimodal-content and VLM-detection focused tests.
- Build and ad-hoc sign an isolated Release Osaurus app under the proof bundle
  identifier and proof root.
- Use Computer Use to confirm settings and visible real-user flows for image,
  video, audio, mixed media, multi-turn recall, media-salt isolation, and
  Thinking on/off with this exact JANGTQ4 bundle.
- Capture TTFT/token rate, Activity Monitor physical footprint, effective
  prefix/paged/L2/TurboQuant telemetry, and a real L2 restore before changing
  this checkpoint from PARTIAL.
- Do not call video Thinking-on passed while it still length-stops, and do not
  force Thinking off to manufacture a pass.
