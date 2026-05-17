# Qwen3VL Video Processor Config Repair - 2026-05-17

Scope: `Qwen3.6-35B-A3B-JANGTQ-CRACK` and any Qwen3VL-family bundle that ships
both `preprocessor_config.json` and `video_preprocessor_config.json`.

## Root Cause

The Swift VLM factory loaded only `preprocessor_config.json` (or
`processor_config.json`). Qwen3VL video-specific settings from
`video_preprocessor_config.json` were ignored, and `Qwen3VLProcessor` resized
video frames with image-only pixel math.

That was not a model coherency issue and not a sampler issue. The runtime was
using the wrong processor contract for video.

## Fix

- `VLMModelFactory.loadProcessorConfig` now embeds
  `video_preprocessor_config.json` under `video_preprocessor_config` before the
  processor is decoded.
- `Qwen3VLProcessorConfiguration` decodes nested `size.shortest_edge` /
  `size.longest_edge` as pixel budgets, while still accepting legacy
  `min_pixels` / `max_pixels`.
- `Qwen3VLProcessor` now applies Qwen3VL video resize math using frame count,
  temporal patch size, spatial factor, and the video pixel budget:
  `ceil(num_frames / temporal_patch_size) * temporal_patch_size * H * W`.
- `MediaProcessing` exposes a CI-frame sequence path so processors can sample
  frames first, then choose the video target size with real frame count before
  converting to `MLXArray`.
- `VLBench.runVideoSmoke` now sets `enable_thinking=false`, matching the other
  VL video rows, so the smoke proves visible answer behavior rather than
  spending a short budget in default thinking mode.

No hidden temperature, top-k, repetition-penalty, forced stop, or fake reasoning
closure was added.

## Live Proof

Local artifact directory:

```text
docs/local/live-model-matrix/20260517T_qwen35_qwen3vl_video_config_fix/
```

Rows:

- `vl_batch_chat_after_video_config.out`: `Qwen35MoE` +
  `Qwen3VLProcessor`; compile OFF and ON both ground the red/blue image and the
  text-only follow-up answers `Red`.
- `vl_video_smoke_final.out`: 1080p fixture loaded through
  `UserInput(videos:)`; `prepare()` attaches `LMInput.video`, emits video pixels
  shape `[560, 1536]`, and the model returns visible coherent video content:
  "A series of colorful, rectangular blocks..."

Verification:

```sh
swift build -c release --product RunBench
VMLINUX_MODEL_FACTORY_TRACE=1 \
  BENCH_MODEL=/Users/eric/models/dealign.ai/Qwen3.6-35B-A3B-JANGTQ-CRACK \
  BENCH_VL_BATCH_CHAT=1 BENCH_MAX_TOKENS=96 \
  .build/release/RunBench
VMLINUX_MODEL_FACTORY_TRACE=1 \
  BENCH_MODEL=/Users/eric/models/dealign.ai/Qwen3.6-35B-A3B-JANGTQ-CRACK \
  BENCH_VL_VIDEO=1 BENCH_MAX_TOKENS=64 \
  .build/release/RunBench
```

`swift test --filter QwenVLIntExtentTests --jobs 2` remains blocked by the
local CLI test toolchain before reaching the targeted test:
`no such module 'Testing'`.

## Remaining Boundary

The full high-resolution `BENCH_VL_MIXED=1` T4 video row is still not a
production-clear row. The local bundle's `video_preprocessor_config.json`
declares a large video budget (`longest_edge=25165824`), so the real processor
contract can still keep the 1080p fixture large. That needs a separate
throughput/scaling gate, not a fake frame clamp.
