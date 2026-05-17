# Media model matrix — every audio/video/image-capable family in vmlx

Pinned to vmlx revision `5adb91b`. Companion to `PARAKEET-RADIO-INTEGRATION.md`
(which is Nemotron-3-specific). This document is the host-team
reference for **every** modality-capable model the engine ships,
their JANGTQ + MXFP4 quant tiers, and the cache topology each one
drives. If your UI is wiring a file picker / drag-drop / streaming
input, this is the table that tells you what to expect.

---

## Quick reference — modality + cache topology

Sorted by media surface. ✓ = native support; ⨯ = unsupported (reject
at host); ◐ = supported via base modality (e.g. video as image stream).

| Model family | model_type | Image | Video | Audio | Cache topology | Hybrid SSM? |
|---|---|---|---|---|---|---|
| **Nemotron-3-Nano-Omni** | `nemotron_h_omni` / `NemotronH_Nano_Omni_Reasoning_V3` | ✓ RADIO ViT | ✓ AVAsset frames | ✓ Parakeet ASR | per-layer mixed: `MambaCache` (23) + `KVCacheSimple` (6 attn) + nil (E layers) | ✓ YES — eager `setHybrid(true)` |
| **Qwen 2 VL** (qwen2-vl-* / qwen2.5-vl-*) | `qwen2_vl`, `qwen2_5_vl` | ✓ ViT + window-attn | ✓ temporal patch | ⨯ | dense `KVCacheSimple` per layer | no |
| **Qwen 3 VL** (qwen3-vl-9b/30b-vl-*) | `qwen3_vl` | ✓ ViT | ✓ config-driven FPS | ⨯ | dense `KVCacheSimple` | no |
| **Qwen 3.5 / 3.6 MoE** (qwen3.5/3.6-VL bundles) | `qwen3_5`, `qwen3_5_moe` | ✓ via Qwen35 path | ✓ video tokens | ⨯ | per-expert routed; **eager `setHybrid(true)`** via osaurus matcher (substring qwen3.5/qwen3.6) | gated SSM in some layers |
| **Gemma 3** | `gemma3` | ✓ SigLIP-ish | ⨯ | ⨯ | dense + sliding | sliding-window only |
| **Gemma 4** | `gemma4` | ✓ MobileNet vision | ⨯ | ⨯ | sliding + full mixed | SWA hybrid (not Mamba) |
| **SmolVLM 2** | `smolvlm` | ✓ Idefics3 | ✓ fps from config | ⨯ | dense | no |
| **Mistral 3** / **Mistral 3.5** | `mistral3` (+ inner `ministral3` / `mistral4`) | ✓ Pixtral ViT | ⨯ | ⨯ | per-layer mixed: `RotatingKVCache` (sliding) + `KVCacheSimple` (full) | SWA hybrid (not Mamba) |
| **Mistral 4 VLM** | inner `mistral4` (via `mistral3` outer dispatch) | ✓ Pixtral | ⨯ | ⨯ | dense + sliding | no |
| **Pixtral** (standalone) | `pixtral` | ✓ | ⨯ | ⨯ | dense | no |
| **PaliGemma** | `paligemma` | ✓ | ⨯ | ⨯ | dense | no |
| **Idefics 3** | `idefics3` | ✓ | ⨯ | ⨯ | dense | no |
| **GLM OCR** | `glm_ocr` | ✓ OCR-tuned ViT | ⨯ | ⨯ | dense | no |
| **LFM2-VL** | `lfm2_vl` / `lfm2-vl` | ✓ | ⨯ | ⨯ | dense | no |
| **FastVLM / Llava-Qwen2** | `fastvlm` / `llava_qwen2` | ✓ | ⨯ | ⨯ | dense | no |
| **MiniMax M2 / M2.7** | `minimax`, `minimax_m2` | ⨯ | ⨯ | ⨯ | per-layer mixed; **eager `setHybrid(true)`** via osaurus matcher | gated SSM |
| **Laguna XS.2 / S.3** | `laguna` (engine-pending) | ⨯ | ⨯ | ⨯ | per-layer mixed: `RotatingKVCache` (SWA, 64 layers) + `KVCacheSimple` (full, 48 layers) | SWA hybrid (not Mamba) |

LLM-only (text):

| Model family | model_type | Cache topology | Hybrid SSM? |
|---|---|---|---|
| **Qwen 3.5 / 3.6 MoE (text)** | `qwen3_5`, `qwen3_5_moe` | per-expert routed; eager hybrid | gated SSM |
| **Holo 3** | aliased into `qwen3_5_moe` | same as Qwen 3.5/3.6 MoE | gated SSM |
| **NemotronH (text-only)** | `nemotron_h` | mixed Mamba + KVCacheSimple per-layer | ✓ YES |
| **Nemotron Cascade-2** | `nemotron_h` (Cascade lineage) | same | ✓ YES |
| **DSV3 / DSV4-Flash** | `deepseek_v3`, `deepseek_v4` | RotatingKVCache (DSV4 default `windowSize=128`); override via `DSV4_KV_MODE=full` | DSV4 = Compressor/Indexer hybrid (not Mamba; separate cache topology) |
| **Kimi K2 / K2.5** | `kimi`, `kimi_k2` (DSV3 lineage) | dense + sliding | no |
| **Mistral 4** | `mistral4` | dense + sliding | sliding-window only |
| **Gemma 2 / 3 / 4 (LLM)** | `gemma`, `gemma3`, `gemma4` | dense + sliding | sliding-window only |
| **GPT-OSS** | `gpt_oss` | dense | no |
| **NanoChat / Lille** | `nanochat`, `lille-130m` | dense | no |
| **OLMo / OLMoE** | `olmo2`, `olmo3`, `olmoe` | dense | no |
| **Apertus** | `apertus` | dense | no |
| **LFM2 / LFM2-MoE** | `lfm2`, `lfm2_moe` | dense | no |
| **Bailing MoE** | `bailing_moe` | dense + per-expert | no |
| **Jamba** | `jamba_3b` | mixed Mamba + Attn | ✓ YES |
| **AfMoE** | `afmoe` | dense + per-expert | no |

---

## Quant-tier matrix (which JANGTQ classes route which model_types)

vmlx dispatches JANGTQ vs MXFP4 vs full-precision based on
`jang_config.json.weight_format`:

- `"mxtq"` → JANGTQ codebook routing (sidecar `jangtq_runtime.safetensors` REQUIRED)
- `"mxfp4"` → standard MLX MXFP4 quantization (no sidecar needed)
- absent / `"none"` → full-precision (BF16/FP16/FP32 from safetensors)

`jang_config.json.quantization.profile` further selects bit width
(`JANGTQ4` → 4 bits, `JANGTQ2` → 2 bits, `JANG_4M` / `JANG_2L` →
mixed). The dispatch table:

| model_type (JANGTQ path) | Routing class | model_types served |
|---|---|---|
| `nemotron_h` + `weight_format=mxtq` | `NemotronHJANGTQModel` | Nemotron-3, Nemotron-Cascade-2, Nemotron-Hyper |
| `qwen3_5_moe` + `mxtq` | `Qwen35JANGTQModel` (LLMModelFactory:56) | Qwen 3.5/3.6 MoE, Holo3 |
| `minimax_m2` + `mxtq` | `MiniMaxJANGTQModel` (LLMModelFactory:117) | MiniMax M2 / M2.7 |
| `deepseek_v3` + `mxtq` | `DeepseekV3JANGTQModel` | Kimi K2/K2.5, DSV3 |
| `deepseek_v4` + `mxtq` | `DeepseekV4JANGTQModel` (with `mxtqBits` from jang_config) | DSV4-Flash, DSV4-Pro |

JANGTQ for new families requires: (a) a quant-aware model class, (b)
factory dispatch entry, (c) the bundle's `jang_config.json` populating
`weight_format=mxtq` + `quantization.bit_widths_used` or `profile`.

Bundles missing the sidecar `jangtq_runtime.safetensors` but
declaring `weight_format=mxtq` will trigger osaurus's
`validateJANGTQSidecarIfRequired` preflight (host-side guard at
`ModelRuntime.swift`, prevents vmlx from hitting an `abort()` in
`TurboQuantSwitchLinear`).

### What VLM JANGTQ looks like

`VLMModelFactory.swift:362-403` merges `jang_config.json` fields into
`config.json` BEFORE decoding so omni / VL configurations see
`weight_format` + `mxtq_bits` + `mxtq_seed` and opt their inner
language model into the JANGTQ path.

For Nemotron-Omni specifically: the bundle reports
`model_type=nemotron_h` (LLM-only), but the presence of
`config_omni.json` flips dispatch to `NemotronH_Nano_Omni_Reasoning_V3`
which constructs a `NemotronHOmni` whose inner text decoder is
JANGTQ-aware via the same merge.

---

## Per-family video processing details

### Nemotron-3-Nano-Omni (`nemotron_h_omni`)

- Frame extraction: `nemotronOmniExtractVideoFrames` (`Preprocessors.swift:577`)
- Default frame budget: 8 frames sampled uniformly
- Override: `processing.frameStride` on `UserInput.Processing`
- Resize: NOT host-side. RADIO's bilinear resize handles arbitrary input.
- Per frame: ~256 vision tokens after spatial-merge 2×2 pixel-shuffle

### Qwen 2 VL / 2.5 VL (`qwen2_vl`, `qwen2_5_vl`)

- Temporal patch size: from `config.temporalPatchSize` (typically 2 — every 2 frames merged)
- ViT window size: `config.windowSize` (typically 8)
- Token IDs: `image_token_id`, `video_token_id`
- FPS hint: derived from frame count / duration; not a fixed config

### Qwen 3 VL (`qwen3_vl`)

- targetFPS: `video_preprocessor_config.json.fps` when present, defaulting to
  Qwen3VL's 2 fps processor default.
- Resize: frame-count-aware video smart resize using
  `video_preprocessor_config.json.size.shortest_edge` /
  `size.longest_edge` as the model's temporal pixel budget.
- Frame extraction loops `input.videos`, samples frames before resizing so the
  target H/W can use the real frame count.
- Padding token: `<|video_pad|>`
- Video token defaults: id=151_656

### Qwen 3.5 / 3.6 (VL bundles via `qwen3_5` model_type)

- Aliased through `Qwen35` class — inherits Qwen 3 VL's video pipeline
- 30B-A3B MoE topology: each frame routes through 6-of-128 experts
- **Important**: hybrid SSM eagerly flipped via osaurus
  `isKnownHybridModel` matcher (substring `qwen3.5` / `qwen3.6`)

### SmolVLM 2 (`smolvlm`)

- FPS from `config.videoSampling.fps` (config-driven, NOT hardcoded)
- Adaptive sampling: 1 fps for duration ≥ 10 s, multiplier for shorter clips
  (see `SmolVLM2.swift:317`)

### Mistral 3 / 3.5 / 4 VLM (`mistral3`)

- **No video support**. Image only via Pixtral ViT.
- If a `video_url` content part is sent, vmlx will reject at the
  preprocessor (no `preprocess(videos:)` implementation).
- Host UI must reject or fall back to image-frame-from-video for
  these families.

### Gemma 3 / 4 (`gemma3`, `gemma4`)

- **No video support**. Image only.
- Same host-side rejection contract as Mistral 3.x.

---

## Per-family audio processing details

| Family | Audio class | Format | Sample rate | Notes |
|---|---|---|---|---|
| **Nemotron-3-Nano-Omni** | `NemotronHParakeetEncoder` | wav/mp3/m4a/flac/ogg via AVAudioConverter | resampled to 16 kHz mono Float32 internally | only model with native audio today |
| All others | — | ⨯ | — | host should reject `input_audio` content parts and surface "model does not support audio" |

Audio is exclusively a Nemotron-3-Nano-Omni capability in the current
engine. Future families with native ASR will need:

1. New encoder class (mirror of `NemotronHParakeetEncoder`)
2. `preprocess(audios:)` on the model's processor
3. Token plumbing for the audio-token id in the chat template

---

## Cache + memory topology by family

### Hybrid SSM (Mamba) — eager `setHybrid(true)` required

These families need the SSM-state companion cache to round-trip across
admission:

- Nemotron-3 Nano Omni
- Nemotron-Cascade-2
- Nemotron-3 reasoning bundles (any future variant)
- Qwen 3.5 / 3.6 MoE family + Holo3
- MiniMax M2 / M2.7
- Jamba 3B

Osaurus `ModelRuntime.installCacheCoordinator` flips this eagerly via
`isKnownHybridModel(name:)`. BatchEngine also auto-flips on first
slot admission via `MambaCache | ArraysCache` detection. Belt + suspenders.

### SWA hybrid (sliding window + full attention) — NOT Mamba hybrid

These have per-layer mixed `RotatingKVCache` + `KVCacheSimple` but
**do not** need `setHybrid(true)` (no SSM-state companion):

- Mistral 3 / 3.5 (per `Mistral3Text.newCache:355-363`)
- Mistral 4
- Gemma 2 / 3 / 4
- Laguna XS.2 / S.3 (engine-pending)

The `setHybrid` flag is specifically for SSM-state round-trip; SWA
hybrids skip it.

### Boundary trap (real bug surface)

`Mistral3Text.newCache`:

```swift
if layer.useSliding, let slidingWindow = args.slidingWindow {
    return RotatingKVCache(maxSize: slidingWindow)
} else {
    return KVCacheSimple()  // ← silent wrong-cache fallthrough
}
```

If a config has `useSliding=true` but `sliding_window` field is
missing/null, the optional binding fails and the layer silently falls
through to `KVCacheSimple`. **Wrong cache type, no error.** Tracked
for engine-side refactor under MC/DC strategy doc §"deferred coverage".

### Dense — single-tier `KVCacheSimple`

Everything else: PaliGemma, Idefics3, FastVLM, Pixtral standalone, GLM
OCR, LFM2-VL, SmolVLM2, Qwen 2/2.5/3 VL (non-MoE), plus all dense LLMs
(GPT-OSS, NanoChat, OLMo, Apertus, LFM2, Bailing MoE, AfMoE).

### DSV4 — dedicated cache topology

DeepseekV4-Flash uses Compressor/Indexer hybrid attention, **not**
Mamba. Per-layer cache list contains custom `DeepseekV4Cache`. vmlx's
auto-flip only matches `MambaCache | ArraysCache`, so DSV4 is
intentionally NOT in the hybrid family list. Cache topology is
controlled via `DSV4_KV_MODE` env (`full` switches new caches to
`KVCacheSimple` for full prompt visibility — set unconditionally at
osaurus launch).

---

## Memory profile (M5 Max, MXFP4 baselines)

Empirical from OmniBench:

| Workload | Model | Peak RAM | Decode tok/s |
|---|---|---|---|
| Text-only 4K prompt | Nemotron-3 30B MXFP4 | ~22 GB | ~65 |
| + 1 image (1024×1024) | Nemotron-3 30B MXFP4 | ~24 GB | ~62 |
| + 8-frame video (256×256) | Nemotron-3 30B MXFP4 | ~28 GB | ~58 |
| + 30 s audio | Nemotron-3 30B MXFP4 | ~25 GB | ~60 |
| All three modalities | Nemotron-3 30B MXFP4 | ~32 GB | ~55 |
| Text-only 16k prompt | Qwen 3.6 35B MXFP4 | ~30 GB | ~50 |
| + 8-frame video (256×256) | Qwen 3.5 VL 9B 8-bit | ~12 GB | ~80 |
| Text-only | DSV4-Flash JANGTQ_2L | ~52 GB | ~22 |
| Text-only | Mistral 3.5 128B MXFP4 | ~74 GB | ~18 (projected) |

---

## Host UI contract — what to enforce per family

### Universal

1. **Don't pre-resize images/videos.** Each family's vision tower has
   a config-driven resize step that's deterministic and respects
   aspect ratios. Host-side resizing loses information.
2. **Don't pre-decode audio.** `nemotronOmniLoadAudioFile` does
   sinc-quality resampling via AVAudioConverter; rolling your own
   loses fidelity.
3. **Don't pre-extract video frames.** vmlx's frame extractor uses
   correct AVAssetImageGenerator timestamping.

### Per-family rejection rules

When a user attaches a media file the host can't route, surface a
clear error rather than letting vmlx throw:

| Attachment | Allowed for | Reject for |
|---|---|---|
| Image (jpeg/png/heic/webp) | All VLM families above | LLM-only families |
| Video | Nemotron-3 omni, Qwen 2/2.5/3 VL, Qwen 3.5/3.6 VL bundles, SmolVLM 2 | Mistral 3.x, Gemma 3/4, Pixtral standalone, PaliGemma, Idefics3, GlmOcr, LFM2-VL, FastVLM, MiniMax, Laguna, all LLM-only |
| Audio | Nemotron-3 omni only | All others |

### Format extension hints (audio path canonicalization)

The audit-fix in `materializeMediaDataUrl` ensures audio mime
canonicalization fires only when the data URL header starts with
`audio/`. For audio attachments:

| Client format string | Temp file ext | Notes |
|---|---|---|
| wav / wave / x-wav | .wav | |
| mp3 / mpeg / x-mpeg | .mp3 | |
| m4a / x-m4a | .m4a | mp4 audio also canonicalizes here |
| flac | .flac | passthrough (default arm) |
| ogg / opus | .ogg | passthrough |

Video attachments keep their native extension — `data:video/mp4` →
`.mp4`, `data:video/quicktime` → `.quicktime` (Bug-fix locked in
`MaterializeMediaDataUrlMCDCTests.test_d6_videoMp4_keepsMp4Extension`).

---

## API surface — single entry point for all media

```swift
import MLXLMCommon

let userInput = MLXLMCommon.UserInput(
    chat: chatMessages,
    tools: tools,
    additionalContext: ["enable_thinking": true],
    audios: [.url(audioURL)],   // Nemotron-3 only — host pre-rejects elsewhere
    videos: [.url(videoURL)],   // Nemotron-3 + Qwen 2/2.5/3 + Qwen 3.5 + SmolVLM 2
    images: [.url(imageURL)]    // every VLM family
)

for await event in await batchEngine.generate(input: userInput, ...) { ... }
```

The `additionalContext` map carries Jinja kwargs (`enable_thinking`,
`reasoning_effort`, etc.) through to the model's chat_template render.
`disableThinking` from the host model-options layer flips
`additionalContext["enable_thinking"]` (verified at
`MLXBatchAdapter.swift:273-278`).

---

## Streaming + cancellation

- `BatchEngine.generate` returns `AsyncThrowingStream<GenerationEvent, Error>`
- Event types are uniform across modalities: `.chunk`, `.reasoning`,
  `.toolCall`, `.usage`
- Audio + video preprocessing is **pre-prefill**. Cancellation during
  the "transcribing audio…" / "extracting frames…" spinner is safe;
  decode work is in `Task.detached` and respects cooperative
  cancellation
- TTFT (time-to-first-token) measured via `TTFTTrace` includes all
  preprocessing — host can show "loading…" until first chunk arrives

---

## Coverage checklist for the host team

Before exposing a new media family in the UI, verify:

- [ ] Family is in this matrix's "modality" column with ✓ for the modality
- [ ] Cache topology row identifies whether `setHybrid(true)` is needed
- [ ] JANGTQ tier is supported (or the bundle ships only as MXFP4 / full-precision)
- [ ] Host pre-rejection logic handles unsupported attachment types
- [ ] Format extension canonicalization handles the audio mime types the model expects
- [ ] Streaming events render cleanly (no marker leaks into `.chunk`)
- [ ] `enable_thinking` toggle plumbs through to the model's chat template
- [ ] Memory budget tested at the model's expected concurrent media load
- [ ] OmniBench / per-model bench scenario added to the regression matrix
