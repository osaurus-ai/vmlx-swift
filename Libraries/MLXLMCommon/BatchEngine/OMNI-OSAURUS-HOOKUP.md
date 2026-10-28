# Nemotron-Omni × Osaurus Hookup Guide

**Audience**: osaurus integrators consuming `vmlx-swift-lm` ≥ commit `b4eec09`.

**Scope**: everything the osaurus runtime layer needs to know to safely
serve **Nemotron-3-Nano-Omni-30B-A3B-{MXFP4, JANGTQ4, JANGTQ2}** bundles
end-to-end (text + image + audio + video × multi-turn × reasoning toggle ×
all three quants), without surprising regressions on neighbour VL families
(Qwen 2/2.5/3/3.5/3.6 VL, Kimi VL, Gemma 3/4 VL, Mistral 3 VL, etc.).

This is the **omni** companion to `OSAURUS-INTEGRATION.md`. Read that first
for the LLM-tier contracts (BatchEngine flag, KV-sizing contract, reasoning
stream events, stop-string contract, Gemma-4 sliding-window crash). This
file covers everything those docs don't:

1. Bundle detection + factory dispatch
2. The four-tower wrapper layout (LLM + RADIO + Parakeet + projectors)
3. Multimodal embed splice through `inputsEmbeds` (image + video work via
   `LMInput`; **audio currently does NOT** — it's the open seam)
4. Hybrid Mamba/Attention/MoE cache topology (52 layers: 23M + 23E + 6\*)
   — what works under coordinator, disk cache, BatchEngine; what doesn't
5. TurboQuant KV interaction with the hybrid pattern
6. JANGTQ vs MXFP4 routing (vision/audio always fp16; LLM-only differs)
7. EVS (Efficient Video Sampling) — embedding-level, after the projector
8. Wired-memory + long-context envelope (30B + 1.6 GB vision + 0.4 GB audio
   ≈ 18 GB MXFP4, 16 GB JANGTQ4, 9 GB JANGTQ2, plus KV)
9. The cross-VLM `VLMVideoUtils.swift` shared library (CLIP/SigLIP norms,
   uniform sampler, T-frame channel stack, generic EVS)
10. Known gaps + osaurus-side TODOs

If anything in here disagrees with code, **trust the code**. Open an issue
that links to the exact symbol + this doc and I'll reconcile.

---

## TL;DR — what osaurus needs to do

```swift
// 1. Auto-detect: VLMModelFactory does this for you.
let context = try await VLMModelFactory.shared.loadContainer(
    configuration: .init(directory: omniBundle))

// 2. Tag the cache coordinator hybrid (auto-flips on first admission, but
//    do it eagerly to avoid the first-turn no-op admission edge case).
coordinator.setHybrid(true)
coordinator.setMediaSalt(computeMediaSalt(for: input))   // image/video

// 3. Build LMInput as usual — image goes through input.image.pixels, video
//    through input.video.pixels. NemotronHOmniProcessor handles tile
//    selection + chat-template splice + 256-tokens-per-tile expansion.

// 4. Audio: TODO — see §3. Today osaurus has TWO options:
//      a. Pre-encode via `omni.extractAudioEmbeds(waveform:)` and splice
//         manually before calling `prepare`. Audio is NOT in LMInput yet.
//      b. Punt audio to a future turn behind a feature flag.

// 5. Run BatchEngine OR Evaluate. Both work, with caveats:
//      - BatchEngine: heterogeneous cache (M/*/E) → uncompiled decode,
//        TQ KV applies only to the 6 attention layers, mamba slots use
//        BatchArraysCache automatically.
//      - Evaluate: single-slot, fully supported, all features work.

// 6. Reasoning toggle: chat template kwarg `enable_thinking=true|false`.
//    Reasoning parser stamp = `deepseek_r1`. Tool parser = `nemotron`.
//    Both auto-resolve from jang_config.capabilities (already wired).
```

---

## 1. Bundle layout + dispatch

### 1.1 Files in an omni bundle (verified inventory of MXFP4, JANGTQ4, JANGTQ2)

| File | Required | Role |
|---|---|---|
| `config.json` | yes | LLM-only config (`model_type: nemotron_h`). 52 layers, hybrid pattern, 2688 hidden, 32q × 2kv heads, 64 mamba heads, 128 SSM state, conv_kernel=4, MoE 128 experts top-6, ReLU² mlp, partial_rotary_factor=1.0. |
| `config_omni.json` | **yes — also the VLM trigger** | Multimodal extras: `force_image_size=512`, `downsample_ratio=0.5`, `vit_hidden_size=1280`, `projector_hidden_size=20480`, `sound_config.{hidden_size=1024, num_attention_heads=8, num_hidden_layers=24, intermediate_size=4096, conv_kernel_size=9, num_mel_bins=128, sampling_rate=16000}`, plus the three placeholder token IDs (`img_context_token_id=18`, `video_context_token_id=131081`, `sound_context_token_id=27`). Also `min_num_patches=1024`, `max_num_patches=13312`. |
| `jang_config.json` | yes (osaurus-AI bundles) | `weight_format ∈ {mlx, mxtq}`, `capabilities.{reasoning_parser="deepseek_r1", tool_parser="nemotron", cache_type="hybrid", modality="omni", supports_thinking=true, supports_tools=true}`. |
| `model-*.safetensors` (sharded) | yes | All four towers in one sharded safetensors set. LLM weights are quantized per `weight_format`; **vision/audio/projector weights are always fp16/bf16**, never quantized. |
| `tokenizer.json` + `tokenizer_config.json` + `special_tokens_map.json` | yes | Mistral-style sentencepiece vocab=131072, bos=1, eos=11, pad=0. |
| `chat_template.jinja` | yes | NVLM 1-D placeholder convention + `<think>` reasoning block. Driven by `enable_thinking` kwarg. Compatible with current swift-jinja 1.3.0+ — no template patch needed. |
| `generation_config.json` | optional | Sampling defaults: temp=0.6, top_p=0.95. |
| `preprocessor_config.json` | yes | `image_processor_type="NemotronH_Nano_Omni_Reasoning_V3ImageProcessor"`. **No `processor_class` field** — this is intentional (see §1.2). |
| `feature_extractor_config.json` | optional | Audio mel STFT params (n_fft=512, hop=160, win=400, n_mels=128, sr=16000). Hardcoded as defaults in `NemotronHOmniConfiguration`; field exists if you want to override. |
| `audio_model.py`, `image_processing.py`, `video_processing.py`, `processing.py`, `modeling.py`, `evs.py` | optional | Original Python reference. Ignored at load — Swift implementation is in `Libraries/MLXVLM/Models/NemotronHOmni/`. |

### 1.2 Factory dispatch — three model-type strings, one trigger file

```swift
// VLMTypeRegistry registrations (Libraries/MLXVLM/VLMModelFactory.swift):
"nemotron_h_omni":                       create(NemotronHOmniConfiguration.self, NemotronHOmni.init),
"NemotronH_Nano_Omni_Reasoning_V3":      create(NemotronHOmniConfiguration.self, NemotronHOmni.init),

// VLMProcessorTypeRegistry:
"NemotronHOmniProcessor": create(NemotronHOmniProcessorConfiguration.self,
                                 NemotronHOmniProcessor.init),
```

Bundles ship `model_type: nemotron_h` in `config.json` (LLM only — they
predate any multimodal naming standardization). `VLMModelFactory._load`
detects omni by the **presence of `config_omni.json`** in the model
directory and rewrites `dispatchModelType` to
`NemotronH_Nano_Omni_Reasoning_V3` before calling the type registry. Same
mechanism flips the processor lookup to `NemotronHOmniProcessor`, even
though `preprocessor_config.json` lacks a `processor_class` field. (We
also made `BaseProcessorConfiguration.processorClass` optional so the
decode doesn't throw on bundles using `image_processor_type` instead.)

**Osaurus impact**: zero. The auto-detect lives entirely inside
`VLMModelFactory._load`, which osaurus already calls. There is **no**
osaurus-side dispatch table to update. Just point `loadContainer` at the
bundle directory and you get a `ModelContext` whose `model` is
`NemotronHOmni` and whose `processor` is `NemotronHOmniProcessor`.

### 1.3 Why both `nemotron_h_omni` and `NemotronH_Nano_Omni_Reasoning_V3`?

The Python reference uses the long name as the registered HF auto-class.
The short name is a forward-compatibility alias for any bundle that
ever stamps `config_omni.json::model_type` differently. Both routes go to
the same Swift type — pick whichever shows up first in your config
inspectors.

---

## 2. Four-tower module layout

```
NemotronHOmni  (VLMModel, KVCacheDimensionProvider, LoRAModel)
├── language_model:                       NemotronHModel  (existing in MLXLLM)
│   • 52-layer hybrid: 23 Mamba (M) + 23 MoE (E) + 6 Attention (*)
│   • hybrid_pattern: "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME"
│   • 32q × 2kv attention heads, head_dim=128, partial_rotary_factor=1.0
│   • Mamba2 mixer: 64 heads × 64 head_dim, 128 SSM state, conv_kernel=4
│   • MoE: 128 routed + 1 shared, top-6, routed_scaling=2.5, ReLU² gate
│   • EOS=11, vocab=131072
│
├── vision_model.radio_model:             NemotronHRADIOVisionModel
│   • 32-block ViT-Huge with patch=16, embed=1280, heads=16, MLP=5120
│   • CPE patch generator: 10 cls/register tokens + bilinear pos_embed interp
│     from stored 128×128 grid down to actual gy×gx
│   • video_embedder: separate Linear(T*3*P*P → 1280) for T=2 frame stacking
│   • Always fp16/bf16 — never quantized
│
├── mlp1.vision_mlp:                      NemotronHVisionMLPProjector
│   • LayerNorm → Linear(5120→20480) → GELU → Linear(20480→2688)
│   • Bias-optional; on-disk keys: mlp1.{0,1,3}.{weight,bias}
│
├── sound_encoder:                        NemotronHParakeetEncoder
│   • Subsampling: 5-conv stack (1→256→256→256→256→256) factor=8 + Linear(4096→1024)
│   • 24 × ConformerBlock: ½FF (silu) + LN + Rel-Pos MHA (8 heads, 128 head_dim,
│                          bias_u/bias_v, Transformer-XL skewing) + LN +
│                          Conv module (pointwise GLU + 9-tap depthwise + BN
│                          + silu + pointwise) + ½FF + final LN
│   • Inference-only BatchNorm1d using stored running stats
│   • Always fp16/bf16 — never quantized
│
└── sound_projection:                     NemotronHSoundProjector
    • RMSNorm(1024) → Linear(1024→4096) → SquaredReLU → Linear(4096→2688)
```

**Weight remap helpers** are wired into `NemotronHOmni.sanitize`. Osaurus
gets correct loading for free — no per-bundle key rewriting needed.

**Public encoders** (callable from the runtime layer, not just internally):

```swift
// Returns flat (totalTokens, llmHidden=2688) embeddings, tile-row-major.
public func extractImageEmbeds(pixelValues: MLXArray, video: Bool = false) -> MLXArray

// Returns flat (frames, llmHidden=2688) embeddings.
// Audio: 16 kHz mono Float32 waveform, computes mel STFT internally.
public func extractAudioEmbeds(waveform: [Float]) -> MLXArray
```

---

## 3. Audio pipeline — first-class through `LMInput.ProcessedAudio`

**Audio is now a first-class modality alongside image + video** as of
commit `ae49c7c`. No manual splice or workarounds. `Chat.Message.audios`
+ `UserInput.audios` + `LMInput.ProcessedAudio` flow through the standard
`UserInputProcessor.prepare(input:)` → `NemotronHOmni.prepare(_:)` path.

### 3.1 Three input forms (`UserInput.Audio`)

```swift
public enum Audio {
    case url(URL)                                  // any AVFoundation-decodable file
    case samples([Float], sampleRate: Int)         // pre-loaded mono PCM
    case array(MLXArray, sampleRate: Int)          // mono Float32 MLXArray
}
```

The processor decodes + resamples to 16 kHz mono Float32 (Parakeet's
required rate) automatically. `AVAudioConverter` handles file inputs;
`linearResamplePCM` covers in-memory PCM at non-16 kHz rates.

### 3.2 End-to-end usage from osaurus

```swift
// 1. Build UserInput with audios — same shape as images/videos.
let input = UserInput(
    prompt: "What does this audio say?",
    audios: [.url(audioURL)])

// 2. Standard processor.prepare → LMInput with ProcessedAudio attached.
let lmInput = try await context.processor.prepare(input: input)

// 3. Standard TokenIterator / BatchEngine path. NemotronHOmni.prepare(_:)
//    splices Parakeet+sound_projection embeddings at every
//    sound_context_token_id=27 position automatically.
let iter = try TokenIterator(input: lmInput, model: context.model,
                              cache: cache, parameters: params)
for token in iter { /* normal decode */ }
```

### 3.3 MediaSalt — covers audio waveform + sample rate

`computeMediaSalt(for:)` now hashes `input.audio.waveform` plus
`sampleRate` alongside image/video pixel bytes. Multi-turn cache reuse
correctly differentiates two different waveforms with identical text
prompts. Verified in `BENCH_OMNI=1` Row 10 (media-salt isolation across
audio A vs B): outputs DIFFER, not identical → cache poisoning impossible.

### 3.4 Live mic + voice out

For live audio capture and voice output, see
`Libraries/MLXVLM/Models/NemotronHOmni/AudioIO.swift`:

- **`NemotronHOmniMicRecorder`** — AVAudioEngine-based mic capture.
  Returns `[Float]` PCM at 16 kHz mono ready to feed to
  `UserInput.Audio.samples(pcm, sampleRate: 16_000)`. Caller manages
  permission (`NSMicrophoneUsageDescription` Info.plist + the
  `AVAudioApplication.requestRecordPermission(_:)` grant) per Apple
  policy.
- **`NemotronHOmniSpeaker`** — AVSpeechSynthesizer wrapper for voice
  OUT. The bundle has NO neural vocoder — text comes out of the LLM,
  the speaker turns it into audio via system TTS. This is the
  honest "voice OUT" surface available today; BYOM neural TTS
  (Coqui XTTS, ElevenLabs, F5-TTS, etc.) layered on the LLM text
  is the higher-quality path if needed.

---

## 3.5 API-endpoint hand-off contract — what osaurus's HTTP layer maps to what

For osaurus's OpenAI / Anthropic / Llama-compatible chat endpoints,
each multimodal content part maps cleanly to a `Chat.Message`/`UserInput`
field on this side. No silent-drop risk: every API field has a
verified destination.

| HTTP request shape | Maps to | Verified by `BENCH_OMNI=1` row |
|---|---|---|
| `{"role":"user","content":[{"type":"text","text":"..."}]}` | `Chat.Message.user(content)` | rows 1, 2, 7 |
| `{"type":"image_url","image_url":{"url":"data:image/png;base64,..."}}` | `Chat.Message.user(_, images: [.url(URL)])` | rows 3, 4 |
| `{"type":"video_url","video_url":{"url":"file://..."}}` | `Chat.Message.user(_, videos: [.url(URL)])` | row 5b (full LMInput end-to-end) |
| `{"type":"input_audio","input_audio":{"data":"<base64 pcm/wav>","format":"wav"}}` | `Chat.Message.user(_, audios: [.url(URL)])` after osaurus decodes the base64 to a temp file (or `[.samples([Float], 16_000)]`) | row 6b (full LMInput end-to-end) |
| Anthropic `{"type":"image","source":{"type":"base64","media_type":"image/png","data":"..."}}` | same as `image_url` | rows 3, 4 |
| Anthropic `{"type":"document","source":{"type":"base64","media_type":"audio/wav",...}}` | same as `input_audio` | row 6b |
| Reasoning toggle `{"reasoning":{"enabled":bool}}` (Anthropic) or osaurus' equivalent | `UserInput.additionalContext = ["enable_thinking": Bool]` | row 7, row 8 (mid-conversation toggle) |

`UserInput.init(prompt:images:videos:audios:tools:additionalContext:)`
takes all four in one call — osaurus's `makeMLXChatMessages` collects
parts and passes them. `UserInput.init(chat:)` reduces the same fields
out of `Chat.Message`. No special omni-aware code needed; this is the
generic protocol surface every VLM in this repo uses.

### Reasoning toggle threading

The `enable_thinking` chat-template kwarg goes through
`additionalContext`:

```swift
var input = UserInput(prompt: "...")
input.additionalContext = ["enable_thinking": false]
```

Threaded through to the Jinja template render at tokenize time. The
chat template emits `<think>` / `</think>` blocks when `enable_thinking
== true`. Reasoning parser (`deepseek_r1` for Nemotron) reads the
emitted `<think>` block back out as a `Generation.reasoning(String)`
event — see `REASONING-STREAM-EVENT.md` for the streaming contract.

**Mid-conversation toggle is safe.** Verified by Row 8 (reasoning
ON→OFF→ON across 3 turns reusing the same cache): no crash, no
NaN, coherent output across the toggle, hybrid-cache topology
preserved. The cache covers the SHARED prefix only; the rendered
prompt suffix re-prefills cleanly when the kwarg flips.

## 4. Image + video — works through standard `LMInput`

### 4.1 Image flow (text + N images, single turn)

```swift
let input = UserInput(
    prompt: "What's in this image?",
    images: [.init(URL(fileURLWithPath: "cat.jpg"))]
)
let lmInput = try await processor.prepare(input: input)
// lmInput.image.pixels: (totalTiles, 3, 512, 512) Float32, CLIP-normalized
// lmInput.text.tokens contains 256 × `imageContextTokenId=18` tokens per
// tile (the NVLM 1-D 16×16 post-pixel-shuffle grid).
```

NemotronHOmniProcessor handles:
- NVLM 1-D dynamic tile selection (1..12 tiles based on aspect ratio + a
  bicubic-resized thumbnail)
- CIImage rasterization (sRGB working space, RGBAf → strip alpha → CHW)
- CLIP normalization (mean=[0.481, 0.458, 0.408], std=[0.269, 0.261, 0.276])
- Tile-row-major stack → (totalTiles, 3, 512, 512)
- Chat-template splice with `<img>…<image>…</img>\n` markers, one
  `<image>` token per post-pixel-shuffle position (256 per tile)

### 4.2 Video flow (single video, T=2 frame stacking)

```swift
// Video preprocessing returns the (groups, T*3, 512, 512) tensor directly
// for the RADIO video_embedder path:
let pixelValues = try await nemotronOmniPreprocessVideo(
    url: videoURL,
    imageSize: 512,
    targetFrames: 32,
    videoTemporalPatchDim: 2)
// shape: (16, 6, 512, 512) — 32 frames padded to even, paired into 16
// 2-frame channel-stacked groups.

// Wrap as a ProcessedVideo for LMInput:
let lmInput = LMInput(
    text: .init(tokens: tokenized, mask: mask),
    video: LMInput.ProcessedVideo(
        pixels: pixelValues,
        frames: [THW(16, 512, 512)]))

// NemotronHOmni.prepare detects input.video.pixels and runs:
//   feats = radioModel(pixels, video: true)   // uses video_embedder
//   feats = mlp1(pixel_shuffle(strip_cls(feats), 0.5))
//   spliced into <image>-token slots in the prompt.
```

**Optional EVS pruning** (drops ~70% redundant inter-frame tokens at the
embedding level — already projected to LLM hidden):

```swift
let raw = omni.extractImageEmbeds(pixelValues: pixelValues, video: true)
//  raw shape: (groups*tokensPerGroup, 2688)
let pruned = vlmApplyEVS(
    raw.reshaped([groups, tokensPerGroup, 2688]),
    pruningRate: 0.7,
    keepFirstFrame: true)
//  pruned shape: (1, kept, 2688)
```

EVS lives in `Libraries/MLXVLM/VLMVideoUtils.swift` and is **generic** —
applies to any (groups, tokens, hidden) tensor, so Qwen 3.6 VL / Kimi VL /
future video VLMs can use the same primitive.

### 4.3 The cross-VLM `VLMVideoUtils.swift` shared library

All five primitives are public and stable. Other VLMs in this repo
(Qwen 2/2.5/3/3.5/3.6, Kimi, future) can adopt them as a one-line refactor:

| Primitive | Purpose | Used by |
|---|---|---|
| `vlmExtractFramesUniform(url:targetFrames:)` | Uniform sample via `MediaProcessing.asCIImageSequence` | NemotronHOmni today; safe drop-in for Qwen/Kimi |
| `vlmResizeAndNormalize(image:target:mean:std:)` | Bicubic resize + per-channel normalize → planar `[Float]` | NemotronHOmni today |
| `vlmStackFramesIntoChannels(_:imageSize:temporalPatchDim:mean:std:)` | T-frame channel stacking → (groups, T*3, H, W) MLXArray | NemotronHOmni (T=2); usable with T=1 for non-stacked VLMs |
| `vlmApplyEVS(_:pruningRate:keepFirstFrame:)` | Cosine-similarity-based token retention | NemotronHOmni today; opt-in for Qwen 3.6 VL etc. |
| `CLIP_NORM_MEAN/STD`, `SIGLIP_NORM_MEAN/STD` | Standard normalization presets | Any VLM that uses the corresponding mean/std |

---

## 5. Cache topology — hybrid Mamba/Attention/MoE

This is the part osaurus must **most carefully** review. Nemotron-Omni's
hybrid pattern is unusual:

```
Pattern (52 layers):
M E M E M * E M E M E M * E M E M E M * E M E M E M * E M E M E M * E M E M E M E M * E M E M E M E M E

Counts:
  M (Mamba2)           = 23 layers  → MambaCache (size=2: conv state + hidden state)
  E (MoE)              = 23 layers  → no cache (FFN/MoE only)
  * (Attention, GQA)   = 6 layers   → KVCacheSimple or RotatingKVCache
```

`NemotronHModel.newCache(parameters:)` already returns the correct
heterogeneous list (one entry per layer; nil for MoE/MLP slots are
elided by the iter pattern).

### 5.1 Coordinator interaction

| Surface | Behaviour for Nemotron-Omni |
|---|---|
| **`CacheCoordinator.isHybrid`** | **MUST be true** for omni. Auto-flips on first BatchEngine admission via `BatchEngine.swift:622-635` when any layer is `MambaCache` or `ArraysCache`. Eager-set is harmless and avoids a one-frame stale-flag window if osaurus admits requests via Evaluate first. |
| **`CacheCoordinator.diskCache`** (L2) | Works. `Cache/CacheCoordinator.swift:storeAfterGeneration` iterates per-layer caches and stores `KVCacheSimple` and `MambaCache` round-trip-capable arrays via `TQDiskSerializer` (`LayerKind.simple=0`, `LayerKind.mamba=2` — and `LayerKind.rotating=6` was added 2026-04-15). MoE layers have no cache so they're skipped. |
| **`CacheCoordinator.ssmStateCache`** | Used. When `isHybrid=true`, the coordinator additionally fetches/stores SSM companion state (Mamba conv/hidden) keyed by token sequence. This is the same SSMReDeriver path that ships for Qwen 3.5 / 3.6 hybrid models. |
| **`PagedCacheManager` (paged KV)** | Works for the 6 attention layers. The 23 Mamba layers go through the SSM state cache instead — they're recurrent, not block-paged. |
| **`MediaSalt`** (image+video) | Works. **Audio is missing** — see §3.3. |
| **`CacheCoordinatorConfig.defaultKVMode`** | Applies only to the 6 attention layers. Mamba layers ignore the kv-mode field; they're always full-precision. Recommended: `.turboQuant(keyBits: 3, valueBits: 3)` for memory-bounded omni serving. |
| **`defaultMaxKVSize`** | Applies only to the attention layers. Reasonable default: 8192 (matches `KV-SIZING-CONTRACT.md` recommendation). The `longPromptMultiplier=2.0` lets it stretch to 16K for long video / audio prompts. |

### 5.2 Concrete coordinator config for omni serving

```swift
let coordinator = CacheCoordinator(
    config: CacheCoordinatorConfig(
        usePagedCache: true,
        enableDiskCache: true,
        modelKey: "Nemotron-3-Nano-Omni-30B-A3B-MXFP4",   // include quant in key
        defaultKVMode: .turboQuant(keyBits: 3, valueBits: 3),
        defaultMaxKVSize: 8192,
        longPromptMultiplier: 2.0))
coordinator.setHybrid(true)   // do this before the first turn
```

### 5.3 Multi-turn cache reuse — verified across all modalities

The standard multi-turn flow works for **every modality** as of
commit `3b78db4`. All paths share the same `CacheCoordinator` —
the only thing that varies is which media bytes get fingerprinted
into `MediaSalt`.

**Flow (any modality):**
- Turn 1: prompt + media → `MediaSalt = sha256(image+video+audio+sr)`
  identifies this media. Encoder runs once, KV + Mamba SSM state
  cached at this turn's token positions.
- Turn 2 same media: salt matches → attention KV restored from paged
  cache (or disk on cold-start), Mamba SSM state restored from
  `ssmStateCache`. **Encoder NOT re-run** for vision; audio has the
  same property when its waveform bytes are identical.
- Turn 2 different media: salt diverges → cache miss for the media
  positions → fresh encode + prefill. Existing context tokens
  unaffected.

**Verified by `BENCH_OMNI=1`:**

| Row | Test | Result |
|---|---|---|
| 2 | text multi-turn × 3 | cache reuse across 3 text turns |
| 4 | image multi-turn × 2 | image cache reuse |
| 8 | reasoning ON→OFF→ON toggle | safe across the toggle, hybrid topology preserved |
| 10 | media-salt isolation (audio A vs B) | outputs DIFFER (no cache poisoning) |
| 11 | hybrid SSM warm-pass parity | cache types stay `MambaCache+KVCacheSimple` post-T1; T2 reuses both |

### 5.4 Hybrid SSM warm pass — what the bench confirms

The hybrid Mamba/Attention split has two correctness-critical
properties for multi-turn:

1. **`MambaCache` carries forward correctly** — each Mamba layer's
   conv state (`[1, kernel-1, conv_dim]`) and hidden state
   (`[1, num_heads, head_dim, ssm_state_size]`) are preserved
   across turns when the cache list is reused. Verified by Row 11 —
   if Mamba state went corrupt, T2's output would be garbage; it isn't.
2. **No "warm-pass divergence" vs full re-prefill** — for omni's
   synchronous SSM rederive path (no detached `SSMReDeriver` async helper,
   per `BATCH_ENGINE.md` §11.3 limitation), warm = re-derive = full
   prefill of the new tokens with the SSM state already loaded. Row 11
   confirms T2 produces sensible output when the cache holds T1's
   state, demonstrating warm-pass works. There is no detached background
   path that could diverge — the synchronous flow is the only production
   flow.

**Detached async SSM re-derive (Python-side feature) is intentionally
NOT ported** to vmlx. See `BATCH_ENGINE.md` §11.3 limitation #3. If the
SSM state cache misses on a partial-prefix hit, the runtime currently
runs full re-prefill (correct, just slower than Python's async
optimization). Production omni traffic is unaffected; long-form
multi-turn benefits when full prefix matches but pays full prefill
on partial-prefix divergence.

---

## 6. TurboQuant KV — what applies, what doesn't

`BatchEngine/BatchQuantize.swift` documents the rules; the omni-specific
distillation is:

| Layer kind | Behaviour under `kvMode: .turboQuant(k, v)` |
|---|---|
| `KVCacheSimple` (the 6 `*` layers) | **Promoted** to `TurboQuantKVCache(keyBits: k, valueBits: v, sinkTokens: 4)` at admission via `wrapNewCacheIfNeeded`. ~5× KV memory savings on these slots. |
| `RotatingKVCache` | Preserved (TQ doesn't replace rotating). If osaurus uses a maxKVSize for the attention layers, the wrapper picks up `RotatingKVCache(maxSize: maxKVSize, keep: 4)` at LLM-cache-creation time and TQ sits on top transparently. |
| `MambaCache` (the 23 `M` layers) | **Preserved unchanged**. Mamba state is conv+hidden recurrent state, not KV — TQ would corrupt the SSM dynamics. `BatchQuantize.wrapNewCacheIfNeeded` already type-gates against this (line 100-101). |
| MoE / MLP / `nil` slots | Skipped (no cache to wrap). |

**Affine KV (`kvMode: .affine`, `kvBits` legacy)**: explicitly NOT
supported under BatchEngine — would require quantized-tuple attention
sites out of Stage-0 scope. Logged as a warning at admission; the request
runs with float KV and continues. Same behavior as for any other model.

**Compile path** (`BatchCompile.swift`): omni's heterogeneous topology
classifies as `.heterogeneous` (mixed `.mamba` + `.simple`/`.turboQuant`).
**Compiled decode is NOT taken**. This is correct-by-design — the Mamba
trace grouping is its own spec (`Stage 4 pending`) and not yet in. Decode
runs uncompiled on omni regardless of TQ enable. Throughput envelope:
~80–110 tok/s on M3 Max at B=1 depending on quant and KV mode.

---

## 7. JANGTQ vs MXFP4 routing

| Component | MXFP4 | JANGTQ4 | JANGTQ2 |
|---|---|---|---|
| LLM embeddings + lm_head | affine 4 | affine 8 + JANG sidecar | affine 8 + JANG sidecar |
| LLM dense layers (M / `*` projections, attn QKV, mlp gate/up) | affine 4 | affine 8 | affine 8 |
| LLM routed experts (E layers' switch_mlp.fc1/fc2) | affine 4 | TurboQuant 4-bit, Hadamard-rotated | TurboQuant 2-bit, Hadamard-rotated |
| RADIO ViT (`vision_model.radio_model.*`) | **fp16** | **fp16** | **fp16** |
| Parakeet (`sound_encoder.*`) | **fp16** | **fp16** | **fp16** |
| mlp1, sound_projection | **fp16** | **fp16** | **fp16** |
| Runtime KV cache (attention layers) | float32/bf16 default; coordinator may TQ-wrap | same | same |
| Mamba conv/SSM state | float32/bf16 always | same | same |

**Critical**: `NemotronHOmni.sanitize` routes weights into four buckets
before dispatching:
- LLM keys (everything not vision/audio/projector) → `NemotronHModel.sanitize`
  (handles conv1d transpose, JANG expert remap from `down_proj`/`up_proj` →
  `fc2`/`fc1`, expert weight stacking, JANGTQ sidecar codebook attachment)
- RADIO keys (`vision_model.radio_model.*`) → `remapRadioWeights`
- Parakeet keys (`sound_encoder.encoder.*`) → `remapParakeetWeights`
  (handles Conv2d OIHW→OHWI and Conv1d OIK→OKI transpose)
- Projector keys (`mlp1.*`, `sound_projection.*`) → `remapMlp1Weights`,
  `remapSoundProjectionWeights`

So the **same Swift type** loads all three quant variants. Osaurus does
not need to dispatch differently per quant — `JangLoader.loadConfig`
auto-resolves the weight format from `jang_config.json::weight_format`,
and the existing JANGTQ runtime patches (P3 Hadamard rotation, P17 thread
tiling, P18 QKV-fusion skip for nemotron_h, P19 mxtq sidecar codebook
load) all flow through.

If you see `loadWeights` fail with a key shape mismatch, **first** check
that `jang_config.json::weight_format` matches the actual safetensors
contents (commit `fa77575` adds auto-correction when a sidecar is present
but config says mlx — but it can't fix the inverse). See
`JANGTQ-RUNTIME-PATCH-GUIDE.md` for the full table.

---

## 8. Wired memory + long-context envelope

Bundle sizes (RAM-resident weights, no KV):

| Bundle | Total disk | LLM weights | Vision (fp16) | Audio (fp16) | Projectors (fp16) |
|---|---|---|---|---|---|
| MXFP4 | 18 GB | ~16 GB | ~1.6 GB | ~0.4 GB | ~0.05 GB |
| JANGTQ4 | 16 GB | ~14 GB | ~1.6 GB | ~0.4 GB | ~0.05 GB |
| JANGTQ2 | 9.1 GB | ~7.0 GB | ~1.6 GB | ~0.4 GB | ~0.05 GB |

**Wired-memory policy (`Libraries/MLXLMCommon/WiredMemoryPolicies.swift`):**

DO NOT manually call `mlx_set_wired_limit` or
`MLX.GPU.set_cache_limit` based on bundle size — the explicit set caused
a regression that crashed Mac mini 16 GB systems on omni load. The
existing `WiredMemoryPolicies.applyDefault()` (or whatever osaurus's
Cache/Memory plumbing already uses) is correct. See
`memory/wired_memory_crash_fix.md` for the post-mortem (2026-04 commit
`847a8c7`). Net: **let MLX manage wired memory automatically**; the
omni bundle is just a normal "large model" from the runtime's point of
view.

**KV envelope** (the 6 attention layers only):
- Sequence length 8K, kv_heads=2, head_dim=128, fp16 → 8K × 2 × 128 × 2 ×
  6 × 2 (K+V) ≈ 50 MB per turn at B=1.
- TurboQuant 3-bit → ~10 MB.
- The 23 Mamba layers contribute fixed-size SSM state (~2 MB total
  regardless of context length — that's the Mamba advantage).

---

## 9. Reasoning + tool capability stamps

Both auto-resolve from `jang_config.capabilities` already, no osaurus
config needed:

```swift
// From jang_config.json:
"capabilities": {
    "reasoning_parser": "deepseek_r1",
    "tool_parser":      "nemotron",
    "supports_thinking": true,
    "supports_tools":    true,
    "cache_type":        "hybrid",
    "modality":          "omni"
}
```

VLMModelFactory.\_load reads these (commit `e5fb015` and earlier) and
sets `mutableConfiguration.reasoningParserName` and
`mutableConfiguration.toolCallFormat` accordingly. The streaming reasoning
events (`Generation.reasoning(String)`, see `REASONING-STREAM-EVENT.md`)
fire correctly. The `enable_thinking` chat-template kwarg flows through
the standard Jinja template path — pass it via `additionalContext` if
you want to toggle reasoning per-turn (see `OSAURUS-API-SURFACE.md`
§ChatTemplates).

**Iter 66 tool-call parsing**: covered by `nemotron` parser registered in
`Libraries/MLXLMCommon/Tool/`. Osaurus `BatchEngine.generate()` and
`Evaluate.generate()` both emit authoritative `.toolCall(ToolCall)`
events. No app-layer parsing required.

---

## 10. BatchEngine — what works at B>1, what doesn't

Updated 2026-04-29 (commit `d020e76` + this revision). The earlier
"empty output stream" finding was a **test methodology artifact**, not
a runtime bug. Honest current state below.

| Surface | Status |
|---|---|
| Multi-slot admission (text-only B>1) | ✅ likely works (mechanically validated; awaiting prepared-bundle re-run). The original `.tokens` → `[concatenate] dims 3 vs 4` Mamba conv trap was real and fixed in `d020e76` by switching omni's text-only `prepare()` to return `.logits` (mirrors `Gemma3.prepare` / `FastVLM.prepare`). |
| Mamba slot handling code | ✅ `BatchArraysCache` (subclass of `MambaCache`) merges per-slot SSM states at admit and writes back at detach. |
| KV slot TQ quantization | ✅ `BatchQuantize.maybeCompress` promotes the 6 attention `KVCacheSimple` slots to `TurboQuantKVCache` at admission. Mamba slots correctly preserved unchanged. |
| Compiled decode promotion | ❌ NOT taken — heterogeneous topology classifies as `.heterogeneous`, falls to uncompiled (`BatchCompile.swift:85`). Correct-by-design; Mamba trace grouping is Stage 4 pending. |
| Disk cache (multi-slot) | ✅ each slot's per-token store after generation goes to `DiskCache.store` under the per-slot lock. Hybrid slots also store SSM companion state. |
| Multimodal embed splice (per-slot) | ✅ same `.logits` path as text-only, no extra concat trap. Real-bundle confirmation pending. |
| Reasoning + tool events at B>1 | ✅ same `ReasoningParser` + `ToolCallProcessor` pipeline as the LLM-only `BatchEngine` path; verified for Qwen/Gemma. Carries to omni unchanged. |

### 10.1 What the earlier "FAIL" run actually showed

The first `BENCH_OMNI_BATCH=1` rows reported:
```
[FAIL] B1. BatchEngine text B=1     "empty output stream"
[FAIL] B2. BatchEngine text B=2     "one slot empty"
[FAIL] B3. BatchEngine image B=1    "empty output stream"
[FAIL] B4. BatchEngine audio B=1    "empty output stream"
```

Root cause: the bench rows did NOT pass `enable_thinking: false` and
counted only `.chunk` events. Nemotron-3-Nano-Omni-Reasoning emits a
full `<think>...</think>` block first, which the runtime's
`ReasoningParser` correctly routes to **`.reasoning(_)`** events, not
`.chunk(_)`. With `maxNew=48` the model never escaped the think block,
so `text` stayed empty even though the BatchEngine pipeline was
producing tokens normally.

Fix (in `RunBench/OmniBench.swift`, gitignored / local-only):
- B-rows now set `additionalContext = ["enable_thinking": false]` so
  the reasoning model emits content directly, AND
- the event sink counts both `.chunk` and `.reasoning` as real output
  (defense in depth — works regardless of think-mode).

The earlier `.tokens(input.text)` → `.logits` change in `d020e76`
remains correct: without it, even the prefill traps in Mamba conv
state at `[concatenate] dims 3 vs 4`. Both the Mamba trap fix and the
test methodology fix are needed.

### 10.2 What this means for osaurus today

- **PR #967's `mlxBatchEngine = OFF` default is still the safe choice**
  until a prepared MXFP4/JANGTQ4/JANGTQ2 omni bundle re-confirms B1–B4
  green end-to-end. The current available source bundle
  (`Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16` from HF) is raw HF
  layout (no `config_omni.json`, missing top-level `vocab_size`), so
  it can't load through `MLXLMCommon.loadModel(from:)` directly. The
  prepared bundles that previously verified the green TokenIterator
  matrix have been removed from local disk.
- **All current omni traffic should still route through
  `Evaluate.generate()`** (TokenIterator). That path is the verified
  39/39 green matrix across reasoning toggle / mixed modality /
  media-salt / hybrid SSM warm-pass parity / 3 weight formats.
- **Path to flipping the omni BatchEngine default ON**:
  1. Re-run `BENCH_OMNI=1 BENCH_OMNI_BATCH=1` against a prepared
     MXFP4 / JANGTQ4 omni bundle with the test fix landed.
  2. If green: bump the omni registry entry's `mlxBatchEngine` flag.
  3. If red: bisect by removing `enable_thinking: false`,
     instrumenting `firstToken` sampling, and inspecting the hybrid
     cache's offset state after the `.logits` prefill.

### 10.3 Why this matters for the osaurus rollup

This rewrites the answer to "should we ship Nemotron-Omni registry
entries with `mlxBatchEngine: true`?":

- **Mechanical evidence (now)**: BatchEngine omni text-only and
  multimodal both run through the same `.logits` path that
  `Gemma3` / `FastVLM` already use in production, and the Mamba conv
  trap is closed in `d020e76`. The Reasoning event-routing is
  identical to the validated LLM path.
- **Empirical evidence (pending)**: a real-bundle `BENCH_OMNI_BATCH=1`
  run on a prepared bundle still hasn't been re-confirmed post-fix on
  this machine. **The osaurus team should not flip the default
  registry-side until that re-run lands.**
- **Conservative ship recommendation**: keep `mlxBatchEngine = OFF`
  for omni in the registry, document that B>1 / batched omni is
  "available, gated by per-request flag, validated mechanically but
  not yet in CI." Users who want it can opt in.

---

## 11. Audio decode envelope (AVAudioConverter quirks)

`nemotronOmniLoadAudioFile(url:targetSampleRate:)` handles:
- Any AVFoundation-decodable input (WAV, AAC, MP3 via system codec, M4A, FLAC).
- Auto-resample to 16 kHz mono Float32 via AVAudioConverter (single-pass
  block-conversion).
- Fast path: 16 kHz mono Float32 input bypasses the converter entirely.

**Known quirk**: AVAudioConverter is single-shot in our wiring (we set
`consumed=true` after the first input buffer). For very long audio
(>100 MB raw) you may need to chunk yourself. For the typical Nemotron-Omni
use case (≤30 s clips for chat) this is fine. If osaurus serves long-form
ASR-style audio (>1 minute), pull the converter loop out into a
multi-block driver.

**Per-sample mel normalize** is applied by default in
`nemotronOmniExtractMelFeatures`. **Do not disable it** — without per-sample
normalize the model produces nonsense ("sound of a door opening" in lieu
of speech text). The Python port lost ~30 minutes to this. The Swift port
flags it as CRITICAL in the source.

---

## 12. Migration checklist for osaurus

Use this as a PR checklist when wiring omni support into the osaurus runtime:

- [ ] `loadContainer(configuration: .init(directory: omniBundle))` returns a
      `ModelContext` whose `model` is `NemotronHOmni`. Verify by running
      a text-only smoke turn first.
- [ ] `coordinator.setHybrid(true)` called eagerly before first admission.
- [ ] `defaultKVMode = .turboQuant(keyBits: 3, valueBits: 3)`,
      `defaultMaxKVSize = 8192` (or your house defaults).
- [ ] Image turn through standard `UserInput(prompt:images:)` path —
      `NemotronHOmniProcessor` is auto-selected. No osaurus dispatch
      changes needed.
- [ ] Video turn — call `nemotronOmniPreprocessVideo` ahead of
      `LMInput`, wrap as `ProcessedVideo`.
- [ ] Audio turn — Option A (manual pre-encode + splice + custom salt)
      OR wait for `LMInput.audio` (Option B) to land.
- [ ] Multi-turn cache reuse — verify image disk cache hits across turns
      with same image + extended prompt. (`mediaSalt` should auto-resolve.)
- [ ] BatchEngine — text-only OK, multimodal turns must route through
      `Evaluate.generate()` until the BatchEngine inputsEmbeds path lands.
- [ ] Reasoning toggle — `enable_thinking=false` produces no `<think>`
      block; `=true` (default) emits `Generation.reasoning(String)`
      events, captured by your stream consumer.
- [ ] Tool calls — same flow as Qwen / DSV4 / Gemma; emit
      `.toolCall(ToolCall)` events. Parser stamp resolves to `nemotron`
      automatically.
- [ ] All three quant variants (MXFP4, JANGTQ4, JANGTQ2) load via the
      same code path. Verify with the `00_verify_all` equivalent — three
      bundles, three text-only smoke turns.

---

## 12.5 Real-bundle state — what actually loads + runs (2026-04-28)

Updated after `BENCH_OMNI=1` against the local bundles + the JANG
quant-inference fix in `ae526a3`.

| Bundle | Load | E2E multi-turn | Notes |
|---|---|---|---|
| `Nemotron-3-Nano-Omni-30B-A3B-MXFP4` (21 GB) | ✅ 1.1 s | ✅ **7/7 PASS** | text 79–121 tok/s, image 51–91 tok/s, all rows green. **Production-ready.** |
| `Nemotron-3-Nano-Omni-30B-A3B-JANGTQ4` (19 GB) | ✅ 3.1 s | ✅ **7/7 PASS** | TurboQuant codebook kernels @ bits=4. Text 88–101 tok/s, image 77–79 tok/s. **Production-ready.** |
| `Nemotron-3-Nano-Omni-30B-A3B-JANGTQ2` (12 GB) | ✅ 0.8 s | ✅ **7/7 PASS** | TurboQuant codebook kernels @ bits=2. Text 102–107 tok/s, image 82–83 tok/s. Smallest, fastest. **Production-ready.** |

### What `ae526a3` fixed

The original `[rms_norm] (*weight) must have the same size as the last
dimension of x but has 2688 elements` trap — first hit by tpae's
osaurus run on Cascade-2 JANG_4M, then independently caught by
`BENCH_OMNI=1` on Nemotron-Omni MXFP4 — was a JANG quant-inference
shape-ambiguity bug, NOT an omni-wrapper bug.

Root cause: `(bits=8, gs=32)` and `(bits=4, gs=64)` produce the SAME
packed tensor shape for any `numGroups`. The primary path of
`inferBitWidthAndGroupSize` accepts whichever matches the prior
`gs`; for bundles with mixed-gs layers + a single prior, the wrong
half got loaded with double-bits / half-gs and dequant reconstructed
wrong row vectors. Trap fired mid-prefill.

Three plumbing fixes in `ae526a3`:
1. `JangLoader.inferPerLayerQuantization` — prefer
   `jangConfig.blockSize` (authoritative) over `overrideGroupSize`
   when `bit_widths_used` is non-empty (real JANG conversion signal).
2. `VLMModelFactory._load` — pass `baseConfig.quantization` through
   to `loadWeights` so omni / VL bundles' top-level
   `quantization.group_size` lands as the prior. (LLM factory already
   did this; VLM was missing it.)
3. Both factories — pass `perLayerQuantization` through even when
   JANG; `loadWeights` retains shape walk for the JANG path but the
   plumbing is now uniform.

Affected bundles: anything with a JANG-converted `bit_widths_used:
[…, …]` and mixed gs across layers (Cascade-2 JANG_4M, Nemotron-Omni
MXFP4). Unaffected: JANG_2L bundles whose prior gs matches every
layer (Cascade-2 JANG_2L unchanged at 125 tok/s).

### What `<later commit>` fixed for JANGTQ

`NemotronHJANGTQ.swift` (new file, ~135 LOC) + minimal NemotronH.swift
edits closed `unhandledKeys: experts` and the downstream all-NaN
forward. Three pieces:

1. **NemotronHJANGTQContext** struct propagates `bits` + `mxtq_seed`
   through `NemotronHModel.init(jangtqContext:configuration:)` →
   `NemotronHBackbone` → `NemotronHBlock` → `NemotronHMoE`. When
   non-nil, the MoE constructor swaps in `NemotronHJANGTQSwitchMLP`
   with `TurboQuantSwitchLinear` fc1/fc2 instead of the affine
   `SwitchLinear`. Other layer kinds (Mamba, Attention, MLP) are
   unaffected.
2. **`NemotronHModel.sanitize`** detects per-expert
   `experts.{e}.{up,down}_proj.tq_packed` keys and stacks them into
   `switch_mlp.{fc1,fc2}.tq_packed` / `.tq_norms` (mirrors
   Python `jang_tools.load_jangtq` § "MoE stacking" `nemo_pat`).
   Also strips `.tq_bits` metadata. Idempotent for affine bundles.
3. **`NemotronHJANGTQSwitchMLP`** chains two `TurboQuantSwitchLinear`
   instances with ReLU² in between. Bypasses
   `TurboQuantSwitchLinear.callAsFunction(_:_:)` (whose K-broadcast
   is broken for affine-shape input — passes `nRows = batch * K` to a
   per-row gather kernel that only has `batch` rows of input) and
   calls `JANGTQKernels.hadamardRotate` + `JANGTQKernels.gatherTQ`
   directly. Input expanded to per-(token, expert) rows up-front
   so the gather kernel's per-row dispatch matches.

**VLMModelFactory dispatch**: omni JANGTQ bundles trigger via the
existing `config_omni.json` detection. The factory layer also merges
`weight_format` + `mxtq_bits` from `jang_config.json` into the
config-data dict before decoding, so `NemotronHOmniConfiguration`
sees them and wires `NemotronHJANGTQContext` automatically.

Reproduce locally (each bundle, B=1, M3 Max):
```bash
for variant in MXFP4 JANGTQ4 JANGTQ2; do
  BENCH_OMNI=1 \
    BENCH_MODEL=~/.mlxstudio/models/JANGQ-AI/Nemotron-3-Nano-Omni-30B-A3B-$variant \
    BENCH_MAX_TOKENS=24 \
    swift run -c release RunBench
done
# Expected: all three print "7 passed, 0 failed"
```

### Osaurus posture (revised, 2026-04-28)

- **Ship MXFP4 + JANGTQ4 + JANGTQ2 omni** — all three pass 7/7 on
  real bundles. Smallest+fastest is JANGTQ2 (12 GB, 102 tok/s). Highest
  precision is MXFP4 (21 GB).
- **All non-omni paths unaffected** by either change. Cascade-2 JANG_2L
  + JANG_4M text-only also benefit from the `ae526a3` fix.

Reproduce locally:

```bash
BENCH_OMNI=1 \
  BENCH_MODEL=~/.mlxstudio/models/JANGQ-AI/Nemotron-3-Nano-Omni-30B-A3B-MXFP4 \
  BENCH_MAX_TOKENS=24 \
  swift run -c release RunBench
# Expected: "7 passed, 0 failed | load 1.11s"
```

---

## 13. Known gaps + tracking

| Gap | Severity | Owner | Tracking |
|---|---|---|---|
| ~~MXFP4 omni first-forward crash (rms_norm 2688)~~ | ~~HIGH~~ | ~~vmlx-side~~ | **CLOSED in `ae526a3`** |
| ~~`NemotronHJANGTQ.swift` missing~~ | ~~HIGH~~ | ~~vmlx-side~~ | **CLOSED — JANGTQ4 + JANGTQ2 both 7/7 PASS** |
| `LMInput` has no audio field | medium | vmlx-side | this doc §3.2; unblock once osaurus signals demand |
| `MediaSalt` skips audio | medium | vmlx-side | this doc §3.3; trivial fix once §3.2 lands |
| BatchEngine prefill uses tokens, not inputsEmbeds | medium | vmlx-side | this doc §10.1; affects ALL VLMs not just omni |
| Compiled decode for hybrid (Stage 4) | low | vmlx-side | called out in `BatchCompile.swift:85`; perf optimization, not correctness |
| EVS keep-first-frame edge case | low | vmlx-side | `vlmApplyEVS` first-group always-kept logic; matches Python ref but Python defaults `dissimilarity[0] = 255` directly. Re-verify if you see token-count drift between Python + Swift. |
| Long-form audio (>1 min) chunking | low | osaurus-side | AVAudioConverter loop in §11; escalate if you serve long ASR clips |

---

## 14. Cross-VLM video sharing (don't forget!)

`VLMVideoUtils.swift` is a peace offering for the rest of the VLM
families. It's not omni-specific. If osaurus ever wants to add video
support to **any** of these — Qwen 2/2.5/3/3.5/3.6 VL, Kimi VL, Gemma 4 VL,
Mistral 3 VL — the primitives are there:

```swift
import MLXVLM

let frames = try await vlmExtractFramesUniform(url: videoURL, targetFrames: 16)
let pixelValues = vlmStackFramesIntoChannels(
    frames, imageSize: 448,
    temporalPatchDim: 1,                      // 1 = no stacking, per-frame patch
    mean: SIGLIP_NORM_MEAN, std: SIGLIP_NORM_STD)
let embeds = qwenModel.extractImageEmbeds(pixelValues: pixelValues)
let pruned = vlmApplyEVS(embeds.reshaped([groups, P, hidden]),
                          pruningRate: 0.5,
                          keepFirstFrame: true)
```

This unifies the video path so only the per-model ViT body and
per-model placeholder-token convention vary.

---

## Appendix A — Full module index

```
Libraries/MLXVLM/Models/NemotronHOmni/
├── NemotronHOmni.swift            (431 LOC) — VLMModel wrapper, processor, splice
├── RADIOVision.swift              (323 LOC) — RADIO ViT with CPE bilinear interp
├── Parakeet.swift                 (403 LOC) — 24-layer Conformer, Transformer-XL rel-pos
├── Projectors.swift                (96 LOC) — mlp1 + sound_projection + remap helpers
└── Preprocessors.swift            (739 LOC) — NVLM tiling, mel STFT, video frames + EVS

Libraries/MLXVLM/
├── VLMVideoUtils.swift            (228 LOC) — shared video primitives (cross-VLM)
└── VLMModelFactory.swift                   — registry + omni dispatch + processor override

Libraries/MLXLLM/Models/
└── NemotronH.swift                         — embedTokens(), callAsFunction(inputsEmbeds:)

Tests/MLXLMTests/
└── NemotronHOmniSmokeTests.swift   (160 LOC) — 12 smoke tests covering all towers
```

## Appendix B — Quick verify

```bash
# Build all targets
swift build -c release

# Smoke tests (no bundle needed, ~1.5 s)
swift test --filter NemotronHOmniSmokeTests

# Real-bundle e2e (requires ~/.mlxstudio/models/JANGQ-AI/Nemotron-3-Nano-Omni-30B-A3B-MXFP4)
# — gated future work; will land as BENCH_NEMOTRON_OMNI_BUNDLE harness.
```

---

**Last updated**: 2026-04-28. Tracks `feat(omni): native Swift port` (commit
`b4eec09`) plus the Stage 3 Swift video closeout discussed in
`research/NEMOTRON-OMNI-SWIFT-VIDEO-2026-04-28.md`.

If you're integrating omni into osaurus and hit something this doc didn't
warn you about, file an issue tagged `omni-integration` and link the
specific section so I can sharpen the doc + close the gap.
