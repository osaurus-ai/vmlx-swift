# vMLX-Flux (native mFLUX) — osaurus integration spec

**Audience:** osaurus engineers wiring native on-device image/video generation.
**Engine repo:** `jjang-ai/vmlx-flux` (standalone SwiftPM) — vendored into
`vmlx-swift/Libraries/vMLXFlux*` so the whole vMLX stack shares one MLX runtime.
**Status date:** 2026-06-15. **Owner:** Eric.

> This is engineering documentation for the vmlx-flux API surface. It is NOT
> wiki content and contains no secrets — safe to share with osaurus teammates
> and to live in the public `jjang-ai/vmlx-flux` repo. Do not copy private wiki
> pages into it.

---

## 0. TL;DR for the integrator

- One import: `import vMLXFlux`. One actor: `FluxEngine`. Four call sites:
  `load`, `generate`, `edit`, `upscale` (+ `generateVideo`, future).
- Everything is **streaming**: each call returns
  `AsyncThrowingStream<ImageGenEvent, Error>`; the terminal `.completed(url:seed:)`
  carries the saved PNG path.
- **No silent downloads, ever.** `FluxEngine.load` requires a local weights dir.
  osaurus's download/staging layer must place weights first (same rule as LLMs).
- **mflux packaging** is first-class: quantized `*-mflux-4bit` model dirs with the
  Diffusers/MFlux component layout (`transformer/`, `text_encoder/`, `vae/`,
  `tokenizer/`) load directly via `MLXStudioModelStore`.
- **GPU concurrency:** image-gen MLX eval races LLM eval on the shared Metal
  command buffer exactly like the Model2Vec embedder did. osaurus MUST route
  image generation through the same `MetalGate` exclusion it already uses for
  embeddings (see §7). This is the single most important wiring correctness note.
- **Per-model status:** `z-image-turbo` = native pipeline implemented (real text
  encoder + DiT + VAE). `qwen-image` / `qwen-image-edit` / `flux2-klein` /
  `flux1-*` / `seedvr2` / `wan-2.x` = registered but throw `notImplemented`
  pending per-model ports. See §6.

---

## 1. Module / build layout

Vendored into vmlx-swift's monorepo `Package.swift` as in-tree targets (they
reuse the in-tree `MLX`, `MLXNN`, `MLXRandom`, `MLXLMCommon`, `VMLXTokenizers`
targets so there is exactly one MLX binary across LLM + image + video):

| Product | Path | Role |
|---|---|---|
| `vMLXFlux` | `Libraries/vMLXFlux` | Umbrella. `@_exported import`s the three below. The one import callers need. |
| `vMLXFluxKit` | `Libraries/vMLXFluxKit` | Core: protocols, `FluxEngine` types, requests/events, registry, schedulers, VAE, weight loader, `MLXStudioModelStore`, JANG bridge. |
| `vMLXFluxModels` | `Libraries/vMLXFluxModels` | Concrete models (ZImage native, Flux1/2, Qwen, FIBO, SeedVR2). |
| `vMLXFluxVideo` | `Libraries/vMLXFluxVideo` | WAN 2.1/2.2 video (scaffold). |
| `vmlxflux-probe` (exe, target `vMLXFluxProbe`) | `tools/vMLXFluxProbe` | Scan / load / generate CLI for local bundles. |

**Integration note for a standalone SwiftPM consumer (e.g. osaurus if it depends
on the published `jjang-ai/vmlx-flux` package instead of vendoring):** the engine
package imports `Tokenizers` and `MLXLMCommon` from `swift-transformers` and
`vmlx-swift-lm`. Inside vmlx-swift's monorepo those become `VMLXTokenizers` and
the in-tree `MLXLMCommon` — the one source delta when vendoring is
`import Tokenizers` → `import VMLXTokenizers` in `ZImageNative.swift`.

---

## 2. FluxEngine API

```swift
public actor FluxEngine {
    public init()

    // Load by canonical name from an explicit local weights dir.
    public func load(name: String, modelPath: URL, quantize: Int? = nil) async throws

    // Resolve + load a local bundle from ~/.mlxstudio/models/image (or a custom store).
    @discardableResult
    public func load(name: String, from store: MLXStudioModelStore = .init()) async throws -> LocalFluxModel

    public func unload()

    public func generate(_ r: ImageGenRequest)  -> AsyncThrowingStream<ImageGenEvent, Error>
    public func edit(_ r: ImageEditRequest)      -> AsyncThrowingStream<ImageGenEvent, Error>
    public func upscale(_ r: UpscaleRequest)     -> AsyncThrowingStream<ImageGenEvent, Error>
    public func generateVideo(_ r: VideoGenRequest) -> AsyncThrowingStream<VideoGenEvent, Error> // throws notImplemented
}
```

- **Actor-isolated.** MLX ops are not thread-safe across one allocator and the
  generation loop holds a persistent latent buffer, so every entry point hops the
  actor executor. Hold ONE `FluxEngine` per process.
- `load(name:from:)` enforces "no silent downloads": it resolves a *local* dir via
  `MLXStudioModelStore`, rejects incomplete bundles (`FluxError.localModelIncomplete`),
  and only then constructs the model. Use this for the osaurus model-picker path.
- Dispatch is capability-based: `generate` requires the loaded model conform to
  `ImageGenerator`, `edit`→`ImageEditor`, `upscale`→`ImageUpscaler`. Mismatch →
  `FluxError.wrongModelKind`.

### Errors (`FluxError`)
`unknownModel`, `notLoaded`, `wrongModelKind(expected:actual:)`, `weightsNotFound`,
`localModelNotFound(name,root)`, `localModelIncomplete(url,reasons:)`,
`notImplemented(String)`, `invalidRequest(String)`. All `CustomStringConvertible`.

---

## 3. Request + Event types

### ImageGenRequest (text→image)
```swift
ImageGenRequest(prompt:, negativePrompt:nil, width:1024, height:1024,
                steps:20, guidance:3.5, seed:UInt64?, numImages:1,
                outputDir:URL, outputFormat:.png)
```
Notes: width/height must be divisible by 16 for Z-Image (else
`invalidRequest`). `guidance == 0` ⇒ no CFG pass (turbo models). `seed == nil` ⇒
nondeterministic. `numImages > 1` is not yet batched (see §6 open work).

### ImageEditRequest (image[+mask]→image)
```swift
ImageEditRequest(prompt:, sourceImage:URL, mask:URL?, strength:0.75,
                 width:nil, height:nil, steps:20, guidance:3.5, seed:,
                 outputDir:, outputFormat:.png)
```
`mask`: white=edit, black=keep. `strength`: 0..1 deviation from source. `width/height nil`⇒match source.

### UpscaleRequest (SeedVR2)
```swift
UpscaleRequest(sourceImage:URL, scale:4, steps:10, seed:, outputDir:, outputFormat:.png)
```

### VideoGenRequest (WAN, future)
```swift
VideoGenRequest(prompt:, negativePrompt:nil, width:1280, height:720,
                numFrames:121, fps:24, steps:50, guidance:5.0, seed:, outputDir:)
```

### Events
```swift
enum ImageGenEvent {
    case step(step:Int, total:Int, etaSeconds:Double?)   // 1-indexed step counter for the UI
    case preview(pngData:Data, step:Int)                 // optional partial decode (not all models)
    case completed(url:URL, seed:UInt64)                 // terminal success → saved image path
    case failed(message:String, hfAuth:Bool)             // hfAuth=true ⇒ 401/403 → show "add HF token" CTA
    case cancelled
}
enum VideoGenEvent { case step; case preview(pngData:,frame:); case completed(url:,seed:,fps:,frameCount:); case failed; case cancelled }
enum ImageFormat: String { case png, jpeg, webp }   // png is the only fully-wired writer today
```
Cancellation: the generation loop checks `Task.isCancelled` per step, so cancelling
the consuming `Task` stops generation at the next step boundary.

---

## 4. Model registry + canonical names

`ModelRegistry` is a decentralized canonical-name → loader map. Each model
self-registers in its own file via a `static let _register` idiom; call
`VMLXFluxModels.registerAll()` + `VMLXFluxVideo.registerAll()` once at startup
(the `FluxEngine.load(name:from:)` path does this for you).

`ModelEntry`: `name`, `displayName`, `kind` (`imageGen|imageEdit|imageUpscale|videoGen`),
`defaultSteps`, `defaultGuidance`, `supportsLoRA`, `loader`.

Lookup is fuzzy (`lookupFuzzy`): lowercases, strips HF org prefix (`org/model`),
strips `-4bit`/`-8bit`/`-3bit` suffix, collapses `flux.1`→`flux1`, `_`→`-`. So
`"Tongyi/Z-Image-Turbo-mflux-4bit"` resolves to canonical `"z-image-turbo"`.

| Canonical | Display | Kind | Default steps / guidance |
|---|---|---|---|
| `flux1-schnell` | FLUX.1 Schnell | imageGen | 4 / 0.0 |
| `flux1-dev` | FLUX.1 Dev | imageGen | 20 / 3.5 |
| `flux2-klein` | FLUX.2 Klein | imageGen | 20 / 3.5 |
| `z-image-turbo` | Z-Image Turbo | imageGen | 4 / 0.0 |
| `qwen-image` | Qwen-Image | imageGen | — |
| `fibo` | FIBO | imageGen | — |
| `flux1-kontext` | FLUX.1 Kontext | imageEdit (prompt-only) | — |
| `flux1-fill` | FLUX.1 Fill | imageEdit (mask) | — |
| `flux2-klein-edit` | FLUX.2 Klein Edit | imageEdit | — |
| `qwen-image-edit` | Qwen-Image-Edit | imageEdit | — |
| `seedvr2` | SeedVR2 | imageUpscale | 10 |
| `wan-2.1`, `wan-2.2` | Wan 2.x | videoGen | scaffold |

---

## 5. mflux packaging + local model store

`MLXStudioModelStore(root: ~/.mlxstudio/models/image)` scans for local image
bundles before any download path is considered. **Exact local directory names win
over canonical family aliases** (so `Z-Image-Turbo-mflux-4bit` and `Z-Image-Turbo`
both map to `z-image-turbo` but resolve to their own dirs).

`scan()` → `[LocalFluxModel]` with:
- `directory`, `directoryName`, `canonicalName?`, `displayName`, `kind?`
- `quantizationBits?` (parsed from `-Nbit` / mflux dir naming)
- `components: Set<{root,tokenizer,transformer,scheduler,textEncoder,vae,assets}>`
- `safetensorCount`, `totalBytes`, `hasModelIndex`
- `readiness: {loadableScaffold, incomplete, unknown}`, `blockedReasons`
- `canEnterNativeLoadPath` (== `readiness == .loadableScaffold`)

**mflux component layout** (what the WeightLoader expects):
```
<model>/
  tokenizer/        # AutoTokenizer.from(modelFolder:) — HF tokenizer.json
  text_encoder/     # (Qwen-style for Z-Image) safetensors
  transformer/      # DiT weights, possibly *.safetensors.index.json sharded
  vae/              # AutoencoderKL decoder weights
  scheduler/, assets/   # optional
```
`WeightLoader.load(from:)` enumerates shards per component (prefers
`model.safetensors.index.json` / `diffusion_pytorch_model.safetensors.index.json`,
else globs `*.safetensors`), merges into a flat `[String: MLXArray]` *and* a
`componentWeights["transformer"|"text_encoder"|"vae"]` map, and detects JANG via
`JangBridge` (reuses vmlx-swift-lm's `JangLoader`/`JangConfig`) so JANG-quantized
bundles carry their per-layer quant metadata. The `*-mflux-4bit` bundles decode
their 4-bit linears through scale tensors at load time inside the model.

---

## 6. Per-model implementation status (the truth matrix)

| Canonical | Native runtime status | What's real | What's missing |
|---|---|---|---|
| **z-image-turbo** | `native_pipeline_implemented` | Full native port: Qwen-style text encoder, patchify+caption-concat DiT (noise/context refiners + unified layers, RoPE, adaLN, timestep embed), real `AutoencoderKL` VAE decode, real 4-bit weight decode, PNG out. | Live same-seed prompt-sensitivity proof (see §8). 1024px tuning. |
| qwen-image / qwen-image-edit | `not_implemented` | Bundle scans + loads. | Qwen text-encoder port + transformer weight key-map. Body throws `notImplemented`. |
| flux2-klein / flux2-klein-edit | `not_implemented` | Bundle scans + loads; `FluxDiTConfig.flux2Klein` preset exists. | T5 (single-encoder) port + weight key-map + 3-axis RoPE. |
| flux1-schnell/dev/kontext/fill | `not_implemented` | Scheduler, VAE, DiT topology + 40-line BFL key-map docblock exist in `FluxDiT.swift`. | T5-XXL + CLIP-L ports + `Module.update` key mapping + Flux 3-axis RoPE. |
| seedvr2 | scaffold | registered | upscale arch (different family). |
| wan-2.1 / wan-2.2 | scaffold | full pipeline scaffolded (WanVAE3D + WanDiT + MP4 writer) with random weights. | real weight key-map, real Conv3d (currently a Conv2d shim), windowed attention for >3-4s clips. |

**Shared P0 blockers for the Flux/Qwen family** (z-image already cleared these via
its own native port): (1) safetensors→module key-map + `Module.update`, (2) T5-XXL
text encoder (shared Flux/Qwen/Wan), (3) CLIP-L (Flux1), (4) Flux 3-axis RoPE.

**mflux feature surface (scope = "mflux type, not just flux type"):**
- ✅ Quantized mflux packaging (`*-mflux-4bit`) load + decode — implemented for z-image.
- ✅ Image *edit* protocol (`ImageEditor`, mask/strength) — API present; per-model bodies pending.
- ⬜ LoRA — `ModelEntry.supportsLoRA` flag exists (false everywhere today); loader hook TBD.
- ⬜ ControlNet / img2img strength conditioning — request fields exist (`strength`, `mask`); wiring pending per model.

---

## 7. GPU concurrency — REQUIRED osaurus wiring (do not skip)

osaurus already learned this with the Model2Vec embedder (PR #1507): a second MLX
graph eval racing the LLM eval on the **shared Metal command buffer** triggers
`addCompletedHandler: unrecognized selector` SIGABRT. **Image generation is a
second MLX graph and has the same hazard.**

Required: route every image/video generation through the same `MetalGate`
exclusion osaurus uses for embeddings. Treat image-gen like the embedder —
**EXCLUSIVE** with respect to LLM generations (acquire at submit, release after the
producer fully drains incl. the final VAE decode eval). Do NOT run image gen
concurrently with token generation on the same device. Acquire the gate in the
osaurus bridge *around* the `for try await event in engine.generate(...)` drain,
not inside vmlx-flux (the engine is deliberately gate-agnostic, same as the LLM
eval hot path).

---

## 8. Sandbox / cache / memory notes for osaurus

- **No silent downloads.** `FluxEngine.load` needs a staged local dir. Wire
  osaurus's DownloadManager to stage the full mflux bundle (all components) before
  calling load; surface `FluxError.localModelIncomplete.reasons` to the UI.
- **Storage matters for the GPU watchdog.** MLX mmaps safetensors lazily; if the
  weights live on a slow volume (external USB), the first transformer forward
  stalls the Metal command buffer on IO and trips the GPU timeout
  (`kIOGPUCommandBufferCallbackErrorTimeout`). **Stage image weights on the
  internal SSD.** (Observed 2026-06-15: 5.5GB 4-bit z-image at 512px timed out from
  USB, fine from SSD.)
- **RAM.** Full bundles are large (z-image 32GB, qwen 25GB, flux2-klein 52GB). The
  4-bit mflux variants (z-image 5.8GB) are the on-device-friendly path. Gate model
  load against the memory-safety plan like LLMs; `unload()` between model switches.
- **No paged/degenerate cache.** Image gen has no KV cache; the only persistent
  buffer is the per-job latent. Nothing to SSD-cache. Don't confuse with LLM
  prefix cache.
- **Output dir** is caller-supplied per request; the engine writes PNG via a
  `@MainActor` `ImageIO.writePNG` and returns the URL in `.completed`.

---

## 9. How osaurus bridges (recommended shape)

Mirror the documented `FluxBackend` pattern (the bridge lives on the osaurus
`Engine`, NOT in vmlx-flux — the engine has zero knowledge of osaurus):

1. Hold a lazily-created `FluxEngine` on the osaurus engine actor (store as `Any?`
   so non-image files don't import vMLXFlux).
2. `generateImage(prompt:model:settings:)`: acquire MetalGate (§7) → ensure the
   requested model is loaded (`load(name:from:)`, load-if-different) → build
   `ImageGenRequest` from osaurus settings → drain the stream → translate
   `ImageGenEvent` → osaurus's UI event type → return final URL → release gate.
3. `editImage(...)` / `upscaleImage(...)`: same, via `edit` / `upscale`.
4. Fan events to the UI via a per-job bridge so the chat surface can subscribe
   without holding a direct flux reference.
5. Event-shape translation: vmlx-flux emits separate `.step` and `.preview`; if
   osaurus uses a unified `.step(step:total:preview:)`, coalesce them in the bridge.

---

## 10. Probe tool (verification harness)

`vmlxflux-probe` (built: `swift build --product vmlxflux-probe`):
```
vmlxflux-probe --root <dir> --model <name|dir> --generate --json \
  --seed N --width W --height H --steps S \
  --turn "prompt A" --turn "prompt B" ... \
  --artifacts <dir> --output-dir <dir>
# or --matrix to scan+load+gen across every local bundle and emit a
# compatibility-matrix.{json,md}
```
Per-turn seed = `--seed` if given (fixes ALL turns to one seed — use for
same-seed/different-prompt sensitivity tests), else `turnIndex+1`. Artifacts:
per-model `*-facts.json`, `*-load.json` (with per-turn image SHA256 + dims),
`scan.{json,md}`, `compatibility-matrix.{json,md}`.

---

## 11. Live-proof verdict (z-image-turbo, 4-bit) — ✅ PASS (2026-06-15)

Probe: `Z-Image-Turbo-mflux-4bit` from internal SSD, seed 7, 512×512, 8 steps,
3 turns (apple / mountain / apple). Artifacts in
`docs/local/vmlx-flux-probes/PROOF-zimage-4bit-ssd/`.

| Check | Result |
|---|---|
| **Determinism** (turn1 apple seed7 vs turn3 apple seed7) | byte-identical SHA `d101a8…` ✅ |
| **Prompt-sensitivity** (turn2 mountain seed7 vs apple) | different SHA `0336dc…` ✅ — text encoder conditions output |
| **Visual coherence** | turn1/3 = coherent photo of a red apple on a wooden table; turn2 = coherent watercolor of a snowy blue mountain ✅ |
| **Style control** | "photograph" vs "watercolor painting" both honored ✅ |
| **Speed** | ~4.1–4.5 s per 512px/8-step; ~18.7 s per 1024px/8-step (4-bit, M5 Max) |
| **Native 1024px** | coherent photorealistic cabin-in-pine-forest-at-sunset, prompt-accurate ✅ |
| **Stability** | no GPU timeout from SSD; USB-resident weights DO time out (§8) |

**Expanded coverage (2026-06-15, same bundle):**

| Dimension | Result |
|---|---|
| **CFG / negative-prompt path** (`guidance 3.0` + negative) | coherent prompt-accurate apple; 7.6 s vs 4.2 s at guidance 0 — the ~1.8× confirms two forward passes per step (real classifier-free guidance). Previously-untested `negativeEncodings` branch ✅ |
| **Seed sensitivity** (same prompt, seed 100 vs 200) | distinct coherent images per seed ✅ |
| **Resolution/speed curve** (8-step, guidance 0) | 256²→1.3 s · 512²→4.3 s · 768²→10.6 s · 1024²→20.8 s |
| **Unit tests** (`swift test --filter vMLXFluxTests`, Xcode toolchain) | 19/19 green |

Probe gained `--guidance` and `--negative` flags to drive the CFG path.

**Conclusion:** the native Swift Z-Image pipeline (text encoder + DiT + VAE + 4-bit
decode) produces real, deterministic, prompt-conditioned images across resolutions,
seeds, and the CFG path. z-image-turbo is **production-compatible** — the May-16
"scaffold_generates_png_noise" verdict is superseded; that predated `ZImageNative.swift`.

---

## 12. Open work / follow-ups (prioritized)

1. Land + commit the vendored engine in vmlx-swift (local-only fork — never pushed)
   and push the ahead-of-repo native work (ZImageNative, LocalModelStore, probe,
   loader edits) back to `jjang-ai/vmlx-flux`.
2. Prove z-image-turbo prompt-sensitivity live (§11); promote to production-compatible.
3. Port the shared T5-XXL + CLIP-L encoders → unblocks Flux1/Flux2/Qwen at once.
4. Per-model weight key-maps (start Flux1-Schnell; Qwen-Image-Edit for the edit path).
5. LoRA loader hook (`supportsLoRA`), img2img/controlnet conditioning.
6. `numImages > 1` batching; webp/jpeg writers; preview-decode cadence.
7. Wire MetalGate exclusion in the osaurus bridge (§7) before shipping.
