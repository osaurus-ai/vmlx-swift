# vMLX-Flux (native mFLUX) — osaurus integration spec

**Audience:** osaurus engineers wiring native on-device image/video generation.
**Engine repo:** `jjang-ai/vmlx-flux` (standalone SwiftPM) — vendored into
`vmlx-swift/Libraries/vMLXFlux*` so the whole vMLX stack shares one MLX runtime.
**Status date:** 2026-06-16. **Owner:** Eric.

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
- **Per-model status:** Osaurus `vmlx-origin/main` runtime-proof baseline
  `a188a2ccecc92c8a5993506acc83df16f83c7420` has fresh live proof for
  `z-image-turbo` 4/8-bit, `flux1-schnell` 4/8-bit, `qwen-image` 4/6-bit,
  `qwen-image-edit` q4/q5, and staged `ideogram-4-fp8`. Load matrix:
  `docs/local/vmlx-flux-probes/2026-06-16-current-a188-load-matrix/compatibility-matrix.json`.
  Generation/edit roots:
  `2026-06-16-current-a188-zimage-4bit-gen/`,
  `2026-06-16-current-a188-zimage-8bit-gen/`,
  `2026-06-16-current-a188-flux-schnell-4bit-gen/`,
  `2026-06-16-current-a188-flux-schnell-8bit-gen/`,
  `2026-06-16-current-a188-qwen-image-4bit-gen20/`,
  `2026-06-16-current-a188-qwen-image-6bit-gen20/`,
  `2026-06-16-current-a188-qwen-edit-q4-gen20/`,
  `2026-06-16-current-a188-qwen-edit-q5-gen20/`, and
  `2026-06-16-current-a188-ideogram-fp8-object-strict/`.
  qwen-edit q3 is incomplete because its text-encoder index references missing
  `text_encoder/3.safetensors`; q6 is incomplete; qwen masks are intentionally
  hidden because the mflux qwen-edit reference has no qwen mask/inpaint
  argument or path. qwen-edit q5 is the cleanest edit row; current q4 proof is
  deterministic and color-sensitive but weaker on shape-changing green-pear
  prompts. `ideogram-4-fp8` is implemented/testable for the staged
  `cocktailpeanut/ideogram-4-fp8` mirror with readable HELLO/BANANA typography
  proof plus current clean 512px object-icon proof. Official `ideogram-ai/*`
  downloads remain approval-gated for the current account (`hf download ... --dry-run`
  returned access denied for fp8 and nf4 on 2026-06-16), and nf4 is incomplete.
  See §6 and §7b.

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
the in-tree `MLXLMCommon` — when syncing from standalone, replace
`import Tokenizers` with `import VMLXTokenizers` in the native image files
(`ZImageNative.swift`, `Flux1Native.swift`, `QwenImageNative.swift`, and
`QwenImageEditSupport.swift`).

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

### ImageEditRequest (image(s)[+mask where supported]→image)
```swift
ImageEditRequest(prompt:, sourceImage:URL, mask:URL?, strength:0.75,
                 width:nil, height:nil, steps:20, guidance:3.5, seed:,
                 outputDir:, outputFormat:.png)

try ImageEditRequest(prompt:, sourceImages:[URL], mask:URL?, strength:0.75,
                     width:nil, height:nil, steps:20, guidance:3.5, seed:,
                     outputDir:, outputFormat:.png)
```
`sourceImage` is the legacy single-image path. `sourceImages` is the ordered
multi-reference path; it throws `FluxError.invalidRequest` for an empty array.
For qwen-image-edit, the last source image drives the mflux sizing plan and all
source images feed Qwen-VL prompt features plus VAE conditioning latents.
`mask`: white=edit, black=keep for models that really support masks.
Qwen-image-edit does not; hide qwen mask controls and reject non-null qwen masks.
`strength`: 0..1 deviation from source. `width/height nil`=>match source.

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
| `ideogram` | Ideogram 4 | imageGen | 20 / 7 (fp8 weights: ideogram-ai/ideogram-4-fp8) |
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
- `components: Set<{root,tokenizer,transformer,unconditionalTransformer,scheduler,textEncoder,vae,assets}>`
- `safetensorCount`, `totalBytes`, `hasModelIndex`
- `readiness: {loadableScaffold, incomplete, unknown}`, `blockedReasons`
- `canEnterNativeLoadPath` (== `readiness == .loadableScaffold`)

Nested quant bundles are expanded into loadable variants. The staged
`Qwen-Image-Edit-mflux` folder contains `q3`, `q4`, `q5`, and `q6`; osaurus
should present/request the exact local variant IDs:
`Qwen-Image-Edit-mflux-q3`, `Qwen-Image-Edit-mflux-q4`,
`Qwen-Image-Edit-mflux-q5`, and `Qwen-Image-Edit-mflux-q6`. Current disk state:
q4/q5 are loadable local bundles with live text-image edit proof, q3 is
incomplete because `text_encoder/model.safetensors.index.json` references
missing `text_encoder/3.safetensors`, and q6 is incomplete because it lacks
transformer/VAE shards and indexed text-encoder shards. The current q4
load-only artifact:
`docs/local/vmlx-flux-probes/2026-06-16-qwen-edit-q4-manifest-load/Qwen-Image-Edit-mflux-q4-load.json`.
Current q4 edit-preprocess artifact:
`docs/local/vmlx-flux-probes/2026-06-16-qwen-edit-q4-preprocess-live/Qwen-Image-Edit-mflux-q4-load.json`.
Current q4 prompt-token artifact:
`docs/local/vmlx-flux-probes/2026-06-16-qwen-edit-q4-prompt-live/Qwen-Image-Edit-mflux-q4-load.json`
(`input_ids_shape=1x276`, `image_token_id=151655`, `image_token_count=196`,
`template_drop_index=64`).
Current q4 VL encode artifact:
`docs/local/vmlx-flux-probes/2026-06-16-qwen-edit-q4-vl-encode-live/Qwen-Image-Edit-mflux-q4-load.json`
(`feature_shape=196x3584`, `prompt_embeds_shape=1x212x3584`, finite stats).
Current q4 VAE-conditioning artifact:
`docs/local/vmlx-flux-probes/2026-06-16-qwen-edit-q4-conditioning-after-cond-fix/Qwen-Image-Edit-mflux-q4-load.json`
(`conditioning_width=384`, `conditioning_height=384`, `patch_rows=24`,
`patch_columns=24`, `latents_shape=1x576x64`, `image_ids_shape=1x576x3`,
finite stats).
Current q4 first transformer velocity artifact:
`docs/local/vmlx-flux-probes/2026-06-16-qwen-edit-q4-denoise-after-cond-fix/Qwen-Image-Edit-mflux-q4-load.json`
(`target_latent_count=1024`, `conditioning_latent_count=576`,
`combined_velocity_shape=1x1600x64`, `target_velocity_shape=1x1024x64`,
`image_shapes=[[1,32,32],[1,24,24]]`, finite stats).
Current q4 same-seed edit proof:
`docs/local/vmlx-flux-probes/2026-06-16-qwen-edit-q4-determinism-after-cond-fix/Qwen-Image-Edit-mflux-q4-load.json`
(`load_status=loaded`; turn 1/3 blue-apple prompt SHA
`005ab8baddfe9b7a94aa83f8ddd22d192e7e5a0275c556dcf2ead76a565e474a`; turn 2
green-pear prompt SHA
`815711be73a9e89599b3e97f9f15196115875103f9407d7b1b61bab33de8e3b4`).
Viewed output is coherent and prompt-sensitive. Source trace: mflux passes
`vl_width/vl_height` into edit conditioning and uses those dimensions for the
source-image VAE encode when present; Swift now matches that path.
Current q5 same-seed edit proof:
`docs/local/vmlx-flux-probes/2026-06-16-qwen-edit-q5-determinism/Qwen-Image-Edit-mflux-q5-load.json`
(`load_status=loaded`; turn 1/3 blue-apple prompt SHA
`5cd5d9197bd659bd8b59b4a2f2bca413266146ad4e08249289d5fa6a8025fa4e`; turn 2
green-pear prompt SHA
`d2c6c4eb4a19dcf48122b5216fc15ac37b9f5aa49c15f596acd1276a4df57034`).
Viewed output is coherent and prompt-sensitive. Current q3 live-load blocker:
`docs/local/vmlx-flux-probes/2026-06-16-qwen-edit-q3-determinism/Qwen-Image-Edit-mflux-q3-load.json`
(`load_status=failed`, missing `text_encoder/3.safetensors`). Scanner artifact:
`docs/local/vmlx-flux-probes/2026-06-16-qwen-edit-q3q5-after-proof/scan.json`
reports q3 incomplete, q4/q5 implemented, and q6 incomplete.

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
| **z-image-turbo** | `native_pipeline_implemented` | Full native port: Qwen-style text encoder, patchify+caption-concat DiT (noise/context refiners + unified layers, RoPE, adaLN, timestep embed), real `AutoencoderKL` VAE decode, real 4/8-bit weight decode, PNG out. Fresh 2026-06-16 proof: 4-bit + 8-bit live load, 3 completed turns, same-prompt SHA match, different-prompt SHA change, viewed coherent apple/mountain images. | 1024px tuning. |
| **qwen-image** | `native_pipeline_implemented` | Full native pipeline `QwenImageNative.swift`: Qwen2.5 LM text encoder (GQA), 60-layer MM-DiT (joint attention + 3-axis RoPE), 3D causal-conv VAE, mflux guidance-rescaled CFG. Current a188 4-bit proof: `docs/local/vmlx-flux-probes/2026-06-16-current-a188-qwen-image-4bit-gen20/qwen-image-mflux-4bit-load.json` (apple/repeat SHA `2c7b1c35af73ed66d45a958e7c2204c635ac95bef5e6a167d81e1285b7579b12`, mountain `cae848046c34d06a6407e28e9d7cd820c9721906a8defd92de5b23666599ecb8`; viewed coherent). Current a188 6-bit proof: `docs/local/vmlx-flux-probes/2026-06-16-current-a188-qwen-image-6bit-gen20/Qwen-Image-mflux-6bit-load.json` (apple/repeat `e5865d8b2a90bb759d4c6f5a647b03e0ba7542e2f4d7ed667ec8920c9f25b866`, mountain `e41ca45ea175cfa965db565e4d8f8d0aa84b3db908dd10fb9efccbb418839b50`; viewed coherent). | Public mflux 8-bit not found/staged; full not staged/proven. Three port bugs fixed: VAE conv weights are MLX channels-last (not PyTorch); qwen timestep is raw sigma (QwenTimesteps applies ×1000 internally); qwen txt2img mod-linear keys can be either `img_mod_linear`/`txt_mod_linear` or nested `img_norm1.mod_linear`/`txt_norm1.mod_linear`. |
| qwen-image-edit | `native_pipeline_implemented` for `Qwen-Image-Edit-mflux-q4` and `Qwen-Image-Edit-mflux-q5`; `native_pipeline_partial` for incomplete q3/q6 | q4/q5 scan as loadable local bundles and have live text-image edit proof after the indexed-shard scanner fix. q3 scans incomplete because its text-encoder index references missing `text_encoder/3.safetensors`; q6 scans incomplete because it lacks transformer/VAE shards and indexed text-encoder shards. `Qwen-Image-Edit-mflux-q4` passes manifest-gated engine load against tokenizer files, Qwen LM keys, Qwen-VL vision keys, transformer keys, and VAE encode/decode keys. Live probes prove real tokenizer image-pad expansion, Qwen2.5-VL prompt-image encode, fixed VL-size VAE static image latents, first edit-shaped transformer velocity, and ImageEditor scheduler/decode/PNG output. Current a188 q4 proof: `docs/local/vmlx-flux-probes/2026-06-16-current-a188-qwen-edit-q4-gen20/Qwen-Image-Edit-mflux-q4-load.json` (blue/repeat SHA `5d658df3259f85502188c4432896682708ce96f3ebaab3c1970678585f746be9`, green prompt `91df895dcd3d8eea33fbb464642d0d1ad1febecf646257f46e20efaac400ab7d`; viewed color/shape-sensitive but noisier/weaker for green-pear shape change). Current a188 q5 proof: `docs/local/vmlx-flux-probes/2026-06-16-current-a188-qwen-edit-q5-gen20/Qwen-Image-Edit-mflux-q5-load.json` (blue/repeat `5265852e90c727b45c224f887254763152c8612f55a48b7932fa1d12327d98c8`, green pear `41732444e47cbc028dac25035e70ffdf216203c94e3d6a7685d4bb729d20ea19`; viewed clean blue apple and green pear). Non-null qwen masks are rejected before the edit pipeline loads and covered by `QwenImageEditSupportTests.testQwenImageEditRejectsMaskBeforePipelineLoad`. | q3/q6 need complete local bundles before UI promotion; qwen mask/inpaint is unsupported by the current mflux reference; q4 should be labeled lower quality than q5 for shape-changing edits; broader Osaurus production matrix still pending. |
| flux2-klein / flux2-klein-edit | `not_implemented` | Bundle scans + loads; `FluxDiTConfig.flux2Klein` preset exists. | T5 (single-encoder) port + weight key-map + 3-axis RoPE. |
| **flux1-schnell** | `native_pipeline_implemented` | Full native pipeline `Flux1Native.swift`: T5-XXL + CLIP-L encoders, full DiT (19 joint + 38 single blocks, 24h×128, 3-axis RoPE), AutoencoderKL VAE, mflux decode. Fresh 2026-06-16 proof: 4-bit + 8-bit live load, 3 completed turns, same-prompt SHA match, different-prompt SHA change, viewed coherent apple/mountain images. | tokenizer.json must be staged (mflux ships slow tokenizers — convert; see port plan). Full precision pending. |
| flux1-dev/kontext/fill | `not_implemented` | dev = schnell + guidance embedder (small add); kontext/fill = edit variants. | wire guidance + edit conditioning on the working schnell pipeline. |
| **ideogram** (Ideogram 4) | `native_pipeline_implemented` for staged fp8 mirror; nf4 incomplete | Strong text/typography renderer. mflux-compatible official weights: `ideogram-ai/ideogram-4-fp8` or `ideogram-ai/ideogram-4-nf4` (4-bit). Official HF metadata is reachable, but official downloads still require approval for the current account. A complete fp8 mirror bundle, `cocktailpeanut/ideogram-4-fp8`, is staged at `~/.mlxstudio/models/image/ideogram-4-fp8`; scan artifact `docs/local/vmlx-flux-probes/2026-06-16-current-a188-load-matrix/scan.json` reports `readiness=loadableScaffold`, 4 safetensors, 27,526,985,054 bytes, and tokenizer/text_encoder/transformer/unconditional_transformer/vae present. Native source runs Qwen3 text encoder, conditional and unconditional 34-layer MM-DiT, fp8 `weight_scale` linears, mflux default 20-step guidance schedule, Flux2 VAE decode, and PNG output. Source fix: both Ideogram rotary helpers match mflux `rotate_half` (`[-secondHalf, firstHalf]`). Typography proof artifact: `docs/local/vmlx-flux-probes/2026-06-16-ideogram-fp8-native-gen20-current-source/ideogram-4-fp8-load.json` (HELLO turn 1/3 SHA `6534f016378a94add5ccc29397decf45c4dada6c1d82260bdd51517390cf4205`; BANANA `b02464bd06e689ea6fc7aeb33dbc70bb1e1eb5b08c92668abc1832f61239f0b5`; viewed readable). Current a188 strict object proof: `docs/local/vmlx-flux-probes/2026-06-16-current-a188-ideogram-fp8-object-strict/ideogram-4-fp8-load.json` (apple/repeat `c62b3b71a82ebcb0964be709c03678271364d381dd4ae8029af7b85d4bf02264`, mountains `d193163f8584ad6040bc71d42960c98ac7864391f76f79c485cf8eca6905b2c1`; viewed clean). Boundary: `docs/local/vmlx-flux-probes/2026-06-16-current-a188-ideogram-fp8-object-determinism/ideogram-4-fp8-load.json` hallucinated text on a broader no-text apple prompt. | Expose only the staged fp8 mirror for testing until official access is approved if the product requires official `ideogram-ai/*`; keep normal UI/API wording scoped to typography and strict object-icon prompt coverage until broader no-text object rows pass; keep nf4 hidden until a complete local bundle has load/generation proof; broader Osaurus production matrix still pending. |
| seedvr2 | scaffold | registered | upscale arch (different family). |
| wan-2.1 / wan-2.2 | scaffold | full pipeline scaffolded (WanVAE3D + WanDiT + MP4 writer) with random weights. | real weight key-map, real Conv3d (currently a Conv2d shim), windowed attention for >3-4s clips. |

**Qwen-image-edit multi-image addendum (2026-06-16):** vMLX source includes
ordered `ImageEditRequest.sourceImages` and the probe accepts repeated
`--source-image`. Source trace: mflux qwen-edit
accepts `image_paths: list[str]`, computes sizing from `image_paths[-1]`, and
concatenates per-source prompt-image features and VAE conditioning latents.
Swift mirrors that in `QwenImageEditSupport.swift`: the preprocessing plan uses
the last source image for sizing, `QwenImageEditPromptImageEncoder` concatenates
per-image Qwen2.5-VL features and prompt image-token runs, and
`QwenImageEditConditioner` concatenates per-image VAE conditioning latents and
IDs. q5 shape proof:
`docs/local/vmlx-flux-probes/2026-06-16-qwen-edit-q5-multi-image-shape-proof/Qwen-Image-Edit-mflux-q5-load.json`
(`image_count=2`, `latents_shape=[1,1152,64]`,
`image_token_counts=[196,196]`,
`combined_velocity_shape=[1,1728,64]`,
`image_shapes=[[1,24,24],[1,24,24],[1,24,24]]`). q4 live proof:
`docs/local/vmlx-flux-probes/2026-06-16-qwen-edit-q4-multi-image-live/Qwen-Image-Edit-mflux-q4-load.json`
(turn 1/3 SHA
`e43910a505ab090bfbd4ec3a00f6e58fcd97df65c6b61e6b973754511bc740be`,
turn 2 SHA
`16ecc1fec4bdff1e5aecb9c0875569b2bebed53a81c686bf23e806faa6e2b893`).
q5 live proof:
`docs/local/vmlx-flux-probes/2026-06-16-qwen-edit-q5-multi-image-live/Qwen-Image-Edit-mflux-q5-load.json`
(turn 1/3 SHA
`ec2e49d6f300849cb46940b793670bee007f4aae8f04e97a31289670758519c9`,
turn 2 SHA
`8dfacb52aa81c6e0a8a6827c4377f27bc5d9396e39a4f8662a47db93798b767a`).
Viewed exact current-source output PNGs recorded in the q4/q5 load artifacts
(the existing contact sheet is
`docs/local/vmlx-flux-outputs/2026-06-16-qwen-edit-multi-image-contact-sheet.png`).
Visual boundary: q4 uses both images but is rougher than q5; q5 strongly uses
the mountain and green-pear prompt, while the first q5 apple prompt leans
mountain-only. This is still not mask/inpaint support; qwen masks remain
unsupported because the mflux qwen-edit reference exposes no mask path.

**Shared P0 blockers for the Flux/Qwen family** (z-image already cleared these via
its own native port): (1) safetensors→module key-map + `Module.update`, (2) T5-XXL
text encoder (shared Flux/Qwen/Wan), (3) CLIP-L (Flux1), (4) Flux 3-axis RoPE.

**mflux feature surface (scope = "mflux type, not just flux type"):**
- ✅ Quantized mflux packaging (`*-mflux-4bit`) load + decode — implemented for z-image.
- ✅ Image *edit* protocol (`ImageEditor`, `sourceImage`/`sourceImages`, `mask`/`strength`) — qwen-edit q4/q5 single-image and ordered multi-image text-image paths are live-proven; expose masks only for models with a real mask path.
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

## 7b. Quant matrix (live-proven 2026-06-16) + model-resolution fix
| Model | 4-bit | 5/6-bit | 8-bit | full |
|---|---|---|---|---|
| z-image-turbo | ✅ proven | — | ✅ proven | (not staged) |
| flux1-schnell | ✅ proven | — | ✅ proven | (not staged) |
| qwen-image | ✅ proven | ✅ 6-bit proven | public mflux 8-bit not found/staged | (not staged) |
| qwen-image-edit | ✅ q4 single/multi-image text edit proven; q3 incomplete; q4 weaker on shape change | ✅ q5 single/multi-image text edit proven; q6 incomplete | (not staged) | (not staged) |
| ideogram | ⬜ nf4 incomplete | ✅ staged fp8 mirror proven for typography + strict object-icon prompts; broader no-text apple prompt can hallucinate text | official access gated | (not staged) |
Proven rows are deterministic (same seed+prompt -> identical), prompt-sensitive,
and coherent. z-image-turbo and flux1-schnell 8-bit and 4-bit produce visibly
distinct images (genuine quant), ~3-4s/512px/4-step. qwen-image 4-bit and
6-bit are live-proven text-to-image rows; qwen-image 8-bit remains unproven
because no public mflux 8-bit bundle was found in the current HF search.
qwen-image-edit q4/q5 are live-proven single-image and ordered multi-image
text-image edit rows after the VL-grid conditioning fix; q3/q6 remain hidden
until local bundles are complete. Qwen masks remain unsupported unless upstream
mflux adds a real qwen mask path or a separate fill/inpaint model is wired.
Current q4 proof is deterministic and color-sensitive but weaker on
shape-changing green-pear prompts; q5 is the cleaner edit row. Staged Ideogram
fp8 is live-proven for typography and strict object-icon prompts, but a broader
current-a188 no-text apple prompt hallucinated text; official `ideogram-ai/*`
access and nf4 remain separate blockers.

Runtime-proof refresh after the latest main image pipeline baseline:
`vmlx-origin/main` `a188a2ccecc92c8a5993506acc83df16f83c7420` was rebuilt and
live-probed from
`/Users/eric/vmlx-swift-fluxwt`. Fresh artifacts:
`docs/local/vmlx-flux-probes/2026-06-16-current-a188-load-matrix/`,
`docs/local/vmlx-flux-probes/2026-06-16-current-a188-zimage-4bit-gen/`,
`docs/local/vmlx-flux-probes/2026-06-16-current-a188-zimage-8bit-gen/`,
`docs/local/vmlx-flux-probes/2026-06-16-current-a188-flux-schnell-4bit-gen/`,
`docs/local/vmlx-flux-probes/2026-06-16-current-a188-flux-schnell-8bit-gen/`,
`docs/local/vmlx-flux-probes/2026-06-16-current-a188-qwen-image-4bit-gen20/`,
`docs/local/vmlx-flux-probes/2026-06-16-current-a188-qwen-image-6bit-gen20/`,
`docs/local/vmlx-flux-probes/2026-06-16-current-a188-qwen-edit-q4-gen20/`,
`docs/local/vmlx-flux-probes/2026-06-16-current-a188-qwen-edit-q5-gen20/`, and
`docs/local/vmlx-flux-probes/2026-06-16-current-a188-ideogram-fp8-object-strict/`.
Contact sheets were viewed at
`docs/local/vmlx-flux-outputs/2026-06-16-current-a188-contact-sheet.png` and
`docs/local/vmlx-flux-outputs/2026-06-16-current-a188-ideogram-strict-sheet.png`.
Boundary artifact
`docs/local/vmlx-flux-probes/2026-06-16-current-a188-ideogram-fp8-object-determinism/`
hallucinated text on the broader no-text apple prompt.

**Model-resolution bug fixed:** `MLXStudioModelStore.resolve(name:)` normalized away the
`-Nbit` suffix, so requesting `...-8bit` collapsed onto a co-installed `...-4bit` dir
(loaded the wrong quant). Fixed by adding a literal case-insensitive directory-name match
FIRST. **osaurus must request the exact bundle directory name** (or canonical+quant) — if a
user has both 4-bit and 8-bit of a model installed, exact-name resolution is required.

**Tokenizer staging (flux/qwen):** mflux bundles ship SLOW tokenizers (CLIP vocab.json+
merges.txt; T5 spiece.model); swift-transformers needs `tokenizer.json`. Convert once via
`transformers` (`AutoTokenizer.from_pretrained(dir, use_fast=True).save_pretrained(dir)`) for
`tokenizer/` and `tokenizer_2/`. z-image ships tokenizer.json already. osaurus download/stage
must ensure tokenizer.json exists.

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

vmlxflux-probe --model Qwen-Image-Edit-mflux-q4 --edit \
  --source-image <png> --turn "make the background blue" \
  --artifacts <dir>
# Repeat --source-image for ordered multi-reference qwen edits; the last source
# image drives mflux sizing and all sources feed prompt/conditioning features.
# qwen-image-edit q4/q5 record completed edit turns and write PNGs; current
# proof includes same-seed deterministic repeat plus prompt-sensitive viewed
# outputs for single-image and multi-image rows.
# Add --mask-image <png> to prove the current unsupported-mask path: the engine
# loads the staged bundle then emits failed_event before the edit pipeline runs.

vmlxflux-probe --model Qwen-Image-Edit-mflux-q4 --qwen-edit-prompt \
  --source-image <png> --turn "make the background blue" \
  --artifacts <dir>
# Repeat --source-image to record per-image token counts and concatenated
# tokenizer image-pad expansion.

vmlxflux-probe --model Qwen-Image-Edit-mflux-q4 --qwen-edit-conditioning \
  --source-image <png> \
  --artifacts <dir>
# qwen-image-edit conditioning records the mflux-compatible VL-size VAE encode
# and packed static latents per source image (square source: 384x384 -> 24x24
# -> 576 tokens).

vmlxflux-probe --model Qwen-Image-Edit-mflux-q4 --qwen-edit-vision \
  --source-image <png> --turn "make the background blue" \
  --artifacts <dir>
# Repeat --source-image to record per-source Qwen2.5-VL grids/features and
# image-token splice into Qwen text embeddings.

vmlxflux-probe --model Qwen-Image-Edit-mflux-q4 --qwen-edit-denoise \
  --source-image <png> --width 256 --height 256 --steps 1 --seed 7 \
  --turn "make the background blue" \
  --artifacts <dir>
# qwen-image-edit denoise records one live edit-shaped transformer velocity
# forward and target velocity slice with target+one shape per conditioning image.
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

1. Wire the osaurus app/server bridge: `/v1/images/models`,
   `/v1/images/generations`, `/v1/images/edits`, progress SSE, output path
   policy, exact directory-name resolution, and the required `MetalGate`
   exclusion around the full stream drain.
2. Ideogram 4 follow-through: staged fp8 native generation is source-wired and
   live-proven for typography plus strict object-icon prompts. Keep broader
   object-renderer wording hidden until the no-text hallucination row is fixed,
   keep official `ideogram-ai/*` exposure gated until access requirements are
   resolved, and keep nf4 hidden until a complete bundle is staged and proven.
3. Qwen-Image-Edit follow-through: q4/q5 single-image and ordered multi-image
   text-image edit are live-proven after the VL-grid conditioning fix. Keep
   q3/q6 hidden/blocked until the local bundles are complete. Prefer q5 in UI
   examples; q4 is weaker on shape-changing edits. Keep qwen mask controls
   hidden unless upstream mflux adds a real qwen mask path or a separate
   fill/inpaint model is wired; do not fake masks with post-blends.
4. Full-precision z-image/flux-schnell: stage weights, run the same proof
   matrix pattern, and visually inspect outputs before promotion.
5. LoRA loader hook (`supportsLoRA`), img2img/controlnet conditioning.
6. `numImages > 1` batching; webp/jpeg writers; preview-decode cadence.
