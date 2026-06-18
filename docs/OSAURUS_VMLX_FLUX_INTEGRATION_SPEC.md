# vMLX-Flux (native mFLUX) — osaurus integration spec

**Audience:** osaurus engineers wiring native on-device image/video generation.
**Engine repo:** `jjang-ai/vmlx-flux` (standalone SwiftPM) — vendored into
`vmlx-swift/Libraries/vMLXFlux*` so the whole vMLX stack shares one MLX runtime.
**Status date:** 2026-06-18. **Owner:** Eric.

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
- **Per-model status:** Osaurus `vmlx-origin/main` has current scanner/load
  proof for all local image rows. 2026-06-18 stress proof:
  `docs/local/vmlx-flux-probes/20260618-image-stress2/current-proof-summary.json`
  reports `status=passed`, matrix `14/14` loaded, and deterministic
  repeat/prompt-sensitivity checks passing for `z-image-turbo` 4/8-bit,
  `flux1-schnell` 4/8-bit, `qwen-image` 4/6/8-bit,
  qwen-edit q4/q5/q6/q8, and staged `ideogram-4-fp8`/`ideogram-4-nf4`
  JSON-caption rows. Viewed
  proof-run contact sheet:
  `docs/local/vmlx-flux-outputs/20260618-image-stress2/current-proof-turn-contact-sheet.png`.
  The proof runner now raises the soft file-descriptor limit before the
  all-model matrix because loading all 14 heavyweight MLX bundles in one
  process can exceed macOS's default `ulimit -n 256`.
  qwen-edit q3 is loadable after staging `q3/text_encoder/3.safetensors`, but
  remains hidden because its viewed 20-step edit output is high-noise and not a
  clean prompt-following edit. qwen masks are intentionally hidden because the
  mflux qwen-edit reference has no qwen mask/inpaint argument or path.
  qwen-edit q5 and q8 are clean prompt-following edit rows in the 2026-06-18
  stress proof; q4 remains deterministic and prompt-sensitive but visibly noisy.
  q5 can still show a square table/composition patch on some source/edit
  prompts, so keep edit-composition cleanup on the follow-up list. Ideogram 4
  is implemented/testable for the staged fp8/NF4 mirrors when fed structured
  JSON captions. Plain prompts can reproduce mflux's warning/failure behavior
  as gray/text-card outputs; expose Ideogram as JSON-caption-first, not as a
  general plain-prompt object renderer.
  Official `ideogram-ai/*` downloads remain approval-gated for the current
  account (`hf download ... --dry-run` returned access denied for fp8 and nf4
  on 2026-06-16).
  UI/server teams should consume `docs/OSAURUS_IMAGE_UI_MANIFEST.json` as the
  machine-readable model/control/exposure/proof manifest; this integration spec
  remains the source for Swift engine semantics and concurrency requirements.
  `docs/OSAURUS_IMAGE_OPENAPI.json` is the route/schema contract for the
  `/v1/images/*` bridge, and `scripts/vmlx-image-openapi-manifest-check.sh`
  verifies the manifest/OpenAPI pair. See §6 and §7b.

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
| `ideogram` | Ideogram 4 | imageGen | 20 / 7 (fp8 or NF4 weights: ideogram-ai/ideogram-4-fp8, ideogram-ai/ideogram-4-nf4) |
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
`Qwen-Image-Edit-mflux` folder contains `q3`, `q4`, `q5`, `q6`, and `q8`;
osaurus should present/request the exact local variant IDs:
`Qwen-Image-Edit-mflux-q3`, `Qwen-Image-Edit-mflux-q4`,
`Qwen-Image-Edit-mflux-q5`, `Qwen-Image-Edit-mflux-q6`, and
`Qwen-Image-Edit-mflux-q8`. Current disk/proof state: q4/q5/q6/q8 are loadable
local bundles with live text-image edit proof; q3 is loadable after staging
`q3/text_encoder/3.safetensors`, but stays hidden because the viewed 20-step
edit output is high-noise and not a clean prompt-following edit. Earlier
qwen-edit-only staging matrix, superseded for current overall status by
`docs/local/vmlx-flux-probes/2026-06-16-current-ee9-status-load-matrix/compatibility-matrix.json`:
`docs/local/vmlx-flux-probes/2026-06-16-qwen-edit-q6q8-status-refresh-load-matrix/compatibility-matrix.json`
(13 scanned, 13 loaded). The current q4
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
Viewed output is coherent and prompt-sensitive. Current q6 same-seed edit proof:
`docs/local/vmlx-flux-probes/2026-06-16-qwen-edit-q6-after-download-gen20/Qwen-Image-Edit-mflux-q6-load.json`
(`load_status=loaded`; blue apple/repeat SHA
`475cdbc7e3066d74c646245cdb99d23c52fcaca070376447bdfee3d295a97330`; green
pear SHA `d4a05f4e424f1679441bce643be16dabbdb425efb0c866d91415efce17f96271`;
viewed clean). Current q8 same-seed edit proof:
`docs/local/vmlx-flux-probes/2026-06-16-qwen-edit-q8-after-download-gen20/Qwen-Image-Edit-mflux-q8-load.json`
(`load_status=loaded`; blue apple/repeat SHA
`72862df44d35d2db7e386cde402dd8c602d48c0e6d07ba55d4839ae1e438b743`; green
pear SHA `9eb6868adb9e3d0f601bbca537522a39a672051a3938a3df17995a2d52fe8678`;
viewed clean). Current q3 boundary proof:
`docs/local/vmlx-flux-probes/2026-06-16-qwen-edit-q3-after-shard-gen20/Qwen-Image-Edit-mflux-q3-load.json`
(`load_status=loaded`; blue apple/repeat SHA
`bbc6e873ae7aeab37bdcb5943ae843044fa990f44bf6478394fae87899ce296f`; green
pear SHA `f2ae4e292a7a5432ef64d07f048a4589a17a7af6d61190fa92475ae17cd223c9`;
viewed high-noise, so keep q3 hidden).

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
| **z-image-turbo** | `native_pipeline_implemented` | Full native port: Qwen-style text encoder, patchify+caption-concat DiT, real `AutoencoderKL` VAE decode, real 4/8-bit weight decode, PNG out. 2026-06-18 stress proof covers 4-bit and 8-bit at `docs/local/vmlx-flux-probes/20260618-image-stress2/current-proof-summary.json`; both rows passed 3-turn completion, deterministic repeat, prompt sensitivity, and viewed coherent apple/mountain outputs. | 1024px tuning; full precision not staged/proven. |
| **qwen-image** | `native_pipeline_implemented` | Full native pipeline `QwenImageNative.swift`: Qwen2.5 LM text encoder (GQA), 60-layer MM-DiT, 3D causal-conv VAE, mflux guidance-rescaled CFG. 2026-06-18 stress proof covers 4-bit and 8-bit at `docs/local/vmlx-flux-probes/20260618-image-stress2/current-proof-summary.json`; both rows passed 3-turn completion, deterministic repeat, prompt sensitivity, and viewed coherent apple/mountain outputs. Retained 6-bit proof remains from the earlier 103be artifact. | Full not staged/proven. Current fixes include mflux-compatible Qwen scheduler sigmas, keyed seed noise, profiled RGB reads for edit conditioning parity, VAE channels-last handling, raw-sigma timestep use, and quantized text-encoder embedding/linear loading. |
| qwen-image-edit | `native_pipeline_implemented` for `Qwen-Image-Edit-mflux-q4`, `Qwen-Image-Edit-mflux-q5`, `Qwen-Image-Edit-mflux-q6`, and `Qwen-Image-Edit-mflux-q8`; `native_pipeline_partial` for q3 | q4/q5/q6/q8 scan as loadable local bundles. 2026-06-18 stress proof covers q4/q5/q8 at `docs/local/vmlx-flux-probes/20260618-image-stress2/current-proof-summary.json`; all three passed completion, deterministic repeat, and prompt sensitivity. q5/q8 viewed as clean prompt-following blue-apple/green-pear edits; q4 remains visibly noisy but deterministic/prompt-sensitive. q6 remains clean from the earlier staged-shard artifact. q3 loads after staging `q3/text_encoder/3.safetensors`, but viewed output is high-noise, so it stays hidden. Non-null qwen masks are rejected before the edit pipeline loads and covered by `QwenImageEditSupportTests.testQwenImageEditRejectsMaskBeforePipelineLoad`. | q3 should stay hidden until a real coherent edit row exists; qwen mask/inpaint is unsupported by the current mflux reference; q5 can show a square table/composition patch on some source/edit prompts; broader Osaurus HTTP/UI production matrix still pending. |
| flux2-klein / flux2-klein-edit | `not_implemented` | Bundle scans + loads; `FluxDiTConfig.flux2Klein` preset exists. | T5 (single-encoder) port + weight key-map + 3-axis RoPE. |
| **flux1-schnell** | `native_pipeline_implemented` | Full native pipeline `Flux1Native.swift`: T5-XXL + CLIP-L encoders, full DiT, AutoencoderKL VAE, mflux decode. 2026-06-18 stress proof covers 4-bit and 8-bit at `docs/local/vmlx-flux-probes/20260618-image-stress2/current-proof-summary.json`; both rows passed completion, deterministic repeat, prompt sensitivity, and viewed coherent apple/mountain outputs. | tokenizer.json must be staged (mflux ships slow tokenizers; convert; see port plan). Full precision pending. |
| flux1-dev/kontext/fill | `not_implemented` | dev = schnell + guidance embedder (small add); kontext/fill = edit variants. | wire guidance + edit conditioning on the working schnell pipeline. |
| **ideogram** (Ideogram 4) | `native_pipeline_implemented` for staged fp8 and NF4 mirrors | Strong text/typography renderer. Complete mirror bundles are staged at `~/.mlxstudio/models/image/ideogram-4-fp8` and `~/.mlxstudio/models/image/ideogram-4-nf4`. Native source runs Qwen3 text encoder, conditional and unconditional 34-layer MM-DiT, fp8 `weight_scale` linears or bitsandbytes NF4 linears, mflux default 20-step guidance schedule, Flux2 VAE decode, and PNG output. 2026-06-18 stress proof covers fp8/NF4 structured JSON-caption apple and mountain rows at `docs/local/vmlx-flux-probes/20260618-image-stress2/current-proof-summary.json`; both passed completion, deterministic repeat, prompt sensitivity, and viewed coherent outputs. | Expose staged fp8 and NF4 mirrors as JSON-caption-first. Plain prompts can reproduce mflux warning/failure behavior as gray/text-card outputs. Official `ideogram-ai/*` downloads remain approval-gated for the current account. Broader Osaurus HTTP/UI production matrix still pending. |
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
- ✅ Image *edit* protocol (`ImageEditor`, `sourceImage`/`sourceImages`, `mask`/`strength`) — qwen-edit q4/q5/q6/q8 single-image and ordered multi-image text-image paths are live-proven where listed; expose masks only for models with a real mask path.
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

## 7b. Quant matrix (live-proven 2026-06-18) + model-resolution fix
| Model | 4-bit | 5/6-bit | 8-bit | full |
|---|---|---|---|---|
| z-image-turbo | 20260618 stress proven | - | 20260618 stress proven | (not staged) |
| flux1-schnell | 20260618 stress proven | - | 20260618 stress proven | (not staged) |
| qwen-image | 20260618 stress proven | retained 103be 6-bit proof | 20260618 stress proven | (not staged) |
| qwen-image-edit | 20260618 q4 proven but visibly noisy; q3 loadable but hidden due high-noise output | 20260618 q5 proven clean with square-patch caveat; retained ed84 q6 clean proof | 20260618 q8 proven clean | (not staged) |
| ideogram | 20260618 staged NF4 JSON-caption proof | 20260618 staged fp8 JSON-caption proof | official access gated | (not staged) |
2026-06-18 refreshed rows are deterministic (same seed+prompt -> identical),
prompt-sensitive, and coherent in the recorded artifacts. Proof summary:
`docs/local/vmlx-flux-probes/20260618-image-stress2/current-proof-summary.json`.
Viewed contact sheet:
`docs/local/vmlx-flux-outputs/20260618-image-stress2/current-proof-turn-contact-sheet.png`.
qwen-image 8-bit remains staged from `AbstractFramework/qwen-image-8bit` as
`qwen-image-mflux-8bit`. qwen-image-edit q5/q8 are clean on the blue-apple and
green-pear rows; q4 remains visibly noisier. q3 loads but remains hidden because
viewed q3 output is high-noise and not a clean prompt-following edit. Qwen masks
remain unsupported unless upstream mflux adds a real qwen mask path or a separate
fill/inpaint model is wired. Ideogram rows are JSON-caption proof rows; plain
prompts can still reproduce mflux warning/failure behavior as gray/text-card
outputs, so UI/API wording must be JSON-caption-first. Official `ideogram-ai/*`
access remains a separate blocker if official bundles are required.

Retained historical runtime-proof refresh after the earlier main image pipeline baselines:
`vmlx-origin/main` `103be4375d88f7f8249b39853feedcf390d41465` was rebuilt and
live-probed from `/Users/eric/vmlx-swift-fluxwt`. Historical 103be artifacts:
`docs/local/vmlx-flux-probes/2026-06-16-current-103be-load-matrix/`,
`docs/local/vmlx-flux-probes/2026-06-16-current-103be-zimage-4bit-gen/`,
`docs/local/vmlx-flux-probes/2026-06-16-current-103be-zimage-8bit-gen/`,
`docs/local/vmlx-flux-probes/2026-06-16-current-103be-flux-schnell-4bit-gen/`,
`docs/local/vmlx-flux-probes/2026-06-16-current-103be-flux-schnell-8bit-gen/`,
`docs/local/vmlx-flux-probes/2026-06-16-current-103be-qwen-image-4bit-gen20/`,
`docs/local/vmlx-flux-probes/2026-06-16-current-103be-qwen-image-6bit-gen20/`,
`docs/local/vmlx-flux-probes/2026-06-16-current-103be-qwen-edit-q4-gen20/`,
`docs/local/vmlx-flux-probes/2026-06-16-current-103be-qwen-edit-q5-gen20/`,
`docs/local/vmlx-flux-probes/2026-06-16-qwen-edit-q6-after-download-gen20/`,
`docs/local/vmlx-flux-probes/2026-06-16-qwen-edit-q8-after-download-gen20/`,
`docs/local/vmlx-flux-probes/2026-06-16-current-103be-ideogram-fp8-bb4-exact/`,
and
`docs/local/vmlx-flux-probes/2026-06-16-current-103be-ideogram-nf4-bb4-exact/`.
Viewed contact sheet:
`docs/local/vmlx-flux-outputs/2026-06-16-current-103be-contact-sheet.png`.
Ideogram boundary contact sheet:
`docs/local/vmlx-flux-outputs/2026-06-16-current-103be-ideogram-exact-and-boundary-sheet.png`.
Historical 5c7 artifacts retained for comparison:
`docs/local/vmlx-flux-probes/2026-06-16-current-5c7-load-matrix/`,
`docs/local/vmlx-flux-probes/2026-06-16-current-5c7-zimage-4bit-gen/`,
`docs/local/vmlx-flux-probes/2026-06-16-current-5c7-zimage-8bit-gen/`,
`docs/local/vmlx-flux-probes/2026-06-16-current-5c7-flux-schnell-4bit-gen/`,
`docs/local/vmlx-flux-probes/2026-06-16-current-5c7-flux-schnell-8bit-gen/`,
`docs/local/vmlx-flux-probes/2026-06-16-current-5c7-qwen-image-4bit-gen20/`,
`docs/local/vmlx-flux-probes/2026-06-16-current-5c7-qwen-edit-q5-gen20/`, and
`docs/local/vmlx-flux-probes/2026-06-16-ideogram-nf4-strict-object/`.
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
2. Use `docs/OSAURUS_IMAGE_OPENAPI.json` as the route/schema source and
   `docs/OSAURUS_IMAGE_UI_MANIFEST.json` as the live model/control exposure
   source. Run `scripts/vmlx-image-openapi-manifest-check.sh` after edits; set
   `VMLX_REQUIRE_LOCAL_PROOF=1` when the local `docs/local` proof artifacts are
   present and you want proof-path/SHA verification too.
3. `generateImage(prompt:model:settings:)`: acquire MetalGate (§7) → ensure the
   requested model is loaded (`load(name:from:)`, load-if-different) → build
   `ImageGenRequest` from osaurus settings → drain the stream → translate
   `ImageGenEvent` → osaurus's UI event type → return final URL → release gate.
4. `editImage(...)` / `upscaleImage(...)`: same, via `edit` / `upscale`.
5. Fan events to the UI via a per-job bridge so the chat surface can subscribe
   without holding a direct flux reference.
6. Event-shape translation: vmlx-flux emits separate `.step` and `.preview`; if
   osaurus uses a unified `.step(step:total:preview:)`, coalesce them in the bridge.

---

## 10. Probe tool (verification harness)

Use `scripts/vmlx-image-current-proof.sh` to refresh the current source-side
load/generation proof set in one run. It builds `vmlxflux-probe`, runs the
load-only matrix plus z-image-turbo 4/8-bit, flux-schnell 4/8-bit,
qwen-image 4/8-bit, qwen-image-edit q4/q8, and staged Ideogram fp8/NF4
same-seed proof rows, writes repo-local ignored proof artifacts, and fails if
a row does not complete or the SHA repeat/sensitivity checks fail. The script
does not replace visual inspection or Osaurus HTTP/UI proof.

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
# qwen-image-edit q4/q5/q6/q8 record completed edit turns and write PNGs;
# current proof includes same-seed deterministic repeat plus prompt-sensitive
# viewed outputs for single-image rows. q4/q5 also retain multi-image proof.
# q3 loads but stays hidden because viewed output is high-noise.
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
   exclusion around the full stream drain. Use `docs/OSAURUS_IMAGE_OPENAPI.json`
   for route/schema wiring and run
   `scripts/vmlx-image-openapi-manifest-check.sh` after any manifest/API edits.
2. Ideogram 4 follow-through: staged fp8 and NF4 native generation are
   source-wired. fp8 is live-proven for typography plus strict object-icon
   prompts; NF4 is live-proven for strict object-icon prompts. Keep broader
   object-renderer wording hidden until the no-text hallucination row is fixed,
   and keep official `ideogram-ai/*` exposure gated until access requirements
   are resolved.
3. Qwen-Image-Edit follow-through: q4/q5/q6/q8 single-image text-image edit
   rows are live-proven after the VL-grid conditioning fix; q4/q5 ordered
   multi-image proof is retained. Prefer q5/q6/q8 in UI examples; q4 is weaker
   on shape-changing edits. Keep q3 hidden until a clean visual edit row exists.
   Keep qwen mask controls hidden unless upstream mflux adds a real qwen mask
   path or a separate fill/inpaint model is wired; do not fake masks with
   post-blends.
4. Full-precision z-image/flux-schnell: stage weights, run the same proof
   matrix pattern, and visually inspect outputs before promotion.
5. LoRA loader hook (`supportsLoRA`), img2img/controlnet conditioning.
6. `numImages > 1` batching; webp/jpeg writers; preview-decode cadence.
