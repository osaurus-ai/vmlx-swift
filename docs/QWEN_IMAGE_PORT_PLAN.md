# Qwen-Image / Qwen-Image-Edit native port plan (vMLXFlux)

**Purpose:** a concrete, executable plan to take `qwen-image` and `qwen-image-edit`
from `notImplemented` to a proven native pipeline, using the **already-proven
`ZImageNative.swift`** as the reference template. For osaurus teammates + future
porting sessions.

**Status (2026-06-15):** registered, scans/loads as a local mflux bundle, but
`generate`/`edit` throw `FluxError.notImplemented`. The `qwen-image-mflux-4bit`
(25 GB) bundle is NOT currently on disk — staging it onto the internal SSD
(`~/.mlxstudio/models/image/`, per the GPU-watchdog rule) is **step 0** and is
required before any live proof.

> Do NOT mark this done without a live same-seed/different-prompt proof (the HARD
> RULE). The Z-Image proof in `OSAURUS_VMLX_FLUX_INTEGRATION_SPEC.md` §11 is the bar.

---

## Why Z-Image is the right template

Qwen-Image and Z-Image share the modern text-to-image recipe that
`vMLXFluxKit` + `ZImageNative.swift` already implement end-to-end:

- A **decoder-LM text encoder** (Z-Image: Qwen-style 2560-dim encoder; Qwen-Image:
  Qwen2.5-VL text tower) producing per-token hidden states as conditioning —
  `ZImageTextEncoder` is a near-drop-in pattern (embed_tokens → N× {RMSNorm,
  q/k/v/o_proj with q/k norm, RoPE attention, gated MLP}).
- An **MM-DiT** transformer that concatenates caption + image streams with
  RoPE position ids and adaLN-zero timestep modulation — `ZImageTransformer`
  (patchify + caption-concat + noise/context refiners + unified blocks +
  final layer + unpatchify) is the template.
- An **AutoencoderKL** VAE decoder — `ZImageVAEDecoder` (conv_in → mid_block →
  up_blocks → conv_out) is reusable; Qwen-Image's VAE has the same family shape.
- The **FlowMatch Euler scheduler**, **mflux 4-bit weight decode** (scale
  tensors), **tokenizer bridge** (`AutoTokenizer.from(modelFolder:)`), and
  **PNG IO** are all model-agnostic in `vMLXFluxKit` and already used by Z-Image.

So the port is mostly: (a) parse Qwen-Image's config, (b) map its checkpoint keys
to module properties, (c) match its exact text-encoder + DiT topology, (d) wire
image conditioning for `-edit`.

---

## Step-by-step

### 0. Stage weights + capture ground truth
- Copy `qwen-image-mflux-4bit` to `~/.mlxstudio/models/image/` (internal SSD).
- Run the **reference mflux (Python)** generation once with a fixed seed + prompt
  to get a ground-truth image + the intermediate shapes (text-encoder output dim,
  latent channels, patch size, VAE scale/shift). These are the oracle for the port.
- `vmlxflux-probe --model qwen-image-mflux-4bit --json` (scan only) to confirm the
  component layout (`transformer/`, `text_encoder/`, `vae/`, `tokenizer/`).

### 1. Config parse
- Read `transformer/config.json` + `text_encoder/config.json` + `vae/config.json`.
- Pull: hidden dim, num layers, num heads, head dim, patch size, in-channels,
  text hidden dim, RoPE theta, VAE scale/shift. Mirror the `ZImageNative` static
  constants block with Qwen-Image's numbers (do NOT reuse Z-Image's 3840/2560/30).

### 2. Weight key-map (the hard part)
- Enumerate `transformer/*.safetensors` keys (use `WeightLoader` which already
  merges component shards into `componentWeights["transformer"|"text_encoder"|"vae"]`).
- Build a `QwenImageWeightStore` modeled on `ZImageWeightStore` with the
  candidate-key fallbacks for Qwen-Image's naming (Qwen-Image uses
  `transformer_blocks.{i}.attn.*`, `img_mod`/`txt_mod`, `img_mlp`/`txt_mlp` —
  verify against the actual checkpoint, do not assume).
- For 4-bit mflux: reuse `ZImageWeightStore.linear(component:prefix:...)`'s
  scale-tensor dequant path (Qwen-Image mflux-4bit packs the same way).

### 3. Text encoder
- Qwen-Image uses a Qwen2.5-VL text tower. Start from `ZImageTextEncoder`; adjust
  dim/layers/heads, the prompt template (Qwen-Image has its own chat-style
  template — capture it from the reference), and the pooled vs per-token output
  contract. Feed per-token hidden states as `capFeats`.

### 4. Transformer (MM-DiT)
- Clone `ZImageTransformer`; match Qwen-Image's block structure (it has dual
  img/txt modulation per block — closer to Flux's double-stream than Z-Image's
  unified blocks, so cross-reference `FluxDiT.swift`'s `FluxDoubleStreamBlock`).
- 3-axis RoPE: confirm whether Qwen-Image uses Z-Image-style 1-grid RoPE or
  Flux-style 3-axis (time,H,W) — this is a known open TODO in `FluxDiT.swift`.

### 5. VAE
- Reuse `ZImageVAEDecoder`; verify channel progression + scale/shift against
  `vae/config.json` (Qwen-Image VAE scale/shift differ from Z-Image's 0.3611/0.1159).

### 6. `qwen-image-edit` image conditioning
- Implement `ImageEditor.edit(_:)`: VAE-**encode** the source image to a latent,
  blend with noise per `strength`, optionally apply the mask (white=edit), then
  run the same denoise loop. Needs the VAE **encoder** (the kit only has the
  decoder today — add `ZImageVAEEncoder`-style conv stack or reuse Qwen-Image's).

### 7. Live proof (gate — do not skip)
- `vmlxflux-probe --model qwen-image-mflux-4bit --generate --seed S --steps N
  --turn "A" --turn "B" --turn "A"` → turns 1≡3 byte-identical (determinism),
  turn 2 ≠ 1 (prompt-sensitivity), all coherent + prompt-accurate.
- For `-edit`: prove the output preserves source structure and applies the prompt
  edit (e.g. recolor) — compare against the masked region.
- Compare against the step-0 reference image for fidelity.

---

## Effort + risk

- **Biggest risk:** the weight key-map (step 2) and the exact text-encoder template
  (step 3) — wrong keys/topology produce coherent-looking but prompt-insensitive or
  garbled output. The same-seed/different-prompt + reference-image checks catch this.
- **Reuse ratio:** ~70% of the machinery (scheduler, VAE decoder pattern, weight
  loader, tokenizer, IO, mflux dequant, patchify/RoPE/adaLN scaffolding) is shared.
- **Shared payoff:** the T5-XXL encoder needed for `flux1-*`/`flux2-klein` is a
  separate port; Qwen-Image does NOT need T5 (it uses Qwen2.5-VL), so Qwen-Image is
  the better *next* target than the Flux family.

## Reference files
- Template: `Libraries/vMLXFluxModels/ZImage/ZImageNative.swift` (proven).
- Flux double-stream block (for MM-DiT): `Libraries/vMLXFluxKit/FluxDiT.swift`.
- Shared kit: `FlowMatchScheduler`, `VAE`, `WeightLoader`, `MathOps`, `LatentSpace`.
- API contract + proof bar: `docs/OSAURUS_VMLX_FLUX_INTEGRATION_SPEC.md`.
