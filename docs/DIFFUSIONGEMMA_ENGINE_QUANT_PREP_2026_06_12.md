# DiffusionGemma Engine + Quant Prep

Date: 2026-06-12

## Source

Target source bundle:

- HF repo: `google/diffusiongemma-26B-A4B-it`
- Local path: `/Users/eric/models/google/diffusiongemma-26B-A4B-it`
- Downloaded size: 48 GB on disk, 20 files, 11 safetensors shards
- Architecture: `DiffusionGemmaForBlockDiffusion`
- `model_type`: `diffusion_gemma`
- Text config: `diffusion_gemma_text`, 30 layers, 128 experts, top-8 experts, 1024-token sliding window
- Canvas: 256 tokens
- Generation config: `max_new_tokens=256`, `max_denoising_steps=48`, entropy-bound sampler, `confidence_threshold=0.005`, `stability_threshold=1`

This is not a normal autoregressive Gemma 4 decode row. The runtime has an
encoder/prompt KV path plus a bidirectional denoising canvas. A plain
`TokenIterator` next-token loop is the wrong engine.

## Modality Gate

The bundle has a real VL path:

- `vision_config.model_type = gemma4_vision`
- `image_token_id = 258880`
- `vision_soft_tokens_per_image = 280`

The bundle does not expose a usable audio path:

- `audio_config = null`
- `audio_token_id = null`
- MLX-VLM's DiffusionGemma processor reference rejects audio inputs

Do not mark audio supported for this model unless a later source bundle adds an
audio config/token and the vMLX runtime proves real audio payload generation.
The immediate modality gate is text + image/VL. `processor_config.json` has a
video processor object, but `config.json` has no `video_token_id`; video remains
unproven until processor/token/runtime evidence is present and a real video
payload row passes.

## Engine Work

Required vMLX work before speed or release claims:

1. Add a `diffusion_gemma` model family instead of aliasing to Gemma 4 AR.
2. Port the block-diffusion generation loop:
   - prompt encoder prefill and cache,
   - 256-token canvas initialization,
   - decoder forward with bidirectional canvas attention,
   - self-conditioning logits,
   - entropy-bound accept/renoise,
   - adaptive stopping,
   - append finalized canvas to the encoder cache.
3. Reuse Gemma 4 MoE and proportional/full/sliding RoPE pieces where tensor
   names and config shape match.
4. Wire the Gemma 4 vision tower/image soft-token path before claiming VL.
5. Add tests that fail if `diffusion_gemma` silently routes through AR decode.

Useful upstream references:

- HF Transformers: `src/transformers/models/diffusion_gemma/generation_diffusion_gemma.py`
- HF Transformers: `src/transformers/models/diffusion_gemma/modeling_diffusion_gemma.py`
- MLX-VLM: `mlx_vlm/models/diffusion_gemma/`

## First-Party Quant Plan

The required output targets are first-party bundles:

- `/Users/eric/models/OsaurusAI/diffusiongemma-26B-A4B-it-MXFP4`
- `/Users/eric/models/OsaurusAI/diffusiongemma-26B-A4B-it-MXFP8`

`mlx-community/diffusiongemma-26B-A4B-it-4bit` is useful only as metadata
comparison. It is not the deliverable.

Converter:

- `scripts/vmlx-convert-diffusiongemma-mxfp.py`
- Uses native MLX MXFP calls: `mx.quantize(..., mode="mxfp4")` or
  `mx.quantize(..., mode="mxfp8")`
- Emits MLX `weight` / `scales` companions. It does not emit old affine 4-bit
  `biases` as the primary format.

Current quant policy:

- MXFP4 profile: decoder attention and MoE experts use MXFP4; dense MLP and
  router projections use MXFP8 overrides.
- MXFP8 profile: decoder attention, dense MLP, router projections, and MoE
  experts use MXFP8.
- Keep norms, layer scalars, router control scalars, embeddings, multimodal
  projector/vision path, and self-conditioning path in fp16/bf16 until parity is
  proven.
- Use `group_size=32` to match current vMLX MXFP loader expectations.
- Stamp `weight_format=mxfp4` / `weight_format=mxfp8` and
  `quantization.mode=mxfp4` / `mxfp8` in generated configs.

Run the deterministic prep once the BF16 source download is complete:

```sh
scripts/vmlx-diffusiongemma-quant-prep.sh \
  --src /Users/eric/models/google/diffusiongemma-26B-A4B-it \
  --out-root /Users/eric/models/OsaurusAI
```

This writes manifests under `docs/local/diffusiongemma-quant-prep/` and fails
closed if the source is incomplete.

Run dry-runs:

```sh
python3 scripts/vmlx-convert-diffusiongemma-mxfp.py \
  --src /Users/eric/models/google/diffusiongemma-26B-A4B-it \
  --bits 4 \
  --dry-run \
  --quiet

python3 scripts/vmlx-convert-diffusiongemma-mxfp.py \
  --src /Users/eric/models/google/diffusiongemma-26B-A4B-it \
  --bits 8 \
  --dry-run \
  --quiet
```

Dry-run proof from the downloaded source:

- Source tensors: 1,047
- MXFP4 profile: 175 tensors MXFP4, 120 tensors MXFP8, 752 tensors fp16
  passthrough
- MXFP8 profile: 295 tensors MXFP8, 752 tensors fp16 passthrough
- Quantized source-weight mass: about 45.62 GiB
- Passthrough source-weight mass: about 2.48 GiB

Run actual conversion with the JANG MLX venv:

```sh
/Users/eric/jang/jang-tools/.venv/bin/python \
  scripts/vmlx-convert-diffusiongemma-mxfp.py \
  --src /Users/eric/models/google/diffusiongemma-26B-A4B-it \
  --out /Users/eric/models/OsaurusAI/diffusiongemma-26B-A4B-it-MXFP4 \
  --bits 4 \
  --group-size 32 \
  --replace

/Users/eric/jang/jang-tools/.venv/bin/python \
  scripts/vmlx-convert-diffusiongemma-mxfp.py \
  --src /Users/eric/models/google/diffusiongemma-26B-A4B-it \
  --out /Users/eric/models/OsaurusAI/diffusiongemma-26B-A4B-it-MXFP8 \
  --bits 8 \
  --group-size 32 \
  --replace
```

Actual conversion proof with the JANG MLX venv:

- MXFP4 output:
  `/Users/eric/models/OsaurusAI/diffusiongemma-26B-A4B-it-MXFP4`
- MXFP4 log: `/tmp/diffusiongemma-mxfp4-convert-20260612T191053Z.log`
- MXFP4 result: 1,047/1,047 source tensors processed, 295 quantized tensors,
  752 passthrough tensors, 0 skipped tensors, 1,342 indexed output tensors,
  15 shards, 15 GB on disk, `total_size=15944852880`
- MXFP4 timing: 74.38 seconds wall, 14,044,954,624-byte max RSS, 0 swaps
- MXFP8 output:
  `/Users/eric/models/OsaurusAI/diffusiongemma-26B-A4B-it-MXFP8`
- MXFP8 log: `/tmp/diffusiongemma-mxfp8-convert-20260612T191232Z.log`
- MXFP8 result: 1,047/1,047 source tensors processed, 295 quantized tensors,
  752 passthrough tensors, 0 skipped tensors, 1,342 indexed output tensors,
  23 shards, 26 GB on disk, `total_size=27918936056`
- MXFP8 timing: 65.80 seconds wall, 14,894,694,400-byte max RSS, 0 swaps

Final bundle verification passed for both MXFP4 and MXFP8:

- `config.json`, `jang_config.json`, `model.safetensors.index.json`, and
  `diffusiongemma_mxfp_manifest.json` agree on `weight_format`, quant mode,
  quant bits, `group_size=32`, shard count, indexed tensor count, and total
  size.
- Every index entry resolves to an existing shard and every indexed tensor key
  is present in its safetensors shard.
- Modality metadata remains `image=true`, `video=true`, `audio=false`.
- Sampled tensor layout:
  - MXFP4 attention: `q_proj.weight (4096, 352) uint32`,
    `q_proj.scales (4096, 88) uint8`
  - MXFP4 dense MLP override:
    `mlp.down_proj.weight (2816, 528) uint32`,
    `mlp.down_proj.scales (2816, 66) uint8`
  - MXFP4 experts: `experts.down_proj.weight (128, 2816, 88) uint32`,
    `experts.down_proj.scales (128, 2816, 22) uint8`
  - MXFP8 attention: `q_proj.weight (4096, 704) uint32`,
    `q_proj.scales (4096, 88) uint8`
  - MXFP8 dense MLP: `mlp.down_proj.weight (2816, 528) uint32`,
    `mlp.down_proj.scales (2816, 66) uint8`
  - MXFP8 experts: `experts.down_proj.weight (128, 2816, 176) uint32`,
    `experts.down_proj.scales (128, 2816, 22) uint8`
  - Embeddings stay passthrough:
    `model.decoder.embed_tokens.weight (262144, 2816) float16`

Synthetic conversion proof also passed with the JANG MLX venv:

- output `weight_format=mxfp4`
- attention/expert tensors emitted `weight` + `scales`
- dense MLP in the MXFP4 profile emitted `mode=mxfp8` per-layer override
- vision/self-conditioning passthrough layers were stamped as skip overrides

## Current Boundary

The source download and manifest prep do not make the runtime working. The
engine remains `PARTIAL` until native block-diffusion text generation works,
then `PARTIAL` until image/VL payloads work, and only then should MXFP4/MXFP8
speed numbers be trusted.
