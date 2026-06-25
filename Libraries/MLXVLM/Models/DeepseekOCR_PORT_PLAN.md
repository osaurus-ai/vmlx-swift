# DeepSeek-OCR / Unlimited-OCR port to vmlx-swift (MLXVLM)

Port of https://github.com/Blaizzy/mlx-vlm/tree/main/mlx_vlm/models/deepseekocr
Covers **both** `deepseek-ai/DeepSeek-OCR` and `baidu/Unlimited-OCR` (identical
arch: `DeepseekOCRForCausalLM`, top `model_type: deepseek_vl_v2`). Community MLX
weights already exist (`sahilchachra/unlimited-ocr-8bit-mlx`, `LoJexLLM/Unlimited-OCR-MLX`).

## Architecture (verified against deepseek-ai/DeepSeek-OCR/config.json + weights)
- **DeepEncoder** (the novel part): two parallel vision towers over a 1024×1024 image
  - `sam_model`: SAM-ViT-B — width 768, 12 layers, heads 12, patch 16, window 14,
    global-attn layers {2,5,8,11}, neck downsample → channels [512,1024]. Weights:
    `model.sam_model.blocks.*` (168), `model.sam_model.neck.*` (6), patch_embed/pos.
  - `vision_model`: CLIP-L/14-224 — width 1024, 24 layers, heads 16, patch 14.
    Weights: `model.vision_model.transformer.*` (288). Takes `patch_embeds=SAM output`.
  - feature concat: `concat(clip[:,1:], sam.flatten(1,2), axis=-1)` → input_dim 2048.
- **projector** (`model.projector`): linear 2048 → n_embed 1280 (projector_type=linear).
- **language_model**: DeepSeek-V2 MoE decoder, **standard attention (use_mla=false,
  qk_nope=0 ⇒ LlamaAttention path)** — hidden 1280, 12 layers, heads 10, head_dim 128,
  vocab 129280, n_routed_experts 64, n_shared_experts 2, num_experts_per_tok 6,
  moe_intermediate 896, first_k_dense_replace 1 (layer 0 dense, 1-11 MoE),
  rope_theta 10000, topk_method greedy, scoring softmax. Weights `model.layers.*`,
  `model.embed_tokens`, `model.norm`, `lm_head`. Mirrors existing DeepseekV3.swift
  WITHOUT the MLA branch.
- **2D tiling**: `image_newline` + `view_separator` learned embeddings (n_embed). Global
  view (always) + local crop views (Gundam/"unlimited" mode for big docs); rows joined
  with `image_newline`, views joined with `view_separator`. tile_tag "2D",
  global_view_pos "head". image_token_index 128815.

## File plan (MLXVLM globs Models/, so new files auto-compile)
- `DeepseekOCR.swift` — config structs (port config.py), `MlpProjector`, top `Model`
  (port deepseekocr.py get_input_embeddings + sanitize), factory glue. [lead]
- `DeepseekOCRLanguage.swift` — DeepSeek-V2 MoE decoder (port language.py; DeepseekV3.swift idiom).
- `DeepseekOCRSAM.swift` — SAM-ViT-B encoder + neck (port sam.py).
- `DeepseekOCRVision.swift` — CLIP vision tower + DeepEncoder glue (port vision.py).
- processor — image preprocessing/tiling (port processing_deepseekocr.py) into the
  vmlx `UserInputProcessor` pattern (see GlmOcr.swift / Qwen3VL.swift).

## sanitize (from deepseekocr.py transform_key)
`model.layers→language_model.model.layers`, `model.embed_tokens→language_model.model.embed_tokens`,
`model.norm→language_model.model.norm`, `model.vision_model→vision_model`,
`model.sam_model→sam_model`, `model.projector→projector`,
`model.view_seperator→view_separator` (sic), `model.image_newline→image_newline`,
`lm_head.weight→language_model.lm_head.weight`.

## Registration
VLMModelFactory `_creators`: `"deepseek_vl_v2": create(DeepseekOCRConfiguration.self, DeepseekOCR.init)`
(+ alias `"deepseekocr"` if any pack uses it). Processor registry: DeepseekOCRProcessor.
osaurus: add to VLM family detection + ModelFamilyNames + media capabilities.

## Verification (correctness gate — behavioral, not eyeballed)
Install mlx-vlm Python, run DeepSeek-OCR on a known test image (`<image>\nFree OCR.`),
capture reference OCR text. Swift port MUST reproduce the same text on the dev app.
RAM: model ~6.7GB bf16 / smaller quant; single resident, ram_feasibility gated.

## Status
- [x] branches off main (osaurus 161e6ca5, vmlx b68502a), model downloaded, arch verified
- [ ] config + scaffold / language / sam / vision / processor / top-model glue
- [ ] build, behavioral verify vs mlx-vlm, live OCR on dev app, PRs
