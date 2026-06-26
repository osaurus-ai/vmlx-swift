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

## STATUS 2026-06-25 — engine compiles; CRITICAL correctness finding

### Done
- All 6 Swift files written; `swift build --target MLXVLM` is GREEN (commit feccec2).
- osaurus repinned to feccec2; dev-app build in progress.
- Ground-truth OCR captured from OFFICIAL PyTorch (baidu/Unlimited-OCR, torch+transformers `model.infer`, trust_remote_code), saved `/tmp/ocr_reference_output.txt`:
  - test img1 → "OSAURUS OCR TEST" / "DeepSeek-OCR 2026"
  - test img2 → "Invoice #2026-0042" / "Date: June 25, 2026" / "Total: $1,337.00 USD"
  - Output format is `<|det|>LABEL [x1,y1,x2,y2]<|/det|>TEXT` triples (bbox 0–1000 grid); recognized text is the segment after each `<|/det|>`. Needs no_repeat_ngram. Defaults base_size=1024, image_size=640, crop_mode=True (gundam).
  - Good prompts: `<image>\n<|grounding|>OCR this image.` or `<image>\nFree OCR.`

### CRITICAL: the mlx-vlm reference is BUGGY (do not match it)
mlx-vlm 0.6.3's `deepseekocr` collapses to a single repeated token (id 31670 "筆") the moment IMAGE embeddings are injected — bf16 AND int8, every prompt. Text-only works; vision outputs are non-NaN. The bug is in mlx-vlm's `get_input_embeddings` image-feature assembly/injection — **the exact code our Swift port mirrors**. So our port most likely reproduces the same collapse.

=> CORRECTNESS GATE CHANGES: verify/fix our `inputEmbeddings` against the OFFICIAL PyTorch `modeling_deepseekocr.py` + `deepencoder.py` (downloaded at /tmp/dsocr_code/), NOT mlx-vlm. Suspects to diff vs official: (a) the concat order `concat(clip[:,1:], sam.flatten(1,2))` and which axis, (b) the global/local tiling reassembly + image_newline/view_separator placement, (c) the projector input layout (downsample vs linear), (d) the image-token COUNT vs feature count alignment (processor↔model), (e) SAM/CLIP feature normalization. The token-collapse signature = image features landing wrong / shape or position mismatch.

### Arch note
Both deepseek-ai/DeepSeek-OCR and baidu/Unlimited-OCR are DeepseekOCRForCausalLM / model_type deepseek_vl_v2. The 8-bit MLX (/tmp/ocr_models/unlimited-ocr-8bit-mlx) is the load target. mlx-vlm crashed loading deepseek-ai/DeepSeek-OCR's Conv2d but loaded baidu/Unlimited-OCR — minor config delta to watch.

### Next phase (needs careful/fresh attention — numerical)
1. Build dev app done → load 8-bit OCR model in osaurus → run OCR → observe (expect collapse like mlx-vlm).
2. Diff our inputEmbeddings/feature-assembly vs official modeling_deepseekocr.py; fix until Swift reproduces the PyTorch ground-truth text on the 2 test images. THIS is the gate.
3. Then osaurus OCR spawn/delegate component: new OCR delegate tool (mirror local_delegate + image-gen): model detection, default-OCR-model setting, single-residency handoff vs keep-loaded per RAM-safety setting, prompt+context pass-through after OCR, multiturn. Settings UI like image-gen/edit/spawn/text-delegate.
4. UI E2E via gpt-5.5 computer-use: model loads, OCR correct, multiturn, reasoning on/off, spawn-tool-for-OCR usable by other models.

### LIVE STATE 2026-06-25 (dev app built + OCR engine loaded)
- osaurus dev app built with engine (vmlx feccec2), OCR model `unlimited-ocr-8bit-mlx` (model_type `deepseekocr`) is RECOGNIZED + routed as VLM (factory alias works).
- First load blocker (concrete, weight-key): `Unhandled keys ["neck"] in sam_model in DeepseekOCRSAMEncoder` — the SAM `neck` module structure (built as explicit neck.0..3 members, cross-checked against deepseek-ai/DeepSeek-OCR) does NOT match the 8-bit `unlimited-ocr-8bit-mlx` neck key layout. Fix the SAM neck module to consume the actual `sam_model.neck.*` keys of the load target (the 8-bit MLX pack — confirm its exact keys; it may differ from the bf16 deepseek-ai pack the subagent checked). Likely the rest of the load surfaces a few more key/shape mismatches (net_2/net_3, pos_embed, quant scales/biases for 8-bit).
- THEN the numerical image-injection gate (mlx-vlm-buggy; match official PyTorch).
NEXT-SESSION START HERE: fix weight-key mapping (neck first) → load clean → run OCR → fix image-injection vs official → osaurus OCR delegate component + settings/RAM + UI E2E.
