# DeepSeek-OCR — true architecture, cache (engine + osaurus), and the real divergences

Synthesized from the OFFICIAL HF reference (deepencoder.py / modeling_deepseekocr.py /
modeling_deepseekv2.py) cross-read against our Swift port, the vmlx cache, and osaurus serving.

## A. Forward path (what actually happens)
Image → **DeepEncoder** (two towers, run on each view):
- **SAM-ViT-B** (`sam_model`): Conv2d patch16 → +pos_embed → 12 blocks (windowed attn win=14 except global at {2,5,8,11}, decomposed rel-pos) → neck (Conv 768→256, **LayerNorm2d over channels**, Conv 256→256, LayerNorm2d) → net_2 (Conv→512 stride2) → net_3 (Conv→1024 stride2). Global 1024² → **[B,16,16,1024]** (256 tokens); crop 640² → **[B,10,10,1024]** (100 tokens).
- **CLIP-L/14** (`vision_model`): does NOT run its own patch conv — it takes the SAM output as `patch_embeds` (`flatten(2).transpose` → [B,256,1024]), prepends CLS → +pos → pre_layrnorm → 24 blocks (quick_gelu, fused qkv) → **[B,257,1024]**.
- **Fuse**: `cat(clip[:,1:], sam.flatten(spatial), axis=-1)` → **[B,256,2048]** (CLIP first, SAM second).
- **Projector**: linear 2048→1280.
- **2D tiling**: per row append one `image_newline` (→ [16,17,1280] flatten 272); local crops reassembled to crop grid then newline per row; final seq = `cat([local, global, view_separator])` (single uncropped = global+separator = 273 rows).
- **Inject**: `masked_scatter` the feature rows into text embeds at `image_token_id` positions (count(mask)==num_feature_rows is a HARD invariant; the per-row newline + separator tokens are baked into input_ids by the processor).

Text → **DeepSeek-V2 decoder** (12 layers): standard **Llama MHA + RoPE** (use_mla=false ⇒ the MLA class is DEAD CODE), head_dim 128, 10 heads (no GQA), rope_theta 1e4 non-traditional; layer 0 dense MLP, 1-11 MoE (64 routed +2 shared, top-6 greedy softmax, scaling 1.0, no norm_topk_prob); RMSNorm eps 1e-6; untied lm_head. **Our decoder port is FAITHFUL — no fix needed there.**

## B. The collapse bug is in the VISION TOWERS + zero-init params, NOT the assembly
Concat order / projector / tiling / injection are faithful in our port — do NOT chase them. Ranked real causes:
1. **H2 (concrete, Swift-specific): `image_newline`/`view_separator` are `MLXArray.zeros` in DeepseekOCR.swift init.** They are LEARNED checkpoint params inserted as actual feature rows (every 17th token + the separator). If `sanitize` doesn't map the checkpoint keys onto these `@ParameterInfo` keys, they stay ZERO → structural corruption at every newline/separator → collapse. **Verify the real key names in the 8-bit pack (`image_newline` vs `model.image_newline`) and assert they load non-zero.** Cheapest + highest-value check.
2. **H1: SAM neck `LayerNorm2d`** normalizes over the CHANNEL axis (NCHW). Our port uses standard `LayerNorm(outChans)` over the last axis (NHWC). Axis coincides only if conv outputs are truly NHWC at the neck — fragile; verify reduction axis + layout. Corrupted SAM half of the 2048 concat → noise.
3. **H3: `getAbsPosSAM` is an identity no-op even when src≠tgt** → wrong positional enc for 640 crop tiles (the common multi-tile OCR path). Fix: interpolate when sizes differ (the rel-pos resize branch already exists).
4. **H4: CLIP `patch_embeds` token order** — official flattens NCHW `flatten(2).transpose`; ours flattens NHWC. If SAM emits a transposed grid the 256 tokens are spatially scrambled (fatal for position-sensitive OCR).
5. **H5: mask-count vs feature-count** — assert `imageIndices.count == globalLocalFeatures.dim(0)` in the scatter.

Plus the **current load blocker**: SAM `neck` built as flat `@ModuleInfo(key:"neck.0".."neck.3")` members → MLX reports `neck` as an unhandled key group. Fix: make `neck` a proper container so `sam_model.neck.{0,1,2,3}.*` map.

### Fix/verify order
load fix (neck container) → H2 (assert newline/separator non-zero; fix sanitize keys) → H5 (assert counts) → H1 (neck LN axis) → H3 (crop abs-pos) → H4 (CLIP token order). Gate each against the PyTorch ground-truth text (/tmp/ocr_reference_output.txt).

## C. KV cache — engine (vmlx) side
- `KVCache` protocol (Libraries/MLXLMCommon/KVCache.swift); concrete: **KVCacheSimple** (default, growable, full attention), RotatingKVCache (sliding window), Quantized/Chunked/Paged.
- Per-layer `newCache`; attention calls `cache.update` via `attentionWithCacheUpdate`; RoPE offset by `cache.offset`. Prefill (L>1) appends L; decode (L==1) appends 1.
- BatchEngine: per-request cache; VLM `prepare()` returns **`.logits`** (whole prompt incl. image tokens prefilled in one forward); multi-tier prefix/paged/disk restore keyed by **SHA(tokens + modelKey + mediaSalt)** so same-text+same-image hits, different-image misses. Rotating caches skip partial restore. VLM image fusion is **B=1** (not co-batchable).
- **PORT FIX**: DeepseekOCR `newCache` uses `RotatingKVCache` when `maxKVSize` set — but this decoder is FULL attention, so rotating is semantically wrong AND defeats prefix/paged reuse. **Drop the rotating branch → always `KVCacheSimple`.**

## D. KV cache + serving — osaurus side
- Load: `ModelRuntime.loadContainer` → vmlx `loadModelContainer` (VLM factory tried first; VLM iff config has `vision_config` / model_type in `VLMModelFactory.supportedModelTypes`). RAM gate `checkRAMFeasibility` is ADVISORY (logs, proceeds); MetalGate makes model-load EXCLUSIVE; strict single-residency eviction + double GPU drain on unload.
- KV: default native fp16 (TurboQuant opt-in only — auto-on regressed Gemma). Prefix cache content-addressed (not session-bound). Paged opt-in. Disk/L2 (DiskCache.swift, content-addressed, survives unload, orphan-row eviction, host-capped 25% free disk). Memory-safety slider rewrites cache caps (defaultMaxKVSize strict 16k→perf 131k, prefix caps, paged/disk off when prefix off) and is fed RESOLVED into the coordinator.
- VLM image path: HTTP image part → `extractImageSources` → `UserInput.Image.ciImage` → model `prepare()` → `LMInput`; `ModelMediaCapabilities`/`VLMDetection` gate UI/agents (glm-ocr already image-only family — add deepseek-ocr). media salt keys KV per-image.

## E. osaurus OCR spawn/delegate blueprint (after OCR works)
New `LocalOCRDelegateTool` (Tools/), mirrors `LocalTextDelegateTool` but VLM + image:
- gate `ocrDelegationEnabled` (+ global `agentDelegationEnabled`); resolve a VLM model (filter via `VLMDetection.isVLM`/`ModelMediaCapabilities.supportsImage`, reject non-VLM); `ChatResidencyHandoff.memoryPreflight` (refuse-before-evict 1.3×+3GB) → unload resident chat model → run → `restoreBestEffort`; load policy `unloadAfterJob` default (VLM weights large) vs keep-warm per RAM setting; seed `ChatMessage(contentParts:[.text(instruction), .imageUrl(dataURL)])` → `ChatEngine.completeChat` under `LocalTextDelegateContext.$isActive` (no recursive spawn); return compact OCR digest; multiturn/context pass via the same budget loop.
- Settings to add: `ocrDelegationEnabled`, `defaultOCRModelId`, `permissionDefaults.ocrExtract`, `ocrJobLoadPolicy`; register `LocalOCRDelegateTool()` in ToolRegistry; settings UI picker filtered to VLM/image-capable installed models. Gate residency on ACTUAL residency (avoids the SpawnTool SIGABRT class).

## F. VERIFIED WORKING 2026-06-25 (engine gate PASSED, commit 9a52851)
Independently re-verified via tools/DeepseekOCRSmoke on 3 images, both code paths:
- img1 global/`Free OCR.` → "OSAURUS OCR TEST" / "DeepSeek-OCR 2026" ✓
- img2 crop/multi-tile/`<|grounding|>OCR this image.` → "Invoice #2026-0042" / "Date: June 25, 2026" / "Total: $1,337.00 USD" ✓
- img3 unseen → "jumps over 13 lazy dogs." / "Receipt: GBP 42.50" ✓ (generalizes; numbers+currency exact)
~105-121 tok/s, bbox grounding correct. The decisive port bug was: prepare() took the pixel
working dtype from the (quantized→uint32) embed weights, zeroing the global view and silently
dropping image injection; fixed to bf16 (imageNewline.dtype). Plus SAM neck array-container load,
LayerNorm2dNHWC channel-axis, crop abs-pos bicubic, KVCacheSimple, plain sft prompt.

### Two known items that are NOT engine bugs (for serving/model-card, team-owned):
1. **Tokenizer packaging defect in `sahilchachra/unlimited-ocr-8bit-mlx`**: its tokenizer.json ships a
   SentencePiece `Metaspace` pretokenizer over a GPT-2 ByteLevel vocab → crashes on plain ASCII. The base
   `baidu/Unlimited-OCR` tokenizer.json (identical vocab/merges) is correct and was swapped in
   (orig backed up `tokenizer.json.broken-metaspace.bak`). **The shipped OCR pack must include the correct
   ByteLevel tokenizer.json.** Not a vmlx bug.
2. **Bare `Free OCR.` + crop path can emit a leading EOS → empty output** (known DeepSeek-OCR greedy quirk;
   official mitigates with no_repeat_ngram_size). The `<|grounding|>OCR this image.` prompt is clean on all
   paths. Serving should default to the grounding prompt (or add no-repeat-ngram). Not a port bug
   (global path + grounding prompt are clean).
