# vMLX-Flux (native mFLUX image gen) — HANDOFF

**For:** the next engineer/agent continuing the native mFLUX image-generation port.
**Date:** 2026-06-16. **Owner:** Eric. **Status:** z-image-turbo and flux-schnell
are live-proven for 4/8-bit; qwen-image 4-bit is live-proven; qwen-image-edit
has q4 load + prompt-image + conditioning + transformer + scheduler + decode
plumbing that writes PNGs, but the viewed q4 edit outputs are noise-like, so
edited-image quality remains `PARTIAL`.

**2026-06-16 continuation evidence:** live baseline probes were rerun from
`/Users/eric/vmlx-swift` so MLX could resolve `default.metallib`; the standalone
repo itself does not contain that Metal library. Fresh standalone artifacts are
under `docs/local/vmlx-flux-{probes,outputs}/2026-06-16-*`. The Osaurus
monorepo worktree also has fresh load artifacts from `/Users/eric/vmlx-swift-fluxwt`:
`docs/local/vmlx-flux-probes/2026-06-16-osaurus-qwen-image-q4-load-final/qwen-image-mflux-4bit-load.json`
(`load_status=loaded`, `native_runtime_status=native_pipeline_implemented`) and
`docs/local/vmlx-flux-probes/2026-06-16-osaurus-qwen-edit-q4-load-final/Qwen-Image-Edit-mflux-q4-load.json`
(`load_status=loaded`, `native_runtime_status=native_pipeline_partial`).

This is the single starting doc. Read it top to bottom, then the per-model port plans.

---

## 0. TL;DR — what works, what's next

| Model | 4-bit | 8-bit | full | Native pipeline file |
|---|---|---|---|---|
| **z-image-turbo** | ✅ proven | ✅ proven | ⬜ (weights gone) | `Libraries/vMLXFluxModels/ZImage/ZImageNative.swift` |
| **flux-schnell** | ✅ proven | ✅ proven | ⬜ (not staged) | `Libraries/vMLXFluxModels/Flux1/Flux1Native.swift` |
| **qwen-image** (txt2img) | ✅ proven | ⬜ | ⬜ | `Libraries/vMLXFluxModels/Common/QwenImageNative.swift` |
| qwen-image-edit | PARTIAL q4 ImageEditor loop completes and writes PNGs, but viewed outputs are noise-like; q3/q4/q5 scan loadable | — | — | debug prompt-image/edit quality root cause; coherent edited-image proof missing |
| ideogram (4) | ⬜ scaffold | — | — | `Libraries/vMLXFluxModels/Ideogram4/Ideogram4.swift` (fp8) |
| flux1-dev/kontext/fill, flux2-klein, fibo, seedvr2, wan | ⬜ scaffold | — | — | registered, throw `notImplemented` |

"Proven" = live-generated a coherent, prompt-accurate image that is **deterministic** (same seed+prompt -> byte-identical) and **prompt-sensitive** (different prompt same seed -> different coherent image). Per Eric's HARD RULE: *do not trust/claim a model works until you have generated and visually checked a real image.* 2026-06-16 rerun: z-image 4/8 and flux-schnell 4/8 passed live load + three-turn generate + SHA determinism/prompt-sensitivity + visual inspection. Qwen-image 4-bit now also passed live load + 20-step generation + three-turn SHA determinism/prompt-sensitivity + visual inspection after the mflux guidance rescale fix; turn 1/3 apple SHA `2f1c27c68993fe9a537bca2cc019ac3d32d59818b92c606c00726104661bcea7`, turn 2 mountain SHA `2bf77ce59c8ed99c1b1aa5fb8940c9d35948b1763fbd360e14f577032b62f060`, artifact `docs/local/vmlx-flux-probes/2026-06-16-qwen-image-q4-guidance-proof/qwen-image-mflux-4bit-load.json`.

Qwen-image-edit q4 passed manifest-gated load plus prompt-token, Qwen2.5-VL prompt-image encode, VAE conditioning, first transformer velocity, scheduler loop, and VAE decode/source PNG write plumbing. The live ImageEditor artifacts completed and wrote PNGs, but viewed outputs were noise-like, so this is **not** a working/coherent edit row. Current status artifact: `docs/local/vmlx-flux-probes/2026-06-16-qwen-edit-q4-partial-status-live/Qwen-Image-Edit-mflux-q4-load.json` (`load_status=loaded`, `native_runtime_status=native_pipeline_partial`). Current edit artifacts: `docs/local/vmlx-flux-probes/2026-06-16-qwen-edit-q4-edit-4step-guidance-live/Qwen-Image-Edit-mflux-q4-load.json` (256x256, 4 steps, output SHA `afeb1714...`) and `docs/local/vmlx-flux-probes/2026-06-16-qwen-edit-q4-edit-512-4step-live/Qwen-Image-Edit-mflux-q4-load.json` (512x512, 4 steps, output SHA `c9f919c...`). Earlier boundary artifacts remain useful for isolating the failing edit-quality path: `docs/local/vmlx-flux-probes/2026-06-16-qwen-edit-q4-prompt-live/Qwen-Image-Edit-mflux-q4-load.json`, `docs/local/vmlx-flux-probes/2026-06-16-qwen-edit-q4-vl-encode-live/Qwen-Image-Edit-mflux-q4-load.json`, `docs/local/vmlx-flux-probes/2026-06-16-qwen-edit-q4-conditioning-live/Qwen-Image-Edit-mflux-q4-load.json`, and `docs/local/vmlx-flux-probes/2026-06-16-qwen-edit-q4-denoise-live/Qwen-Image-Edit-mflux-q4-load.json`.

**Next work, in priority order:**
1. **qwen-image-edit** — debug why the now-wired prompt-image + conditioning + denoise + decode loop produces noise-like PNGs. Do not call it working until a coherent edited image is viewed.
2. **Ideogram 4** — needs an **fp8 quant path** (different from the MLX group-quant used by the others) + Qwen3 encoder + 34-layer DiT.
3. **Full-precision** flux-schnell + z-image (download + prove with existing pipelines — should "just work").
4. Consolidated PR of all the new models to `osaurus-ai/vmlx-swift` main.

---

## 1. Where the code lives

- **vmlx-swift integration worktree:** `/Users/eric/vmlx-swift-fluxwt` — clean
  Osaurus monorepo worktree for this lane. Current integration branch:
  `codex/mflux-qwen-edit-main`, based on `vmlx-origin/main`.
- **Dirty local dev tree:** `/Users/eric/vmlx-swift` — branch
  `codex/mimo-v25-cache-contract` carries unrelated MLXPress/MiMo/Gemma/JANG
  WIP; do not commit the image-gen integration from that dirty checkout.
- **Pushable remotes:**
  - `jjang-ai/vmlx-flux` (standalone SwiftPM engine) — **all native work is pushed here** on branch `native-zimage-proven`. This is the durable home. Latest: branch HEAD.
  - `osaurus-ai/vmlx-swift` (the monorepo) — z-image engine vendored + merged via **PR #63** (`90e64687`). Current qwen/flux-schnell integration is on `codex/mflux-qwen-edit-main` for PR/merge to main. Remote name `vmlx-origin`. (Note: the `osaurus-upstream` remote is DO_NOT_PUSH — only the mlx-swift fork.)
- **Standalone clone (for vmlx-flux pushes):** `/Users/eric/vmlx-flux-push` (sibling to `../vmlx-swift-lm` so its path-deps resolve).

### Module layout (vendored in `vmlx-swift/Package.swift` as in-tree targets)
- `vMLXFluxKit` — `FluxEngine` types, `ModelRegistry`, requests/events, `FlowMatchEulerScheduler`, `VAE`, `WeightLoader`, `MLXStudioModelStore` (`LocalModelStore.swift`), JANG bridge.
- `vMLXFluxModels` — concrete models. **`Common/`** holds the shared, reusable pieces:
  - `MFluxQuant.swift` — `MFluxStore` + `MFluxLinear`/`MFluxEmbedding`/`MFluxRMSNorm`/`MFluxLayerNorm`/`MFluxGroupNorm`/`MFluxConv2D`. **This is the foundation every port builds on.**
  - `T5XXL.swift` (flux), `CLIPText.swift` (flux), `QwenImageNative.swift` (qwen full pipeline).
  - `Flux1/Flux1Native.swift` (flux DiT + VAE + pipeline; also defines `FluxAdaNormContinuous` reused by qwen), `ZImage/ZImageNative.swift` (z-image, has its own private store — left untouched to avoid regressing the proven path).
- `vMLXFluxVideo` — WAN 2.x scaffold (`WanVAE3D.swift` has a `CausalConv3d` shim).
- `vmlxflux-probe` (`tools/vMLXFluxProbe/main.swift`) — the verification CLI (see §4).

---

## 2. Build & run

```bash
cd /Users/eric/vmlx-swift-fluxwt
swift build --product vmlxflux-probe          # warm .build; ~3-40s. Default CommandLineTools toolchain is fine.
# Unit tests need the Xcode toolchain (XCTest):
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer swift test --filter vMLXFluxTests
```
- The repo's CI **skips** `mac_build_and_test` (known) — local build is the only gate before merging to main.
- Main is **tools-version 6.1 / Swift 6 language mode**. The flux/qwen targets are Swift-6-clean (verified) — no `.swiftLanguageMode(.v5)` pin needed.

---

## 3. Models on disk (staged at `~/.mlxstudio/models/image/`)
- `Z-Image-Turbo-mflux-4bit` (5.5GB), `Z-Image-Turbo-mflux-8bit` (10GB)
- `FLUX.1-schnell-mflux-4bit` (9GB), `FLUX.1-schnell-mflux-8bit` (12GB)
- `qwen-image-mflux-4bit` (24GB)
- `Qwen-Image-Edit-mflux` (89GB) with nested `q3`, `q4`, `q5`, `q6` variants.
  The scanner now expands these as `Qwen-Image-Edit-mflux-q3/q4/q5/q6`; `q3`,
  `q4`, and `q5` have tokenizer/text_encoder/transformer/vae, while `q6` is
  incomplete on the current disk image. `Qwen-Image-Edit-mflux-q4` also passes
  the manifest-gated engine load contract (tokenizer files + Qwen LM keys +
  Qwen-VL vision keys + transformer keys + VAE encode/decode keys) and a live
  source-image preprocess request (`output=1024x1024`, `vl=384x384`,
  `vae=1024x1024`, `conditioning=64x64`, `vision_patches=784x1176`,
  `vision_grid=1x28x28`, `vae_input=1x3x1024x1024`). The q4 prompt probe
  tokenizes the mflux edit prompt with 196 repeated image-pad tokens
  (`input_ids_shape=1x276`, `image_token_id=151655`, `image_token_count=196`,
  `template_drop_index=64`). The q4 conditioning probe also live-encodes the
  source through the Qwen 3D VAE and packs static image latents
  (`latents_shape=1x4096x64`, `image_ids_shape=1x4096x3`, finite stats). The q4
  VL encode probe runs the real Qwen2.5-VL vision transformer and image-token
  splice (`feature_shape=196x3584`, `prompt_embeds_shape=1x212x3584`,
  `token_image_count=196`, finite stats). The q4 denoise probe concatenates
  256 target latents with 4096 static conditioning latents, runs the edit-shaped
  transformer RoPE grids, and slices the target velocity
  (`combined_velocity_shape=1x4352x64`, `target_velocity_shape=1x256x64`,
  finite stats). `ImageEditor` now runs the scheduler loop, decodes, and writes
  PNGs for q4, but the viewed 256px/512px edit outputs are still noise-like.
  Treat qwen-image-edit as `PARTIAL` until prompt-image/edit fidelity is fixed
  and a coherent edited image is live-proven.

**Downloadable mflux-compatible weights (HF):**
- flux: `dhairyashil/FLUX.1-schnell-mflux-{4,8}bit`; full = `black-forest-labs/FLUX.1-schnell` (GATED).
- z-image: `Tongyi-MAI/Z-Image-Turbo` (full), `carsenk/z-image-turbo-mflux-8bit`, `filipstrand/Z-Image-Turbo-mflux-4bit`.
- qwen: `carsenk/qwen-image-mflux-4bit` (txt2img), `fcreait/Qwen-Image-Edit-mflux` (87GB full edit model).
- ideogram: `ideogram-ai/ideogram-4-fp8` (the mflux canonical), `ideogram-ai/ideogram-4-nf4` (4-bit).

**TOKENIZER GOTCHA:** mflux bundles ship SLOW tokenizers (CLIP vocab.json+merges, T5 spiece.model). swift-transformers' `AutoTokenizer.from(modelFolder:)` needs `tokenizer.json` (fast). Convert once:
```python
from transformers import AutoTokenizer
AutoTokenizer.from_pretrained(dir, use_fast=True).save_pretrained(dir)   # for tokenizer/ and tokenizer_2/
```
(qwen + z-image bundles already ship tokenizer.json; flux needs the conversion.) `pip install --user transformers tokenizers sentencepiece protobuf` is already done on this box.

---

## 4. The probe (verification harness)
`tools/vMLXFluxProbe/main.swift`. The canonical proof command (same-seed determinism + prompt-sensitivity):
```bash
.build/debug/vmlxflux-probe --model <DIR-NAME> --generate --json \
  --seed 7 --width 512 --height 512 --steps <N> \
  --artifacts <art> --output-dir <out> \
  --turn "a red apple on a wooden table, photo" \
  --turn "a snowy blue mountain landscape, watercolor" \
  --turn "a red apple on a wooden table, photo"
# turn1≡turn3 (byte-identical sha) ⇒ deterministic; turn2≠turn1 ⇒ prompt-sensitive; then VIEW the PNGs.

.build/debug/vmlxflux-probe --model Qwen-Image-Edit-mflux-q4 --edit \
  --source-image <png> --turn "make the background blue" \
  --artifacts <art>
# Current qwen-edit status: live load + ImageEditor loop + PNG write complete,
# but viewed q4 outputs are noise-like; coherent edit proof is missing.

.build/debug/vmlxflux-probe --model Qwen-Image-Edit-mflux-q4 --qwen-edit-prompt \
  --source-image <png> --turn "make the background blue" \
  --artifacts <art>
# Current qwen-edit prompt status: live q4 load + real tokenizer image-pad expansion.

.build/debug/vmlxflux-probe --model Qwen-Image-Edit-mflux-q4 --qwen-edit-conditioning \
  --source-image <png> --artifacts <art>
# Current qwen-edit conditioning status: live q4 load + source-image VAE encode + packed static latents.

.build/debug/vmlxflux-probe --model Qwen-Image-Edit-mflux-q4 --qwen-edit-vision \
  --source-image <png> --turn "make the background blue" \
  --artifacts <art>
# Current qwen-edit VL status: live q4 load + Qwen2.5-VL vision features + image-token splice into Qwen text embeddings.

.build/debug/vmlxflux-probe --model Qwen-Image-Edit-mflux-q4 --qwen-edit-denoise \
  --source-image <png> --width 256 --height 256 --steps 1 --seed 7 \
  --turn "make the background blue" --artifacts <art>
# Current qwen-edit denoise status: live q4 load + prompt-image encode + VAE
# conditioning + first transformer velocity slice.
```
- Flags: `--guidance`, `--negative` (added for CFG), `--width/height/steps/seed/turn/root/model/output-dir/artifacts/--matrix`.
- `--model` must be the **exact directory name** (the resolution bug — §6 — is fixed so `-8bit` no longer collapses onto `-4bit`).
- Per-turn seed = `--seed` if given (so all turns share it). Pipelines print `[qwen]`/`[flux]` stderr stats (shape/mean/max/finite) per stage — the **de-risk signal**: if a stage isn't finite, that's where the bug is.

---

## 5. Architecture cheat-sheet (how the pipelines are built)

All on `MFluxStore` (loads safetensors via `WeightLoader`, builds quant-aware layers from exact checkpoint keys). mflux **linear** weights are PyTorch `(out,in)` (handled by `MFluxLinear` via `matmul(x, weight.T)`); mflux **conv** weights are **MLX channels-last** `(out,[kt,]kh,kw,in)` (see §6 bug 1).

- **flux-schnell** (`Flux1Native.swift`): T5-XXL (`T5XXL.swift`) → per-token `prompt_embeds`; CLIP-L (`CLIPText.swift`) → pooled vector. `FluxTransformer` = 19 joint + 38 single blocks, 24h×128, 3-axis RoPE (`FluxRoPE`), `FluxTimeTextEmbed`, `FluxAdaNormZero/Single/Continuous`. `FluxVAEDecoder` (AutoencoderKL). FlowMatch Euler, 4 steps, guidance 0. **timestep passed = sigma×1000** (flux time-proj has no internal scale).
- **z-image-turbo** (`ZImageNative.swift`): Qwen-style text encoder + Lumina-style DiT (noise/context refiners) + AutoencoderKL VAE. Proven; uses its own private store. 4 steps, guidance 0.
- **qwen-image** (`QwenImageNative.swift`):
  - `QwenTextEncoder` = Qwen2.5 LM (28-layer, GQA 28q/4kv, standard RoPE θ1e6, SwiGLU, **causal**). Tokenize with the gen template; **drop the first 34 tokens** of the output → prompt embeds.
  - `QwenTransformer` = 60-layer MM-DiT (dual-stream `QwenBlock`: img/txt `mod_linear` 3072→18432 split into mod1(attn)/mod2(mlp), each shift/scale/gate; `QwenAttn` joint img+txt with RMSNorm q/k + complex-pair RoPE `QwenRoPE` axes[16,56,56] θ1e4 scale_rope; `QwenFF` gelu_approx 4×). img_in 64→3072, txt_norm+txt_in 3584→3072, `QwenTimeEmbed`, norm_out=`FluxAdaNormContinuous`, proj_out→64.
  - `Qwen3DVAEEncoder`/`Qwen3DVAEDecoder` = 3D causal-conv VAE **operated in 2D since T=1** (each causal Conv3d → 2D conv on the last temporal kernel slice; decoder resamplers do spatial nearest-2× + conv; encoder downsamplers pad bottom/right then stride-2 conv). Per-channel `LATENTS_MEAN/STD` (16-vectors). Decoder channel flow 384→192→192→96→3 over 3 upsamples (8×). Qwen-edit q4 VAE conditioning encode/pack and edit-loop PNG plumbing are live-proven, but coherent edit quality is not.
  - Pipeline: noise (flux-style pack, 1,hw,64) → loop[CFG: pos+neg transformer passes → mflux guidance rescale (`combined = neg + g*(pos-neg)`, then rescale to positive-noise norm) → FlowMatch step] → unpack → 5D → VAE decode → PNG. **timestep passed = RAW sigma** (`QwenTimesteps` applies ×1000 internally — see §6 bug 2). ~20 steps, guidance ~4 (CFG).

Full per-model transcription specs are in `docs/FLUX_SCHNELL_PORT_PLAN.md` and `docs/QWEN_IMAGE_PORT_PLAN.md` (grounded from the mflux Python source).

---

## 6. Bugs found & fixed (don't reintroduce these)
1. **Conv weight layout.** mflux stores conv weights in **MLX channels-last** `(out, [kt,] kh, kw, in)`, NOT PyTorch `(out, in, k...)`. Assuming PyTorch → wrong reshape/transpose → load-time crash (`reshape 442368→(1152,1)`). Linear weights ARE PyTorch `(out,in)`.
2. **Qwen timestep double-scale.** mflux passes the raw sigma to `QwenTimesteps(scale=1000)` which multiplies internally. Passing `sigma×1000` double-scales → the transformer denoises to pure noise. Pass the **raw sigma**.
3. **Model resolution / quant collision.** `MLXStudioModelStore.resolve` normalized away the `-Nbit` suffix → requesting `...-8bit` loaded a co-installed `...-4bit`. Fixed with a literal case-insensitive directory-name match first. **osaurus must request the exact bundle directory name.**
4. **Tokenizer format** — see §3 (need fast `tokenizer.json`).
5. **GPU watchdog** — MLX mmaps weights lazily; running gen with weights on a **slow volume (USB)** stalls the Metal command buffer → `kIOGPUCommandBufferCallbackErrorTimeout`. **Stage weights on the internal SSD.**

---

## 7. RULES (Eric's, non-negotiable)
- **No AI attribution** in any commit/PR/GitHub-visible content (no `Co-Authored-By: Claude`, no "Generated with"). All commits are Eric's.
- **Live-prove everything.** Do not claim a model/feature works until you've generated a real image and visually verified it's coherent + prompt-accurate (+ deterministic + prompt-sensitive). "Builds clean" and "stats are finite" are necessary but NOT sufficient.
- **No fake guards / fake behavior.** Real fixes only.
- `jjang-ai/wiki` is a PRIVATE repo — never copy wiki content into project repos. Never store secrets in the wiki.
- Don't push `vmlx-swift` to the `osaurus-upstream` remote (DO_NOT_PUSH). Use `vmlx-origin` for the monorepo, `origin` for vmlx-flux.

---

## 8. osaurus integration (for the UI/server team)
- `docs/OSAURUS_VMLX_FLUX_INTEGRATION_SPEC.md` — engine API (`FluxEngine` actor: load/generate/edit/upscale), `ImageGenRequest`/events, model registry, per-model status, the **required MetalGate exclusion** (image-gen MLX eval races LLM eval on the shared Metal command buffer — same SIGABRT hazard as the Model2Vec embedder, osaurus PR #1507 — so gate it), quant matrix, gotchas.
- `docs/OSAURUS_IMAGE_API_SPEC.md` — UI-facing HTTP contract: `GET /v1/images/models`, `POST /v1/images/{generations,edits,upscale}`, every request setting (prompt/negative/steps/guidance/strength/size/seed/n/format), and the SSE **progress events** (`queued`→`loading_model`→`step{step,total,progress,eta}`→`completed`) so the UI shows "Step N/M" and never looks stuck.
- The HTTP layer is a **proposed contract** — the engine is real, but the `/v1/images/*` endpoints aren't built in osaurus yet.

---

## 9. How to continue (concrete next steps)
1. **qwen-image-edit:** the txt2img pipeline works, and the q4 prompt-token, Qwen2.5-VL prompt-image encode, source-image VAE conditioning, first transformer velocity, scheduler loop, and PNG-write boundaries are live-proven. Read `/tmp/mflux-ref/src/mflux/models/qwen/variants/edit/` + transformer edit code next with the specific goal of finding the edit-quality mismatch. The prompt/image side matches the mflux edit template (`edit_template_start_idx=64`, `Picture N:` image prefix, `<|vision_start|><|image_pad|><|vision_end|>`) and splices Qwen-VL image features into the text-token stream at `image_token_id=151655`; the ImageEditor path also concatenates target latents with static image latents and runs the edit-shaped transformer RoPE grids. Current blocker: q4 live edit PNGs complete but are noise-like, so coherent edited-image proof is missing.
   - Current staged bundle is already present at `~/.mlxstudio/models/image/Qwen-Image-Edit-mflux`; use the `q4` variant first (`Qwen-Image-Edit-mflux-q4`) for implementation/proof. Current q4 proof artifacts: `docs/local/vmlx-flux-probes/2026-06-16-qwen-edit-q4-manifest-load/Qwen-Image-Edit-mflux-q4-load.json` (`load_status=loaded`, `generate_requested=false`), `docs/local/vmlx-flux-probes/2026-06-16-qwen-edit-q4-partial-status-live/Qwen-Image-Edit-mflux-q4-load.json` (`native_runtime_status=native_pipeline_partial`), `docs/local/vmlx-flux-probes/2026-06-16-qwen-edit-q4-prompt-live/Qwen-Image-Edit-mflux-q4-load.json` (`qwen_edit_prompt_tokens[0].status=tokenized`, `image_token_count=196`, `template_drop_index=64`), `docs/local/vmlx-flux-probes/2026-06-16-qwen-edit-q4-vl-encode-live/Qwen-Image-Edit-mflux-q4-load.json` (`qwen_edit_vision_language[0].status=encoded`, `feature_shape=196x3584`, `prompt_embeds_shape=1x212x3584`, finite stats), `docs/local/vmlx-flux-probes/2026-06-16-qwen-edit-q4-conditioning-live/Qwen-Image-Edit-mflux-q4-load.json` (`qwen_edit_conditioning.status=encoded`, `latents_shape=1x4096x64`, `image_ids_shape=1x4096x3`, finite stats), `docs/local/vmlx-flux-probes/2026-06-16-qwen-edit-q4-denoise-live/Qwen-Image-Edit-mflux-q4-load.json` (`qwen_edit_denoise[0].status=predicted`, `combined_velocity_shape=1x4352x64`, `target_velocity_shape=1x256x64`, finite stats), `docs/local/vmlx-flux-probes/2026-06-16-qwen-edit-q4-edit-4step-guidance-live/Qwen-Image-Edit-mflux-q4-load.json` (`edit_turns[0].status=completed`, 256x256 PNG), and `docs/local/vmlx-flux-probes/2026-06-16-qwen-edit-q4-edit-512-4step-live/Qwen-Image-Edit-mflux-q4-load.json` (`edit_turns[0].status=completed`, 512x512 PNG).
2. **Ideogram 4:** download `ideogram-ai/ideogram-4-nf4`. Port = Qwen3 text encoder (close to the qwen LM encoder) + 34-layer DiT (emb 4608, 18 heads, `llm_features 4096×13` = multi-layer Qwen3 hidden states, rope θ5e6) + VAE. **Build an fp8 dequant/matmul path** in `MFluxStore` (the transformer is fp8, not group-quant). Ref: `/tmp/mflux-ref/src/mflux/models/ideogram4/`.
3. **Full precision** flux/z-image: download, run the probe — existing pipelines (`MFluxLinear` handles non-quant). Should just work.
4. **Consolidated osaurus PR:** keep `codex/mflux-qwen-edit-main` rebased on
   current `vmlx-origin/main`, keep standalone imports rewritten to
   `VMLXTokenizers` in the monorepo, verify build/tests (Swift 6), then open or
   update the PR to main. The mlx-swift / swift-transformers fork pins must
   match `../vmlx-swift-lm` (mlx-swift `0a56f904`, swift-transformers osaurus
   fork `087a66b1`) — see vmlx-flux Package.swift.

**Reference:** the mflux Python source (the source of truth for every arch + weight key) is at `/tmp/mflux-ref` (clone of `github.com/filipstrand/mflux`). Re-clone if gone.

---

## 10. GH PR / commit references
- `osaurus-ai/vmlx-swift` **PR #63** — z-image engine vendored + merged to main (`36aebd42→90e64687`).
- `osaurus-ai/vmlx-swift` branch **`codex/mflux-qwen-edit-main`** — current
  Osaurus monorepo sync branch, based on `vmlx-origin/main` `90e64687`. Its
  HEAD vendors the standalone flux-schnell + qwen-image pipelines, keeps
  qwen-image-edit marked `PARTIAL`, and updates the osaurus image API /
  integration docs for team wiring.
- `jjang-ai/vmlx-flux` branch **`native-zimage-proven`** — all native work: `9915417` (z-image vendor+proof), `4a88089` (resolution fix), `a2c1a28` (flux-schnell working), `f82dd1b` (probe flags), `fc6e5b1` (qwen-image working + ideogram scaffold), `f7014e0` (handoff); current HEAD adds qwen-edit nested scan, manifest-gated q4 load, q4 source-image tensor preprocess proof, q4 VAE conditioning latent proof, q4 Qwen2.5-VL prompt-image encode proof, q4 first transformer velocity proof, qwen mflux guidance rescale, and q4 edit-loop PNG plumbing. Open a PR from this branch to vmlx-flux main when ready.
- Wiki note (private `jjang-ai/wiki`): `notes/2026-06-15-vmlx-flux-native-z-image-proven-fork-lockstep.md`.
- Per-project memory: `~/.claude/projects/-Users-eric-vmlx-swift/memory/vmlx-flux-native-zimage-integration.md`.
- Proof artifacts (gitignored): `docs/local/vmlx-flux-{outputs,probes}/` (PROOF-*, FLUX-proof, QWEN-proof, Q8b-*).
