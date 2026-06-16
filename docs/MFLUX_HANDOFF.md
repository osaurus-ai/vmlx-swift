# vMLX-Flux (native mFLUX image gen) — HANDOFF

**For:** the next engineer/agent continuing the native mFLUX image-generation port.
**Date:** 2026-06-15. **Author:** Eric (+ Claude). **Status:** 3 of N models live-proven; clear runway for the rest.

This is the single starting doc. Read it top to bottom, then the per-model port plans.

---

## 0. TL;DR — what works, what's next

| Model | 4-bit | 8-bit | full | Native pipeline file |
|---|---|---|---|---|
| **z-image-turbo** | ✅ proven | ✅ proven | ⬜ (weights gone) | `Libraries/vMLXFluxModels/ZImage/ZImageNative.swift` |
| **flux-schnell** | ✅ proven | ✅ proven | ⬜ (not staged) | `Libraries/vMLXFluxModels/Flux1/Flux1Native.swift` |
| **qwen-image** (txt2img) | ✅ proven | ⬜ | ⬜ | `Libraries/vMLXFluxModels/Common/QwenImageNative.swift` |
| qwen-image-edit | ⬜ scaffold | — | — | needs Qwen2.5-VL vision tower on top of qwen-image |
| ideogram (4) | ⬜ scaffold | — | — | `Libraries/vMLXFluxModels/Ideogram4/Ideogram4.swift` (fp8) |
| flux1-dev/kontext/fill, flux2-klein, fibo, seedvr2, wan | ⬜ scaffold | — | — | registered, throw `notImplemented` |

"Proven" = live-generated a coherent, prompt-accurate image that is **deterministic** (same seed+prompt → byte-identical) and **prompt-sensitive** (different prompt same seed → different coherent image). Per Eric's HARD RULE: *do not trust/claim a model works until you have generated and visually checked a real image.*

**Next work, in priority order:**
1. **qwen-image-edit** — add the Qwen2.5-VL vision tower + image conditioning on top of the working qwen-image txt2img pipeline.
2. **Ideogram 4** — needs an **fp8 quant path** (different from the MLX group-quant used by the others) + Qwen3 encoder + 34-layer DiT.
3. **Full-precision** flux-schnell + z-image (download + prove with existing pipelines — should "just work").
4. Consolidated PR of all the new models to `osaurus-ai/vmlx-swift` main.

---

## 1. Where the code lives

- **Working repo (local-only build):** `/Users/eric/vmlx-swift` — current dev tree. Branch `codex/mimo-v25-cache-contract` carries unrelated WIP; the flux files are untracked there. Build with the warm `.build` here.
- **Pushable remotes:**
  - `jjang-ai/vmlx-flux` (standalone SwiftPM engine) — **all native work is pushed here** on branch `native-zimage-proven`. This is the durable home. Latest: `fc6e5b1`.
  - `osaurus-ai/vmlx-swift` (the monorepo) — z-image engine vendored + merged via **PR #63** (`codex/native-mflux-zimage`). Remote name `vmlx-origin`. (Note: the `osaurus-upstream` remote is DO_NOT_PUSH — only the mlx-swift fork.)
- **Clean commit worktree:** `/Users/eric/vmlx-swift-fluxwt` (branch `codex/native-mflux-zimage`, off `main`) — used to make clean vmlx-swift PRs without the mimo WIP.
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
cd /Users/eric/vmlx-swift
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
  - `Qwen3DVAEDecoder` = 3D causal-conv VAE **operated in 2D since T=1** (each causal Conv3d → 2D conv on the last temporal kernel slice; resamplers do spatial nearest-2× + a conv that halves channels). Per-channel `LATENTS_MEAN/STD` (16-vectors). Channel flow 384→192→192→96→3 over 3 upsamples (8×).
  - Pipeline: noise (flux-style pack, 1,hw,64) → loop[CFG: pos+neg transformer passes → guided = neg+g·(pos−neg) → FlowMatch step] → unpack → 5D → VAE decode → PNG. **timestep passed = RAW sigma** (`QwenTimesteps` applies ×1000 internally — see §6 bug 2). ~20 steps, guidance ~4 (CFG).

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
1. **qwen-image-edit:** the txt2img pipeline works. Read `/tmp/mflux-ref/src/mflux/models/qwen/variants/edit/` + `qwen_text_encoder/qwen_vision_*` + `tokenizer/qwen_vision_language_tokenizer.py` (edit template, `edit_template_start_idx=64`, `Picture N:` image prefix, `<|vision_start|><|image_pad|><|vision_end|>`). Add the Qwen2.5-VL **vision transformer** (`qwen_vision_*`) → image features spliced into the text-token stream at `image_token_id=151655`; VAE-encode the source image to a conditioning latent; concat to the noise latent. Download `fcreait/Qwen-Image-Edit-mflux` (87GB) or find a quantized edit bundle.
2. **Ideogram 4:** download `ideogram-ai/ideogram-4-nf4`. Port = Qwen3 text encoder (close to the qwen LM encoder) + 34-layer DiT (emb 4608, 18 heads, `llm_features 4096×13` = multi-layer Qwen3 hidden states, rope θ5e6) + VAE. **Build an fp8 dequant/matmul path** in `MFluxStore` (the transformer is fp8, not group-quant). Ref: `/tmp/mflux-ref/src/mflux/models/ideogram4/`.
3. **Full precision** flux/z-image: download, run the probe — existing pipelines (`MFluxLinear` handles non-quant). Should just work.
4. **Consolidated osaurus PR:** rebase `codex/native-mflux-zimage` onto current `vmlx-origin/main`, copy the new model files in (remember `import Tokenizers`→`import VMLXTokenizers` for the monorepo), verify build (Swift 6), open PR to main. The mlx-swift / swift-transformers fork pins must match `../vmlx-swift-lm` (mlx-swift `0a56f904`, swift-transformers osaurus fork `087a66b1`) — see vmlx-flux Package.swift.

**Reference:** the mflux Python source (the source of truth for every arch + weight key) is at `/tmp/mflux-ref` (clone of `github.com/filipstrand/mflux`). Re-clone if gone.

---

## 10. GH PR / commit references
- `osaurus-ai/vmlx-swift` **PR #63** — z-image engine vendored + merged to main (`36aebd42→90e64687`).
- `jjang-ai/vmlx-flux` branch **`native-zimage-proven`** — all native work: `9915417` (z-image vendor+proof), `4a88089` (resolution fix), `a2c1a28` (flux-schnell working), `f82dd1b` (probe flags), `fc6e5b1` (qwen-image working + ideogram scaffold). Open a PR from this branch to vmlx-flux main when ready.
- Wiki note (private `jjang-ai/wiki`): `notes/2026-06-15-vmlx-flux-native-z-image-proven-fork-lockstep.md`.
- Per-project memory: `~/.claude/projects/-Users-eric-vmlx-swift/memory/vmlx-flux-native-zimage-integration.md`.
- Proof artifacts (gitignored): `docs/local/vmlx-flux-{outputs,probes}/` (PROOF-*, FLUX-proof, QWEN-proof, Q8b-*).
