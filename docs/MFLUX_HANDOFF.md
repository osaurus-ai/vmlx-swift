# vMLX-Flux (native mFLUX image gen) — HANDOFF

**For:** the next engineer/agent continuing the native mFLUX image-generation port.
**Date:** 2026-06-16. **Owner:** Eric. **Status:** z-image-turbo and flux-schnell
are live-proven for 4/8-bit; qwen-image 4-bit is live-proven; qwen-image-edit
q4/q5 are live-proven for text-image edit after the conditioning-grid fix. qwen-edit
q3 is incomplete (`text_encoder/3.safetensors` missing from its index), q6 is incomplete on disk, masks are
not wired, and Ideogram has no complete staged local bundle/proof because the
current HF account is not approved for `ideogram-ai/ideogram-4-nf4` or
`ideogram-ai/ideogram-4-fp8`.

**2026-06-16 continuation evidence:** live baseline probes were rerun from
`/Users/eric/vmlx-swift` so MLX could resolve `default.metallib`; the standalone
repo itself does not contain that Metal library. Fresh standalone artifacts are
under `docs/local/vmlx-flux-{probes,outputs}/2026-06-16-*`. The Osaurus
monorepo worktree also has fresh load artifacts from `/Users/eric/vmlx-swift-fluxwt`:
`docs/local/vmlx-flux-probes/2026-06-16-osaurus-qwen-image-q4-load-final/qwen-image-mflux-4bit-load.json`
(`load_status=loaded`, `native_runtime_status=native_pipeline_implemented`) and
`docs/local/vmlx-flux-probes/2026-06-16-osaurus-qwen-edit-q4-load-final/Qwen-Image-Edit-mflux-q4-load.json`
(`load_status=loaded`, pre-fix `native_runtime_status=native_pipeline_partial`).

**2026-06-16 Osaurus PR #64 live-proof refresh:** rerun from
`/Users/eric/vmlx-swift-fluxwt` on branch `codex/mflux-qwen-edit-main` at
pre-doc commit `81fae70e`. Text-to-image artifacts:
`docs/local/vmlx-flux-probes/2026-06-16-goal-zimage-4bit-explicit-gen/Z-Image-Turbo-mflux-4bit-load.json`,
`docs/local/vmlx-flux-probes/2026-06-16-goal-zimage-8bit-explicit-gen/Z-Image-Turbo-mflux-8bit-load.json`,
`docs/local/vmlx-flux-probes/2026-06-16-goal-flux-schnell-4bit-explicit-gen/FLUX.1-schnell-mflux-4bit-load.json`,
`docs/local/vmlx-flux-probes/2026-06-16-goal-flux-schnell-8bit-explicit-gen/FLUX.1-schnell-mflux-8bit-load.json`,
and
`docs/local/vmlx-flux-probes/2026-06-16-goal-qwen-image-4bit-gen20/qwen-image-mflux-4bit-load.json`.
Viewed contact sheet:
`docs/local/vmlx-flux-outputs/2026-06-16-goal-contact-sheet-explicit-turn-order.png`.
Qwen-edit q4 pre-fix apple-blue artifact:
`docs/local/vmlx-flux-probes/2026-06-16-goal-qwen-edit-q4-apple-blue/Qwen-Image-Edit-mflux-q4-load.json`
(`edit_turns[0].status=completed`, output SHA
`5fc2b04436eb0a8ad0e7a61265f962ec0dc67027efa417804bc39c80ab37cb13`);
viewed output is source-like and failed the requested edit. Root cause was the
edit conditioning grid: Swift used 1024-area VAE dimensions for static source
latents, but mflux passes `vl_width/vl_height` and encodes those latents at the
VL size. Post-fix proof:
`docs/local/vmlx-flux-probes/2026-06-16-qwen-edit-q4-determinism-after-cond-fix/Qwen-Image-Edit-mflux-q4-load.json`
(`load_status=loaded`; turn 1 and turn 3 same blue-apple prompt SHA
`005ab8baddfe9b7a94aa83f8ddd22d192e7e5a0275c556dcf2ead76a565e474a`;
turn 2 green-pear prompt SHA
`815711be73a9e89599b3e97f9f15196115875103f9407d7b1b61bab33de8e3b4`).
The viewed PNGs are coherent/prompt-sensitive and the repeated prompt is
byte-identical. Shape proof:
`docs/local/vmlx-flux-probes/2026-06-16-qwen-edit-q4-conditioning-after-cond-fix/Qwen-Image-Edit-mflux-q4-load.json`
(`conditioning_width=384`, `conditioning_height=384`, `patch_rows=24`,
`patch_columns=24`, `latents_shape=1x576x64`) and
`docs/local/vmlx-flux-probes/2026-06-16-qwen-edit-q4-denoise-after-cond-fix/Qwen-Image-Edit-mflux-q4-load.json`
(`target_latent_count=1024`, `conditioning_latent_count=576`,
`combined_velocity_shape=1x1600x64`, `image_shapes=[[1,32,32],[1,24,24]]`).
Current local scan: `docs/local/vmlx-flux-probes/2026-06-16-goal-current-scan/scan.json`.
No complete Ideogram bundle was found under the scanned local image-model roots.
The HF dry-runs left local `ideogram-4-nf4` and `ideogram-4-fp8` asset/cache
skeleton directories only; the scanner reports them as incomplete Ideogram rows.
Ideogram has no live load/generation evidence on this machine. `hf download`
dry-runs for both `ideogram-ai/ideogram-4-nf4` and `ideogram-ai/ideogram-4-fp8`
returned `Access denied. This repository requires approval.` on 2026-06-16.

This is the single starting doc. Read it top to bottom, then the per-model port plans.

---

## 0. TL;DR — what works, what's next

| Model | 4-bit | 8-bit | full | Native pipeline file |
|---|---|---|---|---|
| **z-image-turbo** | ✅ proven | ✅ proven | ⬜ (weights gone) | `Libraries/vMLXFluxModels/ZImage/ZImageNative.swift` |
| **flux-schnell** | ✅ proven | ✅ proven | ⬜ (not staged) | `Libraries/vMLXFluxModels/Flux1/Flux1Native.swift` |
| **qwen-image** (txt2img) | ✅ proven | ⬜ | ⬜ | `Libraries/vMLXFluxModels/Common/QwenImageNative.swift` |
| qwen-image-edit | ✅ q4/q5 text-image edit proven; q3/q6 incomplete | — | — | `Libraries/vMLXFluxModels/QwenImage/QwenImageEditSupport.swift`; masks/inpaint pending |
| ideogram (4) | ⬜ scaffold, no local bundle staged/proven | — | — | `Libraries/vMLXFluxModels/Ideogram4/Ideogram4.swift` (fp8/nf4 path missing) |
| flux1-dev/kontext/fill, flux2-klein, fibo, seedvr2, wan | ⬜ scaffold | — | — | registered, throw `notImplemented` |

"Proven" = live-generated a coherent, prompt-accurate image that is **deterministic** (same seed+prompt -> byte-identical) and **prompt-sensitive** (different prompt same seed -> different coherent image). Per Eric's HARD RULE: *do not trust/claim a model works until you have generated and visually checked a real image.* 2026-06-16 rerun: z-image 4/8 and flux-schnell 4/8 passed live load + three-turn generate + SHA determinism/prompt-sensitivity + visual inspection. Qwen-image 4-bit now also passed live load + 20-step generation + three-turn SHA determinism/prompt-sensitivity + visual inspection after the mflux guidance rescale fix; turn 1/3 apple SHA `2f1c27c68993fe9a537bca2cc019ac3d32d59818b92c606c00726104661bcea7`, turn 2 mountain SHA `2bf77ce59c8ed99c1b1aa5fb8940c9d35948b1763fbd360e14f577032b62f060`, artifact `docs/local/vmlx-flux-probes/2026-06-16-qwen-image-q4-guidance-proof/qwen-image-mflux-4bit-load.json`.

Qwen-image-edit q4/q5 are live-proven after fixing the source-image conditioning grid to match mflux. Source trace: mflux `qwen_image_edit.py` passes `vl_width/vl_height` into `QwenEditUtil.create_image_conditioning_latents`, and `qwen_edit_util.py` uses those VL dimensions for the source-image VAE encode when present. Swift now mirrors that in `QwenImageEditSupport.swift`: square source images encode conditioning at 384x384, pack 24x24=576 static latents, and denoise with 1024 target latents + 576 conditioning latents. q4 live proof artifact: `docs/local/vmlx-flux-probes/2026-06-16-qwen-edit-q4-determinism-after-cond-fix/Qwen-Image-Edit-mflux-q4-load.json` (blue prompt SHA `005ab8baddfe9b7a94aa83f8ddd22d192e7e5a0275c556dcf2ead76a565e474a`, green-pear prompt SHA `815711be73a9e89599b3e97f9f15196115875103f9407d7b1b61bab33de8e3b4`, repeated blue prompt same SHA). q5 live proof artifact: `docs/local/vmlx-flux-probes/2026-06-16-qwen-edit-q5-determinism/Qwen-Image-Edit-mflux-q5-load.json` (blue prompt SHA `5cd5d9197bd659bd8b59b4a2f2bca413266146ad4e08249289d5fa6a8025fa4e`, green-pear prompt SHA `d2c6c4eb4a19dcf48122b5216fc15ac37b9f5aa49c15f596acd1276a4df57034`, repeated blue prompt same SHA). Viewed q4/q5 outputs are coherent and prompt-sensitive. Boundary artifacts remain useful: `docs/local/vmlx-flux-probes/2026-06-16-qwen-edit-q4-prompt-live/Qwen-Image-Edit-mflux-q4-load.json`, `docs/local/vmlx-flux-probes/2026-06-16-qwen-edit-q4-vl-encode-live/Qwen-Image-Edit-mflux-q4-load.json`, `docs/local/vmlx-flux-probes/2026-06-16-qwen-edit-q4-conditioning-after-cond-fix/Qwen-Image-Edit-mflux-q4-load.json`, and `docs/local/vmlx-flux-probes/2026-06-16-qwen-edit-q4-denoise-after-cond-fix/Qwen-Image-Edit-mflux-q4-load.json`.

**Next work, in priority order:**
1. **Ideogram 4** — get HF approval for `ideogram-ai/ideogram-4-nf4` or `ideogram-ai/ideogram-4-fp8`, stage one local bundle, then implement the **fp8/nf4 quant path** (different from the MLX group-quant used by the others) + Qwen3 encoder + 34-layer DiT + unconditional transformer.
2. **qwen-image-edit follow-through** — wire masks/inpaint semantics. q3/q6 need complete local bundles before they can be exposed.
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
  The scanner now expands these as `Qwen-Image-Edit-mflux-q3/q4/q5/q6`; q4 and
  q5 have complete tokenizer/text_encoder/transformer/vae bundles and are
  live-proven. q3 is incomplete because `text_encoder/model.safetensors.index.json`
  references missing `text_encoder/3.safetensors`; q6 is incomplete on the
  current disk image (`missing transformer`, `missing vae`, and missing indexed
  text-encoder shards). `Qwen-Image-Edit-mflux-q4` also passes
  the manifest-gated engine load contract (tokenizer files + Qwen LM keys +
  Qwen-VL vision keys + transformer keys + VAE encode/decode keys) and a live
  source-image preprocess request (`output=1024x1024`, `vl=384x384`,
  `vae=1024x1024`, `conditioning=24x24`, `vision_patches=784x1176`,
  `vision_grid=1x28x28`, `vae_input=1x3x384x384`). The q4 prompt probe
  tokenizes the mflux edit prompt with 196 repeated image-pad tokens
  (`input_ids_shape=1x276`, `image_token_id=151655`, `image_token_count=196`,
  `template_drop_index=64`). The q4 conditioning probe live-encodes the source
  through the Qwen 3D VAE at the VL size and packs static image latents
  (`conditioning_width=384`, `conditioning_height=384`,
  `latents_shape=1x576x64`, `image_ids_shape=1x576x3`, finite stats). The q4 VL
  encode probe runs the real Qwen2.5-VL vision transformer and image-token
  splice (`feature_shape=196x3584`, `prompt_embeds_shape=1x212x3584`,
  `token_image_count=196`, finite stats). The q4 denoise probe concatenates
  1024 target latents with 576 static conditioning latents, runs the edit-shaped
  transformer RoPE grids, and slices the target velocity
  (`combined_velocity_shape=1x1600x64`, `target_velocity_shape=1x1024x64`,
  finite stats). `ImageEditor` runs the scheduler loop, decodes, and writes
  coherent prompt-sensitive PNGs for q4. `Qwen-Image-Edit-mflux-q5` also has
  live same-seed text-image edit proof at
  `docs/local/vmlx-flux-probes/2026-06-16-qwen-edit-q5-determinism/Qwen-Image-Edit-mflux-q5-load.json`.
  q3/q6 require complete local bundles before UI promotion, and masks are not
  wired.

**Downloadable mflux-compatible weights (HF):**
- flux: `dhairyashil/FLUX.1-schnell-mflux-{4,8}bit`; full = `black-forest-labs/FLUX.1-schnell` (GATED).
- z-image: `Tongyi-MAI/Z-Image-Turbo` (full), `carsenk/z-image-turbo-mflux-8bit`, `filipstrand/Z-Image-Turbo-mflux-4bit`.
- qwen: `carsenk/qwen-image-mflux-4bit` (txt2img), `fcreait/Qwen-Image-Edit-mflux` (87GB full edit model).
- ideogram: `ideogram-ai/ideogram-4-fp8` (the mflux canonical), `ideogram-ai/ideogram-4-nf4` (4-bit).
  Current account state: both repos are visible through `hf models info`, but
  `hf download --dry-run` is approval-gated (`Access denied. This repository
  requires approval.`), so live Ideogram load/generation is blocked until access
  is granted.

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
# Current qwen-edit q4 status: live load + ImageEditor loop + PNG write +
# coherent same-seed prompt-sensitive edit proof after the VL-grid conditioning fix.

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
  - `Qwen3DVAEEncoder`/`Qwen3DVAEDecoder` = 3D causal-conv VAE **operated in 2D since T=1** (each causal Conv3d → 2D conv on the last temporal kernel slice; decoder resamplers do spatial nearest-2× + conv; encoder downsamplers pad bottom/right then stride-2 conv). Per-channel `LATENTS_MEAN/STD` (16-vectors). Decoder channel flow 384→192→192→96→3 over 3 upsamples (8×). Qwen-edit q4/q5 VAE conditioning encode/pack and edit-loop PNG output are live-proven after the VL-grid conditioning fix; masks and incomplete q3/q6 bundles are pending.
  - Pipeline: noise (flux-style pack, 1,hw,64) → loop[CFG: pos+neg transformer passes → mflux guidance rescale (`combined = neg + g*(pos-neg)`, then rescale to positive-noise norm) → FlowMatch step] → unpack → 5D → VAE decode → PNG. **timestep passed = RAW sigma** (`QwenTimesteps` applies ×1000 internally — see §6 bug 2). ~20 steps, guidance ~4 (CFG).

Full per-model transcription specs are in `docs/FLUX_SCHNELL_PORT_PLAN.md` and `docs/QWEN_IMAGE_PORT_PLAN.md` (grounded from the mflux Python source).

---

## 6. Bugs found & fixed (don't reintroduce these)
1. **Conv weight layout.** mflux stores conv weights in **MLX channels-last** `(out, [kt,] kh, kw, in)`, NOT PyTorch `(out, in, k...)`. Assuming PyTorch → wrong reshape/transpose → load-time crash (`reshape 442368→(1152,1)`). Linear weights ARE PyTorch `(out,in)`.
2. **Qwen timestep double-scale.** mflux passes the raw sigma to `QwenTimesteps(scale=1000)` which multiplies internally. Passing `sigma×1000` double-scales → the transformer denoises to pure noise. Pass the **raw sigma**.
3. **Qwen-edit conditioning grid.** mflux computes a 1024-area VAE plan for the output target but passes `vl_width/vl_height` into `QwenEditUtil.create_image_conditioning_latents`; when present, the source-image conditioning VAE encode uses those VL dimensions. Swift must use `vlWidth/vlHeight` for qwen-edit conditioning latents, not `vaeWidth/vaeHeight`. The fixed square q4 path is 384x384 -> 24x24 -> 576 static tokens, not 1024x1024 -> 64x64 -> 4096 tokens.
4. **Indexed safetensor completeness.** A component with “some safetensors” is not enough. `Qwen-Image-Edit-mflux-q3` looked loadable until live load failed on missing `text_encoder/3.safetensors`; scanner readiness now validates `*.safetensors.index.json` shard references.
5. **Model resolution / quant collision.** `MLXStudioModelStore.resolve` normalized away the `-Nbit` suffix → requesting `...-8bit` loaded a co-installed `...-4bit`. Fixed with a literal case-insensitive directory-name match first. **osaurus must request the exact bundle directory name.**
6. **Tokenizer format** — see §3 (need fast `tokenizer.json`).
7. **GPU watchdog** — MLX mmaps weights lazily; running gen with weights on a **slow volume (USB)** stalls the Metal command buffer → `kIOGPUCommandBufferCallbackErrorTimeout`. **Stage weights on the internal SSD.**

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
1. **qwen-image-edit:** the q4/q5 text-image edit paths are live-proven. Source-image conditioning now follows mflux's VL-size path (`vlWidth/vlHeight`) instead of the 1024-area VAE target grid. Current proof artifacts: `docs/local/vmlx-flux-probes/2026-06-16-qwen-edit-q4-determinism-after-cond-fix/Qwen-Image-Edit-mflux-q4-load.json`, `docs/local/vmlx-flux-probes/2026-06-16-qwen-edit-q5-determinism/Qwen-Image-Edit-mflux-q5-load.json`, `docs/local/vmlx-flux-probes/2026-06-16-qwen-edit-q4-conditioning-after-cond-fix/Qwen-Image-Edit-mflux-q4-load.json` (`latents_shape=1x576x64`, `image_ids_shape=1x576x3`), and `docs/local/vmlx-flux-probes/2026-06-16-qwen-edit-q4-denoise-after-cond-fix/Qwen-Image-Edit-mflux-q4-load.json` (`combined_velocity_shape=1x1600x64`). Next qwen-edit work is masks/inpaint semantics and complete q3/q6 bundle staging if those variants matter.
   - Current staged bundle is already present at `~/.mlxstudio/models/image/Qwen-Image-Edit-mflux`; use `Qwen-Image-Edit-mflux-q4` or `Qwen-Image-Edit-mflux-q5` for current Osaurus wiring. Keep q3/q6 hidden/blocked until their indexed shards/components are complete.
2. **Ideogram 4:** get HF approval, then stage `ideogram-ai/ideogram-4-nf4` or `ideogram-ai/ideogram-4-fp8` locally before live proof. Port = Qwen3 text encoder (close to the qwen LM encoder) + 34-layer DiT (emb 4608, 18 heads, `llm_features 4096×13` = multi-layer Qwen3 hidden states, rope θ5e6) + unconditional transformer + VAE. **Build an fp8/nf4 dequant/matmul path** in `MFluxStore` (the transformer is fp8 in the mflux canonical bundle, not group-quant). Ref: `/tmp/mflux-ref/src/mflux/models/ideogram4/`.
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
  HEAD vendors the standalone flux-schnell + qwen-image pipelines, fixes
  qwen-image-edit q4 conditioning-grid parity with mflux, and updates the
  osaurus image API / integration docs for team wiring.
- `jjang-ai/vmlx-flux` branch **`native-zimage-proven`** — all native work: `9915417` (z-image vendor+proof), `4a88089` (resolution fix), `a2c1a28` (flux-schnell working), `f82dd1b` (probe flags), `fc6e5b1` (qwen-image working + ideogram scaffold), `f7014e0` (handoff); current HEAD adds qwen-edit nested scan, manifest-gated q4 load, q4 source-image tensor preprocess proof, q4 VAE conditioning latent proof, q4 Qwen2.5-VL prompt-image encode proof, q4 first transformer velocity proof, qwen mflux guidance rescale, and q4 edit-loop PNG plumbing. Open a PR from this branch to vmlx-flux main when ready.
- Wiki note (private `jjang-ai/wiki`): `notes/2026-06-15-vmlx-flux-native-z-image-proven-fork-lockstep.md`.
- Per-project memory: `~/.claude/projects/-Users-eric-vmlx-swift/memory/vmlx-flux-native-zimage-integration.md`.
- Proof artifacts (gitignored): `docs/local/vmlx-flux-{outputs,probes}/` (PROOF-*, FLUX-proof, QWEN-proof, Q8b-*).
