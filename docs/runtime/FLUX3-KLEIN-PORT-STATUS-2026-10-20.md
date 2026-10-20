# FLUX 3 Image Klein checkpoint — round 2, 2026-10-20

**PARTIAL: native source prepared, GPU quality and Osaurus GUI unproven.**
The round-2 GPU run is authorized, but this tool session exposes no Metal device.
The local mflux import fails before model construction. Clean Swift/Osaurus tests
also stop at GitHub DNS/dependency resolution. No generated PNG or measured image
throughput exists for this port yet.

Report: `<local path>`.
Reference: `<local path>` at `d587aac3d999eb9b2b8508a4481e65672d4b4ba4`.
Contract: `the FLUX 3 Klein integration guide/INTEGRATION-VMLX-SWIFT-OSAURUS.md`.

## Prepared source

- Native quant-aware Qwen3 language encoder, FLUX 3 transformer, current/target
  time embeddings, flow-map Euler scheduler and AutoencoderKLFlux2 encode/decode.
- Seeded BF16-rounded noise with Float32 Euler state; single/multiple ordered
  VAE references, reference layer IDs, first-reference unsized output and crop-back.
- Stamp/header preflight before tensor loading, effective parameters, cancellation,
  PNG output and optional previews. LoRA remains optional and unadvertised.
- Osaurus direct composer, image panel, tool/delegated job parameter paths and
  advisory estimates; hidden negative/guidance controls, ignored negative prompts,
  custom chat/panel sizes, typed chat steps and installed-bundle generation defaults.

## Current evidence

- **17/17** Foundation/CPU Swift tests pass, including every legal side length
  64...4096 by 16, variable steps, local 4/6/8-bit bundle headers, Pillow pixel
  fixtures, EXIF/RGB decoding, scheduler/geometry and negative-prompt policy.
- **20/20** reduced Osaurus contracts pass: 11 XCTest cases (including four
  actual delegated argument-decoder tests) and nine existing Qwen bridge tests.
  The reduced runner excludes actor/UI execution and model resolution.
- Native Swift Models/facade static compilation and the new opt-in GPU test
  source typecheck pass against disclosed cached MLX/Common dependencies.
- The offline comparator passes CPU fixture checks for F32/BF16 tensors,
  RGB8 MAE and mismatched-seed rejection. This is not model parity evidence.

## Remaining gates

1. Orchestrator commits the uncommitted Swift source as Eric. Git metadata is
   read-only here. Then run Osaurus `scripts/pin-flux3-worktree.py --apply` to
   update the manifest and all three lockfiles to that actual HEAD. The script
   refuses an uncommitted port; the current old pin is not a FLUX 3 pin.
2. Restore exact dependencies/submodules and run clean focused Swift/Osaurus
   tests, then build a fresh app. Cached-module compilation does not prove a
   clean pinned C++ build or app link.
3. Run `tools/flux3/README.md` in a Metal-capable session: generation and edit
   at 512², four steps, seed 42, then ordered multiple references. Each backend
   records noise/text/reference/step-0/final tensors, PNGs and timings in a fresh
   process. Compare matching inputs and inspect images; free GPU memory between runs.
4. Complete the report's GUI checklist: all three quants, direct composer,
   single/multiple/unsized edits, explicit size/steps/seeds, ignored negative
   prompts, chat image tool, spawned child image jobs, cancellation and parent
   continuation. Record effective parameters, physical footprint and throughput.

All source stays uncommitted. No push, PR, tag, release, Hub write, BF16/source
bundle load or Osaurus HTTP request was performed.
