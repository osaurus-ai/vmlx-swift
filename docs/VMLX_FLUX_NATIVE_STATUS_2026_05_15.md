# vMLX Flux Native Status - 2026-05-15

This records the current state of native MFlux/Flux-family image support in
the consolidated `vmlx-swift` package. It is based on source inspection plus
live probes against local bundles under `~/.mlxstudio/models/image`.

## Integrated Surface

- `jjang-ai/vmlx-flux` was cloned and inspected as the partial standalone
  Swift project for native MFlux-style image/video generation.
- Its Swift targets are now vendored into this package as local products:
  `vMLXFlux`, `vMLXFluxKit`, `vMLXFluxModels`, and `vMLXFluxVideo`.
- The umbrella `VMLX` product re-exports `vMLXFlux`.
- `vmlxflux-probe` is a local executable for scan/load/generation probes. It
  writes artifacts under ignored `docs/local/vmlx-flux-probes/`. Use
  `--matrix` to scan, load, and run the multi-turn generation probe across all
  detected local image bundles.
- `MLXStudioModelStore` scans `~/.mlxstudio/models/image` and resolves local
  Diffusers/MFlux component layouts before any silent download path is
  considered. Exact local directory names win over canonical family aliases.

## Local Bundle Inventory

Latest scan artifact:
`docs/local/vmlx-flux-probes/20260516T-native-mflux-full-matrix/scan.json`

Latest load/generation matrix:
`docs/local/vmlx-flux-probes/20260516T-native-mflux-full-matrix/compatibility-matrix.json`

| Local directory | Native family | Bytes | Runtime status |
| --- | --- | ---: | --- |
| `FLUX.2-klein-9B` | `flux2-klein` | 52,864,013,750 | `not_implemented` |
| `qwen-image-mflux-4bit` | `qwen-image` | 25,895,424,132 | `not_implemented` |
| `Z-Image-Turbo` | `z-image-turbo` | 32,832,339,790 | `scaffold_generates_png_noise` |
| `Z-Image-Turbo-mflux-4bit` | `z-image-turbo` | 5,891,426,229 | `scaffold_generates_png_noise` |

All four bundles have enough component files to be discovered as local image
bundles. None are production-compatible yet.

## Live Probe Results

- `Z-Image-Turbo-mflux-4bit` load/generate:
  `docs/local/vmlx-flux-probes/20260516T-native-mflux-full-matrix/`
  completed three PNG turns at 128x128, but output is colored noise. This is
  not a compatibility pass.
- `Z-Image-Turbo` load/generate:
  `docs/local/vmlx-flux-probes/20260516T-native-mflux-full-matrix/`
  completed three PNG turns at 128x128, also through the same scaffold path.
- `qwen-image-mflux-4bit`:
  `docs/local/vmlx-flux-probes/20260516T-native-mflux-full-matrix/`
  loads the local bundle but every generation turn throws
  `FluxError.notImplemented`.
- `FLUX.2-klein-9B`:
  `docs/local/vmlx-flux-probes/20260516T-native-mflux-full-matrix/`
  loads the local bundle but every generation turn throws
  `FluxError.notImplemented`.

Matrix result:

| Local directory | Load | Multi-turn generation | Gate |
| --- | --- | --- | --- |
| `FLUX.2-klein-9B` | `loaded` | `0/3` | `blocked_after_load` |
| `qwen-image-mflux-4bit` | `loaded` | `0/3` | `blocked_after_load` |
| `Z-Image-Turbo` | `loaded` | `3/3` noise PNGs | `blocked_after_load` |
| `Z-Image-Turbo-mflux-4bit` | `loaded` | `3/3` noise PNGs | `blocked_after_load` |

## Current Blockers

- Z-Image has an end-to-end PNG-producing scaffold, but it feeds zero-tensor
  text embeddings and does not apply loaded safetensors into `FluxDiTModel` or
  `VAEDecoder`.
- The current `loaded` status means the native constructor accepted the local
  directory and opened safetensors metadata/arrays. It does not prove real
  prompt-conditioned weights are resident or used during generation.
- Flux2 Klein and Qwen-Image model bodies still throw `notImplemented`.
- Shared T5/CLIP text encoder ports are missing.
- Safetensors-to-module key mapping is missing.
- Flux 3-axis RoPE is still TODO in `FluxDiT`.
- `swift package describe --type json`, `swift build --target vMLXFlux`, and
  `swift build --product vmlxflux-probe` succeed.
- Full `swift test --filter vMLXFluxTests` is currently blocked before reaching
  the Flux tests by test-module availability on this toolchain: `XCTest` is
  missing for `CmlxTests` and `Testing` is missing for `MLXPressPolicyTests`.

## Next Engineering Path

1. Implement real safetensors key mapping and `Module.update` for the smallest
   runnable path first: `Z-Image-Turbo-mflux-4bit`.
2. Port or reuse the required text encoder path so prompts affect conditioning.
3. Replace the scaffold Z-Image pass criteria with image-output checks that
   reject noise/prompt-insensitive rows.
4. Only after Z-Image is prompt-sensitive, port Flux2 Klein and Qwen-Image
   model bodies and run the same probe matrix on their local bundles.

Until those rows pass with real prompt-sensitive images, the status is:
`not production-compatible`.
