# Workflow media spec implementation

Active scope: the entire https://gist.github.com/tpae/bb9f4a8384abe07446448128bac24c7b, authorized 2026-10-14. Each item requires engine implementation, Osaurus consumption, automated contract tests and affected live-app evidence before promotion. No release is part of this lane.

## Ledger

| Item | Status | Remaining acceptance |
| --- | --- | --- |
| 1.1 cancellable upscale | Delivered for SeedVR2 3B | Reuse existing app cancellation/recovery proof |
| 1.2 native batch | Engine600 and app3098 merged | Complete; reuse byte parity, encoder count, cancellation and app evidence |
| 1.3/1.4 contracts | Engine601/app3099 implemented | Qwen edit rounding replay passed; final app adoption pending |
| 1.5 samplers | Engine602/app3099 implemented | Actual non-Turbo Euler/Heun comparison passed |
| 1.6 adapters | Engine605/app3101 implemented | Real adapter zero/cache/removal parity passed; integrated app controls |
| 1.7 previews | Engine603/app3100 implemented, corrected in607 | Z512 3 previews, byte parity, paired ratio0.969 passed; integrated app |
| 1.8 memory | Engine604 implemented, calibration corrected in607 | Z/Qwen2.1 plus Flux/Ideogram/legacy Qwen measured; SeedVR2 512/1024/4096 calibration passed; final app admission pending |
| 1.9 local video | Engine607/app3103 implemented | Real T2V49-frame480p and I2V9-frame512×288 rendered; cancel/exact duration passed; app pending |
| Stage2 | Engine606 implemented | Parity/reuse/mix/cancel/stale/hires/lifetime passed; reuse recorded evidence |

Preserve existing seeds and independent random streams. Do not confuse reported sampler metadata with selectable implementations. Do not advertise a scaffold as an implemented backend. Keep Turbo's bundle-defined schedule. Existing singleton image admission and chat-model unload/restore remain authoritative. A source-only or RunBench row is not live-app proof.

Private evidence and historical proof: `~/vmlx-private-evidence/flux-workflows-20261014/`. Full-scope source starts on fresh main in `feat/flux-full-spec-oct14`, using the existing isolated engine/app worktrees; prior dirty default checkouts are untouched. Allosaurus PRs599/3097 remain separately held and their evidence is preserved.

## Remaining-family parameter contracts
Implemented engine limits and effective output metadata for the existing Flux Schnell, Z-Image, Qwen-Image, Qwen-Image-Edit and Ideogram pipelines. Unsupported registry scaffolds still expose no invented contract. Explicit seeds keep the existing noise paths; unspecified seeds are resolved once and reported for replay. Guidance thresholds describe the actual negative-conditioning math (Z-Image >0; Qwen always runs both branches).

Focused limit validation and backward-compatible decoding pass. Real512² seeded replay is byte-exact for Flux Schnell4steps, Z-Image4steps, Qwen30steps and Ideogram20steps; each checks emitted step count and actual PNG size. Qwen edit's 1000-to-992 rounding replay also passed with byte-identical output; see the dated calibration receipt below. Raw evidence: ~/vmlx-private-evidence/flux-workflows-20261014/*contract-replay.log. Engine evidence does not imply Osaurus family-wide parameter adoption is complete.

User changed the proof cadence: use focused change-specific checks and reuse prior evidence, then one integrated live application check per completed stage. Do not repeat unrelated whole-model matrices for every edit.

### Sampling controls (stage 1.5, engine implementation)

Requests and model limits now carry sampler, scheduler and shift choices. Qwen-Image-2.1 implements Euler and Heun, its native schedule and a linear schedule, and an explicit exponential shift. Turbo advertises only Euler with its bundle sigma grid; unsupported overrides return typed errors. Other implemented families advertise their existing Euler/native schedule, without claiming alternate samplers or shift support.

The default Euler arithmetic and seed sequence are preserved. Two focused tests cover native shift replay and bundle-specific validation; the full package compiles. Actual non-Turbo Euler/Heun output passed in the dated acceptance receipt below. Osaurus controls/API forwarding are implemented; integrated app acceptance remains pending. Existing image batching proofs are reused; they do not prove the new Heun path.

Per the latest user direction, implementation takes priority over repeated proof runs: run focused tests when behavior changes, retain existing evidence, and perform one integrated live app check per completed stage. Expand verification only for a failure or material risk.

### Preview implementation (stage 1.7)

Optional request preview cadence and maximum pixels now reach every currently implemented native generation/edit pipeline. The real VAE decodes a reduced latent. Graph submission stays on the sampling lane; PNG encoding runs on a utility worker with one in-flight preview per request. Busy work and memory pressure drop previews. Outputs stay below 64 KiB, within the requested size, and stop before terminal events. Defaults remain off.

The engine target compiles. The corrected real preview run emitted three bounded previews, retained final PNG byte parity and measured a 0.969 wall-time ratio; see the dated receipt and its measurement limits below. Unsupported/stub model entries are not claimed as implemented.

### Request memory estimates (stage 1.8, calibration pending)

Generation/edit/upscale request queries now resolve actual dimensions, native parallel width, reference conditioning and preview workspace. Weight payload bytes are read from the same shard groups as WeightLoader without loading tensors. The API also accepts a model URL for pre-load admission callers.

The per-family activation/VAE coefficients are calibrated estimates, not hard bounds. The measured families and sizes are listed below; arbitrary dimensions/adapters/devices do not have a universal ±15% guarantee. SeedVR2 calibration is also recorded below. Allocator cache and other resident models remain separate host accounting concerns.

### 1.6 Request-scoped LoRA adapters (implementation)

`ImageGenRequest` and `ImageEditRequest` accept local safetensor adapters with scale and transformer/text-encoder/both targeting. Native MFlux and Z-Image linear layers apply low-rank deltas without changing base packed weights. PEFT A/B and Kohya down/up pairs are matched against actual registered layer names and dimensions; unsupported targets/tensors fail explicitly. Qwen edit's separately constructed VLM text encoder is not an adapter target yet.

An engine operation lease keeps load/unload and concurrent generate/edit operations from replacing adapter state while a stream is active. The last set is cached by file SHA256, scale and target. Scale zero takes the base arithmetic path and does not open an adapter file. Clearing the request removes deltas; it never subtracts rounded fused weights. `adapterLoadCount` counts actual installations. Generation/edit through FluxEngine is the supported adapter entry point.

Compilation and the real Z-Image adapter runtime checks pass: scale-zero/base byte parity, repeated-set load count and removal parity are recorded below. This is engine runtime evidence; Osaurus adapter wiring still needs integrated app acceptance. No seed changes.

### Stage 2 — native Qwen-Image-2.1/Turbo handles (implementation)

The engine owns opaque conditioning and spatial latent handles. `encodePrompt`, `encodeImage`, `emptyLatent`, weighted `mix`, `sample`, nearest/bilinear/bislerp `resize`, PNG `decode`, and typed `release` are implemented for Qwen-Image-2.1/Turbo against the same native sampling/VAE code as normal generation. Other families currently return an explicit unsupported-stage error.

Prompt cache keys include adapter content identity; changing an adapter after encoding invalidates sampling rather than pairing old text conditioning with new weights. Handles are cleared on model replacement/unload. Operations share the engine lease and never install a second model copy through the stage loader. Sampling builds a new output and leaves the input latent available after cancellation. A zero-denoise request returns the original latent values. Multiple latent images are sampled sequentially with the same base-seed-plus-index contract as generation.

Stage sample/decode estimates include retained handles, conservatively counting aliases; estimator calibration remains pending. Array buffers are evaluated before storage, and release drops engine ownership without clearing the host allocator cache. A handle shared by another handle naturally remains allocated until the last owner is released.

The engine target compiles (`stages-build.log`). The dated runtime receipts below cover byte parity, handle memory release, cancellation, hires-fix peak memory and visual output. Stage support remains limited to Qwen-Image-2.1/Turbo; this does not claim all-family support or an Osaurus workflow graph editor.

### 1.9 Native Wan 2.1 video (implementation)

Replaces the former random-weight/zero-text WAN facade with original-weight T2V/I2V inference: UMT5-XXL, full-dimension Q/K RMS normalization, three-axis rotary attention, separate image/text cross-attention for I2V, CLIP's penultimate visual layer and a real causal Conv3d VAE with per-layer temporal history. Sampling uses explicitly reported flow Euler and bundle `sample_sigmas` when supplied. Wan 2.2 dual-expert routing remains unsupported and fails at load instead of rendering noise.

Bundle layout: original `config.json`; original transformer names in `transformer/*.safetensors`; original Wan UMT5 and VAE names in `text_encoder/` and `vae/`; HF UMT5 files in `tokenizer/`; I2V additionally has original `visual.*` CLIP tensors under `image_encoder/`. Convolution arrays preserve PyTorch axes on disk and are transposed once in the native reader. `scripts/flux/convert-wan-original.py` converts the original `.pth` components with `torch.load(weights_only=True)`; it never overwrites an existing bundle. This is not support for arbitrary diffusers/MLX-video layouts without conversion.

Video requests accept sourceImage and cooperative interrupts. Completion includes actual dimensions, scheduler, steps, seed, FPS and frame count. Frame count uses `(frames-1)/4+1` latent frames. MP4 writing runs off MainActor, checks cancellation/backpressure, sets the exact end time and removes partial files on failure. The engine owns the exclusive operation lease through producer drain.

A related cancellation correction keeps the engine consuming the inner image/video stream after the outer consumer cancels, until the producer acknowledges its interrupt. Cancelling the consumer Task early would release adapter state while a producer still used it. Legacy native image denoisers now also check the shared interrupt at step boundaries.

Source reference: Apple MLX examples at `796f5b53cab69a3d48a44233ce21aae889e94a08`, video/wan2.1. License retained in `docs/third-party/mlx-examples-WAN-LICENSE.txt`.

Engine compilation and actual original-checkpoint T2V/I2V runs pass, as recorded below. The 49-frame T2V image content was inspected; exact clock correction was verified on a short clip rather than repeating generation. Osaurus local-video acceptance remains pending after the discovery and duration-validation corrections.

## Consolidated runtime update — 2026-10-15

Do not repeat passed cases from `workflow-integration.log`: Qwen Turbo4 stage/generate PNG parity at seeds7/8, conditioning reuse, zero-weight mixing, cancellation preserving its input, and stale-handle rejection. Real ostris Z-Image adapter v2 passed scale-zero parity, repeated-set one-load reuse and byte-identical base restoration.

The initial preview ratio was1.0828 (FAIL); a128px internal draft decode now produces3 previews under64KiB with byte-identical final output,10.981s versus11.330s control. This is one Debug probe pair with host compilation, not a normalized performance claim or app proof.

Z-Image8bit actual peaks:14,749,918,440 bytes at512² and25,140,704,712 at1024²; live weights after output10,814MB. The old two-feature-map estimate materially understated the untiled fp32 VAE/convolution workspace. The corrected Z-only spatial coefficient is calibrated from those two sizes. Other family estimates are still uncalibrated; do not claim the all-family ±15% gate passed.

Native original Wan2.1 T2V1.3B,49frames832×480,50Euler steps,seed7 completed in522.164s. Start/middle/end frames show the requested red sailboat on a blue lake. Interrupt after first480p step returned cancelled in0.000261s with no output. Exact MP4 duration required both movie and track time scales to be multiples of FPS: the default600Hz track rounded16fps timestamps. The corrected actual5-frame run reports0.3125s exactly. No50step regeneration was needed for this writer-only correction. Original49-frame movie has old rounded3.063333s duration and is not passed off as exact.

App3103 adds local video discovery/zero-cost jobs/API/tool/delegation, shared parent swap/restore policy and actual completion metadata. Live app acceptance remains pending; CLI video output does not establish it. Source/CI checks continue while the final distinct integration rows run.

### Follow-up acceptance and bounded stage decoding

Non-Turbo Qwen2.1 4-bit, seed7,512²,8steps, linear schedule/shift1 produced distinct Euler/Heun images with matching effective metadata. Heun output was visually inspected; the requested red teapot is present. Runtime8.288s/13.398s (Debug probe, host build active; not an app speed claim).

Hires initially produced the correct1536² image but did NOT beat fresh peak:41.2597GB versus41.2593GB. Stage VAE decode now materializes each residual block separately, reducing live convolution workspace without tiling or changing precision/math. The failed-row rerun passed at36.1011GB versus41.2593GB fresh; old/new hires PNG SHA256 both56f93c0101f0557575d713515437c48d7053dc7758f5271b38979ff80bdee37b. Releasing the final handle dropped active memory by exactly1,179,648bytes, matching `retainedBytes(for:)`. Ordinary generate/decode and seeds are unchanged.

Qwen2.1's estimate now uses its144-channel decoder instead of the96-channel encoder and the observed convolution workspace. The512²/1536² measured rows are within8%;1024²/max-size and other families remain uncalibrated. Admission wiring remains under app review; do not promote the all-family memory guarantee yet.

### Final family calibration and I2V check

Qwen-Image-Edit1000² rounding replay passed:992²,30steps,seed7,byte-identical output (620.722s combined test). Missing family memory rows then identified additional decoder-workspace underestimates: Flux1024² estimated19.087GB against32.265GB measured, and Ideogram2048² estimated31.296GB against76.615GB. The decoder coefficients now include the measured untiled convolution workspace. Metadata-only queries against the same measured runs yield errors of−2.3%/+2.4% for Flux512/1024;−3.3%/−3.3%/+2.5% for Ideogram512/1024/2048;−3.9%/+0.9% for Qwen512/1024;−2.5%/+1.2% for QwenEdit512/1024. This calibration does not claim a guarantee for arbitrary dimensions, adapters or devices. Adapter payload bytes are now included instead of omitted. SeedVR2 remains the uncalibrated family.

Original Wan-AI/Wan2.1-I2V-14B-480P revision6b73f84e66371cdfe870c72acd6826e1d61cf279 passed a real source-image run:512×288,9frames,16fps,20Euler steps,guidance5,seed7,123.890s. Duration0.5625s is exact; start/middle/end frames preserve the red teapot source with camera motion. Peak80.796GB. This is engine I2V evidence, not app or480p I2V proof. Raw wan-i2v-source.log and contact.png remain private.

### Final SeedVR2 calibration and corrected app replay

SeedVR2's tiled decoder was incorrectly estimated as an untiled two-feature-map VAE. Real output sizes 512/1024/4096 measured MLX peaks 15,490,898,234 / 15,605,734,466 / 15,889,849,410 bytes. The corrected model accounts for the bounded 512px convolution workspace, whole-output buffers, and the transformer's 2x2 patch packing after VAE/8. Metadata-only estimates against those same completed runs are 15,495,044,870 / 15,608,291,078 / 18,153,529,094 bytes: +0.027%, +0.016%, +14.25%. No inference was repeated to evaluate the estimator correction. Raw `memory-seedvr.log`, `seedvr-corrected-estimates.log`, and `seedvr-memory-comparison.json` remain in the private evidence directory. This is MLX peak calibration, not a bound on host color-processing allocations or physical footprint.

Osaurus Release head 128edcaa0, engine 3a0674611, replayed the failed native video scenario successfully: Diplodocus Orchestrator spawned Raptor Helper, the video tool completed in 8.18s, the parent displayed the MP4 artifact and resumed with Diplodocus loaded. File inspection confirms 160x96, 5 frames, 16fps, exactly 0.3125s. The model initially omitted the required duration; typed validation rejected that call, and the next call supplied it. No sampling or prompt override was added. A separate existing file-claim advisory incorrectly retried the successful media answer once; the app follow-up now recognizes completed media artifact receipts. That classification correction is covered by focused app tests, not a repeated video-generation matrix.
