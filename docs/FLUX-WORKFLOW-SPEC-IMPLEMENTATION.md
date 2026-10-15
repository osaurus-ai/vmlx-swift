# Workflow media spec implementation

Active scope: the entire https://gist.github.com/tpae/bb9f4a8384abe07446448128bac24c7b, authorized 2026-10-14. Each item requires engine implementation, Osaurus consumption, automated contract tests and affected live-app evidence before promotion. No release is part of this lane.

## Ledger

| Item | Status | Remaining acceptance |
| --- | --- | --- |
| 1.1 cancellable upscale | Delivered for SeedVR2 3B | Preserve existing cancellation/recovery proof |
| 1.3 effective parameters | Delivered for Qwen2.1/Turbo and SeedVR2 | Other implemented families; per-family replay |
| 1.4 typed limits | Delivered for Qwen2.1/Turbo and SeedVR2 | Other implemented families; validation before weight loading |
| 1.2 native batch | In progress | Encode once, indexed completions, explicit seed list, cancellation, individual-image byte parity, wall time |
| 1.5 sampler/scheduler/shift | Pending | Real selectable implementations, capability query and invalid-field errors |
| 1.7 previews | Pending | Bounded cadence/size, cancellation, overhead <=5% |
| 1.8 memory estimates | Pending | Per-request estimates within 15% of measured peaks at required dimensions |
| 1.6 adapters | Pending | Supported targets, content identity, reuse, zero-scale and removal parity |
| 1.9 video | Pending | Real weights/text encoding; 49-frame480p clip, metadata and cancellation |
| Stage2 | Pending | Conditioning/latent handles, encode/mix/resize/sample/decode/release; parity, ownership, memory release |

Preserve existing seeds and independent random streams. Do not confuse reported sampler metadata with selectable implementations. Do not advertise a scaffold as an implemented backend. Keep Turbo's bundle-defined schedule. Existing singleton image admission and chat-model unload/restore remain authoritative. A source-only or RunBench row is not live-app proof.

Private evidence and historical proof: `~/vmlx-private-evidence/flux-workflows-20261014/`. Full-scope source starts on fresh main in `feat/flux-full-spec-oct14`, using the existing isolated engine/app worktrees; prior dirty default checkouts are untouched. Allosaurus PRs599/3097 remain separately held and their evidence is preserved.

## Remaining-family parameter contracts
Implemented engine limits and effective output metadata for the existing Flux Schnell, Z-Image, Qwen-Image, Qwen-Image-Edit and Ideogram pipelines. Unsupported registry scaffolds still expose no invented contract. Explicit seeds keep the existing noise paths; unspecified seeds are resolved once and reported for replay. Guidance thresholds describe the actual negative-conditioning math (Z-Image >0; Qwen always runs both branches).

Focused limit validation and backward-compatible decoding pass. Real512² seeded replay is byte-exact for Flux Schnell4steps, Z-Image4steps, Qwen30steps and Ideogram20steps; each checks emitted step count and actual PNG size. Qwen edit's1000-to992 rounding test is implemented but live execution remains pending for the integrated stage check. Raw evidence: ~/vmlx-private-evidence/flux-workflows-20261014/*contract-replay.log. Engine evidence does not imply Osaurus family-wide parameter adoption is complete.

User changed the proof cadence: use focused change-specific checks and reuse prior evidence, then one integrated live application check per completed stage. Do not repeat unrelated whole-model matrices for every edit.

### Sampling controls (stage 1.5, engine implementation)

Requests and model limits now carry sampler, scheduler and shift choices. Qwen-Image-2.1 implements Euler and Heun, its native schedule and a linear schedule, and an explicit exponential shift. Turbo advertises only Euler with its bundle sigma grid; unsupported overrides return typed errors. Other implemented families advertise their existing Euler/native schedule, without claiming alternate samplers or shift support.

The default Euler arithmetic and seed sequence are preserved. Two focused tests cover native shift replay and bundle-specific validation; the full package compiles. Actual alternate-sampler output, Osaurus controls/API forwarding, and integrated app verification are pending. No model generation was run for this change yet. Existing image batching proofs are reused; they do not prove the new Heun path.

Per the latest user direction, implementation takes priority over repeated proof runs: run focused tests when behavior changes, retain existing evidence, and perform one integrated live app check per completed stage. Expand verification only for a failure or material risk.

### Preview implementation (stage 1.7)

Optional request preview cadence and maximum pixels now reach every currently implemented native generation/edit pipeline. The real VAE decodes a reduced latent. Graph submission stays on the sampling lane; PNG encoding runs on a utility worker with one in-flight preview per request. Busy work and memory pressure drop previews. Outputs stay below 64 KiB, within the requested size, and stop before terminal events. Defaults remain off.

The engine target compiles. The consolidated generation pass still needs to measure preview count, image quality and <=5% overhead; those are not proven by compilation. Unsupported/stub model entries are not claimed as implemented.

### Request memory estimates (stage 1.8, calibration pending)

Generation/edit/upscale request queries now resolve actual dimensions, native parallel width, reference conditioning and preview workspace. Weight payload bytes are read from the same shard groups as WeightLoader without loading tensors. The API also accepts a model URL for pre-load admission callers.

The per-family activation/VAE coefficients are analytical estimates, not measured bounds. The ±15% acceptance comparison is pending the consolidated run; Osaurus must not use this draft to replace admission decisions yet. Allocator cache and other resident models remain separate host accounting concerns.

### 1.6 Request-scoped LoRA adapters (implementation)

`ImageGenRequest` and `ImageEditRequest` accept local safetensor adapters with scale and transformer/text-encoder/both targeting. Native MFlux and Z-Image linear layers apply low-rank deltas without changing base packed weights. PEFT A/B and Kohya down/up pairs are matched against actual registered layer names and dimensions; unsupported targets/tensors fail explicitly. Qwen edit's separately constructed VLM text encoder is not an adapter target yet.

An engine operation lease keeps load/unload and concurrent generate/edit operations from replacing adapter state while a stream is active. The last set is cached by file SHA256, scale and target. Scale zero takes the base arithmetic path and does not open an adapter file. Clearing the request removes deltas; it never subtracts rounded fused weights. `adapterLoadCount` counts actual installations. Generation/edit through FluxEngine is the supported adapter entry point.

Compilation passes. A real adapter image, scale-zero/base byte parity, repeated-set load count and removal parity remain part of the combined integration pass. This is not yet a live-accepted capability. No seed changes.
