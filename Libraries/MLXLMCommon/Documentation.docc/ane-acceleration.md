# ANE Acceleration Roadmap - 2026-05-07

Scope: production path for a user-facing acceleration flag that can improve
token/s or prompt-processing/s without regressing the existing MLX/Metal
runtime, cache stack, reasoning streams, or batching.

## Current Truth

- The vendored MLX device API exposes CPU and GPU only. There is no MLX
  `Device(.ane)` or equivalent direct ANE stream today.
- Custom kernels in this repo are Metal kernels, not ANE kernels.
- Public ANE execution is through Core ML scheduling. Core ML can choose CPU,
  GPU, and Neural Engine, and Xcode/Instruments can report where operators
  actually ran.
- A production flag must therefore select a validated Core ML subgraph island,
  not try to run the whole MLX decoder on ANE.

Do not add a user-facing "force ANE decode" flag until at least one island has
a live benchmark showing a win and byte/cosine parity against the MLX path. A
no-op speed flag would be worse than no flag because Osaurus would surface
misleading performance controls.

## Proposed Flag Contract

Runtime names reserved for the future Osaurus/vMLX setting:

| Value | Meaning |
| --- | --- |
| `metal` | Default. Current MLX + custom Metal kernel path. |
| `auto` | Use ANE islands only when the loaded model family and local device have a validated manifest row. |
| `ane-coreml` | Require a validated Core ML island. Fail closed if no island exists for the active model path. |

Suggested environment variable for local benches:

```sh
VMLX_ACCELERATOR=auto
```

Implemented API shape:

```swift
public enum AccelerationMode: Sendable {
    case metal
    case auto
    case aneCoreML
    case invalid(String)
}
```

The actual implementation lives in `RuntimeAcceleration.swift` and
`GenerateParameters.accelerationMode`. `auto` keeps the current Metal path until
a validated island manifest is supplied. `ane-coreml` fails closed for text
decode today because no text-decode Core ML island is validated.

## Candidate ANE Islands

### 1. VL / Audio Encoders

Best first target.

- Candidates: Nemotron-Omni RADIO vision encoder, Parakeet audio encoder,
  Gemma/Qwen/Mistral vision towers.
- Why: encoder work is prefill/TTFT-heavy, has bounded tensor shapes, and does
  not mutate KV cache every token.
- Cache impact: media salt must include model hash, Core ML package hash,
  preprocessing version, media bytes hash, and encoder-output dtype/shape.
- Acceptance: same-media prefix hit still hits; different media misses; text
  follow-up emits no raw media/reasoning markers; ANE island improves wall
  time or reduces GPU contention.

### 2. Dense Fixed-Shape Prefill Blocks

Possible but high risk.

- Candidate: fixed prompt chunks for dense transformer blocks, especially
  small/medium models where ANE can stay resident.
- Risk: MLX to Core ML tensor copies and shape bucketing can erase the win.
- Cache impact: outputs must re-enter the normal MLX KV/cache path with exact
  dtype and position semantics.

### 3. MoE Expert MLP Islands

Research target, not first production target.

- Candidate: MiniMax/Qwen/Nemotron routed expert MLP blocks.
- Risk: routing is dynamic, selected experts change per token, and current
  JANGTQ custom Metal kernels already avoid many slow paths.
- Requirements: precompiled expert-pack Core ML functions, stable top-k routing
  input contract, and a fallback for unsupported expert sets.

### 4. Full Decoder On ANE

Not a current production path.

- KV cache mutation, dynamic sequence length, sampling, paged-prefix state,
  TurboQuant KV, ZAYA CCA state, Ling recurrent state, and DSV4 CSA/HSA/SWA
  cache classes all need exact semantics.
- Whole-decoder Core ML conversion would duplicate model-loading, quantization,
  cache, reasoning, tool-call, and batching behavior that already works in
  vmlx-swift-lm.

## Cache And Batching Constraints

Any ANE island must obey these invariants:

- `KVCacheSimple`, `RotatingKVCache`, `TurboQuantKVCache`, `CacheList`,
  `ArraysCache`, `ZayaCCACache`, and DSV4 cache classes remain owned by the
  MLX runtime unless the island explicitly owns the same state representation.
- Paged-prefix and disk L2 keys must include the accelerator island manifest
  hash. A Core ML encoder output is not interchangeable with an MLX encoder
  output unless parity is proven and the manifest says so.
- BatchEngine slot isolation must remain byte-identical for slot 0 vs solo.
- Hybrid families stay conservative:
  - MiniMax is normal MoE/KV, not hybrid.
  - Ling/Bailing carries recurrent ArraysCache state.
  - ZAYA carries KV plus CCA convolution state plus previous hidden state.
  - DSV4 carries CSA/HSA/SWA-specific cache policy.
  - Omni/VL cache keys include media salt.

## Measurement Gates

For every island and model family:

1. Core ML performance report or Instruments row proves ANE participation.
2. MLX-vs-ANE output parity is measured on real inputs.
3. TTFT, prompt-processing/s, decode token/s, peak RSS, and GPU memory pressure
   are logged.
4. BatchEngine B=2 isolation passes.
5. Prefix cache hit/miss semantics pass.
6. Disk L2 restore either passes or is explicitly disabled for that island.
7. Reasoning/tool/media marker streams stay clean.

## Immediate Work Order

1. Keep `metal` as the only active runtime path for text decode.
2. Use `ANEProbe` as the isolated Core ML capability check. It is deliberately
   outside the hot decode path and reports Core ML / Neural Engine visibility.
3. Convert one media encoder first, preferably Nemotron-Omni RADIO or Parakeet,
   and benchmark `BENCH_VL_CHAT_CACHE` with `VMLX_ACCELERATOR=metal` vs
   `VMLX_ACCELERATOR=ane-coreml`.
4. Only after a measured win, wire `AccelerationMode` into generation options
   and Osaurus settings.
5. Leave dense/MoE decoder ANE islands as research until the media island is
   proven production-safe.

## Current Flag Behavior

`VMLX_ACCELERATOR` is parsed by `AccelerationRuntime.requestedMode()`:

| Value | Current behavior |
| --- | --- |
| unset / `metal` | Text decode uses MLX/Metal. |
| `auto` | Text decode uses MLX/Metal until a validated Core ML island manifest is registered. |
| `ane-coreml` | Text decode fails closed with `AccelerationError.noValidatedCoreMLIsland`. |
| invalid | Generation fails closed with `AccelerationError.invalidMode`. |

This gives Osaurus a stable settings contract without claiming an ANE speedup
before a real Core ML subgraph exists.

## Local Probe Result

2026-05-07 on the local M5 Max:

```text
VMLX_ANE_PROBE_VERSION=1
VMLX_ACCELERATOR_REQUESTED=metal
VMLX_TEXT_DECODE_ACCELERATOR=metal
VMLX_TEXT_DECODE_REASON=explicit-metal
COREML_AVAILABLE=YES
MLX_DIRECT_ANE_DEVICE=NO
COREML_COMPUTE_UNITS=all
COREML_COMPUTE_DEVICE_COUNT=3
COREML_COMPUTE_DEVICE_0=<MLNeuralEngineComputeDevice: ...>
COREML_COMPUTE_DEVICE_1=<MLGPUComputeDevice: ...> Apple M5 Max
COREML_COMPUTE_DEVICE_2=<MLCPUComputeDevice: ...>
COREML_NEURAL_ENGINE_VISIBLE=YES
```
