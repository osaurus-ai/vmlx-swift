# Osaurus Memory Safety Slider Spec

This is the vMLX/Osaurus contract for a user-visible memory safety control.
It is not a hardcoded RAM rejection rule and must not change model behavior to
hide memory issues.

## Goals

- Give Osaurus UI, CLI, and API one persisted memory safety setting.
- Resolve that setting into real engine controls: load cap, allocator cache
  cap, mmap safetensors, cache policy, concurrency, and typed admission issues.
- Keep Safe Auto as the default for 24 GB-class hosts without blocking every
  large-model attempt from a single free-RAM snapshot.
- Fail before unsafe MLX/Metal allocation when a real estimate and selected
  policy require refusal.
- Preserve model defaults from `generation_config.json`.

## Non-Goals

- Do not expose MLXPress/JangPress as the primary user setting.
- Do not use sampler, prompt, parser, reasoning, or output stripping as memory
  fixes.
- Do not silently truncate context unless the user explicitly selected a
  truncation policy.
- Do not infer TurboQuant KV, media support, native MTP, or companion cache
  support from model names alone.
- Do not catch/retry after native process-fatal MLX/Metal failures.

## User Modes

| Slider | Mode | UI label | Behavior |
| --- | --- | --- | --- |
| 0 | `performance` | Performance | Higher memory pressure, fewer automatic caps, warnings over budget. |
| 1 | `balanced` | Balanced | Moderate caps, paged KV, block disk L2, 1-2 concurrent sequences. |
| 2 | `safe_auto` | Safe Auto | Default. Conservative caps, mmap safetensors, paged KV, block disk L2, one sequence. |
| 3 | `strict` | Strict | Low-RAM mode. Blocks unknown or over-budget estimates before load/decode. |
| 4 | `diagnostic_dangerous` | Diagnostic | Explicit advanced/custom settings only. |

Safe Auto is the default. It should not be described as a guarantee; it is a
conservative plan with typed warnings/refusals when estimates prove unsafe.

## vMLX API

`VMLXServerRuntimeSettings` owns the contract:

```swift
public struct VMLXServerRuntimeSettings: Codable, Sendable, Equatable {
    public var memorySafety: VMLXMemorySafetySettings

    public func resolvedMemorySafetyPlan(
        baseLoadConfiguration: LoadConfiguration = .default,
        bundleFacts: LoadBundleFacts? = nil,
        host: MemoryStatus? = nil,
        request: VMLXMemoryRequestEstimate? = nil
    ) -> VMLXResolvedMemorySafetyPlan
}
```

Persist this setting in Osaurus:

```swift
public struct VMLXMemorySafetySettings: Codable, Sendable, Equatable {
    public var mode: VMLXMemorySafetyMode
    public var slider: Int
    public var allowExperimentalMLXPress: Bool
    public var failClosedWhenEstimateUnknown: Bool
    public var customPhysicalMemoryFraction: Double?
    public var customAllocatorCacheBytes: UInt64?
    public var customDefaultMaxKVSize: Int?
    public var customMaxConcurrentSequences: Int?
}
```

Request-time estimates:

```swift
public struct VMLXMemoryRequestEstimate: Sendable, Equatable {
    public var workingSetBytes: UInt64?
    public var promptTokens: Int?
    public var maxNewTokens: Int?
}
```

Resolved plan:

```swift
public struct VMLXResolvedMemorySafetyPlan: Sendable, Equatable {
    public var loadConfiguration: LoadConfiguration
    public var cache: VMLXServerCacheSettings
    public var concurrency: VMLXServerConcurrencySettings
    public var resolvedPhysicalMemoryBytes: UInt64
    public var resolvedLoadBudgetBytes: UInt64?
    public var warnings: [String]
    public var blockingIssues: [VMLXServerSettingsIssue]
    public var displaySummary: String
    public var allowed: Bool
}
```

## Current Resolver Mapping

| Mode | Load cap | Allocator cache cap | Max concurrent | Prefix cap | Default KV cap | Blocking |
| --- | --- | --- | --- | --- | --- | --- |
| `performance` | 0.90 physical unless custom | unlimited unless custom | 2 unless set | existing or 20 percent | existing/custom | warn over budget |
| `balanced` | 0.75 physical unless custom | 1 GiB unless custom | 2 unless set | 512 MB, 15 percent | 8192 | warn over budget |
| `safe_auto` | 0.70 physical unless custom | 128 MiB unless custom | 1 unless set | 128 MB, 15 percent | 8192 | warn over budget |
| `strict` | 0.60 physical unless custom | 128 MiB unless custom | 1 unless set | 128 MB, 10 percent | 4096 | block unknown/over-budget request estimates |
| `diagnostic_dangerous` | 1.0 physical unless custom | unlimited unless custom | 1 unless set | existing | existing/custom | explicit advanced only |

The resolver always enables mmap safetensors, paged KV, and block disk L2 in
safe plans. It disables legacy disk when paged KV is enabled.

## MLXPress/JangPress Boundary

Memory safety is not MLXPress.

- Dense Gemma/MXFP rows must keep `jangPress == .disabled`.
- JANG filenames do not automatically enable MLXPress.
- MLXPress/JangPress can pass through only when:
  - `allowExperimentalMLXPress == true`,
  - bundle facts prove a routed JANG/JANGTQ row,
  - the selected bundle lane is already proven or explicitly diagnostic.
- The UI should show MLXPress as disabled, eligible, active, or refused with a
  reason, but not as the primary slider.

## Model-Family Requirements

Gemma 4 MXFP4/JANG_4M:

- Preserve bundle sampler defaults.
- Do not claim TurboQuant KV when telemetry reports zero TurboQuant KV layers.
- Preserve rotating/full KV and disk-backed restore.
- Image, audio, and video must remain per-modality claims with real media proof.

Qwen MTP / hybrid SSM:

- Native MTP is separate from memory safety.
- Keep `native_mtp_status` and tuning requirements separate from the slider.
- Cache proof requires KV plus SSM companion rederive/hits.

MiMo V2.5:

- TurboQuant KV can apply only to proven full-attention KV components.
- It must not replace SWA/rotating state.
- Cache proof must show rotating topology and disk-backed restore.

VL/audio/video/omni:

- Include media tokens and processor-side tensors in estimates.
- Cache keys must include media hash/salt, modality metadata, processor state,
  and template identity.
- Text-only proof does not promote media support.

## Osaurus UI/CLI Work

Osaurus should:

- Persist `memorySafety` with the existing runtime settings.
- Add CLI/API fields for `mode`, `slider`, and advanced custom values.
- Show the selected mode and resolved plan.
- Say when a setting takes effect only on next model load.
- Return typed user-facing refusals from `blockingIssues`.
- Never convert warnings into hidden sampler/template/parser changes.

Required status/admin fields:

- `memory_safety.mode`
- `memory_safety.slider`
- `memory_safety.allowed`
- `memory_safety.display_summary`
- `memory_safety.resolved_physical_memory_bytes`
- `memory_safety.resolved_load_budget_bytes`
- `memory_safety.load_configuration.memory_limit`
- `memory_safety.load_configuration.max_resident_bytes`
- `memory_safety.load_configuration.use_mmap_safetensors`
- `memory_safety.load_configuration.jang_press_policy`
- `memory_safety.cache.prefix_enabled`
- `memory_safety.cache.paged_kv_enabled`
- `memory_safety.cache.block_disk_enabled`
- `memory_safety.cache.live_kv_codec`
- `memory_safety.cache.default_max_kv_size`
- `memory_safety.cache.enable_ssm_rederive`
- `memory_safety.concurrency.max_concurrent_sequences`
- `memory_safety.warnings`
- `memory_safety.blocking_issues`
- `memory_safety.active_runtime.cache_topology`
- `memory_safety.active_runtime.mlxpress_status`
- `memory_safety.active_runtime.physical_footprint_bytes` when available

## Required Proof

- Settings encode/decode with Safe Auto default.
- Invalid slider/custom values produce validation issues.
- Each mode resolves expected load/cache/concurrency settings.
- Strict unknown-estimate and over-budget estimate return typed refusals.
- Dense Gemma does not enable MLXPress.
- Proven routed JANG/JANGTQ can opt into MLXPress only through explicit
  advanced/proven settings.
- Qwen hybrid cache preserves SSM companion rederive.
- Gemma rotating cache does not force TurboQuant KV.
- Osaurus UI/CLI/status proves the selected setting changes the next resolved
  plan.
- Low-RAM model/request rows either run or refuse gracefully before unsafe
  native allocation.
