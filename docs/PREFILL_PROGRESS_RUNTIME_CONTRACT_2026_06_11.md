# Prefill Progress Runtime Contract

Date: 2026-06-11

This spec covers runtime and Osaurus UI wiring for showing prompt-prefill
progress before the first generated token.

## Problem

Long prompt prefill is currently blind. Users can wait 30 seconds or more before
the first token and cannot tell whether the model is frozen, loading, restoring
cache, processing media, or simply running slowly on their compute. This is
especially visible on large Gemma BF16/source rows and lower-RAM machines.

The fix must be runtime-backed. A UI timer or guessed spinner would hide the
real bottleneck and would not help users understand slow compute.

## Goals

- Emit prefill progress for every normal generation path before first token.
- Make progress visible through Osaurus chat and server/API streaming.
- Keep progress monotonic and bounded: `0.0 <= fraction <= 1.0`.
- Include enough metadata for the UI to render useful labels such as
  "Restoring cache", "Processing media", and "Prefilling prompt".
- Preserve normal text, reasoning, tool-call, and final-info streaming behavior.
- Keep model defaults, samplers, reasoning/tool parsing, and cache policy
  untouched.

## Non-Goals

- Do not invent speed estimates when no measurement exists.
- Do not change generation quality or prompt templates.
- Do not use paged RAM cache just to make progress look better.
- Do not make the UI responsible for calculating model-internal progress.

## Runtime API Shape

Add a public progress payload shared by token and text streams:

```swift
public struct PrefillProgress: Sendable, Equatable {
    public enum Stage: String, Sendable, Equatable {
        case queued
        case cacheLookup
        case cacheRestore
        case prefill
        case complete
    }

    public let stage: Stage
    public let completedUnitCount: Int
    public let totalUnitCount: Int
    public let detail: String?

    public var fractionCompleted: Double
    public var percentCompleted: Double
}
```

Extend stream enums:

```swift
public enum Generation: Sendable {
    case prefillProgress(PrefillProgress)
    case chunk(String)
    case reasoning(String)
    case info(GenerateCompletionInfo)
    case toolCall(ToolCall)
}

public enum TokenGeneration: Sendable {
    case prefillProgress(PrefillProgress)
    case token(Int)
    case info(GenerateCompletionInfo)
}

public enum BatchGeneration: Sendable {
    case prefillProgress(PrefillProgress)
    case token(Int)
    case info(GenerateCompletionInfo)
}
```

Existing callers that switch exhaustively will need a source update. That is
acceptable because Osaurus must explicitly handle the new event and old clients
should not silently drop progress in a user-facing release.

## Calculation

The denominator is the effective number of prompt tokens the runtime will
process for the current request after cache policy is resolved.

For token-only LLM paths:

- `totalTokens = inputForPrepare.text.tokens.size`
- `prefillStepSize = effectivePrefillWindow(...)`
- `chunkCount = ceil(totalTokens / prefillStepSize)`
- After each completed prefill chunk, emit:
  - `processedTokens = min(totalTokens, completedChunks * prefillStepSize)`
  - `fraction = processedTokens / totalTokens`

For cache-hit paths:

- Cache lookup emits stage `.cacheLookup` with `fraction = 0`.
- Cache restore emits `.cacheRestore` with:
  - `restoredTokens = restored token count`
  - `totalTokens = restoredTokens + remainingTokens.count`
  - `processedTokens = restoredTokens`
  - `fraction = restoredTokens / totalTokens`
- Prompt prefill then counts only `remainingTokens.count`, but the displayed
  fraction uses the full prompt denominator so users see that cache restored
  real work.
- If correctness rules roll back to full prefill for media placeholders, hybrid
  SSM partial hits, or missing companion state, emit a stage message and reset
  `processedTokens` to `0` against the full prompt denominator.

For VLM/media paths:

- Media preprocessing is a stage, not token progress. Emit media-stage events
  before prompt prefill starts once the model path exposes that boundary.
- If the model can expose image/video/audio patch counts, use them in the
  message and keep `fraction` below the prompt-prefill range until media is
  done. If not, emit stage-level `mediaEncode` start and completion events.
- After media embeddings are spliced and effective prompt tokens are known,
  prompt prefill uses the effective token count reported by `PrepareResult`
  when available.

For hybrid SSM, ZAYA/CCA, DSV4, and other path-dependent caches:

- Progress follows the same cache-restore and prompt-prefill math, but rollback
  rules are authoritative. If partial restore is unsafe, do not show the cache
  hit as saved work; emit the rollback message and full-prefill denominator.
- SSM/companion rederive after prompt boundary is a cache-maintenance stage. It
  may be shown as a short post-prefill stage only if it happens before first
  token; work intentionally deferred after first token should not block the
  displayed prefill percentage.

For TurboQuant KV:

- TurboQuant compression after prefill is part of the prefill pipeline when it
  blocks first token. Emit a final `.promptPrefill` or `.firstToken` transition
  only after required compression/materialization has completed.
- The progress event must not claim TurboQuant KV is active unless cache
  topology proves the cache actually contains TurboQuant layers.

## Wiring Points

vMLX runtime:

- Add `PrefillProgress` and enum cases in `Evaluate.swift` and
  `BatchEngine/BatchTypes.swift`.
- Thread a progress sink through `TokenIterator` and `BatchEngine`.
- Add a prepare-progress observer so `LanguageModel.prepare(...)` can report
  chunk completion without changing model outputs.
- For model implementations that already chunk internally, call the observer at
  each real chunk boundary.
- For black-box/full-prompt VLM prepare implementations, emit truthful
  stage-level start/end until finer-grained media/token counts are available.

Osaurus:

- Update the MLX generation adapter to forward `prefillProgress` events through
  chat/API streaming.
- Add a UI prefill progress row on the assistant message while no first token
  has arrived.
- Replace the progress row with normal streaming output once `.chunk`,
  `.reasoning`, `.toolCall`, or final `.info` arrives.
- For API streaming, include a typed progress event rather than encoding it as
  assistant text.

## Model Suggestion Interaction

Model suggestions should be recalibrated around maximum useful capability that
fits the device:

- Rank higher-capability Gemma rows only when estimated load footprint,
  prefill working set, and KV growth fit the host memory profile.
- Prefer quantized Gemma QAT/MXFP/JANG rows on lower-RAM machines when BF16
  source rows would make prefill unreasonably slow or memory-risky.
- Keep the recommendation explanation honest: show that a model is suggested
  because it balances capability, RAM, prefill latency, and expected token/s.
- Do not silently alter generation defaults to make a suggested model look
  better.

## Tests

Focused vMLX tests:

- `Generation` and `BatchGeneration` expose progress events without swallowing
  chunk/reasoning/tool/info events.
- Token-only prefill progress is monotonic and ends at 100% before first token.
- Cache-hit progress accounts for restored tokens and remaining prefill tokens.
- Rollback paths reset progress to the full-prefill denominator.
- Explicit cancellation during prefill finishes streams without stale progress
  events or orphan slots.

Osaurus tests/proof:

- Chat UI shows prefill progress on a long prompt before first token.
- Server/API streaming emits typed progress events.
- Gemma BF16/source and quantized rows record prefill time, token/s, cache
  topology, and final harness score.

## Status

- Spec written.
- vMLX implementation partial:
  - Public `PrefillProgress` is added to `Generation`, `TokenGeneration`, and
    `BatchGeneration`.
  - `BatchEngine` emits real stage/token-boundary progress before first token:
    `queued`, `cacheLookup`, optional `cacheRestore`, `prefill`, and `complete`.
  - Current batch implementation is intentionally stage-level around
    `model.prepare(...)`; deeper per-chunk percentages require adding a prepare
    progress observer to model implementations that chunk internally.
- Source build passes for `MLXLMCommon`.
- Osaurus UI/API implementation is patched in the integration checkout; focused
  tests are blocked until Osaurus pins a vMLX SHA containing
  `Generation.prefillProgress`.
- Live proof pending.

## Completion Gate

This feature is not done from source tests alone. The final PR must prove the
dev Osaurus app can load a Gemma model, show prefill progress before first
token, complete a normal chat turn, complete a tool-call turn, and report
token/s plus RAM/physical-footprint observations during the run.
