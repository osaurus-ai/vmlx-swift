// Copyright © 2025 JANG. All rights reserved.
//
// Synchronous "prompt-only" SSM state re-derivation for hybrid
// (Mamba + attention) models on thinking-template turns.
//
// Note: `MLX.eval(...)` below is the MLX Swift API for materializing
// lazy MLXArray computation graphs — it is NOT JavaScript eval and
// has nothing to do with arbitrary code execution.
//
// THE PROBLEM
// ===========
// Hybrid models (Nemotron-H, Qwen3.5-A3B, Jamba, FalconH1, Qwen3-Next,
// GraniteMoeHybrid, MiMoV2Flash, BaichuanM1, LFM2 / LFM2MoE) interleave
// Mamba SSM layers with attention layers. The SSM state is cumulative
// and path-dependent: after a forward pass over `prompt + generated`
// the SSM state reflects that full position, not just `prompt`.
//
// When the chat template adds a generation-prompt suffix
// (`<|im_start|>assistant\n` / `<|turn>model\n` / etc.), the cache
// coordinator strips those trailing tokens from the hash key
// (`CacheCoordinator.fetch(genPromptLen:)`) so multi-turn chat can
// reuse prior prefill state. But the stored SSM state is the live
// post-generation state — its position is `prompt + generated`, not
// the stripped prompt. Restoring it on the next turn would feed the
// model an SSM state ahead of the matched token position → garbled
// output.
//
// Python v1.3.36 mitigates this by SKIPPING SSM storage when
// `gen_prompt_len > 0` (see `CacheCoordinator.shouldSkipSSMStorage`).
// Correct, but wastes cache: every thinking-model turn re-computes
// the whole hybrid prefix from scratch.
//
// THIS FIX
// ========
// `reDeriveSSMStates(model:, tokens:, prefillStepSize:)` runs a FRESH
// forward pass on just the prompt-only tokens, through a fresh cache,
// and extracts the resulting SSM state. That state's position matches
// the stripped prompt hash key exactly — no contamination, no offset
// mismatch, and the next turn can cache-hit cleanly.
//
// Called synchronously from `cacheStoreAction` at turn-end after the
// stream has yielded completion `.info`. The user does not pay this as
// an end-of-stream spinner, but the work remains serialized with the
// generation task because the old detached async helper was reverted after
// a Metal command-encoder race.
//
// Cost: one extra chunked-prefill pass over the prompt (no decode
// loop, no sampling). On a 2K prompt at ~1000 prefill tok/s that's
// ~2 seconds. On long contexts it dominates, so production hosts can
// disable the extra pass with `CacheCoordinatorConfig.enableSSMReDerive`
// for controlled A/B rows. The synchronous prompt-boundary path is the
// shipped path; the old detached async helper was reverted after a
// Metal command-encoder race.
//
// PARITY MAP — vmlx (Python) GH issues #103/#105/#107/#109/#110
// =============================================================
// Tracking equivalent treatment for hybrid SSM models on the Swift
// side. Pinned 2026-04-24 against vmlx 1.3.86 / `fa1fdb8`.
//
// • #103 deferred re-derive starves queued requests
//   Python: scheduler runs re-derive when `running.empty` but ignores
//   `waiting`, so a freshly arrived request eats N × re-derive TTFT.
//   Swift parity: detached async re-derive is not active. The production
//   path stores prompt-boundary SSM state synchronously at turn-end through
//   `Evaluate.swift` / `BatchEngine.swift`, then `BatchEngine` admission
//   restores it only when the boundary is safe. A host can disable the
//   extra prompt-only pass with `CacheCoordinatorConfig.enableSSMReDerive`
//   for matrix rows or latency-sensitive deployments.
//
// • #105 `mx.contiguous(mx.array(a))` redundant wrap
//   Python: three sites in scheduler/mllm_batch_generator clone SSM
//   layers with a redundant `mx.array` wrap.
//   Swift parity: N/A. `SSMStateCache.store` clones via `arr * 1`
//   (line ~127) which is idiomatic Swift for forcing materialization;
//   no equivalent double-wrap exists.
//
// • #107 PLD auto-tune kill-switch first-window false positive
//   Python: scheduler disables PLD on the first 1-token summary window
//   because n-gram indices haven't been built yet.
//   Swift parity: PLD is settings-only in Swift (see
//   `SettingsTypes.swift:332-333`). The scheduler-side PLD logic is
//   not yet ported, so the kill-switch hysteresis bug doesn't apply
//   here. When PLD lands, mirror the streak-based fix: only disable
//   after 2 consecutive zero-attempt windows.
//
// • #109 capture-during-prefill (clean SSM state at prompt boundary)
//   Python: proposes capturing SSM state at the prefill→decode
//   transition (when `num_computed_tokens == prompt_len - gen_prompt_len`)
//   so the next turn doesn't have to re-derive at all.
//   Swift parity: PARTIAL. `captureCleanSSMStateInline` exists as the
//   source-compatible hook for live-cache capture, but the production
//   store path currently uses `reDeriveAndStoreSSMStatesForPromptBoundaries`
//   so full-block and exact prompt boundaries can be captured together
//   for paged prefix reuse. There is no detached async helper.
//
// • #110 SSM companion disk persistence (L2 write-through)
//   Python: proposes safetensors + JSON sidecar disk store for
//   SSMCompanionCache so a stable-system-prompt workload doesn't pay
//   full prefill on every cold start.
//   Swift parity: IMPLEMENTED. `SSMCompanionDiskStore` is wired by
//   `CacheCoordinator` whenever disk cache is enabled, and
//   `SSMStateCache` write-through/read-through shares the coordinator
//   model key and media salt isolation with the KV tiers.

import Foundation
import MLX
import MLXNN

/// Re-derive the SSM companion state for a hybrid model by running a
/// fresh prompt-only forward pass.
/// Convenience wrapper used by the `cacheStoreAction` closures in
/// `Evaluate.swift`. Decides whether re-derive should fire based on
/// the feature flag + model hybridness + genPromptLen, runs the
/// prompt-only forward pass, and stores the clean state directly in
/// `coordinator.ssmStateCache`. Swallows errors so re-derive never
/// breaks the main generation path.
public func maybeReDeriveSSMState(
    coordinator: CacheCoordinator,
    model: any LanguageModel,
    promptTokenIds: [Int],
    genPromptLen: Int,
    enableSSMReDerive: Bool
) {
    // iter-45: stderr breadcrumb so users + operators can see whether
    // the SSM helper watcher actually fires. The counter (reDerives)
    // only bumps on the happy path; without logs, a user seeing
    // reDerives=0 couldn't tell if the feature is disabled, the model
    // isn't hybrid, the prompt has no gen-suffix, or the MLX prep
    // step threw. These `cache/ssm-rederive` lines make the decision
    // tree visible.
    func log(_ status: String) {
        let line = "[vmlx][cache/ssm-rederive] \(status) hybrid=\(coordinator.isHybrid) genGP=\(genPromptLen) promptLen=\(promptTokenIds.count) enabled=\(enableSSMReDerive)\n"
        FileHandle.standardError.write(Data(line.utf8))
    }

    guard enableSSMReDerive else { log("skip/disabled"); return }
    guard coordinator.isHybrid else { log("skip/not-hybrid"); return }
    guard genPromptLen > 0 else { log("skip/no-gen-prompt"); return }
    guard promptTokenIds.count > genPromptLen else {
        log("skip/prompt-shorter-than-gp"); return
    }

    let stripped = Array(promptTokenIds.prefix(promptTokenIds.count - genPromptLen))
    guard !stripped.isEmpty else { log("skip/empty-stripped"); return }

    let states = reDeriveAndStoreSSMStatesForPromptBoundaries(
        coordinator: coordinator,
        model: model,
        promptTokenIds: stripped)
    guard let states, !states.isEmpty else {
        log("skip/no-ssm-states")
        return
    }
    log("ok/stored stateCount=\(states.count) boundaryMode=prompt-and-paged-blocks")
}

/// §440 — capture-during-prefill (Python #109 native port). When the
/// caller is already running prefill on the FULL prompt (including any
/// generation-prompt suffix tokens), we can capture the clean SSM
/// state at the prompt-only boundary BY POSITION instead of running a
/// second fresh prefill at turn-end (`reDeriveSSMStates` above). This
/// path zeroes the re-derive cost for thinking-template multi-turn
/// chat — the user's prompt forward pass is the only forward pass.
///
/// Contract:
///   • Caller has just finished prefilling `boundaryTokens` worth of
///     tokens through the live `[KVCache]`. The cache layers already
///     contain the SSM state at the boundary because Mamba caches are
///     state-replacing (O(1) per token), not appending — the live
///     cache IS the snapshot.
///   • We deep-copy via `extractSSMStates(from:)` (which calls
///     `state.value.copy()` per CacheHelpers.swift) so subsequent
///     mutation by generation-prompt tokens doesn't alias.
///   • Storage matches `maybeReDeriveSSMState`: keyed on the stripped
///     prefix hash + modelKey so multi-turn fetch hits exactly.
///
/// Why this is correct vs. `reDeriveSSMStates`:
///   • Both produce SSM state for the SAME `boundaryTokens` (i.e. the
///     prompt minus the generation suffix).
///   • `reDeriveSSMStates` runs a second forward pass; this skips the
///     pass entirely and reads the live cache.
///   • Determinism: both must produce bit-identical state because
///     they're driving the same model with the same input through the
///     same kernels. The §439 fp32 cast in `computeDt` is what
///     guarantees the bit-identity — without it, the prefill chunked
///     SSD path and the decode L=1 path can diverge by exp(precision
///     loss). With §439 in place, capture-during-prefill = re-derive
///     in the limit.
///
/// Used by Evaluate.swift's prefill loop: when `genPromptLen > 0` AND
/// the model is hybrid AND the feature flag is on, the loop calls this
/// at the `prompt.count - genPromptLen` boundary inline.
public func captureCleanSSMStateInline(
    coordinator: CacheCoordinator,
    liveCache: [KVCache],
    promptTokenIds: [Int],
    genPromptLen: Int,
    enableSSMReDerive: Bool
) {
    func log(_ status: String) {
        let line = "[vmlx][cache/ssm-rederive] inline-capture/\(status) hybrid=\(coordinator.isHybrid) genGP=\(genPromptLen) promptLen=\(promptTokenIds.count) enabled=\(enableSSMReDerive)\n"
        FileHandle.standardError.write(Data(line.utf8))
    }

    guard enableSSMReDerive else { log("skip/disabled"); return }
    guard coordinator.isHybrid else { log("skip/not-hybrid"); return }
    guard genPromptLen > 0 else { log("skip/no-gen-prompt"); return }
    guard promptTokenIds.count > genPromptLen else {
        log("skip/prompt-shorter-than-gp"); return
    }

    let stripped = Array(promptTokenIds.prefix(promptTokenIds.count - genPromptLen))
    guard !stripped.isEmpty else { log("skip/empty-stripped"); return }

    let states = extractSSMStates(from: liveCache)
    guard !states.isEmpty else { log("skip/no-ssm-states"); return }

    coordinator.ssmStateCache.store(
        ssmStates: states,
        tokens: stripped,
        boundary: stripped.count
    )
    coordinator.ssmStateCache.markReDeriveFired()
    log("ok/captured stateCount=\(states.count)")
}

public func reDeriveSSMStates(
    model: any LanguageModel,
    tokens: [Int],
    prefillStepSize: Int = 512
) throws -> [MLXArray]? {
    guard !tokens.isEmpty else { return nil }

    let freshCache: [KVCache] = model.newCache(parameters: nil)

    // Pure-attention model check: if no cache layer carries path-dependent
    // state, there's nothing to re-derive and we bail immediately.
    // ZayaCCACache fits the same contract — its conv_state + prev_hs are
    // path-dependent alongside the KV pair (see CacheHelpers.swift:293-300).
    let hasSSMLayer = freshCache.contains { cache in
        let desc = String(describing: type(of: cache))
        return desc.contains("Mamba") || desc.contains("Arrays") || desc.contains("ZayaCCA")
    }
    guard hasSSMLayer else { return nil }

    // Shape tokens as [1, L] — every prefill call site in Evaluate.swift
    // and BatchEngine expects the batch axis for sliding-window attention.
    let tokenArray = MLXArray(tokens.map { Int32($0) })
        .reshaped([1, tokens.count])
    let input = LMInput(text: LMInput.Text(tokens: tokenArray))

    // iter-45 FIX: the default `LLMModel.prepare` only runs chunks of
    // length `prefillStepSize` through the forward pass, and returns
    // any remaining tail as a `PrepareResult.tokens(...)` for the
    // caller to run. Prior re-derive code discarded that tail — so
    // for short prompts (len ≤ prefillStepSize, the common case for
    // chat single-turn requests) ZERO forward passes happened and the
    // SSM cache stayed empty, producing `skip/no-ssm-states` on every
    // attempt. Fix: collect the prepare tail and explicitly run it
    // through `callAsFunction` so the whole prompt touches every
    // mamba layer, writing `conv_state` + `hidden_state` into the
    // fresh cache.
    let result = try model.prepare(
        input, cache: freshCache, windowSize: prefillStepSize)
    switch result {
    case .tokens(let tail):
        if tail.tokens.size > 0 {
            let tailInput: MLXArray
            if tail.tokens.shape.count >= 2 {
                tailInput = tail.tokens
            } else {
                tailInput = tail.tokens.reshaped([1, tail.tokens.size])
            }
            _ = model.callAsFunction(tailInput, cache: freshCache)
        }
    case .logits:
        break
    @unknown default:
        break
    }
    MLX.eval(freshCache)

    let states = extractSSMStates(from: freshCache)
    return states.isEmpty ? nil : states
}

/// Re-derive SSM state once while capturing snapshots at multiple token
/// boundaries. Used for hybrid BlockDisk: the KV tier restores only full
/// paged blocks, so a prompt of 284 cacheable tokens with a 64-token block
/// needs an SSM snapshot at 256 as well as the exact 284-token boundary.
public func reDeriveSSMStatesAtBoundaries(
    model: any LanguageModel,
    tokens: [Int],
    boundaries: [Int],
    prefillStepSize: Int = 512
) throws -> [Int: [MLXArray]] {
    guard !tokens.isEmpty else { return [:] }
    let captureBoundaries = Array(Set(boundaries.filter { $0 > 0 && $0 <= tokens.count })).sorted()
    guard !captureBoundaries.isEmpty else { return [:] }

    let freshCache: [KVCache] = model.newCache(parameters: nil)
    let hasSSMLayer = freshCache.contains { cache in
        let desc = String(describing: type(of: cache))
        return desc.contains("Mamba") || desc.contains("Arrays") || desc.contains("ZayaCCA")
    }
    guard hasSSMLayer else { return [:] }

    var out: [Int: [MLXArray]] = [:]
    var cursor = 0
    let step = max(1, prefillStepSize)

    for boundary in captureBoundaries {
        while cursor < boundary {
            let end = min(boundary, cursor + step)
            let chunk = Array(tokens[cursor..<end])
            let tokenArray = MLXArray(chunk.map { Int32($0) })
                .reshaped([1, chunk.count])
            _ = model.callAsFunction(tokenArray, cache: freshCache)
            MLX.eval(freshCache)
            cursor = end
            Memory.clearCache()
        }

        let states = extractSSMStates(from: freshCache)
        if !states.isEmpty {
            out[boundary] = states
        }
    }

    return out
}

/// Re-derive clean SSM companion states for every prefix boundary the
/// attention KV cache may later restore.
///
/// Hybrid models need SSM snapshots at the same token boundary as the KV
/// prefix hit. A stored prompt of 138 tokens with a 64-token paged block
/// size may later match only 128 tokens when the next prompt diverges inside
/// the final partial block. Storing only the 138-token SSM state forces the
/// coordinator to reject that otherwise valid 128-token KV hit. This helper
/// captures both full-block boundaries and the exact prompt boundary in one
/// fresh prefill pass.
@discardableResult
public func reDeriveAndStoreSSMStatesForPromptBoundaries(
    coordinator: CacheCoordinator,
    model: any LanguageModel,
    promptTokenIds: [Int],
    mediaSalt: String? = nil,
    prefillStepSize: Int = 512
) -> [MLXArray]? {
    guard coordinator.isHybrid, !promptTokenIds.isEmpty else { return nil }

    var boundaries = Set<Int>()
    boundaries.insert(promptTokenIds.count)
    if promptTokenIds.count > 1 {
        boundaries.insert(promptTokenIds.count - 1)
    }

    if coordinator.pagedCache != nil, !coordinator.isPagedIncompatible {
        let blockSize = max(1, coordinator.config.pagedBlockSize)
        var boundary = blockSize
        while boundary < promptTokenIds.count {
            boundaries.insert(boundary)
            boundary += blockSize
        }
    }

    do {
        let statesByBoundary = try reDeriveSSMStatesAtBoundaries(
            model: model,
            tokens: promptTokenIds,
            boundaries: Array(boundaries).sorted(),
            prefillStepSize: prefillStepSize)

        for boundary in boundaries.sorted() {
            guard let states = statesByBoundary[boundary], !states.isEmpty else {
                continue
            }
            coordinator.ssmStateCache.store(
                ssmStates: states,
                tokens: promptTokenIds,
                boundary: boundary,
                mediaSalt: mediaSalt)
        }

        if !statesByBoundary.isEmpty {
            coordinator.ssmStateCache.markReDeriveFired()
        }
        return statesByBoundary[promptTokenIds.count]
    } catch {
        let line = "[vmlx][cache/ssm-rederive] prompt-boundaries/fail \(error) promptLen=\(promptTokenIds.count)\n"
        FileHandle.standardError.write(Data(line.utf8))
        return nil
    }
}
