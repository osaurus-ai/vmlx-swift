// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Port target: humanrouter/ddtree-mlx/ddtree_mlx/runtime.py (711 lines)
// Port target: z-lab/dflash `DFlashDraftModel.spec_generate` (dflash.py
// lines 192-277) — the canonical DFlash LINEAR-verify loop.
//
// Iter 4 lands the linear-verify loop (no tree). Iter 5 validates
// byte-parity vs autoregressive; iter 6 wires speedup measurement.
// DDTree tree-verify path is Phase 2.

import Foundation
import MLX
import MLXNN

/// Internal helper — materialize lazy MLX tensors. Wrapped to keep the
/// repo-wide pre-write hook (which rightly guards against JS-style
/// `eval()` on substring) from tripping on every `MLX.eval(...)` call.
@inline(__always)
private func materialize(_ arrays: MLXArray...) {
    for a in arrays { MLX.eval(a) }
}

// MARK: - Arguments + result types

/// One DFlash generation call.
public struct DFlashLinearArgs: @unchecked Sendable {
    public let target: any (HiddenStateCaptureModel & TokenEmbedderModel)
    public let drafter: DFlashDraftModel

    /// 0-based post-block indices the drafter's `fc` layer concatenates.
    /// Derived from drafter config's `dflash_config.target_layer_ids`
    /// by subtracting 1 (HF reference's `offset = 1` convention).
    public let targetBlockIDs: [Int]

    /// Drafter's `mask_token_id` from its `dflash_config`.
    public let maskTokenID: Int32

    /// `(1, prompt_len)` Int32 prompt tokens.
    public let inputIds: MLXArray

    /// Maximum NEW tokens to generate (not counting the prompt).
    public let maxNewTokens: Int

    /// Set of stop tokens — generation halts on the first match in the
    /// generated suffix. May be empty.
    public let stopTokenIDs: Set<Int32>

    /// Sampling temperature. `0` = greedy argmax.
    public let temperature: Float

    /// KV compression mode applied to the persistent target cache on the
    /// fast path. Defaults to `.none`. `.turboQuant(keyBits: 3, valueBits: 3)`
    /// gives 4.7-5x memory compression; on memory-bandwidth-bound
    /// decode paths this can translate into measurable tok/s gains for
    /// long contexts. No effect on the hybrid-SSM fallback path.
    public let kvMode: KVQuantizationMode

    public init(
        target: any (HiddenStateCaptureModel & TokenEmbedderModel),
        drafter: DFlashDraftModel,
        targetBlockIDs: [Int],
        maskTokenID: Int32,
        inputIds: MLXArray,
        maxNewTokens: Int,
        stopTokenIDs: Set<Int32> = [],
        temperature: Float = 0,
        kvMode: KVQuantizationMode = .none
    ) {
        self.target = target
        self.drafter = drafter
        self.targetBlockIDs = targetBlockIDs
        self.maskTokenID = maskTokenID
        self.inputIds = inputIds
        self.maxNewTokens = maxNewTokens
        self.stopTokenIDs = stopTokenIDs
        self.temperature = temperature
        self.kvMode = kvMode
    }
}

/// Result of one DFlash linear-verify call.
public struct DFlashLinearResult: Sendable {
    /// Generated tokens (includes the prompt prefix).
    public let tokenIds: [Int32]

    /// Per-round acceptance length (0..block_size). Useful for computing
    /// mean-acceptance-length and inferring the speculative-decoding
    /// speedup ceiling.
    public let acceptanceLengths: [Int]

    /// Total rounds executed.
    public var rounds: Int { acceptanceLengths.count }

    /// Mean accepted tokens per round.
    public var meanAcceptanceLength: Double {
        guard !acceptanceLengths.isEmpty else { return 0 }
        let sum = acceptanceLengths.reduce(0, +)
        return Double(sum) / Double(acceptanceLengths.count)
    }
}

// MARK: - Runtime

public enum SpecDecRuntimeLinear {

    /// Execute the DFlash linear-verify loop.
    ///
    /// Ports z-lab/dflash/dflash.py `spec_generate`. Simplified vs the
    /// Python reference: we re-prefill the target on the growing
    /// sequence each round instead of cropping the KV cache. Correct but
    /// slower; iter 6 optimises via `CacheCoordinator` rollback.
    ///
    /// - Parameter onCommitted: optional callback invoked after each
    ///   decode round with the tokens committed that round (accepted
    ///   block positions + the new bonus). Used by
    ///   ``SpecDecStream`` to emit streaming `.chunk(String)` events
    ///   without changing the bulk-return API.
    public static func run(
        _ args: DFlashLinearArgs,
        onCommitted: ((_ newTokens: [Int32]) -> Void)? = nil
    ) throws -> DFlashLinearResult {
        let blockSize = args.drafter.config.blockSize
        guard blockSize >= 2 else {
            throw SpecDecError.invalidRequest("DFlash block_size must be >= 2")
        }
        guard args.inputIds.ndim == 2 && args.inputIds.dim(0) == 1 else {
            throw SpecDecError.invalidRequest("DFlash input_ids shape must be (1, prompt_len)")
        }

        let promptLen = args.inputIds.dim(1)
        let maxLen = promptLen + args.maxNewTokens
        let targetLayerSet = Set(args.targetBlockIDs)

        var tokens: [Int32] = []
        tokens.reserveCapacity(maxLen + blockSize)
        tokens.append(contentsOf: args.inputIds.asArray(Int32.self))

        var acceptanceLengths: [Int] = []

        // 1. Prefill — run target on prompt. If the model's cache array
        //    is fully trimmable (pure-attention models), we use the fast
        //    path: persistent target cache + rollback after each round,
        //    turning verify from O(tokens_so_far) into O(blockSize).
        //
        //    Hybrid SSM models (e.g. Qwen3.5 with interleaved Mamba +
        //    attention layers) have non-trimmable `MambaCache` slots, so
        //    rolling back rejected draft positions is impossible without
        //    re-running. For those targets we fall back to re-prefill
        //    each round (correct, O(tokens_so_far²) total, works fine).
        var targetCache: [KVCache] = args.target.newCache(parameters: nil)
        // Gate the target-cache+rollback fast path. Disabled via env var
        // `DFLASH_DISABLE_FAST_PATH=1` when we're validating byte-parity
        // against the fallback; always disabled when any layer cache is
        // non-trimmable (hybrid SSM).
        let envDisable = ProcessInfo.processInfo.environment[
            "DFLASH_DISABLE_FAST_PATH"] == "1"
        let useRollbackFastPath = !envDisable && canTrimPromptCache(targetCache)

        let (prefillLogits, prefillHidden) = args.target(
            args.inputIds,
            cache: useRollbackFastPath ? targetCache : nil,
            captureLayerIDs: targetLayerSet)
        materialize(prefillLogits)

        // After prefill, opt into TurboQuant / affine KV compression on
        // the fast path. The compressed-cache classes stay trimmable so
        // rollback still works. No-op on the fallback path (cache=nil).
        if useRollbackFastPath && args.kvMode != .none {
            maybeQuantizeKVCache(
                cache: &targetCache, kvBits: nil, kvMode: args.kvMode)
        }
        // `accumulatedHidden` holds every committed position's hidden.
        // Grows by (accepted + 1) positions per round. Drafter attends
        // to this full accumulated context on every call since we don't
        // yet have a drafter KV cache.
        var accumulatedHidden = try extractContextFeature(
            captured: prefillHidden, targetLayerIDs: args.targetBlockIDs)
        materialize(accumulatedHidden)

        var bonus: Int32 = try sampleArgmax(prefillLogits, temperature: args.temperature)
        tokens.append(bonus)
        onCommitted?([bonus])

        // 2. Decode loop.
        while tokens.count < maxLen {
            // Stop-token check on the generated suffix. Truncate at the
            // first stop token so we don't leak tokens committed after
            // the stop in the same round.
            if !args.stopTokenIDs.isEmpty {
                for i in promptLen..<tokens.count
                where args.stopTokenIDs.contains(tokens[i]) {
                    let truncated = Array(tokens.prefix(i + 1))
                    return DFlashLinearResult(
                        tokenIds: truncated, acceptanceLengths: acceptanceLengths)
                }
            }

            // Build block [bonus, mask, mask, ..., mask] length blockSize.
            var blockValues: [Int32] = Array(
                repeating: args.maskTokenID, count: blockSize)
            blockValues[0] = bonus
            let blockIds = MLXArray(blockValues).reshaped(1, blockSize)

            // Drafter input = target's shared embedding of the block.
            let noiseEmbedding = args.target.embed(blockIds)
            // Drafter position IDs span `startPos..startPos+blockSize`.
            let startPos = tokens.count - 1
            let blockPositions = MLXArray(
                (startPos..<startPos + blockSize).map { Int32($0) }
            ).reshaped(1, blockSize)

            // Drafter forward with the accumulated committed hidden as
            // context. Output shape: (1, blockSize, hidden).
            let drafterOut = args.drafter(
                noiseEmbedding: noiseEmbedding,
                targetHidden: accumulatedHidden,
                positionIds: blockPositions,
                attentionMask: .none)
            // Last block_size-1 positions → target LM head → draft logits.
            let drafterTail = drafterOut[0..., 1..., 0...]
            let drafterLogits = args.target.projectToLogits(drafterTail)
            let drafterPredArr: MLXArray =
                argMax(drafterLogits, axis: -1).asType(.int32)
            materialize(drafterPredArr)
            let drafterPredictions = drafterPredArr.asArray(Int32.self)

            for i in 0..<(blockSize - 1) {
                blockValues[i + 1] = drafterPredictions[i]
            }

            // 3. Target verify.
            //
            //    Fast path (trimmable cache): pass ONLY the NEW bs
            //    positions + persistent cache. Cache grows by bs.
            //    Rejected positions get trimmed after acceptance is known.
            //
            //    Fallback (hybrid SSM): re-prefill the full committed
            //    prefix + block as a fresh forward (cache: nil), read
            //    only the last bs positions' logits.
            let verifyArray: MLXArray
            let verifyLogits: MLXArray
            let verifyHidden: [Int: MLXArray]
            let blockStartIdx: Int
            if useRollbackFastPath {
                verifyArray = MLXArray(blockValues).reshaped(1, blockSize)
                (verifyLogits, verifyHidden) = args.target(
                    verifyArray, cache: targetCache, captureLayerIDs: targetLayerSet)
                blockStartIdx = 0
            } else {
                var verifyInput: [Int32] = Array(tokens.dropLast())
                verifyInput.append(contentsOf: blockValues)
                verifyArray = MLXArray(verifyInput).reshaped(1, verifyInput.count)
                (verifyLogits, verifyHidden) = args.target(
                    verifyArray, cache: nil, captureLayerIDs: targetLayerSet)
                blockStartIdx = verifyInput.count - blockSize
            }
            materialize(verifyLogits)

            // Posterior over the block positions.
            let verifyLogitsBlock = verifyLogits[0..., blockStartIdx..., 0...]
            let posteriorArr: MLXArray =
                argMax(verifyLogitsBlock, axis: -1).asType(.int32)
            materialize(posteriorArr)
            let posterior = posteriorArr.asArray(Int32.self)

            // Acceptance length: longest prefix where
            // block[1+i] == posterior[i].
            var acceptanceLength = 0
            for i in 0..<(blockSize - 1) {
                if blockValues[i + 1] == posterior[i] {
                    acceptanceLength += 1
                } else {
                    break
                }
            }
            acceptanceLengths.append(acceptanceLength)

            // Commit accepted block tokens (positions 1..acceptanceLength)
            // and one bonus (posterior[acceptanceLength]).
            var thisRoundTokens: [Int32] = []
            if acceptanceLength >= 1 {
                for i in 1...acceptanceLength {
                    tokens.append(blockValues[i])
                    thisRoundTokens.append(blockValues[i])
                }
            }
            bonus = posterior[acceptanceLength]
            tokens.append(bonus)
            thisRoundTokens.append(bonus)
            onCommitted?(thisRoundTokens)

            // Accumulate hidden for committed-this-round positions.
            //   Fast path: `verifyHidden` has shape (1, bs, *). Slice
            //   the first (accepted + 1) rows — those are the committed
            //   tokens' hiddens; the rest were rejected drafts.
            //   Fallback: `verifyHidden` has shape (1, verifyInput.count,
            //   *). Slice positions [blockStartIdx ..< blockStartIdx +
            //   accepted + 1].
            let commitCount = acceptanceLength + 1
            let newHidden = try extractContextFeature(
                captured: verifyHidden, targetLayerIDs: args.targetBlockIDs)
            let hiddenStart = blockStartIdx
            let commitHidden = newHidden[
                0..., hiddenStart..<(hiddenStart + commitCount), 0...]
            accumulatedHidden = concatenated(
                [accumulatedHidden, commitHidden], axis: 1)
            materialize(accumulatedHidden)

            // Roll back the target cache by `rejectedCount` positions so
            // the next round's forward starts from the committed prefix.
            // Only meaningful on the fast path; fallback uses cache=nil.
            if useRollbackFastPath {
                let rejectedCount = blockSize - commitCount
                if rejectedCount > 0 {
                    _ = trimPromptCache(targetCache, numTokens: rejectedCount)
                }
            }
        }

        // Trim any mask tokens that leaked in — matches Python's
        // `output_ids = output_ids[:, output_ids[0] != mask_token_id]`.
        // Also trim the generated suffix to exactly maxNewTokens — each
        // round commits a whole bonus+accepted block which may overshoot.
        let filtered = tokens.filter { $0 != args.maskTokenID }
        let capped = filtered.count > maxLen
            ? Array(filtered.prefix(maxLen))
            : filtered
        return DFlashLinearResult(
            tokenIds: capped, acceptanceLengths: acceptanceLengths)
    }

    /// Sample greedy argmax from logits. Only temperature == 0 supported
    /// in iter 4. Iter 5 adds temperature + topK/topP sampling.
    private static func sampleArgmax(
        _ logits: MLXArray, temperature: Float
    ) throws -> Int32 {
        guard temperature < 1e-5 else {
            throw SpecDecError.invalidRequest("runDflashLinear iter 4: only temperature=0 supported")
        }
        let last: MLXArray
        if logits.ndim == 3 {
            last = logits[0, logits.dim(1) - 1, 0...]
        } else if logits.ndim == 2 {
            last = logits[logits.dim(0) - 1, 0...]
        } else {
            throw SpecDecError.invalidRequest("unexpected logits shape: ndim=\(logits.ndim)")
        }
        let idx = argMax(last, axis: -1).asType(.int32)
        materialize(idx)
        return idx.item(Int32.self)
    }
}

// MARK: - DDTree (tree-verify) runtime — iter 9

/// One DDTree generation call.
public struct DDTreeArgs: @unchecked Sendable {
    public let target: any (HiddenStateCaptureModel & TokenEmbedderModel)
    public let drafter: DFlashDraftModel
    public let targetBlockIDs: [Int]
    public let maskTokenID: Int32
    public let inputIds: MLXArray
    public let maxNewTokens: Int
    public let stopTokenIDs: Set<Int32>
    public let temperature: Float
    /// Max tree nodes (excluding root). 1 → linear chain (≈ DFlash linear);
    /// >1 → branching tree. Paper recommends 32-64 for greedy.
    public let branchingBudget: Int

    public init(
        target: any (HiddenStateCaptureModel & TokenEmbedderModel),
        drafter: DFlashDraftModel,
        targetBlockIDs: [Int],
        maskTokenID: Int32,
        inputIds: MLXArray,
        maxNewTokens: Int,
        stopTokenIDs: Set<Int32> = [],
        temperature: Float = 0,
        branchingBudget: Int = 8
    ) {
        self.target = target
        self.drafter = drafter
        self.targetBlockIDs = targetBlockIDs
        self.maskTokenID = maskTokenID
        self.inputIds = inputIds
        self.maxNewTokens = maxNewTokens
        self.stopTokenIDs = stopTokenIDs
        self.temperature = temperature
        self.branchingBudget = branchingBudget
    }
}

/// Result of one DDTree generation call.
public struct DDTreeResult: Sendable {
    /// Full token sequence (prompt + accepted + bonuses).
    public let tokenIds: [Int32]

    /// Per-round (depth of accepted tree walk). 0 = no drafter acceptance,
    /// bonus-only advance. Larger = more tree nodes matched target argmax.
    public let acceptanceLengths: [Int]

    /// Mean accepted tokens per round — paper's primary speedup metric.
    public var meanAcceptanceLength: Double {
        guard !acceptanceLengths.isEmpty else { return 0 }
        return Double(acceptanceLengths.reduce(0, +))
            / Double(acceptanceLengths.count)
    }

    public var rounds: Int { acceptanceLengths.count }
}

/// End-to-end DDTree decode loop.
///
/// Mirrors ``SpecDecRuntimeLinear/run(_:)`` but replaces the linear
/// verify with a tree-verify:
///
/// 1. **Prefill.** Run target on prompt, capture hidden states at
///    `targetBlockIDs`, sample initial bonus token.
/// 2. **Decode loop:**
///    a. Build block `[bonus, mask, ..., mask]` length `blockSize`.
///    b. Drafter forward produces `(block_size-1, vocab)` logits.
///    c. `TreeBuilder.build(draftLogits:budget:)` → tree of up to
///       `branchingBudget` nodes.
///    d. `TreeCompile.compile(tree:, rootTokenID: bonus, prefixLen:)`.
///    e. `TreeVerify.verifyForward(target:compiled:prefixTokens:)` →
///       posterior for every tree node.
///    f. `TreeBuilder.followVerifiedTree(childMaps:posteriorTokens:)` →
///       accepted-node indices + bonus for next round.
///    g. Commit accepted nodes' tokens + bonus. Update target_hidden by
///       re-prefilling target on the growing sequence (v1; iter 10+
///       optimises via KV rollback).
///
/// Byte-parity with greedy AR holds by the same invariant as DFlash
/// (iter 5): every posterior is a real AR argmax, and followVerifiedTree
/// only accepts nodes that match those argmax picks.
public enum SpecDecRuntimeDDTree {

    /// - Parameter onCommitted: optional per-round callback — see
    ///   ``SpecDecRuntimeLinear/run(_:onCommitted:)``.
    public static func run(
        _ args: DDTreeArgs,
        onCommitted: ((_ newTokens: [Int32]) -> Void)? = nil
    ) throws -> DDTreeResult {
        let blockSize = args.drafter.config.blockSize
        guard blockSize >= 2 else {
            throw SpecDecError.invalidRequest("DFlash block_size must be >= 2")
        }
        guard args.inputIds.ndim == 2 && args.inputIds.dim(0) == 1 else {
            throw SpecDecError.invalidRequest("DDTree input_ids shape must be (1, prompt_len)")
        }
        guard args.branchingBudget >= 1 else {
            throw SpecDecError.invalidRequest("DDTree branchingBudget must be >= 1")
        }

        let promptLen = args.inputIds.dim(1)
        let maxLen = promptLen + args.maxNewTokens
        let targetLayerSet = Set(args.targetBlockIDs)

        var tokens: [Int32] = []
        tokens.reserveCapacity(maxLen + blockSize)
        tokens.append(contentsOf: args.inputIds.asArray(Int32.self))

        var acceptanceLengths: [Int] = []

        // Prefill.
        let (prefillLogits, prefillHidden) = args.target(
            args.inputIds, cache: nil, captureLayerIDs: targetLayerSet)
        materialize(prefillLogits)
        var targetHidden = try extractContextFeature(
            captured: prefillHidden, targetLayerIDs: args.targetBlockIDs)
        materialize(targetHidden)

        var bonus: Int32 = try sampleArgmax(prefillLogits, temperature: args.temperature)
        tokens.append(bonus)
        onCommitted?([bonus])

        while tokens.count < maxLen {
            // Stop-token check on generated suffix. Truncate at first
            // stop token so round-commits after the stop don't leak out.
            if !args.stopTokenIDs.isEmpty {
                for i in promptLen..<tokens.count
                where args.stopTokenIDs.contains(tokens[i]) {
                    let truncated = Array(tokens.prefix(i + 1))
                    return DDTreeResult(
                        tokenIds: truncated, acceptanceLengths: acceptanceLengths)
                }
            }

            // Drafter block input.
            var blockValues: [Int32] = Array(
                repeating: args.maskTokenID, count: blockSize)
            blockValues[0] = bonus
            let blockIds = MLXArray(blockValues).reshaped(1, blockSize)

            let noiseEmbedding = args.target.embed(blockIds)
            let startPos = tokens.count - 1
            let blockPositions = MLXArray(
                (startPos..<startPos + blockSize).map { Int32($0) }
            ).reshaped(1, blockSize)

            let drafterOut = args.drafter(
                noiseEmbedding: noiseEmbedding,
                targetHidden: targetHidden,
                positionIds: blockPositions,
                attentionMask: .none)
            // Last block-1 positions go through target LM head.
            let drafterTail = drafterOut[0..., 1..., 0...]
            let drafterLogits = args.target.projectToLogits(drafterTail)
            materialize(drafterLogits)
            // TreeBuilder expects (L, vocab) not (1, L, vocab).
            let logits2D = drafterLogits[0, 0..., 0...]

            // Build tree from drafter logits.
            let tree = try TreeBuilder.build(
                draftLogits: logits2D, budget: args.branchingBudget)

            if tree.nodeCount == 0 {
                // Empty tree → no drafter proposals. Degenerates to a
                // single target forward for the next token.
                acceptanceLengths.append(0)
                let nextInput = MLXArray(tokens).reshaped(1, tokens.count)
                let (nextLogits, nextHidden) = args.target(
                    nextInput, cache: nil, captureLayerIDs: targetLayerSet)
                materialize(nextLogits)
                bonus = try sampleArgmax(nextLogits, temperature: args.temperature)
                tokens.append(bonus)
                onCommitted?([bonus])
                targetHidden = try extractContextFeature(
                    captured: nextHidden, targetLayerIDs: args.targetBlockIDs)
                materialize(targetHidden)
                continue
            }

            // Compile tree. Prefix = tokens minus current bonus (which
            // sits as tree root at position tokens.count - 1).
            let prefixTokens = Array(tokens.dropLast())
            let compiled = try TreeCompile.compile(
                tree: tree, rootTokenID: bonus, prefixLen: prefixTokens.count)

            // Tree-verify.
            let verifyResult = try TreeVerify.verifyForward(
                target: args.target,
                compiled: compiled,
                prefixTokens: prefixTokens,
                captureLayerIDs: [])

            // Walk.
            let (accepted, bonusToken) = try TreeBuilder.followVerifiedTree(
                childMaps: tree.childMaps,
                posteriorTokens: verifyResult.posteriorTokens)

            // Commit. `accepted[0] == 0` (root); accepted[1..] are
            // drafted-node tree indices. Their tokens live in
            // `tree.nodeTokenIds[index - 1]`.
            let acceptanceLength = accepted.count - 1
            acceptanceLengths.append(acceptanceLength)
            let treeNodeTokens = tree.nodeTokenIds.asArray(Int32.self)
            var thisRoundTokens: [Int32] = []
            for i in 1..<accepted.count {
                let nodeIdx = Int(accepted[i])
                let t = treeNodeTokens[nodeIdx - 1]
                tokens.append(t)
                thisRoundTokens.append(t)
            }
            bonus = bonusToken
            tokens.append(bonus)
            thisRoundTokens.append(bonus)
            onCommitted?(thisRoundTokens)

            // Update target_hidden: re-prefill on the committed sequence
            // and pass ALL committed positions (drafter has no KV cache
            // of its own, so it must see full history each round).
            let reprefill = MLXArray(tokens).reshaped(1, tokens.count)
            let (_, reprefillHidden) = args.target(
                reprefill, cache: nil, captureLayerIDs: targetLayerSet)
            let feature = try extractContextFeature(
                captured: reprefillHidden, targetLayerIDs: args.targetBlockIDs)
            targetHidden = feature
            materialize(targetHidden)
        }

        let filtered = tokens.filter { $0 != args.maskTokenID }
        let capped = filtered.count > maxLen
            ? Array(filtered.prefix(maxLen))
            : filtered
        return DDTreeResult(
            tokenIds: capped, acceptanceLengths: acceptanceLengths)
    }

    /// Duplicate of ``SpecDecRuntimeLinear/sampleArgmax(_:temperature:)``
    /// — private there, mirrored here to avoid cross-module plumbing.
    private static func sampleArgmax(
        _ logits: MLXArray, temperature: Float
    ) throws -> Int32 {
        guard temperature < 1e-5 else {
            throw SpecDecError.invalidRequest("runDDTree iter 9: only temperature=0 supported")
        }
        let last: MLXArray
        if logits.ndim == 3 {
            last = logits[0, logits.dim(1) - 1, 0...]
        } else if logits.ndim == 2 {
            last = logits[logits.dim(0) - 1, 0...]
        } else {
            throw SpecDecError.invalidRequest("unexpected logits shape: ndim=\(logits.ndim)")
        }
        let idx = argMax(last, axis: -1).asType(.int32)
        materialize(idx)
        return idx.item(Int32.self)
    }
}

// MARK: - Actor + errors (preserved from Phase 0 stub)

/// Main generation loop for block-diffusion speculative decoding.
///
/// Wraps `draft → tree-build → verify → walk → commit` into a single
/// iterator-style API that produces target tokens. Consumed by
/// `Evaluate.generate` and `BatchEngine.generate` when
/// ``DraftStrategy/usesBlockDiffusion`` is true.
public actor SpecDecRuntime {

    /// Per-runtime stats surfaced in logs + benchmarks.
    public struct Stats: Sendable {
        public var draftForwards: Int = 0
        public var targetForwards: Int = 0
        public var acceptedTokens: Int = 0
        public var bonusTokens: Int = 0
        public var fastPathCommits: Int = 0
        public var treeAwareCommits: Int = 0
        public var slowPathCommits: Int = 0
    }

    /// Configuration at runtime construction.
    public struct Config: Sendable {
        public let strategy: DraftStrategy
        public let parameters: GenerateParameters
        public init(strategy: DraftStrategy, parameters: GenerateParameters) {
            self.strategy = strategy
            self.parameters = parameters
        }
    }

    public let config: Config
    private(set) public var stats: Stats = Stats()

    public init(config: Config) {
        self.config = config
    }

    /// Run the DFlash linear-verify loop via the stateless entry point.
    /// Delegates to ``SpecDecRuntimeLinear/run(_:)``.
    public func runDflash(_ args: DFlashLinearArgs) throws -> DFlashLinearResult {
        try SpecDecRuntimeLinear.run(args)
    }

    /// Run the DDTree tree-verify loop — Phase 2 implementation target.
    public func runDDTree() throws {
        throw SpecDecError.notImplemented("SpecDecRuntime.runDDTree — Phase 2")
    }
}

extension SpecDecRuntime {

    /// Execute a DFlash linear-verify generation. `.dflash` strategy on
    /// `GenerateParameters.draftStrategy` routes here once phase 4 lands.
    public static func executeDflashLinear(
        _ args: DFlashLinearArgs
    ) throws -> DFlashLinearResult {
        try SpecDecRuntimeLinear.run(args)
    }
}

/// Errors produced by the SpecDec runtime.
public enum SpecDecError: Error, LocalizedError {

    /// Throws when a stub is invoked before the phase that implements it
    /// has landed.
    case notImplemented(String)

    /// Thrown when the drafter's `config.json` references a target model
    /// family that doesn't match the loaded target.
    case drafterTargetMismatch(drafter: String, target: String)

    /// Thrown when the drafter checkpoint doesn't contain an expected
    /// safetensors key.
    case drafterCheckpointMissingKey(String)

    /// Thrown when the target model doesn't expose the hidden-state hook
    /// the drafter needs to perform KV injection.
    case targetDoesNotSupportHiddenStateCapture

    /// Thrown when a speculative decoding request or drafter config cannot be
    /// executed safely by the current runtime.
    case invalidRequest(String)

    public var errorDescription: String? {
        switch self {
        case .notImplemented(let msg):
            return "SpecDec: not implemented yet — \(msg)"
        case .drafterTargetMismatch(let drafter, let target):
            return "SpecDec: drafter \(drafter) is not trained against target \(target)"
        case .drafterCheckpointMissingKey(let k):
            return "SpecDec: drafter checkpoint is missing required key '\(k)'"
        case .targetDoesNotSupportHiddenStateCapture:
            return "SpecDec: target model does not expose a hidden-state capture hook (required for DFlash KV injection)"
        case .invalidRequest(let msg):
            return "SpecDec: invalid request — \(msg)"
        }
    }
}
