import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Single-sequence runtime bridge. The reference graph's explicit padding API
/// remains available, but generic generation cannot yet persist padded positions
/// alongside companion state or wrap that state for multi-sequence decoding.
extension NaiveN05FlashModel: LLMModel {
    enum RuntimeFailure: Error, LocalizedError {
        case unsupportedInput, unsupportedPadding, invalidCache, invalidPrefillSize
        var errorDescription: String? {
            switch self {
            case .unsupportedInput: "Naive-N0.5 runtime requires nonempty single-sequence text input."
            case .unsupportedPadding: "Naive-N0.5 runtime does not yet support padded generation input."
            case .invalidCache: "Naive-N0.5 runtime requires complete, aligned model-owned KV/indexer caches."
            case .invalidPrefillSize: "Naive-N0.5 prefill chunk size must be positive."
            }
        }
    }

    // Router weights/correction retain native F32 while ordinary BF16 weights stay BF16.
    var preservesCheckpointParameterDTypes: Bool { true }
    var maximumSupportedDecodeBatchSize: Int? { 1 }
    var supportsWholeForwardCompilation: Bool { false }
    var cacheStorageDTypeIdentity: String? { "naive-n05-paired-v1" }
    var loraLayers: [Module] { model.layers }

    func newCache(parameters: GenerateParameters?) -> [KVCache] { newCache() }

    private func runtimeTokens(_ text: LMInput.Text) throws -> MLXArray {
        let tokens = text.tokens
        guard tokens.size > 0, tokens.ndim == 1 || (tokens.ndim == 2 && tokens.dim(0) == 1)
        else { throw RuntimeFailure.unsupportedInput }
        if let mask = text.mask {
            guard mask.shape == tokens.shape || mask.shape == [tokens.size] || mask.shape == [1, tokens.size],
                mask.asType(.bool).all().item(Bool.self)
            else { throw RuntimeFailure.unsupportedPadding }
        }
        return tokens.reshaped(1, -1)
    }

    /// Validate every layer before any layer appends. Generic cache substitution
    /// and wrong model geometries must fail without partially advancing a prefix.
    private func runtimeCache(_ cache: [KVCache]?) throws -> [NaiveN05FlashCache]? {
        guard let cache, !cache.isEmpty else { return nil }
        guard let typed = cache as? [NaiveN05FlashCache], typed.count == config.layerCount
        else { throw RuntimeFailure.invalidCache }
        let offset = typed[0].offset
        for (layer, entry) in typed.enumerated() {
            let sparse = config.attentionKinds[layer] == .sparse
            let geometry = sparse ? config.fullAttention : config.slidingAttention
            guard entry.offset == offset, entry.requiresIndexer == sparse,
                entry.window == (sparse ? nil : config.window)
            else { throw RuntimeFailure.invalidCache }
            let rows = entry.state
            if rows.isEmpty {
                guard offset == 0 else { throw RuntimeFailure.invalidCache }
                continue
            }
            let retained = sparse ? offset : min(offset, config.window - 1)
            guard rows.count == (sparse ? 3 : 2),
                rows[0].shape == [1, geometry.kvHeads, retained, geometry.keyDimensions],
                rows[1].shape == [1, geometry.kvHeads, retained, geometry.valueDimensions]
            else { throw RuntimeFailure.invalidCache }
            if sparse {
                guard rows[2].shape == [1, 1, retained, config.indexerDimensions],
                    rows[2].dtype == .float32
                else { throw RuntimeFailure.invalidCache }
            }
        }
        return typed
    }

    private func runtimeForward(_ tokens: MLXArray, cache: [NaiveN05FlashCache]?) throws -> MLXArray {
        // append replaces immutable arrays; retaining references does not copy KV.
        let snapshots = cache?.map { ($0.state, $0.metaState, $0.offset) }
        do {
            return try self(tokens, padding: nil, cache: cache)
        } catch {
            if let cache, let snapshots {
                for (entry, snapshot) in zip(cache, snapshots) {
                    precondition(entry.restoreDiskCacheState(snapshot.0,
                        metadata: snapshot.1, offset: snapshot.2), "Invalid owned cache snapshot")
                }
            }
            throw error
        }
    }

    /// Prefill chunk: the caller's value, except the library default (`GenerateParameters.prefillStepSize` = 512,
    /// indistinguishable from "unset"), which becomes 4,096 — Naive-N0.5's measured best with the blocked/sparse
    /// prefill attention and the expert-sorted tiles (5k prompt: 483-543 tok/s at 512, 608-633 at 2,048, 652-663 at
    /// 4,096; vMLX Python also prefills N0.5 at 4,096). `VMLX_NAIVE_PREFILL_STEP=<n>` overrides both.
    static func prefillStep(requested: Int?) -> Int {
        if let raw = ProcessInfo.processInfo.environment["VMLX_NAIVE_PREFILL_STEP"], let n = Int(raw), n > 0 {
            return n
        }
        guard let requested, requested != 512 else { return 4096 }
        return requested
    }

    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        try Task.checkCancellation()
        guard input.image == nil, input.video == nil, input.audio == nil
        else { throw RuntimeFailure.unsupportedInput }
        var tokens = try runtimeTokens(input.text).reshaped(-1)
        let owned = try runtimeCache(cache)
        let step = Self.prefillStep(requested: windowSize)
        guard step > 0 else { throw RuntimeFailure.invalidPrefillSize }
        // With no cache, retain the entire prompt for the generation forward.
        // Consuming chunks here would silently discard their context.
        while owned != nil && tokens.size > step {
            try Task.checkCancellation()
            _ = try runtimeForward(tokens[..<step].reshaped(1, -1), cache: owned)
            MLX.eval(cache)
            tokens = tokens[step...]
            PrefillProgressReporter.reportCompletedUnits(input.text.tokens.size - tokens.size)
            Memory.clearCache()
        }
        try Task.checkCancellation()
        return .tokens(LMInput.Text(tokens: tokens))
    }

    func replayForward(_ tokens: MLXArray, cache: [KVCache]?) throws -> MLXArray {
        try Task.checkCancellation()
        let normalized = try runtimeTokens(LMInput.Text(tokens: tokens))
        let owned = try runtimeCache(cache)
        return try runtimeForward(normalized, cache: owned)
    }

    func callAsFunction(_ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?) -> LMOutput {
        // LanguageModel's generation API is nonthrowing. Input admission belongs
        // to throwing prepare/replay; a bypass is a violated runtime invariant,
        // never a reason to manufacture logits or drop a supplied mask.
        do {
            let tokens = try runtimeTokens(input)
            let owned = try runtimeCache(cache)
            return LMOutput(logits: try runtimeForward(tokens, cache: owned))
        } catch {
            preconditionFailure("Invalid Naive-N0.5 generation admission: \(error)")
        }
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        self(LMInput.Text(tokens: inputs), cache: cache, state: nil).logits
    }
}
