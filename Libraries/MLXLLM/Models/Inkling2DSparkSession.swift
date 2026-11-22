// Copyright © 2026 Jinho Jang. All rights reserved.
//
// Inkling-2-Mini in the generic dSpark runtime (`SpecDec/DSpark.swift`, run inside TokenIterator):
// the target conforms to `DSparkSpeculativeModel`, the drafter lives in the bundle itself
// (`mtp.safetensors`, loaded with the model), and the draft state lives in the request's cache
// array so ordinary decode steps, verify commits and prefix-cache restores keep it in sync:
//
//   cache[0 ..< 2*slots]            the target (48 KV + 48 packed conv states)
//   cache[2*slots + 0 ..< +layers]  the drafter's context KV per draft layer (KVCacheSimple)
//   cache[2*slots + layers]         the last committed row's 4 captured target features
//                                   (MambaCache, one slot) -> f_rnn
//
// Every target forward over such a cache captures the layer-visit inputs, appends their context
// KV and refreshes the feature slot. A verify forward additionally records conv inputs, so the
// commit trims KV rows and rewinds conv states to the accepted prefix in place.

import Foundation
import MLX
import MLXLMCommon

private let f32 = DType.float32

/// Verify bookkeeping between `beginVerifyCapture` and `commitVerifiedPrefix`.
final class Inkling2PendingVerify: @unchecked Sendable {
    let recorder = Inkling2ConvRecorder()
    var captures: MLXArray?  // [n, 4, H] of the verify rows
    var rows = 0
}

extension Inkling2LoopModel {
    /// The bundle ships the native drafter and this runtime implements its head type.
    public var dsparkAvailable: Bool { drafter != nil && hasDrafterWeights }

    var draftSlotCount: Int { (drafter?.cfg.layers ?? 0) + 1 }

    func cacheHasDraftState(_ cache: [KVCache]?) -> Bool {
        guard let cache, drafter != nil else { return false }
        return cache.count == 2 * args.numCacheSlots + draftSlotCount
    }

    /// The trunk over a cache that may carry draft state: plain forward, or capture + draft-context
    /// maintenance (+ verify recording when a verify is pending). Returns the final normed hidden.
    func trunk(_ ids: MLXArray, cache: [KVCache]?, embeddings: MLXArray? = nil) -> MLXArray {
        guard let cache, cacheHasDraftState(cache), let d = drafter else {
            return model.forward(ids, cache: cache, capture: nil, recorder: nil, embeddings: embeddings)
        }
        d.prepare(lmHeadWeight: denseUnembedWeight())
        let slots = 2 * args.numCacheSlots
        let target = Array(cache[0 ..< slots])
        let ctx = (0 ..< d.cfg.layers).map { cache[slots + $0] as! KVCacheSimple }
        let feat = cache[slots + d.cfg.layers] as! MambaCache
        let cap = Inkling2Capture(ids: d.cfg.targetLayerIds)
        let pending = pendingVerify
        let hidden = model.forward(
            ids, cache: target, capture: cap, recorder: pending?.recorder, embeddings: embeddings)
        let caps = cap.stacked().asType(.bfloat16)  // [T, 4, H]
        let T = caps.dim(0)
        d.writeContext(caps, pos0: ctx[0].offset, caches: ctx)
        feat[0] = caps[(T - 1)...].asType(f32)
        feat.offset += T
        if let pending {
            pending.captures = caps
            pending.rows = T
        }
        return hidden
    }
}

extension Inkling2LoopModel: DSparkSpeculativeModel {
    /// Drafted tokens per round: the block's rows minus the anchor.
    public var dsparkBlockSize: Int? { dsparkAvailable ? (drafter!.cfg.blockSize - 1) : nil }

    /// The drafter is part of the bundle and loads with the model.
    public func attachDSparkDraft(directory: URL) throws -> Bool { dsparkAvailable }

    public func makeDSparkDraftSession() -> (any DSparkDraftSession)? {
        dsparkAvailable ? Inkling2DSparkDraftSession(model: self) : nil
    }

    public func dsparkCacheUnsupportedReason(_ cache: [KVCache]) -> String? {
        cacheHasDraftState(cache)
            ? nil : "the request cache carries no Inkling draft context (built without dSpark)"
    }
}

final class Inkling2DSparkDraftSession: DSparkDraftSession, @unchecked Sendable {
    unowned let model: Inkling2LoopModel

    init(model: Inkling2LoopModel) { self.model = model }

    var vocabularySize: Int? { model.drafter?.cfg.unpaddedVocab }
    var commitsAnyPrefix: Bool { true }

    private func block(seed: Int, cache: [KVCache]) -> (MLXArray, MLXArray) {
        let d = model.drafter!
        let slots = 2 * model.args.numCacheSlots
        let ctx = (0 ..< d.cfg.layers).map { cache[slots + $0] as! KVCacheSimple }
        let feat = cache[slots + d.cfg.layers] as! MambaCache
        let P = ctx[0].offset
        let anchorEmb = model.model.embed(MLXArray([Int32(seed)]).reshaped([1, 1])).reshaped([-1])
        let h = d.blockHidden(anchorEmbedding: anchorEmb, P: P, caches: ctx)
        let fRnn = d.fRnn(feat[0]!.reshaped([-1, d.cfg.hidden]).asType(.bfloat16))
        return (h, fRnn)
    }

    func draft(seed: Int, cache: [KVCache]) -> [Int] {
        draft(seed: seed, cache: cache, count: model.drafter!.cfg.blockSize - 1)
    }

    func draft(seed: Int, cache: [KVCache], count: Int) -> [Int] {
        let (h, fRnn) = block(seed: seed, cache: cache)
        let d = model.drafter!
        return Array(
            d.walk(
                h: h, unembed: { model.lmHead($0) }, anchor: seed, fRnn: fRnn,
                steps: min(count, d.cfg.blockSize - 1)))
    }

    func draft(seed: Int, cache: [KVCache], noise: MLXArray, inverseTemperature: Float) -> [Int] {
        draft(
            seed: seed, cache: cache, count: model.drafter!.cfg.blockSize - 1, noise: noise,
            inverseTemperature: inverseTemperature)
    }

    func draft(seed: Int, cache: [KVCache], count: Int, noise: MLXArray, inverseTemperature: Float)
        -> [Int]
    {
        let (h, fRnn) = block(seed: seed, cache: cache)
        let d = model.drafter!
        return d.walk(
            h: h, unembed: { model.lmHead($0) }, anchor: seed, fRnn: fRnn,
            steps: min(count, d.cfg.blockSize - 1), noise: noise,
            inverseTemperature: inverseTemperature)
    }

    func beginVerifyCapture(cache: [KVCache]) -> Bool {
        guard model.cacheHasDraftState(cache) else { return false }
        model.pendingVerify = Inkling2PendingVerify()
        return true
    }

    func commitVerifiedPrefix(
        cache: [KVCache], checkpoint: SpeculativeCacheListCheckpoint, keep: Int
    ) -> Bool {
        guard let pending = model.pendingVerify, let caps = pending.captures, keep >= 1,
            keep <= pending.rows
        else { return false }
        let drop = pending.rows - keep
        if drop > 0 {
            // Every attention slot (target visits, then the drafter's context) rolls back the rejected rows. A
            // windowed slot holds every row the verify appended, so this is exact for any width; a shortfall
            // would leave slots at different positions, so it is reported and the caller restores the checkpoint.
            let d = model.drafter!
            let attention = Array(0 ..< model.args.numCacheSlots)
                + Array((2 * model.args.numCacheSlots) ..< (2 * model.args.numCacheSlots + d.cfg.layers))
            for i in attention where cache[i].trim(drop) != drop {
                model.pendingVerify = nil
                return false
            }
        }
        pending.recorder.rewind(keep: keep)
        let d = model.drafter!
        let feat = cache[2 * model.args.numCacheSlots + d.cfg.layers] as! MambaCache
        feat[0] = caps[(keep - 1) ..< keep].asType(f32)
        feat.offset -= drop
        model.pendingVerify = nil
        return true
    }

    func endVerifyCapture(cache: [KVCache]) { model.pendingVerify = nil }
}
