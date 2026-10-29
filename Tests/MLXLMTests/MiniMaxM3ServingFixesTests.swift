// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Cmlx
import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

/// Fixes from the Osaurus GUI proof of MiniMax-M3.1 (m5-max, 2026-10-29): what the serving path did that the harness
/// path did not.
///
/// - Memory: Safe Auto set MLX's scheduling limit to the resident weights + 4 GiB; past ~26k tokens of M3 context every
///   eval waited for the queue to drain. M3 now takes the resident-pool policy (MLX's native ceiling), like DSV4 / GLM.
/// - Prefill step: M3 recommends 2048 (the measured fastest step); the host default was 512.
/// - DSpark rounds: the checkpoint was a second owner of every lane buffer, so each verify copied every layer's whole
///   K / V / index-key buffer. The row checkpoint keeps the buffers donatable; tokens are unchanged.
/// - Lifetime: the fused MoE kernels' per-layer view lived in a process-wide table and kept a released model's routed
///   banks alive (a reload then held two models). It now lives on the block.
@Suite("MiniMax-M3 serving-path fixes", .serialized)
struct MiniMaxM3ServingFixesTests {
    typealias DS = MiniMaxM3DSparkTests

    // MARK: memory policy

    static let gib = UInt64(1 << 30)

    static func m3Facts(_ type: String = "minimax_m3_vl") -> LoadBundleFacts {
        LoadBundleFacts(
            totalSafetensorsBytes: 100_223_033_936, isRouted: true, physicalMemory: 128 * gib, modelType: type,
            hasJangConfig: true, numRoutedExperts: 128, topK: 4)
    }

    static func host() -> MemoryStatus {
        MemoryStatus(
            memoryLimit: Int(128 * gib), cacheLimit: Int(128 * gib), recommendedWorkingSetBytes: 121_634_816_000,
            physicalMemory: 128 * gib, currentRSS: 0)
    }

    @Test("Safe Auto: an M3 bundle keeps MLX's native ceiling; another routed bundle keeps weights + 4 GiB (control)")
    func m3BundleIsNotThrottledBySafeAuto() {
        for type in ["minimax_m3_vl", "minimax_m3"] {
            let facts = Self.m3Facts(type)
            #expect(facts.isMiniMaxM3Routed && facts.requiresUncappedResidentPools, "\(type)")
            let plan = VMLXServerRuntimeSettings().resolvedMemorySafetyPlan(
                baseLoadConfiguration: .osaurusProduction, bundleFacts: facts, host: Self.host())
            #expect(plan.loadConfiguration.memoryLimit == .unlimited, "\(type): \(plan.loadConfiguration.memoryLimit)")
        }
        // Control: the same bundle under another routed model type gets the weights + 4 GiB allowance the GUI proof ran
        // under (104,518,001,232 bytes), so the M3 branch is what changed it.
        let other = Self.m3Facts("qwen3_moe")
        #expect(!other.requiresUncappedResidentPools)
        let plan = VMLXServerRuntimeSettings().resolvedMemorySafetyPlan(
            baseLoadConfiguration: .osaurusProduction, bundleFacts: other, host: Self.host())
        #expect(plan.loadConfiguration.memoryLimit == .absolute(100_223_033_936 + (4 << 30)))
        // A dense M3-typed config is not a routed bundle.
        let dense = LoadBundleFacts(
            totalSafetensorsBytes: 8 * Self.gib, isRouted: false, physicalMemory: 128 * Self.gib,
            modelType: "minimax_m3")
        #expect(!dense.isMiniMaxM3Routed)
    }

    @Test("Loading an M3 bundle applies MLX's native ceiling to the process (not the stale allowance)")
    func m3LoadAppliesNativeCeiling() {
        let priorMemoryLimit = MLX.Memory.memoryLimit
        let priorCacheLimit = MLX.Memory.cacheLimit
        defer {
            MLX.Memory.memoryLimit = priorMemoryLimit
            MLX.Memory.cacheLimit = priorCacheLimit
        }
        MLX.Memory.memoryLimit = 100_223_033_936 + (4 << 30)
        let applied = applyResidentPoolProcessMemoryLimitsIfNeeded(
            facts: Self.m3Facts(), allocatorCacheLimit: 2 << 30, recommendedWorkingSetBytes: 121_634_816_000)
        let physical = Int(128 * Self.gib)
        let physical95 = (physical / 20) * 19 + ((physical % 20) * 19) / 20
        #expect(applied == min(physical95, 121_634_816_000 + 121_634_816_000 / 2))
        #expect(MLX.Memory.memoryLimit == applied)
        #expect(MLX.Memory.cacheLimit == 2 << 30)
    }

    // MARK: prefill step

    @Test("M3 recommends the measured fastest prefill step (2048)")
    func recommendedPrefillStep() throws {
        let model = try DS.m3Model()
        #expect(model.recommendedPrefillStepSize == 2048)
        #expect(ModelContainer(context: ModelContext(
            configuration: ModelConfiguration(id: "m3"), model: model, processor: StandInProcessor(),
            tokenizer: DS.FixtureTokenizer())).recommendedPrefillStepSize == 2048)
    }

    struct StandInProcessor: UserInputProcessor {
        func prepare(input: UserInput) async throws -> LMInput { LMInput(tokens: MLXArray([Int32(1)])) }
    }

    // MARK: DSpark rounds keep the lanes donatable

    /// The address of an evaluated array's buffer (its first byte).
    static func address(_ a: MLXArray) -> UInt {
        UInt(bitPattern: mlx_array_data_uint8(a.ctx))
    }

    /// Per layer: (K buffer, V buffer, index-key buffer) addresses and capacities.
    static func lanes(_ cache: [KVCache]) -> [[(UInt, Int)]] {
        cache.map { layer in
            if let s = layer as? MiniMaxM3SparseCache {
                return [s.kv.keys, s.kv.values, s.indexStorage].compactMap { $0 }.map { (address($0), $0.dim(2)) }
            }
            if let k = layer as? KVCacheSimple {
                return [k.keys, k.values].compactMap { $0 }.map { (address($0), $0.dim(2)) }
            }
            return []
        }
    }

    struct CopyCount { var copied = 0, inPlace = 0, grown = 0 }

    /// Counts lane buffers that moved between two snapshots: grown (capacity changed: an expected copy), copied (same
    /// capacity, new buffer: copy on write), in place (same buffer). `sparse` / `dense`: layers whose cache is / is not a
    /// `MiniMaxM3SparseCache` (the tiny model has one dense-attention layer; M3.1's 60 layers are all sparse).
    static func count(
        _ before: [[(UInt, Int)]], _ after: [[(UInt, Int)]], cache: [KVCache], sparse: inout CopyCount,
        dense: inout CopyCount
    ) {
        for (i, (b, a)) in zip(before, after).enumerated() {
            for (x, y) in zip(b, a) {
                if cache[i] is MiniMaxM3SparseCache {
                    if x.1 != y.1 { sparse.grown += 1 } else if x.0 != y.0 { sparse.copied += 1 } else { sparse.inPlace += 1 }
                } else {
                    if x.1 != y.1 { dense.grown += 1 } else if x.0 != y.0 { dense.copied += 1 } else { dense.inPlace += 1 }
                }
            }
        }
    }

    /// The DSpark loop of `MiniMaxM3DSparkTests.dsparkRun` with every forward evaluated and the lane buffers compared
    /// before / after it.
    static func countedRun(_ model: MiniMaxM3Model, prompt: [Int], reference: [Int], count n: Int)
        throws -> (tokens: [Int], verify: CopyCount, ar: CopyCount, dense: CopyCount, stats: DSparkGenerationStats)
    {
        let inner = try #require(model.makeDSparkDraftSession() as? MiniMaxM3DSparkSession)
        let scripted = DS.ScriptedSession(inner: inner, reference: prompt + reference) { $0 % 5 == 4 }
        var decoder = DSparkDecoder(
            session: scripted, blockSize: DS.draftBlock, stopTokenIDs: [], governor: DS.alwaysSpeculate,
            ladderTop: model.dsparkLadderTop, pinnedWidth: nil)
        let cache = model.newCache(parameters: nil)
        var y = DS.argmax(DS.prefill(model, prompt, cache: cache))
        var verify = CopyCount(), ar = CopyCount(), dense = CopyCount()
        let forward: ([Int]) -> MLXArray = { ids in
            let before = lanes(cache)
            let logits = model(DS.tokens(ids), cache: cache)
            eval(logits)
            eval(cache.flatMap { $0.innerState() })
            if ids.count > 1 {
                count(before, lanes(cache), cache: cache, sparse: &verify, dense: &dense)
            } else {
                count(before, lanes(cache), cache: cache, sparse: &ar, dense: &dense)
            }
            return logits
        }
        var queue: [Int] = []
        var out: [Int] = []
        while out.count < n {
            if !queue.isEmpty {
                out.append(queue.removeFirst())
                continue
            }
            let seed = y
            switch decoder.step(seed: seed, cache: cache, remaining: n - out.count, forward: forward) {
            case .autoregressive:
                y = DS.argmax(forward([seed])[0, -1])
                out.append(seed)
            case .committed(let committed, let next):
                y = next.item(Int.self)
                queue = Array(committed.dropFirst())
                out.append(committed[0])
            }
        }
        decoder.finish()
        return (out, verify, ar, dense, decoder.stats)
    }

    @Test("DSpark verify rounds append in place with the row checkpoint; the reference checkpoint copies (control)",
          arguments: [(773, 11), (1029, 13)])
    func dsparkRoundsKeepLanesDonatable(promptLength: Int, seed: Int) throws {
        let directory = DS.temporaryDirectory("donation")
        defer { try? FileManager.default.removeItem(at: directory) }
        try DS.writeReleaseDrafter(directory)
        try Device.withDefaultDevice(.gpu) {
            let model = try DS.attachedModel(directory)
            let prompt = DS.ids(promptLength, seed: UInt64(seed))
            let (reference, _) = DS.greedyAR(model, prompt, count: 45)
            let row = try MiniMaxM3SparseCache.$referenceCheckpointOverride.withValue(false) {
                try Self.countedRun(model, prompt: prompt, reference: reference, count: 45)
            }
            let ref = try MiniMaxM3SparseCache.$referenceCheckpointOverride.withValue(true) {
                try Self.countedRun(model, prompt: prompt, reference: reference, count: 45)
            }
            print("[donation] prompt \(promptLength): sparse lanes, row checkpoint verify \(row.verify) ar \(row.ar); "
                + "reference checkpoint verify \(ref.verify) ar \(ref.ar); dense layer (reference in both) \(row.dense); "
                + "rounds \(row.stats.specSteps)")
            #expect(row.tokens == reference && ref.tokens == reference, "DSpark output != AR")
            #expect(row.stats.specSteps > 3 && row.stats.accepted < row.stats.drafted)
            // Sparse layers (K, V, index-key lanes): every verify forward appends in place.
            #expect(row.verify.copied == 0, "row checkpoint: \(row.verify.copied) lane buffers copied on write")
            #expect(row.verify.inPlace == 3 * 3 * row.stats.specSteps && row.ar.copied == 0)
            // Control: the reference checkpoint makes every round's append copy each of them.
            #expect(ref.verify.copied == 3 * 3 * ref.stats.specSteps && ref.verify.inPlace == 0)
            // The dense layer's KVCacheSimple keeps the generic reference checkpoint in both arms (its two lanes copy).
            #expect(row.dense.copied == 2 * row.stats.specSteps)
        }
    }

    // MARK: fused MoE view lifetime

    @Test("The fused MoE kernels' layer view dies with its block (and the kernels still cover it)")
    func fusedMoELayerViewDiesWithBlock() throws {
        try Device.withDefaultDevice(.gpu) {
            weak var weakShared: MiniMaxM3MLP?
            weak var weakRouted: JANGTRoutedExpertLayer?
            weak var weakBox: MiniMaxM3FusedMoE.LayerBox?
            let activeBefore = MLX.Memory.activeMemory
            do {
                let g = (kgu: 2.0, kdn: 3.0, nog: 0, nod: 0, seed: 200)
                let (routed, _) = try MiniMaxM3FusedMoETests.routed(
                    kgu: g.kgu, kdn: g.kdn, nog: g.nog, nod: g.nod, seed: g.seed)
                var digests: [String: String] = [:]
                let block = try MiniMaxM3FusedMoETests.block(routed, seed: g.seed, digests: &digests)
                let (x, r) = MiniMaxM3FusedMoETests.rows(seed: g.seed)
                #expect(MiniMaxM3FusedMoE.shared.covers(block, x: x[0..., ..<1]))
                let out = MiniMaxM3FusedMoE.shared.residual(block, xNormed: x[0..., ..<1], resid: r[0..., ..<1])
                eval(out)
                weakShared = block.sharedExperts
                weakRouted = routed
                weakBox = block._fusedMoELayer
                #expect(weakBox?.layer != nil)
            }
            MLX.Memory.clearCache()
            #expect(weakBox == nil, "the fused view outlived its block")
            #expect(weakShared == nil && weakRouted == nil, "the block's modules outlived the block")
            print("[fused MoE lifetime] MLX active before \(activeBefore) after \(MLX.Memory.activeMemory)")
        }
    }
}
