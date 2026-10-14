// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// DSpark drafter (MiniCPM-V 4.7 `dspark/`): config contract, module topology and draft-row mapping.

import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLMCommon

final class DSparkDrafterTests: XCTestCase {

    /// The MiniCPM-V 4.7 1B DSpark config as shipped (contract split between the root and `dflash_config`).
    private func dsparkConfig(markovHeadType: String = "vanilla", extra: [String: Any] = [:]) -> [String: Any] {
        var root: [String: Any] = [
            "architectures": ["Qwen3DSparkModel"], "block_size": 15, "hidden_size": 64,
            "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
            "intermediate_size": 128, "vocab_size": 300, "rms_norm_eps": 1e-6,
            "rope_parameters": ["rope_theta": 10_000_000, "rope_type": "default"],
            "layer_types": ["full_attention", "full_attention"], "sliding_window": NSNull(),
            "markov_rank": 8, "markov_head_type": markovHeadType, "enable_confidence_head": true,
            "dflash_config": [
                "mask_token_id": 299, "target_layer_ids": [1, 3], "projector_type": "dspark",
                "markov_rank": 8, "enable_confidence_head": true,
            ] as [String: Any],
        ]
        for (k, v) in extra { root[k] = v }
        return root
    }

    func testDSparkContractNormalizesBlockToVerifyWidth() throws {
        let c = try DFlash2Configuration(json: dsparkConfig())
        XCTAssertTrue(c.isDSpark)
        XCTAssertFalse(c.isDFlash2)
        XCTAssertTrue(c.isRunnable)
        XCTAssertTrue(c.anchorRowDrafts)
        // Declared 15 counts drafts (the anchor row drafts too); the lane speaks verify width.
        XCTAssertEqual(c.blockSize, 16)
        XCTAssertEqual(c.markovRank, 8)
        XCTAssertEqual(c.ropeTheta, 10_000_000)
        XCTAssertFalse(c.ropeTraditional, "MiniCPM-V 4.7 DSpark is neox")
        XCTAssertTrue(c.hasConfidenceHead)
        XCTAssertNil(c.slidingWindow)
    }

    func testInterleavedRopeIsReadFromTheStamp() throws {
        let c = try DFlash2Configuration(json: dsparkConfig(extra: ["rope_is_neox_style": false]))
        XCTAssertTrue(c.ropeTraditional)
    }

    func testUnportedMarkovHeadIsRefusedByName() {
        XCTAssertThrowsError(try DFlash2Configuration(json: dsparkConfig(markovHeadType: "gated"))) { error in
            guard case DFlash2LoadError.unsupportedMarkovHead(let kind) = error else {
                return XCTFail("unexpected \(error)")
            }
            XCTAssertEqual(kind, "gated")
        }
    }

    func testDSparkModelHasMarkovAndConfidenceHeadsAndNoDFlash2Modules() throws {
        let model = DFlash2DraftModel(try DFlash2Configuration(json: dsparkConfig()))
        let keys = Set(model.parameters().flattened().map(\.0))
        XCTAssertTrue(keys.contains("markov_head.markov_w1.weight"))
        XCTAssertTrue(keys.contains("markov_head.markov_w2.weight"))
        XCTAssertTrue(keys.contains("confidence_head.proj.weight"))
        XCTAssertTrue(keys.contains("fc.weight"))
        XCTAssertFalse(keys.contains { $0.hasPrefix("candidate_selector.") })
        XCTAssertFalse(keys.contains { $0.contains("attention_conv") || $0.contains("mlp_conv") })
        XCTAssertFalse(model.supportsLattice, "a DSpark drafter drafts one chain, no tree lattice")
    }

    func testDSparkExtremeBlockRejectsWithoutOverflow() throws {
        let config = dsparkConfig(extra: ["block_size": Int.max])
        XCTAssertThrowsError(try DFlash2Configuration(json: config))
    }

    func testDSparkExtremeConfidenceRankRejectsWithoutOverflow() throws {
        var config = dsparkConfig()
        var inner = config["dflash_config"] as! [String: Any]
        inner["markov_rank"] = Int.max
        config["dflash_config"] = inner
        XCTAssertThrowsError(try DFlash2ArtifactMetadata.requiredShapes(
            configData: JSONSerialization.data(withJSONObject: config)))
    }

    func testDSparkLadderStaysWithinTheTrainedBlock() {
        XCTAssertEqual(DFlash2WidthChooser.dsparkLadder(trained: 16), [3, 5, 8, 16])
        XCTAssertEqual(DFlash2WidthChooser.dsparkLadder(trained: 6), [3, 5, 6])
    }

    func testNativeCompanionFileRoundtripAndIdentityRejections() throws {
        let config = try DFlash2Configuration(json: dsparkConfig())
        let tokens = [1, 2, 3]
        for dtype: DType in [.float16, .bfloat16, .float32] {
            let cache: [KVCache] = (0..<config.numHiddenLayers).map { index in
                let layer = KVCacheSimple()
                let state = MLXArray((0..<(2 * 3 * 16)).map { Float($0 + index) / 16 })
                    .reshaped(1, 2, 3, 16).asType(dtype)
                layer.state = [state, state + MLXArray(Float(1)).asType(dtype)]
                return layer
            }
            let snapshot = try XCTUnwrap(DSparkNativeCompanionSnapshot.capture(
                cache: cache, config: config, draftIdentity: "draft-v1", targetIdentity: "target-v1",
                embeddingDType: dtype, tokens: tokens, mediaSalt: "media-v1"))
            let url = FileManager.default.temporaryDirectory
                .appendingPathComponent("dspark-reference-\(UUID().uuidString).safetensors")
            defer { try? FileManager.default.removeItem(at: url) }
            var payload = TQDiskSerializer.serialize(cache: cache, preserveStandardKVStorageDType: true)
            payload.merge(snapshot.arrays) { original, _ in original }
            try MLX.save(arrays: payload, url: url)
            let loaded = try MLX.loadArrays(url: url)
            func restore(_ arrays: [String: MLXArray], draft: String = "draft-v1",
                         target: String = "target-v1", prefix: [Int] = [1, 2, 3],
                         salt: String = "media-v1", embedding: DType? = nil) -> [KVCache]? {
                DSparkNativeCompanionSnapshot.restore(arrays: arrays, config: config,
                    draftIdentity: draft, targetIdentity: target, embeddingDType: embedding ?? dtype,
                    tokens: prefix, mediaSalt: salt)
            }
            let restored = try XCTUnwrap(restore(loaded))
            for (original, actual) in zip(cache, restored) {
                XCTAssertEqual(actual.offset, tokens.count)
                for (expected, value) in zip(original.state, actual.state) {
                    XCTAssertEqual(value.dtype, dtype)
                    XCTAssertEqual(MLX.max(MLX.abs(expected - value)).item(Float.self), 0)
                }
            }
            XCTAssertNil(restore(loaded, draft: "changed-draft"))
            XCTAssertNil(restore(loaded, target: "changed-target"))
            XCTAssertNil(restore(loaded, prefix: [1, 2]))
            XCTAssertNil(restore(loaded, prefix: [1, 2, 4]))
            XCTAssertNil(restore(loaded, salt: "other-media"))
            XCTAssertNil(restore(loaded, embedding: .int32))
            for key in [DSparkNativeCompanionSnapshot.metadataKey, "dspark_ref_1_v"] {
                var missing = loaded
                missing.removeValue(forKey: key)
                XCTAssertNil(restore(missing))
                var targetCaches: [KVCache] = [KVCacheSimple(), KVCacheSimple()]
                XCTAssertEqual(restoreFromDiskArrays(missing, into: &targetCaches), 3)
                XCTAssertEqual(targetCaches[0].state[0].dtype, dtype)
            }
            var incompatible = loaded
            incompatible["dspark_ref_0_k"] = loaded["dspark_ref_0_k"]!.asType(.int8)
            XCTAssertNil(restore(incompatible))
            let int8Companion = incompatible
            incompatible["dspark_ref_0_k"] = MLXArray.zeros([1, 2, 2, 16], dtype: dtype)
            XCTAssertNil(restore(incompatible))
            let wrongShapeCompanion = incompatible

            // A stale companion: captured for the shorter prefix [1, 2] of an
            // earlier generation, but riding in the payload keyed by [1, 2, 3].
            let staleCache: [KVCache] = cache.map { layer in
                let stale = KVCacheSimple()
                stale.state = layer.state.map { $0[0..., 0..., ..<2, 0...] }
                return stale
            }
            let staleSnapshot = try XCTUnwrap(DSparkNativeCompanionSnapshot.capture(
                cache: staleCache, config: config, draftIdentity: "draft-v1", targetIdentity: "target-v1",
                embeddingDType: dtype, tokens: [1, 2], mediaSalt: "media-v1"))
            var stale = TQDiskSerializer.serialize(cache: cache, preserveStandardKVStorageDType: true)
            stale.merge(staleSnapshot.arrays) { original, _ in original }
            XCTAssertNil(restore(stale))
            XCTAssertEqual(try XCTUnwrap(restore(stale, prefix: [1, 2]))[0].offset, 2,
                           "the stale companion itself is intact; only its boundary no longer matches")
            // Corrupt header bytes: same key/dtype/shape class, undecodable content.
            var corruptHeader = loaded
            corruptHeader[DSparkNativeCompanionSnapshot.metadataKey] = MLXArray([Int32](repeating: 300, count: 8))
            XCTAssertNil(restore(corruptHeader))
            var wrongHeaderDType = loaded
            wrongHeaderDType[DSparkNativeCompanionSnapshot.metadataKey] =
                loaded[DSparkNativeCompanionSnapshot.metadataKey]!.asType(.float32)
            XCTAssertNil(restore(wrongHeaderDType))

            // Item 8 of the handoff: every rejected companion (stale OR incompatible,
            // not only missing) must leave the SAME payload's target restore usable on
            // the iterator's `requirePromptBoundary: true` path. The reference never
            // refuses a target prefix; it falls back to target restore + suffix-only
            // proposal context.
            let rejectedButTargetUsable: [(String, [String: MLXArray])] = [
                ("stale-boundary", stale), ("int8-kv", int8Companion),
                ("wrong-shape-kv", wrongShapeCompanion), ("corrupt-header", corruptHeader),
                ("wrong-header-dtype", wrongHeaderDType),
                ("changed-draft-identity", loaded), ("changed-target-identity", loaded),
            ]
            for (label, arrays) in rejectedButTargetUsable {
                switch label {
                case "changed-draft-identity": XCTAssertNil(restore(arrays, draft: "changed-draft"), label)
                case "changed-target-identity": XCTAssertNil(restore(arrays, target: "changed-target"), label)
                default: XCTAssertNil(restore(arrays), label)
                }
                var targetCaches: [KVCache] = [KVCacheSimple(), KVCacheSimple()]
                XCTAssertEqual(
                    restoreFromDiskArrays(arrays, into: &targetCaches, requirePromptBoundary: true), 3,
                    "\(label): target restore must survive companion rejection")
                for (expected, actual) in zip(cache, targetCaches) {
                    XCTAssertEqual(actual.offset, 3, label)
                    for (want, got) in zip(expected.state, actual.state) {
                        XCTAssertEqual(got.dtype, dtype, label)
                        XCTAssertEqual(MLX.max(MLX.abs(want - got)).item(Float.self), 0, label)
                    }
                }
            }
            _ = cache[0].trim(1)
            XCTAssertEqual(try XCTUnwrap(restore(snapshot.arrays))[0].offset, 3,
                           "later target/draft rollback cannot change the published boundary snapshot")
        }
    }

    private final class ContextTarget: Module, LanguageModel, TokenEmbedderModel {
        let embedding: MLXArray
        init(dtype: DType) {
            self.embedding = MLXArray((0..<(300 * 64)).map { Float(sin(Double($0) * 0.013)) })
                .reshaped(300, 64).asType(dtype)
            super.init()
        }
        func embed(_ tokens: MLXArray) -> MLXArray { MLX.take(embedding, tokens, axis: 0) }
        func projectToLogits(_ hidden: MLXArray) -> MLXArray { hidden.matmul(embedding.T) }
        func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
            projectToLogits(embed(inputs))
        }
        func newCache(parameters: GenerateParameters?) -> [KVCache] { [] }
        func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
            .tokens(input.text)
        }
    }

    /// Same context projection batch shapes on both arms: only the moment at
    /// which K/V are appended differs. No sampler or RNG override is involved.
    func testDSparkContextOnlyAppendMatchesProposalContext() throws {
        try checkContextOnlyAppend(dtype: .float32)
        try checkContextOnlyAppend(dtype: .bfloat16)
    }

    private func checkContextOnlyAppend(dtype: DType) throws {
        let model = DFlash2DraftModel(try DFlash2Configuration(json: dsparkConfig()))
        model.update(parameters: model.parameters().mapValues { $0.asType(dtype) })
        let target = ContextTarget(dtype: dtype)
        let inputs = MLXArray([Int32(1), 299, 299, 299]).reshaped(1, 4)
        let direct = model.makeCache()
        let seeded = model.makeCache()
        let hidden = MLXArray((0..<(7 * 128)).map { Float(cos(Double($0) * 0.021)) })
            .reshaped(1, 7, 128).asType(dtype)
        var directLast: MLXArray?
        for range in [0..<3, 3..<7] {
            let chunk = hidden[0..., range, 0...]
            directLast = model.hiddenStates(inputs: inputs, targetHidden: chunk, cache: direct,
                                             embedder: target)
            model.appendContext(targetHidden: chunk, dtype: dtype, cache: seeded)
        }
        let empty = MLXArray.zeros([1, 0, 128])
        let actual = model.hiddenStates(inputs: inputs, targetHidden: empty, cache: seeded,
                                        embedder: target)
        MLX.eval(actual, directLast!)
        XCTAssertEqual(MLX.max(MLX.abs(actual - directLast!)).item(Float.self), 0)
        for (a, b) in zip(direct, seeded) {
            XCTAssertEqual(a.offset, 7)
            XCTAssertEqual(b.offset, 7, "empty pending features must not append twice")
            for (x, y) in zip(a.state, b.state) {
                XCTAssertEqual(x.dtype, y.dtype)
                XCTAssertEqual(y.dtype, dtype)
                XCTAssertEqual(MLX.max(MLX.abs(x - y)).item(Float.self), 0)
            }
        }
    }

    func testChunkedContextRequiresFullAttentionDSpark() throws {
        let full = DFlash2DraftModel(try DFlash2Configuration(json: dsparkConfig()))
        XCTAssertTrue(full.supportsChunkedContextPrefill)
        let sliding = DFlash2DraftModel(try DFlash2Configuration(json: dsparkConfig(extra: [
            "layer_types": ["full_attention", "sliding_attention"], "sliding_window": 16,
        ])))
        XCTAssertFalse(sliding.supportsChunkedContextPrefill)
    }

    /// Diagnostic, not a bit-equality gate: splitting projection M may choose
    /// different GEMM reductions. Report the observed difference explicitly.
    func testDSparkContextProjectionBatchShapeDiagnostic() throws {
        let rows = 2053
        for dtype: DType in [.float32, .bfloat16] {
            let model = DFlash2DraftModel(try DFlash2Configuration(json: dsparkConfig()))
            model.update(parameters: model.parameters().mapValues { $0.asType(dtype) })
            let target = ContextTarget(dtype: dtype)
            let inputs = MLXArray([Int32(1), 299, 299, 299]).reshaped(1, 4)
            let hidden = MLXArray((0..<(rows * 128)).map { Float(cos(Double($0) * 0.021)) })
                .reshaped(1, rows, 128).asType(dtype)
            let wholeCache = model.makeCache()
            let whole = model.hiddenStates(inputs: inputs, targetHidden: hidden, cache: wholeCache,
                                           embedder: target)
            let wholeLogits = target.projectToLogits(whole)
            MLX.eval(whole, wholeLogits)
            for chunkSize in [7, 64, 512] {
                let chunked = model.makeCache()
                for start in stride(from: 0, to: rows, by: chunkSize) {
                    let end = min(rows, start + chunkSize)
                    model.appendContext(targetHidden: hidden[0..., start..<end, 0...],
                                        dtype: dtype, cache: chunked)
                    MLX.eval(chunked)
                }
                let actual = model.hiddenStates(inputs: inputs,
                    targetHidden: MLXArray.zeros([1, 0, 128], dtype: dtype), cache: chunked,
                    embedder: target)
                let actualLogits = target.projectToLogits(actual)
                let hiddenDelta = MLX.max(MLX.abs(actual - whole)).item(Float.self)
                let logitDelta = MLX.max(MLX.abs(actualLogits - wholeLogits)).item(Float.self)
                var kvDelta: Float = 0
                for (reference, observed) in zip(wholeCache, chunked) {
                    XCTAssertEqual(observed.offset, rows)
                    for (x, y) in zip(reference.state, observed.state) {
                        XCTAssertEqual(x.shape, y.shape)
                        XCTAssertEqual(y.dtype, dtype)
                        kvDelta = max(kvDelta, MLX.max(MLX.abs(x - y)).item(Float.self))
                    }
                }
                let ids = MLX.argMax(actualLogits, axis: -1).asArray(Int32.self)
                let referenceIDs = MLX.argMax(wholeLogits, axis: -1).asArray(Int32.self)
                let mismatches = zip(ids, referenceIDs).filter { $0 != $1 }.count
                XCTAssertTrue(hiddenDelta.isFinite && logitDelta.isFinite && kvDelta.isFinite)
                print("[DSparkContextShape] dtype=\(dtype) rows=\(rows) chunk=\(chunkSize)"
                    + " kvMaxAbs=\(kvDelta) hiddenMaxAbs=\(hiddenDelta) logitsMaxAbs=\(logitDelta)"
                    + " argmaxMismatches=\(mismatches)/\(ids.count)")
            }
        }
    }
}
