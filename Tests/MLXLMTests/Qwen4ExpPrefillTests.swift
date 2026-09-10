import Foundation
import CryptoKit
import MLX
import MLXLMCommon
import MLXNN
import MLXRandom
import Testing

@testable import MLXVLM

@Suite("Flash text prefill window", .serialized)
struct Qwen4ExpPrefillTests {
    private final class Progress: @unchecked Sendable {
        private let lock = NSLock()
        private var values: [Int] = []
        func append(_ value: Int) { lock.withLock { values.append(value) } }
        func read() -> [Int] { lock.withLock { values } }
    }

    private func withFixture(routedBits: [Int] = [], _ body: (Qwen4Exp) throws -> Void) throws {
        // Entire geometry is bounded before construction; no installed model is read.
        let data = Data(
            """
            {
              "model_type":"qwen4_exp","eos_token_id":1,
              "text_config":{
                "model_type":"qwen4_exp_text","dtype":"float32",
                "hidden_size":64,"num_hidden_layers":2,"intermediate_size":64,
                "num_attention_heads":4,"num_key_value_heads":1,"head_dim":16,
                "linear_num_value_heads":4,"linear_num_key_heads":1,
                "linear_key_head_dim":16,"linear_value_head_dim":16,
                "linear_conv_kernel_dim":4,"vocab_size":128,
                "num_experts":8,"num_experts_per_tok":2,
                "moe_intermediate_size":16,"shared_expert_intermediate_size":16,
                "layer_types":["linear_attention","full_attention"],
                "hc_count":4,"hc_lowrank":8,"ple_layer_ids":[1],
                "ple_embed_dim":64,"ple_conv_kernel_size":4,
                "ngram_size":3,"heads_per_ngram":2,"ngram_vocab_size_base":101,
                "make_ngram_vocab_size_divisible_by":128,"seed":1234,
                "split_ngram_parts":4,"indexer_n_heads":2,"indexer_kv_heads":1,
                "indexer_head_dim":8,"indexer_budget":32,"indexer_compress_ratio":4,
                "mrope_section":[1,1,1],"mtp_num_hidden_layers":0
              }
            }
            """.utf8)
        var fixtureData = data
        var routedSpecs: [String: Int] = [:]
        if !routedBits.isEmpty {
            try #require(routedBits.count == 3 && (routedBits == [0, 0, 0] || routedBits.allSatisfy { [2, 3, 4, 6].contains($0) }))
            var root = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
            var text = try #require(root["text_config"] as? [String: Any])
            text["dtype"] = "bfloat16"
            text["moe_intermediate_size"] = 64
            text["shared_expert_intermediate_size"] = 64
            root["text_config"] = text
            var quantization: [String: Any] = ["bits": 8, "group_size": 64]
            for layer in 0..<2 {
                for (projection, bits) in zip(["gate_proj", "up_proj", "down_proj"], routedBits) {
                    if bits == 0 { continue } // BF16 dense control, same geometry.
                    let path = "language_model.layers.\(layer).mlp.switch_mlp.\(projection)"
                    routedSpecs[path] = bits
                    quantization[path] = ["bits": bits, "group_size": 64]
                }
            }
            if !routedSpecs.isEmpty { root["quantization"] = quantization }
            fixtureData = try JSONSerialization.data(withJSONObject: root)
        }
        let config = try JSONDecoder().decode(Qwen4ExpConfiguration.self, from: fixtureData)
        let text = config.base.textConfiguration
        try #require(text.hiddenSize == 64 && text.hiddenLayers == 2)
        try #require(text.vocabularySize == 128 && text.numExperts == 8)
        MLXRandom.seed(0)
        let model = Qwen4Exp(config)
        let count = model.parameters().flattened().reduce(0) { $0 + $1.1.size }
        try #require(count < 2_000_000, "refuse fixture drift before MLX evaluation")
        if !routedBits.isEmpty {
            // Match the retained live parameter-domain contract: ordinary
            // parameters BF16, router weights FP32. This is still a tiny
            // generated fixture, not a shipped-weight numerical reference.
            model.update(parameters: ModuleParameters.unflattened(
                model.parameters().flattened().map { path, value in
                    (path, path.hasSuffix(".mlp.gate.weight") ? value : value.asType(.bfloat16))
                }))
        }
        if !routedSpecs.isEmpty {
            quantize(model: model, filter: { path, _ in
                guard let bits = routedSpecs[path] else { return nil }
                return (groupSize: 64, bits: bits, mode: QuantizationMode.affine)
            })
            let leaves = Dictionary(uniqueKeysWithValues: model.leafModules().flattened())
            for (path, bits) in routedSpecs {
                let projection = try #require(leaves[path] as? any Quantized)
                try #require(projection.bits == bits && projection.groupSize == 64)
            }
            model.update(parameters: ModuleParameters.unflattened(
                model.parameters().flattened().compactMap { path, value in
                    guard path.hasSuffix(".scales") || path.hasSuffix(".biases") else { return nil }
                    return (path, value.asType(.float16))
                }))
        }

        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("flash-prefill-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        var arrays: [String: MLXArray] = [:]
        var map: [String: String] = [:]
        var specs: [String: [String: Int]] = [:]
        for shard in 0 ..< 4 {
            let base = "language_model.layers.0.ple.ngram_embedding.shards.\(shard)"
            arrays[base + ".weight"] = MLXArray(
                (0 ..< (128 * 2)).map { UInt32(truncatingIfNeeded: ($0 + shard) * 0x12345) }
            ).reshaped(128, 2)
            arrays[base + ".scales"] = MLXArray.full([128, 4], values: MLXArray(Float(0.01)))
                .asType(.float16)
            arrays[base + ".biases"] = MLXArray.full([128, 4], values: MLXArray(Float(-0.05)))
                .asType(.float16)
            specs[base + ".weight"] = ["bits": 4, "group_size": 4]
            for suffix in [".weight", ".scales", ".biases"] {
                map[base + suffix] = "model.safetensors"
            }
        }
        try MLX.save(arrays: arrays, url: directory.appendingPathComponent("model.safetensors"))
        try JSONSerialization.data(withJSONObject: ["weight_map": map])
            .write(to: directory.appendingPathComponent("model.safetensors.index.json"))
        try JSONSerialization.data(withJSONObject: ["jang_config": ["bit_map": specs]])
            .write(to: directory.appendingPathComponent("config.json"))
        try model.configure(modelDirectory: directory)
        try body(model)
    }

    @Test("caller window bounds Flash prefill and reports completed chunks")
    func flashPrepareHonorsWindowAndReportsProgress() throws {
        try MLXMetalTestLock.withLock {
            try withFixture { model in
                let cache = model.newCache(parameters: nil)
                let tokens = MLXArray([Int32(2), 3, 4, 5, 6, 7, 8]).reshaped(1, 7)
                let progress = Progress()
                let result = try PrefillProgressReporter.withHandler({ progress.append($0) }) {
                    try model.prepare(
                        LMInput(text: .init(tokens: tokens)),
                        cache: cache, windowSize: 3)
                }
                guard case .logits(let output) = result else {
                    Issue.record("Flash prepare must return last-chunk logits")
                    return
                }
                MLX.eval(output.logits)
                MLX.eval(cache)
                #expect(progress.read() == [3, 6])
                #expect(output.logits.shape == [1, 1, 128])
                #expect(cache.allSatisfy { $0.offset == 7 })
                #expect(cache.first is MambaCache)
                #expect(cache.last is QSAKVCache)
            }
        }
    }

    private func expectEqual(_ actual: MLXArray, _ expected: MLXArray, _ label: String) {
        #expect(actual.shape == expected.shape, "\(label) shape")
        guard actual.shape == expected.shape else { return }
        let a = actual.asType(.float32)
        let b = expected.asType(.float32)
        let error = MLX.max(abs(a - b)).item(Float.self)
        #expect(
            allClose(a, b, rtol: 1e-5, atol: 1e-5).item(Bool.self),
            "\(label) maxAbs=\(error)")
    }

    @Test("chunked suffix matches explicit forwards including restored PLE and QSA")
    func chunkedSuffixMatchesExplicitForwards() throws {
        try MLXMetalTestLock.withLock {
            try withFixture { model in
                for prefixLength in [0, 35] {
                    let prefix = model.newCache(parameters: nil)
                    if prefixLength > 0 {
                        let ids = MLXArray((0 ..< prefixLength).map { Int32(2 + $0 % 100) })
                            .reshaped(1, prefixLength)
                        MLX.eval(model(ids, cache: prefix))
                        MLX.eval(prefix)
                    }
                    // copy() discards derived QSA pools, as a restore does;
                    // raw index keys and PLE token/conv history must suffice.
                    let actualCache = prefix.map { $0.copy() }
                    let referenceCache = prefix.map { $0.copy() }
                    let tokens = MLXArray((0 ..< 17).map { Int32(3 + $0 % 100) }).reshaped(1, 17)
                    let result = try model.prepare(
                        LMInput(text: .init(tokens: tokens)),
                        cache: actualCache, windowSize: 8)
                    guard case .logits(let output) = result else {
                        Issue.record("missing logits")
                        return
                    }
                    for range in [0 ..< 8, 8 ..< 16] {
                        MLX.eval(model(tokens[0..., range], cache: referenceCache))
                        MLX.eval(referenceCache)
                    }
                    let expected = model(tokens[0..., 16...], cache: referenceCache)
                    expectEqual(output.logits, expected, "prefix\(prefixLength) last logits")
                    for (layer, pair) in zip(actualCache, referenceCache).enumerated() {
                        #expect(pair.0.offset == prefixLength + 17)
                        #expect(pair.0.offset == pair.1.offset)
                        #expect(pair.0.state.count == pair.1.state.count)
                        for (slot, arrays) in zip(pair.0.state, pair.1.state).enumerated() {
                            expectEqual(arrays.0, arrays.1, "layer\(layer) slot\(slot)")
                        }
                    }
                    let next = MLXArray([Int32(31)]).reshaped(1, 1)
                    expectEqual(
                        model(next, cache: actualCache),
                        model(next, cache: referenceCache), "next decode")
                }
            }
        }
    }

    @Test("opt-in preprocessing decode matches native multi-token continuation", .enabled(
        if: ProcessInfo.processInfo.environment["VMLX_QWEN4_EXP_COMPILE_GDN_PREPROCESS"] == "1"
            || ProcessInfo.processInfo.environment["VMLX_RUN_GDN_PREPROCESS_CONTROL"] == "1"))
    func compiledPreprocessingContinuationParity() throws {
        try MLXMetalTestLock.withLock {
            try withFixture(routedBits: [2, 3, 4]) { model in
                let prefix = model.newCache(parameters: nil)
                let ids = MLXArray((0 ..< 35).map { Int32(2 + $0 % 100) }).reshaped(1, 35)
                MLX.eval(model(ids, cache: prefix)); MLX.eval(prefix)
                let decoded = prefix.map { $0.copy() }
                let native = prefix.map { $0.copy() }
                let tokens = MLXArray([Int32(41), 42, 43, 44]).reshaped(1, 4)
                let expected = model(tokens, cache: native)
                MLX.eval(expected); MLX.eval(native)
                var pieces: [MLXArray] = []
                for index in 0 ..< 4 {
                    let output = model(tokens[0..., index ..< index + 1], cache: decoded)
                    MLX.eval(output); MLX.eval(decoded)
                    pieces.append(output)
                }
                let actual = concatenated(pieces, axis: 1)
                func digest(_ arrays: [MLXArray]) -> String {
                    var hash = SHA256()
                    for array in arrays {
                        let values = array.asType(.float32).asArray(Float.self)
                        values.withUnsafeBytes { hash.update(bufferPointer: $0) }
                    }
                    return hash.finalize().map { String(format: "%02x", $0) }.joined()
                }
                print("[GDNPreprocessIntegration] enabled=\(Qwen4ExpCompiledGDNInputs.preprocessEnabled)"
                    + " decode_sha=\(digest([actual])) native_sha=\(digest([expected]))"
                    + " decode_cache_sha=\(digest(decoded.flatMap { $0.state }))"
                    + " native_cache_sha=\(digest(native.flatMap { $0.state }))")
                expectEqual(actual, expected, "compiled decode logits")
                for (layer, pair) in zip(decoded, native).enumerated() {
                    #expect(pair.0.offset == 39 && pair.1.offset == 39)
                    #expect(pair.0.state.count == pair.1.state.count)
                    for (slot, values) in zip(pair.0.state, pair.1.state).enumerated() {
                        expectEqual(values.0, values.1, "compiled layer\(layer) slot\(slot)")
                    }
                }
            }
        }
    }

    // MLX's default TF32 path uses reduced precision for float32 matrix ops.
    // Keep the 1e-5 FP32 oracle strict and explicitly run this row with
    // MLX_ENABLE_TF32=0; a default run must report it skipped, not proven.
    @Test(
        "one-shot and chunked prefill agree across QSA budget",
        .enabled(
            if: ProcessInfo.processInfo.environment["MLX_ENABLE_TF32"] == "0",
            "Strict FP32 oracle requires MLX_ENABLE_TF32=0; default TF32 drift is recorded separately"
        ))
    func chunkedMatchesOneShot() throws {
        try MLXMetalTestLock.withLock {
            try withFixture { model in
                let tokens = MLXArray((0 ..< 44).map { Int32(2 + $0 % 100) }).reshaped(1, 44)
                let whole = model.newCache(parameters: nil)
                let chunked = model.newCache(parameters: nil)
                let expected = model(tokens, cache: whole)
                MLX.eval(expected)
                let result = try model.prepare(
                    LMInput(text: .init(tokens: tokens)),
                    cache: chunked, windowSize: 8)
                guard case .logits(let output) = result else {
                    Issue.record("missing logits")
                    return
                }
                expectEqual(output.logits, expected[0..., 40..., 0...], "one-shot last chunk")
                let next = MLXArray([Int32(47)]).reshaped(1, 1)
                expectEqual(
                    model(next, cache: chunked), model(next, cache: whole),
                    "one-shot next decode")
            }
        }
    }

    @Test(
        "live structural boundary capture preserves Flash PLE QSA continuation",
        .enabled(
            if: ProcessInfo.processInfo.environment["MLX_ENABLE_TF32"] == "0",
            "Strict FP32 segmentation oracle requires TF32 off; production policy is unchanged"
        ))
    func structuralBoundaryCaptureParity() throws {
        try MLXMetalTestLock.withLock {
            try withFixture { model in
                let ids = MLXArray((0..<44).map { Int32(2 + $0 % 100) }).reshaped(1, 44)
                let baseline = model.newCache(parameters: nil)
                let captured = model.newCache(parameters: nil)
                let headReference = model.newCache(parameters: nil)
                _ = try model.prepare(LMInput(text: .init(tokens: ids)), cache: baseline, windowSize: 8)
                let head = LMInput(text: .init(tokens: ids[0..., 0..<37]))
                _ = try model.prepare(head, cache: captured, windowSize: 8)
                let owned = captured.map { $0.copy() }
                MLX.eval(owned)
                let file = FileManager.default.temporaryDirectory
                    .appendingPathComponent("flash-ple-boundary-\(UUID().uuidString).safetensors")
                defer { try? FileManager.default.removeItem(at: file) }
                let serialized = TQDiskSerializer.serialize(cache: owned)
                try MLX.save(arrays: serialized, url: file)
                let persisted = try MLX.loadArrays(url: file)
                var diskRestored = model.newCache(parameters: nil)
                #expect(restoreFromDiskArrays(persisted, into: &diskRestored) == 37)
                let ple = try #require(diskRestored.first as? MambaCache)
                #expect(ple.persistentStateSlotCount == 4 && ple.state.count == 4)
                _ = try model.prepare(
                    LMInput(text: .init(tokens: ids[0..., 37...])), cache: diskRestored, windowSize: 8)
                _ = try model.prepare(
                    LMInput(text: .init(tokens: ids[0..., 37...])), cache: captured, windowSize: 8)
                _ = try model.prepare(head, cache: headReference, windowSize: 8)
                for (index, pair) in zip(owned, headReference).enumerated() {
                    #expect(pair.0.offset == 37 && pair.1.offset == 37)
                    #expect(pair.0.state.count == pair.1.state.count)
                    for (slot, state) in zip(pair.0.state, pair.1.state).enumerated() {
                        expectEqual(state.0, state.1, "owned layer\(index) slot\(slot)")
                    }
                }
                let next = MLXArray([Int32(47)]).reshaped(1, 1)
                let expectedNext = model(next, cache: baseline)
                expectEqual(model(next, cache: captured), expectedNext,
                            "structural split next-token logits")
                expectEqual(model(next, cache: diskRestored), expectedNext,
                            "disk-restored PLE QSA next-token logits")
            }
        }
    }

    @Test(
        "staged Flash acceptance preserves PLE GDN QSA state and next logits",
        .enabled(
            if: ProcessInfo.processInfo.environment["MLX_ENABLE_TF32"] == "0",
            "Strict comparison requires TF32 off; no tolerance relaxation for mixed quantization"
        ),
        arguments: [[], [0, 0, 0], [2, 2, 2], [2, 3, 3], [4, 4, 4], [4, 4, 6]])
    func stagedAcceptedPrefixMatchesSequential(routedBits: [Int]) throws {
        try MLXMetalTestLock.withLock {
            try withFixture(routedBits: routedBits) { model in
                let start = Date()
                var checkedRows = 0
                for prefixLength in [0, 35, 71, 127] {
                    let prefix = model.newCache(parameters: nil)
                    if prefixLength > 0 {
                        let ids = MLXArray((0..<prefixLength).map { Int32(2 + $0 % 100) })
                            .reshaped(1, prefixLength)
                        _ = try model.prepare(
                            LMInput(text: .init(tokens: ids)), cache: prefix, windowSize: 8)
                        MLX.eval(prefix)
                    }
                    for width in [2, 3, 4] {
                        let ids = MLXArray((0..<width).map { Int32(47 + $0) }).reshaped(1, width)
                        for accepted in 1...width {
                            let staged = prefix.map { $0.copy() }
                            let reference = prefix.map { $0.copy() }
                            let before = staged.map { $0.copy() }
                            MLX.eval(before)
                            let verified = NativeMTPVerifierStatePolicy.withVerifierMode("input_capture_staged") {
                                model.nativeBackboneMTPVerifyForward(ids, cache: staged)
                            }
                            MLX.eval(verified.logits, verified.hiddenStates)
                            MLX.eval(staged)
                            let qsa = try #require(staged.last as? QSAKVCache)
                            let pooledBeforeTrim = qsa.derivedPooledBlockCount
                            let mamba = try #require(staged.first as? MambaCache)
                            #expect(mamba.offset == prefixLength)
                            #expect(mamba.verifyStagingReady)
                            #expect(mamba[4] != nil && mamba[5] != nil)
                            let prior = try #require(before.first as? MambaCache)
                            for slot in 0..<4 {
                                if let expected = prior[slot] {
                                    let actual = try #require(mamba[slot])
                                    expectEqual(actual, expected, "uncommitted slot\(slot)")
                                } else {
                                    #expect(mamba[slot] == nil)
                                }
                            }
                            if accepted < width {
                                for layer in staged where layer.isTrimmable {
                                    #expect(layer.trim(width - accepted) == width - accepted)
                                }
                            }
                            try #require(model.commitStagedVerifiedBlock(
                                cache: staged, acceptedInputs: accepted, blockLength: width))
                            // Fixture compression ratio is 4. Completed blocks
                            // preceding the accepted boundary must not be rebuilt
                            // merely because the speculative suffix was rejected.
                            #expect(qsa.derivedPooledBlockCount == min(
                                pooledBeforeTrim, (prefixLength + accepted) / 4))
                            #expect(mamba[4] == nil && mamba[5] == nil)
                            for row in 0..<accepted {
                                MLX.eval(model(ids[0..., row..<(row + 1)], cache: reference))
                                MLX.eval(reference)
                            }
                            for (layer, pair) in zip(staged, reference).enumerated() {
                                #expect(pair.0.offset == prefixLength + accepted)
                                #expect(pair.0.offset == pair.1.offset)
                                #expect(pair.0.state.count == pair.1.state.count)
                                for (slot, arrays) in zip(pair.0.state, pair.1.state).enumerated() {
                                    if prefixLength == 35 && width == 2 && accepted == 1 {
                                        let delta = MLX.max(abs(arrays.0.asType(.float32) - arrays.1.asType(.float32))).item(Float.self)
                                        print("STAGED-STATE-DELTA bits=\(routedBits) layer=\(layer) slot=\(slot) maxAbs=\(delta)")
                                    }
                                    expectEqual(arrays.0, arrays.1,
                                        "prefix\(prefixLength) width\(width) accepted\(accepted) layer\(layer) slot\(slot)")
                                }
                            }
                            let next = MLXArray([Int32(73)]).reshaped(1, 1)
                            if prefixLength == 35 && width == 2 && accepted == 1 {
                                let patched = staged.map { $0.copy() }
                                let control = reference.map { $0.copy() }
                                let patchedMamba = try #require(patched.first as? MambaCache)
                                let controlMamba = try #require(control.first as? MambaCache)
                                let recurrent = try #require(controlMamba[1]) * 1
                                MLX.eval(recurrent)
                                patchedMamba[1] = recurrent
                                let a = model(next, cache: patched).asType(.float32)
                                let b = model(next, cache: control).asType(.float32)
                                let delta = MLX.max(abs(a - b)).item(Float.self)
                                print("STAGED-RECURRENT-TRANSPLANT bits=\(routedBits) nextMaxAbs=\(delta); diagnostic only")
                            }
                            // Both copies drop derived QSA pools. Keep the original
                            // comparison below: this is localization, not a bypass.
                            let stagedFreshPools = staged.map { $0.copy() }
                            let referenceFreshPools = reference.map { $0.copy() }
                            MLX.eval(stagedFreshPools)
                            MLX.eval(referenceFreshPools)
                            expectEqual(model(next, cache: staged), model(next, cache: reference),
                                "prefix\(prefixLength) width\(width) accepted\(accepted) next logits")
                            expectEqual(
                                model(next, cache: stagedFreshPools),
                                model(next, cache: referenceFreshPools),
                                "freshPools prefix\(prefixLength) width\(width) accepted\(accepted) next logits")
                            if prefixLength >= 35 {
                                let rebuilt = try #require(stagedFreshPools.last as? QSAKVCache)
                                let retainedPool = try #require(qsa.derivedPooledBlocks)
                                let rebuiltPool = try #require(rebuilt.derivedPooledBlocks)
                                expectEqual(retainedPool, rebuiltPool, "retained vs rebuilt pooled keys")
                                // Exercise sparse selection on those real indexer
                                // pools with fixed, varied query directions. This
                                // supplements the actual model next-logit checks.
                                for direction in [-1, 1] {
                                    let query = MLXArray((0..<16).map {
                                        Float(direction * ($0 - 7)) / 8
                                    }).reshaped(1, 2, 1, 8).asType(retainedPool.dtype)
                                    let mask = try #require(Qwen4ExpQSA.selectedTokenMask(
                                        query: query, pooledKeys: expandedDimensions(retainedPool, axis: 1),
                                        pastLen: qsa.offset - 1, compressRatio: 4,
                                        blockTopK: 8, keyLen: qsa.offset))
                                    let full = try #require(Qwen4ExpQSA.selectedTokenMask(
                                        query: query, pooledKeys: expandedDimensions(rebuiltPool, axis: 1),
                                        pastLen: rebuilt.offset - 1, compressRatio: 4,
                                        blockTopK: 8, keyLen: rebuilt.offset))
                                    #expect(arrayEqual(mask, full).item(Bool.self))
                                }
                            }
                            checkedRows += accepted
                        }
                    }
                }
                print("FLASH-STAGED-FIXTURE routedBits=\(routedBits) checkedAcceptedRows=\(checkedRows) fixtureRowsPerSecond=\(Double(checkedRows) / max(Date().timeIntervalSince(start), 1e-9)); not model decode throughput")
            }
        }
    }

    @Test("cancelled text prefill does no cache work")
    func cancelledPrefillDoesNotAdvanceCache() async throws {
        try await MLXMetalTestLock.withLock {
            try withFixture { model in
                let cache = model.newCache(parameters: nil)
                let progress = Progress()
                let tokens = MLXArray([Int32(2), 3, 4, 5]).reshaped(1, 4)
                withUnsafeCurrentTask { $0?.cancel() }
                do {
                    _ = try PrefillProgressReporter.withHandler({ progress.append($0) }) {
                        try model.prepare(
                            LMInput(text: .init(tokens: tokens)),
                            cache: cache, windowSize: 2)
                    }
                    Issue.record("expected cancellation")
                } catch is CancellationError {
                    #expect(progress.read().isEmpty)
                    #expect(cache.allSatisfy { $0.offset == 0 })
                }
            }
        }
    }

    @Test("cancellation after one completed chunk prevents subsequent chunks")
    func cancellationAtChunkBoundary() async throws {
        try await MLXMetalTestLock.withLock {
            try withFixture { model in
                let cache = model.newCache(parameters: nil)
                let progress = Progress()
                let tokens = MLXArray([Int32(2), 3, 4, 5, 6]).reshaped(1, 5)
                do {
                    _ = try PrefillProgressReporter.withHandler({ value in
                        progress.append(value)
                        withUnsafeCurrentTask { $0?.cancel() }
                    }) {
                        try model.prepare(
                            LMInput(text: .init(tokens: tokens)),
                            cache: cache, windowSize: 2)
                    }
                    Issue.record("expected cancellation at second chunk")
                } catch is CancellationError {
                    #expect(progress.read() == [2])
                    #expect(cache.allSatisfy { $0.offset == 2 })
                }
            }
        }
    }

    @Test("short prompts and disabled windows retain single-shot behavior")
    func shortAndDisabledWindows() throws {
        try MLXMetalTestLock.withLock {
            try withFixture { model in
                for window: Int? in [nil, 0, -1, 4, 8] {
                    let cache = model.newCache(parameters: nil)
                    let progress = Progress()
                    let tokens = MLXArray([Int32(2), 3, 4, 5]).reshaped(1, 4)
                    let result = try PrefillProgressReporter.withHandler({ progress.append($0) }) {
                        try model.prepare(
                            LMInput(text: .init(tokens: tokens)),
                            cache: cache, windowSize: window)
                    }
                    guard case .logits(let output) = result else {
                        Issue.record("missing logits")
                        return
                    }
                    MLX.eval(output.logits)
                    #expect(progress.read().isEmpty)
                    #expect(output.logits.shape == [1, 4, 128])
                    #expect(cache.allSatisfy { $0.offset == 4 })
                }
            }
        }
    }
}
