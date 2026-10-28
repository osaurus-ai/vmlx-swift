// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import CryptoKit
import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

/// MiniMax-M3 aligned-chunk prefix cache (`AlignedPrefillPrefix.swift`, `MiniMaxM3Model.prepare`), end to end through
/// the real Swift generation and cache path: `TokenIterator` with a `CacheCoordinator` built from the serving
/// configuration (the solo path every Osaurus request takes), its end-of-turn store, the L2 disk tier (the only tier
/// that holds M3 state: the paged tier cannot carry the three-lane sparse cache), a restart onto the same disk store,
/// the DSpark lane, and `BatchEngine` (solo `generate` and queued `submit`).
///
/// Tiny synthetic M3 with the released MSA geometry: blocks of 128 keys, top-16, so selection turns sparse past 2,048
/// keys; 16 query heads over 2 KV groups (8 per group, the MSA prefill kernel's shape) so the chunk kernel, the decode
/// kernel and the R8 selection kernels all run (counted); MoE routed (8 experts, top-4) + shared expert on layers 2-3;
/// MXFP8 / affine trunk as an M3.1 JANGHT bundle. The cold reference of every request is the same request through
/// `TokenIterator` with no coordinator (no lookup, no store), itself checked against the model's own forward over the
/// same absolute pieces.
///
/// Exactness is bit-equality of the logits of every forward that can feed the sampler (the last prefill piece and
/// every decode step) plus the greedy tokens. Prefill steps 320 (5 x 64, not a multiple of 128 / 256 / 2048) and 2048;
/// tool outputs 517 / 333 / 1291 / 77 / 2113 (no multiple of 64 / 128 / 2048). Invariants on every restore: every layer
/// and lane holds exactly the reused rows, the prefill starts there and covers the rest of the prompt, and the reuse is
/// a multiple of the step or the request's own store boundary.
///
/// Parity with the vMLX Python runtime: `Fixtures/MiniMaxM3/aligned-prefix/python-counts.json`, written by
/// `scripts/minimax_m3_fixtures/generate_aligned_prefix_fixtures.py` from a clean export of the Python branch commit
/// (ef84c03f): the same prompts give the same per-turn cached / prefilled counts. Counts depend only on lengths and the
/// rule; `VMLX_M3_ALIGNED_PY_WEIGHTS=<dir>` (the generator's `--weights` output) also loads Python's exact weights and
/// compares tokens and first-step logits with Python's.
///
/// Controls that must FAIL (each first asserts it really reused a prefix the rule would not): the previous token-only
/// reuse without aligned chunking (`VMLX_M3_ALIGNED_PREFIX=0`), aligned chunking with a misaligned continuation start,
/// and exact repeats without the cold prefill's cut at the store boundary.
@Suite("MiniMax-M3 aligned-chunk prefix cache", .serialized)
struct MiniMaxM3AlignedPrefixCacheTests {
    typealias DSpark = MiniMaxM3DSparkTests

    static let vocabulary = 1024
    static let gen = [vocabulary - 5, vocabulary - 4, vocabulary - 3]
    static let end = [vocabulary - 6]
    static let maxTokens = 7

    // MARK: - Fixture

    struct Fixture {
        let raw: [String: Any]
        var firstPrompt: [Int] { raw["first_prompt"] as! [Int] }
        var tools: [[Int]] { raw["tools"] as! [[Int]] }
        func loop(_ step: Int) -> [String: Any] { (raw["loops"] as! [String: Any])["\(step)"] as! [String: Any] }
        func memory(_ step: Int) -> [String: Any] {
            (raw["memory_tier"] as! [String: Any])["\(step)"] as! [String: Any]
        }
        var shortAndRestart: [String: Any] { raw["short_and_restart"] as! [String: Any] }

        static func load() throws -> Fixture {
            let url = DSpark.fixtures.appendingPathComponent("aligned-prefix/python-counts.json")
            let raw = try #require(try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any])
            return Fixture(raw: raw)
        }
    }

    static func turns(_ section: [String: Any]) -> [[String: Any]] { section["turns"] as! [[String: Any]] }
    static func int(_ row: [String: Any], _ key: String) -> Int { (row[key] as! NSNumber).intValue }

    // MARK: - Model

    static func configuration() throws -> MiniMaxM3Configuration {
        let text: [String: Any] = [
            "model_type": "minimax_m3", "hidden_size": 512, "num_hidden_layers": 4, "intermediate_size": 256,
            "dense_intermediate_size": 512, "shared_intermediate_size": 256, "num_attention_heads": 16,
            "num_key_value_heads": 2, "head_dim": 64, "rotary_dim": 32, "rope_theta": 10_000.0, "rms_norm_eps": 1e-6,
            "vocab_size": vocabulary, "num_local_experts": 8, "num_experts_per_tok": 4, "routed_scaling_factor": 2.0,
            "norm_topk_prob": true, "swiglu_alpha": 1.702, "swiglu_limit": 7.0, "moe_layer_freq": [0, 0, 1, 1],
            "tie_word_embeddings": false,
            "sparse_attention_config": [
                "sparse_num_index_heads": 2, "sparse_index_dim": 64, "sparse_block_size": 128,
                "sparse_topk_blocks": 16, "sparse_local_block": 1, "sparse_attention_freq": [0, 1, 1, 1],
            ],
        ]
        let json: [String: Any] = [
            "model_type": "minimax_m3_vl", "architectures": ["MiniMaxM3SparseForConditionalGeneration"],
            "text_config": text,
        ]
        return try JSONDecoder.json5().decode(
            MiniMaxM3Configuration.self, from: JSONSerialization.data(withJSONObject: json))
    }

    /// Python's exact weights when `VMLX_M3_ALIGNED_PY_WEIGHTS` names the generator's `--weights` directory.
    static var pythonWeights: URL? {
        ProcessInfo.processInfo.environment["VMLX_M3_ALIGNED_PY_WEIGHTS"].map {
            URL(fileURLWithPath: $0).appendingPathComponent("model.safetensors")
        }
    }

    /// The synthetic M3: Python's unquantized weights (opt-in) or random weights like the DSpark tests', then the M3.1
    /// JANGHT per-module quantization plan (the Python test's `_plan`).
    static func model(seed: UInt64 = 5, python: Bool = false) throws -> MiniMaxM3Model {
        let model = MiniMaxM3Model(try configuration())
        var update: [(String, MLXArray)] = []
        if python, let url = pythonWeights {
            let weights = try loadArrays(url: url)
            let names = Set(model.parameters().flattened().map(\.0))
            #expect(Set(weights.keys) == names, "Python weight names differ from the Swift model's")
            update = weights.map { ($0.key, $0.value) }
        } else {
            var key = MLXRandom.key(seed)
            for (name, value) in model.parameters().flattened() {
                let split = MLXRandom.split(key: key)
                key = split.0
                let r = MLXRandom.normal(value.shape, key: split.1)
                if name.hasSuffix("norm.weight") {
                    update.append((name, (r * Float(0.1)).asType(.bfloat16)))
                } else if name.hasSuffix("e_score_correction_bias") {
                    update.append((name, (r * Float(0.05)).asType(.float32)))
                } else if name.hasSuffix(".mlp.gate.weight") {
                    update.append((name, (r * Float(0.3)).asType(.float32)))
                } else if name.contains("embed_tokens") {
                    update.append((name, r.asType(.bfloat16)))
                } else {
                    update.append((name, (r * Float(1.5 / Double(value.dim(-1)).squareRoot())).asType(.bfloat16)))
                }
            }
        }
        try model.update(parameters: ModuleParameters.unflattened(update), verify: .all)
        quantize(model: model, filter: DSpark.plan)
        eval(model)
        return model
    }

    static func ids(_ n: Int, seed: UInt64) -> [Int] {
        MLXRandom.randInt(0 ..< Int32(vocabulary - 8), [n], key: MLXRandom.key(seed)).asArray(Int32.self)
            .map(Int.init)
    }

    static func boundary(_ prompt: [Int]) -> Int { prompt.count - gen.count - 1 }

    // MARK: - Coordinator (the serving configuration)

    static func coordinator(_ dir: URL, step: Int) -> CacheCoordinator {
        var config = VMLXServerRuntimeSettings().cacheCoordinatorConfig(
            modelKey: "minimax-m3-aligned-test-\(step)", diskCacheDirectory: dir)
        config.enableDiskCache = true
        let coordinator = CacheCoordinator(config: config)
        // What the host derives at load (ModelContainer): the template's generation-prompt suffix.
        coordinator.setGenPromptSuffixTokens(gen)
        return coordinator
    }

    static func temporaryDirectory(_ tag: String) -> URL {
        FileManager.default.temporaryDirectory.appendingPathComponent("minimax-m3-aligned-\(tag)-\(UUID().uuidString)")
    }

    // MARK: - Running requests

    /// Everything the hooks saw, across requests.
    final class Recorder: @unchecked Sendable {
        private let lock = NSLock()
        private var _logits: [MLXArray] = []
        private var _restores: [AlignedPrefillRestoreRecord] = []
        private var _prefills: [AlignedPrefillPrefillRecord] = []
        private var _msa: [MiniMaxM3MSA.Path] = []
        private var _select: [MiniMaxM3MSA.SelectPath] = []

        var logits: [MLXArray] { lock.withLock { _logits } }
        var restores: [AlignedPrefillRestoreRecord] { lock.withLock { _restores } }
        var prefills: [AlignedPrefillPrefillRecord] { lock.withLock { _prefills } }
        var msa: [MiniMaxM3MSA.Path] { lock.withLock { _msa } }
        var select: [MiniMaxM3MSA.SelectPath] { lock.withLock { _select } }

        func install() {
            AlignedPrefill.hooks.logitsObserver = { [self] x in
                let row = x[0, -1].asType(.float32)
                lock.withLock { _logits.append(row) }
            }
            AlignedPrefill.hooks.restoreObserver = { [self] r in lock.withLock { _restores.append(r) } }
            AlignedPrefill.hooks.prefillObserver = { [self] r in lock.withLock { _prefills.append(r) } }
            MiniMaxM3MSA.hooks.pathObserver = { [self] p in lock.withLock { _msa.append(p) } }
            MiniMaxM3MSA.hooks.selectObserver = { [self] p in lock.withLock { _select.append(p) } }
        }

        static func uninstall() {
            AlignedPrefill.hooks.logitsObserver = nil
            AlignedPrefill.hooks.restoreObserver = nil
            AlignedPrefill.hooks.prefillObserver = nil
            AlignedPrefill.hooks.controlReuse = nil
            AlignedPrefill.hooks.controlOmitsBoundaryCut = false
            MiniMaxM3MSA.hooks.pathObserver = nil
            MiniMaxM3MSA.hooks.selectObserver = nil
        }
    }

    /// One request: its tokens, the logits of every forward that fed the sampler, what was restored and prefilled.
    struct Run {
        let prompt: Int
        let tokens: [Int]
        let logits: [MLXArray]
        let restores: [AlignedPrefillRestoreRecord]
        let prefills: [AlignedPrefillPrefillRecord]
        let cache: [KVCache]
        let dspark: DSparkGenerationStats?
        /// Rows the prefill started from (what was reused).
        var cached: Int { prefills.first?.start ?? -1 }
    }

    /// The request a chat template would produce: message boundaries (`cachePrefixTokenCounts`, the last one the
    /// generation-prompt-stripped history) and the stable system prefix.
    static func input(_ prompt: [Int], boundaries: [Int], stable: [Int]) -> LMInput {
        LMInput(
            tokens: MLXArray(prompt.map { Int32($0) }).reshaped(1, prompt.count), tokenIds: prompt,
            cachePrefixTokenCounts: boundaries.filter { $0 > 0 && $0 < prompt.count },
            cacheStablePrefixTokenCounts: stable.filter { $0 > 0 && $0 < prompt.count })
    }

    /// One turn through the solo path: `TokenIterator` with the coordinator (nil = the cold reference: no lookup, no
    /// store), greedy, then the end-of-turn store the generate loop runs (a length stop: no post-answer row).
    static func run(
        _ model: MiniMaxM3Model, _ prompt: [Int], coordinator: CacheCoordinator?, step: Int, recorder: Recorder,
        boundaries: [Int]? = nil, stable: [Int] = [], maxTokens: Int = maxTokens, drafter: URL? = nil
    ) throws -> Run {
        let before = (recorder.logits.count, recorder.restores.count, recorder.prefills.count)
        var parameters = GenerateParameters(maxTokens: maxTokens, temperature: 0)
        parameters.prefillStepSize = step
        if let drafter { parameters.draftStrategy = .dspark(drafterPath: drafter) }
        let request = input(prompt, boundaries: boundaries ?? [prompt.count - gen.count], stable: stable)
        var iterator = try TokenIterator(
            input: request, model: model, parameters: parameters, cacheCoordinator: coordinator)
        var out: [Int] = []
        while let token = iterator.next() { out.append(token) }
        iterator.storeCacheAfterGeneration(generatedTokenIds: out, includeGeneratedBoundary: false)
        // No wait for the disk writer: as in serving, the next request may be served the write-behind payload.
        let logits = Array(recorder.logits[before.0...])
        eval(logits)
        return Run(
            prompt: prompt.count, tokens: out, logits: logits, restores: Array(recorder.restores[before.1...]),
            prefills: Array(recorder.prefills[before.2...]), cache: iterator.cache,
            dspark: iterator.dsparkGenerationStats)
    }

    static func equal(_ a: MLXArray, _ b: MLXArray) -> Bool { a.shape == b.shape && arrayEqual(a, b).item(Bool.self) }

    /// Bit-identical: the same greedy tokens and the same logits at every sampled step.
    static func same(_ a: Run, _ b: Run) -> Bool {
        a.tokens == b.tokens && a.logits.count == b.logits.count && zip(a.logits, b.logits).allSatisfy(equal)
    }

    static func firstLogitGap(_ a: Run, _ b: Run) -> Float {
        guard let x = a.logits.first, let y = b.logits.first else { return .nan }
        return abs(x - y).max().item(Float.self)
    }

    static func sha256(_ x: MLXArray) -> String {
        let bytes = x.asType(.float32).asArray(Float.self).withUnsafeBufferPointer { Data(buffer: $0) }
        return SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined()
    }

    /// The invariants of an aligned request: one prefill, on the plan, from exactly the reused rows to the prompt end;
    /// a restore (when one was kept) left every layer and lane at the reused rows, a chunk boundary or this request's
    /// own store boundary.
    static func checkInvariants(_ run: Run, prompt: [Int], step: Int, context: String) {
        let k = boundary(prompt)
        guard run.prefills.count == 1, let prefill = run.prefills.first else {
            Issue.record("\(context): \(run.prefills.count) prefills")
            return
        }
        #expect(prefill.aligned, "\(context): prefill did not follow the plan")
        #expect(prefill.plan == AlignedPrefillPlan(step: step, boundary: k, promptTokenCount: prompt.count), "\(context)")
        #expect(prefill.end == prompt.count && prefill.start == run.cached, "\(context)")
        #expect(prefill.laneRowsBefore.allSatisfy { $0 == prefill.start }, "\(context): lanes \(prefill.laneRowsBefore)")
        #expect(
            prefill.pieces == AlignedPrefillGeometry.pieces(start: prefill.start, end: prompt.count, step: step, boundary: k),
            "\(context): pieces \(prefill.pieces)")
        let kept = run.restores.filter { $0.reuse > 0 }
        if run.cached > 0 {
            guard kept.count == 1, let restore = kept.first else {
                Issue.record("\(context): reused \(run.cached) rows without one kept restore (\(run.restores.count))")
                return
            }
            #expect(restore.reuse == run.cached, "\(context)")
            #expect(restore.offsets.allSatisfy { $0 == run.cached } && restore.laneRows.allSatisfy { $0 == run.cached },
                "\(context): offsets \(restore.offsets) lanes \(restore.laneRows)")
            #expect(run.cached % step == 0 || run.cached == k, "\(context): reuse \(run.cached) (step \(step), K \(k))")
            if run.cached % step != 0 {
                #expect(restore.exactGeometry && run.cached == k, "\(context): a non-chunk reuse that is not exact")
            }
        } else {
            #expect(kept.isEmpty, "\(context)")
        }
    }

    /// The model's own forward over the same absolute pieces: the logits of the last piece and the cache.
    static func directPrefill(_ model: MiniMaxM3Model, _ prompt: [Int], step: Int) -> (MLXArray, [KVCache]) {
        let cache = model.newCache(parameters: nil)
        var last = MLXArray(0)
        for piece in AlignedPrefillGeometry.pieces(start: 0, end: prompt.count, step: step, boundary: boundary(prompt)) {
            last = model(DSpark.tokens(Array(prompt[piece])), cache: cache)
            eval(cache.flatMap(\.state))
        }
        let row = last[0, -1].asType(.float32)
        eval(row)
        return (row, cache)
    }

    static func withRecorder<R>(_ body: (Recorder) throws -> R) throws -> R {
        try MLXMetalTestLock.withLock {
            try Device.withDefaultDevice(.gpu) {
                let recorder = Recorder()
                recorder.install()
                defer { Recorder.uninstall() }
                return try body(recorder)
            }
        }
    }

    // MARK: - The rule

    @Test("Geometry: absolute pieces, reuse limits, store boundary, the stored tag (Python prefill_geometry values)")
    func geometryRule() throws {
        let P = AlignedPrefillGeometry.pieces
        #expect(P(0, 1013, 320, 1009, false) == [0 ..< 320, 320 ..< 640, 640 ..< 960, 960 ..< 1009, 1009 ..< 1013])
        #expect(P(640, 1013, 320, 1009, false) == [640 ..< 960, 960 ..< 1009, 1009 ..< 1013])
        #expect(P(1009, 1013, 320, 1009, false) == [1009 ..< 1013])
        #expect(P(0, 700, 320, 640, false) == [0 ..< 320, 320 ..< 640, 640 ..< 700])
        #expect(P(0, 1013, 320, 1009, true) == [0 ..< 1009, 1009 ..< 1013])
        let R = AlignedPrefillGeometry.reuseLimit
        #expect(R(1009, 1009, 320, true, false, 1) == 1009)
        #expect(R(1009, 1009, 320, false, false, 1) == 960)
        #expect(R(1000, 1500, 320, true, false, 1) == 960)
        #expect(R(1009, 1009, 320, true, true, 1) == 1009)
        #expect(R(960, 1009, 320, true, true, 1) == 0)
        #expect(R(1009, 1009, 1000, false, false, 64) == 0)
        let prompt = Array(0 ..< 100) + Self.gen
        #expect(
            AlignedPrefillGeometry.storeBoundary(
                promptTokenIds: prompt, cachePrefixTokenCounts: [], generationPromptSuffix: Self.gen) == 99)
        #expect(
            AlignedPrefillGeometry.storeBoundary(
                promptTokenIds: prompt, cachePrefixTokenCounts: [40, 100], generationPromptSuffix: []) == 99)
        #expect(
            AlignedPrefillGeometry.storeBoundary(
                promptTokenIds: prompt, cachePrefixTokenCounts: [], generationPromptSuffix: []) == 102)
        #expect(
            AlignedPrefillGeometry.storeBoundary(
                promptTokenIds: [7], cachePrefixTokenCounts: [], generationPromptSuffix: []) == nil)
        #expect(AlignedPrefillGeometry.isEnabled([:]) && !AlignedPrefillGeometry.isEnabled(["VMLX_M3_ALIGNED_PREFIX": "0"]))
        let tag = AlignedPrefillTag(step: 320, boundary: 1009)
        #expect(AlignedPrefillTag(metaState: tag.metaState) == tag && tag.cleanPrefix == 960)
        #expect(AlignedPrefillTag(metaState: ["aligned-prefill-v0", "step=320", "boundary=1009"]) == nil)

        // The sparse cache carries its tag through metaState, copy, trim and a disk-state restore; a foreign tag is
        // refused, an untagged (pre-aligned) entry restores with none.
        try MLXMetalTestLock.withLock {
            let cache = MiniMaxM3SparseCache()
            let k = MLXRandom.uniform(low: -1, high: 1, [1, 2, 300, 8]).asType(.bfloat16)
            _ = cache.update(keys: k, values: k)
            _ = cache.updateIndexKeys(MLXRandom.uniform(low: -1, high: 1, [1, 1, 300, 8]).asType(.bfloat16))
            cache.alignedPrefillTag = tag
            #expect(Array(cache.metaState.suffix(3)) == tag.metaState)
            #expect((cache.copy() as? MiniMaxM3SparseCache)?.alignedPrefillTag == tag)
            cache.trim(37)
            #expect(cache.alignedPrefillTag == tag && cache.alignedPrefillLaneRows == [263, 263])
            let restored = MiniMaxM3SparseCache()
            #expect(restored.restoreDiskCacheState(cache.state, metadata: cache.metaState, offset: 263))
            #expect(restored.alignedPrefillTag == tag && restored.alignedPrefillLaneRows == [263, 263])
            let untagged = MiniMaxM3SparseCache()
            #expect(untagged.restoreDiskCacheState(cache.state, metadata: [""], offset: 263))
            #expect(untagged.alignedPrefillTag == nil)
            #expect(!MiniMaxM3SparseCache().restoreDiskCacheState(
                cache.state, metadata: ["", "aligned-prefill-v9", "step=1", "boundary=2"], offset: 263))
        }
    }

    // MARK: - (a) + (b) + (c): the agentic tool loop

    @Test(
        "6-turn tool loop: cached == cold bit for bit, reuse = Python's per turn, exact repeat, partial hit mid-chunk, a cached turn's stored state == a direct cold prefill",
        arguments: [320, 2048])
    func agenticToolLoop(step: Int) throws {
        let fixture = try Fixture.load()
        let expected = fixture.loop(step)
        let memoryTier = step == 320 ? fixture.memory(320) : nil
        let dir = Self.temporaryDirectory("loop-\(step)")
        defer { try? FileManager.default.removeItem(at: dir) }
        try Self.withRecorder { recorder in
            let model = try Self.model()
            let coordinator = Self.coordinator(dir, step: step)
            var prompt = fixture.firstPrompt
            // Message boundaries as the processor declares them: system, first user, then each assistant / tool end.
            var bounds = [1100, 1157]
            var prev: [Int] = []
            var table: [(Int, Int, Int)] = []
            var crossedSparse = false
            for (turn, row) in Self.turns(expected).enumerated() {
                let warm = try Self.run(
                    model, prompt, coordinator: coordinator, step: step, recorder: recorder, boundaries: bounds,
                    stable: [1100])
                let cold = try Self.run(
                    model, prompt, coordinator: nil, step: step, recorder: recorder, boundaries: bounds,
                    stable: [1100])
                // A real, non-empty baseline (fail closed).
                try #require(cold.cached == 0 && cold.tokens.count == Self.maxTokens && cold.logits.count == 8)
                if turn == 0 || turn == 5 {
                    let (direct, _) = Self.directPrefill(model, prompt, step: step)
                    #expect(Self.equal(direct, cold.logits[0]), "turn \(turn + 1): cold arm != the model's own pieces")
                }
                Self.checkInvariants(warm, prompt: prompt, step: step, context: "turn \(turn + 1)")
                Self.checkInvariants(cold, prompt: prompt, step: step, context: "turn \(turn + 1) cold")
                #expect(Self.same(warm, cold), "turn \(turn + 1): cached != cold (step \(step))")
                #expect(prompt.count == Self.int(row, "prompt"))
                #expect(warm.cached == Self.int(row, "cached"),
                    "turn \(turn + 1): reused \(warm.cached), Python \(Self.int(row, "cached"))")
                if let memoryTier {
                    #expect(warm.cached == Self.int(Self.turns(memoryTier)[turn], "cached"))
                }
                if turn > 0 {
                    #expect(warm.cached == (Self.boundary(prev) / step) * step)
                }
                if warm.cached < 2048 && 2048 < prompt.count { crossedSparse = true }
                table.append((prompt.count, warm.cached, prompt.count - warm.cached))
                prev = prompt
                if turn < 5 {
                    let assistantEnd = prompt.count + warm.tokens.count + Self.end.count
                    prompt = prompt + warm.tokens + Self.end + fixture.tools[turn] + Self.gen
                    bounds += [assistantEnd, prompt.count - Self.gen.count]
                }
            }
            #expect(crossedSparse, "no continuation crossed the 2,048-key dense -> sparse switch")
            // The paged tier cannot hold M3's three-lane cache: every restore came from the disk tier.
            #expect(coordinator.isPagedIncompatible)
            let details = Set(recorder.restores.map(\.detail))
            #expect(details == ["disk"], "restore tiers \(details)")

            // The state a CACHED turn (turn 6) stored, restored from disk, equals a direct cold prefill row for row.
            _ = coordinator.waitForPendingDiskWrites()
            var parameters = GenerateParameters(maxTokens: Self.maxTokens, temperature: 0)
            parameters.prefillStepSize = step
            let salt = computeCacheSalt(
                for: Self.input(prev, boundaries: bounds, stable: [1100]), parameters: parameters, model: model)
            guard case .hit(let matched, _, let detail, _, _, let arrays?) = coordinator.fetch(
                tokens: prev, mediaSalt: salt, skipExactDiskBoundary: true)
            else {
                Issue.record("INVALID: the cached turn's prompt state is not on disk")
                return
            }
            #expect(detail == .disk && matched == prev.count - 1)
            var stored = model.newCache(parameters: nil)
            try #require(restoreFromDiskArrays(arrays, into: &stored, requirePromptBoundary: true) == matched)
            #expect(AlignedPrefill.tag(of: stored) == AlignedPrefillTag(step: step, boundary: Self.boundary(prev)))
            let (_, reference) = Self.directPrefill(model, prev, step: step)
            let k = Self.boundary(prev)
            for layer in stored { layer.trim(layer.offset - k) }
            for layer in reference { layer.trim(layer.offset - k) }
            let problems = DSpark.cacheProblems(stored, reference)
            #expect(problems.isEmpty, "cached turn's stored state != direct cold prefill: \(problems.prefix(3))")

            // Exact repeat: restores this request's own store boundary (the cold prefill cut there).
            let repeatRun = try Self.run(
                model, prev, coordinator: coordinator, step: step, recorder: recorder, boundaries: bounds,
                stable: [1100])
            let repeatCold = try Self.run(
                model, prev, coordinator: nil, step: step, recorder: recorder, boundaries: bounds, stable: [1100])
            Self.checkInvariants(repeatRun, prompt: prev, step: step, context: "exact repeat")
            #expect(repeatRun.cached == Self.boundary(prev)
                && repeatRun.cached == Self.int(expected["exact_repeat"] as! [String: Any], "cached"))
            #expect(Self.same(repeatRun, repeatCold), "exact repeat != cold")

            // (c) Partial hit: the last tool result replaced from a mid-chunk position on.
            let partial = expected["partial"] as! [String: Any]
            let cut = Self.int(partial, "divergence")
            try #require(cut % step != 0 && cut % 64 != 0)
            let variant = Array(prev[..<cut]) + (partial["tail"] as! [Int]) + Self.gen
            let variantBounds = bounds.filter { $0 < cut } + [variant.count - Self.gen.count]
            let part = try Self.run(
                model, variant, coordinator: coordinator, step: step, recorder: recorder, boundaries: variantBounds,
                stable: [1100])
            let partCold = try Self.run(
                model, variant, coordinator: nil, step: step, recorder: recorder, boundaries: variantBounds,
                stable: [1100])
            Self.checkInvariants(part, prompt: variant, step: step, context: "partial hit")
            #expect(Self.same(part, partCold), "partial hit != cold")
            #expect(part.cached > 0 && part.cached % step == 0 && part.cached < cut)
            // Whole-prompt entries (Swift's disk tier, Python's memory tier) reuse the last chunk boundary of the
            // longest entry that IS a prefix; Python's block tier also matches blocks inside a longer entry.
            let pythonWhole = memoryTier.map { Self.int($0["partial"] as! [String: Any], "cached") }
            if let pythonWhole { #expect(part.cached == pythonWhole) }

            let msa = Dictionary(grouping: recorder.msa, by: { $0 }).mapValues(\.count)
            let select = Dictionary(grouping: recorder.select, by: { $0 }).mapValues(\.count)
            #expect((msa[.decode] ?? 0) > 0 && (select[.nax] ?? 0) + (select[.fma] ?? 0) > 0, "\(msa) \(select)")
            #expect((msa[.kernel] ?? 0) > 0, "the chunk kernel never ran: \(msa)")
            print(
                "[M3 aligned] step \(step) turns (prompt, cached, prefill): \(table); repeat \(repeatRun.cached); "
                    + "partial cut \(cut) cached \(part.cached) (Python block tier \(Self.int(partial, "cached")), "
                    + "whole-entry tier \(pythonWhole.map(String.init) ?? "n/a")); MSA \(msa) select \(select)")
        }
    }

    // MARK: - (d) + (e): short prompt, the 2048 switch, restart onto the disk store

    @Test("Short prompt (< step), its extension, a long prompt's continuation, then a restart onto the same disk store")
    func shortPromptAndRestart() throws {
        let fixture = try Fixture.load()
        let section = fixture.shortAndRestart
        let rows = section["rows"] as! [String: [String: Any]]
        let step = Self.int(section, "step")
        let dir = Self.temporaryDirectory("restart")
        defer { try? FileManager.default.removeItem(at: dir) }
        try Self.withRecorder { recorder in
            let model = try Self.model()
            func check(_ name: String, _ prompt: [Int], _ coordinator: CacheCoordinator) throws -> Run {
                let warm = try Self.run(model, prompt, coordinator: coordinator, step: step, recorder: recorder)
                let cold = try Self.run(model, prompt, coordinator: nil, step: step, recorder: recorder)
                try #require(cold.tokens.count == Self.maxTokens && cold.cached == 0)
                Self.checkInvariants(warm, prompt: prompt, step: step, context: name)
                #expect(Self.same(warm, cold), "\(name): cached != cold")
                let row = try #require(rows[name])
                #expect(prompt.count == Self.int(row, "prompt") && warm.cached == Self.int(row, "cached"),
                    "\(name): prompt \(prompt.count) cached \(warm.cached), Python \(Self.int(row, "cached"))")
                return warm
            }
            var coordinator = Self.coordinator(dir, step: step)
            let short = section["short"] as! [Int]
            let first = try check("short_cold", short, coordinator)
            _ = try check("short_repeat", short, coordinator)
            let longer = short + first.tokens + Self.end + (section["extension_tail"] as! [Int]) + Self.gen
            _ = try check("extension", longer, coordinator)
            let long = section["long"] as! [Int]
            let firstLong = try check("long_first", long, coordinator)
            let next = long + firstLong.tokens + Self.end + (section["next_tail"] as! [Int]) + Self.gen
            let warmRAM = try check("next_warm_ram", next, coordinator)
            #expect(warmRAM.cached > 2048 || next.count < 2048)
            // Restart: a new coordinator on the same disk store (nothing in memory).
            _ = coordinator.waitForPendingDiskWrites()
            coordinator = Self.coordinator(dir, step: step)
            let fromDisk = try check("restart_exact_from_disk", next, coordinator)
            #expect(fromDisk.restores.first?.detail == "disk" && fromDisk.restores.first?.exactGeometry == true)
            #expect(Self.same(fromDisk, warmRAM))
            let third = next + fromDisk.tokens + Self.end + (section["third_tail"] as! [Int]) + Self.gen
            _ = try check("restart_continuation", third, coordinator)
        }
    }

    // MARK: - A step that is not a multiple of the 64-token block

    @Test("Step 1000 (not a block multiple): reuse at 1000-token chunk boundaries, = Python's whole-entry tier")
    func stepNotABlockMultiple() throws {
        let fixture = try Fixture.load()
        let expected = fixture.memory(1000)
        let step = 1000
        let dir = Self.temporaryDirectory("step1000")
        defer { try? FileManager.default.removeItem(at: dir) }
        try Self.withRecorder { recorder in
            let model = try Self.model()
            let coordinator = Self.coordinator(dir, step: step)
            var prompt = fixture.firstPrompt
            var reused: [Int] = []
            for (turn, row) in Self.turns(expected).prefix(4).enumerated() {
                let warm = try Self.run(model, prompt, coordinator: coordinator, step: step, recorder: recorder)
                let cold = try Self.run(model, prompt, coordinator: nil, step: step, recorder: recorder)
                Self.checkInvariants(warm, prompt: prompt, step: step, context: "step 1000 turn \(turn + 1)")
                #expect(Self.same(warm, cold), "step 1000 turn \(turn + 1)")
                #expect(warm.cached == Self.int(row, "cached"))
                reused.append(warm.cached)
                prompt = prompt + warm.tokens + Self.end + fixture.tools[turn] + Self.gen
            }
            #expect(reused.filter { $0 > 0 }.count >= 2 && reused.allSatisfy { $0 % step == 0 }, "\(reused)")
        }
    }

    // MARK: - (f) The DSpark lane

    @Test("DSpark lane: cached == cold (tokens, first logits, cache) and == the plain lane; DSpark verify ran")
    func dsparkLane() throws {
        let fixture = try Fixture.load()
        let step = 320
        let dir = Self.temporaryDirectory("dspark")
        let drafter = Self.temporaryDirectory("dspark-drafter")
        defer {
            try? FileManager.default.removeItem(at: dir)
            try? FileManager.default.removeItem(at: drafter)
        }
        try DSpark.writeReleaseDrafter(drafter)
        try Self.withRecorder { recorder in
            let model = try Self.model()
            let plainModel = try Self.model()
            let coordinator = Self.coordinator(dir, step: step)
            var prompt = fixture.firstPrompt
            var prev: [Int] = []
            var specSteps = 0
            for turn in 0 ..< 4 {
                let warm = try Self.run(
                    model, prompt, coordinator: coordinator, step: step, recorder: recorder, maxTokens: 12,
                    drafter: drafter)
                let cold = try Self.run(
                    model, prompt, coordinator: nil, step: step, recorder: recorder, maxTokens: 12, drafter: drafter)
                let plain = try Self.run(
                    plainModel, prompt, coordinator: nil, step: step, recorder: recorder, maxTokens: 12)
                let stats = try #require(warm.dspark, "turn \(turn + 1): the DSpark lane did not run")
                specSteps += stats.specSteps
                try #require(cold.tokens.count == 12 && plain.tokens.count == 12)
                Self.checkInvariants(warm, prompt: prompt, step: step, context: "dspark turn \(turn + 1)")
                #expect(warm.tokens == cold.tokens && Self.equal(warm.logits[0], cold.logits[0]),
                    "dspark turn \(turn + 1): cached != cold")
                let problems = DSpark.cacheProblems(warm.cache, cold.cache)
                #expect(problems.isEmpty, "dspark turn \(turn + 1): caches differ: \(problems.prefix(3))")
                #expect(plain.tokens == cold.tokens && Self.equal(plain.logits[0], cold.logits[0]),
                    "dspark turn \(turn + 1): DSpark != plain lane")
                if turn > 0 { #expect(warm.cached == (Self.boundary(prev) / step) * step && warm.cached > 0) }
                prev = prompt
                if turn < 3 { prompt = prompt + warm.tokens + Self.end + fixture.tools[turn] + Self.gen }
            }
            #expect(specSteps > 0, "no DSpark verify step ran")
            let repeatRun = try Self.run(
                model, prev, coordinator: coordinator, step: step, recorder: recorder, maxTokens: 12, drafter: drafter)
            let plain = try Self.run(plainModel, prev, coordinator: nil, step: step, recorder: recorder, maxTokens: 12)
            #expect(repeatRun.cached == Self.boundary(prev))
            #expect(repeatRun.tokens == plain.tokens && Self.equal(repeatRun.logits[0], plain.logits[0]))
            print("[M3 aligned] DSpark lane: 4 turns + repeat == cold == plain, verify steps \(specSteps)")
        }
    }

    // MARK: - BatchEngine: solo generate and queued submit

    static func engine(_ model: MiniMaxM3Model, coordinator: CacheCoordinator?) -> BatchEngine {
        let processor = TestInputProcessor(
            tokenizer: TestTokenizer(vocabularySize: vocabulary),
            configuration: ModelConfiguration(id: "minimax-m3-aligned"), messageGenerator: DefaultMessageGenerator())
        let context = ModelContext(
            configuration: processor.configuration, model: model, processor: processor, tokenizer: processor.tokenizer)
        return BatchEngine(context: context, maxBatchSize: 1, cacheCoordinator: coordinator)
    }

    /// One request through `BatchEngine.submit` (the scheduler's slot path); tokens.
    static func submitTurn(_ engine: BatchEngine, _ prompt: [Int], step: Int) async -> [Int] {
        var parameters = GenerateParameters(maxTokens: maxTokens, temperature: 0)
        parameters.prefillStepSize = step
        let (_, stream) = await engine.submit(
            input: input(prompt, boundaries: [prompt.count - gen.count], stable: []), parameters: parameters)
        var tokens: [Int] = []
        for await event in stream {
            if case .token(let t) = event { tokens.append(t) }
        }
        return tokens
    }

    /// One request through `BatchEngine.generate` (the solo fast path Osaurus takes); drains the stream.
    static func generateTurn(_ engine: BatchEngine, _ prompt: [Int], step: Int) async {
        var parameters = GenerateParameters(maxTokens: maxTokens, temperature: 0)
        parameters.prefillStepSize = step
        let stream = await engine.generate(
            input: input(prompt, boundaries: [prompt.count - gen.count], stable: []), parameters: parameters)
        for await _ in stream {}
    }

    @Test("BatchEngine: solo generate and queued submit reuse aligned prefixes; cached == cold, logits every step")
    func batchEnginePaths() async throws {
        let mlxLock = lockSerializedMLXTest()
        defer { mlxLock.unlock() }
        let fixture = try Fixture.load()
        let step = 320
        let dir = Self.temporaryDirectory("engine")
        defer { try? FileManager.default.removeItem(at: dir) }
        let recorder = Recorder()
        recorder.install()
        defer { Recorder.uninstall() }
        let model = try Self.model()
        nonisolated(unsafe) let shared = model
        let warmEngine = Self.engine(shared, coordinator: Self.coordinator(dir, step: step))
        let coldEngine = Self.engine(shared, coordinator: nil)
        var prompt = fixture.firstPrompt
        var lines: [String] = []
        for turn in 0 ..< 4 {
            let solo = turn % 2 == 0
            func arm(_ engine: BatchEngine) async -> (tokens: [Int], logits: [MLXArray], prefill: AlignedPrefillPrefillRecord?) {
                let (l0, p0) = (recorder.logits.count, recorder.prefills.count)
                var tokens: [Int] = []
                if solo {
                    await Self.generateTurn(engine, prompt, step: step)
                } else {
                    tokens = await Self.submitTurn(engine, prompt, step: step)
                }
                let logits = Array(recorder.logits[l0...])
                eval(logits)
                if solo { tokens = logits.prefix(Self.maxTokens).map { argMax($0).item(Int.self) } }
                return (tokens, logits, recorder.prefills.count > p0 ? recorder.prefills[p0] : nil)
            }
            let warm = await arm(warmEngine)
            let cold = await arm(coldEngine)
            try #require(!cold.tokens.isEmpty && !cold.logits.isEmpty && cold.prefill?.start == 0)
            let restored = warm.prefill?.start ?? -1
            #expect(warm.prefill?.aligned == true && cold.prefill?.aligned == true)
            #expect(warm.tokens == cold.tokens && warm.logits.count == cold.logits.count
                && zip(warm.logits, cold.logits).allSatisfy(Self.equal),
                "turn \(turn + 1) (\(solo ? "solo generate" : "submit")): cached != cold")
            if turn > 0 {
                #expect(restored == Self.int(Self.turns(fixture.loop(step))[turn], "cached") && restored > 0,
                    "turn \(turn + 1): reused \(restored)")
            }
            lines.append("\(solo ? "generate" : "submit") \(prompt.count)/\(restored)")
            prompt = prompt + warm.tokens + Self.end + fixture.tools[turn] + Self.gen
        }
        print("[M3 aligned] BatchEngine (prompt/cached): \(lines.joined(separator: ", "))")
    }

    // MARK: - Controls (must fail)

    /// Turns of the tool loop, warm vs cold: (prompt, cached, identical, first-logit gap, same tokens).
    static func controlLoop(
        _ model: MiniMaxM3Model, coordinator: CacheCoordinator, step: Int, recorder: Recorder, turns: Int = 6
    ) throws -> [(Int, Int, Bool, Float, Bool)] {
        let fixture = try Fixture.load()
        var prompt = fixture.firstPrompt
        var rows: [(Int, Int, Bool, Float, Bool)] = []
        for turn in 0 ..< turns {
            let warm = try Self.run(model, prompt, coordinator: coordinator, step: step, recorder: recorder)
            let cold = try Self.run(model, prompt, coordinator: nil, step: step, recorder: recorder)
            try #require(cold.tokens.count == Self.maxTokens && cold.cached == 0)
            rows.append(
                (prompt.count, warm.cached, Self.same(warm, cold), Self.firstLogitGap(warm, cold),
                    warm.tokens == cold.tokens))
            if turn < turns - 1 { prompt = prompt + warm.tokens + Self.end + fixture.tools[turn] + Self.gen }
        }
        return rows
    }

    @Test("Control: VMLX_M3_ALIGNED_PREFIX=0 (token-only reuse, chunks from the restored offset) is NOT bit-identical")
    func controlTokenOnlyReuseDrifts() throws {
        let step = 320
        let dir = Self.temporaryDirectory("control-old")
        defer { try? FileManager.default.removeItem(at: dir) }
        try Self.withRecorder { recorder in
            setenv(AlignedPrefillGeometry.environmentKey, "0", 1)
            defer { unsetenv(AlignedPrefillGeometry.environmentKey) }
            let model = try Self.model()
            try #require(model.alignedPrefillSchema == nil, "INVALID: the lever did not turn aligned prefill off")
            let rows = try Self.controlLoop(
                model, coordinator: Self.coordinator(dir, step: step), step: step, recorder: recorder)
            let noneAligned = recorder.restores.isEmpty && recorder.prefills.allSatisfy { !$0.aligned }
            try #require(noneAligned)
            // It really reused prefixes the rule would not: every later turn, not a chunk boundary.
            let unaligned = rows.dropFirst().filter { $0.1 > 0 && $0.1 % step != 0 }
            #expect(unaligned.count == rows.count - 1, "INVALID: \(rows)")
            #expect(!rows.dropFirst().allSatisfy { $0.2 }, "the control did not fail: \(rows)")
            print("[M3 aligned control] token-only, unaligned (prompt, cached, identical, gap, same tokens): \(rows)")
        }
    }

    @Test("Control: aligned chunking with a misaligned continuation start (reuse not capped) is NOT bit-identical")
    func controlMisalignedContinuationDrifts() throws {
        let step = 320
        let dir = Self.temporaryDirectory("control-misaligned")
        defer { try? FileManager.default.removeItem(at: dir) }
        try Self.withRecorder { recorder in
            // Keep every restored row the entry holds up to this request's own boundary.
            AlignedPrefill.hooks.controlReuse = { min($0.matched, $0.plan.boundary ?? 0) }
            let model = try Self.model()
            let rows = try Self.controlLoop(
                model, coordinator: Self.coordinator(dir, step: step), step: step, recorder: recorder)
            let allAligned = recorder.prefills.allSatisfy { $0.aligned }
            #expect(allAligned)
            let unaligned = rows.dropFirst().filter { $0.1 > 0 && $0.1 % step != 0 }
            #expect(unaligned.count == rows.count - 1, "INVALID: \(rows)")
            #expect(!rows.dropFirst().allSatisfy { $0.2 }, "the control did not fail: \(rows)")
            print("[M3 aligned control] misaligned start (prompt, cached, identical, gap, same tokens): \(rows)")
        }
    }

    @Test("Control: exact repeats without the cold prefill's cut at the store boundary are NOT bit-identical")
    func controlNoBoundaryCutDrifts() throws {
        let step = 320
        let dir = Self.temporaryDirectory("control-nocut")
        defer { try? FileManager.default.removeItem(at: dir) }
        try Self.withRecorder { recorder in
            AlignedPrefill.hooks.controlOmitsBoundaryCut = true
            let model = try Self.model()
            let coordinator = Self.coordinator(dir, step: step)
            var rows: [(Int, Int, Bool, Float)] = []
            for (seed, n) in [(3, 1500), (4, 2711), (5, 4466)] as [(UInt64, Int)] {
                let prompt = Self.ids(n, seed: seed) + Self.gen
                let first = try Self.run(model, prompt, coordinator: coordinator, step: step, recorder: recorder)
                let again = try Self.run(model, prompt, coordinator: coordinator, step: step, recorder: recorder)
                try #require(first.cached == 0 && first.tokens.count == Self.maxTokens)
                // It really continued from the store boundary, which the cold prefill did not cut at.
                #expect(again.cached == Self.boundary(prompt) && again.cached % step != 0, "INVALID: \(again.cached)")
                rows.append((n, again.cached, Self.same(again, first), Self.firstLogitGap(again, first)))
            }
            #expect(!rows.allSatisfy { $0.2 }, "the control did not fail: \(rows)")
            print("[M3 aligned control] no cut at K, exact repeats (n, cached, identical, gap): \(rows)")
        }
    }

    // MARK: - Same weights as Python (opt-in)

    @Test("Python's exact weights: per-turn counts, greedy tokens and first-step logits vs the Python runtime",
          .enabled(if: pythonWeights.map { FileManager.default.fileExists(atPath: $0.path) } ?? false))
    func sameWeightsAsPython() throws {
        let fixture = try Fixture.load()
        let step = 320
        let expected = fixture.loop(step)
        let dir = Self.temporaryDirectory("python-weights")
        defer { try? FileManager.default.removeItem(at: dir) }
        try Self.withRecorder { recorder in
            let model = try Self.model(python: true)
            let coordinator = Self.coordinator(dir, step: step)
            let strictF32 = ProcessInfo.processInfo.environment["MLX_ENABLE_TF32"] == "0"
            var prompt = fixture.firstPrompt
            var lines: [String] = []
            for (turn, row) in Self.turns(expected).enumerated() {
                let warm = try Self.run(model, prompt, coordinator: coordinator, step: step, recorder: recorder)
                let cold = try Self.run(model, prompt, coordinator: nil, step: step, recorder: recorder)
                #expect(Self.same(warm, cold))
                #expect(warm.cached == Self.int(row, "cached"))
                let pyTokens = row["cold_tokens"] as! [Int]
                let pyDigest = row["first_logits_sha256"] as! String
                let digest = Self.sha256(cold.logits[0])
                lines.append("turn \(turn + 1): tokens \(cold.tokens == pyTokens ? "==" : "!=") Python, "
                    + "first logits \(digest == pyDigest ? "==" : "!=") Python")
                // The fixture was written with MLX_ENABLE_TF32=0: under strict f32 the two runtimes run the same
                // pieces through the same kernels (with the -fno-fast-math metallib), so they must agree bit for bit.
                if strictF32 {
                    #expect(cold.tokens == pyTokens && digest == pyDigest, "turn \(turn + 1): Swift != Python")
                }
                // Python's prompts carry Python's tokens: follow them, so every turn is the same prompt as Python's.
                if turn < 5 { prompt = prompt + (row["tokens"] as! [Int]) + Self.end + fixture.tools[turn] + Self.gen }
            }
            print("[M3 aligned] same weights as Python (MLX_ENABLE_TF32=\(strictF32 ? "0" : "default")): "
                + lines.joined(separator: "; "))
        }
    }
}
