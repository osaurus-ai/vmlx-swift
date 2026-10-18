import Foundation
import MLX
import MLXNN
import MLXRandom
import XCTest
@testable import MLXLMCommon
@testable import MLXVLM

/// Checks confirmed MTP head history against an independently reconstructed
/// real Qwen4Exp head cache through greedy, sampled and staged commits.
/// Fresh process: VMLX_NATIVE_MTP_AR_SAFETY=0 VMLX_MTP_VERIFY_PREFETCH=0
/// and VMLX_MTP_ALIGNED_HEAD_CACHE unset/1. Diagnostic controls only.
final class Qwen4ExpMTPHeadChainTests: XCTestCase {
    // These rows reconstruct the head cache from an empty history; prompt priming has its own test
    // (NativeMTPPromptPrimingTests).
    private var primingWasEnabled = true
    override func setUp() {
        super.setUp()
        primingWasEnabled = NativeMTPPromptPrimer.enabled
        NativeMTPPromptPrimer.enabled = false
    }
    override func tearDown() {
        NativeMTPPromptPrimer.enabled = primingWasEnabled
        super.tearDown()
    }

    func testSampledChunkCommittedHeadMatchesOracleAcrossCycles() throws {
        _ = try exercise(depth: 3, seed: 9041, vocabulary: 128,
                         sequential: false, sampled: true, cycles: 2)
    }

    func testSampledChunkRejectedPartialAndFullCommitsMatchOracle() throws {
        for depth in [2, 3] {
            var observed: Set<String> = []
            // Fixed independent grid; all rows run, with no target-ID forcing.
            for vocabulary in [2, 128] {
                for seed in 9041...9056 {
                    let accepted = try exercise(depth: depth, seed: seed,
                        vocabulary: vocabulary, sequential: false, sampled: true)
                    observed.insert(accepted == 0 ? "reject" : accepted == depth ? "full" : "partial")
                }
            }
            print("MTP-SAMPLED-CHUNK-CATEGORIES depth=\(depth) seeds=9041...9056 vocabularies=[2,128] observed=\(observed.sorted())")
            XCTAssertEqual(observed, Set(["reject", "partial", "full"]))
        }
    }

    func testDepthOneCommittedHeadMatchesOracle() throws { _ = try exercise(depth: 1) }
    func testDepthThreeInitialCommittedHeadMatchesOracle() throws { _ = try exercise(depth: 3) }

    func testSequentialGDNFirstRejectedCommitMatchesOracle() throws {
        let accepted = try exercise(depth: 3, sequential: true)
        XCTAssertLessThan(accepted, 3, "Seed9041 must actually reject; no target outcome is forced")
    }
    func testBoundedDepthTwoThreeAcceptanceCategories() throws { try categories(sequential: false) }
    func testBoundedSequentialDepthTwoThreeAcceptanceCategories() throws { try categories(sequential: true) }

    func testSampledSequentialDepthTwoThreeCommittedHeadMatchesOracle() throws {
        for depth in [2, 3] {
            var observed: Set<String> = []
            for seed in 9041...9048 {
                let accepted = try exercise(depth: depth, seed: seed, vocabulary: 2,
                                            sequential: true, sampled: true)
                observed.insert(accepted == 0 ? "reject" : accepted == depth ? "full" : "partial")
            }
            print("MTP-SAMPLED-CATEGORIES depth=\(depth) randomSeeds=9041...9048 observed=\(observed.sorted())")
        }
    }

    func testTwoCycleConfirmedHeadContinuity() throws {
        for depth in [1, 3] {
            for mode in 0..<3 {
                _ = try exercise(depth: depth, seed: 9041, vocabulary: 2,
                    sequential: mode != 0, sampled: mode == 2,
                    prompt: [0, 1, 0, 1], cycles: 2)
            }
        }
    }

    /// One additional actual staged hybrid row; the original 89 rows stay unchanged.
    /// This facade adds capability only for this row, leaving sequential fixtures sequential.
    func testStagedHybridDepthThreeTwoCycleConfirmedHeadContinuity() throws {
        let env = ProcessInfo.processInfo.environment
        try XCTSkipIf(env["VMLX_NATIVE_MTP_AR_SAFETY"] != "0"
            || env["VMLX_MTP_VERIFY_PREFETCH"] != "0",
            "Requires isolated diagnostic governor/prefetch settings")
        try XCTSkipIf(["0", "false", "no", "off"].contains(
            env["VMLX_MTP_ALIGNED_HEAD_CACHE"]?.lowercased() ?? ""))
        try XCTSkipIf(env["VMLX_MTP_COMPILED_VERIFY"] == "1",
            "This row isolates eager default staged dispatch")
        for key in ["VMLX_NATIVE_MTP_HYBRID_VERIFY", "VMLINUX_NATIVE_MTP_HYBRID_VERIFY"] {
            try XCTSkipIf(env[key] != nil, "Default staged dispatch requires no verifier-mode override")
        }
        try MLXMetalTestLock.withLock {
            let depth = 3
            let inputIDs: [Int32] = [0, 1, 0, 1]
            MLXRandom.seed(9041)
            let json = Self.config
                .replacingOccurrences(of: "\"vocab_size\":128", with: "\"vocab_size\":2")
                .replacingOccurrences(of: "[\"full_attention\",\"full_attention\"]",
                                      with: "[\"linear_attention\",\"full_attention\"]")
            let config = try JSONDecoder().decode(Qwen4ExpConfiguration.self, from: Data(json.utf8))
            let real = Qwen4Exp(config)
            // Hybrid topology is independent of the recorder's sequential capture mode.
            // The staged block oracle must capture the actual verifier block instead.
            let recorder = ConfirmedHeadRecorder(real, depth: depth,
                sequential: false, sampled: false, independentBlock: true)
            let staged = StagedConfirmedHeadRecorder(recorder)
            var parameters = GenerateParameters(maxTokens: 24, temperature: 0)
            parameters.draftStrategy = .nativeMTP(depth: depth)
            parameters.nativeMTPDepthPolicy = .fixed
            let start = ProcessInfo.processInfo.systemUptime
            var iterator = try NativeMTPTokenIterator(
                input: LMInput(tokens: MLXArray(inputIDs)),
                model: staged, parameters: parameters, depth: depth)
            XCTAssertTrue(iterator.cache.contains { $0 is MambaCache })
            XCTAssertEqual(recorder.calls, depth)
            var emitted: [Int] = []
            var priorAccepted = 0
            var expectedOffset = 1
            for cycle in 1...2 {
                while iterator.verifyCalls < cycle, emitted.count < 16, let token = iterator.next() {
                    emitted.append(token)
                }
                let totalAccepted = iterator.acceptedByDepth.reduce(0) { $0 + $1.key * $1.value }
                let accepted = totalAccepted - priorAccepted
                XCTAssertEqual(iterator.verifyCalls, cycle)
                XCTAssertEqual(iterator.chunkVerifierCount, cycle)
                XCTAssertEqual(iterator.stagedVerifierCommitCount, cycle)
                XCTAssertEqual(iterator.sequentialVerifierCount, 0)
                XCTAssertEqual(iterator.prefixCommitCount, cycle)
                XCTAssertEqual(recorder.verifyWidths, Array(repeating: depth + 1, count: cycle))
                XCTAssertEqual(staged.ordinaryCommitCalls, 0)
                XCTAssertEqual(staged.stagedCommitAcceptedInputs.count, cycle)
                XCTAssertEqual(staged.stagedCommitAcceptedInputs.last, accepted + 1)
                XCTAssertEqual(staged.stagedCommitBlockLengths, Array(repeating: depth + 1, count: cycle))
                XCTAssertEqual(staged.stagedCommitResults, Array(repeating: true, count: cycle))
                XCTAssertEqual(recorder.comparisonCount, cycle)
                XCTAssertEqual(recorder.calls, depth * (cycle + 1))
                XCTAssertEqual(recorder.firstPreOffsets, [expectedOffset])
                XCTAssertEqual(recorder.oraclePreOffsets, [expectedOffset])
                XCTAssertEqual(recorder.committedWidth, accepted + 1)
                XCTAssertEqual(recorder.firstPostOffsets, [expectedOffset + accepted + 1])
                XCTAssertEqual(recorder.firstPostOffsets, recorder.oraclePostOffsets)
                XCTAssertEqual(recorder.seedLogits, recorder.oracleSeedLogits)
                XCTAssertFalse(recorder.actualLogits.isEmpty)
                XCTAssertTrue(recorder.actualLogits.allSatisfy { $0.isFinite })
                XCTAssertTrue(recorder.expectedLogits.allSatisfy { $0.isFinite })
                XCTAssertEqual(recorder.actualLogits, recorder.expectedLogits)
                XCTAssertEqual(recorder.actualState, recorder.expectedState)
                XCTAssertTrue(recorder.blockPrefixMatches, "Head must consume the actual verified target prefix")
                XCTAssertEqual(recorder.blockInputCount, depth + 1)
                let error = zip(recorder.actualLogits, recorder.expectedLogits)
                    .map { abs($0 - $1) }.max() ?? .infinity
                let aState = recorder.actualState.flatMap { $0 }
                let eState = recorder.expectedState.flatMap { $0 }
                let cacheError: Float = aState.count == eState.count
                    ? (zip(aState, eState).map { abs($0 - $1) }.max() ?? 0) : .infinity
                let elapsed = ProcessInfo.processInfo.systemUptime - start
                let category = accepted == 0 ? "reject" : accepted == depth ? "full" : "partial"
                print("MTP-STAGED-HEAD-CHAIN cycle=\(cycle) plannedCycles=2 prompt=\(inputIDs) seed=9041 vocabulary=2 depth=3 accepted=\(accepted) category=\(category) stagedCommits=\(iterator.stagedVerifierCommitCount) nativeAcceptedInputs=\(staged.stagedCommitAcceptedInputs) nativeBlockLengths=\(staged.stagedCommitBlockLengths) verifierInputs=\(recorder.verifiedInputs) before=\(recorder.firstPreOffsets) oracle=\(recorder.oraclePreOffsets) maxLogitError=\(error) cacheError=\(cacheError) actualStateCounts=\(recorder.actualState.map { $0.count }) oracleStateCounts=\(recorder.expectedState.map { $0.count }) emitted=\(emitted) fixtureTokS=\(Double(emitted.count)/max(elapsed,1e-9)) modelSpeedProof=false")
                priorAccepted = totalAccepted
                expectedOffset += accepted + 1
            }
        }
    }

    private func categories(sequential: Bool) throws {
        for depth in [2, 3] {
            var observed: Set<String> = []
            // Original fixed seed subset retained in full. Report its own
            // coverage before the independent, preregistered prompt extension.
            for seed in 9041...9048 {
                let accepted = try exercise(depth: depth, seed: seed, vocabulary: 2, sequential: sequential)
                observed.insert(accepted == 0 ? "reject" : accepted == depth ? "full" : "partial")
            }
            let required = Set(["reject", "partial", "full"])
            print("MTP-ORIGINAL-SUBSET-COVERAGE sequential=\(sequential) depth=\(depth) prompt=[1,1,1,1] seeds=9041...9048 observed=\(observed.sorted()) originalGatePass=\(observed == required)")
            if sequential && depth == 3 {
                // Independent set fixed before execution; run ALL 32 rows,
                // even after coverage is met. No seed search or early stop.
                print("MTP-PROMPT-EXTENSION-REGISTER depth=3 sequential=true seeds=9041...9048 prompts=\(Self.additionalBinaryPrompts)")
                for prompt in Self.additionalBinaryPrompts {
                    for seed in 9041...9048 {
                        let accepted = try exercise(depth: depth, seed: seed, vocabulary: 2,
                            sequential: true, prompt: prompt)
                        observed.insert(accepted == 0 ? "reject" : accepted == depth ? "full" : "partial")
                    }
                }
            }
            print("MTP-AGGREGATE-COVERAGE sequential=\(sequential) depth=\(depth) observed=\(observed.sorted())")
            XCTAssertEqual(observed, required)
        }
    }

    // Four independent binary sequences; original [1,1,1,1] remains above.
    private static let additionalBinaryPrompts: [[Int32]] = [
        [0, 0, 0, 0], [0, 1, 0, 1], [1, 0, 1, 0], [0, 0, 1, 1]
    ]

    private func exercise(depth: Int, seed: Int = 9041, vocabulary: Int = 128,
                          sequential: Bool = false, sampled: Bool = false,
                          prompt: [Int32]? = nil, cycles: Int = 1) throws -> Int {
        let env = ProcessInfo.processInfo.environment
        try XCTSkipIf(env["VMLX_NATIVE_MTP_AR_SAFETY"] != "0"
            || env["VMLX_MTP_VERIFY_PREFETCH"] != "0",
            "Requires isolated diagnostic governor/prefetch settings")
        try XCTSkipIf(["0", "false", "no", "off"].contains(
            env["VMLX_MTP_ALIGNED_HEAD_CACHE"]?.lowercased() ?? ""))
        return try MLXMetalTestLock.withLock {
            MLXRandom.seed(UInt64(seed))
            var json = Self.config.replacingOccurrences(of: "\"vocab_size\":128", with: "\"vocab_size\":\(vocabulary)")
            if sequential {
                json = json.replacingOccurrences(of: "[\"full_attention\",\"full_attention\"]", with: "[\"linear_attention\",\"full_attention\"]")
            }
            let config = try JSONDecoder().decode(Qwen4ExpConfiguration.self, from: Data(json.utf8))
            let real = Qwen4Exp(config)
            let recorder = ConfirmedHeadRecorder(real, depth: depth, sequential: sequential, sampled: sampled, independentBlock: cycles > 1)
            var parameters = GenerateParameters(maxTokens: cycles > 1 ? 24 : 16, temperature: sampled ? 1 : 0)
            if sampled { parameters.randomSeed = UInt64(seed) }
            parameters.draftStrategy = .nativeMTP(depth: depth)
            parameters.nativeMTPDepthPolicy = .fixed
            let inputIDs = prompt ?? [3, 7, 11, 5].map { Int32($0 % vocabulary) }
            XCTAssertFalse(inputIDs.isEmpty)
            XCTAssertTrue(inputIDs.allSatisfy { $0 >= 0 && Int($0) < vocabulary })
            let start = ProcessInfo.processInfo.systemUptime
            var iterator = try NativeMTPTokenIterator(
                input: LMInput(tokens: MLXArray(inputIDs)),
                model: recorder, parameters: parameters, depth: depth)
            XCTAssertEqual(recorder.calls, depth)
            var emitted: [Int] = []
            var priorAccepted = 0
            var expectedOffset = 1
            for cycle in 1...cycles {
                while iterator.verifyCalls < cycle, emitted.count < 16, let token = iterator.next() {
                    emitted.append(token)
                }
                let totalAccepted = iterator.acceptedByDepth.reduce(0) { $0 + $1.key * $1.value }
                let accepted = totalAccepted - priorAccepted
                XCTAssertEqual(iterator.verifyCalls, cycle)
                XCTAssertEqual(iterator.sequentialVerifierCount, sequential ? cycle : 0)
                XCTAssertEqual(iterator.prefixCommitCount, cycle)
                XCTAssertEqual(recorder.verifyWidths, sequential ? [] : Array(repeating: depth + 1, count: cycle))
                if sequential {
                    XCTAssertEqual(recorder.confirmedTokens.count, accepted + 1)
                    XCTAssertEqual(recorder.confirmedHidden.count, accepted + 1)
                    XCTAssertEqual(recorder.confirmedInputs.count, accepted + 1)
                }
                XCTAssertEqual(recorder.comparisonCount, cycle)
                XCTAssertEqual(recorder.firstPreOffsets, [expectedOffset], "Head history differs from confirmed prefix at this commit")
                XCTAssertEqual(recorder.oraclePreOffsets, [expectedOffset])
                XCTAssertEqual(recorder.committedWidth, accepted + 1)
                XCTAssertEqual(recorder.firstPostOffsets, recorder.oraclePostOffsets)
                // Identical operations and native inputs should agree. This strict
                // comparison is not relaxed if initialization exposes another defect.
                XCTAssertEqual(recorder.seedLogits, recorder.oracleSeedLogits)
                XCTAssertFalse(recorder.actualLogits.isEmpty)
                XCTAssertTrue(recorder.actualLogits.allSatisfy { $0.isFinite })
                XCTAssertTrue(recorder.expectedLogits.allSatisfy { $0.isFinite })
                XCTAssertEqual(recorder.actualLogits, recorder.expectedLogits)
                XCTAssertEqual(recorder.actualState, recorder.expectedState)
                let error = zip(recorder.actualLogits, recorder.expectedLogits)
                    .map { abs($0 - $1) }.max() ?? .infinity
                let aState = recorder.actualState.flatMap { $0 }
                let eState = recorder.expectedState.flatMap { $0 }
                let cacheError: Float = aState.count == eState.count
                    ? (zip(aState, eState).map { abs($0 - $1) }.max() ?? 0) : .infinity
                let elapsed = ProcessInfo.processInfo.systemUptime - start
                print("MTP-COMMIT-CHAIN cycle=\(cycle) plannedCycles=\(cycles) prompt=\(inputIDs) seed=\(seed) vocabulary=\(vocabulary) sequential=\(sequential) sampled=\(sampled) verifierInputs=\(sequential ? recorder.confirmedInputs : recorder.verifiedInputs) depth=\(depth) accepted=\(accepted) before=\(recorder.firstPreOffsets) oracle=\(recorder.oraclePreOffsets) maxLogitError=\(error) cacheError=\(cacheError) actualStateCounts=\(recorder.actualState.map { $0.count }) oracleStateCounts=\(recorder.expectedState.map { $0.count }) emitted=\(emitted) fixtureTokS=\(Double(emitted.count)/max(elapsed,1e-9)) modelSpeedProof=false")
                if cycles > 1 && !sequential {
                    XCTAssertTrue(recorder.blockPrefixMatches, "Head commit must consume verified backbone prefix")
                    XCTAssertEqual(recorder.blockInputCount, depth + 1)
                }
                priorAccepted = totalAccepted
                expectedOffset += accepted + 1
            }
            return priorAccepted
        }
    }

    // Existing tiny configuration shape, with all-full-attention/no-PLE to
    // isolate normal trim/commit from recurrent repair and staged dispatch.
    private static let config = #"""
    {"model_type":"qwen4_exp","text_config":{
      "model_type":"qwen4_exp_text","dtype":"float32","mamba_ssm_dtype":"float32",
      "mtp_num_hidden_layers":1,"hidden_size":64,"num_hidden_layers":2,
      "intermediate_size":64,"num_attention_heads":4,"num_key_value_heads":1,"head_dim":16,
      "linear_num_value_heads":4,"linear_num_key_heads":1,"linear_key_head_dim":16,
      "linear_value_head_dim":16,"linear_conv_kernel_dim":4,"vocab_size":128,
      "num_experts":8,"num_experts_per_tok":2,"moe_intermediate_size":16,
      "shared_expert_intermediate_size":16,"layer_types":["full_attention","full_attention"],
      "hc_count":4,"hc_lowrank":8,"ple_layer_ids":[],"ple_embed_dim":64,
      "ple_conv_kernel_size":4,"ngram_size":3,"heads_per_ngram":2,"ngram_vocab_size_base":101,
      "make_ngram_vocab_size_divisible_by":128,"seed":1234,"split_ngram_parts":4,
      "indexer_n_heads":2,"indexer_kv_heads":1,"indexer_head_dim":8,
      "indexer_budget":32,"indexer_compress_ratio":4}}
    """#
}

private final class ConfirmedHeadRecorder: Module, NativeMTPModel {
    let real: Qwen4Exp
    let depth: Int
    let sequential: Bool
    let sampled: Bool
    let independentBlock: Bool
    var capturedCycle = -1
    var verifiedHidden: MLXArray?
    var verifiedInputs: [Int32] = []
    var blockPrefixMatches = false
    var blockInputCount = 0
    var confirmedInputs: [Int32] = []
    var confirmedHidden: [MLXArray] = []
    var confirmedTokens: [Int32] = []
    let oracle: [KVCache]
    var calls = 0
    var comparisonCount = 0
    var verifyWidths: [Int] = []
    var firstPreOffsets: [Int] = [], oraclePreOffsets: [Int] = []
    var firstPostOffsets: [Int] = [], oraclePostOffsets: [Int] = []
    var committedWidth = 0
    var seedLogits: [Float] = [], oracleSeedLogits: [Float] = []
    var actualLogits: [Float] = [], expectedLogits: [Float] = []
    var actualState: [[Float]] = [], expectedState: [[Float]] = []
    init(_ real: Qwen4Exp, depth: Int, sequential: Bool, sampled: Bool, independentBlock: Bool) {
        self.independentBlock = independentBlock
        self.sequential = sequential
        self.sampled = sampled
        self.real = real; self.depth = depth; self.oracle = real.makeNativeMTPCache()
        super.init()
    }
    var nativeMTPAvailable: Bool { real.nativeMTPAvailable }
    func newCache(parameters: GenerateParameters?) -> [KVCache] { real.newCache(parameters: parameters) }
    func makeNativeMTPCache() -> [KVCache] { real.makeNativeMTPCache() }
    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        try real.prepare(input, cache: cache, windowSize: windowSize)
    }
    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        real.nativeBackboneForward(inputs, cache: cache).logits
    }
    func nativeBackboneForward(_ inputs: MLXArray, cache: [KVCache]?) -> NativeMTPForwardResult {
        let result = real.nativeBackboneForward(inputs, cache: cache)
        if sequential && calls > 0 && calls % depth == 0 {
            let cycle = calls / depth
            if capturedCycle != cycle {
                confirmedHidden.removeAll()
                confirmedInputs.removeAll()
                confirmedTokens.removeAll()
                capturedCycle = cycle
            }
            // Capture actual target hidden states. Greedy uses the independent
            // target argmax; sampled uses accepted next inputs plus the real
            // correction/bonus supplied at the next head call.
            confirmedHidden.append(result.hiddenStates)
            // Later verifier inputs exist only after actual acceptance.
            confirmedInputs.append(contentsOf: inputs.asType(.int32).asArray(Int32.self))
            if !sampled {
                let id = argMax(result.logits[0..., -1, 0...], axis: -1).item(Int.self)
                confirmedTokens.append(Int32(id))
            }
        }
        return result
    }
    func nativeBackboneMTPVerifyForward(_ inputs: MLXArray, cache: [KVCache]?) -> NativeMTPForwardResult {
        verifyWidths.append(inputs.dim(1))
        let result = real.nativeBackboneMTPVerifyForward(inputs, cache: cache)
        if independentBlock {
            verifiedHidden = result.hiddenStates
            verifiedInputs = inputs.asType(.int32).asArray(Int32.self)
        }
        return result
    }
    func nativeMTPForward(hiddenStates: MLXArray, nextTokenIds: MLXArray, cache: [KVCache]?) -> NativeMTPForwardResult {
        calls += 1
        let compare = calls > depth && (calls - 1) % depth == 0
        if compare {
            if sequential && sampled {
                // Each retained hidden_i pairs with the NEXT confirmed input.
                // The last correction/bonus is the actual token forwarded by
                // the iterator into this next head call, not an oracle sample.
                let finalID = nextTokenIds[0, -1].item(Int32.self)
                confirmedTokens = Array(confirmedInputs.dropFirst()) + [finalID]
            }
            comparisonCount += 1
            firstPreOffsets = cache?.map { $0.offset } ?? []
            oraclePreOffsets = oracle.map { $0.offset }
            committedWidth = nextTokenIds.dim(1)
        }
        let output = real.nativeMTPForward(hiddenStates: hiddenStates, nextTokenIds: nextTokenIds, cache: cache)
        if calls == 1 || compare {
            // Oracle sees the first confirmed bridge and then the actual
            // committed backbone pairs. It NEVER consumes recursive head rows.
            var oracleHidden = compare && sequential
                ? concatenated(confirmedHidden, axis: 1) : hiddenStates
            var oracleTokens = compare && sequential
                ? MLXArray(confirmedTokens).reshaped(1, confirmedTokens.count) : nextTokenIds
            if compare && !sequential && independentBlock {
                // Width is independently checked against acceptedByDepth by
                // the test. Reconstruct every pair from the actual verified
                // block; final correction/bonus is the real next-head token.
                guard let verifiedHidden else {
                    XCTFail("Missing actual verifier block")
                    return output
                }
                let width = nextTokenIds.dim(1)
                oracleHidden = verifiedHidden[0..., ..<width, 0...]
                let ids = Array(verifiedInputs.dropFirst().prefix(width - 1))
                    + [nextTokenIds[0, -1].item(Int32.self)]
                oracleTokens = MLXArray(ids).reshaped(1, ids.count)
                blockInputCount = verifiedInputs.count
                blockPrefixMatches = oracleHidden.asType(.float32).asArray(Float.self)
                    == hiddenStates.asType(.float32).asArray(Float.self)
                    && ids == nextTokenIds.asType(.int32).asArray(Int32.self)
            }
            let reference = real.nativeMTPForward(hiddenStates: oracleHidden, nextTokenIds: oracleTokens, cache: oracle)
            MLX.eval(output.logits, reference.logits)
            if calls == 1 {
                seedLogits = output.logits.asType(.float32).asArray(Float.self)
                oracleSeedLogits = reference.logits.asType(.float32).asArray(Float.self)
            } else {
                actualLogits = (sequential ? output.logits[0..., -1, 0...] : output.logits).asType(.float32).asArray(Float.self)
                expectedLogits = (sequential ? reference.logits[0..., -1, 0...] : reference.logits).asType(.float32).asArray(Float.self)
                firstPostOffsets = cache?.map { $0.offset } ?? []
                oraclePostOffsets = oracle.map { $0.offset }
                actualState = (cache ?? []).flatMap { $0.state }.map { $0.asType(.float32).asArray(Float.self) }
                expectedState = oracle.flatMap { $0.state }.map { $0.asType(.float32).asArray(Float.self) }
            }
        }
        return output
    }
}

/// Adds the real rollback capability only to the separately registered staged row.
/// Every model forward and head oracle remains in the original recorder.
private final class StagedConfirmedHeadRecorder: Module, NativeMTPModel,
    DFlash2StagedVerifyRollbackModel {
    let recorder: ConfirmedHeadRecorder
    var ordinaryCommitCalls = 0
    var stagedCommitAcceptedInputs: [Int] = []
    var stagedCommitBlockLengths: [Int] = []
    var stagedCommitResults: [Bool] = []

    init(_ recorder: ConfirmedHeadRecorder) {
        self.recorder = recorder
        super.init()
    }
    var nativeMTPAvailable: Bool { recorder.nativeMTPAvailable }
    func newCache(parameters: GenerateParameters?) -> [KVCache] { recorder.newCache(parameters: parameters) }
    func makeNativeMTPCache() -> [KVCache] { recorder.makeNativeMTPCache() }
    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        try recorder.prepare(input, cache: cache, windowSize: windowSize)
    }
    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        recorder(inputs, cache: cache)
    }
    func nativeBackboneForward(_ inputs: MLXArray, cache: [KVCache]?) -> NativeMTPForwardResult {
        recorder.nativeBackboneForward(inputs, cache: cache)
    }
    func nativeBackboneMTPVerifyForward(_ inputs: MLXArray, cache: [KVCache]?) -> NativeMTPForwardResult {
        recorder.nativeBackboneMTPVerifyForward(inputs, cache: cache)
    }
    func nativeMTPForward(hiddenStates: MLXArray, nextTokenIds: MLXArray, cache: [KVCache]?) -> NativeMTPForwardResult {
        recorder.nativeMTPForward(hiddenStates: hiddenStates, nextTokenIds: nextTokenIds, cache: cache)
    }
    func commitVerifiedBlock(cache: [KVCache], acceptedInputs: Int) -> Bool {
        ordinaryCommitCalls += 1
        return recorder.real.commitVerifiedBlock(cache: cache, acceptedInputs: acceptedInputs)
    }
    func commitStagedVerifiedBlock(cache: [KVCache], acceptedInputs: Int, blockLength: Int) -> Bool {
        let result = recorder.real.commitStagedVerifiedBlock(
            cache: cache, acceptedInputs: acceptedInputs, blockLength: blockLength)
        stagedCommitAcceptedInputs.append(acceptedInputs)
        stagedCommitBlockLengths.append(blockLength)
        stagedCommitResults.append(result)
        return result
    }
}
