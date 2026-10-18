import Foundation
import MLX
import MLXNN
import MLXRandom
import XCTest
@testable import MLXLMCommon
@testable import MLXVLM

/// Native-MTP prompt priming on a real (toy) Qwen4Exp: the head cache holds one row per prompt position (plus the
/// first draft), its rows equal the teacher-forced head over the prompt, and greedy output does not change.
final class NativeMTPPromptPrimingTests: XCTestCase {
    private func model(seed: UInt64) throws -> Qwen4Exp {
        MLXRandom.seed(seed)
        let config = try JSONDecoder().decode(Qwen4ExpConfiguration.self, from: Data(Self.config.utf8))
        return Qwen4Exp(config)
    }

    private func run(_ model: Qwen4Exp, prompt: [Int32], primed: Bool, tokens: Int)
        throws -> (emitted: [Int], primedRows: Int, headOffset: Int, head: [KVCache])
    {
        let previous = NativeMTPPromptPrimer.enabled
        NativeMTPPromptPrimer.enabled = primed
        defer { NativeMTPPromptPrimer.enabled = previous }
        var parameters = GenerateParameters(maxTokens: tokens, temperature: 0)
        parameters.draftStrategy = .nativeMTP(depth: 1)
        parameters.nativeMTPDepthPolicy = .fixed
        var iterator = try NativeMTPTokenIterator(
            input: LMInput(tokens: MLXArray(prompt)), model: model, parameters: parameters, depth: 1)
        let primedRows = iterator.promptPrimedRows
        let headOffset = iterator.mtpCache.first?.offset ?? -1
        let head = iterator.mtpCache
        var emitted: [Int] = []
        while let token = iterator.next() { emitted.append(token) }
        return (emitted, primedRows, headOffset, head)
    }

    func testPrimedHeadHoldsThePromptAndOutputIsUnchanged() throws {
        try MLXMetalTestLock.withLock {
            let real = try model(seed: 7101)
            // 9 tokens: not a multiple of any block / chunk size.
            let prompt: [Int32] = [3, 7, 11, 5, 19, 2, 23, 13, 29]
            let off = try run(real, prompt: prompt, primed: false, tokens: 12)
            let on = try run(real, prompt: prompt, primed: true, tokens: 12)
            XCTAssertEqual(off.primedRows, 0)
            XCTAssertEqual(off.headOffset, 1, "unprimed: only the first draft row")
            XCTAssertEqual(on.primedRows, prompt.count, "n-1 prompt pairs plus (h_{n-1}, first token)")
            XCTAssertEqual(on.headOffset, prompt.count + 1, "primed rows plus the first draft row")
            XCTAssertFalse(on.emitted.isEmpty)
            XCTAssertEqual(on.emitted, off.emitted, "the head only proposes: priming must not change output")
        }
    }

    func testPrimedRowsMatchTheTeacherForcedHead() throws {
        try MLXMetalTestLock.withLock {
            let real = try model(seed: 7102)
            let prompt: [Int32] = [4, 9, 1, 16, 25, 36, 49]
            let on = try run(real, prompt: prompt, primed: true, tokens: 2)
            let first = Int32(on.emitted[0])
            // Teacher-forced oracle in the primer's call shape: the prefill window's (h_i, x_{i+1}) pairs in one head
            // call, then (h_{n-1}, first token) on its own (1-row head calls take the decode-shaped paths).
            let ids = prompt + [first]
            let n = prompt.count
            let trunk = real.nativeBackboneForward(
                MLXArray(prompt).reshaped(1, -1), cache: real.newCache(parameters: GenerateParameters()))
            let oracle = real.makeNativeMTPCache()
            _ = real.nativeMTPForward(
                hiddenStates: trunk.hiddenStates[0..., 0 ..< (n - 1)],
                nextTokenIds: MLXArray(Array(ids[1 ..< n])).reshaped(1, -1), cache: oracle)
            _ = real.nativeMTPForward(
                hiddenStates: trunk.hiddenStates[0..., (n - 1)...],
                nextTokenIds: MLXArray([first]).reshaped(1, 1), cache: oracle)
            MLX.eval(oracle)
            let primedKeys = on.head[0].state[0][0..., 0..., 0 ..< n, 0...]
            let oracleKeys = oracle[0].state[0][0..., 0..., 0 ..< n, 0...]
            XCTAssertEqual(primedKeys.shape, oracleKeys.shape)
            XCTAssertEqual(abs(primedKeys - oracleKeys).max().item(Float.self), 0,
                           "primed head rows must be the teacher-forced head's rows")
        }
    }

    func testRequiredToolSelectionRestoresQwenHybridPrefixAndMatchesCold() throws {
        try MLXMetalTestLock.withLock {
            let real = try model(seed: 7103)
            let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
            defer { try? FileManager.default.removeItem(at: root) }
            let suffix = [120, 121]
            func coordinator() -> CacheCoordinator {
                let result = CacheCoordinator(config: .init(
                    usePagedCache: false, enableDiskCache: true, diskCacheDir: root,
                    modelKey: "qwen-required-tool-priming"))
                result.setHybrid(true, requiresRecurrentSSMCompanion: true, requiresSeparateRecurrentPayload: false)
                result.setGenPromptSuffixTokens(suffix)
                return result
            }
            func input(_ body: [Int], required: Bool) -> LMInput {
                let ids = body + suffix
                return LMInput(tokens: MLXArray(ids.map(Int32.init)).reshaped(1, ids.count), tokenIds: ids,
                    cachePrefixTokenCounts: [3, body.count], cacheStablePrefixTokenCounts: [3],
                    cacheRestorePolicy: required ? .freshRequiredToolSelection : .standard)
            }
            var parameters = GenerateParameters(maxTokens: 12, temperature: 0, prefillStepSize: 3)
            parameters.draftStrategy = .nativeMTP(depth: 1)
            parameters.nativeMTPDepthPolicy = .fixed
            let body = [3, 7, 11, 5, 19, 2, 23, 13, 29]
            let writer = coordinator()
            var initial = try NativeMTPTokenIterator(
                input: input(body, required: false), model: real, parameters: parameters,
                depth: 1, cacheCoordinator: writer)
            initial.storeCacheAfterGeneration(generatedTokenIds: [], includeGeneratedBoundary: false)

            let next = input(body + [17, 31, 43], required: true)
            var warm = try NativeMTPTokenIterator(input: next, model: real, parameters: parameters,
                depth: 1, cacheCoordinator: coordinator())
            // Full prompt = 14; restored boundary = 9. Priming only five suffix
            // positions proves that the forced-tool path actually reused disk.
            XCTAssertEqual(warm.promptPrimedRows, 5)
            var cold = try NativeMTPTokenIterator(input: next, model: real, parameters: parameters, depth: 1)
            XCTAssertEqual(cold.promptPrimedRows, 14)
            var warmTokens: [Int] = []; var coldTokens: [Int] = []
            while let token = warm.next() { warmTokens.append(token) }
            while let token = cold.next() { coldTokens.append(token) }
            XCTAssertEqual(warmTokens.count, 12)
            XCTAssertEqual(warmTokens, coldTokens)
            XCTAssertFalse(real.statesBeforeFirstMediaTokenAreMediaIndependent,
                           "Required-tool reuse must not enable cross-media prefix reuse")
        }
    }

    private static let config = #"""
    {"model_type":"qwen4_exp","text_config":{
      "model_type":"qwen4_exp_text","dtype":"float32","mamba_ssm_dtype":"float32",
      "mtp_num_hidden_layers":1,"hidden_size":64,"num_hidden_layers":2,
      "intermediate_size":64,"num_attention_heads":4,"num_key_value_heads":1,"head_dim":16,
      "linear_num_value_heads":4,"linear_num_key_heads":1,"linear_key_head_dim":16,
      "linear_value_head_dim":16,"linear_conv_kernel_dim":4,"vocab_size":128,
      "num_experts":8,"num_experts_per_tok":2,"moe_intermediate_size":16,
      "shared_expert_intermediate_size":16,"layer_types":["linear_attention","full_attention"],
      "hc_count":4,"hc_lowrank":8,"ple_layer_ids":[],"ple_embed_dim":64,
      "ple_conv_kernel_size":4,"ngram_size":3,"heads_per_ngram":2,"ngram_vocab_size_base":101,
      "make_ngram_vocab_size_divisible_by":128,"seed":1234,"split_ngram_parts":4,
      "indexer_n_heads":2,"indexer_kv_heads":1,"indexer_head_dim":8,
      "indexer_budget":32,"indexer_compress_ratio":4}}
    """#
}
