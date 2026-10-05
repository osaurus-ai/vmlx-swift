import Foundation
import MLX
import MLXNN
import MLXRandom
import XCTest
@testable import MLXLMCommon
@testable import MLXVLM

/// Actual native iterator regression for Qwen4Exp lazy verifier PLE state.
/// The tiny model and disk table use natural full/rejected draft outcomes;
/// no target IDs, logits or acceptance are forced.
final class Qwen4ExpPLELazyCommitTests: XCTestCase {
    func testActualIteratorPLECanonicalStateAfterFirstCycle() throws {
        let env = ProcessInfo.processInfo.environment
        try XCTSkipIf(env["VMLX_NATIVE_MTP_AR_SAFETY"] != "0" || env["VMLX_MTP_VERIFY_PREFETCH"] != "0",
                      "Run isolated: VMLX_NATIVE_MTP_AR_SAFETY=0 VMLX_MTP_VERIFY_PREFETCH=0")
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwen4-ple-lazy-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try MLXMetalTestLock.withLock {
            let table = directory.appendingPathComponent("tiny-table", isDirectory: true)
            try Self.writeTable(table)
            var categories: [String: Set<String>] = [:]
            // Fixed finite grid: run all eight rows, never stop when categories appear.
            for mode in ["lazy_repair"] {
                for seed in 9041...9048 {
                    MLXRandom.seed(UInt64(seed))
                    let config = try JSONDecoder().decode(Qwen4ExpConfiguration.self, from: Data(Self.config.utf8))
                    let real = Qwen4Exp(config)
                    try real.configure(modelDirectory: table)
                    let recorder = PLELazyCommitRecorder(real)
                    var parameters = GenerateParameters(maxTokens: 16, temperature: 0)
                    parameters.draftStrategy = .nativeMTP(depth: 1, verifierMode: mode)
                    parameters.nativeMTPDepthPolicy = .fixed
                    let start = ProcessInfo.processInfo.systemUptime
                    var iterator = try NativeMTPTokenIterator(
                        input: LMInput(tokens: MLXArray([Int32(0), 1, 0, 1])),
                        model: recorder, parameters: parameters, depth: 1)
                    var emitted: [Int] = []
                    while iterator.verifyCalls < 1, emitted.count < 8, let token = iterator.next() {
                        emitted.append(token)
                    }
                    let accepted = iterator.acceptedByDepth.reduce(0) { $0 + $1.key * $1.value }
                    let category = accepted == 1 ? "full" : "reject"
                    categories[mode, default: []].insert(category)
                    XCTAssertEqual(iterator.verifyCalls, 1)
                    XCTAssertEqual(iterator.chunkVerifierCount, 1)
                    XCTAssertEqual(iterator.sequentialVerifierCount, 0)
                    XCTAssertEqual(recorder.inputs.count, 2)
                    let before = try XCTUnwrap(recorder.before)
                    let oracle = before.map { $0.copy() }
                    let prefix = Array(recorder.inputs.prefix(accepted + 1))
                    let target = real.nativeBackboneForward(MLXArray(prefix).reshaped(1, prefix.count), cache: oracle)
                    MLX.eval(target.logits, target.hiddenStates)
                    let actualRows = Self.snapshot(iterator.cache)
                    let expectedRows = Self.snapshot(oracle)
                    let canonicalEqual = actualRows == expectedRows
                    // Probe the same target-selected next token on copies; never mutate iterator state.
                    let next = argMax(target.logits[0..., -1, 0...], axis: -1).item(Int32.self)
                    let actualNext = real.nativeBackboneForward(MLXArray([next]).reshaped(1, 1), cache: iterator.cache.map { $0.copy() })
                    let oracleNext = real.nativeBackboneForward(MLXArray([next]).reshaped(1, 1), cache: oracle.map { $0.copy() })
                    let a = actualNext.logits.asType(.float32).asArray(Float.self)
                    let b = oracleNext.logits.asType(.float32).asArray(Float.self)
                    let error = zip(a, b).map { abs($0 - $1) }.max() ?? 0
                    let elapsed = ProcessInfo.processInfo.systemUptime - start
                    print("PLE-LAZY-COMMIT mode=\(mode) seed=\(seed) category=\(category) canonical=\(canonicalEqual) nextExact=\(a == b) nextMaxAbs=\(error) fixtureTokS=\(Double(emitted.count)/max(elapsed,1e-9))")
                    XCTAssertEqual(iterator.cache.map { $0.offset }, oracle.map { $0.offset })
                    XCTAssertTrue(canonicalEqual, "\(mode)/\(seed)/\(category): committed cache differs from accepted ordinary prefix")
                    XCTAssertEqual(a, b, "\(mode)/\(seed)/\(category): next logits differ")
                }
            }
            for mode in ["lazy_repair"] {
                XCTAssertEqual(categories[mode], Set(["full", "reject"]), "Missing natural acceptance category; do not claim it tested")
            }
        }
    }

    private static func snapshot(_ cache: [KVCache]) -> [[[Float]]] {
        cache.map { $0.state.map { $0.asType(.float32).asArray(Float.self) } }
    }
    private static func writeTable(_ directory: URL) throws {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let base = "language_model.layers.0.ple.ngram_embedding.shards.0"
        let names = [base + ".weight", base + ".scales", base + ".biases"]
        let rows = 2048, dimensions = 16
        let wBytes = rows * dimensions / 8 * 4, mBytes = rows * 2
        let header: [String: Any] = [
            names[0]: ["dtype": "U32", "shape": [rows, 2], "data_offsets": [0, wBytes]],
            names[1]: ["dtype": "F16", "shape": [rows, 1], "data_offsets": [wBytes, wBytes + mBytes]],
            names[2]: ["dtype": "F16", "shape": [rows, 1], "data_offsets": [wBytes + mBytes, wBytes + 2*mBytes]]]
        let h = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
        var data = Data(), n = UInt64(h.count).littleEndian
        withUnsafeBytes(of: &n) { data.append(contentsOf: $0) }; data.append(h)
        for row in 0..<rows {
            for word in 0..<2 {
                var packed: UInt32 = 0
                for col in 0..<8 { packed |= UInt32((row * 7 + word * 8 + col * 3) % 16) << (4*col) }
                var little = packed.littleEndian
                withUnsafeBytes(of: &little) { data.append(contentsOf: $0) }
            }
        }
        for row in 0..<rows {
            var bits = Float16(0.02 + Float(row % 97) * 0.0003).bitPattern.littleEndian
            withUnsafeBytes(of: &bits) { data.append(contentsOf: $0) }
        }
        for row in 0..<rows {
            var bits = Float16(-0.2 + Float(row % 43) * 0.001).bitPattern.littleEndian
            withUnsafeBytes(of: &bits) { data.append(contentsOf: $0) }
        }
        try data.write(to: directory.appendingPathComponent("model.safetensors"))
        try JSONSerialization.data(withJSONObject: ["weight_map": Dictionary(uniqueKeysWithValues: names.map { ($0, "model.safetensors") })])
            .write(to: directory.appendingPathComponent("model.safetensors.index.json"))
        try JSONSerialization.data(withJSONObject: ["jang_config": ["bit_map": ["language_model.layers.*.ple.ngram_embedding.shards.0.weight": ["bits": 4, "group_size": 16]]]])
            .write(to: directory.appendingPathComponent("config.json"))
    }
    private static let config = #"""
    {"model_type":"qwen4_exp","text_config":{
      "model_type":"qwen4_exp_text","dtype":"float32","mamba_ssm_dtype":"float32",
      "mtp_num_hidden_layers":1,"hidden_size":64,"num_hidden_layers":2,
      "intermediate_size":64,"num_attention_heads":4,"num_key_value_heads":1,"head_dim":16,
      "linear_num_value_heads":4,"linear_num_key_heads":1,"linear_key_head_dim":16,
      "linear_value_head_dim":16,"linear_conv_kernel_dim":4,"vocab_size":2,
      "num_experts":8,"num_experts_per_tok":2,"moe_intermediate_size":16,
      "shared_expert_intermediate_size":16,"layer_types":["linear_attention","full_attention"],
      "hc_count":4,"hc_lowrank":8,"ple_layer_ids":[1],"ple_embed_dim":64,
      "ple_conv_kernel_size":4,"ngram_size":3,"heads_per_ngram":2,"ngram_vocab_size_base":101,
      "make_ngram_vocab_size_divisible_by":128,"seed":1234,"split_ngram_parts":1,
      "indexer_n_heads":2,"indexer_kv_heads":1,"indexer_head_dim":8,
      "indexer_budget":32,"indexer_compress_ratio":4}}
    """#
}

private final class PLELazyCommitRecorder: Module, NativeMTPModel, DFlash2StagedVerifyRollbackModel {
    let real: Qwen4Exp
    var before: [KVCache]?
    var inputs: [Int32] = []
    init(_ real: Qwen4Exp) { self.real = real; super.init() }
    var nativeMTPAvailable: Bool { real.nativeMTPAvailable }
    func newCache(parameters: GenerateParameters?) -> [KVCache] { real.newCache(parameters: parameters) }
    func makeNativeMTPCache() -> [KVCache] { real.makeNativeMTPCache() }
    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult { try real.prepare(input, cache: cache, windowSize: windowSize) }
    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray { real.nativeBackboneForward(inputs, cache: cache).logits }
    func nativeBackboneForward(_ inputs: MLXArray, cache: [KVCache]?) -> NativeMTPForwardResult { real.nativeBackboneForward(inputs, cache: cache) }
    func nativeBackboneMTPVerifyForward(_ inputs: MLXArray, cache: [KVCache]?) -> NativeMTPForwardResult {
        self.inputs = inputs.asType(.int32).asArray(Int32.self)
        before = cache?.map { $0.copy() }
        return real.nativeBackboneMTPVerifyForward(inputs, cache: cache)
    }
    func nativeMTPForward(hiddenStates: MLXArray, nextTokenIds: MLXArray, cache: [KVCache]?) -> NativeMTPForwardResult {
        real.nativeMTPForward(hiddenStates: hiddenStates, nextTokenIds: nextTokenIds, cache: cache)
    }
    func commitVerifiedBlock(cache: [KVCache], acceptedInputs: Int) -> Bool { real.commitVerifiedBlock(cache: cache, acceptedInputs: acceptedInputs) }
    func commitStagedVerifiedBlock(cache: [KVCache], acceptedInputs: Int, blockLength: Int) -> Bool {
        real.commitStagedVerifiedBlock(cache: cache, acceptedInputs: acceptedInputs, blockLength: blockLength)
    }
}
