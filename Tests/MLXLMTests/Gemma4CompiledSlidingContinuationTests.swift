import Foundation
import MLX
import MLXLLM
import MLXVLM
import XCTest

@testable import MLXLMCommon

/// Real model forwards, including the mixed sliding/full and shared-KV owners.
/// These tiny random fixtures prove cache/compile parity, not prose quality.
final class Gemma4CompiledSlidingContinuationTests: XCTestCase {
    private static let textJSON = """
        {"model_type":"gemma4_text","hidden_size":64,"intermediate_size":128,
         "num_hidden_layers":4,"num_attention_heads":2,"num_key_value_heads":1,
         "num_global_key_value_heads":1,"head_dim":32,"global_head_dim":32,
         "vocab_size":64,"sliding_window":8,"num_kv_shared_layers":2,
         "attention_k_eq_v":true,"enable_moe_block":true,"num_experts":4,
         "top_k_experts":2,"moe_intermediate_size":64,
         "layer_types":["sliding_attention","full_attention","sliding_attention","full_attention"]}
        """

    private static func continuation(vlm: Bool) throws {
        guard HardwareInfo.isCompiledDecodeSupported else { throw XCTSkip("Compiled decode unsupported") }
        let decoder = JSONDecoder()
        let model: any LanguageModel
        if vlm {
            let json = "{\"model_type\":\"gemma4\",\"text_config\":\(textJSON)}"
            model = Gemma4(try decoder.decode(Gemma4Configuration.self, from: Data(json.utf8)))
        } else {
            model = Gemma4TextModel(try decoder.decode(Gemma4TextConfiguration.self, from: Data(textJSON.utf8)))
        }
        eval(model)
        func prefill() throws -> [any KVCache] {
            let caches = model.newCache(parameters: nil)
            XCTAssertEqual(caches.count, 2, "Shared layers must reuse the two KV owners")
            XCTAssertTrue(caches[0] is RotatingKVCache)
            XCTAssertTrue(caches[1] is KVCacheSimple)
            // Same final three-token chunk that leaves window + 2 temporary rows.
            for tokens in [Array(Int32(1) ... Int32(8)), Array(Int32(9) ... Int32(11))] {
                let result = try model.prepare(LMInput(tokens: MLXArray(tokens, [1, tokens.count])), cache: caches, windowSize: nil)
                switch result {
                case .logits(let output): eval(output.logits)
                case .tokens(let remaining):
                    let output = model(remaining[text: .newAxis], cache: caches, state: nil)
                    eval(output.logits)
                }
                eval(caches)
            }
            XCTAssertEqual(caches[0].state[0].dim(2), 10)
            return caches
        }
        let reference = try prefill()
        let seed = try prefill()
        let sourceMetadata = seed.map(\.metaState)
        let promoted: [any KVCache] = seed.map { layer in
            if let ring = layer as? RotatingKVCache { return CompilableRotatingKVCache(from: ring) }
            return CompilableKVCache(from: layer, maxLength: 64)
        }
        eval(promoted)
        // Fail the baseline safely at the owning invariant before an invalid C array cascades.
        XCTAssertEqual(promoted[0].innerState()[0].dim(2), 8)
        guard promoted[0].innerState()[0].dim(2) == 8 else { return }
        let forward: @Sendable ([MLXArray]) -> [MLXArray] = compile(inputs: promoted, outputs: promoted) { args in
            CompiledDecodeTrace.withActive {
                [model(LMInput.Text(tokens: args[0])[text: .newAxis], cache: promoted, state: nil).logits]
            }
        }
        for step in 0 ..< 20 {
            let tokens = MLXArray([Int32(12 + step)])
            let expected = model(LMInput.Text(tokens: tokens)[text: .newAxis], cache: reference, state: nil).logits
            let actual = forward([tokens])[0]
            eval(expected, actual)
            XCTAssertEqual(actual.shape, expected.shape)
            XCTAssertTrue(MLX.all(isFinite(actual)).item(Bool.self))
            let error = abs(actual.asType(.float32) - expected.asType(.float32)).max().item(Float.self)
            let scale = expected.abs().max().item(Float.self)
            XCTAssertLessThanOrEqual(error, 1e-4 + 1e-4 * scale, "vlm=\(vlm), step=\(step)")
            let ring = try XCTUnwrap(promoted[0] as? CompilableRotatingKVCache)
            XCTAssertEqual(ring.offsetArray.item(Int.self), 12 + step)
            XCTAssertEqual(reference[0].offset, 12 + step)
        }
        XCTAssertEqual(seed.map(\.metaState), sourceMetadata)
    }

    func testTextMixedSharedKVCompiledContinuationAfterPrefillTail() throws {
        try MLXMetalTestLock.withLock { try Self.continuation(vlm: false) }
    }

    func testVLMMixedSharedKVCompiledContinuationAfterPrefillTail() throws {
        try MLXMetalTestLock.withLock { try Self.continuation(vlm: true) }
    }
}
