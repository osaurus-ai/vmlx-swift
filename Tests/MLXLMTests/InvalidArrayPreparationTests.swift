import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLMCommon
@testable import MLXVLM

/// Model-layer repros for the Sentry EXC_BREAKPOINT groups whose faulting
/// frame sits inside a model forward during preparation. In every one the
/// array being subscripted is the empty result of an MLX operation that had
/// already failed (ndim 0 under a recovering error handler); the model then
/// trapped the process in `MLXArray` indexing before `TokenIterator` could
/// run its `error.check()`. These tests feed exactly such an array into the
/// production modules and require the forward to return (invalid) instead of
/// trapping, and the request-level `TokenIterator` to throw the originating
/// `MLXError`.
final class InvalidArrayPreparationTests: XCTestCase {

    /// Deliberate shape error -> the empty array a failed op returns.
    private func invalidArray() -> MLXArray {
        matmul(MLXArray.zeros([2, 3]), MLXArray.zeros([4, 2]))
    }

    private func assertOriginatingError(
        _ error: Error, file: StaticString = #filePath, line: UInt = #line
    ) {
        guard case .caught(let message)? = error as? MLXError else {
            XCTFail("expected MLXError.caught, got \(error)", file: file, line: line)
            return
        }
        XCTAssertTrue(
            message.lowercased().contains("matmul"),
            "first error must be the originating matmul failure, got: \(message)",
            file: file, line: line)
    }

    // MARK: APPLE-MACOS-36B / 1N7 — Qwen3VLVision.Attention `parts[0]` on an empty split

    func testQwen3VLVisionAttentionWithInvalidHiddenStatesReturnsInvalid() throws {
        try MLXMetalTestLock.withLock {
            let attention = Qwen3VLVision.Attention(dim: 16, numHeads: 2)
            // 4 patches, one image: cu_seqlens [0, 4]; rotary freqs [seq, headDim / 2].
            let cuSeqlens = MLXArray([Int32(0), 4])
            let freqs = MLXArray.zeros([4, 4])

            // Control: a valid input still produces [seq, dim].
            let valid = attention(MLXArray.zeros([4, 16]), cuSeqlens: cuSeqlens, rotaryPosEmb: freqs)
            eval(valid)
            XCTAssertEqual(valid.shape, [4, 16])

            XCTAssertThrowsError(
                try withError { error in
                    // What `patchEmbed(pixelValues) + posEmbeds` hands the first block
                    // after a failed op upstream (grid/pixel mismatch, bad patch dim).
                    let poisoned = invalidArray()
                    let out = attention(poisoned, cuSeqlens: cuSeqlens, rotaryPosEmb: freqs)
                    XCTAssertFalse(out.isValid)
                    try error.check()
                }
            ) { assertOriginatingError($0) }

            // An empty cu_seqlens (no grid) must not trap the mask loop either.
            XCTAssertThrowsError(
                try withError { error in
                    let out = attention(
                        invalidArray(), cuSeqlens: MLXArray(0, dtype: .int32), rotaryPosEmb: freqs)
                    XCTAssertFalse(out.isValid)
                    try error.check()
                }
            ) { assertOriginatingError($0) }
        }
    }

    // MARK: APPLE-MACOS-2X8 / 1ZB — Qwen35Language.GatedDeltaNet conv tail subscript

    private func tinyQwen35TextConfiguration() throws -> Qwen35Configuration.TextConfiguration {
        try JSONDecoder.json5().decode(
            Qwen35Configuration.self,
            from: """
                {
                  "model_type": "qwen3_5_moe",
                  "text_config": {
                    "model_type": "qwen3_5_moe_text",
                    "hidden_size": 32,
                    "num_hidden_layers": 4,
                    "intermediate_size": 64,
                    "num_attention_heads": 4,
                    "num_key_value_heads": 2,
                    "linear_num_value_heads": 4,
                    "linear_num_key_heads": 2,
                    "linear_key_head_dim": 8,
                    "linear_value_head_dim": 8,
                    "linear_conv_kernel_dim": 4,
                    "head_dim": 8,
                    "full_attention_interval": 4,
                    "vocab_size": 100,
                    "rms_norm_eps": 1e-6,
                    "rope_parameters": {
                      "rope_type": "default",
                      "rope_theta": 100000.0,
                      "partial_rotary_factor": 0.25,
                      "mrope_section": [1, 1, 1]
                    }
                  },
                  "vocab_size": 100
                }
                """.data(using: .utf8)!
        ).textConfiguration
    }

    func testGatedDeltaNetWithInvalidInputsReturnsInvalid() throws {
        let text = try tinyQwen35TextConfiguration()
        try MLXMetalTestLock.withLock {
            let gdn = Qwen35Language.GatedDeltaNet(text)

            // Control: a valid prefill produces [B, S, hidden] and primes the conv slot.
            let cache = MambaCache()
            let valid = gdn(MLXArray.zeros([1, 5, 32]), cache: cache)
            eval(valid)
            XCTAssertEqual(valid.shape, [1, 5, 32])
            // convDim = 2 * keyDim + valueDim = 2 * (2 * 8) + (4 * 8) = 64; the slot keeps K - 1 rows.
            XCTAssertEqual(cache[0]?.shape, [1, 3, 64])

            XCTAssertThrowsError(
                try withError { error in
                    // The decoder layer's input after a failed op earlier in the layer
                    // (e.g. a quantized in_proj that rejected its operands).
                    let out = gdn(invalidArray(), cache: MambaCache())
                    XCTAssertFalse(out.isValid)
                    try error.check()
                }
            ) { assertOriginatingError($0) }
        }
    }

    // MARK: APPLE-MACOS-14W — the sampling-row subscript inside preparation

    /// A model whose `prepare` indexes a poisoned tensor the way every model
    /// forward does, then returns the poisoned logits.
    private final class PoisonedPrepareModel: Module, LanguageModel, @unchecked Sendable {
        var vocabularySize: Int { 4 }

        func newCache(parameters: GenerateParameters?) -> [KVCache] { [] }

        func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
            // A failed op somewhere in the forward...
            let hidden = matmul(MLXArray.zeros([2, 3]), MLXArray.zeros([4, 2]))
            // ...followed by the model's own shape-dependent subscripts.
            let rotated = hidden[.ellipsis, ..<2] + hidden[.ellipsis, 2...]
            let tail = rotated[0..., 0 ..< 1, 0...]
            return .logits(LMOutput(logits: tail))
        }

        func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
            MLXArray([Float(0), 1, 2, 3]).reshaped(1, 1, 4)
        }
    }

    func testTokenIteratorThrowsOriginatingErrorWhenModelIndexesInvalidArray() throws {
        try MLXMetalTestLock.withLock {
            XCTAssertThrowsError(
                try TokenIterator(
                    input: LMInput(tokens: MLXArray([Int32(1), 2, 3])), model: PoisonedPrepareModel(),
                    parameters: GenerateParameters(maxTokens: 1, temperature: 0))
            ) { assertOriginatingError($0) }
        }
    }
}
