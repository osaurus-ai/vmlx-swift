import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLMCommon

final class DFlash2NoRopeTests: XCTestCase {
    private func configuration(_ noRope: Any? = nil) throws -> DFlash2Configuration {
        var json: [String: Any] = [
            "hidden_size": 8, "num_hidden_layers": 2, "num_attention_heads": 2,
            "num_key_value_heads": 1, "head_dim": 4, "intermediate_size": 16,
            "vocab_size": 256, "rms_norm_eps": 1e-6, "max_position_embeddings": 4096,
            "num_target_layers": 1, "is_causal": false,
            "layer_types": ["full_attention", "full_attention"],
            "dflash_config": [
                "block_size": 4, "conv_kernel_size": 2, "conv_group_size": 2,
                "mask_token_id": 250, "selector_rank": 4, "selector_top_k": 2,
                "target_layer_ids": [0],
            ],
        ]
        if let noRope { json["no_rope_layers"] = noRope }
        return try DFlash2Configuration(json: json)
    }

    func testDefaultAndExplicitLayerSelection() throws {
        for value: Any? in [nil, [Int]()] {
            let config = try configuration(value)
            XCTAssertTrue(config.noRopeLayers.isEmpty)
            XCTAssertTrue(DFlash2Attention(config, layerIndex: 0).usesRope)
            XCTAssertTrue(DFlash2Attention(config, layerIndex: 1).usesRope)
        }
        let config = try configuration([1])
        XCTAssertEqual(config.noRopeLayers, [1])
        XCTAssertTrue(DFlash2Attention(config, layerIndex: 0).usesRope)
        XCTAssertFalse(DFlash2Attention(config, layerIndex: 1).usesRope)
    }

    func testMalformedAndOutOfRangeListsAreRejected() {
        for value: Any in [["1"], [true], [0.5], 0, NSNull(), [-1], [2]] {
            XCTAssertThrowsError(try configuration(value), "unexpectedly accepted \(value)")
        }
    }

    /// Inspect actual cached keys after a nonzero position: selected layers keep unrotated keys.
    func testSelectedLayerSkipsRotationWhileOrdinaryLayerRetainsIt() throws {
        let lock = lockSerializedMLXTest()
        defer { lock.unlock() }
        let config = try configuration([1])
        let plain = DFlash2Attention(config, layerIndex: 1)
        let rotated = DFlash2Attention(config, layerIndex: 0)
        rotated.update(parameters: plain.parameters())
        let rope = RoPE(dimensions: 4, traditional: false, base: 10000)
        let context = MLXArray(Array(1 ... 16).map(Float.init)).reshaped(1, 2, 8)
        let query = context[0..., ..<1, 0...]
        let expected = plain.kNorm(plain.wk(context).reshaped(1, 2, 1, 4)).transposed(0, 2, 1, 3)
        let a = KVCacheSimple()
        let b = KVCacheSimple()
        for cache in [a, b] {
            let prefix = MLXArray.zeros([1, 1, 7, 4])
            _ = cache.update(keys: prefix, values: prefix)
        }
        eval(plain(query, context: context, rope: rope, cache: a))
        eval(rotated(query, context: context, rope: rope, cache: b))
        let keysA = a.state[0][0..., 0..., 7..<9, 0...]
        let keysB = b.state[0][0..., 0..., 7..<9, 0...]
        XCTAssertLessThan(abs(keysA - expected).max().item(Float.self), 1e-5)
        XCTAssertLessThan(abs(keysB - rope(expected, offset: 7)).max().item(Float.self), 1e-5)
        XCTAssertGreaterThan(abs(keysB - expected).max().item(Float.self), 1e-3)
    }
}
