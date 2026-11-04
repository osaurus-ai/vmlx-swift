import Foundation
import MLX
import MLXNN
import MLXVLM
import XCTest

final class Gemma4UnifiedVisionLayerNormTests: XCTestCase {
    private func model(unified: Bool = true) throws -> Gemma4 {
        let config = """
            {"model_type":"\(unified ? "gemma4_unified" : "gemma4")",
             "text_config":{"model_type":"gemma4_text","hidden_size":8,"intermediate_size":16,
              "num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":1,
              "head_dim":4,"global_head_dim":4,"vocab_size":32,"sliding_window":4,
              "layer_types":["sliding_attention"]},
             "vision_config":{"model_type":"\(unified ? "gemma4_unified_vision" : "gemma4_vision")",
              "hidden_size":8,"intermediate_size":16,"num_hidden_layers":1,
              "num_attention_heads":2,"head_dim":4,"patch_size":2,"model_patch_size":6,
              "pooling_kernel_size":3,"mm_embed_dim":8,"mm_posemb_size":16,
              "num_soft_tokens":4,"default_output_length":4,"output_proj_dims":8,
              "rms_norm_eps":0.000001}}
            """
        return try Gemma4(
            JSONDecoder().decode(Gemma4Configuration.self, from: Data(config.utf8)),
            requesting: [.text, .vision])
    }

    func testUnifiedVisionLayerNormsMatchReferenceOnLowVariancePixelsAndFeatures() throws {
        try MLXMetalTestLock.withLock {
            let model = try model()
            let norms = Dictionary(
                uniqueKeysWithValues: model.leafModules().flattened().compactMap {
                    key, module -> (String, LayerNorm)? in
                    guard key.hasPrefix("vision_embedder."), let norm = module as? LayerNorm else {
                        return nil
                    }
                    return (key, norm)
                })
            XCTAssertEqual(
                Set(norms.keys),
                Set([
                    "vision_embedder.patch_ln1", "vision_embedder.patch_ln2",
                    "vision_embedder.pos_norm",
                ]))
            for (name, norm) in norms {
                // Upstream unified vision uses nn.LayerNorm's 1e-5 default,
                // independently of the 1e-6 used by Gemma's RMSNorms.
                XCTAssertEqual(norm.eps, 1e-5, name)
                let values = (0 ..< norm.dimensions).map { Float($0 % 7 - 3) * 0.0001 }
                let mean = values.reduce(0.0) { $0 + Double($1) } / Double(values.count)
                let variance =
                    values.reduce(0.0) { $0 + pow(Double($1) - mean, 2) } / Double(values.count)
                let expected = values.map { Float((Double($0) - mean) / sqrt(variance + 1e-5)) }
                let actual = norm(MLXArray(values).reshaped(1, -1)).asArray(Float.self)
                let error = zip(actual, expected).map { abs($0 - $1) }.max() ?? .infinity
                XCTAssertLessThan(error, 1e-5, "\(name) low-variance reference error \(error)")
            }
        }
    }

    func testUnifiedVisionCacheIdentityRejectsOldNormalizationSnapshots() throws {
        try MLXMetalTestLock.withLock {
            let unified = try model()
            let ordinary = try model(unified: false)
            XCTAssertEqual(
                unified.cacheStorageDTypeIdentity,
                "gemma4-native-kv-dtype-v1|unified-vision-ln-1e-5-v1")
            XCTAssertEqual(ordinary.cacheStorageDTypeIdentity, "gemma4-native-kv-dtype-v1")
        }
    }
}
