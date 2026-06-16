// Pin throwing-contract source coverage for all three Mistral-family VLMs.
//
// Background (mirrors Gemma deep-trace §7.3):
//
// `Mistral3`, `Mistral3JANGTQ`, and `Mistral4VLM` all had a private
// `mergeInputIdsWithImageFeatures` (or `mergeImageFeatures`) helper that
// `fatalError`'d when `imagePositions.count != numImagePatches` — config/
// processor-stamp drift the caller should be able to recover from rather
// than process-abort on first image. The helpers + their callers
// (`getInputEmbeddings`, `prepare`) now thread `throws` and emit
// `VLMError.processing` with a diagnostic message naming the actual
// counts.
//
// Source-coverage style: pin the contract textually so a regression
// reintroducing `fatalError` is caught at test time without needing a
// Metal runner / loaded model.

import Foundation
@testable import MLXVLM
import Testing

@Suite("Mistral-family VLM image-mismatch throwing-contract source coverage")
struct MistralVLMImageMismatchThrowsTests {

    private static func source(_ relativePath: String) throws -> String {
        let repo = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        return try String(contentsOf: repo.appendingPathComponent(relativePath), encoding: .utf8)
    }

    @Test("Mistral3 mergeInputIdsWithImageFeatures throws VLMError.processing on count mismatch")
    func mistral3Throws() throws {
        let source = try Self.source("Libraries/MLXVLM/Models/Mistral3.swift")

        // Helper signature is `throws`.
        #expect(
            source.contains("private func mergeInputIdsWithImageFeatures(") &&
            source.contains("inputIds: MLXArray\n    ) throws -> MLXArray"),
            "Mistral3 mergeInputIdsWithImageFeatures must throw — config-drift mismatch should be recoverable.")

        // Mismatch path throws VLMError.processing with diagnostic.
        #expect(
            source.contains("throw VLMError.processing(") &&
            source.contains("Mistral3 image token count"),
            "Mismatch path must throw VLMError.processing with diagnostic message.")

        // `getInputEmbeddings` cascades the throw.
        #expect(
            source.contains("private func getInputEmbeddings(") &&
            source.contains("imageSizes: [(Int, Int)]?\n    ) throws -> MLXArray"),
            "Mistral3 getInputEmbeddings must throw to cascade.")

        // `prepare` caller uses `try`.
        #expect(
            source.contains("let embeddings = try getInputEmbeddings("),
            "Mistral3.prepare must use `try getInputEmbeddings(...)`.")

        // The prior `fatalError("Image token count ... does not match ...")` form must NOT come back.
        // (Other unrelated fatalErrors elsewhere in the file are out of scope.)
        #expect(
            !source.contains(#"fatalError("Image token count"#),
            "Mistral3 must not reintroduce fatalError on image-token count mismatch.")
    }

    @Test("Mistral3JANGTQ mergeInputIdsWithImageFeatures throws VLMError.processing on count mismatch")
    func mistral3JangtqThrows() throws {
        let source = try Self.source("Libraries/MLXVLM/Models/Mistral3VLMJANGTQ.swift")

        #expect(
            source.contains("private func mergeInputIdsWithImageFeatures(") &&
            source.contains("inputIds: MLXArray\n    ) throws -> MLXArray"),
            "Mistral3JANGTQ mergeInputIdsWithImageFeatures must throw.")

        #expect(
            source.contains("throw VLMError.processing(") &&
            source.contains("Mistral3JANGTQ image token count"),
            "Mismatch path must throw VLMError.processing with diagnostic.")

        #expect(
            source.contains("private func getInputEmbeddings(") &&
            source.contains("imageSizes: [(Int, Int)]?\n    ) throws -> MLXArray"),
            "Mistral3JANGTQ getInputEmbeddings must throw.")

        #expect(
            source.contains("let embeddings = try getInputEmbeddings("),
            "Mistral3JANGTQ.prepare must use `try getInputEmbeddings(...)`.")
    }

    @Test("Mistral4VLM mergeImageFeatures throws VLMError.processing on count mismatch")
    func mistral4Throws() throws {
        let source = try Self.source("Libraries/MLXVLM/Models/Mistral4VLM.swift")

        #expect(
            source.contains("private func mergeImageFeatures(imageFeatures: MLXArray, inputsEmbeds: MLXArray, inputIds: MLXArray) throws -> MLXArray"),
            "Mistral4VLM mergeImageFeatures must throw.")

        #expect(
            source.contains("throw VLMError.processing(") &&
            source.contains("Mistral4VLM image token count"),
            "Mistral4VLM mismatch path must throw VLMError.processing with diagnostic.")

        #expect(
            source.contains("private func getInputEmbeddings(inputIds: MLXArray?, pixelValues: MLXArray?, imageSizes: [(Int, Int)]?) throws -> MLXArray"),
            "Mistral4VLM getInputEmbeddings must throw.")

        #expect(
            source.contains("let embeddings = try getInputEmbeddings("),
            "Mistral4VLM.prepare must use `try getInputEmbeddings(...)`.")

        #expect(
            !source.contains(#"fatalError("Image token count"#),
            "Mistral4VLM must not reintroduce fatalError on count mismatch.")
    }

    @Test("Mistral4VLM projector bridge preserves required Mistral3 rope parameters")
    func mistral4ProjectorBridgePreservesRopeParameters() throws {
        let json = """
        {
          "model_type": "mistral4",
          "vocab_size": 64,
          "text_config": {
            "model_type": "mistral4",
            "hidden_size": 8,
            "num_hidden_layers": 1,
            "intermediate_size": 16,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "rms_norm_eps": 0.00001,
            "vocab_size": 64,
            "head_dim": 4,
            "rope_parameters": { "rope_theta": 10000.0 },
            "kv_lora_rank": 4,
            "q_lora_rank": 4,
            "qk_rope_head_dim": 2,
            "v_head_dim": 4,
            "qk_nope_head_dim": 2,
            "n_routed_experts": 1,
            "num_experts_per_tok": 1,
            "n_shared_experts": 1,
            "moe_intermediate_size": 16
          },
          "vision_config": {
            "model_type": "pixtral",
            "hidden_size": 8,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "intermediate_size": 16,
            "patch_size": 4,
            "image_size": 8,
            "num_channels": 3,
            "head_dim": 4,
            "rope_theta": 10000.0
          }
        }
        """
        let data = try #require(json.data(using: .utf8))
        let config = try JSONDecoder().decode(Mistral4VLMConfiguration.self, from: data)

        _ = try Mistral4VLM(config)
    }
}
