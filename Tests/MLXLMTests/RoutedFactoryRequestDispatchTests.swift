// Copyright © 2026 osaurus-eval contributors

import Foundation
import MLXLMCommon
import Testing

@testable import MLXVLM

/// The registry is the ONLY route from a request to a model. These tests go through it rather than
/// calling an initialiser, because the defect they guard against was never in the initialisers:
/// the modality-aware `init(_:requesting:)` existed and was correct, and nothing called it. A test
/// that constructs a model directly passes just as happily with the registration table unwired.
@Suite("routed factory: the request reaches the model")
struct RoutedFactoryRequestDispatchTests {

    /// qwen3_5 with a vision tower in the config.
    /// Reused verbatim from `Qwen35VLMGatedDeltaTests` — a fixture already known to decode,
    /// so a failure here is the dispatch and never the JSON.
    static let configJSON = #"""
        {"model_type":"qwen3_5_moe","text_config":{"model_type":"qwen3_5_moe_text","hidden_size":32,"num_hidden_layers":4,"intermediate_size":64,"num_attention_heads":4,"num_key_value_heads":2,"linear_num_value_heads":4,"linear_num_key_heads":2,"linear_key_head_dim":8,"linear_value_head_dim":8,"linear_conv_kernel_dim":2,"head_dim":8,"full_attention_interval":4,"vocab_size":100,"rms_norm_eps":1e-06,"rope_parameters":{"rope_type":"default","rope_theta":100000.0,"partial_rotary_factor":0.25,"mrope_section":[1,1,1]}},"vision_config":{"model_type":"qwen3_vl","depth":1,"hidden_size":16,"intermediate_size":32,"out_hidden_size":32,"num_heads":4,"patch_size":2,"spatial_merge_size":2,"temporal_patch_size":1,"num_position_embeddings":32},"vocab_size":100,"image_token_id":98,"video_token_id":97}
        """#

    @Test("a text-only request builds no vision tower and says so")
    func textOnlyRequestIsHonoured() async throws {
        let data = Data(Self.configJSON.utf8)
        let model = try await VLMTypeRegistry.shared.createModel(
            configuration: data, modelType: "qwen3_5", requesting: [.text])
        let bearing = try #require(model as? any ModalityBearing)

        // If the registration table still routed to the plain `init`, this would be
        // [.text, .vision, .video] — the request would have been accepted and discarded.
        #expect(bearing.modalities == [.text], "the request must survive the trip through the registry")
    }

    @Test("no request builds everything the config offers")
    func nilRequestBuildsEverything() async throws {
        let data = Data(Self.configJSON.utf8)
        let model = try await VLMTypeRegistry.shared.createModel(
            configuration: data, modelType: "qwen3_5", requesting: nil)
        #expect((model as? any ModalityBearing)?.modalities == [.text, .vision, .video])
    }

    /// `nil` is the default, so every existing caller keeps the old behaviour.
    @Test("the legacy two-argument call is unchanged")
    func legacyCallIsUnchanged() async throws {
        let data = Data(Self.configJSON.utf8)
        let model = try await VLMTypeRegistry.shared.createModel(
            configuration: data, modelType: "qwen3_5")
        #expect((model as? any ModalityBearing)?.modalities == [.text, .vision, .video])
    }

    @Test("asking for a lane the config cannot build is refused, not silently dropped")
    func unsupportedLaneThrows() async throws {
        let textOnly = #"""
            {"model_type":"qwen3_5_moe","text_config":{"model_type":"qwen3_5_moe_text","hidden_size":32,"num_hidden_layers":4,"intermediate_size":64,"num_attention_heads":4,"num_key_value_heads":2,"linear_num_value_heads":4,"linear_num_key_heads":2,"linear_key_head_dim":8,"linear_value_head_dim":8,"linear_conv_kernel_dim":2,"head_dim":8,"full_attention_interval":4,"vocab_size":100,"rms_norm_eps":1e-06,"rope_parameters":{"rope_type":"default","rope_theta":100000.0,"partial_rotary_factor":0.25,"mrope_section":[1,1,1]}},"vocab_size":100}
            """#
        // Deliberately NOT `throws: (any Error).self` — that passes on a malformed fixture, which
        // is how this test first went green while proving nothing.
        await #expect(throws: UnconstructibleModalities.self) {
            _ = try await VLMTypeRegistry.shared.createModel(
                configuration: Data(textOnly.utf8), modelType: "qwen3_5", requesting: [.vision])
        }
    }
}
