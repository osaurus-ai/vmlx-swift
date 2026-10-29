// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXLMCommon
import MLXVLM
import Testing

@testable import MLXLLM

/// A MiniMax-M3.1 JANGHT bundle (`minimax_m3_vl` with a `vision_config`, JANGT routed experts declared per module as
/// `{"mode": "jangt", "kbits": ...}` WITHOUT `bits`) as an app loads it: `loadModelContainer` tries the VLM factory,
/// then the LLM factory, and surfaces the first "real" error.
///
/// Before the fix the VLM factory's strict `BaseConfiguration` decode threw "Missing field
/// 'quantization.model.layers.N.mlp.switch_mlp.down_proj.bits'" for every JANGT entry, a real error that masked the
/// LLM factory's: Osaurus showed that parse error whenever the LLM load failed for any reason (the GUI proof's
/// per-file symlink farm), never the LLM's own reason. Now per-module entries without `bits` decode (a custom format's
/// `mode` as `.formatOwned`, MLX's microscaling modes with their fixed geometry), the VLM factory reports the model
/// type as unsupported, and the LLM factory's result (a load, or its real error) is what the caller sees.
@Suite("MiniMax-M3.1 bundle load routing (VLM + LLM factories)", .serialized)
struct MiniMaxM3BundleLoadRoutingTests {
    static let fixtures = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        .appendingPathComponent("Fixtures/MiniMaxM3")

    struct NullTokenizer: Tokenizer {
        var bosToken: String? { nil }
        var eosToken: String? { nil }
        var unknownToken: String? { nil }
        func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }
        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
        func convertTokenToId(_ token: String) -> Int? { nil }
        func convertIdToToken(_ id: Int) -> String? { nil }
        func applyChatTemplate(
            messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?
        ) throws -> [Int] { [] }
    }

    struct NullTokenizerLoader: TokenizerLoader {
        func load(from directory: URL) async throws -> any Tokenizer { NullTokenizer() }
    }

    /// A private copy of the tiny M3.1 JANGHT bundle, with the released bundle's VLM keys (a `vision_config`, the VL
    /// architecture) as the real config.json carries them; returns (scratch root, bundle dir).
    static func bundle(vision: Bool = true) throws -> (URL, URL) {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("m31-routing-\(UUID().uuidString)")
        let dir = root.appendingPathComponent("real")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        try FileManager.default.copyItem(at: fixtures.appendingPathComponent("m31-janght"), to: dir)
        if vision {
            let url = dir.appendingPathComponent("config.json")
            var config = try #require(try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any])
            config["architectures"] = ["MiniMaxM3SparseForConditionalGeneration"]
            config["vision_config"] = [
                "model_type": "minimax_m3_vision", "hidden_size": 64, "num_hidden_layers": 1,
                "num_attention_heads": 2, "patch_size": 14, "image_size": 224,
            ]
            config["image_token_index"] = 200_025
            try JSONSerialization.data(withJSONObject: config).write(to: url)
        }
        return (root, dir)
    }

    @Test("Per-module entries without bits decode: custom formats as .formatOwned, MLX microscaling modes with their geometry")
    func perModuleEntriesWithoutBits() throws {
        let json = """
            {"model_type": "minimax_m3_vl", "quantization": {
              "group_size": 64, "bits": 8,
              "model.layers.1.mlp.switch_mlp.down_proj": {"mode": "jangt", "kbits": 1.5},
              "model.layers.1.self_attn.indexer.index_q_proj": {"mode": "mxfp8"},
              "model.layers.1.mlp.shared_experts.up_proj": {"mode": "nvfp4"},
              "model.layers.1.mlp.shared_experts.down_proj": {"mode": "mxfp4", "group_size": 32},
              "model.layers.1.self_attn.q_proj": {"bits": 4},
              "model.layers.1.self_attn.o_proj": {"group_size": 32, "bits": 8, "mode": "mxfp8"}}}
            """
        let config = try JSONDecoder.json5().decode(BaseConfiguration.self, from: Data(json.utf8))
        let plan = try #require(config.perLayerQuantization)
        #expect(plan.perLayerQuantization["model.layers.1.mlp.switch_mlp.down_proj"] == .formatOwned(mode: "jangt"))
        #expect(plan.quantization(layer: "model.layers.1.mlp.switch_mlp.down_proj") == nil)
        #expect(plan.quantization(layer: "model.layers.1.self_attn.indexer.index_q_proj") == BaseConfiguration.Quantization(groupSize: 32, bits: 8, mode: .mxfp8))
        #expect(plan.quantization(layer: "model.layers.1.mlp.shared_experts.up_proj") == BaseConfiguration.Quantization(groupSize: 16, bits: 4, mode: .nvfp4))
        #expect(plan.quantization(layer: "model.layers.1.mlp.shared_experts.down_proj") == BaseConfiguration.Quantization(groupSize: 32, bits: 4, mode: .mxfp4))
        #expect(plan.quantization(layer: "model.layers.1.self_attn.q_proj") == BaseConfiguration.Quantization(groupSize: 64, bits: 4, mode: .affine))
        #expect(plan.quantization(layer: "model.layers.1.self_attn.o_proj") == BaseConfiguration.Quantization(groupSize: 32, bits: 8, mode: .mxfp8))
        // Controls: affine has no fixed width (bits stay required); an unknown mode WITH bits is still an error.
        for bad in [#"{"mode": "affine"}"#, #"{"mode": "jangt", "bits": 2, "group_size": 64}"#] {
            let doc = #"{"model_type": "x", "quantization": {"group_size": 64, "bits": 8, "m.p": "# + bad + "}}"
            #expect(throws: DecodingError.self) {
                _ = try JSONDecoder.json5().decode(BaseConfiguration.self, from: Data(doc.utf8))
            }
        }
    }

    @Test("The VLM factory reports the M3.1 JANGHT bundle as an unsupported model type, not a config.json parse error")
    func vlmFactoryDeclinesCleanly() async throws {
        let (root, dir) = try Self.bundle()
        defer { try? FileManager.default.removeItem(at: root) }
        do {
            _ = try await VLMModelFactory.shared.load(from: dir, using: NullTokenizerLoader())
            Issue.record("the VLM factory built an M3 model")
        } catch let error as ModelFactoryError {
            switch error {
            case .unsupportedModelType(let type):
                #expect(type.contains("minimax_m3"))
            default:
                Issue.record("VLM factory: \(error.localizedDescription)")
            }
        }
    }

    @Test("Through the factory registry: the bundle, and a symlinked bundle directory, load as MiniMaxM3Model")
    func registryLoads() async throws {
        let (root, dir) = try Self.bundle()
        defer { try? FileManager.default.removeItem(at: root) }
        let link = root.appendingPathComponent("linked")
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: dir)
        for target in [dir, link] {
            let container = try await loadModelContainer(
                from: target, using: NullTokenizerLoader(), configuration: ModelConfiguration(directory: target),
                loadConfiguration: LoadConfiguration())
            let type = await container.perform { String(describing: Swift.type(of: $0.model)) }
            #expect(type == "MiniMaxM3Model", "\(target.lastPathComponent): \(type)")
        }
    }

    @Test("A failing LLM load surfaces its own reason (a symlinked shard file), not the VLM factory's parse error")
    func llmFailureIsNotMasked() async throws {
        let (root, dir) = try Self.bundle()
        defer { try? FileManager.default.removeItem(at: root) }
        // The GUI proof's staging: a directory of per-file symlinks into the bundle. JANGH reads its files in place
        // (O_NOFOLLOW, identity-checked), so the load must fail, and say why.
        let farm = root.appendingPathComponent("farm")
        try FileManager.default.createDirectory(at: farm, withIntermediateDirectories: true)
        for name in try FileManager.default.contentsOfDirectory(atPath: dir.path) {
            try FileManager.default.createSymbolicLink(
                at: farm.appendingPathComponent(name), withDestinationURL: dir.appendingPathComponent(name))
        }
        do {
            _ = try await loadModelContainer(
                from: farm, using: NullTokenizerLoader(), configuration: ModelConfiguration(directory: farm),
                loadConfiguration: LoadConfiguration())
            Issue.record("a per-file symlink farm loaded")
        } catch {
            let message = "\(error.localizedDescription) | \(error)"
            #expect(!message.contains("Missing field"), "the VLM parse error masked the LLM's: \(message)")
            #expect(message.contains("cannot open JANGH metadata file"), "\(message)")
            #expect(message.contains("symbolic link"), "\(message)")
        }
    }
}
