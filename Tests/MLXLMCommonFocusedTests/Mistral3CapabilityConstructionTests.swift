// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Capability-driven construction for the mistral3 family, checked without a GPU where possible and
// against a real bundle where one is present. Same contract as Muse Glimmer — deliberately, since
// a convention that only one family follows is a proposal rather than an architecture.

import Foundation
import MLX
import MLXLMCommon
import MLXVLM
import Testing

@Suite("Mistral3 capability-driven construction")
struct Mistral3CapabilityConstructionTests {

    typealias Modalities = Set<ModelRuntimeRequestModality>

    /// A text-only Mistral 3 conversion: the same document minus its vision section. Until
    /// `visionConfig` became optional this could not DECODE at all.
    private static let textOnlyJSON = """
        {"model_type":"mistral3",
         "text_config":{"model_type":"mistral","hidden_size":128,"num_hidden_layers":2,
           "intermediate_size":256,"num_attention_heads":4,"num_key_value_heads":2,
           "rms_norm_eps":1e-6,"vocab_size":1000,"rope_theta":10000.0,"head_dim":32,
           "max_position_embeddings":4096}}
        """

    private func textOnlyConfig() throws -> Mistral3VLMConfiguration {
        try JSONDecoder.json5().decode(
            Mistral3VLMConfiguration.self, from: Data(Self.textOnlyJSON.utf8))
    }

    @Test("a config with no vision_config decodes, and builds text only")
    func textOnlyDecodes() throws {
        let cfg = try textOnlyConfig()
        #expect(cfg.visionConfig == nil)
        #expect(Mistral3VLM.constructibleModalities(of: cfg) == [.text])
        #expect(Mistral3VLM(cfg).modalities == [.text])
    }

    /// Pixtral is image-only. Declaring `.video` here would let a video request pass construction
    /// and fail later at `prepare()`, which is exactly the failure the modality set exists to move
    /// earlier.
    @Test("the vision lane is image-only — no video is claimed")
    func noVideoLaneIsClaimed() throws {
        guard let cfg = try Self.realBundleConfig() else { return }
        #expect(Mistral3VLM.constructibleModalities(of: cfg) == [.text, .vision])
        #expect(!Mistral3VLM.constructibleModalities(of: cfg).contains(.video))
    }

    @Test("asking a text-only config for vision fails at construction, not at prepare()")
    func visionRequestOnTextOnlyThrows() throws {
        let cfg = try textOnlyConfig()
        #expect(throws: UnconstructibleModalities.self) {
            _ = try Mistral3VLM(cfg, requesting: [.text, .vision])
        }
    }

    @Test("a caller may ask for less than the bundle offers")
    func narrowerRequestIsHonoured() throws {
        guard let cfg = try Self.realBundleConfig() else { return }
        let narrowed = try Mistral3VLM(cfg, requesting: [.text])
        #expect(narrowed.modalities == [.text])
        let full = Mistral3VLM(cfg)
        #expect(full.modalities == [.text, .vision])
    }

    // MARK: the real bundle

    /// The SMALLEST locally held mistral3 bundle, if any. Absent on a machine without one, so
    /// these skip.
    ///
    /// Smallest rather than first-found: these tests construct the model, and the qualifying
    /// bundles here span 2.6 GB to 91 GB. Any of them proves the same thing about construction,
    /// so there is no reason to pay for the largest — and "first alphabetically" was picking a
    /// 119B purely by accident of the org name.
    static func realBundleConfig() throws -> Mistral3VLMConfiguration? {
        guard let dir = smallestBundleDirectory(modelType: "mistral3") else { return nil }
        let url = dir.appendingPathComponent("config.json")
        guard let data = try? Data(contentsOf: url) else { return nil }
        return try? JSONDecoder.json5().decode(Mistral3VLMConfiguration.self, from: data)
    }

    /// Bundle directories are ranked by total `.safetensors` bytes, which is the cost that
    /// actually matters here — config decoding is free either way.
    static func smallestBundleDirectory(modelType: String) -> URL? {
        let root = URL(fileURLWithPath: NSHomeDirectory()).appendingPathComponent("Library/MLModels")
        let fm = FileManager.default
        guard let orgs = try? fm.contentsOfDirectory(atPath: root.path) else { return nil }
        var best: (url: URL, bytes: UInt64)? = nil
        for org in orgs.sorted() {
            let orgDir = root.appendingPathComponent(org)
            guard let kids = try? fm.contentsOfDirectory(atPath: orgDir.path) else { continue }
            for kid in kids.sorted() {
                let dir = orgDir.appendingPathComponent(kid)
                guard let data = try? Data(contentsOf: dir.appendingPathComponent("config.json")),
                    let obj = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any],
                    obj["model_type"] as? String == modelType,
                    obj["vision_config"] != nil,
                    let files = try? fm.contentsOfDirectory(atPath: dir.path)
                else { continue }
                var bytes: UInt64 = 0
                for f in files where f.hasSuffix(".safetensors") {
                    let attrs = try? fm.attributesOfItem(atPath: dir.appendingPathComponent(f).path)
                    bytes += (attrs?[.size] as? UInt64) ?? 0
                }
                if best == nil || bytes < best!.bytes { best = (dir, bytes) }
            }
        }
        return best?.url
    }

    /// The payoff, on a real bundle: a text-only load allocates no vision tower and no projector.
    /// Pixtral's tower is a flat ~0.75 GiB whatever the text model's size, so this is a smaller
    /// win than Muse Glimmer's 3.84 GB — but it is the same win, and it is measured.
    @Test("text-only construction omits the vision tower entirely")
    func textOnlyOmitsTheTower() throws {
        guard let cfg = try Self.realBundleConfig() else { return }
        func count(_ m: Mistral3VLM) -> Int {
            m.parameters().flattened().reduce(0) { $0 + $1.1.size }
        }
        let full = count(Mistral3VLM(cfg))
        let text = count(try Mistral3VLM(cfg, requesting: [.text]))
        #expect(text < full)
        print("full  : \(full) params")
        print("text  : \(text) params")
        print("spared: \(full - text) params")
    }

    /// A text-only instance must expose no vision parameters at all — not merely fewer.
    @Test("a text-only instance exposes no vision keys")
    func textOnlyHasNoVisionKeys() throws {
        guard let cfg = try Self.realBundleConfig() else { return }
        let text = try Mistral3VLM(cfg, requesting: [.text])
        let keys = text.parameters().flattened().map(\.0)
        #expect(!keys.isEmpty)
        for k in keys {
            #expect(!k.contains("vision_tower"), "vision parameter present: \(k)")
            #expect(!k.contains("multi_modal_projector"), "projector parameter present: \(k)")
        }
    }

    /// The bundle still SHIPS its vision weights. Sanitize must drop them rather than rehome them
    /// onto modules that were never built, which would leave keys with no destination.
    @Test("sanitize drops the vision weights a text-only instance cannot receive")
    func sanitizeDropsVisionWeights() throws {
        guard let cfg = try Self.realBundleConfig() else { return }
        let text = try Mistral3VLM(cfg, requesting: [.text])
        let out = text.sanitize(weights: [
            "vision_tower.transformer.layers.0.attention.wq.weight": MLXArray([0.0] as [Float]),
            "model.multi_modal_projector.linear_1.weight": MLXArray([0.0] as [Float]),
            "model.language_model.layers.0.self_attn.q_proj.weight": MLXArray([0.0] as [Float]),
        ])
        #expect(out.keys.first { $0.contains("vision_tower") } == nil)
        #expect(out.keys.first { $0.contains("multi_modal_projector") } == nil)
        // The language weight survives, so the drop is targeted rather than wholesale.
        #expect(out.count == 1)
    }
}
