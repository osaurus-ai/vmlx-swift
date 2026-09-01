// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// The bug this design exists to make impossible: `requesting: [.video]` was accepted (video IS
// constructible) and then built nothing, because construction asked `modalities.contains(.vision)`.
// Both checks were right about different questions. Lanes and modules are now different types.

import Foundation
import MLX
import MLXLMCommon
import MLXVLM
import Testing

@Suite("Model construction plan")
struct ModelConstructionPlanTests {

    /// A stand-in for a qwen3_5-shaped family: one tower serving both image and video.
    private struct SharedTowerFamily: ModelComponentMapping {
        struct Configuration { let hasVision: Bool }

        static func constructibleModalities(of c: Configuration)
            -> Set<ModelRuntimeRequestModality>
        { c.hasVision ? [.text, .vision, .video] : [.text] }

        static func components(
            for requested: Set<ModelRuntimeRequestModality>, of c: Configuration
        ) -> Set<ModelComponent> {
            var out: Set<ModelComponent> = []
            // The dependency, stated once: EITHER media lane needs the tower.
            if requested.contains(.vision) || requested.contains(.video) { out.insert(.visionTower) }
            return out
        }

        static func servedModalities(
            by components: Set<ModelComponent>, of c: Configuration
        ) -> Set<ModelRuntimeRequestModality> {
            var m: Set<ModelRuntimeRequestModality> = []
            if components.contains(.languageCore) { m.insert(.text) }
            if components.contains(.visionTower) { m.formUnion([.vision, .video]) }
            return m
        }
    }

    private static let multimodal = SharedTowerFamily.Configuration(hasVision: true)

    @Test("a video-only request builds the shared tower — the bug this replaces")
    func videoOnlyBuildsTheTower() throws {
        let plan = try SharedTowerFamily.resolveConstruction(Self.multimodal, requesting: [.video])
        #expect(plan.requested == [.video], "the lane stays literal — video is not an image request")
        #expect(plan.builds(.visionTower), "video needs the tower")
        #expect(plan.builds(.languageCore))
    }

    @Test("an image-only request builds the same tower")
    func visionOnlyBuildsTheTower() throws {
        let plan = try SharedTowerFamily.resolveConstruction(Self.multimodal, requesting: [.vision])
        #expect(plan.requested == [.vision])
        #expect(plan.builds(.visionTower))
    }

    @Test("a text-only request builds no tower")
    func textOnlyBuildsNoTower() throws {
        let plan = try SharedTowerFamily.resolveConstruction(Self.multimodal, requesting: [.text])
        #expect(plan.components == [.languageCore])
    }

    @Test("the language core is present even when no lane names it")
    func languageCoreAlwaysBuilt() throws {
        for req: Set<ModelRuntimeRequestModality> in [[.text], [.vision], [.video], [.text, .video]] {
            let plan = try SharedTowerFamily.resolveConstruction(Self.multimodal, requesting: req)
            #expect(plan.builds(.languageCore), "\(req)")
        }
    }

    @Test("nil requests everything the configuration offers")
    func nilRequestsAll() throws {
        let plan = try SharedTowerFamily.resolveConstruction(Self.multimodal, requesting: nil)
        #expect(plan.requested == [.text, .vision, .video])
        #expect(plan.builds(.visionTower))
    }

    @Test("empty and unsupported selections are refused")
    func invalidSelectionsRefused() {
        #expect(throws: (any Error).self) {
            try SharedTowerFamily.resolveConstruction(Self.multimodal, requesting: [])
        }
        #expect(throws: (any Error).self) {
            try SharedTowerFamily.resolveConstruction(Self.multimodal, requesting: [.audio])
        }
        let textOnly = SharedTowerFamily.Configuration(hasVision: false)
        #expect(throws: (any Error).self) {
            try SharedTowerFamily.resolveConstruction(textOnly, requesting: [.vision])
        }
    }

    /// Gemma 4's two audio paths: the same lane, different modules. This is the case that showed
    /// `constructibleModalities` cannot be a list of towers.
    @Test("one audio lane maps to different modules depending on the path")
    func audioLaneMapsPerPath() throws {
        struct Gemma4Like: ModelComponentMapping {
            struct Configuration { let conformer: Bool; let unifiedAudio: Bool }
            static func constructibleModalities(of c: Configuration)
                -> Set<ModelRuntimeRequestModality>
            { c.conformer || c.unifiedAudio ? [.text, .vision, .audio] : [.text, .vision] }
            static func components(
                for requested: Set<ModelRuntimeRequestModality>, of c: Configuration
            ) -> Set<ModelComponent> {
                var out: Set<ModelComponent> = []
                if requested.contains(.vision) { out.insert(.visionTower) }
                if requested.contains(.audio) {
                    out.insert(.audioProjection)               // both paths need it
                    if c.conformer { out.insert(.audioTower) } // only the encoder path
                }
                return out
            }
            static func servedModalities(
                by components: Set<ModelComponent>, of c: Configuration
            ) -> Set<ModelRuntimeRequestModality> {
                var m: Set<ModelRuntimeRequestModality> = []
                if components.contains(.languageCore) { m.insert(.text) }
                if components.contains(.visionTower) { m.insert(.vision) }
                if components.contains(.audioProjection) { m.insert(.audio) }
                return m
            }
        }
        let conformer = try Gemma4Like.resolveConstruction(
            .init(conformer: true, unifiedAudio: false), requesting: [.audio])
        #expect(conformer.components == [.languageCore, .audioTower, .audioProjection])

        let unified = try Gemma4Like.resolveConstruction(
            .init(conformer: false, unifiedAudio: true), requesting: [.audio])
        #expect(unified.components == [.languageCore, .audioProjection],
                "the unified path has no conformer, and must not be rejected for lacking one")
    }
    // MARK: - The real family, not a stand-in

    /// A vision-capable qwen3_5 config, decoded rather than hand-built so the shapes are real.
    private static func qwen35Config() throws -> Qwen35Configuration {
        let json = """
            {"model_type":"qwen3_5","text_config":{"hidden_size":64,"num_hidden_layers":2,
              "intermediate_size":128,"num_attention_heads":4,"num_key_value_heads":2,
              "rms_norm_eps":1e-6,"vocab_size":100,"rope_theta":10000.0},
             "vision_config":{"model_type":"qwen3_5","depth":2,"hidden_size":32,"num_heads":2,"in_channels":3,
              "intermediate_size":64,"out_hidden_size":64,"patch_size":16,"spatial_merge_size":2,
              "temporal_patch_size":2,"num_position_embeddings":64}}
            """
        return try JSONDecoder.json5().decode(
            Qwen35Configuration.self, from: Data(json.utf8))
    }

    /// The bug, on the class that had it: `[.video]` was accepted and built NO tower.
    @Test("Qwen35: a video-only request builds the vision tower")
    func qwen35VideoOnlyBuildsTower() throws {
        let cfg = try Self.qwen35Config()
        let plan = try Qwen35.resolveConstruction(cfg, requesting: [.video])
        #expect(plan.builds(.visionTower), "video needs the shared tower")

        let model = try Qwen35(cfg, requesting: [.video])
        let visionParams = model.parameters().flattened().map(\.0)
            .filter { $0.contains("vision_tower") }
        #expect(!visionParams.isEmpty, "video-only instance allocated NO vision parameters")
    }

    /// The control: text-only must still allocate nothing.
    @Test("Qwen35: a text-only request allocates no vision parameters")
    func qwen35TextOnlyAllocatesNothing() throws {
        let model = try Qwen35(try Self.qwen35Config(), requesting: [.text])
        let visionParams = model.parameters().flattened().map(\.0)
            .filter { $0.contains("vision_tower") }
        #expect(visionParams.isEmpty, "text-only leaked \(visionParams.count) vision parameters")
    }

    /// And the default is unchanged: everything the config offers.
    @Test("Qwen35: the default construction still builds the tower")
    func qwen35DefaultUnchanged() throws {
        let model = Qwen35(try Self.qwen35Config())
        #expect(model.plan.builds(.visionTower))
        #expect(model.modalities == [.text, .vision, .video], "default serves every lane")
    }

    /// Muse Glimmer had the identical defect, with the request Jang cited: `[.text, .video]`.
    @Test("MuseGlimmer: a text+video request builds the shared tower")
    func museGlimmerTextVideoBuildsTower() throws {
        let json = """
            {"model_type":"muse_glimmer","hidden_size":64,"num_hidden_layers":2,
             "intermediate_size":128,"num_attention_heads":4,"num_key_value_heads":2,
             "rms_norm_eps":1e-6,"vocab_size":100,"rope_theta":10000.0,
             "vision_config":{"model_type":"muse_glimmer","hidden_size":32,"num_hidden_layers":2,
               "intermediate_size":64,"num_attention_heads":2,"patch_size":16,"image_size":64,
               "merge_size":2}}
            """
        let cfg = try JSONDecoder.json5().decode(
            MuseGlimmerConfiguration.self, from: Data(json.utf8))
        let plan = try MuseGlimmer.resolveConstruction(cfg, requesting: [.text, .video])
        #expect(plan.builds(.visionTower), "text+video needs the shared tower")

        let model = try MuseGlimmer(cfg, requesting: [.text, .video])
        let visionParams = model.parameters().flattened().map(\.0)
            .filter { $0.lowercased().contains("vision") }
        #expect(!visionParams.isEmpty, "text+video instance allocated NO vision parameters")

        let textOnly = try MuseGlimmer(cfg, requesting: [.text])
        let none = textOnly.parameters().flattened().map(\.0)
            .filter { $0.lowercased().contains("vision") }
        #expect(none.isEmpty, "text-only leaked \(none.count) vision parameters")
    }

    /// Jang's blocker #4, against REAL bundle configs on this machine: a unified-audio bundle can
    /// process audio through an encoder-free projection, and was being told `.audio` is not
    /// constructible because it has no conformer tower.
    @Test("Gemma4: both audio paths accept an .audio request, with different modules")
    func gemma4BothAudioPathsAccepted() throws {
        let root = URL(fileURLWithPath: NSHomeDirectory())
            .appendingPathComponent("Library/MLModels")
        var sawUnified = false, sawConformer = false
        let fm = FileManager.default
        for org in (try? fm.contentsOfDirectory(atPath: root.path))?.sorted() ?? [] {
            let orgDir = root.appendingPathComponent(org)
            for kid in (try? fm.contentsOfDirectory(atPath: orgDir.path))?.sorted() ?? [] {
                let cfgURL = orgDir.appendingPathComponent(kid).appendingPathComponent("config.json")
                guard let data = try? Data(contentsOf: cfgURL),
                    let obj = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any],
                    (obj["model_type"] as? String)?.hasPrefix("gemma4") == true,
                    let audio = obj["audio_config"] as? [String: Any],
                    let cfg = try? JSONDecoder.json5().decode(Gemma4Configuration.self, from: data)
                else { continue }
                let unified = (audio["model_type"] as? String) == "gemma4_unified_audio"
                let plan = try Gemma4.resolveConstruction(cfg, requesting: [.audio])
                #expect(plan.builds(.audioProjection), "\(kid): audio needs the projection")
                if unified {
                    sawUnified = true
                    #expect(!plan.builds(.audioTower), "\(kid): unified has no conformer to build")
                } else {
                    sawConformer = true
                    #expect(plan.builds(.audioTower), "\(kid): E-series needs the conformer")
                }
            }
        }
        // A bundle-gated assertion that never ran looks exactly like one that passed.
        print("gemma4 audio paths exercised — unified: \(sawUnified), conformer: \(sawConformer)")
        #expect(sawUnified || sawConformer, "no gemma4 audio bundle on this machine")
    }

    /// The second-order form of the same conflation: a video-only instance HAS the tower, so
    /// `sanitize` must keep its weights. Asking about the LANE there would drop the weights of a
    /// tower that was just built — a model with an allocated but unloaded vision tower.
    @Test("MuseGlimmer: a video-only instance keeps its vision weights through sanitize")
    func museGlimmerVideoKeepsVisionWeights() throws {
        let json = """
            {"model_type":"muse_glimmer","hidden_size":64,"num_hidden_layers":2,
             "intermediate_size":128,"num_attention_heads":4,"num_key_value_heads":2,
             "rms_norm_eps":1e-6,"vocab_size":100,"rope_theta":10000.0,
             "vision_config":{"model_type":"muse_glimmer","hidden_size":32,"num_hidden_layers":2,
               "intermediate_size":64,"num_attention_heads":2,"patch_size":16,"image_size":64,
               "merge_size":2}}
            """
        let cfg = try JSONDecoder.json5().decode(
            MuseGlimmerConfiguration.self, from: Data(json.utf8))
        let weights: [String: MLXArray] = [
            "model.vision_tower.blocks.0.attn.qkv.weight": MLXArray([1.0] as [Float]),
            "model.language_model.norm.weight": MLXArray([2.0] as [Float]),
        ]
        let videoOnly = try MuseGlimmer(cfg, requesting: [.text, .video])
        let kept = videoOnly.sanitize(weights: weights)
        #expect(
            kept.keys.contains { $0.lowercased().contains("vision") },
            "video-only built the tower but sanitize dropped its weights: \(kept.keys.sorted())")

        let textOnly = try MuseGlimmer(cfg, requesting: [.text])
        let dropped = textOnly.sanitize(weights: weights)
        #expect(
            !dropped.keys.contains { $0.lowercased().contains("vision") },
            "text-only has no tower, so vision weights must be dropped")
    }

    /// Blocker #2: no PUBLIC path may construct an empty or unsupported instance.
    ///
    /// Asserted on the source rather than by calling the removed overloads — a test that could call
    /// them would not compile, which is the point, but a compile error is not a regression signal.
    /// This fails loudly if someone re-widens one.
    @Test("no public initializer takes an unchecked modality set")
    func noPublicUncheckedConstructor() throws {
        let families = [
            "Libraries/MLXVLM/Models/Qwen35.swift",
            "Libraries/MLXVLM/Models/MuseGlimmer.swift",
            "Libraries/MLXVLM/Models/Gemma4.swift",
            "Libraries/MLXVLM/Models/Qwen4Exp.swift",
            "Libraries/MLXVLM/Models/DiffusionGemmaVLM.swift",
        ]
        var offenders: [String] = []
        for file in families {
            guard let src = try? String(contentsOfFile: file, encoding: .utf8) else {
                Issue.record("missing \(file)"); continue
            }
            let lines = src.split(separator: "\n", omittingEmptySubsequences: false).map(String.init)
            for (n, line) in lines.enumerated() {
                // a public init/func whose parameter list carries a raw modality set
                guard line.contains("public init(") || line.contains("public func make") else { continue }
                let window = lines[n...min(n + 4, lines.count - 1)].joined(separator: " ")
                if window.contains("modalities: Set<ModelRuntimeRequestModality>") {
                    offenders.append("\(file.split(separator: "/").last!):\(n + 1)")
                }
            }
        }
        #expect(
            offenders.isEmpty,
            "public unchecked constructor(s): \(offenders.joined(separator: ", "))")
    }

    /// rcfa's point: `modalities` must report what the model CAN DO, and that is family-specific —
    /// never "the request plus text". A speech-to-speech model has no text lane to append.
    @Test("served modalities come from the family, not from a blanket text rule")
    func servedIsFamilySpecific() throws {
        /// A hypothetical voice-in/voice-out model: its core serves AUDIO, not text.
        struct VoiceFamily: ModelComponentMapping {
            struct Configuration { let hasAudio: Bool }
            static func constructibleModalities(of c: Configuration)
                -> Set<ModelRuntimeRequestModality>
            { c.hasAudio ? [.audio] : [] }
            static func components(
                for requested: Set<ModelRuntimeRequestModality>, of c: Configuration
            ) -> Set<ModelComponent> {
                requested.contains(.audio) ? [.audioProjection] : []
            }
            static func servedModalities(
                by components: Set<ModelComponent>, of c: Configuration
            ) -> Set<ModelRuntimeRequestModality> {
                // NO `.text`. The core here is not a text LM.
                components.contains(.audioProjection) ? [.audio] : []
            }
        }
        let plan = try VoiceFamily.resolveConstruction(.init(hasAudio: true), requesting: [.audio])
        #expect(plan.served == [.audio], "a blanket +.text rule would describe this model wrongly")
        #expect(plan.builds(.languageCore), "the core is still built; it just serves no text lane")

        // And a text family still reports text, because ITS core is a text LM.
        let text = try SharedTowerFamily.resolveConstruction(Self.multimodal, requesting: [.video])
        #expect(text.served == [.text, .vision, .video],
                "a video request still yields a model that can also take text and images")
    }

}
