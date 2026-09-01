// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Every dual-path family now answers the same question the same way. The point of this suite is
// UNIVERSALITY: a convention a few families follow is a proposal, and the next person adding a
// multimodal model should find a pattern rather than a special case. So these tests assert the
// SHAPE across families rather than one family's behaviour.
//
// Configurations come from real local bundles, with `vision_config` removed where a text-only case
// is needed — which is what a text-only conversion of that bundle would look like. Hand-written
// fixtures were tried first and drifted from the real schemas immediately; deriving from the
// bundles keeps the test honest and removes a whole class of maintenance.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import MLXVLM
import Testing

@Suite("Capability-driven construction across the dual-path families")
struct UniversalCapabilityConstructionTests {

    typealias Modalities = Set<ModelRuntimeRequestModality>

    /// The raw `config.json` of the SMALLEST local bundle with this `model_type`, if any.
    ///
    /// Smallest rather than first-found: several of these tests construct the model, and the
    /// qualifying bundles span 2.6 GB to 91 GB. Any of them proves the same thing, so ranking by
    /// weight size keeps the suite cheap — "first alphabetically" was selecting a 119B by accident
    /// of the org name.
    static func rawConfig(modelType: String, needsVision: Bool = true) -> [String: Any]? {
        let root = URL(fileURLWithPath: NSHomeDirectory()).appendingPathComponent("Library/MLModels")
        let fm = FileManager.default
        guard let orgs = try? fm.contentsOfDirectory(atPath: root.path) else { return nil }
        var best: (obj: [String: Any], bytes: UInt64)? = nil
        for org in orgs.sorted() {
            let orgDir = root.appendingPathComponent(org)
            guard let kids = try? fm.contentsOfDirectory(atPath: orgDir.path) else { continue }
            for kid in kids.sorted() {
                let dir = orgDir.appendingPathComponent(kid)
                guard let data = try? Data(contentsOf: dir.appendingPathComponent("config.json")),
                    let obj = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any],
                    obj["model_type"] as? String == modelType,
                    !needsVision || obj["vision_config"] != nil,
                    let files = try? fm.contentsOfDirectory(atPath: dir.path)
                else { continue }
                var bytes: UInt64 = 0
                for f in files where f.hasSuffix(".safetensors") {
                    let attrs = try? fm.attributesOfItem(atPath: dir.appendingPathComponent(f).path)
                    bytes += (attrs?[.size] as? UInt64) ?? 0
                }
                if best == nil || bytes < best!.bytes { best = (obj, bytes) }
            }
        }
        return best?.obj
    }

    /// The same document with its vision section removed.
    static func visionStripped(_ cfg: [String: Any]) -> Data {
        var c = cfg
        c.removeValue(forKey: "vision_config")
        if var t = c["text_config"] as? [String: Any] {
            t.removeValue(forKey: "vision_config")
            c["text_config"] = t
        }
        return try! JSONSerialization.data(withJSONObject: c)
    }

    private static func asData(_ cfg: [String: Any]) -> Data {
        try! JSONSerialization.data(withJSONObject: cfg)
    }

    /// Which families this run actually exercised.
    ///
    /// Every test here is bundle-gated, and a gate that closes silently is indistinguishable from a
    /// passing test — not hypothetical: a mutation to Gemma3's construction passed the whole suite
    /// because no `gemma3` bundle exists on this machine, so its assertions never ran.
    ///
    /// The report distinguishes THREE states, because two of them look alike and are not. The
    /// vision-bearing tests need a bundle WITH a `vision_config`; a text-only bundle of the same
    /// family satisfies neither. An earlier version of this test asked only whether any bundle
    /// existed, which would have reported a family as exercised while every one of its assertions
    /// skipped — the precise failure this test exists to prevent, in the test itself.
    ///
    /// Note also that a family's text-only bundles often carry a DIFFERENT `model_type` entirely
    /// (`gemma3_text` rather than `gemma3`, `mistral` rather than `mistral3`) and route to the LLM
    /// factory, so they are not this suite's subject at all.
    @Test("report which families this machine can actually exercise")
    func coverageIsLegible() {
        // The nine dual-path families PLUS qwen4_exp, which is VLM-only and therefore not in
        // DualPathFamilies — but the modality convention is about multimodal construction, not
        // about dual registration, so it belongs in this suite's scope.
        let families = [
            "gemma3", "gemma4", "gemma4_unified", "qwen3_5", "qwen3_5_moe",
            "mistral3", "ministral3", "muse_glimmer", "diffusion_gemma", "qwen4_exp",
        ]
        var withVision: [String] = []
        var textOnlyOnly: [String] = []
        var absent: [String] = []
        for f in families {
            if Self.rawConfig(modelType: f) != nil {
                withVision.append(f)
            } else if Self.rawConfig(modelType: f, needsVision: false) != nil {
                textOnlyOnly.append(f)
            } else {
                absent.append(f)
            }
        }
        print("exercised          : \(withVision.sorted().joined(separator: ", "))")
        if !textOnlyOnly.isEmpty {
            print("NOT exercised      : \(textOnlyOnly.sorted().joined(separator: ", "))"
                + "  — a bundle exists but carries no vision_config, so the assertions skip")
        }
        print("no bundle at all   : \(absent.sorted().joined(separator: ", "))")
        #expect(!withVision.isEmpty, "no vision bundle for any dual-path family; this suite tested nothing")
    }

    // MARK: the lanes each family declares

    /// The declared set must describe what `prepare` will actually accept, and the families DIFFER.
    /// That difference is the evidence the image/video split earns its keep: one "vision" flag
    /// could not tell Qwen 3.5 from Pixtral, and would let a video request pass construction on a
    /// family that refuses video outright.
    @Test("Qwen 3.5 claims video, because it actually consumes it")
    func qwenClaimsVideo() throws {
        guard let raw = Self.rawConfig(modelType: "qwen3_5") else { return }
        let cfg = try JSONDecoder.json5().decode(
            MLXVLM.Qwen35Configuration.self, from: Self.asData(raw))
        #expect(Qwen35.constructibleModalities(of: cfg) == [.text, .vision, .video])
    }

    @Test("Mistral 3 claims vision without video, because Pixtral is image-only")
    func mistralClaimsVisionNotVideo() throws {
        guard let raw = Self.rawConfig(modelType: "mistral3") else { return }
        let cfg = try JSONDecoder.json5().decode(
            Mistral3VLMConfiguration.self, from: Self.asData(raw))
        #expect(Mistral3VLM.constructibleModalities(of: cfg) == [.text, .vision])
        #expect(!Mistral3VLM.constructibleModalities(of: cfg).contains(.video))
    }

    /// Gemma 4 refuses video in `prepare` ("no proven vMLX video path yet"), so it must not claim
    /// the lane — and it may claim `.audio`, but only for the conformer tower.
    @Test("Gemma 4 never claims video, and claims audio only for a conformer tower")
    func gemma4LanesAreHonest() throws {
        guard let raw = Self.rawConfig(modelType: "gemma4") else { return }
        let cfg = try JSONDecoder.json5().decode(Gemma4Configuration.self, from: Self.asData(raw))
        // Asserted through the public surface only: Gemma4Configuration's own fields are
        // internal, and `constructibleModalities` is the thing callers actually consult.
        let lanes = Gemma4.constructibleModalities(of: cfg)
        #expect(lanes.contains(.text))
        #expect(lanes.contains(.vision))
        #expect(!lanes.contains(.video))
        // Audio is claimed only for the conformer tower; whether THIS bundle has one is a
        // property of the bundle, so pin only that the claim is one of the two coherent answers.
        #expect(lanes.isSubset(of: [.text, .vision, .audio]))
    }

    /// qwen4_exp is the first family to ARRIVE after the convention existed, which makes it the
    /// real test of whether the convention is one. Like qwen3_5 it genuinely consumes video —
    /// `prepare` reads `input.video` and threads video frames into the tower — so it claims the
    /// video lane rather than the image-only shape the Gemma and Pixtral families use.
    @Test("qwen4_exp declares text, vision and video, and narrows like the rest")
    func qwen4ExpFollowsTheConvention() throws {
        guard let raw = Self.rawConfig(modelType: "qwen4_exp") else { return }
        let cfg = try JSONDecoder.json5().decode(Qwen4ExpConfiguration.self, from: Self.asData(raw))
        #expect(Qwen4Exp.constructibleModalities(of: cfg) == [.text, .vision, .video])

        let narrowed = try Qwen4Exp(cfg, requesting: [.text])
        #expect(narrowed.modalities == [.text])
        #expect(Qwen4Exp(cfg).modalities == [.text, .vision, .video])

        // Narrowing must SKIP the tower, not merely declare a smaller set.
        let visionKeys = narrowed.parameters().flattened().map(\.0).filter { $0.contains("visual") }
        #expect(visionKeys.isEmpty)

        // And the weights it cannot receive are dropped rather than handed to nothing.
        let out = narrowed.sanitize(weights: [
            "visual.patch_embed.proj.weight": MLXArray([0.0] as [Float]),
            "model.language_model.layers.0.self_attn.q_proj.weight": MLXArray([0.0] as [Float]),
        ])
        #expect(out.keys.first { $0.contains("visual") } == nil)
        #expect(out.count == 1)
    }

    // MARK: a vision-less config decodes, and builds text only

    /// Before this work a text-only bundle of any of these families could not DECODE at all — the
    /// vision section was non-optional. That, not the tower, was the original reason each family
    /// needed a second registration.
    @Test("a vision-stripped config decodes and builds text only, in every family")
    func textOnlyEverywhere() throws {
        if let raw = Self.rawConfig(modelType: "gemma3") {
            let cfg = try JSONDecoder.json5().decode(
                Gemma3Configuration.self, from: Self.visionStripped(raw))
            #expect(cfg.visionConfiguration == nil)
            #expect(Gemma3.constructibleModalities(of: cfg) == [.text])
            #expect(Gemma3(cfg).modalities == [.text])
        }
        if let raw = Self.rawConfig(modelType: "mistral3") {
            let cfg = try JSONDecoder.json5().decode(
                Mistral3VLMConfiguration.self, from: Self.visionStripped(raw))
            #expect(cfg.visionConfig == nil)
            #expect(Mistral3VLM(cfg).modalities == [.text])
        }
        if let raw = Self.rawConfig(modelType: "gemma4") {
            let cfg = try JSONDecoder.json5().decode(
                Gemma4Configuration.self, from: Self.visionStripped(raw))
            #expect(!Gemma4.constructibleModalities(of: cfg).contains(.vision))
        }
        if let raw = Self.rawConfig(modelType: "qwen3_5") {
            let cfg = try JSONDecoder.json5().decode(
                MLXVLM.Qwen35Configuration.self, from: Self.visionStripped(raw))
            #expect(cfg.visionConfiguration == nil)
            #expect(Qwen35.constructibleModalities(of: cfg) == [.text])
        }
    }

    // MARK: the subset rule holds everywhere

    @Test("asking a vision-stripped config for vision throws, in every family")
    func overRequestThrowsEverywhere() throws {
        if let raw = Self.rawConfig(modelType: "gemma3") {
            let cfg = try JSONDecoder.json5().decode(
                Gemma3Configuration.self, from: Self.visionStripped(raw))
            #expect(throws: UnconstructibleModalities.self) {
                _ = try Gemma3(cfg, requesting: [.text, .vision])
            }
        }
        if let raw = Self.rawConfig(modelType: "mistral3") {
            let cfg = try JSONDecoder.json5().decode(
                Mistral3VLMConfiguration.self, from: Self.visionStripped(raw))
            #expect(throws: UnconstructibleModalities.self) {
                _ = try Mistral3VLM(cfg, requesting: [.text, .vision])
            }
        }
        if let raw = Self.rawConfig(modelType: "qwen3_5") {
            let cfg = try JSONDecoder.json5().decode(
                MLXVLM.Qwen35Configuration.self, from: Self.visionStripped(raw))
            #expect(throws: UnconstructibleModalities.self) {
                _ = try Qwen35(cfg, requesting: [.text, .vision])
            }
        }
    }

    /// A caller may narrow, which is the whole point — and the narrowed instance must SAY so.
    @Test("a narrowed instance reports the narrower set, in every family")
    func narrowingIsReportedEverywhere() throws {
        if let raw = Self.rawConfig(modelType: "mistral3") {
            let cfg = try JSONDecoder.json5().decode(
                Mistral3VLMConfiguration.self, from: Self.asData(raw))
            #expect(try Mistral3VLM(cfg, requesting: [.text]).modalities == [.text])
            #expect(Mistral3VLM(cfg).modalities == [.text, .vision])
        }
        if let raw = Self.rawConfig(modelType: "gemma3") {
            let cfg = try JSONDecoder.json5().decode(
                Gemma3Configuration.self, from: Self.asData(raw))
            #expect(try Gemma3(cfg, requesting: [.text]).modalities == [.text])
        }
    }

    /// Declaring a narrower set and BUILDING a narrower model are different claims, and only the
    /// second one saves anything. A mutation that built the tower regardless of the request passed
    /// every other test in this suite, which is exactly why this one exists: assert the parameters,
    /// not the declaration.
    @Test("a narrowed instance allocates no vision parameters, in every family")
    func narrowingActuallySkipsTheTower() throws {
        func visionKeys(_ m: Module) -> [String] {
            m.parameters().flattened().map(\.0).filter {
                $0.contains("vision_tower") || $0.contains("vision_model")
                    || $0.contains("multi_modal_projector") || $0.contains("vision_embedder")
            }
        }

        if let raw = Self.rawConfig(modelType: "gemma3") {
            let cfg = try JSONDecoder.json5().decode(
                Gemma3Configuration.self, from: Self.asData(raw))
            #expect(!visionKeys(Gemma3(cfg)).isEmpty, "the full load should carry vision")
            #expect(visionKeys(try Gemma3(cfg, requesting: [.text])).isEmpty)
        }
        if let raw = Self.rawConfig(modelType: "mistral3") {
            let cfg = try JSONDecoder.json5().decode(
                Mistral3VLMConfiguration.self, from: Self.asData(raw))
            #expect(!visionKeys(Mistral3VLM(cfg)).isEmpty, "the full load should carry vision")
            #expect(visionKeys(try Mistral3VLM(cfg, requesting: [.text])).isEmpty)
        }
        if let raw = Self.rawConfig(modelType: "gemma4") {
            let cfg = try JSONDecoder.json5().decode(
                Gemma4Configuration.self, from: Self.asData(raw))
            #expect(!visionKeys(Gemma4(cfg)).isEmpty, "the full load should carry vision")
            #expect(visionKeys(try Gemma4(cfg, requesting: [.text])).isEmpty)
        }
        if let raw = Self.rawConfig(modelType: "qwen3_5") {
            let cfg = try JSONDecoder.json5().decode(
                MLXVLM.Qwen35Configuration.self, from: Self.asData(raw))
            #expect(!visionKeys(Qwen35(cfg)).isEmpty, "the full load should carry vision")
            #expect(visionKeys(try Qwen35(cfg, requesting: [.text])).isEmpty)
        }
    }

    /// diffusion_gemma was already closest to the target: its VLM creator returns the SAME type the
    /// LLM factory registers, gaining a vision lane only when a tower is installed. So its modality
    /// set is a `var` — and it still refuses an impossible request.
    @Test("diffusion_gemma declares and refuses like the rest")
    func diffusionGemmaFollowsTheConvention() throws {
        guard let raw = Self.rawConfig(modelType: "diffusion_gemma") else { return }
        let full = try JSONDecoder.json5().decode(
            DiffusionGemmaVLMConfiguration.self, from: Self.asData(raw))
        #expect(diffusionGemmaConstructibleModalities(of: full).contains(.vision))
        #expect(makeDiffusionGemmaVLM(full).modalities.contains(.vision))

        let stripped = try JSONDecoder.json5().decode(
            DiffusionGemmaVLMConfiguration.self, from: Self.visionStripped(raw))
        #expect(diffusionGemmaConstructibleModalities(of: stripped) == [.text])
        #expect(makeDiffusionGemmaVLM(stripped).modalities == [.text])
        #expect(throws: UnconstructibleModalities.self) {
            _ = try makeDiffusionGemmaVLM(stripped, requesting: [.text, .vision])
        }
    }
}
