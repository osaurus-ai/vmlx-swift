// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// `Mistral3TextModel` is hard to reach with a published bundle: it needs an outer `model_type` of
// mistral3/ministral3 WITHOUT a `vision_config`, and in practice every bundle declaring that outer
// type ships a vision tower — `Mistral3ForConditionalGeneration` IS the multimodal wrapper. So the
// class is exercised here the way the code path itself would be: take a real multimodal bundle,
// remove its vision section (what a text-only conversion of it would look like), and run the LLM
// dispatch and sanitize over the bundle's OWN weight-map keys.
//
// This is the test that would otherwise wait on a bundle that may never be published.

import Foundation
import MLX
@testable import MLXLLM
@testable import MLXLMCommon
import XCTest

final class Mistral3TextRealBundleSanitizeTests: XCTestCase {

    private struct Candidate {
        let name: String
        let config: [String: Any]
        let keys: [String]
        /// `tie_word_embeddings` as the DECODER resolves it — `text_config` first. See the note
        /// at the assignment: one real bundle contradicts itself between the two levels.
        let tiesEmbeddings: Bool
    }

    /// Bundles that would land on `Mistral3TextModel` once vision is removed. Excluded: a
    /// `text_config.model_type` of `mistral4` (routes to Mistral4Model) and `weight_format: mxtq`
    /// (routes to Mistral3TextJANGTQModel). Absent on a machine without these models, so the tests
    /// skip rather than fail.
    private func candidates() -> [Candidate] {
        let root = URL(fileURLWithPath: NSHomeDirectory()).appendingPathComponent("Library/MLModels")
        let fm = FileManager.default
        guard let orgs = try? fm.contentsOfDirectory(atPath: root.path) else { return [] }
        var out: [Candidate] = []
        for org in orgs {
            let orgDir = root.appendingPathComponent(org)
            guard let kids = try? fm.contentsOfDirectory(atPath: orgDir.path) else { continue }
            for kid in kids {
                let dir = orgDir.appendingPathComponent(kid)
                let cfgURL = dir.appendingPathComponent("config.json")
                let idxURL = dir.appendingPathComponent("model.safetensors.index.json")
                guard fm.fileExists(atPath: cfgURL.path), fm.fileExists(atPath: idxURL.path),
                    let cfgData = try? Data(contentsOf: cfgURL),
                    let cfg = (try? JSONSerialization.jsonObject(with: cfgData)) as? [String: Any],
                    let outer = cfg["model_type"] as? String,
                    outer == "mistral3" || outer == "ministral3"
                else { continue }

                let textCfg = cfg["text_config"] as? [String: Any] ?? [:]
                if textCfg["model_type"] as? String == "mistral4" { continue }
                let wf = (cfg["weight_format"] ?? textCfg["weight_format"]) as? String
                if wf?.lowercased() == "mxtq" { continue }

                guard let idxData = try? Data(contentsOf: idxURL),
                    let idx = (try? JSONSerialization.jsonObject(with: idxData)) as? [String: Any],
                    let map = idx["weight_map"] as? [String: Any]
                else { continue }

                // Precedence is not cosmetic here. `Mistral3TextConfiguration.init(from:)` reads
                // EVERY field from `text_config` when the bundle has one, so the text decoder's
                // own flag wins over the outer VLM wrapper's. Magistral-Small-2509 declares the
                // two CONTRADICTORILY — outer `true`, `text_config` `false` — and the nested one
                // is the honest answer: the checkpoint ships a real `language_model.lm_head.weight`
                // that a tied model would not carry. Mirror the decoder, not the outer key.
                let tie = (textCfg["tie_word_embeddings"] as? Bool)
                    ?? (cfg["tie_word_embeddings"] as? Bool) ?? false
                out.append(
                    Candidate(name: kid, config: cfg, keys: Array(map.keys), tiesEmbeddings: tie))
            }
        }
        return out
    }

    /// The bundle as a text-only conversion of itself would look: same everything, no vision.
    private func visionStripped(_ cfg: [String: Any]) -> Data {
        var c = cfg
        c.removeValue(forKey: "vision_config")
        if var t = c["text_config"] as? [String: Any] {
            t.removeValue(forKey: "vision_config")
            c["text_config"] = t
        }
        return try! JSONSerialization.data(withJSONObject: c)
    }

    /// Step 1 of `dispatchMistral3LLM` throws `unsupportedModelType` while `vision_config` is
    /// present, so the LLM route is unreachable for these bundles as shipped. Removing it is what
    /// makes the vanilla text model the destination — not the JANGTQ variant, not Mistral4Model.
    func testStrippingVisionRoutesToMistral3TextModel() throws {
        let found = candidates()
        try XCTSkipIf(found.isEmpty, "no qualifying mistral3 bundle on this machine")
        for c in found {
            // A TEXT-ONLY mistral3 bundle (Ministral-3-3B/8B) ships no `vision_config`, so the gate
            // has nothing to refuse and the LLM route is reachable as shipped. Asserting the throw
            // for those would assert the gate fires on bundles it was never meant to see.
            if c.config["vision_config"] != nil {
                let asShipped = try! JSONSerialization.data(withJSONObject: c.config)
                XCTAssertThrowsError(
                    try LLMTypeRegistry.dispatchMistral3LLM(data: asShipped),
                    "\(c.name): the vision gate should refuse the bundle as shipped")
            }

            let model = try LLMTypeRegistry.dispatchMistral3LLM(data: visionStripped(c.config))
            XCTAssertTrue(
                model is Mistral3TextModel,
                "\(c.name): expected Mistral3TextModel, got \(type(of: model))")
        }
    }

    /// The claim patch 0035 rests on, checked against real tensor names rather than a simulation:
    /// `sanitize` returns a non-empty dictionary. It also pins WHAT survives — the language tower,
    /// with its `language_model.` prefix stripped, and nothing from the vision side, which the
    /// unwrap discards implicitly.
    func testSanitizeOverRealKeysKeepsExactlyTheLanguageTower() throws {
        let found = candidates()
        try XCTSkipIf(found.isEmpty, "no qualifying mistral3 bundle on this machine")
        for c in found {
            let model = try LLMTypeRegistry.dispatchMistral3LLM(data: visionStripped(c.config))
            let text = try XCTUnwrap(model as? Mistral3TextModel, "\(c.name)")

            // One element each: sanitize is a function of the KEYS, and this keeps the test cheap.
            var weights: [String: MLXArray] = [:]
            for k in c.keys { weights[k] = MLXArray([0.0] as [Float]) }

            let out = text.sanitize(weights: weights)

            // 0035: the empty-result fallback that used to sit here could never fire.
            XCTAssertFalse(out.isEmpty, "\(c.name): sanitize returned nothing for a real bundle")

            let lm = c.keys.filter { $0.hasPrefix("language_model.") }
            XCTAssertFalse(lm.isEmpty, "\(c.name): expected a language_model subtree")
            // Tying only costs a key when the checkpoint SHIPS one. Ministral-3-3B/8B tie and
            // ship no `lm_head` at all, so nothing is dropped and the count is unchanged — the
            // same condition the tie test below already guards on. Deriving the delta from the
            // keys rather than from the flag alone is what makes this hold for both shapes.
            let dropsHead = c.tiesEmbeddings && lm.contains { $0.hasSuffix(".lm_head.weight") }
            let expected = dropsHead ? lm.count - 1 : lm.count
            XCTAssertEqual(out.count, expected, "\(c.name)")

            for k in out.keys {
                XCTAssertFalse(k.hasPrefix("language_model."), "\(c.name): prefix not stripped: \(k)")
                XCTAssertFalse(k.hasPrefix("vision_tower."), "\(c.name): vision weight survived: \(k)")
                XCTAssertFalse(
                    k.hasPrefix("multi_modal_projector."), "\(c.name): projector survived: \(k)")
            }
        }
    }

    /// The NEGATIVE direction, on real bundles. Every bundle that actually SHIPS an `lm_head`
    /// resolves to untied once `text_config` wins, so this pins that the tensor survives when it
    /// should. The bundles that do resolve to tied ship no `lm_head` to drop, which is why the
    /// positive direction still needs a forced flag — see the test below.
    func testTiedEmbeddingsDropMatchesTheBundlesDeclaration() throws {
        let found = candidates()
        try XCTSkipIf(found.isEmpty, "no qualifying mistral3 bundle on this machine")
        for c in found {
            // Only meaningful where the checkpoint actually ships the tensor.
            guard c.keys.contains("language_model.lm_head.weight") else { continue }
            let model = try LLMTypeRegistry.dispatchMistral3LLM(data: visionStripped(c.config))
            let text = try XCTUnwrap(model as? Mistral3TextModel, "\(c.name)")

            var weights: [String: MLXArray] = [:]
            for k in c.keys { weights[k] = MLXArray([0.0] as [Float]) }
            let out = text.sanitize(weights: weights)

            if c.tiesEmbeddings {
                XCTAssertNil(
                    out["lm_head.weight"],
                    "\(c.name): ties embeddings, so lm_head.weight must be dropped")
            } else {
                XCTAssertNotNil(
                    out["lm_head.weight"],
                    "\(c.name): does not tie embeddings, so lm_head.weight must survive")
            }
        }
    }

    /// The POSITIVE direction. No local bundle exercises the drop: the ones that resolve to tied
    /// (Ministral-3-3B/8B) carry no `lm_head` for it to remove, and the ones that carry one
    /// resolve to untied. So the branch would otherwise go unexercised. Force the flag on an
    /// otherwise real config and keep the real key set, so it is tested against the tensor names
    /// it actually has to handle.
    func testForcedTieDropsLmHeadOverRealKeys() throws {
        let found = candidates()
        try XCTSkipIf(found.isEmpty, "no qualifying mistral3 bundle on this machine")
        for c in found {
            guard c.keys.contains("language_model.lm_head.weight") else { continue }

            var cfg = c.config
            cfg.removeValue(forKey: "vision_config")
            if var t = cfg["text_config"] as? [String: Any] {
                t["tie_word_embeddings"] = true
                t.removeValue(forKey: "vision_config")
                cfg["text_config"] = t
            } else {
                cfg["tie_word_embeddings"] = true
            }
            let data = try JSONSerialization.data(withJSONObject: cfg)
            let model = try LLMTypeRegistry.dispatchMistral3LLM(data: data)
            let text = try XCTUnwrap(model as? Mistral3TextModel, "\(c.name)")

            var weights: [String: MLXArray] = [:]
            for k in c.keys { weights[k] = MLXArray([0.0] as [Float]) }
            let out = text.sanitize(weights: weights)

            XCTAssertNil(
                out["lm_head.weight"],
                "\(c.name): with tie_word_embeddings forced on, lm_head.weight must be dropped")
            // And nothing else went missing with it.
            let lm = c.keys.filter { $0.hasPrefix("language_model.") }
            XCTAssertEqual(out.count, lm.count - 1, "\(c.name)")
        }
    }
}
