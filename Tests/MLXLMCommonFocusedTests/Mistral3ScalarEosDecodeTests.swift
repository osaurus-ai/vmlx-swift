// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// `eos_token_id` is spelled both ways in the wild. Bundles that use the scalar form were
// undecodable — not degraded, undecodable: the whole configuration failed, so the model could not
// load at all. Found by running a real bundle rather than by reading the type.

import Foundation
import MLXLMCommon
import MLXVLM
import Testing

@Suite("Mistral3 eos_token_id accepts either spelling")
struct Mistral3ScalarEosDecodeTests {

    private static func config(eos: String) -> String {
        """
        {"model_type":"mistral3", \(eos)
         "text_config":{"model_type":"mistral","hidden_size":128,"num_hidden_layers":2,
           "intermediate_size":256,"num_attention_heads":4,"num_key_value_heads":2,
           "rms_norm_eps":1e-6,"vocab_size":1000,"rope_theta":10000.0,"head_dim":32,
           "max_position_embeddings":4096,"sliding_window":4096},
         "vision_config":{"model_type":"pixtral","hidden_size":64,"num_hidden_layers":2,
           "num_attention_heads":4,"intermediate_size":128,"patch_size":14,"image_size":224}}
        """
    }

    private func decode(_ json: String) throws -> Mistral3VLMConfiguration {
        try JSONDecoder.json5().decode(Mistral3VLMConfiguration.self, from: Data(json.utf8))
    }

    /// The shape that used to fail. Devstral-Small-2-24B and Mistral-Small-4-119B-2603 both ship it.
    @Test("a scalar eos_token_id decodes, and normalises to a one-element list")
    func scalarDecodes() throws {
        let cfg = try decode(Self.config(eos: #""eos_token_id": 2,"#))
        #expect(cfg.eosTokenId == [2])
    }

    @Test("an array eos_token_id still decodes unchanged")
    func arrayStillDecodes() throws {
        let cfg = try decode(Self.config(eos: #""eos_token_id": [2, 7],"#))
        #expect(cfg.eosTokenId == [2, 7])
    }

    @Test("an absent eos_token_id stays nil rather than becoming empty")
    func absentStaysNil() throws {
        let cfg = try decode(Self.config(eos: ""))
        #expect(cfg.eosTokenId == nil)
    }

    /// The real bundle that motivated this, if present — the test that would have caught it.
    @Test("every local mistral3 bundle's config decodes")
    func realBundlesDecode() throws {
        let root = URL(fileURLWithPath: NSHomeDirectory()).appendingPathComponent("Library/MLModels")
        let fm = FileManager.default
        guard let orgs = try? fm.contentsOfDirectory(atPath: root.path) else { return }
        var checked = 0
        var skipped = 0
        for org in orgs.sorted() {
            let orgDir = root.appendingPathComponent(org)
            guard let kids = try? fm.contentsOfDirectory(atPath: orgDir.path) else { continue }
            for kid in kids.sorted() {
                let url = orgDir.appendingPathComponent(kid).appendingPathComponent("config.json")
                guard let data = try? Data(contentsOf: url),
                    let obj = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any],
                    obj["model_type"] as? String == "mistral3",
                    (obj["text_config"] as? [String: Any])?["model_type"] as? String != "mistral4"
                else { continue }
                do {
                    _ = try JSONDecoder.json5().decode(Mistral3VLMConfiguration.self, from: data)
                    checked += 1
                } catch let e as DecodingError {
                    // A TEXT-ONLY mistral3 bundle (Ministral-3-3B/8B) ships no `vision_config`, and
                    // `visionConfig` is not optional here — so it cannot decode, for a reason that has
                    // nothing to do with `eos_token_id`. Failing on it would make this test assert
                    // something it does not claim. Anything else IS this test's business.
                    guard case .keyNotFound(let key, _) = e, key.stringValue == "vision_config" else {
                        throw e
                    }
                    skipped += 1
                }
            }
        }
        print("mistral3 bundle configs decoded: \(checked), text-only skipped: \(skipped)")
    }
}
