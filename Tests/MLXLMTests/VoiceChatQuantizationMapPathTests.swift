import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLMCommon
@testable import MLXVLM

/// The quantization map in a published bundle is keyed by BUNDLE tensor paths,
/// but the Swift module tree renames several of them. If the loader's remap
/// ever drifts from the sanitizers', every per-module recipe misses and each
/// module is quantized with the GLOBAL default instead of its own — which on a
/// mixed-bit JANG bundle is a hard shape error at the first matmul
/// (`w.shape == (4480, 2450)` vs `scales.shape == (4480, 490)`), and on a
/// uniform bundle would be silent quality loss.
///
/// This suite pins the two together against the real bundles: every module the
/// map names must resolve to a module that actually exists in the model.
public class VoiceChatQuantizationMapPathTests: XCTestCase {

    private static var bundles: [(name: String, path: String)] {
        let home = FileManager.default.homeDirectoryForCurrentUser.path
        return [
            ("MXFP8", "\(home)/models/JANGQ-AI/NemotronLabs-VoiceChat-11B-MXFP8"),
            ("JANG_4", "\(home)/models/OsaurusAI/NemotronLabs-VoiceChat-11B-JANG_4"),
            ("JANG_2", "\(home)/models/OsaurusAI/NemotronLabs-VoiceChat-11B-JANG_2"),
        ]
    }

    /// Remapping is a pure string transform — checkable without any bundle.
    func testKnownRenamesAreApplied() {
        let map = VoiceChatQuantizationMap.modulePath(forBundlePath:)
        XCTAssertEqual(
            map("stt_model.llm.layers.17.mixer.down_proj"),
            "stt_model.llm.backbone.layers.17.mixer.down_proj")
        XCTAssertEqual(map("stt_model.lm_head"), "stt_model.llm.lm_head")
        XCTAssertEqual(map("stt_model.embed_tokens"), "stt_model.llm.backbone.embeddings")
        XCTAssertEqual(
            map("stt_model.rnnt_joint.joint_net.2"), "stt_model.rnnt_joint.joint_net.out")
        // Paths that must pass through untouched.
        XCTAssertEqual(
            map("stt_model.perception.encoder.layers.0.self_attn.linear_q"),
            "stt_model.perception.encoder.layers.0.self_attn.linear_q")
        XCTAssertEqual(
            map("tts_model.tts_model.backbone.layers.3.mlp.up_proj"),
            "tts_model.tts_model.backbone.layers.3.mlp.up_proj")
    }

    /// Every quantized module named by a real bundle must exist in the model
    /// tree under its remapped path — this is the check that would have caught
    /// the shape crash before it happened.
    func testEveryMappedModuleResolvesInTheModelTree() throws {
        var checked = 0
        for (name, path) in Self.bundles {
            let dir = URL(fileURLWithPath: path)
            let configURL = dir.appendingPathComponent("config.json")
            guard FileManager.default.fileExists(atPath: configURL.path) else { continue }

            let data = try Data(contentsOf: configURL)
            let config = try JSONDecoder().decode(
                NemotronVoiceChatConfiguration.self, from: data)
            let root = try JSONSerialization.jsonObject(with: data) as? [String: Any]
            guard let map = VoiceChatQuantizationMap(root?["quantization"] as? [String: Any])
            else {
                XCTFail("\(name): bundle has no per-module quantization map")
                continue
            }

            let model = NemotronVoiceChatModel(config)
            let modules = Set(model.leafModules().flattened().map(\.0))
            XCTAssertFalse(modules.isEmpty)

            var unresolved = [String]()
            for (bundlePath, _) in map.perModule {
                let modulePath = VoiceChatQuantizationMap.modulePath(forBundlePath: bundlePath)
                if !modules.contains(modulePath) { unresolved.append(bundlePath) }
            }
            XCTAssertTrue(
                unresolved.isEmpty,
                "\(name): \(unresolved.count) mapped module(s) do not resolve, e.g. "
                    + unresolved.sorted().prefix(3).joined(separator: ", "))
            checked += 1

            // Sanity: the map really is per-module and mixed, not a single
            // global recipe wearing a hat.
            XCTAssertGreaterThan(map.perModule.count, 100, "\(name): map looks truncated")
        }
        try XCTSkipIf(checked == 0, "no VoiceChat quant bundles present")
    }
}
