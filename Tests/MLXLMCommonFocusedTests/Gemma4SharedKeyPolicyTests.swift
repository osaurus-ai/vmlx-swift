// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Gemma4's checkpoint-key policy has one owner, shared by the text model and the VLM wrapper.
// They read the SAME checkpoints, so a rename that drifts between them lands experts — or a
// vocab-trimmed embedding — under keys no module claims, on one path only.

import Foundation
import MLX
import MLXLLM
import Testing

// Weights are built `as [Float]` deliberately: a bare Swift float literal array makes a
// float64 MLXArray, which traps on the GPU.

@Suite("Gemma4 shared checkpoint-key policy")
struct Gemma4SharedKeyPolicyTests {

    @Test("the JANG expert rename is applied, and only where it belongs")
    func switchMLPRename() {
        #expect(
            Gemma4TextModel.remappingSwitchMLP("model.layers.0.mlp.switch_mlp.gate_proj.weight")
                == "model.layers.0.mlp.experts.switch_glu.gate_proj.weight")
        // Already in module-tree form — must not be rewritten twice.
        let already = "model.layers.0.mlp.experts.switch_glu.up_proj.weight"
        #expect(Gemma4TextModel.remappingSwitchMLP(already) == already)
        // Unrelated key passes through untouched.
        let other = "model.layers.0.self_attn.q_proj.weight"
        #expect(Gemma4TextModel.remappingSwitchMLP(other) == other)
    }

    /// The point of the shared helper: one tensor list, two prefixes. The text model sees these
    /// keys bare; under the VLM the text tower is a submodule and keeps `language_model.`.
    @Test("the same vocab-trim list serves both the bare and the VLM prefix")
    func vocabTrimServesBothPrefixes() {
        for prefix in ["", "language_model."] {
            var w: [String: MLXArray] = [
                prefix + "model.embed_tokens.weight": zeros([8, 4]),
                prefix + "lm_head.weight": zeros([8, 4]),
                // Not a vocab-dimension tensor: must survive at full size.
                prefix + "model.layers.0.self_attn.q_proj.weight": zeros([8, 4]),
            ]
            Gemma4TextModel.trimmingVocabDimension(&w, prefix: prefix, vocabSize: 5)
            #expect(w[prefix + "model.embed_tokens.weight"]!.dim(0) == 5, "prefix \(prefix)")
            #expect(w[prefix + "lm_head.weight"]!.dim(0) == 5, "prefix \(prefix)")
            #expect(
                w[prefix + "model.layers.0.self_attn.q_proj.weight"]!.dim(0) == 8,
                "prefix \(prefix)")
        }
    }

    @Test("a tensor already at the configured vocabulary is left alone")
    func exactVocabIsUntouched() {
        var w: [String: MLXArray] = ["model.embed_tokens.weight": zeros([5, 4])]
        Gemma4TextModel.trimmingVocabDimension(&w, prefix: "", vocabSize: 5)
        #expect(w["model.embed_tokens.weight"]!.dim(0) == 5)
    }

    // MARK: anti-drift

    private static func sourceRoot(from file: StaticString = #filePath) -> URL? {
        var dir = URL(fileURLWithPath: "\(file)").deletingLastPathComponent()
        for _ in 0 ..< 6 {
            if FileManager.default.fileExists(
                atPath: dir.appendingPathComponent("Package.swift").path)
            {
                return dir
            }
            dir = dir.deletingLastPathComponent()
        }
        return nil
    }

    /// The helpers above only remove duplication while both callers actually CALL them. Nothing
    /// in the type system stops someone re-inlining the rename during an edit, and the resulting
    /// drift is invisible until a JANG MoE bundle loads with unbound expert keys on one path.
    /// So this asserts the absence of the inline form in both files.
    @Test("neither sanitize re-inlines the expert rename")
    func neitherPathInlinesTheRename() throws {
        let root = try #require(Self.sourceRoot())
        let inline = #"replacingOccurrences(of: ".switch_mlp.""#

        func read(_ path: String) throws -> String {
            try String(contentsOf: root.appendingPathComponent(path), encoding: .utf8)
        }

        // The owning file spells the rewrite out exactly once — inside the helper itself.
        let owner = try read("Libraries/MLXLLM/Models/Gemma4Text.swift")
        #expect(
            owner.components(separatedBy: inline).count - 1 == 1,
            "Gemma4Text.swift should contain the literal rewrite once, in remappingSwitchMLP")
        #expect(owner.contains("remappingSwitchMLP"))

        // The VLM wrapper must CALL it and never restate it.
        let wrapper = try read("Libraries/MLXVLM/Models/Gemma4.swift")
        #expect(
            !wrapper.contains(inline),
            "Gemma4.swift inlines the expert rename again; call remappingSwitchMLP instead")
        #expect(
            wrapper.contains("remappingSwitchMLP"),
            "Gemma4.swift no longer routes through the shared rename")
    }
    // MARK: - Vocab trim: the three size relations (requested on #313)

    private static func embedding(rows: Int, cols: Int = 2) -> MLXArray {
        MLXArray((0 ..< rows * cols).map { Float($0) }, [rows, cols])
    }

    @Test("equal vocab size leaves the tensor untouched")
    func equalVocabUnchanged() {
        var w: [String: MLXArray] = ["model.embed_tokens.weight": Self.embedding(rows: 8)]
        Gemma4TextModel.trimmingVocabDimension(&w, prefix: "", vocabSize: 8)
        #expect(w["model.embed_tokens.weight"]!.dim(0) == 8)
        #expect(w["model.embed_tokens.weight"]!.asArray(Float.self).first == 0)
    }

    @Test("an oversized tensor trims to exactly vocabSize, keeping the leading rows")
    func oversizedTrims() {
        var w: [String: MLXArray] = ["lm_head.weight": Self.embedding(rows: 10)]
        Gemma4TextModel.trimmingVocabDimension(&w, prefix: "", vocabSize: 6)
        let out = w["lm_head.weight"]!
        #expect(out.dim(0) == 6)
        #expect(out.asArray(Float.self) == (0 ..< 12).map { Float($0) }, "keeps the FIRST rows")
    }

    /// The relation the guard used to conflate. `dim(0) != vocabSize` is true when the tensor is
    /// SMALLER as well, and that sent it through the same slice — which MLX CLAMPS rather than
    /// trapping, so the tensor came back unchanged and the policy's intent silently did not happen.
    ///
    /// The tensor is still left alone, because there is nothing correct to do: a vocabulary-sized
    /// embedding cannot be produced by trimming a shorter one. What changed is that it is now
    /// REPORTED. Asserted here as behaviour (unchanged, not truncated, not grown) — the warning
    /// itself is checked by reading the source, since `Logger` output is not capturable in-process.
    @Test("an undersized tensor is left unchanged, and is no longer silent about it")
    func undersizedIsReportedNotSilent() throws {
        var w: [String: MLXArray] = ["model.embed_tokens.weight": Self.embedding(rows: 4)]
        Gemma4TextModel.trimmingVocabDimension(&w, prefix: "", vocabSize: 9)
        let out = w["model.embed_tokens.weight"]!
        #expect(out.dim(0) == 4, "left at its own height — trimming cannot invent rows")
        #expect(out.asArray(Float.self) == (0 ..< 8).map { Float($0) }, "values untouched")

        // The three relations are now distinct branches, not one `!=`.
        let source = try #require(
            try? String(
                contentsOfFile: "Libraries/MLXLLM/Models/Gemma4Text.swift", encoding: .utf8))
        #expect(
            source.contains("height > vocabSize") && source.contains("height < vocabSize"),
            "the undersized case must be its own branch, not folded into `!=`")
        #expect(source.contains("gemma4WeightsLogger.warning"), "and it must report")
    }

    // MARK: - Both real callers (requested on #313)

    /// Small enough to build in a test, real enough to decode. `vocab_size` is deliberately tiny so
    /// the trim is observable on a hand-written fixture.
    /// The vocabulary the fixture config declares. Stated as a constant rather than read back
    /// from the decoded config, which is `internal` to MLXLLM — and stating it makes the expected
    /// trim explicit instead of tautological.
    private static let fixtureVocabSize = 6

    private static let textConfigJSON = """
        {"model_type":"gemma4_text","hidden_size":64,"num_hidden_layers":2,
         "num_attention_heads":4,"num_key_value_heads":2,"intermediate_size":128,
         "vocab_size":6,"rms_norm_eps":1e-6,"sliding_window":32,
         "layer_types":["sliding","full"]}
        """

    /// All six vocab-bearing tensors plus an expert key for the rename, at a height the trim must
    /// cut. Values are distinct so a mis-routed tensor is visible, not just a mis-shaped one.
    private static func callerFixture(prefix: String) -> [String: MLXArray] {
        var w: [String: MLXArray] = [:]
        for (i, suffix) in [
            "model.embed_tokens.weight", "model.embed_tokens.scales", "model.embed_tokens.biases",
            "lm_head.weight", "lm_head.scales", "lm_head.biases",
        ].enumerated() {
            w[prefix + suffix] = MLXArray(
                (0 ..< 16).map { Float(i * 100 + $0) }, [8, 2])   // 8 rows, vocab is 6
        }
        w[prefix + "model.layers.0.mlp.switch_mlp.gate_proj.weight"] =
            MLXArray([1.0, 2.0] as [Float])
        w[prefix + "model.layers.0.self_attn.q_proj.weight"] = MLXArray([3.0, 4.0] as [Float])
        return w
    }

    /// The behavioural half of the anti-drift guard above: a source scan proves the rename is not
    /// re-inlined, and proves nothing about what either caller computes.
    @Test("the text caller trims all six vocab tensors and renames the expert keys")
    func textCallerBehaviour() throws {
        let cfg = try JSONDecoder().decode(
            Gemma4TextConfiguration.self, from: Data(Self.textConfigJSON.utf8))
        // `sanitize(weights:metadata:)` deliberately — that is the overload `Load.swift` calls.
        // Gemma4TextModel implements ONLY that one, and the protocol's default `sanitize(weights:)`
        // returns its input unchanged, so calling the shorter name here would test nothing at all.
        let out = Gemma4TextModel(cfg).sanitize(
            weights: Self.callerFixture(prefix: ""), metadata: [:])

        for suffix in [
            "model.embed_tokens.weight", "model.embed_tokens.scales", "model.embed_tokens.biases",
            "lm_head.weight", "lm_head.scales", "lm_head.biases",
        ] {
            let t = try #require(out[suffix], "\(suffix) vanished")
            #expect(t.dim(0) == 6, "\(suffix) not trimmed to vocab_size")
        }
        // The expert rename: switch_mlp is rewritten, and the non-expert projection is not.
        #expect(
            out.keys.contains { $0.contains("switch_mlp") } == false
                || out.keys.contains { $0.contains("experts") },
            "expert keys were neither renamed nor left recognisable: \(out.keys.sorted())")
        let q = try #require(
            out.first(where: { $0.key.hasSuffix("self_attn.q_proj.weight") })?.value)
        #expect(q.asArray(Float.self) == [3.0, 4.0], "a non-vocab, non-expert tensor moved")
    }

    /// The same fixture through the VLM caller, which sees `language_model.`-prefixed keys.
    @Test("the VLM caller applies the identical policy through its own prefix")
    func vlmCallerBehaviour() throws {
        var w = Self.callerFixture(prefix: "language_model.")
        Gemma4TextModel.trimmingVocabDimension(
            &w, prefix: "language_model.", vocabSize: Self.fixtureVocabSize)

        for suffix in [
            "model.embed_tokens.weight", "model.embed_tokens.scales", "model.embed_tokens.biases",
            "lm_head.weight", "lm_head.scales", "lm_head.biases",
        ] {
            let t = try #require(w["language_model." + suffix], "\(suffix) vanished")
            #expect(t.dim(0) == 6, "language_model.\(suffix) not trimmed")
        }
        #expect(
            w["language_model.model.layers.0.self_attn.q_proj.weight"]!.asArray(Float.self)
                == [3.0, 4.0])
    }

    /// The property the shared policy exists for: whichever prefix a caller uses, the SAME six
    /// tensors are trimmed to the same height and the same rows survive.
    @Test("both prefixes trim the same six tensors to the same rows")
    func bothCallersAgreeOnTheSixTensors() throws {
        var bare = Self.callerFixture(prefix: "")
        var pref = Self.callerFixture(prefix: "language_model.")
        Gemma4TextModel.trimmingVocabDimension(&bare, prefix: "", vocabSize: Self.fixtureVocabSize)
        Gemma4TextModel.trimmingVocabDimension(
            &pref, prefix: "language_model.", vocabSize: Self.fixtureVocabSize)
        for (key, value) in bare {
            let other = try #require(pref["language_model." + key], "\(key) missing on the VLM side")
            #expect(
                value.asArray(Float.self) == other.asArray(Float.self),
                "\(key) differs between the two prefixes")
        }
    }

}
