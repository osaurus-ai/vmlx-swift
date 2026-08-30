// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Apertus 1.5 routes its whole language tower through one prefix normalisation: a 70B quantisation
// carries 2,084 body tensors that way. The rule is `Weights.stripLanguageModelPrefix`, and the
// reason to call it rather than re-derive it is not brevity — the helper resolves a checkpoint that
// carries BOTH spellings of a key to a FIXED winner, where a local `dict[key] = value` loop lets
// Dictionary iteration order decide. That order is seeded per process, so the same bundle can bind
// a different tensor from run to run: it loads, it generates, and it is wrong some fraction of the
// time. Nothing downstream reports that, so the guard is here.

import Foundation
import MLX
import MLXLMCommon
import Testing

@Suite("Apertus 1.5 prefix normalisation")
struct Apertus1p5PrefixNormalisationTests {

    private static let source: String = {
        (try? String(
            contentsOfFile: "Libraries/MLXVLM/Models/Apertus1p5.swift", encoding: .utf8)) ?? ""
    }()

    @Test("sanitize delegates to the shared helper")
    func delegatesToSharedHelper() throws {
        try #require(!Self.source.isEmpty, "source not found — the path is wrong, not the code")
        #expect(Self.source.contains("Weights.stripLanguageModelPrefix("))
    }

    /// The specific shape that must not come back: normalising inside the routing loop, which both
    /// duplicates the rule and reintroduces the order-dependent bind.
    @Test("the prefix rule is not re-derived locally")
    func ruleIsNotReDerived() throws {
        try #require(!Self.source.isEmpty)
        let reDerived = [
            #"key = "model." + key.dropFirst("model.language_model.".count)"#,
            #"key = String(key.dropFirst("language_model.".count))"#,
        ]
        for fragment in reDerived {
            #expect(!Self.source.contains(fragment), "re-derived: \(fragment)")
        }
    }

    /// The behaviour the delegation buys, asserted on the helper itself: both layouts land on the
    /// same destination, and a checkpoint carrying both binds the unprefixed key REGARDLESS of
    /// insertion order. Built twice in opposite orders, because that is the only thing a local loop
    /// would get wrong.
    @Test("both layouts normalise, and a contested key binds the same way either order")
    func contestedKeyIsOrderIndependent() {
        let hf = Weights.stripLanguageModelPrefix(["model.language_model.layers.0.w": MLXArray([1])])
        #expect(hf.keys.sorted() == ["model.layers.0.w"])

        let mlx = Weights.stripLanguageModelPrefix(["language_model.model.layers.0.w": MLXArray([1])])
        #expect(mlx.keys.sorted() == ["model.layers.0.w"])

        let unprefixed = MLXArray([7])
        let prefixed = MLXArray([9])
        var forward: [String: MLXArray] = [:]
        forward["model.layers.0.w"] = unprefixed
        forward["model.language_model.layers.0.w"] = prefixed
        var backward: [String: MLXArray] = [:]
        backward["model.language_model.layers.0.w"] = prefixed
        backward["model.layers.0.w"] = unprefixed

        let a = Weights.stripLanguageModelPrefix(forward)["model.layers.0.w"]
        let b = Weights.stripLanguageModelPrefix(backward)["model.layers.0.w"]
        #expect(a?.item(Int.self) == 7, "the unprefixed key must win")
        #expect(b?.item(Int.self) == 7, "…and win the same way in the other order")
    }
}
