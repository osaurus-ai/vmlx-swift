// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// A MoE expert's down projection must resolve its quantization from TENSOR GEOMETRY, not from the
// bundle's declaration — because on an ambiguous packed width the declaration is self-consistent
// even when it is wrong, so consulting it first cannot fail loudly.
//
// This is a source-ordering assertion. The behaviour it guards needs a 119B bundle to observe, and
// the failure it prevents is a hard crash inside gather_qmm rather than a bad number, so there is
// nothing subtler to assert against.

import Foundation
import Testing

@Suite("Expert down-projection quantization is derived before it is declared")
struct ExpertDownProjectionQuantOrderTests {

    private static func loaderSource(from file: StaticString = #filePath) throws -> String {
        var dir = URL(fileURLWithPath: "\(file)").deletingLastPathComponent()
        for _ in 0 ..< 6 {
            let candidate = dir.appendingPathComponent("Libraries/MLXLMCommon/JangLoader.swift")
            if FileManager.default.fileExists(atPath: candidate.path) {
                return try String(contentsOf: candidate, encoding: .utf8)
            }
            dir = dir.deletingLastPathComponent()
        }
        throw CocoaError(.fileNoSuchFile)
    }

    /// The whole bug, in one assertion.
    ///
    /// `packedDim = 256` with `numGroups = 32` satisfies BOTH `(bits=4, gs=64)` and
    /// `(bits=8, gs=32)`. Mistral-Small-4-119B declares 8-bit and ships 4-bit tensors. While the
    /// declaration was consulted first, it won — and `gather_qmm` died on a 1024-wide matrix fed a
    /// 2048-wide input. The sibling-derived intermediate width settles it correctly.
    @Test("the expert-intermediate hint is consulted before the declared per-layer option")
    func hintPrecedesDeclaration() throws {
        let src = try Self.loaderSource()
        guard let hint = src.range(
            of: "} else if isExpertDownProjection,\n                      let expertIntermediateSize")
        else {
            Issue.record("the expert down-projection hint branch is gone"); return
        }
        guard let declaration = src.range(of: "} else if let dq = explicitForLayer,") else {
            Issue.record("the declared per-layer fallback is gone"); return
        }
        // Below the declaration, a self-consistent-but-wrong declaration wins on every
        // ambiguous packed width — which is how the 8-bit stamp beat 4-bit tensors.
        #expect(hint.lowerBound < declaration.lowerBound)
    }

    /// Both other tensor-derived hints already sit above the declaration. The expert hint was the
    /// odd one out, and this pins the general rule rather than the single case.
    @Test("every tensor-derived hint precedes the declared fallback")
    func allDerivedHintsPrecedeDeclaration() throws {
        let src = try Self.loaderSource()
        guard let declaration = src.range(of: "} else if let dq = explicitForLayer,") else {
            Issue.record("the declared per-layer fallback is gone"); return
        }
        for derived in [
            "} else if isLanguageHiddenAnchor,",
            "} else if isHiddenInputProjection,",
            "} else if isExpertDownProjection,",
        ] {
            guard let r = src.range(of: derived) else {
                Issue.record("missing branch: \(derived)"); continue
            }
            #expect(r.lowerBound < declaration.lowerBound, "\(derived)")
        }
    }
}
