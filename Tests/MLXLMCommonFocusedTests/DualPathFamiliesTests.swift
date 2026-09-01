// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// The two model registries, checked against each other.
//
// A family that ships both a text-only and a multimodal implementation is registered twice — once in
// `LLMModelFactory`, once in `VLMModelFactory` — in files that share no symbol and never mention each
// other. Nothing links the two lines, so a change to one is silently not made to the other, and the
// result is a bundle that looks registered and quietly has no vision.
//
// `DualPathFamilies` states that invariant explicitly. These tests are what make stating it useful:
// the declaration and the two registries have to agree, in BOTH directions.

import Foundation
import MLXLLM
import MLXLMCommon
import MLXVLM
import Testing

@Suite("Dual-path model families")
struct DualPathFamiliesTests {

    private func registeredInBoth() async -> Set<String> {
        let llm = await LLMTypeRegistry.shared.registeredModelTypes
        let vlm = await VLMTypeRegistry.shared.registeredModelTypes
        return llm.intersection(vlm)
    }

    /// The contract, as one exact equality. Both directions in a single assertion — but reported
    /// as two named sets, because "not equal" does not tell the next author which file to edit.
    @Test("the declaration matches what the factories actually register")
    func declarationMatchesRegistries() async {
        let llm = await LLMTypeRegistry.shared.registeredModelTypes
        let vlm = await VLMTypeRegistry.shared.registeredModelTypes
        let d = DualPathFamilies.divergence(llm: llm, vlm: vlm)
        let missingFromDeclaration = d.missingFromDeclaration
        let missingFromRegistry = d.missingFromRegistry
        #expect(
            d.isEmpty,
            """
            registered in both factories but NOT declared (add to DualPathFamilies): \
            \(missingFromDeclaration.sorted().joined(separator: ", ").ifEmpty("none")); \
            declared but NOT registered in both (a second registration went missing): \
            \(missingFromRegistry.sorted().joined(separator: ", ").ifEmpty("none"))
            """)
    }

    /// The direction that fails silently in production.
    ///
    /// Adding a multimodal implementation for a family that already had a text one is a one-line
    /// edit to `VLMModelFactory` — and nothing anywhere else changes. This is what notices.
    @Test("a family registered in both factories is declared")
    func nothingIsDualWithoutBeingDeclared() async {
        let undeclared = await registeredInBoth().subtracting(DualPathFamilies.modelTypes)
        #expect(
            undeclared.isEmpty,
            """
            registered in BOTH factories but not declared in DualPathFamilies: \
            \(undeclared.sorted().joined(separator: ", ")). If that is intentional, add it there; \
            the list is what tells the next person the family has two entry points.
            """)
    }

    /// The other direction: a family declared dual whose second registration went missing.
    @Test("a declared family is registered in both factories")
    func everyDeclaredFamilyIsActuallyDual() async {
        let llm = await LLMTypeRegistry.shared.registeredModelTypes
        let vlm = await VLMTypeRegistry.shared.registeredModelTypes
        for family in DualPathFamilies.modelTypes.sorted() {
            #expect(llm.contains(family), "\(family) is declared dual but LLMTypeRegistry lacks it")
            #expect(vlm.contains(family), "\(family) is declared dual but VLMTypeRegistry lacks it")
        }
    }

    /// Guards the guard: if either registry came back empty the comparisons above would pass
    /// vacuously, and this whole suite would be decoration.
    ///
    /// Asserted with non-emptiness plus a known dual-path sentinel rather than a registry SIZE. A
    /// count threshold is unrelated to the invariant and becomes maintenance noise the moment
    /// registrations move or are conditionally compiled — it would fail on a change that is not a
    /// violation, and that is the kind of test people delete.
    @Test("neither registry is empty, so the comparisons cannot pass vacuously")
    func registriesAreNotEmpty() async {
        let llm = await LLMTypeRegistry.shared.registeredModelTypes
        let vlm = await VLMTypeRegistry.shared.registeredModelTypes
        #expect(!llm.isEmpty, "LLMTypeRegistry is empty")
        #expect(!vlm.isEmpty, "VLMTypeRegistry is empty")
        #expect(llm.contains("gemma4"), "LLMTypeRegistry lost its dual-path sentinel")
        #expect(vlm.contains("gemma4"), "VLMTypeRegistry lost its dual-path sentinel")
    }
}

/// The contract exercised against SYNTHETIC registries.
///
/// The live registries agree with the declaration by construction, so a suite that only compares
/// them proves the declaration is currently correct and proves nothing about the check. These are
/// the cases that must fail — each is a real edit someone could make.
@Suite("Dual-path declaration, against synthetic registries")
struct DualPathDivergenceTests {

    @Test("a family registered in both but undeclared is reported, and named")
    func undeclaredDualIsCaught() {
        let d = DualPathFamilies.divergence(
            llm: ["gemma4", "newfam"], vlm: ["gemma4", "newfam"], declared: ["gemma4"])
        #expect(!d.isEmpty)
        #expect(d.missingFromDeclaration == ["newfam"])
        #expect(d.missingFromRegistry.isEmpty, "nothing is missing from the registries here")
    }

    @Test("a family declared but registered in only ONE factory is reported")
    func declaredButOnlyHalfRegisteredIsCaught() {
        let d = DualPathFamilies.divergence(
            llm: ["gemma4", "halfway"], vlm: ["gemma4"], declared: ["gemma4", "halfway"])
        #expect(!d.isEmpty)
        #expect(d.missingFromRegistry == ["halfway"])
        #expect(d.missingFromDeclaration.isEmpty)
    }

    @Test("a fake key in only one registry does not make it dual-path")
    func singleRegistryKeyIsNotDual() {
        let d = DualPathFamilies.divergence(
            llm: ["gemma4", "llm_only"], vlm: ["gemma4"], declared: ["gemma4"])
        #expect(d.isEmpty, "registered once is not dual-path: \(d)")
    }

    @Test("both directions are reported at once, so one run names every edit")
    func bothDirectionsReportedTogether() {
        let d = DualPathFamilies.divergence(
            llm: ["a", "b"], vlm: ["a", "b"], declared: ["a", "c"])
        #expect(d.missingFromDeclaration == ["b"])
        #expect(d.missingFromRegistry == ["c"])
    }

    /// Why the non-emptiness assertion in the live suite is not redundant: with empty registries
    /// AND an empty declaration, divergence is legitimately empty. Vacuity is a separate property
    /// from agreement, and only the live suite can check it.
    @Test("empty registries agree with an empty declaration — vacuity needs its own guard")
    func emptyAgreesWithEmpty() {
        #expect(DualPathFamilies.divergence(llm: [], vlm: [], declared: []).isEmpty)
        #expect(!DualPathFamilies.divergence(llm: [], vlm: [], declared: ["gemma4"]).isEmpty)
    }
}

extension String {
    /// Keeps an empty diagnostic list from reading as a truncated message.
    fileprivate func ifEmpty(_ replacement: String) -> String { isEmpty ? replacement : self }
}
