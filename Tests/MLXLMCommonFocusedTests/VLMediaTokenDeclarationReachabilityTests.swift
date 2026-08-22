// Copyright © 2026 Osaurus AI. All rights reserved.
//
// The VL reuse mechanism is correct and almost nothing reaches it.
//
// `VLGrowingConversationReuseTests` builds every `LMInput` with
// `mediaTokenIds: [imageToken]` and shows a text follow-up resuming from the
// cached prefix. That is what the mechanism does when it is fed. It is not
// what ships: of the VLM processors that construct an `LMInput` carrying
// media, only **three** declare `mediaTokenIds` — Audex, DeepSeek-OCR and
// Nemotron-H Omni. Qwen3-VL, Qwen2.5-VL, Gemma 4, Muse Glimmer, LFM2-VL,
// Mistral 3, GLM-4V, Zaya1-VL and the rest pass none.
//
// Without declared ids `cacheHitSuffixContainsMediaPlaceholder` takes its
// `guard let mediaTokenIds else { return true }` branch and rolls back on ANY
// non-empty suffix. So on those families even a pure-text follow-up — the case
// the characterization tests show resuming — re-prefills the whole prompt,
// vision tower included.
//
// This file asserts the reachability rather than the mechanism, because the
// mechanism already has tests and they pass regardless. It is the same shape
// as the guards that shipped correct-but-unreached twice before.
//
// These are characterization assertions about TODAY. When a family starts
// declaring its ids, the count here SHOULD be updated deliberately rather than
// quietly relaxed.

import Foundation
import MLX
import Testing

@testable import MLXLMCommon

@Suite("VL media-token declaration reachability")
struct VLMediaTokenDeclarationReachabilityTests {

    private static func repoRoot() -> URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }

    private static func vlmSources() throws -> [(name: String, text: String)] {
        let dir = repoRoot().appendingPathComponent("Libraries/MLXVLM/Models")
        let urls = try FileManager.default.subpathsOfDirectory(atPath: dir.path)
            .filter { $0.hasSuffix(".swift") }
            .map { dir.appendingPathComponent($0) }
        return try urls.map {
            (name: $0.lastPathComponent, text: try String(contentsOf: $0, encoding: .utf8))
        }
    }

    /// The undeclared case is not "slightly worse" — it is a blanket rollback.
    /// Pin the branch itself so a refactor cannot quietly make undeclared mean
    /// "no media in the suffix", which would substitute the wrong image.
    @Test("an undeclared media token set rolls back on ANY non-empty suffix")
    func undeclaredIdsRollBackUnconditionally() {
        FocusedMLXTestSupport.withLock {
            let pixels = MLXArray((0..<12).map { Float($0) }).reshaped([1, 3, 2, 2])
            MLX.eval(pixels)

            let undeclared = LMInput(
                text: .init(tokens: MLXArray([Int32(1), 27, 27, 2, 5, 6])),
                image: .init(pixels: pixels))
            let declared = LMInput(
                text: .init(tokens: MLXArray([Int32(1), 27, 27, 2, 5, 6])),
                image: .init(pixels: pixels),
                mediaTokenIds: [27])

            // Identical prompt, identical pixels, pure-text suffix [5, 6].
            // The only difference is whether the processor declared its ids.
            #expect(undeclared.cacheHitSuffixContainsMediaPlaceholder([5, 6]))
            #expect(!declared.cacheHitSuffixContainsMediaPlaceholder([5, 6]))

            // Declaring ids does NOT weaken the real rollback: a suffix that
            // still carries a placeholder is rejected either way.
            #expect(declared.cacheHitSuffixContainsMediaPlaceholder([5, 27, 6]))

            // And a prompt with no media is untouched by any of this.
            let textOnly = LMInput(text: .init(tokens: MLXArray([Int32(1), 2, 3])))
            #expect(!textOnly.cacheHitSuffixContainsMediaPlaceholder([2, 3]))
        }
    }

    /// The reachability claim itself: which processors feed the mechanism.
    @Test("only three VLM processors declare mediaTokenIds today")
    func onlyThreeProcessorsDeclareMediaTokenIds() throws {
        let declaring =
            try Self.vlmSources()
            .filter { $0.text.contains("mediaTokenIds:") }
            .map(\.name)
            .sorted()

        #expect(
            declaring == ["Audex.swift", "DeepseekOCRProcessor.swift", "NemotronHOmni.swift"],
            "declaring processors changed: \(declaring)")
    }

    /// Named individually so the failure message says WHICH family regained or
    /// lost the declaration, instead of only that a count moved.
    @Test("the mainstream VL families still declare nothing")
    func mainstreamFamiliesDeclareNothing() throws {
        let sources = try Self.vlmSources()
        let mainstream = [
            "Qwen3VL.swift", "Qwen25VL.swift", "Qwen2VL.swift", "Gemma4.swift",
            "Gemma3.swift", "MuseGlimmerProcessor.swift", "LFM2VL.swift",
            "Mistral3.swift", "Glm4v.swift", "Zaya1VL.swift", "Idefics3.swift",
            "Pixtral.swift", "SmolVLM2.swift", "FastVLM.swift",
        ]

        for name in mainstream {
            guard let file = sources.first(where: { $0.name == name }) else { continue }
            #expect(
                !file.text.contains("mediaTokenIds:"),
                "\(name) now declares mediaTokenIds — update the reuse table in the audit")
        }
    }

    /// The second half of why VL prompts do not resume, and the one that is
    /// easy to miss: Qwen3-VL's TEXT branch computes canonical boundaries and
    /// passes them, while its MEDIA branch returns an `LMInput` with neither
    /// `cachePrefixTokenCounts` nor `cacheStablePrefixTokenCounts`. A prompt
    /// that declares no boundaries has nothing for the coordinator to probe,
    /// so the rollback above is never even reached — there is no candidate hit
    /// to roll back.
    @Test("Qwen3-VL declares cache boundaries on the text path but not the media path")
    func qwen3VLMediaPathDeclaresNoBoundaries() throws {
        let source = try String(
            contentsOf: Self.repoRoot()
                .appendingPathComponent("Libraries/MLXVLM/Models/Qwen3VL.swift"),
            encoding: .utf8)

        // The text-only branch is the one that carries boundaries.
        #expect(source.contains("cachePrefixTokenCounts: cacheBoundaries.all"))
        #expect(source.contains("cacheStablePrefixTokenCounts: cacheBoundaries.stable"))

        // Exactly one construction site does, and it is not the media one.
        let boundaryDeclarations = source.components(
            separatedBy: "cachePrefixTokenCounts: cacheBoundaries.all"
        ).count - 1
        #expect(
            boundaryDeclarations == 1,
            "expected the single text-path declaration, found \(boundaryDeclarations)")

        // The media construction passes image/video and stops there.
        #expect(source.contains("image: processedImage"))
        #expect(source.contains("video: processedVideo"))
    }
}
