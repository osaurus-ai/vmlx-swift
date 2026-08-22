// Copyright © 2026 Osaurus AI. All rights reserved.
//
// The VL reuse mechanism is correct and almost nothing reaches it.
//
// `VLGrowingConversationReuseTests` builds every `LMInput` with
// `mediaTokenIds: [imageToken]` and shows a text follow-up resuming from the
// cached prefix. That is what the mechanism does when it is fed. It is not
// what shipped: of the VLM processors that construct an `LMInput` carrying
// media, only **three** declared `mediaTokenIds` — Audex, DeepSeek-OCR and
// Nemotron-H Omni. Qwen3-VL, Qwen2.5-VL, Qwen2-VL, Gemma 4, Muse Glimmer,
// LFM2-VL, Mistral 3, GLM-4V, Zaya1-VL and the rest passed none.
//
// The three Qwen VL processors now declare theirs through
// `QwenVL.mediaTokenIds`. Every other family still does not, so the assertions
// below are half regression guard and half worklist.
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
@testable import MLXVLM

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
    ///
    /// Every processor whose placeholder tokens could be established now
    /// declares them, through the shared `MediaTokenIds.resolve`.
    @Test("every processor with knowable placeholder tokens declares them")
    func declaringSetCoversEveryKnowableProcessor() throws {
        let declaring =
            try Self.vlmSources()
            .filter { $0.text.contains("mediaTokenIds:") }
            .map(\.name)
            .sorted()

        #expect(
            declaring == [
                "Audex.swift", "DeepseekOCRProcessor.swift", "FastVLM.swift",
                "Gemma4.swift", "LFM2VL.swift", "Mistral3.swift",
                "MuseGlimmerProcessor.swift", "NemotronHOmni.swift", "Pixtral.swift",
                "Qwen25VL.swift", "Qwen2VL.swift", "Qwen3VL.swift", "SmolVLM2.swift",
                "Zaya1VL.swift",
            ],
            "declaring processors changed: \(declaring)")
    }

    /// The remaining three carry their placeholder only as a numeric config id
    /// the processor cannot see, so there is no string to round-trip. They keep
    /// the conservative blanket rollback rather than a guessed id — a WRONG id
    /// is worse than none, because it would wave through a suffix that really
    /// does carry a placeholder.
    @Test("config-id-only families are deliberately left undeclared")
    func configIdOnlyFamiliesRemainUndeclared() throws {
        let sources = try Self.vlmSources()
        for name in ["Gemma3.swift", "Glm4v.swift", "Idefics3.swift"] {
            guard let file = sources.first(where: { $0.name == name }) else { continue }
            #expect(
                !file.text.contains("mediaTokenIds:"),
                "\(name) declared ids — verify the token string round-trips first")
        }
    }

    /// A family that emits more than one KIND of placeholder must declare all
    /// of them. Declaring only some is the dangerous direction: a suffix
    /// carrying the undeclared kind matches nothing, reads as media-free, and
    /// resumes onto the wrong media.
    @Test("multi-modality families declare every placeholder kind they emit")
    func multiModalityFamiliesDeclareEveryKind() throws {
        let sources = try Self.vlmSources()

        func text(_ name: String) -> String {
            sources.first(where: { $0.name == name })?.text ?? ""
        }

        // Gemma 4 builds an LMInput carrying image AND audio.
        let gemma4 = text("Gemma4.swift")
        #expect(gemma4.contains(#""<|image|>""#))
        #expect(gemma4.contains(#""<|audio|>""#))

        // Muse Glimmer expands `<|patch|>` for stills and `<|video|>` for clips.
        let muse = text("MuseGlimmerProcessor.swift")
        #expect(muse.contains(#""<|patch|>""#))
        #expect(muse.contains(#""<|video|>""#))

        // Qwen VL: image and video pads.
        for name in ["Qwen3VL.swift", "Qwen25VL.swift", "Qwen2VL.swift"] {
            #expect(text(name).contains(#""<|image_pad|>", "<|video_pad|>""#), "\(name)")
        }

        // Mistral/Pixtral repeat [IMG] and separate spans with BREAK/END.
        for name in ["Mistral3.swift", "Pixtral.swift"] {
            let t = text(name)
            #expect(t.contains(#""[IMG]""#), "\(name)")
            #expect(t.contains(#""[IMG_BREAK]""#), "\(name)")
            #expect(t.contains(#""[IMG_END]""#), "\(name)")
        }
    }

    /// Named individually so the failure message says WHICH family regained or
    /// lost the declaration, instead of only that a count moved.
    /// Resolution goes through the one shared helper, so the round-trip and
    /// nil-not-empty guarantees hold everywhere rather than per copy.
    @Test("every declaration resolves through the shared helper")
    func everyDeclarationUsesTheSharedResolver() throws {
        let declaring = try Self.vlmSources().filter { $0.text.contains("mediaTokenIds:") }

        for file in declaring {
            #expect(
                file.text.contains("MediaTokenIds.resolve")
                    || file.text.contains("QwenVL.mediaTokenIds")
                    || file.name == "Audex.swift"
                    || file.name == "DeepseekOCRProcessor.swift"
                    || file.name == "NemotronHOmni.swift",
                "\(file.name) hand-rolls its ids instead of using MediaTokenIds.resolve")
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

/// A tokenizer whose special-token behaviour is scripted, so the resolver's
/// guards can be exercised without a real bundle.
private struct ScriptedTokenizer: MLXLMCommon.Tokenizer {
    /// token text -> ids it encodes to.
    let table: [String: [Int]]
    /// id -> token text it decodes back to.
    let reverse: [Int: String]

    func encode(text: String, addSpecialTokens: Bool) -> [Int] { table[text] ?? [] }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        tokenIds.compactMap { reverse[$0] }.joined()
    }
    func convertTokenToId(_ token: String) -> Int? { table[token]?.first }
    func convertIdToToken(_ id: Int) -> String? { reverse[id] }

    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var eosTokenId: Int? { nil }
    var unknownToken: String? { "<unk>" }
    var unknownTokenId: Int? { 0 }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [] }
}

@Suite("QwenVL media token id resolution")
struct QwenVLMediaTokenIdResolutionTests {

    private static let image = "<|image_pad|>"
    private static let video = "<|video_pad|>"

    @Test("a tokenizer that knows both specials yields both ids")
    func bothSpecialsResolve() {
        let tokenizer = ScriptedTokenizer(
            table: [Self.image: [151_655], Self.video: [151_656]],
            reverse: [151_655: Self.image, 151_656: Self.video])

        let ids = QwenVL.mediaTokenIds(
            tokenizer: tokenizer, tokens: [Self.image, Self.video])
        #expect(ids == [151_655, 151_656])
    }

    /// The dangerous case. A tokenizer without these specials maps them to
    /// `<unk>` — a single, real id that also appears in ordinary text. Taking
    /// it would declare a "media placeholder" that matches unrelated tokens and
    /// suppress reuse far more aggressively than declaring nothing. The decode
    /// round-trip is what rejects it; the `count == 1` check alone would not,
    /// because unk IS one token.
    @Test("an unk-mapping tokenizer resolves to nothing, not to unk")
    func unkMappingResolvesToNil() {
        let tokenizer = ScriptedTokenizer(
            table: [Self.image: [0], Self.video: [0]],
            reverse: [0: "<unk>"])

        #expect(
            QwenVL.mediaTokenIds(tokenizer: tokenizer, tokens: [Self.image, Self.video]) == nil)
    }

    /// A tokenizer that splits the literal into ordinary pieces is also not a
    /// declaration — those ids are pieces of text, not a placeholder.
    @Test("a token that splits into pieces is not declared")
    func splitTokenIsNotDeclared() {
        let tokenizer = ScriptedTokenizer(
            table: [Self.image: [10, 11, 12]],
            reverse: [10: "<|", 11: "image_pad", 12: "|>"])

        #expect(QwenVL.mediaTokenIds(tokenizer: tokenizer, tokens: [Self.image]) == nil)
    }

    /// Partial knowledge is still useful: an image-only bundle declares the
    /// image id and simply has no video id to declare.
    @Test("one resolvable token is enough")
    func partialResolutionStillDeclares() {
        let tokenizer = ScriptedTokenizer(
            table: [Self.image: [151_655], Self.video: [0]],
            reverse: [151_655: Self.image, 0: "<unk>"])

        #expect(
            QwenVL.mediaTokenIds(tokenizer: tokenizer, tokens: [Self.image, Self.video])
                == [151_655])
    }

    /// Returning `nil` rather than `[]` matters: an empty array takes the
    /// `!mediaTokenIds.isEmpty` branch and would report "no media in the
    /// suffix" for every suffix, which is the unsafe direction.
    @Test("nil, never empty — empty would disable the rollback entirely")
    func nilRatherThanEmpty() {
        FocusedMLXTestSupport.withLock {
            let pixels = MLXArray((0..<12).map { Float($0) }).reshaped([1, 3, 2, 2])
            MLX.eval(pixels)

            let empty = LMInput(
                text: .init(tokens: MLXArray([Int32(1), 27, 2])),
                image: .init(pixels: pixels),
                mediaTokenIds: [])
            // Empty declares "nothing is a placeholder" — a suffix carrying the
            // real placeholder is waved through. This is exactly what the
            // resolver must never produce.
            #expect(!empty.cacheHitSuffixContainsMediaPlaceholder([27]))

            let tokenizer = ScriptedTokenizer(table: [Self.image: [0]], reverse: [0: "<unk>"])
            #expect(QwenVL.mediaTokenIds(tokenizer: tokenizer, tokens: [Self.image]) == nil)
        }
    }
}
