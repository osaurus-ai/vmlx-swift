// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation

/// Model families registered in BOTH `LLMTypeRegistry` and `VLMTypeRegistry`.
///
/// ## Why this exists
///
/// Two files register the same `model_type`, share no symbol, and never refer to each other:
///
///     LLMModelFactory   "gemma4": create(Gemma4TextConfiguration.self, Gemma4TextModel.init)
///     VLMModelFactory   "gemma4": create(Gemma4Configuration.self,     Gemma4.init)
///
/// Nothing connects those two lines, so a change made to one is silently not made to the other and
/// the failure surfaces as "this model has no vision" at run time, on a bundle that looks
/// registered. That is exactly how a Muse Glimmer fix came to wire the text path and miss the
/// multimodal one — the normal case for that bundle. This declares the invariant the two registries
/// encode implicitly, so it can be CHECKED in both directions.
///
/// ## What the second registration actually expresses
///
/// Not a second BUNDLE SHAPE, which is the easy assumption. `ModelFactoryRegistry` seeds itself
/// with an `MLXVLM` trampoline ahead of an `MLXLLM` one, and `load(loader:)` falls through on any
/// error, so a routed load reaches the LLM entry only when the VLM factory declines the bundle.
///
/// This type declares registry OVERLAP and nothing more. It does not model, and must not be read
/// as predicting, which registration a given bundle selects at run time: that is a property of the
/// routed loader and of the bundle, and it is not checked here.
///
/// The LLM-side entry is reached by callers who ask `LLMModelFactory` directly, to mean "text
/// only" — an INTENT, expressed as a registration because there was no parameter for it. Where a
/// family's model supports capability-driven construction, that intent has a better home. A
/// natively text-only RELEASE is separately trained, often at a different parameter count, and
/// keeps needing this path.
///
/// ## What is, and is not, written twice
///
/// The type side is clean: `protocol VisionLanguageModelProtocol: LanguageModel`, so a family's
/// model is not reimplemented. Its CHECKPOINT-KEY POLICY was another matter — `muse_glimmer`
/// carried the centered-norm `+1` fold in both paths, and `gemma4` carried the JANG expert rename
/// and the vocab trim in both. Those are consolidated now (`foldCenteredNorms`,
/// `remappingSwitchMLP`, `trimmingVocabDimension`), each with one owner and a guard. `gemma3`,
/// `qwen3_5_moe` and `diffusion_gemma` never had the problem — they delegate, subclass, and return
/// the same type respectively. `mistral3` genuinely differs between paths and is left alone.
///
/// ## What it does not do
///
/// It does not unify the registries. Note the reason is NOT that no module can see both: this file
/// lives in `MLXLMCommon`, and `ModelFactoryRegistry` already reaches into both `MLXVLM` and
/// `MLXLLM` without importing either, through `NSClassFromString` trampolines. The compile-time
/// direction (`MLXVLM` depends on `MLXLLM`, never the reverse) holds and does prevent a single
/// table naming `Gemma4TextModel` and `Gemma4` as symbols — but a unified registry is reachable by
/// the route the trampolines already use. It is simply a much larger change than declaring the
/// invariant, and declaring it is what catches the bug.
public enum DualPathFamilies {

    /// `model_type` values registered in BOTH `LLMTypeRegistry` and `VLMTypeRegistry`.
    ///
    /// Adding a multimodal implementation for an existing text-only family means adding its
    /// `model_type` here as well; the conformance test fails until the registration and this list
    /// agree, in both directions.
    ///
    /// This list was NOT compiled by reading the two factories. It was compiled by asking the built
    /// registries at run time, because reading them got it wrong: a source scan for `"type": create(`
    /// found seven and missed `mistral3` and `ministral3`, which are registered through named
    /// functions (`dispatchMistral3LLM` / `dispatchMistral3VLM`) instead of an inline creator. The
    /// test caught that on its first run. Maintain this list from a test failure, never from a grep.
    public static let modelTypes: Set<String> = [
        "diffusion_gemma",   // DiffusionGemmaModel      / makeDiffusionGemmaVLM
        "gemma3",            // Gemma3TextModel          / Gemma3
        "gemma4",            // Gemma4TextModel          / Gemma4
        "gemma4_unified",    // Gemma4TextModel          / Gemma4
        "ministral3",        // dispatchMistral3LLM      / dispatchMistral3VLM
        "mistral3",          // dispatchMistral3LLM      / dispatchMistral3VLM
        "muse_glimmer",      // MuseGlimmerTextModel     / MuseGlimmer
        "qwen3_5",           // Qwen35Model              / Qwen35
        "qwen3_5_moe",       // Qwen35MoE                / Qwen35MoE
    ]
    /// How a pair of registries diverges from the declaration.
    ///
    /// A pure function of its three inputs so the contract can be exercised against synthetic
    /// registries. Comparing only the live singletons would leave the comparison itself untested:
    /// every case that MATTERS is one where the registries and the declaration disagree, and the
    /// live ones agree by construction.
    public struct Divergence: Equatable, Sendable {
        /// Registered in both factories, absent from the declaration — a family that quietly became
        /// dual-path. This is the direction that fails silently in production.
        public let missingFromDeclaration: Set<String>

        /// Declared dual-path, not registered in both — a second registration that went missing.
        public let missingFromRegistry: Set<String>

        public var isEmpty: Bool { missingFromDeclaration.isEmpty && missingFromRegistry.isEmpty }
    }

    public static func divergence(
        llm: Set<String>, vlm: Set<String>, declared: Set<String> = DualPathFamilies.modelTypes
    ) -> Divergence {
        let both = llm.intersection(vlm)
        return Divergence(
            missingFromDeclaration: both.subtracting(declared),
            missingFromRegistry: declared.subtracting(both))
    }

}
