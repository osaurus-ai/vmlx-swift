// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Construction-time modality selection, checked without a GPU or a bundle.

import Foundation
import MLXLMCommon
import MLXVLM
import Testing

@Suite("Model construction modalities")
struct ModelConstructionModalitiesTests {

    typealias Modalities = Set<ModelRuntimeRequestModality>

    // MARK: the subset rule

    @Test("no request means everything the configuration can build")
    func nilRequestTakesAll() throws {
        let constructible: Modalities = [.text, .vision, .video]
        #expect(
            try Modalities.resolveForConstruction(requested: nil, constructible: constructible)
                == constructible)
    }

    @Test("a caller may ask for LESS — that is the point")
    func narrowerRequestIsHonoured() throws {
        // This is the load that is impossible today for any family nobody hand-registered twice:
        // a vision bundle opened text-only, leaving its tower unallocated.
        let got = try Modalities.resolveForConstruction(
            requested: [.text], constructible: [.text, .vision, .video])
        #expect(got == [.text])
        #expect(!got.contains(.vision))
    }

    @Test("asking for MORE than can be built is an error, and an early one")
    func widerRequestThrows() {
        #expect(throws: UnconstructibleModalities.self) {
            try Modalities.resolveForConstruction(requested: [.text, .vision], constructible: [.text])
        }
    }

    @Test("text is NOT implied — a speech-to-speech model must be expressible")
    func textIsNotImplied() throws {
        // audio in, audio out, no text anywhere. Unlikely, not impossible; implying text would make
        // it inexpressible, and the cost of not implying is that callers say what they mean.
        let got = try Modalities.resolveForConstruction(requested: [.audio], constructible: [.audio])
        #expect(got == [.audio])
        #expect(!got.contains(.text))
    }

    @Test("an empty configuration or request is rejected, not silently accepted")
    func emptyIsRejected() {
        // With every component optional, "nothing" is reachable by a config that merely failed to
        // parse. A model that can do nothing is a bug far more often than an intent.
        #expect(throws: EmptyModalitySelection.self) {
            try Modalities.resolveForConstruction(requested: nil, constructible: [])
        }
        #expect(throws: EmptyModalitySelection.self) {
            try Modalities.resolveForConstruction(requested: [], constructible: [.text])
        }
    }

    /// The vocabulary is `ModelRuntimeRequestModality` — shared with
    /// `ModelRuntimeCapabilitySnapshot.validate`, and CLOSED. A downstream package cannot mint a
    /// modality; a new one is a case added upstream, which is deliberate: the exhaustive switch in
    /// `support(for:)` then forces it to be wired through everywhere before it can be requested.
    /// What this test pins is that the resolver itself is modality-agnostic — it hardcodes no
    /// subset, so a new case costs nothing here.
    @Test("the resolver covers the whole vocabulary, with no lane special-cased")
    func resolverIsModalityAgnostic() throws {
        for modality in ModelRuntimeRequestModality.allCases {
            let got = try Modalities.resolveForConstruction(
                requested: [modality], constructible: Set(ModelRuntimeRequestModality.allCases))
            #expect(got == [modality])
        }
    }

    @Test("vision and video are separate, because bundles differ on exactly that")
    func visionAndVideoAreDistinct() {
        // JangLoader's own comment: "many VLMs accept images but not videos", and Gemma4 throws on
        // video input. One vision flag would let such a request pass here and fail at prepare().
        #expect(ModelRuntimeRequestModality.vision != ModelRuntimeRequestModality.video)
        #expect(throws: UnconstructibleModalities.self) {
            try Modalities.resolveForConstruction(
                requested: [.video], constructible: [.text, .vision])
        }
    }

    @Test("the error names what is missing, not just that something is")
    func errorIsActionable() {
        let e = UnconstructibleModalities(
            requested: [.text, .audio], constructible: [.text, .vision])
        #expect(e.excess == [.audio])
        #expect(e.description.contains("audio"))
    }

    // MARK: the two questions are deliberately distinct

    /// `ModelRuntimeCapabilitySnapshot` says what a bundle was TRAINED for; this says what a config
    /// can BUILD. Keeping them apart is what lets a broken conversion — a `supports_vision` stamp
    /// over a config with no vision section — stay visible instead of being averaged away.
    @Test("a bundle may declare vision while its config cannot build it")
    func declarationAndConstructibilityCanDisagree() throws {
        let json = """
            {"model_type":"muse_glimmer","hidden_size":128,"num_hidden_layers":2,
             "intermediate_size":256,"num_attention_heads":4,"num_key_value_heads":2,
             "rms_norm_eps":1e-6,"vocab_size":1000,"rope_theta":10000.0}
            """
        let cfg = try JSONDecoder.json5().decode(
            MuseGlimmerConfiguration.self, from: Data(json.utf8))

        // The config cannot build a vision tower...
        #expect(MuseGlimmer.constructibleModalities(of: cfg) == [.text])
        // ...and construction refuses, regardless of what any stamp elsewhere claims.
        #expect(throws: UnconstructibleModalities.self) {
            _ = try MuseGlimmer(cfg, requesting: [.text, .vision])
        }
    }

    // MARK: the change that removes the need for a second registration

    /// A text-only Muse Glimmer bundle has no `vision_config`, and until it became optional this
    /// configuration could not DECODE such a bundle at all.
    ///
    /// That does NOT make the LLM-side `muse_glimmer` entry redundant. `muse_glimmer` names two
    /// bundle shapes, and the two registrations serve one each — `ModelFactory` routes between
    /// them by trying the VLM factory first and falling through on any error. See
    /// `MuseGlimmerConfiguration.visionConfiguration`. What this test pins is only that a
    /// vision-less config decodes and builds nothing but text.
    @Test("a config with no vision_config decodes, and builds text only")
    func textOnlyBundleDecodes() throws {
        let json = """
            {"model_type":"muse_glimmer","hidden_size":128,"num_hidden_layers":2,
             "intermediate_size":256,"num_attention_heads":4,"num_key_value_heads":2,
             "rms_norm_eps":1e-6,"vocab_size":1000,"rope_theta":10000.0}
            """
        let cfg = try JSONDecoder.json5().decode(
            MuseGlimmerConfiguration.self, from: Data(json.utf8))
        #expect(cfg.visionConfiguration == nil)

        let model = MuseGlimmer(cfg)
        #expect(model.modalities == [.text])
    }
}
