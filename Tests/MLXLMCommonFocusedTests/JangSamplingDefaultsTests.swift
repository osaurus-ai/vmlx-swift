// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// `chat.sampling_defaults` was decoded into three fields, one of which (`max_new_tokens`) no
// observed bundle carries, while `top_k` — which every observed bundle carrying the block sets —
// was dropped. Nothing read the result either way, so neither gap could surface at runtime.

import Foundation
import MLXLMCommon
import Testing

@Suite("Jang chat sampling defaults")
struct JangSamplingDefaultsTests {

    /// The shape real bundles ship. `source` and `mode` are provenance, not sampler knobs, and
    /// must not become one; DeepSeek-V4-Flash additionally copies generation_config residue
    /// (`_from_model_config`, `bos_token_id`, `transformers_version`) into the same block.
    @Test("every sampler key a real bundle carries survives the decode")
    func realBundleKeysSurvive() {
        let d = JangChatSamplingDefaults(
            temperature: 0.6, topP: 0.95, topK: 20, minP: 0.0,
            repetitionPenalty: 1.0, presencePenalty: 0.0)
        #expect(d.temperature == 0.6)
        #expect(d.topK == 20, "top_k is carried by every observed bundle with a sampling block")
        #expect(d.minP == 0.0)
        #expect(d.repetitionPenalty == 1.0)
        #expect(d.presencePenalty == 0.0)
        #expect(!d.isEmpty)
    }

    /// A block of pure provenance resolves to nothing actionable, rather than to a struct that
    /// looks applicable and would overwrite the caller's sampler with nils-turned-defaults.
    @Test("a provenance-only block is empty")
    func provenanceOnlyIsEmpty() {
        #expect(JangChatSamplingDefaults().isEmpty)
    }

    /// The overlay replaces exactly what the bundle declares and leaves the rest of the caller's
    /// configuration alone — including fields that have nothing to do with sampling.
    @Test("applied(to:) overlays only the declared knobs")
    func overlayIsScoped() {
        var base = GenerateParameters()
        base.temperature = 1.0
        base.topP = 1.0
        base.topK = 0
        base.maxTokens = 4096
        base.prefillStepSize = 999          // untouched by any sampling default

        let partial = JangChatSamplingDefaults(temperature: 0.6, topK: 20)
        let out = partial.applied(to: base)
        #expect(out.temperature == 0.6)
        #expect(out.topK == 20)
        #expect(out.topP == 1.0, "not declared, so the caller's value stands")
        #expect(out.maxTokens == 4096, "not declared, so the caller's cap stands")
        #expect(out.prefillStepSize == 999, "sampling defaults must not reach non-sampler fields")
    }

    /// Both spellings of the length cap land on `maxTokens`. Bundles write `max_tokens`; the type
    /// has always called it `maxNewTokens`, and no bundle observed uses that spelling.
    @Test("the length cap lands on maxTokens")
    func lengthCapApplies() {
        let out = JangChatSamplingDefaults(maxNewTokens: 20480).applied(to: GenerateParameters())
        #expect(out.maxTokens == 20480)
    }

    /// An empty block must be a no-op, not a reset.
    @Test("an empty block changes nothing")
    func emptyIsNoOp() {
        var base = GenerateParameters()
        base.temperature = 0.42
        base.topK = 7
        let out = JangChatSamplingDefaults().applied(to: base)
        #expect(out.temperature == 0.42)
        #expect(out.topK == 7)
    }
}
