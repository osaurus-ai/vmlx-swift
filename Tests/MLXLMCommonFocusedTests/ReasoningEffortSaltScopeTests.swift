// Copyright © 2026 Osaurus AI. All rights reserved.
//
// What `effort=` in the cache scope salt actually costs.
//
// This file exists because of a wrong claim. The first pass looked at four
// chat templates, wrote "all 8 templates that consume reasoning_effort render
// it into the system message before anything else", and concluded that
// changing effort mid-conversation always re-prefills and that the salt was
// therefore free. Measured against the real library:
//
//   - 97 chat templates in ~/models
//   - 8 files consume `reasoning_effort` — but only **2 distinct bodies**
//     (6 + 2 copies across quant / CRACK variants, both in-house)
//   - 75 use `enable_thinking`; 3 use `reasoning_strength` (Muse Glimmer)
//
// Two templates out of ninety-seven is not a property of the library, and for
// the other ~89 the conclusion INVERTS. Their template never references
// `reasoning_effort`, so the rendered prompt cannot depend on it — output
// cannot vary with a variable the template never reads. The host still injects
// `reasoning_effort` into `additionalContext` for the Qwen / Nemotron / Zaya /
// generic branches, so the salt changes while the tokens do not.
//
// Identical tokens, different key, guaranteed miss. That is reuse thrown away
// for nothing, which is the opposite of "the salt is free".
//
// These tests pin the mechanism so the cost is visible. If someone narrows the
// salt to key off what the template actually consumes, the expectations here
// SHOULD change deliberately rather than quietly.

import Foundation
import MLX
import Testing

@testable import MLXLMCommon

@Suite("reasoning-effort salt scope")
struct ReasoningEffortSaltScopeTests {

    /// The mechanism itself: two efforts, same everything else, different salt.
    @Test("effort level alone changes the cache scope salt")
    func effortLevelAloneChangesTheScopeSalt() {
        let low = cacheScopeSalt(from: ["enable_thinking": true, "reasoning_effort": "low"])
        let high = cacheScopeSalt(from: ["enable_thinking": true, "reasoning_effort": "high"])

        #expect(low == "reasoning=on|effort=low")
        #expect(high == "reasoning=on|effort=high")
        #expect(low != high)
    }

    /// And it reaches the composed key, which is what the tiers actually probe.
    /// A component that changed the scope string but washed out of the final
    /// hash would cost nothing; this one does not wash out.
    @Test("the effort difference survives into the composed cache salt")
    func effortDifferenceSurvivesIntoTheComposedSalt() {
        let tokens = MLXArray([Int32(1), 2, 3])
        let lowInput = LMInput(
            text: .init(tokens: tokens),
            cacheScopeSalt: cacheScopeSalt(from: ["reasoning_effort": "low"]))
        let highInput = LMInput(
            text: .init(tokens: tokens),
            cacheScopeSalt: cacheScopeSalt(from: ["reasoning_effort": "high"]))

        let lowSalt = computeCacheSalt(for: lowInput)
        let highSalt = computeCacheSalt(for: highInput)

        #expect(lowSalt != nil)
        #expect(lowSalt != highSalt)
    }

    /// Absent effort must NOT invent a component. A model that never receives
    /// `reasoning_effort` has to keep the salt it would have had, or every
    /// non-reasoning model would fragment its own cache.
    @Test("no effort key means no effort component")
    func absentEffortAddsNothing() {
        #expect(cacheScopeSalt(from: ["enable_thinking": true]) == "reasoning=on")
        #expect(cacheScopeSalt(from: [:]) == nil)
        #expect(cacheScopeSalt(from: nil) == nil)

        // Whitespace-only is absent, not a distinct level.
        #expect(cacheScopeSalt(from: ["reasoning_effort": "   "]) == nil)
    }

    /// Case and surrounding whitespace must not fragment the cache: "High",
    /// "high" and " HIGH " are one level, not three cache scopes.
    @Test("effort is normalized so equivalent spellings share a scope")
    func effortSpellingsShareOneScope() {
        let a = cacheScopeSalt(from: ["reasoning_effort": "High"])
        let b = cacheScopeSalt(from: ["reasoning_effort": " HIGH "])
        let c = cacheScopeSalt(from: ["reasoning_effort": "high"])
        #expect(a == c)
        #expect(b == c)
    }

    /// Muse Glimmer is the family Eric named, and it is genuinely different:
    /// it has no thinking tags, ignores `reasoning_effort` entirely, and reads
    /// `reasoning_strength` — which the template DOES render, so there the
    /// prompt really does change and the salt really is earning its keep.
    @Test("Muse reasoning_strength is a separate, template-rendered scope")
    func museStrengthIsItsOwnScope() {
        let low = cacheScopeSalt(from: ["reasoning_strength": "low"])
        let xhigh = cacheScopeSalt(from: ["reasoning_strength": "xhigh"])

        #expect(low == "strength=low")
        #expect(xhigh == "strength=xhigh")
        #expect(low != xhigh)

        // It composes with, rather than replaces, the reasoning flag.
        #expect(
            cacheScopeSalt(from: ["enable_thinking": true, "reasoning_strength": "medium"])
                == "reasoning=on|strength=medium")
    }

    /// Turning thinking on and off is the case where the salt is unambiguously
    /// correct: 75 of 97 templates branch on `enable_thinking`, so the rendered
    /// prompt genuinely differs and the two scopes must not share KV.
    @Test("thinking on/off remains a legitimately distinct scope")
    func thinkingOnOffStaysDistinct() {
        #expect(cacheScopeSalt(from: ["enable_thinking": true]) == "reasoning=on")
        #expect(cacheScopeSalt(from: ["enable_thinking": false]) == "reasoning=off")
    }
}
