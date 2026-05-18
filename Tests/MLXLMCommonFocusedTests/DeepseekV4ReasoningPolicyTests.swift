// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import MLXLMCommon
import Testing

@Suite("DeepseekV4 reasoning policy")
struct DeepseekV4ReasoningPolicyTests {
    @Test("public max passes through without hidden downgrade")
    func maxPassesThroughByDefault() {
        let context = DeepseekV4ReasoningPolicy.normalizedAdditionalContext(
            [
                "enable_thinking": true,
                "reasoning_effort": "max",
            ],
            modelType: "deepseek_v4",
            environment: [:]
        )

        #expect(context?["enable_thinking"] as? Bool == true)
        #expect(context?["reasoning_effort"] as? String == "max")
        #expect(cacheScopeSalt(from: context) == "reasoning=on|effort=max")
    }

    @Test("legacy raw max environment no longer gates max pass-through")
    func rawMaxEnvironmentDoesNotChangeMaxPassThrough() {
        let context = DeepseekV4ReasoningPolicy.normalizedAdditionalContext(
            ["reasoning_effort": "max"],
            modelType: "deepseek_v4",
            environment: [:]
        )

        #expect(context?["enable_thinking"] as? Bool == true)
        #expect(context?["reasoning_effort"] as? String == "max")
        #expect(cacheScopeSalt(from: context) == "reasoning=on|effort=max")
    }

    @Test("low medium high efforts are preserved instead of aliased")
    func lowMediumHighPassThrough() {
        for effort in ["low", "medium", "high"] {
            let context = DeepseekV4ReasoningPolicy.normalizedAdditionalContext(
                ["reasoning_effort": effort],
                modelType: "deepseek_v4",
                environment: [:]
            )
            #expect(context?["enable_thinking"] as? Bool == true)
            #expect(context?["reasoning_effort"] as? String == effort)
        }
    }

    @Test("direct rail efforts remove reasoning effort")
    func directRailEffortsDisableThinking() {
        let context = DeepseekV4ReasoningPolicy.normalizedAdditionalContext(
            [
                "enable_thinking": true,
                "reasoning_effort": "instruct",
            ],
            modelType: "deepseek_v4",
            environment: [:]
        )

        #expect(context?["enable_thinking"] as? Bool == false)
        #expect(context?["reasoning_effort"] == nil)
        #expect(cacheScopeSalt(from: context) == "reasoning=off")
    }

    @Test("force direct environment does not override explicit reasoning request")
    func forceDirectRailEnvironmentDoesNotOverrideRequest() {
        let context = DeepseekV4ReasoningPolicy.normalizedAdditionalContext(
            ["reasoning_effort": "max"],
            modelType: "deepseek_v4",
            environment: [DeepseekV4ReasoningPolicy.forceDirectRailEnvironmentKey: "true"]
        )

        #expect(context?["enable_thinking"] as? Bool == true)
        #expect(context?["reasoning_effort"] as? String == "max")
    }

    @Test("non DSV4 context is not rewritten")
    func nonDSV4ContextIsUnchanged() {
        let context = DeepseekV4ReasoningPolicy.normalizedAdditionalContext(
            ["reasoning_effort": "max"],
            modelType: "qwen3",
            environment: [:]
        )

        #expect(context?["reasoning_effort"] as? String == "max")
        #expect(context?["enable_thinking"] == nil)
    }
}
