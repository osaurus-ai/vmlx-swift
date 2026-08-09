// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import MLXLMCommon
import Testing

@testable import MLXLLM

@Suite("DeepseekV4 reasoning policy")
struct DeepseekV4ReasoningPolicyTests {
    @Test("public max passes through without hidden downgrade")
    func maxPassesThroughByDefault() throws {
        let context = try DeepseekV4ReasoningPolicy.normalizedAdditionalContext(
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
    func rawMaxEnvironmentDoesNotChangeMaxPassThrough() throws {
        let context = try DeepseekV4ReasoningPolicy.normalizedAdditionalContext(
            ["reasoning_effort": "max"],
            modelType: "deepseek_v4",
            environment: [:]
        )

        #expect(context?["enable_thinking"] as? Bool == true)
        #expect(context?["reasoning_effort"] as? String == "max")
        #expect(cacheScopeSalt(from: context) == "reasoning=on|effort=max")
    }

    @Test("0731 accepts exactly low high and max efforts")
    func canonicalEffortsPassThrough() throws {
        for effort in ["low", "high", "max"] {
            let context = try DeepseekV4ReasoningPolicy.normalizedAdditionalContext(
                ["reasoning_effort": effort],
                modelType: "deepseek_v4",
                environment: [:]
            )
            #expect(context?["enable_thinking"] as? Bool == true)
            #expect(context?["reasoning_effort"] as? String == effort)
        }
    }

    @Test("invalid and legacy effort aliases fail explicitly")
    func invalidEffortsFailExplicitly() {
        for effort in ["medium", "maximum", "instruct", "off"] {
            do {
                _ = try DeepseekV4ReasoningPolicy.normalizedAdditionalContext(
                    ["reasoning_effort": effort],
                    modelType: "deepseek_v4",
                    environment: [:]
                )
                Issue.record("Expected reasoning_effort=\(effort) to fail")
            } catch let error as DeepseekV4ReasoningPolicyError {
                #expect(error == .invalidReasoningEffort(effort))
            } catch {
                Issue.record("Unexpected error for reasoning_effort=\(effort): \(error)")
            }
        }
    }

    @Test("explicit thinking false overrides every valid effort")
    func explicitFalseOverridesEffort() throws {
        for effort in ["low", "high", "max"] {
            let context = try DeepseekV4ReasoningPolicy.normalizedAdditionalContext(
                [
                    "enable_thinking": false,
                    "reasoning_effort": effort,
                ],
                modelType: "deepseek_v4",
                environment: [:]
            )

            #expect(context?["enable_thinking"] as? Bool == false)
            #expect(context?["reasoning_effort"] == nil)
            #expect(cacheScopeSalt(from: context) == "reasoning=off")
        }
    }

    @Test("thinking true without effort renders the enforced low rail")
    func thinkingTrueWithoutEffortRemainsLow() throws {
        let context = try DeepseekV4ReasoningPolicy.normalizedAdditionalContext(
            ["enable_thinking": true],
            modelType: "deepseek_v4",
            environment: [:]
        )

        #expect(context?["enable_thinking"] as? Bool == true)
        #expect(context?["reasoning_effort"] == nil)
        #expect(cacheScopeSalt(from: context) == "reasoning=on")

        let prompt = try DeepseekV4ChatEncoder.renderOpenAIChat(
            messages: [["role": "user", "content": "hi"]],
            tools: nil,
            additionalContext: context,
            addGenerationPrompt: true
        )
        #expect(prompt.contains("Reasoning Effort: Low"),
            "absent effort is the low default and must carry the enforcement preface")
        #expect(!prompt.contains("Reasoning Effort: Absolute"))
        #expect(!prompt.contains("Reasoning Effort: Beyond"))
        #expect(prompt.hasSuffix(DeepseekV4Tokens.assistant + DeepseekV4Tokens.thinkStart))
    }

    @Test("absent thinking controls default to the bundle's thinking mode")
    func absentThinkingControlsDefaultToThinking() throws {
        // DSV4's jang_config declares default_mode="thinking" with
        // default_effort="low" (now enforced via the low preface). A request
        // that carries NO enable_thinking and NO reasoning_effort must render
        // the thinking rail — an open <think> tail — not silently fall to
        // chat mode while the UI reports the "Low" default. Explicit
        // enable_thinking=false remains the only path to the chat rail.
        let prompt = try DeepseekV4ChatEncoder.renderOpenAIChat(
            messages: [["role": "user", "content": "hi"]],
            tools: nil,
            additionalContext: nil,
            addGenerationPrompt: true
        )
        #expect(prompt.contains("Reasoning Effort: Low"))
        #expect(prompt.hasSuffix(DeepseekV4Tokens.assistant + DeepseekV4Tokens.thinkStart))

        let emptyContext = try DeepseekV4ChatEncoder.renderOpenAIChat(
            messages: [["role": "user", "content": "hi"]],
            tools: nil,
            additionalContext: [:],
            addGenerationPrompt: true
        )
        #expect(emptyContext.hasSuffix(DeepseekV4Tokens.assistant + DeepseekV4Tokens.thinkStart))

        let explicitOff = try DeepseekV4ChatEncoder.renderOpenAIChat(
            messages: [["role": "user", "content": "hi"]],
            tools: nil,
            additionalContext: ["enable_thinking": false],
            addGenerationPrompt: true
        )
        #expect(explicitOff.hasSuffix(DeepseekV4Tokens.assistant + DeepseekV4Tokens.thinkEnd))
    }

    @Test("force direct environment does not override explicit reasoning request")
    func forceDirectRailEnvironmentDoesNotOverrideRequest() throws {
        let context = try DeepseekV4ReasoningPolicy.normalizedAdditionalContext(
            ["reasoning_effort": "max"],
            modelType: "deepseek_v4",
            environment: [DeepseekV4ReasoningPolicy.forceDirectRailEnvironmentKey: "true"]
        )

        #expect(context?["enable_thinking"] as? Bool == true)
        #expect(context?["reasoning_effort"] as? String == "max")
    }

    @Test("non DSV4 context is not rewritten")
    func nonDSV4ContextIsUnchanged() throws {
        let context = try DeepseekV4ReasoningPolicy.normalizedAdditionalContext(
            ["reasoning_effort": "max"],
            modelType: "qwen3",
            environment: [:]
        )

        #expect(context?["reasoning_effort"] as? String == "max")
        #expect(context?["enable_thinking"] == nil)
    }

    @Test("LLM processor merge preserves bundle defaults and request precedence")
    func processorMergePreservesDefaultsAndExplicitFalse() throws {
        let omitted = try llmMergedAdditionalContext(
            defaultAdditionalContext: nil,
            requestAdditionalContext: nil,
            modelType: "deepseek_v4"
        )
        #expect(omitted == nil)

        let requestEnabled = try llmMergedAdditionalContext(
            defaultAdditionalContext: ["enable_thinking": false],
            requestAdditionalContext: ["enable_thinking": true],
            modelType: "deepseek_v4"
        )
        #expect(requestEnabled?["enable_thinking"] as? Bool == true)
        #expect(requestEnabled?["reasoning_effort"] == nil)

        let requestDisabled = try llmMergedAdditionalContext(
            defaultAdditionalContext: ["enable_thinking": true],
            requestAdditionalContext: [
                "enable_thinking": false,
                "reasoning_effort": "max",
            ],
            modelType: "deepseek_v4"
        )
        #expect(requestDisabled?["enable_thinking"] as? Bool == false)
        #expect(requestDisabled?["reasoning_effort"] == nil)
    }

    @Test("LLM processor merge propagates invalid effort errors")
    func processorMergeRejectsMedium() {
        do {
            _ = try llmMergedAdditionalContext(
                defaultAdditionalContext: nil,
                requestAdditionalContext: ["reasoning_effort": "medium"],
                modelType: "deepseek_v4"
            )
            Issue.record("Expected processor merge to reject reasoning_effort=medium")
        } catch let error as DeepseekV4ReasoningPolicyError {
            #expect(error == .invalidReasoningEffort("medium"))
        } catch {
            Issue.record("Unexpected processor merge error: \(error)")
        }
    }
}
