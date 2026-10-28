import Foundation
import MLXLMCommon
import Testing

@testable import MLXLLM
@testable import MLXVLM

@Suite("Split bundle thinking and effort defaults")
struct SplitThinkingDefaultsTests {
    private func fixture(_ enabled: Bool = true) -> [String: Any] {
        [
            "format": "jang",
            "capabilities": ["supports_thinking": true],
            "chat": [
                "thinking": [
                    "supported": true, "template_flag": "enable_thinking", "default": enabled,
                ],
                "reasoning_effort": ["values": ["low", "high", "max"], "default": "high"],
            ],
        ]
    }

    @Test(arguments: [true, false])
    func declaredDefaultsReachBothFactories(_ enabled: Bool) throws {
        let config = try JangLoader.parseConfig(from: fixture(enabled))
        let chat = try #require(config.chat)
        #expect(chat.reasoning?.reasoningEffortLevels == ["low", "high", "max"])
        #expect(chat.reasoning?.defaultEffort == "high")
        #expect(chat.reasoning?.defaultMode == (enabled ? "thinking" : "chat"))
        let llm = try #require(
            llmDefaultAdditionalContext(
                modelType: "deepseek_v41", capabilities: config.capabilities,
                generationConfig: nil, chatConfig: chat))
        let vlm = try #require(
            VLMDefaultContextUserInputProcessor.defaultContext(
                capabilities: config.capabilities, chatConfig: chat))
        for context in [llm, vlm] {
            #expect(context["enable_thinking"] as? Bool == enabled)
            #expect(context["reasoning_effort"] as? String == (enabled ? "high" : nil))
        }
        if enabled { #expect(DeepseekV4ChatEncoder.reasoningBudget(from: llm) == 75) }
    }

    @Test
    func explicitRequestAndGenerationConfigWin() throws {
        let config = try JangLoader.parseConfig(from: fixture())
        let defaults = try #require(
            llmDefaultAdditionalContext(
                modelType: "deepseek_v41", capabilities: config.capabilities,
                generationConfig: nil, chatConfig: config.chat))
        let low = try #require(
            llmMergedAdditionalContext(
                defaultAdditionalContext: defaults,
                requestAdditionalContext: ["reasoning_effort": "low"], modelType: "deepseek_v41"))
        #expect(DeepseekV4ChatEncoder.reasoningBudget(from: low) == 50)
        let off = try #require(
            llmMergedAdditionalContext(
                defaultAdditionalContext: defaults,
                requestAdditionalContext: ["enable_thinking": false], modelType: "deepseek_v41"))
        #expect(off["enable_thinking"] as? Bool == false)
        let generationOff = try #require(
            llmDefaultAdditionalContext(
                modelType: "deepseek_v41", capabilities: config.capabilities,
                generationConfig: .init(defaultChatTemplateKwargs: .init(enableThinking: false)),
                chatConfig: config.chat))
        #expect(generationOff["enable_thinking"] as? Bool == false)
        #expect(generationOff["reasoning_effort"] == nil)
    }

    @Test
    func canonicalBlocksAndExplicitDefaultsKeepPrecedence() throws {
        var root = fixture()
        var chat = try #require(root["chat"] as? [String: Any])
        chat["reasoning"] = ["default_mode": "chat", "default_effort": "low"]
        root["chat"] = chat
        let canonical = try JangLoader.parseConfig(from: root)
        #expect(canonical.chat?.reasoning?.defaultMode == "chat")
        #expect(canonical.chat?.reasoning?.defaultEffort == "low")
        #expect(canonical.chat?.templateKwargsDefaults == nil)

        chat.removeValue(forKey: "reasoning")
        chat["template_kwargs_defaults"] = ["enable_thinking": false]
        root["chat"] = chat
        let explicit = try JangLoader.parseConfig(from: root)
        let context = try #require(
            VLMDefaultContextUserInputProcessor.defaultContext(
                capabilities: explicit.capabilities, chatConfig: explicit.chat))
        #expect(context["enable_thinking"] as? Bool == false)
        #expect(context["reasoning_effort"] == nil)
    }

    @Test(arguments: ["unknown", "empty", "duplicate", "missingDefault"])
    func incompleteOrUnsupportedContractsDoNotInventDefaults(_ kind: String) throws {
        var root = fixture()
        var chat = try #require(root["chat"] as? [String: Any])
        var thinking = try #require(chat["thinking"] as? [String: Any])
        var effort = try #require(chat["reasoning_effort"] as? [String: Any])
        switch kind {
        case "unknown": thinking["template_flag"] = "unsupported_flag"
        case "empty": effort["values"] = [String]()
        case "duplicate": effort["values"] = ["low", " LOW "]
        default: thinking.removeValue(forKey: "default")
        }
        chat["thinking"] = thinking
        chat["reasoning_effort"] = effort
        root["chat"] = chat
        let config = try JangLoader.parseConfig(from: root)
        #expect(config.chat?.reasoning == nil)
        #expect(config.chat?.templateKwargsDefaults == nil)
    }

    @Test(arguments: ["qwen3_5", "nemotron_h", "k2_horizon", "deepseek_v41"])
    func noNewDefaultsWithoutADeclaration(_ model: String) {
        #expect(
            llmDefaultAdditionalContext(
                modelType: model, capabilities: .init(supportsThinking: true),
                generationConfig: nil, chatConfig: nil) == nil)
        #expect(
            VLMDefaultContextUserInputProcessor.defaultContext(
                capabilities: .init(supportsThinking: true)) == nil)
    }
}
