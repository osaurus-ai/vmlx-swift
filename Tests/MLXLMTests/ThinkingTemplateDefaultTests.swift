import Testing
@testable import MLXLLM
import MLXLMCommon

@Suite("Native omitted thinking reaches processor defaults")
struct ThinkingTemplateDefaultTests {
    private let native = """
        {% if add_generation_prompt %}<|im_start|>assistant
        {% if enable_thinking is defined %}
        {% if enable_thinking is false %}<think>\n\n</think>\n\n
        {% elif enable_thinking is true %}<think>\n{% endif %}
        {% endif %}{% endif %}
        """

    @Test func nativeOmissionWinsOverStaleCapability() {
        let context = llmDefaultAdditionalContext(
            modelType: "llama", capabilities: .init(supportsThinking: false),
            generationConfig: nil, chatConfig: nil, chatTemplate: native)
        #expect(context == nil)
    }

    @Test(arguments: [true, false]) func declaredDefaultsRemainEffective(_ value: Bool) throws {
        let context = try #require(llmDefaultAdditionalContext(
            modelType: "llama", capabilities: .init(supportsThinking: false),
            generationConfig: .init(defaultChatTemplateKwargs: .init(enableThinking: value)),
            chatConfig: nil, chatTemplate: native))
        #expect(context["enable_thinking"] as? Bool == value)
    }

    @Test(arguments: [true, false]) func explicitRequestsWin(_ value: Bool) throws {
        let defaults = llmDefaultAdditionalContext(
            modelType: "llama", capabilities: .init(supportsThinking: false),
            generationConfig: nil, chatConfig: nil, chatTemplate: native)
        let resolved = try llmMergedAdditionalContext(
            defaultAdditionalContext: defaults,
            requestAdditionalContext: ["enable_thinking": value], modelType: "llama")
        let merged = try #require(resolved)
        #expect(merged["enable_thinking"] as? Bool == value)
    }

    @Test func legacyCapabilityAndExplicitOnlyDetection() throws {
        #expect(ThinkingTemplateContract.preservesOmittedThinking(native))
        for template in ["plain", native.replacingOccurrences(of: "{% endif %}{% endif %}", with: "{% else %}<think>{% endif %}{% endif %}"),
                         "{% set enable_thinking = false %}" + native] {
            #expect(!ThinkingTemplateContract.preservesOmittedThinking(template))
            let context = try #require(llmDefaultAdditionalContext(
                modelType: "example", capabilities: .init(supportsThinking: false),
                generationConfig: .init(defaultChatTemplateKwargs: .init(enableThinking: true)),
                chatConfig: nil, chatTemplate: template))
            #expect(context["enable_thinking"] as? Bool == false)
        }
    }
}
