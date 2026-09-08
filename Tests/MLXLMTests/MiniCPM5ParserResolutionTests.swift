import Testing
@testable import MLXLMCommon

@Suite("MiniCPM5 template-derived parser resolution")
struct MiniCPM5ParserResolutionTests {
    @Test func dialectNotArchitectureSelectsParser() {
        let template = #"<function name="f"><param name="s"><![CDATA[x]]></param></function>"#
        let capabilities = JangCapabilities(toolParser: "llama")
        #expect(ParserResolution.toolCall(capabilities: capabilities, modelType: "llama", chatTemplate: template).format == .minicpm5)
        #expect(ParserResolution.toolCall(capabilities: nil, modelType: "llama", chatTemplate: template).format == .minicpm5)
        #expect(ParserResolution.toolCall(capabilities: nil, modelType: "llama", chatTemplate: "plain prose").format == nil)
        // Explicit non-generic publisher declarations retain priority.
        #expect(ParserResolution.toolCall(capabilities: .init(toolParser: "dsml"), modelType: "llama", chatTemplate: template).format == .dsml)
        let reasoningTemplate = template + "<think></think> enable_thinking"
        let reasoning = ParserResolution.reasoning(capabilities: capabilities, modelType: "llama", chatTemplate: reasoningTemplate)
        #expect(reasoning.parser?.preservesXMLFunctionPayloads == true)
        #expect(reasoning.source == .chatTemplate)
        #expect(ParserResolution.reasoning(capabilities: .init(reasoningParser: "none"), modelType: "llama", chatTemplate: reasoningTemplate).parser == nil)
    }
}
