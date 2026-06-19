import Foundation
import MLXLMCommon
import Testing

/// Regression coverage for non-JANG Qwen3 tool-call detection.
///
/// Symptom (2026-06-19, reporter tpae): a plain Qwen3 bundle errored with
/// "tool calling as unsupported" because `ParserResolution.toolCall` skipped the
/// chat-template signal for non-JANG models and `ToolCallFormat.infer("qwen3" /
/// "qwen3_moe")` returns nil (qwen3_moe is shared by instruct=Hermes/.json and
/// Qwen3-Coder=.xmlFunction, so model_type alone cannot disambiguate).
///
/// The fragments below are the REAL tool-call envelopes copied from upstream
/// Hugging Face `chat_template`s:
///   - Qwen/Qwen3-4B (model_type=qwen3) and Qwen/Qwen3-30B-A3B (qwen3_moe):
///       <tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>
///   - Qwen/Qwen3-Coder-30B-A3B-Instruct (qwen3_moe):
///       <tool_call>\n<function=name>\n<parameter=key>\nvalue\n</parameter>\n</function>\n</tool_call>
struct Qwen3ToolFormatTemplateDetectTests {
    private let qwen3InstructTemplate = """
        You are a helpful assistant. For each function call return a json object \
        with function name and arguments within <tool_call></tool_call> XML tags:
        <tool_call>
        {"name": <function-name>, "arguments": <args-json-object>}
        </tool_call>
        """

    private let qwen3CoderTemplate = """
        Make tool calls inside <tool_call> tags using the function/parameter form:
        <tool_call>
        <function=example_function_name>
        <parameter=example_parameter_1>
        value_1
        </parameter>
        </function>
        </tool_call>
        """

    @Test("qwen3 dense instruct template → .json via chatTemplate")
    func qwen3DenseInstruct() {
        let r = ParserResolution.toolCall(
            capabilities: nil, modelType: "qwen3", chatTemplate: qwen3InstructTemplate)
        #expect(r.format == .json)
        #expect(r.source == .chatTemplate)
    }

    @Test("qwen3_moe instruct template → .json via chatTemplate")
    func qwen3MoeInstruct() {
        let r = ParserResolution.toolCall(
            capabilities: nil, modelType: "qwen3_moe", chatTemplate: qwen3InstructTemplate)
        #expect(r.format == .json)
        #expect(r.source == .chatTemplate)
    }

    @Test("qwen3_moe coder template → .xmlFunction via chatTemplate")
    func qwen3MoeCoder() {
        let r = ParserResolution.toolCall(
            capabilities: nil, modelType: "qwen3_moe", chatTemplate: qwen3CoderTemplate)
        #expect(r.format == .xmlFunction)
        #expect(r.source == .chatTemplate)
    }

    @Test("recognised model_type still wins over template (no behaviour change)")
    func recognisedModelTypeUnchanged() {
        // qwen2 is resolved by the model_type heuristic; the template fallback must
        // not be consulted (source stays .modelTypeHeuristic).
        let r = ParserResolution.toolCall(
            capabilities: nil, modelType: "qwen2", chatTemplate: qwen3CoderTemplate)
        #expect(r.format == .json)
        #expect(r.source == .modelTypeHeuristic)
    }

    @Test("no template + unrecognised model_type stays unresolved")
    func unresolvedWithoutTemplate() {
        let r = ParserResolution.toolCall(
            capabilities: nil, modelType: "qwen3", chatTemplate: nil)
        #expect(r.format == nil)
        #expect(r.source == .none)
    }
}
