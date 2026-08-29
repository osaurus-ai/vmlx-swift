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

    // MARK: - Mis-stamped `tool_parser: "hermes"` recovery (2026-08-28)
    //
    // A batch of published Qwen 3.8 JANG bundles was accidentally stamped
    // `capabilities.tool_parser = "hermes"` (a vLLM parser name) instead of
    // `"qwen"`. The whole safety story rests on "hermes" being UNRECOGNISED:
    // the stamp resolves nil, so the ladder falls through to model_type /
    // chat-template detection instead of trusting a wrong format. These pin
    // that recovery so a future "hermes" alias can never silently flip it
    // into an authoritative-but-wrong `.json` stamp for coder-style bundles.

    @Test("'hermes' is not a recognised capability alias")
    func hermesStampIsUnrecognised() {
        #expect(ToolCallFormat.fromCapabilityName("hermes") == nil)
        #expect(ToolCallFormat.fromCapabilityName("Hermes") == nil)
        #expect(ToolCallFormat.fromCapabilityName("hermes_json") == nil)
    }

    @Test("hermes-stamped qwen3_5 bundle recovers .xmlFunction via model_type")
    func hermesStampedQwen35Recovers() {
        // Qwen3.8-27B bundles: model_type=qwen3_5. The bad stamp must fall
        // through to the model_type heuristic, which knows the family.
        let r = ParserResolution.toolCall(
            capabilities: JangCapabilities(toolParser: "hermes"),
            modelType: "qwen3_5",
            chatTemplate: qwen3CoderTemplate)
        #expect(r.format == .xmlFunction)
        #expect(r.source == .modelTypeHeuristic)
    }

    @Test("hermes-stamped qwen4_exp bundle recovers .xmlFunction via template")
    func hermesStampedQwen4ExpRecovers() {
        // Qwen3.8 Flash-Next bundles: model_type=qwen4_exp, which the
        // heuristic does not know — the bundle's own chat template (real
        // envelope: <function=/<parameter=) is the ground truth.
        let r = ParserResolution.toolCall(
            capabilities: JangCapabilities(toolParser: "hermes"),
            modelType: "qwen4_exp",
            chatTemplate: qwen3CoderTemplate)
        #expect(r.format == .xmlFunction)
        #expect(r.source == .chatTemplate)
    }

    @Test("hermes-stamped genuine Hermes-style instruct still lands .json")
    func hermesStampedInstructStillJson() {
        // A bundle whose template actually IS the Hermes bare-JSON envelope
        // (plain Qwen3 instruct): the wrong stamp does no harm there either.
        let r = ParserResolution.toolCall(
            capabilities: JangCapabilities(toolParser: "hermes"),
            modelType: "qwen3",
            chatTemplate: qwen3InstructTemplate)
        #expect(r.format == .json)
        #expect(r.source == .chatTemplate)
    }

    @Test("the CORRECTED stamps resolve .xmlFunction authoritatively")
    func correctedStampsResolve() {
        // The repo fix restamps `qwen` (and converter-era bundles carry
        // `qwen3_coder`) — both must resolve directly, template not needed.
        for stamp in ["qwen", "qwen3_coder", "qwen3_5"] {
            let r = ParserResolution.toolCall(
                capabilities: JangCapabilities(toolParser: stamp),
                modelType: "qwen3_5",
                chatTemplate: nil)
            #expect(r.format == .xmlFunction, "stamp '\(stamp)' must resolve")
            #expect(r.source == .jangStamped)
        }
    }
}
