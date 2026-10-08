import Foundation
import Testing

@testable import MLXLMCommon

/// A math-expression argument (Osaurus `calculate(expression:)`) is a single
/// string full of characters several wire formats treat as syntax: `=` and
/// `;` (assignment-style parsers), `,` inside parentheses (pythonic argument
/// lists), `%`, `^`, `×`/`÷`, newlines, and numeric-looking text that a
/// type-coercing parser could turn into an integer. Every format must hand the
/// tool the exact string the model wrote, through both the one-shot parsers
/// and the streaming processor split at every boundary.
@Suite("Math expression tool arguments")
struct MathExpressionToolArgumentTests {

    static let tool: [String: any Sendable] = [
        "type": "function",
        "function": [
            "name": "calculate",
            "description": "Evaluate math exactly.",
            "parameters": [
                "type": "object",
                "properties": [
                    "expression": ["type": "string"] as [String: any Sendable],
                    "angle_unit": ["type": "string", "enum": ["radians", "degrees"]] as [String: any Sendable],
                ] as [String: any Sendable],
                "required": ["expression"],
            ] as [String: any Sendable],
        ] as [String: any Sendable],
    ]

    static let expressions = [
        "x^2 - 5x + 6 = 0",
        "r = 3; pi * r^2",
        "15% of 80 × 3 ÷ 2",
        "max(2, 3) * sqrt(16)",
        "a = 3\nb = 4\nsqrt(a^2 + b^2)",
        "1250000",
        "007",
        "2*(3+4)/5",
    ]

    static let dsml = DeepseekV4Tokens.dsml

    static func jsonString(_ s: String) -> String {
        let data = try! JSONSerialization.data(withJSONObject: [s], options: [.withoutEscapingSlashes])
        let array = String(decoding: data, as: UTF8.self)
        return String(array.dropFirst().dropLast())
    }

    static func pythonString(_ s: String) -> String {
        "\"" + s.replacingOccurrences(of: "\\", with: "\\\\").replacingOccurrences(of: "\"", with: "\\\"")
            .replacingOccurrences(of: "\n", with: "\\n") + "\""
    }

    /// The text each format's chat template teaches the model to emit.
    static func wire(_ format: ToolCallFormat, _ e: String) -> String {
        let jsonArgs = "{\"expression\": \(jsonString(e))}"
        switch format {
        case .json:
            return "<tool_call>{\"name\": \"calculate\", \"arguments\": \(jsonArgs)}</tool_call>"
        case .mimo:
            // MiMo's native template adds no framing newlines inside parameter tags.
            return "<tool_call><function=calculate><parameter=expression>\(e)</parameter></function></tool_call>"
        case .xmlFunction, .step, .nemotron:
            return "<tool_call>\n<function=calculate>\n<parameter=expression>\n\(e)\n</parameter>\n</function>\n</tool_call>"
        case .zayaXml:
            return "<zyphra_tool_call>\n<function=calculate>\n<parameter=expression>\n\(e)\n</parameter>\n</function>\n</zyphra_tool_call>"
        case .glm4:
            return "<tool_call>calculate<arg_key>expression</arg_key><arg_value>\(e)</arg_value></tool_call>"
        case .hunyuan:
            return "<tool_calls>\n<tool_call>calculate<tool_sep>\n<arg_key>expression</arg_key><arg_value>\(jsonString(e))</arg_value>\n</tool_call>\n</tool_calls>"
        case .minimaxM2:
            return "<minimax:tool_call>\n<invoke name=\"calculate\">\n<parameter name=\"expression\">\(e)</parameter>\n</invoke>\n</minimax:tool_call>"
        case .minicpm5:
            return "<function name=\"calculate\"><param name=\"expression\">\(e)</param></function>"
        case .atem:
            return "<atem:function_calls>\n<atem:invoke name=\"calculate\">\n<atem:parameter name=\"expression\">\(e)</atem:parameter>\n</atem:invoke>\n</atem:function_calls>"
        case .dsml:
            return "<\(dsml)tool_calls>\n<\(dsml)invoke name=\"calculate\">\n<\(dsml)parameter name=\"expression\" string=\"true\">\(e)</\(dsml)parameter>\n</\(dsml)invoke>\n</\(dsml)tool_calls>"
        case .kimiK2:
            return "<|tool_calls_section_begin|><|tool_call_begin|>functions.calculate:0<|tool_call_argument_begin|>\(jsonArgs)<|tool_call_end|><|tool_calls_section_end|>"
        case .k2Horizon:
            return "<ifm|tool_calls><ifm|tool_call>calculate<ifm|arg_key>expression</ifm|arg_key><ifm|arg_value>\(e)</ifm|arg_value></ifm|tool_call></ifm|tool_calls>"
        case .llama3:
            return "{\"name\": \"calculate\", \"parameters\": \(jsonArgs)}"
        case .mistral:
            return "[TOOL_CALLS]calculate[ARGS]\(jsonArgs)"
        case .lfm2:
            return "<|tool_call_start|>[calculate(expression=\(pythonString(e)))]<|tool_call_end|>"
        case .gemma:
            return "<start_function_call>call:calculate{expression:<escape>\(e)<escape>}<end_function_call>"
        case .gemma4:
            return "<|tool_call>call:calculate{expression:<|\"|>\(e)<|\"|>}<tool_call|>"
        }
    }

    static var cases: [(ToolCallFormat, String)] {
        ToolCallFormat.allCases.flatMap { format in expressions.map { (format, $0) } }
    }

    @Test("one-shot parsers keep the expression byte-exact", arguments: cases)
    func oneShot(_ sample: (format: ToolCallFormat, expression: String)) {
        let parser = sample.format.createParser()
        let text = Self.wire(sample.format, sample.expression)
        let call = parser.parse(content: text, tools: [Self.tool]) ?? parser.parseEOS(text, tools: [Self.tool]).first
        #expect(call?.function.name == "calculate", "\(sample.format.rawValue): no call parsed from \(text)")
        #expect(
            call?.function.arguments["expression"] == .string(sample.expression),
            "\(sample.format.rawValue): \(String(describing: call?.function.arguments["expression"])) != \(sample.expression)")
    }

    /// Token-sized streaming: one character per chunk.
    @Test("streaming processor keeps the expression one character at a time", arguments: ToolCallFormat.allCases)
    func streamedPerCharacter(_ format: ToolCallFormat) {
        for expression in Self.expressions {
            let text = Self.wire(format, expression)
            let processor = ToolCallProcessor(format: format, tools: [Self.tool])
            var visible = ""
            for character in text { visible += processor.processChunk(String(character)) ?? "" }
            visible += processor.processEOS() ?? ""
            #expect(processor.toolCalls.count == 1, "\(format.rawValue): \(processor.toolCalls.count) calls")
            #expect(
                processor.toolCalls.first?.function.arguments["expression"] == .string(expression),
                "\(format.rawValue): \(String(describing: processor.toolCalls.first?.function.arguments["expression"]))")
            #expect(visible.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty, "\(format.rawValue) leaked: \(visible)")
        }
    }

    /// The live path: the processor sees the call in streamed pieces. Split at
    /// every character boundary so a format whose tag/argument scanning breaks
    /// mid-token is caught.
    @Test("streaming processor keeps the expression at every split", arguments: ToolCallFormat.allCases)
    func streamed(_ format: ToolCallFormat) {
        for expression in Self.expressions {
            let text = Self.wire(format, expression)
            for split in stride(from: 0, through: text.count, by: max(1, text.count / 40)) {
                let processor = ToolCallProcessor(format: format, tools: [Self.tool])
                let index = text.index(text.startIndex, offsetBy: split)
                var visible = processor.processChunk(String(text[..<index])) ?? ""
                visible += processor.processChunk(String(text[index...])) ?? ""
                visible += processor.processEOS() ?? ""
                #expect(processor.toolCalls.count == 1, "\(format.rawValue) split \(split): \(processor.toolCalls.count) calls")
                #expect(
                    processor.toolCalls.first?.function.arguments["expression"] == .string(expression),
                    "\(format.rawValue) split \(split): \(String(describing: processor.toolCalls.first?.function.arguments["expression"]))")
                #expect(
                    visible.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
                    "\(format.rawValue) split \(split) leaked markup: \(visible)")
            }
        }
    }
}
