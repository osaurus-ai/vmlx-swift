import Foundation
import Testing
import VMLXJinja

@testable import MLXLMCommon

@Suite("K2-Horizon native parser and history contract")
struct K2HorizonParserContractTests {
    @Test(.enabled(if: ProcessInfo.processInfo.environment["DIPLODOC_TEMPLATE"] != nil))
    func preview3TemplateMatchesPythonHighLowAndHistory() throws {
        let env = ProcessInfo.processInfo.environment
        let path = try #require(env["DIPLODOC_TEMPLATE"])
        let probes = try #require(env["DIPLODOC_PROBES"])
        let template = try Template(
            String(contentsOfFile: path, encoding: .utf8),
            with: .init(lstripBlocks: true, trimBlocks: true))
        let rows = try #require(JSONSerialization.jsonObject(
            with: Data(contentsOf: URL(fileURLWithPath: probes))) as? [[String: Any]])
        #expect(rows.count == 9)
        for row in rows {
            var context: [String: Any] = [
                "messages": row["messages"]!, "bos_token": "<|ifm|begin_of_text|>",
                "eos_token": "<|ifm|endoftext|>", "add_generation_prompt": true,
                "reasoning_effort": row["effort"]!, "tool_call_format": "xml",
            ]
            if let tools = row["tools"], !(tools is NSNull) { context["tools"] = tools }
            let rendered = try template.render(
                try context.mapValues { try VMLXJinja.Value(any: $0) })
            #expect(rendered == row["rendered"] as? String, "\(row["name"]!)")
        }
    }

    private let tools: [[String: any Sendable]] = {
        let account: [String: any Sendable] = ["type": "string"]
        let count: [String: any Sendable] = ["type": "integer"]
        let literal: [String: any Sendable] = ["type": ["string", "null"]]
        let properties: [String: any Sendable] = [
            "account": account, "count": count, "literal": literal,
        ]
        let parameters: [String: any Sendable] = ["type": "object", "properties": properties]
        let function: [String: any Sendable] = ["name": "lookup", "parameters": parameters]
        return [["type": "function", "function": function]]
    }()
    private func call(_ body: String) -> String {
        "<ifm|tool_calls><ifm|tool_call>\(body)</ifm|tool_call></ifm|tool_calls>"
    }
    private func argument(_ key: String, _ value: String, type: String? = nil) -> String {
        "<ifm|arg_key>\(key)</ifm|arg_key>"
            + (type.map { "<ifm|arg_type>\($0)</ifm|arg_type>" } ?? "")
            + "<ifm|arg_value>\(value)</ifm|arg_value>"
    }
    @Test func nativeXMLTypedJSONAndLiteralStrings() {
        let parser = K2HorizonToolCallParser()
        for type in [nil, "string"] as [String?] {
            for literal in [
                "007", "3.10", "null", "true", "  &amp;\\n <ifm|think>data</ifm|think>  ",
            ] {
                let text = call(
                    "lookup" + argument("account", literal, type: type)
                        + argument("count", "3", type: type == nil ? nil : "integer"))
                let parsed = parser.parse(content: text, tools: tools)
                #expect(parsed?.function.arguments["account"] == .string(literal))
                #expect(parsed?.function.arguments["count"] == .int(3))
            }
        }
        let json = call(#"{"name":"lookup","arguments":{"account":"007","count":3}}"#)
        #expect(
            parser.parse(content: json, tools: tools)?.function.arguments["account"]
                == .string("007"))
        #expect(
            parser.parse(content: call("lookup" + argument("literal", "null")), tools: tools)?
                .function.arguments["literal"] == .string("null"))
    }
    @Test func typedArraysAndFlatSchemasPreserveLiteralFields() {
        let parser = K2HorizonToolCallParser()
        let typed = call("lookup" + argument("items", "[1,2]", type: "array[integer]"))
        #expect(
            parser.parse(content: typed, tools: nil)?.function.arguments["items"]
                == .array([.int(1), .int(2)]))
        let flat = tools.compactMap { $0["function"] as? [String: any Sendable] }
        let string = call("lookup" + argument("account", "007", type: "integer"))
        #expect(
            parser.parse(content: string, tools: flat)?.function.arguments["account"]
                == .string("007"))
        let wrong = call("lookup" + argument("count", "true"))
        #expect(parser.parse(content: wrong, tools: tools) == nil)
    }

    @Test func repeatedLeadingGroupOpenersPreserveValuesAtEveryBoundary() {
        let value = "literal <ifm|tool_calls> remains data"
        let member = "<ifm|tool_call>lookup" + argument("account", value) + "</ifm|tool_call>"
        let text = "<ifm|tool_calls>\n<ifm|tool_calls>\n" + member + "</ifm|tool_calls>"
        for split in 0 ... text.count {
            let processor = ToolCallProcessor(format: .k2Horizon, tools: tools)
            let index = text.index(text.startIndex, offsetBy: split)
            var visible = processor.processChunk(String(text[..<index])) ?? ""
            visible += processor.processChunk(String(text[index...])) ?? ""
            visible += processor.processEOS() ?? ""
            #expect(visible.isEmpty)
            #expect(processor.toolCalls.count == 1)
            #expect(processor.toolCalls.first?.function.arguments["account"] == .string(value))
        }
        let parser = K2HorizonToolCallParser()
        // A repeated group AFTER a member is not the admitted prefix defect.
        let nested = "<ifm|tool_calls>" + member + "<ifm|tool_calls>" + member + "</ifm|tool_calls>"
        #expect(parser.parseEOS(nested, tools: tools).isEmpty)
        #expect(parser.parseEOS("<ifm|tool_calls>\n<ifm|tool_calls>" + member, tools: tools).isEmpty)
    }

    @Test func JSONQuotedProtocolMarkersAreDataAtEveryBoundary() throws {
        let literal = #"</ifm|tool_call></ifm|tool_calls><ifm|think>\""#
        let json = try JSONSerialization.data(withJSONObject: [
            "name": "lookup", "arguments": ["account": literal],
        ])
        let input = call(String(decoding: json, as: UTF8.self))
        for split in 0 ... input.count {
            let processor = ToolCallProcessor(format: .k2Horizon, tools: tools)
            let index = input.index(input.startIndex, offsetBy: split)
            var visible = processor.processChunk(String(input[..<index])) ?? ""
            visible += processor.processChunk(String(input[index...])) ?? ""
            visible += processor.processEOS() ?? ""
            #expect(visible.isEmpty)
            #expect(processor.toolCalls.count == 1)
            #expect(processor.toolCalls.first?.function.arguments["account"] == .string(literal))
        }
    }

    @Test func everyBoundaryReasoningAndMultipleCalls() throws {
        let payload = "  <ifm|think>literal</ifm|think>  "
        let block =
            "<ifm|tool_calls><ifm|tool_call>lookup" + argument("account", payload)
            + "</ifm|tool_call><ifm|tool_call>{\"name\":\"lookup\",\"arguments\":{\"account\":\"007\"}}</ifm|tool_call></ifm|tool_calls>"
        for closer in ["</ifm|think>", "</ifm|think_fast>", "</ifm|think_faster>"] {
            let input = "private work" + closer + "answer" + block + "tail"
            for split in 0 ... input.count {
                var reasoning = try #require(ReasoningParser.fromCapabilityName("k2_horizon"))
                let processor = ToolCallProcessor(format: .k2Horizon, tools: tools)
                var thought = ""
                var visible = ""
                let index = input.index(input.startIndex, offsetBy: split)
                var segments = reasoning.feed(String(input[..<index]))
                segments += reasoning.feed(String(input[index...]))
                segments += reasoning.flush()
                for segment in segments {
                    switch segment {
                    case .reasoning(let text): thought += text
                    case .content(let text): visible += processor.processChunk(text) ?? ""
                    }
                }
                visible += processor.processEOS() ?? ""
                #expect(thought == "private work")
                #expect(visible == "answertail")
                #expect(processor.toolCalls.count == 2)
                #expect(
                    processor.toolCalls.first?.function.arguments["account"] == .string(payload))
            }
        }
    }
    @Test func malformedAndTruncatedCallsNeverExecuteOrLeak() {
        let valid = call("lookup" + argument("account", "007"))
        let invalid =
            [
                call("lookup<ifm|arg_key>x</ifm|arg_key>"),
                call("lookup" + argument("account", "x") + argument("account", "y")),
                call("lookup garbage"),
                call(#"{"name":"lookup","arguments":[]}"#),
                "<ifm|tool_calls><ifm|tool_call>lookup</ifm|tool_call><ifm|tool_call>broken",
            ] + (15 ..< valid.count).map { String(valid.prefix($0)) }
        for text in invalid {
            for split in 0 ... text.count {
                let processor = ToolCallProcessor(format: .k2Horizon, tools: tools)
                let index = text.index(text.startIndex, offsetBy: split)
                var visible = processor.processChunk(String(text[..<index])) ?? ""
                visible += processor.processChunk(String(text[index...])) ?? ""
                visible += processor.processEOS() ?? ""
                #expect(visible.isEmpty)
                #expect(processor.toolCalls.isEmpty)
            }
        }
    }
    private func fixedReasoningMetadata(effort: String = "high") throws -> Data {
        try JSONSerialization.data(withJSONObject: [
            "capabilities": ["supports_reasoning_toggle": false, "reasoning_efforts": []]
                as [String: Any],
            "chat": [
                "reasoning": [
                    "supported": true, "default_mode": "think", "modes": ["think"],
                    "template_flag": "reasoning_effort",
                    "mode_kwargs": ["think": ["reasoning_effort": effort]],
                ]
            ] as [String: Any],
        ])
    }

    @Test func reasoningValidationUsesBundleDeclarationInsteadOfFamilyDefault() throws {
        let high = try fixedReasoningMetadata()
        let highDeclaration = K2HorizonTemplateContract.fixedReasoningDeclaration(
            metadata: high, modelType: "k2_horizon")
        #expect(highDeclaration?.effort == "high")
        #expect(
            K2HorizonTemplateContract.fixedReasoningDeclaration(
                metadata: nil, modelType: "k2_horizon") == nil)
        #expect(
            K2HorizonTemplateContract.fixedReasoningDeclaration(metadata: high, modelType: "qwen3")
                == nil)
        try K2HorizonTemplateContract.validateContext(
            ["reasoning_effort": "medium"], modelType: "k2_horizon", declaration: nil)
        try K2HorizonTemplateContract.validateContext(
            ["reasoning_effort": "medium"], modelType: "qwen3", declaration: highDeclaration)
        let future = K2HorizonTemplateContract.fixedReasoningDeclaration(
            metadata: try fixedReasoningMetadata(effort: "medium"), modelType: "k2_horizon")
        #expect(future?.effort == "medium")
        try K2HorizonTemplateContract.validateContext(
            ["reasoning_effort": "medium"], modelType: "k2_horizon", declaration: future)
        #expect(throws: K2HorizonTemplateContract.ContractError.self) {
            try K2HorizonTemplateContract.validateContext(
                ["reasoning_effort": "high"], modelType: "k2_horizon", declaration: future)
        }
    }

    @Test func historyPreservesReasoningAndNativeHighOnly() throws {
        let source: [Message] = [
            ["role": "assistant", "content": "A", "reasoning_content": "actual"],
            ["role": "assistant", "content": "B"],
            ["role": "assistant", "content": "C", "think": "alias"],
        ]
        let result = K2HorizonTemplateContract.prepare(messages: source, modelType: "k2_horizon")
        #expect(result[0]["reasoning_content"] as? String == "actual")
        #expect(result[1]["reasoning_content"] as? String == "")
        #expect(result[2]["think"] as? String == "alias")
        #expect(result[2]["reasoning_content"] == nil)
        #expect(
            K2HorizonTemplateContract.prepare(messages: source, modelType: "qwen3")[1][
                "reasoning_content"] == nil)
        let declaration = K2HorizonTemplateContract.fixedReasoningDeclaration(
            metadata: try fixedReasoningMetadata(), modelType: "k2_horizon")
        try K2HorizonTemplateContract.validateContext(
            nil, modelType: "k2_horizon", declaration: declaration)
        try K2HorizonTemplateContract.validateContext(
            ["reasoning_effort": "high"], modelType: "k2_horizon", declaration: declaration)
        #expect(throws: K2HorizonTemplateContract.ContractError.self) {
            try K2HorizonTemplateContract.validateContext(
                ["reasoning_effort": "low"], modelType: "k2_horizon", declaration: declaration)
        }
        #expect(throws: K2HorizonTemplateContract.ContractError.self) {
            try K2HorizonTemplateContract.validateContext(
                ["enable_thinking": false], modelType: "k2_horizon", declaration: declaration)
        }
        #expect(ToolCallFormat.infer(from: "k2_horizon") == .k2Horizon)
        #expect(ToolCallFormat.fromCapabilityName("k2_horizon") == .k2Horizon)
        #expect(reasoningStampFromModelType("k2_horizon") == "k2_horizon")
        #expect(!ToolCallFormat.k2Horizon.parsesToolCallsFromReasoningChannel)
    }

    /// Local qualification opts in explicitly; an enabled row requires its fixture.
    @Test(.enabled(if: ProcessInfo.processInfo.environment["K2_HORIZON_TEMPLATE"] != nil))
    func installedNativeTemplateRendersHistoryAndAllToolFormats() throws {
        let path = try #require(ProcessInfo.processInfo.environment["K2_HORIZON_TEMPLATE"])
        let template = try Template(String(contentsOfFile: path, encoding: .utf8))
        let metadataURL = URL(fileURLWithPath: path).deletingLastPathComponent()
            .appendingPathComponent("jang_config.json")
        let metadata = try Data(contentsOf: metadataURL)
        let declaration = try #require(
            K2HorizonTemplateContract.fixedReasoningDeclaration(
                metadata: metadata, modelType: "k2_horizon"))
        #expect(declaration.effort == "high")
        let arguments: [String: any Sendable] = ["account": "007"]
        let function: [String: any Sendable] = ["name": "lookup", "arguments": arguments]
        let toolCalls: [[String: any Sendable]] = [["function": function]]
        for format in ["xml", "xml_typed", "json"] {
            let messages: [Message] = [
                ["role": "user", "content": "lookup"],
                [
                    "role": "assistant", "content": "",
                    "reasoning_content": "retain this reasoning",
                    "tool_calls": toolCalls,
                ],
                ["role": "tool", "content": "found"], ["role": "user", "content": "next"],
            ]
            let context: [String: Any] = [
                "messages": messages, "tools": tools,
                "bos_token": "", "add_generation_prompt": true, "tool_call_format": format,
            ]
            let rendered = try template.render(
                try context.mapValues { try VMLXJinja.Value(any: $0) })
            #expect(rendered.contains("retain this reasoning</ifm|think>"))
            #expect(rendered.contains("007"))
            #expect(rendered.contains("<|ifm|im_start|>tool\nfound<|ifm|im_end|>"))
            #expect(rendered.hasSuffix("<|ifm|im_start|>assistant\n<ifm|think>\n"))
        }
    }
}
