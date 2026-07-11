import Foundation
@testable import MLXLMCommon
import Testing

@Suite("Hy3 parser and capability dispatch")
struct Hy3ParserDispatchTests {
    @Test("Hy3 tool parser aliases resolve to Hunyuan parser")
    func hy3ToolParserAliasesResolveToHunyuan() throws {
        for stamp in ["hunyuan", "tencent", "hy3", "hy_v3", "hy-v3"] {
            #expect(ToolCallFormat.fromCapabilityName(stamp) == .hunyuan)
        }
        for modelType in ["hy_v3", "hy3", "hy-v3"] {
            #expect(ToolCallFormat.infer(from: modelType) == .hunyuan)
        }
        #expect(ToolCallFormat.hunyuan.createParser() is HunyuanToolCallParser)
    }

    @Test("Hy3 reasoning parser aliases resolve to think XML")
    func hy3ReasoningParserAliasesResolveToThinkXML() throws {
        for stamp in ["hy_v3", "hy-v3", "hy3", "hunyuan", "tencent"] {
            #expect(ReasoningParser.fromCapabilityName(stamp) != nil)
        }
        for modelType in ["hy_v3", "hy-v3", "hy3", "Hy3"] {
            #expect(reasoningStampFromModelType(modelType) == "hy_v3")
        }
    }

    @Test("Hy3 Hunyuan parser extracts multiple calls and scalar arguments")
    func hy3HunyuanParserExtractsMultipleCalls() throws {
        let parser = HunyuanToolCallParser()
        let searchProperties: [String: any Sendable] = [
            "query": ["type": "string"] as [String: any Sendable],
            "limit": ["type": "integer"] as [String: any Sendable],
            "safe": ["type": "boolean"] as [String: any Sendable],
        ]
        let searchParameters: [String: any Sendable] = [
            "type": "object",
            "properties": searchProperties,
        ]
        let searchFunction: [String: any Sendable] = [
            "name": "search_web",
            "parameters": searchParameters,
        ]
        let openProperties: [String: any Sendable] = [
            "path": ["type": "string"] as [String: any Sendable],
        ]
        let openParameters: [String: any Sendable] = [
            "type": "object",
            "properties": openProperties,
        ]
        let openFunction: [String: any Sendable] = [
            "name": "open_file",
            "parameters": openParameters,
        ]
        let tools: [[String: any Sendable]] = [
            [
                "type": "function",
                "function": searchFunction,
            ],
            [
                "type": "function",
                "function": openFunction,
            ],
        ]

        let calls = parser.parseEOS(
            """
            <tool_calls>
            <tool_call>search_web<tool_sep>
            <arg_key>query</arg_key><arg_value>"hy3 runtime"</arg_value>
            <arg_key>limit</arg_key><arg_value>3</arg_value>
            <arg_key>safe</arg_key><arg_value>true</arg_value>
            </tool_call>
            <tool_call>open_file<tool_sep>
            <arg_key>path</arg_key><arg_value>"/tmp/a b.txt"</arg_value>
            </tool_call>
            </tool_calls>
            """,
            tools: tools)

        #expect(calls.count == 2)
        #expect(calls[0].function.name == "search_web")
        #expect(calls[0].function.arguments["query"] == .string("hy3 runtime"))
        #expect(calls[0].function.arguments["limit"] == .int(3))
        #expect(calls[0].function.arguments["safe"] == .bool(true))
        #expect(calls[1].function.name == "open_file")
        #expect(calls[1].function.arguments["path"] == .string("/tmp/a b.txt"))
    }

    @Test("Hy3 JANG capability kwargs resolve to reasoning and Hunyuan tool parser")
    func hy3JangCapabilityKwargsResolve() throws {
        let cfg = try JangLoader.parseConfig(from: [
            "format": "jang",
            "format_version": "2.0",
            "model_family": "hy_v3",
            "capabilities": [
                "family": "hy_v3",
                "reasoning_parser": "qwen3",
                "tool_parser": "hunyuan",
                "supports_tools": true,
                "supports_thinking": true,
                "think_in_template": true,
                "modality": "text",
                "cache_type": "kv",
            ] as [String: Any],
            "chat": [
                "reasoning": [
                    "supported": true,
                    "default_mode": "high",
                    "reasoning_effort_levels": ["no_think", "low", "high"],
                ] as [String: Any],
                "tool_calling": [
                    "supported": true,
                    "parser": "hunyuan",
                ] as [String: Any],
            ] as [String: Any],
        ])

        #expect(cfg.modelFamily == "hy_v3")
        #expect(cfg.capabilities?.supportsThinking == true)
        #expect(cfg.capabilities?.supportsTools == true)
        #expect(cfg.capabilities?.thinkInTemplate == true)
        #expect(cfg.capabilities?.cacheType == "kv")
        #expect(ReasoningParser.fromCapabilityName(cfg.capabilities?.reasoningParser) != nil)
        #expect(ToolCallFormat.fromCapabilityName(cfg.capabilities?.toolParser) == .hunyuan)
        #expect(cfg.chat?.reasoning?.reasoningEffortLevels == ["no_think", "low", "high"])
        #expect(ToolCallFormat.fromCapabilityName(cfg.chat?.toolCalling?.parser) == .hunyuan)
    }

    @Test("Hy3 reasoning and Hunyuan tool-call pipeline does not leak markers")
    func hy3ReasoningAndHunyuanPipelineDoesNotLeakMarkers() throws {
        var reasoningParser = ReasoningParser.fromCapabilityName("qwen3")
        let toolProcessor = ToolCallProcessor(format: .hunyuan)
        let stream = """
            <think>choose the lookup tool</think>
            <tool_calls>
            <tool_call>search_web<tool_sep>
            <arg_key>query</arg_key><arg_value>"hy3 swift"</arg_value>
            </tool_call>
            <tool_call>open_file<tool_sep>
            <arg_key>path</arg_key><arg_value>"/tmp/hy3.md"</arg_value>
            </tool_call>
            </tool_calls>
            Final answer after tools.
            """

        var visible = ""
        var reasoning = ""
        for scalar in stream {
            if var parser = reasoningParser {
                for segment in parser.feed(String(scalar)) {
                    switch segment {
                    case .reasoning(let text):
                        reasoning += text
                    case .content(let text):
                        if let chunk = toolProcessor.processChunk(text) {
                            visible += chunk
                        }
                    }
                }
                reasoningParser = parser
            }
        }
        if var parser = reasoningParser {
            for segment in parser.flush() {
                switch segment {
                case .reasoning(let text):
                    reasoning += text
                case .content(let text):
                    if let chunk = toolProcessor.processChunk(text) {
                        visible += chunk
                    }
                }
            }
        }
        toolProcessor.processEOS()

        #expect(reasoning.contains("choose the lookup tool"))
        #expect(toolProcessor.toolCalls.map(\.function.name) == ["search_web", "open_file"])
        #expect(toolProcessor.toolCalls[0].function.arguments["query"] == .string("hy3 swift"))
        #expect(toolProcessor.toolCalls[1].function.arguments["path"] == .string("/tmp/hy3.md"))
        #expect(visible.contains("Final answer after tools."))
        #expect(!visible.contains("<think>"))
        #expect(!visible.contains("</think>"))
        #expect(!visible.contains("choose the lookup tool"))
        #expect(!visible.contains("<tool_calls>"))
        #expect(!visible.contains("<tool_call>"))
        #expect(!visible.contains("<arg_key>"))
        #expect(!visible.contains("<arg_value>"))
    }
}
