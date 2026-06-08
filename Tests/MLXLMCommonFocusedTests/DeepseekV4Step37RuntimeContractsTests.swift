// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLXLMCommon
import Testing

@Suite("DSV4 step 3.7 runtime contracts", .serialized)
struct DeepseekV4Step37RuntimeContractsTests {
    @Test("DSV4 native composite cache uses disk L2 and skips generic paged KV")
    func dsv4NativeCompositeCacheUsesDiskL2AndSkipsGenericPagedKV() throws {
        let model = try Self.source("Libraries/MLXLLM/Models/DeepseekV4.swift")
        let serializer = try Self.source("Libraries/MLXLMCommon/Cache/TQDiskSerializer.swift")
        let quantize = try Self.source("Libraries/MLXLMCommon/BatchEngine/BatchQuantize.swift")
        let coordinator = try Self.source("Libraries/MLXLMCommon/Cache/CacheCoordinator.swift")

        #expect(model.contains("`cr > 0`  (every other layer) → `DeepseekV4Cache(window=128, cr=cr)`"))
        #expect(model.contains("Caller-level `GenerateParameters.kvMode = .turboQuant` is\n    /// intentionally NOT enough to switch DSV4 into `\"tq\"` mode."))
        #expect(model.contains("case \"full\", \"tq\":\n                return KVCacheSimple()"))
        #expect(model.contains("return DeepseekV4Cache(\n                        slidingWindow: config.slidingWindow,\n                        compressRatio: cr)"))

        #expect(quantize.contains("Preserves `RotatingKVCache`, `DeepseekV4Cache`, `MambaCache`,"))
        #expect(quantize.contains("DSV4's `DeepseekV4Cache` is also\n///   skipped"))
        #expect(quantize.contains("unless the model was explicitly loaded\n///   with `DSV4_KV_MODE=tq`"))

        #expect(serializer.contains("dsv4_{i}_pool_comp"))
        #expect(serializer.contains("dsv4_\\(i)_pool_comp"))
        #expect(serializer.contains("dsv4_\\(i)_pool_idx"))
        #expect(serializer.contains("dsv4_\\(i)_buf_comp_kv"))
        #expect(serializer.contains("deserializeDeepseekV4Layer"))

        #expect(coordinator.contains("isPagedIncompatible"))
        #expect(coordinator.contains("let skipPaged = isPagedIncompatible"))
        #expect(coordinator.contains("if !skipPaged,\n           let pagedCache"))
        #expect(coordinator.contains("if !isPagedIncompatible, let pagedCache"))
        #expect(coordinator.contains("if config.enableDiskCache"))
        #expect(!coordinator.contains("if config.defaultKVMode"))
    }

    @Test("DSV4 required tools keep ordinary assistant tail and schema-aware parser fallbacks")
    func dsv4RequiredToolsKeepOrdinaryAssistantTailAndSchemaAwareParserFallbacks() {
        let rendered = DeepseekV4ChatEncoder().encode(
            messages: [
                .init(role: .system, content: "", tools: [Self.lineCountToolSpec()]),
                .init(role: .user, content: "Use line_count on red\ngreen\nblue."),
                .init(
                    role: .assistant,
                    toolCalls: [
                        .init(
                            id: "call_lines",
                            name: "line_count",
                            arguments: #"{"text":"red\ngreen\nblue"}"#)
                    ]),
                .init(role: .tool, content: #"{"lines":3}"#, toolCallId: "call_lines"),
                .init(role: .user, content: "How many lines? Do not call another tool."),
                .init(role: .assistant, content: "Three lines were counted."),
                .init(role: .user, content: "Now use line_count on one\ntwo."),
            ],
            thinkingMode: .chat,
            toolChoiceRequired: true,
            toolChoiceName: "line_count")

        #expect(rendered.contains("The active API tool_choice is required"))
        #expect(rendered.contains("Use the `line_count` function."))
        #expect(rendered.contains("<\u{FF5C}latest_reminder\u{FF5C}>"))
        #expect(rendered.hasSuffix("<\u{FF5C}Assistant\u{FF5C}></think>"))
        #expect(!rendered.contains("<\u{FF5C}action\u{FF5C}>"))

        let processor = ToolCallProcessor(format: .dsml, tools: [Self.lineCountToolSpec()])
        let output = #"line_count("text": "one\ntwo") extra prose that must not leak"#
        var visible = ""
        for ch in output {
            visible += processor.processChunk(String(ch)) ?? ""
        }
        visible += processor.processEOS() ?? ""

        #expect(processor.toolCalls.count == 1)
        #expect(processor.toolCalls.first?.function.name == "line_count")
        #expect(processor.toolCalls.first?.function.arguments["text"] == .string("one\ntwo"))
        #expect(visible.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
    }

    private static func source(_ relativePath: String) throws -> String {
        try String(contentsOfFile: relativePath, encoding: .utf8)
    }

    private static func lineCountToolSpec() -> [String: any Sendable] {
        [
            "type": "function",
            "function": [
                "name": "line_count",
                "description": "Count newline-separated lines.",
                "parameters": [
                    "type": "object",
                    "properties": [
                        "text": ["type": "string"] as [String: any Sendable],
                    ] as [String: any Sendable],
                    "required": ["text"] as [String],
                    "additionalProperties": false,
                ] as [String: any Sendable],
            ] as [String: any Sendable],
        ] as [String: any Sendable]
    }
}
