// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import VMLXJinja
import XCTest

private extension Template {
    func renderSnapshotContext(_ context: [String: Any]) throws -> String {
        var values: [String: Value] = [:]
        for (key, value) in context {
            values[key] = try Value(any: value)
        }
        return try render(values)
    }
}

final class LocalChatTemplateFamilySnapshotTests: XCTestCase {

    private struct TemplateCase {
        let name: String
        let candidatePaths: [String]
        let expectedFragments: [String]
        let expectedNormalizedFragments: [String]

        init(
            name: String,
            candidatePaths: [String],
            expectedFragments: [String],
            expectedNormalizedFragments: [String] = []
        ) {
            self.name = name
            self.candidatePaths = candidatePaths
            self.expectedFragments = expectedFragments
            self.expectedNormalizedFragments = expectedNormalizedFragments
        }
    }

    private let cases: [TemplateCase] = [
        TemplateCase(
            name: "mistral-jang-2l",
            candidatePaths: [
                "models/Mistral-Medium-3.5-128B-JANG_2L/chat_template.jinja",
                "models/Mistral-Small-4-119B-JANG_2L/chat_template.jinja",
                ".cache/huggingface/hub/models--JANGQ-AI--Mistral-Medium-3.5-128B-JANG_2L/snapshots/*/chat_template.jinja",
                ".cache/huggingface/hub/models--JANGQ-AI--Mistral-Small-4-119B-A6B-JANG_2L/snapshots/*/chat_template.jinja",
            ],
            expectedFragments: [
                "[MODEL_SETTINGS]",
                "[INST]Say ping.[/INST]",
            ]),
        TemplateCase(
            name: "minimax-m27-jangtq",
            candidatePaths: [
                "models/JANGQ/MiniMax-M2.7-JANGTQ/chat_template.jinja",
                "models/JANGQ/MiniMax-M2.7-JANGTQ_K/chat_template.jinja",
                "models/JANGQ/MiniMax-M2.7-Small-JANGTQ/chat_template.jinja",
                "models/dealign.ai/MiniMax-M2.7-JANGTQ_K-CRACK/chat_template.jinja",
                "models/dealign.ai/MiniMax-M2.7-JANG_K-CRACK/chat_template.jinja",
                ".cache/huggingface/hub/models--JANGQ-AI--MiniMax-M2.7-JANGTQ_K/snapshots/*/chat_template.jinja",
            ],
            expectedFragments: [
                "]~b]system",
                "]~b]user\nSay ping.",
                "]~b]ai\n",
            ]),
        TemplateCase(
            name: "kimi-k26-jangtq",
            candidatePaths: [
                "models/JANGQ/Kimi-K2.6-Small-JANGTQ/chat_template.jinja",
            ],
            expectedFragments: [
                "<|im_user|>user<|im_middle|>",
                "Say ping.",
                "<|im_assistant|>assistant<|im_middle|>",
            ]),
        TemplateCase(
            name: "qwen35",
            candidatePaths: [
                "models/Qwen3.5-35B-A3B-4bit/chat_template.jinja",
                "models/dealign.ai/Qwen3.6-35B-A3B-MXFP4-CRACK-MTP/chat_template.jinja",
                "models/dealign.ai/Qwen3.6-35B-A3B-MXFP8-CRACK-MTP/chat_template.jinja",
            ],
            expectedFragments: [
                "<|im_start|>user\nSay ping.<|im_end|>",
                "<|im_start|>assistant\n<think>\n\n</think>",
            ]),
        TemplateCase(
            name: "qwen36-jang-4m",
            candidatePaths: [
                "models/dealign.ai/Qwen3.6-27B-JANG_4M-CRACK/chat_template.jinja",
                "models/dealign.ai/Qwen3.6-35B-A3B-JANGTQ-CRACK/chat_template.jinja",
            ],
            expectedFragments: [
                "<|im_start|>user\nSay ping.<|im_end|>",
                "<|im_start|>assistant\n<think>\n\n</think>",
            ]),
        TemplateCase(
            name: "ling-bailing-hybrid",
            candidatePaths: [
                "models/JANGQ/Ling-2.6-flash-JANGTQ/chat_template.jinja",
                "models/dealign.ai/Ling-2.6-flash-MXFP4-CRACK/chat_template.jinja",
                ".mlxstudio/models/dealignai/Ling-2.6-flash-JANGTQ2-CRACK/chat_template.jinja",
            ],
            expectedFragments: [
                "<role>HUMAN</role>Say ping.<|role_end|>",
                "<role>ASSISTANT</role>",
            ]),
        TemplateCase(
            name: "nemotron",
            candidatePaths: [
                "models/NVIDIA-Nemotron-3-Ultra-550B-A55B-JANGTQ_1L/chat_template.jinja",
                ".mlxstudio/models/Nemotron-Cascade-2-30B-A3B-JANG_2L/chat_template.jinja",
                ".mlxstudio/models/Nemotron-3-Super-120B-A12B-JANG_2L/chat_template.jinja",
                ".mlxstudio/models/dealignai/Nemotron-Omni-Nano-JANGTQ-CRACK/chat_template.jinja",
                "models/dealign.ai/Nemotron-Omni-Nano-JANGTQ-CRACK/chat_template.jinja",
            ],
            expectedFragments: [
                "<think></think>",
                "<|im_start|>assistant",
            ],
            expectedNormalizedFragments: [
                "<|im_start|>user Say ping.",
            ]),
        TemplateCase(
            name: "zaya",
            candidatePaths: [
                "models/Osaurus/ZAYA1-8B-MXFP4/chat_template.jinja",
                "models/JANGQ/ZAYA1-8B-JANGTQ_K/chat_template.jinja",
                "models/JANGQ/ZAYA1-8B-JANGTQ4/chat_template.jinja",
                "models/Zyphra/ZAYA1-8B-JANGTQ2/chat_template.jinja",
                "models/Zyphra/ZAYA1-8B-JANGTQ4/chat_template.jinja",
                "models/Zyphra/ZAYA1-8B-MXFP4/chat_template.jinja",
            ],
            expectedFragments: [
                "<bos>\n<|im_start|>system",
                "<|im_start|>user\nSay ping.<|im_end|>",
                "<|im_start|>assistant\n<think>\n</think>",
            ]),
    ]

    func testLocalFamilyTemplatesRenderStableMinimalChatSnapshots() throws {
        var renderedCount = 0
        var missing: [String] = []

        for item in cases {
            guard let templateURL = firstExistingTemplate(for: item) else {
                missing.append(item.name)
                continue
            }

            let source = try String(contentsOf: templateURL, encoding: .utf8)
            let rendered = try Template(source).renderSnapshotContext(snapshotContext)
            renderedCount += 1

            for fragment in item.expectedFragments {
                XCTAssertTrue(
                    rendered.contains(fragment),
                    "\(item.name) snapshot from \(templateURL.path) missing \(fragment.debugDescription). Rendered: \(rendered)"
                )
            }
            let normalizedRendered = Self.normalizedWhitespace(rendered)
            for fragment in item.expectedNormalizedFragments {
                XCTAssertTrue(
                    normalizedRendered.contains(Self.normalizedWhitespace(fragment)),
                    "\(item.name) normalized snapshot from \(templateURL.path) missing \(fragment.debugDescription). Rendered: \(rendered)"
                )
            }
        }

        if renderedCount == 0 {
            throw XCTSkip("No local ~/models chat templates were available for snapshot coverage.")
        }

        XCTAssertGreaterThanOrEqual(
            renderedCount, 6,
            "Expected broad local chat-template coverage; missing: \(missing.joined(separator: ", "))"
        )
    }

    private var snapshotContext: [String: Any] {
        [
            "messages": [
                ["role": "system", "content": "You are a test assistant."],
                ["role": "user", "content": "Say ping."],
            ],
            "tools": [] as [Any],
            "tools_ts_str": "",
            "add_generation_prompt": true,
            "enable_thinking": false,
            "thinking": false,
            "preserve_thinking": false,
            "truncate_history_thinking": true,
            "reasoning_effort": "none",
            "reasoning_budget": 0,
            "add_vision_id": false,
            "bos_token": "<bos>",
            "eos_token": "<|im_end|>",
        ]
    }

    private func firstExistingTemplate(for item: TemplateCase) -> URL? {
        let home = FileManager.default.homeDirectoryForCurrentUser
        for path in item.candidatePaths {
            let expanded = home.appendingPathComponent(path).path
            if path.contains("*") {
                let matches = glob(expanded)
                if let first = matches.first {
                    return URL(fileURLWithPath: first)
                }
            } else if FileManager.default.fileExists(atPath: expanded) {
                return URL(fileURLWithPath: expanded)
            }
        }
        return nil
    }

    private func glob(_ pattern: String) -> [String] {
        var globResult = glob_t()
        defer { globfree(&globResult) }
        guard Darwin.glob(pattern, 0, nil, &globResult) == 0,
              let glPathv = globResult.gl_pathv
        else { return [] }

        var paths: [String] = []
        for i in 0..<Int(globResult.gl_matchc) {
            if let cString = glPathv[i] {
                paths.append(String(cString: cString))
            }
        }
        return paths.sorted()
    }

    private static func normalizedWhitespace(_ text: String) -> String {
        text.split(whereSeparator: \.isWhitespace).joined(separator: " ")
    }
}
