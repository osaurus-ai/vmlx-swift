// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLXLMCommon

/// Bailing/Ling chat templates do not read the generic `enable_thinking`
/// Jinja kwarg. They use the system prompt contract from the upstream
/// template instead: a system message containing "detailed thinking on"
/// enables `<think>...</think>` output, while "detailed thinking off"
/// disables it. Keep the translation in the model package so hosts can keep
/// passing the standard `additionalContext["enable_thinking"]` knob.
///
enum BailingThinkingTemplateContext {
    private static let thinkingOnDirective = "detailed thinking on"
    private static let thinkingOffDirective = "detailed thinking off"

    static func applies(to modelType: String?) -> Bool {
        modelType?.lowercased().hasPrefix("bailing") == true
    }

    /// Ling 3.0's template reads the standard kwarg (`{%- if enable_thinking is
    /// defined %}`) and renders "detailed thinking on/off" itself at the END
    /// of the SYSTEM turn, after the tools block — the trained position. The
    /// prepend below puts it at the START of the system content (and the
    /// template then skips its own), which is not what the model saw in
    /// training. When the template reads the kwarg, leave the messages alone
    /// and let the kwarg travel.
    static func templateReadsEnableThinking(_ chatTemplate: String?) -> Bool {
        guard let chatTemplate else { return false }
        return chatTemplate.contains("enable_thinking is defined")
            || chatTemplate.contains("enable_thinking is not defined")
    }

    static func apply(
        to messages: [Message],
        modelType: String?,
        additionalContext: [String: any Sendable]?,
        templateReadsEnableThinking: Bool = false
    ) -> [Message] {
        guard applies(to: modelType), !templateReadsEnableThinking else {
            return messages
        }

        var directives: [String] = []
        if let enableThinking = additionalContext?["enable_thinking"] as? Bool {
            directives.append(enableThinking ? thinkingOnDirective : thinkingOffDirective)
        }
        guard !directives.isEmpty else {
            return messages
        }

        let directive = directives.joined(separator: "\n")
        var out = messages
        if let index = out.firstIndex(where: { ($0["role"] as? String) == "system" }),
           let content = out[index]["content"] as? String
        {
            var system = out[index]
            let cleaned = removeExistingDirectives(from: content)
            system["content"] = cleaned.isEmpty ? directive : "\(directive)\n\n\(cleaned)"
            out[index] = system
        } else {
            out.insert(["role": "system", "content": directive], at: 0)
        }
        return out
    }

    private static func removeExistingDirectives(from content: String) -> String {
        content
            .split(separator: "\n", omittingEmptySubsequences: false)
            .filter { line in
                let normalized = String(line)
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                    .lowercased()
                return normalized != thinkingOnDirective
                    && normalized != thinkingOffDirective
            }
            .joined(separator: "\n")
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

/// Nemotron-H / Omni native tokenizer templates advertise XML function calls.
/// Keep `tool_choice` in `additionalContext` and tool schemas, but do not add
/// synthetic system prompt directives here. Required tool-choice rows must be
/// proven by the model/template path itself rather than prompt coercion.
enum NemotronToolChoiceTemplateContext {
    static func applies(to modelType: String?) -> Bool {
        guard let modelType else { return false }
        let normalized = modelType.lowercased()
        return normalized == "nemotron" || normalized.hasPrefix("nemotron_")
    }

    static func apply(
        to messages: [Message],
        modelType: String?,
        additionalContext: [String: any Sendable]?
    ) -> [Message] {
        _ = modelType
        _ = additionalContext
        return messages
    }
}
