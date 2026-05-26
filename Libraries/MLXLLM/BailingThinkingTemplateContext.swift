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
/// The same native template also ignores generic `tool_choice`. When callers
/// explicitly request OpenAI-style `required`, surface that contract through
/// the same system-message lane instead of adding sampler or decode coercion.
enum BailingThinkingTemplateContext {
    private static let thinkingOnDirective = "detailed thinking on"
    private static let thinkingOffDirective = "detailed thinking off"
    private static let requiredToolDirective =
        "For this assistant turn, return exactly one <tool_call> JSON object for one available function and no prose before the tool result."

    static func applies(to modelType: String?) -> Bool {
        modelType?.lowercased().hasPrefix("bailing") == true
    }

    static func apply(
        to messages: [Message],
        modelType: String?,
        additionalContext: [String: any Sendable]?
    ) -> [Message] {
        guard applies(to: modelType) else {
            return messages
        }

        var directives: [String] = []
        if let enableThinking = additionalContext?["enable_thinking"] as? Bool {
            directives.append(enableThinking ? thinkingOnDirective : thinkingOffDirective)
        }
        if additionalContext?["tool_choice"] as? String == "required" {
            directives.append(requiredToolDirective)
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
                    && normalized != requiredToolDirective.lowercased()
            }
            .joined(separator: "\n")
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

/// Nemotron-H / Omni native tokenizer templates advertise XML function calls,
/// but they do not consume the generic OpenAI `tool_choice=required` kwarg.
/// Translate only that explicit API contract into the model's existing
/// ChatML/system-message lane. This is deliberately separate from fallback
/// template rails and does not alter sampler defaults, thinking mode, or
/// parser visibility.
enum NemotronToolChoiceTemplateContext {
    private static let requiredToolDirective =
        "For this assistant turn, return exactly one <tool_call> XML function call for one available function and no prose before the tool result. Include every required <parameter=...> value exactly as requested."

    private static func requiredToolDirective(targetName: String?) -> String {
        guard let targetName,
              !targetName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        else {
            return requiredToolDirective
        }
        return "For this assistant turn, return exactly one <tool_call> XML function call for the \(targetName) function and no prose before the tool result. Include every required <parameter=...> value exactly as requested."
    }

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
        guard applies(to: modelType),
              additionalContext?["tool_choice"] as? String == "required"
        else {
            return messages
        }

        var out = messages
        let directive = requiredToolDirective(
            targetName: additionalContext?["tool_choice_name"] as? String)
        if let index = out.firstIndex(where: { ($0["role"] as? String) == "system" }),
           let content = out[index]["content"] as? String
        {
            var system = out[index]
            let cleaned = removeExistingDirective(from: content)
            system["content"] = cleaned.isEmpty
                ? directive
                : "\(directive)\n\n\(cleaned)"
            out[index] = system
        } else {
            out.insert(["role": "system", "content": directive], at: 0)
        }
        return out
    }

    private static func removeExistingDirective(from content: String) -> String {
        content
            .split(separator: "\n", omittingEmptySubsequences: false)
            .filter { line in
                let normalized = String(line)
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                    .lowercased()
                return normalized != requiredToolDirective.lowercased()
                    && !normalized.hasPrefix(
                        "for this assistant turn, return exactly one <tool_call> xml function call for the ")
            }
            .joined(separator: "\n")
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
