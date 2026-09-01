// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation

/// Where a bundle's chat template came from.
///
/// Worth reporting rather than discarding: a template that fails to resolve is replaced by a
/// generic one, and the model then generates fluent, plausible text through the wrong prompt
/// format. Generation cannot distinguish "template applied" from "template silently replaced", so
/// the provenance has to be observable somewhere or the failure has no tell at all.
public enum ChatTemplateSource: String, Equatable, Sendable {
    /// A sibling `chat_template.jinja`. Several VLMs ship the template ONLY here and leave
    /// `tokenizer_config.json` without one — GLM-4.5V is the case that prompted this.
    case sidecar

    /// The `chat_template` string inside `tokenizer_config.json`.
    case tokenizerConfig
}

public struct ResolvedChatTemplate: Equatable, Sendable {
    public let text: String
    public let source: ChatTemplateSource

    public init(text: String, source: ChatTemplateSource) {
        self.text = text
        self.source = source
    }
}

/// Resolving a bundle's chat template, in one place.
///
/// This rule was written twice — once in `LLMModelFactory`, once in `VLMModelFactory` — with
/// identical bodies. Identical today; the point of a single owner is that a bundle loaded through
/// the text path and the same bundle loaded through the multimodal path cannot start disagreeing
/// about which file its prompt format comes from.
public enum ChatTemplateResolver {
    /// The sidecar takes precedence over the inline template.
    ///
    /// That order is not arbitrary: a converter that rewrites the template writes the sidecar and
    /// commonly leaves a stale inline copy behind, so preferring the sidecar prefers the newer of
    /// the two. Bundles carrying only one are unaffected either way.
    public static func resolve(modelDirectory: URL) -> ResolvedChatTemplate? {
        let sidecarURL = modelDirectory.appending(component: "chat_template.jinja")
        if let text = try? String(contentsOf: sidecarURL, encoding: .utf8) {
            return ResolvedChatTemplate(text: text, source: .sidecar)
        }
        let tokenizerConfigURL = modelDirectory.appending(component: "tokenizer_config.json")
        guard
            let data = try? Data(contentsOf: tokenizerConfigURL),
            let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let text = object["chat_template"] as? String
        else {
            return nil
        }
        return ResolvedChatTemplate(text: text, source: .tokenizerConfig)
    }

    /// The resolved template text, or `nil` when the bundle carries neither source.
    public static func text(modelDirectory: URL) -> String? {
        resolve(modelDirectory: modelDirectory)?.text
    }
}
