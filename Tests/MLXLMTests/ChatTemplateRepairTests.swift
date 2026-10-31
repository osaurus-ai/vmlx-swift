// Copyright © 2026 Apple Inc.

import Foundation
import Testing
import VMLXJinja
import VMLXTokenizers

@Suite("Selected chat templates retain their model-owned boundaries")
struct ChatTemplateRepairTests {
    private static let legacyMXFP = "{%- elif not (ns_tr_out.flag and not has_content) -%}"
    private static let legacy8Bit =
        "{%- if not (message['tool_responses'] and not message['content']) -%}"

    @Test("Legacy conditions can be diagnosed without rewriting the template")
    func legacyTemplatesAreNotSilentlyChanged() {
        for condition in [Self.legacyMXFP, Self.legacy8Bit] {
            let source = "before \(condition) after"
            #expect(ChatTemplateRepair.needsRepair(source))
            #expect(ChatTemplateRepair.repaired(source) == source)
        }
    }

    @Test("An in-memory rewrite must not erase a close before a new user")
    func explicitUserBoundarySurvives() throws {
        let source = """
            {%- set ns_tr_out = namespace(flag=true) -%}
            {%- set has_content = true -%}
            {{- '<tool_response|>' -}}
            {%- if false -%}
            \(Self.legacyMXFP)
                {{- '<turn|>\\n' -}}
            {%- endif -%}
            {{- '<|turn>user\\nNext user<turn|>\\n' -}}
            """
        let rendered = try Template(ChatTemplateRepair.repaired(source)).render([:])
        #expect(rendered == "<tool_response|><turn|>\n<|turn>user\nNext user<turn|>\n")
    }

    @Test("Templates without legacy spellings remain byte-identical")
    func otherTemplateFamiliesAreUnchanged() {
        for source in [
            "{{ bos_token }}{% for m in messages %}{{ m['content'] }}{% endfor %}",
            "<|im_start|>assistant\n<think>",
            "[INST] {{ messages[0].content }} [/INST]",
        ] {
            #expect(!ChatTemplateRepair.needsRepair(source))
            #expect(ChatTemplateRepair.repaired(source) == source)
        }
    }

    @Test("Repeated calls preserve the original source")
    func repeatedCallsDoNotMutateSource() {
        let source = "x \(Self.legacyMXFP) y \(Self.legacy8Bit) z"
        #expect(ChatTemplateRepair.repaired(source) == source)
        #expect(ChatTemplateRepair.repaired(ChatTemplateRepair.repaired(source)) == source)
    }

    @Test("The tokenizer compiles the selected source, without automatic repairs")
    func tokenizerKeepsSelectedTemplateAuthoritative() throws {
        let root =
            ProcessInfo.processInfo.environment["VMLX_RUNTIME_SOURCE_ROOT"]
            ?? FileManager.default.currentDirectoryPath
        let url = URL(fileURLWithPath: root).appendingPathComponent(
            "Vendors/swift-transformers/Sources/Tokenizers/Tokenizer.swift")
        let source = try String(contentsOf: url, encoding: .utf8)
        #expect(source.contains("compiledTemplate(for: selectedChatTemplate)"))
        #expect(!source.contains("ChatTemplateRepair.repaired(selectedChatTemplate)"))
    }
}
