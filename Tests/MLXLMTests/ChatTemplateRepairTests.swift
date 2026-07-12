// Copyright © 2026 Apple Inc.

import Foundation
import Testing
import VMLXTokenizers

/// The gemma-4 templates close the assistant turn when a history message carries BOTH content
/// AND tool calls, and `add_generation_prompt` then skips `<|turn>model` because the last thing
/// emitted was a tool response. Generation starts after a CLOSED turn with no role header — the
/// model has nothing telling it who is speaking, improvises the scaffolding it has seen around
/// assistant turns, and locks into repeating it:
///
///     thought
///     <channel|>£thought
///     <channel|>o'thought
///     <channel|>o'thought          ← forever
///
/// It only fires after real tool use. A turn with tool calls but no content leaves the turn
/// open, which is why plain chat repros look clean and this survived for months.
///
/// Patching the bundle on disk does not hold — the loader re-fetches `tokenizer_config.json`
/// from the Hub and silently restores the broken template — so the repair happens in memory at
/// the point every path selects the template.
@Suite("Chat template repair: a tool turn must not strand generation")
struct ChatTemplateRepairTests {

    /// Both dialects that ship in the wild spell the same mistake.
    private static let brokenMXFP = "{%- elif not (ns_tr_out.flag and not has_content) -%}"
    private static let broken8Bit = "{%- if not (message['tool_responses'] and not message['content']) -%}"

    @Test("The MXFP/JANG dialect's stranded-turn close is repaired")
    func repairsMXFPDialect() {
        let template = """
            {%- for message in messages -%}
                {%- if ns.prev_message_type == 'tool_call' and not ns_tr_out.flag -%}
                    {{- '<|tool_response>' -}}
                \(Self.brokenMXFP)
                    {{- '<turn|>\\n' -}}
                {%- endif -%}
            {%- endfor -%}
            """

        #expect(ChatTemplateRepair.needsRepair(template))

        let fixed = ChatTemplateRepair.repaired(template)
        #expect(fixed.contains("{%- elif not ns_tr_out.flag -%}"))
        #expect(
            !fixed.contains(Self.brokenMXFP),
            "a turn that emitted a tool response must stay OPEN whether or not it had content"
        )
    }

    @Test("The 8-bit retention dialect's stranded-turn close is repaired")
    func repairs8BitDialect() {
        let template = """
            {%- for message in messages -%}
                \(Self.broken8Bit)
                    {{- '<turn|>\\n' -}}
                {%- endif -%}
            {%- endfor -%}
            """

        #expect(ChatTemplateRepair.needsRepair(template))

        let fixed = ChatTemplateRepair.repaired(template)
        #expect(fixed.contains("{%- if not message['tool_responses'] -%}"))
        #expect(!fixed.contains(Self.broken8Bit))
    }

    @Test("A template without the bug is returned untouched")
    func healthyTemplateIsUnchanged() {
        // Nothing else may be rewritten — this runs on EVERY model's template, not just gemma.
        let healthy = """
            {%- for message in messages -%}
                {{- '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n' -}}
            {%- endfor -%}
            {%- if add_generation_prompt -%}{{- '<|im_start|>assistant\\n' -}}{%- endif -%}
            """
        #expect(!ChatTemplateRepair.needsRepair(healthy))
        #expect(ChatTemplateRepair.repaired(healthy) == healthy)
    }

    @Test("Repair is idempotent")
    func repairIsIdempotent() {
        // The loader may repair the same template more than once across reloads; doing so must
        // not corrupt it.
        let broken = "x \(Self.brokenMXFP) y \(Self.broken8Bit) z"
        let once = ChatTemplateRepair.repaired(broken)
        let twice = ChatTemplateRepair.repaired(once)
        #expect(once == twice)
        #expect(!ChatTemplateRepair.needsRepair(once))
    }
}
