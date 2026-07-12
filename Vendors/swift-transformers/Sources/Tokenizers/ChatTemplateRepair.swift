// Copyright © 2026 Apple Inc.

import Foundation

/// Repairs a chat template that would strand generation outside of any turn.
///
/// ## The failure
///
/// The gemma-4 templates close the assistant turn when a history message carries BOTH
/// content AND tool calls, and then — because the last thing emitted was a tool response —
/// `add_generation_prompt` skips the `<|turn>model` role header. Generation therefore
/// begins *after a closed turn, with no role*. The model has nothing telling it who is
/// speaking, so it improvises the scaffolding it has seen around assistant turns: it emits
/// `thought` / `<channel|>` openers. In a long agent context it locks onto that and repeats
/// the header line forever:
///
///     thought
///     <channel|>£thought
///     <channel|>o'thought
///     <channel|>o'thought          ← until the token budget runs out
///
/// It only fires after real tool use — a turn with tool calls but no content leaves the turn
/// open, so plain chat repros stay clean, which is exactly why this survived so long.
///
/// ## Why repair it here
///
/// Patching the bundle on disk does not hold: the loader re-fetches `tokenizer_config.json`
/// and `chat_template.jinja` from the Hub, silently restoring the broken template. Repairing
/// the template string in memory, at the one point where every path selects it, fixes every
/// already-downloaded bundle without a re-download and cannot be undone by a refetch.
///
/// This is not a behavioural workaround. It biases no logits, injects no thinking tags, and
/// coerces no output — it hands the model a well-formed turn instead of a malformed one. The
/// bundles should still be corrected at the source; this keeps users working until they are.
public enum ChatTemplateRepair {

    /// A turn that emitted a tool response must stay OPEN, whether or not it also had
    /// content — otherwise generation starts after `<turn|>` with no role header.
    ///
    /// Two dialects ship in the wild; both spell the same mistake.
    private static let closeRepairs: [(broken: String, fixed: String)] = [
        // MXFP4 / JANG / MXFP8 dialect
        (
            "{%- elif not (ns_tr_out.flag and not has_content) -%}",
            "{%- elif not ns_tr_out.flag -%}"
        ),
        // 8-bit retention dialect
        (
            "{%- if not (message['tool_responses'] and not message['content']) -%}",
            "{%- if not message['tool_responses'] -%}"
        ),
    ]

    /// Does this template contain the stranded-turn bug?
    public static func needsRepair(_ template: String) -> Bool {
        closeRepairs.contains { template.contains($0.broken) }
    }

    /// Return a repaired template, or the original when there is nothing to fix.
    ///
    /// Only the turn-close condition is rewritten. The other half of the original bug — the
    /// assistant's content rendering *after* the tool response, inverting the chronology the
    /// model was trained on — is a block reordering that cannot be done safely with string
    /// surgery on an arbitrary template, and it is the *close* that strands generation. Fixing
    /// the close alone is what stops the runaway; the ordering is corrected at the source in
    /// the bundles.
    public static func repaired(_ template: String) -> String {
        var out = template
        for repair in closeRepairs where out.contains(repair.broken) {
            out = out.replacingOccurrences(of: repair.broken, with: repair.fixed)
        }
        return out
    }
}
