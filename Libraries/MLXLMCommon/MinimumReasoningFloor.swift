// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX

/// Minimum visible-reasoning floor for the DSV4 enforced-low thinking rail.
///
/// DSV4's official chat template renders assistant history as
/// `<think></think>` adjacently, so under greedy decode the model's prior for
/// emitting `</think>` immediately after `<think>` is overwhelming: live runs
/// produced `think=0` on the very first token even with the strongest
/// enforcement preface (both the v2 and v3 wordings were verified to fail).
/// Prompt-side steering can never beat that trained bigram.
///
/// The floor is the decode-side complement of the low preface: while the
/// first ``tokenCount`` tokens are sampled, the `</think>` token id is masked
/// so a short thinking block must land before the model may close it. The
/// ban lifts after the window and the model closes naturally; the preface's
/// "one or two sentences" keeps the block brief.
///
/// This can only DELAY the close token — it never forces or biases toward
/// close. It is the inverse of the removed hidden close-bias mechanism that
/// `NoHiddenReasoningCloseBiasFocusedTests` bans, and it arms only when BOTH
/// hold:
///  - the prompt tail ends inside a freshly opened think block (`…<think>`),
///    which the chat rail (`…</think>` tail) never matches, and
///  - the prompt head carries one of the enforced-effort marker lines
///    (Low, High, or Max preface), which no other model family renders.
///
/// High and Max are floored too: their prefaces make instant close unlikely,
/// but the trained bigram prior is rail-independent and the mask costs
/// nothing when the model was going to think anyway — every effort level is
/// guaranteed a visible think block.
enum MinimumReasoningFloor {

    struct Armed {
        let closeTokenID: Int
        let tokenCount: Int
    }

    static let defaultTokenCount = 16

    /// Leading sampled-token window during which `</think>` stays masked.
    ///
    /// `VMLX_DSV4_REASONING_FLOOR` overrides the window (`0` disables the
    /// floor entirely) so a build can be A/B'd against an unfloored run
    /// without a rebuild.
    static var tokenCount: Int {
        guard
            let raw = ProcessInfo.processInfo.environment[
                "VMLX_DSV4_REASONING_FLOOR"],
            let parsed = Int(raw.trimmingCharacters(in: .whitespaces))
        else { return defaultTokenCount }
        return max(0, parsed)
    }

    /// Arming markers for the floor.
    ///
    /// EMPTY under the V4.1 (1017) numeric reasoning-effort contract. The floor
    /// was a decode-side compensation for the 0731 text "enforced-low" preface,
    /// which made the model close `<think>` on the first token. With the correct
    /// V4.1 numeric budget contract the model self-modulates and emits a visible
    /// thinking block on its own (verified against the vmlx-py reference:
    /// "Say hello in exactly one word." → `<think>We need answer exactly one
    /// word. "Hello" is one word.</think>Hello`). Keeping the floor armed under
    /// the numeric contract is actively HARMFUL: on prompts with little to
    /// reason about it forbids the natural `</think>` for 16 tokens and the
    /// model degenerates into a repetition loop. Deliberately empty so
    /// `railRequiresFloor` never arms; the mechanism below is retained for
    /// A/B'ing via `VMLX_DSV4_REASONING_FLOOR` should a future rail need it.
    static var enforcedEffortMarkers: [String] { [] }

    static func railRequiresFloor(promptHead: String?, promptTail: String?) -> Bool {
        guard let promptTail, promptTail.hasSuffix("<think>") else { return false }
        guard let promptHead else { return false }
        return enforcedEffortMarkers.contains { promptHead.contains($0) }
    }

    static func armIfNeeded(
        input: LMInput,
        tokenizer: any Tokenizer,
        promptTail: String?
    ) -> Armed? {
        guard promptTail?.hasSuffix("<think>") == true else { return nil }
        let tokenIds =
            input.text.tokenIds
            ?? input.text.tokens.reshaped(-1).asArray(Int.self)
        guard !tokenIds.isEmpty else { return nil }
        let head = tokenizer.decode(
            tokenIds: Array(tokenIds.prefix(48)), skipSpecialTokens: false)
        guard railRequiresFloor(promptHead: head, promptTail: promptTail) else { return nil }
        // Round-trip check: convertTokenToId returns the <unk> id for unknown
        // tokens, so id existence alone is not proof the vocab has </think>.
        guard let closeID = tokenizer.convertTokenToId("</think>"),
            tokenizer.convertIdToToken(closeID) == "</think>"
        else { return nil }
        let window = Self.tokenCount
        guard window > 0 else { return nil }
        return Armed(closeTokenID: closeID, tokenCount: window)
    }
}
