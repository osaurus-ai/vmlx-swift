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

    /// First lines of the enforced-effort prefaces — the arming markers.
    /// Derived from the encoder constants so a preface reword can never
    /// desync the floor from the rails that require it.
    static var enforcedEffortMarkers: [String] {
        [
            DeepseekV4ChatEncoder.reasoningEffortLowPreface,
            DeepseekV4ChatEncoder.reasoningEffortHighPreface,
            DeepseekV4ChatEncoder.reasoningEffortMaxPreface,
        ].compactMap { $0.components(separatedBy: "\n").first }
    }

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
