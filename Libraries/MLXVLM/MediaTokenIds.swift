// Copyright © 2026 Osaurus AI. All rights reserved.

import Foundation
import MLXLMCommon

/// Resolves the placeholder token ids a processor declares as
/// ``LMInput/mediaTokenIds``.
///
/// Declaring these is what lets a cache hit resume. Without them
/// ``LMInput/cacheHitSuffixContainsMediaPlaceholder(_:)`` takes its
/// `guard let mediaTokenIds else { return true }` branch and rolls back on ANY
/// non-empty suffix, so a text follow-up about the same picture re-prefills the
/// whole prompt, vision tower included. Declaring them narrows that blanket
/// rollback to a precise one; a suffix that genuinely carries a placeholder is
/// still rejected.
///
/// Measured on Qwen3.8 27B: follow-up TTFT 3.40s → 0.59s median.
public enum MediaTokenIds {

    /// Ids for `tokens`, or `nil` if none resolve.
    ///
    /// A token counts only if it encodes to exactly one id **and** that id
    /// decodes back to the same token. Both halves are load-bearing:
    ///
    /// - Encoding to several ids means the tokenizer split the literal into
    ///   ordinary text pieces — those are not a placeholder.
    /// - Encoding to one id is not enough. A tokenizer without the special
    ///   maps it to `<unk>`, which IS a single real id and also appears
    ///   throughout ordinary text. Declaring it would mark unrelated tokens as
    ///   media placeholders and suppress reuse far more aggressively than
    ///   declaring nothing. The decode check is what rules that out. Note
    ///   `convertTokenToId` cannot be used here for the same reason — it
    ///   returns the unk id rather than nil for an unknown token.
    ///
    /// Returns `nil` rather than `[]` on failure: an empty array takes the
    /// `!mediaTokenIds.isEmpty` branch and would report "no media in the
    /// suffix" for every suffix, which is the unsafe direction. `nil` keeps
    /// the conservative full-prefill rollback, i.e. today's behaviour.
    public static func resolve(tokenizer: any Tokenizer, tokens: [String]) -> [Int]? {
        var ids: [Int] = []
        for token in tokens {
            let encoded = tokenizer.encode(text: token, addSpecialTokens: false)
            guard encoded.count == 1, let id = encoded.first else { continue }
            guard tokenizer.decode(tokenIds: [id], skipSpecialTokens: false) == token else {
                continue
            }
            if !ids.contains(id) { ids.append(id) }
        }
        return ids.isEmpty ? nil : ids
    }
}
