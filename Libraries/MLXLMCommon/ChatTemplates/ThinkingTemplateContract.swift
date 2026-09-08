import Foundation

/// Detect the native three-way generation-tail contract without a model-name table.
public enum ThinkingTemplateContract {
    /// Omitted `enable_thinking` is distinct from both explicit bool values.
    /// A normalization/default assignment or an absent-kwarg else branch is
    /// not this contract. This inspection never modifies the rendered prompt.
    public static func preservesOmittedThinking(_ template: String?) -> Bool {
        guard let lower = template?.lowercased(),
            !lower.contains("set enable_thinking"),
            let generation = lower.range(of: "if add_generation_prompt"),
            let regex = try? NSRegularExpression(
                pattern: #"\{%-?\s*(.*?)\s*-?%\}"#,
                options: [.dotMatchesLineSeparators])
        else { return false }
        let tail = String(lower[generation.lowerBound...]) as NSString
        let tags = regex.matches(in: tail as String, range: NSRange(location: 0, length: tail.length))
        func body(_ match: NSTextCheckingResult) -> String {
            tail.substring(with: match.range(at: 1)).trimmingCharacters(in: .whitespacesAndNewlines)
        }
        guard let start = tags.firstIndex(where: {
            ["if enable_thinking is defined", "if (enable_thinking is defined)"].contains(body($0))
        }) else { return false }
        var depth = 0
        let contentStart = NSMaxRange(tags[start].range)
        for tag in tags.dropFirst(start + 1) {
            let keyword = body(tag).split(whereSeparator: \.isWhitespace).first
            if keyword == "if" {
                depth += 1
            } else if keyword == "else" || keyword == "elif" {
                if depth == 0 { return false }
            } else if keyword == "endif" {
                if depth == 0 {
                    let branch = tail.substring(with: NSRange(
                        location: contentStart, length: tag.range.location - contentStart))
                    return branch.contains("enable_thinking is true")
                        && branch.contains("enable_thinking is false")
                        && branch.contains("<think>") && branch.contains("</think>")
                }
                depth -= 1
            }
        }
        return false
    }
}
