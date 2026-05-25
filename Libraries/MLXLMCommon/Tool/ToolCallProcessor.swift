// Copyright © 2025 Apple Inc.

import Foundation

/// Processes generated text to detect and extract tool calls during streaming generation.
///
/// `ToolCallProcessor` handles the streaming detection of tool calls in model output,
/// buffering partial content and extracting complete tool calls when detected.
///
/// Example:
/// ```swift
/// let processor = ToolCallProcessor(format: .lfm2)
/// for chunk in generatedChunks {
///     if let text = processor.processChunk(chunk) {
///         // Regular text to display
///         print(text)
///     }
/// }
/// // After generation completes:
/// for toolCall in processor.toolCalls {
///     // Handle extracted tool calls
///     print(toolCall.function.name)
/// }
/// ```
public class ToolCallProcessor {

    // MARK: - Properties

    private let parser: any ToolCallParser
    private let tools: [[String: any Sendable]]?
    private var state = State.normal
    private var toolCallBuffer = ""
    private var leadingTextBeforeToolCall = ""

    /// The tool calls extracted during processing.
    public var toolCalls: [ToolCall] = []

    // MARK: - State Enum

    private enum State {
        case normal
        case potentialToolCall
        case collectingToolCall
        case collectingInlineToolCall
    }

    // MARK: - Initialization

    /// Initialize with a specific tool call format.
    /// - Parameters:
    ///   - format: The tool call format to use (defaults to `.json` for standard JSON format)
    ///   - tools: Optional tool schemas for type-aware parsing
    public init(format: ToolCallFormat = .json, tools: [[String: any Sendable]]? = nil) {
        self.parser = format.createParser()
        self.tools = tools
    }

    // MARK: - Computed Properties

    /// Whether this processor uses inline format (no start tag).
    private var isInlineFormat: Bool {
        parser.startTag == nil
    }

    /// The first character of the start tag for quick detection.
    private var startTagFirstChar: Character? {
        parser.startTagAliases.first?.first ?? parser.startTagPrefixes.first?.first
    }

    // MARK: - Public Methods

    /// Process a generated text chunk and extract any tool call content.
    /// - Parameter chunk: The text chunk to process
    /// - Returns: Regular text that should be displayed (non-tool call content), or `nil` if buffering
    public func processChunk(_ chunk: String) -> String? {
        if isInlineFormat {
            return processInlineChunk(chunk)
        }
        return processTaggedChunk(chunk)
    }

    /// Process end-of-sequence, parsing any buffered content as tool call(s).
    ///
    /// Call this when generation ends (e.g., on EOS token) to handle formats
    /// whose end tag is never delivered as text (e.g., Mistral where `</s>`
    /// is intercepted at the token ID level).
    ///
    /// For formats with end tags that appear in the text stream, the buffer
    /// will already be empty at generation end, making this a no-op.
    @discardableResult
    public func processEOS() -> String? {
        guard
            state == .collectingToolCall || state == .potentialToolCall
                || state == .collectingInlineToolCall
        else { return nil }
        guard !toolCallBuffer.isEmpty else {
            state = .normal
            return nil
        }

        let parsed = parser.parseEOS(toolCallBuffer, tools: tools)
        toolCalls.append(contentsOf: parsed)
        let suppressUnparsedInlineToolIntent =
            parsed.isEmpty
            && state == .collectingInlineToolCall
            && parser.supportsInlineJSONToolFallback
            && looksLikeExplicitInlineToolIntent(toolCallBuffer)
        let unparsedText =
            parsed.isEmpty && !suppressUnparsedInlineToolIntent
            ? leadingTextBeforeToolCall + toolCallBuffer
            : nil

        toolCallBuffer = ""
        leadingTextBeforeToolCall = ""
        state = .normal
        return unparsedText?.isEmpty == false ? unparsedText : nil
    }

    // MARK: - Private Methods

    /// Process chunk for inline formats (no wrapper tags).
    ///
    /// Uses brace counting to detect when output looks like a JSON tool call.
    /// While braces are unbalanced the content is buffered (returns `nil`)
    /// so partial JSON is never leaked to the UI.
    private func processInlineChunk(_ chunk: String) -> String? {
        switch state {
        case .normal:
            // Check if this chunk starts what looks like a JSON tool call
            if let braceIndex = chunk.firstIndex(of: "{") {
                let leading = String(chunk[..<braceIndex])
                let jsonPart = String(chunk[braceIndex...])
                toolCallBuffer = jsonPart
                state = .collectingInlineToolCall

                if let toolCall = parser.parse(content: toolCallBuffer, tools: tools) {
                    toolCalls.append(toolCall)
                    toolCallBuffer = ""
                    state = .normal
                    return leading.isEmpty ? nil : leading
                }

                // Still collecting — check if braces are balanced (would mean parse
                // failed on complete JSON, so it's not a tool call)
                if jsonBracesBalanced(toolCallBuffer) {
                    state = .normal
                    let buffer = toolCallBuffer
                    toolCallBuffer = ""
                    return leading + buffer
                }

                return leading.isEmpty ? nil : leading
            }

            // No brace seen — pass through as regular text
            return chunk

        case .potentialToolCall, .collectingToolCall, .collectingInlineToolCall:
            toolCallBuffer += chunk

            if let toolCall = parser.parse(content: toolCallBuffer, tools: tools) {
                toolCalls.append(toolCall)
                toolCallBuffer = ""
                state = .normal
                return nil
            }

            // If braces are balanced but parse failed, this isn't a tool call — flush
            if jsonBracesBalanced(toolCallBuffer) {
                state = .normal
                let buffer = toolCallBuffer
                toolCallBuffer = ""
                return buffer
            }

            // Still collecting
            return nil
        }
    }

    /// Check whether open/close braces are balanced in the string.
    private func jsonBracesBalanced(_ text: String) -> Bool {
        var depth = 0
        for ch in text {
            if ch == "{" { depth += 1 } else if ch == "}" { depth -= 1 }
        }
        return depth == 0
    }

    private func looksLikeExplicitInlineToolIntent(_ text: String) -> Bool {
        let compact = text.unicodeScalars.filter {
            !CharacterSet.whitespacesAndNewlines.contains($0)
        }.map(String.init).joined()
        guard compact.hasPrefix("{") else { return false }
        return compact.hasPrefix(#"{"tool":"#)
            || compact.hasPrefix(#"{"tool_name":"#)
            || compact.hasPrefix(#"{"function":{"#)
            || compact.contains(#","tool":"#)
            || compact.contains(#","tool_name":"#)
            || compact.contains(#","function":{"#)
    }

    /// Process chunk for tagged formats.
    private func processTaggedChunk(_ chunk: String) -> String? {
        if parser.supportsInlineJSONToolFallback {
            switch state {
            case .collectingInlineToolCall:
                return processInlineChunk(chunk)
            case .normal:
                if let braceIndex = chunk.firstIndex(of: "{") {
                    let taggedIndex = startTagFirstChar.flatMap { chunk.firstIndex(of: $0) }
                    if taggedIndex == nil || braceIndex < taggedIndex! {
                        return processInlineChunk(chunk)
                    }
                }
            case .potentialToolCall, .collectingToolCall:
                break
            }
        }

        let startTags = parser.startTagAliases
        let startTagPrefixes = parser.startTagPrefixes
        guard (!startTags.isEmpty || !startTagPrefixes.isEmpty),
            let startChar = startTagFirstChar
        else {
            return chunk
        }

        guard (state == .normal && chunk.contains(startChar)) || state != .normal else {
            return chunk
        }

        toolCallBuffer += chunk
        var leadingToken: String?

        switch state {
        case .normal:
            // Change state to potential tool call
            state = .potentialToolCall

            leadingToken = separateToken(
                from: &toolCallBuffer, separator: String(startChar), returnLeading: true)
            if let leadingToken {
                leadingTextBeforeToolCall += leadingToken
            }

            fallthrough
        case .potentialToolCall:
            if partialMatch(buffer: toolCallBuffer, tags: startTags, prefixes: startTagPrefixes) {
                if startsWithCompleteTag(buffer: toolCallBuffer, tags: startTags, prefixes: startTagPrefixes) {
                    state = .collectingToolCall
                    fallthrough
                } else {
                    return nil
                }
            } else {
                // Otherwise, return the collected text and reset the state
                state = .normal
                let buffer = toolCallBuffer
                toolCallBuffer = ""
                let visible = leadingTextBeforeToolCall + buffer
                leadingTextBeforeToolCall = ""
                return visible
            }
        case .collectingToolCall:
            let endTags = parser.endTagAliases
            let endTagPrefixes = parser.endTagPrefixes
            guard !endTags.isEmpty || !endTagPrefixes.isEmpty else {
                return nil
            }

            guard parser.isValidPartialContent(toolCallBuffer) else {
                state = .normal
                let buffer = toolCallBuffer
                toolCallBuffer = ""
                let visible = leadingTextBeforeToolCall + buffer
                leadingTextBeforeToolCall = ""
                return visible.isEmpty ? nil : visible
            }

            if let matchedEndTag = firstTag(
                in: toolCallBuffer,
                tags: endTags,
                prefixes: endTagPrefixes
            ) {
                // Separate the trailing token
                let trailingToken = separateToken(
                    from: &toolCallBuffer, separator: matchedEndTag, returnLeading: false)

                // Parse the completed wrapper. Some formats, including Hy3 /
                // Hunyuan, can carry multiple calls inside one outer block.
                toolCalls.append(contentsOf: parser.parseEOS(toolCallBuffer, tools: tools))

                state = .normal
                toolCallBuffer = ""

                // If the token contains the start character, there may be more tool calls to come
                let leading = leadingTextBeforeToolCall
                leadingTextBeforeToolCall = ""
                if let trailingToken, let startChar = startTagFirstChar,
                    trailingToken.contains(startChar)
                {
                    let trailing = processChunk(trailingToken) ?? ""
                    let visible = leading + trailing
                    return visible.isEmpty ? nil : visible
                } else {
                    // Otherwise, return the collected token, or nil if it's empty
                    let visible = leading + (trailingToken ?? "")
                    return visible.isEmpty ? nil : visible
                }
            } else {
                return nil
            }
        case .collectingInlineToolCall:
            return processInlineChunk(chunk)
        }
    }

    /// Separates a token from a string buffer based on a separator
    /// - Parameters:
    ///   - buffer: The string buffer to modify
    ///   - separator: The separator string to search for
    ///   - returnLeading: If true, returns text before separator; if false, returns text after
    /// - Returns: The separated token, or nil if separator not found
    private func separateToken(from buffer: inout String, separator: String, returnLeading: Bool)
        -> String?
    {
        guard let range = buffer.range(of: separator) else { return nil }

        let token: String
        if returnLeading {
            token = String(buffer[..<range.lowerBound])
            buffer = String(buffer[range.lowerBound...])
        } else {
            token = String(buffer[range.upperBound...])
            buffer = String(buffer[..<range.upperBound])
        }

        return token
    }

    private func partialMatch(buffer: String, tags: [String], prefixes: [String]) -> Bool {
        tags.contains { partialMatch(buffer: buffer, tag: $0) }
            || prefixes.contains { partialMatch(buffer: buffer, tag: $0) || buffer.starts(with: $0) }
    }

    private func partialMatch(buffer: String, tag: String) -> Bool {
        for (tagIndex, bufferIndex) in zip(tag.indices, buffer.indices) {
            if buffer[bufferIndex] != tag[tagIndex] {
                return false
            }
        }

        return true
    }

    private func startsWithCompleteTag(
        buffer: String,
        tags: [String],
        prefixes: [String]
    ) -> Bool {
        tags.contains(where: { buffer.starts(with: $0) })
            || prefixes.contains {
                completedPrefixedTag(
                    in: buffer,
                    prefix: $0,
                    range: buffer.startIndex ..< buffer.endIndex
                )?.lowerBound == buffer.startIndex
            }
    }

    private func firstTag(in text: String, tags: [String], prefixes: [String]) -> String? {
        let exactMatch = tags
            .compactMap { tag -> (String, Range<String.Index>)? in
                text.range(of: tag).map { (tag, $0) }
            }
            .min { $0.1.lowerBound < $1.1.lowerBound }

        let prefixedMatch = prefixes
            .compactMap { prefix -> (String, Range<String.Index>)? in
                guard
                    let match = completedPrefixedTag(
                        in: text,
                        prefix: prefix,
                        range: text.startIndex ..< text.endIndex
                    )
                else { return nil }
                return (String(text[match]), match)
            }
            .min { $0.1.lowerBound < $1.1.lowerBound }

        switch (exactMatch, prefixedMatch) {
        case (nil, nil):
            return nil
        case let (exact?, nil):
            return exact.0
        case let (nil, prefixed?):
            return prefixed.0
        case let (exact?, prefixed?):
            return exact.1.lowerBound <= prefixed.1.lowerBound ? exact.0 : prefixed.0
        }
    }

    private func completedPrefixedTag(
        in text: String,
        prefix: String,
        range: Range<String.Index>
    ) -> Range<String.Index>? {
        guard
            let prefixRange = text.range(of: prefix, range: range),
            let close = text[prefixRange.upperBound...].firstIndex(of: ">")
        else { return nil }
        return prefixRange.lowerBound ..< text.index(after: close)
    }
}
