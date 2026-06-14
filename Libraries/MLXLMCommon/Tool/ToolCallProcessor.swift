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
    /// When true, the processor still detects and strips tool-call control
    /// markers from visible text, but never surfaces extracted tool calls.
    /// Used when the model emits tool-call syntax (e.g. a hallucinated call
    /// under thinking) while the request offered no tools — the markers are
    /// template control tokens that must not leak, but no tool was requested
    /// so none is reported.
    private let stripOnly: Bool
    public let parsesToolCallsFromReasoningChannel: Bool
    public let usesTaggedOnlyReasoningExtraction: Bool
    public let preservesReasoningTextAroundToolCalls: Bool
    private var state = State.normal
    private var toolCallBuffer = ""
    private var leadingTextBeforeToolCall = ""
    private var inlineToolCallKind = InlineToolCallKind.json
    private var suppressingTextAfterInlineToolCall = false

    /// The tool calls extracted during processing.
    public var toolCalls: [ToolCall] = []

    /// Record extracted tool calls unless running strip-only (markers are
    /// stripped from visible text either way; in strip-only mode the call
    /// itself is discarded because no tools were offered).
    private func recordToolCall(_ call: ToolCall) {
        guard !stripOnly else { return }
        toolCalls.append(canonicalizedToolName(call))
    }

    private func recordToolCalls(_ calls: [ToolCall]) {
        guard !stripOnly else { return }
        toolCalls.append(contentsOf: calls.map(canonicalizedToolName))
    }

    /// Registered tool names from the request's tool schemas.
    private func registeredToolNames() -> [String] {
        guard let tools else { return [] }
        return tools.compactMap { tool in
            let function = (tool["function"] as? [String: any Sendable]) ?? tool
            return function["name"] as? String
        }
    }

    /// Canonicalize a parsed tool-call name to the exact registered spelling
    /// when it differs only by case. Models (e.g. some Gemma quantizations)
    /// occasionally emit `Get_weather` for a tool registered as `get_weather`;
    /// downstream dispatch (`ToolCall.execute`, OpenAI clients) matches names
    /// case-sensitively, so an un-canonicalized name fails to invoke the tool.
    /// Conservative: only rewrites on a case-insensitive EXACT match to a
    /// registered name, so it never invents or guesses a different tool.
    private func canonicalizedToolName(_ call: ToolCall) -> ToolCall {
        let registered = registeredToolNames()
        guard !registered.isEmpty else { return call }
        let name = call.function.name
        if registered.contains(name) { return call }
        guard let canonical = registered.first(where: {
            $0.compare(name, options: .caseInsensitive) == .orderedSame
        }) else { return call }
        return ToolCall(
            id: call.id,
            function: .init(name: canonical, arguments: call.function.arguments))
    }

    // MARK: - State Enum

    private enum State {
        case normal
        case potentialToolCall
        case collectingToolCall
        case collectingInlineToolCall
    }

    private enum InlineToolCallKind {
        case json
        case actionJSON
        case functionCall
        case bareCall
        case requestToolXML
        case embeddedAPIToolJSON
        case bareNameJSON
        case bareNameKeyValue
    }

    private enum BareNameKeyValueTailState {
        case none
        case partial
        case started
    }

    // MARK: - Initialization

    /// Initialize with a specific tool call format.
    /// - Parameters:
    ///   - format: The tool call format to use (defaults to `.json` for standard JSON format)
    ///   - tools: Optional tool schemas for type-aware parsing
    public init(
        format: ToolCallFormat = .json,
        tools: [[String: any Sendable]]? = nil,
        stripOnly: Bool = false
    ) {
        self.parser = format.createParser()
        self.tools = tools
        self.stripOnly = stripOnly
        self.parsesToolCallsFromReasoningChannel = format.parsesToolCallsFromReasoningChannel
        self.usesTaggedOnlyReasoningExtraction = format.usesTaggedOnlyReasoningExtraction
        self.preservesReasoningTextAroundToolCalls =
            format.preservesReasoningTextAroundToolCalls
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
        if suppressingTextAfterInlineToolCall {
            return nil
        }
        if isInlineFormat {
            return processInlineChunk(chunk)
        }
        return processTaggedChunk(chunk, allowInlineFallback: true)
    }

    /// Process a chunk using only the parser's explicit wrapper tags. This is
    /// used for reasoning-channel extraction in formats whose prose can mention
    /// bare function-looking text before the real native tool envelope.
    public func processTaggedProtocolChunk(_ chunk: String) -> String? {
        processTaggedChunk(chunk, allowInlineFallback: false)
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
        defer {
            suppressingTextAfterInlineToolCall = false
            inlineToolCallKind = .json
        }
        if state == .normal, !leadingTextBeforeToolCall.isEmpty {
            let visible = leadingTextBeforeToolCall
            leadingTextBeforeToolCall = ""
            return visible.isEmpty ? nil : visible
        }
        guard
            state == .collectingToolCall || state == .potentialToolCall
                || state == .collectingInlineToolCall
        else { return nil }
        guard !toolCallBuffer.isEmpty else {
            state = .normal
            return nil
        }

        let parsed = parser.parseEOS(toolCallBuffer, tools: tools)
        recordToolCalls(parsed)
        let suppressUnparsedInlineToolIntent =
            parsed.isEmpty
            && state == .collectingInlineToolCall
            && parser.supportsInlineJSONToolFallback
            && looksLikeExplicitInlineToolIntent(toolCallBuffer)
        let suppressUnparsedTaggedProtocolTail =
            parsed.isEmpty
            && (state == .collectingToolCall || state == .potentialToolCall)
            && looksLikeTaggedProtocolTail(toolCallBuffer)
        let unparsedText: String?
        if parsed.isEmpty {
            if suppressUnparsedInlineToolIntent {
                unparsedText = nil
            } else if suppressUnparsedTaggedProtocolTail {
                unparsedText =
                    leadingTextBeforeToolCall.trimmingCharacters(in: .whitespacesAndNewlines)
                        .isEmpty
                    ? nil
                    : leadingTextBeforeToolCall
            } else {
                unparsedText = leadingTextBeforeToolCall + toolCallBuffer
            }
        } else {
            unparsedText = nil
        }

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
            let inlineText = leadingTextBeforeToolCall + chunk
            leadingTextBeforeToolCall = ""

            if let callIndex = firstInlineActionJSONToolCallStart(in: inlineText) {
                let leading = String(inlineText[..<callIndex])
                inlineToolCallKind = .actionJSON
                toolCallBuffer = String(inlineText[callIndex...])
                state = .collectingInlineToolCall

                if inlineToolCallComplete(toolCallBuffer),
                    let toolCall = parser.parse(content: toolCallBuffer, tools: tools)
                {
                    recordToolCall(toolCall)
                    toolCallBuffer = ""
                    state = .normal
                    suppressingTextAfterInlineToolCall = true
                }
                return visibleInlineLeading(leading)
            }

            if let pendingIndex = partialInlineActionJSONToolCallStart(in: inlineText) {
                let visible = String(inlineText[..<pendingIndex])
                leadingTextBeforeToolCall = String(inlineText[pendingIndex...])
                return visible.isEmpty ? nil : visible
            }

            if let callIndex = firstInlineEmbeddedAPIToolJSONStart(in: inlineText) {
                let leading = String(inlineText[..<callIndex])
                inlineToolCallKind = .embeddedAPIToolJSON
                toolCallBuffer = String(inlineText[callIndex...])
                state = .collectingInlineToolCall

                if inlineToolCallComplete(toolCallBuffer),
                    let toolCall = parser.parse(content: toolCallBuffer, tools: tools)
                {
                    recordToolCall(toolCall)
                    toolCallBuffer = ""
                    state = .normal
                    suppressingTextAfterInlineToolCall = true
                }
                return visibleInlineLeading(leading)
            }

            if let pendingIndex = partialInlineEmbeddedAPIToolJSONStart(in: inlineText) {
                let visible = String(inlineText[..<pendingIndex])
                leadingTextBeforeToolCall = String(inlineText[pendingIndex...])
                return visible.isEmpty ? nil : visible
            }

            if let callIndex = firstInlinePythonicCallListStart(in: inlineText) {
                let leading = String(inlineText[..<callIndex])
                inlineToolCallKind = .functionCall
                toolCallBuffer = String(inlineText[callIndex...])
                state = .collectingInlineToolCall

                if let toolCall = parser.parse(content: toolCallBuffer, tools: tools) {
                    recordToolCall(toolCall)
                    toolCallBuffer = ""
                    state = .normal
                    suppressingTextAfterInlineToolCall = true
                }
                return visibleInlineLeading(leading)
            }

            if let pendingIndex = partialInlinePythonicCallListStart(in: inlineText) {
                let visible = String(inlineText[..<pendingIndex])
                leadingTextBeforeToolCall = String(inlineText[pendingIndex...])
                return visible.isEmpty ? nil : visible
            }

            if let callIndex = firstInlineFunctionToolCallStart(in: inlineText) {
                let leading = String(inlineText[..<callIndex])
                inlineToolCallKind = .functionCall
                toolCallBuffer = String(inlineText[callIndex...])
                state = .collectingInlineToolCall

                if let toolCall = parser.parse(content: toolCallBuffer, tools: tools) {
                    recordToolCall(toolCall)
                    toolCallBuffer = ""
                    state = .normal
                    suppressingTextAfterInlineToolCall = true
                }
                return visibleInlineLeading(leading)
            }

            if let pendingIndex = partialInlineFunctionToolCallStart(in: inlineText) {
                let visible = String(inlineText[..<pendingIndex])
                leadingTextBeforeToolCall = String(inlineText[pendingIndex...])
                return visible.isEmpty ? nil : visible
            }

            if let callIndex = firstInlineRequestToolXMLToolCallStart(in: inlineText) {
                let leading = String(inlineText[..<callIndex])
                inlineToolCallKind = .requestToolXML
                toolCallBuffer = String(inlineText[callIndex...])
                state = .collectingInlineToolCall

                if requestToolXMLCallComplete(toolCallBuffer),
                    let toolCall = parser.parse(content: toolCallBuffer, tools: tools)
                {
                    recordToolCall(toolCall)
                    toolCallBuffer = ""
                    state = .normal
                    suppressingTextAfterInlineToolCall = true
                }
                return visibleInlineLeading(leading)
            }

            if let pendingIndex = partialInlineRequestToolXMLToolCallStart(in: inlineText) {
                let visible = String(inlineText[..<pendingIndex])
                leadingTextBeforeToolCall = String(inlineText[pendingIndex...])
                return visible.isEmpty ? nil : visible
            }

            if let callIndex = firstInlineBareNameJSONToolCallStart(in: inlineText) {
                let leading = String(inlineText[..<callIndex])
                inlineToolCallKind = .bareNameJSON
                toolCallBuffer = String(inlineText[callIndex...])
                state = .collectingInlineToolCall

                if let toolCall = parser.parse(content: toolCallBuffer, tools: tools) {
                    recordToolCall(toolCall)
                    toolCallBuffer = ""
                    state = .normal
                    suppressingTextAfterInlineToolCall = true
                }
                return visibleInlineLeading(leading)
            }

            if let pendingIndex = partialInlineBareNameJSONToolCallStart(in: inlineText) {
                let visible = String(inlineText[..<pendingIndex])
                leadingTextBeforeToolCall = String(inlineText[pendingIndex...])
                return visible.isEmpty ? nil : visible
            }

            if let callIndex = firstInlineBareNameKeyValueToolCallStart(in: inlineText) {
                let leading = String(inlineText[..<callIndex])
                inlineToolCallKind = .bareNameKeyValue
                toolCallBuffer = String(inlineText[callIndex...])
                state = .collectingInlineToolCall

                if bareNameKeyValueCallComplete(toolCallBuffer),
                    let toolCall = parser.parse(content: toolCallBuffer, tools: tools)
                {
                    recordToolCall(toolCall)
                    toolCallBuffer = ""
                    state = .normal
                    suppressingTextAfterInlineToolCall = true
                }
                return visibleInlineLeading(leading)
            }

            if let pendingIndex = partialInlineBareNameKeyValueToolCallStart(in: inlineText) {
                let visible = String(inlineText[..<pendingIndex])
                leadingTextBeforeToolCall = String(inlineText[pendingIndex...])
                return visible.isEmpty ? nil : visible
            }

            // Check if this chunk starts what looks like a JSON tool call
            if let braceIndex = inlineText.firstIndex(of: "{") {
                let leading = String(inlineText[..<braceIndex])
                let jsonPart = String(inlineText[braceIndex...])
                inlineToolCallKind = .json
                toolCallBuffer = jsonPart
                state = .collectingInlineToolCall

                if let toolCall = parser.parse(content: toolCallBuffer, tools: tools) {
                    recordToolCall(toolCall)
                    toolCallBuffer = ""
                    state = .normal
                    return visibleInlineLeading(leading)
                }

                // Still collecting — check if braces are balanced (would mean parse
                // failed on complete JSON, so it's not a tool call)
                if jsonBracesBalanced(toolCallBuffer) {
                    state = .normal
                    let buffer = toolCallBuffer
                    toolCallBuffer = ""
                    return leading + buffer
                }

                return visibleInlineLeading(leading)
            }

            // No brace seen — pass through as regular text
            return inlineText

        case .potentialToolCall, .collectingToolCall, .collectingInlineToolCall:
            toolCallBuffer += chunk

            if shouldAttemptInlineToolParse(toolCallBuffer),
                let toolCall = parser.parse(content: toolCallBuffer, tools: tools)
            {
                recordToolCall(toolCall)
                toolCallBuffer = ""
                state = .normal
                if inlineToolCallKind == .actionJSON
                    || inlineToolCallKind == .functionCall
                    || inlineToolCallKind == .bareCall
                    || inlineToolCallKind == .bareNameJSON
                    || inlineToolCallKind == .bareNameKeyValue
                {
                    suppressingTextAfterInlineToolCall = true
                }
                return nil
            }

            // If the explicit inline candidate is complete but parse failed,
            // keep tool-shaped DSV4 fallback bytes out of the visible answer.
            if inlineToolCallComplete(toolCallBuffer) {
                let suppressUnparsedInlineToolIntent =
                    parser.supportsInlineJSONToolFallback
                    && looksLikeExplicitInlineToolIntent(toolCallBuffer)
                state = .normal
                let buffer = toolCallBuffer
                toolCallBuffer = ""
                inlineToolCallKind = .json
                return suppressUnparsedInlineToolIntent ? nil : buffer
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

    private func inlineToolCallComplete(_ text: String) -> Bool {
        switch inlineToolCallKind {
        case .json, .actionJSON:
            return jsonBracesBalanced(text)
        case .embeddedAPIToolJSON:
            return embeddedAPIToolJSONCallBalanced(text)
        case .functionCall:
            return functionCallParenthesesBalanced(text)
        case .bareCall:
            return bareCallBracesBalanced(text)
        case .requestToolXML:
            return requestToolXMLCallComplete(text)
        case .bareNameJSON:
            return bareNameJSONCallBalanced(text)
        case .bareNameKeyValue:
            return bareNameKeyValueCallComplete(text)
        }
    }

    private func shouldAttemptInlineToolParse(_ text: String) -> Bool {
        switch inlineToolCallKind {
        case .actionJSON, .requestToolXML, .embeddedAPIToolJSON, .bareNameKeyValue:
            return inlineToolCallComplete(text)
        case .json, .functionCall, .bareCall, .bareNameJSON:
            return true
        }
    }

    private func bareCallBracesBalanced(_ text: String) -> Bool {
        guard let open = text.firstIndex(of: "{") else { return false }
        return jsonBracesBalanced(String(text[open...]))
    }

    private func bareNameJSONCallBalanced(_ text: String) -> Bool {
        guard let open = text.firstIndex(of: "{") else { return false }
        return jsonBracesBalanced(String(text[open...]))
    }

    private func functionCallParenthesesBalanced(_ text: String) -> Bool {
        guard let open = text.firstIndex(of: "(") else { return false }
        var depth = 0
        var quote: Character?
        var escaped = false
        var sawClose = false
        var cursor = open
        while cursor < text.endIndex {
            let ch = text[cursor]
            defer { cursor = text.index(after: cursor) }
            if escaped {
                escaped = false
                continue
            }
            if ch == "\\" {
                escaped = quote != nil
                continue
            }
            if let activeQuote = quote {
                if ch == activeQuote { quote = nil }
                continue
            }
            if ch == "\"" || ch == "'" {
                quote = ch
                continue
            }
            if ch == "(" {
                depth += 1
            } else if ch == ")" {
                depth -= 1
                if depth == 0 {
                    sawClose = true
                    break
                }
                if depth < 0 { return false }
            }
        }
        return sawClose
    }

    private func looksLikeExplicitInlineToolIntent(_ text: String) -> Bool {
        let compact = text.unicodeScalars.filter {
            !CharacterSet.whitespacesAndNewlines.contains($0)
        }.map(String.init).joined()
        if compact.hasPrefix("{") {
            return inlineFunctionToolNames().contains { name in
                compact.contains(#""tool":"\#(name)""#)
                    || compact.contains(#""tool_name":"\#(name)""#)
                    || compact.contains(#""function":{"name":"\#(name)""#)
                    || compact.contains(#""function":{"#)
                        && compact.contains(#""name":"\#(name)""#)
            }
        }
        if compact.hasPrefix("action:{") || compact.hasPrefix("action:json{")
            || compact.hasPrefix(":{")
        {
            return inlineFunctionToolNames().contains { name in
                compact.contains(#""tool":"\#(name)""#)
                    || compact.contains(#""tool_name":"\#(name)""#)
                    || compact.contains(#""name":"\#(name)""#)
                    || compact.contains(#""function":{"name":"\#(name)""#)
            }
        }
        return inlineFunctionToolNames().contains {
            compact.hasPrefix("\($0)(")
                || compact.hasPrefix("[\($0)(")
                || compact.hasPrefix("\($0){")
                || compact.hasPrefix("\($0):json{")
                || compact.hasPrefix("\($0)```")
        } || firstInlineRequestToolXMLToolCallStart(in: text) != nil
            || partialInlineRequestToolXMLToolCallStart(in: text) != nil
            || firstInlineEmbeddedAPIToolJSONStart(in: text) != nil
            || partialInlineEmbeddedAPIToolJSONStart(in: text) != nil
            || firstInlineBareNameKeyValueToolCallStart(in: text) != nil
            || partialInlineBareNameKeyValueToolCallStart(in: text) != nil
    }

    private func looksLikeTaggedProtocolTail(_ text: String) -> Bool {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.hasPrefix("<") || trimmed.hasPrefix("[") else { return false }

        let tags =
            parser.startTagAliases + parser.endTagAliases
            + parser.startTagPrefixes + parser.endTagPrefixes

        if tags.contains(where: { trimmed.hasPrefix($0) }) {
            return true
        }

        guard trimmed.count >= 2 else { return false }
        return tags.contains { tag in
            tag.hasPrefix(trimmed)
        }
    }

    private func firstInlineFunctionToolCallStart(in text: String) -> String.Index? {
        inlineFunctionToolNames()
            .compactMap { name -> String.Index? in
                var searchStart = text.startIndex
                let needle = "\(name)("
                while let range = text.range(of: needle, range: searchStart ..< text.endIndex) {
                    if isInlineFunctionBoundary(text, before: range.lowerBound) {
                        return range.lowerBound
                    }
                    searchStart = range.upperBound
                }
                return nil
            }
            .min()
    }

    private func firstBareCallToolCallStart(in text: String) -> String.Index? {
        var searchStart = text.startIndex
        while let range = text.range(of: "call:", range: searchStart ..< text.endIndex) {
            if isInlineFunctionBoundary(text, before: range.lowerBound),
               bareCallTailStartsKnownTool(in: text, afterPrefix: range.upperBound)
            {
                return range.lowerBound
            }
            searchStart = range.upperBound
        }
        return nil
    }

    private func partialBareCallToolCallStart(in text: String) -> String.Index? {
        let prefix = "call:"
        for length in stride(from: min(prefix.count - 1, text.count), through: 1, by: -1) {
            let suffix = String(text.suffix(length))
            if prefix.hasPrefix(suffix) {
                return text.index(text.endIndex, offsetBy: -length)
            }
        }
        guard let range = text.range(of: prefix),
              isInlineFunctionBoundary(text, before: range.lowerBound)
        else {
            return nil
        }
        let tail = text[range.upperBound...]
        if tail.firstIndex(of: "{") == nil {
            return range.lowerBound
        }
        return nil
    }

    private func bareCallTailStartsKnownTool(in text: String, afterPrefix: String.Index) -> Bool {
        let tail = text[afterPrefix...]
        guard let brace = tail.firstIndex(of: "{") else { return false }
        let name = tail[..<brace].trimmingCharacters(in: .whitespacesAndNewlines)
        guard !name.isEmpty else { return false }
        if let tools, !tools.isEmpty {
            return inlineFunctionToolNames().contains(String(name))
        }
        return name.allSatisfy { $0 == "_" || $0.isLetter || $0.isNumber }
    }

    private func firstInlineActionJSONToolCallStart(in text: String) -> String.Index? {
        var searchStart = text.startIndex
        while let range = text.range(of: "action:", range: searchStart ..< text.endIndex) {
            if isInlineFunctionBoundary(text, before: range.lowerBound),
                inlineActionJSONTailStartsObject(in: text, afterPrefix: range.upperBound)
            {
                return range.lowerBound
            }
            searchStart = range.upperBound
        }
        return nil
    }

    private func partialInlineActionJSONToolCallStart(in text: String) -> String.Index? {
        guard !text.isEmpty else { return nil }
        var cursor = text.startIndex
        while cursor < text.endIndex {
            if isInlineFunctionBoundary(text, before: cursor) {
                let suffix = String(text[cursor...])
                if "action:".hasPrefix(suffix) && suffix.count < "action:".count {
                    return cursor
                }
                if suffix.hasPrefix("action:") {
                    guard
                        let afterPrefix = text.index(
                            cursor,
                            offsetBy: "action:".count,
                            limitedBy: text.endIndex)
                    else { return nil }
                    let objectStart = inlineActionJSONTailObjectStart(
                        in: text,
                        afterPrefix: afterPrefix)
                    if objectStart == text.endIndex {
                        return cursor
                    }
                }
            }
            cursor = text.index(after: cursor)
        }
        return nil
    }

    private func inlineActionJSONTailStartsObject(
        in text: String,
        afterPrefix: String.Index
    ) -> Bool {
        let objectStart = inlineActionJSONTailObjectStart(in: text, afterPrefix: afterPrefix)
        return objectStart < text.endIndex && text[objectStart] == "{"
    }

    private func inlineActionJSONTailObjectStart(
        in text: String,
        afterPrefix: String.Index
    ) -> String.Index {
        var cursor = afterPrefix
        while cursor < text.endIndex, isInlineWhitespace(text[cursor]) {
            cursor = text.index(after: cursor)
        }
        if text[cursor...].lowercased().hasPrefix("json") {
            guard
                let afterJSON = text.index(cursor, offsetBy: 4, limitedBy: text.endIndex)
            else { return cursor }
            cursor = afterJSON
            while cursor < text.endIndex, isInlineWhitespace(text[cursor]) {
                cursor = text.index(after: cursor)
            }
        }
        return cursor
    }

    private func firstInlineEmbeddedAPIToolJSONStart(in text: String) -> String.Index? {
        var searchStart = text.startIndex
        while let range = text.range(
            of: Self.embeddedAPIToolJSONPrefix,
            range: searchStart ..< text.endIndex)
        {
            let objectStart = embeddedAPIToolJSONTailObjectStart(
                in: text,
                afterPrefix: range.upperBound)
            if objectStart < text.endIndex, text[objectStart] == "{" {
                return range.lowerBound
            }
            searchStart = range.upperBound
        }
        return nil
    }

    private func partialInlineEmbeddedAPIToolJSONStart(in text: String) -> String.Index? {
        guard !text.isEmpty else { return nil }
        var cursor = text.startIndex
        while cursor < text.endIndex {
            if isInlineFunctionBoundary(text, before: cursor) || text[cursor] == "_" {
                let suffix = String(text[cursor...])
                if Self.embeddedAPIToolJSONPrefix.hasPrefix(suffix)
                    && suffix.count < Self.embeddedAPIToolJSONPrefix.count
                {
                    return cursor
                }
                if suffix.hasPrefix(Self.embeddedAPIToolJSONPrefix) {
                    guard
                        let afterPrefix = text.index(
                            cursor,
                            offsetBy: Self.embeddedAPIToolJSONPrefix.count,
                            limitedBy: text.endIndex)
                    else { return nil }
                    let objectStart = embeddedAPIToolJSONTailObjectStart(
                        in: text,
                        afterPrefix: afterPrefix)
                    if objectStart == text.endIndex {
                        return cursor
                    }
                }
            }
            cursor = text.index(after: cursor)
        }
        return nil
    }

    private func embeddedAPIToolJSONTailObjectStart(
        in text: String,
        afterPrefix: String.Index
    ) -> String.Index {
        var cursor = afterPrefix
        while cursor < text.endIndex, isInlineWhitespace(text[cursor]) {
            cursor = text.index(after: cursor)
        }
        return cursor
    }

    private func embeddedAPIToolJSONCallBalanced(_ text: String) -> Bool {
        guard let open = text.firstIndex(of: "{") else { return false }
        return jsonBracesBalanced(String(text[open...]))
    }

    private func partialInlineFunctionToolCallStart(in text: String) -> String.Index? {
        guard !text.isEmpty else { return nil }
        let needles = inlineFunctionToolNames().map { "\($0)(" }
        var best: String.Index?
        var cursor = text.startIndex
        while cursor < text.endIndex {
            if isInlineFunctionBoundary(text, before: cursor) {
                let suffix = String(text[cursor...])
                if needles.contains(where: { $0.hasPrefix(suffix) && suffix.count < $0.count }) {
                    best = cursor
                }
            }
            cursor = text.index(after: cursor)
        }
        return best
    }

    private func firstInlinePythonicCallListStart(in text: String) -> String.Index? {
        var cursor = text.startIndex
        while let bracket = text[cursor...].firstIndex(of: "[") {
            let afterBracket = text.index(after: bracket)
            var nameStart = afterBracket
            while nameStart < text.endIndex, isInlineWhitespace(text[nameStart]) {
                nameStart = text.index(after: nameStart)
            }
            for name in inlineFunctionToolNames() {
                if text[nameStart...].hasPrefix("\(name)(") {
                    return bracket
                }
            }
            cursor = afterBracket
        }
        return nil
    }

    private func partialInlinePythonicCallListStart(in text: String) -> String.Index? {
        guard !text.isEmpty else { return nil }
        var cursor = text.startIndex
        while let bracket = text[cursor...].firstIndex(of: "[") {
            let suffix = String(text[bracket...])
            if suffix.trimmingCharacters(in: .whitespacesAndNewlines) == "[" {
                return bracket
            }
            let afterBracket = text.index(after: bracket)
            var nameStart = afterBracket
            while nameStart < text.endIndex, isInlineWhitespace(text[nameStart]) {
                nameStart = text.index(after: nameStart)
            }
            let tail = String(text[nameStart...])
            for name in inlineFunctionToolNames() {
                let needle = "\(name)("
                if needle.hasPrefix(tail) && tail.count < needle.count {
                    return bracket
                }
            }
            cursor = afterBracket
        }
        return nil
    }

    private func firstInlineRequestToolXMLToolCallStart(in text: String) -> String.Index? {
        Self.requestToolXMLPrefixes
            .compactMap { text.range(of: $0)?.lowerBound }
            .min()
    }

    private func partialInlineRequestToolXMLToolCallStart(in text: String) -> String.Index? {
        guard !text.isEmpty else { return nil }
        var best: String.Index?
        var cursor = text.startIndex
        while cursor < text.endIndex {
            if isInlineFunctionBoundary(text, before: cursor) || text[cursor] == "_" {
                let suffix = String(text[cursor...])
                if Self.requestToolXMLPrefixes.contains(where: {
                    $0.hasPrefix(suffix) && suffix.count < $0.count
                }) {
                    best = cursor
                    break
                }
            }
            cursor = text.index(after: cursor)
        }
        return best
    }

    private func requestToolXMLCallComplete(_ text: String) -> Bool {
        text.contains("</invoke>")
    }

    private func firstInlineBareNameJSONToolCallStart(in text: String) -> String.Index? {
        var best: String.Index?
        var cursor = text.startIndex
        while cursor < text.endIndex {
            if isInlineFunctionBoundary(text, before: cursor) {
                let (candidate, afterName) = inlineIdentifier(in: text, at: cursor)
                if !candidate.isEmpty,
                    inlineFunctionToolNames().contains(where: {
                        resolvesInlineFunctionName(candidate, to: $0)
                    }),
                    bareNameJSONTailStartsObject(in: text, afterName: afterName)
                {
                    best = minIndex(best, cursor)
                }
            }
            cursor = text.index(after: cursor)
        }
        return best
    }

    private func partialInlineBareNameJSONToolCallStart(in text: String) -> String.Index? {
        guard !text.isEmpty else { return nil }
        var best: String.Index?
        var cursor = text.startIndex
        while cursor < text.endIndex {
            if isInlineFunctionBoundary(text, before: cursor) {
                let suffix = String(text[cursor...])
                for name in inlineFunctionToolNames() {
                    if inlineFunctionNamePrefix(suffix, matches: name) && suffix.count < name.count {
                        best = cursor
                        break
                    }
                    let (candidate, afterName) = inlineIdentifier(in: text, at: cursor)
                    guard !candidate.isEmpty,
                        resolvesInlineFunctionName(candidate, to: name)
                    else { continue }
                    let tailCursor = bareNameJSONTailObjectStart(in: text, afterName: afterName)
                    if tailCursor == text.endIndex {
                        best = cursor
                        break
                    }
                    if text[tailCursor] != "{" {
                        continue
                    }
                }
            }
            cursor = text.index(after: cursor)
        }
        return best
    }

    private func inlineIdentifier(
        in text: String,
        at start: String.Index
    ) -> (String, String.Index) {
        var cursor = start
        while cursor < text.endIndex, isInlineKeyValueIdentifierCharacter(text[cursor]) {
            cursor = text.index(after: cursor)
        }
        return (String(text[start..<cursor]), cursor)
    }

    private func minIndex(_ lhs: String.Index?, _ rhs: String.Index) -> String.Index {
        guard let lhs else { return rhs }
        return rhs < lhs ? rhs : lhs
    }

    private func inlineFunctionNamePrefix(_ suffix: String, matches expected: String) -> Bool {
        if expected.hasPrefix(suffix) { return true }
        return expected.replacingOccurrences(of: "c", with: "")
            .hasPrefix(suffix.replacingOccurrences(of: "c", with: ""))
    }

    private func resolvesInlineFunctionName(_ raw: String, to expected: String) -> Bool {
        if raw == expected { return true }
        if raw.replacingOccurrences(of: "c", with: "")
            == expected.replacingOccurrences(of: "c", with: "")
        {
            return true
        }
        return editDistanceAtMostOne(raw, expected)
    }

    private func editDistanceAtMostOne(_ lhs: String, _ rhs: String) -> Bool {
        let left = Array(lhs)
        let right = Array(rhs)
        if abs(left.count - right.count) > 1 { return false }

        var i = 0
        var j = 0
        var edits = 0
        while i < left.count && j < right.count {
            if left[i] == right[j] {
                i += 1
                j += 1
                continue
            }
            edits += 1
            if edits > 1 { return false }
            if left.count > right.count {
                i += 1
            } else if right.count > left.count {
                j += 1
            } else {
                i += 1
                j += 1
            }
        }
        if i < left.count || j < right.count { edits += 1 }
        return edits <= 1
    }

    private func firstInlineBareNameKeyValueToolCallStart(in text: String) -> String.Index? {
        inlineFunctionToolNames()
            .compactMap { name -> String.Index? in
                var searchStart = text.startIndex
                while let range = text.range(of: name, range: searchStart ..< text.endIndex) {
                    if isInlineFunctionBoundary(text, before: range.lowerBound)
                        && bareNameKeyValueTailState(
                            in: text,
                            toolName: name,
                            afterName: range.upperBound) == .started
                    {
                        return range.lowerBound
                    }
                    searchStart = range.upperBound
                }
                return nil
            }
            .min()
    }

    private func partialInlineBareNameKeyValueToolCallStart(in text: String) -> String.Index? {
        guard !text.isEmpty else { return nil }
        var best: String.Index?
        var cursor = text.startIndex
        while cursor < text.endIndex {
            if isInlineFunctionBoundary(text, before: cursor) {
                let suffix = String(text[cursor...])
                for name in inlineFunctionToolNames() {
                    if name.hasPrefix(suffix) && suffix.count < name.count {
                        best = cursor
                        break
                    }
                    guard suffix.hasPrefix(name),
                        let afterName = text.index(
                            cursor,
                            offsetBy: name.count,
                            limitedBy: text.endIndex)
                    else { continue }
                    if bareNameKeyValueTailState(
                        in: text,
                        toolName: name,
                        afterName: afterName) == .partial
                    {
                        best = cursor
                        break
                    }
                }
            }
            cursor = text.index(after: cursor)
        }
        return best
    }

    private func bareNameKeyValueTailState(
        in text: String,
        toolName: String,
        afterName: String.Index
    ) -> BareNameKeyValueTailState {
        guard afterName <= text.endIndex else { return .none }
        guard afterName < text.endIndex else { return .partial }
        guard isInlineWhitespace(text[afterName]) else { return .none }

        var cursor = afterName
        while cursor < text.endIndex, isInlineWhitespace(text[cursor]) {
            cursor = text.index(after: cursor)
        }
        guard cursor < text.endIndex else { return .partial }
        guard text[cursor] != "{", !text[cursor...].hasPrefix("```") else { return .none }

        let lineEnd = text[cursor...].firstIndex(where: \.isNewline) ?? text.endIndex
        let line = text[cursor ..< lineEnd].trimmingCharacters(in: .whitespaces)
        guard !line.isEmpty else { return .partial }
        if isInlineKeyValueArgumentLine(line, toolName: toolName) {
            return .started
        }
        if isPartialInlineKeyValueArgumentLine(line, toolName: toolName) {
            return .partial
        }
        if line.allSatisfy(isInlineKeyValueIdentifierCharacter) {
            return .partial
        }
        return .none
    }

    private func bareNameKeyValueCallComplete(_ text: String) -> Bool {
        var leadingStart = text.startIndex
        while leadingStart < text.endIndex, isInlineWhitespace(text[leadingStart]) {
            leadingStart = text.index(after: leadingStart)
        }
        let trimmed = String(text[leadingStart...])
        guard
            let name = inlineFunctionToolNames().first(where: { name in
                trimmed.hasPrefix(name)
                    && trimmed.index(trimmed.startIndex, offsetBy: name.count) <= trimmed.endIndex
            }),
            let afterName = trimmed.index(
                trimmed.startIndex,
                offsetBy: name.count,
                limitedBy: trimmed.endIndex),
            afterName < trimmed.endIndex,
            isInlineWhitespace(trimmed[afterName])
        else { return false }

        let tail = String(trimmed[afterName...])
        let lines = tail.split(omittingEmptySubsequences: false, whereSeparator: \.isNewline)
        let endsWithNewline = tail.last?.isNewline == true
        var sawKeyValue = false
        for (index, rawLine) in lines.enumerated() {
            let isLastLine = index == lines.count - 1
            let line = rawLine.trimmingCharacters(in: .whitespaces)
            if line.isEmpty {
                if isLastLine { return false }
                if sawKeyValue { return true }
                continue
            }
            if isInlineKeyValueArgumentLine(line, toolName: name) {
                sawKeyValue = true
                continue
            }
            if sawKeyValue {
                if isLastLine && !endsWithNewline
                    && (line.allSatisfy(isInlineKeyValueIdentifierCharacter)
                        || isPartialInlineKeyValueArgumentLine(line, toolName: name))
                {
                    return false
                }
                return true
            }
            return false
        }
        return false
    }

    private func isInlineKeyValueArgumentLine(_ line: String, toolName: String) -> Bool {
        guard let separator = firstInlineKeyValueSeparator(in: line) else { return false }
        let key = line[..<separator].trimmingCharacters(in: .whitespaces)
        guard !key.isEmpty, key.allSatisfy(isInlineKeyValueIdentifierCharacter) else {
            return false
        }
        let value = line[line.index(after: separator)...]
            .trimmingCharacters(in: .whitespaces)
        guard !value.isEmpty else { return false }
        return isAllowedInlineKeyValueArgumentName(String(key), toolName: toolName)
    }

    private func isPartialInlineKeyValueArgumentLine(_ line: String, toolName: String) -> Bool {
        guard let separator = firstInlineKeyValueSeparator(in: line) else { return false }
        let key = line[..<separator].trimmingCharacters(in: .whitespaces)
        guard !key.isEmpty, key.allSatisfy(isInlineKeyValueIdentifierCharacter) else {
            return false
        }
        let value = line[line.index(after: separator)...]
            .trimmingCharacters(in: .whitespaces)
        return value.isEmpty && isAllowedInlineKeyValueArgumentName(String(key), toolName: toolName)
    }

    private func isAllowedInlineKeyValueArgumentName(_ key: String, toolName: String) -> Bool {
        if let names = inlineArgumentNames(for: toolName), names.contains(key) {
            return true
        }
        if Self.schemaLessKeyValueArgumentNames.contains(key) {
            return true
        }
        if let names = inlineArgumentNames(for: toolName), !names.isEmpty {
            return names.contains(key)
        }
        return false
    }

    private func inlineArgumentNames(for toolName: String) -> Set<String>? {
        guard let tools else { return nil }
        for tool in tools {
            let function = (tool["function"] as? [String: any Sendable]) ?? tool
            guard function["name"] as? String == toolName else { continue }
            guard
                let parameters = function["parameters"] as? [String: any Sendable],
                let properties = parameters["properties"] as? [String: any Sendable]
            else { return nil }
            return Set(properties.keys)
        }
        return nil
    }

    private func firstInlineKeyValueSeparator(in line: String) -> String.Index? {
        let equals = line.firstIndex(of: "=")
        let colon = line.firstIndex(of: ":")
        switch (equals, colon) {
        case (let e?, let c?):
            return e < c ? e : c
        case (let e?, nil):
            return e
        case (nil, let c?):
            return c
        case (nil, nil):
            return nil
        }
    }

    private func isInlineKeyValueIdentifierCharacter(_ character: Character) -> Bool {
        character == "_" || character.isLetter || character.isNumber
    }

    private static let schemaLessKeyValueArgumentNames: Set<String> = [
        "path",
        "start_line",
        "end_line",
        "content",
        "command",
        "query",
        "pattern",
        "replacement",
        "old",
        "new",
        "message",
        "cwd",
        "recursive",
    ]

    private static let requestToolXMLPrefixes = [
        "_only:request_tool<invoke>",
        "request_tool<invoke>",
    ]

    private static let embeddedAPIToolJSONPrefix = "_only_call_one_tools_without_parameters"

    private func bareNameJSONTailStartsObject(in text: String, afterName: String.Index) -> Bool {
        let cursor = bareNameJSONTailObjectStart(in: text, afterName: afterName)
        return cursor < text.endIndex && text[cursor] == "{"
    }

    private func bareNameJSONTailObjectStart(
        in text: String,
        afterName: String.Index
    ) -> String.Index {
        var cursor = afterName
        while cursor < text.endIndex, isInlineWhitespace(text[cursor]) {
            cursor = text.index(after: cursor)
        }
        cursor = consumeOptionalJSONLabel(in: text, at: cursor)
        guard cursor < text.endIndex else { return cursor }
        guard text[cursor...].hasPrefix("```") else { return cursor }

        var fenceCursor = cursor
        for _ in 0..<3 {
            guard fenceCursor < text.endIndex else { return cursor }
            fenceCursor = text.index(after: fenceCursor)
        }
        while fenceCursor < text.endIndex,
            text[fenceCursor] != "\n",
            text[fenceCursor] != "\r"
        {
            fenceCursor = text.index(after: fenceCursor)
        }
        while fenceCursor < text.endIndex, isInlineWhitespace(text[fenceCursor]) {
            fenceCursor = text.index(after: fenceCursor)
        }
        return fenceCursor
    }

    private func consumeOptionalJSONLabel(in text: String, at cursor: String.Index) -> String.Index {
        guard cursor < text.endIndex, text[cursor] == ":" else { return cursor }
        var labelStart = text.index(after: cursor)
        while labelStart < text.endIndex, isInlineWhitespace(text[labelStart]) {
            labelStart = text.index(after: labelStart)
        }
        guard text[labelStart...].lowercased().hasPrefix("json") else { return cursor }
        guard
            let labelEnd = text.index(labelStart, offsetBy: 4, limitedBy: text.endIndex)
        else { return cursor }
        if labelEnd < text.endIndex,
            !isInlineWhitespace(text[labelEnd]),
            text[labelEnd] != "{",
            !text[labelEnd...].hasPrefix("```")
        {
            return cursor
        }
        var next = labelEnd
        while next < text.endIndex, isInlineWhitespace(text[next]) {
            next = text.index(after: next)
        }
        return next
    }

    private func isInlineWhitespace(_ character: Character) -> Bool {
        character.unicodeScalars.allSatisfy {
            CharacterSet.whitespacesAndNewlines.contains($0)
        }
    }

    private func isInlineFunctionBoundary(_ text: String, before index: String.Index) -> Bool {
        guard index > text.startIndex else { return true }
        let previous = text[text.index(before: index)]
        return !(previous.isLetter || previous.isNumber || previous == "_")
    }

    private func inlineFunctionToolNames() -> [String] {
        let schemaNames = tools?.compactMap { tool -> String? in
            let function = (tool["function"] as? [String: any Sendable]) ?? tool
            return function["name"] as? String
        } ?? []
        if !schemaNames.isEmpty {
            return schemaNames.sorted { $0.count > $1.count }
        }
        return [
            "file_search",
            "file_read",
            "file_tree",
            "file_write",
            "file_edit",
            "shell_run",
            "git_status",
            "git_diff",
            "git_commit",
        ]
    }

    private func visibleInlineLeading(_ leading: String) -> String? {
        // Emit leading text verbatim. Returning nil only for *empty* leading
        // (never for whitespace-only) keeps ordinary prose byte-exact: a chunk
        // like " {" must not lose its leading space when the brace triggers a
        // (possibly false) inline tool-call probe.
        leading.isEmpty ? nil : leading
    }

    /// Process chunk for tagged formats.
    private func processTaggedChunk(_ chunk: String, allowInlineFallback: Bool) -> String? {
        let startTags = parser.startTagAliases
        let startTagPrefixes = parser.startTagPrefixes
        guard (!startTags.isEmpty || !startTagPrefixes.isEmpty),
            let startChar = startTagFirstChar
        else {
            return chunk
        }

        let hasTaggedStartCandidate: Bool = {
            guard state == .normal,
                let taggedStart = chunk.firstIndex(of: startChar)
            else { return false }
            let suffix = String(chunk[taggedStart...])
            guard suffix.count > 1 else { return false }
            return partialMatch(buffer: suffix, tags: startTags, prefixes: startTagPrefixes)
        }()

        if allowInlineFallback && parser.supportsBareCallToolFallback && !hasTaggedStartCandidate {
            switch state {
            case .collectingInlineToolCall where inlineToolCallKind == .bareCall:
                return processInlineChunk(chunk)
            case .collectingInlineToolCall where inlineToolCallKind == .bareNameJSON:
                return processInlineChunk(chunk)
            case .normal:
                let candidate = leadingTextBeforeToolCall + chunk
                if let callIndex = firstInlineBareNameJSONToolCallStart(in: candidate) {
                    let leading = String(candidate[..<callIndex])
                    inlineToolCallKind = .bareNameJSON
                    toolCallBuffer = String(candidate[callIndex...])
                    leadingTextBeforeToolCall = ""
                    state = .collectingInlineToolCall

                    if shouldAttemptInlineToolParse(toolCallBuffer),
                       let toolCall = parser.parse(content: toolCallBuffer, tools: tools)
                    {
                        recordToolCall(toolCall)
                        toolCallBuffer = ""
                        state = .normal
                        suppressingTextAfterInlineToolCall = true
                    }
                    return visibleInlineLeading(leading)
                }
                if let fragmentIndex = partialInlineBareNameJSONToolCallStart(in: candidate) {
                    let leading = String(candidate[..<fragmentIndex])
                    leadingTextBeforeToolCall = String(candidate[fragmentIndex...])
                    return leading.isEmpty ? nil : leading
                }
                if let callIndex = firstBareCallToolCallStart(in: candidate) {
                    let leading = String(candidate[..<callIndex])
                    inlineToolCallKind = .bareCall
                    toolCallBuffer = String(candidate[callIndex...])
                    leadingTextBeforeToolCall = ""
                    state = .collectingInlineToolCall

                    if shouldAttemptInlineToolParse(toolCallBuffer),
                       let toolCall = parser.parse(content: toolCallBuffer, tools: tools)
                    {
                        recordToolCall(toolCall)
                        toolCallBuffer = ""
                        state = .normal
                        suppressingTextAfterInlineToolCall = true
                    }
                    return visibleInlineLeading(leading)
                }
                if let fragmentIndex = partialBareCallToolCallStart(in: candidate) {
                    let leading = String(candidate[..<fragmentIndex])
                    leadingTextBeforeToolCall = String(candidate[fragmentIndex...])
                    return leading.isEmpty ? nil : leading
                }
                // A previously buffered bare-call fragment (a trailing "c"/
                // "ca"/"cal"/"call" that looked like the start of `call:`) did
                // not continue into a tool call. Flush the held text together
                // with this chunk, in order, so ordinary prose is never dropped
                // or reordered. Gemma is the only family with bare-call
                // fallback and it never enables inline-JSON fallback, so no
                // later stage would otherwise consume this candidate.
                if !leadingTextBeforeToolCall.isEmpty {
                    leadingTextBeforeToolCall = ""
                    return candidate.isEmpty ? nil : candidate
                }
            case .collectingInlineToolCall:
                break
            case .potentialToolCall, .collectingToolCall:
                break
            }
        }

        if allowInlineFallback && parser.supportsInlineJSONToolFallback && !hasTaggedStartCandidate {
            switch state {
            case .collectingInlineToolCall:
                return processInlineChunk(chunk)
            case .normal:
                if !chunk.contains(startChar) {
                    let candidate = leadingTextBeforeToolCall + chunk
                    if shouldBufferPotentialBareToolMarker(candidate) {
                        leadingTextBeforeToolCall = candidate
                        return nil
                    }
                    if !leadingTextBeforeToolCall.isEmpty {
                        if firstInlineActionJSONToolCallStart(in: candidate) != nil
                            || partialInlineActionJSONToolCallStart(in: candidate) != nil
                            || firstInlineEmbeddedAPIToolJSONStart(in: candidate) != nil
                            || partialInlineEmbeddedAPIToolJSONStart(in: candidate) != nil
                            || firstInlineFunctionToolCallStart(in: candidate) != nil
                            || partialInlineFunctionToolCallStart(in: candidate) != nil
                            || firstInlinePythonicCallListStart(in: candidate) != nil
                            || partialInlinePythonicCallListStart(in: candidate) != nil
                            || firstInlineRequestToolXMLToolCallStart(in: candidate) != nil
                            || partialInlineRequestToolXMLToolCallStart(in: candidate) != nil
                            || firstInlineBareNameJSONToolCallStart(in: candidate) != nil
                            || partialInlineBareNameJSONToolCallStart(in: candidate) != nil
                            || firstInlineBareNameKeyValueToolCallStart(in: candidate) != nil
                            || partialInlineBareNameKeyValueToolCallStart(in: candidate) != nil
                        {
                            return processInlineChunk(chunk)
                        }
                        let visible = leadingTextBeforeToolCall + chunk
                        leadingTextBeforeToolCall = ""
                        return visible
                    }
                }

                let bufferedBareToolMarker =
                    !leadingTextBeforeToolCall.isEmpty
                    && bareToolMarkerFragment(in: leadingTextBeforeToolCall) != nil
                if (!leadingTextBeforeToolCall.isEmpty && !bufferedBareToolMarker)
                    || firstInlineActionJSONToolCallStart(in: chunk) != nil
                    || partialInlineActionJSONToolCallStart(in: chunk) != nil
                    || firstInlineEmbeddedAPIToolJSONStart(in: chunk) != nil
                    || partialInlineEmbeddedAPIToolJSONStart(in: chunk) != nil
                    || firstInlineFunctionToolCallStart(in: chunk) != nil
                    || partialInlineFunctionToolCallStart(in: chunk) != nil
                    || firstInlinePythonicCallListStart(in: chunk) != nil
                    || partialInlinePythonicCallListStart(in: chunk) != nil
                    || firstInlineRequestToolXMLToolCallStart(in: chunk) != nil
                    || partialInlineRequestToolXMLToolCallStart(in: chunk) != nil
                    || firstInlineBareNameJSONToolCallStart(in: chunk) != nil
                    || partialInlineBareNameJSONToolCallStart(in: chunk) != nil
                    || firstInlineBareNameKeyValueToolCallStart(in: chunk) != nil
                    || partialInlineBareNameKeyValueToolCallStart(in: chunk) != nil
                {
                    return processInlineChunk(chunk)
                }
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

        var effectiveChunk = chunk
        if state == .normal,
            let matchedStart = firstTag(in: chunk, tags: startTags, prefixes: startTagPrefixes),
            let matchedRange = chunk.range(of: matchedStart)
        {
            let leading = String(chunk[..<matchedRange.lowerBound])
            if !leading.isEmpty {
                leadingTextBeforeToolCall += leading
            }
            effectiveChunk = String(chunk[matchedRange.lowerBound...])
        }

        guard (state == .normal && effectiveChunk.contains(startChar)) || state != .normal else {
            return effectiveChunk
        }

        toolCallBuffer += effectiveChunk
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
                let parsed = parser.parseEOS(toolCallBuffer, tools: tools)
                recordToolCalls(parsed)

                state = .normal
                toolCallBuffer = ""

                // If the token contains the start character, there may be more tool calls to come
                let leading = visibleLeadingTextBeforeToolCall(parsedToolCalls: parsed)
                leadingTextBeforeToolCall = ""
                if let trailingToken, let startChar = startTagFirstChar,
                    trailingToken.contains(startChar)
                {
                    let trailing = processTaggedChunk(
                        trailingToken,
                        allowInlineFallback: allowInlineFallback) ?? ""
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

    private func shouldBufferPotentialBareToolMarker(_ text: String) -> Bool {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmed.isEmpty,
            Self.requestToolXMLPrefixes.contains(where: {
                $0.hasPrefix(trimmed) && trimmed.count < $0.count
            })
        {
            return true
        }
        if !trimmed.isEmpty {
            for name in inlineFunctionToolNames() {
                if name.hasPrefix(trimmed) { return true }
                guard trimmed.hasPrefix(name) else { continue }
                let afterName = String(trimmed.dropFirst(name.count))
                if afterName.isEmpty { return true }
                let compactTail = afterName.unicodeScalars.filter {
                    !CharacterSet.whitespacesAndNewlines.contains($0)
                }.map(String.init).joined().lowercased()
                if compactTail.isEmpty { return true }
                if ":json".hasPrefix(compactTail) { return true }
                if compactTail.hasPrefix(":json") {
                    let afterLabel = String(compactTail.dropFirst(5))
                    return afterLabel.isEmpty
                }
            }
        }
        guard let fragment = bareToolMarkerFragment(in: text) else { return false }
        return inlineFunctionToolNames().contains { name in
            name.hasPrefix(fragment) || fragment == name
        }
    }

    private func visibleLeadingTextBeforeToolCall(parsedToolCalls: [ToolCall]) -> String {
        guard parser.supportsInlineJSONToolFallback,
            let fragment = bareToolMarkerFragment(in: leadingTextBeforeToolCall),
            parsedToolCalls.contains(where: { $0.function.name == fragment })
        else {
            return leadingTextBeforeToolCall
        }
        return ""
    }

    private func bareToolMarkerFragment(in text: String) -> String? {
        var marker = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !marker.isEmpty else { return nil }

        var removedMarkerPrefix = false
        while let first = marker.first,
            first == "-" || first == "*" || first == "`" || first == ":"
        {
            removedMarkerPrefix = true
            marker.removeFirst()
            marker = marker.trimmingCharacters(in: .whitespacesAndNewlines)
        }

        guard !marker.isEmpty || removedMarkerPrefix else { return nil }
        guard marker.allSatisfy({ $0 == "_" || $0.isLetter || $0.isNumber }) else {
            return nil
        }
        return marker
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
