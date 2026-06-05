// Copyright © 2026 Osaurus AI. All rights reserved.

public enum GenerationTextChannel {
    case content
    case reasoning
}

public func routeGenerationText(
    _ text: String,
    channel: GenerationTextChannel,
    through toolCallProcessor: ToolCallProcessor?
) -> [Generation] {
    guard let toolCallProcessor else {
        guard !text.isEmpty else { return [] }
        switch channel {
        case .content:
            return [.chunk(text)]
        case .reasoning:
            return [.reasoning(text)]
        }
    }
    if channel == .reasoning, !toolCallProcessor.parsesToolCallsFromReasoningChannel {
        return text.isEmpty ? [] : [.reasoning(text)]
    }

    var events: [Generation] = []
    let toolCallCountBeforeChunk = toolCallProcessor.toolCalls.count
    let visibleChunk: String?
    if channel == .reasoning, toolCallProcessor.usesTaggedOnlyReasoningExtraction {
        visibleChunk = toolCallProcessor.processTaggedProtocolChunk(text)
    } else {
        visibleChunk = toolCallProcessor.processChunk(text)
    }
    if let visible = visibleChunk {
        let parsedToolCallInChunk = toolCallProcessor.toolCalls.count > toolCallCountBeforeChunk
        switch channel {
        case .content:
            events.append(.chunk(visible))
        case .reasoning:
            if !parsedToolCallInChunk || toolCallProcessor.preservesReasoningTextAroundToolCalls {
                events.append(.reasoning(visible))
            }
        }
    }
    events.append(contentsOf: drainToolCallEvents(from: toolCallProcessor))
    return events
}

public func drainToolCallEvents(from toolCallProcessor: ToolCallProcessor) -> [Generation] {
    guard !toolCallProcessor.toolCalls.isEmpty else { return [] }
    let calls = toolCallProcessor.toolCalls
    toolCallProcessor.toolCalls.removeAll(keepingCapacity: true)
    return calls.map { .toolCall($0) }
}

public func flushGenerationText(
    channel: GenerationTextChannel,
    through toolCallProcessor: ToolCallProcessor?
) -> [Generation] {
    guard let toolCallProcessor else { return [] }
    if channel == .reasoning, !toolCallProcessor.parsesToolCallsFromReasoningChannel {
        return []
    }

    var events: [Generation] = []
    if let visible = toolCallProcessor.processEOS() {
        switch channel {
        case .content:
            events.append(.chunk(visible))
        case .reasoning:
            events.append(.reasoning(visible))
        }
    }
    events.append(contentsOf: drainToolCallEvents(from: toolCallProcessor))
    return events
}
