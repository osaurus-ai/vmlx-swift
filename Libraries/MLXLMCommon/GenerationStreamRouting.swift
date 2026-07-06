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
    let collectingBeforeChunk = toolCallProcessor.collectingToolCallText
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
    // Surface the growth of an in-flight tool-call envelope as a progress
    // delta, before the completed call drains below. Diffing the processor's
    // collecting buffer around the chunk keeps every per-family parser state
    // machine untouched: a call that completed (or reverted to plain text)
    // within this chunk reports no residual buffer and emits nothing here.
    // `hasPrefix` guards the boundary where one call closed and the next
    // opened inside the same chunk — the buffer then holds unrelated text, so
    // the whole new buffer is the delta.
    if let after = toolCallProcessor.collectingToolCallText, !after.isEmpty {
        if let before = collectingBeforeChunk, after.hasPrefix(before) {
            if after.count > before.count {
                events.append(.toolCallProgress(String(after.dropFirst(before.count))))
            }
        } else {
            events.append(.toolCallProgress(after))
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
