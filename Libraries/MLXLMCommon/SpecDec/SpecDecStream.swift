// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Streaming wrapper around SpecDecRuntimeLinear / SpecDecRuntimeDDTree
// that emits `Generation` events (.chunk + .info) exactly like the
// non-speculative `Evaluate.generate` path does. This is what osaurus
// integrates with — same stream contract, same event types.
//
// Iter 10 deliverable: osaurus can flip `GenerateParameters.draftStrategy`
// to `.dflash(...)` or `.ddtree(...)` and consume the resulting
// `AsyncStream<Generation>` without any other API changes.

import Foundation
import MLX

/// Helper that runs a SpecDec runtime and surfaces its per-round
/// commits as a streaming `AsyncStream<Generation>`.
///
/// The runtime runs on a background `Task`; each committed batch of
/// tokens is detokenized via `NaiveStreamingDetokenizer` and yielded as
/// `.chunk(String)` and `.reasoning(String)`. A final `.info(GenerateCompletionInfo)` fires on
/// completion.
///
/// Tool-call and reasoning parsing are applied to the detokenized
/// stream using the same `ToolCallProcessor` + `ReasoningParser`
/// pipeline the non-speculative path uses (see
/// `Evaluate.TextToolTokenLoopHandler`). That keeps osaurus's chunk
/// contract byte-identical whether speculative decoding is on or off.
public enum SpecDecStream {

    /// Run a DFlash-linear generation and stream `Generation` events.
    ///
    /// - Parameters:
    ///   - args: the runtime args.
    ///   - tokenizer: used to detokenize committed tokens into chunks.
    ///   - toolCallFormat: tool-call wire format for the stream's
    ///     `.toolCall(ToolCall)` events. Pass `.json` default.
    ///   - reasoningParserName: optional JANG capability stamp for
    ///     `<think>`-style reasoning strip.
    /// - Returns: `AsyncStream<Generation>` — same event shape as
    ///   `Evaluate.generate(input:cache:parameters:context:)`.
    public static func streamDflashLinear(
        args: DFlashLinearArgs,
        tokenizer: any Tokenizer,
        toolCallFormat: ToolCallFormat = .json,
        reasoningParserName: String? = nil,
        toolSchemas: [ToolSpec]? = nil
    ) -> AsyncStream<Generation> {
        AsyncStream<Generation> { continuation in
            Task {
                do {
                    let startTime = Date()
                    let promptTokenCount = args.inputIds.dim(1)
                    var detokenizer = NaiveStreamingDetokenizer(
                        tokenizer: tokenizer)
                    let activeToolSchemas = toolSchemas?.isEmpty == false ? toolSchemas : nil
                    // No tools offered: still strip tagged tool-call markers so a
                    // model that emits an envelope anyway cannot leak raw markup.
                    let toolCallProcessor: ToolCallProcessor? =
                        activeToolSchemas.map { ToolCallProcessor(format: toolCallFormat, tools: $0) }
                        ?? (toolCallFormat.hasTaggedToolMarkers
                            ? ToolCallProcessor(format: toolCallFormat, tools: nil, stripOnly: true)
                            : nil)
                    var reasoningParser = ReasoningParser
                        .fromCapabilityName(reasoningParserName)

                    let onCommitted: ([Int32]) -> Void = { batch in
                        pushBatch(
                            tokens: batch,
                            detokenizer: &detokenizer,
                            toolCallProcessor: toolCallProcessor,
                            reasoningParser: &reasoningParser,
                            continuation: continuation)
                    }

                    let result = try SpecDecRuntimeLinear.run(
                        args, onCommitted: onCommitted)

                    // Flush any buffered content in the reasoning parser
                    // + tool-call processor before finishing.
                    flush(
                        detokenizer: &detokenizer,
                        toolCallProcessor: toolCallProcessor,
                        reasoningParser: &reasoningParser,
                        continuation: continuation)

                    let elapsed = Date().timeIntervalSince(startTime)
                    let generatedCount = result.tokenIds.count - promptTokenCount
                    let info = GenerateCompletionInfo(
                        promptTokenCount: promptTokenCount,
                        generationTokenCount: max(0, generatedCount),
                        promptTime: 0,
                        generationTime: elapsed,
                        stopReason: .length)
                    continuation.yield(.info(info))
                    continuation.finish()
                } catch {
                    // Terminate the stream on error — callers observe
                    // completion without an info event.
                    continuation.finish()
                }
            }
        }
    }

    /// Run a DDTree generation and stream `Generation` events. See
    /// ``streamDflashLinear(args:tokenizer:toolCallFormat:reasoningParserName:)``
    /// for the event contract.
    public static func streamDDTree(
        args: DDTreeArgs,
        tokenizer: any Tokenizer,
        toolCallFormat: ToolCallFormat = .json,
        reasoningParserName: String? = nil,
        toolSchemas: [ToolSpec]? = nil
    ) -> AsyncStream<Generation> {
        AsyncStream<Generation> { continuation in
            Task {
                do {
                    let startTime = Date()
                    let promptTokenCount = args.inputIds.dim(1)
                    var detokenizer = NaiveStreamingDetokenizer(
                        tokenizer: tokenizer)
                    let activeToolSchemas = toolSchemas?.isEmpty == false ? toolSchemas : nil
                    // No tools offered: still strip tagged tool-call markers so a
                    // model that emits an envelope anyway cannot leak raw markup.
                    let toolCallProcessor: ToolCallProcessor? =
                        activeToolSchemas.map { ToolCallProcessor(format: toolCallFormat, tools: $0) }
                        ?? (toolCallFormat.hasTaggedToolMarkers
                            ? ToolCallProcessor(format: toolCallFormat, tools: nil, stripOnly: true)
                            : nil)
                    var reasoningParser = ReasoningParser
                        .fromCapabilityName(reasoningParserName)

                    let onCommitted: ([Int32]) -> Void = { batch in
                        pushBatch(
                            tokens: batch,
                            detokenizer: &detokenizer,
                            toolCallProcessor: toolCallProcessor,
                            reasoningParser: &reasoningParser,
                            continuation: continuation)
                    }

                    let result = try SpecDecRuntimeDDTree.run(
                        args, onCommitted: onCommitted)

                    flush(
                        detokenizer: &detokenizer,
                        toolCallProcessor: toolCallProcessor,
                        reasoningParser: &reasoningParser,
                        continuation: continuation)

                    let elapsed = Date().timeIntervalSince(startTime)
                    let generatedCount = result.tokenIds.count - promptTokenCount
                    let info = GenerateCompletionInfo(
                        promptTokenCount: promptTokenCount,
                        generationTokenCount: max(0, generatedCount),
                        promptTime: 0,
                        generationTime: elapsed,
                        stopReason: .length)
                    continuation.yield(.info(info))
                    continuation.finish()
                } catch {
                    continuation.finish()
                }
            }
        }
    }

    // MARK: - Strategy-driven dispatch (consumed by Evaluate.generate)

    /// Build an `AsyncStream<Generation>` for a block-diffusion draft
    /// strategy without requiring the caller to pre-load the drafter.
    ///
    /// Used by `Evaluate.generate(input:cache:parameters:context:…)` when
    /// `parameters.draftStrategy ∈ {.dflash, .ddtree}`. Returns `nil`
    /// when the strategy is `.none` / `.autoregressive` / `nil` or the
    /// target model doesn't conform to both
    /// ``HiddenStateCaptureModel`` and ``TokenEmbedderModel`` — callers
    /// fall back to the existing non-speculative path.
    ///
    /// Drafter loading happens asynchronously inside the stream's Task,
    /// so the dispatch site stays synchronous and doesn't block on
    /// disk I/O.
    public static func streamViaStrategy(
        strategy: DraftStrategy,
        inputIds: MLXArray,
        context: ModelContext,
        maxNewTokens: Int,
        stopTokenIDs: Set<Int32> = [],
        temperature: Float = 0,
        toolSchemas: [ToolSpec]? = nil,
        resolver: SpecDecDrafterResolver = .shared
    ) -> AsyncStream<Generation>? {
        guard strategy.usesBlockDiffusion else { return nil }
        guard let targetCheck = context.model
            as? any (HiddenStateCaptureModel & TokenEmbedderModel)
        else { return nil }

        let (stream, continuation) = AsyncStream<Generation>.makeStream()
        let toolCallFormat = context.configuration.toolCallFormat ?? .json
        let reasoningParserName = context.configuration.reasoningParserName
        let tokenizer = context.tokenizer
        // Decode the prompt tail so the reasoning parser starts in
        // the correct state (inside reasoning if the chat template
        // prefilled an opener). See
        // `Libraries/MLXLMCommon/BatchEngine/RALPH-EDGE-TASK.md` B1.
        let promptTailText: String? = {
            let n = inputIds.ndim == 1 ? inputIds.dim(0) : inputIds.dim(inputIds.ndim - 1)
            guard n > 0 else { return nil }
            let tailLen = min(64, n)
            let start = n - tailLen
            let tail = inputIds.ndim == 1
                ? inputIds[start ..< n]
                : inputIds[.ellipsis, start ..< n]
            let ints = tail.asArray(Int32.self).map { Int($0) }
            return tokenizer.decode(tokenIds: ints, skipSpecialTokens: false)
        }()
        // Boxed non-Sendable references — SpecDec types are @unchecked
        // Sendable but the Task-capture compile-check needs a synthetic
        // nonisolated wrapper.
        let box = _SpecDecDispatchBox(
            inputIds: inputIds, target: targetCheck)

        Task { @Sendable in
            do {
                let resolved: ResolvedDrafter
                do {
                    resolved = try await resolver.resolve(strategy: strategy)
                } catch {
                    continuation.finish()
                    return
                }
                let startTime = Date()
                let promptTokenCount = box.inputIds.dim(1)
                var detokenizer = NaiveStreamingDetokenizer(
                    tokenizer: tokenizer)
                let activeToolSchemas = toolSchemas?.isEmpty == false ? toolSchemas : nil
                // No tools offered: still strip tagged tool-call markers so a
                // model that emits an envelope anyway cannot leak raw markup.
                let toolCallProcessor: ToolCallProcessor? =
                    activeToolSchemas.map { ToolCallProcessor(format: toolCallFormat, tools: $0) }
                    ?? (toolCallFormat.hasTaggedToolMarkers
                        ? ToolCallProcessor(format: toolCallFormat, tools: nil, stripOnly: true)
                        : nil)
                var reasoningParser = ReasoningParser
                    .forPrompt(
                        stampName: reasoningParserName,
                        promptTail: promptTailText)

                let onCommitted: ([Int32]) -> Void = { batch in
                    pushBatch(
                        tokens: batch,
                        detokenizer: &detokenizer,
                        toolCallProcessor: toolCallProcessor,
                        reasoningParser: &reasoningParser,
                        continuation: continuation)
                }

                let resultCount: Int
                switch strategy {
                case .dflash(_, let blockSize):
                    _ = blockSize  // blockSize is a drafter-config field;
                                   // the runtime reads it from the loaded
                                   // drafter model (ignored here).
                    let args = DFlashLinearArgs(
                        target: box.target,
                        drafter: resolved.model,
                        targetBlockIDs: resolved.targetBlockIDs,
                        maskTokenID: resolved.maskTokenID,
                        inputIds: box.inputIds,
                        maxNewTokens: maxNewTokens,
                        stopTokenIDs: stopTokenIDs,
                        temperature: temperature)
                    let r = try SpecDecRuntimeLinear.run(
                        args, onCommitted: onCommitted)
                    resultCount = r.tokenIds.count
                case .ddtree(_, let branchingBudget, _):
                    let args = DDTreeArgs(
                        target: box.target,
                        drafter: resolved.model,
                        targetBlockIDs: resolved.targetBlockIDs,
                        maskTokenID: resolved.maskTokenID,
                        inputIds: box.inputIds,
                        maxNewTokens: maxNewTokens,
                        stopTokenIDs: stopTokenIDs,
                        temperature: temperature,
                        branchingBudget: branchingBudget)
                    let r = try SpecDecRuntimeDDTree.run(
                        args, onCommitted: onCommitted)
                    resultCount = r.tokenIds.count
                default:
                    continuation.finish()
                    return
                }

                flush(
                    detokenizer: &detokenizer,
                    toolCallProcessor: toolCallProcessor,
                    reasoningParser: &reasoningParser,
                    continuation: continuation)

                let elapsed = Date().timeIntervalSince(startTime)
                let info = GenerateCompletionInfo(
                    promptTokenCount: promptTokenCount,
                    generationTokenCount: max(0, resultCount - promptTokenCount),
                    promptTime: 0,
                    generationTime: elapsed,
                    stopReason: .length)
                continuation.yield(.info(info))
                continuation.finish()
            } catch {
                continuation.finish()
            }
        }
        return stream
    }

    // MARK: - Internals

    /// Feed one round's committed tokens through the detokenizer +
    /// reasoning parser + tool-call processor, yielding any produced
    /// events to the caller.
    private static func pushBatch(
        tokens: [Int32],
        detokenizer: inout NaiveStreamingDetokenizer,
        toolCallProcessor: ToolCallProcessor?,
        reasoningParser: inout ReasoningParser?,
        continuation: AsyncStream<Generation>.Continuation
    ) {
        for t in tokens {
            detokenizer.append(token: Int(t))
            guard let chunk = detokenizer.next() else { continue }

            // 1. Reasoning pass (if configured) — peels off <think>…
            //    segments. `.reasoning(String)` events are emitted on
            //    the stream so callers can render a think-pane UI;
            //    content continues into the tool-call processor.
            let contentPieces: [String]
            if var parser = reasoningParser {
                var pieces: [String] = []
                for segment in parser.feed(chunk) {
                    switch segment {
                    case .content(let c):
                        pieces.append(c)
                    case .reasoning(let r):
                        for event in routeGenerationText(
                            r,
                            channel: .reasoning,
                            through: toolCallProcessor
                        ) {
                            continuation.yield(event)
                        }
                    }
                }
                reasoningParser = parser
                contentPieces = pieces
            } else {
                contentPieces = [chunk]
            }

            // 2. Tool-call pass — same contract as non-speculative path.
            for piece in contentPieces {
                for event in routeGenerationText(
                    piece,
                    channel: .content,
                    through: toolCallProcessor
                ) {
                    continuation.yield(event)
                }
            }
        }
    }

    /// End-of-stream flush — same idea as
    /// `Evaluate.TextToolTokenLoopHandler.onGenerationEnd`.
    private static func flush(
        detokenizer: inout NaiveStreamingDetokenizer,
        toolCallProcessor: ToolCallProcessor?,
        reasoningParser: inout ReasoningParser?,
        continuation: AsyncStream<Generation>.Continuation
    ) {
        if var parser = reasoningParser {
            for segment in parser.flush() {
                switch segment {
                case .content(let c):
                    for event in routeGenerationText(
                        c,
                        channel: .content,
                        through: toolCallProcessor
                    ) {
                        continuation.yield(event)
                    }
                case .reasoning(let r):
                    for event in routeGenerationText(
                        r,
                        channel: .reasoning,
                        through: toolCallProcessor
                    ) {
                        continuation.yield(event)
                    }
                }
            }
            reasoningParser = parser
        }
        for event in flushGenerationText(
            channel: reasoningParser?.isInsideReasoning == true ? .reasoning : .content,
            through: toolCallProcessor
        ) {
            continuation.yield(event)
        }
    }
}

/// Sendable wrapper for the strategy-driven dispatch's non-Sendable
/// references. MLXArray and the target composite protocol are both
/// `@unchecked Sendable` at the type level but Swift's closure-capture
/// sending-safety check needs a struct-level escape hatch.
private struct _SpecDecDispatchBox: @unchecked Sendable {
    let inputIds: MLXArray
    let target: any (HiddenStateCaptureModel & TokenEmbedderModel)
}
