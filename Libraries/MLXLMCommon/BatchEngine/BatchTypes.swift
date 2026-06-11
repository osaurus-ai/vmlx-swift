// Copyright 2025 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX

// MARK: - Batch Engine Types

/// Unique identifier for a request submitted to ``BatchEngine``.
///
/// Each call to ``BatchEngine/submit(input:parameters:)`` generates a new ID
/// that can be used to track or cancel the request.
public struct BatchRequestID: Hashable, Sendable, CustomStringConvertible {
    let value: UUID

    public init() {
        self.value = UUID()
    }

    public var description: String { value.uuidString.prefix(8).lowercased() }
}

/// A token-level event yielded by ``BatchEngine`` for each active request.
///
/// Consumers iterate an `AsyncStream<BatchGeneration>` to receive tokens
/// as they are generated, and a final `.info` event with completion metrics.
///
/// ## Example
/// ```swift
/// let stream = await engine.submit(input: lmInput, parameters: params)
/// for await event in stream {
///     switch event {
///     case .token(let id):
///         // Feed to a StreamingDetokenizer
///         detokenizer.append(token: id)
///     case .info(let completionInfo):
///         print(completionInfo.summary())
///     }
/// }
/// ```
public enum BatchGeneration: Sendable {
    /// A single generated token ID.
    case token(Int)

    /// Prompt-processing progress before the first decoded token.
    case prefillProgress(PrefillProgress)

    /// Completion information with metrics. This is the final event before
    /// the stream closes.
    case info(GenerateCompletionInfo)
}

// MARK: - Internal Request Wrapper

/// Internal representation of a submitted request before it becomes an active slot.
struct BatchPendingRequest {
    let id: BatchRequestID
    let input: LMInput
    // `var` (not `let`) so the admission path can apply
    // `CacheCoordinatorConfig.resolveKVPolicy(...)` defaults before the
    // slot's cache is allocated. Per-request values set by the caller
    // always win; the coordinator only fills nils.
    var parameters: GenerateParameters
    let continuation: AsyncStream<BatchGeneration>.Continuation
    let submittedAt: Date

    init(
        input: LMInput,
        parameters: GenerateParameters,
        continuation: AsyncStream<BatchGeneration>.Continuation
    ) {
        self.id = BatchRequestID()
        self.input = input
        self.parameters = parameters
        self.continuation = continuation
        self.submittedAt = Date()
    }
}
