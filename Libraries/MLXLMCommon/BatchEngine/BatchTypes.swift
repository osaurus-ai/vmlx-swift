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

/// One actor-consistent view of a ``BatchEngine``'s admission capacity.
///
/// Read this through ``BatchEngine/capacitySnapshot`` when a serving layer
/// needs active, queued, and configured-capacity values from the same actor
/// turn. Reading the engine's individual diagnostic properties separately can
/// observe different scheduling turns.
///
/// `nominalAvailableCount` is configured sequence headroom, not a reservation.
/// Architecture-specific admission rules can still serialize work (for
/// example, hybrid-pool cache requests), and another request can consume the
/// headroom immediately after the snapshot is returned.
public struct BatchEngineCapacitySnapshot: Sendable, Equatable {
    /// Configured maximum number of simultaneously active sequences.
    public let configuredMaximum: Int

    /// Currently active sequences, including the direct B=1 solo path.
    public let activeCount: Int

    /// Requests waiting for engine admission.
    public let pendingCount: Int

    /// Configured headroom at this instant, or zero after shutdown.
    public let nominalAvailableCount: Int

    /// Whether the engine accepts new requests.
    public let isAcceptingRequests: Bool

    /// Whether terminal engine shutdown has begun.
    public let isShutdown: Bool

    /// Maximum concurrent active sequence count observed since engine creation.
    public let activeCountHighWatermark: Int

    public init(
        configuredMaximum: Int,
        activeCount: Int,
        pendingCount: Int,
        nominalAvailableCount: Int,
        isAcceptingRequests: Bool,
        isShutdown: Bool,
        activeCountHighWatermark: Int
    ) {
        self.configuredMaximum = configuredMaximum
        self.activeCount = activeCount
        self.pendingCount = pendingCount
        self.nominalAvailableCount = nominalAvailableCount
        self.isAcceptingRequests = isAcceptingRequests
        self.isShutdown = isShutdown
        self.activeCountHighWatermark = activeCountHighWatermark
    }
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
