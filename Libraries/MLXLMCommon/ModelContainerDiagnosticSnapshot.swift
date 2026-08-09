// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

/// The output-head implementation that returned logits for a loaded model.
public enum ModelOutputHeadPath: String, Codable, Sendable, Equatable {
    /// Fused quantized matmul over the source quantized head.
    case qmm
    /// Exact FP32 materialization and matmul.
    case exact
    /// Exact FP32 matmul using a prepared, model-local cached weight.
    case exactCached
}

/// A small, source-backed diagnostic snapshot exposed by ``ModelContainer``.
///
/// Model implementations capture their load-time configuration and derived
/// state. Reading this value does not allocate arrays, create request caches,
/// write logs, or inspect stderr.
public struct ModelContainerDiagnosticSnapshot: Codable, Sendable, Equatable {
    public let modelType: String
    public let cacheRequested: Bool
    public let shadowRequested: Bool
    public let sourceQuantized: Bool
    public let sourceSupported: Bool
    public let prepared: Bool
    public let cacheIdentity: UInt64?
    public let logicalBytes: Int
    public let configuredLMHeadMode: String
    public let effectivePath: ModelOutputHeadPath

    public init(
        modelType: String,
        cacheRequested: Bool,
        shadowRequested: Bool,
        sourceQuantized: Bool,
        sourceSupported: Bool,
        prepared: Bool,
        cacheIdentity: UInt64?,
        logicalBytes: Int,
        configuredLMHeadMode: String
    ) {
        self.modelType = modelType
        self.cacheRequested = cacheRequested
        self.shadowRequested = shadowRequested
        self.sourceQuantized = sourceQuantized
        self.sourceSupported = sourceSupported
        self.prepared = prepared
        self.cacheIdentity = cacheIdentity
        self.logicalBytes = logicalBytes
        self.configuredLMHeadMode = configuredLMHeadMode.lowercased()
        self.effectivePath = Self.resolvePath(
            configuredLMHeadMode: self.configuredLMHeadMode,
            shadowRequested: shadowRequested,
            sourceQuantized: sourceQuantized,
            prepared: prepared)
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case cacheRequested = "cache_requested"
        case shadowRequested = "shadow_requested"
        case sourceQuantized = "source_quantized"
        case sourceSupported = "source_supported"
        case prepared
        case cacheIdentity = "cache_identity"
        case logicalBytes = "logical_bytes"
        case configuredLMHeadMode = "configured_lm_head_mode"
        case effectivePath = "effective_path"
    }

    private static func resolvePath(
        configuredLMHeadMode: String,
        shadowRequested: Bool,
        sourceQuantized: Bool,
        prepared: Bool
    ) -> ModelOutputHeadPath {
        if prepared {
            // Shadow mode computes and returns the exact baseline after it
            // compares the cached result. The cache remains observable through
            // `prepared`, `cacheIdentity`, and `logicalBytes`.
            return shadowRequested ? .exact : .exactCached
        }
        if sourceQuantized && configuredLMHeadMode != "exact" {
            return .qmm
        }
        return .exact
    }
}

/// A model-owned, read-only diagnostic provider used by ``ModelContainer``.
public protocol ModelContainerDiagnosticSnapshotProvider: AnyObject {
    func modelContainerDiagnosticSnapshot() -> ModelContainerDiagnosticSnapshot
}
