import Foundation

/// Original-file repair requires both explicit opt-in and a direct user Send.
/// Background consumers must keep `.disabled`, including parent-model restore.
public enum AlignmentRepairAuthorization: Sendable, Equatable {
    case disabled
    case directUserSend
}

/// Per-shard preparation, not a transaction across the entire model bundle.
public struct AlignmentRepairProgress: Sendable, Equatable {
    public enum Stage: Sendable, Equatable { case copying, verifying, installed, fallback }
    public let bundle: URL
    public let shard: URL
    public let stage: Stage
    public let copiedBytes: UInt64
    public let totalBytes: UInt64

    public init(bundle: URL, shard: URL, stage: Stage, copiedBytes: UInt64, totalBytes: UInt64) {
        self.bundle = bundle
        self.shard = shard
        self.stage = stage
        self.copiedBytes = copiedBytes
        self.totalBytes = totalBytes
    }

    /// Scoped to the load task. Hosts must additionally match their load identity
    /// before displaying an event; an observer never authorizes a rewrite.
    @TaskLocal public static var observer: (@Sendable (AlignmentRepairProgress) -> Void)?
}
