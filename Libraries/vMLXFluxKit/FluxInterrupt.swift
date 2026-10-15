import Foundation

/// Cooperative producer cancellation. The consumer must drain the stream before
/// releasing its GPU lease; cancelling a Swift task alone does not stop GPU work.
public final class FluxInterrupt: @unchecked Sendable {
    private let lock = NSLock()
    private var interrupted = false

    public init() {}

    public func cancel() { lock.withLock { interrupted = true } }
    public var isCancelled: Bool { lock.withLock { interrupted } }
    public func check() throws { if isCancelled { throw CancellationError() } }
}

/// Inherited by native pipeline tasks; the engine drains the producer before
/// releasing its operation lease, even if the consumer abandons its stream.
public enum FluxInterruptScope {
    @TaskLocal public static var current: FluxInterrupt?
    public static func check() throws { try current?.check() }
}
