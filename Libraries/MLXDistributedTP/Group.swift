import Foundation
import MLX  // pulls Cmlx symbols into the link

/// Swift wrapper around an MLX `mlx_distributed_group` handle. Created
/// once per process via `Group.init(strict:backend:)`; subsequent
/// `split` returns child groups that share the same lifecycle.
///
/// Phase 5 scope:
/// - rank, size, split queryable.
/// - Lifecycle: leaks on shutdown (mlx-c does NOT expose a public
///   `_free` for groups; the C++ Group is shared_ptr-managed but the
///   wrapper allocates with `new` and the only deleter is private).
///   Single Group per process is the expected pattern, so this
///   shouldn't matter in practice.
public struct Group: Sendable {
    /// Opaque mlx_distributed_group handle (just a void* ctx wrapper).
    public let handle: MLXDistributedGroupHandle
    private let rankOverride: Int?
    private let sizeOverride: Int?

    /// Initialise the global distributed group for this process.
    /// - Parameters:
    ///   - strict: when true, throws if the requested backend can't init
    ///     (env vars missing, etc). When false, returns a trivial size-1
    ///     group on failure — same semantics as Python's
    ///     `mx.distributed.init()`.
    ///   - backend: optional backend hint ("jaccl", "ring", "mpi",
    ///     "nccl"). nil lets MLX pick.
    public init(strict: Bool = false, backend: String? = nil) {
        if let backend {
            self.handle = backend.withCString { bk in
                MLXDistributedGroupHandle(_mlx_distributed_init(strict, bk))
            }
        } else {
            self.handle = MLXDistributedGroupHandle(_mlx_distributed_init(strict, nil))
        }
        self.rankOverride = nil
        self.sizeOverride = nil
    }

    /// Number of ranks in this group.
    public var size: Int {
        if let sizeOverride { return sizeOverride }
        return Int(_mlx_distributed_group_size(handle.raw))
    }

    /// Local rank within this group.
    public var rank: Int {
        if let rankOverride { return rankOverride }
        return Int(_mlx_distributed_group_rank(handle.raw))
    }

    /// Single-process test group for sharding transforms and shape checks.
    /// Do not use this for collective correctness: there are no peer ranks.
    public static func singleProcessTest(rank: Int, size: Int) -> Group {
        precondition(size > 0, "test group size must be positive")
        precondition(rank >= 0 && rank < size, "test group rank must be in [0, size)")
        return Group(
            handle: MLXDistributedGroupHandle(_MLXDistributedGroupRaw()),
            rankOverride: rank,
            sizeOverride: size)
    }

    /// Returns true if this group has more than one rank — i.e. real
    /// multi-host work is happening. False means a no-op "group of 1".
    public var isMultiRank: Bool { size > 1 }

    /// Split into a sub-group; ranks with the same `color` end up in
    /// the same returned group, with `key` controlling rank order.
    /// On a size-1 group this is a no-op (mlx-c rejects splits of the
    /// trivial group; we return self rather than expose the empty
    /// handle that would result).
    public func split(color: Int, key: Int) -> Group {
        guard isMultiRank else { return self }
        let raw = _mlx_distributed_group_split(handle.raw, Int32(color), Int32(key))
        return Group(handle: MLXDistributedGroupHandle(raw))
    }

    private init(handle: MLXDistributedGroupHandle) {
        self.handle = handle
        self.rankOverride = nil
        self.sizeOverride = nil
    }

    private init(handle: MLXDistributedGroupHandle, rankOverride: Int, sizeOverride: Int) {
        self.handle = handle
        self.rankOverride = rankOverride
        self.sizeOverride = sizeOverride
    }
}

/// Sendable wrapper around the mlx_distributed_group struct (just a
/// `void* ctx` pointer). The underlying C++ Group is reference-counted
/// internally; passing this handle by value is safe.
public struct MLXDistributedGroupHandle: @unchecked Sendable {
    let raw: _MLXDistributedGroupRaw

    init(_ raw: _MLXDistributedGroupRaw) { self.raw = raw }
}

/// Mirror of the C-level `mlx_distributed_group` struct (a struct with
/// a single `void*` ctx field). Layout-compatible so we can pass by
/// value across the @_silgen_name boundary.
public struct _MLXDistributedGroupRaw {
    public var ctx: UnsafeMutableRawPointer?
    public init(ctx: UnsafeMutableRawPointer? = nil) { self.ctx = ctx }
}

// MARK: - C symbol forward declarations
//
// mlx-swift doesn't export Cmlx as a public product so we can't
// `import Cmlx`. These declarations bind to symbols pulled in via
// our MLX dependency.

@_silgen_name("mlx_distributed_init")
private func _mlx_distributed_init(
    _ strict: Bool, _ backend: UnsafePointer<CChar>?
) -> _MLXDistributedGroupRaw

@_silgen_name("mlx_distributed_group_rank")
private func _mlx_distributed_group_rank(_ g: _MLXDistributedGroupRaw) -> Int32

@_silgen_name("mlx_distributed_group_size")
private func _mlx_distributed_group_size(_ g: _MLXDistributedGroupRaw) -> Int32

@_silgen_name("mlx_distributed_group_split")
private func _mlx_distributed_group_split(
    _ g: _MLXDistributedGroupRaw, _ color: Int32, _ key: Int32
) -> _MLXDistributedGroupRaw
