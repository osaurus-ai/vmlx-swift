import Cmlx
import Foundation
import MLX
import CmlxDistributedShim

// mlx-c default-stream handle. Private to this file. Cached lazily so
// every collective dispatches against the same stream.
@_silgen_name("mlx_default_cpu_stream_new")
private func _mlx_default_cpu_stream_new() -> UnsafeMutableRawPointer?

/// Construct the default stream lazily on each collective. The
/// underlying mlx-c stream handles are heap-allocated descriptors —
/// constructing one per call is cheap (~µs) and sidesteps Swift 6
/// strict-concurrency global-mutable-state diagnostics. If profiling
/// shows this is hot we can switch to a `@unchecked Sendable` wrapper
/// or per-actor caching.
@inline(__always) private func defaultStream() -> UnsafeMutableRawPointer? {
    _mlx_default_cpu_stream_new()
}

/// Tensor-level collectives. Each is a no-op on a size-1 group (matches
/// Python's `mx.distributed.all_sum` etc. semantics) so call sites don't
/// need to gate on `group.isMultiRank`.
public enum Collectives {

    /// All-reduce sum: every rank ends up with the elementwise sum of
    /// `x` across the group.
    public static func allSum(_ x: MLXArray, group: Group) -> MLXArray {
        if !group.isMultiRank { return x }
        return invoke(x, group: group) { res, src, grp, str in
            vmlx_all_sum(res, src, grp, str)
        }
    }

    /// All-gather: returns concatenation of every rank's `x` along the
    /// 0th axis (Python ref: `mx.distributed.all_gather`).
    public static func allGather(_ x: MLXArray, group: Group) -> MLXArray {
        if !group.isMultiRank { return x }
        return invoke(x, group: group) { res, src, grp, str in
            vmlx_all_gather(res, src, grp, str)
        }
    }

    /// Sum-scatter: every rank gets a slice of the elementwise sum
    /// (Python ref: `mx.distributed.sum_scatter`).
    public static func sumScatter(_ x: MLXArray, group: Group) -> MLXArray {
        if !group.isMultiRank { return x }
        return invoke(x, group: group) { res, src, grp, str in
            vmlx_sum_scatter(res, src, grp, str)
        }
    }

    /// Send `x` to `dst`; returns the input array (used for graph
    /// dependency in MLX's lazy-eval model).
    public static func send(_ x: MLXArray, to dst: Int, group: Group) -> MLXArray {
        if !group.isMultiRank { return x }
        return invokeWithInt(x, intArg: dst, group: group) { res, src, n, grp, str in
            vmlx_send(res, src, Int32(n), grp, str)
        }
    }

    /// Receive an array shaped like `like` from `src` rank.
    public static func recvLike(_ like: MLXArray, from src: Int, group: Group) -> MLXArray {
        if !group.isMultiRank { return like }
        return invokeWithInt(like, intArg: src, group: group) { res, lik, n, grp, str in
            vmlx_recv_like(res, lik, Int32(n), grp, str)
        }
    }

    // MARK: - Internal bridging

    /// Calls a 4-arg collective (res, x, group, stream).
    ///
    /// The mlx-c distributed entry points reject a null `mlx_stream`
    /// (see `mlx-c/mlx/c/distributed.cpp:82`). We pass MLX's default
    /// CPU stream — these collectives are CPU-side coordination ops
    /// (the heavy compute happens via the buffer-shared metal arrays
    /// that are pre-evaluated before the collective fires).
    private static func invoke(
        _ x: MLXArray,
        group: Group,
        _ call: (
            UnsafeMutablePointer<UnsafeMutableRawPointer?>,
            UnsafeMutableRawPointer?,
            UnsafeMutableRawPointer?,
            UnsafeMutableRawPointer?
        ) -> Int32
    ) -> MLXArray {
        var resPtr: UnsafeMutableRawPointer? = nil
        let srcPtr = x.ctx.ctx
        let grpPtr = group.handle.raw.ctx
        let stmPtr = defaultStream()
        let rc = call(&resPtr, srcPtr, grpPtr, stmPtr)
        precondition(rc == 0, "mlx-c collective failed with rc=\(rc)")
        return mlxArrayFromCtx(resPtr)
    }

    /// Calls a 5-arg collective with an int between x and group.
    private static func invokeWithInt(
        _ x: MLXArray,
        intArg: Int,
        group: Group,
        _ call: (
            UnsafeMutablePointer<UnsafeMutableRawPointer?>,
            UnsafeMutableRawPointer?,
            Int,
            UnsafeMutableRawPointer?,
            UnsafeMutableRawPointer?
        ) -> Int32
    ) -> MLXArray {
        var resPtr: UnsafeMutableRawPointer? = nil
        let srcPtr = x.ctx.ctx
        let grpPtr = group.handle.raw.ctx
        let stmPtr = defaultStream()
        let rc = call(&resPtr, srcPtr, intArg, grpPtr, stmPtr)
        precondition(rc == 0, "mlx-c send/recv failed with rc=\(rc)")
        return mlxArrayFromCtx(resPtr)
    }
}

/// Wraps the `mlx_array` handle a collective returned. `MLXArray` is a
/// class that owns an `mlx_array`, so the handle must go through its
/// designated initializer. Reinterpreting a one-pointer struct as the class
/// (the previous `unsafeBitCast`) read the C handle as a Swift object and
/// crashed on first use; it only ran once a real multi-rank group existed.
internal func mlxArrayFromCtx(_ ptr: UnsafeMutableRawPointer?) -> MLXArray {
    MLXArray(mlx_array(ctx: ptr))
}
