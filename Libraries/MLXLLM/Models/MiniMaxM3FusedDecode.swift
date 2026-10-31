// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXLMCommon

/// Integration seam for MiniMax-M3 / M3.1 fused decode kernels ("megakernels"), 1...16 rows. The default path is
/// `MiniMaxM3FusedKernelPath` (vMLX's K1-K5 attention kernels, `MiniMaxM3FusedAttention`; the MoE half declines until
/// K6-K8 are ported); tests may register another.
///
/// The fused path replaces whole halves of a decoder layer for small calls: the attention block (input RMSNorm,
/// q / k / v and index projections, head norms + partial RoPE + casts, K / V / index-key appends, indexer selection,
/// MSA attention, o_proj, + residual) and the MLP block (post-attention RMSNorm, router, routed JANGT experts +
/// shared expert, combine, + residual). The kernels come from vMLX Python (`vmlx_engine/models/minimax_m3/kernels/`
/// standalone .metal sources: K1-K5 attention on `perf/minimax-m3-fused-attn`, K6-K8 MoE on
/// `perf/minimax-m3-verify-multirow`) and run through `MLXFast.metalKernel` with the identical source.
///
/// Contract a path must keep:
/// - **Row invariance.** A row's output depends only on that row and the cache before it, never on how many rows share
///   the call. The same kernels then serve a one-token decode step and a DSpark verify of W rows, so verify == decode
///   bit for bit and greedy DSpark == plain. Both the decode forward and `verifyForward` consult this seam with the
///   same admission rule; a path that declines (returns nil) must decline for that layer at every row count, or the
///   two would run different arithmetic.
/// - **Cache effects.** The attention block appends this call's K / V and index-key rows exactly as the stock path does
///   (the caches stay in lockstep; DSpark rollback trims them).
/// - **Decline is free.** Returning nil runs the stock path for that block unchanged.
///
/// Off by default: `VMLX_M3_FUSED_DECODE=1` turns it on. Only single-sequence calls are offered: decode-forward calls
/// of 1...16 rows (prefill chunks keep the chunk kernels) and DSpark verify forwards of any width (a path runs rows
/// past 16 in chunks, so a wide verify stays on the decode step's arithmetic).
public enum MiniMaxM3FusedDecode {
    public static let environmentKey = "VMLX_M3_FUSED_DECODE"
    /// Rows a fused call may carry: a decode step (1) up to DSpark's verify of the ladder top (2 x block 7 + 1 = 15).
    public static let maxRows = 16

    static let enabledByEnvironment: Bool = {
        let value = (ProcessInfo.processInfo.environment[environmentKey] ?? "0")
            .trimmingCharacters(in: .whitespaces).lowercased()
        return ["1", "true", "on", "yes"].contains(value)
    }()

    private static let lock = NSLock()
    nonisolated(unsafe) private static var _path: (any MiniMaxM3FusedPath)?
    nonisolated(unsafe) private static var _forceEnabled: Bool?

    /// The registered fused path (nil = the default `MiniMaxM3FusedKernelPath`).
    static var path: (any MiniMaxM3FusedPath)? {
        get { lock.withLock { _path } }
        set { lock.withLock { _path = newValue } }
    }

    /// Tests / harness only: overrides the environment switch (nil = environment).
    static var forceEnabled: Bool? {
        get { lock.withLock { _forceEnabled } }
        set { lock.withLock { _forceEnabled = newValue } }
    }

    /// The path to use for a call of `rows` rows in a batch of `batch` (`verify`: a DSpark verify forward, any width),
    /// or nil (stock path).
    static func active(rows: Int, batch: Int, verify: Bool = false) -> (any MiniMaxM3FusedPath)? {
        guard batch == 1, rows >= 1, verify || rows <= maxRows else { return nil }
        let (path, force) = lock.withLock { (_path, _forceEnabled) }
        guard force ?? enabledByEnvironment else { return nil }
        return path ?? MiniMaxM3FusedKernelPath.shared
    }
}

/// A fused implementation of the two halves of a MiniMax-M3 decoder layer for 1...16 rows (see `MiniMaxM3FusedDecode`).
protocol MiniMaxM3FusedPath: AnyObject, Sendable {
    /// `x + attention(input_layernorm(x))` for `x` `[1, L, hidden]` at the cache's offset, appending this call's K / V /
    /// index-key rows to `cache`; `verify` is true inside a DSpark verify forward (every row must equal the row a
    /// one-token step at its position computes). nil declines (the stock path runs).
    func attentionBlock(
        layer: Int, module: MiniMaxM3DecoderLayer, x: MLXArray, cache: KVCache?, verify: Bool
    ) -> MLXArray?

    /// `h + mlp(post_attention_layernorm(h))` (dense MLP on the first layers, router + routed + shared elsewhere). nil
    /// declines.
    func mlpBlock(layer: Int, module: MiniMaxM3DecoderLayer, h: MLXArray, verify: Bool) -> MLXArray?
}
