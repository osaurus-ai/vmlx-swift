// Copyright 2025 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Stage 2 (real) — `CompilableTurboQuantKVCache`.
//
// The Stage 2 rollback (iteration 8) surfaced that `TurboQuantKVCache`'s
// compressed-phase append path is NOT compile-traceable as-is:
// `windowOffset` is a Swift Int, so `compile()` captures it at
// trace-build time and every subsequent compiled call writes at the same
// position. Only the first decode token is correct.
//
// This subclass fixes the path by:
//   - Tracking `writePosArray` as `MLXArray[1] int32` instead of Int.
//   - Writing new tokens via `dynamicSliceUpdate` + `_updateInternal` at
//     the MLXArray write position, so the compile tracer follows the
//     updates through the computation graph.
//   - Returning the FULL unified buffer (not a `..<totalTokens` dynamic
//     slice) and emitting an `.array` attention mask via `makeMask` that
//     restricts attention to valid positions. This mirrors the Overflow
//     Bin pattern used by `CompilableKVCache`, which passed a 50-step
//     multi-step compile-correctness probe at 5e-7 relative FP diff.
//
// Initialise via ``init(from:)`` on an already-compressed
// `TurboQuantKVCache`. Fill-phase append delegates to the parent class
// (fill phase runs during prefill only, which is uncompiled anyway).

import Foundation
import MLX
import MLXNN

/// Compile-traceable specialisation of ``TurboQuantKVCache`` for the
/// batched compile path (Stage 2 proper).
///
/// Compared to the parent:
///
/// - `writePosArray: MLXArray` replaces Swift-Int `windowOffset`. Advance
///   happens via `_updateInternal` so the compile tracer sees the growth.
/// - `offsetArray: MLXArray` mirrors `offset` (which stays as an Int for
///   compatibility with code that reads `.offset` outside the trace, e.g.
///   cache-coordinator code). Inside the trace, `offsetArray` drives
///   ``makeMask(n:windowSize:returnArray:)``.
/// - `update(keys:values:)` routes `.compressed` to an MLXArray-typed
///   write path. `.fill` delegates to `super.update` because fill-phase
///   allocation is not itself compile-traced — the compile promotion in
///   `BatchEngine.maybePromoteToCompiledDecode` only runs after prefill
///   has already produced a compressed cache.
/// - `makeMask` always returns an `.array` mask over the full unified
///   buffer, matching the shape the return value advertises. Without
///   this the attention kernel would attend to the uninitialised tail of
///   the window, producing garbage.
///
/// ## Memory
///
/// No new allocations beyond two small `MLXArray[1]` counters and a
/// pre-computed mask index array. The large unified buffer is inherited
/// from the parent — the subclass works in-place on the same storage.
public final class CompilableTurboQuantKVCache: TurboQuantKVCache, @unchecked Sendable {

    /// Position of the next window write, as `MLXArray[1] int32`. Starts
    /// at 0 after promotion; advances by `newTokens` per `update` call.
    public var writePosArray: MLXArray

    /// Total valid tokens = `prefixTokenCount + writePosArray`, as
    /// `MLXArray[1] int32`. Used by `makeMask` as the causal upper bound.
    ///
    /// Iter-10 measured: removing `writePosArray` and computing the start
    /// via `offsetArray` alone INCREASED multi-step drift from ~7% to
    /// ~13%. Keeping both MLXArray counters in the traced state surface
    /// — even though they advance in lockstep — appears to help the
    /// compile tracer stay coherent across invocations.
    public var offsetArray: MLXArray

    /// Pre-computed column indices `[0, 1, ..., bufferLen-1]` used by
    /// `makeMask` to build a causal mask over the full unified buffer.
    /// The buffer length is fixed at promotion time; lazy so we avoid
    /// work when only `update()` is called.
    private lazy var maskRinds: MLXArray = {
        let bufferLen = unifiedKeys?.dim(2) ?? 0
        return MLXArray(Int32(0) ..< Int32(bufferLen))
    }()

    // MARK: - Init

    /// Public convenience initialiser that builds a
    /// `CompilableTurboQuantKVCache` directly in the compressed phase
    /// from `keyBits` / `valueBits` / `sinkTokens`. Callers normally use
    /// ``init(from:)`` instead — this direct init exists primarily for
    /// testing.
    public override init(
        keyBits: Int = 3,
        valueBits: Int = 3,
        sinkTokens: Int = 4,
        residualTokens: Int = TurboQuantKVCache.defaultResidualTokens
    ) {
        self.writePosArray = MLXArray([Int32(0)])
        self.offsetArray = MLXArray([Int32(0)])
        super.init(
            keyBits: keyBits,
            valueBits: valueBits,
            sinkTokens: sinkTokens,
            residualTokens: residualTokens)
    }

    /// Promote an existing ``TurboQuantKVCache`` (in `.compressed` phase)
    /// to a compile-traceable variant. Shares the parent's compressed /
    /// decoded / unified buffers — no data copy, just a new type tag and
    /// two `MLXArray[1]` counter properties.
    ///
    /// - Parameter tq: Source cache. Precondition: `tq.phase == .compressed`.
    public convenience init(from tq: TurboQuantKVCache) {
        precondition(tq.phase == .compressed,
            "CompilableTurboQuantKVCache(from:) requires source to be in .compressed phase")

        self.init(
            keyBits: tq.keyBits,
            valueBits: tq.valueBits,
            sinkTokens: tq.sinkTokens,
            residualTokens: tq.residualTokens)

        // Copy state references from tq. All fields are `internal` now so
        // same-module subclass access works.
        self.floatKeys = tq.floatKeys
        self.floatValues = tq.floatValues
        self.decodedKeyBuffer = tq.decodedKeyBuffer
        self.decodedValueBuffer = tq.decodedValueBuffer
        self.unifiedKeys = tq.unifiedKeys
        self.unifiedValues = tq.unifiedValues
        self.prefixTokenCount = tq.prefixTokenCount
        self.windowOffset = tq.windowOffset
        self.encoderState = tq.encoderState
        self.valueEncoderState = tq.valueEncoderState

        // Promote phase via restoreCompressed's side effects would
        // re-allocate unifiedKeys. Instead, flip phase directly via the
        // parent's protocol-level state mechanism.
        //
        // `phase` is `public private(set)` on the parent so we can't
        // assign from here. However parent's `update()` dispatches based
        // on phase, and `.fill` is the default. Since we copied the
        // compressed state manually above, we need phase == .compressed.
        //
        // Workaround: call `restoreCompressed` which sets phase =
        // .compressed and re-builds unifiedKeys at the same shape. This
        // is slightly wasteful (re-decodes the compressed prefix) but
        // happens exactly once per promotion and preserves correctness.
        if let ck = tq.compressedKeys, let cv = tq.compressedValues {
            self.restoreCompressed(
                encodedKeys: ck, encodedValues: cv, sourceOffset: tq.offset)
        }

        // After restoreCompressed, `unifiedKeys` is re-allocated at the
        // same shape as before. windowOffset is reset to 0. Our
        // MLXArray counters track from this reset baseline.
        self.writePosArray = MLXArray([Int32(0)])
        self.offsetArray = MLXArray([Int32(self.prefixTokenCount)])
        // `offset` (super, Int) was set to `tq.offset` by restoreCompressed.
    }

    // MARK: - KVCache protocol overrides

    /// Compile-traceable update for the compressed phase. The fill phase
    /// delegates to super because it's not on the compiled hot path —
    /// compile promotion only runs after the cache is already compressed.
    public override func update(
        keys: MLXArray, values: MLXArray
    ) -> (MLXArray, MLXArray) {
        switch phase {
        case .fill:
            return super.update(keys: keys, values: values)
        case .compressed:
            return compiledAppendDecode(keys: keys, values: values)
        }
    }

    /// Compressed-phase append rewritten to be compile-traceable.
    ///
    /// Key invariants:
    ///  - Buffer shape stays constant (no realloc inside the trace).
    ///  - Writes use `dynamicSliceUpdate` at an `MLXArray` position.
    ///  - Counters advance via `_updateInternal` so the tracer follows
    ///    them through the graph.
    ///  - Return value is the FULL unified buffer — `makeMask` restricts
    ///    attention to valid positions.
    private func compiledAppendDecode(
        keys: MLXArray, values: MLXArray
    ) -> (MLXArray, MLXArray) {
        let newTokens = keys.dim(2)

        let prev = offsetArray
        let advance = MLXArray([Int32(newTokens)])
        let newOffset = prev + advance

        unifiedKeys!._updateInternal(
            dynamicSliceUpdate(
                unifiedKeys!, update: keys,
                start: prev, axes: [2]))
        unifiedValues!._updateInternal(
            dynamicSliceUpdate(
                unifiedValues!, update: values,
                start: prev, axes: [2]))

        writePosArray._updateInternal(writePosArray + advance)
        offsetArray._updateInternal(newOffset)

        // Swift-Int mirrors for consumers outside the trace (e.g. cache
        // coordinator post-gen store). These are ONLY read outside the
        // trace — never inside, so capturing at trace-build is fine.
        offset += newTokens
        windowOffset += newTokens

        // Full buffer — mask handles validity.
        return (unifiedKeys!, unifiedValues!)
    }

    /// Build a causal attention mask over the full unified buffer.
    ///
    /// Always returns `.array(mask)` — the full-buffer return from
    /// `compiledAppendDecode` means uninitialised positions must be
    /// masked out or attention will read garbage. The mask is built
    /// entirely from `MLXArray` ops so the compile tracer follows the
    /// per-step changes to `offsetArray`.
    public override func makeMask(
        n: Int, windowSize: Int?, returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        // Query positions — shape [n, 1] after reshape. For n=1 decode
        // step, the query position is the current offset.
        //
        // Critical timing: Llama and peers call
        // `createAttentionMask(h:cache:)` BEFORE any layer invokes
        // `cache.update(...)`. So when `makeMask` runs, `offsetArray`
        // reflects the PRE-update state (position about to be written).
        // The `.>=` semantic below means "attend to positions [0, linds]"
        // — which after the layer's update writes at position `linds`
        // matches the post-update valid region. This mirrors
        // `CompilableKVCache.makeMask` exactly; a `.>` variant was tried
        // in iter 9 and measurably worsened per-step divergence because
        // it excluded the newly-written token from attention.
        let linds: MLXArray
        if n == 1 {
            linds = offsetArray.reshaped(1, 1)
        } else {
            linds = (MLXArray(Int32(0) ..< Int32(n)) + offsetArray).reshaped(n, 1)
        }

        let bufferLen = unifiedKeys?.dim(2) ?? 0
        let rinds = maskRinds.reshaped(1, bufferLen)

        var mask = linds .>= rinds

        if let windowSize {
            let windowStart = linds - Int32(windowSize - 1)
            mask = mask & (rinds .>= windowStart)
        }

        // Return 2D `[n, bufferLen]` mask — exactly matching
        // `CompilableKVCache.makeMask`, which passes Stage 1B.2's
        // multi-step correctness test at 5e-7. Adding explicit [1, 1]
        // broadcast dims (as tried in iter 10) introduced drift — MLX's
        // attention kernel broadcasts [n, bufferLen] correctly to
        // [B, H, n, bufferLen] and diverges slightly from the 4D form.
        return .array(mask)
    }

    // MARK: - innerState override

    /// Expose MLXArray counters so `compile()`'s inputs/outputs can find
    /// and track them. The parent's innerState is extended to include
    /// our two counters — if those aren't in the trace's state set, the
    /// tracer treats them as uncaptured and refuses to compile.
    /// Return ONLY state that mutates during decode: the unified
    /// buffers and the two MLXArray counters. The parent's `innerState`
    /// also returns compressed-key tuples (`indicesPacked`, `qjlPacked`,
    /// `residualNorms`, `vectorNorms`, sink data) — these are immutable
    /// after promotion and were included in an earlier draft (iter 9).
    /// Under compile they appear as state inputs+outputs even though the
    /// trace never writes to them. That mismatch between "declared
    /// mutable" and "actually immutable" measurably correlates with
    /// multi-step drift in the iter-10 probes.
    ///
    /// Skipping them mirrors what `CompilableKVCache.innerState` does
    /// for its own implementation: only the things that MUTATE per call.
    public override func innerState() -> [MLXArray] {
        var state = [MLXArray]()
        if let uk = unifiedKeys { state.append(uk) }
        if let uv = unifiedValues { state.append(uv) }
        state.append(writePosArray)
        state.append(offsetArray)
        return state
    }
}
