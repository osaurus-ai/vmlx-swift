// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon

/// MTP verify-tile M-padding (workplan W2a).
///
/// Native MTP verify forwards run `depth + 1` rows (2–6). MLX only takes the
/// NAX qmm tile path when `M >= get_qmv_batch_limit` — 10–12 for this trunk's
/// K/N on an M5 Max (`Source/Cmlx/mlx/mlx/backend/metal/quantized.cpp:87`,
/// dispatch at `:1406-1425`); below the limit every row runs the qmv vector
/// kernel, which re-streams the full dense weight per row. Zero-padding ONLY
/// the M dimension of row-independent weight matmuls up to the tile turns
/// those S weight passes into one 64-row NAX M-tile pass, so the extra verify
/// rows become ~free where weight streaming dominates.
///
/// Safety: the pad wraps pure matmuls (QKV/out projections, GDN input/output
/// projections, lm_head) and the padded rows are sliced away before ANY cache
/// write, norm-state recording, or attention interaction — KV/GDN caches and
/// emitted logits only ever see the real rows. The real rows' values are
/// mathematically exact (rows of a linear map are independent); the kernel
/// switch (qmv → qmm/NAX) can shift low-order bits versus the unpadded path,
/// which is why the gate defaults OFF and exists for a measured A/B.
///
/// Gate: `VMLX_MTP_VERIFY_TILE=<rows>` (16 recommended; 8 stays BELOW the
/// qmv→qmm floor for the large trunk matmuls on M5-class devices and only
/// helps if the floor is lower on the running device), `off`/`0`/unset
/// disables. Family-scoped by call site: qwen4_exp (QSA q/k/v/o + indexer,
/// GDN, lm_head) and qwen3_5 (attention q/k/v/o, GDN, lm_head) each pass
/// their own `family` tag so the one-shot activation log names the trunk
/// that engaged; the shared GatedDeltaNet participates only when constructed
/// with `verifyTilePadding: true`. Routed MoE never consults this.
enum Qwen4ExpVerifyTile {
    /// Activation-log tags. One line per family per process.
    enum Family {
        static let qwen4Exp = "Qwen4Exp"
        static let qwen35 = "Qwen35"
    }

    /// Rows to pad verify-width matmuls to; `nil` = disabled (the default).
    /// Read once from the environment; mutable ONLY so focused tests can
    /// exercise the padded path without process-level env plumbing.
    nonisolated(unsafe) static var rows: Int? = parse(
        RuntimeEnvironment.value("VMLX_MTP_VERIFY_TILE"))

    static func parse(_ raw: String?) -> Int? {
        guard
            let raw = raw?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased(),
            !raw.isEmpty, raw != "off", raw != "0"
        else { return nil }
        guard let value = Int(raw), (2 ... 64).contains(value) else {
            FileHandle.standardError.write(
                Data(
                    ("[Qwen4Exp] verify_tile: ignoring VMLX_MTP_VERIFY_TILE=\(raw) "
                        + "(expected 2-64, or off)\n").utf8))
            return nil
        }
        return value
    }

    private static let activationLogLock = NSLock()
    private nonisolated(unsafe) static var reportedFamilies: Set<String> = []

    private static func reportActivation(tile: Int, family: String) {
        activationLogLock.lock()
        defer { activationLogLock.unlock() }
        guard reportedFamilies.insert(family).inserted else { return }
        FileHandle.standardError.write(
            Data(
                "[\(family)] verify_tile=\(tile) path=m-pad\n".utf8))
    }

    /// Runs `body` — a row-independent weight matmul over the axis-1 rows of
    /// a `[1, S, *]` activation — with S zero-padded up to `tile`, then
    /// slices the result back to the original S rows. Falls through untouched
    /// when the gate is off, `S == 1` (AR decode: qmv IS the right kernel),
    /// `S >= tile` (prefill / already at the tile), or batch > 1.
    static func padded(
        _ x: MLXArray, tile: Int? = rows, family: String = Family.qwen4Exp,
        _ body: (MLXArray) -> MLXArray
    ) -> MLXArray {
        guard let tile, x.ndim >= 2, x.dim(0) == 1 else { return body(x) }
        let s = x.dim(1)
        guard s > 1, s < tile else { return body(x) }
        var padShape = x.shape
        padShape[1] = tile - s
        let out = body(
            concatenated(
                [x, MLXArray.zeros(padShape, dtype: x.dtype)], axis: 1))
        reportActivation(tile: tile, family: family)
        return out[0..., ..<s]
    }

    /// Two-input form for the GDN output stage, where the projection consumes
    /// the scan output and its gate together. Both are padded on axis 1 and
    /// the result is sliced back to the real rows.
    static func padded(
        _ x: MLXArray, _ gate: MLXArray, tile: Int? = rows,
        family: String = Family.qwen4Exp,
        _ body: (MLXArray, MLXArray) -> MLXArray
    ) -> MLXArray {
        guard let tile, x.ndim >= 2, x.dim(0) == 1, gate.ndim >= 2,
            gate.dim(0) == 1, gate.dim(1) == x.dim(1)
        else { return body(x, gate) }
        let s = x.dim(1)
        guard s > 1, s < tile else { return body(x, gate) }
        var xPad = x.shape
        xPad[1] = tile - s
        var gatePad = gate.shape
        gatePad[1] = tile - s
        let out = body(
            concatenated([x, MLXArray.zeros(xPad, dtype: x.dtype)], axis: 1),
            concatenated([gate, MLXArray.zeros(gatePad, dtype: gate.dtype)], axis: 1))
        reportActivation(tile: tile, family: family)
        return out[0..., ..<s]
    }
}
