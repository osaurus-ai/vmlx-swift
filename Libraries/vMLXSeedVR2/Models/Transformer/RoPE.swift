// SPDX-License-Identifier: MIT (port). SeedVR2 model/weights: Apache-2.0 / ByteDance.
// Port of mflux seedvr2_transformer/rope.py (freqs_for == "lang" path).
//
// Axial 3D rotary embeddings. `freqs` (e.g. 21 values for rope_dim=128) is stored in
// the checkpoint, so we load it (not recompute). Shapes are passed as Swift Ints
// ([[t,h,w]] per window, [txtLen] per window) — the partitioner produces them.
import Foundation
import MLX
import MLXNN

public final class SeedVR2RoPE: Module {
    @ParameterInfo(key: "freqs") var freqs: MLXArray
    let ropeAxes = 3

    public init(dim: Int = 128) {
        // freqs = 1 / theta^(arange(0,freqDim,2)[:freqDim/2] / freqDim); loaded from checkpoint,
        // but initialise the right length so update() matches.
        let freqDim = dim / ropeAxes
        let n = freqDim / 2
        self._freqs.wrappedValue = MLXArray.zeros([n])
        super.init()
    }

    private var freqDimPerAxis: Int { freqs.shape[0] * 2 }

    /// Axial freqs over a grid `dims`, with an optional offset for the first (temporal) axis.
    /// Returns shape `dims + [freqDimPerAxis * dims.count]`.
    private func axialFreqs(_ dims: [Int], temporalOffset: Int = 0) -> MLXArray {
        let fdpa = freqDimPerAxis
        let f32 = freqs.asType(.float32)
        var all: [MLXArray] = []
        for (ind, d) in dims.enumerated() {
            let start = ind == 0 ? temporalOffset : 0
            let pos = MLXArray(start ..< (start + d)).asType(.float32)
            var af = outer(pos, f32)  // [d, n]
            af = repeated(af, count: 2, axis: -1)  // [d, 2n]
            var shape = Array(repeating: 1, count: dims.count) + [fdpa]
            shape[ind] = d
            af = af.reshaped(shape)
            af = broadcast(af, to: dims + [fdpa])
            all.append(af)
        }
        return concatenated(all, axis: -1)  // [dims..., fdpa*dims.count]
    }

    private static func rotateHalf(_ x: MLXArray) -> MLXArray {
        let s = x.shape
        var r = x.reshaped(Array(s.dropLast()) + [-1, 2])
        let x1 = r[.ellipsis, 0]
        let x2 = r[.ellipsis, 1]
        r = stacked([-x2, x1], axis: -1)
        return r.reshaped(s)
    }

    /// freqs: [..., rotDim]; t: [N, heads, headDim]. Rotate the first rotDim dims.
    private static func applyRotary(_ freqs: MLXArray, _ t: MLXArray) -> MLXArray {
        let rotDim = freqs.shape[freqs.ndim - 1]
        let dim = t.shape[t.ndim - 1]
        let tMid = t[.ellipsis, 0 ..< rotDim].asType(.float32)
        let f = freqs.asType(.float32)
        var out = tMid * cos(f) + rotateHalf(tMid) * sin(f)
        out = out.asType(t.dtype)
        if dim > rotDim {
            let tRight = t[.ellipsis, rotDim ..< dim]
            return concatenated([out, tRight], axis: -1)
        }
        return out
    }

    /// Video-only RoPE (rope_on_text == false). windowShapes: [[t,h,w]] per window.
    public func applyVid(_ vidQ: MLXArray, _ vidK: MLXArray, windowShapes: [[Int]]) -> (
        MLXArray, MLXArray
    ) {
        var parts: [MLXArray] = []
        for s in windowShapes {
            let vf = axialFreqs(s).reshaped([-1, freqDimPerAxis * ropeAxes])
            parts.append(vf)
        }
        let vidFreqs = concatenated(parts, axis: 0).expandedDimensions(axis: 1)  // [Ntok,1,rot]
        return (Self.applyRotary(vidFreqs, vidQ), Self.applyRotary(vidFreqs, vidK))
    }

    /// Multi-modal RoPE (rope_on_text == true): vid temporal positions are offset by txtLen.
    public func applyMM(
        _ vidQ: MLXArray, _ vidK: MLXArray, windowShapes: [[Int]],
        _ txtQ: MLXArray, _ txtK: MLXArray, txtLens: [Int]
    ) -> (MLXArray, MLXArray, MLXArray, MLXArray) {
        var vidParts: [MLXArray] = []
        var txtParts: [MLXArray] = []
        for (i, s) in windowShapes.enumerated() {
            let (f, h, w) = (s[0], s[1], s[2])
            let tl = txtLens[i]
            // full grid over (tl+f, h, w), keep the vid temporal slice [tl, tl+f)
            let full = axialFreqs([tl + f, h, w])
            let vidSlice = full[tl ..< (tl + f)].reshaped([-1, freqDimPerAxis * ropeAxes])
            vidParts.append(vidSlice)
            // text: 1D axial over tl, tiled across the 3 axes
            let txt1d = axialFreqs([tl])  // [tl, fdpa]
            txtParts.append(tiled(txt1d, repetitions: [1, ropeAxes]))  // [tl, fdpa*3]
        }
        let vidFreqs = concatenated(vidParts, axis: 0).expandedDimensions(axis: 1)
        let txtFreqs = concatenated(txtParts, axis: 0).expandedDimensions(axis: 1)
        return (
            Self.applyRotary(vidFreqs, vidQ), Self.applyRotary(vidFreqs, vidK),
            Self.applyRotary(txtFreqs, txtQ), Self.applyRotary(txtFreqs, txtK)
        )
    }
}
