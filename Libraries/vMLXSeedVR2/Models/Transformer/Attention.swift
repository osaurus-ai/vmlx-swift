// SPDX-License-Identifier: MIT (port). SeedVR2 model/weights: Apache-2.0 / ByteDance.
// Port of mflux seedvr2_transformer/attention.py (MMAttention), B==1 upscale path.
//
// Windowed multi-modal attention: partition vid tokens into variable-size windows,
// replicate the full text token block into every window, RoPE both, run SDPA per
// window over [vid_window ++ text], then coalesce (vid scattered back; text averaged
// across the windows it was replicated into).
import Foundation
import MLX
import MLXFast
import MLXNN

public final class MMAttention: Module {
    @ModuleInfo(key: "proj_qkv_vid") var projQkvVid: Linear
    @ModuleInfo(key: "proj_out_vid") var projOutVid: Linear
    @ModuleInfo(key: "norm_q_vid") var normQVid: SeedVR2RMSNorm
    @ModuleInfo(key: "norm_k_vid") var normKVid: SeedVR2RMSNorm
    @ModuleInfo(key: "proj_qkv_txt") var projQkvTxt: Linear
    @ModuleInfo(key: "proj_out_txt") var projOutTxt: Linear
    @ModuleInfo(key: "norm_q_txt") var normQTxt: SeedVR2RMSNorm
    @ModuleInfo(key: "norm_k_txt") var normKTxt: SeedVR2RMSNorm
    @ModuleInfo(key: "rope") var rope: SeedVR2RoPE

    let heads: Int, headDim: Int, scale: Float, window: [Int], ropeOnText: Bool, shift: Bool

    public init(
        vidDim: Int, txtDim: Int, heads: Int = 20, headDim: Int = 128,
        ropeDim: Int = 128, ropeOnText: Bool = true, window: [Int] = [4, 3, 3], shift: Bool = false
    ) {
        self.heads = heads
        self.headDim = headDim
        self.scale = powf(Float(headDim), -0.5)
        self.window = window
        self.ropeOnText = ropeOnText
        self.shift = shift
        let inner = heads * headDim
        self._projQkvVid.wrappedValue = Linear(vidDim, 3 * inner, bias: false)
        self._projOutVid.wrappedValue = Linear(inner, vidDim, bias: true)
        self._normQVid.wrappedValue = SeedVR2RMSNorm(headDim)
        self._normKVid.wrappedValue = SeedVR2RMSNorm(headDim)
        self._projQkvTxt.wrappedValue = Linear(txtDim, 3 * inner, bias: false)
        self._projOutTxt.wrappedValue = Linear(inner, txtDim, bias: true)
        self._normQTxt.wrappedValue = SeedVR2RMSNorm(headDim)
        self._normKTxt.wrappedValue = SeedVR2RMSNorm(headDim)
        self._rope.wrappedValue = SeedVR2RoPE(dim: ropeDim)
        super.init()
    }

    /// vid [1,L,vidDim], txt [1,Lt,txtDim]. vidShape [[t,h,w]] (B==1). txtLen scalar.
    public func callAsFunction(_ vid: MLXArray, _ txt: MLXArray, vidShape: [[Int]], txtLen: Int)
        -> (MLXArray, MLXArray)
    {
        let (B, L) = (vid.shape[0], vid.shape[1])
        let (Bt, Lt) = (txt.shape[0], txt.shape[1])
        precondition(B == 1 && Bt == 1, "upscale path is single-image")
        let inner = heads * headDim

        // 1. project to qkv: [N, 3, heads, headDim]
        var qkvVid = projQkvVid(vid.reshaped([-1, vid.shape[2]])).reshaped([-1, 3, heads, headDim])
        let qkvTxt = projQkvTxt(txt.reshaped([-1, txt.shape[2]])).reshaped([-1, 3, heads, headDim])

        let part = WindowPartitioner(vidShape: vidShape, window: window, shift: shift)
        qkvVid = part.partition(qkvVid)

        // 2. normalize q,k; replicate text into every window
        let qVid = normQVid(qkvVid[0..., 0])
        let kVid = normKVid(qkvVid[0..., 1])
        let vVid = qkvVid[0..., 2]
        var qTxt = normQTxt(qkvTxt[0..., 0])
        var kTxt = normKTxt(qkvTxt[0..., 1])
        var vTxt = qkvTxt[0..., 2]
        let nWin = part.windowShapes.count
        // tile each text tensor [Lt,heads,hd] -> [nWin*Lt, heads, hd] (block per window)
        qTxt = tiled(qTxt, repetitions: [nWin, 1, 1])
        kTxt = tiled(kTxt, repetitions: [nWin, 1, 1])
        vTxt = tiled(vTxt, repetitions: [nWin, 1, 1])

        // 3. RoPE
        var qV = qVid
        var kV = kVid
        var qT = qTxt
        var kT = kTxt
        if ropeOnText {
            let txtLens = Array(repeating: txtLen, count: nWin)
            (qV, kV, qT, kT) = rope.applyMM(
                qVid, kVid, windowShapes: part.windowShapes, qTxt, kTxt, txtLens: txtLens)
        } else {
            (qV, kV) = rope.applyVid(qVid, kVid, windowShapes: part.windowShapes)
        }

        // 4. per-window SDPA over [vid ++ text]
        let vidLens = part.windowShapes.map { $0[0] * $0[1] * $0[2] }
        var vidOutBlocks: [MLXArray] = []
        var txtOutBlocks: [MLXArray] = []
        var vOff = 0
        for i in 0 ..< nWin {
            let vl = vidLens[i]
            let tOff = i * Lt
            func win(_ vidPart: MLXArray, _ txtPart: MLXArray) -> MLXArray {
                concatenated(
                    [vidPart[vOff ..< (vOff + vl)], txtPart[tOff ..< (tOff + Lt)]], axis: 0)
            }
            // [N, heads, hd] -> [1, heads, N, hd]
            let q = win(qV, qT).expandedDimensions(axis: 0).transposed(0, 2, 1, 3)
            let k = win(kV, kT).expandedDimensions(axis: 0).transposed(0, 2, 1, 3)
            let v = win(vVid, vTxt).expandedDimensions(axis: 0).transposed(0, 2, 1, 3)
            var o = MLXFast.scaledDotProductAttention(
                queries: q, keys: k, values: v, scale: scale, mask: .none)
            o = o.transposed(0, 2, 1, 3).squeezed(axis: 0).reshaped([-1, inner])  // [N, inner]
            vidOutBlocks.append(o[0 ..< vl])
            txtOutBlocks.append(o[vl ..< (vl + Lt)])
            vOff += vl
        }

        // 5. coalesce: vid scattered back to original order; text averaged across windows
        let vidOut = part.reverse(concatenated(vidOutBlocks, axis: 0))  // [L, inner]
        let txtOut = mean(stacked(txtOutBlocks, axis: 0), axis: 0)  // [Lt, inner]
        return (
            projOutVid(vidOut).reshaped([B, L, -1]),
            projOutTxt(txtOut).reshaped([Bt, Lt, -1])
        )
    }
}
