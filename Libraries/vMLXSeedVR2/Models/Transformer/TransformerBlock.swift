// SPDX-License-Identifier: MIT (port). SeedVR2 model/weights: Apache-2.0 / ByteDance.
// Port of mflux seedvr2_transformer/transformer_block.py.
import MLX
import MLXFast
import MLXNN

/// Weightless RMSNorm (ones weight) — mflux `TransformerBlock._rms_norm`. Not a parameter.
func rmsNormOnes(_ x: MLXArray, eps: Float) -> MLXArray {
    MLXFast.rmsNorm(x, weight: MLXArray.ones([x.shape[x.ndim - 1]]).asType(x.dtype), eps: eps)
}

public final class TransformerBlock: Module {
    @ModuleInfo(key: "attn") var attn: MMAttention
    @ModuleInfo(key: "mlp") var mlp: MMSwiGLU
    @ModuleInfo(key: "ada") var ada: AdaModulation
    let isLastLayer: Bool
    let eps: Float

    public init(
        vidDim: Int = 2560, txtDim: Int = 2560, heads: Int = 20, headDim: Int = 128,
        expandRatio: Int = 4, normEps: Float = 1e-5, ropeDim: Int = 128,
        ropeOnText: Bool = true, shared: Bool = false, isLastLayer: Bool = false,
        window: [Int] = [4, 3, 3], shift: Bool = false
    ) {
        self.isLastLayer = isLastLayer
        self.eps = normEps
        self._attn.wrappedValue = MMAttention(
            vidDim: vidDim, txtDim: txtDim, heads: heads,
            headDim: headDim, ropeDim: ropeDim, ropeOnText: ropeOnText, window: window, shift: shift
        )
        self._mlp.wrappedValue = MMSwiGLU(
            vidDim: vidDim, txtDim: txtDim, expandRatio: expandRatio,
            shared: shared, isLastLayer: isLastLayer)
        self._ada.wrappedValue = AdaModulation(
            dim: vidDim, shared: shared, isLastLayer: isLastLayer)
        super.init()
    }

    public func callAsFunction(
        _ vidIn: MLXArray, _ txtIn: MLXArray, emb: MLXArray,
        vidShape: [[Int]], txtLen: Int
    ) -> (MLXArray, MLXArray) {
        var vid = vidIn
        var txt = txtIn

        var vidAttn = ada.modulateVid(rmsNormOnes(vid, eps: eps), emb, .attn, .modIn)
        var txtAttn = ada.modulateTxt(rmsNormOnes(txt, eps: eps), emb, .attn, .modIn)
        (vidAttn, txtAttn) = attn(vidAttn, txtAttn, vidShape: vidShape, txtLen: txtLen)
        vidAttn = ada.modulateVid(vidAttn, emb, .attn, .modOut)
        txtAttn = ada.modulateTxt(txtAttn, emb, .attn, .modOut)
        vid = vid + vidAttn
        if !isLastLayer { txt = txt + txtAttn }

        var vidMlp = ada.modulateVid(rmsNormOnes(vid, eps: eps), emb, .mlp, .modIn)
        let txtNorm = isLastLayer ? txt : rmsNormOnes(txt, eps: eps)
        var txtMlp = ada.modulateTxt(txtNorm, emb, .mlp, .modIn)
        (vidMlp, txtMlp) = mlp(vidMlp, txtMlp)
        vidMlp = ada.modulateVid(vidMlp, emb, .mlp, .modOut)
        txtMlp = ada.modulateTxt(txtMlp, emb, .mlp, .modOut)
        vid = vid + vidMlp
        if !isLastLayer { txt = txt + txtMlp }

        return (vid, txt)
    }
}
