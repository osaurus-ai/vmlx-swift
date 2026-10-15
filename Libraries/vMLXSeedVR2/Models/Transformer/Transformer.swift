// SPDX-License-Identifier: MIT (port). SeedVR2 model/weights: Apache-2.0 / ByteDance.
// Port of mflux seedvr2_transformer/transformer.py (top-level assembly).
import MLX
import MLXNN

public final class SeedVR2Transformer: Module {
    @ModuleInfo(key: "vid_in") var vidIn: PatchIn
    @ModuleInfo(key: "txt_in") var txtIn: Linear
    @ModuleInfo(key: "emb_in") var embIn: TimeEmbedding
    @ModuleInfo(key: "blocks") var blocks: [TransformerBlock]
    @ModuleInfo(key: "vid_out_norm") var vidOutNorm: SeedVR2RMSNorm
    @ParameterInfo(key: "out_shift") var outShift: MLXArray
    @ParameterInfo(key: "out_scale") var outScale: MLXArray
    @ModuleInfo(key: "vid_out") var vidOut: PatchOut

    let vidDim: Int

    public init(_ cfg: SeedVR2Config = .r3B) {
        self.vidDim = cfg.vidDim
        let embDim = 6 * cfg.vidDim
        self._vidIn.wrappedValue = PatchIn(
            inChannels: cfg.vidInChannels, patchSize: cfg.patchSize, dim: cfg.vidDim)
        self._txtIn.wrappedValue = Linear(cfg.txtInDim, cfg.vidDim)
        self._embIn.wrappedValue = TimeEmbedding(
            sinusoidalDim: 256, hiddenDim: cfg.vidDim, outputDim: embDim)

        var blk: [TransformerBlock] = []
        for i in 0 ..< cfg.numLayers {
            let shared = i >= cfg.mmLayers
            let isLast = i == cfg.numLayers - 1  // 3B last_layer_vid_only
            let shift = i % 2 == 1
            blk.append(
                TransformerBlock(
                    vidDim: cfg.vidDim, txtDim: cfg.vidDim, heads: cfg.heads,
                    headDim: cfg.headDim, expandRatio: cfg.expandRatio, normEps: cfg.normEps,
                    ropeDim: cfg.ropeDim, ropeOnText: cfg.ropeOnText, shared: shared,
                    isLastLayer: isLast,
                    window: cfg.window, shift: shift))
        }
        self._blocks.wrappedValue = blk

        self._vidOutNorm.wrappedValue = SeedVR2RMSNorm(cfg.vidDim, eps: cfg.normEps)
        self._outShift.wrappedValue = MLXArray.zeros([cfg.vidDim])
        self._outScale.wrappedValue = MLXArray.ones([cfg.vidDim])
        self._vidOut.wrappedValue = PatchOut(
            outChannels: cfg.vidOutChannels, patchSize: cfg.patchSize, dim: cfg.vidDim)
        super.init()
    }

    /// vid [B,Cin,T,H,W], txt [B,Lt,txtInDim], timestep scalar. Returns [B,Cout,T,H,W].
    public func callAsFunction(
        _ vidIn0: MLXArray, _ txt0: MLXArray, timestep: MLXArray, checkpoint: () throws -> Void = {}
    ) throws -> MLXArray {
        try checkpoint()
        var txt = txtIn(txt0)
        let txtLen = txt.shape[1]
        var (vid, vidShape) = vidIn(vidIn0)  // [B,L,dim], [[t,h,w]]
        var emb = embIn(timestep)
        emb = emb.reshaped([-1, vidDim, 2, 3])

        for block in blocks {
            (vid, txt) = block(vid, txt, emb: emb, vidShape: [vidShape], txtLen: txtLen)
            try checkedEval(vid, txt)
            try checkpoint()
        }

        vid = vidOutNorm(vid)
        // output ada: hidden * (scale_a + out_scale) + (shift_a + out_shift), from emb[:,:,0,0:2]
        let mod = emb[0..., 0..., 0]  // [B,dim,3]
        let shiftA = mod[.ellipsis, 0].expandedDimensions(axis: 1)
        let scaleA = mod[.ellipsis, 1].expandedDimensions(axis: 1)
        vid = vid * (scaleA + outScale) + (shiftA + outShift)

        return vidOut(vid, vidShape: vidShape)
    }
}
