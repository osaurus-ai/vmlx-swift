// SPDX-License-Identifier: MIT (port). SeedVR2 model/weights: Apache-2.0 / ByteDance.
// Port of mflux seedvr2_vae/{encoder,decoder,vae}.py.
import MLX
import MLXNN

public final class Encoder3D: Module {
    @ModuleInfo(key: "conv_in") var convIn: CausalConv3d
    @ModuleInfo(key: "down_blocks") var downBlocks: [DownBlock3D]
    @ModuleInfo(key: "mid_block") var midBlock: MidBlock3D
    @ModuleInfo(key: "conv_norm_out") var convNormOut: GroupNorm
    @ModuleInfo(key: "conv_out") var convOut: CausalConv3d

    public init(
        inChannels: Int = 3, outChannels: Int = 16,
        blockOut: [Int] = [128, 256, 512, 512], layersPerBlock: Int = 2, temporalDownBlocks: Int = 2
    ) {
        self._convIn.wrappedValue = CausalConv3d(inChannels, blockOut[0])
        var blocks: [DownBlock3D] = []
        var outCh = blockOut[0]
        let n = blockOut.count
        for (i, ch) in blockOut.enumerated() {
            let inCh = outCh
            outCh = ch
            let isFinal = i == n - 1
            let temporalDown = (i >= n - temporalDownBlocks - 1) && !isFinal
            blocks.append(
                DownBlock3D(
                    inCh, outCh, numLayers: layersPerBlock, addDownsample: !isFinal,
                    temporalDown: temporalDown))
        }
        self._downBlocks.wrappedValue = blocks
        self._midBlock.wrappedValue = MidBlock3D(blockOut[n - 1])
        self._convNormOut.wrappedValue = groupNorm32(blockOut[n - 1])
        self._convOut.wrappedValue = CausalConv3d(blockOut[n - 1], 2 * outChannels)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = convIn(x)
        for b in downBlocks { h = b(h) }
        h = midBlock(h)
        h = silu(vaeGroupNorm(h, convNormOut))
        return convOut(h)
    }
}

public final class Decoder3D: Module {
    @ModuleInfo(key: "conv_in") var convIn: CausalConv3d
    @ModuleInfo(key: "mid_block") var midBlock: MidBlock3D
    @ModuleInfo(key: "up_blocks") var upBlocks: [UpBlock3D]
    @ModuleInfo(key: "conv_norm_out") var convNormOut: GroupNorm
    @ModuleInfo(key: "conv_out") var convOut: CausalConv3d

    public init(
        inChannels: Int = 16, outChannels: Int = 3,
        blockOut: [Int] = [128, 256, 512, 512], layersPerBlock: Int = 3, temporalUpBlocks: Int = 2
    ) {
        let rev = Array(blockOut.reversed())
        self._convIn.wrappedValue = CausalConv3d(inChannels, rev[0])
        self._midBlock.wrappedValue = MidBlock3D(rev[0])
        var blocks: [UpBlock3D] = []
        var outCh = rev[0]
        let n = rev.count
        for (i, ch) in rev.enumerated() {
            let inCh = outCh
            outCh = ch
            let isFinal = i == n - 1
            let temporalUp = i < temporalUpBlocks
            blocks.append(
                UpBlock3D(
                    inCh, outCh, numLayers: layersPerBlock, addUpsample: !isFinal,
                    temporalUp: temporalUp))
        }
        self._upBlocks.wrappedValue = blocks
        self._convNormOut.wrappedValue = groupNorm32(rev[n - 1])
        self._convOut.wrappedValue = CausalConv3d(rev[n - 1], outChannels)
        super.init()
    }

    public func callAsFunction(_ z: MLXArray) -> MLXArray {
        var h = convIn(z)
        h = midBlock(h)
        for b in upBlocks { h = b(h) }
        h = silu(vaeGroupNorm(h, convNormOut))
        return convOut(h)
    }
}

public final class SeedVR2VAE: Module {
    @ModuleInfo(key: "encoder") var encoder: Encoder3D
    @ModuleInfo(key: "decoder") var decoder: Decoder3D
    let scalingFactor: Float = 0.9152
    let latentChannels = 16

    public override init() {
        self._encoder.wrappedValue = Encoder3D()
        self._decoder.wrappedValue = Decoder3D()
        super.init()
    }

    /// x [B,3,T,H,W] (or [B,3,H,W]) -> latent [B,16,T,H/8,W/8].
    public func encode(_ xIn: MLXArray) -> MLXArray {
        let x = xIn.ndim == 4 ? xIn.expandedDimensions(axis: 2) : xIn
        let h = encoder(x)
        let mean = h[0..., 0 ..< latentChannels]
        return mean * scalingFactor
    }

    /// z [B,16,T,H,W] -> image [B,3,T,H*8,W*8].
    public func decode(_ zIn: MLXArray) -> MLXArray {
        var z = zIn.ndim == 4 ? zIn.expandedDimensions(axis: 2) : zIn
        z = z / scalingFactor
        return decoder(z)
    }
}

extension SeedVR2VAE {
    /// Evaluate bounded stages so a cancelled job stops before submitting the next stage.
    public func encode(_ image: MLXArray, checkpoint: () throws -> Void) throws -> MLXArray {
        try checkpoint()
        let input = image.ndim == 4 ? image.expandedDimensions(axis: 2) : image
        var h = encoder.convIn(input)
        try checkedEval(h)
        try checkpoint()
        for block in encoder.downBlocks {
            h = block(h)
            try checkedEval(h)
            try checkpoint()
        }
        h = encoder.midBlock(h)
        try checkedEval(h)
        try checkpoint()
        h = encoder.convOut(silu(vaeGroupNorm(h, encoder.convNormOut)))
        h = h[0..., 0 ..< latentChannels] * scalingFactor
        try checkedEval(h)
        try checkpoint()
        return h
    }

    public func decode(_ latent: MLXArray, checkpoint: () throws -> Void) throws -> MLXArray {
        try checkpoint()
        let input = latent.ndim == 4 ? latent.expandedDimensions(axis: 2) : latent
        var h = decoder.convIn(input / scalingFactor)
        h = decoder.midBlock(h)
        try checkedEval(h)
        try checkpoint()
        for block in decoder.upBlocks {
            h = block(h)
            try checkedEval(h)
            try checkpoint()
        }
        h = decoder.convOut(silu(vaeGroupNorm(h, decoder.convNormOut)))
        try checkedEval(h)
        try checkpoint()
        return h
    }
}
