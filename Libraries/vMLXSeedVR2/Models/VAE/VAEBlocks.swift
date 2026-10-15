// SPDX-License-Identifier: MIT (port). SeedVR2 model/weights: Apache-2.0 / ByteDance.
// Port of mflux seedvr2_vae blocks (resnet/attention/down/up/mid).
import Foundation
import MLX
import MLXFast
import MLXNN

func groupNorm32(_ dims: Int) -> GroupNorm {
    GroupNorm(groupCount: 32, dimensions: dims, eps: 1e-6, pytorchCompatible: true)
}

/// ResnetBlock3D (identical for encoder & decoder).
public final class ResnetBlock3D: Module {
    @ModuleInfo(key: "norm1") var norm1: GroupNorm
    @ModuleInfo(key: "norm2") var norm2: GroupNorm
    @ModuleInfo(key: "conv1") var conv1: CausalConv3d
    @ModuleInfo(key: "conv2") var conv2: CausalConv3d
    @ModuleInfo(key: "conv_shortcut") var convShortcut: CausalConv3d?

    public init(_ inCh: Int, _ outCh: Int) {
        self._norm1.wrappedValue = groupNorm32(inCh)
        self._norm2.wrappedValue = groupNorm32(outCh)
        self._conv1.wrappedValue = CausalConv3d(inCh, outCh)
        self._conv2.wrappedValue = CausalConv3d(outCh, outCh)
        self._convShortcut.wrappedValue =
            inCh != outCh
            ? CausalConv3d(inCh, outCh, kernel: (1, 1, 1), padding: (0, 0, 0)) : nil
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = conv1(silu(vaeGroupNorm(x, norm1)))
        h = conv2(silu(vaeGroupNorm(h, norm2)))
        let residual = convShortcut?(x) ?? x
        return h + residual
    }
}

/// Attention3D — spatial self-attention within the VAE mid block.
public final class Attention3D: Module {
    @ModuleInfo(key: "group_norm") var groupNorm: GroupNorm
    @ModuleInfo(key: "to_q") var toQ: Linear
    @ModuleInfo(key: "to_k") var toK: Linear
    @ModuleInfo(key: "to_v") var toV: Linear
    @ModuleInfo(key: "to_out") var toOut: [Linear]
    let scale: Float

    public init(_ channels: Int) {
        self._groupNorm.wrappedValue = groupNorm32(channels)
        self._toQ.wrappedValue = Linear(channels, channels)
        self._toK.wrappedValue = Linear(channels, channels)
        self._toV.wrappedValue = Linear(channels, channels)
        self._toOut.wrappedValue = [Linear(channels, channels)]
        self.scale = powf(Float(channels), -0.5)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let s = x.shape
        let (B, C, T, H, W) = (s[0], s[1], s[2], s[3], s[4])
        let residual = x
        var h = x.transposed(0, 2, 1, 3, 4).reshaped([B * T, C, H * W]).transposed(0, 2, 1)  // [B*T, HW, C]
        h = groupNorm(h.asType(.float32)).asType(VAEPrecision.dtype)
        let q = toQ(h).expandedDimensions(axis: 1)
        let k = toK(h).expandedDimensions(axis: 1)
        let v = toV(h).expandedDimensions(axis: 1)
        var o = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: .none)
        o = o.squeezed(axis: 1)
        o = toOut[0](o).transposed(0, 2, 1).reshaped([B, T, C, H, W]).transposed(0, 2, 1, 3, 4)
        return o + residual
    }
}

public final class Downsample3D: Module {
    @ModuleInfo(key: "conv") var conv: CausalConv3d
    public init(_ channels: Int, spatialOnly: Bool) {
        let (kt, st, pt) = spatialOnly ? (1, 1, 0) : (3, 2, 1)
        self._conv.wrappedValue = CausalConv3d(
            channels, channels, kernel: (kt, 3, 3),
            stride: (st, 2, 2), padding: (pt, 0, 0))
        super.init()
    }
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let padded = MLX.padded(
            x, widths: [.init((0, 0)), .init((0, 0)), .init((0, 0)), .init((0, 1)), .init((0, 1))])
        return conv(padded)
    }
}

public final class Upsample3D: Module {
    @ModuleInfo(key: "conv") var conv: CausalConv3d
    @ModuleInfo(key: "upscale_conv") var upscaleConv: CausalConv3d
    let sf = 2, tf: Int
    public init(_ channels: Int, temporalUp: Bool) {
        self.tf = temporalUp ? 2 : 1
        let total = sf * sf * tf
        self._conv.wrappedValue = CausalConv3d(channels, channels, usePaddingCausal: true)
        self._upscaleConv.wrappedValue = CausalConv3d(
            channels, channels * total, kernel: (1, 1, 1), padding: (0, 0, 0))
        super.init()
    }
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let s = x.shape
        let (B, C, T, H, W) = (s[0], s[1], s[2], s[3], s[4])
        var h = upscaleConv(x)
        h = h.reshaped([B, sf, sf, tf, C, T, H, W]).transposed(0, 4, 5, 3, 6, 1, 7, 2)
        h = h.reshaped([B, C, T * tf, H * sf, W * sf])
        if T == 1 && tf > 1 { h = h[0..., 0..., 0 ..< 1] }
        return conv(h)
    }
}

public final class MidBlock3D: Module {
    @ModuleInfo(key: "attentions") var attentions: [Attention3D]
    @ModuleInfo(key: "resnets") var resnets: [ResnetBlock3D]
    public init(_ channels: Int) {
        self._attentions.wrappedValue = [Attention3D(channels)]
        self._resnets.wrappedValue = [
            ResnetBlock3D(channels, channels), ResnetBlock3D(channels, channels),
        ]
        super.init()
    }
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        resnets[1](attentions[0](resnets[0](x)))
    }
}

public final class DownBlock3D: Module {
    @ModuleInfo(key: "resnets") var resnets: [ResnetBlock3D]
    @ModuleInfo(key: "downsamplers") var downsamplers: [Downsample3D]
    public init(_ inCh: Int, _ outCh: Int, numLayers: Int, addDownsample: Bool, temporalDown: Bool)
    {
        self._resnets.wrappedValue = (0 ..< numLayers).map {
            ResnetBlock3D($0 == 0 ? inCh : outCh, outCh)
        }
        self._downsamplers.wrappedValue =
            addDownsample ? [Downsample3D(outCh, spatialOnly: !temporalDown)] : []
        super.init()
    }
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for r in resnets { h = r(h) }
        for d in downsamplers { h = d(h) }
        return h
    }
}

public final class UpBlock3D: Module {
    @ModuleInfo(key: "resnets") var resnets: [ResnetBlock3D]
    @ModuleInfo(key: "upsamplers") var upsamplers: [Upsample3D]
    public init(_ inCh: Int, _ outCh: Int, numLayers: Int, addUpsample: Bool, temporalUp: Bool) {
        self._resnets.wrappedValue = (0 ..< numLayers).map {
            ResnetBlock3D($0 == 0 ? inCh : outCh, outCh)
        }
        self._upsamplers.wrappedValue =
            addUpsample ? [Upsample3D(outCh, temporalUp: temporalUp)] : []
        super.init()
    }
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for r in resnets { h = r(h) }
        for u in upsamplers { h = u(h) }
        return h
    }
}
