// SPDX-License-Identifier: MIT (port). SeedVR2 model/weights: Apache-2.0 / ByteDance.
// Port of mflux seedvr2_transformer/patch_in.py + patch_out.py.
import MLX
import MLXNN

/// Patchify `[B,C,T,H,W]` by `patchSize` and project to `dim`. Returns (tokens, [T',H',W']).
public final class PatchIn: Module {
    @ModuleInfo(key: "proj") var proj: Linear
    let patch: [Int]

    public init(inChannels: Int = 33, patchSize: [Int] = [1, 2, 2], dim: Int = 2560) {
        self.patch = patchSize
        let (t, h, w) = (patchSize[0], patchSize[1], patchSize[2])
        self._proj.wrappedValue = Linear(inChannels * t * h * w, dim)
        super.init()
    }

    public func callAsFunction(_ vidIn: MLXArray) -> (MLXArray, [Int]) {
        let (t, h, w) = (patch[0], patch[1], patch[2])
        let s = vidIn.shape
        let (B, C, T, H, W) = (s[0], s[1], s[2], s[3], s[4])
        let (tp, hp, wp) = (T / t, H / h, W / w)

        var vid = vidIn.reshaped([B, C, tp, t, hp, h, wp, w])
        vid = vid.transposed(0, 2, 4, 6, 3, 5, 7, 1)
        vid = vid.reshaped([B, tp, hp, wp, t * h * w * C])
        vid = proj(vid)
        vid = vid.reshaped([B, -1, vid.shape[vid.ndim - 1]])
        return (vid, [tp, hp, wp])
    }
}

/// Inverse of PatchIn: project tokens back to `[B,C_out,T,H,W]`.
public final class PatchOut: Module {
    @ModuleInfo(key: "proj") var proj: Linear
    let patch: [Int]

    public init(outChannels: Int = 16, patchSize: [Int] = [1, 2, 2], dim: Int = 2560) {
        self.patch = patchSize
        let (t, h, w) = (patchSize[0], patchSize[1], patchSize[2])
        self._proj.wrappedValue = Linear(dim, outChannels * t * h * w)
        super.init()
    }

    public func callAsFunction(_ vidIn: MLXArray, vidShape: [Int]) -> MLXArray {
        let (t, h, w) = (patch[0], patch[1], patch[2])
        var vid = proj(vidIn)
        let B = vid.shape[0]
        let (tp, hp, wp) = (vidShape[0], vidShape[1], vidShape[2])
        let C = vid.shape[vid.ndim - 1] / (t * h * w)
        vid = vid.reshaped([B, tp, hp, wp, t, h, w, C])
        vid = vid.transposed(0, 7, 1, 4, 2, 5, 3, 6)
        vid = vid.reshaped([B, C, tp * t, hp * h, wp * w])
        return vid
    }
}
