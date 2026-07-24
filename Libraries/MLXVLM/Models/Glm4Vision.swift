//
//  Glm4Vision.swift
//  mlx-swift-lm
//
//  Shared vision helpers for the GLM-4V family (glm4v, glm4v_moe). These two
//  ports carried byte-identical copies of the position-embedding interpolation;
//  the numerically sensitive part lives here so there is exactly one copy to be
//  correct.
//

import Foundation
import MLX

enum Glm4SharedVision {

    /// `grid_sample` with `mode="bicubic"`, `align_corners=False`,
    /// `padding_mode="border"` — matching `torch.nn.functional.grid_sample` as GLM-4V's
    /// `Glm4vVisionEmbeddings.forward` invokes it (`interpolated_method = "bicubic"`, hardcoded
    /// in HF transformers `modeling_glm4v.py` / `modeling_glm4v_moe.py`).
    ///
    /// The earlier port used BILINEAR here. That diverges from the reference at every
    /// non-grid-aligned sample — i.e. across the whole interior of any upsampled position grid,
    /// not just the borders — so it produced subtly wrong position embeddings on any image whose
    /// patch grid differs from the learned 24×24. (The `padding_mode="border"` clamp below is
    /// correct and matches the reference; only the interpolation kernel was wrong.)
    ///
    /// - Parameters:
    ///   - x: source, `(B, H, W, C)`.
    ///   - grid: sample coordinates, `(B, gN, gM, 2)`, last axis `(gx, gy)` in `[-1, 1]`.
    /// - Returns: `(B, gN, gM, C)`.
    static func gridSampleBicubic(_ x: MLXArray, grid: MLXArray) -> MLXArray {
        let B = x.dim(0)
        let H = x.dim(1)
        let W = x.dim(2)
        let C = x.dim(3)
        let gN = grid.dim(1)
        let gM = grid.dim(2)

        // Un-normalize from [-1, 1] to pixel coordinates (align_corners=False).
        let gx = grid[.ellipsis, 0]  // (B, gN, gM)
        let gy = grid[.ellipsis, 1]
        let ix = ((gx + 1) * Float(W) - 1) / 2
        let iy = ((gy + 1) * Float(H) - 1) / 2

        let ixFloor = floor(ix)
        let iyFloor = floor(iy)
        let tx = ix - ixFloor  // fractional part in [0, 1)
        let ty = iy - iyFloor

        // Cubic-convolution weights for the four taps at offsets {-1, 0, 1, 2}.
        let wx = cubicWeights(tx)
        let wy = cubicWeights(ty)

        let xFlat = x.reshaped(B, H * W, C)

        func clampX(_ v: MLXArray) -> MLXArray {
            clip(v, min: MLXArray(Int32(0)), max: MLXArray(Int32(W - 1))).asType(.int32)
        }
        func clampY(_ v: MLXArray) -> MLXArray {
            clip(v, min: MLXArray(Int32(0)), max: MLXArray(Int32(H - 1))).asType(.int32)
        }

        // Gather x[b, clamp(iyc), clamp(ixc)] for the whole (gN, gM) grid → (B, gN, gM, C).
        // Border padding = clamping the tap indices to the valid range.
        func gather(_ iyc: MLXArray, _ ixc: MLXArray) -> MLXArray {
            let flatIdx = (iyc * Int32(W) + ixc).reshaped(B, gN * gM)
            var slices = [MLXArray]()
            for b in 0 ..< B {
                let idxB = flatIdx[b]
                let g = xFlat[b][idxB]  // (gN*gM, C)
                slices.append(g[.newAxis])
            }
            return concatenated(slices, axis: 0).reshaped(B, gN, gM, C)
        }

        // Separable bicubic: interpolate the four sampled rows in x, then combine in y.
        let offsets = [-1, 0, 1, 2]
        var out: MLXArray?
        for (j, dy) in offsets.enumerated() {
            let iyc = clampY(iyFloor + Float(dy))
            var row: MLXArray?
            for (i, dx) in offsets.enumerated() {
                let ixc = clampX(ixFloor + Float(dx))
                let v = gather(iyc, ixc)
                let w = wx[i][.ellipsis, .newAxis]  // (B, gN, gM, 1)
                row = row.map { $0 + v * w } ?? v * w
            }
            let wyj = wy[j][.ellipsis, .newAxis]
            out = out.map { $0 + row! * wyj } ?? row! * wyj
        }
        return out!
    }

    /// Keys cubic-convolution weights (`A = -0.75`, matching PyTorch's
    /// `get_cubic_upsample_coefficients`) for taps at `floor-1, floor, floor+1, floor+2`,
    /// given the fractional offset `t ∈ [0, 1)`.
    private static func cubicWeights(_ t: MLXArray) -> [MLXArray] {
        let a: Float = -0.75
        // conv1 for |x| <= 1, conv2 for 1 < |x| < 2.
        func conv1(_ x: MLXArray) -> MLXArray { ((a + 2) * x - (a + 3)) * x * x + 1 }
        func conv2(_ x: MLXArray) -> MLXArray { ((a * x - 5 * a) * x + 8 * a) * x - 4 * a }
        return [conv2(t + 1), conv1(t), conv1(1 - t), conv2(2 - t)]
    }
}
