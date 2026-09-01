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
import MLXLMCommon

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
    /// Validate the grid invariants the vision path relies on, at a point where throwing is
    /// allowed.
    ///
    /// The forward path reshapes each frame to `(h / merge, merge, w / merge, merge)`. If `h` or
    /// `w` is not divisible by the merge size the element count does not match and MLX traps; if
    /// the merge size is zero it divides by zero. Both are configuration mistakes with no
    /// recovery, but a trap takes the process down with a message about shapes rather than about
    /// the bundle.
    ///
    /// Checking once here rather than guarding each site is deliberate: `rotaryPositionEmbedding`
    /// and the rope-index walk are NOT throwing, and making them so would change signatures across
    /// the tower to restate an invariant that is fixed before any of them runs. `prepare` already
    /// throws, so the invariant is established at the boundary and relied on inside.
    public static func validateGrid(_ frames: [THW], spatialMergeSize merge: Int) throws {
        guard merge > 0 else {
            throw VLMError.processing(
                "vision config declares spatial_merge_size = \(merge); it must be positive")
        }
        for (i, f) in frames.enumerated() where f.h % merge != 0 || f.w % merge != 0 {
            throw VLMError.processing(
                "frame \(i) is \(f.h)x\(f.w), which spatial_merge_size \(merge) does not "
                    + "divide. The vision tower reshapes each frame by the merge size, so a "
                    + "non-divisible grid cannot be formed. This is a property of the bundle's "
                    + "preprocessor configuration, not of the request.")
        }
    }

    /// The M-RoPE position-index walk, shared by both GLM-4V towers.
    ///
    /// `glm4v` and `glm4v_moe` carried byte-identical copies of this, 89 lines each, differing
    /// only in the model name inside one `precondition` message. The walk depends on nothing but
    /// the token ids, the image grid, and two configuration scalars, so there was never a reason
    /// for there to be two of it — and a divergence here would show up as subtly wrong positions
    /// rather than as a failure.
    public static func ropeIndex(
        inputIds: MLXArray, imageGridThw: [THW]?, spatialMergeSize: Int, imageTokenId: Int,
        family: String
    ) -> (MLXArray, MLXArray) {
        let batchSize = inputIds.dim(0)
        let seqLength = inputIds.dim(1)

        guard let imageGridThw, !imageGridThw.isEmpty else {
            let positions = MLXArray(0 ..< Int32(seqLength)).expandedDimensions(axis: 0)
            let positionIds = tiled(
                broadcast(positions, to: [batchSize, seqLength]).expandedDimensions(axis: 0),
                repetitions: [3, 1, 1])
            let deltas = MLXArray(Int32(0))
            return (positionIds, deltas)
        }

        precondition(batchSize == 1, "\(family) getRopeIndex only supports batchSize == 1")
        let positionIds = zeros([3, batchSize, seqLength], type: Int32.self)
        var imageIndex = 0
        var mropePositionDelta: Int = 0

        for batchIdx in 0 ..< batchSize {
            let inputTokens: [Int32] = inputIds[batchIdx].asArray(Int32.self)

            var dimT = [Int32]()
            var dimH = [Int32]()
            var dimW = [Int32]()
            dimT.reserveCapacity(seqLength)
            dimH.reserveCapacity(seqLength)
            dimW.reserveCapacity(seqLength)
            var st = 0
            var lastMax: Int32 = -1

            let appendTextPositions = { (count: Int) in
                guard count > 0 else { return }
                let base: Int32 = lastMax + 1
                for j in 0 ..< count {
                    let pos = base + Int32(j)
                    dimT.append(pos)
                    dimH.append(pos)
                    dimW.append(pos)
                }
                lastMax = base + Int32(count) - 1
            }

            while imageIndex < imageGridThw.count {
                guard let ed = inputTokens[st...].firstIndex(of: Int32(imageTokenId)) else {
                    break
                }

                let frame = imageGridThw[imageIndex]
                let llmGridT = frame.t
                let llmGridH = frame.h / spatialMergeSize
                let llmGridW = frame.w / spatialMergeSize
                imageIndex += 1

                appendTextPositions(ed - st)

                let imgOffset: Int32 = lastMax + 1
                for t in 0 ..< llmGridT {
                    for h in 0 ..< llmGridH {
                        for w in 0 ..< llmGridW {
                            dimT.append(Int32(t) + imgOffset)
                            dimH.append(Int32(h) + imgOffset)
                            dimW.append(Int32(w) + imgOffset)
                        }
                    }
                }
                let tMax = Int32(llmGridT - 1) + imgOffset
                let hMax = Int32(llmGridH - 1) + imgOffset
                let wMax = Int32(llmGridW - 1) + imgOffset
                lastMax = max(tMax, max(hMax, wMax))

                st = ed + llmGridT * llmGridH * llmGridW
            }

            appendTextPositions(inputTokens.count - st)

            positionIds[0, batchIdx] = MLXArray(dimT)
            positionIds[1, batchIdx] = MLXArray(dimH)
            positionIds[2, batchIdx] = MLXArray(dimW)

            mropePositionDelta = Int(lastMax) + 1 - inputTokens.count
        }

        let deltas = MLXArray(Int32(mropePositionDelta))
        return (positionIds, deltas)
    }
}
