import Foundation
import MLX
import Testing

@testable import MLXVLM

/// Pins `Glm4SharedVision.gridSampleBicubic` to `torch.nn.functional.grid_sample(mode="bicubic",
/// align_corners=False, padding_mode="border")` — the call GLM-4V's vision embeddings make.
///
/// Two of these anchors are independent ground truth (they hold for any cubic-convolution constant
/// `A`, so they'd survive a wrong kernel): sampling exactly on a pixel center returns that pixel, and
/// a constant field stays constant (the weights partition to 1). The fractional-interior values are
/// the `A`-sensitive ones; they're pinned to a numpy reimplementation of PyTorch's Keys `A = -0.75`
/// coefficients (a torch cross-check is noted in the PR).
@Suite("Glm4SharedVision.gridSampleBicubic")
struct Glm4VisionTests {

    /// Grid coordinate of pixel-center `c` of `n` under align_corners=False: `(2c+1)/n − 1`.
    private static func center(_ c: Int, _ n: Int) -> Float { (2 * Float(c) + 1) / Float(n) - 1 }

    /// `(H, W)` single-channel image → `(1, H, W, 1)`.
    private static func image(_ rows: [[Float]]) -> MLXArray {
        MLXArray(rows.flatMap { $0 }, [1, rows.count, rows[0].count, 1])
    }

    /// One sample at grid `(gx, gy)` → scalar.
    private static func sampleOne(_ img: MLXArray, gx: Float, gy: Float) -> Float {
        let grid = MLXArray([gx, gy], [1, 1, 1, 2])
        let out = Glm4SharedVision.gridSampleBicubic(img, grid: grid)
        return out.reshaped(-1).item(Float.self)
    }

    // Computed, not stored: a static `let MLXArray` is a non-Sendable global and Swift 6 strict
    // concurrency rejects it.
    private static var arange4x4: MLXArray {
        image([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]])
    }

    // MARK: - Independent ground truth (holds for any A)

    @Test("sampling a pixel center returns that pixel exactly")
    func integerGridIsExact() {
        // At t=0 the cubic weights are exactly [0, 1, 0, 0] regardless of A.
        for (r, c, want) in [(0, 0, Float(0)), (1, 2, 6), (3, 3, 15)] {
            let got = Self.sampleOne(
                Self.arange4x4, gx: Self.center(c, 4), gy: Self.center(r, 4))
            #expect(abs(got - want) < 1e-4, "pixel (\(r),\(c)): got \(got), want \(want)")
        }
    }

    @Test("a constant field stays constant (weights partition to 1)")
    func constantFieldIsPreserved() {
        let cst = Self.image(Array(repeating: Array(repeating: Float(7), count: 4), count: 4))
        #expect(abs(Self.sampleOne(cst, gx: 0.13, gy: -0.42) - 7) < 1e-4)
    }

    // MARK: - A-sensitive interior (pinned to numpy Keys A = -0.75; cross-check vs torch)

    @Test("fractional interior samples match the bicubic reference")
    func fractionalInteriorMatchesReference() {
        let cases: [(Float, Float, Float)] = [
            (0.10, -0.30, 5.087_000),
            (-0.55, 0.25, 8.316_000),
            (0.333_333, 0.777_778, 14.378_987),
        ]
        for (gx, gy, want) in cases {
            let got = Self.sampleOne(Self.arange4x4, gx: gx, gy: gy)
            #expect(abs(got - want) < 1e-3, "gx=\(gx) gy=\(gy): got \(got), want \(want)")
        }
    }

    // MARK: - Border padding (the reference passes padding_mode="border")

    @Test("out-of-bounds samples clamp to the edge, not zero")
    func borderPaddingClampsToEdge() {
        // Row-ramp [0,1,2,3] per row; sampling past the right edge must approach the edge column
        // value (~3), not fall toward 0 as zero-padding would.
        let ramp = Self.image(Array(repeating: [Float(0), 1, 2, 3], count: 4))
        let got = Self.sampleOne(ramp, gx: 0.95, gy: 0.0)
        #expect(abs(got - 3.108) < 1e-3, "border sample: got \(got)")
        #expect(got > 2.5, "zero-padding would pull this well below the edge value")
    }
}
