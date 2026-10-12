import Foundation
import MLX
import XCTest

@testable import MLXLLM

/// Two-dispatch mHC collapse against the graph form it replaces (`DeepseekV4Math.hcPreGraph`), at GLM-5.3 shapes
/// (4 streams x 4096, 24 mix entries, 20 Sinkhorn rounds) for a decode row and an 8-row verify slab.
final class MHCFusedDecodeTests: XCTestCase {
    func testMatchesGraphCollapse() throws {
        MLXRandom.seed(3)
        let d = 4096
        let fn = (MLXRandom.normal([24, 4 * d]) * 0.02).asType(.bfloat16)
        let base = MLXRandom.normal([24]) * 0.5
        let scale = MLXArray([Float(0.8), 0.6, 1.2])
        for rows in [1, 8] {
            let streams = (MLXRandom.normal([1, rows, 4, d]) * 3).asType(.bfloat16)
            let graph = DeepseekV4Math.hcPreGraph(
                streams, fn: fn, scale: scale, base: base, hcMult: 4, hiddenSize: d, iters: 20, eps: 1e-6,
                normEps: 1e-5)
            let fused = try XCTUnwrap(MHCFusedDecode.collapse(
                streams, fn: fn, scale: scale, base: base, iters: 20, eps: 1e-6, normEps: 1e-5))
            func maxAbs(_ a: MLXArray, _ b: MLXArray) -> Float {
                abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
            }
            XCTAssertEqual(fused.x.shape, graph.x.shape)
            XCTAssertEqual(fused.post.shape, graph.post.shape)
            XCTAssertEqual(fused.comb.shape, graph.comb.shape)
            let px = maxAbs(fused.post, graph.post), cx = maxAbs(fused.comb, graph.comb)
            let xx = maxAbs(fused.x, graph.x), xmax = abs(graph.x.asType(.float32)).max().item(Float.self)
            print("mHC rows=\(rows): post \(px) comb \(cx) collapsed \(xx) (max |x| \(xmax))")
            XCTAssertLessThan(px, 1e-5)
            XCTAssertLessThan(cx, 1e-5)
            XCTAssertLessThan(xx, xmax * 1e-2)
        }
    }
}
