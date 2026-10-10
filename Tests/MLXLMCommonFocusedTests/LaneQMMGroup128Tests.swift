// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Lane matmul on group-128 affine weights (Diplodocus preview 3 stamps nearly every projection at
// group 128). 4-bit and direct 8-bit kernels read GS = 128 natively; 5/6-bit (and 8-bit without the
// direct kernel) run at 64 with each group's scale and bias repeated, which is the same dequantization.

import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLMCommon

private final class LaneHolder: Module {
    @ModuleInfo(key: "proj") var proj: QuantizedLinear
    init(_ p: QuantizedLinear) { _proj.wrappedValue = p }
}

final class LaneQMMGroup128Tests: XCTestCase {

    /// Every Diplodocus projection shape (N, K): q/o, k/v, gate/up, down; plus a small shape.
    private let shapes = [(4096, 4096), (1024, 4096), (12288, 4096), (4096, 12288), (256, 1024)]

    private func relativeError(_ a: MLXArray, _ b: MLXArray) -> Float {
        let diff = abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
        let scale = abs(b.asType(.float32)).max().item(Float.self)
        return diff / max(scale, 1e-6)
    }

    /// Lane output against an fp32 matmul of the exact dequantized weight, for every row count the
    /// verify uses, tiled and untiled. MLX's own quantized matmul is measured against the same
    /// reference so the bound reflects bf16 output rounding, not a lane-specific allowance.
    func testGroup128LaneMatchesDequantizedReference() throws {
        try FocusedMLXTestSupport.withLock {
            guard LaneQMM.available() else { throw XCTSkip("no Metal-4 tensor ops") }
            MLXRandom.seed(128)
            defer { LaneQMM.direct8Enabled = true }
            for bits in [4, 5, 8] {
                for direct8 in bits == 8 ? [true, false] : [true] {
                    LaneQMM.direct8Enabled = direct8
                    for (n, k) in shapes {
                        let w = (MLXRandom.normal([n, k]) * 0.04).asType(.bfloat16)
                        let (q, s, b) = MLX.quantized(w, groupSize: 128, bits: bits)
                        let layer = QuantizedLinear(
                            weight: q, bias: nil, scales: s.asType(.bfloat16),
                            biases: b!.asType(.bfloat16), groupSize: 128, bits: bits, mode: .affine)
                        let group = try XCTUnwrap(LaneQMM.laneGroup(for: layer), "bits \(bits)")
                        XCTAssertTrue(LaneQMM.takes(layer), "bits \(bits) not admitted")
                        let repeats = 128 / group
                        let sbt = repeats == 1
                            ? LaneQMM.packScales(layer.scales, layer.biases!)
                            : LaneQMM.packScales(
                                repeated(layer.scales, count: repeats, axis: 1),
                                repeated(layer.biases!, count: repeats, axis: 1))
                        let tiledQ = LaneQMM.tileWeight(q, bits: bits, group: group)
                        let dense = dequantized(
                            q, scales: layer.scales, biases: layer.biases!, groupSize: 128, bits: bits
                        ).asType(.float32)
                        for rows in [1, 2, 3, 4, 7, 8, 16] {
                            let x = MLXRandom.normal([rows, k]).asType(.bfloat16)
                            let reference = matmul(x.asType(.float32), dense.T)
                            let mlx = quantizedMM(
                                x, q, scales: layer.scales, biases: layer.biases!, transpose: true,
                                groupSize: 128, bits: bits, mode: .affine)
                            let bound = max(4 * relativeError(mlx, reference), 0.01)
                            for (weight, tiled) in [(q, false), (tiledQ, true)] {
                                let lane = LaneQMM.laneMatmul(
                                    x, weight: weight, sbt: sbt, bits: bits, group: group, tiled: tiled)
                                XCTAssertEqual(lane.shape, [rows, n])
                                let err = relativeError(lane, reference)
                                XCTAssertLessThanOrEqual(
                                    err, bound,
                                    "bits \(bits) direct8 \(direct8) (\(n),\(k)) rows \(rows) tiled \(tiled)")
                            }
                        }
                    }
                }
            }
        }
    }

    /// The installed module answers through the lane for verify-width calls and through MLX's own
    /// matmul (bundle scales, untiled weight) for prefill-width calls; both agree with the original.
    func testInstalledGroup128ModuleKeepsBundleScalesForWideCalls() throws {
        try FocusedMLXTestSupport.withLock {
            guard LaneQMM.available() else { throw XCTSkip("no Metal-4 tensor ops") }
            MLXRandom.seed(7)
            for bits in [4, 5, 8] {
                let w = (MLXRandom.normal([1024, 4096]) * 0.04).asType(.bfloat16)
                let (q, s, b) = MLX.quantized(w, groupSize: 128, bits: bits)
                let original = QuantizedLinear(
                    weight: q, bias: nil, scales: s.asType(.bfloat16), biases: b!.asType(.bfloat16),
                    groupSize: 128, bits: bits, mode: .affine)
                let holder = LaneHolder(original)
                let report = LaneQMM.install(model: holder)
                XCTAssertEqual(report.lane, 1, "bits \(bits)")
                let lane = try XCTUnwrap(holder.proj as? LaneQuantizedLinear)
                XCTAssertEqual(lane.groupSize, 128)
                for rows in [3, 8, 200] {
                    let x = MLXRandom.normal([rows, 4096]).asType(.bfloat16)
                    let reference = quantizedMM(
                        x, q, scales: original.scales, biases: original.biases!, transpose: true,
                        groupSize: 128, bits: bits, mode: .affine)
                    let err = relativeError(lane(x), reference)
                    XCTAssertLessThanOrEqual(err, 0.02, "bits \(bits) rows \(rows)")
                }
            }
        }
    }
}
