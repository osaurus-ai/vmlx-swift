import Foundation
import MLX
import XCTest

@testable import MLXLMCommon

final class JANGTFusedRotationTests: XCTestCase {
    func testOptInFusedRotationMicrobenchmark() throws {
        guard ProcessInfo.processInfo.environment["VMLX_TEST_JANGT_FUSED_BENCH"] == "1" else {
            throw XCTSkip("Opt-in diagnostic, not a model-speed qualification")
        }
        try MLXMetalTestLock.withLock {
            var rng = JANGTLowBitKernelTests.Rng(state: 137)
            let experts = 8
            for width in [1.5, 2.5] {
                let columns = Array(repeating: [Int32(17), -1], count: experts)
                let gate = try JANGTLowBitKernelTests.Projection(
                    &rng, k: width, experts: experts, outputs: 640, inputs: 2560,
                    columns: columns, columnScales: false
                ).mlx()
                let up = try JANGTLowBitKernelTests.Projection(
                    &rng, k: width, experts: experts, outputs: 640, inputs: 2560,
                    columns: columns, columnScales: false
                ).mlx()
                let down = try JANGTLowBitKernelTests.Projection(
                    &rng, k: width, experts: experts, outputs: 2560, inputs: 640,
                    columns: columns, columnScales: true
                ).mlx()
                let guSigns =
                    gate.signs
                    * JANGTRoutedExpertLayer.keepMask(
                        width: 2560, columns: gate.outlierColumnIndices[0 ..< 1])[0]
                let downSigns =
                    down.columnScales! * down.signs.expandedDimensions(axis: 0)
                    * JANGTRoutedExpertLayer.keepMask(
                        width: 640, columns: down.outlierColumnIndices)
                for tokens in [1, 4] {
                    let input = MLXArray(
                        (0 ..< tokens * 2560).map { _ in rng.normal() }, [tokens, 2560]
                    ).asType(.bfloat16)
                    let ids = MLXArray(
                        (0 ..< tokens * experts).map { UInt32($0 % experts) }, [tokens, experts])
                    let weights =
                        MLXArray.ones([tokens, experts], dtype: .float32) / Float(experts)
                    func run(fused: Bool) -> Double {
                        var x = input
                        let start = ProcessInfo.processInfo.systemUptime
                        for _ in 0 ..< 16 {
                            let h =
                                fused
                                ? JANGTKernels.gateUpWithRotation(
                                    x: x, signs: guSigns, indices: ids, gate: gate, up: up,
                                    limit: 0, outputDType: .float32)
                                : JANGTKernels.gateUp(
                                    xr: JANGTKernels.rotateRows(x, signs: guSigns), x: x,
                                    indices: ids, gate: gate, up: up, limit: 0,
                                    outputDType: .float32)
                            let y =
                                fused
                                ? JANGTKernels.downWithRotation(
                                    h: h, signs: downSigns, indices: ids, weights: weights,
                                    projection: down)
                                : JANGTKernels.down(
                                    hr: JANGTKernels.rotateRows(
                                        h, signs: downSigns, rowExperts: ids.flattened()), h: h,
                                    indices: ids, weights: weights, projection: down)
                            x = (x.asType(.float32) + 0.001 * y).asType(.bfloat16)
                        }
                        MLX.eval(x)
                        return ProcessInfo.processInfo.systemUptime - start
                    }
                    _ = run(fused: false)
                    _ = run(fused: true)
                    var old: [Double] = []
                    var candidate: [Double] = []
                    for i in 0 ..< 4 {
                        if i % 2 == 0 {
                            old.append(run(fused: false))
                            candidate.append(run(fused: true))
                        } else {
                            candidate.append(run(fused: true))
                            old.append(run(fused: false))
                        }
                    }
                    let a = old.sorted()[2]
                    let b = candidate.sorted()[2]
                    print(
                        "FUSED_ROTATION_MICRO K=\(width) rows=\(tokens) layers=16 samples=4 old_ms=\(a * 1000) fused_ms=\(b * 1000) ratio=\(a / b) scope=hot-bank-host-plus-gpu"
                    )
                }
            }
        }
    }

    func testFusedRotationPreservesDecodeForTrellisWidthsTailsAndVerifyRows() throws {
        try MLXMetalTestLock.withLock {
            var rng = JANGTLowBitKernelTests.Rng(state: 79)
            for width in [1.0, 1.5, 2, 2.5, 3] {
                for withOutliers in [false, true] {
                    let guColumns: [[Int32]] =
                        withOutliers ? [[17, -1], [17, -1]] : [[-1], [-1]]
                    let downColumns: [[Int32]] =
                        withOutliers ? [[7, -1], [12, -1]] : [[-1], [-1]]
                    let gate = try JANGTLowBitKernelTests.Projection(
                        &rng, k: width, experts: 2, outputs: 640, inputs: 2560,
                        columns: guColumns, columnScales: false
                    ).mlx()
                    let up = try JANGTLowBitKernelTests.Projection(
                        &rng, k: width, experts: 2, outputs: 640, inputs: 2560,
                        columns: guColumns, columnScales: false
                    ).mlx()
                    let down = try JANGTLowBitKernelTests.Projection(
                        &rng, k: width, experts: 2, outputs: 2560, inputs: 640,
                        columns: downColumns, columnScales: true
                    ).mlx()
                    let guSigns =
                        gate.signs
                        * JANGTRoutedExpertLayer.keepMask(
                            width: 2560, columns: gate.outlierColumnIndices[0 ..< 1])[0]
                    let downSigns =
                        down.columnScales! * down.signs.expandedDimensions(axis: 0)
                        * JANGTRoutedExpertLayer.keepMask(
                            width: 640, columns: down.outlierColumnIndices)
                    for dtype: DType in [.float32, .float16, .bfloat16] {
                        for tokens in [1, 4] {
                            let magnitude: Float = tokens == 4 ? 4096 : 1
                            let x = MLXArray(
                                (0 ..< tokens * 2560).map { _ in magnitude * rng.normal() },
                                [tokens, 2560]
                            ).asType(dtype)
                            // Duplicate experts and unequal route weights must preserve slot accumulation order.
                            let ids = MLXArray(
                                (0 ..< tokens * 8).map { UInt32(($0 / 3) % 2) }, [tokens, 8])
                            let weights = MLXArray(
                                (0 ..< tokens * 8).map { Float($0 % 8 + 1) / 36 }, [tokens, 8])
                            for limit: Float in [0, 1.5] {
                                let reference = JANGTKernels.gateUp(
                                    xr: JANGTKernels.rotateRows(x, signs: guSigns), x: x,
                                    indices: ids,
                                    gate: gate, up: up, limit: limit, outputDType: .float32)
                                let fused = JANGTKernels.gateUpWithRotation(
                                    x: x, signs: guSigns, indices: ids, gate: gate, up: up,
                                    limit: limit, outputDType: .float32)
                                MLX.eval(reference, fused)
                                let label =
                                    "K=\(width) outliers=\(withOutliers) dtype=\(dtype) rows=\(tokens) limit=\(limit)"
                                XCTAssertEqual(
                                    reference.asArray(Float.self).map(\.bitPattern),
                                    fused.asArray(Float.self).map(\.bitPattern), label)
                                let expected = JANGTKernels.down(
                                    hr: JANGTKernels.rotateRows(
                                        reference, signs: downSigns, rowExperts: ids.flattened()),
                                    h: reference, indices: ids, weights: weights, projection: down)
                                let actual = JANGTKernels.downWithRotation(
                                    h: reference, signs: downSigns, indices: ids, weights: weights,
                                    projection: down)
                                MLX.eval(expected, actual)
                                XCTAssertEqual(
                                    expected.asArray(Float.self).map(\.bitPattern),
                                    actual.asArray(Float.self).map(\.bitPattern), label)
                            }
                        }
                    }
                }
            }
        }
    }
}
