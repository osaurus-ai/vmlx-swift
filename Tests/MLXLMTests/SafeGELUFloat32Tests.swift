// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Cmlx
import CmlxGraphShim
import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLMCommon

/// `safeGeluApproximateFloat32` and `SafeGELU` against a host oracle: the same formula in host
/// `Float`, rounded once to the input's dtype (round to nearest even on the bit pattern). The
/// oracle runs no MLX code. Each check runs on the GPU stream, where bf16 and fp16 take the fused
/// Metal kernel, and on a CPU stream, where they take the float32 chain.
final class SafeGELUFloat32Tests: XCTestCase {
    private static var forms: [(name: String, apply: (MLXArray) -> MLXArray)] {
        let layer = SafeGELU()
        return [
            ("safeGeluApproximateFloat32", { safeGeluApproximateFloat32($0) }),
            ("SafeGELU()", { layer($0) }),
        ]
    }
    private static let devices: [Device] = [.gpu, .cpu]

    // MARK: - The host oracle

    /// `safeGeluApproximate`'s formula and constants, in host `Float`.
    private static func oracle(_ x: Float) -> Float {
        0.5 * x * (1 + tanhf((2 / Float.pi).squareRoot() * (x + 0.044715 * x * x * x)))
    }

    /// The derivative of the same formula, in host `Double`.
    private static func derivative(_ x: Float) -> Float {
        let (d, c, a) = (Double(x), (2 / Double.pi).squareRoot(), 0.044715)
        let t = tanh(c * (d + a * d * d * d))
        return Float(0.5 * (1 + t) + 0.5 * d * (1 - t * t) * c * (1 + 3 * a * d * d))
    }

    private static func value(_ bits: UInt16, _ dtype: DType) -> Float {
        dtype == .bfloat16
            ? Float(bitPattern: UInt32(bits) << 16) : Float(Float16(bitPattern: bits))
    }

    /// `x` rounded once to `dtype`, to nearest even.
    private static func rounded(_ x: Float, _ dtype: DType) -> UInt16 {
        guard dtype == .bfloat16 else { return Float16(x).bitPattern }
        if x.isNaN { return 0x7FC0 }
        let bits = x.bitPattern
        return UInt16(truncatingIfNeeded: (bits &+ 0x7FFF &+ ((bits >> 16) & 1)) >> 16)
    }

    /// Every bit pattern of `dtype` whose value is in [-3, 6], -0 and +0 included. Below -3,
    /// 1 + tanh cancels so far that host and GPU float32 may round to different values. Metal
    /// flushes float32 subnormals to zero, today's bf16 path included, so on the GPU the bf16
    /// values with 0 < |x| < 2^-125 (their result is a float32 subnormal) are left out too. The
    /// CPU does not flush, so a CPU stream gets all of them. The specials test covers both ends
    /// of that band on both streams.
    private static func domain(_ dtype: DType, on device: Device) -> [UInt16] {
        let flushes = device.deviceType == .gpu && dtype == .bfloat16
        return (0 ... UInt16.max).filter { bits in
            let x = value(bits, dtype)
            return !x.isNaN && x >= -3 && x <= 6 && !(flushes && x != 0 && abs(x) < 0x1p-125)
        }
    }

    /// The bit patterns ordered as the values are, so that -0 and +0 are the same point.
    private static func distance(_ a: UInt16, _ b: UInt16) -> Int {
        func key(_ v: UInt16) -> Int { v & 0x8000 == 0 ? Int(v) : -Int(v & 0x7FFF) }
        return abs(key(a) - key(b))
    }

    /// `f` of the bit patterns `bits` as `dtype`, on `device`: the output dtype and bit patterns.
    private static func run(
        _ f: (MLXArray) -> MLXArray, _ bits: [UInt16], _ dtype: DType, on device: Device
    ) -> (dtype: DType, bits: [UInt16]) {
        Device.withDefaultDevice(device) {
            let y = f(MLXArray(bits).view(dtype: dtype))
            guard y.dtype == dtype else { return (y.dtype, []) }
            return (y.dtype, y.view(dtype: .uint16).asArray(UInt16.self))
        }
    }

    /// Checks that `actual` has the dtype `dtype` and is within bit distance 1 of the oracle at
    /// each of `inputs`, and prints how many values are at distance 1 and beyond.
    private static func checkNearOracle(
        _ label: String, _ inputs: [UInt16], _ actual: (dtype: DType, bits: [UInt16]),
        _ dtype: DType, file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertEqual(actual.dtype, dtype, "\(label): output dtype", file: file, line: line)
        guard actual.dtype == dtype else { return }
        let expected = inputs.map { rounded(oracle(value($0, dtype)), dtype) }
        let distances = zip(actual.bits, expected).map { distance($0, $1) }
        let worst = distances.indices.max { distances[$0] < distances[$1] }!
        print(
            "SafeGELUFloat32Tests: \(label): \(inputs.count) values, "
                + "\(distances.filter { $0 == 1 }.count) at bit distance 1, "
                + "\(distances.filter { $0 > 1 }.count) beyond, max \(distances[worst])")
        XCTAssertLessThanOrEqual(
            distances[worst], 1,
            "\(label): bit distance to the oracle at x = \(value(inputs[worst], dtype)): "
                + "\(value(actual.bits[worst], dtype)) against \(value(expected[worst], dtype))",
            file: file, line: line)
    }

    // MARK: - The tests

    func testBFloat16IsTheFloat32ResultRoundedOnce() {
        MLXMetalTestLock.withLock { Self.checkAgainstOracle(.bfloat16) }
    }

    func testFloat16IsTheFloat32ResultRoundedOnce() {
        MLXMetalTestLock.withLock { Self.checkAgainstOracle(.float16) }
    }

    private static func checkAgainstOracle(
        _ dtype: DType, file: StaticString = #filePath, line: UInt = #line
    ) {
        for form in forms {
            for device in devices {
                let inputs = domain(dtype, on: device)
                let label = "\(dtype) \(form.name) on \(device.deviceType?.rawValue ?? "?")"
                checkNearOracle(
                    label, inputs, run(form.apply, inputs, dtype, on: device), dtype,
                    file: file, line: line)
            }
        }
    }

    func testFloat32InputIsUnchanged() {
        MLXMetalTestLock.withLock {
            let points =
                (0 ..< 4096).map { -8 + 16 * Float($0) / 4095 } + [0, .infinity, -.infinity, .nan]
            for form in Self.forms {
                for device in Self.devices {
                    let label = "\(form.name) on \(device.deviceType?.rawValue ?? "?")"
                    let (actual, reference) = Device.withDefaultDevice(device) {
                        let x = MLXArray(points)
                        return (form.apply(x), safeGeluApproximate(x))
                    }
                    XCTAssertEqual(actual.dtype, .float32, "\(label): output dtype")
                    guard actual.dtype == .float32 else { continue }
                    let a = actual.asArray(Float.self)
                    let r = reference.asArray(Float.self)
                    let mismatches = zip(a, r).filter { pair in
                        pair.0.isNaN != pair.1.isNaN
                            || (!pair.0.isNaN && pair.0.bitPattern != pair.1.bitPattern)
                    }.count
                    XCTAssertEqual(
                        mismatches, 0, "\(label): bit patterns that differ from safeGeluApproximate"
                    )
                }
            }
        }
    }

    func testSpecialValuesKeepTheirMeaning() {
        MLXMetalTestLock.withLock {
            // +0, -0, +inf, -inf, NaN, +max finite, -max finite, +min subnormal, -min subnormal;
            // for bf16 also ±(the largest value below 2^-125), the top of the band that Metal
            // flushes to zero.
            let specials: [(DType, [UInt16])] = [
                (
                    .bfloat16,
                    [
                        0x0000, 0x8000, 0x7F80, 0xFF80, 0x7FC0, 0x7F7F, 0xFF7F, 0x0001, 0x8001,
                        0x00FF, 0x80FF,
                    ]
                ),
                (
                    .float16,
                    [0x0000, 0x8000, 0x7C00, 0xFC00, 0x7E00, 0x7BFF, 0xFBFF, 0x0001, 0x8001]
                ),
            ]
            for (dtype, inputs) in specials {
                for form in Self.forms {
                    for device in Self.devices {
                        let label = "\(dtype) \(form.name) on \(device.deviceType?.rawValue ?? "?")"
                        let actual = Self.run(form.apply, inputs, dtype, on: device)
                        let reference = Self.run(safeGeluApproximate, inputs, dtype, on: device)
                        XCTAssertEqual(actual.dtype, dtype, "\(label): output dtype")
                        guard actual.dtype == dtype else { continue }
                        for (i, (a, r)) in zip(actual.bits, reference.bits).enumerated() {
                            let x = Self.value(inputs[i], dtype)
                            let (va, vr) = (Self.value(a, dtype), Self.value(r, dtype))
                            let same =
                                va.isNaN == vr.isNaN && va.isInfinite == vr.isInfinite
                                && (!va.isInfinite || va.sign == vr.sign)
                                && (va.isNaN || va.isInfinite || Self.distance(a, r) <= 1)
                            XCTAssertTrue(
                                same,
                                "\(label): x = \(x) gives \(va); safeGeluApproximate gives \(vr)")
                        }
                    }
                }
            }
        }
    }

    func testFusedAndChainAgree() {
        MLXMetalTestLock.withLock {
            for dtype in [DType.bfloat16, .float16] {
                // The GPU's domain: in the band that the GPU flushes, the two cannot agree.
                let inputs = Self.domain(dtype, on: .gpu)
                for form in Self.forms {
                    let label = "\(dtype) \(form.name)"
                    let fused = Self.run(form.apply, inputs, dtype, on: .gpu)
                    let chain = Self.run(form.apply, inputs, dtype, on: .cpu)
                    XCTAssertEqual(fused.dtype, dtype, "\(label): GPU output dtype")
                    XCTAssertEqual(chain.dtype, dtype, "\(label): CPU output dtype")
                    guard fused.dtype == dtype, chain.dtype == dtype else { continue }
                    let distances = zip(fused.bits, chain.bits).map { Self.distance($0, $1) }
                    let worst = distances.indices.max { distances[$0] < distances[$1] }!
                    print(
                        "SafeGELUFloat32Tests: \(label) GPU against CPU: \(inputs.count) values, "
                            + "\(distances.filter { $0 == 1 }.count) at bit distance 1, "
                            + "\(distances.filter { $0 > 1 }.count) beyond, max \(distances[worst])"
                    )
                    XCTAssertLessThanOrEqual(
                        distances[worst], 1,
                        "\(label): GPU against CPU at x = \(Self.value(inputs[worst], dtype)): "
                            + "\(Self.value(fused.bits[worst], dtype)) against \(Self.value(chain.bits[worst], dtype))"
                    )
                }
            }
        }
    }

    /// The dispatch, read from the output's graph before it runs. On the GPU stream bf16 and fp16
    /// take the fused kernel, whose graph has no AsType node. On a CPU stream they take the
    /// float32 chain, which has two: the up-cast and the final cast.
    func testKernelRunsOnTheGPUAndTheChainOnTheCPU() throws {
        // A compiled chain is one node, so its casts do not show in the graph.
        try XCTSkipIf(
            HardwareInfo.isCompiledDecodeSupported, "VMLX_ENABLE_UNSAFE_COMPILE compiles the chain")
        MLXMetalTestLock.withLock {
            for dtype in [DType.bfloat16, .float16] {
                for form in Self.forms {
                    for (device, expected) in [(Device.gpu, Int32(0)), (Device.cpu, Int32(2))] {
                        let label = "\(dtype) \(form.name) on \(device.deviceType?.rawValue ?? "?")"
                        let (status, casts): (Int32, Int32) = Device.withDefaultDevice(device) {
                            let bits = Self.domain(dtype, on: device)
                            let y = form.apply(MLXArray(bits).view(dtype: dtype))
                            var nodes: Int32 = -1
                            var casts: Int32 = -1
                            let status = vmlx_graph_stats(y.ctx.ctx, &nodes, &casts)
                            return (status, casts)
                        }
                        XCTAssertEqual(status, 0, "\(label): vmlx_graph_stats status")
                        XCTAssertEqual(casts, expected, "\(label): AsType nodes in the graph")
                    }
                }
            }
        }
    }

    /// Inside a compiled function the input is a tracer, so the kernel does not run and the
    /// float32 chain gives the result.
    func testCompiledCallIsTheFloat32ResultRoundedOnce() {
        MLXMetalTestLock.withLock {
            for dtype in [DType.bfloat16, .float16] {
                let inputs = Self.domain(dtype, on: .gpu)
                for form in Self.forms {
                    let label = "\(dtype) \(form.name) compiled on gpu"
                    // vmlxTrustedCompile traces even when VMLX_ENABLE_UNSAFE_COMPILE is not set;
                    // compile(_:) then calls the function directly, without a trace.
                    let compiled = vmlxTrustedCompile { (a: [MLXArray]) in [form.apply(a[0])] }
                    Self.checkNearOracle(
                        label, inputs, Self.run({ compiled([$0])[0] }, inputs, dtype, on: .gpu),
                        dtype)
                }
            }
        }
    }

    /// Under valueAndGrad the input is a tracer, so the kernel (which has no derivative) does not
    /// run. The float32 chain gives the value and the gradient.
    func testValueAndGradTakesTheChain() {
        MLXMetalTestLock.withLock {
            let points: [Float] = [-3, -2, -1.5, -1, -0.5, -0.25, 0, 0.25, 0.5, 1, 1.5, 2, 3, 4, 6]
            for dtype in [DType.bfloat16, .float16] {
                let inputs = points.map { Self.rounded($0, dtype) }
                for form in Self.forms {
                    let label = "\(dtype) \(form.name) under valueAndGrad on gpu"
                    let f = valueAndGrad { (a: [MLXArray]) -> [MLXArray] in
                        let y = form.apply(a[0])
                        return [y.asType(.float32).sum(), y]
                    }
                    let result: ([MLXArray], [MLXArray])
                    do {
                        result = try Device.withDefaultDevice(.gpu) {
                            try withError { f([MLXArray(inputs).view(dtype: dtype)]) }
                        }
                    } catch {
                        XCTFail("\(label): \(error)")
                        continue
                    }
                    guard result.0.count == 2, result.1.count == 1 else {
                        XCTFail("\(label): \(result.0.count) values, \(result.1.count) gradients")
                        continue
                    }
                    let (y, g) = (result.0[1], result.1[0])
                    XCTAssertEqual(y.dtype, dtype, "\(label): value dtype")
                    XCTAssertEqual(g.dtype, dtype, "\(label): gradient dtype")
                    guard y.dtype == dtype, g.dtype == dtype else { continue }
                    let yBits = y.view(dtype: .uint16).asArray(UInt16.self)
                    let gBits = g.view(dtype: .uint16).asArray(UInt16.self)
                    for (i, bits) in inputs.enumerated() {
                        let x = Self.value(bits, dtype)
                        let (ey, eg) = (
                            Self.rounded(Self.oracle(x), dtype),
                            Self.rounded(Self.derivative(x), dtype)
                        )
                        XCTAssertLessThanOrEqual(
                            Self.distance(yBits[i], ey), 1,
                            "\(label): value at x = \(x): \(Self.value(yBits[i], dtype)) "
                                + "against \(Self.value(ey, dtype))")
                        XCTAssertLessThanOrEqual(
                            Self.distance(gBits[i], eg), 1,
                            "\(label): gradient at x = \(x): \(Self.value(gBits[i], dtype)) "
                                + "against \(Self.value(eg, dtype))")
                    }
                }
            }
        }
    }

    /// A transposed input and a zero-stride broadcast input, on the GPU stream.
    func testTransposedAndBroadcastInputs() {
        MLXMetalTestLock.withLock {
            let dtype = DType.bfloat16
            let all = Self.domain(dtype, on: .gpu)
            // 32516 = 44 * 739: every value of the domain, as a [44, 739] array.
            let (rows, columns) = (44, all.count / 44)
            XCTAssertEqual(rows * columns, all.count, "the domain fills the array")
            let transposed = (0 ..< columns).flatMap { c in
                (0 ..< rows).map { r in all[r * columns + c] }
            }
            // x = -3 (the largest error of today's path) and 255 values spread over the domain.
            let row = [Self.rounded(-3, dtype)] + (0 ..< 255).map { all[$0 * (all.count / 255)] }
            let broadcastRows = Array(repeating: row, count: 4).flatMap { $0 }
            for form in Self.forms {
                Self.checkNearOracle(
                    "bf16 \(form.name) transposed on gpu", transposed,
                    Self.run(
                        { form.apply($0.reshaped(rows, columns).transposed()) }, all, dtype,
                        on: .gpu), dtype)
                Self.checkNearOracle(
                    "bf16 \(form.name) broadcast on gpu", broadcastRows,
                    Self.run(
                        { form.apply(broadcast($0.reshaped(1, row.count), to: [4, row.count])) },
                        row, dtype, on: .gpu), dtype)
            }
        }
    }

    func testEmptyInputKeepsItsShapeAndDtype() {
        MLXMetalTestLock.withLock {
            for dtype in [DType.bfloat16, .float16] {
                for form in Self.forms {
                    for device in Self.devices {
                        let label = "\(dtype) \(form.name) on \(device.deviceType?.rawValue ?? "?")"
                        // The sum forces the empty output. Its bytes are not read back:
                        // asArray of an empty array traps on its nil data pointer.
                        let (shape, outputType, sum) = Device.withDefaultDevice(device) {
                            let y = form.apply(MLXArray.zeros([0, 3072], dtype: dtype))
                            return (y.shape, y.dtype, y.asType(.float32).sum().item(Float.self))
                        }
                        XCTAssertEqual(shape, [0, 3072], "\(label): output shape")
                        XCTAssertEqual(outputType, dtype, "\(label): output dtype")
                        XCTAssertEqual(sum, 0, "\(label): sum of the output")
                    }
                }
            }
        }
    }
}
