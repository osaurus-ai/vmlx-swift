// Copyright © 2026 Osaurus contributors.

import Cmlx
import Foundation
import XCTest
@testable import MLX

final class MLX0322IntegrationTests: XCTestCase {
    override func setUp() {
        prepareMLXMetallibForTests()
    }

    func testLinkedCoreVersion() {
        var version = mlx_string_new()
        defer { mlx_string_free(version) }
        XCTAssertEqual(mlx_version(&version), 0)
        let actual = String(cString: mlx_string_data(version))
        XCTAssertEqual(actual, "0.32.3")
        print("MLX0322 linked_core_version=\(actual)")
    }

    func testSharedStreamAcrossOSWorkers() {
        // Creation, evaluation and final synchronization occur on different
        // OS threads. This exercises the native cross-thread encoder registry.
        let stream = Stream(Device.gpu)
        let done = DispatchGroup()
        for worker in 0 ..< 2 {
            done.enter()
            Thread.detachNewThread {
                defer { done.leave() }
                for iteration in 0 ..< 12 {
                    let input = MLXArray([Float(worker), Float(iteration), 2, 3])
                    let output = MLX.add(input, MLXArray(Float(7)), stream: .stream(stream))
                    eval(output)
                    XCTAssertEqual(output.asArray(Float.self),
                                   [Float(worker + 7), Float(iteration + 7), 9, 10])
                }
            }
        }
        XCTAssertEqual(done.wait(timeout: .now() + 30), .success)
        stream.synchronize()
    }

    func testDefaultStreamSelectionAndDrain() {
        func checkCurrentDefault() {
            let expected = StreamOrDevice.default.stream
            let selected = MLX.Stream()
            // Queue identity makes the negative control deterministic: an
            // unrelated queue may finish first, but cannot be our drain.
            XCTAssertEqual(selected, expected)
            let input = MLXArray((0 ..< 8192).map { Float($0 % 7) })
            let output = input + 3
            asyncEval(output)
            selected.synchronize()
            var available = false
            XCTAssertEqual(_mlx_array_is_available(&available, output.ctx), 0)
            XCTAssertTrue(available, "Default-stream drain must finish submitted work")
            // Cleanup even in the failing control; readback must not disguise
            // the missing drain because availability was checked first.
            expected.synchronize()
            XCTAssertEqual(output.asArray(Float.self),
                           (0 ..< 8192).map { Float($0 % 7) + 3 })
        }
        for device in [Device.cpu, Device.gpu] {
            Device.withDefaultDevice(device) {
                let original = StreamOrDevice.default.stream
                checkCurrentDefault()
                Stream.withNewDefaultStream(device: device) {
                    let scoped = StreamOrDevice.default.stream
                    XCTAssertNotEqual(scoped, original)
                    checkCurrentDefault()
                    Stream.withNewDefaultStream(device: device) {
                        XCTAssertNotEqual(StreamOrDevice.default.stream, scoped)
                        checkCurrentDefault()
                    }
                    XCTAssertEqual(StreamOrDevice.default.stream, scoped)
                }
                XCTAssertEqual(StreamOrDevice.default.stream, original)
            }
        }
        print("MLX0322 default_stream_drain cpu_gpu_nested_cases=6")
    }

    func testRunWithRestoresTargetDeviceAndNestedDefaults() {
        func nativeDefault(_ device: Device) -> MLX.Stream {
            var result = mlx_stream_new()
            XCTAssertEqual(mlx_get_default_stream(&result, device.ctx), 0)
            return Stream(result)
        }
        let oldCPU = nativeDefault(.cpu)
        let oldGPU = nativeDefault(.gpu)
        let cpu = Stream(Device.cpu)
        let gpu = Stream(Device.gpu)
        cpu.runWith {
            XCTAssertEqual(nativeDefault(.cpu), cpu)
            XCTAssertEqual(nativeDefault(.gpu), oldGPU)
            gpu.runWith {
                XCTAssertEqual(nativeDefault(.cpu), cpu)
                XCTAssertEqual(nativeDefault(.gpu), gpu)
            }
            XCTAssertEqual(nativeDefault(.gpu), oldGPU)
        }
        XCTAssertEqual(nativeDefault(.cpu), oldCPU)
        XCTAssertEqual(nativeDefault(.gpu), oldGPU)
    }

    private final class CompileOwner: @unchecked Sendable {
        let lock = NSLock()
        var function: CompiledFunction?
        var cacheIDs: Set<UInt64> = []
    }

    func testCompiledFunctionErasesEveryWorkerCacheOnForeignThread() throws {
        guard ProcessInfo.processInfo.environment["MLX_DISABLE_COMPILE"] == nil else {
            throw XCTSkip("Requires trusted compilation, not the hard-disable control")
        }
        let owner = CompileOwner()
        weak var released: CompiledFunction?
        let id: UInt = autoreleasepool {
            let function = CompiledFunction(inputs: [], outputs: [], shapeless: false,
                                            trusted: true) { [$0[0] * 2] }
            owner.function = function
            released = function
            return UInt(bitPattern: Unmanaged.passUnretained(function).toOpaque())
        }
        let populated = DispatchGroup()
        let finished = DispatchGroup()
        let resume = [DispatchSemaphore(value: 0), DispatchSemaphore(value: 0)]
        for worker in 0 ..< 2 {
            populated.enter()
            finished.enter()
            Thread.detachNewThread {
                defer { finished.leave() }
                autoreleasepool {
                    let function = owner.lock.withLock { owner.function! }
                    let result = function.call([MLXArray(Float(5))])
                    XCTAssertEqual(result[0].item(Float.self), 10)
                    owner.lock.withLock { _ = owner.cacheIDs.insert(mlx_detail_compile_cache_id()) }
                }
                populated.leave()
                guard resume[worker].wait(timeout: .now() + 30) == .success else {
                    XCTFail("Timed out waiting for foreign-thread deinit")
                    return
                }
                // Reuse exactly the old function id on the SAME live worker,
                // but with different math. A stale cache returns 10 instead
                // of 35 without tracing the replacement. This tests actual
                // cache erasure, not merely Swift object deallocation.
                evalLock.withLock {
                    var cache = mlx_compile_cache_new()
                    XCTAssertEqual(mlx_detail_compile_cache(&cache), 0)
                    let replacement = new_mlx_closure { [$0[0] * 7] }
                    var compiled = mlx_closure_new()
                    let args = new_mlx_vector_array([MLXArray(Float(5))])
                    var result = mlx_vector_array_new()
                    defer {
                        mlx_detail_compile_erase(cache, id)
                        mlx_compile_cache_free(cache)
                        mlx_closure_free(compiled)
                        mlx_closure_free(replacement)
                        mlx_vector_array_free(args)
                        mlx_vector_array_free(result)
                    }
                    XCTAssertEqual(mlx_detail_compile(&compiled, replacement, id, false, [], 0), 0)
                    XCTAssertEqual(mlx_closure_apply(&result, compiled, args), 0)
                    let output = mlx_vector_array_values(result)
                    XCTAssertEqual(output.count, 1)
                    if let first = output.first { XCTAssertEqual(first.item(Float.self), 35) }
                }
            }
        }
        let ready = populated.wait(timeout: .now() + 30)
        XCTAssertEqual(ready, .success)
        owner.lock.withLock { owner.function = nil }
        XCTAssertNil(released)
        XCTAssertEqual(owner.lock.withLock { owner.cacheIDs.count }, 2)
        resume.forEach { $0.signal() }
        XCTAssertEqual(finished.wait(timeout: .now() + 30), .success)
    }

    func testNestedTrustedCompileUsesRecursiveLockOrder() {
        let inner = CompiledFunction(inputs: [], outputs: [], shapeless: false,
                                     trusted: true) { [$0[0] * 3] }
        let outer = CompiledFunction(inputs: [], outputs: [], shapeless: false,
                                     trusted: true) { inner.call($0).map { $0 + 1 } }
        XCTAssertEqual(outer.call([MLXArray(Float(5))])[0].item(Float.self), 16)
    }

    func testCustomKernelSourceIdentityAndCompiledOutputShapes() {
        let add = MLXFast.metalKernel(
            name: "mlx0322_same_name", inputNames: ["inp"], outputNames: ["out"],
            source: "uint i = thread_position_in_grid.x; out[i] = inp[i] + 1.0f;")
        let multiply = MLXFast.metalKernel(
            name: "mlx0322_same_name", inputNames: ["inp"], outputNames: ["out"],
            source: "uint i = thread_position_in_grid.x; out[i] = inp[i] * 3.0f;")
        for width in [4, 8] {
            let input = MLXArray((0 ..< width).map(Float.init))
            let pendingAdd = add([input], grid: (width, 1, 1), threadGroup: (4, 1, 1),
                                 outputShapes: [[width]], outputDTypes: [.float32])[0]
            let product = multiply([input], grid: (width, 1, 1), threadGroup: (4, 1, 1),
                                   outputShapes: [[width]], outputDTypes: [.float32])[0]
            // Same name, different source, opposite evaluation order, with
            // the first lazy graph still alive when the second is evaluated.
            assertEqual(product, input * 3, rtol: 0, atol: 0)
            assertEqual(pendingAdd, input + 1, rtol: 0, atol: 0)
            let compiled = CompiledFunction(inputs: [], outputs: [], shapeless: false,
                                            trusted: true) { args in
                add(args, grid: (width, 1, 1), threadGroup: (4, 1, 1),
                    outputShapes: [[width]], outputDTypes: [.float32])
            }
            assertEqual(compiled.call([input])[0], input + 1, rtol: 0, atol: 0)
        }
    }

    func testFFTNormalizationAndDataSafetensorsRoundTrip() throws {
        let input = MLXArray([Float(1), 2, 3, 4, 5, 6, 7, 8])
        for stream in [StreamOrDevice.cpu, .gpu] {
            for norm in [FFTNorm.backward, .ortho, .forward] {
                let spectrum = MLXFFT.rfft(input, norm: norm, stream: stream)
                let restored = MLXFFT.irfft(spectrum, norm: norm, stream: stream)
                XCTAssertTrue(allClose(restored, input, atol: 1e-5).item(Bool.self))
            }
        }
        let bytes = try saveToData(arrays: ["input": input], metadata: ["test": "mlx0322"])
        let (arrays, metadata) = try loadArraysAndMetadata(data: bytes)
        XCTAssertEqual(metadata["test"], "mlx0322")
        XCTAssertEqual(try XCTUnwrap(arrays["input"]).asArray(Float.self), input.asArray(Float.self))
    }

    func testAffineBitWidthsDenseAndGatheredAcrossDispatchBoundaries() throws {
        var cases = 0
        var reducedPrecisionCases = 0
        var maxReducedPrecisionError: Float = 0
        // Run this matrix in TWO processes: default policy and explicit
        // MLX_ENABLE_TF32=0. The latter retains the strict F32 reference check
        // for every row. The former also qualifies NAX's TF32 error contract;
        // do not silently disable the production backend to make it pass.
        let permitsTF32 = ProcessInfo.processInfo.environment["MLX_ENABLE_TF32"] != "0"
        for bits in [1, 2, 3, 4, 6, 8] {
            for k in [64, 128, 256, 1024] {
                for n in [16, 33] {
                    let weights: MLXArray
                    let scales: MLXArray
                    let biases: MLXArray
                    let unpacked: MLXArray
                    if bits == 1 {
                        // JANG1L loads already-packed affine weights; neither
                        // old nor new affine_quantize creates one-bit weights.
                        // Construct LSB-packed words and an independent scalar
                        // reference instead of exercising an unsupported API.
                        let codes = (0 ..< 3 * n * k).map { UInt32(($0 * 13 + $0 / k) % 2) }
                        var words = [UInt32](repeating: 0, count: codes.count / 32)
                        for i in codes.indices { words[i / 32] |= codes[i] << UInt32(i % 32) }
                        let scaleValues = (0 ..< codes.count / 64).map { Float($0 % 4 + 1) / 4 }
                        let biasValues = (0 ..< codes.count / 64).map { -Float($0 % 3 + 1) / 8 }
                        weights = MLXArray(words, [3, n, k / 32])
                        scales = MLXArray(scaleValues, [3, n, k / 64])
                        biases = MLXArray(biasValues, [3, n, k / 64])
                        unpacked = MLXArray(codes.indices.map {
                            Float(codes[$0]) * scaleValues[$0 / 64] + biasValues[$0 / 64]
                        }, [3, n, k])
                        let decoded = dequantized(weights, scales: scales, biases: biases,
                                                  groupSize: 64, bits: bits)
                        assertEqual(decoded, unpacked, rtol: 0, atol: 0)
                    } else {
                        let raw = (0 ..< 3 * n * k).map { Float(($0 * 13 + $0 / k) % 17) / 16 - 0.5 }
                        let source = MLXArray(raw, [3, n, k])
                        let quantizedWeights = quantized(source, groupSize: 64, bits: bits)
                        weights = quantizedWeights.0
                        scales = quantizedWeights.1
                        biases = try XCTUnwrap(quantizedWeights.2)
                        unpacked = dequantized(weights, scales: scales, biases: biases,
                                               groupSize: 64, bits: bits, stream: .cpu)
                    }
                    eval(weights, scales, biases, unpacked)
                    for m in [1, 2, 8, 33] {
                        let x = MLXArray((0 ..< m * k).map { Float(($0 * 7) % 11) / 16 - 0.25 }, [m, k])
                        let dense = quantizedMM(x, weights[1], scales: scales[1], biases: biases[1],
                                                transpose: true, groupSize: 64, bits: bits)
                        let reference = matmul(x, unpacked[1].transposed(), stream: .cpu)
                        XCTAssertEqual(dense.dtype, .float32)
                        XCTAssertTrue(allClose(dense, reference, rtol: 2e-4, atol: 2e-4).item(Bool.self),
                                      "dense bits=\(bits) K=\(k) N=\(n) M=\(m) max_abs=\(abs(dense - reference).max().item(Float.self))")
                        let gathered = gatherQuantizedMM(
                            x.reshaped(1, m, k), weights, scales: scales, biases: biases,
                            rhsIndices: MLXArray([2, 0]), transpose: true, groupSize: 64, bits: bits)
                        let gatheredReference = stacked([
                            matmul(x, unpacked[2].transposed(), stream: .cpu),
                            matmul(x, unpacked[0].transposed(), stream: .cpu),
                        ])
                        let difference = abs(gathered - gatheredReference)
                        if permitsTF32 && m == 33 {
                            // This matrix's 33-row gathered products admit
                            // NAX. Use a conservative TF32-class budget
                            // (10 fraction bits) plus F32 accumulation per output,
                            // including cancellation; a relative-output-only
                            // tolerance is not a valid error model. This is
                            // a qualification budget, not a claim about the
                            // Metal hardware's exact internal representation.
                            // The separate strict process still checks these
                            // SAME weights at rtol/atol 2e-4 with no allowance.
                            let magnitudes = stacked([
                                matmul(abs(x), abs(unpacked[2]).transposed(), stream: .cpu),
                                matmul(abs(x), abs(unpacked[0]).transposed(), stream: .cpu),
                            ])
                            let unit: Float = 1.0 / 1024
                            let relativeBound = 2 * unit + unit * unit + Float(k + 2) * Float.ulpOfOne
                            let bound = magnitudes * relativeBound + Float(2e-4)
                            XCTAssertTrue((difference .<= bound).all().item(Bool.self),
                                          "TF32 bound bits=\(bits) K=\(k) N=\(n) M=\(m)")
                            maxReducedPrecisionError = Swift.max(maxReducedPrecisionError,
                                                                 difference.max().item(Float.self))
                            reducedPrecisionCases += 1
                        } else {
                            XCTAssertTrue(allClose(gathered, gatheredReference, rtol: 2e-4, atol: 2e-4).item(Bool.self),
                                          "strict gathered bits=\(bits) K=\(k) N=\(n) M=\(m)")
                        }
                        cases += 2
                    }
                }
            }
        }
        print("MLX0322 affine_bit_width_dispatch_cases=\(cases) tf32_bound_cases=\(reducedPrecisionCases) "
              + "max_tf32_abs=\(maxReducedPrecisionError) strict_f32_control=\(!permitsTF32)")
    }

    func testRMSNormDtypesTailsWeightsAndStrides() {
        var cases = 0
        for dtype in [DType.float32, .float16, .bfloat16] {
            for width in [31, 128, 257, 2560, 8193] {
                for strided in [false, true] {
                    let values = (0 ..< 3 * width * 2).map { Float(($0 * 13) % 127) / 128 - 0.5 }
                    let base = MLXArray(values, [3, width * 2]).asType(dtype)
                    let input = strided ? base[0..., .stride(by: 2)] : base[0..., ..<width]
                    let weight = MLXArray((0 ..< width).map { Float($0 % 7 + 1) / 8 }).asType(dtype)
                    for weighted in [false, true] {
                        var result = mlx_array_new()
                        XCTAssertEqual(mlx_fast_rms_norm(
                            &result, input.ctx, (weighted ? weight : MLXArray.mlxNone).ctx,
                            1e-5, StreamOrDevice.gpu.ctx), 0)
                        let actual = MLXArray(result)
                        let x32 = input.asType(.float32, stream: .cpu)
                        let squares = multiply(x32, x32, stream: .cpu)
                        let inverse = rsqrt(mean(squares, axis: -1, keepDims: true, stream: .cpu) + 1e-5,
                                            stream: .cpu)
                        var expected = multiply(x32, inverse, stream: .cpu).asType(dtype, stream: .cpu)
                        if weighted { expected = multiply(expected, weight, stream: .cpu) }
                        let tolerance = dtype == .bfloat16 ? 0.025 : (dtype == .float16 ? 0.003 : 2e-5)
                        assertEqual(actual, expected, rtol: tolerance, atol: tolerance)
                        XCTAssertEqual(actual.dtype, dtype)
                        cases += 1
                    }
                }
            }
        }
        print("MLX0322 rms_norm_reference_cases=\(cases)")
    }
}
