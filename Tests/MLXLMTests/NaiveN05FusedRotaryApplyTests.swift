import Foundation
#if canImport(Darwin)
import Darwin
#endif
import MLX
import MLXLMCommon
import MLXNN
import MLXRandom
import XCTest
@testable import MLXLLM

/// Strict native storage fixtures, including real typed-cache runtime paths.
/// Source preparation is separate from the owner's native execution and A/B.
final class NaiveN05FusedRotaryApplyTests: XCTestCase {
    private static func configuration(indexerPrecision: String = "fp8_e4m3",
                                      overrides: [String: Any] = [:]) throws -> NaiveN05ArchitectureContract {
        var values: [String: Any] = [
            "model_type": "naive_n05_flash", "hidden_size": 8, "intermediate_size": 12,
            "num_hidden_layers": 2, "vocab_size": 16, "num_attention_heads": 64,
            "num_key_value_heads": 4, "head_dim": 192, "v_head_dim": 128,
            "swa_num_attention_heads": 64, "swa_num_key_value_heads": 8,
            "swa_head_dim": 192, "swa_v_head_dim": 128, "partial_rotary_factor": 0.334,
            "n_routed_experts": 4, "num_experts_per_tok": 2, "moe_intermediate_size": 6,
            "hybrid_layer_pattern": [0, 1], "moe_layer_freq": [0, 1],
            "index_n_heads": 16, "index_head_dim": 128, "index_top_k": 3,
            "indexer_activation_dtype": indexerPrecision,
            "sliding_window": 3, "attention_value_scale": 0.707,
        ]
        values.merge(overrides) { _, override in override }
        return try JSONDecoder().decode(NaiveN05ArchitectureContract.self,
            from: JSONSerialization.data(withJSONObject: values))
    }

    private static func models(indexerPrecision: String = "fp8_e4m3") throws -> (NaiveN05FlashModel, NaiveN05FlashModel) {
        MLXRandom.seed(20261001)
        let baseline = try NaiveN05FlashModel(configuration(indexerPrecision: indexerPrecision),
            allowedMaskGPUArange: true,
            fusedRotaryApply: false)
        let candidate = try NaiveN05FlashModel(configuration(indexerPrecision: indexerPrecision),
            allowedMaskGPUArange: true,
            fusedRotaryApply: true)
        // Keep the router's checkpoint F32 contract; ordinary parameters are BF16.
        let parameters = ModuleParameters.unflattened(baseline.parameters().flattened().map {
            ($0.0, $0.1.asType($0.0.contains(".mlp.gate.") ? .float32 : .bfloat16))
        })
        try baseline.update(parameters: parameters, verify: [.all])
        try candidate.update(parameters: parameters, verify: [.all])
        return (baseline, candidate)
    }

    private static func assertStorage(_ actual: MLXArray, _ expected: MLXArray,
                                     file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(actual.dtype, expected.dtype, file: file, line: line)
        XCTAssertEqual(actual.shape, expected.shape, file: file, line: line)
        XCTAssertEqual(actual.asData().data, expected.asData().data, file: file, line: line)
    }

    /// Independent ordinary binary primitives, with product stores explicitly
    /// evaluated before the add/subtract. There is no compiled/custom oracle.
    private static func stagedReference(_ x: MLXArray, cosine: MLXArray,
                                        sine: MLXArray) -> MLXArray {
        let a = x[.ellipsis, ..<32], b = x[.ellipsis, 32..<64]
        let ac = a * cosine, bs = b * sine, bc = b * cosine, aS = a * sine
        MLX.eval(ac, bs, bc, aS)
        return concatenated([ac - bs, bc + aS, x[.ellipsis, 64...]], axis: -1)
    }

    private static let roles = [(4, 192), (8, 192), (64, 192), (1, 128), (16, 128)]

    private static func input(heads: Int, width: Int, strided: Bool) -> MLXArray {
        let values = (0 ..< heads * width).map { Float(($0 * 37) % 257 - 128) / 64 }
        if !strided { return MLXArray(values).reshaped(1, heads, 1, width).asType(.bfloat16) }
        let columnMajor = (0 ..< width).flatMap { d in (0 ..< heads).map { h in values[h * width + d] } }
        return MLXArray(columnMajor).reshaped(1, width, 1, heads)
            .asType(.bfloat16).transposed(0, 3, 2, 1)
    }

    private static func assertCaches(_ actual: [NaiveN05FlashCache], _ expected: [NaiveN05FlashCache],
                                    file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(actual.count, expected.count, file: file, line: line)
        for (a, b) in zip(actual, expected) {
            XCTAssertEqual(a.offset, b.offset, file: file, line: line)
            XCTAssertEqual(a.keyOffset, b.keyOffset, file: file, line: line)
            XCTAssertEqual(a.metaState, b.metaState, file: file, line: line)
            XCTAssertEqual(a.state.count, b.state.count, file: file, line: line)
            for (row, reference) in zip(a.state, b.state) {
                assertStorage(row, reference, file: file, line: line)
            }
        }
    }

    private struct Snapshot {
        let identity: ObjectIdentifier
        let rows: [MLXArray]
        let metadata: [String]
        let offset: Int
        let keyOffset: Int
        init(_ cache: NaiveN05FlashCache) {
            identity = ObjectIdentifier(cache)
            rows = cache.state
            metadata = cache.metaState
            offset = cache.offset
            keyOffset = cache.keyOffset
        }
    }

    private static func assertUnchanged(_ cache: [NaiveN05FlashCache], _ before: [Snapshot],
                                       file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(cache.count, before.count, file: file, line: line)
        for (entry, snapshot) in zip(cache, before) {
            XCTAssertEqual(ObjectIdentifier(entry), snapshot.identity, file: file, line: line)
            XCTAssertEqual(entry.offset, snapshot.offset, file: file, line: line)
            XCTAssertEqual(entry.keyOffset, snapshot.keyOffset, file: file, line: line)
            XCTAssertEqual(entry.metaState, snapshot.metadata, file: file, line: line)
            XCTAssertEqual(entry.state.count, snapshot.rows.count, file: file, line: line)
            for (a, b) in zip(entry.state, snapshot.rows) {
                assertStorage(a, b, file: file, line: line)
            }
        }
    }

    func testPolicyDefaultsToQualifiedM5IsImmutableAndKeepsNumericalCacheIdentity() throws {
        XCTAssertFalse(NaiveN05FusedRotaryApply.requested(environment: [:]))
        let c = try Self.configuration()
        XCTAssertTrue(NaiveN05FusedRotaryApply.defaultEnabled(c, backend: .gpu,
            metalDeviceName: "Apple M5 Max"))
        let otherBackends: [DeviceType?] = [.cpu, nil]
        for backend in otherBackends {
            XCTAssertFalse(NaiveN05FusedRotaryApply.defaultEnabled(c, backend: backend,
                metalDeviceName: "Apple M5 Max"))
        }
        let otherNames: [String?] = [nil, "", "Unknown", "Apple M5", "Apple M5 Pro", "Apple M4 Max",
            "applegpu_g17s", "Apple M5 Max ", "apple m5 max", "AMD Radeon"]
        for name in otherNames {
            XCTAssertFalse(NaiveN05FusedRotaryApply.defaultEnabled(c, backend: .gpu, metalDeviceName: name))
        }
        let otherGeometries: [[String: Any]] = [
            ["num_attention_heads": 32], ["num_key_value_heads": 8],
            ["swa_num_attention_heads": 32], ["swa_num_key_value_heads": 4],
            ["head_dim": 128], ["swa_head_dim": 128],
            ["partial_rotary_factor": 0.25], ["index_n_heads": 8], ["index_head_dim": 64],
        ]
        for overrides in otherGeometries {
            XCTAssertFalse(NaiveN05FusedRotaryApply.defaultEnabled(try Self.configuration(overrides: overrides),
                backend: .gpu, metalDeviceName: "Apple M5 Max"))
        }
        XCTAssertTrue(NaiveN05FusedRotaryApply.requested(environment: [:], defaultEnabled: true))
        for value in ["0", "true", "false", "yes", " 1 ", "", "2"] {
            XCTAssertFalse(NaiveN05FusedRotaryApply.requested(
                environment: ["VMLX_NAIVE_FUSED_ROTARY_APPLY": value]))
            XCTAssertFalse(NaiveN05FusedRotaryApply.requested(
                environment: ["VMLX_NAIVE_FUSED_ROTARY_APPLY": value], defaultEnabled: true))
        }
        XCTAssertTrue(NaiveN05FusedRotaryApply.requested(
            environment: ["VMLX_NAIVE_FUSED_ROTARY_APPLY": "1"]))
        try MLXMetalTestLock.withLock {
            let (baseline, candidate) = try Self.models()
            let flag = "VMLX_NAIVE_FUSED_ROTARY_APPLY"
            let original = getenv(flag).map { String(cString: $0) }
            defer {
                if let original { setenv(flag, original, 1) } else { unsetenv(flag) }
            }
            unsetenv(flag)
            let automatic = try NaiveN05FlashModel(Self.configuration())
            XCTAssertEqual(automatic.fusedRotaryApply,
                NaiveN05FusedRotaryApply.modelRequested(c, environment: [:]))
            let capturedDefault = automatic.fusedRotaryApply
            let metalName = NaiveN05FusedRotaryApply.nativeMetalDeviceName() ?? "unavailable"
            let backend = Device.defaultDevice().deviceType?.rawValue ?? "unavailable"
            print("[NaiveRotaryPolicy] metal_device_name=\(String(reflecting: metalName)) backend=\(backend) default_enabled=\(capturedDefault)")
            setenv(flag, "0", 1)
            let environmentOff = try NaiveN05FlashModel(c)
            setenv(flag, "1", 1)
            let environmentOn = try NaiveN05FlashModel(c)
            setenv(flag, "true", 1)
            let environmentUnrecognized = try NaiveN05FlashModel(c)
            XCTAssertFalse(environmentOff.fusedRotaryApply)
            XCTAssertTrue(environmentOn.fusedRotaryApply)
            XCTAssertFalse(environmentUnrecognized.fusedRotaryApply)
            XCTAssertEqual(automatic.fusedRotaryApply, capturedDefault)
            CompiledDecodeTrace.withActive {
                XCTAssertFalse(NaiveN05FusedRotaryApply.modelRequested(c, environment: [:]))
            }
            Device.withDefaultDevice(.cpu) {
                XCTAssertFalse(NaiveN05FusedRotaryApply.modelRequested(c, environment: [:]))
            }
            for model in [automatic, environmentOff, environmentOn, environmentUnrecognized] {
                XCTAssertEqual(model.cacheStorageDTypeIdentity, baseline.cacheStorageDTypeIdentity)
                XCTAssertEqual(model.newCache().map(\.diskCacheStateIdentifier),
                    baseline.newCache().map(\.diskCacheStateIdentifier))
            }
            XCTAssertFalse(baseline.fusedRotaryApply)
            XCTAssertTrue(candidate.fusedRotaryApply)
            XCTAssertFalse(NaiveN05FusedRotaryApply.requested(
                environment: ["VMLX_NAIVE_FUSED_ROTARY_APPLY": "0"]))
            XCTAssertTrue(candidate.fusedRotaryApply)
            XCTAssertEqual(candidate.cacheStorageDTypeIdentity, baseline.cacheStorageDTypeIdentity)
            XCTAssertEqual(candidate.cacheStorageDTypeIdentity, "naive-n05-paired-v1")
            XCTAssertEqual(candidate.newCache().map(\.diskCacheStateIdentifier),
                baseline.newCache().map(\.diskCacheStateIdentifier))
            XCTAssertTrue(candidate.allowedMaskGPUArange)
            XCTAssertFalse(candidate.supportsWholeForwardCompilation)
            let c = candidate.config
            XCTAssertEqual(c.fullAttention.heads, 64)
            XCTAssertEqual(c.fullAttention.kvHeads, 4)
            XCTAssertEqual(c.slidingAttention.heads, 64)
            XCTAssertEqual(c.slidingAttention.kvHeads, 8)
            XCTAssertEqual(c.fullAttention.keyDimensions, 192)
            XCTAssertEqual(c.slidingAttention.keyDimensions, 192)
            XCTAssertEqual(c.fullAttention.rotaryDimensions, 64)
            XCTAssertEqual(c.slidingAttention.rotaryDimensions, 64)
            XCTAssertEqual(c.indexerHeads, 16)
            XCTAssertEqual(c.indexerDimensions, 128)
            let position = MLXArray([Int32(9)]).reshaped(1, 1)
            let tables = NaiveN05FlashMath.RotaryTables(positions: position,
                fusedApply: candidate.fusedRotaryApply)
            XCTAssertTrue(tables.fusedApply)
            XCTAssertFalse(NaiveN05FlashMath.RotaryTables(positions: position).fusedApply)
            // Actual model-owned projection/norm wrappers must yield admitted
            // dtype/layout metadata; equal logits alone cannot prove activation.
            let states = MLXArray.ones([1, 1, c.hiddenDimensions], dtype: .bfloat16)
            for layer in candidate.model.layers {
                let attention = layer.attention, g = attention.geometry
                let operands = [
                    attention.query(states).reshaped(1, 1, g.heads, g.keyDimensions).transposed(0, 2, 1, 3),
                    attention.key(states).reshaped(1, 1, g.kvHeads, g.keyDimensions).transposed(0, 2, 1, 3),
                ]
                let phases = tables.phases(dimensions: g.rotaryDimensions,
                    theta: g.ropeTheta, dtype: .bfloat16)
                for operand in operands {
                    XCTAssertTrue(NaiveN05FusedRotaryApply.admits(operand,
                        cosine: phases.0, sine: phases.1, dimensions: g.rotaryDimensions))
                }
                if let indexer = attention.indexer {
                    let operands = [
                        indexer.wq(states).reshaped(1, 1, c.indexerHeads, c.indexerDimensions).transposed(0, 2, 1, 3),
                        indexer.keyNorm(indexer.wk(states)).expandedDimensions(axis: 1),
                    ]
                    for operand in operands {
                        XCTAssertTrue(NaiveN05FusedRotaryApply.admits(operand,
                            cosine: phases.0, sine: phases.1, dimensions: c.fullAttention.rotaryDimensions))
                    }
                }
            }
            XCTAssertEqual(tables.count, 2)
        }
    }

    func testBoundedMetadataAdmissionAndExactFallbacks() throws {
        try MLXMetalTestLock.withLock {
            XCTAssertEqual(Device.defaultDevice().deviceType, .gpu,
                "Native Metal qualification is required; this suite has no skips")
            let phase = MLXArray.ones([1, 1, 1, 32], dtype: .bfloat16)
            for (heads, width) in Self.roles {
                let x = Self.input(heads: heads, width: width, strided: false)
                XCTAssertTrue(NaiveN05FusedRotaryApply.admits(x,
                    cosine: phase, sine: phase, dimensions: 64))
                XCTAssertNotNil(NaiveN05FusedRotaryApply.apply(x,
                    cosine: phase, sine: phase, dimensions: 64))
                XCTAssertLessThanOrEqual(x.size, 64 * 192)
            }
            for shape in [[1, 2, 1, 192], [1, 4, 1, 128], [1, 16, 1, 192],
                          [1, 64, 1, 128], [1, 4, 1, 64], [1, 4, 1, 256],
                          [2, 4, 1, 192], [1, 4, 2, 192], [1, 0, 1, 192],
                          [1, 4, 0, 192], [0, 4, 1, 192], [4, 1, 192]] {
                let x = MLXArray.zeros(shape, dtype: .bfloat16)
                XCTAssertNil(NaiveN05FusedRotaryApply.apply(x,
                    cosine: phase, sine: phase, dimensions: 64), "shape=\(shape)")
            }
            let x = Self.input(heads: 4, width: 192, strided: false)
            for dimensions in [0, 2, 32, 62, 66, 96, 128, 192, Int.max] {
                XCTAssertNil(NaiveN05FusedRotaryApply.apply(x,
                    cosine: phase, sine: phase, dimensions: dimensions))
            }
            for bad in [MLXArray.ones([32], dtype: .bfloat16),
                        MLXArray.ones([1, 1, 1, 31], dtype: .bfloat16),
                        MLXArray.ones([1, 1, 2, 32], dtype: .bfloat16),
                        phase.asType(.float32), phase.asType(.float16)] {
                XCTAssertNil(NaiveN05FusedRotaryApply.apply(x,
                    cosine: bad, sine: phase, dimensions: 64))
                XCTAssertNil(NaiveN05FusedRotaryApply.apply(x,
                    cosine: phase, sine: bad, dimensions: 64))
            }
            CompiledDecodeTrace.withActive {
                XCTAssertNil(NaiveN05FusedRotaryApply.apply(x,
                    cosine: phase, sine: phase, dimensions: 64))
            }
            Device.withDefaultDevice(.cpu) {
                XCTAssertFalse(NaiveN05FusedRotaryApply.admits(x,
                    cosine: phase, sine: phase, dimensions: 64))
            }
            for (shape, dimensions, dtype) in [
                ([1, 2, 1, 192], 64, DType.bfloat16),
                ([1, 4, 1, 192], 96, .bfloat16),
                ([2, 4, 1, 192], 64, .bfloat16),
                ([1, 4, 3, 192], 64, .bfloat16),
                ([1, 4, 1, 192], 64, .float16),
                ([1, 4, 1, 192], 64, .float32),
            ] {
                let x = (MLXArray(0 ..< shape.reduce(1, *)).asType(.float32) / 64)
                    .reshaped(shape).asType(dtype)
                let positions = MLXArray(0 ..< shape[0] * shape[2]).reshaped(shape[0], shape[2])
                let on = NaiveN05FlashMath.RotaryTables(positions: positions, fusedApply: true)
                let off = NaiveN05FlashMath.RotaryTables(positions: positions)
                Self.assertStorage(NaiveN05FlashMath.rotary(x, positions: positions,
                    dimensions: dimensions, theta: 10_000_000, tables: on),
                    NaiveN05FlashMath.rotary(x, positions: positions,
                    dimensions: dimensions, theta: 10_000_000, tables: off))
            }
        }
    }

    func testDirectKernelPreservesEveryBF16InputWordAndTypedRoundingBoundary() throws {
        try MLXMetalTestLock.withLock {
            // Explicit BF16 bytes cover signed zero, subnormal/normal boundaries,
            // infinities and NaN payloads, without a CPU conversion/FTZ oracle.
            let phases: [UInt16] = [0x0000, 0x8000, 0x0001, 0x007f, 0x0080, 0x0081,
                0x3e80, 0x3f00, 0x3f01, 0x3f02, 0x3f7f, 0x3f80, 0x3f81,
                0x3f82, 0x3f83, 0xbe80, 0xbf00, 0xbf01, 0xbf02, 0xbf7f,
                0xbf80, 0xbf81, 0xbf82, 0xbf83, 0x3d80, 0xbd80, 0x3e00,
                0xbe00, 0x3f40, 0xbf40, 0x3f60, 0xbf60]
            let cosine = MLXArray(phases).view(dtype: .bfloat16).reshaped(1, 1, 1, 32)
            let sine = MLXArray(Array(phases.reversed())).view(dtype: .bfloat16).reshaped(1, 1, 1, 32)
            let count = 64 * 192
            for chunk in 0 ..< 16 {
                // All 65,536 words occur in the rotated prefix, not merely in
                // the pass-through suffix. Each prefix has 64 words/head.
                let words = (0 ..< count).map { element -> UInt16 in
                    let head = element / 192, feature = element % 192
                    return UInt16(truncatingIfNeeded: chunk * 4096 + head * 64 + feature % 64)
                }
                let x = MLXArray(words).view(dtype: .bfloat16).reshaped(1, 64, 1, 192)
                let fused = try XCTUnwrap(NaiveN05FusedRotaryApply.apply(x,
                    cosine: cosine, sine: sine, dimensions: 64))
                Self.assertStorage(fused, Self.stagedReference(x, cosine: cosine, sine: sine))
                Self.assertStorage(fused[.ellipsis, 64...], x[.ellipsis, 64...])
            }
            // Witness that FP32 products rounded only once are an invalid oracle:
            // BF16(1.0078125 * 0.50390625) == 0.5078125, hence low output +0.
            let a = MLXArray([UInt16](repeating: 0x3f81, count: 32))
                .view(dtype: .bfloat16).reshaped(1, 1, 1, 32)
            let b = MLXArray.ones(a.shape, dtype: .bfloat16)
            let x = concatenated([a, b, MLXArray.zeros([1, 1, 1, 64], dtype: .bfloat16)], axis: -1)
            let c = MLXArray([UInt16](repeating: 0x3f01, count: 32)).view(dtype: .bfloat16).reshaped(a.shape)
            let s = MLXArray([UInt16](repeating: 0x3f02, count: 32)).view(dtype: .bfloat16).reshaped(a.shape)
            let reference = Self.stagedReference(x, cosine: c, sine: s)
            let wrongLow = (a.asType(.float32) * c.asType(.float32)
                - b.asType(.float32) * s.asType(.float32)).asType(.bfloat16)
            XCTAssertNotEqual(reference[.ellipsis, ..<32].asData().data, wrongLow.asData().data)
            XCTAssertEqual(reference[.ellipsis, ..<32].view(dtype: .uint16).asArray(UInt16.self),
                [UInt16](repeating: 0, count: 32))
            Self.assertStorage(try XCTUnwrap(NaiveN05FusedRotaryApply.apply(x,
                cosine: c, sine: s, dimensions: 64)), reference)
        }
    }

    func testOriginalPrecisePhasesExplicitPositionsStridesAndLazyOwnership() throws {
        try MLXMetalTestLock.withLock {
            for (heads, width) in Self.roles {
                for strided in [false, true] {
                    let x = Self.input(heads: heads, width: width, strided: strided)
                    for position in [Int32(-3), 0, 1, 127, 8192, 10290, 16_777_217, Int32.max] {
                        let positions = MLXArray([position]).reshaped(1, 1)
                        let on = NaiveN05FlashMath.RotaryTables(positions: positions, fusedApply: true)
                        let off = NaiveN05FlashMath.RotaryTables(positions: positions)
                        for theta in [10_000.0, 10_000_000.0] {
                            let phases = on.phases(dimensions: 64, theta: theta, dtype: .bfloat16)
                            XCTAssertTrue(NaiveN05FusedRotaryApply.admits(x,
                                cosine: phases.0, sine: phases.1, dimensions: 64))
                            Self.assertStorage(NaiveN05FlashMath.rotary(x, positions: positions,
                                dimensions: 64, theta: theta, tables: on),
                                NaiveN05FlashMath.rotary(x, positions: positions,
                                dimensions: 64, theta: theta, tables: off))
                        }
                        XCTAssertEqual(on.count, 2)
                    }
                }
            }
            // Fractional supplied positions still produce the original precise
            // phases. Neither the apply kernel nor its admission inspects them.
            for position in [Float(-0.25), 1.5, 127.75] {
                let positions = MLXArray([position]).reshaped(1, 1)
                let x = Self.input(heads: 16, width: 128, strided: true)
                Self.assertStorage(NaiveN05FlashMath.rotary(x, positions: positions, dimensions: 64,
                    theta: 10_000_000, tables: .init(positions: positions, fusedApply: true)),
                    NaiveN05FlashMath.rotary(x, positions: positions, dimensions: 64,
                    theta: 10_000_000, tables: .init(positions: positions)))
            }
            let phaseBase = MLXArray((0 ..< 64).map { Float($0 - 32) / 64 })
                .reshaped(32, 2).asType(.bfloat16)
            let cosine = phaseBase[0..., 0].reshaped(1, 1, 1, 32)
            let sine = phaseBase[0..., 1].reshaped(1, 1, 1, 32)
            let x = Self.input(heads: 8, width: 192, strided: true)
            Self.assertStorage(try XCTUnwrap(NaiveN05FusedRotaryApply.apply(x,
                cosine: cosine, sine: sine, dimensions: 64)),
                Self.stagedReference(x, cosine: cosine, sine: sine))

            func delayed(_ enabled: Bool, position: Int32) -> MLXArray {
                let x = Self.input(heads: 64, width: 192, strided: true)
                let positions = MLXArray([position]).reshaped(1, 1)
                let tables = NaiveN05FlashMath.RotaryTables(positions: positions, fusedApply: enabled)
                return NaiveN05FlashMath.rotary(x, positions: positions,
                    dimensions: 64, theta: 10_000_000, tables: tables)
            }
            // No eval/readback occurs before inputs and the forward-local tables
            // leave scope. A second position cannot reuse the first lazy phases.
            let first = delayed(true, position: 8192)
            let second = delayed(true, position: 10290)
            Self.assertStorage(first, delayed(false, position: 8192))
            Self.assertStorage(second, delayed(false, position: 10290))
            XCTAssertNotEqual(first.asData().data, second.asData().data)
            let positions = MLXArray([Int32(9)]).reshaped(1, 1)
            XCTAssertEqual(NaiveN05FlashMath.RotaryTables(positions: positions, fusedApply: true).count, 0)
        }
    }

    func testSerialRuntimeChunkDecodeAndDiskRestorePreserveAllStorage() throws {
        try MLXMetalTestLock.withLock {
            for precision in ["fp8_e4m3", "bf16"] {
                let (baseline, candidate) = try Self.models(indexerPrecision: precision)
                let expected = baseline.newCache(), actual = candidate.newCache()
                let first = LMInput.Text(tokens: MLXArray([1]))
                Self.assertStorage(candidate(first, cache: nil, state: nil).logits,
                    baseline(first, cache: nil, state: nil).logits)
                let tokens = MLXArray([1, 2, 3, 4, 5])
                // The runtime consumes two chunks, then passes its one-token tail to
                // generation with nil padding/positions. Explicit all-true input
                // masks are admitted and normalized by the real runtime bridge.
                guard case .tokens(let a) = try candidate.prepare(
                    LMInput(tokens: tokens, mask: MLXArray.ones([5], dtype: .bool)),
                    cache: actual, windowSize: 2),
                    case .tokens(let b) = try baseline.prepare(
                        LMInput(tokens: tokens, mask: MLXArray.ones([5], dtype: .bool)),
                        cache: expected, windowSize: 2)
                else { return XCTFail("Expected one-token prompt tails") }
                XCTAssertEqual(a.tokens.shape, [1])
                Self.assertStorage(a.tokens, b.tokens)
                Self.assertCaches(actual, expected)
                XCTAssertEqual(actual.map(\.offset), [4, 4])
                Self.assertStorage(candidate(a, cache: actual, state: nil).logits,
                    baseline(b, cache: expected, state: nil).logits)
                Self.assertCaches(actual, expected)
                for values in [[6], [7, 8, 9], [10], [11]] {
                    Self.assertStorage(try candidate.replayForward(MLXArray(values), cache: actual),
                        try baseline.replayForward(MLXArray(values), cache: expected))
                    Self.assertCaches(actual, expected)
                }
                XCTAssertEqual(actual.map(\.offset), [11, 11])
                XCTAssertEqual(actual[0].state.map { $0.dim(2) }, [11, 11, 11])
                XCTAssertEqual(actual[0].state[2].dtype, .float32)
                XCTAssertEqual(actual[1].state.map { $0.dim(2) }, [2, 2])

                let path = FileManager.default.temporaryDirectory
                    .appendingPathComponent("naive-fused-rotary-\(UUID().uuidString).safetensors")
                defer { try? FileManager.default.removeItem(at: path) }
                try MLX.save(arrays: TQDiskSerializer.serialize(cache: actual), url: path)
                let disk = try MLX.loadArrays(url: path)
                // The same exact serialized tuple may continue under either policy.
                var restoredOn: [KVCache] = candidate.newCache()
                var restoredOff: [KVCache] = baseline.newCache()
                XCTAssertEqual(restoreFromDiskArrays(disk, into: &restoredOn, requirePromptBoundary: true), 11)
                XCTAssertEqual(restoreFromDiskArrays(disk, into: &restoredOff, requirePromptBoundary: true), 11)
                let typedOn = try XCTUnwrap(restoredOn as? [NaiveN05FlashCache])
                let typedOff = try XCTUnwrap(restoredOff as? [NaiveN05FlashCache])
                Self.assertCaches(typedOn, actual)
                Self.assertCaches(typedOff, expected)
                XCTAssertTrue(validateRestoredCacheBoundary(restoredOn, matchedTokens: 11, restoredTokens: 11))
                XCTAssertTrue(validateRestoredCacheBoundary(restoredOff, matchedTokens: 11, restoredTokens: 11))
                let next = MLXArray([12])
                let reference = try baseline.replayForward(next, cache: expected)
                Self.assertStorage(try candidate.replayForward(next, cache: actual), reference)
                Self.assertStorage(try candidate.replayForward(next, cache: typedOn), reference)
                Self.assertStorage(try baseline.replayForward(next, cache: typedOff), reference)
                Self.assertCaches(actual, expected)
                Self.assertCaches(typedOn, expected)
                Self.assertCaches(typedOff, expected)
            }
        }
    }

    func testExplicitPaddingPositionsBatchAndMultiqueryKeepReferenceStorage() throws {
        try MLXMetalTestLock.withLock {
            let (baseline, candidate) = try Self.models()
            let expected = baseline.newCache(), actual = candidate.newCache()
            for (values, paddingValues, positionValues) in [
                ([1, 2, 3], [false, true, true] as [Bool]?, nil as [Int32]?),
                ([4], [false, true, true, true], nil),
                ([5], nil, [99]),
                ([6], [true, true, true, true, true, true], [16_777_217]),
            ] {
                let tokens = MLXArray(values).reshaped(1, -1)
                let padding = paddingValues.map { MLXArray($0).reshaped(1, -1) }
                let positions = positionValues.map { MLXArray($0).reshaped(1, -1) }
                Self.assertStorage(try candidate(tokens, padding: padding, positions: positions, cache: actual),
                    try baseline(tokens, padding: padding, positions: positions, cache: expected))
                Self.assertCaches(actual, expected)
            }
            for (length, paddingValues, positionValues) in [
                (1, nil as [Bool]?, nil as [Int32]?),
                (3, [false, true, true, true, false, true], nil),
                (3, nil, [0, 1, 5, 7, -1, 2]),
            ] {
                let tokens = MLXArray((0 ..< 2 * length).map { $0 + 1 }).reshaped(2, length)
                let padding = paddingValues.map { MLXArray($0).reshaped(2, length) }
                let positions = positionValues.map { MLXArray($0).reshaped(2, length) }
                let actual = candidate.newCache(), expected = baseline.newCache()
                Self.assertStorage(try candidate(tokens, padding: padding, positions: positions, cache: actual),
                    try baseline(tokens, padding: padding, positions: positions, cache: expected))
                Self.assertCaches(actual, expected)
            }
        }
    }

    func testLateAppendFailureAndMisalignedCachesKeepNextPositionAndStorage() throws {
        try MLXMetalTestLock.withLock {
            let (baseline, candidate) = try Self.models()
            let expected = baseline.newCache(), actual = candidate.newCache()
            Self.assertStorage(try candidate.replayForward(MLXArray([1, 2, 3, 4]), cache: actual),
                try baseline.replayForward(MLXArray([1, 2, 3, 4]), cache: expected))
            Self.assertCaches(actual, expected)
            let originalOn = actual.map(Snapshot.init), originalOff = expected.map(Snapshot.init)
            for (model, cache) in [(candidate, actual), (baseline, expected)] {
                let late = cache[1]
                XCTAssertTrue(late.restoreDiskCacheState(late.state.map { $0.asType(.float16) },
                    metadata: late.metaState, offset: late.offset))
                let before = cache.map(Snapshot.init)
                // runtimeCache admits this topology; layer zero appends, then
                // layer one's dtype mismatch throws. runtimeForward restores all.
                XCTAssertThrowsError(try model.replayForward(MLXArray([5]), cache: cache)) {
                    XCTAssertTrue($0 is NaiveN05FlashCache.Failure)
                }
                Self.assertUnchanged(cache, before)
                XCTAssertEqual(cache.map(\.offset), [4, 4])
            }
            for (cache, snapshot) in [(actual, originalOn), (expected, originalOff)] {
                for (entry, saved) in zip(cache, snapshot) {
                    XCTAssertTrue(entry.restoreDiskCacheState(saved.rows,
                        metadata: saved.metadata, offset: saved.offset))
                }
            }
            Self.assertStorage(try candidate.replayForward(MLXArray([5]), cache: actual),
                try baseline.replayForward(MLXArray([5]), cache: expected))
            Self.assertCaches(actual, expected)
            for (model, cache) in [(candidate, actual), (baseline, expected)] {
                XCTAssertEqual(cache[0].trim(1), 1)
                let before = cache.map(Snapshot.init)
                XCTAssertThrowsError(try model.replayForward(MLXArray([6]), cache: cache))
                // The direct graph's alignment guard also precedes construction.
                XCTAssertThrowsError(try model(MLXArray([6]).reshaped(1, 1), cache: cache))
                Self.assertUnchanged(cache, before)
            }
        }
    }

    func testCancellationBeforeSerialForwardPreservesCompleteCompanionTuple() async throws {
        try await Task.detached {
            try MLXMetalTestLock.withLock {
                let (baseline, candidate) = try Self.models()
                let expected = baseline.newCache(), actual = candidate.newCache()
                Self.assertStorage(try candidate.replayForward(MLXArray([1, 2, 3, 4]), cache: actual),
                    try baseline.replayForward(MLXArray([1, 2, 3, 4]), cache: expected))
                Self.assertCaches(actual, expected)
                let beforeOn = actual.map(Snapshot.init), beforeOff = expected.map(Snapshot.init)
                // Deterministic current-task cancellation; no sleeps or race.
                withUnsafeCurrentTask { task in
                    XCTAssertNotNil(task)
                    task?.cancel()
                }
                for (model, cache, before) in [(candidate, actual, beforeOn), (baseline, expected, beforeOff)] {
                    XCTAssertThrowsError(try model.replayForward(MLXArray([5]), cache: cache)) {
                        XCTAssertTrue($0 is CancellationError)
                    }
                    XCTAssertThrowsError(try model.prepare(LMInput(tokens: MLXArray([5, 6, 7])),
                        cache: cache, windowSize: 1)) { XCTAssertTrue($0 is CancellationError) }
                    Self.assertUnchanged(cache, before)
                    XCTAssertEqual(cache.map(\.offset), [4, 4])
                }
                Self.assertCaches(actual, expected)
            }
        }.value
    }
}
