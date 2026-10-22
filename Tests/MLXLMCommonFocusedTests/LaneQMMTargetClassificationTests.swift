import MLX
import MLXNN
import XCTest

@testable import MLXLMCommon

// Custom routed banks deliberately keep their mmap tensors outside Module.parameters.
// The classifier must not infer lane coverage from an empty parameter dictionary.
private final class OpaqueRoutedBank: Module, WeightedRoutedExpertLayer {
    func callRouted(_ input: MLXArray, indices: MLXArray, scores: MLXArray) -> MLXArray {
        input
    }
}

private final class DenseLaneTarget: Module {
    @ModuleInfo(key: "proj") var proj: QuantizedLinear
    init(_ projection: QuantizedLinear) { _proj.wrappedValue = projection }
}

private final class RoutedLaneTarget: Module {
    @ModuleInfo(key: "proj") var proj: QuantizedLinear
    @ModuleInfo(key: "experts") var experts: OpaqueRoutedBank
    init(_ projection: QuantizedLinear) {
        _proj.wrappedValue = projection
        _experts.wrappedValue = OpaqueRoutedBank()
    }
}

final class LaneQMMTargetClassificationTests: XCTestCase {
    func testOpaqueRoutedBankKeepsNarrowAdaptiveCandidate() throws {
        try FocusedMLXTestSupport.withLock {
            guard LaneQMM.available() else { throw XCTSkip("no Metal-4 tensor ops") }
            func projection() -> QuantizedLinear {
                let (w, s, b) = MLX.quantized(
                    MLXArray.zeros([256, 1024], dtype: .bfloat16), groupSize: 64, bits: 4)
                return QuantizedLinear(weight: w, scales: s, biases: b!, groupSize: 64, bits: 4)
            }
            // Retain both targets together so process-wide installation identities cannot be reused.
            let dense = DenseLaneTarget(projection())
            let routed = RoutedLaneTarget(projection())
            XCTAssertTrue(routed.experts.parameters().flattenedValues().isEmpty)
            LaneQMM.installForDFlash2Target(dense)
            LaneQMM.installForDFlash2Target(routed)
            XCTAssertTrue(dense.proj is LaneQuantizedLinear)
            XCTAssertTrue(routed.proj is LaneQuantizedLinear)
            XCTAssertTrue(LaneQMM.isLaneFlat(dense), "Fully lane-backed dense control stays flat")
            XCTAssertFalse(LaneQMM.isLaneFlat(routed), "Opaque routed banks do not execute on the lane")
            XCTAssertEqual(
                DFlash2WidthChooser.plan(trained: 16, laneFlat: LaneQMM.isLaneFlat(routed)),
                [1, 5, 16, 32])
        }
    }
}

// Mirrors the Qwen3.5/MiniCPM compiled shared expert: it consumes the
// projection's raw packed weights rather than invoking its module override.
private final class SharedExpertProjection: Module {
    @ModuleInfo(key: "gate_proj") var projection: QuantizedLinear
    init(_ projection: QuantizedLinear) { _projection.wrappedValue = projection }
}
private final class SharedExpertMLP: Module {
    @ModuleInfo(key: "shared_expert") var shared: SharedExpertProjection
    init(_ projection: QuantizedLinear) { _shared.wrappedValue = SharedExpertProjection(projection) }
}
private final class RawSharedExpertReader: Module {
    @ModuleInfo(key: "mlp") var mlp: SharedExpertMLP
    @ModuleInfo(key: "attention") var attention: QuantizedLinear
    var projection: QuantizedLinear { mlp.shared.projection }

    init(_ projection: QuantizedLinear, _ attention: QuantizedLinear) {
        _mlp.wrappedValue = SharedExpertMLP(projection)
        _attention.wrappedValue = attention
    }

    func rawForward(_ input: MLXArray) -> MLXArray {
        quantizedMM(input, projection.weight, scales: projection.scales,
                    biases: projection.biases, transpose: true,
                    groupSize: projection.groupSize, bits: projection.bits)
    }
}

extension LaneQMMTargetClassificationTests {
    func testInstallingLanePreservesRawSharedExpertWeightLayout() throws {
        try FocusedMLXTestSupport.withLock {
            guard LaneQMM.available() else { throw XCTSkip("no Metal-4 tensor ops") }
            func projection() -> QuantizedLinear {
                // Nonuniform deterministic values expose a tile permutation;
                // zero weights would let the old bug pass.
                let data: [Float] = (0..<(256 * 256)).map { index in
                    let value = (index * 17 + index / 256) % 113 - 56
                    return Float(value) / Float(113)
                }
                let values = MLXArray(data).reshaped(256, 256).asType(.bfloat16)
                let (w, s, b) = MLX.quantized(values, groupSize: 64, bits: 8)
                return QuantizedLinear(weight: w, scales: s, biases: b!, groupSize: 64, bits: 8)
            }
            let target = RawSharedExpertReader(projection(), projection())
            let inputData: [Float] = (0..<256).map { Float($0 % 19 - 9) / Float(19) }
            let input = MLXArray(inputData).reshaped(1, 256).asType(.bfloat16)
            let expected = target.rawForward(input)
            eval(expected)
            LaneQMM.installForDFlash2Target(target)
            XCTAssertTrue(target.projection is LaneQuantizedLinear,
                          "Keep accelerated multi-row dispatch; preserve only its raw weight layout")
            XCTAssertTrue(target.attention is LaneQuantizedLinear)
            XCTAssertTrue(arrayEqual(expected, target.rawForward(input)).item(Bool.self),
                          "A fused raw reader must see the original MLX packing after lane installation")
        }
    }
}
