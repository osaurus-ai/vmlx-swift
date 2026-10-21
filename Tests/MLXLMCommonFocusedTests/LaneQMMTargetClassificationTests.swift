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
