// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// LaneQMM.installForDFlash2Drafter: opt-in, idempotent, and numerically the quantized projection.

import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLMCommon

final class LaneQMMDrafterInstallTests: XCTestCase {

    private final class TinyDrafter: Module {
        @ModuleInfo(key: "up") var up: Linear
        @ModuleInfo(key: "down") var down: Linear
        override init() {
            _up.wrappedValue = Linear(256, 512, bias: false)
            _down.wrappedValue = Linear(512, 256, bias: false)
        }
        func callAsFunction(_ x: MLXArray) -> MLXArray { down(up(x)) }
    }

    private func quantizedDrafter() -> TinyDrafter {
        MLXRandom.seed(17)
        let model = TinyDrafter()
        let weights = model.parameters().flattened().map { ($0.0, 0.05 * MLXRandom.normal($0.1.shape)) }
        try! model.update(parameters: ModuleParameters.unflattened(Dictionary(uniqueKeysWithValues: weights)), verify: [.all])
        model.apply { $0.asType(.bfloat16) }
        quantize(model: model, groupSize: 64, bits: 4)
        eval(model)
        return model
    }

    private func laneCount(_ model: Module) -> Int {
        model.leafModules().flattened().filter { $0.1 is LaneQuantizedLinear }.count
    }

    func testDefaultLeavesTheDrafterOnMLX() {
        unsetenv("VMLX_LANE_QMM_DRAFTER")
        let model = quantizedDrafter()
        LaneQMM.installForDFlash2Drafter(model)
        XCTAssertEqual(laneCount(model), 0)
    }

    func testOptInRoutesProjectionsThroughLaneAndKeepsTheirOutput() throws {
        guard LaneQMM.available() else { throw XCTSkip("lane matmul unavailable on this device") }
        setenv("VMLX_LANE_QMM_DRAFTER", "1", 1)
        defer { unsetenv("VMLX_LANE_QMM_DRAFTER") }
        let model = quantizedDrafter()
        var references: [Int: MLXArray] = [:]
        for rows in [8, 16] {
            MLXRandom.seed(UInt64(rows))
            let x = MLXRandom.normal([rows, 256]).asType(.bfloat16)
            references[rows] = model(x)
            eval(references[rows]!)
        }
        LaneQMM.installForDFlash2Drafter(model)
        XCTAssertEqual(laneCount(model), 2)
        LaneQMM.installForDFlash2Drafter(model)  // idempotent
        XCTAssertEqual(laneCount(model), 2)
        for rows in [8, 16] {
            MLXRandom.seed(UInt64(rows))
            let x = MLXRandom.normal([rows, 256]).asType(.bfloat16)
            let lane = model(x).asType(.float32)
            let reference = references[rows]!.asType(.float32)
            // Lane accumulates in a different order than MLX's qmm (fp32 inside, bf16 out): compare
            // relative to the output scale, a few bf16 ulps.
            let scale = abs(reference).max().item(Float.self)
            let diff = abs(lane - reference).max().item(Float.self)
            XCTAssertLessThan(diff, scale * 0.02, "rows \(rows): max diff \(diff) vs scale \(scale)")
        }
    }
}
