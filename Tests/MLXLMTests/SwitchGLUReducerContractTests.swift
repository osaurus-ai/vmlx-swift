import MLX
import MLXNN
import XCTest

@testable import MLXLMCommon

final class SwitchGLUReducerContractTests: XCTestCase {
    private let inputs = 2048, hidden = 512, experts = 8

    private func install(_ routed: SwitchGLU) throws {
        func projection(_ input: Int, _ output: Int, _ bias: Float) -> QuantizedSwitchLinear {
            QuantizedSwitchLinear(
                inputDims: input, outputDims: output, numExperts: experts,
                weight: MLXArray.zeros([experts, output, input / 8], dtype: .uint32),
                scales: MLXArray.zeros([experts, output, input / 64], dtype: .bfloat16),
                biases: MLXArray.full([experts, output, input / 64], values: MLXArray(bias)).asType(.bfloat16),
                groupSize: 64, bits: 4, mode: .affine)
        }
        try routed.update(modules: ModuleChildren.unflattened([
            ("gate_proj", projection(inputs, hidden, 1) as Module),
            ("up_proj", projection(inputs, hidden, 2) as Module),
            ("down_proj", projection(hidden, inputs, 1 / Float(hidden)) as Module),
        ]), verify: .all)
    }

    private func reduce(_ routed: SwitchGLU) -> MLXArray? {
        var values = [Float](repeating: 0, count: inputs)
        values[0] = 1
        return routed.qwen4ExpReduced(
            MLXArray(values, [1, 1, inputs]).asType(.bfloat16),
            indices: MLXArray((0 ..< experts).map(UInt32.init), [1, 1, experts]),
            scores: MLXArray.full([1, 1, experts], values: MLXArray(Float(1) / Float(experts))).asType(.bfloat16))
    }

    func testUnknownMathRefusesQualifiedReducer() throws {
        try MLXMetalTestLock.withLock {
            let makers: [() -> SwitchGLU] = [
                { SwitchGLU(inputDims: self.inputs, hiddenDims: self.hidden, numExperts: self.experts,
                            activation: .custom { $0 * 3 }) },
                { SwitchGLU(inputDims: self.inputs, hiddenDims: self.hidden, numExperts: self.experts,
                            activation: .geluApproximate) },
                { SwitchGLU(inputDims: self.inputs, hiddenDims: self.hidden, numExperts: self.experts,
                            glue: { $0 + $1 }) },
                { SwitchGLU(inputDims: self.inputs, hiddenDims: self.hidden, numExperts: self.experts,
                            scoredGlue: { gate, up, scores in gate * up * scores }) },
                { SwitchGLU(inputDims: self.inputs, hiddenDims: self.hidden, numExperts: self.experts,
                            activation: .swiGLU(limit: 10), glue: { $0 - $1 }) },
                { SwitchGLU(inputDims: self.inputs, hiddenDims: self.hidden, numExperts: self.experts,
                            swigluLimit: 10) },
                { SwitchGLU(inputDims: self.inputs, hiddenDims: self.hidden, numExperts: self.experts,
                            activation: .swiGLU(limit: 10), swigluLimit: 5) },
            ]
            for make in makers {
                let routed = make()
                try install(routed)
                XCTAssertNil(reduce(routed), "Shape eligibility must not authorize unknown math")
            }
        }
    }

    func testKnownPlainAndTypedSwiGLURetainReducer() throws {
        try MLXMetalTestLock.withLock {
            for activation: SwitchGLUActivation in [.silu, .swiGLU(limit: nil), .swiGLU(limit: 10)] {
                let routed = SwitchGLU(inputDims: inputs, hiddenDims: hidden, numExperts: experts,
                                       activation: activation)
                try install(routed)
                let output = try XCTUnwrap(reduce(routed))
                XCTAssertEqual(output.dtype, .bfloat16)
                let values = output.asType(.float32).asArray(Float.self)
                XCTAssertEqual(values.count, inputs)
                // Explicit bank above yields g=1,u=2, and the down rows average hidden channels.
                for value in values { XCTAssertEqual(value, 1.4621172, accuracy: 0.02) }
            }
        }
    }

    func testTypedGlueKeepsAsymmetricClampAndOpaquePrecedence() throws {
        try MLXMetalTestLock.withLock {
            let g = MLXArray([Float(-25), 25, 1, -1])
            let u = MLXArray([Float(1), 1, 400, -400])
            for dtype in [DType.float32, .float16, .bfloat16] {
                let typed = SwitchGLU(inputDims: 4, hiddenDims: 4, numExperts: 1,
                                      activation: .swiGLU(limit: 10))
                let output = try XCTUnwrap(typed.glue)(g.asType(dtype), u.asType(dtype))
                XCTAssertEqual(output.dtype, dtype)
                let values = output.asType(.float32).asArray(Float.self)
                XCTAssertEqual(values[0], -3.471986e-10, accuracy: dtype == .float16 ? 1e-7 : 1e-11)
                XCTAssertEqual(values[1], 9.999546, accuracy: 0.05)
                XCTAssertEqual(values[2], 7.310586, accuracy: 0.05)
                XCTAssertEqual(values[3], 2.689414, accuracy: 0.05)
            }
            let opaque = SwitchGLU(inputDims: 4, hiddenDims: 4, numExperts: 1,
                                   activation: .swiGLU(limit: 10), glue: { $0 + $1 })
            XCTAssertEqual(try XCTUnwrap(opaque.glue)(g, u).asArray(Float.self), [-24, 26, 401, -401])
        }
    }
}
