// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

#if os(Linux)
    import Foundation
    import MLX
    import Testing

    /// A float32 function MLX evaluates with its own polynomial, and its Double reference.
    struct PolynomialCase: CustomStringConvertible, Sendable {
        let name: String
        let inputs: @Sendable () -> [Float]
        let apply: @Sendable (MLXArray) -> MLXArray
        let reference: @Sendable (Double) -> Double
        /// The absolute floor of a unit (`unitsError`). erf is accurate near 0 only in absolute terms
        /// (about 1e-7); at 2^-33 its worst "unit" would be that rounding noise, over a thousand units
        /// that move between correct evaluations, and 2 units of margin would mean nothing.
        var floor: Double = 0x1p-33
        var description: String { name }
    }

    // MLX's exp maps x > 88 to +inf, before float32's true limit near 88.72: MLX's own behaviour
    // on every path, and outside this grid.
    let polynomialCases: [PolynomialCase] = [
        PolynomialCase(
            name: "exp",
            inputs: { grid(-87, 88, count: 200_000, seed: 1, extras: [0, 1, -1, 88]) },
            apply: { MLX.exp($0) }, reference: { Foundation.exp($0) }),
        PolynomialCase(
            name: "sin", inputs: { grid(-300, 300, count: 200_000, seed: 2, extras: [.pi, -.pi]) },
            apply: { MLX.sin($0) }, reference: { Foundation.sin($0) }),
        PolynomialCase(
            name: "cos", inputs: { grid(-300, 300, count: 200_000, seed: 3, extras: [.pi / 2]) },
            apply: { MLX.cos($0) }, reference: { Foundation.cos($0) }),
        PolynomialCase(
            name: "sinWide", inputs: { grid(-10000, 10000, count: 200_000, seed: 8) },
            apply: { MLX.sin($0) }, reference: { Foundation.sin($0) }),
        PolynomialCase(
            name: "cosWide", inputs: { grid(-10000, 10000, count: 200_000, seed: 9) },
            apply: { MLX.cos($0) }, reference: { Foundation.cos($0) }),
        PolynomialCase(
            name: "erf",
            inputs: { grid(-5, 5, count: 200_000, seed: 4, extras: [0, 1e-6, -1e-6]) },
            apply: { MLX.erf($0) }, reference: { Foundation.erf($0) }, floor: 0x1p-24),
        PolynomialCase(
            name: "erfInverse",
            inputs: { grid(-0.99999, 0.99999, count: 200_000, seed: 5, extras: [0, 0.5]) },
            apply: { MLX.erfInverse($0) }, reference: erfInverseReference),
        PolynomialCase(
            name: "sigmoid", inputs: { grid(-50, 50, count: 200_000, seed: 6, extras: [0]) },
            apply: { MLX.sigmoid($0) }, reference: { 1 / (1 + Foundation.exp(-$0)) }),
    ]

    /// The absolute floor of a unit (`unitsError`) that `name`'s baseline was measured with.
    func polynomialFloor(_ name: String) -> Double {
        polynomialCases.first { $0.name == name }?.floor ?? 0x1p-33
    }

    @Suite(.serialized) struct PolynomialAccuracyTests {
        static var measuring: Bool {
            ProcessInfo.processInfo.environment["VMLX_ULP_BASELINE"] == "1"
        }

        /// Judges `measured` against the baseline. Measuring mode only prints it, and records an issue,
        /// so that a measuring run can never pass for a judged one.
        static func judge(_ name: String, _ measured: Double, worst: String) throws {
            if measuring {
                print("ULP_BASELINE \(name) \(measured)")
                print("  worst \(worst)")
                Issue.record(
                    "VMLX_ULP_BASELINE=1 measures \(name) (\(measured) units), and judges nothing")
                return
            }
            let baseline = try #require(
                ULPBaseline.units[name],
                "no baseline for \(name): measure one with VMLX_ULP_BASELINE=1 (see ULPBaseline)")
            #expect(
                measured <= baseline + 2,
                "\(name): \(measured) units against a baseline of \(baseline), \(worst)")
        }

        @Test(arguments: polynomialCases)
        func staysWithinTwoUnitsOfTheScalarCode(_ c: PolynomialCase) throws {
            try KernelLock.run {
                let inputs = c.inputs()
                let got = c.apply(MLXArray(inputs, [inputs.count])).asArray(Float.self)
                var worst = (units: 0.0, at: "")
                for (x, y) in zip(inputs, got) {
                    let truth = c.reference(Double(x))
                    let units = unitsError(y, truth, floor: c.floor)
                    if units > worst.units {
                        worst = (units, "at x = \(x): \(y), float64 gives \(truth)")
                    }
                }
                try Self.judge(c.name, worst.units, worst: worst.at)
            }
        }

        @Test func logAddExpStaysWithinTwoUnitsOfTheScalarCode() throws {
            try KernelLock.run {
                let side = grid(-50, 50, count: 500, seed: 7)
                let a = side.flatMap { x in side.map { _ in x } }
                let b = side.flatMap { _ in side }
                let got = MLX.logAddExp(MLXArray(a, [a.count]), MLXArray(b, [b.count])).asArray(
                    Float.self)
                var worst = (units: 0.0, at: "")
                for i in 0 ..< a.count {
                    let (x, y) = (Double(a[i]), Double(b[i]))
                    let truth = max(x, y) + Foundation.log1p(Foundation.exp(-abs(x - y)))
                    let units = unitsError(got[i], truth)
                    if units > worst.units {
                        worst = (units, "at (\(a[i]), \(b[i])): \(got[i]), float64 gives \(truth)")
                    }
                }
                try Self.judge("logAddExp", worst.units, worst: worst.at)
            }
        }
    }
#endif
