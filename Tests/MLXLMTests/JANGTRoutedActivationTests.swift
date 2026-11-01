// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import XCTest

@testable import MLXLMCommon

/// The routed-expert layer with a non-SiLU activation over JANGH codebook banks. The JANGH fused gate/up epilogues are
/// SiLU only, so a SwiGLU-OAI model (MiniMax-M3 / M3.1) runs codebook gate and up as single-weight passes and applies
/// its variant in F32, in decode and in the sorted prefill. Reference: the banks' own codes and levels in Double, with
/// the Hadamard-32 input basis written out. Control: the same banks with the SiLU activation miss it.
final class JANGTRoutedActivationTests: XCTestCase {
    static let hidden = 256, inter = 128, experts = 4, routes = 2
    static let limit: Float = 1, alpha: Float = 1.702
    static let gateBits = 2, downBits = 4

    struct Rng {
        var state: UInt64
        mutating func next() -> UInt64 {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            return state
        }
        mutating func word() -> UInt32 { UInt32(truncatingIfNeeded: next() >> 17) }
        mutating func uniform() -> Float { Float(next() >> 40) / Float(1 << 24) }
        mutating func normal() -> Float {
            let u = max(uniform(), 1e-7), v = uniform()
            return sqrt(-2 * log(u)) * cos(2 * .pi * v)
        }
    }

    /// odd-cubic codebook levels u * (alpha + beta * u^2), u = q - (2^bits - 1) / 2.
    static func levels(bits: Int, alpha: Double) -> [Double] {
        (0 ..< (1 << bits)).map { q in (Double(q) - Double((1 << bits) - 1) / 2) * alpha }
    }

    struct Bank {
        let bits: Int, outputs: Int, inputs: Int
        let packed: [UInt32], scales: [Float16]
        var words: Int { inputs * bits / 32 }

        func weight(_ e: Int, _ n: Int, _ c: Int, _ book: [Double]) -> Double {
            let bit = c * bits
            let code = (packed[(e * outputs + n) * words + bit >> 5] >> UInt32(bit & 31)) & UInt32((1 << bits) - 1)
            return book[Int(code)] * Double(scales[e * outputs + n])
        }
    }

    static func h32(_ v: [Double]) -> [Double] {
        var x = v
        for block in stride(from: 0, to: x.count, by: 32) {
            var h = 1
            while h < 32 {
                for i in stride(from: block, to: block + 32, by: 2 * h) {
                    for j in i ..< i + h {
                        (x[j], x[j + h]) = (x[j] + x[j + h], x[j] - x[j + h])
                    }
                }
                h *= 2
            }
        }
        return x.map { $0 / 32.0.squareRoot() }
    }

    /// A one-layer JANGH bundle (codebook gate / up / down) at `model.layers.0.mlp.switch_mlp`, aligned shard + index.
    static func bundle(_ directory: URL, banks: [String: Bank]) throws {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        var header: [String: Any] = [:], payload = Data(), index: [String: String] = [:]
        let file = "model-00001-of-00001.safetensors"
        func put(_ name: String, _ dtype: String, _ shape: [Int], _ data: Data) {
            while payload.count % 64 != 0 { payload.append(0) }
            header[name] = ["dtype": dtype, "shape": shape, "data_offsets": [payload.count, payload.count + data.count]]
            payload.append(data)
            index[name] = file
        }
        var plan: [String: Any] = ["group_size": 64, "bits": 8]
        for (role, b) in banks.sorted(by: { $0.key < $1.key }) {
            let module = "model.layers.0.mlp.switch_mlp.\(role)"
            put(module + ".tq2_packed", "U32", [experts, b.outputs, b.words], b.packed.withUnsafeBytes { Data($0) })
            put(module + ".tq2_scales", "F16", [experts, b.outputs], b.scales.withUnsafeBytes { Data($0) })
            plan[module] = ["mode": "jangtq2", "bits": b.bits, "rotation": "hadamard32"]
        }
        var encoded = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
        encoded.append(Data(repeating: 0x20, count: ((encoded.count + 8 + 4095) / 4096) * 4096 - 8 - encoded.count))
        var length = UInt64(encoded.count).littleEndian
        var out = withUnsafeBytes(of: &length) { Data($0) }
        out.append(encoded)
        out.append(payload)
        try out.write(to: directory.appendingPathComponent(file))
        try JSONSerialization.data(withJSONObject: ["weight_map": index]).write(
            to: directory.appendingPathComponent("model.safetensors.index.json"))
        let books: [String: Any] = Dictionary(uniqueKeysWithValues: [gateBits, downBits].map { bits in
            (String(bits), ["alpha": 0.5, "beta": 0.0, "levels": levels(bits: bits, alpha: 0.5)] as [String: Any])
        })
        let config: [String: Any] = [
            "model_type": "minimax_m3_vl", "quantization": plan,
            "jangtq": ["version": 2, "packing": "lsb-bitstream", "scale_dtype": "float16", "rotation": "hadamard32",
                       "codebook_family": "odd-cubic", "codebooks": books],
        ]
        try JSONSerialization.data(withJSONObject: config).write(to: directory.appendingPathComponent("config.json"))
    }

    func testCodebookGateUpRunsTheModelActivation() throws {
        let h = Self.hidden, i = Self.inter, e = Self.experts, k = Self.routes
        var rng = Rng(state: 17)
        func bank(bits: Int, outputs: Int, inputs: Int, scale: Float) -> Bank {
            Bank(bits: bits, outputs: outputs, inputs: inputs,
                 packed: (0 ..< e * outputs * inputs * bits / 32).map { _ in rng.word() },
                 scales: (0 ..< e * outputs).map { _ in Float16(scale * (0.5 + rng.uniform())) })
        }
        let banks = [
            "gate_proj": bank(bits: Self.gateBits, outputs: i, inputs: h, scale: 0.15),
            "up_proj": bank(bits: Self.gateBits, outputs: i, inputs: h, scale: 0.15),
            "down_proj": bank(bits: Self.downBits, outputs: h, inputs: i, scale: 0.02),
        ]
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("jangt-activation-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: directory) }
        try Self.bundle(directory, banks: banks)
        let preparation = try JANGHModelPreparation(
            directory: directory, configuration: Data(contentsOf: directory.appendingPathComponent("config.json")),
            sidecar: nil,
            layout: .init(modelType: "minimax_m3_vl", hiddenSize: h, intermediateSize: i, expertCount: e,
                          sparseLayers: [0], routesPerToken: k))
        let oai = try XCTUnwrap(
            preparation.makeRoutedExperts(activationLimit: Self.limit, activation: .oai(alpha: Self.alpha))[0])
        // The JANGH-only layer of a SiLU model keeps its own layer; a non-SiLU model needs the variant-aware one.
        XCTAssertTrue(oai is JANGTRoutedExpertLayer)
        let silu = try XCTUnwrap(preparation.makeRoutedExperts(activationLimit: Self.limit)[0])
        XCTAssertTrue(silu is JANGHRoutedExpertLayer)

        // Dense (E, N, K) weights decoded from the banks' own codes, levels and scales.
        func dense(_ b: Bank) -> [Double] {
            let book = Self.levels(bits: b.bits, alpha: 0.5)
            var out = [Double](repeating: 0, count: e * b.outputs * b.inputs)
            for ex in 0 ..< e {
                for n in 0 ..< b.outputs {
                    for c in 0 ..< b.inputs { out[(ex * b.outputs + n) * b.inputs + c] = b.weight(ex, n, c, book) }
                }
            }
            return out
        }
        let wg = dense(banks["gate_proj"]!), wu = dense(banks["up_proj"]!), wd = dense(banks["down_proj"]!)
        func reference(x: [Float], tokens: Int, ids: [UInt32], w: [Float]) -> (y: [Float], clamped: Int) {
            var y = [Float](repeating: 0, count: tokens * h)
            var clamped = 0
            for t in 0 ..< tokens {
                let xr = Self.h32((0 ..< h).map { Double(x[t * h + $0]) })
                for s in 0 ..< k {
                    let ex = Int(ids[t * k + s])
                    var hidden = [Double](repeating: 0, count: i)
                    for n in 0 ..< i {
                        var g = 0.0, u = 0.0
                        let row = (ex * i + n) * h
                        for c in 0 ..< h {
                            g += xr[c] * wg[row + c]
                            u += xr[c] * wu[row + c]
                        }
                        let l = Double(Self.limit)
                        if g > l || abs(u) > l { clamped += 1 }
                        g = min(g, l)
                        u = min(max(u, -l), l)
                        hidden[n] = g / (1 + Foundation.exp(-Double(Self.alpha) * g)) * (u + 1)
                    }
                    let hr = Self.h32(hidden)
                    for n in 0 ..< h {
                        var acc = 0.0
                        let row = (ex * h + n) * i
                        for c in 0 ..< i { acc += hr[c] * wd[row + c] }
                        y[t * h + n] += Float(Double(w[t * k + s]) * acc)
                    }
                }
            }
            return (y, clamped)
        }
        func relL2(_ a: [Float], _ b: [Float]) -> Double {
            var num = 0.0, den = 0.0
            for j in b.indices {
                num += Double(a[j] - b[j]) * Double(a[j] - b[j])
                den += Double(b[j]) * Double(b[j])
            }
            XCTAssertGreaterThan(den, 0)
            return (num / den).squareRoot()
        }
        func inputs(tokens: Int) -> ([Float], [UInt32], [Float]) {
            let x = (0 ..< tokens * h).map { _ in rng.normal() }
            var ids: [UInt32] = []
            for _ in 0 ..< tokens {
                let first = UInt32(rng.next() % UInt64(e))
                ids += [first, (first + 1 + UInt32(rng.next() % UInt64(e - 1))) % UInt32(e)]
            }
            return (x, ids, (0 ..< tokens * k).map { _ in 0.25 + rng.uniform() })
        }

        // Decode route (3 tokens): F32 input, single-weight QMVs + F32 activation + weighted JANGH down.
        do {
            let tokens = 3
            let (x, ids, w) = inputs(tokens: tokens)
            let (want, clamped) = reference(x: x, tokens: tokens, ids: ids, w: w)
            XCTAssertGreaterThan(clamped, 0, "the clamp must bite")
            let run = { (layer: any WeightedRoutedExpertLayer) in
                layer.callRouted(MLXArray(x, [tokens, h]), indices: MLXArray(ids, [tokens, k]),
                                 scores: MLXArray(w, [tokens, k])).asArray(Float.self)
            }
            let err = relL2(run(oai), want), control = relL2(run(silu), want)
            print("[JANGTRoutedActivationTests] decode: rel-L2 \(err), SiLU control \(control)")
            XCTAssertLessThan(err, 1e-4)
            XCTAssertGreaterThan(control, 100 * 1e-4)
        }

        // Sorted prefill route (300 tokens x 2 routes = 600 pairs >= 512): BF16 rows, single-weight NAX tiles.
        guard JANGTPrefillKernel.available else {
            throw XCTSkip("the sorted prefill tiles need an M5-class GPU")
        }
        do {
            let tokens = 300
            let (x, ids, w) = inputs(tokens: tokens)
            let xb = MLXArray(x, [tokens, h]).asType(.bfloat16)
            let (want, clamped) = reference(
                x: xb.asType(.float32).asArray(Float.self), tokens: tokens, ids: ids, w: w)
            XCTAssertGreaterThan(clamped, 0)
            let run = { (layer: any WeightedRoutedExpertLayer) in
                layer.callRouted(xb, indices: MLXArray(ids, [tokens, k]), scores: MLXArray(w, [tokens, k]))
                    .asType(.float32).asArray(Float.self)
            }
            let err = relL2(run(oai), want), control = relL2(run(silu), want)
            print("[JANGTRoutedActivationTests] prefill: rel-L2 \(err), SiLU control \(control)")
            XCTAssertLessThan(err, 2e-2)
            XCTAssertGreaterThan(control, 10 * 2e-2)
        }
    }
}
