// Copyright © 2026 Osaurus AI. All rights reserved.

import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon

/// Inkling-2-Mini JANGHT bundles (2026-11-23): every routed projection group (gate/up together, down alone) of every
/// layer is a `jangtq2` codebook bank, a `jangt` trellis bank, or the source checkpoint's own NVFP4 / MXFP8 bytes under
/// runtime names. Synthetic weights: these prove admission, refusal, construction, the loader key layout and the
/// half-layer routed math, not trained-model quality (the fixture bundle's parity probe covers that).
final class Inkling2JANGHTBundleTests: XCTestCase {
    private let f = Ink2HTFixture()

    private func withBundle<T>(
        seed: UInt64 = 1, mutateConfig: (inout [String: Any]) -> Void = { _ in },
        mutateTensors: (inout [String: Ink2HTFixture.Raw]) -> Void = { _ in },
        _ body: (URL, Data, Ink2HTFixture.Bundle) throws -> T
    ) throws -> T {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("ink2ht-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        var bundle = f.makeBundle(seed: seed)
        mutateTensors(&bundle.raw)
        let data = try f.write(directory, bundle: bundle, mutateConfig: mutateConfig)
        return try body(directory, data, bundle)
    }

    // MARK: - admission

    func testMixedLayersAreAdmittedAndConstructed() throws {
        try MLXMetalTestLock.withLock {
            try withBundle { directory, data, _ in
                XCTAssertTrue(JANGHModelPreparation.declaresCustomFormat(configuration: data, sidecar: nil))
                let prepared = try XCTUnwrap(
                    Inkling2LoopJANGHPreparation.loadIfDeclared(directory: directory, configurationData: data))
                let banks = prepared.banks
                XCTAssertFalse(banks.routedProjectionsAreAllCodebook)
                let p = { (l: Int, r: String) in "model.layers.\(l).mlp.switch_mlp.\(r)" }
                XCTAssertEqual(banks.nativeRoutedModules, [
                    p(1, "gate_proj"): .nvfp4, p(1, "up_proj"): .nvfp4, p(1, "down_proj"): .nvfp4,
                    p(2, "down_proj"): .mxfp8, p(3, "gate_proj"): .nvfp4, p(3, "up_proj"): .nvfp4,
                ])
                // L0 trellis gate+up (2 x 6) + codebook down (2); L2 codebook gate+up (4); L3 trellis down (6);
                // L4 codebook gate+up (4) + trellis down (6).
                XCTAssertEqual(banks.excludedTensorNames.count, 34)
                XCTAssertTrue(banks.excludedTensorNames.allSatisfy { $0.contains(".tq2_") || $0.contains(".jt_") })
                // The ordinary plan skips every routed projection (custom and native alike) and keeps the trunk.
                let ordinary = try XCTUnwrap(
                    JSONSerialization.jsonObject(with: banks.ordinaryConfiguration) as? [String: Any])
                let q = try XCTUnwrap(ordinary["quantization"] as? [String: Any])
                for layer in 0 ..< f.layers {
                    for role in ["gate_proj", "up_proj", "down_proj"] {
                        XCTAssertEqual(q[p(layer, role)] as? Bool, false, p(layer, role))
                    }
                }
                XCTAssertEqual(q["bits"] as? Int, 8)

                let model = try prepared.construct(requesting: [.text])
                let mlps = model.model.layers.map(\.mlp)
                XCTAssertNotNil(mlps[0].mixedExperts)
                XCTAssertNil(mlps[0].mixedExperts?.nativeGateUp)
                XCTAssertNil(mlps[0].mixedExperts?.nativeDown)
                XCTAssertNotNil(mlps[1].switchMLP, "an all-native layer runs the native experts")
                XCTAssertNil(mlps[1].mixedExperts)
                XCTAssertNotNil(mlps[2].mixedExperts?.nativeDown)
                XCTAssertNil(mlps[2].mixedExperts?.nativeGateUp)
                XCTAssertNotNil(mlps[3].mixedExperts?.nativeGateUp)
                XCTAssertNil(mlps[3].mixedExperts?.nativeDown)
                XCTAssertNotNil(mlps[4].mixedExperts)
                XCTAssertTrue(mlps.allSatisfy { $0.janghExperts == nil })
                XCTAssertTrue(model.model.decodeSupported, "every layer keeps the batch-1 decode step")
                for name in banks.excludedTensorNames where !name.hasSuffix(".jt_cs") || name.contains("down_proj") {
                    XCTAssertTrue(model.excludeFromGenericSafetensorsLoad(key: name), name)
                }
                for name in [p(1, "gate_proj.weight"), p(2, "down_proj.scales"), p(3, "up_proj.global_scale")] {
                    XCTAssertFalse(model.excludeFromGenericSafetensorsLoad(key: name), name)
                }
            }
        }
    }

    /// The generic loader's view: sanitize brings each native group to its layer's module (`switch_mlp` for an
    /// all-native layer, `mixed_mlp` for a half layer), and the update finds a parameter for every key.
    func testNativeGroupsLoadUnderTheirLayerModules() throws {
        try MLXMetalTestLock.withLock {
            try withBundle { directory, data, bundle in
                let prepared = try XCTUnwrap(
                    Inkling2LoopJANGHPreparation.loadIfDeclared(directory: directory, configurationData: data))
                let model = try prepared.construct(requesting: [.text])
                let sanitized = model.sanitize(weights: bundle.native)
                XCTAssertEqual(Set(sanitized.keys), [
                    "model.layers.1.mlp.switch_mlp.gate_up_proj.codes",
                    "model.layers.1.mlp.switch_mlp.gate_up_proj.block_scales",
                    "model.layers.1.mlp.switch_mlp.gate_up_proj.expert_scale",
                    "model.layers.1.mlp.switch_mlp.down_proj.codes",
                    "model.layers.1.mlp.switch_mlp.down_proj.block_scales",
                    "model.layers.1.mlp.switch_mlp.down_proj.expert_scale",
                    "model.layers.2.mlp.mixed_mlp.down_proj.codes",
                    "model.layers.2.mlp.mixed_mlp.down_proj.block_scales",
                    "model.layers.2.mlp.mixed_mlp.down_proj.expert_scale",
                    "model.layers.3.mlp.mixed_mlp.gate_up_proj.codes",
                    "model.layers.3.mlp.mixed_mlp.gate_up_proj.block_scales",
                    "model.layers.3.mlp.mixed_mlp.gate_up_proj.expert_scale",
                ])
                XCTAssertNoThrow(try model.update(
                    parameters: ModuleParameters.unflattened(sanitized), verify: [.noUnusedKeys]))
                let l3 = try XCTUnwrap(model.model.layers[3].mlp.mixedExperts?.nativeGateUp)
                XCTAssertTrue(l3.isNVFP4)
                XCTAssertEqual(l3.codes.shape, [f.E, 2 * f.I, f.H / 8])
                let l2 = try XCTUnwrap(model.model.layers[2].mlp.mixedExperts?.nativeDown)
                XCTAssertEqual(l2.format, .init(mode: .mxfp8, bits: 8, groupSize: 32))
            }
        }
    }

    // MARK: - refusals

    private func assertRefused(
        _ why: String, mutateConfig: (inout [String: Any]) -> Void = { _ in },
        mutateTensors: (inout [String: Ink2HTFixture.Raw]) -> Void = { _ in },
        file: StaticString = #filePath, line: UInt = #line
    ) throws {
        try MLXMetalTestLock.withLock {
            try withBundle(mutateConfig: mutateConfig, mutateTensors: mutateTensors) { directory, data, _ in
                XCTAssertThrowsError(
                    try Inkling2LoopJANGHPreparation.loadIfDeclared(directory: directory, configurationData: data),
                    why, file: file, line: line)
            }
        }
    }

    private func setQuant(_ module: String, _ entry: Any) -> (inout [String: Any]) -> Void {
        { root in
            var q = root["quantization"] as! [String: Any]
            q[module] = entry
            root["quantization"] = q
        }
    }

    func testGateAndUpInDifferentFormatsAreRefused() throws {
        // L2 gate native, up still a codebook bank (the gate's codebook tensors stay in the index).
        try assertRefused("native gate with codebook up",
                          mutateConfig: setQuant("model.layers.2.mlp.switch_mlp.gate_proj", Ink2HTFixture.nvfp4Entry))
        // L3 gate NVFP4, up MXFP8.
        try assertRefused("NVFP4 gate with MXFP8 up",
                          mutateConfig: setQuant("model.layers.3.mlp.switch_mlp.up_proj", Ink2HTFixture.mxfp8Entry))
    }

    func testNativeEntriesOutsideTheContractAreRefused() throws {
        try assertRefused("NVFP4 group 32", mutateConfig: setQuant(
            "model.layers.3.mlp.switch_mlp.gate_proj", ["mode": "nvfp4", "group_size": 32, "bits": 4]))
        try assertRefused("MXFP8 with an extra key", mutateConfig: setQuant(
            "model.layers.2.mlp.switch_mlp.down_proj",
            ["mode": "mxfp8", "group_size": 32, "bits": 8, "rotation": "hadamard32"]))
        try assertRefused("affine routed projection in a custom stack", mutateConfig: setQuant(
            "model.layers.2.mlp.switch_mlp.down_proj", ["mode": "affine", "group_size": 32, "bits": 8]))
        try assertRefused("declared NVFP4 over MXFP8 tensors", mutateConfig: setQuant(
            "model.layers.2.mlp.switch_mlp.down_proj", Ink2HTFixture.nvfp4Entry))
    }

    func testNativeTensorsOutsideTheContractAreRefused() throws {
        try assertRefused("NVFP4 without global_scale", mutateTensors: {
            $0.removeValue(forKey: "model.layers.3.mlp.switch_mlp.gate_proj.global_scale")
        })
        try assertRefused("MXFP8 with affine biases", mutateTensors: {
            $0["model.layers.2.mlp.switch_mlp.down_proj.biases"] = ("F16", [16], Data(count: 32))
        })
        try assertRefused("MXFP8 with a global_scale", mutateTensors: {
            $0["model.layers.2.mlp.switch_mlp.down_proj.global_scale"] = ("F32", [16], Data(count: 64))
        })
        try assertRefused("NVFP4 scales of the wrong dtype", mutateTensors: {
            let (_, shape, data) = $0["model.layers.3.mlp.switch_mlp.up_proj.scales"]!
            $0["model.layers.3.mlp.switch_mlp.up_proj.scales"] = ("I8", shape, data)
        })
        try assertRefused("MXFP8 codes of the wrong width", mutateTensors: {
            $0["model.layers.2.mlp.switch_mlp.down_proj.weight"] = ("U32", [16, 256, 16], Data(count: 16 * 256 * 64))
        })
        try assertRefused("native tensors for a custom projection", mutateTensors: {
            $0["model.layers.0.mlp.switch_mlp.down_proj.weight"] = ("U32", [16, 256, 16], Data(count: 16 * 256 * 64))
        })
    }

    /// A native NVFP4 gate/up pair whose per-expert global scales differ cannot run as one fused gather: the load is
    /// refused on the tensor contents (never a silently skipped layer). Equal scales (the fixture) are admitted above.
    func testNVFP4GateUpWithDifferentGlobalScalesIsRefused() throws {
        for layer in [1, 3] {
            try assertRefused("L\(layer) gate/up global_scale differ", mutateTensors: {
                let name = "model.layers.\(layer).mlp.switch_mlp.up_proj.global_scale"
                let (dtype, shape, data) = $0[name]!
                var values = data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
                values[5] *= 2
                $0[name] = (dtype, shape, values.withUnsafeBytes { Data($0) })
            })
        }
    }

    /// Other architectures keep the previous contract: a native projection inside a custom stack is refused.
    func testNativeProjectionsNeedTheArchitecturesConsent() throws {
        try MLXMetalTestLock.withLock {
            try withBundle { directory, data, _ in
                let layout = JANGHRoutedModelLayout(
                    modelType: "inkling2_loop", hiddenSize: f.H, intermediateSize: f.I, expertCount: f.E,
                    sparseLayers: Set(0 ..< f.layers), routesPerToken: f.K)
                XCTAssertThrowsError(try JANGHModelPreparation(
                    directory: directory, configuration: data, sidecar: nil, layout: layout))
            }
        }
    }

    // MARK: - routed math

    /// A half layer's two halves composed (`routedHidden` -> `routedDown`) equal the full layer on the same banks, on
    /// the decode kernels (fp32 and bf16 rows) and the expert-sorted prefill tiles (>= 512 routed pairs).
    func testHalfLayersComposeToTheFullLayer() throws {
        try MLXMetalTestLock.withLock {
            try withBundle(seed: 2) { directory, data, _ in
                let prep = try JANGHModelPreparation(
                    directory: directory, configuration: data, sidecar: nil,
                    layout: .init(modelType: "inkling2_loop", hiddenSize: f.H, intermediateSize: f.I,
                                  expertCount: f.E, sparseLayers: Set(0 ..< f.layers), routesPerToken: f.K,
                                  admitsNativeRoutedProjections: true))
                let banks = try prep.mapBanks()
                let activation = JANGTActivation.inkling(gateCap: 4, upCap: 25)
                var rng = JANGTLowBitKernelTests.Rng(state: 9)
                var cases: [(Int, DType, Double)] = [(3, .float32, 1e-5), (3, .bfloat16, 1e-5)]
                if JANGTPrefillKernel.available { cases.append((160, .bfloat16, 1e-2)) }
                // L0: trellis gate/up (outliers) + codebook down; L4: codebook gate/up + trellis down.
                for layer in [0, 4] {
                    let parent = "model.layers.\(layer).mlp.switch_mlp"
                    func make(_ g: JANGTRoutedExpertLayer.Groups) throws -> JANGTRoutedExpertLayer {
                        try JANGTRoutedExpertLayer(
                            banks: banks, parentModule: parent, hiddenSize: f.H, intermediateSize: f.I,
                            activationLimit: nil, activation: activation, groups: g)
                    }
                    let full = try make(.gateUpAndDown), gateUp = try make(.gateUpOnly), down = try make(.downOnly)
                    for (tokens, dtype, bound) in cases {
                        let x = (MLXArray((0 ..< tokens * f.H).map { _ in rng.normal() }, [tokens, f.H]) * 0.5)
                            .asType(dtype)
                        let (ids, scores) = f.routing(&rng, tokens: tokens)
                        let want = full.callRouted(x, indices: ids, scores: scores)
                        let hidden = gateUp.routedHidden(x, indices: ids)
                        XCTAssertEqual(hidden.shape, [tokens * f.K, f.I])
                        XCTAssertEqual(hidden.dtype, .float32)
                        let got = down.routedDown(hidden, indices: ids, scores: scores, outputDType: dtype)
                        XCTAssertEqual(got.shape, want.shape)
                        XCTAssertEqual(got.dtype, dtype)
                        let w = want.asType(.float32).asArray(Float.self)
                        JANGTLowBitKernelTests.assertLive(w, "full layer \(layer)")
                        let err = JANGTLowBitKernelTests.relL2(got.asType(.float32).asArray(Float.self), w)
                        XCTAssertLessThan(err, bound, "layer \(layer) T=\(tokens) \(dtype)")
                    }
                }
            }
        }
    }

    /// The model's half-native layers against a dense reference: L2 (codebook gate/up from the custom half, MXFP8 down
    /// dequantized by MLX) and L3 (NVFP4 gate/up dequantized per expert with its global scale, trellis down from the
    /// custom half). Both the unsorted (decode) and the sorted (>= 64 routed pairs) native gathers.
    func testHalfNativeLayersMatchDenseReference() throws {
        try MLXMetalTestLock.withLock {
            try withBundle(seed: 3) { directory, data, bundle in
                let prepared = try XCTUnwrap(
                    Inkling2LoopJANGHPreparation.loadIfDeclared(directory: directory, configurationData: data))
                let model = try prepared.construct(requesting: [.text])
                try model.update(
                    parameters: ModuleParameters.unflattened(model.sanitize(weights: bundle.native)),
                    verify: [.noUnusedKeys])
                var rng = JANGTLowBitKernelTests.Rng(state: 11)
                let caps: (Float, Float) = (4, 25)
                func act(_ g: MLXArray, _ u: MLXArray) -> MLXArray {
                    tanh(g / caps.0) * caps.0 * sigmoid(g) * (tanh(u / caps.1) * caps.1)
                }
                func dense(_ name: String, _ format: JANGHNativeRoutedFormat) -> MLXArray {
                    let w = bundle.native[name + ".weight"]!, s = bundle.native[name + ".scales"]!
                    if format == .mxfp8 {
                        return MLX.dequantized(w, scales: s, biases: nil, groupSize: 32, bits: 8, mode: .mxfp8, dtype: .float32)
                    }
                    let g = bundle.native[name + ".global_scale"]!
                    return MLX.stacked((0 ..< f.E).map { e in
                        MLX.dequantized(w[e], scales: s[e], biases: nil, groupSize: 16, bits: 4, mode: .nvfp4,
                                        globalScale: g[e ..< (e + 1)], dtype: .float32)
                    }, axis: 0)
                }
                for tokens in [2, 40] {
                    let x = MLXArray((0 ..< tokens * f.H).map { _ in rng.normal() }, [tokens, f.H]) * 0.5
                    let (ids, scores) = f.routing(&rng, tokens: tokens)
                    let flatIDs = ids.reshaped([-1])
                    // L2: custom gate/up (its own half), MXFP8 down.
                    let l2 = try XCTUnwrap(model.model.layers[2].mlp.mixedExperts)
                    let h2 = l2.custom.routedHidden(x, indices: ids)  // [T*k, I]
                    let wd2 = dense("model.layers.2.mlp.switch_mlp.down_proj", .mxfp8)[flatIDs]  // [T*k, H, I]
                    let y2 = MLX.matmul(wd2, h2.expandedDimensions(axis: -1)).squeezed(axis: -1)
                        .reshaped([tokens, f.K, f.H])
                    let want2 = (y2 * scores.expandedDimensions(axis: -1)).sum(axis: 1)
                    let got2 = l2(x, ids, scores)
                    // L3: NVFP4 gate/up, custom down (its own half).
                    let l3 = try XCTUnwrap(model.model.layers[3].mlp.mixedExperts)
                    let wg = dense("model.layers.3.mlp.switch_mlp.gate_proj", .nvfp4)[flatIDs]
                    let wu = dense("model.layers.3.mlp.switch_mlp.up_proj", .nvfp4)[flatIDs]
                    let xs = MLX.repeated(x, count: f.K, axis: 0).expandedDimensions(axis: -1)  // [T*k, H, 1]
                    let h3 = act(MLX.matmul(wg, xs).squeezed(axis: -1), MLX.matmul(wu, xs).squeezed(axis: -1))
                    let want3 = l3.custom.routedDown(h3, indices: ids, scores: scores, outputDType: .float32)
                    let got3 = l3(x, ids, scores)
                    for (got, want, name) in [(got2, want2, "L2"), (got3, want3, "L3")] {
                        let w = want.asArray(Float.self)
                        JANGTLowBitKernelTests.assertLive(w, name)
                        let err = JANGTLowBitKernelTests.relL2(got.asType(.float32).asArray(Float.self), w)
                        XCTAssertLessThan(err, 2e-3, "\(name) T=\(tokens)")
                    }
                }
            }
        }
    }
}

/// Synthetic Inkling-2 JANGHT bundle: 5 layers, hidden 256, intermediate 128, 16 experts, top-4.
///   L0 gate/up jt2 (layer-wide outliers) / down jh2; L1 all NVFP4; L2 gate/up jh4 / down MXFP8;
///   L3 gate/up NVFP4 / down jt2.5 (per-expert outliers); L4 gate/up jh2 / down jt2.
struct Ink2HTFixture {
    typealias Raw = (String, [Int], Data)
    typealias P = JANGTLowBitKernelTests.Projection
    let H = 256, I = 128, E = 16, K = 4, layers = 5

    enum Unit { case jh(Int), jt(Double), nvfp4, mxfp8 }
    let plan: [(Unit, Unit)] = [(.jt(2), .jh(2)), (.nvfp4, .nvfp4), (.jh(4), .mxfp8), (.nvfp4, .jt(2.5)), (.jh(2), .jt(2))]

    static var nvfp4Entry: [String: Any] { ["mode": "nvfp4", "group_size": 16, "bits": 4] }
    static var mxfp8Entry: [String: Any] { ["mode": "mxfp8", "group_size": 32, "bits": 8] }

    struct Bundle {
        var raw: [String: Raw] = [:]
        var quantization: [String: Any] = [:]
        var outliers: [String: Int] = [:]
        /// The native tensors as arrays (the generic loader's input), keyed by bundle name.
        var native: [String: MLXArray] = [:]
    }

    private func bytes<T>(_ v: [T]) -> Data { v.withUnsafeBytes { Data($0) } }

    func makeBundle(seed: UInt64) -> Bundle {
        var rng = JANGTLowBitKernelTests.Rng(state: seed)
        var b = Bundle()
        for (layer, (gateUp, down)) in plan.enumerated() {
            for role in ["gate_proj", "up_proj", "down_proj"] {
                let module = "model.layers.\(layer).mlp.switch_mlp.\(role)"
                let isDown = role == "down_proj"
                let (n, kin) = isDown ? (H, I) : (I, H)
                switch isDown ? down : gateUp {
                case .jh(let bits):
                    let words = kin * bits / 32
                    b.raw[module + ".tq2_packed"] = ("U32", [E, n, words], bytes((0 ..< E * n * words).map { _ in rng.word() }))
                    b.raw[module + ".tq2_scales"] = ("F16", [E, n], bytes((0 ..< E * n).map { _ in Float16(0.02 + 0.02 * rng.uniform()) }))
                    b.quantization[module] = ["mode": "jangtq2", "bits": bits, "rotation": "hadamard32"]
                case .jt(let k):
                    let columns: [[Int32]] = isDown
                        ? (0 ..< E).map { e in e % 2 == 0 ? [Int32(e % kin), -1] : [Int32(kin - 1), Int32(5)] }
                        : (0 ..< E).map { _ in [Int32(3), Int32(kin - 5)] }
                    let p = P(&rng, k: k, experts: E, outputs: n, inputs: kin, columns: columns, columnScales: isDown)
                    b.raw[module + ".jt_packed"] = ("U32", [E, n, p.words], bytes(p.packed))
                    b.raw[module + ".jt_scale"] = ("F32", [E, n], bytes(p.scale))
                    b.raw[module + ".jt_su"] = ("F32", [kin], bytes(p.signs))
                    b.raw[module + ".jt_ocol"] = ("I32", [E, p.outliers], bytes(p.ocol))
                    b.raw[module + ".jt_wout"] = ("F16", [E, n, p.outliers], bytes(p.wout.map { Float16($0) }))
                    if let cs = p.cs { b.raw[module + ".jt_cs"] = ("F32", [E, kin], bytes(cs)) }
                    b.quantization[module] = ["mode": "jangt", "kbits": k]
                    b.outliers[module] = p.outliers
                case .nvfp4:
                    // e2m1 codes are any nibble; e4m3 block scales 2^-4 .. 1 (no NaN encodings).
                    let codes = (0 ..< E * n * kin / 8).map { _ in rng.word() }
                    let scales = (0 ..< E * n * kin / 16).map { _ in UInt8(0x30 + Int(rng.next() % 9)) }
                    // One factor per expert, shared by gate and up (the source's w13 factor).
                    let global = (0 ..< E).map { e in Float(2688) * (0.05 + 0.01 * Float(e % 5)) }
                    put(&b, module + ".weight", "U32", [E, n, kin / 8], codes, MLXArray(codes, [E, n, kin / 8]))
                    put(&b, module + ".scales", "U8", [E, n, kin / 16], scales, MLXArray(scales, [E, n, kin / 16]))
                    put(&b, module + ".global_scale", "F32", [E], global, MLXArray(global))
                    b.quantization[module] = Self.nvfp4Entry
                case .mxfp8:
                    // e4m3 codes without the NaN encodings (|code| <= 0x6F); e8m0 scales 2^-7 .. 1.
                    let codeBytes = (0 ..< E * n * kin).map { _ -> UInt8 in
                        UInt8(rng.next() % 0x70) | (rng.next() % 2 == 0 ? 0 : 0x80)
                    }
                    let codes = codeBytes.withUnsafeBytes { Array($0.bindMemory(to: UInt32.self)) }
                    let scales = (0 ..< E * n * kin / 32).map { _ in UInt8(120 + Int(rng.next() % 8)) }
                    put(&b, module + ".weight", "U32", [E, n, kin / 4], codes, MLXArray(codes, [E, n, kin / 4]))
                    put(&b, module + ".scales", "U8", [E, n, kin / 32], scales, MLXArray(scales, [E, n, kin / 32]))
                    b.quantization[module] = Self.mxfp8Entry
                }
            }
        }
        return b
    }

    private func put<T>(_ b: inout Bundle, _ name: String, _ dtype: String, _ shape: [Int], _ values: [T],
                        _ array: MLXArray) {
        b.raw[name] = (dtype, shape, bytes(values))
        b.native[name] = array
    }

    func configuration(_ bundle: Bundle) -> [String: Any] {
        var q: [String: Any] = ["group_size": 64, "bits": 8, "mode": "affine"]
        q.merge(bundle.quantization) { $1 }
        func levels(_ bits: Int, _ alpha: Double) -> [Double] {
            (0 ..< (1 << bits)).map { (Double($0) - Double((1 << bits) - 1) / 2) * alpha }
        }
        return [
            "model_type": "inkling2_loop", "hidden_size": H, "num_hidden_layers": layers, "vocab_size": 64,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 32, "n_routed_experts": E,
            "num_experts_per_tok": K, "intermediate_size": I, "n_layers_per_attn_res_block": 2,
            "sliding_window_size": 64, "rel_extent": 64,
            "loop_config": ["core_start": 1, "core_end": 3, "num_passes": 2, "gate_rank": 8],
            "quantization": q,
            "jangtq": [
                "version": 2, "packing": "lsb-bitstream", "scale_dtype": "float16", "rotation": "hadamard32",
                "codebook_family": "odd-cubic",
                "codebooks": [
                    "2": ["alpha": 0.9, "beta": 0.0, "levels": levels(2, 0.9)],
                    "4": ["alpha": 0.25, "beta": 0.0, "levels": levels(4, 0.25)],
                ],
            ],
            "jangt": [
                "version": 1, "code": "v2_halfbits", "state_bits": 12, "rotation": "hadamard128-signs",
                "outlier_columns": bundle.outliers,
                "code_multipliers": ["2": 636_231_159, "2.5": 636_231_159],
                "step_bits": ["2": [4, 4], "2.5": [6, 4]],
            ],
        ]
    }

    func write(_ directory: URL, bundle: Bundle, mutateConfig: (inout [String: Any]) -> Void) throws -> Data {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        var index: [String: String] = [:]
        try Q38TFixture().writeAligned(bundle.raw, file: "model-00001-of-00001.safetensors", into: directory,
                                       index: &index)
        try JSONSerialization.data(withJSONObject: ["weight_map": index]).write(
            to: directory.appendingPathComponent("model.safetensors.index.json"))
        var root = configuration(bundle)
        mutateConfig(&root)
        let data = try JSONSerialization.data(withJSONObject: root)
        try data.write(to: directory.appendingPathComponent("config.json"))
        return data
    }

    /// Distinct experts per token, positive weights summing to 1: `ids` `[T, k]` uint32, `scores` `[T, k]` fp32.
    func routing(_ rng: inout JANGTLowBitKernelTests.Rng, tokens: Int) -> (MLXArray, MLXArray) {
        var ids: [UInt32] = [], scores: [Float] = []
        for _ in 0 ..< tokens {
            var pool = Array(0 ..< E)
            for s in 0 ..< K { pool.swapAt(s, s + Int(rng.next() % UInt64(E - s))) }
            ids += pool.prefix(K).map(UInt32.init)
            let w = (0 ..< K).map { _ in 0.05 + rng.uniform() }
            let sum = w.reduce(0, +)
            scores += w.map { $0 / sum }
        }
        return (MLXArray(ids, [tokens, K]), MLXArray(scores, [tokens, K]))
    }
}
