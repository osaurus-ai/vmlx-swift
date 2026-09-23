// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
@testable import MLXLMCommon
import MLXNN
@testable import MLXLLM
import Testing

@Suite("Mixed expert region loading", .serialized)
struct MixedQuantizedExpertCatalogTests {
    @Test("Host load policy recognizes the same resident contract as the model factory")
    func residentAdmissionContract() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        var config: [String: Any] = ["model_type": "mimo_v2",
            "attention_projection_layout": "fused_qkv", "n_routed_experts": 256,
            "quantization": ["mode": "affine", "gate": ["mode": "mxfp4"]]]
        let url = directory.appendingPathComponent("config.json")
        try JSONSerialization.data(withJSONObject: config).write(to: url)
        let facts = LoadBundleFacts.inspect(bundleURL: directory)
        #expect(facts.isMiMoV26MixedQuantized)
        #expect(!facts.resolveMmapSafetensors(requested: true))
        // Residency must not silently raise the caller's allocator limits.
        #expect(facts.resolveMLXMemoryLimit(requested: .default) == .default)
        #expect(facts.resolveMLXAllocatorCacheLimit(requested: .default) == .default)
        config["attention_projection_layout"] = "separate"
        try JSONSerialization.data(withJSONObject: config).write(to: url)
        #expect(!LoadBundleFacts.inspect(bundleURL: directory).requiresResidentSafetensors)
    }

    private func saveAligned(_ arrays: [String: MLXArray], to url: URL) throws {
        try MLX.save(arrays: arrays, url: url)
        let data = try Data(contentsOf: url)
        let length = data.prefix(8).enumerated().reduce(UInt64(0)) {
            $0 | UInt64($1.element) << ($1.offset * 8)
        }
        let padding = (64 - (8 + Int(length)) % 64) % 64
        var size = (length + UInt64(padding)).littleEndian
        var aligned = withUnsafeBytes(of: &size) { Data($0) }
        aligned.append(data[8..<(8 + Int(length))])
        aligned.append(Data(repeating: 32, count: padding))
        aligned.append(data[(8 + Int(length))...])
        try aligned.write(to: url)
    }

    private func bundle() throws -> (URL, [String: MLXArray]) {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        var weights: [String: MLXArray] = [:]
        for (name, bits, group, mode) in [
            ("gate_proj", 4, 32, QuantizationMode.mxfp4),
            ("up_proj", 2, 64, .affine), ("down_proj", 8, 32, .affine),
        ] {
            let input = name == "down_proj" ? 32 : 64, output = name == "down_proj" ? 64 : 32
            let original = MLXArray((0..<(4*output*input)).map { Float($0 % 37 - 18) / 31 })
                .reshaped(4, output, input).asType(.bfloat16)
            let q = quantized(original, groupSize: group, bits: bits, mode: mode)
            let stem = "model.layers.1.mlp.switch_mlp.\(name)"
            weights[stem + ".weight"] = q.wq
            weights[stem + ".scales"] = q.scales
            weights[stem + ".biases"] = q.biases
        }
        let first = weights.filter { $0.key.hasSuffix(".weight") }
        let second = weights.filter { !$0.key.hasSuffix(".weight") }
        try saveAligned(first, to: url.appendingPathComponent("weights.safetensors"))
        try saveAligned(second, to: url.appendingPathComponent("companions.safetensors"))
        let index = weights.mapValues { _ in "companions.safetensors" }
            .merging(first.mapValues { _ in "weights.safetensors" }) { _, b in b }
        try JSONSerialization.data(withJSONObject: ["weight_map": index])
            .write(to: url.appendingPathComponent("model.safetensors.index.json"))
        return (url, weights)
    }

    private func catalog(_ directory: URL) throws -> MixedQuantizedExpertCatalog {
        try MixedQuantizedExpertCatalog(directory: directory, layerIndices: [1],
            expertCount: 4, inputDimensions: 64, hiddenDimensions: 32)
    }

    @Test func crossShardCompanionsAndNativeProjectionParity() throws {
        let (url, weights) = try bundle()
        defer { try? FileManager.default.removeItem(at: url) }
        let catalog = try catalog(url)
        for index in [0, 3] {
            let expert = try catalog.loadExpert(layer: 1, index: index)
            for (name, projection) in [("gate_proj", expert.gate), ("up_proj", expert.up), ("down_proj", expert.down)] {
                let stem = "model.layers.1.mlp.switch_mlp.\(name)"
                let original = try #require(weights[stem + ".weight"])
                let scales = try #require(weights[stem + ".scales"])
                #expect(arrayEqual(projection.weight, original[index]).item(Bool.self))
                #expect(arrayEqual(projection.scales, scales[index]).item(Bool.self))
                #expect(projection.weight.dtype == original.dtype)
                #expect(projection.scales.dtype == scales.dtype)
                let input = MLXArray.ones([1, name == "down_proj" ? 32 : 64], dtype: .bfloat16)
                let expected = quantizedMM(input, original[index], scales: scales[index],
                    biases: weights[stem + ".biases"].map { $0[index] }, groupSize: projection.groupSize,
                    bits: projection.bits, mode: projection.mode)
                let actual = quantizedMM(input, projection.weight, scales: projection.scales,
                    biases: projection.biases, groupSize: projection.groupSize,
                    bits: projection.bits, mode: projection.mode)
                #expect(arrayEqual(expected, actual).item(Bool.self))
            }
        }
        #expect(throws: MixedQuantizedExpertCatalog.InvalidBundle.self) { try catalog.loadExpert(layer: 1, index: 4) }
        #expect(throws: MixedQuantizedExpertCatalog.InvalidBundle.self) { try catalog.loadExpert(layer: 0, index: 0) }
    }

    @Test func residentBanksPreservePackedBitsAndSurviveSourceRemoval() throws {
        let (url, weights) = try bundle()
        defer { try? FileManager.default.removeItem(at: url) }
        let source = try catalog(url)
        let resident = try source.loadExperts(layer: 1, storage: .resident)
        // Adjacent expert views must address adjacent spans of the same bank.
        // Single integer indexing instead produces separately allocated gathers.
        let first = resident[0].gate.weight.asData(access: .noCopy)
        let second = resident[1].gate.weight.asData(access: .noCopy)
        first.data.withUnsafeBytes { a in
            second.data.withUnsafeBytes { b in
                #expect(Int(bitPattern: b.baseAddress!) - Int(bitPattern: a.baseAddress!) == first.data.count)
            }
        }
        let mappedModule = try MixedQuantizedSwitchGLU(catalog: source, layer: 1,
            inputDimensions: 64, storage: .mapped)
        let residentModule = try MixedQuantizedSwitchGLU(catalog: source, layer: 1,
            inputDimensions: 64, storage: .resident)
        let input = MLXArray.ones([1, 1, 64], dtype: .bfloat16)
        let routes = MLXArray([Int32(3), 0]).reshaped(1, 1, 2)
        let expected = mappedModule(input, routes)
        MLX.eval(expected)
        try FileManager.default.removeItem(at: url)
        for (index, expert) in resident.enumerated() {
            for (name, projection) in [("gate_proj", expert.gate), ("up_proj", expert.up), ("down_proj", expert.down)] {
                let stem = "model.layers.1.mlp.switch_mlp.\(name)"
                let weight = try #require(weights[stem + ".weight"])
                let scales = try #require(weights[stem + ".scales"])
                #expect(projection.weight.dtype == weight.dtype)
                #expect(arrayEqual(projection.weight, weight[index]).item(Bool.self))
                #expect(projection.scales.dtype == scales.dtype)
                #expect(arrayEqual(projection.scales, scales[index]).item(Bool.self))
                if let bias = weights[stem + ".biases"] {
                    #expect(arrayEqual(try #require(projection.biases), bias[index]).item(Bool.self))
                } else {
                    #expect(projection.biases == nil)
                }
            }
        }
        #expect(arrayEqual(expected, residentModule(input, routes)).item(Bool.self))
    }

    @Test func rejectsMissingCompanionAndTraversal() throws {
        let (url, _) = try bundle()
        defer { try? FileManager.default.removeItem(at: url) }
        let path = url.appendingPathComponent("model.safetensors.index.json")
        var object = try #require(JSONSerialization.jsonObject(with: Data(contentsOf: path)) as? [String: [String: String]])
        let key = "model.layers.1.mlp.switch_mlp.up_proj.scales"
        object["weight_map"]?.removeValue(forKey: key)
        try JSONSerialization.data(withJSONObject: object).write(to: path)
        #expect(throws: MixedQuantizedExpertCatalog.InvalidBundle.self) { try catalog(url) }
        object["weight_map"]?[key] = "../companions.safetensors"
        try JSONSerialization.data(withJSONObject: object).write(to: path)
        #expect(throws: MixedQuantizedExpertCatalog.InvalidBundle.self) { try catalog(url) }
    }

    @Test func rejectsCompanionGeometryAndTruncatedPayload() throws {
        let (url, weights) = try bundle()
        defer { try? FileManager.default.removeItem(at: url) }
        var companions = weights.filter { !$0.key.hasSuffix(".weight") }
        companions["model.layers.1.mlp.switch_mlp.up_proj.scales"] = MLXArray.ones([4, 32, 3], dtype: .bfloat16)
        try saveAligned(companions, to: url.appendingPathComponent("companions.safetensors"))
        #expect(throws: MixedQuantizedExpertCatalog.InvalidBundle.self) { try catalog(url) }
        let file = try FileHandle(forWritingTo: url.appendingPathComponent("weights.safetensors"))
        try file.truncate(atOffset: 12)
        try file.close()
        #expect(throws: MixedQuantizedExpertCatalog.InvalidBundle.self) { try catalog(url) }
    }

    @Test func mappingFailureThrowsDuringModuleConstruction() throws {
        let (url, _) = try bundle()
        defer { try? FileManager.default.removeItem(at: url) }
        let validated = try catalog(url)
        try FileManager.default.removeItem(at: url.appendingPathComponent("weights.safetensors"))
        #expect(throws: (any Error).self) {
            try MixedQuantizedSwitchGLU(catalog: validated, layer: 1, inputDimensions: 64)
        }
    }

    @Test(arguments: [1, 7, 9])
    func rejectsUnsupportedAffineWidths(bits: Int) throws {
        let (url, original) = try bundle()
        defer { try? FileManager.default.removeItem(at: url) }
        var weights = original
        // Valid payload geometry must not admit a width native MLX rejects.
        weights["model.layers.1.mlp.switch_mlp.up_proj.weight"] =
            MLXArray.zeros([4, 32, 2 * bits], dtype: .uint32)
        try saveAligned(weights, to: url.appendingPathComponent("model.safetensors"))
        try JSONSerialization.data(withJSONObject: ["weight_map": weights.mapValues { _ in "model.safetensors" }])
            .write(to: url.appendingPathComponent("model.safetensors.index.json"))
        #expect(throws: MixedQuantizedExpertCatalog.InvalidBundle.self) { try catalog(url) }
    }

    @Test(arguments: [MixedQuantizedExpertCatalog.Storage.mapped, .resident], [64, 128])
    func eightRegionMetalKernelMatchesNativeProjections(storage: MixedQuantizedExpertCatalog.Storage, upGroup: Int) throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let input = MLXArray((0..<1024).map { Float($0 % 17 - 8) / 13 })
            .asType(.bfloat16)[.stride(by: 2)].reshaped(1, 1, 512)
        let order: [Int32] = [7, 0, 5, 1, 4, 2, 6, 3]
        for mxGate in [false, true] {
            var weights: [String: MLXArray] = [:]
            for name in ["gate_proj", "up_proj", "down_proj"] {
                let mx = name == "gate_proj" && mxGate
                let mode: QuantizationMode = mx ? .mxfp4 : .affine
                let group = mx ? 32 : name == "down_proj" ? 128 : name == "up_proj" ? upGroup : 64
                let count: Int = 8 * 512 * 512
                let values: [Float] = (0..<count).map { (index: Int) -> Float in
                    Float(index % 41 - 20) / Float(83)
                }
                let source = MLXArray(values).reshaped(8, 512, 512).asType(.bfloat16)
                let q = quantized(source, groupSize: group, bits: mx ? 4 : 2, mode: mode)
                let stem = "model.layers.1.mlp.switch_mlp.\(name)"
                weights[stem + ".weight"] = q.wq
                weights[stem + ".scales"] = q.scales
                weights[stem + ".biases"] = q.biases
            }
            try saveAligned(weights, to: directory.appendingPathComponent("model.safetensors"))
            try JSONSerialization.data(withJSONObject: ["weight_map": weights.mapValues { _ in "model.safetensors" }])
                .write(to: directory.appendingPathComponent("model.safetensors.index.json"))
            let catalog = try MixedQuantizedExpertCatalog(directory: directory, layerIndices: [1],
                expertCount: 8, inputDimensions: 512, hiddenDimensions: 512)
            let module = try MixedQuantizedSwitchGLU(catalog: catalog, layer: 1,
                inputDimensions: 512, storage: storage)
            if storage == .resident {
                let bank = try catalog.loadResidentLayer(layer: 1)
                let duplicateRoutes: [Int32] = [7, 0, 7, 1, 4, 2, 0, 3]
                let pair = try #require(MixedQuantizedExpertKernel().pairedGateUp(input,
                    indices: MLXArray(duplicateRoutes).reshaped(1, 1, 8),
                    gate: bank.gate, up: bank.up))
                for (i, name) in ["gate_proj", "up_proj"].enumerated() {
                    let mx = name == "gate_proj" && mxGate
                    let stem = "model.layers.1.mlp.switch_mlp.\(name)"
                    let expected = stacked(duplicateRoutes.map { index in
                        quantizedMM(input, weights[stem + ".weight"]![Int(index)],
                            scales: weights[stem + ".scales"]![Int(index)],
                            biases: weights[stem + ".biases"].map { $0[Int(index)] },
                            groupSize: mx ? 32 : name == "up_proj" ? upGroup : 64, bits: mx ? 4 : 2,
                            mode: mx ? .mxfp4 : .affine)
                    }).reshaped(1, 1, 8, 1, 512)
                    #expect(arrayEqual(expected, pair[i]).item(Bool.self))
                }
                let fused = try #require(MixedQuantizedExpertKernel().fusedGateUp(input,
                    indices: MLXArray(duplicateRoutes).reshaped(1, 1, 8),
                    gate: bank.gate, up: bank.up))
                #expect(arrayEqual(silu(pair[0]) * pair[1], fused).item(Bool.self))
            }
            #expect(module.parameters().flattened().isEmpty)
            let reference = order.map { index in
                func project(_ x: MLXArray, _ name: String) -> MLXArray {
                    let mx = name == "gate_proj" && mxGate
                    let stem = "model.layers.1.mlp.switch_mlp.\(name)"
                    return quantizedMM(x, weights[stem + ".weight"]![Int(index)],
                        scales: weights[stem + ".scales"]![Int(index)],
                        biases: weights[stem + ".biases"].map { $0[Int(index)] },
                        groupSize: mx ? 32 : name == "down_proj" ? 128 : name == "up_proj" ? upGroup : 64,
                        bits: mx ? 4 : 2, mode: mx ? .mxfp4 : .affine)
                }
                return project(MLXNN.silu(project(input, "gate_proj")) * project(input, "up_proj"), "down_proj")
            }
            let expected = stacked(reference).reshaped(1, 1, 8, 512)
            let result = module(input, MLXArray(order).reshaped(1, 1, 8))
            #expect(arrayEqual(expected, result).item(Bool.self))
            eval(result)
        }
    }

    @Test(arguments: [3, 5, 6])
    func nativeAffineWidthsKeepPackingAndRouting(bits: Int) throws {
        // Preserve the installed bundle's three-bit gate / two-bit up/down case.
        try nativeQuantRouting(specs: [
            "gate_proj": (bits, 128, .affine),
            "up_proj": (2, 128, .affine), "down_proj": (2, 128, .affine),
        ], dtype: .bfloat16)
    }

    @Test(arguments: 0..<36)
    func nativeAffineProjectionMatrix(caseIndex: Int) throws {
        let widths = [2, 3, 4, 5, 6, 8]
        let group = [32, 64, 128][caseIndex / 6 % 3]
        let dtype: DType = caseIndex < 18 ? .bfloat16 : .float16
        // Each role sees all six widths, three group sizes, and both scale
        // dtypes. Rotate widths to exercise mixed, not only uniform, banks.
        var specs: [String: (bits: Int, group: Int, mode: QuantizationMode)] = [:]
        for (offset, name) in ["gate_proj", "up_proj", "down_proj"].enumerated() {
            specs[name] = (widths[(caseIndex + offset) % widths.count], group, .affine)
        }
        try nativeQuantRouting(specs: specs, dtype: dtype)
    }

    @Test(arguments: 1..<8, [DType.bfloat16, .float16])
    func nativeMXFP4ProjectionMatrix(mask: Int, dtype: DType) throws {
        // Every nonempty combination of MXFP4 gate/up/down, including all
        // MXFP4. Remaining roles use odd-bit affine companions.
        var specs: [String: (bits: Int, group: Int, mode: QuantizationMode)] = [:]
        for (offset, name) in ["gate_proj", "up_proj", "down_proj"].enumerated() {
            specs[name] = mask & (1 << offset) != 0
                ? (4, 32, .mxfp4) : ([3, 5, 6][offset], 128, .affine)
        }
        try nativeQuantRouting(specs: specs, dtype: dtype)
    }

    @Test(arguments: 1..<8, [DType.bfloat16, .float16])
    func nativeMXFP8ProjectionMatrix(mask: Int, dtype: DType) throws {
        var specs: [String: (bits: Int, group: Int, mode: QuantizationMode)] = [:]
        for (offset, name) in ["gate_proj", "up_proj", "down_proj"].enumerated() {
            specs[name] = mask & (1 << offset) != 0
                ? (8, 32, .mxfp8)
                : offset == 1 ? (4, 32, .mxfp4) : (3, 128, .affine)
        }
        try nativeQuantRouting(specs: specs, dtype: dtype)
    }

    @Test(arguments: 0..<4)
    func rejectsInvalidMXFP8Companions(caseIndex: Int) throws {
        let (url, original) = try bundle()
        defer { try? FileManager.default.removeItem(at: url) }
        var weights = original
        let stem = "model.layers.1.mlp.switch_mlp.gate_proj"
        let bits = caseIndex == 3 ? 6 : 8
        let group = caseIndex == 0 ? 64 : 32
        let dtype: DType = caseIndex == 1 ? .float16 : .uint8
        weights[stem + ".weight"] = MLXArray.zeros([4, 32, 2 * bits], dtype: .uint32)
        weights[stem + ".scales"] = MLXArray.ones([4, 32, 64 / group], dtype: dtype)
        weights[stem + ".biases"] = caseIndex == 2
            ? MLXArray.zeros([4, 32, 64 / group], dtype: .uint8) : nil
        try saveAligned(weights, to: url.appendingPathComponent("model.safetensors"))
        try JSONSerialization.data(withJSONObject: ["weight_map": weights.mapValues { _ in "model.safetensors" }])
            .write(to: url.appendingPathComponent("model.safetensors.index.json"))
        #expect(throws: MixedQuantizedExpertCatalog.InvalidBundle.self) { try catalog(url) }
    }

    @Test(arguments: ["affine", "mxfp4", "mxfp8"])
    func quantizationDoesNotSelectLegacyArchitecture(mode: String) async throws {
        var object = try #require(JSONSerialization.jsonObject(
            with: MiMoV26RuntimeTests.configuration(moe: true)) as? [String: Any])
        object["quantization"] = ["mode": mode, "bits": mode == "mxfp4" ? 4 : 8,
                                   "group_size": mode == "affine" ? 64 : 32]
        let data = try JSONSerialization.data(withJSONObject: object)
        #expect(MiMoV26Contract.matches(data))
        let model = try await LLMTypeRegistry.shared.createModel(configuration: data, modelType: "mimo_v2")
        #expect(model is MiMoV26TextModel)
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try data.write(to: directory.appendingPathComponent("config.json"))
        #expect(LoadBundleFacts.inspect(bundleURL: directory).requiresResidentSafetensors)
    }

    private func nativeQuantRouting(
        specs: [String: (bits: Int, group: Int, mode: QuantizationMode)], dtype: DType
    ) throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        var weights: [String: MLXArray] = [:]
        for (projection, phase) in [("gate_proj", Float(0.3)), ("up_proj", 1.1), ("down_proj", 2.7)] {
            let spec = specs[projection]!
            let values = sin(MLXArray(0..<(8 * 512 * 512)).asType(.float32) * Float(0.031) + phase)
            let packed = quantized((values * Float(0.02)).reshaped(8, 512, 512).asType(dtype),
                groupSize: spec.group, bits: spec.bits, mode: spec.mode)
            let stem = "model.layers.1.mlp.switch_mlp.\(projection)"
            weights[stem + ".weight"] = packed.wq
            weights[stem + ".scales"] = packed.scales
            weights[stem + ".biases"] = packed.biases
        }
        try saveAligned(weights, to: directory.appendingPathComponent("model.safetensors"))
        try JSONSerialization.data(withJSONObject: ["weight_map": weights.mapValues { _ in "model.safetensors" }])
            .write(to: directory.appendingPathComponent("model.safetensors.index.json"))
        let source = try MixedQuantizedExpertCatalog(directory: directory, layerIndices: [1],
            expertCount: 8, inputDimensions: 512, hiddenDimensions: 512)
        let bank = try source.loadResidentLayer(layer: 1)
        for (name, projection) in [("gate_proj", bank.gate), ("up_proj", bank.up), ("down_proj", bank.down)] {
            let stem = "model.layers.1.mlp.switch_mlp.\(name)"
            let spec = specs[name]!
            #expect(projection.bits == spec.bits && projection.groupSize == spec.group)
            #expect(projection.mode == spec.mode)
            for (actual, expected) in [(projection.weight, weights[stem + ".weight"]!),
                                       (projection.scales, weights[stem + ".scales"]!)] {
                #expect(actual.dtype == expected.dtype)
                #expect(arrayEqual(actual, expected).item(Bool.self))
            }
            if let bias = weights[stem + ".biases"] {
                #expect(projection.biases?.dtype == dtype)
                #expect(arrayEqual(try #require(projection.biases), bias).item(Bool.self))
            } else {
                #expect(projection.biases == nil)
            }
        }
        let reference = SwitchGLU(inputDims: 512, hiddenDims: 512, numExperts: 8,
            allowFusedGateUpCache: false)
        let projections: [(String, Module)] = ["gate_proj", "up_proj", "down_proj"].map { name in
            let stem = "model.layers.1.mlp.switch_mlp.\(name)"
            return (name, QuantizedSwitchLinear(inputDims: 512, outputDims: 512, numExperts: 8,
                weight: weights[stem + ".weight"]!, scales: weights[stem + ".scales"]!,
                biases: weights[stem + ".biases"], groupSize: specs[name]!.group,
                bits: specs[name]!.bits, mode: specs[name]!.mode))
        }
        try reference.update(modules: ModuleChildren.unflattened(projections), verify: .all)
        func expertOutput(_ input: MLXArray, _ expert: Int) -> MLXArray {
            func project(_ value: MLXArray, _ name: String) -> MLXArray {
                let stem = "model.layers.1.mlp.switch_mlp.\(name)"
                return quantizedMM(value, weights[stem + ".weight"]![expert],
                    scales: weights[stem + ".scales"]![expert],
                    biases: weights[stem + ".biases"].map { $0[expert] }, groupSize: specs[name]!.group,
                    bits: specs[name]!.bits, mode: specs[name]!.mode)
            }
            return project(silu(project(input, "gate_proj")) * project(input, "up_proj"), "down_proj")
        }
        for storage in [MixedQuantizedExpertCatalog.Storage.mapped, .resident] {
            let module = try MixedQuantizedSwitchGLU(catalog: source, layer: 1,
                inputDimensions: 512, storage: storage)
            for tokens in [1, 8] {
                let x = (sin(MLXArray(0..<(tokens * 512)).asType(.float32) * Float(0.017)) * Float(0.1))
                    .asType(dtype).reshaped(1, tokens, 512)
                let order: [Int32] = (0..<(tokens * 8)).map { Int32(($0 * 3 + $0 / 8) % 8) }
                let routes = MLXArray(order).reshaped(1, tokens, 8)
                let expected: MLXArray
                if tokens == 1 {
                    expected = stacked(order.map { expertOutput(x[0], Int($0)) })
                        .reshaped(1, tokens, 8, 512)
                } else if storage == .resident {
                    // Compare prefill to the native batched SwitchGLU, not
                    // separate QMV calls with a different reduction order.
                    expected = reference(x, routes)
                } else {
                    // Every expert receives each token once in this route
                    // fixture. Compute native QMM independently by expert,
                    // then select each result in the requested router order.
                    let byExpert = (0..<8).map { expertOutput(x[0], $0) }
                    expected = stacked(order.enumerated().map { position, expert in
                        byExpert[Int(expert)][position / 8]
                    }).reshaped(1, tokens, 8, 512)
                }
                let actual = module(x, routes)
                #expect(arrayEqual(expected, actual).item(Bool.self),
                    "specs=\(specs) dtype=\(dtype) storage=\(storage) tokens=\(tokens) maxAbs=\(abs(expected.asType(.float32) - actual.asType(.float32)).max().item(Float.self))")
                if tokens == 1 && [3, 5, 6].contains(specs["gate_proj"]!.bits) {
                    // Existing specialized kernels must decline odd-bit gates.
                    let kernel = MixedQuantizedExpertKernel()
                    #expect(kernel.fusedGateUp(x, indices: routes, gate: bank.gate, up: bank.up) == nil)
                    #expect(kernel.pairedGateUp(x, indices: routes, gate: bank.gate, up: bank.up) == nil)
                }
            }
        }
    }

    @Test(arguments: [false, true], [64, 128])
    func fusedGateUpMatchesNativeAtProductionShape(mxGate: Bool, upGroup: Int) throws {
        let width = 4096, hidden = 2048
        func projection(mx: Bool, group: Int, phase: Float) -> MixedQuantizedExpertCatalog.Projection {
            let positions = MLXArray(0..<(2 * hidden * width)).asType(.float32)
            let source = (sin(positions * Float(0.173) + phase) * Float(0.03))
                .reshaped(2, hidden, width).asType(.bfloat16)
            let mode: QuantizationMode = mx ? .mxfp4 : .affine
            let q = quantized(source, groupSize: group, bits: mx ? 4 : 2, mode: mode)
            eval(q.wq, q.scales)
            return .init(weight: q.wq, scales: q.scales, biases: q.biases,
                bits: mx ? 4 : 2, groupSize: group, mode: mode)
        }
        let gate = projection(mx: mxGate, group: mxGate ? 32 : 64, phase: 0)
        let up = projection(mx: false, group: upGroup, phase: 1.71)
        let indices = MLXArray([Int32(1), 0, 1, 1, 0, 0, 1, 0]).reshaped(1, 1, 8)
        let kernel = MixedQuantizedExpertKernel()
        for magnitude: Float in [0, 1, 16] {
            // Deliberately strided input, duplicate routes, and nonlinear/saturated SiLU ranges.
            let input = (sin(MLXArray(0..<(2 * width)).asType(.float32) * Float(0.071)) * magnitude)
                .asType(.bfloat16)[.stride(by: 2)].reshaped(1, 1, width)
            func native(_ p: MixedQuantizedExpertCatalog.Projection) -> MLXArray {
                gatherQuantizedMM(expandedDimensions(input, axes: [-2, -3]), p.weight,
                    scales: p.scales, biases: p.biases, rhsIndices: indices,
                    groupSize: p.groupSize, bits: p.bits, mode: p.mode)
            }
            let expected = silu(native(gate)) * native(up)
            let result = try #require(kernel.fusedGateUp(input, indices: indices, gate: gate, up: up))
            #expect(arrayEqual(expected, result).item(Bool.self),
                "magnitude=\(magnitude) maxAbs=\(abs(expected.asType(.float32) - result.asType(.float32)).max().item(Float.self))")
        }
    }

    @Test func fusedDownReduceMatchesNativeAtProductionShape() throws {
        let width = 2048, output = 4096, experts = 8
        let kernel = MixedQuantizedExpertKernel()
        let positions = MLXArray(0..<(experts * output * width)).asType(.float32)
        let weights = (sin(positions * Float(0.173)) * Float(0.03))
            .reshaped(experts, output, width).asType(.bfloat16)
        let packed = quantized(weights, groupSize: 64, bits: 2, mode: .affine)
        let bias = try #require(packed.biases)
        eval(packed.wq, packed.scales, bias)
        let scores = MLXArray([Float(0.1), 0.2, 0.05, 0.12, 0.08, 0.15, 0.19, 0.11])
            .reshaped(1, 1, 8)
        for routes: [Int32] in [[0, 1, 2, 3, 4, 5, 6, 7], [7, 2, 7, 0, 3, 2, 1, 7]] {
            let indices = MLXArray(routes).reshaped(1, 1, 8)
            for magnitude: Float in [0, 1, 16] {
                let x = (sin(MLXArray(0..<(16 * width)).asType(.float32) * Float(0.071)) * magnitude)
                    .asType(.bfloat16)[.stride(by: 2)].reshaped(1, 1, 8, 1, width)
                let down = gatherQuantizedMM(x, packed.wq, scales: packed.scales, biases: bias,
                    rhsIndices: indices, groupSize: 64, bits: 2, mode: .affine)
                let expected = (down.squeezed(axis: -2).asType(.float32)
                    * scores[.ellipsis, .newAxis]).sum(axis: -2).asType(.bfloat16)
                let projection = MixedQuantizedExpertCatalog.Projection(weight: packed.wq,
                    scales: packed.scales, biases: bias, bits: 2, groupSize: 64, mode: .affine)
                let actual = try #require(kernel.fusedDownReduce(x, indices: indices,
                    scores: scores, down: projection))
                #expect(arrayEqual(expected, actual).item(Bool.self),
                    "routes=\(routes) magnitude=\(magnitude) maxAbs=\(abs(expected.asType(.float32) - actual.asType(.float32)).max().item(Float.self))")
            }
        }
    }

    @Test func weightedResidentDecodeAndFallbackContracts() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        var weights: [String: MLXArray] = [:]
        for name in ["gate_proj", "up_proj", "down_proj"] {
            let mx = name == "gate_proj"
            let phase: Float = mx ? 0 : name == "up_proj" ? 1.3 : 2.7
            let values = sin(MLXArray(0..<(8 * 512 * 512)).asType(.float32) * Float(0.031) + phase)
            let source = (values * Float(0.01)).reshaped(8, 512, 512).asType(.bfloat16)
            let q = quantized(source, groupSize: mx ? 32 : 64, bits: mx ? 4 : 2,
                mode: mx ? .mxfp4 : .affine)
            let stem = "model.layers.1.mlp.switch_mlp.\(name)"
            weights[stem + ".weight"] = q.wq
            weights[stem + ".scales"] = q.scales
            weights[stem + ".biases"] = q.biases
        }
        try saveAligned(weights, to: directory.appendingPathComponent("model.safetensors"))
        try JSONSerialization.data(withJSONObject: ["weight_map": weights.mapValues { _ in "model.safetensors" }])
            .write(to: directory.appendingPathComponent("model.safetensors.index.json"))
        let catalog = try MixedQuantizedExpertCatalog(directory: directory, layerIndices: [1],
            expertCount: 8, inputDimensions: 512, hiddenDimensions: 512)
        let module = try MixedQuantizedSwitchGLU(catalog: catalog, layer: 1,
            inputDimensions: 512, storage: .resident)
        let x = sin(MLXArray(0..<512).asType(.float32)).asType(.bfloat16).reshaped(1, 1, 512)
        let routes = MLXArray([Int32(7), 1, 0, 7, 3, 1, 4, 2]).reshaped(1, 1, 8)
        let scores = MLXArray([Float(0.1), 0.2, 0.05, 0.12, 0.08, 0.15, 0.19, 0.11]).reshaped(1, 1, 8)
        let result = module.fusedWeightedOutput(x, routes, scores: scores)
        if RuntimeEnvironment.value("VMLX_MIMO_FUSED_DOWN_REDUCE") == "1" {
            let actual = try #require(result)
            let expected = (module(x, routes).asType(.float32) * scores[.ellipsis, .newAxis])
                .sum(axis: -2).asType(.bfloat16)
            #expect(arrayEqual(actual, expected).item(Bool.self))
        } else {
            #expect(result == nil)
        }
        let batch = broadcast(x, to: [1, 2, 512])
        #expect(module.fusedWeightedOutput(batch, routes, scores: scores) == nil)
        #expect(module.fusedWeightedOutput(x, routes, scores: scores.asType(.bfloat16)) == nil)
        let mapped = try MixedQuantizedSwitchGLU(catalog: catalog, layer: 1,
            inputDimensions: 512, storage: .mapped)
        #expect(mapped.fusedWeightedOutput(x, routes, scores: scores) == nil)
        let bank = try catalog.loadResidentLayer(layer: 1)
        let kernel = MixedQuantizedExpertKernel()
        let activated = MLXArray.zeros([1, 1, 8, 1, 512], dtype: .bfloat16)
        #expect(kernel.fusedDownReduce(activated.asType(.float32), indices: routes,
            scores: scores, down: bank.down) == nil)
        #expect(kernel.fusedDownReduce(activated, indices: routes.asType(.float32),
            scores: scores, down: bank.down) == nil)
    }

    @Test(arguments: [QuantizationMode.affine, .mxfp4, .mxfp8])
    func indexedModelLoadAndRingCacheParity(gateMode: QuantizationMode) throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let gateBits = gateMode == .affine ? 3 : gateMode == .mxfp4 ? 4 : 8
        var object = try #require(JSONSerialization.jsonObject(
            with: MiMoV26RuntimeTests.configuration(moe: true)) as? [String: Any])
        var quantization = try #require(object["quantization"] as? [String: Any])
        quantization["model.layers.1.mlp.switch_mlp.gate_proj"] =
            ["bits": gateBits, "group_size": 32, "mode": gateMode.rawValue]
        object["quantization"] = quantization
        let data = try JSONSerialization.data(withJSONObject: object)
        try data.write(to: directory.appendingPathComponent("config.json"))
        let original = try MiMoV26RuntimeTests.model(moe: true)
        try original.update(parameters: ModuleParameters.unflattened(original.parameters().flattened().map {
            ($0.0, $0.0.contains(".mlp.gate.") ? $0.1 + Float(0.001337) : $0.1.asType(.bfloat16))
        }), verify: .all)
        MLXNN.quantize(model: original, filter: { path, module in
            guard module is Quantizable else { return nil }
            if path.hasSuffix("switch_mlp.gate_proj") { return (32, gateBits, gateMode) }
            if path.hasSuffix("switch_mlp.up_proj") { return (64, 2, .affine) }
            if path.hasSuffix("self_attn.o_proj") { return (32, 8, .affine) }
            return (64, 8, .affine)
        })
        let weights = Dictionary(uniqueKeysWithValues: original.parameters().flattened())
        try saveAligned(weights, to: directory.appendingPathComponent("model.safetensors"))
        try JSONSerialization.data(withJSONObject: ["weight_map": weights.mapValues { _ in "model.safetensors" }])
            .write(to: directory.appendingPathComponent("model.safetensors.index.json"))
        let loaded = try MiMoV26RuntimeTests.model(moe: true)
        try loaded.configure(modelDirectory: directory)
        #expect(!loaded.supportsWholeForwardCompilation)
        #expect(loaded.parameters().flattened().allSatisfy { !$0.0.contains(".switch_mlp.") })
        let base = try JSONDecoder().decode(BaseConfiguration.self, from: data)
        try loadWeights(modelDirectory: directory, model: loaded,
                        perLayerQuantization: base.perLayerQuantization)
        let reference = original.newCache(parameters: nil), actual = loaded.newCache(parameters: nil)
        // The long warm chunk crosses the grouped-expert threshold and wraps
        // the rotating cache; following tokens verify restored route ordering.
        for tokens in [[1, 2, 3], Array(10..<50), [4], [5], [6], [7], [8], [9]] {
            let input = MLXArray(tokens).reshaped(1, -1)
            let expected = original(input, cache: reference), result = loaded(input, cache: actual)
            #expect(arrayEqual(expected, result).item(Bool.self),
                    "maxAbs=\(abs(expected.asType(.float32)-result.asType(.float32)).max().item(Float.self))")
        }
    }
}
