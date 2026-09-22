// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXLMCommon
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

    @Test(arguments: [MixedQuantizedExpertCatalog.Storage.mapped, .resident])
    func eightRegionMetalKernelMatchesNativeProjections(storage: MixedQuantizedExpertCatalog.Storage) throws {
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
                let group = mx ? 32 : name == "down_proj" ? 128 : 64
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
            #expect(module.parameters().flattened().isEmpty)
            let reference = order.map { index in
                func project(_ x: MLXArray, _ name: String) -> MLXArray {
                    let mx = name == "gate_proj" && mxGate
                    let stem = "model.layers.1.mlp.switch_mlp.\(name)"
                    return quantizedMM(x, weights[stem + ".weight"]![Int(index)],
                        scales: weights[stem + ".scales"]![Int(index)],
                        biases: weights[stem + ".biases"].map { $0[Int(index)] },
                        groupSize: mx ? 32 : name == "down_proj" ? 128 : 64,
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

    @Test func indexedModelLoadAndRingCacheParity() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let data = try MiMoV26RuntimeTests.configuration(moe: true)
        try data.write(to: directory.appendingPathComponent("config.json"))
        let original = try MiMoV26RuntimeTests.model(moe: true)
        try original.update(parameters: ModuleParameters.unflattened(original.parameters().flattened().map {
            ($0.0, $0.0.contains(".mlp.gate.") ? $0.1 + Float(0.001337) : $0.1.asType(.bfloat16))
        }), verify: .all)
        MLXNN.quantize(model: original, filter: { path, module in
            guard module is Quantizable else { return nil }
            if path.hasSuffix("switch_mlp.gate_proj") { return (32, 4, .mxfp4) }
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
        for tokens in [[1, 2, 3], [4], [5], [6], [7], [8], [9]] {
            let input = MLXArray(tokens).reshaped(1, -1)
            let expected = original(input, cache: reference), result = loaded(input, cache: actual)
            #expect(arrayEqual(expected, result).item(Bool.self),
                    "maxAbs=\(abs(expected.asType(.float32)-result.asType(.float32)).max().item(Float.self))")
        }
    }
}
