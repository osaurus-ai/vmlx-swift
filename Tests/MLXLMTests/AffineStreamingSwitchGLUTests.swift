// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLMCommon

@Suite("native affine active-expert SSD routing", .serialized)
struct AffineStreamingSwitchGLUTests {
    private final class LoaderOrderingModel: Module, LanguageModel,
        SafetensorsLoadKeyExcluding, SafetensorsModelDirectoryConfigurable
    {
        private(set) var configuredDirectory: URL?
        private(set) var exclusionObservedConfiguration = true

        var kvHeads: [Int] { [] }
        var vocabularySize: Int { 1 }

        func configureSafetensorsModelDirectory(_ modelDirectory: URL) throws {
            configuredDirectory = modelDirectory.resolvingSymlinksInPath()
        }

        func excludeFromGenericSafetensorsLoad(key: String) -> Bool {
            exclusionObservedConfiguration =
                exclusionObservedConfiguration && configuredDirectory != nil
            return true
        }

        func prepare(
            _ input: LMInput, cache: [KVCache], windowSize: Int?
        ) throws -> PrepareResult {
            .tokens(input.text)
        }

        func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
            MLXArray.zeros([1, 1, 1])
        }

        func newCache(parameters: GenerateParameters?) -> [KVCache] { [] }
    }

    private final class ConcurrentResults: @unchecked Sendable {
        private let lock = NSLock()
        private(set) var shapes: [[Int]] = []
        private(set) var failures: [String] = []

        func append(shape: [Int]) {
            lock.lock()
            shapes.append(shape)
            lock.unlock()
        }

        func append(failure: String) {
            lock.lock()
            failures.append(failure)
            lock.unlock()
        }
    }

    private struct Fixture {
        let directory: URL
        let tensors: [String: MLXArray]
    }

    private func makeFixture(numExperts: Int = 4, dimensions: Int = 32) throws -> Fixture {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("affine-streaming-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)

        func dense(expert: Int, projection: Int) -> MLXArray {
            let count = dimensions * dimensions
            let values = (0 ..< count).map { index -> Float in
                let centered = Float((index * 13 + expert * 17 + projection * 23) % 101) - 50
                return centered / Float(200 + projection * 40)
            }
            return MLXArray(values, [dimensions, dimensions])
        }

        var tensors: [String: MLXArray] = [:]
        for (projectionIndex, projection) in ["gate_proj", "up_proj", "down_proj"].enumerated() {
            var weights: [MLXArray] = []
            var scales: [MLXArray] = []
            var biases: [MLXArray] = []
            for expert in 0 ..< numExperts {
                let quantized = MLX.quantized(
                    dense(expert: expert, projection: projectionIndex),
                    groupSize: dimensions, bits: 4, mode: .affine)
                weights.append(quantized.wq)
                scales.append(quantized.scales.asType(.float16))
                biases.append(quantized.biases!.asType(.float16))
            }
            let base = "language_model.layers.0.mlp.switch_mlp.\(projection)"
            tensors[base + ".weight"] = stacked(weights, axis: 0)
            tensors[base + ".scales"] = stacked(scales, axis: 0)
            tensors[base + ".biases"] = stacked(biases, axis: 0)
        }
        MLX.eval(Array(tensors.values))

        let shardName = "model.safetensors"
        let shardURL = directory.appendingPathComponent(shardName)
        try MLX.save(arrays: tensors, url: shardURL)
        // Production Qwen4 uses JangPressPrestacker's alignment overlay.
        // Reproduce that contract here by padding the safetensors JSON header
        // so the data segment begins on a page boundary without changing any
        // tensor-relative data_offsets.
        let saved = try Data(contentsOf: shardURL)
        let oldHeaderLength = saved.prefix(8).withUnsafeBytes {
            $0.loadUnaligned(as: UInt64.self).littleEndian
        }
        var header = saved.subdata(in: 8 ..< (8 + Int(oldHeaderLength)))
        let padding = (4096 - ((8 + header.count) % 4096)) % 4096
        header.append(Data(repeating: 0x20, count: padding))
        var aligned = Data()
        var newHeaderLength = UInt64(header.count).littleEndian
        aligned.append(contentsOf: withUnsafeBytes(of: &newHeaderLength) { Array($0) })
        aligned.append(header)
        aligned.append(saved.suffix(from: 8 + Int(oldHeaderLength)))
        try aligned.write(to: shardURL)
        let weightMap = Dictionary(uniqueKeysWithValues: tensors.keys.map { ($0, shardName) })
        let index = ["weight_map": weightMap]
        try JSONSerialization.data(withJSONObject: index, options: [.sortedKeys])
            .write(to: directory.appendingPathComponent("model.safetensors.index.json"))
        return Fixture(directory: directory, tensors: tensors)
    }

    private func makeDeepseekFixture(
        numExperts: Int = 4, dimensions: Int = 32
    ) throws -> Fixture {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("dsv4-affine-streaming-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)

        func dense(expert: Int, projection: Int) -> MLXArray {
            let values = (0 ..< dimensions * dimensions).map { index -> Float in
                let centered = Float((index * 11 + expert * 19 + projection * 29) % 97) - 48
                return centered / Float(180 + projection * 30)
            }
            return MLXArray(values, [dimensions, dimensions])
        }

        var tensors: [String: MLXArray] = [:]
        for expert in 0 ..< numExperts {
            for (projectionIndex, source) in ["w1", "w3", "w2"].enumerated() {
                let bits = [3, 4, 2][projectionIndex]
                let quantized = MLX.quantized(
                    dense(expert: expert, projection: projectionIndex),
                    groupSize: dimensions, bits: bits, mode: .affine)
                let base = "layers.0.ffn.experts.\(expert).\(source)"
                tensors[base + ".weight"] = quantized.wq
                tensors[base + ".scales"] = quantized.scales.asType(.float16)
                tensors[base + ".biases"] = quantized.biases!.asType(.float16)
            }
        }
        MLX.eval(Array(tensors.values))

        let shardName = "model.safetensors"
        let shardURL = directory.appendingPathComponent(shardName)
        try MLX.save(arrays: tensors, url: shardURL)
        let saved = try Data(contentsOf: shardURL)
        let oldHeaderLength = saved.prefix(8).withUnsafeBytes {
            $0.loadUnaligned(as: UInt64.self).littleEndian
        }
        var header = saved.subdata(in: 8 ..< (8 + Int(oldHeaderLength)))
        let padding = (4096 - ((8 + header.count) % 4096)) % 4096
        header.append(Data(repeating: 0x20, count: padding))
        var aligned = Data()
        var newHeaderLength = UInt64(header.count).littleEndian
        aligned.append(contentsOf: withUnsafeBytes(of: &newHeaderLength) { Array($0) })
        aligned.append(header)
        aligned.append(saved.suffix(from: 8 + Int(oldHeaderLength)))
        try aligned.write(to: shardURL)
        let index = [
            "weight_map": Dictionary(
                uniqueKeysWithValues: tensors.keys.map { ($0, shardName) })
        ]
        try JSONSerialization.data(withJSONObject: index, options: [.sortedKeys])
            .write(to: directory.appendingPathComponent("model.safetensors.index.json"))
        return Fixture(directory: directory, tensors: tensors)
    }

    @Test("loader exclusion is exact and does not swallow shared experts")
    func exactLoaderExclusion() {
        #expect(
            AffineStreamingExpertCatalog.isRoutedTensorKey(
                "language_model.layers.4.mlp.switch_mlp.gate_proj.weight"))
        #expect(
            AffineStreamingExpertCatalog.isRoutedTensorKey(
                "model.language_model.layers.4.mlp.switch_mlp.down_proj.biases"))
        #expect(
            !AffineStreamingExpertCatalog.isRoutedTensorKey(
                "language_model.layers.4.mlp.shared_expert.gate_proj.weight"))
        #expect(
            !AffineStreamingExpertCatalog.isRoutedTensorKey(
                "language_model.layers.4.mlp.switch_mlp.gate_proj.tq_packed"))
        #expect(
            AffineStreamingExpertCatalog.isDeepseekV4RoutedTensorKey(
                "layers.4.ffn.experts.255.w2.weight"))
        #expect(
            !AffineStreamingExpertCatalog.isDeepseekV4RoutedTensorKey(
                "layers.4.ffn.shared_experts.w2.weight"))
        #expect(
            !AffineStreamingExpertCatalog.isDeepseekV4RoutedTensorKey(
                "layers.4.ffn.experts.255.w2.tq_packed"))
    }

    @Test("generic loader configures model-owned readers before excluding tensors")
    func loaderConfiguresBeforeExclusion() throws {
        let fixture = try makeDeepseekFixture(numExperts: 1)
        defer { try? FileManager.default.removeItem(at: fixture.directory) }
        let model = LoaderOrderingModel()

        try loadWeights(modelDirectory: fixture.directory, model: model)

        #expect(
            model.configuredDirectory
                == fixture.directory.resolvingSymlinksInPath())
        #expect(model.exclusionObservedConfiguration)
    }

    @Test("DSV4 per-expert files preserve scored-before-down numerical order")
    func deepseekPerExpertScoredParity() throws {
        try MLXMetalTestLock.withLock {
            let fixture = try makeDeepseekFixture()
            defer { try? FileManager.default.removeItem(at: fixture.directory) }
            let catalog = AffineStreamingExpertCatalog(
                numExperts: 4, inputDims: 32, hiddenDims: 32,
                layout: .deepseekV4PerExpert)
            try catalog.configure(modelDirectory: fixture.directory, layerCount: 1)
            let streaming = AffineStreamingSwitchGLU(
                inputDims: 32, hiddenDims: 32, numExperts: 4, layerIndex: 0,
                catalog: catalog, cacheExpertLimit: 2, prefillChunkSize: 1,
                scoredGlue: { gate, up, scores in
                    (silu(gate.asType(.float32)) * up.asType(.float32)
                        * scores.asType(.float32)[.ellipsis, .newAxis, .newAxis])
                        .asType(gate.dtype)
                })

            let input = (MLXArray(0 ..< 32).asType(.float32) / 64).reshaped([1, 1, 32])
            let indices = MLXArray([Int32(3), Int32(1)], [1, 1, 2]).asType(.uint32)
            let scores = MLXArray([Float(0.65), Float(0.35)], [1, 1, 2])
            let actual = streaming.routed(
                input, indices: indices, preDownScores: scores)

            func bank(_ source: String, _ suffix: String) -> MLXArray {
                stacked(
                    (0 ..< 4).map {
                        fixture.tensors["layers.0.ffn.experts.\($0).\(source).\(suffix)"]!
                    }, axis: 0)
            }
            let expanded = expandedDimensions(input, axes: [-2, -3])
            func project(_ source: String, _ value: MLXArray) -> MLXArray {
                let bits = source == "w1" ? 3 : (source == "w2" ? 2 : 4)
                return MLX.gatherQuantizedMM(
                    value, bank(source, "weight"),
                    scales: bank(source, "scales"), biases: bank(source, "biases"),
                    rhsIndices: indices, transpose: true,
                    groupSize: 32, bits: bits, mode: .affine)
            }
            let gate = project("w1", expanded)
            let up = project("w3", expanded)
            let activated =
                (silu(gate.asType(.float32)) * up.asType(.float32)
                * scores.asType(.float32)[.ellipsis, .newAxis, .newAxis]).asType(gate.dtype)
            let expected = squeezed(project("w2", activated), axis: -2)
            MLX.eval(actual, expected)

            #expect(actual.shape == [1, 1, 2, 32])
            #expect(MLX.allClose(actual, expected, rtol: 1e-5, atol: 1e-5).item(Bool.self))
            #expect(streaming.cachedExpertCount == 2)
        }
    }

    @Test("DSV4 catalog fails closed when one routed tensor is absent")
    func deepseekIncompleteCatalogFailsClosed() throws {
        let fixture = try makeDeepseekFixture()
        defer { try? FileManager.default.removeItem(at: fixture.directory) }
        let indexURL = fixture.directory.appendingPathComponent(
            "model.safetensors.index.json")
        let data = try Data(contentsOf: indexURL)
        var index = try #require(
            JSONSerialization.jsonObject(with: data) as? [String: Any])
        var weightMap = try #require(index["weight_map"] as? [String: String])
        let missing = "layers.0.ffn.experts.3.w2.biases"
        #expect(weightMap.removeValue(forKey: missing) != nil)
        index["weight_map"] = weightMap
        try JSONSerialization.data(withJSONObject: index, options: [.sortedKeys])
            .write(to: indexURL)

        let catalog = AffineStreamingExpertCatalog(
            numExperts: 4, inputDims: 32, hiddenDims: 32,
            layout: .deepseekV4PerExpert)
        #expect(throws: AffineStreamingExpertError.self) {
            try catalog.configure(modelDirectory: fixture.directory, layerCount: 1)
        }
    }

    @Test("exact expert regions preserve projection geometry")
    func exactRegionGeometry() throws {
        let fixture = try makeFixture()
        defer { try? FileManager.default.removeItem(at: fixture.directory) }
        let catalog = AffineStreamingExpertCatalog(
            numExperts: 4, inputDims: 32, hiddenDims: 32)
        try catalog.configure(modelDirectory: fixture.directory, layerCount: 1)
        let expert = try catalog.loadExpert(layer: 0, expert: 3)
        #expect(expert.gate.weight.shape == [32, 4])
        #expect(expert.gate.scales.shape == [32, 1])
        #expect(expert.gate.biases.shape == [32, 1])
        #expect(expert.gate.bits == 4)
        #expect(expert.gate.groupSize == 32)
    }

    @Test("selected SSD experts match the full native affine bank")
    func numericalParityAndBoundedCache() throws {
        let fixture = try makeFixture()
        defer { try? FileManager.default.removeItem(at: fixture.directory) }
        let catalog = AffineStreamingExpertCatalog(
            numExperts: 4, inputDims: 32, hiddenDims: 32)
        try catalog.configure(modelDirectory: fixture.directory, layerCount: 1)
        let streaming = AffineStreamingSwitchGLU(
            inputDims: 32, hiddenDims: 32, numExperts: 4, layerIndex: 0,
            catalog: catalog, cacheExpertLimit: 2, prefillChunkSize: 1)

        let input = (MLXArray(0 ..< 32).asType(.float32) / 64).reshaped([1, 1, 32])
        let indices = MLXArray([Int32(3), Int32(1)], [1, 1, 2]).asType(.uint32)
        let scores = MLXArray([Float(0.65), Float(0.35)], [1, 1, 2])
        let actual = streaming.reduced(input, indices: indices, scores: scores)

        func tensor(_ projection: String, _ suffix: String) -> MLXArray {
            fixture.tensors[
                "language_model.layers.0.mlp.switch_mlp.\(projection).\(suffix)"]!
        }
        let expanded = expandedDimensions(input, axes: [-2, -3])
        func project(_ projection: String, _ value: MLXArray) -> MLXArray {
            MLX.gatherQuantizedMM(
                value, tensor(projection, "weight"),
                scales: tensor(projection, "scales"),
                biases: tensor(projection, "biases"), rhsIndices: indices,
                transpose: true, groupSize: 32, bits: 4, mode: .affine)
        }
        let activated =
            silu(project("gate_proj", expanded))
            * project("up_proj", expanded)
        let routed = squeezed(project("down_proj", activated), axis: -2)
        let expected = (routed * scores[.ellipsis, .newAxis]).sum(axis: -2)
        MLX.eval(expected)

        #expect(MLX.allClose(actual, expected, rtol: 1e-5, atol: 1e-5).item(Bool.self))
        #expect(streaming.cachedExpertCount == 2)

        _ = streaming.reduced(
            input,
            indices: MLXArray([Int32(0), Int32(2)], [1, 1, 2]).asType(.uint32),
            scores: scores)
        #expect(streaming.cachedExpertCount == 2)
    }

    @Test("catalog supports concurrent exact-region opens")
    func concurrentRegionAccess() throws {
        let fixture = try makeFixture()
        defer { try? FileManager.default.removeItem(at: fixture.directory) }
        let catalog = AffineStreamingExpertCatalog(
            numExperts: 4, inputDims: 32, hiddenDims: 32)
        try catalog.configure(modelDirectory: fixture.directory, layerCount: 1)
        let results = ConcurrentResults()
        DispatchQueue.concurrentPerform(iterations: 16) { iteration in
            do {
                let arrays = try catalog.loadExpert(layer: 0, expert: iteration % 4)
                results.append(shape: arrays.down.weight.shape)
            } catch {
                results.append(failure: String(describing: error))
            }
        }
        #expect(results.failures.isEmpty)
        #expect(results.shapes.count == 16)
        #expect(results.shapes.allSatisfy { $0 == [32, 4] })
    }

    @Test("optional real Qwen4 overlay validates every affine expert descriptor")
    func optionalRealBundleGeometry() throws {
        guard let path = ProcessInfo.processInfo.environment["QWEN4_REAL_MODEL"],
            !path.isEmpty
        else { return }
        let catalog = AffineStreamingExpertCatalog(
            numExperts: 512, inputDims: 2560, hiddenDims: 640)
        try catalog.configure(
            modelDirectory: URL(fileURLWithPath: path, isDirectory: true),
            layerCount: 48)
        let first = try catalog.loadExpert(layer: 0, expert: 0)
        let last = try catalog.loadExpert(layer: 47, expert: 511)
        #expect(first.gate.weight.shape == [640, 320])
        #expect(first.down.weight.shape == [2560, 80])
        #expect(last.gate.groupSize == 64)
        #expect(last.gate.bits == 4)
        #expect(last.down.groupSize == 64)
        #expect(last.down.bits == 4)
    }

    @Test("optional real DSV4 bundle validates every per-expert affine descriptor")
    func optionalRealDeepseekGeometry() throws {
        guard let path = ProcessInfo.processInfo.environment["DSV4_REAL_MODEL"],
            !path.isEmpty
        else { return }
        let catalog = AffineStreamingExpertCatalog(
            numExperts: 256, inputDims: 4096, hiddenDims: 2048,
            layout: .deepseekV4PerExpert)
        try catalog.configure(
            modelDirectory: URL(fileURLWithPath: path, isDirectory: true),
            layerCount: 43)
        let first = try catalog.loadExpert(layer: 0, expert: 0)
        let last = try catalog.loadExpert(layer: 42, expert: 255)
        #expect(first.gate.weight.shape[0] == 2048)
        #expect(first.down.weight.shape[0] == 4096)
        #expect([1, 2, 3, 4, 5, 6, 8].contains(first.gate.bits))
        #expect([1, 2, 3, 4, 5, 6, 8].contains(last.down.bits))
        #expect(first.gate.groupSize > 0)
        #expect(last.down.groupSize > 0)
    }
}
