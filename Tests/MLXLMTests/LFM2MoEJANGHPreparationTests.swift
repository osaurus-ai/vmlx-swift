import Cmlx
import Foundation
import MLX
import MLXNN
import XCTest
@testable import MLXLLM
@testable import MLXLMCommon

final class LFM2MoEJANGHPreparationTests: XCTestCase {
    private static let parent = "model.layers.1.feed_forward.switch_mlp"
    private static let shard = "model.safetensors"
    private static let index = "model.safetensors.index.json"
    private static func architecture() -> [String: Any] {
        ["model_type":"lfm2_moe", "vocab_size":64, "hidden_size":32,
         "intermediate_size":64, "moe_intermediate_size":32, "num_hidden_layers":2,
         "num_experts":2, "num_experts_per_tok":1, "norm_topk_prob":true,
         "num_attention_heads":2, "num_key_value_heads":1, "max_position_embeddings":131072,
         "use_expert_bias":true, "num_dense_layers":1, "norm_eps":0.00001,
         "conv_bias":false, "conv_L_cache":3, "layer_types":["conv","full_attention"],
         "rope_parameters":["rope_theta":10000000], "routed_scaling_factor":1.0]
    }
    private static func temporaryDirectory() throws -> URL {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:directory,withIntermediateDirectories:true)
        return directory
    }
    private static func fixture(_ directory: URL) throws -> Data {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        var books: [String: Any] = [:]
        var quant: [String: Any] = ["mode": "affine", "bits": 8, "group_size": 32]
        var headers: [String: Any] = [:]
        var map: [String: String] = [:]
        var payload = Data()
        for (role, bits) in [("gate_proj", 2), ("up_proj", 2), ("down_proj", 3)] {
            let module = parent + "." + role
            books[String(bits)] = ["alpha": 0.25, "beta": 0,
                                   "levels": (0..<(1 << bits)).map { (Double($0) - Double((1 << bits) - 1) / 2) * 0.25 }]
            quant[module] = ["mode": "jangtq2", "bits": bits, "rotation": "hadamard32"]
            for (suffix, dtype, shape, data) in [
                ("tq2_packed", "U32", [2, 32, bits], Data(repeating: 0, count: 2 * 32 * bits * 4)),
                ("tq2_scales", "F16", [2, 32], Data((0..<64).flatMap { _ in [UInt8(0), UInt8(0x3c)] })),
            ] {
                let name = module + "." + suffix
                headers[name] = ["dtype": dtype, "shape": shape, "data_offsets": [payload.count, payload.count + data.count]]
                map[name] = shard
                payload.append(data)
            }
        }
        var header = try JSONSerialization.data(withJSONObject: headers, options: .sortedKeys)
        XCTAssertLessThan(header.count, 4088)
        header.append(Data(repeating: 0x20, count: 4088 - header.count))
        var length = UInt64(header.count).littleEndian
        var file = withUnsafeBytes(of: &length) { Data($0) }
        file.append(header)
        file.append(payload)
        try file.write(to: directory.appendingPathComponent(shard), options: .atomic)
        try JSONSerialization.data(withJSONObject: ["weight_map": map]).write(
            to: directory.appendingPathComponent(index), options: .atomic)
        var config: [String: Any] = ["model_type": "lfm2_moe", "jangtq": ["version": 2, "packing": "lsb-bitstream", "scale_dtype": "float16",
                                                       "codebook_family": "odd-cubic", "rotation": "hadamard32", "codebooks": books],
                                     "quantization": quant]
        config.merge(architecture()) { _, new in new }
        return try JSONSerialization.data(withJSONObject: config)
    }

    private static func writeSummary(_ directory: URL, mutate: ([String: Any]) -> [String: Any] = { $0 }) throws {
        let summary: [String: Any] = ["method":"jang", "bits":8, "group_size":32,
            "modes":["affine","jangtq2"], "bit_widths_used":[2,3,8],
            "routed_experts":["gate_up_bits":[2], "down_bits":[3], "mode":"jangtq2"],
            "source_precision":["expert_bias (f32)"], "calibration":"native traces"]
        let sidecar: [String: Any] = ["format":"jang", "format_version":"2.0",
                                   "quantization":mutate(summary)]
        try JSONSerialization.data(withJSONObject:sidecar).write(to:directory.appendingPathComponent("jang_config.json"))
    }

    func testExactLFMHeaderAdmissionWithoutPayloadMapping() throws {
        let directory = try Self.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at:directory) }
        let config = try Self.fixture(directory)
        try Self.writeSummary(directory)
        let before = mlx_safetensors_mmap_tracked_buffer_bytes()
        let preparation = try XCTUnwrap(LFM2MoEJANGHPreparation.loadIfDeclared(directory:directory,configurationData:config))
        XCTAssertEqual(preparation.banks.moduleByLayer,[1:Self.parent])
        XCTAssertEqual(preparation.banks.excludedTensorNames.count,6)
        XCTAssertEqual(mlx_safetensors_mmap_tracked_buffer_bytes(),before)
        let ordinary = try JSONSerialization.jsonObject(with:preparation.banks.ordinaryConfiguration) as! [String:Any]
        let quant = ordinary["quantization"] as! [String:Any]
        XCTAssertEqual(quant[Self.parent + ".gate_proj"] as? Bool,false)
        XCTAssertEqual(quant[Self.parent + ".down_proj"] as? Bool,false)
    }

    func testDescriptiveSummaryModeOrderAndNewNotesDoNotChangeAdmission() throws {
        let directory = try Self.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let config = try Self.fixture(directory)
        try Self.writeSummary(directory) { summary in
            var changed = summary
            changed["modes"] = ["jangtq2", "affine"]
            changed["future_note"] = "descriptive metadata, not a quantization owner"
            return changed
        }
        XCTAssertNotNil(try LFM2MoEJANGHPreparation.loadIfDeclared(directory: directory, configurationData: config))
    }

    func testWrongNamespaceAndSecondExecutableSidecarRefuse() throws {
        let directory = try Self.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at:directory) }
        let config = try Self.fixture(directory)
        let wrong = String(decoding:config,as:UTF8.self).replacingOccurrences(of:"feed_forward.switch_mlp",with:"mlp.switch_mlp")
        XCTAssertThrowsError(try LFM2MoEJANGHPreparation.loadIfDeclared(directory:directory,configurationData:Data(wrong.utf8)))
        try Self.writeSummary(directory) { summary in
            var changed = summary
            changed[Self.parent + ".gate_proj"] = ["mode":"affine", "bits":8]
            return changed
        }
        XCTAssertThrowsError(try LFM2MoEJANGHPreparation.loadIfDeclared(directory:directory,configurationData:config))
    }

    func testAffineLFMIsNotCustomAndUnknownOrdinaryModeRefusesBeforeMapping() throws {
        let directory = try Self.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at:directory) }
        XCTAssertNil(try LFM2MoEJANGHPreparation.loadIfDeclared(directory:directory,
            configurationData:JSONSerialization.data(withJSONObject:Self.architecture())))
        var config = try JSONSerialization.jsonObject(with:Self.fixture(directory)) as! [String:Any]
        var quant = config["quantization"] as! [String:Any]
        quant["model.embed_tokens"] = ["mode":"unsupported", "bits":8]
        config["quantization"] = quant
        let before = mlx_safetensors_mmap_tracked_buffer_bytes()
        XCTAssertThrowsError(try LFM2MoEJANGHPreparation.loadIfDeclared(directory:directory,
            configurationData:JSONSerialization.data(withJSONObject:config)))
        XCTAssertEqual(mlx_safetensors_mmap_tracked_buffer_bytes(),before)
    }

    func testMappedInstallationHasNoDenseExpertDuplicatesAndPreservesOrdinaryModules() throws {
        try MLXMetalTestLock.withLock {
            let directory = try Self.temporaryDirectory()
            defer { try? FileManager.default.removeItem(at:directory) }
            let preparation = try XCTUnwrap(LFM2MoEJANGHPreparation.loadIfDeclared(directory:directory,
                configurationData:Self.fixture(directory)))
            let model = try preparation.construct(requesting:[.text])
            let names = Set(model.parameters().flattened().map(\.0))
            XCTAssertFalse(names.contains { $0.hasPrefix(Self.parent + ".") })
            XCTAssertTrue(names.contains("model.layers.1.feed_forward.gate.weight"))
            XCTAssertTrue(names.contains("model.layers.1.feed_forward.expert_bias"))
            XCTAssertTrue(names.contains("model.layers.0.feed_forward.gate_proj.weight"))
            XCTAssertTrue(names.contains("model.embed_tokens.weight"))
            XCTAssertFalse(names.contains("lm_head.weight"))
            XCTAssertTrue(model.requiresExactTensorMmapBuffers)
            XCTAssertTrue(model.preservesCheckpointParameterDTypes)
            for name in preparation.banks.excludedTensorNames {
                XCTAssertTrue(model.excludeFromGenericSafetensorsLoad(key:name))
            }
            XCTAssertFalse(model.excludeFromGenericSafetensorsLoad(key:"model.layers.1.feed_forward.gate.weight"))
            XCTAssertEqual(model.kvHeads,[0,1])
            for requesting: Set<ModelRuntimeRequestModality> in [[], [.text, .tools, .reasoning], [.tools, .reasoning]] {
                XCTAssertTrue(try preparation.construct(requesting: requesting).requiresExactTensorMmapBuffers)
            }
            for requesting: Set<ModelRuntimeRequestModality> in [[.vision], [.text, .vision], [.video], [.audio]] {
                XCTAssertThrowsError(try preparation.construct(requesting: requesting))
            }
        }
    }

    func testGenericMixedLoadKeepsTiedAffineEmbeddingNativeBiasAndOpaqueBanks() throws {
        try MLXMetalTestLock.withLock {
            let directory = try Self.temporaryDirectory()
            defer { try? FileManager.default.removeItem(at: directory) }
            let config = try Self.fixture(directory)
            let preparation = try XCTUnwrap(LFM2MoEJANGHPreparation.loadIfDeclared(
                directory: directory, configurationData: config))
            let original = try preparation.construct(requesting: [.text])
            try original.update(parameters: ModuleParameters.unflattened(
                original.parameters().flattened().map { ($0.0, $0.1.asType(.bfloat16)) }), verify: [.all])
            let biasName = "model.layers.1.feed_forward.expert_bias"
            let nativeBias = MLXArray([Float(0.1234567), -0.2345678])
            try original.update(parameters: ModuleParameters.unflattened([biasName: nativeBias]), verify: [])
            quantize(model: original, groupSize: 32, bits: 8, mode: .affine,
                filter: { path, _ in path != "model.layers.1.feed_forward.gate" })
            let weights = Dictionary(uniqueKeysWithValues: original.parameters().flattened())
            XCTAssertNotNil(weights["model.embed_tokens.scales"])
            XCTAssertNil(weights["model.layers.1.feed_forward.gate.scales"])
            XCTAssertNil(weights[Self.parent + ".gate_proj.weight"])
            let ordinaryShard = "ordinary.safetensors"
            try MLX.save(arrays: weights, url: directory.appendingPathComponent(ordinaryShard))
            let indexURL = directory.appendingPathComponent(Self.index)
            var index = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: indexURL)) as? [String: Any])
            var map = try XCTUnwrap(index["weight_map"] as? [String: String])
            for key in weights.keys { map[key] = ordinaryShard }
            index["weight_map"] = map
            try JSONSerialization.data(withJSONObject: index).write(to: indexURL)
            // Re-admit the final index; the first lease deliberately becomes stale.
            let finalPreparation = try XCTUnwrap(LFM2MoEJANGHPreparation.loadIfDeclared(
                directory: directory, configurationData: config))
            let loaded = try finalPreparation.construct(requesting: [.text])
            let base = try JSONDecoder().decode(BaseConfiguration.self, from: finalPreparation.banks.ordinaryConfiguration)
            try loadWeights(modelDirectory: directory, model: loaded,
                quantization: base.quantizationContainer?.quantization,
                perLayerQuantization: base.perLayerQuantization)
            XCTAssertTrue(loaded.model.embedTokens is QuantizedEmbedding)
            let actual = Dictionary(uniqueKeysWithValues: loaded.parameters().flattened())
            XCTAssertEqual(Set(actual.keys), Set(weights.keys))
            for (name, expected) in weights {
                let value = try XCTUnwrap(actual[name])
                XCTAssertEqual(value.shape, expected.shape, name)
                XCTAssertEqual(value.dtype, expected.dtype, name)
                XCTAssertTrue((value .== expected).all().item(Bool.self), name)
            }
            XCTAssertEqual(actual[biasName]?.dtype, .float32)
            let moe = try XCTUnwrap(loaded.model.layers[1].feedForward as? Lfm2MoeSparseMoeBlock)
            XCTAssertTrue(moe.experts is JANGHRoutedExpertLayer)
        }
    }

    func testRegistryRequiresDirectoryForCustomBanksAndKeepsAffineTextRoute() async throws {
        try await MLXMetalTestLock.withLock {
            let ordinary = try await LLMTypeRegistry.shared.createModel(
                configuration: JSONSerialization.data(withJSONObject: Self.architecture()),
                modelType: "lfm2_moe", requesting: [.text])
            XCTAssertTrue(ordinary is LFM2MoEModel)
            for requesting: Set<ModelRuntimeRequestModality> in [[], [.text, .tools, .reasoning], [.tools, .reasoning]] {
                let model = try await LLMTypeRegistry.shared.createModel(
                    configuration: JSONSerialization.data(withJSONObject: Self.architecture()),
                    modelType: "lfm2_moe", requesting: requesting)
                XCTAssertTrue(model is LFM2MoEModel)
            }
            for requesting: Set<ModelRuntimeRequestModality> in [[.vision], [.text, .vision], [.video], [.audio]] {
                do {
                    _ = try await LLMTypeRegistry.shared.createModel(
                        configuration: JSONSerialization.data(withJSONObject: Self.architecture()),
                        modelType: "lfm2_moe", requesting: requesting)
                    XCTFail("real media must not construct the text-only LFM runtime")
                } catch {}
            }
            let directory = try Self.temporaryDirectory()
            defer { try? FileManager.default.removeItem(at: directory) }
            do {
                _ = try await LLMTypeRegistry.shared.createModel(configuration: Self.fixture(directory),
                    modelType: "lfm2_moe", requesting: [.text])
                XCTFail("directory-free custom banks must refuse")
            } catch {}
        }
    }

    private final class AffineProtocolAdapter: Module, SwitchGLULayer {
        let original: SwitchGLU
        var calls = 0
        init(_ original: SwitchGLU) { self.original = original; super.init() }
        func callAsFunction(_ input: MLXArray, _ indices: MLXArray) -> MLXArray {
            calls += 1
            return original(input, indices)
        }
    }

    func testOrdinaryBF16ContributionRoundingAndProtocolOnlyAffineReplacement() throws {
        try MLXMetalTestLock.withLock {
            var root = Self.architecture()
            root["num_experts_per_tok"] = 2
            let config = try JSONDecoder().decode(LFM2MoEConfiguration.self,
                from: JSONSerialization.data(withJSONObject: root))
            let moe = Lfm2MoeSparseMoeBlock(config, layerIdx: 1)
            try moe.update(parameters: ModuleParameters.unflattened(
                moe.parameters().flattened().map { name, value in
                    let values = (0 ..< value.size).map { Float(($0 * 13) % 61 - 30) / 31 }
                    return (name, MLXArray(values, value.shape).asType(name == "expert_bias" ? .float32 : .bfloat16))
                }), verify: [.all])
            let affine = try XCTUnwrap(moe.experts as? SwitchGLU)
            for rows in [1, 16] {
                let values = (0 ..< rows * 32).map { Float(($0 * 11) % 53 - 26) / 29 }
                let x = MLXArray(values, [1, rows, 32]).asType(.bfloat16)
                let probabilities = sigmoid(moe.gate(x))
                let selection = probabilities.asType(.float32) + moe.expertBias!
                let indices = argPartition(selection, kth: -2, axis: -1)[.ellipsis, (-2)...]
                let picked = takeAlong(probabilities, indices, axis: -1)
                let scores = (picked / (picked.sum(axis: -1, keepDims: true) + 1e-6)).asType(x.dtype)
                let expected = (affine(x, indices) * scores[.ellipsis, .newAxis]).sum(axis: -2)
                let actual = moe(x)
                XCTAssertTrue(JANGHDenseFastAdmission.bitwiseEqual(actual, expected), "rows=\(rows)")
                let adapter = AffineProtocolAdapter(affine)
                moe.update(modules: ModuleChildren.unflattened([("switch_mlp", adapter)]))
                XCTAssertTrue(JANGHDenseFastAdmission.bitwiseEqual(moe(x), expected), "protocol rows=\(rows)")
                XCTAssertEqual(adapter.calls, 1)
                moe.update(modules: ModuleChildren.unflattened([("switch_mlp", affine)]))
            }
        }
    }

    func testActualBundleHeadersWhenRequested() throws {
        guard let path = ProcessInfo.processInfo.environment["VMLX_LFM_JANGH_HEADER_BUNDLE"] else {
            throw XCTSkip("set VMLX_LFM_JANGH_HEADER_BUNDLE for header-only mounted-bundle audit")
        }
        let directory = URL(fileURLWithPath:path)
        let before = mlx_safetensors_mmap_tracked_buffer_bytes()
        let preparation = try XCTUnwrap(LFM2MoEJANGHPreparation.loadIfDeclared(directory:directory,
            configurationData:Data(contentsOf:directory.appendingPathComponent("config.json"))))
        XCTAssertEqual(preparation.banks.moduleByLayer.count,38)
        XCTAssertEqual(preparation.banks.excludedTensorNames.count,228)
        XCTAssertEqual(preparation.configuration.numExpertsPerToken,4)
        XCTAssertEqual(preparation.configuration.ropeTheta,10000000)
        XCTAssertEqual(mlx_safetensors_mmap_tracked_buffer_bytes(),before)
    }
}
