import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import MLXVLM
import Testing

@Suite("Gemma QAT tied-head precision through the production loader", .serialized)
struct Gemma4TiedHeadLoadDTypeTests {
    private func makeModel(vlm: Bool, perLayerInput: Int) throws -> any LanguageModel {
        let text = """
            {"model_type":"gemma4_text","hidden_size":64,"intermediate_size":64,
             "num_hidden_layers":2,"num_attention_heads":2,"num_key_value_heads":1,
             "head_dim":32,"global_head_dim":32,"vocab_size":64,"sliding_window":4,
             "tie_word_embeddings":true,"hidden_size_per_layer_input":\(perLayerInput),
             "vocab_size_per_layer_input":\(perLayerInput > 0 ? 64 : 0),
             "layer_types":["sliding_attention","full_attention"]}
            """
        if vlm {
            return Gemma4(try JSONDecoder().decode(
                Gemma4Configuration.self,
                from: Data("{\"model_type\":\"gemma4\",\"text_config\":\(text)}".utf8)))
        }
        return Gemma4TextModel(try JSONDecoder().decode(
            Gemma4TextConfiguration.self, from: Data(text.utf8)))
    }

    @Test(arguments: [
        (false, 0, QuantizationMode.mxfp4), (true, 0, QuantizationMode.mxfp4),
        (false, 32, QuantizationMode.mxfp4), (true, 32, QuantizationMode.mxfp4),
        (false, 0, QuantizationMode.affine), (true, 0, QuantizationMode.affine),
        (false, 32, QuantizationMode.affine), (true, 32, QuantizationMode.affine),
    ])
    func headCodecPreservesLoadedActivationAndKVPrecision(
        vlm: Bool, perLayerInput: Int, mode: QuantizationMode
    ) throws {
        try MLXMetalTestLock.withLock {
            let savedPolicy = TiedHeadQuantizationPolicy.current
            defer { TiedHeadQuantizationPolicy.current = savedPolicy }
            try #require(ProcessInfo.processInfo.environment["VMLX_QUANT_TIED_HEAD_BITS"] == nil)
            let directory = FileManager.default.temporaryDirectory
                .appendingPathComponent("gemma-tied-head-load-\(UUID().uuidString)")
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: directory) }

            let original = try makeModel(vlm: vlm, perLayerInput: perLayerInput)
            original.apply { $0.asType(.float16) }
            let format = mode == .mxfp4 ? "mxfp4" : "jang_affine"
            let config: [String: Any] = ["model_type": "gemma4", "weight_format": format]
            try JSONSerialization.data(withJSONObject: config)
                .write(to: directory.appendingPathComponent("config.json"))
            let quantization = BaseConfiguration.Quantization(groupSize: 32, bits: 4, mode: mode)
            var overrides = [String: BaseConfiguration.QuantizationOption]()
            for (path, module) in original.leafModules().flattened() {
                if module is Linear || (mode == .mxfp4 && path.hasSuffix("embed_tokens_per_layer")) {
                    overrides[path] = .quantize(quantization)
                } else if module is Embedding {
                    overrides[path] = .skip
                }
            }
            MLXNN.quantize(model: original, filter: { path, _ in
                if case .quantize(let q) = overrides[path] { return q.asTuple }
                return nil
            })
            try MLX.save(
                arrays: Dictionary(uniqueKeysWithValues: original.parameters().flattened()),
                url: directory.appendingPathComponent("model.safetensors"))
            let perLayer = BaseConfiguration.PerLayerQuantization(
                quantization: quantization, perLayerQuantization: overrides)

            func load(head: TiedHeadQuantizationPolicy.Quantization?) throws -> any LanguageModel {
                TiedHeadQuantizationPolicy.current = head
                let model = try makeModel(vlm: vlm, perLayerInput: perLayerInput)
                try loadWeights(
                    modelDirectory: directory, model: model,
                    perLayerQuantization: perLayer)
                return model
            }
            let passthrough = try load(head: nil)
            let converted = try load(head: .init(bits: 6, groupSize: 64))
            let tokens = MLXArray([Int32(1), 2, 3, 4, 5, 6]).reshaped(1, 6)
            func embedding(_ model: any LanguageModel) throws -> Embedding {
                try #require(model.namedModules().first { $0.0.hasSuffix("embed_tokens") }?.1 as? Embedding)
            }
            let ordinaryRows = try embedding(passthrough)(tokens)
            let convertedRows = try embedding(converted)(tokens)
            #expect(convertedRows.dtype == ordinaryRows.dtype)

            let ordinaryCache = passthrough.newCache(parameters: nil)
            let convertedCache = converted.newCache(parameters: nil)
            for input in [tokens, MLXArray([Int32(7)]).reshaped(1, 1)] {
                let ordinaryLogits = passthrough(input, cache: ordinaryCache)
                let convertedLogits = converted(input, cache: convertedCache)
                eval(ordinaryLogits, convertedLogits, ordinaryCache, convertedCache)
                #expect(convertedLogits.dtype == ordinaryLogits.dtype)
                #expect(ordinaryCache.map(\.offset) == convertedCache.map(\.offset))
                for (ordinary, converted) in zip(ordinaryCache, convertedCache) {
                    try #require(!ordinary.state.isEmpty && ordinary.state.count == converted.state.count)
                    for (a, b) in zip(ordinary.state, converted.state) {
                        #expect(a.dtype == b.dtype)
                    }
                }
            }
        }
    }
}
