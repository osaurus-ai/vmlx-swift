import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLMCommon
@testable import MLXVLM

/// Qwen3.8-Flash-Next JANGHT (2026-10-16): qwen4_exp with every backbone routed projection a JANGT trellis bank
/// (`{"mode": "jangt", "kbits": K}`, K per (layer, gate+up | down) in {1, 1.5, 2, 2.5}), a top-level `jangt` block,
/// NO `jangtq` codebook header, and a TEXT-ONLY bundle (no `vision_config`, no `visual.*`, no preprocessor files).
/// Synthetic weights only: this proves admission, construction, the routed math and the factory route, not
/// trained-model coherence.
final class Qwen4ExpJANGTPreparationTests: XCTestCase {
    private let f = Q38TFixture()
    private var roles: [String] { f.roles }
    private var plan: [(Double, Double)] { f.plan }
    private var hidden: Int { f.hidden }
    private var topK: Int { f.topK }

    // MARK: - admission

    func testPureJANGTTextOnlyBundleIsAdmittedAndConstructed() throws {
        try MLXMetalTestLock.withLock {
            let directory = FileManager.default.temporaryDirectory.appendingPathComponent("q38t-" + UUID().uuidString)
            defer { try? FileManager.default.removeItem(at: directory) }
            let data = try f.fixture(directory, banks: f.makeBanks(seed: 1))
            // The strict ordinary decoder refuses `mode: jangt`; only the custom partition may consume it.
            XCTAssertThrowsError(try JSONDecoder.json5().decode(Qwen4ExpConfiguration.self, from: data))
            XCTAssertTrue(JANGHModelPreparation.declaresCustomFormat(configuration: data, sidecar: nil))
            let prepared = try XCTUnwrap(Qwen4ExpJANGHPreparation.loadIfDeclared(directory: directory,
                                                                                 configurationData: data))
            // 2 layers x 3 roles x 6 trellis suffixes (jt_cs is listed for every role, as the index plan defines).
            XCTAssertEqual(prepared.banks.excludedTensorNames.count, 36)
            XCTAssertTrue(JANGHModelPreparation.isCanonicalRoutedExclusionSet(prepared.banks.excludedTensorNames,
                                                                              routedLayers: [0, 1]))
            let ordinary = try XCTUnwrap(JSONSerialization.jsonObject(with: prepared.banks.ordinaryConfiguration)
                as? [String: Any])
            XCTAssertNil(ordinary["vision_config"])
            XCTAssertNil(ordinary["jangtq"], "no codebook header is invented for a pure-JANGT bundle")
            let q = try XCTUnwrap(ordinary["quantization"] as? [String: Any])
            XCTAssertEqual((q["language_model.embed_tokens"] as? [String: Int])?["bits"], 4)
            for layer in 0 ..< 2 {
                for role in roles { XCTAssertEqual(q["model.layers.\(layer).mlp.switch_mlp.\(role)"] as? Bool, false) }
            }
            let model = try prepared.construct(configurationData: prepared.banks.ordinaryConfiguration,
                                               requesting: nil)
            XCTAssertEqual(model.modalities, [.text], "text-only: no vision tower is constructed")
            let routed = model.namedModules().filter { $0.1 is JANGTRoutedExpertLayer }.map(\.0).sorted()
            XCTAssertEqual(routed, ["language_model.layers.0.mlp.switch_mlp", "language_model.layers.1.mlp.switch_mlp"])
            XCTAssertTrue(model.namedModules().allSatisfy { !($0.1 is JANGHRoutedExpertLayer) })
            // Prompt caches written by other trellis kernels must not restore: JANGT bundles carry a computation
            // identity in the cache key; ordinary bundles keep their existing (nil) identity.
            XCTAssertEqual(model.cacheComputationIdentity, "qwen4_exp:jangt-v2-halfbits-k1-masked-tail-v1")
            var shellRoot = f.configuration()
            shellRoot["quantization"] = ["group_size": 64, "bits": 8]
            let shell = try Qwen4Exp(try JSONDecoder.json5().decode(
                Qwen4ExpConfiguration.self, from: JSONSerialization.data(withJSONObject: shellRoot)), requesting: [.text])
            XCTAssertNil(shell.cacheComputationIdentity)
            for name in prepared.banks.excludedTensorNames where !name.hasSuffix(".jt_cs") || name.contains("down_proj") {
                XCTAssertTrue(model.excludeFromGenericSafetensorsLoad(key: name), name)
            }
            for name in ["lm_head.weight", "language_model.layers.0.mlp.shared_expert.gate_proj.weight",
                         "language_model.layers.0.mlp.gate.weight", "mtp.layers.0.mlp.switch_mlp.gate_proj.weight"] {
                XCTAssertFalse(model.excludeFromGenericSafetensorsLoad(key: name), name)
            }
        }
    }

    /// The routed layers the preparation builds, against the float64 reference: decode kernels (30 slots) and, on
    /// NAX GPUs, the expert-sorted prefill tiles (600 slots >= the 512-slot crossover), at mixed widths per layer.
    func testRoutedLayersMatchDenseReferenceDecodeAndPrefill() throws {
        try MLXMetalTestLock.withLock {
            let directory = FileManager.default.temporaryDirectory.appendingPathComponent("q38t-" + UUID().uuidString)
            defer { try? FileManager.default.removeItem(at: directory) }
            let banks = f.makeBanks(seed: 2)
            let data = try f.fixture(directory, banks: banks)
            let prepared = try XCTUnwrap(Qwen4ExpJANGHPreparation.loadIfDeclared(directory: directory,
                                                                                 configurationData: data))
            let routed = try prepared.banks.makeRoutedExperts(activationLimit: nil)
            XCTAssertEqual(Set(routed.keys), [0, 1])
            var rng = JANGTLowBitKernelTests.Rng(state: 5)
            var cases: [(tokens: Int, dtype: DType, bound: Double)] = [(3, .float32, 5e-3), (3, .bfloat16, 2e-2)]
            if JANGTPrefillKernel.available { cases.append((60, .bfloat16, 2e-2)) }
            for layer in 0 ..< 2 {
                let layerModule = try XCTUnwrap(routed[layer] as? JANGTRoutedExpertLayer)
                let base = "model.layers.\(layer).mlp.switch_mlp."
                let gate = banks.projections[base + "gate_proj"]!, up = banks.projections[base + "up_proj"]!
                let down = banks.projections[base + "down_proj"]!
                for c in cases {
                    let x = JANGTLowBitKernelTests.bf16((0 ..< c.tokens * hidden).map { _ in rng.normal() })
                    let (ids, scores) = f.routing(&rng, tokens: c.tokens)
                    let want = f.reference(x: x, tokens: c.tokens, indices: ids, scores: scores, gate: gate, up: up,
                                         down: down)
                    JANGTLowBitKernelTests.assertLive(want, "routed reference layer \(layer)")
                    let got = layerModule.callRouted(
                        MLXArray(x, [1, c.tokens, hidden]).asType(c.dtype),
                        indices: MLXArray(ids, [1, c.tokens, topK]), scores: MLXArray(scores, [1, c.tokens, topK]))
                    XCTAssertEqual(got.shape, [1, c.tokens, hidden])
                    XCTAssertEqual(got.dtype, c.dtype)
                    let err = JANGTLowBitKernelTests.relL2(got.asType(.float32).asArray(Float.self), want)
                    let route = c.tokens * topK >= JANGTRoutedExpertLayer.sortThreshold
                        && JANGTPrefillKernel.available ? "prefill" : "decode"
                    print("Q38T routed layer \(layer) (gate+up K=\(plan[layer].0), down K=\(plan[layer].1)) "
                          + "\(route) T=\(c.tokens) \(c.dtype): rel-L2 \(err)")
                    XCTAssertLessThan(err, c.bound, "layer \(layer) \(route) T=\(c.tokens) \(c.dtype)")
                }
            }
        }
    }

    func testJANGTOwnsRoutedStacksWhenACodebookHeaderIsPresent() throws {
        try MLXMetalTestLock.withLock {
            let directory = FileManager.default.temporaryDirectory.appendingPathComponent("q38t-" + UUID().uuidString)
            defer { try? FileManager.default.removeItem(at: directory) }
            let data = try f.fixture(directory, banks: f.makeBanks(seed: 3), jangtqHeader: true)
            let prepared = try XCTUnwrap(Qwen4ExpJANGHPreparation.loadIfDeclared(directory: directory,
                                                                                 configurationData: data))
            let routed = try prepared.banks.makeRoutedExperts(activationLimit: nil)
            XCTAssertTrue(routed.values.allSatisfy { $0 is JANGTRoutedExpertLayer })
            XCTAssertEqual(prepared.banks.excludedTensorNames.filter { $0.contains("tq2_") }.count, 0)
        }
    }

    func testMalformedJANGTDeclarationsFailClosed() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("q38t-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let banks = f.makeBanks(seed: 4)
        // A declared multiplier that is not the runtime's code.
        var data = try f.fixture(directory, banks: banks) { root in
            var jt = root["jangt"] as! [String: Any]
            jt["code_multipliers"] = ["1.5": 636_231_159]
            root["jangt"] = jt
        }
        XCTAssertThrowsError(try Qwen4ExpJANGHPreparation.loadIfDeclared(directory: directory, configurationData: data))
        // A codebook projection without its header.
        data = try f.fixture(directory, banks: banks) { root in
            var q = root["quantization"] as! [String: Any]
            q["model.layers.1.mlp.switch_mlp.down_proj"] = ["mode": "jangtq2", "bits": 2, "rotation": "hadamard32"]
            root["quantization"] = q
            root["quantization_config"] = q
        }
        XCTAssertThrowsError(try Qwen4ExpJANGHPreparation.loadIfDeclared(directory: directory, configurationData: data))
        // A width that disagrees with the shard (K=1.5 declared, K=2 rows stored).
        let wrong = f.makeBanks(seed: 4, wordsOverride: ["model.layers.0.mlp.switch_mlp.gate_proj": 2,
                                                        "model.layers.0.mlp.switch_mlp.up_proj": 2])
        data = try f.fixture(directory, banks: wrong) { root in
            var q = root["quantization"] as! [String: Any]
            for role in ["gate_proj", "up_proj"] {
                q["model.layers.0.mlp.switch_mlp.\(role)"] = ["mode": "jangt", "kbits": 1.5]
            }
            root["quantization"] = q
            root["quantization_config"] = q
        }
        XCTAssertThrowsError(try Qwen4ExpJANGHPreparation.loadIfDeclared(directory: directory, configurationData: data))
        // gate and up at different widths (one fused kernel per unit).
        data = try f.fixture(directory, banks: banks) { root in
            var q = root["quantization"] as! [String: Any]
            q["model.layers.0.mlp.switch_mlp.up_proj"] = ["mode": "jangt", "kbits": 2]
            root["quantization"] = q
            root["quantization_config"] = q
        }
        XCTAssertThrowsError(try Qwen4ExpJANGHPreparation.loadIfDeclared(directory: directory, configurationData: data))
        // A missing routed projection.
        data = try f.fixture(directory, banks: banks) { root in
            var q = root["quantization"] as! [String: Any]
            q.removeValue(forKey: "model.layers.1.mlp.switch_mlp.down_proj")
            root["quantization"] = q
            root["quantization_config"] = q
        }
        XCTAssertThrowsError(try Qwen4ExpJANGHPreparation.loadIfDeclared(directory: directory, configurationData: data))
    }

    func testTextOnlyProcessorFallbackIsScopedToTextOnlyQwen4Exp() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("q38t-proc-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        var root = f.configuration()
        let textOnly = try JSONSerialization.data(withJSONObject: root)
        let data = try XCTUnwrap(Qwen4ExpTextOnlyBundle.processorConfiguration(modelDirectory: directory,
                                                                                 configData: textOnly))
        XCTAssertEqual(try JSONDecoder.json5().decode(BaseProcessorConfiguration.self, from: data).processorClass,
                       "Qwen3VLProcessor")
        XCTAssertNoThrow(try JSONDecoder.json5().decode(Qwen3VLProcessorConfiguration.self, from: data))
        // A vision bundle, another family, or any processor file on disk: never synthesized.
        root["vision_config"] = ["model_type": "qwen3_vl"]
        XCTAssertNil(Qwen4ExpTextOnlyBundle.processorConfiguration(
            modelDirectory: directory, configData: try JSONSerialization.data(withJSONObject: root)))
        XCTAssertNil(Qwen4ExpTextOnlyBundle.processorConfiguration(
            modelDirectory: directory,
            configData: try JSONSerialization.data(withJSONObject: ["model_type": "qwen3_5"])))
        try Data("{}".utf8).write(to: directory.appendingPathComponent("preprocessor_config.json"))
        XCTAssertNil(Qwen4ExpTextOnlyBundle.processorConfiguration(modelDirectory: directory, configData: textOnly))
    }

    // MARK: - the factory route Osaurus uses (VLMModelFactory, routed loader)

    private struct FixtureTokenizer: GenerationPromptControllableTokenizer {
        var bosToken: String? { nil }
        var eosToken: String? { "<eos>" }
        var unknownToken: String? { nil }
        func encode(text: String, addSpecialTokens: Bool) -> [Int] { text.utf8.map { 16 + Int($0) % 96 } }
        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { tokenIds.map(String.init).joined(separator: " ") }
        func convertTokenToId(_ token: String) -> Int? { token == "<eos>" ? 1 : nil }
        func convertIdToToken(_ id: Int) -> String? { id == 1 ? "<eos>" : nil }
        func applyChatTemplate(messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
                               additionalContext: [String: any Sendable]?) throws -> [Int] {
            try applyChatTemplate(messages: messages, tools: tools, additionalContext: additionalContext,
                                  addGenerationPrompt: true)
        }
        func applyChatTemplate(messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
                               additionalContext: [String: any Sendable]?, addGenerationPrompt: Bool) throws -> [Int] {
            var ids: [Int] = []
            for m in messages {
                ids.append((m["role"] as? String) == "assistant" ? 3 : 2)
                // The Qwen-VL message generator sends content as a list of typed parts.
                let text = (m["content"] as? String)
                    ?? ((m["content"] as? [[String: any Sendable]])?.compactMap { $0["text"] as? String }.joined()) ?? ""
                ids += encode(text: text, addSpecialTokens: false) + [5]
            }
            if addGenerationPrompt { ids.append(3) }
            return ids
        }
    }

    private struct FixtureTokenizerLoader: TokenizerLoader {
        func load(from directory: URL) async throws -> any Tokenizer { FixtureTokenizer() }
    }

    func testTextOnlyBundleLoadsThroughTheRoutedFactory() async throws {
        let f = Q38TFixture()
        try await MLXMetalTestLock.withLock {
            let directory = FileManager.default.temporaryDirectory.appendingPathComponent("q38t-full-" + UUID().uuidString)
            defer { try? FileManager.default.removeItem(at: directory) }
            try Device.withDefaultDevice(.cpu) { try f.fullBundle(directory) }
            for name in ["preprocessor_config.json", "processor_config.json", "video_preprocessor_config.json"] {
                XCTAssertFalse(FileManager.default.fileExists(atPath: directory.appendingPathComponent(name).path))
            }
            // The routed loader (VLM factory first, then LLM) — the entry Osaurus' model load goes through.
            let context = try await loadModel(from: directory, using: FixtureTokenizerLoader())
            let model = try XCTUnwrap(context.model as? Qwen4Exp, "qwen4_exp must be constructed by the VLM factory")
            XCTAssertEqual(model.modalities, [.text])
            XCTAssertEqual(model.namedModules().filter { $0.1 is JANGTRoutedExpertLayer }.count, 2)
            XCTAssertNil(model.namedModules().first { $0.0.hasPrefix("visual") })

            let input = try await context.processor.prepare(input: UserInput(chat: [.user("Hello, Flash Next.")]))
            let prompt = input.text.tokens.reshaped(-1).asArray(Int.self)
            XCTAssertGreaterThan(prompt.count, 4)
            // 64 tokens x top-10 = 640 routed slots: the prefill route on NAX GPUs; then one decode token.
            let ids = (0 ..< 64).map { prompt[$0 % prompt.count] }
            let cache = model.newCache(parameters: nil)
            let prefill = model(MLXArray(ids).reshaped(1, ids.count), cache: cache)
            let step = model(MLXArray([ids[0]]).reshaped(1, 1), cache: cache)
            eval(prefill, step)
            XCTAssertEqual(prefill.shape, [1, 64, 128])
            XCTAssertEqual(step.shape, [1, 1, 128])
            let values = prefill.asType(.float32).asArray(Float.self) + step.asType(.float32).asArray(Float.self)
            XCTAssertTrue(values.allSatisfy(\.isFinite), "logits must be finite")
            XCTAssertGreaterThan(values.lazy.filter { $0 != 0 }.count, values.count / 2, "logits must be live")
            // Media is refused by a model that carries no tower (never silently dropped).
            XCTAssertFalse(model.modalities.contains(.vision))
        }
    }
}

/// Synthetic Qwen3.8-Flash-Next JANGHT bundle pieces (Sendable, so async tests can build bundles off the test instance).
struct Q38TFixture: Sendable {
    typealias P = JANGTLowBitKernelTests.Projection
    let roles = ["gate_proj", "up_proj", "down_proj"]
    /// Per-layer widths: (gate+up, down), as the Qwen allocation plan mixes them.
    let plan: [(Double, Double)] = [(1.5, 2), (1, 2.5)]
    let hidden = 256, inter = 128, experts = 16, topK = 10

    func configuration() -> [String: Any] {
        [
            "model_type": "qwen4_exp",
            "tie_word_embeddings": false,
            "image_token_id": 120, "video_token_id": 121,
            "vision_start_token_id": 122, "vision_end_token_id": 123,
            "text_config": [
                "model_type": "qwen4_exp_text", "dtype": "bfloat16",
                "hidden_size": hidden, "num_hidden_layers": 2, "intermediate_size": 256,
                "num_attention_heads": 4, "num_key_value_heads": 1, "head_dim": 32,
                "linear_num_value_heads": 4, "linear_num_key_heads": 1,
                "linear_key_head_dim": 32, "linear_value_head_dim": 32,
                "linear_conv_kernel_dim": 4, "vocab_size": 128,
                "num_experts": experts, "num_experts_per_tok": topK, "norm_topk_prob": true,
                "moe_intermediate_size": inter, "shared_expert_intermediate_size": 64,
                "layer_types": ["linear_attention", "full_attention"],
                "hc_count": 4, "hc_lowrank": 8, "ple_layer_ids": [],
                "indexer_n_heads": 2, "indexer_kv_heads": 1, "indexer_head_dim": 16,
                "indexer_budget": 32, "indexer_compress_ratio": 4,
                "max_position_embeddings": 4096, "rms_norm_eps": 1e-6,
            ],
        ]
    }

    func jangtHeader(outliers: [String: Int]) -> [String: Any] {
        [
            "version": 1, "code": "v2_halfbits", "state_bits": 12, "rotation": "hadamard128-signs",
            "outlier_columns": outliers,
            "code_multipliers": ["1": 590_428_815, "1.5": 2_124_823_417, "2": 636_231_159, "2.5": 636_231_159],
            "step_bits": ["1": [2, 2], "1.5": [4, 2], "2": [4, 4], "2.5": [6, 4]],
        ]
    }

    struct Banks {
        var projections: [String: P] = [:]
        var arrays: [String: MLXArray] = [:]
        /// name -> (safetensors dtype, shape, little-endian bytes), for the aligned bank shard.
        var raw: [String: (String, [Int], Data)] = [:]
        var quantization: [String: Any] = [:]
        var outliers: [String: Int] = [:]
    }

    /// Synthetic trellis banks for every backbone routed projection, in the bundle layout of `assemble_q38t.py`.
    func makeBanks(seed: UInt64, wordsOverride: [String: Double] = [:]) -> Banks {
        var rng = JANGTLowBitKernelTests.Rng(state: seed)
        var b = Banks()
        for (layer, widths) in plan.enumerated() {
            // Layer-wide gate/up outlier columns; per-expert, -1-padded down columns.
            let gu = (0 ..< experts).map { _ in [Int32(3), Int32(hidden - 5)] }
            let dn = (0 ..< experts).map { e -> [Int32] in
                e % 3 == 0 ? [Int32(e % inter), -1] : (e % 3 == 1 ? [Int32(inter - 1), Int32(7)] : [-1, -1])
            }
            for role in roles {
                let module = "model.layers.\(layer).mlp.switch_mlp.\(role)"
                let k = role == "down_proj" ? widths.1 : widths.0
                let p = P(&rng, k: wordsOverride[module] ?? k, experts: experts,
                          outputs: role == "down_proj" ? hidden : inter,
                          inputs: role == "down_proj" ? inter : hidden,
                          columns: role == "down_proj" ? dn : gu, columnScales: role == "down_proj")
                b.projections[module] = p
                b.arrays[module + ".jt_packed"] = MLXArray(p.packed, [p.experts, p.outputs, p.words])
                b.arrays[module + ".jt_scale"] = MLXArray(p.scale, [p.experts, p.outputs])
                b.arrays[module + ".jt_su"] = MLXArray(p.signs, [p.inputs])
                b.arrays[module + ".jt_ocol"] = MLXArray(p.ocol, [p.experts, p.outliers])
                b.arrays[module + ".jt_wout"] = MLXArray(p.wout, [p.experts, p.outputs, p.outliers]).asType(.float16)
                if let cs = p.cs { b.arrays[module + ".jt_cs"] = MLXArray(cs, [p.experts, p.inputs]) }
                func bytes<T>(_ v: [T]) -> Data { v.withUnsafeBytes { Data($0) } }
                b.raw[module + ".jt_packed"] = ("U32", [p.experts, p.outputs, p.words], bytes(p.packed))
                b.raw[module + ".jt_scale"] = ("F32", [p.experts, p.outputs], bytes(p.scale))
                b.raw[module + ".jt_su"] = ("F32", [p.inputs], bytes(p.signs))
                b.raw[module + ".jt_ocol"] = ("I32", [p.experts, p.outliers], bytes(p.ocol))
                b.raw[module + ".jt_wout"] = ("F16", [p.experts, p.outputs, p.outliers], bytes(p.wout.map { Float16($0) }))
                if let cs = p.cs { b.raw[module + ".jt_cs"] = ("F32", [p.experts, p.inputs], bytes(cs)) }
                b.quantization[module] = ["mode": "jangt", "kbits": k]
                b.outliers[module] = p.outliers
            }
        }
        return b
    }

    /// Bank shard in the layout the assembler writes (aligned safetensors): page-aligned data section, every tensor
    /// at an offset that is a multiple of its element size (the mapped-bank admission checks it).
    func writeAligned(_ tensors: [String: (String, [Int], Data)], file: String, into directory: URL,
                      index: inout [String: String]) throws {
        var header: [String: Any] = [:]
        var payload = Data()
        for name in tensors.keys.sorted() {
            let (dtype, shape, data) = tensors[name]!
            while payload.count % 64 != 0 { payload.append(0) }
            let start = payload.count
            payload.append(data)
            header[name] = ["dtype": dtype, "shape": shape, "data_offsets": [start, payload.count]]
            index[name] = file
        }
        var encoded = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
        let padded = ((encoded.count + 8 + 4095) / 4096) * 4096 - 8
        encoded.append(Data(repeating: 0x20, count: padded - encoded.count))
        var length = UInt64(encoded.count).littleEndian
        var out = withUnsafeBytes(of: &length) { Data($0) }
        out.append(encoded)
        out.append(payload)
        try out.write(to: directory.appendingPathComponent(file))
    }

    func write(_ arrays: [String: MLXArray], file: String, into directory: URL,
                       index: inout [String: String]) throws {
        try MLX.save(arrays: arrays, url: directory.appendingPathComponent(file))
        for name in arrays.keys { index[name] = file }
    }

    /// Bundle with only the routed banks (admission + routed-layer math). Returns the config data.
    func fixture(_ directory: URL, banks: Banks, jangtqHeader: Bool = false,
                         mutate: (inout [String: Any]) -> Void = { _ in }) throws -> Data {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        var index: [String: String] = [:]
        try writeAligned(banks.raw, file: "model-00001-of-00001.safetensors", into: directory, index: &index)
        try JSONSerialization.data(withJSONObject: ["weight_map": index]).write(
            to: directory.appendingPathComponent("model.safetensors.index.json"))
        var root = configuration()
        var q: [String: Any] = ["group_size": 64, "bits": 8, "language_model.embed_tokens": ["bits": 4, "group_size": 64]]
        q.merge(banks.quantization) { $1 }
        root["quantization"] = q
        root["quantization_config"] = q
        root["jangt"] = jangtHeader(outliers: banks.outliers)
        if jangtqHeader {
            root["jangtq"] = ["version": 2, "packing": "lsb-bitstream", "scale_dtype": "float16",
                              "rotation": "hadamard32", "codebook_family": "odd-cubic",
                              "codebooks": ["2": ["alpha": 0.893, "beta": 0.0,
                                                  "levels": [-1.3395, -0.4465, 0.4465, 1.3395]]]]
        }
        mutate(&root)
        let data = try JSONSerialization.data(withJSONObject: root)
        try data.write(to: directory.appendingPathComponent("config.json"))
        return data
    }

    /// Float64 reference of one routed layer (the `JANGTRoutedExpertLayer` contract) for tokens `x` (T, D).
    func reference(x: [Float], tokens: Int, indices: [UInt32], scores: [Float], gate: P, up: P, down: P)
        -> [Float]
    {
        let d = hidden, i = inter, k = topK
        var keep = [Float](repeating: 1, count: d)
        for (_, c) in gate.columns(0) { keep[c] = 0 }
        var y = [Float](repeating: 0, count: tokens * d)
        for t in 0 ..< tokens {
            let xt = Array(x[t * d ..< (t + 1) * d])
            let xr = JANGTLowBitKernelTests.h128((0 ..< d).map { xt[$0] * gate.signs[$0] * keep[$0] })
            var acc = [Double](repeating: 0, count: d)
            for s in 0 ..< k {
                let e = Int(indices[t * k + s])
                var h = [Float](repeating: 0, count: i)
                gate.values.withUnsafeBufferPointer { gv in
                    up.values.withUnsafeBufferPointer { uv in
                        for n in 0 ..< i {
                            var g = JANGTLowBitKernelTests.dot(gv, (e * i + n) * d, xr) * Double(gate.scale[e * i + n])
                            var u = JANGTLowBitKernelTests.dot(uv, (e * i + n) * d, xr) * Double(up.scale[e * i + n])
                            for (j, c) in gate.columns(e) {
                                g += Double(xt[c]) * Double(gate.wout[(e * i + n) * gate.outliers + j])
                                u += Double(xt[c]) * Double(up.wout[(e * i + n) * up.outliers + j])
                            }
                            h[n] = Float(g / (1 + Foundation.exp(-g)) * u)
                        }
                    }
                }
                var keepE = [Float](repeating: 1, count: i)
                for (_, c) in down.columns(e) { keepE[c] = 0 }
                let hr = JANGTLowBitKernelTests.h128((0 ..< i).map { h[$0] * down.cs![e * i + $0] * down.signs[$0] * keepE[$0] })
                down.values.withUnsafeBufferPointer { dv in
                    for n in 0 ..< d {
                        var v = JANGTLowBitKernelTests.dot(dv, (e * d + n) * i, hr) * Double(down.scale[e * d + n])
                        for (j, c) in down.columns(e) { v += Double(h[c]) * Double(down.wout[(e * d + n) * down.outliers + j]) }
                        acc[n] += Double(scores[t * k + s]) * v
                    }
                }
            }
            for n in 0 ..< d { y[t * d + n] = Float(acc[n]) }
        }
        return y
    }

    func routing(_ rng: inout JANGTLowBitKernelTests.Rng, tokens: Int) -> ([UInt32], [Float]) {
        var ids: [UInt32] = [], scores: [Float] = []
        for _ in 0 ..< tokens {
            var pool = Array(0 ..< experts)
            for s in 0 ..< topK { pool.swapAt(s, s + Int(rng.next() % UInt64(experts - s))) }
            ids += pool.prefix(topK).map(UInt32.init)
            let w = (0 ..< topK).map { _ in 0.05 + rng.uniform() }
            let sum = w.reduce(0, +)
            scores += w.map { $0 / sum }
        }
        return (ids, scores)
    }

    /// A complete text-only bundle: ordinary bf16 weights for everything but the routed experts (from the model's own
    /// parameter tree), JANGT banks for the routed experts, no `vision_config`, no `visual.*`, no preprocessor files.
    func fullBundle(_ directory: URL) throws {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        var shellRoot = configuration()
        shellRoot["quantization"] = ["group_size": 64, "bits": 8]
        let shellConfig = try JSONDecoder.json5().decode(
            Qwen4ExpConfiguration.self, from: JSONSerialization.data(withJSONObject: shellRoot))
        let shell = try Qwen4Exp(shellConfig, requesting: [.text])
        var plan: [String: Any] = ["group_size": 64, "bits": 8, "mode": "affine"]
        var ordinary: [String: MLXArray] = [:]
        for (name, parameter) in shell.parameters().flattened() {
            if name.hasPrefix("language_model.layers."), name.contains(".mlp.switch_mlp.") { continue }
            let salt = name.utf8.reduce(0) { $0 &+ Int($1) }
            var values = [Float]()
            values.reserveCapacity(parameter.size)
            for i in 0 ..< parameter.size { values.append(Float((i &* 7 &+ salt) % 31 - 15) / 512) }
            ordinary[name] = MLXArray(values, parameter.shape).asType(parameter.dtype == .float32 ? .float32 : .bfloat16)
            if name.hasSuffix(".weight") { plan[String(name.dropLast(7))] = false }
        }
        XCTAssertNil(ordinary.keys.first { $0.hasPrefix("visual.") })
        let banks = makeBanks(seed: 6)
        var index: [String: String] = [:]
        try write(ordinary, file: "model-00001-of-00002.safetensors", into: directory, index: &index)
        try writeAligned(banks.raw, file: "model-00002-of-00002.safetensors", into: directory, index: &index)
        try JSONSerialization.data(withJSONObject: ["weight_map": index]).write(
            to: directory.appendingPathComponent("model.safetensors.index.json"))
        plan.merge(banks.quantization) { $1 }
        var root = configuration()
        root["quantization"] = plan
        root["quantization_config"] = plan
        root["jangt"] = jangtHeader(outliers: banks.outliers)
        try JSONSerialization.data(withJSONObject: root).write(to: directory.appendingPathComponent("config.json"))
        try JSONSerialization.data(withJSONObject: ["eos_token_id": [1], "do_sample": true, "temperature": 1.0,
                                                    "top_p": 0.95, "top_k": 20])
            .write(to: directory.appendingPathComponent("generation_config.json"))
    }

}
