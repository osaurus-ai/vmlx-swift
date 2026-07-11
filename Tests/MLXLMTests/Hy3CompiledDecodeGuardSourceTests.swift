import Foundation
import MLXLLM
import MLXLMCommon
import Testing

struct Hy3CompiledDecodeGuardSourceTests {
    private static let repoRoot = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()

    @Test("BatchEngine consults the per-instance compile veto before the name denylist")
    func batchEngineCompileGuardConsultsVeto() throws {
        let source = try String(contentsOf: Self.repoRoot.appendingPathComponent(
            "Libraries/MLXLMCommon/BatchEngine/BatchEngine.swift"))

        #expect(source.contains("private var compiledDecodeDeniedForModel: Bool"))
        #expect(source.contains("context.model as? CompiledDecodeVetoing"))
        #expect(source.contains("context.configuration.toolCallFormat == .hunyuan"))
        #expect(source.contains("!compiledDecodeDeniedForModel && !soloParameters.enableCompiledDecode"))
        #expect(source.contains("guard !compiledDecodeDeniedForModel else { return }"))
    }

    @Test("TokenIterator direct compiled decode consults the veto too")
    func tokenIteratorCompileGuardConsultsVeto() throws {
        let source = try String(contentsOf: Self.repoRoot.appendingPathComponent(
            "Libraries/MLXLMCommon/Evaluate.swift"))

        #expect(source.contains("private static func compiledDecodeDenied(for model: any LanguageModel) -> Bool"))
        #expect(source.contains("model as? CompiledDecodeVetoing"))
        #expect(source.contains("typeName.contains(\"hy3\") || typeName.contains(\"hunyuan\")"))
        #expect(source.contains("effectiveParameters.enableCompiledDecode && !Self.compiledDecodeDenied(for: model)"))
    }

    @Test("Hy3 vetoes compile only for TurboQuant expert packs")
    func hy3VetoFollowsExpertFormat() throws {
        let affine = try hy3Config(weightFormat: "jang-affine-mixed")
        let turboQuant = try hy3Config(weightFormat: "jangtq")

        #expect(!Hy3Model(affine).vetoesCompiledDecode)
        #expect(Hy3Model(turboQuant).vetoesCompiledDecode)
    }

    private func hy3Config(weightFormat: String) throws -> Hy3Configuration {
        let json = """
            {
                "model_type": "hy_v3",
                "hidden_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "intermediate_size": 128,
                "moe_intermediate_size": 32,
                "n_routed_experts": 4,
                "num_experts_per_tok": 2,
                "n_shared_experts": 1,
                "first_k_dense_replace": 1,
                "vocab_size": 128,
                "rms_norm_eps": 1e-5,
                "rope_theta": 10000,
                "weight_format": "\(weightFormat)"
            }
            """
        return try JSONDecoder().decode(
            Hy3Configuration.self, from: Data(json.utf8))
    }
}
