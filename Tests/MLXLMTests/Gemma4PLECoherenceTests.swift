// Pin Gemma4 PLE (per-layer-embedding) config-coherence contract for both
// the LLM (`Gemma4TextConfiguration`) and VLM (`Gemma4.Gemma4TextConfiguration`) decoders.
//
// Background (see `docs/GEMMA4-DEEP-TRACE-2026-05-10.md` §7.6):
//
// Gemma4 E2B / E4B variants opt into Per-Layer Embedding by setting BOTH
// `hidden_size_per_layer_input` AND `vocab_size_per_layer_input` to non-zero.
// Setting a positive hidden size without vocab is structurally invalid:
//   * vocab=0, hidden>0 → `Embedding(embeddingCount: 0, ...)` → garbage rows
// Shipped full Gemma4 rows may set vocab>0 while hidden=0; the decoder treats
// hidden=0 as the authoritative PLE-off signal and normalizes vocab to 0.
// Both fields default to 0 in the decoder (PLE off — base 26B / 31B models).
//
// Before today, the decoder rejected shipped 26B/31B-style configs that carried
// `vocab_size_per_layer_input` even though PLE was disabled by hidden=0.
// It now accepts and normalizes that shape, while still rejecting hidden>0/vocab=0.
//
// Source-coverage style — no MLX runtime needed.

import Foundation
@testable import MLXLLM
import Testing

@Suite("Gemma4 PLE config-coherence contract")
struct Gemma4PLECoherenceTests {

    private static func decode(
        _ json: String,
        file: String = #filePath, line: Int = #line
    ) throws -> Gemma4TextConfiguration {
        let data = json.data(using: .utf8)!
        return try JSONDecoder().decode(Gemma4TextConfiguration.self, from: data)
    }

    /// Both fields zero (PLE off) — this is the base 27B / 31B path and
    /// must continue to decode cleanly.
    @Test("Gemma4 LLM config decodes when PLE is off (both fields zero)")
    func plOffDecodesCleanly() throws {
        let json = """
            {
              "model_type": "gemma4_text",
              "hidden_size": 2304,
              "num_hidden_layers": 4,
              "num_attention_heads": 8,
              "num_key_value_heads": 4,
              "intermediate_size": 9216,
              "vocab_size": 262144,
              "rms_norm_eps": 1e-6,
              "sliding_window": 1024,
              "layer_types": ["sliding","full","sliding","full"]
            }
            """
        let config = try Self.decode(json)
        #expect(config.hiddenSizePerLayerInput == 0)
        #expect(config.vocabSizePerLayerInput == 0)
    }

    /// Both fields positive (PLE on) — the E2B/E4B path.
    @Test("Gemma4 LLM config decodes when PLE is on (both fields positive)")
    func plOnDecodesCleanly() throws {
        let json = """
            {
              "model_type": "gemma4_text",
              "hidden_size": 2304,
              "num_hidden_layers": 4,
              "hidden_size_per_layer_input": 256,
              "vocab_size_per_layer_input": 262144
            }
            """
        let config = try Self.decode(json)
        #expect(config.hiddenSizePerLayerInput == 256)
        #expect(config.vocabSizePerLayerInput == 262144)
    }

    /// Hidden positive but vocab zero — must throw.
    @Test("Gemma4 LLM config rejects hidden>0 with vocab=0")
    func plHiddenWithoutVocabThrows() throws {
        let json = """
            {
              "model_type": "gemma4_text",
              "hidden_size_per_layer_input": 256,
              "vocab_size_per_layer_input": 0
            }
            """
        #expect(throws: DecodingError.self) {
            _ = try Self.decode(json)
        }
    }

    /// Vocab positive but hidden zero — shipped full Gemma4 configs use this
    /// shape; hidden=0 means PLE is off, so vocab is ignored/normalized.
    @Test("Gemma4 LLM config treats vocab>0 with hidden=0 as PLE off")
    func plVocabWithoutHiddenNormalizesToOff() throws {
        let json = """
            {
              "model_type": "gemma4_text",
              "hidden_size_per_layer_input": 0,
              "vocab_size_per_layer_input": 262144
            }
            """
        let config = try Self.decode(json)
        #expect(config.hiddenSizePerLayerInput == 0)
        #expect(config.vocabSizePerLayerInput == 0)
    }

    /// Source-coverage guard for the VLM-side config (Gemma4.swift's
    /// nested `Gemma4TextConfiguration`). The VLM type is fileprivate to MLXVLM, so
    /// we pin its source contract directly rather than constructing the
    /// type — same recipe used for Zaya1VL adapters.
    @Test("Gemma4 VLM (MLXVLM) Gemma4TextConfiguration has the same PLE coherence guard")
    func vlmConfigHasMatchingGuard() throws {
        let repo = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let url = repo.appendingPathComponent("Libraries/MLXVLM/Models/Gemma4.swift")
        let source = try String(contentsOf: url, encoding: .utf8)

        // The normalization is present.
        #expect(
            source.contains("decodedHiddenSizePerLayerInput == 0"),
            "Gemma4.swift (VLM) must normalize hidden=0/vocab>0 to PLE off.")
        // The positive-hidden invalid guard is present.
        #expect(
            source.contains("decodedVocabSizePerLayerInput == 0"),
            "Gemma4.swift (VLM) must still reject hidden>0/vocab=0.")
    }

    @Test("Gemma4 E-series scaled projection accepts quantized checkpoint scales")
    func eSeriesScaledProjectionAcceptsQuantizedScales() throws {
        let candidateRoots = [
            "/Volumes/EricsLLMDrive/hf-stage/gemma4-qat-mxfp4/OsaurusAI/gemma-4-E2B-it-qat-MXFP4",
            "/Volumes/eric/models/JANGQ-AI/gemma-4-E2B-it-qat-MXFP4",
            "/Users/eric/osaurus_models/finished/gemma-4-e2b-it-4bit",
        ]
        guard let modelRoot = candidateRoots.first(where: {
            FileManager.default.fileExists(atPath: "\($0)/model.safetensors.index.json")
        }) else {
            print("SKIP: Gemma4 E2B model index not local")
            return
        }

        let indexURL = URL(fileURLWithPath: modelRoot)
            .appendingPathComponent("model.safetensors.index.json")
        let index = try String(contentsOf: indexURL, encoding: .utf8)
        #expect(index.contains("language_model.model.per_layer_model_projection.weight"))
        #expect(index.contains("language_model.model.per_layer_model_projection.scales"))

        let repo = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let textSource = try String(
            contentsOf: repo.appendingPathComponent("Libraries/MLXLLM/Models/Gemma4Text.swift"),
            encoding: .utf8)
        let vlmSource = try String(
            contentsOf: repo.appendingPathComponent("Libraries/MLXVLM/Models/Gemma4.swift"),
            encoding: .utf8)

        for source in [textSource, vlmSource] {
            #expect(source.contains(#"@ParameterInfo(key: "scales") var scales: MLXArray"#))
            #expect(source.contains(#"@ParameterInfo(key: "biases") var biases: MLXArray?"#))
            #expect(source.contains("MLXArray.mlxNone"))
            #expect(source.contains("!scales.shape.isEmpty"))
            #expect(source.contains("JangLoader.inferBitWidthAndGroupSize"))
            #expect(source.contains("quantizedMM("))
            #expect(source.contains("projected.reshaped(Array(x.shape.dropLast()) + [outputDims])"))
            #expect(source.contains("mode: .mxfp4"))
        }
    }

    @Test("Gemma4 E-series scaled projection initializes JANG affine biases slot")
    func eSeriesScaledProjectionInitializesAffineBiasSlot() throws {
        let repo = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let textSource = try String(
            contentsOf: repo.appendingPathComponent("Libraries/MLXLLM/Models/Gemma4Text.swift"),
            encoding: .utf8)
        let vlmSource = try String(
            contentsOf: repo.appendingPathComponent("Libraries/MLXVLM/Models/Gemma4.swift"),
            encoding: .utf8)

        for source in [textSource, vlmSource] {
            #expect(source.contains("self._biases.wrappedValue = MLXArray.mlxNone"))
        }
    }

    @Test("Gemma4 PLE layer tensors split without advanced indexing")
    func eSeriesPerLayerInputsAvoidAdvancedIndexingTrap() throws {
        let repo = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let textSource = try String(
            contentsOf: repo.appendingPathComponent("Libraries/MLXLLM/Models/Gemma4Text.swift"),
            encoding: .utf8)
        let vlmSource = try String(
            contentsOf: repo.appendingPathComponent("Libraries/MLXVLM/Models/Gemma4.swift"),
            encoding: .utf8)

        for source in [textSource, vlmSource] {
            #expect(source.contains("splitPerLayerInputs"))
            #expect(source.contains("perLayerInputs[0..., 0..., start ..< end]"))
            #expect(source.contains("precondition(width > 0"))
            #expect(!source.contains("let lastDim = perLayerInputs.shape.last ?? 0"))
            #expect(!source.contains("lastDim == layerCount * width"))
            #expect(source.contains("hiddenSizePerLayerInput"))
            #expect(!source.contains("split(indices: boundaries"))
            #expect(!source.contains("split(parts: layers.count, axis: 2)"))
            #expect(!source.contains("perLayerInputs.dim(-1)"))
            #expect(!source.contains("squeezed(axis: 2)"))
            #expect(!source.contains("[0..., 0..., i, 0...]"))
            #expect(!source.contains("[0..., 0..., $0, 0...]"))
        }
    }
}
