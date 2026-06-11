// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Phase 4 iter 10 tests — pin SpecDecStream's event contract:
//   1. `.chunk(String)` events concatenated back decode to the same
//      tokens the non-streaming runtime returns.
//   2. Exactly one `.info(GenerateCompletionInfo)` fires at completion.
//   3. Stream finishes cleanly (continuation closes).

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

/// Deterministic tokenizer: `decode([t1, t2, ...]) = "t1|t2|..."`. Good
/// enough to assert round-trip byte-identity on the streaming path.
private struct DecimalTokenizer: MLXLMCommon.Tokenizer {
    let vocabularySize: Int

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        // Prompts are fed in as MLXArray, not via encode; return empty.
        []
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        tokenIds.map { String($0) }.joined(separator: "|") + "|"
    }

    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { String(id) }

    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [] }
}

@Suite("SpecDec stream — Phase 4 iter 10", .serialized)
struct SpecDecStreamTests {

    // Reuse the same tiny-config pair as the other SpecDec tests.
    private static let hiddenSize = 128
    private static let numAttentionHeads = 4
    private static let numKVHeads = 2
    private static let headDim = 32
    private static let vocabSize = 512
    private static let targetLayers = 4
    private static let draftLayers = 2
    private static let targetBlockIDs = [0, 2]
    private static let blockSize = 4
    private static let maskTokenID: Int32 = 500

    private func tokens(_ values: [Int32]) -> MLXArray {
        MLXArray(values).reshaped(1, values.count)
    }

    private func targetConfig() -> Qwen3Configuration {
        let json = """
        {
          "hidden_size": \(Self.hiddenSize),
          "num_hidden_layers": \(Self.targetLayers),
          "intermediate_size": 256,
          "num_attention_heads": \(Self.numAttentionHeads),
          "rms_norm_eps": 1e-6,
          "vocab_size": \(Self.vocabSize),
          "num_key_value_heads": \(Self.numKVHeads),
          "rope_theta": 1000000,
          "head_dim": \(Self.headDim),
          "tie_word_embeddings": false,
          "max_position_embeddings": 512
        }
        """
        return try! JSONDecoder().decode(
            Qwen3Configuration.self, from: json.data(using: .utf8)!)
    }

    private func drafterConfig() -> DFlashDrafterConfiguration {
        let hfStampIDs = Self.targetBlockIDs.map { $0 + 1 }
        let json = """
        {
          "hidden_size": \(Self.hiddenSize),
          "num_hidden_layers": \(Self.draftLayers),
          "num_attention_heads": \(Self.numAttentionHeads),
          "num_key_value_heads": \(Self.numKVHeads),
          "head_dim": \(Self.headDim),
          "intermediate_size": 256,
          "rms_norm_eps": 1e-6,
          "rope_theta": 1000000,
          "max_position_embeddings": 512,
          "attention_bias": false,
          "hidden_act": "silu",
          "block_size": \(Self.blockSize),
          "dflash_config": {
            "mask_token_id": \(Self.maskTokenID),
            "target_layer_ids": \(hfStampIDs)
          }
        }
        """
        return try! JSONDecoder().decode(
            DFlashDrafterConfiguration.self, from: json.data(using: .utf8)!)
    }

    // MARK: - DFlash linear streaming

    @Test("DFlash stream yields .info + tokens matching the non-streaming run")
    func testDflashStreamYieldsTokens() async throws {
        MLXRandom.seed(0x1010)
        let target = Qwen3Model(targetConfig())
        let drafter = DFlashDraftModel(drafterConfig())
        let prompt: [Int32] = [1, 2, 3, 4]
        let maxNew = 4
        let args = DFlashLinearArgs(
            target: target, drafter: drafter,
            targetBlockIDs: Self.targetBlockIDs,
            maskTokenID: Self.maskTokenID,
            inputIds: tokens(prompt), maxNewTokens: maxNew,
            stopTokenIDs: [], temperature: 0)

        // Run non-streaming first for a ground-truth.
        let bulk = try SpecDecRuntimeLinear.run(args)

        // Stream with the decimal tokenizer.
        MLXRandom.seed(0x1010)
        let target2 = Qwen3Model(targetConfig())
        let drafter2 = DFlashDraftModel(drafterConfig())
        let args2 = DFlashLinearArgs(
            target: target2, drafter: drafter2,
            targetBlockIDs: Self.targetBlockIDs,
            maskTokenID: Self.maskTokenID,
            inputIds: tokens(prompt), maxNewTokens: maxNew,
            stopTokenIDs: [], temperature: 0)
        let stream = SpecDecStream.streamDflashLinear(
            args: args2, tokenizer: DecimalTokenizer(
                vocabularySize: Self.vocabSize))

        var chunks: [String] = []
        var infoCount = 0
        for await ev in stream {
            switch ev {
            case .chunk(let c): chunks.append(c)
            case .prefillProgress:

                break
            case .info: infoCount += 1
            case .reasoning, .prefillProgress, .toolCall: break
            @unknown default: break
            }
        }

        #expect(infoCount == 1, "Exactly one .info event must fire")
        // Decoded chunk sequence decodes back to the bulk token list
        // (minus prompt, since the stream only emits generated tokens).
        let streamedText = chunks.joined()
        let expectedGenerated = bulk.tokenIds.suffix(bulk.tokenIds.count - prompt.count)
        let expectedText = expectedGenerated.map { String($0) }.joined(separator: "|") + (expectedGenerated.isEmpty ? "" : "|")
        #expect(streamedText == expectedText,
            "streamed text must equal decode of generated suffix. streamed=\(streamedText) expected=\(expectedText)")
    }

    // MARK: - DDTree streaming

    @Test("DDTree stream yields .info + tokens matching non-streaming run")
    func testDDTreeStreamYieldsTokens() async throws {
        MLXRandom.seed(0x2020)
        let target = Qwen3Model(targetConfig())
        let drafter = DFlashDraftModel(drafterConfig())
        let prompt: [Int32] = [7, 8, 9]
        let maxNew = 4
        let args = DDTreeArgs(
            target: target, drafter: drafter,
            targetBlockIDs: Self.targetBlockIDs,
            maskTokenID: Self.maskTokenID,
            inputIds: tokens(prompt), maxNewTokens: maxNew,
            stopTokenIDs: [], temperature: 0, branchingBudget: 4)

        let bulk = try SpecDecRuntimeDDTree.run(args)

        MLXRandom.seed(0x2020)
        let target2 = Qwen3Model(targetConfig())
        let drafter2 = DFlashDraftModel(drafterConfig())
        let args2 = DDTreeArgs(
            target: target2, drafter: drafter2,
            targetBlockIDs: Self.targetBlockIDs,
            maskTokenID: Self.maskTokenID,
            inputIds: tokens(prompt), maxNewTokens: maxNew,
            stopTokenIDs: [], temperature: 0, branchingBudget: 4)
        let stream = SpecDecStream.streamDDTree(
            args: args2, tokenizer: DecimalTokenizer(
                vocabularySize: Self.vocabSize))

        var chunks: [String] = []
        var infoCount = 0
        for await ev in stream {
            switch ev {
            case .chunk(let c): chunks.append(c)
            case .prefillProgress:

                break
            case .info: infoCount += 1
            case .reasoning, .prefillProgress, .toolCall: break
            @unknown default: break
            }
        }
        #expect(infoCount == 1)
        let streamedText = chunks.joined()
        let expectedGenerated = bulk.tokenIds.suffix(bulk.tokenIds.count - prompt.count)
        let expectedText = expectedGenerated.map { String($0) }.joined(separator: "|") + (expectedGenerated.isEmpty ? "" : "|")
        #expect(streamedText == expectedText)
    }
}
