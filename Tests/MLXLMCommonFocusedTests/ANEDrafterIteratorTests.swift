// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// NativeMTPTokenIterator end to end with the Neural Engine drafter on a
// tiny random Qwen3.5: greedy output must equal the GPU-head iterator's
// (the verifier is the truth either way), the ANE must actually have
// drafted, and the accept/reject bookkeeping must match the GPU head's
// cycle for cycle. With random weights the head predicts nothing, so
// acceptance itself is a live-model question (see ANE-MTP-DRAFTER.md).
// Skips without an ANE.

import Foundation
import MLX
import MLXNN
import MLXRandom
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

@Suite("ANE drafter in the iterator", .serialized)
struct ANEDrafterIteratorTests {

    private static func makeModel() throws -> Qwen35TextModel {
        let json = """
            {
              "hidden_size": 128,
              "num_attention_heads": 4,
              "num_key_value_heads": 2,
              "head_dim": 32,
              "intermediate_size": 256,
              "vocab_size": 512,
              "num_hidden_layers": 4,
              "full_attention_interval": 2,
              "mtp_num_hidden_layers": 1,
              "num_experts": 0,
              "partial_rotary_factor": 0.25,
              "rms_norm_eps": 1e-6
            }
            """
        let config = try JSONDecoder().decode(Qwen35TextConfiguration.self, from: Data(json.utf8))
        let model = Qwen35TextModel(config)
        MLX.eval(model.parameters())
        return model
    }

    private static func generate(_ model: Qwen35TextModel, ane: Bool, maxTokens: Int) throws -> (tokens: [Int], iterator: NativeMTPTokenIterator) {
        setenv("VMLX_ANE_MTP", ane ? "1" : "0", 1)
        setenv("VMLX_ANE_MTP_VOCAB", "512", 1)
        setenv("VMLX_ANE_MTP_WINDOW", "64", 1)
        defer { unsetenv("VMLX_ANE_MTP") }
        var parameters = GenerateParameters(maxTokens: maxTokens, temperature: 0)
        parameters.draftStrategy = .nativeMTP(depth: 3)
        let prompt = MLXArray([Int32]([11, 42, 7, 99, 3, 250, 8]))
        var iterator = try NativeMTPTokenIterator(
            input: LMInput(text: .init(tokens: prompt)),
            model: model,
            parameters: parameters,
            depth: 3)
        var out: [Int] = []
        while let t = iterator.next(), out.count < maxTokens { out.append(t) }
        return (out, iterator)
    }

    @Test("greedy output and accept/reject bookkeeping match the GPU-head iterator")
    func aneMatchesGPU() throws {
        guard ANEProgram.isAvailable else { print("ANE unavailable; skipping"); return }
        try FocusedMLXTestSupport.withLock {
            MLXRandom.seed(7)
            let model = try Self.makeModel()
            let gpu = try Self.generate(model, ane: false, maxTokens: 24)
            let ane = try Self.generate(model, ane: true, maxTokens: 24)
            print("gpu: \(gpu.tokens)")
            print("ane: \(ane.tokens)")
            print("gpu iterator: mtpForwards \(gpu.iterator.mtpForwardCount) accepted \(gpu.iterator.acceptedByDepth) rejected \(gpu.iterator.rejectedCount)")
            print("ane iterator: aneForwards \(ane.iterator.aneDraftForwardCount) accepted \(ane.iterator.acceptedByDepth) rejected \(ane.iterator.rejectedCount)")
            #expect(ane.tokens == gpu.tokens)
            #expect(ane.iterator.aneDraftForwardCount > 0)
            #expect(ane.iterator.aneDraftForwardCount == gpu.iterator.mtpForwardCount)
            #expect(ane.iterator.acceptedByDepth == gpu.iterator.acceptedByDepth)
            #expect(ane.iterator.rejectedCount == gpu.iterator.rejectedCount)
        }
    }
}
