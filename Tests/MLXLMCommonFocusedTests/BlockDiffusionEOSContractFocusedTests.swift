// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLMCommon

@Suite("Block diffusion native EOS and immutable logits")
struct BlockDiffusionEOSContractFocusedTests {
    @Test("the first canvas may emit EOS with native self-conditioning and untouched logits")
    func firstPositionEOSIsNotRescued() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = EOSDiffusionFixture(firstEOSPosition: 0)
            let logitsAlias = model.decoderLogits
            let before = logitsAlias.asArray(Float.self)
            let prompt = MLXArray([Int32(7), 8, 9])
            let promptAlias = prompt
            var iterator = try BlockDiffusionTokenIterator(
                input: LMInput(tokens: prompt), model: model,
                parameters: GenerateParameters(maxTokens: 8, randomSeed: 42),
                options: model.blockDiffusionDefaults)
            let start = Date()
            var output: [Int] = []
            while let token = iterator.next() { output.append(token) }
            #expect(output == [model.eos])
            #expect(iterator.next() == nil)
            #expect(model.decoderInputs.count >= 2)
            #expect(model.decoderInputs[0] == nil)
            // Unity native schedule: the same unmasked logits self-condition
            // the next step; even first-position EOS must remain untouched.
            #expect(model.decoderInputs[1] == before)
            #expect(model.decoderLogits.asArray(Float.self) == before)
            #expect(logitsAlias.asArray(Float.self) == before)
            #expect(promptAlias.asArray(Int32.self) == [7, 8, 9])
            #expect(model.encoderInputs == [[7, 8, 9]])
            print("NATIVE_DIFFUSION_SYNTHETIC tokens=\(output.count) tokens_per_second=\(Double(output.count) / max(Date().timeIntervalSince(start), 1e-9)) CPU=true real_model_proof=false")
        }
    }

    @Test("native EOS still truncates a canvas and commits only preceding content")
    func nativeEOSAndCacheCommitRemainIntact() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = EOSDiffusionFixture(firstEOSPosition: 1)
            var iterator = try BlockDiffusionTokenIterator(
                input: LMInput(tokens: MLXArray([Int32(7)])), model: model,
                parameters: GenerateParameters(maxTokens: 8, randomSeed: 42),
                options: model.blockDiffusionDefaults)
            let start = Date()
            var output: [Int] = []
            while let token = iterator.next() { output.append(token) }
            #expect(output == [0, model.eos])
            #expect(iterator.tokenCount == 2)
            #expect(model.encoderInputs == [[7], [0]])
            print("NATIVE_DIFFUSION_SYNTHETIC tokens=\(output.count) tokens_per_second=\(Double(output.count) / max(Date().timeIntervalSince(start), 1e-9)) CPU=true real_model_proof=false")
        }
    }

    @Test("ordinary generation routes block diffusion to the unmasked native iterator")
    func defaultDispatchReachability() throws {
        let evaluate = try String(contentsOfFile: "Libraries/MLXLMCommon/Evaluate.swift", encoding: .utf8)
        #expect(evaluate.contains("if let diffusionModel = context.model as? any BlockDiffusionModel"))
        #expect(evaluate.contains("let iterator = try BlockDiffusionTokenIterator("))
        let iterator = try String(
            contentsOfFile: "Libraries/MLXLMCommon/Diffusion/BlockDiffusionTokenIterator.swift", encoding: .utf8)
        #expect(iterator.contains("MLX.categorical(processed, key: denoiserKey)"))
        #expect(iterator.contains("argMax(processed, axis: -1)"))
        #expect(iterator.contains("selfConditioning = processed"))
        #expect(!iterator.contains("sampled[0, 0, eos]"))
    }
}

/// No weights or tokenizer: exercise the real native iterator with a model
/// distribution that prefers EOS, including at the very first position.
private final class EOSDiffusionFixture: Module, BlockDiffusionModel {
    let eos = 2
    let decoderLogits: MLXArray
    var decoderInputs: [[Float]?] = []
    var encoderInputs: [[Int32]] = []
    var diffusionVocabularySize: Int { 4 }
    var blockDiffusionDefaults: BlockDiffusionParameters {
        BlockDiffusionParameters(
            canvasLength: 3, maxDenoisingSteps: 3, tMin: 1, tMax: 1,
            stabilityThreshold: 1, confidenceThreshold: 0.005, eosTokenIds: [eos])
    }

    init(firstEOSPosition: Int) {
        var values = [Float](repeating: -100, count: 12)
        for position in 0 ..< 3 {
            values[position * 4 + (position == firstEOSPosition ? 2 : 0)] = 100
        }
        decoderLogits = MLXArray(values).reshaped(1, 3, 4)
        super.init()
    }

    func newCache(parameters: GenerateParameters?) -> [KVCache] { [] }
    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        throw BlockDiffusionModelError.requiresBlockDiffusionEngine("test-fixture")
    }
    func encoderForward(_ tokens: MLXArray, cache: [KVCache]) {
        encoderInputs.append(tokens.reshaped(-1).asArray(Int32.self))
    }
    func decoderForward(
        canvas: MLXArray, cache: [KVCache], selfConditioningLogits: MLXArray?
    ) -> MLXArray {
        decoderInputs.append(selfConditioningLogits?.asArray(Float.self))
        return decoderLogits
    }
}
