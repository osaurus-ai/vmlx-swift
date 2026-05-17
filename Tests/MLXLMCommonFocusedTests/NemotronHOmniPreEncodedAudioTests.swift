// Copyright © 2026 Osaurus AI. All rights reserved.

import Foundation
import MLX
import MLXLMCommon
import MLXVLM
import Testing

private struct FocusedOmniTokenizer: Tokenizer {
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        [1, 18, 27, 2]
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        tokenIds.map(String.init).joined(separator: " ")
    }

    func convertTokenToId(_ token: String) -> Int? {
        switch token {
        case "<image>": 18
        case "<so_embedding>": 27
        default: nil
        }
    }

    func convertIdToToken(_ id: Int) -> String? {
        switch id {
        case 18: "<image>"
        case 27: "<so_embedding>"
        default: String(id)
        }
    }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        [1, 18, 27, 2]
    }
}

@Suite("Nemotron H Omni pre-encoded audio")
struct NemotronHOmniPreEncodedAudioTests {
    @Test("live audio buffer keeps full snapshot while streaming chunks")
    func liveAudioBufferSnapshotAndStreamingCursor() {
        let buffer = NemotronHOmniLiveAudioBuffer(sampleRate: 4)

        #expect(buffer.snapshot().samples == [])
        #expect(buffer.durationSeconds == 0)

        buffer.append([1, 2, 3])
        let firstChunk = buffer.consumeAvailableSamples()
        #expect(firstChunk.samples == [1, 2, 3])
        #expect(firstChunk.sampleRate == 4)
        #expect(abs(firstChunk.durationSeconds - 0.75) < 0.0001)
        #expect(buffer.consumeAvailableSamples().samples == [])

        buffer.append([4, 5])
        #expect(buffer.consumeAvailableSamples().samples == [4, 5])
        #expect(buffer.snapshot().samples == [1, 2, 3, 4, 5])

        buffer.resetConsumeCursor()
        #expect(buffer.consumeAvailableSamples().samples == [1, 2, 3, 4, 5])

        buffer.clear()
        #expect(buffer.snapshot().samples == [])
        #expect(buffer.retainedSampleCount == 0)
    }

    @Test("processor preserves caller supplied Parakeet embedding")
    func processorPreservesPreEncodedAudioEmbedding() async throws {
        try await FocusedMLXTestSupport.withLock {
            let processor = NemotronHOmniProcessor(
                NemotronHOmniProcessorConfiguration(),
                tokenizer: FocusedOmniTokenizer())
            let samples = [Float](repeating: 0.05, count: 1_600)
            let embedding = MLXArray.zeros([5, 2_688])

            let input = UserInput(
                prompt: "What did the caller say?",
                audios: [
                    .preEncoded(samples: samples, sampleRate: 16_000, embedding: embedding)
                ])
            let lmInput = try await processor.prepare(input: input)

            #expect(lmInput.audio?.waveform.shape == [1, samples.count])
            #expect(lmInput.audio?.sampleRate == 16_000)
            #expect(lmInput.audio?.preEncodedEmbedding?.shape == [5, 2_688])
            #expect(lmInput.mediaTokenIds == [18, 27])
        }
    }

    @Test("video EVS count matches LMInput placeholder contract")
    func videoEVSCountMatchesSourceTokenCount() {
        FocusedMLXTestSupport.withLock {
            let feats = MLXArray.zeros([16, 256, 8])
            let pruned = nemotronOmniApplyEVS(feats, pruningRate: 0.7)
            #expect(pruned.shape == [1, 1228, 8])

            let targetPruned = nemotronOmniApplyEVS(feats, targetTokenCount: 1024)
            #expect(targetPruned.shape == [1, 1024, 8])
            #expect(
                NemotronHOmniProcessor.videoTokenCountAfterEVS(
                    groups: 16, tokensPerGroup: 256, pruningRate: 0.7) == 1228)

            let video = LMInput.ProcessedVideo(
                pixels: MLXArray.zeros([16, 3, 512, 512]),
                frames: nil,
                embeddingTokenCount: 1228)
            #expect(video.embeddingTokenCount == 1228)
        }
    }

    @Test("RADIO pixel shuffle preserves expected downsample shape")
    func radioPixelShuffleScaleHalfShape() {
        FocusedMLXTestSupport.withLock {
            let input = MLXArray.zeros([1, 4, 4, 16])
            let output = nemotronOmniPixelShuffle(input, scaleFactor: 0.5)
            #expect(output.shape == [1, 2, 2, 64])
        }
    }

    @Test("Parakeet relative shift keeps query/key square")
    func parakeetRelativeShiftShape() {
        FocusedMLXTestSupport.withLock {
            let scores = MLXArray(
                (0 ..< (2 * 2 * 3 * 5)).map { Float($0) }
            ).reshaped([2, 2, 3, 5])
            let shifted = nemotronOmniRelShift(scores, seqLen: 3)
            #expect(shifted.shape == [2, 2, 3, 3])
        }
    }

    @Test("projector weight remaps match Nemotron source layout")
    func projectorWeightRemapsMatchSourceLayout() {
        FocusedMLXTestSupport.withLock {
            let mlpRaw: [String: MLXArray] = [
                "mlp1.0.weight": MLXArray.zeros([5120]),
                "mlp1.0.bias": MLXArray.zeros([5120]),
                "mlp1.1.weight": MLXArray.zeros([20_480, 5120]),
                "mlp1.3.weight": MLXArray.zeros([2688, 20_480]),
                "irrelevant.weight": MLXArray.zeros([1]),
            ]
            let mlp = remapMlp1Weights(mlpRaw)
            #expect(mlp["layer_norm.weight"]?.shape == [5120])
            #expect(mlp["layer_norm.bias"]?.shape == [5120])
            #expect(mlp["fc1.weight"]?.shape == [20_480, 5120])
            #expect(mlp["fc2.weight"]?.shape == [2688, 20_480])
            #expect(mlp["irrelevant.weight"] == nil)

            let soundRaw: [String: MLXArray] = [
                "sound_projection.norm.weight": MLXArray.zeros([1024]),
                "sound_projection.linear1.weight": MLXArray.zeros([4096, 1024]),
                "sound_projection.linear2.weight": MLXArray.zeros([2688, 4096]),
                "sound_projection.linear1.bias": MLXArray.zeros([4096]),
                "skip.me": MLXArray.zeros([1]),
            ]
            let sound = remapSoundProjectionWeights(soundRaw)
            #expect(sound["norm.weight"]?.shape == [1024])
            #expect(sound["linear1.weight"]?.shape == [4096, 1024])
            #expect(sound["linear2.weight"]?.shape == [2688, 4096])
            #expect(sound["skip.me"] == nil)
        }
    }

    @Test("Parakeet source weights transpose to MLX layouts")
    func parakeetWeightRemapsTransposeToMLXLayouts() {
        FocusedMLXTestSupport.withLock {
            let conv2d: [String: MLXArray] = [
                "sound_encoder.encoder.subsampling.layers.0.weight":
                    MLXArray.zeros([256, 1, 3, 3]),
                "sound_encoder.encoder.subsampling.layers.0.bias":
                    MLXArray.zeros([256]),
            ]
            let conv2dOut = remapParakeetWeights(conv2d)
            #expect(conv2dOut["subsampling.layers_0.weight"]?.shape == [256, 3, 3, 1])

            let conv1d: [String: MLXArray] = [
                "sound_encoder.encoder.layers.0.conv.pointwise_conv1.weight":
                    MLXArray.zeros([2048, 1024, 1])
            ]
            let conv1dOut = remapParakeetWeights(conv1d)
            #expect(conv1dOut["layers.0.conv.pointwise_conv1.weight"]?.shape == [2048, 1, 1024])
        }
    }

    @Test("audio latency bench uses bundle generation defaults")
    func audioLatencyBenchUsesGenerationConfig() throws {
        let path = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appending(path: "tools/OmniAudioLatencyBench/main.swift")
        let source = try String(contentsOf: path)
        #expect(source.contains(
            "GenerateParameters(\n            generationConfig: context.configuration.generationDefaults)"))
        #expect(!source.contains("params.temperature = 0.0"))
        #expect(source.contains("\"event\": \"sampling\""))
        #expect(source.contains("rounded(Double(samplingProbe.topP), places: 3)"))
    }
}
