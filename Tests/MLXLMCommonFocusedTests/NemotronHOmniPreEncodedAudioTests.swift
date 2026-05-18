// Copyright © 2026 Osaurus AI. All rights reserved.

import Foundation
import CoreImage
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

private struct FocusedOmniMediaTokenizer: Tokenizer {
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        var ids: [Int] = addSpecialTokens ? [1] : []
        var cursor = text.startIndex
        while cursor < text.endIndex {
            let suffix = text[cursor...]
            if suffix.hasPrefix("<so_start>") {
                ids.append(28)
                cursor = text.index(cursor, offsetBy: "<so_start>".count)
            } else if suffix.hasPrefix("<so_end>") {
                ids.append(29)
                cursor = text.index(cursor, offsetBy: "<so_end>".count)
            } else if suffix.hasPrefix("<so_embedding>") {
                ids.append(27)
                cursor = text.index(cursor, offsetBy: "<so_embedding>".count)
            } else if suffix.hasPrefix("<sound>") {
                ids.append(contentsOf: [1060, 95_690, 1062])
                cursor = text.index(cursor, offsetBy: "<sound>".count)
            } else if suffix.hasPrefix("</sound>") {
                ids.append(contentsOf: [1885, 95_690, 1062])
                cursor = text.index(cursor, offsetBy: "</sound>".count)
            } else if suffix.hasPrefix("<image>") {
                ids.append(18)
                cursor = text.index(cursor, offsetBy: "<image>".count)
            } else {
                cursor = text.index(after: cursor)
            }
        }
        if addSpecialTokens { ids.append(2) }
        return ids
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        tokenIds.map(String.init).joined(separator: " ")
    }

    func convertTokenToId(_ token: String) -> Int? {
        switch token {
        case "<image>": 18
        case "<so_embedding>": 27
        case "<so_start>": 28
        case "<so_end>": 29
        default: nil
        }
    }

    func convertIdToToken(_ id: Int) -> String? {
        switch id {
        case 18: "<image>"
        case 27: "<so_embedding>"
        case 28: "<so_start>"
        case 29: "<so_end>"
        default: String(id)
        }
    }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        let text = messages.compactMap { $0["content"].map(String.init(describing:)) }
            .joined(separator: "\n")
        return [1] + encode(text: text, addSpecialTokens: false) + [2]
    }
}

private final class RecordingOmniMediaTokenizer: GenerationPromptControllableTokenizer, @unchecked Sendable {
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }

    private let lock = NSLock()
    private var _messages: [[String: any Sendable]] = []
    private var _historyTokenCount: Int = 0

    var messages: [[String: any Sendable]] {
        lock.lock()
        defer { lock.unlock() }
        return _messages
    }

    var historyTokenCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return _historyTokenCount
    }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        var ids: [Int] = addSpecialTokens ? [1] : []
        var cursor = text.startIndex
        while cursor < text.endIndex {
            let suffix = text[cursor...]
            if suffix.hasPrefix("<image>") {
                ids.append(18)
                cursor = text.index(cursor, offsetBy: "<image>".count)
            } else if suffix.hasPrefix("<so_embedding>") {
                ids.append(27)
                cursor = text.index(cursor, offsetBy: "<so_embedding>".count)
            } else {
                ids.append(7)
                cursor = text.index(after: cursor)
            }
        }
        if addSpecialTokens { ids.append(2) }
        return ids
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
        try applyChatTemplate(
            messages: messages,
            tools: tools,
            additionalContext: additionalContext,
            addGenerationPrompt: true)
    }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools _: [[String: any Sendable]]?,
        additionalContext _: [String: any Sendable]?,
        addGenerationPrompt: Bool
    ) throws -> [Int] {
        lock.lock()
        _messages = messages
        lock.unlock()
        var text = messages.compactMap { $0["content"].map(String.init(describing:)) }
            .joined(separator: "\n")
        let historyTokens = [1] + encode(text: text, addSpecialTokens: false)
        if addGenerationPrompt {
            text += "\n<assistant>"
            let promptTokens = [1] + encode(text: text, addSpecialTokens: false)
            lock.lock()
            _historyTokenCount = historyTokens.count
            lock.unlock()
            return promptTokens
        }
        return historyTokens
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

    @Test("processor uses source-compatible audio wrapper tokens")
    func processorUsesSourceCompatibleAudioWrapperTokens() async throws {
        try await FocusedMLXTestSupport.withLock {
            let processor = NemotronHOmniProcessor(
                NemotronHOmniProcessorConfiguration(),
                tokenizer: FocusedOmniMediaTokenizer())
            let embedding = MLXArray.zeros([5, 2_688])
            let input = UserInput(
                prompt: "Briefly describe what you hear.",
                audios: [
                    .preEncoded(
                        samples: [Float](repeating: 0.0, count: 1_600),
                        sampleRate: 16_000,
                        embedding: embedding)
                ])

            let lmInput = try await processor.prepare(input: input)
            let tokens = lmInput.text.tokens.reshaped(-1).asArray(Int.self)

            #expect(tokens.contains(28))
            #expect(tokens.contains(29))
            #expect(tokens.filter { $0 == 27 }.count == 5)
            #expect(!tokens.contains(95_690))
        }
    }

    @Test("chat media placeholders stay on the media-bearing user turn")
    func chatMediaPlaceholdersStayOnMediaBearingTurn() async throws {
        try await FocusedMLXTestSupport.withLock {
            let tokenizer = RecordingOmniMediaTokenizer()
            let processor = NemotronHOmniProcessor(
                NemotronHOmniProcessorConfiguration(),
                tokenizer: tokenizer)
            let image = CIImage(color: .red).cropped(
                to: CGRect(x: 0, y: 0, width: 16, height: 16))
            let input = UserInput(
                chat: [
                    .user("Describe the image.", images: [.ciImage(image)]),
                    .assistant("It is a red square."),
                    .user("What color did it have?"),
                ],
                additionalContext: ["enable_thinking": false])

            let lmInput = try await processor.prepare(input: input)
            let messages = tokenizer.messages

            #expect(lmInput.image != nil)
            #expect(messages.count == 3)
            let firstUser = String(describing: messages[0]["content"] ?? "")
            let finalUser = String(describing: messages[2]["content"] ?? "")
            #expect(firstUser.contains("<img>"))
            #expect(firstUser.contains("<image>"))
            #expect(firstUser.contains("Describe the image."))
            #expect(!finalUser.contains("<image>"))
            #expect(finalUser == "What color did it have?")
            #expect(lmInput.cachePrefixTokenCounts == [tokenizer.historyTokenCount])
        }
    }

    @Test("image preprocessing keeps both red and blue gradient endpoints")
    func imagePreprocessingKeepsGradientEndpoints() throws {
        try FocusedMLXTestSupport.withLock {
            let image = try Self.syntheticRedBlueGradient(side: 224)
            let (pixels, counts) = try nemotronOmniPreprocessImages([image])

            #expect(counts == [256])
            #expect(pixels.shape == [1, 3, 512, 512])

            let values = pixels.asArray(Float.self)
            let plane = 512 * 512
            func channel(_ c: Int, _ y: Int, _ x: Int) -> Float {
                values[c * plane + y * 512 + x]
            }
            func denormalize(_ value: Float, channel c: Int) -> Float {
                value * NEMOTRON_OMNI_CLIP_STD[c] + NEMOTRON_OMNI_CLIP_MEAN[c]
            }

            let edgeA = (
                r: denormalize(channel(0, 8, 256), channel: 0),
                b: denormalize(channel(2, 8, 256), channel: 2))
            let edgeB = (
                r: denormalize(channel(0, 503, 256), channel: 0),
                b: denormalize(channel(2, 503, 256), channel: 2))

            let hasRedDominantEdge = edgeA.r > edgeA.b + 0.4 || edgeB.r > edgeB.b + 0.4
            let hasBlueDominantEdge = edgeA.b > edgeA.r + 0.4 || edgeB.b > edgeB.r + 0.4
            #expect(hasRedDominantEdge)
            #expect(hasBlueDominantEdge)
        }
    }

    @Test("image preprocessing follows source dynamic-resolution token count")
    func imagePreprocessingFollowsSourceDynamicResolution() throws {
        try FocusedMLXTestSupport.withLock {
            let image = Self.syntheticSolidImage(width: 224, height: 448)
            let (pixels, counts) = try nemotronOmniPreprocessImages([image])

            #expect(counts == [276])
            #expect(pixels.shape == [1, 3, 736, 384])
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

    private static func syntheticRedBlueGradient(side: Int) throws -> CIImage {
        var bytes = [UInt8](repeating: 0, count: side * side * 4)
        for y in 0 ..< side {
            let r = UInt8(255 - (255 * y) / max(side - 1, 1))
            let b = UInt8((255 * y) / max(side - 1, 1))
            for x in 0 ..< side {
                let off = (y * side + x) * 4
                bytes[off + 0] = r
                bytes[off + 1] = 64
                bytes[off + 2] = b
                bytes[off + 3] = 255
            }
        }
        let image = CIImage(
            bitmapData: Data(bytes),
            bytesPerRow: side * 4,
            size: CGSize(width: side, height: side),
            format: .RGBA8,
            colorSpace: CGColorSpace(name: CGColorSpace.sRGB)!)
        return image
    }

    private static func syntheticSolidImage(width: Int, height: Int) -> CIImage {
        var bytes = [UInt8](repeating: 0, count: width * height * 4)
        for y in 0 ..< height {
            for x in 0 ..< width {
                let off = (y * width + x) * 4
                bytes[off + 0] = 180
                bytes[off + 1] = 64
                bytes[off + 2] = 32
                bytes[off + 3] = 255
            }
        }
        return CIImage(
            bitmapData: Data(bytes),
            bytesPerRow: width * 4,
            size: CGSize(width: width, height: height),
            format: .RGBA8,
            colorSpace: CGColorSpace(name: CGColorSpace.sRGB)!)
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
