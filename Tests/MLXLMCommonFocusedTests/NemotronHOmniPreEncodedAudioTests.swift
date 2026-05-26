// Copyright © 2026 Osaurus AI. All rights reserved.

import Foundation
import MLX
import MLXLMCommon
@testable import MLXVLM
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

private struct FocusedOmniTemplateTokenizer: Tokenizer {
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        text.unicodeScalars.map { Int($0.value) }
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        var result = ""
        for id in tokenIds {
            if let scalar = UnicodeScalar(id) {
                result.unicodeScalars.append(scalar)
            }
        }
        return result
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
        var rendered = ""
        for message in messages {
            let role = (message["role"] as? String) ?? "user"
            rendered += "<|im_start|>\(role)\n"
            rendered += String(describing: message["content"] ?? "")
            rendered += "<|im_end|>\n"
        }
        rendered += "<|im_start|>assistant\n"
        if additionalContext?["enable_thinking"] as? Bool == false {
            rendered += "<think></think>"
        } else {
            rendered += "<think>\n"
        }
        return encode(text: rendered, addSpecialTokens: false)
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

    @Test("processor preserves bundle compact no-thinking media tail")
    func processorPreservesCompactNoThinkingMediaTail() async throws {
        try await FocusedMLXTestSupport.withLock {
            let tokenizer = FocusedOmniTemplateTokenizer()
            let processor = NemotronHOmniProcessor(
                NemotronHOmniProcessorConfiguration(),
                tokenizer: tokenizer)
            var input = UserInput(
                prompt: "Briefly describe what you hear.",
                audios: [
                    .preEncoded(
                        samples: [Float](repeating: 0.0, count: 1_600),
                        sampleRate: 16_000,
                        embedding: MLXArray.zeros([5, 2_688]))
                ])
            input.additionalContext = ["enable_thinking": false]

            let lmInput = try await processor.prepare(input: input)
            let rendered = tokenizer.decode(
                tokenIds: lmInput.text.tokens.reshaped(-1).asArray(Int.self),
                skipSpecialTokens: false)

            #expect(rendered.hasSuffix("<|im_start|>assistant\n<think></think>"))
            #expect(!rendered.hasSuffix("<|im_start|>assistant\n<think>\n</think>\n\n"))
        }
    }

    @Test("required tool choice reaches Nemotron Omni VLM native template path")
    func requiredToolChoiceReachesNemotronOmniVLMTemplatePath() {
        var messages: [Message] = [
            ["role": "user", "content": "Use line_count on red\ngreen\nblue."],
        ]

        NemotronHOmniProcessor.addRequiredToolChoiceInstruction(
            to: &messages,
            tools: [lineCountTool()],
            additionalContext: ["tool_choice": "required"])

        #expect(messages.count == 3)
        #expect(messages[0]["role"] as? String == "system")
        let system = messages[0]["content"] as? String
        #expect(system?.contains("exactly one <tool_call> XML function call") == true)
        #expect(system?.contains("Include every required <parameter=...>") == true)
        #expect(messages[1]["content"] as? String == "Use line_count on red\ngreen\nblue.")
        #expect(messages[2]["role"] as? String == "system")
        #expect(messages[2]["content"] as? String == system)
    }

    @Test("required tool choice repeats at the current Nemotron Omni turn after no-tool history")
    func requiredToolChoiceRepeatsAtCurrentNemotronOmniTurnAfterNoToolHistory() throws {
        var messages: [Message] = [
            ["role": "user", "content": "Use line_count on red\ngreen\nblue."],
            [
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    [
                        "id": "call_lines",
                        "type": "function",
                        "function": [
                            "name": "line_count",
                            "arguments": #"{"text":"red\ngreen\nblue"}"#,
                        ] as [String: any Sendable],
                    ] as [String: any Sendable],
                ],
            ] as [String: any Sendable],
            ["role": "tool", "tool_call_id": "call_lines", "content": #"{"lines":3}"#],
            [
                "role": "user",
                "content": "How many lines were counted? Answer plainly. Do not call another tool.",
            ],
            ["role": "assistant", "content": "Three lines were counted."],
            ["role": "user", "content": "Now use line_count on one\ntwo."],
        ]

        NemotronHOmniProcessor.addRequiredToolChoiceInstruction(
            to: &messages,
            tools: [lineCountTool()],
            additionalContext: ["tool_choice": "required"])

        let finalUserIndex = try #require(messages.lastIndex {
            ($0["role"] as? String) == "user"
                && (($0["content"] as? String)?.contains("one\ntwo") == true)
        })
        let tailInstructionIndex = try #require(messages.lastIndex {
            ($0["role"] as? String) == "system"
                && (($0["content"] as? String)?.contains(
                    NemotronHOmniProcessor.requiredToolChoiceInstruction) == true)
        })
        #expect(tailInstructionIndex > finalUserIndex)
    }

    @Test("non-required tool choice leaves Nemotron Omni VLM messages unchanged")
    func nonRequiredToolChoiceLeavesNemotronOmniVLMMessagesUnchanged() {
        var messages: [Message] = [
            ["role": "user", "content": "hello"],
        ]

        NemotronHOmniProcessor.addRequiredToolChoiceInstruction(
            to: &messages,
            tools: [lineCountTool()],
            additionalContext: ["tool_choice": "none"])

        #expect(messages.count == 1)
        #expect(messages[0]["role"] as? String == "user")
        #expect(messages[0]["content"] as? String == "hello")
    }

    private func lineCountTool() -> ToolSpec {
        [
            "type": "function",
            "function": [
                "name": "line_count",
                "description": "Count newline-separated text lines.",
                "parameters": [
                    "type": "object",
                    "properties": [
                        "text": ["type": "string"] as [String: any Sendable],
                    ] as [String: any Sendable],
                    "required": ["text"],
                ] as [String: any Sendable],
            ] as [String: any Sendable],
        ]
    }

    @Test("media no-thinking prompt carries explicit direct-answer instruction")
    func mediaNoThinkingPromptCarriesDirectAnswerInstruction() async throws {
        try await FocusedMLXTestSupport.withLock {
            let tokenizer = FocusedOmniTemplateTokenizer()
            let processor = NemotronHOmniProcessor(
                NemotronHOmniProcessorConfiguration(),
                tokenizer: tokenizer)
            var input = UserInput(
                prompt: "Briefly describe what you hear.",
                audios: [
                    .preEncoded(
                        samples: [Float](repeating: 0.0, count: 1_600),
                        sampleRate: 16_000,
                        embedding: MLXArray.zeros([5, 2_688]))
                ])
            input.additionalContext = ["enable_thinking": false]

            let lmInput = try await processor.prepare(input: input)
            let rendered = tokenizer.decode(
                tokenIds: lmInput.text.tokens.reshaped(-1).asArray(Int.self),
                skipSpecialTokens: false)

            #expect(rendered.contains(
                "Answer directly with only the final visible response."))
            #expect(rendered.contains(
                "Do not include analysis, reasoning, scratchpad steps, or drafts."))
            #expect(rendered.hasSuffix("<|im_start|>assistant\n<think></think>"))
        }
    }

    @Test("media thinking request uses direct-answer media contract")
    func mediaThinkingRequestUsesDirectAnswerMediaContract() async throws {
        try await FocusedMLXTestSupport.withLock {
            let tokenizer = FocusedOmniTemplateTokenizer()
            let processor = NemotronHOmniProcessor(
                NemotronHOmniProcessorConfiguration(),
                tokenizer: tokenizer)
            var input = UserInput(
                prompt: "Briefly describe what you hear.",
                audios: [
                    .preEncoded(
                        samples: [Float](repeating: 0.0, count: 1_600),
                        sampleRate: 16_000,
                        embedding: MLXArray.zeros([5, 2_688]))
                ])
            input.additionalContext = ["enable_thinking": true]

            let lmInput = try await processor.prepare(input: input)
            let rendered = tokenizer.decode(
                tokenIds: lmInput.text.tokens.reshaped(-1).asArray(Int.self),
                skipSpecialTokens: false)

            #expect(rendered.hasSuffix("<|im_start|>assistant\n<think></think>"))
            #expect(!rendered.hasSuffix("<|im_start|>assistant\n<think>\n"))
            #expect(rendered.contains(
                "Answer directly with only the final visible response."))
        }
    }

    @Test("media default prompt uses direct-answer media contract")
    func mediaDefaultPromptUsesDirectAnswerMediaContract() async throws {
        try await FocusedMLXTestSupport.withLock {
            let tokenizer = FocusedOmniTemplateTokenizer()
            let processor = NemotronHOmniProcessor(
                NemotronHOmniProcessorConfiguration(),
                tokenizer: tokenizer)
            let input = UserInput(
                prompt: "Briefly describe what you hear.",
                audios: [
                    .preEncoded(
                        samples: [Float](repeating: 0.0, count: 1_600),
                        sampleRate: 16_000,
                        embedding: MLXArray.zeros([5, 2_688]))
                ])

            let lmInput = try await processor.prepare(input: input)
            let rendered = tokenizer.decode(
                tokenIds: lmInput.text.tokens.reshaped(-1).asArray(Int.self),
                skipSpecialTokens: false)

            #expect(rendered.hasSuffix("<|im_start|>assistant\n<think></think>"))
            #expect(!rendered.hasSuffix("<|im_start|>assistant\n<think>\n"))
            #expect(rendered.contains(
                "Answer directly with only the final visible response."))
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

    @Test("video prompt uses source-style frame labels and keeps full placeholder budget")
    func videoPromptUsesFrameLabelsAndPlaceholderBudget() {
        let media = NemotronHOmniProcessor.videoPromptMedia(
            totalTokens: 4_096,
            groups: 16,
            tokensPerGroup: 256,
            temporalPatchDim: 2)

        #expect(media.hasPrefix("Frame 1 and frame 2: <img><image>"))
        #expect(media.contains("\nFrame 3 and frame 4: <img><image>"))
        #expect(media.contains("\nFrame 31 and frame 32: <img><image>"))
        #expect(!media.contains("<video>"))
        #expect(media.hasSuffix("</img>\n"))
        #expect(media.components(separatedBy: "<image>").count - 1 == 4_096)
    }

    @Test("EVS keep indices retain first group and target count")
    func evsKeepIndicesRetainFirstGroupAndTargetCount() {
        FocusedMLXTestSupport.withLock {
            let feats = MLXArray.zeros([16, 256, 8])
            let keep = nemotronOmniEVSKeepIndices(feats, targetTokenCount: 1_024)
            #expect(keep.count == 1_024)
            #expect(Array(keep.prefix(256)) == Array(0 ..< 256))
            #expect(keep == keep.sorted())
        }
    }

    @Test("video target size preserves aspect ratio and source patch budget")
    func videoTargetSizePreservesAspectRatioAndPatchBudget() {
        let target = nemotronOmniVideoTargetSize(width: 1_920, height: 1_080)

        #expect(target.width == 672)
        #expect(target.height == 384)
        #expect(target.tokens == 252)
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
