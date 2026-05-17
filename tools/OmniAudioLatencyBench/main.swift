import Darwin
import Foundation
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import MLXVLM
@preconcurrency import Tokenizers

@main
struct OmniAudioLatencyBench {
    static func main() async throws {
        setvbuf(stdout, nil, _IONBF, 0)

        let env = ProcessInfo.processInfo.environment
        guard let modelPath = env["BENCH_MODEL"], !modelPath.isEmpty else {
            throw NSError(
                domain: "OmniAudioLatencyBench",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey:
                    "Set BENCH_MODEL to a local Nemotron Omni bundle path"])
        }
        let maxNewTokens = max(1, Int(env["BENCH_MAX_TOKENS"] ?? "8") ?? 8)
        try await OmniAudioLatencyRunner.run(modelPath: modelPath, maxNewTokens: maxNewTokens)
    }
}

enum OmniAudioLatencyRunner {
    static func run(modelPath: String, maxNewTokens: Int) async throws {
        let env = ProcessInfo.processInfo.environment
        let modelDir = URL(fileURLWithPath: modelPath)
        let audioPath = env["BENCH_AUDIO_FILE"]
            ?? "Tests/MLXLMTests/Resources/audio_only.mov"
        let audioURL = URL(fileURLWithPath: audioPath)
        let repeats = max(1, Int(env["BENCH_AUDIO_REPEATS"] ?? "3") ?? 3)
        let prompt = env["BENCH_AUDIO_PROMPT"]
            ?? "Briefly describe what you hear."
        let pathSetting = (env["BENCH_OMNI_AUDIO_PATH"] ?? "batch").lowercased()
        let paths = try audioPaths(from: pathSetting)
        let comparePreEncoded = (env["BENCH_OMNI_AUDIO_PREENCODE"] ?? "1") != "0"
        let enableDiskCache = (env["BENCH_OMNI_AUDIO_DISK_CACHE"] ?? "1") != "0"
        let cacheRoot = URL(fileURLWithPath: env["BENCH_OMNI_AUDIO_CACHE_DIR"]
            ?? "/tmp/vmlx-omni-audio-latency-\(ProcessInfo.processInfo.processIdentifier)")

        guard FileManager.default.fileExists(atPath: modelDir.appending(path: "config_omni.json").path)
        else {
            throw benchError("config_omni.json not found; expected a Nemotron Omni bundle")
        }
        guard FileManager.default.fileExists(atPath: audioURL.path) else {
            throw benchError("audio file not found: \(audioURL.path)")
        }
        if enableDiskCache {
            try? FileManager.default.removeItem(at: cacheRoot)
            try FileManager.default.createDirectory(
                at: cacheRoot, withIntermediateDirectories: true)
        }

        let loadStart = CFAbsoluteTimeGetCurrent()
        let context = try await MLXLMCommon.loadModel(
            from: modelDir, using: #huggingFaceTokenizerLoader())
        let loadMs = elapsedMs(since: loadStart)
        guard let omni = context.model as? NemotronHOmni else {
            throw benchError("loaded \(type(of: context.model)); expected NemotronHOmni")
        }

        printJSON([
            "event": "load",
            "model": modelDir.lastPathComponent,
            "model_path": modelDir.path,
            "load_ms": rounded(loadMs),
            "rss_mib": rounded(currentRSSMiB()),
        ])
        let samplingProbe = generationParameters(
            context: context, maxNewTokens: maxNewTokens)
        let repetitionPenalty: Any = samplingProbe.repetitionPenalty.map {
            rounded(Double($0))
        } ?? NSNull()
        printJSON([
            "event": "sampling",
            "model": modelDir.lastPathComponent,
            "source": "generation_config",
            "max_tokens": samplingProbe.maxTokens ?? -1,
            "temperature": rounded(Double(samplingProbe.temperature), places: 3),
            "top_p": rounded(Double(samplingProbe.topP), places: 3),
            "top_k": samplingProbe.topK,
            "min_p": rounded(Double(samplingProbe.minP), places: 3),
            "repetition_penalty": repetitionPenalty,
        ])

        let fileDecodeStart = CFAbsoluteTimeGetCurrent()
        let pcm = try nemotronOmniLoadAudioFile(audioURL, targetSampleRate: 16_000)
        let fileDecodeMs = elapsedMs(since: fileDecodeStart)
        let audioDurationMs = Double(pcm.count) / 16.0
        printJSON([
            "event": "audio_source",
            "model": modelDir.lastPathComponent,
            "audio_file": audioURL.path,
            "samples": pcm.count,
            "sample_rate": 16_000,
            "audio_duration_ms": rounded(audioDurationMs),
            "file_decode_ms": rounded(fileDecodeMs),
            "rss_mib": rounded(currentRSSMiB()),
        ])

        var scenarios: [(mode: String, audio: UserInput.Audio)] = [
            ("raw_samples", .samples(pcm, sampleRate: 16_000))
        ]
        if comparePreEncoded {
            let encodeStart = CFAbsoluteTimeGetCurrent()
            let audioEmbedding = omni.extractAudioEmbeds(waveform: pcm)
            MLX.eval(audioEmbedding)
            let preencodeMs = elapsedMs(since: encodeStart)
            printJSON([
                "event": "preencode",
                "model": modelDir.lastPathComponent,
                "mode": "preencoded",
                "audio_tokens": audioEmbedding.dim(0),
                "hidden_size": audioEmbedding.dim(1),
                "preencode_ms": rounded(preencodeMs),
                "rss_mib": rounded(currentRSSMiB()),
            ])
            scenarios.append(
                ("preencoded", .preEncoded(
                    samples: pcm, sampleRate: 16_000, embedding: audioEmbedding)))
        }

        for path in paths {
            for scenario in scenarios {
                let scenarioCacheDir = cacheRoot.appending(path: "\(path)-\(scenario.mode)")
                let coordinator = makeCoordinator(
                    modelName: modelDir.lastPathComponent,
                    path: path,
                    mode: scenario.mode,
                    enableDiskCache: enableDiskCache,
                    cacheDir: scenarioCacheDir)
                let engine: BatchEngine?
                if path == "batch" {
                    engine = BatchEngine(
                        context: context,
                        maxBatchSize: 1,
                        cacheCoordinator: coordinator)
                } else {
                    engine = nil
                }

                for turn in 1 ... repeats {
                    let result: TurnResult
                    switch path {
                    case "batch":
                        guard let engine else { continue }
                        result = try await runBatchTurn(
                            prompt: prompt,
                            audio: scenario.audio,
                            context: context,
                            engine: engine,
                            maxNewTokens: maxNewTokens)
                    case "iterator":
                        result = try await runIteratorTurn(
                            prompt: prompt,
                            audio: scenario.audio,
                            context: context,
                            coordinator: coordinator,
                            maxNewTokens: maxNewTokens)
                    default:
                        continue
                    }
                    var fields: [String: Any] = [
                        "event": "turn",
                        "model": modelDir.lastPathComponent,
                        "path": path,
                        "mode": scenario.mode,
                        "turn": turn,
                        "repeats": repeats,
                        "cache_reuse_candidate": turn > 1,
                        "disk_cache": enableDiskCache,
                        "cache_dir": scenarioCacheDir.path,
                        "audio_file": audioURL.path,
                        "audio_duration_ms": rounded(audioDurationMs),
                        "processor_prepare_ms": rounded(result.processorPrepareMs),
                        "iterator_init_ms": rounded(result.iteratorInitMs),
                        "stream_create_ms": rounded(result.streamCreateMs),
                        "first_delta_ms": rounded(result.firstDeltaMs),
                        "total_ms": rounded(result.totalMs),
                        "tokens": result.tokens,
                        "events": result.events,
                        "e2e_tokens_per_s": rounded(result.e2eTokensPerSecond),
                        "rss_mib": rounded(result.rssMiB),
                        "peak_rss_mib": rounded(result.peakRssMiB),
                        "text": result.text,
                    ]
                    fields.merge(result.promptDiagnostics.jsonFields) { _, new in new }
                    printJSON(fields)
                }
                if let engine {
                    await engine.shutdown()
                }
            }
        }
    }

    private struct TurnResult {
        let processorPrepareMs: Double
        let iteratorInitMs: Double
        let streamCreateMs: Double
        let firstDeltaMs: Double
        let totalMs: Double
        let tokens: Int
        let events: Int
        let e2eTokensPerSecond: Double
        let rssMiB: Double
        let peakRssMiB: Double
        let promptDiagnostics: PromptDiagnostics
        let text: String
    }

    private struct PromptDiagnostics {
        let promptTokens: Int
        let mediaTokenIdsKnown: Bool
        let mediaTokenIds: [Int]
        let mediaPlaceholderTokens: Int
        let firstMediaTokenIndex: Int
        let lastMediaTokenIndex: Int
        let blockSize: Int
        let blockSuffixMediaTokens: Int
        let promptMinusOneAfterMedia: Bool

        var jsonFields: [String: Any] {
            [
                "prompt_tokens": promptTokens,
                "media_token_ids_known": mediaTokenIdsKnown,
                "media_token_ids": mediaTokenIds,
                "media_placeholder_tokens": mediaPlaceholderTokens,
                "first_media_token_index": firstMediaTokenIndex,
                "last_media_token_index": lastMediaTokenIndex,
                "cache_block_size": blockSize,
                "block_suffix_media_tokens": blockSuffixMediaTokens,
                "block_suffix_contains_media": blockSuffixMediaTokens > 0,
                "prompt_minus_one_after_media": promptMinusOneAfterMedia,
            ]
        }
    }

    private static func runIteratorTurn(
        prompt: String,
        audio: UserInput.Audio,
        context: ModelContext,
        coordinator: CacheCoordinator,
        maxNewTokens: Int
    ) async throws -> TurnResult {
        let turnStart = CFAbsoluteTimeGetCurrent()
        let input = makeInput(prompt: prompt, audio: audio)
        let prepareStart = CFAbsoluteTimeGetCurrent()
        let lmInput = try await context.processor.prepare(input: input)
        let prepareMs = elapsedMs(since: prepareStart)
        let promptDiagnostics = promptDiagnostics(for: lmInput)
        let params = generationParameters(
            context: context, maxNewTokens: maxNewTokens)

        let iteratorStart = CFAbsoluteTimeGetCurrent()
        let iterator = try TokenIterator(
            input: lmInput,
            model: context.model,
            parameters: params,
            cacheCoordinator: coordinator)
        let iteratorMs = elapsedMs(since: iteratorStart)

        let stops = stopTokenIDs(context: context)
        var tokenIds: [Int] = []
        var firstDeltaMs: Double?
        var peakRSS = currentRSSMiB()
        for token in iterator {
            if stops.contains(token) { break }
            if firstDeltaMs == nil {
                firstDeltaMs = elapsedMs(since: turnStart)
            }
            tokenIds.append(token)
            peakRSS = max(peakRSS, currentRSSMiB())
            if tokenIds.count >= maxNewTokens { break }
        }

        let totalMs = elapsedMs(since: turnStart)
        let text = userVisibleText(
            context: context,
            lmInput: lmInput,
            tokenIds: tokenIds,
            allowReasoningFallback: false)
        return TurnResult(
            processorPrepareMs: prepareMs,
            iteratorInitMs: iteratorMs,
            streamCreateMs: 0,
            firstDeltaMs: firstDeltaMs ?? totalMs,
            totalMs: totalMs,
            tokens: tokenIds.count,
            events: tokenIds.count,
            e2eTokensPerSecond: tokensPerSecond(count: tokenIds.count, totalMs: totalMs),
            rssMiB: currentRSSMiB(),
            peakRssMiB: peakRSS,
            promptDiagnostics: promptDiagnostics,
            text: text.replacingOccurrences(of: "\n", with: " "))
    }

    private static func runBatchTurn(
        prompt: String,
        audio: UserInput.Audio,
        context: ModelContext,
        engine: BatchEngine,
        maxNewTokens: Int
    ) async throws -> TurnResult {
        let turnStart = CFAbsoluteTimeGetCurrent()
        let input = makeInput(prompt: prompt, audio: audio)
        let prepareStart = CFAbsoluteTimeGetCurrent()
        let lmInput = try await context.processor.prepare(input: input)
        let prepareMs = elapsedMs(since: prepareStart)
        let promptDiagnostics = promptDiagnostics(for: lmInput)
        let params = generationParameters(
            context: context, maxNewTokens: maxNewTokens)

        let streamStart = CFAbsoluteTimeGetCurrent()
        nonisolated(unsafe) let sendableInput = lmInput
        let stream = await engine.generate(input: sendableInput, parameters: params)
        let streamCreateMs = elapsedMs(since: streamStart)

        var text = ""
        var events = 0
        var generatedTokens = 0
        var firstDeltaMs: Double?
        var peakRSS = currentRSSMiB()
        for await event in stream {
            switch event {
            case .chunk(let chunk):
                guard !chunk.isEmpty else { continue }
                if firstDeltaMs == nil {
                    firstDeltaMs = elapsedMs(since: turnStart)
                }
                text += chunk
                events += 1
            case .reasoning(let reasoning):
                guard !reasoning.isEmpty else { continue }
                if firstDeltaMs == nil {
                    firstDeltaMs = elapsedMs(since: turnStart)
                }
                text += reasoning
                events += 1
            case .info(let info):
                generatedTokens = info.generationTokenCount
            default:
                break
            }
            peakRSS = max(peakRSS, currentRSSMiB())
        }

        let totalMs = elapsedMs(since: turnStart)
        let count = generatedTokens > 0 ? generatedTokens : events
        return TurnResult(
            processorPrepareMs: prepareMs,
            iteratorInitMs: 0,
            streamCreateMs: streamCreateMs,
            firstDeltaMs: firstDeltaMs ?? totalMs,
            totalMs: totalMs,
            tokens: count,
            events: events,
            e2eTokensPerSecond: tokensPerSecond(count: count, totalMs: totalMs),
            rssMiB: currentRSSMiB(),
            peakRssMiB: peakRSS,
            promptDiagnostics: promptDiagnostics,
            text: text.replacingOccurrences(of: "\n", with: " "))
    }

    private static func promptDiagnostics(for input: LMInput) -> PromptDiagnostics {
        let tokens = input.text.tokens.reshaped(-1).asArray(Int.self)
        let knownIDs = input.mediaTokenIds != nil
        let mediaTokenIds = input.mediaTokenIds ?? []
        let mediaTokenSet = Set(mediaTokenIds)
        let mediaPositions = tokens.enumerated().compactMap { index, token in
            mediaTokenSet.contains(token) ? index : nil
        }
        let blockSize = 64
        let suffixMediaTokens: Int
        if input.hasMediaContent && !knownIDs {
            suffixMediaTokens = -1
        } else {
            suffixMediaTokens = tokens.dropFirst(blockSize).filter {
                mediaTokenSet.contains($0)
            }.count
        }
        let lastMedia = mediaPositions.last ?? -1
        let promptMinusOneAfterMedia = lastMedia >= 0 && (tokens.count - 1) > lastMedia
        return PromptDiagnostics(
            promptTokens: tokens.count,
            mediaTokenIdsKnown: knownIDs,
            mediaTokenIds: mediaTokenIds,
            mediaPlaceholderTokens: mediaPositions.count,
            firstMediaTokenIndex: mediaPositions.first ?? -1,
            lastMediaTokenIndex: lastMedia,
            blockSize: blockSize,
            blockSuffixMediaTokens: suffixMediaTokens,
            promptMinusOneAfterMedia: promptMinusOneAfterMedia)
    }

    private static func makeCoordinator(
        modelName: String,
        path: String,
        mode: String,
        enableDiskCache: Bool,
        cacheDir: URL
    ) -> CacheCoordinator {
        CacheCoordinator(config: CacheCoordinatorConfig(
            usePagedCache: true,
            enableDiskCache: enableDiskCache,
            pagedBlockSize: 64,
            maxCacheBlocks: 512,
            diskCacheMaxGB: 4,
            diskCacheDir: cacheDir,
            ssmMaxEntries: 32,
            enableSSMReDerive: true,
            modelKey: "\(modelName)|omni-audio-latency|\(path)|\(mode)"))
    }

    private static func makeInput(prompt: String, audio: UserInput.Audio) -> UserInput {
        var userInput = UserInput(prompt: prompt, audios: [audio])
        userInput.additionalContext = ["enable_thinking": false]
        return userInput
    }

    private static func generationParameters(
        context: ModelContext,
        maxNewTokens: Int
    ) -> GenerateParameters {
        var params = GenerateParameters(
            generationConfig: context.configuration.generationDefaults)
        params.maxTokens = maxNewTokens
        params.prefillStepSize = 512
        return params
    }

    private static func audioPaths(from setting: String) throws -> [String] {
        switch setting {
        case "both": return ["batch", "iterator"]
        case "batch", "iterator": return [setting]
        default: throw benchError("BENCH_OMNI_AUDIO_PATH must be batch, iterator, or both")
        }
    }

    private static func stopTokenIDs(context: ModelContext) -> Set<Int> {
        var stops = context.configuration.eosTokenIds
        if let eos = context.tokenizer.eosTokenId { stops.insert(eos) }
        if let unknown = context.tokenizer.unknownTokenId { stops.insert(unknown) }
        for token in context.configuration.extraEOSTokens + commonEndTurnTokens {
            if let id = context.tokenizer.convertTokenToId(token) {
                stops.insert(id)
            }
        }
        return stops
    }

    private static let commonEndTurnTokens = [
        "<|im_end|>",
        "<|endoftext|>",
        "<|eot_id|>",
        "<|end_of_text|>",
        "<|end|>",
        "<|end_of_turn|>",
        "<end_of_turn>",
    ]

    private static func userVisibleText(
        context: ModelContext,
        lmInput: LMInput,
        tokenIds: [Int],
        allowReasoningFallback: Bool
    ) -> String {
        let raw = context.tokenizer.decode(tokenIds: tokenIds, skipSpecialTokens: false)
        let promptTokens = lmInput.text.tokens.reshaped(-1).asArray(Int.self)
        let promptTail = context.tokenizer.decode(
            tokenIds: Array(promptTokens.suffix(256)),
            skipSpecialTokens: false)

        guard var parser = ReasoningParser.forPrompt(
            stampName: context.configuration.reasoningParserName,
            promptTail: promptTail)
        else {
            return raw.trimmingCharacters(in: .whitespacesAndNewlines)
        }

        var visible = ""
        var reasoning = ""
        var segments = parser.feed(raw)
        segments.append(contentsOf: parser.flush())
        for segment in segments {
            switch segment {
            case .content(let text): visible += text
            case .reasoning(let text): reasoning += text
            }
        }

        let trimmedVisible = visible.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmedVisible.isEmpty {
            return trimmedVisible
        }
        guard allowReasoningFallback else { return "" }
        return reasoning.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func printJSON(_ fields: [String: Any]) {
        do {
            let data = try JSONSerialization.data(
                withJSONObject: fields, options: [.sortedKeys])
            let json = String(data: data, encoding: .utf8) ?? "{}"
            print("OMNI_AUDIO_LATENCY \(json)")
        } catch {
            print("OMNI_AUDIO_LATENCY {\"event\":\"encode_error\",\"error\":\"\(error)\"}")
        }
    }

    private static func elapsedMs(since start: CFAbsoluteTime) -> Double {
        (CFAbsoluteTimeGetCurrent() - start) * 1000.0
    }

    private static func tokensPerSecond(count: Int, totalMs: Double) -> Double {
        guard totalMs > 0, count > 0 else { return 0 }
        return Double(count) / (totalMs / 1000.0)
    }

    private static func rounded(_ value: Double) -> Double {
        rounded(value, places: 1)
    }

    private static func rounded(_ value: Double, places: Int) -> Double {
        guard value.isFinite else { return value }
        let scale = pow(10.0, Double(max(0, places)))
        return (value * scale).rounded() / scale
    }

    private static func currentRSSMiB() -> Double {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(
            MemoryLayout<mach_task_basic_info>.stride / MemoryLayout<natural_t>.stride)
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(
                    mach_task_self_,
                    task_flavor_t(MACH_TASK_BASIC_INFO),
                    $0,
                    &count)
            }
        }
        guard kr == KERN_SUCCESS else { return -1 }
        return Double(info.resident_size) / (1024.0 * 1024.0)
    }

    private static func benchError(_ message: String) -> NSError {
        NSError(
            domain: "OmniAudioLatencyBench",
            code: 1,
            userInfo: [NSLocalizedDescriptionKey: message])
    }
}
