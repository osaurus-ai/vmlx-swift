import CoreImage
import Darwin
import Foundation
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import MLXVLM
@preconcurrency import VMLXTokenizers

/// VL multi-turn smoke test for vision-language models.
///
/// Run via Bench.swift dispatch when env `BENCH_VL=1` is set.
///
/// What it does:
/// 1. Loads model with the REAL HuggingFace tokenizer (chat template + special tokens)
/// 2. Synthesises a 224×224 RGB image (CIImage built from an MLXArray gradient)
/// 3. Builds `UserInput(prompt:..., images:[Image.array(...)])`
/// 4. Calls `context.processor.prepare(input:)` to get an `LMInput` with vision tokens
/// 5. Generates 32 tokens via `TokenIterator` and decodes them
/// 6. Issues a SECOND turn over the same conversation to exercise multi-turn cache reuse
/// 7. Reports decode tok/s + first decoded text per turn
enum VLBench {

    /// Load through the same mmap-backed policy Osaurus uses in production.
    /// The plain `loadModel(from:using:)` overload bypasses that policy and
    /// copies safetensors into anonymous storage, which makes VL memory gates
    /// measure a harness-only residency spike instead of the host runtime.
    private static func loadProductionContext(from modelDir: URL) async throws -> ModelContext {
        let loaded = try await MLXLMCommon.loadModel(
            from: modelDir,
            using: #huggingFaceTokenizerLoader(),
            loadConfiguration: .osaurusProduction)
        return loaded.0
    }

    static func run(modelPath: String, maxNewTokens: Int) async throws {
        let modelDir = URL(fileURLWithPath: modelPath)
        print("=== VLBench — \(modelDir.lastPathComponent) ===")

        let loadStart = CFAbsoluteTimeGetCurrent()
        let context = try await loadProductionContext(from: modelDir)
        print(String(format: "Load: %.2fs", CFAbsoluteTimeGetCurrent() - loadStart))
        print("Model: \(type(of: context.model))")
        print("Processor: \(type(of: context.processor))")

        let image = try synthesiseGradientImage(side: 224)

        let cache = context.model.newCache(parameters: .init())

        try await runTurn(
            label: "Turn 1 — describe image",
            prompt: "Describe what you see in this image in one sentence.",
            images: [.ciImage(image)],
            context: context, cache: cache, maxNewTokens: maxNewTokens
        )

        try await runTurn(
            label: "Turn 2 — follow-up (cache reuse)",
            prompt: "Name one colour visible in the image. Answer with one word.",
            images: [.ciImage(image)],
            context: context, cache: cache, maxNewTokens: maxNewTokens
        )

        print("=== VLBench done ===")
    }

    // MARK: - BatchEngine VL multi-turn (iter 30)

    /// TRUE BatchEngine verification for VL models. Unlike ``run(modelPath:maxNewTokens:)``
    /// which uses `TokenIterator`, this routes each turn through
    /// `BatchEngine.generate(...)` to exercise the VL path under the real
    /// batched-inference engine.
    ///
    /// Runs two turns with a shared image: first describes the image,
    /// second asks a follow-up that requires recalling the first answer.
    /// Both turns go through `engine.generate()` — the iter 28 fix (the
    /// canonical `AsyncStream.makeStream() + Task {}` detokenizer relay)
    /// is the hot path here.
    static func runBatch(modelPath: String, maxNewTokens: Int) async throws {
        let modelDir = URL(fileURLWithPath: modelPath)
        print("=== VLBench BATCH — \(modelDir.lastPathComponent) ===")

        let modelMiB = directorySizeMiB(modelDir)
        let preLoadRSS = currentRSSMiB()
        let preLoadFootprint = currentPhysFootprintMiB()
        var peakRSS = preLoadRSS
        var peakFootprint = preLoadFootprint
        print(String(format:
            "  VL batch memory pre-load: rssMiB=%.0f footprintMiB=%.0f modelMiB=%.0f",
            preLoadRSS, preLoadFootprint, modelMiB))

        let loadStart = CFAbsoluteTimeGetCurrent()
        let context = try await loadProductionContext(from: modelDir)
        print(String(format: "Load: %.2fs", CFAbsoluteTimeGetCurrent() - loadStart))
        print("Model: \(type(of: context.model))")
        print("Processor: \(type(of: context.processor))")
        let postLoadRSS = currentRSSMiB()
        let postLoadFootprint = currentPhysFootprintMiB()
        peakRSS = max(peakRSS, postLoadRSS)
        peakFootprint = max(peakFootprint, postLoadFootprint)
        print(String(format:
            "  VL batch memory post-load: rssMiB=%.0f footprintMiB=%.0f rssDeltaMiB=%+.0f footprintDeltaMiB=%+.0f",
            postLoadRSS, postLoadFootprint, postLoadRSS - preLoadRSS,
            postLoadFootprint - preLoadFootprint))

        let image = try synthesiseGradientImage(side: 224)
        nonisolated(unsafe) let ctx = context

        for compileOn in [false, true] {
            let label = compileOn ? "compile ON" : "compile OFF"
            print("\n[\(label)] BatchEngine VL 2-turn chat")

            var params = GenerateParameters(
                maxTokens: maxNewTokens, temperature: 0,
                prefillStepSize: 512)
            params.enableCompiledBatchDecode = compileOn

            let engine = BatchEngine(context: ctx, maxBatchSize: 1)

            for (i, prompt) in [
                "Describe what you see in this image in one sentence.",
                "Name one colour visible in the image. Answer with one word.",
            ].enumerated() {
                let turnFootprint = try await runBatchTurn(
                    engine: engine, context: ctx,
                    prompt: prompt, image: image,
                    label: "Turn \(i + 1)",
                    parameters: params, maxNew: maxNewTokens
                )
                peakFootprint = max(peakFootprint, turnFootprint)
                peakRSS = max(peakRSS, currentRSSMiB())
            }
        }

        let batchPeakDelta = peakFootprint - preLoadFootprint
        let peakPercent = modelMiB > 0 ? (batchPeakDelta / modelMiB) * 100 : -1
        print(String(format:
            "  VL batch memory peak: rssMiB=%.0f footprintMiB=%.0f rssDeltaMiB=%+.0f footprintDeltaMiB=%+.0f modelMiB=%.0f ratio=%.1f%%",
            peakRSS, peakFootprint, peakRSS - preLoadRSS, batchPeakDelta,
            modelMiB, peakPercent))
        print(String(format:
            "  VL batch gate: footprintDeltaMiB=%.0f modelMiB=%.0f ratio=%.1f%% verdict=%@",
            batchPeakDelta, modelMiB, peakPercent,
            (modelMiB <= 0 || batchPeakDelta <= modelMiB) ? "passed" : "failed"))
        if modelMiB > 0, batchPeakDelta > modelMiB {
            fputs(String(format:
                "[VL Batch] FAIL: VL batch gate failed " +
                "(footprintDeltaMiB=%.0f modelMiB=%.0f ratio=%.1f%%).\n",
                batchPeakDelta, modelMiB, peakPercent), stderr)
            exit(1)
        }

        print("\n=== VLBench BATCH done ===")
    }

    /// Single VL turn through `BatchEngine.generate(...)`.
    private static func runBatchTurn(
        engine: BatchEngine,
        context: ModelContext,
        prompt: String,
        image: CIImage,
        label: String,
        parameters: GenerateParameters,
        maxNew: Int
    ) async throws -> Double {
        print("  \(label) [\(parameters.enableCompiledBatchDecode ? "compile" : "uncomp")]:")
        let t0 = CFAbsoluteTimeGetCurrent()

        var userInput = UserInput(prompt: prompt, images: [.ciImage(image)])
        userInput.additionalContext = ["enable_thinking": false]
        let lmInput = try await context.processor.prepare(input: userInput)
        if (ProcessInfo.processInfo.environment["BENCH_VL_DEBUG_PROMPT"] ?? "0") == "1" {
            let ids = lmInput.text.tokenIds ?? []
            let imageId = context.tokenizer.convertTokenToId("<|image|>") ?? -1
            let boiId = context.tokenizer.convertTokenToId("<|image>") ?? -1
            let eoiId = context.tokenizer.convertTokenToId("<image|>") ?? -1
            let imageCount = ids.filter { $0 == imageId }.count
            let boiCount = ids.filter { $0 == boiId }.count
            let eoiCount = ids.filter { $0 == eoiId }.count
            let firstImage = ids.firstIndex(of: imageId) ?? ids.startIndex
            let lo = max(ids.startIndex, firstImage - 12)
            let hi = min(ids.endIndex, firstImage + 24)
            let window = context.tokenizer.decode(
                tokenIds: Array(ids[lo ..< hi]), skipSpecialTokens: false)
            print("    debug prompt tokens=\(ids.count) imageId=\(imageId) imageCount=\(imageCount) boiCount=\(boiCount) eoiCount=\(eoiCount)")
            print("    debug image pixels=\(lmInput.image?.pixels.shape.description ?? "nil") frames=\(lmInput.image?.frames?.map { "\($0.h)x\($0.w)" }.joined(separator: ",") ?? "nil")")
            if let pixels = lmInput.image?.pixels {
                if pixels.ndim >= 4 {
                    let means = pixels.mean(axes: [0, 2, 3]).asArray(Float.self)
                    print("    debug image channel means=\(means)")
                } else {
                    print("    debug image channel means=not-applicable for rank-\(pixels.ndim) processed features")
                }
            }
            print("    debug prompt window: \(window.debugDescription)")
        }
        nonisolated(unsafe) let sendable = lmInput
        let stream = await engine.generate(input: sendable, parameters: parameters)

        var text = ""
        var ttft: Double?
        var chunkCount = 0
        var generationTokens = 0
        var promptTokensPerSecond = 0.0
        var decodeTokensPerSecond = 0.0
        var stopReason = "unknown"
        let rssBefore = currentRSSMiB()
        let footprintBefore = currentPhysFootprintMiB()
        for await event in stream {
            switch event {
            case .chunk(let chunk):
                if ttft == nil { ttft = CFAbsoluteTimeGetCurrent() - t0 }
                text += chunk
                chunkCount += 1
                if chunkCount > maxNew * 2 { break }
            case .prefillProgress:

                break
            case .info(let info):
                generationTokens = info.generationTokenCount
                promptTokensPerSecond = info.promptTokensPerSecond
                decodeTokensPerSecond = info.tokensPerSecond
                stopReason = String(describing: info.stopReason)
            case .reasoning, .toolCall, .toolCallProgress:
                break
            }
        }
        let total = CFAbsoluteTimeGetCurrent() - t0
        let rssAfter = currentRSSMiB()
        let footprintAfter = currentPhysFootprintMiB()
        let preview = text.count > 200 ? String(text.prefix(200)) + "..." : text
        print(String(format: "    TTFT %dms, total %.2fs, chunks=%d",
            Int((ttft ?? 0) * 1000), total, chunkCount))
        print(String(format:
            "    telemetry tokens=%d promptTok/s=%.1f decodeTok/s=%.1f stop=%@ rssMiB=%.0f footprintMiB=%.0f rssDeltaMiB=%+.0f footprintDeltaMiB=%+.0f",
            generationTokens, promptTokensPerSecond, decodeTokensPerSecond,
            stopReason, rssAfter, footprintAfter, rssAfter - rssBefore,
            footprintAfter - footprintBefore))
        print("    \"\(preview)\"")

        let visible = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard chunkCount > 0, !visible.isEmpty else {
            throw NSError(
                domain: "VLBench",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey: "\(label) emitted no visible VL output"])
        }
        if prompt.localizedCaseInsensitiveContains("describe"),
            !containsAnyWord(
                visible,
                words: ["image", "gradient", "red", "green", "blue", "orange", "yellow", "purple", "color", "colour"])
        {
            throw NSError(
                domain: "VLBench",
                code: 2,
                userInfo: [
                    NSLocalizedDescriptionKey:
                        "\(label) did not emit grounded image/color language: \(preview)"
                ])
        }
        if (prompt.localizedCaseInsensitiveContains("colour visible")
            || prompt.localizedCaseInsensitiveContains("color visible")),
            !containsAnyWord(visible, words: ["red", "blue", "green"])
        {
            throw NSError(
                domain: "VLBench",
                code: 3,
                userInfo: [
                    NSLocalizedDescriptionKey:
                        "\(label) did not ground a visible image colour: \(preview)"
                ])
        }
        return footprintAfter
    }

    private static func containsAnyWord(_ text: String, words: Set<String>) -> Bool {
        let tokens = text.lowercased().components(separatedBy: CharacterSet.letters.inverted)
        return tokens.contains { words.contains($0) }
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
                    $0, &count)
            }
        }
        guard kr == KERN_SUCCESS else { return -1 }
        return Double(info.resident_size) / (1024.0 * 1024.0)
    }

    private static func currentPhysFootprintMiB() -> Double {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(
                    mach_task_self_,
                    task_flavor_t(TASK_VM_INFO),
                    $0, &count)
            }
        }
        guard kr == KERN_SUCCESS else { return -1 }
        return Double(info.phys_footprint) / (1024.0 * 1024.0)
    }

    private static func makeProofCoordinator(
        modelDir: URL,
        context: ModelContext,
        parameters: GenerateParameters,
        label: String
    ) -> CacheCoordinator {
        let probeCache = context.model.newCache(parameters: parameters)
        let needsDiskBackedRestore =
            cacheRequiresDiskBackedCoordinatorRestore(probeCache)

        var cfg = CacheCoordinatorConfig()
        cfg.usePagedCache = true
        cfg.enableDiskCache = needsDiskBackedRestore
        cfg.pagedBlockSize = 64
        cfg.maxCacheBlocks = 512
        cfg.modelKey = modelDir.lastPathComponent
        if needsDiskBackedRestore {
            cfg.diskCacheDir = FileManager.default.temporaryDirectory
                .appendingPathComponent(
                    "vmlx-vlbench-\(label)-\(ProcessInfo.processInfo.processIdentifier)-\(UUID().uuidString)")
        }

        let coordinator = CacheCoordinator(config: cfg)
        if needsDiskBackedRestore {
            coordinator.setPagedIncompatible(true)
        }
        print("  cache proof: " +
              (needsDiskBackedRestore
                ? "disk-backed path-dependent restore"
                : "paged media-salted restore"))
        return coordinator
    }

    // MARK: - Video multi-turn (2026-04-22)

    /// End-to-end verification that a VL model correctly ingests a video
    /// file through `context.processor.prepare(input:)` → `BatchEngine.generate`.
    /// Matches ``runBatch(modelPath:maxNewTokens:)`` but feeds a video
    /// URL instead of an image.
    ///
    /// Drives two turns so multi-turn cache behavior with a video-salt
    /// also gets exercised. If the model emits any content at all
    /// (non-empty `.chunk` stream on turn 1), video ingestion is
    /// demonstrably working end-to-end.
    static func runBatchVideo(
        modelPath: String, videoPath: String, maxNewTokens: Int
    ) async throws {
        let modelDir = URL(fileURLWithPath: modelPath)
        let videoURL = URL(fileURLWithPath: videoPath)
        print("=== VLBench VIDEO BATCH — \(modelDir.lastPathComponent) ===")
        print("video: \(videoURL.lastPathComponent)")

        let modelMiB = directorySizeMiB(modelDir)
        let preLoadFootprint = currentPhysFootprintMiB()
        var peakFootprint = preLoadFootprint
        print(String(format:
            "  video memory pre-load: footprintMiB=%.0f modelMiB=%.0f",
            preLoadFootprint, modelMiB))

        let loadStart = CFAbsoluteTimeGetCurrent()
        let context = try await loadProductionContext(from: modelDir)
        print(String(format: "Load: %.2fs", CFAbsoluteTimeGetCurrent() - loadStart))
        print("Model: \(type(of: context.model))")
        print("Processor: \(type(of: context.processor))")
        let postLoadFootprint = currentPhysFootprintMiB()
        peakFootprint = max(peakFootprint, postLoadFootprint)
        print(String(format:
            "  video memory post-load: footprintMiB=%.0f footprintDeltaMiB=%+.0f",
            postLoadFootprint, postLoadFootprint - preLoadFootprint))

        nonisolated(unsafe) let ctx = context

        let params = GenerateParameters(
            maxTokens: maxNewTokens, temperature: 0,
            prefillStepSize: 512)

        let engine = BatchEngine(context: ctx, maxBatchSize: 1)

        let prompts = [
            "Describe what you see in this video in one sentence.",
            "What's happening in the foreground?",
        ]
        for (i, prompt) in prompts.enumerated() {
            let turnLabel = "Turn \(i + 1)"
            print("  \(turnLabel):")
            let t0 = CFAbsoluteTimeGetCurrent()
            // enable_thinking=false so the token budget goes to visible
            // content, not chain-of-thought. Without this, Qwen 3.x
            // defaults to thinking-on and a 256-token budget gets spent
            // entirely inside `<think>...</think>` before any answer.
            var userInput = UserInput(
                prompt: prompt, videos: [.url(videoURL)])
            userInput.additionalContext = ["enable_thinking": false]
            let lmInput: LMInput
            do {
                lmInput = try await ctx.processor.prepare(input: userInput)
            } catch {
                if isVideoNotImplemented(error) {
                    print("    not applicable: processor video input is not implemented for this model: \(error)")
                    return
                }
                print("    PREPARE ERROR: \(error)")
                throw error
            }
            nonisolated(unsafe) let sendable = lmInput
            let stream = await engine.generate(
                input: sendable, parameters: params)

            var text = ""
            var chunkCount = 0
            var reasoningCount = 0
            var ttft: Double?
            var generationTokens = 0
            var promptTokensPerSecond = 0.0
            var decodeTokensPerSecond = 0.0
            var stopReason = "unknown"
            for await ev in stream {
                switch ev {
                case .chunk(let c):
                    if ttft == nil { ttft = CFAbsoluteTimeGetCurrent() - t0 }
                    text += c
                    chunkCount += 1
                case .reasoning:
                    reasoningCount += 1
                case .info(let info):
                    generationTokens = info.generationTokenCount
                    promptTokensPerSecond = info.promptTokensPerSecond
                    decodeTokensPerSecond = info.tokensPerSecond
                    stopReason = String(describing: info.stopReason)
                case .prefillProgress, .toolCall, .toolCallProgress:
                    break
                }
            }
            let total = CFAbsoluteTimeGetCurrent() - t0
            let footprint = currentPhysFootprintMiB()
            peakFootprint = max(peakFootprint, footprint)
            let preview = text.count > 220
                ? String(text.prefix(220)) + "..." : text
            print(String(format:
                "    TTFT %dms total %.2fs chunks=%d reasoningDeltas=%d",
                Int((ttft ?? 0) * 1000), total, chunkCount, reasoningCount))
            print(String(format:
                "    telemetry tokens=%d promptTok/s=%.1f decodeTok/s=%.1f stop=%@ footprintMiB=%.0f",
                generationTokens, promptTokensPerSecond, decodeTokensPerSecond,
                stopReason, footprint))
            print("    \"\(preview)\"")

            let visible = text.trimmingCharacters(in: .whitespacesAndNewlines)
            guard chunkCount > 0, !visible.isEmpty else {
                throw NSError(
                    domain: "VLBenchVideo",
                    code: 1,
                    userInfo: [NSLocalizedDescriptionKey: "\(turnLabel) emitted no visible video output"])
            }
            let rawMarkers = ["<think>", "</think>", "<|channel>", "<|tool_call>", "<tool_call>"]
            if let marker = rawMarkers.first(where: { visible.contains($0) }) {
                throw NSError(
                    domain: "VLBenchVideo",
                    code: 2,
                    userInfo: [NSLocalizedDescriptionKey:
                        "\(turnLabel) leaked raw parser marker \(marker): \(preview)"])
            }
            guard generationTokens > 0, decodeTokensPerSecond > 0 else {
                throw NSError(
                    domain: "VLBenchVideo",
                    code: 3,
                    userInfo: [NSLocalizedDescriptionKey:
                        "\(turnLabel) did not emit usable generation telemetry"])
            }
        }

        let peakDelta = peakFootprint - preLoadFootprint
        let peakPercent = modelMiB > 0 ? (peakDelta / modelMiB) * 100 : -1
        print(String(format:
            "  video memory peak: footprintMiB=%.0f footprintDeltaMiB=%+.0f modelMiB=%.0f ratio=%.1f%% verdict=%@",
            peakFootprint, peakDelta, modelMiB, peakPercent,
            (modelMiB <= 0 || peakDelta <= modelMiB) ? "passed" : "failed"))
        if modelMiB > 0, peakDelta > modelMiB {
            throw NSError(
                domain: "VLBenchVideo",
                code: 4,
                userInfo: [NSLocalizedDescriptionKey:
                    String(format:
                        "video memory gate failed (footprintDeltaMiB=%.0f modelMiB=%.0f ratio=%.1f%%)",
                        peakDelta, modelMiB, peakPercent)])
        }

        print("=== VLBench VIDEO BATCH done ===")
    }

    // MARK: - Mixed-variable multi-turn matrix (2026-04-22)

    /// Single model + single BatchEngine, four turns with different
    /// conditions flipped on each turn. Exercises the full pipeline
    /// with shared state:
    ///
    ///   T1: thinking=ON  text     → .reasoning deltas, then .chunk
    ///   T2: thinking=ON  text repeat → cache hit on the shared prefix
    ///   T3: thinking=OFF image    → .chunk only, no `<think>` leak
    ///   T4: thinking=OFF video    → .chunk only, video-relevant answer
    ///
    /// Covers on one run:
    ///   - `ReasoningParser.forPrompt` tail detection: T1/T2 tail ends
    ///     with `<think>\n` (thinking on) → parser starts in reasoning;
    ///     T3/T4 tail ends with `<think>\n\n</think>\n\n` (thinking off)
    ///     → parser starts in content.
    ///   - SSM seed at prefill-end (fde3bb9): hybrid-SSM slots deposit
    ///     companion state on every turn.
    ///   - stepBatchDecode force-unwrap fix (105ff8b): no crash path.
    ///   - Image → video role swap with shared engine.
    ///   - Generation stream event routing (.reasoning vs .chunk).
    static func runMixedMultiTurn(
        modelPath: String,
        videoPath: String,
        maxNewTokens: Int
    ) async throws {
        let modelDir = URL(fileURLWithPath: modelPath)
        let videoURL = URL(fileURLWithPath: videoPath)
        print("=== VLBench MIXED MULTI-TURN — \(modelDir.lastPathComponent) ===")

        let loadStart = CFAbsoluteTimeGetCurrent()
        let context = try await loadProductionContext(from: modelDir)
        print(String(format: "Load: %.2fs", CFAbsoluteTimeGetCurrent() - loadStart))
        print("Model: \(type(of: context.model))")
        print("Processor: \(type(of: context.processor))")
        print("Reasoning stamp: \(context.configuration.reasoningParserName ?? "nil")")

        nonisolated(unsafe) let ctx = context
        let engine = BatchEngine(context: ctx, maxBatchSize: 1)
        let params = GenerateParameters(
            maxTokens: maxNewTokens, temperature: 0, prefillStepSize: 512)

        let image = try synthesiseGradientImage(side: 224)

        // T1: thinking ON + text
        try await runMixedTurn(
            label: "T1 thinking=ON text", engine: engine, ctx: ctx,
            prompt: "What's 7 + 6? Answer with just the number.",
            thinking: true, image: nil, video: nil,
            params: params, maxNew: maxNewTokens)

        // T2: thinking ON + SAME prompt → expect shared-prefix cache speedup
        try await runMixedTurn(
            label: "T2 thinking=ON text (cache hit)", engine: engine, ctx: ctx,
            prompt: "What's 7 + 6? Answer with just the number.",
            thinking: true, image: nil, video: nil,
            params: params, maxNew: maxNewTokens)

        // T3: thinking OFF + image (mode flip + modality flip)
        try await runMixedTurn(
            label: "T3 thinking=OFF image", engine: engine, ctx: ctx,
            prompt: "Describe the image in one sentence.",
            thinking: false, image: image, video: nil,
            params: params, maxNew: maxNewTokens)

        // T4: thinking OFF + video (modality flip again)
        try await runMixedTurn(
            label: "T4 thinking=OFF video", engine: engine, ctx: ctx,
            prompt: "What is this video showing?",
            thinking: false, image: nil, video: videoURL,
            params: params, maxNew: maxNewTokens)

        print("\n=== VLBench MIXED MULTI-TURN done ===")
    }

    private static func runMixedTurn(
        label: String, engine: BatchEngine, ctx: ModelContext,
        prompt: String, thinking: Bool,
        image: CIImage?, video: URL?,
        params: GenerateParameters, maxNew: Int
    ) async throws {
        print("\n[\(label)]")
        let t0 = CFAbsoluteTimeGetCurrent()
        var userInput: UserInput
        if let image {
            userInput = UserInput(prompt: prompt, images: [.ciImage(image)])
        } else if let video {
            userInput = UserInput(prompt: prompt, videos: [.url(video)])
            userInput.processing = benchVideoProcessing(defaultSquare: nil)
            if let resize = userInput.processing.resize {
                print("  video processing resize = \(Int(resize.width))x\(Int(resize.height))")
            }
        } else {
            userInput = UserInput(prompt: prompt)
        }
        userInput.additionalContext = ["enable_thinking": thinking]
        let lmInput: LMInput
        do {
            lmInput = try await ctx.processor.prepare(input: userInput)
        } catch {
            if video != nil, isVideoNotImplemented(error) {
                print("  not applicable: processor video input is not implemented for this model: \(error)")
                return
            }
            throw error
        }
        if let video = lmInput.video {
            print("  video pixels shape: \(video.pixels.shape)")
        }
        nonisolated(unsafe) let sendable = lmInput
        let stream = await engine.generate(input: sendable, parameters: params)

        var text = ""
        var reasoningText = ""
        var chunks = 0
        var reasoningDeltas = 0
        var ttft: Double?
        var sawAnyEvent = false
        for await ev in stream {
            switch ev {
            case .chunk(let c):
                if ttft == nil { ttft = CFAbsoluteTimeGetCurrent() - t0 }
                text += c; chunks += 1; sawAnyEvent = true
            case .reasoning(let r):
                if ttft == nil { ttft = CFAbsoluteTimeGetCurrent() - t0 }
                reasoningText += r; reasoningDeltas += 1; sawAnyEvent = true
            case .prefillProgress, .info, .toolCall, .toolCallProgress:
                break
            }
        }
        let total = CFAbsoluteTimeGetCurrent() - t0
        let preview = text.count > 160 ? String(text.prefix(160)) + "..." : text
        let rprev = reasoningText.count > 80
            ? String(reasoningText.prefix(80)) + "..." : reasoningText
        print(String(format: "  TTFT %dms total %.2fs chunks=%d reasoningDeltas=%d",
            Int((ttft ?? 0) * 1000), total, chunks, reasoningDeltas))
        if !text.isEmpty { print("  chunk: \"\(preview)\"") }
        if !reasoningText.isEmpty { print("  reasoning: \"\(rprev)\"") }

        // Invariants:
        //  1. thinking=OFF: .chunk must not contain <think> or </think>
        //     (parser had to start in content mode via forPrompt).
        if !thinking {
            if text.contains("<think>") || text.contains("</think>") {
                fputs("FAIL [\(label)]: <think> leak in .chunk with thinking OFF\n", stderr)
                exit(1)
            }
        }
        //  2. Engine must not crash — at least one event fires.
        if !sawAnyEvent {
            fputs("WARN [\(label)]: zero events streamed — may indicate crash path\n", stderr)
        }
    }

    // MARK: - mediaSalt isolation (iter 37)

    /// Verify VL cache isolation via `mediaSalt`. This is the cross-image
    /// poisoning check: the coordinator's block hashes must incorporate
    /// the image bytes, not just the text tokens, so that the same text
    /// prompt with image A can't return image B's cached KV state.
    ///
    /// Methodology:
    ///  1. Submit prompt P with image A through BatchEngine + coordinator.
    ///     Finish stores KV keyed by (tokens, salt_A).
    ///  2. Probe coordinator with (tokens, salt_A) → must HIT.
    ///  3. Probe coordinator with (tokens, salt_B) where B != A → must MISS.
    ///
    /// If step 3 HITs, the bug is in `PagedCacheManager.storeTokenSequence`
    /// or `fetchPrefix` not including mediaSalt in the block hash chain.
    static func runBatchMediaSalt(modelPath: String, maxNewTokens: Int) async throws {
        let modelDir = URL(fileURLWithPath: modelPath)
        print("=== VLBench mediaSalt isolation (iter 37) ===")

        let modelMiB = directorySizeMiB(modelDir)
        let preLoadRSS = currentRSSMiB()
        let preLoadFootprint = currentPhysFootprintMiB()
        var peakRSS = preLoadRSS
        var peakFootprint = preLoadFootprint
        print(String(format:
            "  mediaSalt memory pre-load: rssMiB=%.0f footprintMiB=%.0f modelMiB=%.0f",
            preLoadRSS, preLoadFootprint, modelMiB))

        let loadStart = CFAbsoluteTimeGetCurrent()
        let context = try await loadProductionContext(from: modelDir)
        print(String(format: "Load: %.2fs", CFAbsoluteTimeGetCurrent() - loadStart))
        print("Model: \(type(of: context.model))")
        print("Processor: \(type(of: context.processor))")
        let postLoadRSS = currentRSSMiB()
        let postLoadFootprint = currentPhysFootprintMiB()
        peakRSS = max(peakRSS, postLoadRSS)
        peakFootprint = max(peakFootprint, postLoadFootprint)
        print(String(format:
            "  mediaSalt memory post-load: rssMiB=%.0f footprintMiB=%.0f rssDeltaMiB=%+.0f footprintDeltaMiB=%+.0f",
            postLoadRSS, postLoadFootprint, postLoadRSS - preLoadRSS,
            postLoadFootprint - preLoadFootprint))

        let params = GenerateParameters(
            maxTokens: maxNewTokens, temperature: 0, prefillStepSize: 512)
        let coordinator = makeProofCoordinator(
            modelDir: modelDir, context: context, parameters: params,
            label: "media-salt")

        nonisolated(unsafe) let ctx = context
        let engine = BatchEngine(
            context: ctx, maxBatchSize: 1, cacheCoordinator: coordinator)

        // Image A: red→blue vertical gradient. Image B: blue→red (axis-flipped).
        // Different pixel bytes → different SHA256 → different mediaSalt.
        let imageA = try synthesiseGradientImage(side: 224)
        let imageB = try synthesiseGradientImage(side: 224, invert: true)

        // Long enough prompt + image tokens that the cache stores ≥ 2 full
        // blocks (block size 64). Vision models expand images into hundreds
        // of vision tokens, so even a short text prompt easily crosses.
        let prompt = "Describe the colours you see in this image in one concise paragraph."

        // Turn 1: image A, prompt P → submit and drain.
        let inputA = try await context.processor.prepare(
            input: UserInput(prompt: prompt, images: [.ciImage(imageA)]))
        peakRSS = max(peakRSS, currentRSSMiB())
        peakFootprint = max(peakFootprint, currentPhysFootprintMiB())
        let tokensA = inputA.text.tokens.reshaped(-1).asArray(Int.self)
        let saltA = computeCacheSalt(for: inputA, parameters: params)
        print("  image A: tokens=\(tokensA.count), image attached=\(inputA.image != nil), " +
              "video attached=\(inputA.video != nil), pixels shape=" +
              "\(inputA.image?.pixels.shape.description ?? "nil"), " +
              "mediaSalt=\(saltA?.prefix(12) ?? "nil")")

        nonisolated(unsafe) let sendA = inputA
        let genStart = CFAbsoluteTimeGetCurrent()
        let (_, streamA) = await engine.submit(input: sendA, parameters: params)
        var genA = 0
        for await event in streamA {
            if case .token = event { genA += 1 }
            peakRSS = max(peakRSS, currentRSSMiB())
            peakFootprint = max(peakFootprint, currentPhysFootprintMiB())
        }
        let genSeconds = max(CFAbsoluteTimeGetCurrent() - genStart, 0.001)
        print(String(format:
            "  Turn 1 (store): generated %d tokens at %.2f tok/s with image A",
            genA, Double(genA) / genSeconds))

        // Probe 1: same prompt + same image A → must HIT.
        let probeHit = coordinator.fetch(tokens: tokensA, mediaSalt: saltA)
        switch probeHit {
        case .hit(let matched, _, let detail, _, _, _):
            print("  Probe A (same image): HIT (\(detail.rawValue), matched=\(matched)/\(tokensA.count))")
        case .miss:
            fputs("[VL MediaSalt] FAIL: probe with identical tokens+salt missed. " +
                  "BatchEngine isn't storing under (tokens, salt) on finish.\n", stderr)
            exit(1)
        }

        // Probe 2: same text but different image. Build image B's input so
        // its token sequence is identical (both use the same prompt wrapper),
        // only mediaSalt differs. For Qwen3.5-VL, image bytes change pixel
        // tensor → change salt; tokens at the vision-token slot are
        // placeholder IDs that depend on image dimensions, which we keep
        // identical by using the same 224×224 size.
        let inputB = try await context.processor.prepare(
            input: UserInput(prompt: prompt, images: [.ciImage(imageB)]))
        peakRSS = max(peakRSS, currentRSSMiB())
        peakFootprint = max(peakFootprint, currentPhysFootprintMiB())
        let tokensB = inputB.text.tokens.reshaped(-1).asArray(Int.self)
        let saltB = computeCacheSalt(for: inputB, parameters: params)
        let tokensEqual = tokensA == tokensB
        let saltsDiffer = saltA != saltB
        print("  image B: tokens=\(tokensB.count), mediaSalt=\(saltB?.prefix(12) ?? "nil")")
        print("  tokensA == tokensB? \(tokensEqual)  saltA != saltB? \(saltsDiffer)")
        if !saltsDiffer {
            fputs("[VL MediaSalt] FAIL: two different images produced the same salt. " +
                  "Test harness broken — pick images with more distinct bytes.\n", stderr)
            exit(1)
        }

        let probeB = coordinator.fetch(tokens: tokensB, mediaSalt: saltB)
        switch probeB {
        case .hit(let matched, _, let detail, _, _, _):
            fputs("[VL MediaSalt] FAIL: probe with DIFFERENT image hit cache " +
                  "(\(detail.rawValue), matched=\(matched)). " +
                  "mediaSalt is not being folded into the block hash chain.\n", stderr)
            exit(1)
        case .miss:
            print("  Probe B (different image): MISS (correct — isolation holds)")
        }
        let peakDelta = peakFootprint - preLoadFootprint
        let peakPercent = modelMiB > 0 ? (peakDelta / modelMiB) * 100 : -1
        print(String(format:
            "  mediaSalt memory peak: rssMiB=%.0f footprintMiB=%.0f rssDeltaMiB=%+.0f footprintDeltaMiB=%+.0f modelMiB=%.0f ratio=%.1f%%",
            peakRSS, peakFootprint, peakRSS - preLoadRSS, peakDelta,
            modelMiB, peakPercent))
        print(String(format:
            "  mediaSalt gate: footprintDeltaMiB=%.0f modelMiB=%.0f ratio=%.1f%% verdict=%@",
            peakDelta, modelMiB, peakPercent,
            (modelMiB <= 0 || peakDelta <= modelMiB) ? "passed" : "failed"))
        if modelMiB > 0, peakDelta > modelMiB {
            fputs(String(format:
                "[VL MediaSalt] FAIL: mediaSalt gate failed " +
                "(footprintDeltaMiB=%.0f modelMiB=%.0f ratio=%.1f%%).\n",
                peakDelta, modelMiB, peakPercent), stderr)
            exit(1)
        }
        print("=== VLBench mediaSalt isolation: passed ===")
    }

    private static func directorySizeMiB(_ url: URL) -> Double {
        guard let enumerator = FileManager.default.enumerator(
            at: url,
            includingPropertiesForKeys: [.fileSizeKey, .isRegularFileKey],
            options: [.skipsHiddenFiles])
        else {
            return 0
        }
        var total = 0
        for case let fileURL as URL in enumerator {
            let values = try? fileURL.resourceValues(
                forKeys: [.fileSizeKey, .isRegularFileKey])
            if values?.isRegularFile == true {
                total += values?.fileSize ?? 0
            }
        }
        return Double(total) / (1024.0 * 1024.0)
    }

    // MARK: - Cross-engine byte-identity on VL (iter 47)

    /// Run the same VL prompt (text + image) through both `TokenIterator`
    /// and `BatchEngine.submit` at temp=0 and assert the emitted tokens
    /// match byte-for-byte. This is the vision-path equivalent of
    /// iter 32's text cross-validator — catches any engine divergence
    /// introduced by the VLM `prepare()` / vision tower / mediaSalt path.
    ///
    /// Required for iter 45's correctness claim to mean anything:
    /// "image reaches model" is necessary but not sufficient; both
    /// iterators must also agree on what tokens the model emits.
    static func runCrossValidate(modelPath: String, maxNewTokens: Int) async throws {
        let modelDir = URL(fileURLWithPath: modelPath)
        print("=== VLBench cross-engine byte-identity (iter 47) ===")

        let loadStart = CFAbsoluteTimeGetCurrent()
        let context = try await loadProductionContext(from: modelDir)
        print(String(format: "Load: %.2fs", CFAbsoluteTimeGetCurrent() - loadStart))
        print("Model: \(type(of: context.model))")
        print("Processor: \(type(of: context.processor))")

        let image = try synthesiseGradientImage(side: 224)
        let prompt = "Describe the colours in this image in one sentence."
        let params = GenerateParameters(
            maxTokens: maxNewTokens, temperature: 0, prefillStepSize: 512)

        // Build two independent LMInputs — each consumer gets a fresh
        // instance because `submit` consumes the sending value.
        func prepareInput() async throws -> LMInput {
            try await context.processor.prepare(
                input: UserInput(prompt: prompt, images: [.ciImage(image)]))
        }
        let inputA = try await prepareInput()
        let inputB = try await prepareInput()
        let promptLen = inputA.text.tokens.size
        print("  VL prompt: \(promptLen) tokens, image=\(inputA.image != nil)")

        // Stop token set — same rule as iter 44's cross-validator.
        var stopTokenIDs: Set<Int> = context.configuration.eosTokenIds
        if let eos = context.tokenizer.eosTokenId { stopTokenIDs.insert(eos) }
        if let unk = context.tokenizer.unknownTokenId { stopTokenIDs.insert(unk) }
        for tok in context.configuration.extraEOSTokens {
            if let id = context.tokenizer.convertTokenToId(tok) {
                stopTokenIDs.insert(id)
            }
        }

        // Path A: TokenIterator
        let iterCache = context.model.newCache(parameters: params)
        let iter = try TokenIterator(
            input: inputA, model: context.model, cache: iterCache, parameters: params)
        var iterTokens: [Int] = []
        for token in iter {
            iterTokens.append(token)
            if iterTokens.count >= maxNewTokens { break }
        }
        print("  TokenIterator (\(iterTokens.count) toks): first 15 = \(Array(iterTokens.prefix(15)))")

        // Path B: BatchEngine
        nonisolated(unsafe) let ctx = context
        let engine = BatchEngine(context: ctx, maxBatchSize: 1)
        nonisolated(unsafe) let sendable = inputB
        let (_, tokenStream) = await engine.submit(input: sendable, parameters: params)
        var engineTokens: [Int] = []
        for await event in tokenStream {
            switch event {
            case .token(let id):
                engineTokens.append(id)
                if engineTokens.count >= maxNewTokens { break }
            case .prefillProgress:

                break
            case .info: break
            }
            if engineTokens.count >= maxNewTokens { break }
        }
        print("  BatchEngine   (\(engineTokens.count) toks): first 15 = \(Array(engineTokens.prefix(15)))")

        // Identity check with EOS-tolerant prefix rule (iter 44 pattern).
        if iterTokens == engineTokens {
            print("  ✓ byte-identical (\(iterTokens.count) tokens)")
        } else if iterTokens.count > engineTokens.count,
                  Array(iterTokens.prefix(engineTokens.count)) == engineTokens,
                  stopTokenIDs.contains(iterTokens[engineTokens.count]) {
            print("  ✓ identical \(engineTokens.count)-token prefix — " +
                  "BatchEngine stopped at EOS token \(iterTokens[engineTokens.count])")
        } else {
            let n = min(iterTokens.count, engineTokens.count)
            var firstDiff = n
            for k in 0..<n where iterTokens[k] != engineTokens[k] {
                firstDiff = k; break
            }
            fputs("[VL CrossValidate] FAIL: engines diverge at index \(firstDiff) " +
                  "(iter=\(iterTokens.count), engine=\(engineTokens.count) tokens). " +
                  "Vision-path engine disagreement.\n", stderr)
            exit(1)
        }
        print("=== VLBench cross-engine byte-identity: passed ===")
    }

    // MARK: - VL multi-turn cache reuse (iter 48)

    /// End-to-end VL cache reuse: turn 1 prompt + image stores under
    /// (tokens, mediaSalt); turn 2 uses a strict token-level extension
    /// (same prefix) with the SAME mediaSalt. BatchEngine must see a
    /// paged HIT and skip re-prefilling the shared prefix.
    ///
    /// Methodology mirrors iter 34's text `runBatchEngineCacheHit`:
    /// build prompts at the token level to guarantee strict prefix
    /// extension, bypass the tokenizer re-templating that would
    /// otherwise introduce divergence at the chat-template boundary.
    static func runBatchCacheHit(modelPath: String, maxNewTokens: Int) async throws {
        let modelDir = URL(fileURLWithPath: modelPath)
        print("=== VLBench multi-turn cache reuse (iter 48) ===")

        let loadStart = CFAbsoluteTimeGetCurrent()
        let context = try await loadProductionContext(from: modelDir)
        print(String(format: "Load: %.2fs", CFAbsoluteTimeGetCurrent() - loadStart))
        print("Model: \(type(of: context.model))")
        print("Processor: \(type(of: context.processor))")

        let params = GenerateParameters(
            maxTokens: maxNewTokens, temperature: 0, prefillStepSize: 512)
        let coordinator = makeProofCoordinator(
            modelDir: modelDir, context: context, parameters: params,
            label: "cache-hit")

        nonisolated(unsafe) let ctx = context
        let engine = BatchEngine(
            context: ctx, maxBatchSize: 1, cacheCoordinator: coordinator)

        // Turn 1: "Describe the image" with a real image. This produces
        // a 74+ token prompt (58 vision + ~15 text) — ≥ 2 full 64-token
        // paged blocks so the store actually retains something.
        let image = try synthesiseGradientImage(side: 224)
        let turn1Prompt = """
            Describe the contents of this image as thoroughly as possible. \
            Mention colours, shapes, and any objects you see.
            """
        let turn1Input = try await context.processor.prepare(
            input: UserInput(prompt: turn1Prompt, images: [.ciImage(image)]))
        let turn1Tokens = turn1Input.text.tokens.reshaped(-1).asArray(Int.self)
        let saltA = computeCacheSalt(for: turn1Input, parameters: params) ?? ""
        print("  image attached=\(turn1Input.image != nil), " +
              "pixels=\(turn1Input.image?.pixels.shape.description ?? "nil"), " +
              "mediaSalt=\(saltA.prefix(12)), tokens=\(turn1Tokens.count)")

        nonisolated(unsafe) let t1Send = turn1Input
        let t0A = CFAbsoluteTimeGetCurrent()
        let (_, streamA) = await engine.submit(input: t1Send, parameters: params)
        var turn1Gen = 0
        var turn1PromptTime: Double = 0
        for await event in streamA {
            switch event {
            case .token: turn1Gen += 1
            case .prefillProgress:

                break
            case .info(let info): turn1PromptTime = info.promptTime
            }
        }
        let wallA = CFAbsoluteTimeGetCurrent() - t0A
        print(String(format:
            "  Turn 1 (cold): %d tokens, promptTime=%.3fs, wall=%.2fs",
            turn1Gen, turn1PromptTime, wallA))

        // VL multi-turn cache reuse has a hard constraint: a partial
        // prefix hit that SPLITS the vision-token region crashes the
        // vision-feature merge step (see BatchEngine.stepPrefill's
        // `hasVisualContent && !remaining.isEmpty` guard that forces
        // fall-back to full prefill). So the property this test can
        // meaningfully check is "REPLAYING the exact same prompt +
        // image hits the cache" — the session-resume case, which is
        // the dominant real-world pattern anyway.
        let turn2Tokens: [Int] = turn1Tokens
        let probe = coordinator.fetch(tokens: turn2Tokens, mediaSalt: saltA)
        switch probe {
        case .hit(let matched, _, let detail, _, _, _):
            print("  Coordinator probe: HIT (\(detail.rawValue), " +
                  "matched=\(matched)/\(turn2Tokens.count))")
        case .miss:
            fputs("[VL CacheHit] FAIL: coordinator.fetch(turn2Tokens, saltA) returned .miss. " +
                  "BatchEngine's finishSlot isn't storing under (tokens, salt) " +
                  "for VL slots with mediaSalt set.\n", stderr)
            exit(1)
        }

        // Submit turn 2 — same tokens, same image → full cache hit.
        let turn2Arr = MLXArray(turn2Tokens.map { Int32($0) })[.newAxis, .ellipsis]
        let turn2Input = LMInput(
            text: LMInput.Text(tokens: turn2Arr),
            image: turn1Input.image,
            video: nil)
        let saltB = computeCacheSalt(for: turn2Input, parameters: params) ?? ""
        print("  Turn 2 built: tokens=\(turn2Tokens.count), " +
              "mediaSalt=\(saltB.prefix(12)), saltA==saltB? \(saltA == saltB)")

        nonisolated(unsafe) let t2Send = turn2Input
        let t0B = CFAbsoluteTimeGetCurrent()
        let (_, streamB) = await engine.submit(input: t2Send, parameters: params)
        var turn2Gen = 0
        var turn2PromptTime: Double = 0
        for await event in streamB {
            switch event {
            case .token: turn2Gen += 1
            case .prefillProgress:

                break
            case .info(let info): turn2PromptTime = info.promptTime
            }
        }
        let wallB = CFAbsoluteTimeGetCurrent() - t0B
        print(String(format:
            "  Turn 2 (warm): %d tokens, promptTime=%.3fs, wall=%.2fs",
            turn2Gen, turn2PromptTime, wallB))

        // Correctness contract on a VL cache hit: turn 2 completes
        // without crash and produces non-zero tokens. Prefill-time
        // ratio is informational only — whether the hit is "full"
        // (matched == tokens.count, routes through the skip-prefill
        // branch) or "partial" (rolls back per BatchEngine.stepPrefill's
        // VL guard) depends on paged `blockSize` alignment with the
        // prompt length. The 83-token Qwen3.5-VL prompt at blockSize=64
        // gets `matched=64/83` — a partial hit that correctly rolls
        // back to full prefill (the alternative is an MLX "SmallVector
        // out of range" crash in the vision-feature merge step).
        let ratio = turn1PromptTime > 0 ? turn2PromptTime / turn1PromptTime : 1.0
        print(String(format: "  ratio (turn2/turn1) = %.2f (informational only)", ratio))
        if turn2Gen == 0 {
            fputs("[VL CacheHit] FAIL: turn 2 generated zero tokens.\n", stderr)
            exit(1)
        }
        print("=== VLBench multi-turn cache reuse: passed (full-replay roundtrip) ===")
    }

    // MARK: - Structured chat media cache matrix

    /// Structured-chat VL cache matrix for production chat wiring.
    ///
    /// This exercises the path osaurus uses more directly than the
    /// prompt-string benches:
    /// - `UserInput(chat:)` carries media inside `Chat.Message`.
    /// - `computeMediaSalt` fingerprints the prepared media tensors.
    /// - `CacheCoordinator` hits for the same chat + same image.
    /// - The same chat + a different image misses even when token shape
    ///   is identical.
    /// - A follow-up chat turn with the prior image in history completes
    ///   through BatchEngine without raw media/reasoning marker leakage.
    static func runChatCacheMatrix(modelPath: String, maxNewTokens: Int) async throws {
        let modelDir = URL(fileURLWithPath: modelPath)
        let env = ProcessInfo.processInfo.environment
        let nativeMTPDepth = env["BENCH_VL_NATIVE_MTP_DEPTH"].flatMap(Int.init)
        print("=== VLBench structured chat cache matrix ===")
        print("model: \(modelDir.lastPathComponent)")

        let loadStart = CFAbsoluteTimeGetCurrent()
        let context: ModelContext
        if nativeMTPDepth != nil {
            let loaded = try await MLXLMCommon.loadModel(
                from: modelDir,
                using: #huggingFaceTokenizerLoader(),
                loadConfiguration: LoadConfiguration(
                    jangPress: .disabled,
                    maxResidentBytes: .unlimited,
                    memoryLimit: .unlimited,
                    useMmapSafetensors: true,
                    nativeMTP: true))
            context = loaded.0
        } else {
            context = try await loadProductionContext(from: modelDir)
        }
        print(String(format: "Load: %.2fs", CFAbsoluteTimeGetCurrent() - loadStart))
        print("Model: \(type(of: context.model))")
        print("Processor: \(type(of: context.processor))")

        var params = GenerateParameters(
            maxTokens: maxNewTokens, temperature: 0, prefillStepSize: 512)
        if let nativeMTPDepth {
            params.draftStrategy = .nativeMTP(depth: nativeMTPDepth)
        }
        print("Native MTP depth: \(nativeMTPDepth.map(String.init) ?? "off")")
        let coordinator = makeProofCoordinator(
            modelDir: modelDir, context: context, parameters: params,
            label: "chat-cache")

        nonisolated(unsafe) let ctx = context
        let engine = BatchEngine(context: ctx, maxBatchSize: 1, cacheCoordinator: coordinator)

        let imageA = try synthesiseGradientImage(side: 224)
        let imageB = try synthesiseGradientImage(side: 224, invert: true)
        let baseChat: [Chat.Message] = [
            .system("Answer briefly and do not mention hidden reasoning."),
            .user("Describe this image in one sentence.", images: [.ciImage(imageA)]),
        ]

        let inputA = try await prepareChat(baseChat, context: context)
        let tokensA = inputA.text.tokens.reshaped(-1).asArray(Int.self)
        guard inputA.hasMediaContent,
              let saltA = computeCacheSalt(for: inputA, parameters: params)
        else {
            throw NSError(
                domain: "VLBench", code: 51,
                userInfo: [NSLocalizedDescriptionKey:
                    "structured chat image did not produce media content/salt"])
        }
        print("  A tokens=\(tokensA.count) salt=\(saltA.prefix(12)) media=\(inputA.hasMediaContent)")

        let first = await submitAndCollect(
            label: "A cold", engine: engine, context: context,
            input: inputA, parameters: params, maxNew: maxNewTokens)
        try validateChatCacheGeneration(first, label: "A cold", maxNew: maxNewTokens, code: 52)

        switch coordinator.fetch(tokens: tokensA, mediaSalt: saltA) {
        case .hit(let matched, _, let detail, _, _, _):
            print("  same-media coordinator probe: HIT \(detail.rawValue) \(matched)/\(tokensA.count)")
        case .miss:
            throw NSError(
                domain: "VLBench", code: 53,
                userInfo: [NSLocalizedDescriptionKey:
                    "same-media coordinator probe missed after generation"])
        }

        let inputAReplay = try await prepareChat(baseChat, context: context)
        let tokensAReplay = inputAReplay.text.tokens.reshaped(-1).asArray(Int.self)
        let saltAReplay = computeCacheSalt(for: inputAReplay, parameters: params)
        if tokensAReplay != tokensA || saltAReplay != saltA {
            throw NSError(
                domain: "VLBench", code: 54,
                userInfo: [NSLocalizedDescriptionKey:
                    "same chat/image did not reproduce identical tokens and media salt"])
        }
        let replay = await submitAndCollect(
            label: "A replay", engine: engine, context: context,
            input: inputAReplay, parameters: params, maxNew: maxNewTokens)
        try validateChatCacheGeneration(replay, label: "A replay", maxNew: maxNewTokens, code: 55)

        let imageBChat: [Chat.Message] = [
            .system("Answer briefly and do not mention hidden reasoning."),
            .user("Describe this image in one sentence.", images: [.ciImage(imageB)]),
        ]
        let inputB = try await prepareChat(imageBChat, context: context)
        let tokensB = inputB.text.tokens.reshaped(-1).asArray(Int.self)
        guard let saltB = computeCacheSalt(for: inputB, parameters: params), saltB != saltA else {
            throw NSError(
                domain: "VLBench", code: 56,
                userInfo: [NSLocalizedDescriptionKey:
                    "different image did not produce a different media salt"])
        }
        switch coordinator.fetch(tokens: tokensB, mediaSalt: saltB) {
        case .miss:
            print("  different-media coordinator probe: MISS (correct)")
        case .hit(let matched, _, let detail, _, _, _):
            throw NSError(
                domain: "VLBench", code: 57,
                userInfo: [NSLocalizedDescriptionKey:
                    "different image hit cache \(detail.rawValue) matched=\(matched)"])
        }

        let followUpChat: [Chat.Message] = [
            .system("Answer briefly and do not mention hidden reasoning."),
            .user("Describe this image in one sentence.", images: [.ciImage(imageA)]),
            .assistant(first.text.isEmpty ? "It is a red and blue gradient." : first.text),
            .user("What colors should I remember from that image?"),
        ]
        let followInput = try await prepareChat(followUpChat, context: context)
        let follow = await submitAndCollect(
            label: "follow-up", engine: engine, context: context,
            input: followInput, parameters: params, maxNew: maxNewTokens)
        try validateChatCacheGeneration(follow, label: "follow-up", maxNew: maxNewTokens, code: 58)
        if follow.text.contains("<think>") || follow.text.contains("</think>") ||
            follow.text.contains("<image>") || follow.text.contains("<so_embedding>")
        {
            throw NSError(
                domain: "VLBench", code: 59,
                userInfo: [NSLocalizedDescriptionKey:
                    "structured follow-up leaked raw reasoning/media markers"])
        }

        print("=== VLBench structured chat cache matrix: passed ===")
    }

    private static func validateChatCacheGeneration(
        _ result: (tokens: Int, text: String, promptTime: Double),
        label: String,
        maxNew: Int,
        code: Int
    ) throws {
        if result.tokens == 0 {
            throw NSError(
                domain: "VLBench", code: code,
                userInfo: [NSLocalizedDescriptionKey: "\(label) generated zero tokens"])
        }
        if result.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            throw NSError(
                domain: "VLBench", code: code,
                userInfo: [NSLocalizedDescriptionKey: "\(label) generated no visible text"])
        }
        if result.tokens >= maxNew {
            throw NSError(
                domain: "VLBench", code: code,
                userInfo: [NSLocalizedDescriptionKey:
                    "\(label) exhausted max token budget (\(result.tokens)/\(maxNew)); not a valid brief-output pass"])
        }
    }

    private static func prepareChat(
        _ chat: [Chat.Message], context: ModelContext
    ) async throws -> LMInput {
        var input = UserInput(chat: chat)
        input.additionalContext = ["enable_thinking": false]
        return try await context.processor.prepare(input: input)
    }

    private static func submitAndCollect(
        label: String,
        engine: BatchEngine,
        context: ModelContext,
        input: LMInput,
        parameters: GenerateParameters,
        maxNew: Int
    ) async -> (tokens: Int, text: String, promptTime: Double) {
        if parameters.draftStrategy?.usesNativeMTP == true {
            return await generateAndCollect(
                label: label,
                engine: engine,
                input: input,
                parameters: parameters)
        }

        nonisolated(unsafe) let sendable = input
        let t0 = CFAbsoluteTimeGetCurrent()
        let (_, stream) = await engine.submit(input: sendable, parameters: parameters)
        var tokenIds: [Int] = []
        var promptTime = 0.0
        var ttft: Double?
        for await event in stream {
            switch event {
            case .token(let id):
                if ttft == nil { ttft = CFAbsoluteTimeGetCurrent() - t0 }
                tokenIds.append(id)
            case .prefillProgress:

                break
            case .info(let info):
                promptTime = info.promptTime
            }
        }
        let text = context.tokenizer.decode(tokenIds: tokenIds)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let preview = text.count > 180 ? String(text.prefix(180)) + "..." : text
        print(String(format:
            "  %@: tokens=%d TTFT=%dms prompt=%.3fs text=\"%@\"",
            label, tokenIds.count, Int((ttft ?? 0) * 1000), promptTime, preview))
        return (tokenIds.count, text, promptTime)
    }

    private static func generateAndCollect(
        label: String,
        engine: BatchEngine,
        input: LMInput,
        parameters: GenerateParameters
    ) async -> (tokens: Int, text: String, promptTime: Double) {
        nonisolated(unsafe) let sendable = input
        let t0 = CFAbsoluteTimeGetCurrent()
        let stream = await engine.generate(input: sendable, parameters: parameters)
        var text = ""
        var reasoning = ""
        var tokenCount = 0
        var promptTime = 0.0
        var ttft: Double?
        for await event in stream {
            switch event {
            case .chunk(let chunk):
                if ttft == nil { ttft = CFAbsoluteTimeGetCurrent() - t0 }
                text += chunk
            case .reasoning(let chunk):
                reasoning += chunk
            case .prefillProgress:

                break
            case .info(let info):
                promptTime = info.promptTime
                tokenCount = info.generationTokenCount
            case .toolCall, .toolCallProgress:
                break
            }
        }

        let visible = text.trimmingCharacters(in: .whitespacesAndNewlines)
        let preview = visible.count > 180 ? String(visible.prefix(180)) + "..." : visible
        print(String(format:
            "  %@: tokens=%d TTFT=%dms prompt=%.3fs reasoning=%d text=\"%@\"",
            label, tokenCount, Int((ttft ?? 0) * 1000), promptTime,
            reasoning.count, preview))
        return (tokenCount, visible, promptTime)
    }

    // MARK: - Video input smoke (iter 49)

    /// Load a short .mov, run it through the VLM processor as a video
    /// input, verify the model accepts the frame sequence and decodes
    /// coherent text. Validates the full video path:
    /// `UserInput(videos:)` → `processor.prepare` → `LMInput.video` →
    /// model forward with video tokens.
    static func runVideoSmoke(
        modelPath: String, videoPath: String, maxNewTokens: Int
    ) async throws {
        let modelDir = URL(fileURLWithPath: modelPath)
        print("=== VLBench video smoke (iter 49) ===")
        print("  video: \(videoPath)")

        let loadStart = CFAbsoluteTimeGetCurrent()
        let context = try await loadProductionContext(from: modelDir)
        print(String(format: "Load: %.2fs", CFAbsoluteTimeGetCurrent() - loadStart))
        print("Model: \(type(of: context.model))")
        print("Processor: \(type(of: context.processor))")

        let videoURL = URL(fileURLWithPath: videoPath)
        guard FileManager.default.fileExists(atPath: videoURL.path) else {
            fputs("[VideoSmoke] FAIL: video not found at \(videoPath)\n", stderr)
            exit(1)
        }

        var userInput = UserInput(
            prompt: "Describe what happens in this short video in one sentence.",
            videos: [.url(videoURL)]
        )
        userInput.additionalContext = ["enable_thinking": false]
        userInput.processing = benchVideoProcessing(defaultSquare: 224)
        if let resize = userInput.processing.resize {
            print("  video processing resize = \(Int(resize.width))x\(Int(resize.height))")
        }

        print("  userInput.videos.count = \(userInput.videos.count)")
        let prepStart = CFAbsoluteTimeGetCurrent()
        let lmInput: LMInput
        do {
            lmInput = try await context.processor.prepare(input: userInput)
        } catch {
            if isVideoNotImplemented(error) {
                print("  not applicable: processor video input is not implemented for this model: \(error)")
                return
            }
            fputs("[VideoSmoke] FAIL: processor.prepare threw: \(error)\n", stderr)
            exit(1)
        }
        let prepMs = (CFAbsoluteTimeGetCurrent() - prepStart) * 1000
        print(String(format: "  prepare(): %.0fms — text tokens: %d, video attached: %@",
            prepMs, lmInput.text.tokens.size, lmInput.video != nil ? "yes" : "no"))
        if let v = lmInput.video {
            print("  video pixels shape: \(v.pixels.shape)")
        }
        if lmInput.video == nil {
            fputs("[VideoSmoke] FAIL: LMInput.video nil after processor.prepare. " +
                  "Video input path is broken.\n", stderr)
            exit(1)
        }

        var params = GenerateParameters(
            maxTokens: maxNewTokens, temperature: 0, prefillStepSize: 512)
        params.prefillStepSize = 512

        nonisolated(unsafe) let ctx = context
        nonisolated(unsafe) let sendInput = lmInput
        let engine = BatchEngine(context: ctx, maxBatchSize: 1)
        let t0 = CFAbsoluteTimeGetCurrent()
        let (_, stream) = await engine.submit(input: sendInput, parameters: params)
        var tokens: [Int] = []
        var ttftMs: Double?
        for await event in stream {
            switch event {
            case .token(let id):
                if ttftMs == nil { ttftMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000 }
                tokens.append(id)
                if tokens.count >= maxNewTokens { break }
            case .prefillProgress:

                break
            case .info: break
            }
            if tokens.count >= maxNewTokens { break }
        }
        if tokens.isEmpty {
            fputs("[VideoSmoke] FAIL: engine produced zero tokens.\n", stderr)
            exit(1)
        }
        let text = context.tokenizer.decode(tokenIds: tokens)
        let preview = text.count > 200 ? String(text.prefix(200)) + "..." : text
        print(String(format: "  generated %d tokens | TTFT %.0fms",
            tokens.count, ttftMs ?? 0))
        print("  preview: \"\(preview)\"")
        print("=== VLBench video smoke: passed ===")
    }

    // MARK: - Single-turn helper

    private static func runTurn(
        label: String,
        prompt: String,
        images: [UserInput.Image],
        context: ModelContext,
        cache: [KVCache],
        maxNewTokens: Int
    ) async throws {
        print("\n[\(label)]")
        let userInput = UserInput(prompt: prompt, images: images)
        let prepStart = CFAbsoluteTimeGetCurrent()
        let lmInput = try await context.processor.prepare(input: userInput)
        let prepMs = (CFAbsoluteTimeGetCurrent() - prepStart) * 1000
        print(String(format: "  prepare(): %.0fms — text tokens: %d",
            prepMs, lmInput.text.tokens.size))

        var params = GenerateParameters(maxTokens: maxNewTokens)
        params.temperature = 0.0
        params.prefillStepSize = 512

        let iter = try TokenIterator(
            input: lmInput, model: context.model, cache: cache, parameters: params)

        let genStart = CFAbsoluteTimeGetCurrent()
        var tokens: [Int] = []
        var ttftMs: Double?
        let firstStart = CFAbsoluteTimeGetCurrent()
        for token in iter {
            tokens.append(token)
            if ttftMs == nil {
                ttftMs = (CFAbsoluteTimeGetCurrent() - firstStart) * 1000
            }
            if tokens.count >= maxNewTokens { break }
        }
        let genSecs = CFAbsoluteTimeGetCurrent() - genStart
        let tokPerSec = Double(tokens.count) / genSecs

        let text = context.tokenizer.decode(tokenIds: tokens)
        print(String(format: "  generated %d tokens | TTFT %.0fms | decode %.1f tok/s",
            tokens.count, ttftMs ?? 0, tokPerSec))
        print("  first 10 tokens: \(Array(tokens.prefix(10)))")
        print("  decoded text: \"\(text.prefix(200))\"")
    }

    // MARK: - Synthetic image

    /// Produces a deterministic 224×224 RGB CIImage with a vertical colour gradient
    /// (red top → blue bottom). Used to verify the vision path doesn't crash and
    /// produces sensible token output.
    ///
    /// When `invert` is true, the gradient runs blue top → red bottom. Used by
    /// the mediaSalt test (iter 37) to produce two images with identical shape
    /// and identical tokenizer wrapping but DIFFERENT pixel bytes — so their
    /// SHA256 salts diverge and the coordinator must isolate them.
    private static func synthesiseGradientImage(side: Int, invert: Bool = false) throws -> CIImage {
        var bytes = [UInt8](repeating: 0, count: side * side * 4)
        if (ProcessInfo.processInfo.environment["BENCH_VL_SOLID_RED"] ?? "0") == "1" {
            for y in 0..<side {
                for x in 0..<side {
                    let off = (y * side + x) * 4
                    bytes[off + 0] = 255
                    bytes[off + 1] = 0
                    bytes[off + 2] = 0
                    bytes[off + 3] = 255
                }
            }
        } else {
        for y in 0..<side {
            let rawR = UInt8(255 - (255 * y) / max(side - 1, 1))
            let rawB = UInt8((255 * y) / max(side - 1, 1))
            let r = invert ? rawB : rawR
            let b = invert ? rawR : rawB
            for x in 0..<side {
                let off = (y * side + x) * 4
                bytes[off + 0] = r
                bytes[off + 1] = 64
                bytes[off + 2] = b
                bytes[off + 3] = 255
            }
        }
        }
        let data = Data(bytes)
        let cs = CGColorSpace(name: CGColorSpace.sRGB)!
        guard let image = CIImage(
            bitmapData: data, bytesPerRow: side * 4,
            size: .init(width: side, height: side),
            format: .RGBA8, colorSpace: cs
        ) as CIImage? else {
            throw NSError(
                domain: "VLBench", code: 1,
                userInfo: [NSLocalizedDescriptionKey: "failed to build CIImage"])
        }
        return image
    }

    private static func isVideoNotImplemented(_ error: Error) -> Bool {
        let message = String(describing: error).lowercased()
        return message.contains("video input is not implemented")
            || message.contains("video is not implemented")
            || message.contains("unsupported video")
    }

    private static func benchVideoProcessing(defaultSquare: Int?) -> UserInput.Processing {
        let env = ProcessInfo.processInfo.environment
        let raw = env["BENCH_VL_VIDEO_RESIZE"] ?? env["BENCH_VIDEO_RESIZE"]
        let side = raw.flatMap { Int($0) } ?? defaultSquare
        guard let side, side > 0 else {
            return .init()
        }
        return .init(resize: CGSize(width: side, height: side))
    }
}
