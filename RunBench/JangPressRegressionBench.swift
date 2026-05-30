// JangPressRegressionBench.swift
// 2026-05-03
//
// Per-bundle regression sweep that exercises the new LoadConfiguration
// API path end-to-end against real model bundles.
//
// What each row checks:
//   1. Load via the new `loadModel(from:using:loadConfiguration:)`
//      overload (LoadConfiguration.default → auto JangPress + 70%
//      resident cap). Verify status reports `enabled=true` for routed
//      bundles and `enabled=false` for dense ones.
//   2. RSS sample at: pre-load, post-load, post-warm, post-quiesce.
//   3. 3-turn coherency on a fixed prompt set. Looping detector flags
//      runs that emit ≥3 identical 16-char windows back-to-back.
//   4. Quiesce check: after 6s idle, JangPress controller MUST advance
//      to .quiescing or .compressed (the failsafe state machine ticks).
//      Cold routed bytes MUST be > 0 on routed bundles.
//   5. Hybrid SSM models (NemotronH-Omni family) — second turn must
//      complete without re-running prefill from scratch (proxy for the
//      async SSM re-derive warm-pass landing correctly).
//   6. TurboQuant cache encode/decode parity — for JANGTQ bundles,
//      identical prompt twice with disk cache enabled. Both responses
//      must be non-empty.
//
// Designed to be invoked once per model from a shell loop so each
// bundle gets a fresh process (no allocator state leak between runs):
//
//   for M in /path/to/Model1 /path/to/Model2 ...; do
//     swift run RunBench --env BENCH_JPREG=1 --env BENCH_MODEL="$M"
//   done

import Foundation
import Cmlx
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import MLXVLM
@preconcurrency import VMLXTokenizers

#if canImport(Darwin)
import Darwin
#endif

enum JangPressRegressionBench {

    struct Result {
        var bundle: String
        var loadSecs: Double
        var preloadRSS_GB: Double
        var preloadFootprint_GB: Double
        var postloadRSS_GB: Double
        var postloadFootprint_GB: Double
        var postdecodeRSS_GB: Double
        var postdecodeFootprint_GB: Double
        var postquiesceRSS_GB: Double
        var postquiesceFootprint_GB: Double
        var jpEnabled: Bool
        var jpColdFraction: Double?
        var jpStateAfterQuiesce: String
        var jpTotalRoutedBytes_GB: Double
        var jpTilesUnderManagement: Int
        var mmapTrackedBufferBytes_GB: Double
        var turnsCompleted: Int
        var totalGeneratedTokens: Int
        var avgPromptTokS: Double
        var avgDecodeTokS: Double
        var avgWallTokS: Double
        var hadLooping: Bool
        var coherencyDetail: String
        var reasoningOffChars: Int?
        var reasoningOnChars: Int?
        var reasoningProbePassed: Bool?
        var ssmWarmPass2ndSecs: Double?
        var tqRoundTripPassed: Bool?
        var totalSecs: Double
        var failure: String?
    }

    /// `BENCH_JPREG_FORCE=1` flips this to true: forces JangPress on
    /// at compressPct=70 regardless of bundle size, so we can verify
    /// the compression path on bundles that the auto threshold would
    /// otherwise leave disabled (i.e. bundles ≤ 50% of physical RAM).
    /// Default false — production-default `LoadConfiguration.default`
    /// path.
    private static let forceJangPress: Bool = {
        ProcessInfo.processInfo.environment["BENCH_JPREG_FORCE"] == "1"
    }()

    private static let useMmapSafetensors: Bool = {
        ProcessInfo.processInfo.environment["BENCH_JPREG_MMAP"] != "0"
    }()

    private static let holdSeconds: UInt64 = {
        UInt64(ProcessInfo.processInfo.environment["BENCH_JPREG_HOLD_SECONDS"] ?? "0") ?? 0
    }()

    private static let longPromptMode: Bool = {
        ProcessInfo.processInfo.environment["BENCH_JPREG_LONG_PROMPTS"] == "1"
    }()

    private static let longPromptRepeat: Int = {
        Int(ProcessInfo.processInfo.environment["BENCH_JPREG_LONG_REPEAT"] ?? "220") ?? 220
    }()

    private static let loadOnly: Bool = {
        ProcessInfo.processInfo.environment["BENCH_JPREG_LOAD_ONLY"] == "1"
    }()

    /// Bench-only escape hatch for bundles whose weights are valid but whose
    /// tokenizer is not supported by swift-transformers yet. This lets the
    /// load/RSS/JangPress path be validated independently; generation still
    /// requires a real tokenizer and must leave this off.
    private static let useNullTokenizer: Bool = {
        ProcessInfo.processInfo.environment["BENCH_JPREG_NULL_TOKENIZER"] == "1"
    }()

    static func run(modelPath: String, maxNewTokens: Int = 64) async throws {
        let modelDir = URL(fileURLWithPath: modelPath)
        let bundle = modelDir.lastPathComponent
        let bundleParent = modelDir.deletingLastPathComponent().lastPathComponent
        print("=================================================================")
        print("=== JangPressRegressionBench — \(bundleParent)/\(bundle)")
        print("=== max new tokens per turn: \(maxNewTokens)")
        if longPromptMode {
            print("=== long prompt mode: repeat=\(longPromptRepeat)")
        }
        print("=================================================================")

        let t0 = CFAbsoluteTimeGetCurrent()
        var r = Result(
            bundle: bundle,
            loadSecs: 0,
            preloadRSS_GB: rssGB(),
            preloadFootprint_GB: physFootprintGB(),
            postloadRSS_GB: 0,
            postloadFootprint_GB: 0,
            postdecodeRSS_GB: 0,
            postdecodeFootprint_GB: 0,
            postquiesceRSS_GB: 0,
            postquiesceFootprint_GB: 0,
            jpEnabled: false,
            jpColdFraction: nil,
            jpStateAfterQuiesce: "—",
            jpTotalRoutedBytes_GB: 0,
            jpTilesUnderManagement: 0,
            mmapTrackedBufferBytes_GB: 0,
            turnsCompleted: 0,
            totalGeneratedTokens: 0,
            avgPromptTokS: 0,
            avgDecodeTokS: 0,
            avgWallTokS: 0,
            hadLooping: false,
            coherencyDetail: "",
            reasoningOffChars: nil,
            reasoningOnChars: nil,
            reasoningProbePassed: nil,
            ssmWarmPass2ndSecs: nil,
            tqRoundTripPassed: nil,
            totalSecs: 0,
            failure: nil)
        print(String(format: "[t=%5.1fs] pre-load RSS = %.1f GB | footprint = %.1f GB",
            0.0, r.preloadRSS_GB, r.preloadFootprint_GB))

        // -----------------------------------------------------------
        // 1. Load with the new typed LoadConfiguration path.
        // -----------------------------------------------------------
        let tLoad = CFAbsoluteTimeGetCurrent()
        let context: ModelContext
        let runtime: JangPressRuntime
        // Force-on path overrides the auto threshold so we can verify
        // the compression machinery on sub-threshold bundles. The cap
        // stays at the default 70% of physical RAM either way.
        let cfg: LoadConfiguration = forceJangPress
            ? LoadConfiguration(
                jangPress: .enabled(coldFraction: 0.70),
                maxResidentBytes: .fraction(0.70),
                memoryLimit: .fraction(0.70),
                useMmapSafetensors: useMmapSafetensors)
            : LoadConfiguration(useMmapSafetensors: useMmapSafetensors)
        if forceJangPress {
            print("[JPREG_FORCE] JangPress force-enabled at coldFraction=0.70 (overrides auto threshold)")
        }
        print("[JPREG_MMAP] useMmapSafetensors=\(useMmapSafetensors)")
        if useNullTokenizer {
            print("[JPREG_NULL_TOKENIZER] using bench stub tokenizer; load/RSS validation only")
        }
        let tokenizerLoader: any TokenizerLoader = useNullTokenizer
            ? NullTokenizerLoader()
            : #huggingFaceTokenizerLoader()
        do {
            (context, runtime) = try await loadModel(
                from: modelDir,
                using: tokenizerLoader,
                loadConfiguration: cfg)
        } catch {
            r.failure = "load: \(error)"
            r.totalSecs = CFAbsoluteTimeGetCurrent() - t0
            print("FAIL: \(r.failure!)")
            printSummary(r)
            return
        }
        defer { JangPressActivation.deactivate(runtime) }

        r.loadSecs = CFAbsoluteTimeGetCurrent() - tLoad
        r.postloadRSS_GB = rssGB()
        r.postloadFootprint_GB = physFootprintGB()
        let s0 = runtime.status()
        r.jpEnabled = s0.enabled
        r.jpColdFraction = s0.coldFraction
        r.jpTotalRoutedBytes_GB = Double(s0.totalRoutedBytes) / 1_073_741_824.0
        r.jpTilesUnderManagement = s0.tilesUnderManagement
        r.mmapTrackedBufferBytes_GB = mmapTrackedBufferGB()
        print(String(format: "[t=%5.1fs] post-load RSS = %.1f GB (Δ %+.1f) | footprint = %.1f GB (Δ %+.1f) | load %.1fs",
            CFAbsoluteTimeGetCurrent() - t0, r.postloadRSS_GB,
            r.postloadRSS_GB - r.preloadRSS_GB,
            r.postloadFootprint_GB,
            r.postloadFootprint_GB - r.preloadFootprint_GB,
            r.loadSecs))
        print("           Model: \(type(of: context.model)) | Processor: \(type(of: context.processor))")
        print("           JangPress: enabled=\(s0.enabled) cold=\(s0.coldFraction.map { String(format: "%.2f", $0) } ?? "—") backend=\(s0.backend.rawValue) state=\(backendName(s0.backend)) tiles=\(s0.tilesUnderManagement) routed=\(String(format: "%.1f", r.jpTotalRoutedBytes_GB)) GB")
        print(String(format: "           mmap tracked Metal buffers: %.1f GB",
            r.mmapTrackedBufferBytes_GB))
        printAdvisorStatus(prefix: "           [post-load]")
        if loadOnly {
            r.totalSecs = CFAbsoluteTimeGetCurrent() - t0
            r.jpStateAfterQuiesce = backendName(s0.backend)
            print("[JPREG_LOAD_ONLY] stopping after load/footprint sample")
            printSummary(r)
            if holdSeconds > 0 {
                print("[JPREG_HOLD] sleeping \(holdSeconds)s for Activity Monitor inspection")
                try? await Task.sleep(nanoseconds: holdSeconds * 1_000_000_000)
            }
            return
        }
        print("[JPREG] starting multi-turn/cache validation")

        // -----------------------------------------------------------
        // 2. 3-turn multi-turn coherency with looping detector.
        // -----------------------------------------------------------
        let kvDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("jpreg_\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: kvDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: kvDir) }

        // Detect hybrid SSM from the model's actual cache layout.
        // Qwen3.5/3.6 JANGTQ, Qwen3-Next, NemotronH, Granite hybrid,
        // FalconH1, etc. expose Mamba/Arrays state caches for their
        // linear-attention / SSM layers; hard-coding class names misses
        // valid hybrid families.
        let isHybridSSM = hasHybridSSMCache(context.model)

        // Detect JANGTQ by sniff for jangtq_runtime.safetensors next to
        // the bundle. These are the bundles that exercise the
        // TurboQuant disk encode/decode round-trip.
        let isJANGTQ = FileManager.default.fileExists(
            atPath: modelDir.appendingPathComponent("jangtq_runtime.safetensors").path)

        let coord = makeCoord(diskCache: true, modelKey: bundle, kvDir: kvDir)
        nonisolated(unsafe) let ctx = context
        let engine = BatchEngine(
            context: ctx, maxBatchSize: 1,
            cacheCoordinator: coord)

        let turnPrompts = longPromptMode
            ? makeLongTurnPrompts(repeatCount: longPromptRepeat)
            : [
                "List three large rivers in South America.",
                "Of those three rivers, which one is the longest?",
                "Roughly how many kilometers long is that longest river?",
            ]

        // Build a chat transcript across turns so the model has real
        // conversational context. This is the only honest way to
        // verify multi-turn coherency — independent prompts wouldn't
        // exercise referential pronouns ("of those", "that river").
        var transcript: [Chat.Message] = []
        var turnTexts: [String] = []
        var turnSecs: [Double] = []
        var turnInfos: [GenerateCompletionInfo] = []

        for (i, prompt) in turnPrompts.enumerated() {
            transcript.append(.user(prompt))
            let tT = CFAbsoluteTimeGetCurrent()
            do {
                let out = try await ask(engine, ctx: ctx, chat: transcript,
                    maxNewTokens: maxNewTokens)
                let visible = out.visibleText
                turnTexts.append(visible)
                turnSecs.append(CFAbsoluteTimeGetCurrent() - tT)
                if let info = out.info {
                    turnInfos.append(info)
                }
                transcript.append(.assistant(visible))
                r.turnsCompleted = i + 1
                let promptTPS = out.info?.promptTokensPerSecond ?? 0
                let decodeTPS = out.info?.tokensPerSecond ?? 0
                let genTokens = out.info?.generationTokenCount ?? 0
                let genSec = out.info?.generateTime ?? 0
                let wallTPS = turnSecs[i] > 0 ? Double(genTokens) / turnSecs[i] : 0
                print(String(format: "  T%d (wall %.1fs, apiGen %.1fs, %d tok, wall %.1f tok/s, prompt %.1f tok/s, apiDecode %.1f tok/s, %d chars): %@",
                    i + 1, turnSecs[i], genSec, genTokens, wallTPS,
                    promptTPS, decodeTPS, visible.count,
                    String(visible.prefix(80)).replacingOccurrences(of: "\n", with: " ")))
                printBlock("T\(i + 1)_ANSWER", out.text)
                if !out.reasoning.isEmpty {
                    printBlock("T\(i + 1)_REASONING", out.reasoning)
                }
            } catch {
                r.failure = "turn \(i + 1): \(error)"
                print("  T\(i + 1) FAIL: \(error)")
                break
            }
        }

        r.totalGeneratedTokens = turnInfos.reduce(0) { $0 + $1.generationTokenCount }
        if !turnInfos.isEmpty {
            r.avgPromptTokS = turnInfos.reduce(0) { $0 + $1.promptTokensPerSecond }
                / Double(turnInfos.count)
            r.avgDecodeTokS = turnInfos.reduce(0) { $0 + $1.tokensPerSecond }
                / Double(turnInfos.count)
            let paired = min(turnInfos.count, turnSecs.count)
            if paired > 0 {
                r.avgWallTokS = (0..<paired).reduce(0.0) { acc, idx in
                    let sec = turnSecs[idx]
                    guard sec > 0 else { return acc }
                    return acc + Double(turnInfos[idx].generationTokenCount) / sec
                } / Double(paired)
            }
        }

        r.postdecodeRSS_GB = rssGB()
        r.postdecodeFootprint_GB = physFootprintGB()
        let sDec = runtime.status()
        if r.jpEnabled {
            print("           [post-decode] JangPress: backend=\(backendName(sDec.backend)) tiles=\(sDec.tilesUnderManagement) routed=\(String(format: "%.1f", Double(sDec.totalRoutedBytes)/1_073_741_824.0)) GB")
            printAdvisorStatus(prefix: "           [post-decode]")
        }
        // Coherency: every turn produced non-empty text + no obvious
        // 16-char repeating window in the join.
        let join = turnTexts.joined(separator: " ⏎ ")
        r.hadLooping = detectLooping(in: join)
        r.coherencyDetail = turnTexts.enumerated().map { idx, txt in
            "T\(idx + 1)=\(txt.count)c"
        }.joined(separator: " ")

        if isHybridSSM, turnSecs.count >= 2 {
            r.ssmWarmPass2ndSecs = turnSecs[1]
        }

        // -----------------------------------------------------------
        // 2b. Reasoning on/off probe. The main chat above intentionally
        // runs with thinking disabled because that is the common osaurus
        // "plain answer" path. This probe verifies the template/context
        // flag is still wired and that neither mode degenerates into an
        // empty or looping stream.
        // -----------------------------------------------------------
        do {
            let thinkingOffMaxTokens = min(max(maxNewTokens, 48), 96)
            // Some reasoning-native families, including Step 3.7, spend
            // more than 160 tokens inside a valid `<think>` block before
            // closing and emitting visible content. Keep the direct-answer
            // probe tight, but let the thinking-on probe use the caller's
            // configured generation budget so the gate does not create a
            // false reasoning-only failure.
            let thinkingOnMaxTokens = min(max(maxNewTokens, 160), 512)
            let off = try await ask(engine, ctx: ctx,
                chat: [.user("Answer in one short sentence: why is the sky blue?")],
                maxNewTokens: thinkingOffMaxTokens,
                enableThinking: false)
            let on = try await ask(engine, ctx: ctx,
                chat: [.user("Think briefly, then answer in one short sentence: why is the sky blue?")],
                maxNewTokens: thinkingOnMaxTokens,
                enableThinking: true)
            r.reasoningOffChars = off.reasoning.count
            r.reasoningOnChars = on.reasoning.count
            let offAnswer = off.text.trimmingCharacters(in: .whitespacesAndNewlines)
            let onAnswer = on.text.trimmingCharacters(in: .whitespacesAndNewlines)
            let offVisible = off.visibleText
            let onVisible = on.visibleText
            r.reasoningProbePassed =
                !offAnswer.isEmpty && !onAnswer.isEmpty
                && !offVisible.isEmpty && !onVisible.isEmpty
                && !detectLooping(in: offVisible)
                && !detectLooping(in: onVisible)
            print("  Thinking probe: off(reasoning=\(off.reasoning.count)c answer=\(off.text.count)c) on(reasoning=\(on.reasoning.count)c answer=\(on.text.count)c) pass=\(r.reasoningProbePassed == true ? "yes" : "NO")")
            printBlock("THINKING_OFF_ANSWER", off.text)
            if !off.reasoning.isEmpty {
                printBlock("THINKING_OFF_REASONING", off.reasoning)
            }
            printBlock("THINKING_ON_ANSWER", on.text)
            if !on.reasoning.isEmpty {
                printBlock("THINKING_ON_REASONING", on.reasoning)
            }
        } catch {
            r.reasoningProbePassed = false
            print("  Thinking probe FAIL: \(error)")
        }

        // -----------------------------------------------------------
        // 3. TurboQuant cache disk round-trip — JANGTQ only.
        // -----------------------------------------------------------
        if isJANGTQ {
            do {
                let r1 = try await ask(engine, ctx: ctx,
                    prompt: "Name a single bird in one word.", maxNewTokens: 16)
                let r2 = try await ask(engine, ctx: ctx,
                    prompt: "Name a single bird in one word.", maxNewTokens: 16)
                r.tqRoundTripPassed = !r1.visibleText.isEmpty && !r2.visibleText.isEmpty
                printBlock("TQ_ROUNDTRIP_1", r1.visibleText)
                printBlock("TQ_ROUNDTRIP_2", r2.visibleText)
            } catch {
                r.tqRoundTripPassed = false
                print("  TQ round-trip FAIL: \(error)")
            }
        }

        // -----------------------------------------------------------
        // 4. Quiesce window: sleep 6s, then sample state + RSS.
        //    The controller's failsafe state machine MUST advance
        //    past .armed within ~5s of last `didFinishInference`.
        // -----------------------------------------------------------
        print(String(format: "[t=%5.1fs] sleeping 6s for quiesce…",
            CFAbsoluteTimeGetCurrent() - t0))
        try await Task.sleep(nanoseconds: 6_000_000_000)

        r.postquiesceRSS_GB = rssGB()
        r.postquiesceFootprint_GB = physFootprintGB()
        let s1 = runtime.status()
        r.jpStateAfterQuiesce = backendName(s1.backend)
        r.jpTotalRoutedBytes_GB = Double(s1.totalRoutedBytes) / 1_073_741_824.0
        r.jpTilesUnderManagement = s1.tilesUnderManagement
        print(String(format: "[t=%5.1fs] post-quiesce RSS = %.1f GB (Δ since post-load %+.1f) | footprint = %.1f GB (Δ %+.1f)",
            CFAbsoluteTimeGetCurrent() - t0, r.postquiesceRSS_GB,
            r.postquiesceRSS_GB - r.postloadRSS_GB,
            r.postquiesceFootprint_GB,
            r.postquiesceFootprint_GB - r.postloadFootprint_GB))
        print("           JangPress: backend=\(r.jpStateAfterQuiesce) tiles=\(r.jpTilesUnderManagement) routed=\(String(format: "%.1f", r.jpTotalRoutedBytes_GB)) GB")
        printAdvisorStatus(prefix: "           [post-quiesce]", waitForIdle: true)

        r.totalSecs = CFAbsoluteTimeGetCurrent() - t0
        printSummary(r)
        if holdSeconds > 0 {
            print("-----------------------------------------------------------------")
            print("HOLDING loaded model for \(holdSeconds)s so Activity Monitor can inspect RSS.")
            print("Process pid: \(ProcessInfo.processInfo.processIdentifier)")
            print("Press Ctrl-C in this terminal/session to stop early.")
            try await Task.sleep(nanoseconds: holdSeconds * 1_000_000_000)
        }
    }

    // MARK: - Helpers

    private static func printAdvisorStatus(
        prefix: String,
        waitForIdle: Bool = false
    ) {
        if waitForIdle {
            JangPressCanonicalExpertAdvisor.shared.waitUntilIdle()
        }
        let s = JangPressCanonicalExpertAdvisor.shared.snapshot()
        print("\(prefix) RouterAdvice: enabled=\(s.enabled) async=\(s.asyncReadback) warmAdvice=\(s.warmAdvice) symbol=\(s.symbolAvailable) hotPerLayer=\(s.hotPerLayer) hot=\(s.hotExpertCount) pending=\(s.pendingObservations) droppedQueue=\(s.droppedQueueFull) readbacks=\(s.readbacks) skippedLarge=\(s.skippedLargeIndexTensors) skippedTracer=\(s.skippedTracerArrays) rewarms=\(s.rewarms) coldPairs=\(s.distinctColdAdvisedPairs) warmCalls=\(s.warmCalls) coldCalls=\(s.coldCalls) warmBytes=\(byteString(s.warmBytes)) coldBytes=\(byteString(s.coldBytes))")
    }

    private static func byteString(_ bytes: Int64) -> String {
        String(format: "%.2f GB", Double(bytes) / 1_073_741_824.0)
    }

    private static func printBlock(_ label: String, _ text: String) {
        print("--- \(label)_BEGIN ---")
        print(text)
        print("--- \(label)_END ---")
    }

    private struct AskResult {
        var text: String
        var reasoning: String
        var info: GenerateCompletionInfo?

        var visibleText: String { text }
    }

    private static func ask(
        _ engine: BatchEngine, ctx: ModelContext,
        chat: [Chat.Message], maxNewTokens: Int,
        enableThinking: Bool = false
    ) async throws -> AskResult {
        var ui = UserInput(chat: chat)
        ui.additionalContext = ["enable_thinking": enableThinking]
        let lm = try await ctx.processor.prepare(input: ui)
        nonisolated(unsafe) let sendable = lm
        let fallback = GenerateParameters(maxTokens: maxNewTokens)
        var p = GenerateParameters(
            generationConfig: ctx.configuration.generationDefaults,
            fallback: fallback)
        p.maxTokens = maxNewTokens
        p.prefillStepSize = 512
        let stream = await engine.generate(input: sendable, parameters: p)
        // Collect BOTH .chunk and .reasoning, but keep them separate:
        // reasoning-only output is not a visible-answer pass.
        var text = ""
        var reasoning = ""
        var info: GenerateCompletionInfo?
        var chunks = 0
        for await event in stream {
            switch event {
            case .chunk(let c): text += c; chunks += 1
            case .reasoning(let r): reasoning += r; chunks += 1
            case .info(let i): info = i
            default: break
            }
        }
        return AskResult(text: text, reasoning: reasoning, info: info)
    }

    /// Used by the TurboQuant disk round-trip pair to send identical
    /// single-turn requests; building a Chat array for one prompt is
    /// just noise.
    private static func ask(
        _ engine: BatchEngine, ctx: ModelContext,
        prompt: String, maxNewTokens: Int
    ) async throws -> AskResult {
        try await ask(engine, ctx: ctx,
            chat: [.user(prompt)], maxNewTokens: maxNewTokens)
    }

    private static func makeCoord(
        diskCache: Bool, modelKey: String, kvDir: URL
    ) -> CacheCoordinator {
        let cfg = CacheCoordinatorConfig(
            usePagedCache: true,
            enableDiskCache: diskCache,
            pagedBlockSize: 256,
            maxCacheBlocks: 1024,
            diskCacheMaxGB: 10,
            diskCacheDir: diskCache ? kvDir : nil,
            ssmMaxEntries: 64,
            modelKey: modelKey,
            defaultKVMode: .none,
            defaultMaxKVSize: nil,
            longPromptMultiplier: 2.0)
        let c = CacheCoordinator(config: cfg)
        c.setHybrid(true)
        return c
    }

    /// Looping detector: flag if any 16-char window appears 3+ times
    /// in a row anywhere in the joined text. Cheap, conservative —
    /// false positives on legitimately repetitive text (counters,
    /// lists) are accepted as the cost of catching real loops.
    private static func detectLooping(in text: String) -> Bool {
        let chars = Array(text)
        guard chars.count >= 48 else { return false }
        for i in 0...(chars.count - 48) {
            let window = Array(chars[i..<(i + 16)])
            if Array(chars[(i + 16)..<(i + 32)]) == window
                && Array(chars[(i + 32)..<(i + 48)]) == window
            {
                return true
            }
        }
        return false
    }

    private static func backendName(_ b: JangPressLoadOptions.Backend) -> String {
        switch b {
        case .mmap: return "mmap"
        case .none: return "none"
        }
    }

    private static func makeLongTurnPrompts(repeatCount: Int) -> [String] {
        let notes = [
            "Amazon basin note: the Amazon carries the largest discharge of any river and is usually listed as the longest river in South America.",
            "Parana basin note: the Parana and its tributaries drain a large portion of southern Brazil, Paraguay, and Argentina.",
            "Orinoco basin note: the Orinoco is a major northern South American river with a broad delta in Venezuela.",
            "Length note: public references vary, but the Amazon is commonly estimated around 6,400 kilometers depending on the source and measurement convention.",
            "Answer style note: keep the final answer concise, avoid copying the briefing, and preserve the conversation context across turns.",
        ]
        let context = (0..<max(1, repeatCount)).map { notes[$0 % notes.count] }
            .joined(separator: " ")
        return [
            """
            Read this briefing and answer the task at the end.

            \(context)

            Task: list three large rivers in South America, with one short identifying detail for each.
            """,
            "Using your previous answer and the briefing, which of those three rivers is the longest? Answer directly and give one caveat about measurement.",
            "Using the same context, roughly how many kilometers long is that longest river? Answer in one sentence.",
        ]
    }

    private static func hasHybridSSMCache(_ model: any LanguageModel) -> Bool {
        model.newCache(parameters: nil).contains { cache in
            if cache is MambaCache || cache is ArraysCache {
                return true
            }
            let name = String(describing: type(of: cache))
            return name.contains("Mamba")
                || name.contains("ArraysCache")
                || name.contains("CacheList")
        }
    }

    private static func rssGB() -> Double {
        #if canImport(Darwin)
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(
            MemoryLayout<mach_task_basic_info>.size / MemoryLayout<integer_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_,
                    task_flavor_t(MACH_TASK_BASIC_INFO),
                    $0, &count)
            }
        }
        guard kr == KERN_SUCCESS else { return 0 }
        return Double(info.resident_size) / 1_073_741_824.0
        #else
        return 0
        #endif
    }

    private static func physFootprintGB() -> Double {
        #if canImport(Darwin)
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_,
                    task_flavor_t(TASK_VM_INFO),
                    $0, &count)
            }
        }
        guard kr == KERN_SUCCESS else { return 0 }
        return Double(info.phys_footprint) / 1_073_741_824.0
        #else
        return 0
        #endif
    }

    private static func mmapTrackedBufferGB() -> Double {
        Double(mlx_safetensors_mmap_tracked_buffer_bytes()) / 1_073_741_824.0
    }

    private static func printSummary(_ r: Result) {
        print("-----------------------------------------------------------------")
        print("SUMMARY: \(r.bundle)")
        if let f = r.failure {
            print("  FAIL: \(f)")
        }
        print(String(format: "  load %.1fs | total %.1fs | turns %d/3 | looping=%@",
            r.loadSecs, r.totalSecs, r.turnsCompleted, r.hadLooping ? "YES" : "no"))
        print(String(format: "  speed: genTokens=%d avgWall=%.1f tok/s avgPrompt=%.1f tok/s avgApiDecode=%.1f tok/s",
            r.totalGeneratedTokens, r.avgWallTokS, r.avgPromptTokS, r.avgDecodeTokS))
        print(String(format: "  RSS: pre=%.1f post-load=%.1f post-decode=%.1f post-quiesce=%.1f GB",
            r.preloadRSS_GB, r.postloadRSS_GB, r.postdecodeRSS_GB, r.postquiesceRSS_GB))
        print(String(format: "  footprint: pre=%.1f post-load=%.1f post-decode=%.1f post-quiesce=%.1f GB",
            r.preloadFootprint_GB, r.postloadFootprint_GB,
            r.postdecodeFootprint_GB, r.postquiesceFootprint_GB))
        print("  JangPress: enabled=\(r.jpEnabled) cold=\(r.jpColdFraction.map { String(format: "%.2f", $0) } ?? "—") quiesce-state=\(r.jpStateAfterQuiesce) tiles=\(r.jpTilesUnderManagement) routed=\(String(format: "%.1f", r.jpTotalRoutedBytes_GB)) GB")
        print(String(format: "  mmap tracked Metal buffers: %.1f GB",
            r.mmapTrackedBufferBytes_GB))
        if let s = r.ssmWarmPass2ndSecs {
            print(String(format: "  SSM warm 2nd-turn: %.1fs", s))
        }
        if let tq = r.tqRoundTripPassed {
            print("  TQ disk round-trip: \(tq ? "PASS" : "FAIL")")
        }
        if let probe = r.reasoningProbePassed {
            print("  Thinking on/off probe: \(probe ? "PASS" : "FAIL") offReasoning=\(r.reasoningOffChars ?? 0)c onReasoning=\(r.reasoningOnChars ?? 0)c")
        }
        print("  Coherency: \(r.coherencyDetail)")
        print("=================================================================")
    }
}
