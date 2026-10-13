// BENCH_SPEED=1 — Swift twin of the Python campaign probe `mtp_anatomy_probe.py` (vMLX Python, 2026-10-04..06),
// so Swift and Python speeds on the same machine and bundle compare like for like.
//
// Same prompts (prose x3, code x3, easy_code x2, easy_prose x2), greedy, thinking off, max 300 tokens,
// one 16-token warm-up discarded. Decode tok/s uses engine generation tokens / generateTime;
// chunk timestamps are diagnostic only because streaming chunks may be coalesced.
//
// Production path: BatchEngine.generate (the host chat window's route -> solo fast path for MTP/DFlash2).
// Arms (BENCH_SPEED_ARM):
//   default   bundle-aware production selection (the default when no arm is supplied)
//   ar        explicit plain decode
//   adaptive  .nativeMTP(depth: BENCH_SPEED_MTP_DEPTH=3) + .adaptive(maximumDepth: BENCH_SPEED_MTP_MAX=5),
//             the Osaurus app's Adaptive mapping (MLXBatchAdapter.nativeMTPDepthPolicy)
//   dflash2   .dflash2(drafterPath: <bundle>/dflash2, blockSize: BENCH_SPEED_DFLASH2_BLOCK or nil)
// Output: one JSON line per prompt plus a summary (medians) to stdout and BENCH_SPEED_OUT.

import CoreImage
import Foundation
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import MLXVLM
import VMLXTokenizers

private let speedPrompts: [(String, [String])] = [
    (
        "prose",
        [
            "Write a detailed essay of about 400 words on the history of the printing press and its effect on literacy in Europe.",
            "Write a detailed essay of about 400 words on how coral reefs form, why they bleach, and what restoration projects have learned.",
            "Write a detailed essay of about 400 words on the economics of medieval Venetian trade with the eastern Mediterranean.",
        ]
    ),
    (
        "code",
        [
            "Write a Python class implementing an LRU cache with get(key) and put(key, value) in O(1) using an OrderedDict, then a small unittest covering eviction order. Code only, no prose.",
            "Write a Rust function that parses an ISO-8601 date string into (year, month, day) with error handling, plus three unit tests. Code only, no prose.",
            "Write a TypeScript debounce(fn, ms) utility with leading/trailing options and cancel(), plus Jest tests for both modes. Code only, no prose.",
        ]
    ),
    (
        "easy_code",
        [
            "Write Python code that prints the numbers 1 to 100, one print statement per line, no loops. Code only.",
            "Write a JavaScript array literal containing the integers 1 through 150 in order. Code only.",
        ]
    ),
    (
        "easy_prose",
        [
            "Write the sentence: The quick brown fox jumps over the lazy dog. Write it 25 times, one per line.",
            "List the days of the week in order, Monday to Sunday, ten times, separated by commas.",
        ]
    ),
]

func runSpeedProbe(modelPath: String) async throws {
    let env = ProcessInfo.processInfo.environment
    let modelDir = URL(fileURLWithPath: modelPath)
    let arm = env["BENCH_SPEED_ARM"] ?? "default"
    let configData = try Data(contentsOf: modelDir.appendingPathComponent("config.json"))
    let jang = try? JangLoader.loadConfig(at: modelDir)
    let status = try? MTPBundleInspector.inspect(modelDirectory: modelDir, jangConfig: jang)
    let settings = VMLXServerRuntimeSettings()
    let defaultStrategy = settings.resolvedMTPDraftStrategy(
        configData: configData, jangConfig: jang, status: status, bundleDirectory: modelDir)
    let defaultLoad = settings.resolvedLoadConfiguration(
        configData: configData, jangConfig: jang, status: status)
    let maxTokens = Int(env["BENCH_SPEED_MAXTOK"] ?? "300") ?? 300
    let loadStart = CFAbsoluteTimeGetCurrent()
    let context: ModelContext
    if arm == "default" {
        context = try await MLXLMCommon.loadModel(
            from: modelDir, using: #huggingFaceTokenizerLoader(),
            loadConfiguration: defaultLoad
        ).0
    } else if arm == "adaptive" {
        context = try await MLXLMCommon.loadModel(
            from: modelDir, using: #huggingFaceTokenizerLoader(),
            loadConfiguration: LoadConfiguration(nativeMTP: true)
        ).0
    } else {
        // Same load policy as the app (osaurus loads with a LoadConfiguration): wired-limit residency,
        // memory caps. The configuration-less overload skips them, which made "ar" arms measure a
        // different residency state than every "default" arm (91 GB GLM: 1.4-1.8 vs 25 tok/s at random).
        context = try await MLXLMCommon.loadModel(
            from: modelDir, using: #huggingFaceTokenizerLoader(), loadConfiguration: .default
        ).0
    }
    print(
        "[BENCH_SPEED_RESOLUTION] requested=\(arm) bundleDefault=\(String(describing: defaultStrategy))"
    )
    print(
        String(
            format: "[BENCH_SPEED] arm=%@ model=%@ load=%.1fs type=%@", arm,
            modelDir.lastPathComponent,
            CFAbsoluteTimeGetCurrent() - loadStart, String(describing: type(of: context.model))))
    nonisolated(unsafe) let ctx = context
    // BENCH_SPEED_CACHE_DIR=<dir>: a disk-backed prefix cache (what the app runs), so multi-turn
    // smokes exercise restore.
    let coordinator: CacheCoordinator? = env["BENCH_SPEED_CACHE_DIR"].map { dir in
        var cfg = CacheCoordinatorConfig()
        cfg.usePagedCache = false
        cfg.enableDiskCache = true
        cfg.diskCacheDir = URL(fileURLWithPath: dir)
        cfg.modelKey = modelDir.lastPathComponent
        let c = CacheCoordinator(config: cfg)
        c.setPagedIncompatible(true)
        return c
    }
    let engine = BatchEngine(context: ctx, maxBatchSize: 1, cacheCoordinator: coordinator)

    func params(_ budget: Int) -> GenerateParameters {
        var p = GenerateParameters(
            maxTokens: budget, temperature: 0, topP: 1, topK: 0, minP: 0,
            repetitionPenalty: nil)
        // BENCH_SPEED_PREFILL_STEP=<n>: prefill chunk size (the app's per-request prefillStepSize).
        if let step = env["BENCH_SPEED_PREFILL_STEP"].flatMap(Int.init), step > 0 { p.prefillStepSize = step }
        // BENCH_SPEED_SAMPLED=1: the bundle sampler the app sends by default (Qwen3.8: T 1.0, top_p 0.95, top_k 20).
        if env["BENCH_SPEED_SAMPLED"] == "1" {
            p.temperature = 1.0
            p.topP = 0.95
            p.topK = 20
            p.randomSeed = UInt64(env["BENCH_SPEED_SEED"] ?? "1234") ?? 1234
        }
        switch arm {
        case "default":
            p.draftStrategy = defaultStrategy
            if defaultStrategy?.nativeMTPDepth != nil {
                p.nativeMTPDepthPolicy = .adaptive(maximumDepth: 5)
            }
        case "adaptive":
            let depth = Int(env["BENCH_SPEED_MTP_DEPTH"] ?? "3") ?? 3
            let maxDepth = Int(env["BENCH_SPEED_MTP_MAX"] ?? "5") ?? 5
            p.draftStrategy = .nativeMTP(depth: depth)
            p.nativeMTPDepthPolicy = .adaptive(maximumDepth: maxDepth)
        case "dflash2":
            let block = env["BENCH_SPEED_DFLASH2_BLOCK"].flatMap(Int.init)
            p.draftStrategy = .dflash2(
                drafterPath: modelDir.appendingPathComponent("dflash2"), blockSize: block)
        case "dspark":
            let block = env["BENCH_SPEED_DFLASH2_BLOCK"].flatMap(Int.init)
            p.draftStrategy = .dflash2(
                drafterPath: modelDir.appendingPathComponent("dspark"), blockSize: block)
        default:
            break
        }
        return p
    }

    struct Row {
        var cat: String
        var tokens: Int
        var promptTokens = 0
        var decode: Double
        var ttft: Double
        var text: String
        var chunkRate: Double
    }
    // BENCH_SPEED_FILLER=<n>: prepend ~n tokens of unrelated context (in-app prompts carry a
    // ~2.5k-token agent system prompt + history; decode cost depends on context length).
    let filler: String = {
        guard let n = Int(env["BENCH_SPEED_FILLER"] ?? ""), n > 0 else { return "" }
        let sentence = "The archive records list shipping manifests, tide tables and lighthouse logs. "
        return "Background notes (ignore):\n" + String(repeating: sentence, count: max(1, n / 14)) + "\n\n"
    }()
    func run(_ prompt: String, _ budget: Int, cat: String) async throws -> Row {
        var input = UserInput(prompt: filler + prompt)
        input.additionalContext = ["enable_thinking": false]
        let t0 = CFAbsoluteTimeGetCurrent()
        let prepared = try await ctx.processor.prepare(input: input)
        nonisolated(unsafe) let send = prepared
        let stream = await engine.generate(input: send, parameters: params(budget))
        var first: Double?
        var last = 0.0
        var tokens = 0
        var genTime = 0.0
        var mtp = ""
        var text = ""
        for await ev in stream {
            switch ev {
            case .chunk(let c):
                let now = CFAbsoluteTimeGetCurrent()
                if first == nil { first = now }
                last = now
                text += c
            case .reasoning(let r):
                let now = CFAbsoluteTimeGetCurrent()
                if first == nil { first = now }
                last = now
                text += r
            case .info(let info):
                tokens = info.generationTokenCount
                genTime = info.generateTime
                if let d = info.dflash2Stats {
                    mtp = String(
                        format: "dflash2 verify=%d acceptLen=%.2f drafted=%d accepted=%d arFallback=%d widths=%@",
                        d.verifyCalls, d.acceptanceLength, d.draftedTokens, d.acceptedTokens,
                        d.autoregressiveFallbackTokens,
                        d.widthCycles.sorted { $0.key < $1.key }.map { "\($0.key):\($0.value)" }
                            .joined(separator: ","))
                }
                if let m = info.nativeMTPStats {
                    mtp = String(
                        format:
                            "verify=%d commit/verify=%.2f arFallback=%d accByDepth=%@ depth=%d->%d downshifts=%d arTrips=%d/%d mode=%@",
                        m.verifyCalls, m.avgCommittedPerVerify, m.arFallbackTokens,
                        m.acceptedByDepth.description, m.depth, m.activeDepth, m.adaptiveDownshifts,
                        m.arSafetyTrips, m.arSafetyResumes, m.verifierMode)
                }
            default:
                break
            }
        }
        // Primary: the engine's own generation clock (tokens / generateTime, prompt excluded) — chunk
        // stamps are unusable here because the host stream coalesces many tokens into one chunk.
        let decode = genTime > 0 ? Double(tokens) / genTime : 0
        let span = (first.map { last - $0 }) ?? 0
        let chunkRate = span > 0.2 && tokens > 1 ? Double(tokens - 1) / span : 0
        if !mtp.isEmpty { print("[BENCH_SPEED_MTP] \(cat) \(mtp)") }
        var row = Row(
            cat: cat, tokens: tokens, decode: decode, ttft: (first ?? t0) - t0, text: text,
            chunkRate: chunkRate)
        row.promptTokens = send.text.tokens.size
        return row
    }

    // BENCH_SPEED_IMAGES=a.png,b.png: media smoke — one turn per image ("color and digit"), the
    // speculative strategy of the arm, decode rate + strategy stats per turn (development check
    // only; the live app is the proof).
    if let images = env["BENCH_SPEED_IMAGES"] {
        for path in images.split(separator: ",").map(String.init) {
            guard let image = CIImage(contentsOf: URL(fileURLWithPath: path)) else { continue }
            var input = UserInput(
                prompt: "What is the background color of this image and which digit is shown? Then write two sentences about that color.",
                images: [.ciImage(image)])
            input.additionalContext = ["enable_thinking": false]
            let t0 = CFAbsoluteTimeGetCurrent()
            let prepared = try await ctx.processor.prepare(input: input)
            nonisolated(unsafe) let send = prepared
            let stream = await engine.generate(input: send, parameters: params(120))
            var text = ""
            var tokens = 0
            var genTime = 0.0
            var first: Double?
            for await ev in stream {
                switch ev {
                case .chunk(let c):
                    if first == nil { first = CFAbsoluteTimeGetCurrent() }
                    text += c
                case .info(let info):
                    tokens = info.generationTokenCount
                    genTime = info.generateTime
                default: break
                }
            }
            print(String(
                format: "[BENCH_SPEED_IMAGE] %@ tokens=%d decode_tok_s=%.2f ttft=%.2f text=%@",
                URL(fileURLWithPath: path).lastPathComponent, tokens,
                genTime > 0 ? Double(tokens) / genTime : 0, (first ?? t0) - t0,
                String(reflecting: String(text.prefix(160)))))
        }
        // Two-turn conversation: image turn, then a text follow-up with the image in history.
        if let first = images.split(separator: ",").first,
            let image = CIImage(contentsOf: URL(fileURLWithPath: String(first)))
        {
            let q1 = "What background color and which digit are in this image? One sentence."
            var answer = ""
            for turn in 0 ..< 2 {
                var chat: [Chat.Message] = [.user(q1, images: [.ciImage(image)])]
                if turn == 1 {
                    chat += [.assistant(answer), .user("Translate your sentence into French and German.")]
                }
                var input = UserInput(chat: chat)
                input.additionalContext = ["enable_thinking": false]
                let t0 = CFAbsoluteTimeGetCurrent()
                let prepared = try await ctx.processor.prepare(input: input)
                nonisolated(unsafe) let send = prepared
                let stream = await engine.generate(input: send, parameters: params(120))
                var text = ""
                var first: Double?
                var tokens = 0
                var genTime = 0.0
                for await ev in stream {
                    switch ev {
                    case .chunk(let c):
                        if first == nil { first = CFAbsoluteTimeGetCurrent() }
                        text += c
                    case .info(let info):
                        tokens = info.generationTokenCount
                        genTime = info.generateTime
                    default: break
                    }
                }
                if turn == 0 { answer = text }
                print(String(
                    format: "[BENCH_SPEED_CONV] turn=%d tokens=%d decode_tok_s=%.2f ttft=%.2f text=%@",
                    turn + 1, tokens, genTime > 0 ? Double(tokens) / genTime : 0,
                    (first ?? t0) - t0, String(reflecting: String(text.prefix(200)))))
            }
        }
        return
    }
    // BENCH_SPEED_PROSECONV=1: the live-app prose follow-up shape — essay, then a second essay with the first in
    // history. Run with and without BENCH_SPEED_CACHE_DIR to separate "restored prefix" from "history content".
    if env["BENCH_SPEED_PROSECONV"] == "1" {
        _ = try await run("Say hi.", 16, cat: "warmup")
        let q1 = "Write a 400-word essay on how the printing press changed European society. Plain prose, no headings."
        let q2 = "Now write a 400-word essay on how the steam engine changed daily life. Plain prose, no headings."
        var answer = ""
        for turn in 0 ..< 2 {
            var chat: [Chat.Message] = [.user(q1)]
            if turn == 1 { chat += [.assistant(answer), .user(q2)] }
            var input = UserInput(chat: chat)
            input.additionalContext = ["enable_thinking": false]
            let t0 = CFAbsoluteTimeGetCurrent()
            let prepared = try await ctx.processor.prepare(input: input)
            nonisolated(unsafe) let send = prepared
            let stream = await engine.generate(input: send, parameters: params(maxTokens))
            var text = ""
            var first: Double?
            var tokens = 0
            var genTime = 0.0
            for await ev in stream {
                switch ev {
                case .chunk(let c):
                    if first == nil { first = CFAbsoluteTimeGetCurrent() }
                    text += c
                case .info(let info):
                    tokens = info.generationTokenCount
                    genTime = info.generateTime
                default: break
                }
            }
            if turn == 0 { answer = text }
            print(String(
                format: "[BENCH_SPEED_PROSECONV] turn=%d tokens=%d decode_tok_s=%.2f ttft=%.2f text=%@",
                turn + 1, tokens, genTime > 0 ? Double(tokens) / genTime : 0,
                (first ?? t0) - t0, String(reflecting: String(text.prefix(160)))))
        }
        return
    }
    _ = try await run("Say hi.", 16, cat: "warmup")
    var summary: [String: [Double]] = [:]
    var lines: [String] = []
    var texts: [String] = []
    // BENCH_SPEED_CATS=prose,code — run a subset (short runs from a cool start; see the handoff's clock notes).
    let selected = env["BENCH_SPEED_CATS"].map { Set($0.split(separator: ",").map(String.init)) }
    for (cat, prompts) in speedPrompts where selected?.contains(cat) ?? true {
        for prompt in prompts {
            let r = try await run(prompt, maxTokens, cat: cat)
            summary[cat, default: []].append(r.decode)
            let preview = String(r.text.prefix(80)).replacingOccurrences(of: "\n", with: "\\n")
            let line = String(
                format:
                    "{\"cat\":\"%@\",\"tokens\":%d,\"decode_tok_s\":%.2f,\"chunk_tok_s\":%.2f,\"ttft\":%.3f,\"prompt_tokens\":%d,\"prefill_tok_s\":%.1f,\"preview\":%@}",
                cat, r.tokens, r.decode, r.chunkRate, r.ttft, r.promptTokens,
                r.ttft > 0 ? Double(r.promptTokens) / r.ttft : 0, String(reflecting: preview))
            print("[BENCH_SPEED_ROW] " + line)
            lines.append(line)
            texts.append(r.text)
        }
    }
    func median(_ v: [Double]) -> Double {
        let s = v.sorted()
        return s.isEmpty
            ? 0 : (s.count % 2 == 1 ? s[s.count / 2] : (s[s.count / 2 - 1] + s[s.count / 2]) / 2)
    }
    let med = speedPrompts.filter { selected?.contains($0.0) ?? true }.map { cat, _ in
        String(format: "\"%@\": %.2f", cat, median(summary[cat] ?? []))
    }
    .joined(separator: ", ")
    print("[BENCH_SPEED_SUMMARY] arm=\(arm) {\(med)}")
    if let textOut = env["BENCH_SPEED_TEXT_OUT"] {
        // Full generated texts in prompt order (greedy identity check across arms).
        let data = try JSONSerialization.data(withJSONObject: texts, options: [.prettyPrinted])
        try data.write(to: URL(fileURLWithPath: textOut))
    }
    if let out = env["BENCH_SPEED_OUT"] {
        try (lines + ["{\"summary\": {\(med)}}"]).joined(separator: "\n").write(
            toFile: out, atomically: true, encoding: .utf8)
    }
}
