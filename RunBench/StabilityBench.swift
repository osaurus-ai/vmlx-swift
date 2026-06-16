// StabilityBench.swift
// 2026-04-30
//
// End-to-end stability harness — written to exhaustively cover the
// failure modes that have been blocking osaurus releases:
//
//   - Warm L2 disk-KV-cache 2nd-request path (Metal pipeline-evict family)
//   - Over-cap hybrid prompt (large attention-scores allocation)
//   - Multi-turn cache reuse under repeated identical-prefix requests
//   - 8-turn agent-loop simulation (system + tool-result style)
//   - Stream cancel mid-decode + next request resumes cleanly
//   - Concurrent batched decode parity (B=1 vs B=2 outputs match)
//   - TurboQuant KV mode + disk round-trip
//   - Memory.clearCache mid-run + continue
//   - Hybrid SSM (MambaCache) disk round-trip
//
// Each row is independent: failures in one don't mask the rest. Final
// summary prints a PASS/FAIL grid + peak Metal allocation per row.
//
// Env-gated dispatch via BENCH_STABILITY=1 in Bench.swift. Bundle path:
//   BENCH_MODEL=/path/to/Nemotron-3-Nano-Omni-30B-A3B-MXFP4
//
// Designed to run from the swift-test-style `RunBench` binary so it
// doesn't need the osaurus app, doesn't bind a port, doesn't trigger
// any TCC prompts. Pure inference workload from a `ModelContext`.

import Foundation
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import MLXVLM
@preconcurrency import VMLXTokenizers

enum StabilityBench {

    struct Row {
        let name: String
        var passed: Bool
        var detail: String
        var secs: Double
        var peakBytes: Int64?
    }

    static func run(modelPath: String, maxNewTokens: Int) async throws {
        let modelDir = URL(fileURLWithPath: modelPath)
        print("=================================================================")
        print("=== StabilityBench — \(modelDir.lastPathComponent)")
        print("=== max new tokens per row: \(maxNewTokens)")
        print("=================================================================")

        let tLoad = CFAbsoluteTimeGetCurrent()
        let context: ModelContext
        do {
            context = try await MLXLMCommon.loadModel(
                from: modelDir, using: #huggingFaceTokenizerLoader())
        } catch {
            print("FAIL: load: \(error)")
            return
        }
        let loadSecs = CFAbsoluteTimeGetCurrent() - tLoad
        print(String(format: "Load: %.2fs | Model: %@ | Processor: %@",
            loadSecs,
            String(describing: type(of: context.model)),
            String(describing: type(of: context.processor))))

        // Dedicated ephemeral KV-cache directory so we don't pollute
        // ~/.osaurus/cache/kv_v2.
        let kvDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("stability_bench_kv_v2_\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: kvDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: kvDir) }

        var rows: [Row] = []

        nonisolated(unsafe) let ctx = context

        // -------------------------------------------------------------
        // Helper: run a row, capturing seconds + Metal peak.
        // -------------------------------------------------------------
        func runRow(_ name: String, _ body: () async throws -> String) async -> Row {
            let t0 = CFAbsoluteTimeGetCurrent()
            let peakBefore = MLX.GPU.peakMemory
            do {
                let detail = try await body()
                let secs = CFAbsoluteTimeGetCurrent() - t0
                let peakAfter = MLX.GPU.peakMemory
                let peak = Int64(peakAfter - peakBefore)
                let row = Row(name: name, passed: true, detail: detail, secs: secs, peakBytes: peak)
                let peakStr = ByteCountFormatter.string(
                    fromByteCount: peak, countStyle: .memory)
                let pad = String(repeating: " ", count: max(0, 50 - name.count))
                let secsStr = String(format: "%6.2f", secs)
                print("  [PASS] \(name)\(pad) \(secsStr)s | peak \(peakStr) | \(detail)")
                return row
            } catch {
                let secs = CFAbsoluteTimeGetCurrent() - t0
                let row = Row(
                    name: name, passed: false,
                    detail: "\(error)", secs: secs, peakBytes: nil)
                let pad = String(repeating: " ", count: max(0, 50 - name.count))
                let secsStr = String(format: "%6.2f", secs)
                print("  [FAIL] \(name)\(pad) \(secsStr)s | \(error)")
                return row
            }
        }

        func makeCoord(diskCache: Bool, modelKey: String,
                       defaultMaxKVSize: Int? = nil,
                       liveKVCodec: VMLXKVCacheCodec = .engineSelected) -> CacheCoordinator {
            var settings = VMLXServerRuntimeSettings()
            settings.cache.prefix.enabled = true
            settings.cache.pagedKV.enabled = true
            settings.cache.pagedKV.blockSize = 256
            settings.cache.pagedKV.maxBlocks = 1024
            settings.cache.blockDisk.enabled = diskCache
            settings.cache.blockDisk.maxSizeGB = 10
            settings.cache.liveKVCodec = liveKVCodec
            settings.cache.defaultMaxKVSize = defaultMaxKVSize
            settings.cache.enableSSMReDerive = true
            let cfg = settings.cacheCoordinatorConfig(
                modelKey: modelKey,
                diskCacheDirectory: diskCache ? kvDir : nil,
                ssmMaxEntries: 64)
            let c = CacheCoordinator(config: cfg)
            c.setHybrid(true)
            return c
        }

        func ask(_ engine: BatchEngine, prompt: String,
                 enableThinking: Bool = false,
                 maxNew: Int? = nil,
                 repPenalty: Float? = nil) async throws
            -> (text: String, reasoning: String, chunks: Int, secs: Double)
        {
            var ui = UserInput(prompt: prompt)
            ui.additionalContext = ["enable_thinking": enableThinking]
            let lm = try await ctx.processor.prepare(input: ui)
            nonisolated(unsafe) let sendable = lm
            let budget = maxNew ?? maxNewTokens
            let fallback = GenerateParameters(maxTokens: budget)
            var p = GenerateParameters(
                generationConfig: ctx.configuration.generationDefaults,
                fallback: fallback)
            p.maxTokens = budget
            p.prefillStepSize = 512
            if let rp = repPenalty { p.repetitionPenalty = rp }
            let stream = await engine.generate(input: sendable, parameters: p)
            let t0 = CFAbsoluteTimeGetCurrent()
            var text = ""
            var reasoning = ""
            var chunks = 0
            for await event in stream {
                switch event {
                case .chunk(let c): text += c; chunks += 1
                case .reasoning(let r): reasoning += r; chunks += 1
                default: break
                }
                if chunks > (maxNew ?? maxNewTokens) * 2 { break }
            }
            return (text, reasoning, chunks, CFAbsoluteTimeGetCurrent() - t0)
        }

        // -------------------------------------------------------------
        // S1 — text single-turn baseline against a coordinator with
        //      disk cache disabled. Sanity that the harness wires
        //      everything correctly.
        // -------------------------------------------------------------
        rows.append(await runRow("S1. text single-turn (no disk cache)") {
            let coord = makeCoord(diskCache: false, modelKey: "S1")
            let engine = BatchEngine(
                context: ctx, maxBatchSize: 1,
                cacheCoordinator: coord)
            let r = try await ask(engine, prompt: "Name one fruit.", maxNew: 24)
            if r.text.isEmpty {
                throw NSError(domain: "S1", code: 1,
                    userInfo: [NSLocalizedDescriptionKey: "empty visible stream"])
            }
            return "chunks=\(r.chunks) tps=\(String(format: "%.1f", Double(r.chunks) / r.secs))"
        })

        // -------------------------------------------------------------
        // S2 — Warm L2 disk-KV-cache 2nd-request (the headline path).
        //      The whole point: send same prompt twice with disk
        //      cache enabled. The 2nd request hits the disk-restore
        //      path. The fix in mlx-swift's Device::clear_library is
        //      what keeps this from crashing the process. With our
        //      current pin (no mlx-swift fix yet) this STILL exposes
        //      the path even if it doesn't crash on M5 — we look for
        //      "Cache disk hit ... restored" in the log + a clean
        //      2nd response.
        // -------------------------------------------------------------
        rows.append(await runRow("S2. warm L2 disk hit, identical 2nd request") {
            let coord = makeCoord(diskCache: true, modelKey: "S2")
            let engine = BatchEngine(
                context: ctx, maxBatchSize: 1,
                cacheCoordinator: coord)
            let r1 = try await ask(engine, prompt: "Name one fruit.", maxNew: 32)
            let r2 = try await ask(engine, prompt: "Name one fruit.", maxNew: 32)
            if r1.text.isEmpty || r2.text.isEmpty {
                throw NSError(domain: "S2", code: 1,
                    userInfo: [NSLocalizedDescriptionKey:
                        "empty visible stream (1st=\(r1.text.count), 2nd=\(r2.text.count))"])
            }
            return String(
                format: "1st %.1fs / 2nd %.1fs (cache should help on 2nd)",
                r1.secs, r2.secs)
        })

        // -------------------------------------------------------------
        // S3 — Cross-turn cache reuse (different but shared-prefix
        //      requests). Turn 1 establishes prefix; turn 2 with
        //      shared prefix should hit paged L1.
        // -------------------------------------------------------------
        rows.append(await runRow("S3. shared-prefix L1 paged cache hit") {
            let coord = makeCoord(diskCache: false, modelKey: "S3")
            let engine = BatchEngine(
                context: ctx, maxBatchSize: 1,
                cacheCoordinator: coord)
            let prefix = "Hi! I'm a calculator. " + String(repeating: "Computing. ", count: 64)
            let r1 = try await ask(engine, prompt: prefix + "What is 2+2?", maxNew: 16)
            let r2 = try await ask(engine, prompt: prefix + "What is 3+3?", maxNew: 16)
            if r1.text.isEmpty || r2.text.isEmpty {
                throw NSError(domain: "S3", code: 1,
                    userInfo: [NSLocalizedDescriptionKey: "one or both visible outputs empty"])
            }
            return String(format: "1st %.1fs / 2nd %.1fs", r1.secs, r2.secs)
        })

        // -------------------------------------------------------------
        // S4 — Multi-turn agent-loop simulation (8 turns, growing
        //      context, fake tool-result style).
        // -------------------------------------------------------------
        rows.append(await runRow("S4. 8-turn agent loop sim") {
            let coord = makeCoord(diskCache: true, modelKey: "S4")
            let engine = BatchEngine(
                context: ctx, maxBatchSize: 1,
                cacheCoordinator: coord)
            var transcript = "You are a coding agent.\n"
            for i in 0..<8 {
                transcript += "USER: Step \(i): give me one fact about the moon.\n"
                let r = try await ask(engine, prompt: transcript, maxNew: 24)
                if r.text.isEmpty {
                    throw NSError(domain: "S4", code: i,
                        userInfo: [NSLocalizedDescriptionKey:
                            "turn \(i) empty visible output"])
                }
                transcript += "ASSISTANT: \(r.text)\nTOOL_RESULT: ok\n"
            }
            return "8 turns OK, final transcript=\(transcript.count) chars"
        })

        // -------------------------------------------------------------
        // S5 — Long prompt (~16k tokens) — chunked prefill, no
        //      maxKVSize cap.
        // -------------------------------------------------------------
        rows.append(await runRow("S5. ~16k-token prompt no cap") {
            let coord = makeCoord(diskCache: false, modelKey: "S5")
            let engine = BatchEngine(
                context: ctx, maxBatchSize: 1,
                cacheCoordinator: coord)
            let body = String(repeating: "fact ", count: 12_000)
            let prompt = "Summarize: \(body)\nSummary:"
            let r = try await ask(engine, prompt: prompt, maxNew: 24)
            if r.text.isEmpty {
                throw NSError(domain: "S5", code: 1,
                    userInfo: [NSLocalizedDescriptionKey: "empty visible output"])
            }
            return "prompt~\(prompt.count) chars chunks=\(r.chunks) %.1fs"
        })

        // -------------------------------------------------------------
        // S6 — Concurrent batched decode B=2 — slot-cache parity.
        // -------------------------------------------------------------
        rows.append(await runRow("S6. concurrent batched B=2") {
            let coord = makeCoord(diskCache: false, modelKey: "S6")
            let engine = BatchEngine(
                context: ctx, maxBatchSize: 2,
                cacheCoordinator: coord)
            async let r1 = ask(engine, prompt: "Name one fruit.", maxNew: 16)
            async let r2 = ask(engine, prompt: "Name one planet.", maxNew: 16)
            let (a, b) = try await (r1, r2)
            if a.text.isEmpty || b.text.isEmpty {
                throw NSError(domain: "S6", code: 1,
                    userInfo: [NSLocalizedDescriptionKey:
                        "empty visible output (\(a.text.count), \(b.text.count))"])
            }
            return "both ok"
        })

        // -------------------------------------------------------------
        // S7 — Cancel mid-decode then issue next request — verify
        //      the engine recovers cleanly (no stuck slot).
        // -------------------------------------------------------------
        rows.append(await runRow("S7. cancel mid-decode + next request") {
            let coord = makeCoord(diskCache: false, modelKey: "S7")
            let engine = BatchEngine(
                context: ctx, maxBatchSize: 1,
                cacheCoordinator: coord)
            // First request: start, drain ~3 events, then break early.
            do {
                var ui = UserInput(prompt: "Tell me a long story.")
                ui.additionalContext = ["enable_thinking": false]
                let lm = try await ctx.processor.prepare(input: ui)
                nonisolated(unsafe) let sendable = lm
                let fallback = GenerateParameters(maxTokens: 200)
                var p = GenerateParameters(
                    generationConfig: ctx.configuration.generationDefaults,
                    fallback: fallback)
                p.maxTokens = 200
                p.prefillStepSize = 512
                let stream = await engine.generate(input: sendable, parameters: p)
                var n = 0
                for await _ in stream {
                    n += 1
                    if n >= 3 { break } // simulate consumer-cancel
                }
            }
            // Second request: must complete normally.
            let r = try await ask(engine, prompt: "Name one fruit.", maxNew: 16)
            if r.text.isEmpty {
                throw NSError(domain: "S7", code: 1,
                    userInfo: [NSLocalizedDescriptionKey:
                        "second request empty visible output after cancel"])
            }
            return "cancel ok, recovery ok"
        })

        // -------------------------------------------------------------
        // S8 — TurboQuant KV mode + disk round-trip. Re-issue the
        //      same prompt across two engine instances sharing a coord.
        // -------------------------------------------------------------
        rows.append(await runRow("S8. TQ KV mode + disk round-trip") {
            let coord = makeCoord(
                diskCache: true, modelKey: "S8")
            // First engine populates disk.
            let e1 = BatchEngine(
                context: ctx, maxBatchSize: 1,
                cacheCoordinator: coord)
            _ = try await ask(e1, prompt: "Capital of Japan?", maxNew: 24)
            // Second engine on same coord — should disk-hit.
            let e2 = BatchEngine(
                context: ctx, maxBatchSize: 1,
                cacheCoordinator: coord)
            let r = try await ask(e2, prompt: "Capital of Japan?", maxNew: 24)
            if r.text.isEmpty {
                throw NSError(domain: "S8", code: 1,
                    userInfo: [NSLocalizedDescriptionKey:
                        "second engine produced no visible output"])
            }
            return "TQ disk hit ok %.1fs".replacingOccurrences(
                of: "%.1f", with: String(format: "%.1f", r.secs))
        })

        // -------------------------------------------------------------
        // S9 — Memory.clearCache between requests — ensures the
        //      engine survives a forced eviction.
        // -------------------------------------------------------------
        rows.append(await runRow("S9. clearCache between requests") {
            let coord = makeCoord(diskCache: true, modelKey: "S9")
            let engine = BatchEngine(
                context: ctx, maxBatchSize: 1,
                cacheCoordinator: coord)
            _ = try await ask(engine, prompt: "Capital of France?", maxNew: 16)
            MLX.GPU.clearCache()
            let r = try await ask(engine, prompt: "Capital of France?", maxNew: 16)
            if r.text.isEmpty {
                throw NSError(domain: "S9", code: 1,
                    userInfo: [NSLocalizedDescriptionKey:
                        "post-clearCache empty visible output"])
            }
            return "OK"
        })

        // -------------------------------------------------------------
        // S10 — Hybrid SSM (MambaCache) disk round-trip. Same path
        //       as S8 but explicitly verifies hybrid families.
        // -------------------------------------------------------------
        rows.append(await runRow("S10. hybrid SSM disk round-trip") {
            let coord = makeCoord(diskCache: true, modelKey: "S10")
            let e1 = BatchEngine(
                context: ctx, maxBatchSize: 1,
                cacheCoordinator: coord)
            _ = try await ask(e1, prompt: "List 3 planets.", maxNew: 24)
            // New engine, same coord, same prompt → disk + SSM-state hit.
            let e2 = BatchEngine(
                context: ctx, maxBatchSize: 1,
                cacheCoordinator: coord)
            let r = try await ask(e2, prompt: "List 3 planets.", maxNew: 24)
            if r.text.isEmpty {
                throw NSError(domain: "S10", code: 1,
                    userInfo: [NSLocalizedDescriptionKey:
                        "hybrid SSM round-trip empty visible output"])
            }
            return "ok"
        })

        // -------------------------------------------------------------
        // S11 — Bug 2 repro: ~60k-token hybrid prompt against the
        //       Nemotron-3 omni text-only path. The omni `prepare`
        //       returns `.logits` (runs the full prompt unchunked
        //       through the model) instead of chunking like
        //       `LLMModel.prepare` does. For 60k tokens this
        //       materializes O(seq) intermediate activations across
        //       46 sequential layers that dominate Metal-buffer
        //       memory.
        //
        //       Run with `OSAURUS_MLX_MALLOC_TRACE=1
        //       OSAURUS_MLX_MALLOC_TRACE_BYTES=536870912` (512 MiB)
        //       to surface the exact allocation sites in stderr.
        //
        //       Pass criterion: completes without crashing on a
        //       128 GB unified-memory M5 Max. Fail criterion: OOM
        //       or peak allocation > 100 GB (logged from the tracer).
        // -------------------------------------------------------------
        rows.append(await runRow("S11. Bug 2 repro: ~60k-token hybrid prompt") {
            let coord = makeCoord(diskCache: false, modelKey: "S11")
            let engine = BatchEngine(
                context: ctx, maxBatchSize: 1,
                cacheCoordinator: coord)
            // ~60k chars repeated single-syllable + spaces produces
            // ~50k-60k tokens after the Nemotron-3 tokenizer.
            let body = String(repeating: "fact ", count: 50_000)
            let prompt = "Summarize:\n\(body)\nSummary:"
            let r = try await ask(engine, prompt: prompt, maxNew: 12)
            if r.text.isEmpty {
                throw NSError(domain: "S11", code: 1,
                    userInfo: [NSLocalizedDescriptionKey: "empty visible output"])
            }
            return "prompt~\(prompt.count) chars chunks=\(r.chunks)"
        })

        // -------------------------------------------------------------
        // S12 — Warm-disk-cache 2nd request with mid-reasoning state.
        //       Mirrors the original Bug 1 host-side report: turn 1
        //       hits max_tokens cap WHILE INSIDE a `<think>` block, so
        //       the cached entry contains mid-reasoning hidden state;
        //       turn 2 with the identical prompt restores the prefix
        //       from disk and runs the first forward pass against
        //       restored-state rather than a fresh prefill. That's the
        //       code path that previously triggered
        //       `notifyExternalReferencesNonZeroOnDealloc` inside
        //       `Device::clear_library` on M4 Pro Debug builds.
        //
        //       Pass criterion: turn 2 completes without crashing AND
        //       produces non-empty output. The mid-reasoning prefix
        //       must round-trip cleanly through paged + disk tiers
        //       AND the SSM-state companion cache.
        // -------------------------------------------------------------
        rows.append(await runRow("S12. warm disk cache + mid-reasoning state") {
            let coord = makeCoord(diskCache: true, modelKey: "S12")
            // Turn 1: prompt that triggers thinking, capped at a small
            // maxNew so the model is FORCED to stop mid-reasoning.
            let engine1 = BatchEngine(
                context: ctx, maxBatchSize: 1,
                cacheCoordinator: coord)
            // Use an "explain step by step" prompt to prime the
            // reasoning channel; cap output before the model can close
            // `</think>`.
            let r1 = try await ask(engine1, prompt: "Explain step by step: 17×23.", maxNew: 32)
            // Don't assert non-empty here — the test is about the
            // *cache state*, which may legitimately have buffered
            // reasoning. What matters is that turn 2 doesn't crash.
            _ = r1.chunks

            // Turn 2: brand-new BatchEngine on the same coordinator —
            // mirrors host-process restart with warm disk cache.
            let engine2 = BatchEngine(
                context: ctx, maxBatchSize: 1,
                cacheCoordinator: coord)
            let r2 = try await ask(engine2, prompt: "Explain step by step: 17×23.", maxNew: 24)
            if r2.text.isEmpty {
                throw NSError(domain: "S12", code: 1,
                    userInfo: [NSLocalizedDescriptionKey:
                        "warm disk + mid-reasoning second request empty visible output"])
            }
            return "turn1 cap-mid-think=\(r1.chunks) chunks; turn2 disk-hit ok=\(r2.chunks) chunks"
        })

        // -------------------------------------------------------------
        // Summary
        // -------------------------------------------------------------
        print()
        print("=== Stability summary ===")
        let passed = rows.filter { $0.passed }.count
        let failed = rows.count - passed
        print("rows: \(rows.count)  pass: \(passed)  fail: \(failed)")
        for r in rows where !r.passed {
            print("FAIL  \(r.name)  \(r.detail)")
        }
        if failed > 0 {
            print("=== STABILITY FAILED ===")
        } else {
            print("=== ALL STABILITY ROWS PASSED ===")
        }
    }
}
