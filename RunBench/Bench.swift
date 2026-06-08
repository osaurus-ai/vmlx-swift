import Foundation
import CmlxGraphShim
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import MLXNN
import MLXVLM
@preconcurrency import VMLXJinja
@preconcurrency import VMLXTokenizers

// Multi-turn benchmark for gemma-4-26b-a4b-it-4bit
// Loads pre-tokenized turns from /tmp/gemma4_multiturn_tokens.json
// Measures TTFT, prompt processing tok/s, decode tok/s for each turn
// with cache reuse across turns.

@main
struct Bench {
    private static func homePath(_ relativePath: String) -> String {
        FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(relativePath)
            .path
    }

    private static func repoPath(_ relativePath: String) -> String {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent(relativePath)
            .path
    }

    private static func benchFailureExitCode(for error: Error) -> Int32 {
        let code = (error as NSError).code
        guard code > 0, code < 126 else { return 1 }
        return Int32(code)
    }

    static func main() async throws {
        setvbuf(stdout, nil, _IONBF, 0)
        FileHandle.standardError.write("[STDERR] Bench main() started\n".data(using: .utf8)!)

        // Model path and tokens file are configurable via env vars so a single
        // executable serves any model. Defaults preserve historical Qwen3.5 behavior.
        let env = ProcessInfo.processInfo.environment
        let modelPath = env["BENCH_MODEL"] ?? homePath("models/Qwen3.5-35B-A3B-4bit")
        let tokensPath = env["BENCH_TOKENS"] ?? "/tmp/qwen35_multiturn_tokens.json"
        let maxNew = Int(env["BENCH_MAX_TOKENS"] ?? "256") ?? 256
        let compileDecode = (env["BENCH_COMPILE_DECODE"] ?? "0") == "1"
        let compileMaxLen = Int(env["BENCH_COMPILE_MAXLEN"] ?? "16384") ?? 16384
        let modelDir = URL(fileURLWithPath: modelPath)

        if (env["BENCH_MTP_CENSUS"] ?? "0") == "1" {
            let status = try MTPBundleInspector.inspect(modelDirectory: modelDir)
            let data = try JSONEncoder().encode(status)
            let json = String(data: data, encoding: .utf8) ?? "{}"
            print("MTP_CENSUS model=\(modelDir.lastPathComponent) \(json)")
            print(
                "MTP_CENSUS_SUMMARY bundleHasMTP=\(status.bundleHasMTP) complete=\(status.hasCompleteMTPArtifact) canAutoLaunch=\(status.canAutoLaunchMTP) vision=\(status.bundleHasVision) line=\"\(status.statusLine)\""
            )
            return
        }

        // BENCH_JANGPRESS=1 activates the JangPress cold-weight tier
        // (axis E) for THIS bench run. Holds the runtime alive for
        // the full bench so the controller's failsafe state machine
        // ticks during inference. Knobs:
        //   BENCH_JANGPRESS_PCT=70        compress 70% of routed mass
        //   BENCH_JANGPRESS_FORCE=0       soft (madvise DONTNEED) vs
        //                                 force (msync MS_INVALIDATE)
        //   BENCH_JANGPRESS_PREFETCH=1    pre-fault top hot tiles at arm
        let jangPressOn = (env["BENCH_JANGPRESS"] ?? "0") == "1"
        let jangPressOpts: JangPressLoadOptions = jangPressOn ? .init(
            enabled: true,
            compressPct: Int(env["BENCH_JANGPRESS_PCT"] ?? "70") ?? 70,
            backend: .mmap,
            forceMode: (env["BENCH_JANGPRESS_FORCE"] ?? "0") == "1" ? .force : .soft,
            enablePrefetch: (env["BENCH_JANGPRESS_PREFETCH"] ?? "1") == "1"
        ) : .disabled
        var jangPressRuntime = JangPressRuntime.none
        if jangPressOn {
            jangPressRuntime = JangPressActivation.activate(
                bundleURL: modelDir, options: jangPressOpts)
            if jangPressRuntime.isActive {
                FileHandle.standardError.write(Data(
                    "[bench] JangPress active: pct=\(jangPressOpts.compressPct) force=\(jangPressOpts.forceMode == .force) prefetch=\(jangPressOpts.enablePrefetch)\n".utf8))
            } else {
                FileHandle.standardError.write(Data(
                    "[bench] JangPress activation requested but tier returned .none\n".utf8))
            }
        }
        defer {
            if jangPressOn { JangPressActivation.deactivate(jangPressRuntime) }
        }

        // BENCH_VL=1 dispatches to VLBench: loads the real HF tokenizer +
        // synthesises a 224×224 image, runs two turns through the VLM
        // processor + cache to verify vision path + multi-turn cache reuse.
        // CAVEAT: the standard BENCH_VL path uses `TokenIterator`, not
        // BatchEngine. Use BENCH_VL_BATCH_CHAT to verify the actual
        // BatchEngine VL path.
        if (env["BENCH_VL"] ?? "0") == "1" {
            try await VLBench.run(modelPath: modelPath, maxNewTokens: maxNew)
            return
        }

        // BENCH_VL_BATCH_CHAT=1 runs VL multi-turn DIRECTLY through
        // `BatchEngine.generate(...)`. This is the honest VL-through-
        // BatchEngine verification — iter 29 audit flagged that prior
        // BENCH_VL only exercises `TokenIterator`. Added 2026-04-19 (iter 30).
        if (env["BENCH_VL_BATCH_CHAT"] ?? "0") == "1" {
            try await VLBench.runBatch(modelPath: modelPath, maxNewTokens: maxNew)
            return
        }

        // BENCH_VL_MIXED=1 (2026-04-22): single model + single BatchEngine
        // + shared CacheCoordinator, four turns with different variables
        // flipped each turn (thinking on/off, text/image/video modality).
        // Validates SSM seed + stepBatchDecode force-unwrap fix +
        // ReasoningParser.forPrompt tail detection, all in one run.
        //   BENCH_VIDEO=/path/to/file.mov overrides the default video.
        if (env["BENCH_VL_MIXED"] ?? "0") == "1" {
            let videoPath = env["BENCH_VIDEO"]
                ?? "Tests/MLXLMTests/Resources/1080p_30.mov"
            try await VLBench.runMixedMultiTurn(
                modelPath: modelPath, videoPath: videoPath,
                maxNewTokens: maxNew)
            return
        }

        // BENCH_VL_BATCH_VIDEO=1 (2026-04-22): end-to-end VL video ingest
        // through `context.processor.prepare(input:)` with
        // `UserInput(prompt:, videos: [.url(...)])`, then drive multi-
        // turn through BatchEngine.generate. Set
        //   BENCH_VIDEO=/path/to/file.mov
        // to override the default test fixture.
        if (env["BENCH_VL_BATCH_VIDEO"] ?? "0") == "1" {
            let videoPath = env["BENCH_VIDEO"]
                ?? "Tests/MLXLMTests/Resources/1080p_30.mov"
            try await VLBench.runBatchVideo(
                modelPath: modelPath, videoPath: videoPath,
                maxNewTokens: maxNew)
            return
        }

        // BENCH_VL_BATCH_MEDIASALT=1 (iter 37): verify VL cache isolation
        // via `mediaSalt`. Submit prompt P with image A, then the same
        // P with image A (must HIT), then the same P with image B (must
        // MISS because SHA256 of image bytes differs). Catches the
        // cache-poisoning bug class where two different images with the
        // same text prompt would return each other's cached KV state.
        if (env["BENCH_VL_BATCH_MEDIASALT"] ?? "0") == "1" {
            try await VLBench.runBatchMediaSalt(modelPath: modelPath, maxNewTokens: maxNew)
            return
        }

        // BENCH_BATCH_TOOLCALL=1 (iter 66): submit a tool-bearing prompt
        // through `BatchEngine.generate(...)` and assert the pipeline
        // (ReasoningParser → ToolCallProcessor) behaves correctly on a
        // real model:
        //   - `.chunk(String)` text MUST NOT contain raw tool-call
        //     markers (`<tool_call>`, `<|tool_call>`, `call:<name>`,
        //     `[TOOL_CALLS]`, etc.) — if it does, the library failed
        //     to extract the call and osaurus would have to re-parse.
        //   - `.chunk(String)` text MUST NOT contain raw `<think>` /
        //     `</think>` markers — if it does, the reasoning parser
        //     never engaged.
        //   - a `.toolCall` event MUST be emitted for this strict
        //     tool-only prompt. Set BENCH_BATCH_TOOLCALL_RAW=1 to dump
        //     raw token IDs/decoded text when a family-specific parser
        //     needs diagnosis.
        if (env["BENCH_BATCH_TOOLCALL"] ?? "0") == "1" {
            try await runBatchEngineToolCall(modelPath: modelPath, maxNew: maxNew)
            return
        }

        // BENCH_PERF=1 (2026-04-21): deterministic tok/s micro-bench for
        // the perf-regression hunt. Prints one grep-able line per run:
        //   PERF model=<name> variant=<label> genTokens=N genSec=F tokps=F
        // Reads env:
        //   BENCH_PERF_PROMPT  — CoT-free prompt text (default below)
        //   BENCH_MAX_TOKENS   — fixed decode budget (default 128)
        //   BENCH_PERF_VARIANT — label emitted in the output
        //   BENCH_PERF_WARMUP  — run N warmup turns first (default 1)
        //   BENCH_PERF_RUNS    — measurement turns, picks median (default 3)
        //   BENCH_PERF_SEED    — deterministic stochastic-sampler seed
        //   BENCH_PERF_USE_GENERATION_CONFIG=1 — seed sampling from the
        //     bundle's generation_config.json before explicit env overrides.
        if (env["BENCH_PERF"] ?? "0") == "1" {
            try await runPerfBench(
                modelPath: modelPath, maxNew: maxNew,
                variant: env["BENCH_PERF_VARIANT"] ?? "auto",
                warmup: Int(env["BENCH_PERF_WARMUP"] ?? "1") ?? 1,
                runs: Int(env["BENCH_PERF_RUNS"] ?? "3") ?? 3,
                useTokenIterator:
                    (env["BENCH_PERF_PATH"] ?? "batch") == "iter")
            return
        }

        // BENCH_HARMONY_CHECK=1: real-model verification of the 2026-04-20
        // harmony-reasoning fix. Loads a Gemma-4 model, sends a short
        // prompt, asserts at least one .reasoning delta fires AND .chunk
        // contains zero harmony markers.
        if (env["BENCH_HARMONY_CHECK"] ?? "0") == "1" {
            try await runHarmonyReasoningCheck(
                modelPath: modelPath, maxNew: maxNew)
            return
        }

        // BENCH_QWEN_THINKING_CHECK=1: real-model verification of the
        // Qwen3.6 prefilled-<think> fix. Loads a Qwen 3.x model with
        // enable_thinking=true, asserts at least one .reasoning delta
        // fires AND .chunk contains zero <think> markers.
        if (env["BENCH_QWEN_THINKING_CHECK"] ?? "0") == "1" {
            try await runQwenThinkingReasoningCheck(
                modelPath: modelPath, maxNew: maxNew)
            return
        }

        // BENCH_CONFIG_SMOKE=1: metadata-only model audit. Verifies
        // model_type, layers, generation_config EOS, tokenizer EOS,
        // JANG/JANGTQ bit metadata, sidecar presence, and safetensors
        // index layout without loading weights.
        if (env["BENCH_CONFIG_SMOKE"] ?? "0") == "1" {
            try await runConfigSmoke(modelPath: modelPath)
            return
        }

        // BENCH_ZAYA_CONTRACT=1: metadata/tokenizer/cache-contract gate for
        // Zyphra ZAYA bundles. This intentionally does not load weights until
        // the CCA model port lands; it verifies the bundle gives the runtime
        // enough exact signals to reject or route safely.
        if (env["BENCH_ZAYA_CONTRACT"] ?? "0") == "1" {
            try await runZayaContract(modelPath: modelPath)
            return
        }

        // BENCH_ZAYA_TOPK=1: load the real ZAYA model and print next-token
        // top-k logits for a raw or chat-rendered prompt. This is a parity
        // probe for the CCA/router port, not a throughput benchmark.
        if (env["BENCH_ZAYA_TOPK"] ?? "0") == "1" {
            try await runZayaTopK(modelPath: modelPath)
            return
        }

        // BENCH_ZAYA_MOE_BITS=1: load the real ZAYA/ZAYA1-VL model and
        // print the resolved JANGTQ gate/up/down bit widths from both
        // decoded configuration and reflected SwitchGLU modules. This is
        // diagnostics only; it does not alter generation.
        if (env["BENCH_ZAYA_MOE_BITS"] ?? "0") == "1" {
            try await runZayaMoEBits(modelPath: modelPath)
            return
        }

        // BENCH_ZAYA_TQ_KERNEL_PROBE=1: diagnostic-only check for ZAYA
        // JANGTQ_K. It loads one real layer's packed JANGTQ tensors and
        // compares the Metal kernels against a CPU dequant reference. This
        // never alters model generation; it only distinguishes a kernel/runtime
        // bug from a legitimately bad packed artifact.
        if (env["BENCH_ZAYA_TQ_KERNEL_PROBE"] ?? "0") == "1" {
            try await runZayaTQKernelProbe(modelPath: modelPath)
            return
        }

        // BENCH_TEMPLATE_SMOKE=1: tokenizer-only chat-template/Jinja
        // smoke for production model bundles. It exercises plain,
        // thinking on/off, reasoning_effort=max, tools, and multi-turn
        // prompt rendering without loading model weights.
        if (env["BENCH_TEMPLATE_SMOKE"] ?? "0") == "1" {
            try await runTemplateSmoke(modelPath: modelPath)
            return
        }

        // BENCH_DSV4_TEMPLATE_KWARGS=1: verify the DSV4 tokenizer template
        // or Swift fallback threads `enable_thinking` (bool) and
        // `reasoning_effort` ('max'/None) kwargs through the upstream
        // applyChatTemplate → additionalContext path. Without this
        // working, callers can't switch between chat/thinking modes
        // or engage max-effort preface on DSV4 bundles.
        if (env["BENCH_DSV4_TEMPLATE_KWARGS"] ?? "0") == "1" {
            try await runDSV4TemplateKwargsCheck(modelPath: modelPath)
            return
        }

        // BENCH_DSV4_COHERENCE=1: production DSV4 chat coherence gate.
        // Loads once and drives the real BatchEngine + UserInput(chat:)
        // path across:
        //   1. 3-turn multi-turn fact recall with thinking disabled.
        //   2. reasoning off / on / max-effort routing checks.
        //   3. long-context recall of an early buried fact after a
        //      large filler body. This is semantic, not just no-crash.
        if (env["BENCH_DSV4_COHERENCE"] ?? "0") == "1" {
            try await runDSV4CoherenceGate(modelPath: modelPath, maxNew: maxNew)
            return
        }

        // BENCH_DSV4_AGENTIC_TOOL=1: DSV4-specific structured tool row.
        // It proves DSML parser/tool autodetection, bundle-derived sampler
        // defaults, multi-turn tool history, and L2 disk cache reuse.
        if (env["BENCH_DSV4_AGENTIC_TOOL"] ?? "0") == "1" {
            try await runDSV4AgenticToolCheck(modelPath: modelPath, maxNew: maxNew)
            return
        }

        // BENCH_REASONING_TURN_MATRIX=1: generic BatchEngine multi-turn
        // reasoning gate. It keeps one loaded model, appends assistant
        // reasoning_content back into the transcript, and verifies that
        // thinking OFF stays visible-only while thinking ON/effort modes do
        // not leak raw control tokens into .chunk output.
        if (env["BENCH_REASONING_TURN_MATRIX"] ?? "0") == "1" {
            try await runReasoningTurnMatrix(modelPath: modelPath, maxNew: maxNew)
            return
        }

        // BENCH_ORPHAN_SLOT_REPRO=1: reproduce the consumer-cancellation
        // → orphan-slot → next-request-Metal-collision pattern reported
        // 2026-04-27. Submits request A, breaks the consumer loop after
        // 4 tokens (simulates osaurus's `Task.isCancelled break`), then
        // immediately submits request B with the same prompt (would
        // normally hit cache and collide with the orphan-A pipelines).
        // Pre-fix: clear_library assertion mid-prefill of request B.
        // Post-fix: continuation.onTermination on A reaps the slot
        // before B starts, both complete cleanly.
        if (env["BENCH_ORPHAN_SLOT_REPRO"] ?? "0") == "1" {
            try await runOrphanSlotRepro(modelPath: modelPath, maxNew: maxNew)
            return
        }

        // BENCH_THINK_LOOP_PROBE=1: validation-style prompt to test
        // whether a reasoning model emits </think> within budget or
        // gets stuck in self-refinement loops. Uses the bundle's
        // generation_config.json by default so MiniMax / ZAYA / Hy3
        // probes do not accidentally run Qwen-style sampling.
        if (env["BENCH_THINK_LOOP_PROBE"] ?? "0") == "1" {
            try await runThinkingLoopProbe(
                modelPath: modelPath, maxNew: maxNew)
            return
        }

        // BENCH_LAGUNA_LOOP=1: targeted probe for the Osaurus v0.18.7
        // Laguna XS.2 report where thinking-off repeats the same file
        // tree summary forever and thinking-on leaves the answer inside
        // the reasoning stream. Uses the production BatchEngine +
        // chat-template path, not a raw prompt.
        if (env["BENCH_LAGUNA_LOOP"] ?? "0") == "1" {
            try await runLagunaLoopProbe(
                modelPath: modelPath, maxNew: maxNew)
            return
        }

        // BENCH_DSV4_FIM_VS_CHAT=1: side-by-side coherence probe across
        // DSV4's three prompt modes on the same simple factual prompt.
        // Decodes 64 tokens each and prints raw output so a human can
        // judge "does it actually answer the question."
        //
        //   1. FIM (raw): no chat template, just "The capital of France is"
        //   2. CHAT no-think: applyChatTemplate, enable_thinking=false
        //   3. CHAT think: applyChatTemplate, enable_thinking=true
        //
        // Background: HumanEval+ FIM mode pass@1 was previously
        // measured at 67% on JANGTQ_2L. Long-trace chat conversations
        // showed drift past sliding_window=128. This bench answers
        // whether chat template short-output (~64 tokens) is coherent.
        if (env["BENCH_DSV4_FIM_VS_CHAT"] ?? "0") == "1" {
            try await runDSV4FIMvsChat(modelPath: modelPath, maxNew: maxNew)
            return
        }

        // BENCH_QWEN_MULTITURN_TOOL=1: mirrors tpae's 2026-04-20 3:02 /
        // 3:04 PM screenshots — Qwen3.6, first turn asks "create README
        // for my game", turn 2 pretends a file_read tool returned game
        // source, turn 3 asks for a second tool. Asserts ZERO <think>
        // markers in `.chunk` across all 3 turns — the EXACT bug tpae
        // screenshotted.
        if (env["BENCH_QWEN_MULTITURN_TOOL"] ?? "0") == "1" {
            try await runQwenMultiturnToolCheck(
                modelPath: modelPath, maxNew: maxNew)
            return
        }
        // BENCH_OMNI=1 (2026-04-28): full multi-turn matrix for
        // Nemotron-3-Nano-Omni bundles. Tests text-only single, text
        // multi-turn, image single, image multi-turn, video, audio
        // (manual splice), and reasoning toggle in one harness.
        // BENCH_MODEL=/path/to/Nemotron-3-Nano-Omni-30B-A3B-{MXFP4|JANGTQ4|JANGTQ2}
        if (env["BENCH_OMNI"] ?? "0") == "1" {
            try await OmniBench.run(modelPath: modelPath, maxNewTokens: maxNew)
            return
        }

        // BENCH_STABILITY=1 (2026-04-30): exhaustive stability matrix
        // covering the failure modes that have been blocking releases —
        // warm L2 disk-cache 2nd-request, over-cap hybrid prompt,
        // multi-turn agent loop, cancel + recovery, concurrent batched
        // decode, TQ KV mode + disk round-trip, clearCache mid-run,
        // hybrid SSM disk round-trip. Drives BatchEngine directly via
        // a single `ModelContext`. Designed for any hybrid / VLM
        // bundle. No HTTP layer; runs from this binary directly.
        if (env["BENCH_STABILITY"] ?? "0") == "1" {
            try await StabilityBench.run(modelPath: modelPath, maxNewTokens: maxNew)
            return
        }

        // BENCH_JPREG=1 (2026-05-03): per-bundle regression sweep that
        // exercises the new typed `LoadConfiguration` path end-to-end.
        // Loads with `LoadConfiguration.default` (JangPress disabled +
        // 70% caps + mmap), runs 3-turn coherency, samples RSS at four
        // checkpoints, and exercises hybrid-SSM warm-pass plus
        // TurboQuant disk round-trip checks where applicable.
        // Designed to be invoked once per bundle from a shell loop so
        // each model gets a fresh process.
        if (env["BENCH_JPREG"] ?? "0") == "1" {
            try await JangPressRegressionBench.run(
                modelPath: modelPath, maxNewTokens: maxNew)
            return
        }

        // Hoist single-load scenarios above the preamble load so they
        // don't double-allocate the model. Critical for huge bundles
        // (DSV4-Flash JANGTQ at 79.5 GB OOMs with two simultaneous
        // copies on a 128 GB host). Each of these scenarios does its
        // own load via runBatchEngine* which uses the real HF
        // tokenizer — the preamble's NullTokenizerLoader copy isn't
        // needed for any of them.
        if (env["BENCH_BATCH_CACHE_HIT"] ?? "0") == "1" {
            try await runBatchEngineCacheHit(modelPath: modelPath, maxNew: maxNew)
            return
        }
        if (env["BENCH_BATCH_DISK_RESTORE"] ?? "0") == "1" {
            try await runBatchEngineDiskRestore(modelPath: modelPath, maxNew: maxNew)
            return
        }
        if (env["BENCH_GROWING_CHAT_CACHE"] ?? "0") == "1" {
            do {
                try await runGrowingChatCacheReuse(modelPath: modelPath, maxNew: maxNew)
            } catch {
                print("[BENCH_GROWING_CHAT_CACHE] error: \(String(reflecting: error))")
                exit(benchFailureExitCode(for: error))
            }
            return
        }
        if (env["BENCH_BATCH_TQ_B2"] ?? "0") == "1" {
            try await runBatchEngineTurboQuantB2(modelPath: modelPath, maxNew: maxNew)
            return
        }
        if (env["BENCH_COHERENT"] ?? "0") == "1" {
            try await runCoherentMultiTurn(modelPath: modelPath, maxNew: maxNew)
            return
        }
        if (env["BENCH_BATCH_CHAT"] ?? "0") == "1" {
            try await runBatchEngineMultiTurn(modelPath: modelPath, maxNew: maxNew)
            return
        }
        if (env["BENCH_CROSS_VALIDATE"] ?? "0") == "1" {
            try await runCrossEngineValidation(modelPath: modelPath, maxNew: maxNew)
            return
        }
        if (env["BENCH_BATCH_CONCURRENT"] ?? "0") == "1" {
            try await runBatchEngineConcurrent(modelPath: modelPath, maxNew: maxNew)
            return
        }
        if (env["BENCH_BATCH_PERSLOT_SAMPLER"] ?? "0") == "1" {
            try await runBatchEnginePerSlotSampler(modelPath: modelPath, maxNew: maxNew)
            return
        }
        if (env["BENCH_BATCH_B4"] ?? "0") == "1" {
            let b = Int(env["BENCH_B_SIZE"] ?? "4") ?? 4
            try await runBatchEngineBMany(
                modelPath: modelPath, maxNew: maxNew, batchSize: b)
            return
        }
        if (env["BENCH_BATCH_CANCEL"] ?? "0") == "1" {
            try await runBatchEngineCancelMidStream(
                modelPath: modelPath, maxNew: maxNew)
            return
        }
        if (env["BENCH_CRASH_FUZZ"] ?? "0") == "1" {
            try await runCrashFuzz(modelPath: modelPath, maxNew: maxNew)
            return
        }
        if (env["BENCH_CRASH_FUZZ_V2"] ?? "0") == "1" {
            try await runCrashFuzzV2(modelPath: modelPath, maxNew: maxNew)
            return
        }
        if (env["BENCH_OFFICIAL"] ?? "0") == "1" {
            try await runOfficialMultiTurn(modelPath: modelPath, maxNew: maxNew)
            return
        }
        if (env["BENCH_NO_GUARD_SAMPLING"] ?? "0") == "1" {
            try await runNoGuardSamplingProbe(modelPath: modelPath, maxNew: maxNew)
            return
        }
        if (env["BENCH_PROD"] ?? "0") == "1" {
            try await runProdMatrix(modelPath: modelPath, maxNew: maxNew)
            return
        }
        if (env["BENCH_BATCH_LONG_CONTEXT"] ?? "0") == "1" {
            let len = Int(env["BENCH_LONG_LEN"] ?? "2048") ?? 2048
            try await runBatchEngineLongContext(
                modelPath: modelPath, maxNew: maxNew, promptLen: len)
            return
        }
        if (env["BENCH_BATCH_SPECDEC"] ?? "0") == "1" {
            let drafter = env["BENCH_SPECDEC_DRAFTER"]
                ?? "/tmp/ddtree-downloads/Qwen3.5-27B-DFlash"
            try await runBatchSpecDec(
                modelPath: modelPath,
                drafterPath: drafter,
                maxNew: maxNew)
            return
        }
        if (env["BENCH_VL_CROSS_VALIDATE"] ?? "0") == "1" {
            try await VLBench.runCrossValidate(
                modelPath: modelPath, maxNewTokens: maxNew)
            return
        }
        if (env["BENCH_VL_BATCH_CACHE_HIT"] ?? "0") == "1" {
            try await VLBench.runBatchCacheHit(
                modelPath: modelPath, maxNewTokens: maxNew)
            return
        }
        if (env["BENCH_VL_CHAT_CACHE"] ?? "0") == "1" {
            try await VLBench.runChatCacheMatrix(
                modelPath: modelPath, maxNewTokens: maxNew)
            return
        }
        if (env["BENCH_VL_VIDEO"] ?? "0") == "1" {
            let videoPath = env["BENCH_VIDEO_PATH"]
                ?? repoPath("Tests/MLXLMTests/Resources/1080p_30.mov")
            try await VLBench.runVideoSmoke(
                modelPath: modelPath,
                videoPath: videoPath,
                maxNewTokens: maxNew)
            return
        }

        print("=== vmlx-swift-lm — \(modelDir.lastPathComponent) MULTI-TURN ===")
        print("Tokens: \(tokensPath)")
        print("Loading...")

        let loadStart = CFAbsoluteTimeGetCurrent()
        // Use general loader — picks LLM or VLM factory based on model_type
        let context = try await MLXLMCommon.loadModel(from: modelDir, using: NullTokenizerLoader())
        print(String(format: "Load: %.2fs", CFAbsoluteTimeGetCurrent() - loadStart))
        print("Model: \(type(of: context.model))")

        // BENCH_BATCH=1 runs the BatchEngine smoke: single request via
        // BatchEngine, then BatchEngine + compile-on, then BatchEngine
        // + TurboQuant, then B=2 concurrent. Verifies end-to-end
        // behaviour on real weights. Added 2026-04-18 (iter 17).
        if (env["BENCH_BATCH"] ?? "0") == "1" {
            try await runBatchSmoke(context: context, maxNew: maxNew)
            return
        }

        // BENCH_COHERENT=1 runs a real multi-turn conversation through
        // BatchEngine with the actual HF tokenizer so we can visually
        // verify coherent text output across 3 turns with cache reuse.
        // Added 2026-04-18 (iter 19) — the user has repeatedly asked
        // for actual coherence testing, not just synthetic-prompt
        // tok/s measurements.
        //
        // CAVEAT (iter 26): BENCH_COHERENT uses ChatSession which
        // internally uses `TokenIterator`, not `BatchEngine`. For
        // TRUE BatchEngine multi-turn verification use BENCH_BATCH_CHAT.
        if (env["BENCH_COHERENT"] ?? "0") == "1" {
            try await runCoherentMultiTurn(modelPath: modelPath, maxNew: maxNew)
            return
        }

        // BENCH_BATCH_CHAT=1 runs a real multi-turn conversation
        // DIRECTLY through BatchEngine (not ChatSession). Uses the real
        // HF tokenizer + chat template + CacheCoordinator for cross-
        // turn reuse. This is the honest BatchEngine-multi-turn test
        // per the spec §6 multi-turn acceptance path. Added 2026-04-19.
        if (env["BENCH_BATCH_CHAT"] ?? "0") == "1" {
            try await runBatchEngineMultiTurn(modelPath: modelPath, maxNew: maxNew)
            return
        }

        // BENCH_CROSS_VALIDATE=1 (iter 32): run the same prompt through
        // `TokenIterator` AND `BatchEngine.generate(...)` with temp=0 and
        // assert the emitted token IDs match byte-for-byte. This is the
        // strongest single correctness property for the engine — compile
        // on/off identity is already checked by `BENCH_BATCH_CHAT`, but
        // equality with the long-standing single-seq path was only
        // assumed until this bench existed.
        if (env["BENCH_CROSS_VALIDATE"] ?? "0") == "1" {
            try await runCrossEngineValidation(modelPath: modelPath, maxNew: maxNew)
            return
        }

        // BENCH_BATCH_CONCURRENT=1 (iter 33): TWO different prompts
        // submitted to BatchEngine maxBatchSize=2 and iterated
        // CONCURRENTLY (TaskGroup). Exercises the actual batched-decode
        // hot path — unlike synthetic BENCH_BATCH which iterates the
        // streams sequentially, or BENCH_BATCH_CHAT which uses B=1.
        // Verifies both streams complete with coherent output and that
        // both slots finish under EOS/max-tokens. Uses real HF tokenizer.
        if (env["BENCH_BATCH_CONCURRENT"] ?? "0") == "1" {
            try await runBatchEngineConcurrent(modelPath: modelPath, maxNew: maxNew)
            return
        }

        // BENCH_BATCH_CACHE_HIT=1 (iter 34): demonstrate CacheCoordinator
        // cross-turn prefix reuse through BatchEngine. Submits two turns
        // where turn 2 extends turn 1's prompt; asserts turn 2's prompt
        // time is meaningfully lower (cache hit on the shared prefix).
        if (env["BENCH_BATCH_CACHE_HIT"] ?? "0") == "1" {
            try await runBatchEngineCacheHit(modelPath: modelPath, maxNew: maxNew)
            return
        }

        // BENCH_BATCH_DISK_RESTORE=1 (iter 35): verify L2 disk cache
        // round-trips through BatchEngine. Turn 1 stores via finishSlot;
        // the coordinator is then DROPPED and recreated fresh against the
        // same disk dir — modelling an osaurus session restart. Turn 2
        // must still hit and skip prefill. This is the strongest "session
        // persistence across runs" property and the one osaurus relies on.
        if (env["BENCH_BATCH_DISK_RESTORE"] ?? "0") == "1" {
            try await runBatchEngineDiskRestore(modelPath: modelPath, maxNew: maxNew)
            return
        }
        if (env["BENCH_GROWING_CHAT_CACHE"] ?? "0") == "1" {
            do {
                try await runGrowingChatCacheReuse(modelPath: modelPath, maxNew: maxNew)
            } catch {
                print("[BENCH_GROWING_CHAT_CACHE] error: \(String(reflecting: error))")
                exit(benchFailureExitCode(for: error))
            }
            return
        }

        // BENCH_BATCH_PERSLOT_SAMPLER=1 (iter 36): submit two slots with
        // DIFFERENT sampling params into the same B=2 engine. Slot 0
        // temp=0 greedy (deterministic, re-runnable byte-identical).
        // Slot 1 temp=0.8 topP=0.9 (stochastic). Must prove each slot's
        // GenerateParameters flows through to its own sampler — osaurus
        // spec explicitly calls this out.
        if (env["BENCH_BATCH_PERSLOT_SAMPLER"] ?? "0") == "1" {
            try await runBatchEnginePerSlotSampler(modelPath: modelPath, maxNew: maxNew)
            return
        }

        // BENCH_BATCH_TQ_B2=1 (iter 38): concurrent B=2 with heterogeneous
        // kvMode. Slot 0 plain KV, slot 1 turboQuant(3,3). Verifies Stage 0
        // compression per-slot without cross-slot corruption, plus both
        // streams complete with coherent output.
        if (env["BENCH_BATCH_TQ_B2"] ?? "0") == "1" {
            try await runBatchEngineTurboQuantB2(modelPath: modelPath, maxNew: maxNew)
            return
        }

        // BENCH_BATCH_B4=1 (iter 39): four concurrent distinct prompts
        // submitted into `maxBatchSize=4`. Osaurus ships max=4 as the
        // default `mlxBatchEngineMaxBatchSize` — this must work end-to-end
        // with real HF tokenizer. Asserts all four slots complete with
        // coherent non-empty output AND no cross-slot mixing (slot 0's
        // tokens identical to a solo run of the same prompt).
        // Set BENCH_B_SIZE=8 (or other) to stress higher fan-out.
        if (env["BENCH_BATCH_B4"] ?? "0") == "1" {
            let b = Int(env["BENCH_B_SIZE"] ?? "4") ?? 4
            try await runBatchEngineBMany(
                modelPath: modelPath, maxNew: maxNew, batchSize: b)
            return
        }

        // BENCH_BATCH_CANCEL=1 (iter 40): cancel mid-stream under B=3.
        // One slot cancelled after a few tokens; surviving slots must
        // decode to max-tokens. Verifies the `.cancelled` info event
        // and that engine state recovery is clean.
        if (env["BENCH_BATCH_CANCEL"] ?? "0") == "1" {
            try await runBatchEngineCancelMidStream(
                modelPath: modelPath, maxNew: maxNew)
            return
        }

        // BENCH_CRASH_FUZZ=1 (2026-04-23): osaurus-style adversarial fuzz
        // for the tpae Qwen 3.6 27B crash report. One model load, many
        // scenarios. Each scenario prints "SCENARIO N: <name>" up-front
        // so when we crash the last line tells us where. Covers:
        //   1. B=1 baseline
        //   2. B=4 concurrent distinct prompts
        //   3. B=4 + cancellation mid-stream on two slots
        //   4. maxTokens=1 on every slot (near EOS-on-first-token)
        //   5. Single-token prompt
        //   6. Rapid submit + immediate consumer drop (no iteration)
        //   7. Same prompt submitted twice back-to-back (cache contention)
        //   8. B=4 with stop-string that matches immediately
        //   9. B=4 with wildly different lengths (short + long)
        //  10. 5 rapid sequential single-turn submits (connection churn)
        if (env["BENCH_CRASH_FUZZ"] ?? "0") == "1" {
            try await runCrashFuzz(modelPath: modelPath, maxNew: maxNew)
            return
        }

        // BENCH_CRASH_FUZZ_V2=1 (2026-04-23): adversarial multi-turn
        // stress through the FULL `generate()` pipeline (NOT `submit()`
        // like v1). Runs against the real chat template, reasoning
        // parser, tool-call processor, stop-string matcher, and
        // NaiveStreamingDetokenizer on every turn. Covers specifically
        // the tokenizer-decode-shrinkage class of bugs (cleanup
        // substitutions, byte-level BPE emoji completion, adjacent
        // special-token collapse) by asking the model for output that
        // is likely to trigger each pattern.
        if (env["BENCH_CRASH_FUZZ_V2"] ?? "0") == "1" {
            try await runCrashFuzzV2(modelPath: modelPath, maxNew: maxNew)
            return
        }

        // BENCH_OFFICIAL=1 (2026-04-23): final-pass multi-turn harness.
        // For a single model, runs a 6-scenario matrix and reports
        // per-turn TTFT, decode tok/s, reasoning/chunk/tool-call counts,
        // peak RSS, and response-content validation (where applicable).
        // Meant to be invoked per model via shell loop so each model
        // gets its own process (clean GPU memory baseline).
        if (env["BENCH_OFFICIAL"] ?? "0") == "1" {
            try await runOfficialMultiTurn(modelPath: modelPath, maxNew: maxNew)
            return
        }

        // BENCH_PROD=1 (2026-04-23): EXHAUSTIVE production matrix.
        // Expands BENCH_OFFICIAL with:
        //   - Multi-turn tool-call ROUND TRIP (assistant emits tool_call
        //     with valid name+args, we inject a fake tool response, model
        //     continues)
        //   - Reasoning ON→OFF→ON alternation on a single engine
        //   - L2 disk cache: second identical turn hits the disk cache
        //     (not just paged); cache directory is explicitly configured
        //   - SSM state re-derive: on hybrid SSM models, second turn
        //     shares a prompt prefix — prefix hit + SSM seed should
        //     speed up prefill ≥2×
        //   - TurboQuant load + forward: on a JANGTQ bundle, verify
        //     model loads with sidecar and produces tokens
        // Validates content per scenario — math contains the answer,
        // factual contains the expected word, tool-call schema name matches.
        if (env["BENCH_PROD"] ?? "0") == "1" {
            try await runProdMatrix(modelPath: modelPath, maxNew: maxNew)
            return
        }

        // BENCH_BATCH_LONG_CONTEXT=1 (iter 42): submit a 2000+ token
        // prompt single-slot through BatchEngine AND through TokenIterator;
        // assert byte-identical token output. Exercises chunked prefill
        // (multi-pass of `prefillStepSize`=512 over ~2k tokens), memory
        // purge during long decode (`memoryPurgeInterval`=256), and
        // sliding-window interaction near the model's cache budget.
        // Tune prompt length via BENCH_LONG_LEN (default 2048).
        if (env["BENCH_BATCH_LONG_CONTEXT"] ?? "0") == "1" {
            let len = Int(env["BENCH_LONG_LEN"] ?? "2048") ?? 2048
            try await runBatchEngineLongContext(
                modelPath: modelPath, maxNew: maxNew, promptLen: len)
            return
        }

        // BENCH_BATCH_SPECDEC=1 (iter 16): run the same prompt under
        // (a) plain generate, (b) DFlash linear, (c) DDTree tree-verify,
        // all against the same target model. Report per-path tok/s +
        // byte-parity vs plain at temperature 0. Drafter path comes
        // from env var BENCH_SPECDEC_DRAFTER.
        if (env["BENCH_BATCH_SPECDEC"] ?? "0") == "1" {
            let drafter = env["BENCH_SPECDEC_DRAFTER"]
                ?? "/tmp/ddtree-downloads/Qwen3.5-27B-DFlash"
            try await runBatchSpecDec(
                modelPath: modelPath,
                drafterPath: drafter,
                maxNew: maxNew)
            return
        }

        // BENCH_VL_CROSS_VALIDATE=1 (iter 47): run the same VL prompt
        // (text + image) through TokenIterator AND BatchEngine, then
        // assert byte-identical token output. Extends iter 32/44's
        // cross-engine validation from dense/hybrid text to vision path.
        // Depends on iter 45's UserInput fix so images actually reach
        // the processor.
        if (env["BENCH_VL_CROSS_VALIDATE"] ?? "0") == "1" {
            try await VLBench.runCrossValidate(modelPath: modelPath, maxNewTokens: maxNew)
            return
        }

        // BENCH_VL_BATCH_CACHE_HIT=1 (iter 48): VL multi-turn cache
        // reuse end-to-end. Turn 1: describe image. Turn 2: extend the
        // prompt with a follow-up question about the same image. Asserts
        // turn 2 produces a paged HIT on (tokens, mediaSalt) — "user
        // asks another question about the same photo" scenario.
        if (env["BENCH_VL_BATCH_CACHE_HIT"] ?? "0") == "1" {
            try await VLBench.runBatchCacheHit(modelPath: modelPath, maxNewTokens: maxNew)
            return
        }

        // BENCH_VL_CHAT_CACHE=1: structured-chat VL cache matrix.
        // Uses UserInput(chat:) with an image-bearing first turn and a
        // text follow-up, then verifies same-media replay hits and
        // different-media replay misses through CacheCoordinator.
        if (env["BENCH_VL_CHAT_CACHE"] ?? "0") == "1" {
            try await VLBench.runChatCacheMatrix(modelPath: modelPath, maxNewTokens: maxNew)
            return
        }

        // BENCH_VL_VIDEO=1 (iter 49): video input end-to-end. Loads a
        // short .mov via AVFoundation, processes frames through the
        // VLM processor, runs the vision tower on the frame sequence,
        // decodes text. Path: UserInput(videos:) → processor.prepare
        // → LMInput.video tensor → model forward.
        if (env["BENCH_VL_VIDEO"] ?? "0") == "1" {
            let videoPath = env["BENCH_VIDEO_PATH"]
                ?? repoPath("Tests/MLXLMTests/Resources/1080p_30.mov")
            try await VLBench.runVideoSmoke(
                modelPath: modelPath,
                videoPath: videoPath,
                maxNewTokens: maxNew)
            return
        }

        // Simple load-and-generate mode: when BENCH_SIMPLE=1, skip multi-turn
        // tokens and just generate N tokens from a short static prompt. Used
        // to smoke-test new model paths (e.g. .jangspec bundles, JANGTQ).
        // BENCH_COORDINATOR=1 installs a CacheCoordinator matching Osaurus's
        // config (isHybrid=true disk=true maxBlocks=2000) to reproduce its
        // prefill / cache-miss / Metal-concurrency path.
        // BENCH_PROMPT_LEN=1394 seeds a deterministic long prompt to match
        // the exact token count from the user's crash report.
        if (env["BENCH_SIMPLE"] ?? "0") == "1" {
            let promptLen = Int(env["BENCH_PROMPT_LEN"] ?? "10") ?? 10
            let seedTokens: [Int32] = (0..<promptLen).map { Int32($0 % 4096 + 1) }
            let simpleInput = LMInput(text: LMInput.Text(
                tokens: MLXArray(seedTokens)[.newAxis, .ellipsis]))
            var sp = GenerateParameters(maxTokens: maxNew)
            sp.temperature = 0.0
            sp.prefillStepSize = Int(env["BENCH_PREFILL_STEP"] ?? "512") ?? 512
            sp.enableCompiledDecode = compileDecode
            sp.compiledMaxCacheLength = compileMaxLen
            let sCache = context.model.newCache(parameters: sp)
            let sCoord: CacheCoordinator?
            if (env["BENCH_COORDINATOR"] ?? "0") == "1" {
                var cfg = CacheCoordinatorConfig()
                cfg.usePagedCache = true
                cfg.maxCacheBlocks = 2000
                cfg.enableDiskCache = true
                cfg.diskCacheDir = URL(fileURLWithPath: "/tmp/bench_disk_cache")
                let c = CacheCoordinator(config: cfg)
                let isHybrid = sCache.contains { !($0 is KVCacheSimple) && !($0 is RotatingKVCache) }
                c.setHybrid(isHybrid)
                sCoord = c
                print("[Coord] isHybrid=\(isHybrid) disk=true maxBlocks=2000")
            } else {
                sCoord = nil
            }
            // Warmup forward pass: triggers lazy module initializations
            // (e.g. SwitchGLU fused gate+up cache) so the timed runs below
            // don't pay that one-time concatenation cost in TTFT. Use a
            // small prompt + 1 token so the warmup is fast.
            if (env["BENCH_SKIP_WARMUP"] ?? "0") != "1" {
                let warmSeed: [Int32] = [1, 2, 3, 4, 5]
                let warmInput = LMInput(text: LMInput.Text(
                    tokens: MLXArray(warmSeed)[.newAxis, .ellipsis]))
                var wParams = GenerateParameters(maxTokens: 1)
                wParams.temperature = 0.0
                wParams.prefillStepSize = 512
                let warmCache = context.model.newCache(parameters: wParams)
                var warmIter = try TokenIterator(
                    input: warmInput, model: context.model,
                    cache: warmCache, parameters: wParams)
                _ = warmIter.next()
            }

            // BENCH_RUNS=N runs the same prompt N times. With a coordinator
            // installed the first run is a cold fetch (miss → full prefill →
            // store on completion), subsequent runs should hit the paged
            // cache for the prefix and only prefill the remainder.
            let runs = Int(env["BENCH_RUNS"] ?? "1") ?? 1
            let tokenIds = (0..<promptLen).map { Int($0 % 4096 + 1) }
            for runIdx in 0..<runs {
                print("\n[Simple run \(runIdx + 1)/\(runs)] \(maxNew) tokens from \(promptLen)-token prompt (prefillStep=\(sp.prefillStepSize))")
                let runCache = context.model.newCache(parameters: sp)
                let t0 = CFAbsoluteTimeGetCurrent()
                var sIter = try TokenIterator(
                    input: simpleInput, model: context.model, cache: runCache, parameters: sp,
                    cacheCoordinator: sCoord)
                var firstTokT: Double = 0
                var count = 0
                var generated: [Int] = []
                while let tok = sIter.next() {
                    count += 1
                    if count == 1 { firstTokT = CFAbsoluteTimeGetCurrent() - t0 }
                    generated.append(tok)
                }
                let tot = CFAbsoluteTimeGetCurrent() - t0
                let decodeTime = max(tot - firstTokT, 0.001)
                print(String(format:
                    "  generated %d tokens | TTFT %.0fms | decode %.1f tok/s | total %.2fs",
                    count, firstTokT * 1000, Double(count - 1) / decodeTime, tot))
                print("  first 10 tokens: \(Array(generated.prefix(10)))")

                // Manual store for single-request path with stepwise stderr
                // logging to pin down which call crashes — stdout buffering
                // hides the real crash site when the Fatal error takes over.
                // Skip via BENCH_STORE=0.
                if let c = sCoord, (env["BENCH_STORE"] ?? "1") == "1" {
                    FileHandle.standardError.write("[store] evaling cache\n".data(using: .utf8)!)
                    MLX.eval(runCache)
                    FileHandle.standardError.write("[store] extractLayerData\n".data(using: .utf8)!)
                    let perLayer = extractLayerData(from: runCache)
                    FileHandle.standardError.write("[store] extractLayerData done, non-nil=\(perLayer.compactMap{$0}.count)/\(perLayer.count)\n".data(using: .utf8)!)
                    let ssm: [MLXArray]?
                    if c.isHybrid {
                        FileHandle.standardError.write("[store] extractSSMStates\n".data(using: .utf8)!)
                        ssm = extractSSMStates(from: runCache)
                        FileHandle.standardError.write("[store] extractSSMStates done, count=\(ssm?.count ?? 0)\n".data(using: .utf8)!)
                    } else { ssm = nil }
                    if let ssm = ssm {
                        FileHandle.standardError.write("[store] eval ssm\n".data(using: .utf8)!)
                        MLX.eval(ssm)
                    }
                    FileHandle.standardError.write("[store] eval per-layer KV\n".data(using: .utf8)!)
                    for kv in perLayer {
                        if let kv = kv { MLX.eval(kv.keys, kv.values) }
                    }
                    FileHandle.standardError.write("[store] call storeAfterGeneration\n".data(using: .utf8)!)
                    c.storeAfterGeneration(
                        promptTokens: tokenIds,
                        perLayerData: perLayer,
                        ssmStates: ssm,
                        cache: runCache,
                        mediaSalt: nil)
                    FileHandle.standardError.write("[store] done\n".data(using: .utf8)!)
                    print("  [Coord] stored after generation")
                }
            }
            print("\n=== Simple Done ===")
            return
        }

        // Load pre-tokenized multi-turn data
        let tokFile = URL(fileURLWithPath: tokensPath)
        let data = try Data(contentsOf: tokFile)
        let json = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        let turns = json["turns"] as! [[Int]]
        let stubTokens = json["stub_tokens"] as! [Int]

        print("Loaded \(turns.count) turns, sizes: \(turns.map { $0.count })")

        // Diagnostic: print first few module parameter shapes
        var diagCount = 0
        for (key, arr) in context.model.parameters().flattened() {
            if key.contains("layers.0") && (key.contains("q_proj") || key.contains("embed_tokens")) {
                print("  param[\(key)] shape=\(arr.shape) dtype=\(arr.dtype)")
                diagCount += 1
                if diagCount >= 8 { break }
            }
        }

        var params = GenerateParameters(maxTokens: maxNew)
        params.temperature = 0.0
        params.prefillStepSize = 8192
        params.enableCompiledDecode = compileDecode
        params.compiledMaxCacheLength = compileMaxLen
        if compileDecode {
            print("Compiled decode: ON (maxCacheLength=\(compileMaxLen))")
        }

        // Persistent cache across turns
        let cache = context.model.newCache(parameters: params)
        let cacheTypes = Set(cache.map { String(describing: type(of: $0)) })
        print("Cache: \(cache.count) layers, types: \(cacheTypes)")

        // Warmup with a small prompt to prime kernels
        print("\n[Warmup] 64 tokens...")
        let warmupCache = context.model.newCache(parameters: params)
        let warmupTokens = Array(turns[0].prefix(min(25, turns[0].count))).map { Int32($0) }
        // 2D tokens [1, L] — works because step() now detects ndim and skips re-newAxis.
        let warmupTokenArray = MLXArray(warmupTokens)[.newAxis, .ellipsis]
        let warmupInput = LMInput(text: LMInput.Text(tokens: warmupTokenArray))
        var wParams = params
        wParams.maxTokens = 32
        var warmupIter = try TokenIterator(input: warmupInput, model: context.model, cache: warmupCache, parameters: wParams)
        var wCount = 0
        while let _ = warmupIter.next() { wCount += 1 }
        print("  warmup done (\(wCount) tokens)")

        print("\n[Multi-turn] cache reused across turns, generate 256 tokens/turn")
        // Track cumulative tokens fed to cache so we know what's "new" per turn
        var cumulativeTokens = 0
        for (turnIdx, turnTokens) in turns.enumerated() {
            // The "new" tokens this turn are everything beyond what we've already
            // processed. If cache offset == cumulativeTokens (assistant stub from
            // previous turn was injected), feed only the difference.
            let nPrompt = turnTokens.count
            let newTokens: [Int32]
            if turnIdx == 0 {
                newTokens = turnTokens.map { Int32($0) }
            } else {
                // Take only the tokens that aren't already cached.
                // cumulativeTokens reflects: (prev_turn_prompt + assistant_stub).
                // turnTokens[turnIdx] already includes prev user msg + prev stub assistant + new user msg
                let already = cumulativeTokens
                let slice = turnTokens[already...].map { Int32($0) }
                newTokens = Array(slice)
            }

            // 2D tokens [1, L] — TokenIterator handles both 1D and 2D safely.
            let input = LMInput(text: LMInput.Text(tokens: MLXArray(newTokens)[.newAxis, .ellipsis]))

            let t0 = CFAbsoluteTimeGetCurrent()
            var iter = try TokenIterator(input: input, model: context.model, cache: cache, parameters: params)

            var firstTokenTime: Double = 0
            var count = 0
            var generated: [Int] = []
            while let tok = iter.next() {
                count += 1
                if count == 1 { firstTokenTime = CFAbsoluteTimeGetCurrent() - t0 }
                generated.append(tok)
            }
            let totalTime = CFAbsoluteTimeGetCurrent() - t0
            let decodeTime = totalTime - firstTokenTime
            let decodeTps = Double(max(0, count - 1)) / decodeTime
            // Prefill speed = tokens fed in this turn / time-to-first-token
            let prefillTps = Double(newTokens.count) / firstTokenTime

            print(String(format: "  Turn %d: total prompt=%d, NEW=%d | prefill %.0f tok/s (%.0fms TTFT) | decode %.1f tok/s (%d tokens, %.3fs)",
                         turnIdx + 1, nPrompt, newTokens.count, prefillTps, firstTokenTime * 1000, decodeTps, count, decodeTime))

            // After this turn: cache contains [prev_context + new_tokens + decoded_response].
            // Next turn's "already cached" = current turnTokens.count + assistant stub length
            // Note: the stub injected matches the next prompt's pre-existing assistant turn.
            cumulativeTokens = nPrompt + stubTokens.count
        }

        print("\n=== Done ===")
    }
}

// Stub tokenizer — model loading requires one, but this bench bypasses tokenization
// (uses pre-tokenized JSON tokens from Python).
final class NullTokenizerLoader: TokenizerLoader, @unchecked Sendable {
    func load(from directory: URL) async throws -> any MLXLMCommon.Tokenizer {
        return NullTokenizer()
    }
}

private func loadBenchModelContext(from modelDir: URL) async throws -> ModelContext {
    let loaded = try await MLXLMCommon.loadModel(
        from: modelDir,
        using: #huggingFaceTokenizerLoader(),
        loadConfiguration: .osaurusProduction)
    return loaded.0
}

// MARK: - BatchEngine multi-turn chat (iter 26)

/// TRUE BatchEngine multi-turn verification. Unlike
/// `runCoherentMultiTurn` which uses `ChatSession` (backed by
/// `TokenIterator`), this routes each turn through
/// `BatchEngine.generate(...)` with a shared `CacheCoordinator` so
/// cross-turn cache hits are possible.
///
/// Runs three turns with a factual callback (turn 2 should recall
/// turn 1's info), twice — once with compile off, once with compile on.
func runBatchEngineMultiTurn(modelPath: String, maxNew: Int) async throws {
    let modelDir = URL(fileURLWithPath: modelPath)
    print("\n=== BatchEngine multi-turn chat (iter 26, simplified iter 27) ===")
    print("Loading with real HuggingFace tokenizer...")
    let loadStart = CFAbsoluteTimeGetCurrent()
    let context = try await loadBenchModelContext(from: modelDir)
    print(String(format: "Load: %.2fs", CFAbsoluteTimeGetCurrent() - loadStart))
    print("Model: \(type(of: context.model))")

    // Iter 27: use BatchEngine DIRECTLY on the loaded context rather
    // than wrapping in ModelContainer + enableCaching. Iter 26's
    // ModelContainer wrap hung for 8 minutes — likely an interaction
    // between makeBatchEngine's nested perform block and a freshly-
    // constructed container that doesn't match the context's
    // original container.
    nonisolated(unsafe) let ctx = context

    for compileOn in [false, true] {
        let label = compileOn ? "compile ON" : "compile OFF"
        print("\n[\(label)] BatchEngine 3-turn chat")

        var params = GenerateParameters(
            maxTokens: maxNew, temperature: 0,
            prefillStepSize: 512)
        params.enableCompiledBatchDecode = compileOn

        let engine = BatchEngine(context: ctx, maxBatchSize: 1)

        // Accumulate a simple text transcript so turn 2's prompt
        // contains turn 1's response, exercising multi-turn context
        // without needing cache-coordinator prefix matching.
        var history = "You are a helpful assistant. Keep responses very brief."
        for (i, prompt) in [
            "My favorite color is blue.",
            "What is my favorite color?",
            "Is that a warm or cool color?",
        ].enumerated() {
            history += "\n\nUser: \(prompt)\nAssistant:"
            let text = try await runBatchEngineTurn(
                engine: engine, context: ctx,
                fullText: history, label: "Turn \(i+1)",
                parameters: params, maxNew: maxNew)
            history += " \(text)"
        }
    }

    print("\n=== BatchEngine multi-turn done ===")
}

/// Send one UserInput through `BatchEngine.generate(...)`, collect
/// text chunks, return decoded response.
func runBatchEngineTurn(
    engine: BatchEngine,
    context: MLXLMCommon.ModelContext,
    fullText: String,
    label: String,
    parameters: GenerateParameters,
    maxNew: Int
) async throws -> String {
    print("  \(label) [\(parameters.enableCompiledBatchDecode ? "compile" : "uncomp")]:")
    let t0 = CFAbsoluteTimeGetCurrent()

    // Prepare input directly on the loaded context (no container actor).
    let input = try await context.processor.prepare(
        input: UserInput(prompt: fullText))
    nonisolated(unsafe) let sendable = input
    // Iter 28: test the fixed `generate()` path. Iter 27 had to use
    // submit() as a workaround because generate() hung under real HF
    // tokenizer. If this iteration runs to completion, the iter-28 fix
    // (Task.detached in generate) actually worked.
    let stream = await engine.generate(input: sendable, parameters: parameters)

    var text = ""
    var reasoning = ""
    var ttft: Double?
    var chunkCount = 0
    for await event in stream {
        switch event {
        case .chunk(let chunk):
            if ttft == nil { ttft = CFAbsoluteTimeGetCurrent() - t0 }
            text += chunk
            chunkCount += 1
            if chunkCount > maxNew * 2 { break }
        case .reasoning(let r):
            if ttft == nil { ttft = CFAbsoluteTimeGetCurrent() - t0 }
            reasoning += r
            chunkCount += 1
            if chunkCount > maxNew * 2 { break }
        case .info, .toolCall:
            break
        }
    }
    let total = CFAbsoluteTimeGetCurrent() - t0
    let visible = text
    let previewSource = visible.isEmpty ? "[empty visible; reasoning chars=\(reasoning.count)]" : visible
    let preview = previewSource.count > 150 ? String(previewSource.prefix(150)) + "..." : previewSource
    print("    TTFT \(Int((ttft ?? 0) * 1000))ms, total \(String(format: "%.2fs", total))")
    print("    \"\(preview)\"")
    return visible
}

// MARK: - Tool-call pipeline end-to-end (iter 66)

/// Submit a tool-bearing prompt through `BatchEngine.generate(...)` on a
/// real model, collecting `.chunk` / `.toolCall` events. Assert the
/// library-level pipeline contract:
///   - a real tool schema MUST be passed through `UserInput.tools`.
///   - the model MUST emit at least one structured `.toolCall`.
///   - `.chunk` output MUST NOT contain raw tool-call markers.
///   - `.chunk` output MUST NOT contain raw `<think>...</think>` markers.
func runBatchEngineToolCall(modelPath: String, maxNew: Int) async throws {
    let modelDir = URL(fileURLWithPath: modelPath)
    print("\n=== BatchEngine tool-call pipeline (iter 66) ===")
    let loadStart = CFAbsoluteTimeGetCurrent()
    let context = try await loadBenchModelContext(from: modelDir)
    print(String(format: "Load: %.2fs", CFAbsoluteTimeGetCurrent() - loadStart))
    print("Model: \(type(of: context.model))")
    print("Tool format: \(context.configuration.toolCallFormat.map { "\($0)" } ?? "json (default)")")
    print("Reasoning stamp: \(context.configuration.reasoningParserName ?? "nil")")

    nonisolated(unsafe) let ctx = context
    let engine = BatchEngine(context: ctx, maxBatchSize: 1)

    let weatherParams: [String: any Sendable] = [
        "type": "object",
        "properties": [
            "location": ["type": "string", "description": "City or region name"] as [String: any Sendable],
        ] as [String: any Sendable],
        "required": ["location"],
    ]
    let weatherFn: [String: any Sendable] = [
        "name": "get_weather",
        "description": "Get current weather for a location.",
        "parameters": weatherParams,
    ]
    let weatherTool: [String: any Sendable] = [
        "type": "function",
        "function": weatherFn,
    ]

    // Strict tool-only request. A text answer with no `.toolCall` is useful
    // evidence, but it is not a passing tool-call pipeline row.
    let prompt = """
        Call get_weather for location Tokyo.
        Emit only the tool call. Do not answer in prose.
        """

    let fallbackParams = GenerateParameters(maxTokens: maxNew, prefillStepSize: 512)
    var params = GenerateParameters(
        generationConfig: ctx.configuration.generationDefaults,
        fallback: fallbackParams)
    params.enableCompiledBatchDecode = false

    let input = try await ctx.processor.prepare(input: UserInput(
        prompt: prompt,
        tools: [weatherTool],
        additionalContext: ["enable_thinking": false]))

    if (ProcessInfo.processInfo.environment["BENCH_BATCH_TOOLCALL_RAW"] ?? "0") == "1" {
        nonisolated(unsafe) let rawInput = input
        let (_, rawStream) = await engine.submit(input: rawInput, parameters: params)
        var tokenIds = [Int]()
        var info: GenerateCompletionInfo?
        for await event in rawStream {
            switch event {
            case .token(let id):
                tokenIds.append(id)
            case .info(let i):
                info = i
            }
        }
        let decoded = ctx.tokenizer.decode(tokenIds: tokenIds, skipSpecialTokens: false)
        print("  raw token ids: \(tokenIds)")
        print("  raw decoded BEGIN")
        print(decoded)
        print("  raw decoded END")
        if let info {
            print("  raw stop=\(info.stopReason) genTokens=\(info.generationTokenCount)")
        }
        await engine.shutdown()
        return
    }

    nonisolated(unsafe) let sendable = input
    let generationStart = CFAbsoluteTimeGetCurrent()
    let stream = await engine.generate(input: sendable, parameters: params)

    var chunkText = ""
    var toolCallCount = 0
    var toolCallDetails = [String]()
    var reasoningChars = 0
    var info: GenerateCompletionInfo?
    for await event in stream {
        switch event {
        case .chunk(let c):
            chunkText += c
        case .reasoning(let r):
            reasoningChars += r.count
        case .toolCall(let call):
            toolCallCount += 1
            let args = call.function.arguments.mapValues { $0.anyValue }
            let argText: String
            if JSONSerialization.isValidJSONObject(args),
               let data = try? JSONSerialization.data(withJSONObject: args, options: [.sortedKeys]),
               let text = String(data: data, encoding: .utf8) {
                argText = text
            } else {
                argText = String(describing: call.function.arguments)
            }
            toolCallDetails.append("\(call.function.name)(\(argText))")
        case .info(let i):
            info = i
        }
    }

    let total = CFAbsoluteTimeGetCurrent() - generationStart
    let emittedEvents = chunkText.count + reasoningChars + toolCallCount
    let tokps: Double
    if let tokens = info?.generationTokenCount, total > 0 {
        tokps = Double(tokens) / total
    } else if total > 0 {
        tokps = Double(emittedEvents) / total
    } else {
        tokps = 0
    }
    let preview = chunkText.count > 240 ? String(chunkText.prefix(240)) + "..." : chunkText
    let stop = info.map { "\($0.stopReason)" } ?? "nil"
    let genTokens = info.map { "\($0.generationTokenCount)" } ?? "nil"
    print(String(format:
        "  chunks: %d chars, reasoning: %d chars, toolCalls: %d, stop: %@, genTokens: %@, total=%.2fs, tokps=%.2f",
        chunkText.count, reasoningChars, toolCallCount, stop, genTokens, total, tokps))
    print("  tool calls: \(toolCallDetails.joined(separator: "; "))")
    print("  text preview: \"\(preview)\"")

    // Contract assertions. These must hold on *every* family: osaurus
    // relies on pure-text `.chunk` + authoritative `.toolCall` events.
    let leakedMarkers = [
        "<tool_call>",
        "<|tool_call>",
        "<minimax:tool_call>",
        "[TOOL_CALLS]",
        "<think>",
        "</think>",
    ].filter { chunkText.contains($0) }

    if !leakedMarkers.isEmpty {
        print("  FAIL — raw markers leaked into .chunk: \(leakedMarkers)")
        throw NSError(
            domain: "BENCH_BATCH_TOOLCALL", code: 1,
            userInfo: [NSLocalizedDescriptionKey:
                "Raw library-level markers leaked into .chunk: \(leakedMarkers)."])
    }
    if toolCallCount == 0 && chunkText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
        print("  FAIL — no structured tool call and no visible fallback output.")
        throw NSError(
            domain: "BENCH_BATCH_TOOLCALL", code: 2,
            userInfo: [NSLocalizedDescriptionKey:
                "No structured tool call and no visible fallback output."])
    }
    if toolCallCount == 0 {
        print("  FAIL — generation completed without a structured .toolCall event.")
        throw NSError(
            domain: "BENCH_BATCH_TOOLCALL", code: 3,
            userInfo: [NSLocalizedDescriptionKey:
                "Tool schema was supplied, but the generation produced no structured .toolCall event."])
    }
    print("  OK — structured tool call emitted and no raw markers leaked to .chunk.")

    print("\n=== BatchEngine tool-call pipeline done ===")
}

// MARK: - SpecDec bench scenario (iter 16)

/// Run the same short prompt through three paths — plain `Evaluate.generate`,
/// DFlash linear, DDTree tree-verify — and report per-path tok/s +
/// byte-parity vs plain at temperature 0.
///
/// The drafter+target pair must share a hidden_size. For the downloaded
/// snapshots, that means:
///   - drafter `z-lab/Qwen3.5-27B-DFlash` (hidden=5120, 5 layers)
///     pairs with target `mlx-community/Qwen3.5-27B-4bit` (hidden=5120).
///
/// Prints one line per path so the operator can cross-reference tok/s
/// numbers. Checks byte-parity between plain + DFlash + DDTree, failing
/// the run (non-zero exit) if outputs diverge.
func runBatchSpecDec(
    modelPath: String, drafterPath: String, maxNew: Int
) async throws {
    let modelDir = URL(fileURLWithPath: modelPath)
    let drafterDir = URL(fileURLWithPath: drafterPath)
    print("\n=== BatchEngine SpecDec (iter 16) ===")
    print("Target:  \(modelDir.lastPathComponent)")
    print("Drafter: \(drafterDir.lastPathComponent)")

    // Must resolve to a DFlash drafter snapshot.
    guard DFlashDrafterLoader.looksLikeDrafter(at: drafterDir) else {
        print("  [skip] drafter not on disk at \(drafterDir.path)")
        return
    }
    let drafter: DFlashDraftModel
    do {
        drafter = try DFlashDrafterLoader.load(from: drafterDir)
    } catch {
        print("  [skip] drafter load failed: \(error)")
        return
    }
    // HF target_layer_ids ARE 0-based indices into target.model.layers
    // (per z-lab/dflash `_patch_model`). Use them directly — no shift.
    let targetBlockIDs = drafter.config.dflashConfig.targetLayerIds

    // Load target via HF tokenizer so the drafter+target share tokenizer.
    let loadStart = CFAbsoluteTimeGetCurrent()
    let context = try await MLXLMCommon.loadModel(
        from: modelDir, using: #huggingFaceTokenizerLoader())
    print(String(format: "Target load: %.2fs", CFAbsoluteTimeGetCurrent() - loadStart))

    // Deterministic prompt. Override via BENCH_SPECDEC_PROMPT env var.
    let promptText = ProcessInfo.processInfo.environment["BENCH_SPECDEC_PROMPT"]
        ?? "The capital of France is"
    let promptTokens = try context.tokenizer.applyChatTemplate(
        messages: [["role": "user", "content": promptText]])
    let promptInts = promptTokens.map { Int32($0) }
    let promptIds = MLXArray(promptInts).reshaped(1, promptInts.count)

    // Cast target to SpecDec protocols — skip if not conformant.
    guard let target = context.model
        as? any (HiddenStateCaptureModel & TokenEmbedderModel)
    else {
        print("  [skip] target \(type(of: context.model)) does not conform to HiddenStateCaptureModel + TokenEmbedderModel")
        return
    }

    // Plain greedy AR with a PERSISTENT KV cache — the honest baseline.
    // Prefills the whole prompt once (cache fills with promptLen states),
    // then each decode step only processes ONE new token through the
    // cached model (O(1) per step instead of O(N)).
    func materializeLogits(_ a: MLXArray) { MLX.eval(a) }
    func greedyAR() throws -> [Int32] {
        var out = promptInts
        let cache = context.model.newCache(parameters: nil)
        let promptArr = MLXArray(promptInts).reshaped(1, promptInts.count)
        var (logits, _) = target(promptArr, cache: cache, captureLayerIDs: [])
        materializeLogits(logits)
        var nextTok = argMax(
            logits[0, logits.dim(1) - 1, 0...], axis: -1
        ).asType(.int32).item(Int32.self)
        out.append(nextTok)
        for _ in 1..<maxNew {
            let stepIn = MLXArray([nextTok]).reshaped(1, 1)
            (logits, _) = target(stepIn, cache: cache, captureLayerIDs: [])
            materializeLogits(logits)
            nextTok = argMax(
                logits[0, logits.dim(1) - 1, 0...], axis: -1
            ).asType(.int32).item(Int32.self)
            out.append(nextTok)
        }
        return out
    }

    // Measurement harness.
    func measure<T>(_ label: String, _ body: () throws -> (T, Int)) rethrows -> T {
        let t0 = CFAbsoluteTimeGetCurrent()
        let (result, generated) = try body()
        let dt = CFAbsoluteTimeGetCurrent() - t0
        let tps = dt > 0 ? Double(generated) / dt : 0
        print(String(
            format: "  %@: %.2fs / %d tokens / %.1f tok/s",
            label, dt, generated, tps))
        return result
    }

    let arTokens = try measure("plain AR") {
        let tokens = try greedyAR()
        return (tokens, tokens.count - promptInts.count)
    }

    // Optional TurboQuant KV compression on the fast path.
    // BENCH_SPECDEC_KV_TURBOQUANT=1 enables 3-bit TurboQuant.
    let kvMode: KVQuantizationMode
    if ProcessInfo.processInfo.environment[
        "BENCH_SPECDEC_KV_TURBOQUANT"] == "1"
    {
        kvMode = .turboQuant(keyBits: 3, valueBits: 3)
        print("  kv-compression: turboQuant(3,3)")
    } else {
        kvMode = .none
    }

    var dfAcceptance: [Int] = []
    let dfTokens = try measure("DFlash linear") {
        let args = DFlashLinearArgs(
            target: target, drafter: drafter,
            targetBlockIDs: targetBlockIDs,
            maskTokenID: Int32(drafter.config.dflashConfig.maskTokenId),
            inputIds: promptIds, maxNewTokens: maxNew,
            stopTokenIDs: [], temperature: 0, kvMode: kvMode)
        let r = try SpecDecRuntimeLinear.run(args)
        dfAcceptance = r.acceptanceLengths
        return (r.tokenIds, r.tokenIds.count - promptInts.count)
    }
    if !dfAcceptance.isEmpty {
        let mean = Double(dfAcceptance.reduce(0, +)) / Double(dfAcceptance.count)
        let bs = drafter.config.blockSize
        print(String(
            format: "    DFlash acceptance: %d rounds, mean=%.2f / %d, draft tokens/round=%.2f",
            dfAcceptance.count, mean, bs - 1, mean + 1))
    }

    var ddAcceptance: [Int] = []
    let ddTokens = try measure("DDTree (budget=8)") {
        let args = DDTreeArgs(
            target: target, drafter: drafter,
            targetBlockIDs: targetBlockIDs,
            maskTokenID: Int32(drafter.config.dflashConfig.maskTokenId),
            inputIds: promptIds, maxNewTokens: maxNew,
            stopTokenIDs: [], temperature: 0,
            branchingBudget: 8)
        let r = try SpecDecRuntimeDDTree.run(args)
        ddAcceptance = r.acceptanceLengths
        return (r.tokenIds, r.tokenIds.count - promptInts.count)
    }
    if !ddAcceptance.isEmpty {
        let mean = Double(ddAcceptance.reduce(0, +)) / Double(ddAcceptance.count)
        print(String(
            format: "    DDTree acceptance: %d rounds, mean depth=%.2f",
            ddAcceptance.count, mean))
    }

    // Byte-parity vs greedy AR. At temp=0 the *accepted* SpecDec tokens
    // should equal AR argmaxes — but high-precision targets (bf16) can
    // show sub-ULP drift from running SDPA over different (q_len, k_len)
    // shapes, occasionally flipping close argmaxes. Report the match
    // rate rather than crashing so the bench can still complete.
    func matchCount(_ a: [Int32], _ b: [Int32]) -> (Int, Int) {
        let n = min(a.count, b.count)
        var m = 0
        for i in 0..<n where a[i] == b[i] { m += 1 }
        return (m, n)
    }
    let (dfM, dfT) = matchCount(dfTokens, arTokens)
    let (ddM, ddT) = matchCount(ddTokens, arTokens)
    let dfPct = dfT > 0 ? 100.0 * Double(dfM) / Double(dfT) : 100.0
    let ddPct = ddT > 0 ? 100.0 * Double(ddM) / Double(ddT) : 100.0
    print(String(format:
        "  byte-parity vs AR: DFlash=%d/%d (%.1f%%), DDTree=%d/%d (%.1f%%)",
        dfM, dfT, dfPct, ddM, ddT, ddPct))

    print("=== BatchEngine SpecDec done ===")
}

// MARK: - Coherent multi-turn chat (iter 19)

/// Run a real 3-turn conversation through `BatchEngine` using the
/// Hugging Face tokenizer + chat template + cache coordinator. The
/// user has repeatedly asked for coherent multi-turn verification,
/// not just synthetic-prompt tok/s. This harness delivers that.
///
/// - Turn 1: introduces a fact ("My favorite color is blue")
/// - Turn 2: asks the model to recall it
/// - Turn 3: asks a follow-up that should reference turn 2's answer
///
/// Both compile-on and compile-off paths are exercised to confirm
/// coherence doesn't regress when compile engages.
func runCoherentMultiTurn(modelPath: String, maxNew: Int) async throws {
    // Reload with real HF tokenizer (the default Bench load uses
    // NullTokenizer for perf benchmarks; coherence needs the real one).
    let modelDir = URL(fileURLWithPath: modelPath)
    print("\n=== Coherent multi-turn chat (iter 19) ===")
    print("Loading with real HuggingFace tokenizer...")
    let loadStart = CFAbsoluteTimeGetCurrent()
    let context = try await MLXLMCommon.loadModel(
        from: modelDir, using: #huggingFaceTokenizerLoader())
    print(String(format: "Load: %.2fs", CFAbsoluteTimeGetCurrent() - loadStart))
    print("Model: \(type(of: context.model))")

    // Three-turn conversation, same shape as
    // `ChatSessionTests.multiTurnConversation`. Using ChatSession lets
    // us exercise the full chat template + multi-turn cache path.
    for compileOn in [false, true] {
        let label = compileOn ? "compile ON" : "compile OFF"
        print("\n[\(label)] 3-turn chat coherence")

        var params = GenerateParameters(maxTokens: maxNew, temperature: 0)
        params.enableCompiledBatchDecode = compileOn

        let session = ChatSession(
            context,
            instructions: "You are a helpful assistant. Keep responses very brief (one sentence max).",
            generateParameters: params
        )

        try await runChatTurn(session: session, prompt: "My favorite color is blue.", label: "Turn 1")
        try await runChatTurn(session: session, prompt: "What is my favorite color?", label: "Turn 2")
        try await runChatTurn(session: session, prompt: "Is that a warm or cool color?", label: "Turn 3")
    }

    print("\n=== Coherent multi-turn done ===")
}

/// Stream one ChatSession turn; print label, tok/s, and response text
/// (truncated to 200 chars for legibility).
///
/// 2026-05-04: also prints reasoning content separately so DSV4-style
/// always-think models don't appear to "produce nothing" when the
/// reasoning parser strips `<think>...</think>` from the chunk stream
/// before maxTokens is hit. Set `BENCH_HIDE_REASONING=1` to suppress.
func runChatTurn(session: ChatSession, prompt: String, label: String) async throws {
    print("\n  User: \(prompt)")
    let t0 = CFAbsoluteTimeGetCurrent()
    var text = ""
    var reasoning = ""
    var ttft: Double?
    let hideReasoning =
        ProcessInfo.processInfo.environment["BENCH_HIDE_REASONING"] == "1"
    for try await event in session.streamDetails(to: prompt, images: [], videos: []) {
        if ttft == nil { ttft = CFAbsoluteTimeGetCurrent() - t0 }
        if let chunk = event.chunk {
            text += chunk
        } else if !hideReasoning, case .reasoning(let r) = event {
            reasoning += r
        }
    }
    let total = CFAbsoluteTimeGetCurrent() - t0
    print("  \(label) [TTFT \(Int((ttft ?? 0) * 1000))ms | total \(String(format: "%.2fs", total))]:")
    if !reasoning.isEmpty {
        let preview = reasoning.count > 300 ? String(reasoning.prefix(300)) + "..." : reasoning
        print("    REASONING: \"\(preview)\"")
    }
    let preview = text.count > 300 ? String(text.prefix(300)) + "..." : text
    print("    CHUNK:     \"\(preview)\"")
}

// MARK: - BatchEngine real-model smoke (iter 17)

/// Runs 4 scenarios through `BatchEngine` on a real loaded model:
///   1. Baseline (compile off, maxBatchSize=1)
///   2. Stage 1B.3 compile-on (enableCompiledBatchDecode, maxBatchSize=1)
///   3. Stage 0 TurboQuant (kvMode=.turboQuant, maxBatchSize=1)
///   4. B=2 concurrent (uncompiled, maxBatchSize=2)
///
/// Each scenario prints TTFT, decode tok/s, and token IDs for manual
/// coherence inspection. Intended as the real-model counterpart to the
/// synthetic-model unit tests — verifies the BatchEngine changes work
/// on Qwen3-0.6B-8bit (or any other loaded real model via BENCH_MODEL).
func runBatchSmoke(context: MLXLMCommon.ModelContext, maxNew: Int) async throws {
    print("\n=== BatchEngine real-model smoke (iter 17, iter 18 warmup) ===")

    let promptIDs = (1...8).map { Int32($0) }
    let prompt = MLXArray(promptIDs)[.newAxis, .ellipsis]
    let input = LMInput(text: LMInput.Text(tokens: prompt))

    // Warmup pass (iter 18, extended iter 20): the first forward pass
    // pays one-time lazy-module initialisation, AND the first compile
    // trace pays compile-time cost. Warm both the uncompiled path and
    // the compile path separately so each measured scenario starts
    // from a warm state.
    print("\n[Warmup] (not measured)")
    try await runBatchScenario(
        context: context, input: input, label: "warmup-uncompiled",
        params: GenerateParameters(maxTokens: 3, temperature: 0),
        maxBatchSize: 1, silent: true)
    try await runBatchScenario(
        context: context, input: input, label: "warmup-compiled",
        params: GenerateParameters(
            maxTokens: 3, enableCompiledBatchDecode: true, temperature: 0),
        maxBatchSize: 1, silent: true)

    // 1. Baseline: compile off, maxBatchSize=1.
    try await runBatchScenario(
        context: context, input: input, label: "1. Baseline (compile off)",
        params: GenerateParameters(maxTokens: maxNew, temperature: 0),
        maxBatchSize: 1)

    // 2. Compile on — Stage 1B.3 path.
    try await runBatchScenario(
        context: context, input: input, label: "2. Stage 1B.3 compile",
        params: GenerateParameters(
            maxTokens: maxNew,
            enableCompiledBatchDecode: true,
            temperature: 0),
        maxBatchSize: 1)

    // 3. TurboQuant on — Stage 0 path. Compile is silently skipped for
    // TQ (v2 rollback).
    try await runBatchScenario(
        context: context, input: input, label: "3. Stage 0 TurboQuant",
        params: GenerateParameters(
            maxTokens: maxNew,
            kvMode: .turboQuant(keyBits: 3, valueBits: 3),
            temperature: 0),
        maxBatchSize: 1)

    // 4. Two concurrent requests — uncompiled batched decode path.
    print("\n[4. B=2 concurrent uncompiled]")
    nonisolated(unsafe) let ctx4 = context
    let engine4 = BatchEngine(context: ctx4, maxBatchSize: 2)
    _ = LMInput(text: LMInput.Text(
        tokens: MLXArray((10...17).map { Int32($0) })[.newAxis, .ellipsis]))
    let p4 = GenerateParameters(maxTokens: maxNew, temperature: 0)

    // Fresh per-submit inputs to satisfy Swift 6 sending-risks-data-race
    // (LMInput isn't Sendable; each submit consumes its own instance).
    let t0 = CFAbsoluteTimeGetCurrent()
    let i1 = LMInput(text: LMInput.Text(
        tokens: MLXArray((1...8).map { Int32($0) })[.newAxis, .ellipsis]))
    let i2 = LMInput(text: LMInput.Text(
        tokens: MLXArray((10...17).map { Int32($0) })[.newAxis, .ellipsis]))
    let (_, s1) = await engine4.submit(input: i1, parameters: p4)
    let (_, s2) = await engine4.submit(input: i2, parameters: p4)
    var tokens1: [Int] = []
    var tokens2: [Int] = []
    for await e in s1 {
        if case .token(let id) = e { tokens1.append(id) }
    }
    for await e in s2 {
        if case .token(let id) = e { tokens2.append(id) }
    }
    let total = CFAbsoluteTimeGetCurrent() - t0
    print(String(format: "  R1: %d tokens, R2: %d tokens | total %.2fs",
        tokens1.count, tokens2.count, total))
    print("  R1 first 8: \(Array(tokens1.prefix(8)))")
    print("  R2 first 8: \(Array(tokens2.prefix(8)))")

    print("\n=== BatchEngine smoke done ===")
}

/// Run one BatchEngine scenario and print timing + first tokens.
func runBatchScenario(
    context: MLXLMCommon.ModelContext,
    input: LMInput,
    label: String,
    params: GenerateParameters,
    maxBatchSize: Int,
    silent: Bool = false
) async throws {
    if !silent { print("\n[\(label)]") }
    nonisolated(unsafe) let ctx = context
    let engine = BatchEngine(context: ctx, maxBatchSize: maxBatchSize)

    let t0 = CFAbsoluteTimeGetCurrent()
    nonisolated(unsafe) let sendableInput = input
    let (_, stream) = await engine.submit(input: sendableInput, parameters: params)

    var tokens: [Int] = []
    var firstTokAt: Double?
    var stopReason: GenerateStopReason?
    for await event in stream {
        switch event {
        case .token(let id):
            if firstTokAt == nil {
                firstTokAt = CFAbsoluteTimeGetCurrent() - t0
            }
            tokens.append(id)
        case .info(let info):
            stopReason = info.stopReason
        }
    }
    if silent { return }

    let total = CFAbsoluteTimeGetCurrent() - t0
    let decodeTime = max(total - (firstTokAt ?? 0), 0.001)
    let tps = Double(max(tokens.count - 1, 1)) / decodeTime
    print(String(format:
        "  %d tokens | TTFT %.0fms | decode %.1f tok/s | stop=%@",
        tokens.count,
        (firstTokAt ?? 0) * 1000,
        tps,
        String(describing: stopReason ?? .length)))
    print("  first 8 tokens: \(Array(tokens.prefix(8)))")
}

// MARK: - Cross-engine correctness validation (iter 32)

/// Run the same short chat prompt through BOTH `TokenIterator` and
/// `BatchEngine.generate(...)` with identical deterministic parameters
/// (temperature=0) and compare the emitted token IDs. Equality is the
/// property — divergence means one of the paths has a bug.
///
/// This is the strongest single correctness check for the engine. The
/// compile-on/off identity check in `BENCH_BATCH_CHAT` only proves
/// BatchEngine is internally consistent with itself. Cross-validation
/// against TokenIterator proves BatchEngine matches the long-standing
/// single-sequence path used by `ChatSession`.
///
/// Scope: text-only model, greedy sampling (temp=0), no cache coordinator
/// on either side — we want to isolate the engine/iterator, not the
/// multi-tier cache layer.
func runCrossEngineValidation(modelPath: String, maxNew: Int) async throws {
    let modelDir = URL(fileURLWithPath: modelPath)
    print("\n=== Cross-engine validation (iter 32) ===")
    print("Loading with real HuggingFace tokenizer...")
    let loadStart = CFAbsoluteTimeGetCurrent()
    let context = try await MLXLMCommon.loadModel(
        from: modelDir, using: #huggingFaceTokenizerLoader())
    print(String(format: "Load: %.2fs", CFAbsoluteTimeGetCurrent() - loadStart))
    print("Model: \(type(of: context.model))")

    // Deterministic params. Greedy sampling → any divergence between
    // iterators is a real engine bug, not a sampling noise artifact.
    let params = GenerateParameters(
        maxTokens: maxNew, temperature: 0, prefillStepSize: 512)

    // Three short prompts — keep each single-turn so we don't have to
    // replay history. The question is whether, given the same LMInput,
    // both iterators emit the same token stream.
    let prompts = [
        "Write a haiku about rain.",
        "Explain recursion in two sentences.",
        "List three primary colours.",
    ]

    // BatchEngine respects `stopTokenIDs` per-slot and terminates the
    // stream as soon as one is emitted. Raw `TokenIterator` DOES NOT —
    // it decodes until `maxTokens` is reached, letting EOS tokens through
    // as ordinary tokens. So if BatchEngine stops short, we must verify
    // equality only over the prefix BatchEngine actually emitted AND
    // that the next TokenIterator token is one BatchEngine would have
    // treated as EOS. Build the same stop set here.
    var stopTokenIDs: Set<Int> = context.configuration.eosTokenIds
    if let eos = context.tokenizer.eosTokenId { stopTokenIDs.insert(eos) }
    if let unk = context.tokenizer.unknownTokenId { stopTokenIDs.insert(unk) }
    for tok in context.configuration.extraEOSTokens {
        if let id = context.tokenizer.convertTokenToId(tok) {
            stopTokenIDs.insert(id)
        }
    }

    var mismatches = 0
    for (i, prompt) in prompts.enumerated() {
        print("\n[Probe \(i + 1)/\(prompts.count)] \"\(prompt)\"")
        let userInput = UserInput(prompt: prompt)
        let lmInput = try await context.processor.prepare(input: userInput)

        // Snapshot the tokens so we can feed EXACTLY the same input to
        // both paths. LMInput.text.tokens is an MLXArray — eval to make
        // sure the shape is materialized before either consumer reads it.
        let promptLen = lmInput.text.tokens.size
        print("  prompt tokens: \(promptLen)")

        // Path A: TokenIterator (single-sequence, the "baseline").
        let iterCache = context.model.newCache(parameters: params)
        let iter = try TokenIterator(
            input: lmInput, model: context.model, cache: iterCache,
            parameters: params)
        var iterTokens: [Int] = []
        for token in iter {
            iterTokens.append(token)
            if iterTokens.count >= maxNew { break }
        }

        // Path B: BatchEngine.
        nonisolated(unsafe) let ctx = context
        let engine = BatchEngine(context: ctx, maxBatchSize: 1)
        nonisolated(unsafe) let sendable = lmInput
        let (_, tokenStream) = await engine.submit(input: sendable, parameters: params)
        var engineTokens: [Int] = []
        for await event in tokenStream {
            switch event {
            case .token(let id):
                engineTokens.append(id)
                if engineTokens.count >= maxNew { break }
            case .info:
                break
            }
            if engineTokens.count >= maxNew { break }
        }

        // Compare.
        let iterSummary = Array(iterTokens.prefix(20))
        let engSummary = Array(engineTokens.prefix(20))
        print("  TokenIterator (\(iterTokens.count) toks): first 20 = \(iterSummary)")
        print("  BatchEngine   (\(engineTokens.count) toks): first 20 = \(engSummary)")

        // Find first divergence point.
        var firstDiff: Int? = nil
        let n = min(iterTokens.count, engineTokens.count)
        for k in 0..<n where iterTokens[k] != engineTokens[k] {
            firstDiff = k
            break
        }
        if iterTokens == engineTokens {
            print("  ✓ identical (\(iterTokens.count) tokens)")
        } else if firstDiff == nil &&
                  engineTokens.count < iterTokens.count &&
                  iterTokens.count > engineTokens.count &&
                  stopTokenIDs.contains(iterTokens[engineTokens.count]) {
            // Common case on chatty models: BatchEngine stopped at EOS,
            // raw TokenIterator continued through it. Prefix is identical
            // and the "extra" token TokenIterator emitted is in the stop
            // set. That's correct behaviour, not divergence.
            print("  ✓ identical \(engineTokens.count)-token prefix — " +
                  "BatchEngine stopped at EOS token \(iterTokens[engineTokens.count]) " +
                  "which TokenIterator's raw loop ignores")
        } else if let d = firstDiff {
            mismatches += 1
            print("  ✗ diverge at index \(d): iter=\(iterTokens[d]), engine=\(engineTokens[d])")
        } else {
            mismatches += 1
            print("  ✗ length differs: iter=\(iterTokens.count) vs engine=\(engineTokens.count)")
        }
    }

    print("\n=== Cross-engine validation: \(prompts.count - mismatches)/\(prompts.count) matched ===")
    if mismatches > 0 {
        fputs("[CrossValidate] FAIL: \(mismatches) prompt(s) diverged\n", stderr)
        exit(1)
    }
}

// MARK: - B=2 concurrent real-model validation (iter 33)

/// Submit two DIFFERENT prompts to `BatchEngine(maxBatchSize: 2)` and
/// iterate both streams concurrently under a `TaskGroup`. Proves the
/// batched-decode hot path functions end-to-end with real HF tokenizer
/// output — not just the synthetic `NullTokenizer` path covered by
/// `BENCH_BATCH`, and not the serialised stream reads it also does.
///
/// Acceptance:
/// - Both streams complete within max-tokens or EOS.
/// - Each stream produces a coherent non-empty preview.
/// - The engine doesn't crash, hang, or mix tokens between slots.
func runBatchEngineConcurrent(modelPath: String, maxNew: Int) async throws {
    let modelDir = URL(fileURLWithPath: modelPath)
    print("\n=== BatchEngine B=2 concurrent (iter 33) ===")
    print("Loading with real HuggingFace tokenizer...")
    let loadStart = CFAbsoluteTimeGetCurrent()
    let context = try await MLXLMCommon.loadModel(
        from: modelDir, using: #huggingFaceTokenizerLoader())
    print(String(format: "Load: %.2fs", CFAbsoluteTimeGetCurrent() - loadStart))
    print("Model: \(type(of: context.model))")

    let params = GenerateParameters(
        maxTokens: maxNew, temperature: 0, prefillStepSize: 512)

    nonisolated(unsafe) let ctx = context
    let engine = BatchEngine(context: ctx, maxBatchSize: 2)

    // Two semantically-distinct prompts so divergent outputs are expected.
    // If the engine mixes slots, one would "see" the other's tokens and
    // the previews would look suspiciously similar.
    let prompts = [
        "What city is the capital of France? Reply with one word.",
        "What is 2 + 2? Reply with just the number.",
    ]

    // Prepare both inputs ahead of the race so `.processor.prepare`
    // overhead doesn't bias who submits first.
    var inputs: [LMInput] = []
    for p in prompts {
        let input = try await context.processor.prepare(input: UserInput(prompt: p))
        inputs.append(input)
    }

    // Submit both, then await each stream concurrently via TaskGroup.
    // TaskGroup is what forces real concurrency — without it, the caller
    // would iterate s1 to completion before touching s2, and the engine's
    // decode step never sees B=2.
    let t0 = CFAbsoluteTimeGetCurrent()
    nonisolated(unsafe) let send0 = inputs[0]
    nonisolated(unsafe) let send1 = inputs[1]
    let (_, s0) = await engine.submit(input: send0, parameters: params)
    let (_, s1) = await engine.submit(input: send1, parameters: params)

    // Collect decoded text for each slot in parallel. Uses the tokenizer
    // directly rather than NaiveStreamingDetokenizer — the synchronous
    // decode is fine for a small benchmark and avoids the O(n²) relay
    // cost that hammers throughput under HF tokenizers.
    let tokenizer = context.tokenizer
    let results = await collectBatchStreamsWithOverlap(
        engine,
        streams: [(0, s0), (1, s1)],
        maxTokens: maxNew,
        label: "ConcurrentBatch B=2")
    let total = CFAbsoluteTimeGetCurrent() - t0

    // Print side-by-side. Guard against empty (stuck) slots.
    for (slot, ids) in results.sorted(by: { $0.0 < $1.0 }) {
        let text = tokenizer.decode(tokenIds: ids)
        print("  Slot \(slot) prompt: \"\(prompts[slot])\"")
        print("    tokens: \(ids.count), first 12: \(Array(ids.prefix(12)))")
        printDecodedOutput(label: "ConcurrentBatch slot \(slot)", text: text)
        if ids.isEmpty {
            fputs("[ConcurrentBatch] FAIL: slot \(slot) produced zero tokens\n", stderr)
            exit(1)
        }
        let lower = text.lowercased()
        if slot == 0 && !lower.contains("paris") {
            fputs("[ConcurrentBatch] FAIL: slot 0 did not answer its France prompt with Paris.\n",
                  stderr)
            exit(1)
        }
        if slot == 1 && !lower.contains("4") {
            fputs("[ConcurrentBatch] FAIL: slot 1 did not answer its arithmetic prompt.\n",
                  stderr)
            exit(1)
        }
    }
    print(String(format: "  total wall time: %.2fs (both slots)", total))
    await engine.shutdown()
    print("=== BatchEngine B=2 concurrent: passed ===")
}

// MARK: - Cache coordinator cross-turn reuse (iter 34)

/// Verify that `CacheCoordinator` wired into `BatchEngine` actually
/// produces cross-turn cache hits on real prompts. Critical property
/// for osaurus: the coordinator is the whole reason multi-turn chats
/// don't re-prefill from scratch every turn.
///
/// Methodology:
/// 1. Build a `BatchEngine` with an in-memory paged `CacheCoordinator`.
/// 2. Turn 1: prompt "The sky is blue. <Q>" — cold cache → full prefill.
/// 3. Turn 2: prompt "The sky is blue. <Q>. And also <Q2>" — warm cache
///    → prefill should only cover the added suffix.
/// 4. Compare `GenerateCompletionInfo.promptTime` across turns; turn 2
///    must be at least **2× faster** on the prefill (a conservative
///    threshold — real gains on Qwen3-0.6B are typically ≥5×).
///
/// If the threshold isn't met, either:
///   - the coordinator isn't being hit (bug in admission path), or
///   - the prompt isn't actually extending (tokenizer quirk), or
///   - the test harness is mismeasuring.
/// Exit 1 surfaces any of these.
func runBatchEngineCacheHit(modelPath: String, maxNew: Int) async throws {
    let modelDir = URL(fileURLWithPath: modelPath)
    print("\n=== BatchEngine cache-hit verification (iter 34) ===")
    print("Loading with real HuggingFace tokenizer...")
    let loadStart = CFAbsoluteTimeGetCurrent()
    let context = try await loadBenchModelContext(from: modelDir)
    print(String(format: "Load: %.2fs", CFAbsoluteTimeGetCurrent() - loadStart))
    print("Model: \(type(of: context.model))")

    // In-memory paged coordinator — no disk I/O, so timings are clean.
    var cfg = CacheCoordinatorConfig()
    cfg.usePagedCache = true
    cfg.enableDiskCache = false
    cfg.pagedBlockSize = 64
    cfg.maxCacheBlocks = 512
    cfg.modelKey = modelDir.lastPathComponent
    let coordinator = CacheCoordinator(config: cfg)

    var params = GenerateParameters(
        maxTokens: maxNew, temperature: 0, prefillStepSize: 512)
    params.extraStopStrings = ["<END>"]

    nonisolated(unsafe) let ctx = context
    let engine = BatchEngine(
        context: ctx, maxBatchSize: 1, cacheCoordinator: coordinator)

    // Key insight 1: this is a cache-coordinator test, not a chat-template
    // test. Mutating a rendered chat prompt after the assistant generation
    // preface appends text inside model-specific reasoning rails and can turn
    // the cache row into a prompt-quality failure. Build raw text prompts at
    // the TOKEN level so turn 2 is a strict extension of turn 1 while still
    // using the loaded model's own tokenizer.
    //
    // Key insight 2: `PagedCacheManager.storeTokenSequence` stores only
    // complete `blockSize`-sized blocks. With the default blockSize=64, a
    // ~58-token prompt stores ZERO blocks (floor(58/64) = 0). We need at
    // least 2 full blocks stored on turn 1 so there's something to hit
    // on turn 2. 200+ tokens at blockSize=64 gives ≥3 blocks cached.
    let turn1Prompt = String(repeating: """
        Cache proof context: the sky has the color blue; grass has the color \
        green; roses have the color red; lemons have the color yellow; snow \
        has the color white. This paragraph is filler for prefix-cache block \
        coverage and is not a dialogue or a list of tasks.
        """, count: 4) + """

        Task: answer with exactly the single word for the sky color, then write <END>.
        Answer:
        """
    let turn1Tokens = context.tokenizer.encode(
        text: turn1Prompt, addSpecialTokens: true)
    let turn1TokensArr = MLXArray(turn1Tokens.map { Int32($0) })[.newAxis, .ellipsis]
    let turn1Input = LMInput(text: LMInput.Text(tokens: turn1TokensArr))
    // Turn 2 = turn 1 tokens + a model-tokenized suffix. This must not use
    // hard-coded token ids: the same integers are not portable across Qwen,
    // MiniMax, Gemma, etc., and can turn a cache test into a garbage-prompt
    // test for non-Qwen tokenizers.
    let followupText = """

        Task: answer with exactly the single word for the grass color, then write <END>.
        Answer:
        """
    let followup = context.tokenizer.encode(
        text: followupText, addSpecialTokens: false)
    if followup.isEmpty {
        fputs("[CacheHit] FAIL: tokenizer produced empty follow-up tokens.\n", stderr)
        exit(2)
    }
    let turn2Tokens: [Int] = turn1Tokens + followup
    let turn2TokensArr = MLXArray(turn2Tokens.map { Int32($0) })[.newAxis, .ellipsis]
    let turn2Input = LMInput(text: LMInput.Text(tokens: turn2TokensArr))

    func runTurn(label: String, input: sending LMInput) async throws
        -> (Double, Int, GenerateCompletionInfo?, String)
    {
        let tokenCount = input.text.tokens.size
        let t0 = CFAbsoluteTimeGetCurrent()
        let stream = await engine.generate(input: input, parameters: params)
        var promptTime: Double = 0
        var text = ""
        var reasoning = ""
        var toolCalls = 0
        var completionInfo: GenerateCompletionInfo?
        for await event in stream {
            switch event {
            case .chunk(let chunk):
                text += chunk
            case .reasoning(let chunk):
                reasoning += chunk
            case .toolCall:
                toolCalls += 1
            case .info(let info):
                completionInfo = info
                promptTime = info.promptTime
            }
        }
        let wall = CFAbsoluteTimeGetCurrent() - t0
        print(String(format:
            "  %@ : prompt=%d tokens, promptTime=%.3fs, genTokens=%d, wall=%.2fs, reasoning=%d, tools=%d",
            label, tokenCount, promptTime, completionInfo?.generationTokenCount ?? 0, wall,
            reasoning.count, toolCalls))
        printDecodedOutput(
            label: "CacheHit \(label)",
            text: text)
        return (promptTime, tokenCount, completionInfo, text)
    }

    nonisolated(unsafe) let turn1Send = turn1Input
    let (turn1Prompt_s, _, turn1Info, turn1Text) = try await runTurn(
        label: "Turn 1 (cold cache)", input: turn1Send)
    if turn1Info?.stopReason == .length || lagunaLoopHeuristic(turn1Text) {
        fputs("[CacheHit] FAIL: turn 1 did not produce a coherent stop-bounded output. " +
              "This row is a cache+coherency gate, not a structural hit-only gate.\n",
              stderr)
        exit(1)
    }

    // Verify our construction produced a true prefix extension, then
    // probe the coordinator directly.
    let isPrefix = turn1Tokens.count <= turn2Tokens.count &&
        Array(turn2Tokens.prefix(turn1Tokens.count)) == turn1Tokens
    print(String(format: "  turn1.count=%d, turn2.count=%d, turn2 starts with turn1? %@",
        turn1Tokens.count, turn2Tokens.count, isPrefix ? "yes" : "NO"))
    if !isPrefix {
        fputs("[CacheHit] FAIL: test harness broken — turn2Tokens is not a prefix of turn1Tokens.\n",
              stderr)
        exit(2)
    }

    if coordinator.isPagedIncompatible {
        print("  Coordinator is paged-incompatible for this model; " +
              "prefix-extension paged cache is intentionally disabled.")
        print("=== BatchEngine cache-hit: not applicable (paged-incompatible topology) ===")
        await engine.shutdown()
        return
    }

    // Directly probe the coordinator with turn 2's token ids BEFORE
    // submitting turn 2. This isolates "does the coordinator have the
    // turn 1 prefix?" from "does BatchEngine apply the hit correctly?".
    // If fetch() returns `.miss`, BatchEngine isn't storing under a key
    // that turn 2 can look up — that's the real failure mode.
    let turn2MediaSalt = computeCacheSalt(for: turn2Input, parameters: params)
    let probeResult = coordinator.fetch(tokens: turn2Tokens, mediaSalt: turn2MediaSalt)
    switch probeResult {
    case .hit(let matched, let remaining, let detail, _, _, _):
        print(String(format:
            "  Coordinator probe: HIT (%@ tier, matched=%d/%d, remaining=%d)",
            detail.rawValue, matched, turn2Tokens.count, remaining.count))
    case .miss:
        fputs("[CacheHit] FAIL: coordinator.fetch(turn2Tokens) returned .miss. " +
              "BatchEngine is not storing under the key that turn 2 looks up. " +
              "Bug is in `finishSlot`'s storeAfterGeneration call or its token-hash.\n",
              stderr)
        exit(1)
    }

    nonisolated(unsafe) let turn2Send = turn2Input
    let (turn2Prompt_s, _, turn2Info, turn2Text) = try await runTurn(
        label: "Turn 2 (warm cache)", input: turn2Send)
    if turn2Info?.stopReason == .length || lagunaLoopHeuristic(turn2Text) {
        fputs("[CacheHit] FAIL: turn 2 hit cache but did not produce a coherent stop-bounded output. " +
              "Do not count prompt-time reduction as production cache proof.\n",
              stderr)
        exit(1)
    }

    // Turn 2 should also be measurably faster because prefill covers
    // only the remaining tokens. Ratio <= 0.75 catches regressions
    // (the paged cache hitting-but-not-saving-compute path); real
    // gains are typically ≥60% reduction.
    //
    // Hybrid SSM exception: partial-hit on hybrid slots rolls back to
    // full prefill by design (SSM recurrence is path-dependent, same
    // class as VL). The coordinator still reports a probe HIT but
    // BatchEngine falls back for correctness — matching prompt times
    // are expected, not a bug. Detect hybrid by checking `coordinator.isHybrid`
    // which the engine auto-flips on admission of any Mamba/SSM slot.
    let ratio = turn1Prompt_s > 0 ? turn2Prompt_s / turn1Prompt_s : 1.0
    print(String(format: "  ratio (turn2/turn1) = %.2f", ratio))
    if coordinator.isHybrid {
        print("  (hybrid SSM model — partial-hit rollback is correct-by-design; " +
              "ratio is informational only)")
    } else if ratio >= 0.75 {
        fputs("[CacheHit] FAIL: turn2 prompt time not < 75% of turn 1 " +
              "(\(turn2Prompt_s)s vs \(turn1Prompt_s)s). " +
              "Coordinator reports hit but BatchEngine isn't using it to skip prefill.\n",
              stderr)
        exit(1)
    }
    await engine.shutdown()
    print("=== BatchEngine cache-hit: passed ===")
}

// MARK: - Disk cache restore across coordinators (iter 35)

/// Models an osaurus session restart. Turn 1 runs with coordinator A
/// (disk-enabled, pointing at a temp dir). Coordinator A is then
/// DISCARDED. A fresh coordinator B is spun up against the same disk
/// dir — as if a new process just started. Turn 2 is submitted through
/// a new BatchEngine bound to coordinator B. If the disk tier works,
/// turn 2 should hit from disk and skip prefill.
///
/// This is the single strongest "does it survive process restart?"
/// property. Paged (RAM) coordinator state disappears at process exit;
/// only the disk tier persists.
func runBatchEngineDiskRestore(modelPath: String, maxNew: Int) async throws {
    let modelDir = URL(fileURLWithPath: modelPath)
    print("\n=== BatchEngine disk-restore verification (iter 35) ===")
    print("Loading with real HuggingFace tokenizer...")
    let loadStart = CFAbsoluteTimeGetCurrent()
    let context = try await loadBenchModelContext(from: modelDir)
    print(String(format: "Load: %.2fs", CFAbsoluteTimeGetCurrent() - loadStart))
    print("Model: \(type(of: context.model))")

    // Disk cache dir — unique per run. Clean up on exit so repeated
    // runs start fresh.
    let diskDir = FileManager.default.temporaryDirectory
        .appendingPathComponent("vmlx-bench-disk-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: diskDir, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: diskDir) }

    func makeCoordinator() -> CacheCoordinator {
        var cfg = CacheCoordinatorConfig()
        cfg.usePagedCache = true  // keep paged on (we're testing disk NOT instead of it)
        cfg.enableDiskCache = true
        cfg.diskCacheDir = diskDir
        cfg.pagedBlockSize = 64
        cfg.maxCacheBlocks = 512
        cfg.modelKey = modelDir.lastPathComponent
        return CacheCoordinator(config: cfg)
    }

    let params = GenerateParameters(
        maxTokens: maxNew, temperature: 0, prefillStepSize: 512)

    // Build a long prompt — ≥2 paged cache blocks so there's a non-trivial
    // amount of KV state to round-trip through disk.
    //
    // IMPORTANT semantics difference from the paged-cache test (iter 34):
    // `DiskCache.fetch` does **exact-or-one-shorter** match, not prefix
    // extension. That matches its intended use case: session resumption
    // (re-opening the same conversation), not turn extension. So this
    // test replays the EXACT same tokens across sessions.
    let basePrompt = String(repeating: """
        You are a careful assistant. Facts to remember across turns: \
        the sky is blue, grass is green, roses are red, oceans are \
        deep, fire is hot. Answer concisely and precisely.
        """, count: 3) + " Q: What is the colour of the sky?"

    let turn1Input = try await context.processor.prepare(
        input: UserInput(prompt: basePrompt))
    let turn1Tokens = turn1Input.text.tokens.reshaped(-1).asArray(Int.self)
    // Session 2 replays the same tokens. Construct a fresh LMInput from
    // the captured ids so the second BatchEngine sees an independent
    // `LMInput` value (consuming `sending` LMInput).
    let turn2Tokens: [Int] = turn1Tokens

    // --- Session 1: store cache entry to disk --------------------------
    nonisolated(unsafe) let ctx1 = context
    let coordA = makeCoordinator()
    let engineA = BatchEngine(
        context: ctx1, maxBatchSize: 1, cacheCoordinator: coordA)

    nonisolated(unsafe) let t1Send = turn1Input
    let t0A = CFAbsoluteTimeGetCurrent()
    let (_, streamA) = await engineA.submit(input: t1Send, parameters: params)
    var idsA: [Int] = []
    var promptTimeA: Double = 0
    for await event in streamA {
        switch event {
        case .token(let id): idsA.append(id)
        case .info(let info): promptTimeA = info.promptTime
        }
    }
    let wallA = CFAbsoluteTimeGetCurrent() - t0A
    print(String(format:
        "  Session 1 (cold, wrote disk): prompt=%d, promptTime=%.3fs, genTokens=%d, wall=%.2fs",
        turn1Tokens.count, promptTimeA, idsA.count, wallA))
    printDecodedOutput(
        label: "DiskRestore session 1 cold",
        text: context.tokenizer.decode(tokenIds: idsA))
    await engineA.shutdown()

    // Allow any disk-flushing async work to complete. On Darwin the
    // safetensors + sqlite writes are sync under the coordinator lock,
    // but defensive yield here.
    await Task.yield()

    // Confirm the disk dir actually got something written to it.
    let diskContents = (try? FileManager.default.contentsOfDirectory(
        at: diskDir, includingPropertiesForKeys: nil)) ?? []
    print("  Disk dir contents: \(diskContents.map { $0.lastPathComponent }.sorted())")
    if diskContents.isEmpty {
        fputs("[DiskRestore] FAIL: no files written to disk dir after session 1. " +
              "BatchEngine's finishSlot didn't call coordinator.storeAfterGeneration " +
              "with disk-tier enabled.\n", stderr)
        exit(1)
    }

    // --- Session 2: fresh coordinator + engine, same disk dir ---------
    //
    // Drop coordinator A and engine A entirely. Coord B is NEW — its
    // paged cache is empty. Only the disk tier carries across.
    let coordB = makeCoordinator()

    // Probe coord B directly with turn 2's tokens BEFORE submit.
    let turn2TokensArr = MLXArray(turn2Tokens.map { Int32($0) })[.newAxis, .ellipsis]
    let turn2Input = LMInput(
        text: LMInput.Text(tokens: turn2TokensArr), image: nil, video: nil,
        cacheScopeSalt: turn1Input.cacheScopeSalt)
    let turn2MediaSalt = computeCacheSalt(for: turn2Input, parameters: params)
    let probe = coordB.fetch(tokens: turn2Tokens, mediaSalt: turn2MediaSalt)
    switch probe {
    case .hit(let matched, _, let detail, _, _, let disk):
        let label = detail.rawValue
        let diskKey = disk != nil ? "yes" : "no"
        print("  Coord B probe: HIT (\(label), matched=\(matched)/\(turn2Tokens.count), diskArrays=\(diskKey))")
        if detail != .disk {
            fputs("[DiskRestore] FAIL: probe hit came from \(label), not disk. " +
                  "Paged cache should be empty for a freshly-constructed coordinator.\n",
                  stderr)
            exit(1)
        }
    case .miss:
        fputs("[DiskRestore] FAIL: fresh coordinator at same disk dir returned .miss. " +
              "Disk writes are not being read back — check TQDiskSerializer / SQLite index.\n",
              stderr)
        exit(1)
    }

    // Actually run turn 2 through a new BatchEngine bound to coordB.
    // We can't reuse engineA because it carries coordA.
    nonisolated(unsafe) let ctx2 = context
    let engineB = BatchEngine(
        context: ctx2, maxBatchSize: 1, cacheCoordinator: coordB)
    nonisolated(unsafe) let t2Send = turn2Input
    let t0B = CFAbsoluteTimeGetCurrent()
    let (_, streamB) = await engineB.submit(input: t2Send, parameters: params)
    var idsB: [Int] = []
    var promptTimeB: Double = 0
    for await event in streamB {
        switch event {
        case .token(let id): idsB.append(id)
        case .info(let info): promptTimeB = info.promptTime
        }
    }
    let wallB = CFAbsoluteTimeGetCurrent() - t0B
    print(String(format:
        "  Session 2 (warm from disk): prompt=%d, promptTime=%.3fs, genTokens=%d, wall=%.2fs",
        turn2Tokens.count, promptTimeB, idsB.count, wallB))
    printDecodedOutput(
        label: "DiskRestore session 2 warm",
        text: context.tokenizer.decode(tokenIds: idsB))

    // Behavioural correctness only: both sessions must generate >0 tokens
    // and the probe must have reported a disk hit. We DO NOT assert on
    // promptTime ratio: the safetensors deserialize cost during restore
    // can dominate a 1-token prefill on small models like Qwen3-0.6B, so
    // on-wire prefill time can be HIGHER after disk restore despite a
    // real cache hit. On larger models this flips — restore saves
    // hundreds of ms of forward-pass compute. The timing tradeoff is a
    // model-size-dependent operational property, not a correctness
    // property of the engine/cache wiring.
    let ratio = promptTimeA > 0 ? promptTimeB / promptTimeA : 1.0
    print(String(format: "  ratio (session2/session1) = %.2f (informational only)", ratio))
    if idsA.isEmpty || idsB.isEmpty {
        fputs("[DiskRestore] FAIL: at least one session generated zero tokens " +
              "(sessionA=\(idsA.count), sessionB=\(idsB.count)). " +
              "Disk restore may have corrupted the cache.\n", stderr)
        exit(1)
    }
    await engineB.shutdown()
    print("=== BatchEngine disk-restore: passed (disk hit fired, both sessions completed) ===")
}

// MARK: - Growing-chat cache reuse (post-answer boundary)

/// Verify the osaurus multi-turn cache shape for path-dependent caches.
/// Turn 2 starts with turn 1's prompt plus the generated assistant answer,
/// so a prompt-boundary-only store is guaranteed to miss. A pass proves the
/// engine stored a post-answer boundary that the next growing prompt can hit.
///
/// Set `BENCH_GROWING_MMAP=1` to use the same low-footprint mmap load path
/// as the perf harness. The default remains the original resident load path.
func runGrowingChatCacheReuse(modelPath: String, maxNew: Int) async throws {
    let modelDir = URL(fileURLWithPath: modelPath)
    let modelName = modelDir.lastPathComponent
    let env = ProcessInfo.processInfo.environment
    let cacheDir = URL(fileURLWithPath:
        env["BENCH_GROWING_CACHE_DIR"] ??
        "/tmp/vmlx-growing-chat-cache-\(modelName)-\(UUID().uuidString)")
    try? FileManager.default.createDirectory(
        at: cacheDir, withIntermediateDirectories: true)
    defer {
        if env["BENCH_KEEP_GROWING_CACHE"] != "1" {
            try? FileManager.default.removeItem(at: cacheDir)
        }
    }

    print("\n=== BENCH_GROWING_CHAT_CACHE — \(modelName) ===")
    print("Cache dir: \(cacheDir.path)")
    let nativeMTPDepth = env["BENCH_GROWING_NATIVE_MTP_DEPTH"].flatMap(Int.init)
    let useMmap = env["BENCH_GROWING_MMAP"] == "1"
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
    } else if useMmap {
        let loaded = try await MLXLMCommon.loadModel(
            from: modelDir,
            using: #huggingFaceTokenizerLoader(),
            loadConfiguration: LoadConfiguration(useMmapSafetensors: true))
        context = loaded.0
    } else {
        context = try await MLXLMCommon.loadModel(
            from: modelDir, using: #huggingFaceTokenizerLoader())
    }
    print(String(format: "Load: %.2fs  Model: %@",
        CFAbsoluteTimeGetCurrent() - loadStart,
        String(describing: type(of: context.model))))
    print("Load mode: \(useMmap || nativeMTPDepth != nil ? "mmap" : "resident")")
    print("Tool format: \(context.configuration.toolCallFormat.map { "\($0)" } ?? "json")")
    print("Reasoning stamp: \(context.configuration.reasoningParserName ?? "nil")")
    print("Native MTP depth: \(nativeMTPDepth.map(String.init) ?? "off")")

    let coordinator = CacheCoordinator(config: CacheCoordinatorConfig(
        usePagedCache: true,
        enableDiskCache: true,
        pagedBlockSize: 64,
        maxCacheBlocks: 512,
        diskCacheMaxGB: 4.0,
        diskCacheDir: cacheDir,
        ssmMaxEntries: 64,
        modelKey: modelName))

    let budget = max(maxNew, 48)
    var params: GenerateParameters
    if env["BENCH_GROWING_BUNDLE_DEFAULTS"] == "1" {
        let fallback = GenerateParameters(
            maxTokens: budget,
            randomSeed: env["BENCH_GROWING_SEED"].flatMap(UInt64.init),
            prefillStepSize: 512)
        params = GenerateParameters(
            generationConfig: context.configuration.generationDefaults,
            fallback: fallback)
        params.maxTokens = budget
        params.randomSeed = env["BENCH_GROWING_SEED"].flatMap(UInt64.init)
        params.prefillStepSize = 512
    } else {
        params = GenerateParameters(
            maxTokens: budget, temperature: 0, prefillStepSize: 512)
    }
    if let nativeMTPDepth {
        params.draftStrategy = .nativeMTP(depth: nativeMTPDepth)
    }
    print(String(format:
        "Sampling: mode=%@ maxTokens=%d temp=%.3f topP=%.3f topK=%d minP=%.3f rep=%@",
        env["BENCH_GROWING_BUNDLE_DEFAULTS"] == "1" ? "bundle-defaults" : "explicit-greedy",
        params.maxTokens ?? -1,
        Double(params.temperature),
        Double(params.topP),
        params.topK,
        Double(params.minP),
        params.repetitionPenalty.map { String(format: "%.3f", Double($0)) } ?? "nil"))

    let topology = ModelCacheTopologySnapshot(cache: context.model.newCache(parameters: params))
    print("  Cache topology: \(topology.topologyTags.joined(separator: ","))")
    nonisolated(unsafe) let ctx = context
    let engine = BatchEngine(
        context: ctx, maxBatchSize: 1, cacheCoordinator: coordinator)

    let prefixRepeat = Int(env["BENCH_GROWING_PREFIX_REPEAT"] ?? "0") ?? 0
    let longPrefix = prefixRepeat > 0
        ? String(repeating:
            "Project context: Osaurus is testing cache reuse, reasoning routing, tool safety, and stable multi-turn local inference. ",
            count: prefixRepeat)
        : ""
    let recallPhrase = env["BENCH_GROWING_RECALL_PHRASE"] ?? "vmlx-cache-green"
    let firstTurnPrompt = longPrefix
        + "Reply with exactly this phrase and nothing else: \(recallPhrase)"
    let messages: [[String: any Sendable]] = [
        ["role": "user", "content": firstTurnPrompt]
    ]
    let promptTokens = try context.tokenizer.applyChatTemplate(
        messages: messages,
        tools: nil,
        additionalContext: ["enable_thinking": false])
    let historyBoundaryTokens = try? (
        context.tokenizer as? GenerationPromptControllableTokenizer
    )?.applyChatTemplate(
        messages: messages,
        tools: nil,
        additionalContext: ["enable_thinking": false],
        addGenerationPrompt: false)
    let cachePrefixTokenCounts: [Int]
    if let historyBoundaryTokens,
       !historyBoundaryTokens.isEmpty,
       historyBoundaryTokens.count < promptTokens.count,
       promptTokens.prefix(historyBoundaryTokens.count).elementsEqual(historyBoundaryTokens)
    {
        cachePrefixTokenCounts = [historyBoundaryTokens.count]
    } else {
        cachePrefixTokenCounts = []
    }
    let promptArray = MLXArray(promptTokens.map { Int32($0) })
        .reshaped(1, promptTokens.count)
    let turn1 = LMInput(
        text: LMInput.Text(tokens: promptArray),
        cacheScopeSalt: cacheScopeSalt(from: ["enable_thinking": false]),
        cachePrefixTokenCounts: cachePrefixTokenCounts)
    print("  Cache history-boundary counts: \(cachePrefixTokenCounts)")

    func run(label: String, input: sending LMInput) async throws
        -> (tokens: [Int], info: GenerateCompletionInfo?, wall: Double)
    {
        let promptSize = input.text.tokens.size
        let t0 = CFAbsoluteTimeGetCurrent()
        var out: [Int] = []
        var info: GenerateCompletionInfo?

        if nativeMTPDepth != nil {
            let (stream, task) = try generateTokensTask(
                input: input,
                parameters: params,
                context: ctx,
                cacheCoordinator: coordinator)
            for await event in stream {
                switch event {
                case .token(let id):
                    out.append(id)
                case .info(let i):
                    info = i
                }
            }
            await task.value
        } else {
            let (_, stream) = await engine.submit(input: input, parameters: params)
            for await event in stream {
                switch event {
                case .token(let id):
                    out.append(id)
                case .info(let i):
                    info = i
                }
            }
        }
        let wall = CFAbsoluteTimeGetCurrent() - t0
        let text = context.tokenizer.decode(tokenIds: out)
        let tokps = wall > 0 ? Double(out.count) / wall : 0
        print(String(format:
            "  %@: prompt=%d gen=%d finish=%@ promptTime=%.3fs wall=%.2fs tokps=%.2f text=\"%@\"",
            label,
            promptSize,
            out.count,
            String(describing: info?.stopReason ?? .cancelled),
            info?.promptTime ?? -1,
            wall,
            tokps,
            String(text.prefix(120)).replacingOccurrences(of: "\n", with: "\\n")))
        return (out, info, wall)
    }

    nonisolated(unsafe) let t1Send = turn1
    let r1 = try await run(label: "Turn 1 cold", input: t1Send)
    guard let info1 = r1.info else {
        throw NSError(domain: "BENCH_GROWING_CHAT_CACHE", code: 1,
            userInfo: [NSLocalizedDescriptionKey: "turn 1 emitted no completion info"])
    }
    guard info1.stopReason == .stop else {
        throw NSError(domain: "BENCH_GROWING_CHAT_CACHE", code: 2,
            userInfo: [NSLocalizedDescriptionKey:
                "turn 1 ended with \(info1.stopReason), not .stop; post-answer boundary is intentionally not stored for length/cancel"])
    }
    guard !r1.tokens.isEmpty else {
        throw NSError(domain: "BENCH_GROWING_CHAT_CACHE", code: 3,
            userInfo: [NSLocalizedDescriptionKey: "turn 1 emitted zero visible tokens"])
    }

    func writeDiagnostic(_ message: String) {
        FileHandle.standardError.write(Data((message + "\n").utf8))
    }

    func diskArraySummary(_ arrays: [String: MLXArray]?) -> String {
        guard let arrays else { return "no" }
        let keys = arrays.keys
        let dsv4Keys = keys.filter { $0.contains("_keys") }.count
        let dsv4Values = keys.filter { $0.contains("_values") }.count
        let dsv4PoolComp = keys.filter { $0.contains("_pool_comp") }.count
        let dsv4PoolIdx = keys.filter { $0.contains("_pool_idx") }.count
        let dsv4BufferComp = keys.filter { $0.contains("_buf_comp") }.count
        let dsv4BufferIdx = keys.filter { $0.contains("_buf_idx") }.count
        if dsv4PoolComp > 0 || dsv4PoolIdx > 0 {
            return "dsv4(keys=\(dsv4Keys),values=\(dsv4Values),poolComp=\(dsv4PoolComp),poolIdx=\(dsv4PoolIdx),bufComp=\(dsv4BufferComp),bufIdx=\(dsv4BufferIdx),total=\(arrays.count))"
        }
        return "yes(total=\(arrays.count))"
    }

    func describeProbe(_ label: String, tokens: [Int], mediaSalt: String?) {
        switch coordinator.fetch(tokens: tokens, mediaSalt: mediaSalt) {
        case .hit(let matched, let remaining, let detail, _, _, let diskArrays):
            writeDiagnostic(String(format:
                "  %@: HIT tier=%@ matched=%d/%d remaining=%d diskArrays=%@",
                label,
                detail.rawValue,
                matched,
                tokens.count,
                remaining.count,
                diskArraySummary(diskArrays)))
        case .miss:
            writeDiagnostic("  \(label): MISS tokens=\(tokens.count)")
        }
    }

    func printCoordinatorStats(_ label: String) {
        let snapshot = coordinator.snapshotStats()
        let paged = snapshot.pagedStats.map {
            "hits=\($0.cacheHits),misses=\($0.cacheMisses),allocated=\($0.allocatedBlocks),free=\($0.freeBlocks),evictions=\($0.evictions)"
        } ?? "disabled"
        let disk = snapshot.diskStats.map {
            "hits=\($0.hits),misses=\($0.misses),stores=\($0.stores),maxBytes=\($0.maxSizeBytes)"
        } ?? "disabled"
        let ssm = snapshot.ssmStats
        print(
            "  \(label) cache stats: hybrid=\(snapshot.isHybrid) pagedIncompatible=\(snapshot.isPagedIncompatible) paged{\(paged)} disk{\(disk)} ssm{hits=\(ssm.hits),misses=\(ssm.misses),reDerives=\(ssm.reDerives)}")
    }

    let turn1Text = context.tokenizer.decode(tokenIds: r1.tokens)
        .trimmingCharacters(in: .whitespacesAndNewlines)
    if !turn1Text.lowercased().contains(recallPhrase.lowercased()) {
        throw NSError(domain: "BENCH_GROWING_CHAT_CACHE", code: 10,
            userInfo: [NSLocalizedDescriptionKey:
                "turn 1 did not emit recall phrase \(recallPhrase.debugDescription); got \(String(turn1Text.prefix(160)).debugDescription)"])
    }
    let turn2Messages: [[String: any Sendable]] = [
        ["role": "user", "content": firstTurnPrompt],
        ["role": "assistant", "content": turn1Text],
        ["role": "user", "content": "What exact phrase did you just answer? Reply with only that phrase."],
    ]
    let turn2Tokens = try context.tokenizer.applyChatTemplate(
        messages: turn2Messages,
        tools: nil,
        additionalContext: ["enable_thinking": false])
    let turn2RenderedTail = context.tokenizer.decode(
        tokenIds: Array(turn2Tokens.suffix(160)),
        skipSpecialTokens: false)
        .replacingOccurrences(of: "\n", with: "\\n")
    print("  Turn 2 rendered tail: \"\(turn2RenderedTail)\"")
    let turn2Array = MLXArray(turn2Tokens.map { Int32($0) })
        .reshaped(1, turn2Tokens.count)
    let turn2 = LMInput(
        text: LMInput.Text(tokens: turn2Array),
        cacheScopeSalt: turn1.cacheScopeSalt)

    func commonPrefixCount(_ lhs: [Int], _ rhs: [Int]) -> Int {
        let limit = Swift.min(lhs.count, rhs.count)
        var idx = 0
        while idx < limit, lhs[idx] == rhs[idx] {
            idx += 1
        }
        return idx
    }

    func decodedWindow(_ tokens: [Int], around index: Int) -> String {
        let lower = Swift.max(0, index - 16)
        let upper = Swift.min(tokens.count, index + 16)
        guard lower < upper else { return "" }
        return context.tokenizer.decode(
            tokenIds: Array(tokens[lower..<upper]),
            skipSpecialTokens: false
        )
        .replacingOccurrences(of: "\n", with: "\\n")
    }

    let turn2Salt = computeCacheSalt(for: turn2, parameters: params)
    let boundaryTokens = promptTokens + r1.tokens
    let promptCommon = commonPrefixCount(turn2Tokens, promptTokens)
    let boundaryCommon = commonPrefixCount(turn2Tokens, boundaryTokens)
    writeDiagnostic(
        "  Native turn-2 common prefix: prompt=\(promptCommon)/\(promptTokens.count) postAnswer=\(boundaryCommon)/\(boundaryTokens.count)")
    if promptCommon < promptTokens.count {
        writeDiagnostic(
            "  Native turn-2 diverged before prompt boundary at token \(promptCommon)")
        writeDiagnostic(
            "    stored prompt window: \"\(decodedWindow(promptTokens, around: promptCommon))\"")
        writeDiagnostic(
            "    turn-2 prompt window: \"\(decodedWindow(turn2Tokens, around: promptCommon))\"")
    } else if boundaryCommon < boundaryTokens.count {
        writeDiagnostic(
            "  Native turn-2 diverged before post-answer boundary at token \(boundaryCommon)")
        writeDiagnostic(
            "    stored post-answer window: \"\(decodedWindow(boundaryTokens, around: boundaryCommon))\"")
        writeDiagnostic(
            "    turn-2 prompt window: \"\(decodedWindow(turn2Tokens, around: boundaryCommon))\"")
    }
    writeDiagnostic("  Coordinator flags: hybrid=\(coordinator.isHybrid) pagedIncompatible=\(coordinator.isPagedIncompatible)")
    describeProbe("Prompt-boundary salted probe", tokens: promptTokens + [-1], mediaSalt: turn2Salt)
    describeProbe("Post-answer salted probe", tokens: boundaryTokens + [-1], mediaSalt: turn2Salt)
    describeProbe("Prompt-boundary nil-salt probe", tokens: promptTokens + [-1], mediaSalt: nil)
    describeProbe("Post-answer nil-salt probe", tokens: boundaryTokens + [-1], mediaSalt: nil)
    printCoordinatorStats("Before turn 2")
    let probe = coordinator.fetch(tokens: turn2Tokens, mediaSalt: turn2Salt)
    switch probe {
    case .hit(let matched, let remaining, let detail, _, _, let diskArrays):
        print(String(format:
            "  Coordinator probe before turn 2: HIT tier=%@ matched=%d/%d remaining=%d diskArrays=%@",
            detail.rawValue,
            matched,
            turn2Tokens.count,
            remaining.count,
            diskArraySummary(diskArrays)))
        let expectedPromptBoundary = promptCommon < promptTokens.count
            ? (cachePrefixTokenCounts.max() ?? promptTokens.count)
            : promptTokens.count
        let expectedPostAnswerBoundary = promptTokens.count + r1.tokens.count
        if matched < expectedPromptBoundary {
            throw NSError(domain: "BENCH_GROWING_CHAT_CACHE", code: 4,
                userInfo: [NSLocalizedDescriptionKey:
                    "probe hit did not reach the safe prompt/cache boundary: matched=\(matched), expected at least \(expectedPromptBoundary)"])
        }
        if matched < promptTokens.count {
            print("  Probe note: native turn-2 chat template diverged from the generation prompt; matched canonical history boundary \(matched) instead of full prompt \(promptTokens.count).")
        } else if matched < expectedPostAnswerBoundary {
            print("  Probe note: native turn-2 chat template matched the prompt boundary but not the raw post-answer boundary; this is expected when the template re-wraps the assistant turn.")
        }
    case .miss:
        if promptCommon < promptTokens.count {
            throw NSError(domain: "BENCH_GROWING_CHAT_CACHE", code: 11,
                userInfo: [NSLocalizedDescriptionKey:
                    "native turn-2 chat template diverged from the cached turn-1 generation prompt before the prompt boundary; storing the raw turn-1 KV under turn-2 tokens would be unsafe"])
        }
        if boundaryCommon < boundaryTokens.count {
            throw NSError(domain: "BENCH_GROWING_CHAT_CACHE", code: 12,
                userInfo: [NSLocalizedDescriptionKey:
                    "native turn-2 chat template matched the prompt boundary but diverged before the raw post-answer boundary; this family needs a canonical history-boundary cache repair before stateless growing-chat L2 can hit"])
        }
        throw NSError(domain: "BENCH_GROWING_CHAT_CACHE", code: 5,
            userInfo: [NSLocalizedDescriptionKey:
                "coordinator missed turn 2 growing prompt; post-answer boundary was not stored under the key turn 2 uses"])
    }

    nonisolated(unsafe) let t2Send = turn2
    let r2 = try await run(label: "Turn 2 growing", input: t2Send)
    guard let info2 = r2.info, info2.stopReason == .stop else {
        throw NSError(domain: "BENCH_GROWING_CHAT_CACHE", code: 6,
            userInfo: [NSLocalizedDescriptionKey:
                "turn 2 ended with \(String(describing: r2.info?.stopReason)), not .stop"])
    }
    let turn2Text = context.tokenizer.decode(tokenIds: r2.tokens)
    let turn2Lower = turn2Text.lowercased()
    let contradictionMarkers = [
        "didn't provide a previous answer",
        "did not provide a previous answer",
        "don't have previous answers",
        "do not have previous answers",
        "no previous answer",
        "newly created assistant",
    ]
    if contradictionMarkers.contains(where: { turn2Lower.contains($0) }) {
        throw NSError(domain: "BENCH_GROWING_CHAT_CACHE", code: 9,
            userInfo: [NSLocalizedDescriptionKey:
                "turn 2 contradicted available assistant history \(turn1Text.debugDescription); got \(String(turn2Text.prefix(180)).debugDescription)"])
    }
    guard !turn2Lower.contains("\nuser:") && !turn2Lower.contains("\nassistant:") else {
        throw NSError(domain: "BENCH_GROWING_CHAT_CACHE", code: 7,
            userInfo: [NSLocalizedDescriptionKey:
                "turn 2 leaked raw role markers into output: \(String(turn2Text.prefix(160)))"])
    }
    if !turn2Lower.contains(recallPhrase.lowercased()) {
        throw NSError(domain: "BENCH_GROWING_CHAT_CACHE", code: 8,
            userInfo: [NSLocalizedDescriptionKey:
                "turn 2 did not refer to previous answer \(turn1Text.debugDescription); got \(String(turn2Text.prefix(160)).debugDescription)"])
    }
    let ratio = (r1.info?.promptTime ?? 0) > 0
        ? (r2.info?.promptTime ?? 0) / (r1.info?.promptTime ?? 1)
        : 1.0
    print(String(format: "  promptTime ratio turn2/turn1 = %.2f", ratio))
    printCoordinatorStats("After turn 2")
    await engine.shutdown()
    print("=== BENCH_GROWING_CHAT_CACHE: passed ===")
}

// MARK: - Per-slot sampling divergence (iter 36)

/// Submit two slots to the same B=2 BatchEngine where each slot carries
/// DIFFERENT `GenerateParameters`. Verify both slots' samplers actually
/// fire with their respective settings:
///
/// - Slot 0: temp=0 (greedy). Re-running the same B=2 shape with the same
///   companion slot must produce byte-identical tokens.
/// - Slot 1: temp=0.8 topP=0.9 with a fixed seed. Re-running the same B=2
///   shape must produce byte-identical tokens, and the output must differ from
///   the greedy path by at least one token (otherwise the sampler didn't
///   actually kick in, or both slots share a sampler instance).
func runBatchEnginePerSlotSampler(modelPath: String, maxNew: Int) async throws {
    let modelDir = URL(fileURLWithPath: modelPath)
    print("\n=== BatchEngine per-slot sampler (iter 36) ===")
    print("Loading with real HuggingFace tokenizer...")
    let loadStart = CFAbsoluteTimeGetCurrent()
    let context = try await MLXLMCommon.loadModel(
        from: modelDir, using: #huggingFaceTokenizerLoader())
    print(String(format: "Load: %.2fs", CFAbsoluteTimeGetCurrent() - loadStart))
    print("Model: \(type(of: context.model))")

    // Same prompt into both slots — eliminates "different outputs came
    // from different inputs" as an alternative explanation. Any divergence
    // must come from the sampler.
    let prompt = "Write five words describing a red apple."
    let baseInput = try await context.processor.prepare(
        input: UserInput(prompt: prompt))
    let promptTokens = baseInput.text.tokens.reshaped(-1).asArray(Int.self)
    print("  prompt: \"\(prompt)\" (\(promptTokens.count) tokens)")

    func freshInput() -> LMInput {
        let arr = MLXArray(promptTokens.map { Int32($0) })[.newAxis, .ellipsis]
        return LMInput(
            text: LMInput.Text(tokens: arr), image: nil, video: nil)
    }

    // Two distinct parameter profiles.
    var greedyParams = GenerateParameters(
        maxTokens: maxNew, temperature: 0, prefillStepSize: 512)
    greedyParams.topP = 1.0
    var stochasticParams = GenerateParameters(
        maxTokens: maxNew, temperature: 0.8, prefillStepSize: 512)
    stochasticParams.topP = 0.9
    stochasticParams.randomSeed = 36

    nonisolated(unsafe) let ctx = context

    func runPair(label: String) async -> ([Int], [Int]) {
        let engine = BatchEngine(context: ctx, maxBatchSize: 2)
        nonisolated(unsafe) let in0 = freshInput()
        nonisolated(unsafe) let in1 = freshInput()
        let (_, s0) = await engine.submit(input: in0, parameters: greedyParams)
        let (_, s1) = await engine.submit(input: in1, parameters: stochasticParams)
        let results = await collectBatchStreamsWithOverlap(
            engine,
            streams: [(0, s0), (1, s1)],
            maxTokens: maxNew,
            label: label)
        await engine.shutdown()
        return (results[0].1, results[1].1)
    }

    // Submit both concurrently — same engine, same model, same prompt,
    // different params.
    let (greedyTokens, stochasticTokens) = await runPair(label: "PerSlotSampler B=2")
    print("  slot 0 (temp=0)    first 15: \(Array(greedyTokens.prefix(15)))")
    print("  slot 1 (temp=0.8, seed=36) first 15: \(Array(stochasticTokens.prefix(15)))")

    // Re-run the identical B=2 shape with a fresh engine. This is the
    // load-bearing determinism check: it avoids conflating true per-slot
    // sampler/state bugs with legitimate B=1-vs-B=2 low-bit tie-breaking.
    let (greedyRecheck, stochasticRecheck) = await runPair(
        label: "PerSlotSampler B=2 recheck")
    print("  slot 0 B=2 re-run  first 15: \(Array(greedyRecheck.prefix(15)))")
    print("  slot 1 B=2 re-run  first 15: \(Array(stochasticRecheck.prefix(15)))")

    // Assertions.
    // 1. Greedy path must be deterministic across identical B=2 runs.
    if greedyTokens != greedyRecheck {
        fputs("[PerSlotSampler] FAIL: greedy slot 0 diverged between runs — " +
              "\(greedyTokens.count) vs \(greedyRecheck.count) tokens, " +
              "first diff at \(firstDiffIndex(greedyTokens, greedyRecheck) ?? -1). " +
              "Either temp=0 isn't really greedy or slot state bled across identical B=2 submissions.\n",
              stderr)
        exit(1)
    }
    // 2. Fixed-seed stochastic path must also be deterministic across
    //    identical B=2 runs. This proves the sampler's random state is
    //    request-local rather than shared across slots.
    if stochasticTokens != stochasticRecheck {
        fputs("[PerSlotSampler] FAIL: fixed-seed stochastic slot diverged between B=2 runs — " +
              "\(stochasticTokens.count) vs \(stochasticRecheck.count) tokens, " +
              "first diff at \(firstDiffIndex(stochasticTokens, stochasticRecheck) ?? -1). " +
              "Sampler RNG state is not request-local.\n",
              stderr)
        exit(1)
    }
    // 3. Stochastic path must differ from greedy on at least one token.
    //    (Not guaranteed on every run — sampling MAY happen to match the
    //    greedy choice — but on a real prompt with temp=0.8 and ≥20 tokens
    //    the probability of byte-for-byte match is effectively zero.)
    if stochasticTokens == greedyTokens {
        fputs("[PerSlotSampler] WARN: stochastic slot 1 matched greedy byte-for-byte. " +
              "Either slot 1's params didn't apply (BUG) or sampling happened to " +
              "agree with greedy on every token (very unlikely — try maxNew>=30).\n",
              stderr)
        exit(1)
    }
    let firstDiff = firstDiffIndex(greedyTokens, stochasticTokens) ?? -1
    print("  first divergence greedy vs stochastic: index \(firstDiff)")
    print("=== BatchEngine per-slot sampler: passed ===")
}

/// Return the index of the first position where `a[i] != b[i]`, or nil
/// if they agree on every overlapping index and have the same length.
private func firstDiffIndex(_ a: [Int], _ b: [Int]) -> Int? {
    let n = min(a.count, b.count)
    for i in 0..<n where a[i] != b[i] { return i }
    return a.count == b.count ? nil : n
}

/// Drain several BatchEngine raw token streams while concurrently observing
/// the scheduler high-water mark. Starting the consumers before the overlap
/// gate matters for models whose scheduling loop only makes progress once the
/// returned streams are being drained.
private func collectBatchStreamsWithOverlap(
    _ engine: BatchEngine,
    streams: [(Int, AsyncStream<BatchGeneration>)],
    maxTokens: Int,
    label: String,
    atLeast expected: Int? = nil
) async -> [(Int, [Int])] {
    let required = expected ?? streams.count
    async let overlap = observeActiveSlotOverlap(engine, atLeast: required, label: label)
    let results = await withTaskGroup(of: (Int, [Int]).self) { group in
        for (slot, stream) in streams {
            group.addTask {
                var ids: [Int] = []
                for await e in stream {
                    if case .token(let id) = e { ids.append(id) }
                    if ids.count >= maxTokens { break }
                }
                return (slot, ids)
            }
        }
        var collected: [(Int, [Int])] = []
        for await result in group {
            collected.append(result)
        }
        return collected.sorted { $0.0 < $1.0 }
    }
    if !(await overlap) {
        exit(1)
    }
    return results
}

private func collectBatchStreamsWithOverlapAndInfo(
    _ engine: BatchEngine,
    streams: [(Int, AsyncStream<BatchGeneration>)],
    label: String,
    atLeast expected: Int? = nil
) async -> [(Int, [Int], GenerateCompletionInfo?)] {
    let required = expected ?? streams.count
    async let overlap = observeActiveSlotOverlap(engine, atLeast: required, label: label)
    let results = await withTaskGroup(of: (Int, [Int], GenerateCompletionInfo?).self) { group in
        for (slot, stream) in streams {
            group.addTask {
                var ids: [Int] = []
                var info: GenerateCompletionInfo?
                for await e in stream {
                    switch e {
                    case .token(let id):
                        ids.append(id)
                    case .info(let completionInfo):
                        info = completionInfo
                    }
                }
                return (slot, ids, info)
            }
        }
        var collected: [(Int, [Int], GenerateCompletionInfo?)] = []
        for await result in group {
            collected.append(result)
        }
        return collected.sorted { $0.0 < $1.0 }
    }
    if !(await overlap) {
        exit(1)
    }
    return results
}

/// Require the scheduler to show real multi-slot admission before a
/// BatchEngine row is allowed to claim B>1 coverage. This does not prove every
/// decode step had width B, but it rejects the common false positive where a
/// concurrent-looking harness accidentally drains one request at a time.
private func requireActiveSlotOverlap(
    _ engine: BatchEngine,
    atLeast expected: Int,
    label: String,
    timeoutSeconds: Double = 5
) async {
    if !(await observeActiveSlotOverlap(engine, atLeast: expected, label: label,
                                       timeoutSeconds: timeoutSeconds)) {
        exit(1)
    }
}

private func observeActiveSlotOverlap(
    _ engine: BatchEngine,
    atLeast expected: Int,
    label: String,
    timeoutSeconds: Double = 5
) async -> Bool {
    guard expected > 1 else { return true }

    let deadline = CFAbsoluteTimeGetCurrent() + timeoutSeconds
    var maxActive = 0
    var highWatermark = 0
    var maxPending = 0
    while CFAbsoluteTimeGetCurrent() < deadline {
        let active = await engine.activeCount
        let observed = await engine.activeCountHighWatermarkForDiagnostics
        let pending = await engine.pendingCount
        maxActive = max(maxActive, active)
        highWatermark = max(highWatermark, observed)
        maxPending = max(maxPending, pending)
        if active >= expected || observed >= expected {
            print("  \(label): active-slot overlap confirmed " +
                  "(active=\(active), highWatermark=\(observed), pending=\(pending))")
            return true
        }
        try? await Task.sleep(nanoseconds: 20_000_000)
    }

    fputs("[\(label)] FAIL: activeCount never reached \(expected) within \(timeoutSeconds)s " +
          "(maxActive=\(maxActive), highWatermark=\(highWatermark), maxPending=\(maxPending)). " +
          "This row cannot be used as real continuous-batching proof.\n", stderr)
    return false
}

private func printDecodedOutput(label: String, text: String) {
    print("    BEGIN_FULL_TEXT[\(label)]")
    if text.isEmpty {
        print("    <empty>")
    } else {
        print(text)
    }
    print("    END_FULL_TEXT[\(label)]")
}

// MARK: - TurboQuant under B=2 (iter 38)

/// Two concurrent slots on the same BatchEngine with heterogeneous
/// `kvMode`: slot 0 runs with plain float KV, slot 1 runs with
/// `turboQuant(keyBits: 3, valueBits: 3)`. Verifies:
/// - Stage 0 compression (`BatchQuantize.maybeCompress`) fires per-slot
///   post-prefill, does not cross-contaminate.
/// - Both streams complete with stop-bounded coherent output.
/// - Running two TQ slots concurrently (the second pass below) also
///   completes without corruption — proving per-slot `TurboQuantKVCache`
///   state is independent.
///
/// We can't cheaply introspect the cache type post-run from outside the
/// BatchEngine actor, but the "no crash + stop-bounded coherent text" combo
/// plus the existing `CompilableTurboQuantKVCacheTests` FP precision probes
/// give a high-confidence end-to-end check.
func runBatchEngineTurboQuantB2(modelPath: String, maxNew: Int) async throws {
    let modelDir = URL(fileURLWithPath: modelPath)
    print("\n=== BatchEngine TurboQuant B=2 (iter 38) ===")
    print("Loading with real HuggingFace tokenizer...")
    let loadStart = CFAbsoluteTimeGetCurrent()
    let context = try await MLXLMCommon.loadModel(
        from: modelDir, using: #huggingFaceTokenizerLoader())
    print(String(format: "Load: %.2fs", CFAbsoluteTimeGetCurrent() - loadStart))
    print("Model: \(type(of: context.model))")

    nonisolated(unsafe) let ctx = context
    let tokenizer = context.tokenizer

    // Prompt pair long enough to leave exact sink + exact recent tail tokens
    // while still crossing the TurboQuant middle-compression threshold. The
    // final instruction stays in the exact tail; older filler is the region
    // the live KV codec is supposed to compress.
    let tqFiller = Array(repeating:
        "Background cache-validation note: keep following the final user instruction exactly.",
        count: 12
    ).joined(separator: " ")
    let prompts = [
        "\(tqFiller)\nFinal instruction: Reply with exactly these five country names, one per line, and no other text: France; Germany; Italy; Spain; Sweden.",
        "\(tqFiller)\nFinal instruction: Reply with exactly these five adjectives, comma-separated, and no other text: warm, bright, fresh, quiet, golden.",
    ]
    func makeVisibleAnswerInput(_ prompt: String) async throws -> LMInput {
        var input = UserInput(prompt: prompt)
        input.additionalContext = ["enable_thinking": false]
        return try await context.processor.prepare(input: input)
    }

    func requireStopBounded(
        label: String,
        tokenIds ids: [Int],
        info: GenerateCompletionInfo?,
        text: String
    ) {
        if ids.isEmpty {
            fputs("[TQ B=2] FAIL: \(label) produced zero tokens.\n", stderr)
            exit(1)
        }
        guard let info else {
            fputs("[TQ B=2] FAIL: \(label) did not emit GenerateCompletionInfo.\n", stderr)
            exit(1)
        }
        guard info.stopReason == .stop else {
            fputs("[TQ B=2] FAIL: \(label) ended with \(info.stopReason), not .stop. " +
                  "This row is a coherence gate, not a structural non-empty gate.\n",
                  stderr)
            exit(1)
        }
        if info.unclosedReasoning {
            fputs("[TQ B=2] FAIL: \(label) ended with unclosed reasoning.\n", stderr)
            exit(1)
        }
        if lagunaLoopHeuristic(text) {
            fputs("[TQ B=2] FAIL: \(label) triggered loop heuristic.\n", stderr)
            exit(1)
        }
    }

    var inputs: [LMInput] = []
    for p in prompts {
        inputs.append(try await makeVisibleAnswerInput(p))
    }

    let plainParams = GenerateParameters(
        maxTokens: maxNew, temperature: 0, prefillStepSize: 512)
    // kvMode defaults to plain on `plainParams`.
    // 4-bit TurboQuant. 3-bit TQ is too aggressive for small models like
    // Qwen3-0.6B — observed garbage output ("repeated newlines",
    // "ائيةء إلى إلى") on B=1 and B=2. 4-bit TQ is the minimum that
    // preserves coherence at this model size. Larger models (≥7B) handle
    // 3-bit TQ; the probe suite (CompilableTurboQuantKVCacheTests) uses
    // synthetic tensors where quantization noise doesn't matter.
    var tqParams = GenerateParameters(
        maxTokens: maxNew, temperature: 0, prefillStepSize: 512)
    tqParams.kvMode = .turboQuant(keyBits: 4, valueBits: 4)
    tqParams.quantizedKVStart = 8

    // Reference run: slot 0 plain KV ALONE. Gives us the expected
    // plain-output baseline to compare against "slot 0 plain beside a
    // TQ neighbour" — cross-slot corruption would show up as drift.
    print("\n[Reference] slot 0 plain KV, solo (B=1)")
    let engineRef = BatchEngine(context: ctx, maxBatchSize: 1)
    var refInputs: [LMInput] = []
    for p in prompts {
        refInputs.append(try await makeVisibleAnswerInput(p))
    }
    nonisolated(unsafe) let inRef = refInputs[0]
    let (_, streamRef) = await engineRef.submit(input: inRef, parameters: plainParams)
    var refTokens: [Int] = []
    var refInfo: GenerateCompletionInfo?
    for await e in streamRef {
        switch e {
        case .token(let id):
            refTokens.append(id)
        case .info(let info):
            refInfo = info
        }
    }
    await engineRef.shutdown()
    let refText = tokenizer.decode(tokenIds: refTokens)
    print("  reference plain solo: \(refTokens.count) tokens, first 8: \(Array(refTokens.prefix(8)))")
    printDecodedOutput(label: "TQ reference plain solo", text: refText)
    requireStopBounded(
        label: "reference plain solo",
        tokenIds: refTokens,
        info: refInfo,
        text: refText)

    // Reference run: same B=2 scheduler shape as the heterogeneous probe, but
    // both slots use plain KV. This is the isolation baseline. Some quantized
    // families have legitimate B=1-vs-B=2 numeric tie breaks deep in an open
    // generation; a TurboQuant neighbour should still match this B=2 plain
    // reference exactly for the plain slot.
    print("\n[Reference] slot 0 plain KV beside slot 1 plain KV (B=2)")
    let engineRefB2 = BatchEngine(context: ctx, maxBatchSize: 2)
    var refB2Inputs: [LMInput] = []
    for p in prompts {
        refB2Inputs.append(try await makeVisibleAnswerInput(p))
    }
    nonisolated(unsafe) let inRefB20 = refB2Inputs[0]
    nonisolated(unsafe) let inRefB21 = refB2Inputs[1]
    let (_, streamRefB20) = await engineRefB2.submit(input: inRefB20, parameters: plainParams)
    let (_, streamRefB21) = await engineRefB2.submit(input: inRefB21, parameters: plainParams)
    let refB2Results = await collectBatchStreamsWithOverlapAndInfo(
        engineRefB2,
        streams: [(0, streamRefB20), (1, streamRefB21)],
        label: "TQ B=2 plain/plain reference")
    await engineRefB2.shutdown()
    let refB2Slot0 = refB2Results[0].1
    for (slot, ids, info) in refB2Results {
        let text = tokenizer.decode(tokenIds: ids)
        print(String(format: "  Plain/plain slot %d: %d tokens, first 8: %@",
            slot, ids.count, "\(Array(ids.prefix(8)))"))
        printDecodedOutput(label: "TQ reference B2 plain slot \(slot)", text: text)
        requireStopBounded(
            label: "B=2 plain/plain reference slot \(slot)",
            tokenIds: ids,
            info: info,
            text: text)
    }

    // --- Pass A: plain KV  +  TurboQuant  (heterogeneous) ----------------
    print("\n[Pass A] slot 0 = plain KV, slot 1 = TurboQuant(4,4)")
    let engineA = BatchEngine(context: ctx, maxBatchSize: 2)

    nonisolated(unsafe) let inA0 = inputs[0]
    nonisolated(unsafe) let inA1 = inputs[1]
    let (_, streamA0) = await engineA.submit(input: inA0, parameters: plainParams)
    let (_, streamA1) = await engineA.submit(input: inA1, parameters: tqParams)
    let resultsA = await collectBatchStreamsWithOverlapAndInfo(
        engineA,
        streams: [(0, streamA0), (1, streamA1)],
        label: "TQ B=2 pass A")
    let compatibilitySplitsA = await engineA.decodeCompatibilitySplitCountForDiagnostics
    let compressionCountA = await engineA.turboQuantCompressionCountForDiagnostics
    for (slot, ids, info) in resultsA {
        let tag = slot == 0 ? "plain" : "TQ(4,4)"
        let text = tokenizer.decode(tokenIds: ids)
        print(String(format: "  Slot %d (%@) : %d tokens, first 8: %@",
            slot, tag, ids.count, "\(Array(ids.prefix(8)))"))
        printDecodedOutput(label: "TQ pass A slot \(slot) \(tag)", text: text)
        requireStopBounded(
            label: "Pass A slot \(slot) \(tag)",
            tokenIds: ids,
            info: info,
            text: text)
    }
    await engineA.shutdown()

    // --- Pass B: both slots TurboQuant (parallel TQ) -------------------
    print("\n[Pass B] slot 0 = TurboQuant(4,4), slot 1 = TurboQuant(4,4)")
    let engineB = BatchEngine(context: ctx, maxBatchSize: 2)
    // Fresh inputs — LMInput is consumed by submit.
    var inputsB: [LMInput] = []
    for p in prompts {
        inputsB.append(try await makeVisibleAnswerInput(p))
    }
    nonisolated(unsafe) let inB0 = inputsB[0]
    nonisolated(unsafe) let inB1 = inputsB[1]
    let (_, streamB0) = await engineB.submit(input: inB0, parameters: tqParams)
    let (_, streamB1) = await engineB.submit(input: inB1, parameters: tqParams)
    let resultsB = await collectBatchStreamsWithOverlapAndInfo(
        engineB,
        streams: [(0, streamB0), (1, streamB1)],
        label: "TQ B=2 pass B")
    let compressionCountB = await engineB.turboQuantCompressionCountForDiagnostics
    for (slot, ids, info) in resultsB {
        let text = tokenizer.decode(tokenIds: ids)
        print(String(format: "  Slot %d (TQ) : %d tokens, first 8: %@",
            slot, ids.count, "\(Array(ids.prefix(8)))"))
        printDecodedOutput(label: "TQ pass B slot \(slot)", text: text)
        requireStopBounded(
            label: "Pass B slot \(slot)",
            tokenIds: ids,
            info: info,
            text: text)
    }
    await engineB.shutdown()

    // ISOLATION CHECK — the actual iter 38 bug class.
    //
    // Pass A slot 0 (plain KV) ran concurrently with Pass A slot 1
    // (TurboQuant). If cross-slot corruption existed, slot 0's output
    // would differ from the shape-matched B=2 plain/plain reference. We EXPECT byte-identical
    // equality because both plain-KV decodes are deterministic at temp=0
    // and each slot's cache should be fully isolated from its neighbour.
    let a0 = resultsA[0].1
    let soloOk = a0 == refTokens
    if !soloOk {
        let firstSoloDiff = firstDiffIndex(a0, refTokens) ?? -1
        print("  Diagnostic: B=2 plain slot differs from B=1 solo at index \(firstSoloDiff); " +
              "using B=2 plain/plain as the isolation baseline.")
    }
    let isolationOk = a0 == refB2Slot0
    let isolatedByScheduler = compatibilitySplitsA > 0 && a0 == refTokens
    let isolationLabel: String
    if isolationOk {
        isolationLabel = "IDENTICAL to B=2 plain/plain reference"
    } else if isolatedByScheduler {
        isolationLabel = "ISOLATED to B=1 plain reference after compatibility split"
    } else {
        isolationLabel = "DIVERGED"
    }
    print(String(format:
        "\n  Slot 0 plain with TQ neighbour isolation: %@ (%d vs B2 %d, solo %d tokens; compatibilitySplits=%d; tqCompressionsA=%d; tqCompressionsB=%d)",
        isolationLabel,
        a0.count, refB2Slot0.count, refTokens.count, compatibilitySplitsA,
        compressionCountA, compressionCountB))
    if compressionCountA == 0 || compressionCountB < 2 {
        fputs("[TQ B=2] FAIL: TurboQuant mode did not actually compress the expected slots " +
              "(passA=\(compressionCountA), passB=\(compressionCountB)).\n",
              stderr)
        exit(1)
    }
    if !(isolationOk || isolatedByScheduler) {
        fputs("[TQ B=2] FAIL: slot 0 plain output drifted from the B=2 plain/plain reference " +
              "or the scheduler-isolated B=1 reference when slot 1 used TurboQuant. Cross-slot corruption.\n",
              stderr)
        let firstDiff = firstDiffIndex(a0, refB2Slot0) ?? -1
        fputs("  first divergence index = \(firstDiff)\n", stderr)
        exit(1)
    }
    print("=== BatchEngine TurboQuant B=2 (isolation verified): passed ===")
}

// MARK: - B=N concurrent stress (iter 39)

/// Submit `batchSize` distinct prompts to a single BatchEngine and drain
/// all streams concurrently via `TaskGroup`. Proves:
/// 1. All slots complete with coherent non-empty output.
/// 2. Per-slot outputs don't cross-contaminate — slot 0's tokens match
///    its solo-reference run byte-for-byte.
/// 3. Wall time is less than `batchSize × single-slot time` (real batching).
func runBatchEngineBMany(modelPath: String, maxNew: Int, batchSize: Int) async throws {
    let modelDir = URL(fileURLWithPath: modelPath)
    print("\n=== BatchEngine B=\(batchSize) concurrent stress (iter 39) ===")
    print("Loading with real HuggingFace tokenizer...")
    let loadStart = CFAbsoluteTimeGetCurrent()
    let context = try await MLXLMCommon.loadModel(
        from: modelDir, using: #huggingFaceTokenizerLoader())
    print(String(format: "Load: %.2fs", CFAbsoluteTimeGetCurrent() - loadStart))
    print("Model: \(type(of: context.model))")

    // Prompts: 8 distinct short questions so we can support batchSize up
    // to 8 without duplication. Caller can go higher by asking through
    // BENCH_B_SIZE but we cycle after 8.
    let promptPool = [
        "Name one country in Asia.",
        "What color is the sun at noon?",
        "Give one word for the number 2.",
        "Name one fruit that is red.",
        "What is the tallest animal on land?",
        "What is the smallest planet?",
        "Name one musical instrument with strings.",
        "What is H₂O commonly called?",
    ]
    let prompts = (0..<batchSize).map {
        "\(promptPool[$0 % promptPool.count]) Reply briefly. Slot marker \($0)."
    }

    let params = GenerateParameters(
        maxTokens: maxNew, temperature: 0, prefillStepSize: 512)

    nonisolated(unsafe) let ctx = context
    let tokenizer = context.tokenizer

    // --- Solo reference for slot 0 (B=1) ------------------------------
    let engineRef = BatchEngine(context: ctx, maxBatchSize: 1)
    let refInput = try await context.processor.prepare(input: UserInput(prompt: prompts[0]))
    nonisolated(unsafe) let inRef = refInput
    let t0Ref = CFAbsoluteTimeGetCurrent()
    let (_, streamRef) = await engineRef.submit(input: inRef, parameters: params)
    var refTokens: [Int] = []
    for await e in streamRef {
        if case .token(let id) = e { refTokens.append(id) }
        if refTokens.count >= maxNew { break }
    }
    let soloWall = CFAbsoluteTimeGetCurrent() - t0Ref
    await engineRef.shutdown()
    print(String(format: "[Reference] B=1 solo, prompt[0]: %d tokens in %.2fs",
        refTokens.count, soloWall))
    printDecodedOutput(
        label: "BMany reference slot 0",
        text: tokenizer.decode(tokenIds: refTokens))

    // --- Actual B=N stress --------------------------------------------
    let engine = BatchEngine(context: ctx, maxBatchSize: batchSize)
    var inputs: [LMInput] = []
    for p in prompts {
        inputs.append(try await context.processor.prepare(input: UserInput(prompt: p)))
    }

    let t0 = CFAbsoluteTimeGetCurrent()
    var streams: [AsyncStream<BatchGeneration>] = []
    streams.reserveCapacity(batchSize)
    for input in inputs {
        nonisolated(unsafe) let sendable = input
        let (_, s) = await engine.submit(input: sendable, parameters: params)
        streams.append(s)
    }
    let configuredMaxBatchSize = await engine.maxBatchSize
    let results = await collectBatchStreamsWithOverlap(
        engine,
        streams: Array(streams.enumerated()).map { ($0.offset, $0.element) },
        maxTokens: maxNew,
        label: "B=\(batchSize) stress",
        atLeast: min(batchSize, configuredMaxBatchSize))
    let batchedWall = CFAbsoluteTimeGetCurrent() - t0

    // Report & validate.
    for (i, ids) in results {
        let text = tokenizer.decode(tokenIds: ids)
        print("  slot \(i) (\"\(prompts[i])\"): \(ids.count) tokens")
        print("    first 8: \(Array(ids.prefix(8)))")
        printDecodedOutput(label: "BMany slot \(i)", text: text)
        if ids.isEmpty {
            fputs("[B=\(batchSize)] FAIL: slot \(i) produced zero tokens.\n", stderr)
            exit(1)
        }
    }

    // Slot 0 under B=N must match its solo-reference run byte-for-byte.
    let slot0 = results[0].1
    let slot0Ok = slot0 == refTokens
    print(String(format:
        "\n  slot 0 (B=%d) vs solo B=1 reference: %@ (%d vs %d tokens)",
        batchSize,
        slot0Ok ? "IDENTICAL ✓" : "DIVERGED ✗",
        slot0.count, refTokens.count))
    if !slot0Ok {
        let d = firstDiffIndex(slot0, refTokens) ?? -1
        fputs("[B=\(batchSize)] FAIL: slot 0 under concurrent B=\(batchSize) " +
              "diverged from solo reference at index \(d). Cross-slot corruption.\n",
              stderr)
        exit(1)
    }

    // Wall-time speedup. With `batchSize` slots decoding concurrently,
    // the fully-serial projection is `batchSize × solo wall time`.
    // Real batching must beat that — but only when ALL slots actually
    // decode the same number of tokens. Some models/prompts EOS
    // immediately (e.g. Gemma-4 short-answer: "Japan", "Two") so one
    // slot dominates wall time and the serial projection becomes a
    // misleading floor. Only assert when every slot ran to max — the
    // batched-decode stress test has nothing to measure otherwise.
    let allReachedMax = results.allSatisfy { $0.1.count == maxNew }
    let speedupRatio = batchedWall / (soloWall * Double(batchSize))
    print(String(format:
        "  solo wall = %.2fs, B=%d wall = %.2fs → batched/serial-proj ratio = %.2f " +
        "(1.0 = no batching, 0.0 = perfect)",
        soloWall, batchSize, batchedWall, speedupRatio))
    if allReachedMax {
        if speedupRatio >= 0.95 {
            fputs("[B=\(batchSize)] FAIL: batched wall (\(batchedWall)s) ≥ 95% of serial projection " +
                  "(\(soloWall * Double(batchSize))s). Engine isn't sharing forward passes.\n",
                  stderr)
            exit(1)
        }
    } else {
        let dist = results.map { $0.1.count }
        print("  speedup assertion skipped — slots had uneven token counts \(dist); " +
              "correctness check (slot 0 vs solo reference) is the load-bearing assertion here")
    }
    await engine.shutdown()
    print("=== BatchEngine B=\(batchSize) stress: passed ===")
}

// MARK: - Cancel mid-stream (iter 40)

/// Submit 3 concurrent requests, let them decode a few tokens, then call
/// `engine.cancel(id)` on slot 1. Verify:
/// - Slot 1 stream yields a trailing `.info` event with `stopReason=.cancelled`
///   within a reasonable grace period.
/// - Slots 0 and 2 continue decoding and finish normally (`.stop` or
///   `.length`), each producing `maxNew` tokens.
/// - No crash, no deadlock — the engine's loopTask doesn't hang.
///
/// This is the primitive osaurus relies on for "close a chat window
/// mid-stream without Metal crash" (`ModelLease` keeps the model pinned
/// while the engine unwinds). We're only exercising the engine side
/// here — lease / Metal lifetime are osaurus-side concerns.
func runBatchEngineCancelMidStream(modelPath: String, maxNew: Int) async throws {
    let modelDir = URL(fileURLWithPath: modelPath)
    print("\n=== BatchEngine cancel mid-stream (iter 40) ===")
    print("Loading with real HuggingFace tokenizer...")
    let loadStart = CFAbsoluteTimeGetCurrent()
    let context = try await MLXLMCommon.loadModel(
        from: modelDir, using: #huggingFaceTokenizerLoader())
    print(String(format: "Load: %.2fs", CFAbsoluteTimeGetCurrent() - loadStart))
    print("Model: \(type(of: context.model))")

    // Long enough maxNew that slots 0 and 2 will still be decoding by
    // the time we fire `cancel()` on slot 1. Tweak via BENCH_MAX_TOKENS.
    let perSlotMax = max(maxNew, 60)

    let params = GenerateParameters(
        maxTokens: perSlotMax, temperature: 0, prefillStepSize: 512)

    nonisolated(unsafe) let ctx = context
    let engine = BatchEngine(context: ctx, maxBatchSize: 3)

    let prompts = [
        "Explain photosynthesis in a short paragraph.",
        "Describe the process of evaporation.",
        "What are the three states of matter?",
    ]
    var inputs: [LMInput] = []
    for p in prompts {
        inputs.append(try await context.processor.prepare(input: UserInput(prompt: p)))
    }

    nonisolated(unsafe) let in0 = inputs[0]
    nonisolated(unsafe) let in1 = inputs[1]
    nonisolated(unsafe) let in2 = inputs[2]
    let (_, s0) = await engine.submit(input: in0, parameters: params)
    let (id1, s1) = await engine.submit(input: in1, parameters: params)
    let (_, s2) = await engine.submit(input: in2, parameters: params)
    await requireActiveSlotOverlap(engine, atLeast: 3, label: "Cancel B=3")

    // Track results concurrently via TaskGroup. Each slot yields
    // (index, tokens, stopReason). stopReason is nil if stream closed
    // without an `.info` event (engine bug).
    enum Outcome { case finished(stop: GenerateStopReason?) }
    let results = await withTaskGroup(
        of: (Int, [Int], GenerateStopReason?).self
    ) { group in
        for (i, s) in [(0, s0), (1, s1), (2, s2)] {
            group.addTask {
                var ids: [Int] = []
                var stop: GenerateStopReason? = nil
                for await e in s {
                    switch e {
                    case .token(let id):
                        ids.append(id)
                    case .info(let info):
                        stop = info.stopReason
                    }
                    if ids.count >= perSlotMax { break }
                }
                return (i, ids, stop)
            }
        }
        // Schedule a cancel on slot 1 after a short delay, long enough
        // for decode to have started on all three slots but well short
        // of maxTokens.
        group.addTask {
            try? await Task.sleep(nanoseconds: 100_000_000)  // 100 ms
            await engine.cancel(id1)
            return (-1, [], nil)  // sentinel — ignored in collection
        }
        var collected: [(Int, [Int], GenerateStopReason?)] = []
        for await r in group where r.0 >= 0 {
            collected.append(r)
        }
        return collected.sorted { $0.0 < $1.0 }
    }
    if results.count != 3 {
        fputs("[Cancel] FAIL: expected 3 slot results, got \(results.count).\n", stderr)
        exit(1)
    }

    // Report & validate.
    for (i, ids, stop) in results {
        let stopStr = stop.map { "\($0)" } ?? "nil"
        print("  slot \(i): \(ids.count) tokens, stop=\(stopStr)")
        printDecodedOutput(
            label: "Cancel slot \(i)",
            text: context.tokenizer.decode(tokenIds: ids))
    }

    let r0 = results[0]
    let r1 = results[1]
    let r2 = results[2]

    // Slot 1 must report cancelled.
    if r1.2 != .cancelled {
        fputs("[Cancel] FAIL: slot 1 stopReason=\(String(describing: r1.2)) — " +
              "expected .cancelled. cancel() didn't reach the slot.\n", stderr)
        exit(1)
    }
    // Slot 1 should have produced strictly fewer tokens than maxTokens
    // (otherwise the cancel fired after decode already completed).
    if r1.1.count >= perSlotMax {
        fputs("[Cancel] FAIL: slot 1 reached max tokens before cancel landed " +
              "(\(r1.1.count)/\(perSlotMax)). cancel is too late to be interesting — " +
              "this row cannot be used as mid-stream cancellation proof.\n",
              stderr)
        exit(1)
    }
    // Slots 0 and 2 must reach max-tokens and report `.length` (not
    // `.cancelled`) — they were unaffected by slot 1's cancel.
    for (label, r) in [("slot 0", r0), ("slot 2", r2)] {
        if r.1.count < perSlotMax {
            fputs("[Cancel] FAIL: \(label) stopped early (\(r.1.count)/\(perSlotMax) tokens). " +
                  "cancelling one slot disturbed its neighbours.\n", stderr)
            exit(1)
        }
        if r.2 == .cancelled {
            fputs("[Cancel] FAIL: \(label) reported .cancelled — cross-slot cancel bled over.\n",
                  stderr)
            exit(1)
        }
    }
    await engine.shutdown()
    print("=== BatchEngine cancel mid-stream: passed ===")
}

// MARK: - Long-context prefill (iter 42)

/// Build a synthetic long prompt with deterministic safe token ids and
/// run it through both TokenIterator and BatchEngine. Asserts:
/// - Prefill chunking doesn't break for prompts much larger than the
///   default `prefillStepSize` (512) → multi-pass prefill works.
/// - Output is byte-identical across the two engines at temp=0 —
///   extending the iter 32 cross-engine check to long-context regime.
/// - Memory purge runs during decode (memoryPurgeInterval=256) without
///   corrupting in-flight state.
/// - No hang, no OOM, wall time scales reasonably.
func runBatchEngineLongContext(
    modelPath: String, maxNew: Int, promptLen: Int
) async throws {
    let modelDir = URL(fileURLWithPath: modelPath)
    print("\n=== BatchEngine long-context prefill (iter 42, prompt \(promptLen) tokens) ===")
    let loadStart = CFAbsoluteTimeGetCurrent()
    let context = try await loadBenchModelContext(from: modelDir)
    print(String(format: "Load: %.2fs", CFAbsoluteTimeGetCurrent() - loadStart))
    print("Model: \(type(of: context.model))")

    // Deterministic, bounded-range token ids. Avoid specials — stay in
    // [100, 50_000) which is safe for any vocab ≥ 50k. Size chosen so
    // both engines see byte-identical input.
    let seedIds: [Int32] = (0..<promptLen).map { Int32(100 + ($0 * 37) % 49_000) }
    let tokensArr = MLXArray(seedIds)[.newAxis, .ellipsis]
    let longInput = LMInput(text: LMInput.Text(tokens: tokensArr))
    print("  synthetic prompt: \(promptLen) tokens, vocab range [100, 50100)")

    let params = GenerateParameters(
        maxTokens: maxNew, temperature: 0, prefillStepSize: 512)

    // Match BatchEngine stop semantics. Raw TokenIterator in this harness
    // yields through EOS until the caller breaks at maxNew, while
    // BatchEngine stops before surfacing EOS tokens.
    var stopTokenIDs: Set<Int> = context.configuration.eosTokenIds
    if let eos = context.tokenizer.eosTokenId { stopTokenIDs.insert(eos) }
    if let unk = context.tokenizer.unknownTokenId { stopTokenIDs.insert(unk) }
    for tok in context.configuration.extraEOSTokens {
        if let id = context.tokenizer.convertTokenToId(tok) {
            stopTokenIDs.insert(id)
        }
    }

    // --- TokenIterator (baseline) ---------------------------------------
    print("\n[Path A] TokenIterator")
    let t0A = CFAbsoluteTimeGetCurrent()
    let iterCache = context.model.newCache(parameters: params)
    let iter = try TokenIterator(
        input: longInput, model: context.model, cache: iterCache, parameters: params)
    var iterTokens: [Int] = []
    var iterFirstTokenTime: Double? = nil
    for token in iter {
        if iterFirstTokenTime == nil {
            iterFirstTokenTime = CFAbsoluteTimeGetCurrent() - t0A
        }
        iterTokens.append(token)
        if iterTokens.count >= maxNew { break }
    }
    let wallA = CFAbsoluteTimeGetCurrent() - t0A
    print(String(format: "  iterator: %d tokens, TTFT %.0fms, wall %.2fs",
        iterTokens.count,
        (iterFirstTokenTime ?? 0) * 1000, wallA))

    // --- BatchEngine ----------------------------------------------------
    print("\n[Path B] BatchEngine.submit")
    nonisolated(unsafe) let ctx = context
    let engine = BatchEngine(context: ctx, maxBatchSize: 1)
    // Fresh LMInput — the one from path A is already consumed.
    let tokensArrB = MLXArray(seedIds)[.newAxis, .ellipsis]
    nonisolated(unsafe) let longInputB = LMInput(text: LMInput.Text(tokens: tokensArrB))
    let t0B = CFAbsoluteTimeGetCurrent()
    let (_, stream) = await engine.submit(input: longInputB, parameters: params)
    var engineTokens: [Int] = []
    var engineFirstTokenTime: Double? = nil
    var promptTime: Double = 0
    for await event in stream {
        switch event {
        case .token(let id):
            if engineFirstTokenTime == nil {
                engineFirstTokenTime = CFAbsoluteTimeGetCurrent() - t0B
            }
            engineTokens.append(id)
            if engineTokens.count >= maxNew { break }
        case .info(let info):
            promptTime = info.promptTime
        }
    }
    let wallB = CFAbsoluteTimeGetCurrent() - t0B
    print(String(format:
        "  engine:   %d tokens, TTFT %.0fms, wall %.2fs, promptTime %.2fs",
        engineTokens.count,
        (engineFirstTokenTime ?? 0) * 1000, wallB, promptTime))

    // --- Byte-identity check -------------------------------------------
    print("\n  first 10 iter:   \(Array(iterTokens.prefix(10)))")
    print("  first 10 engine: \(Array(engineTokens.prefix(10)))")
    if iterTokens == engineTokens {
        print("  ✓ byte-identical across both engines (\(iterTokens.count) tokens)")
    } else if engineTokens.count < iterTokens.count &&
              Array(iterTokens.prefix(engineTokens.count)) == engineTokens &&
              stopTokenIDs.contains(iterTokens[engineTokens.count]) {
        print("  ✓ identical \(engineTokens.count)-token prefix — BatchEngine stopped at EOS token \(iterTokens[engineTokens.count])")
    } else {
        let d = firstDiffIndex(iterTokens, engineTokens) ?? -1
        fputs("[LongContext] FAIL: engines diverged at token \(d) " +
              "(iter=\(iterTokens.count), engine=\(engineTokens.count) tokens). " +
              "Long-context prefill chunking is broken somewhere.\n", stderr)
        exit(1)
    }
    print("=== BatchEngine long-context prefill: passed ===")
}

struct NullTokenizer: MLXLMCommon.Tokenizer {
    var bosToken: String? { nil }
    var eosToken: String? { "<end_of_turn>" }
    var unknownToken: String? { nil }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
    func convertTokenToId(_ token: String) -> Int? { token == "<end_of_turn>" ? 1 : nil }
    func convertIdToToken(_ id: Int) -> String? { nil }
    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [] }
}


// MARK: - Harmony reasoning check (2026-04-20 harmony fix)

/// Real-model regression for the Gemma-4 harmony reasoning bug.
/// Loads a Gemma-4 model, sends a short prompt that likely elicits
/// chain-of-thought, and asserts:
///   1. At least one `.reasoning(String)` event fires.
///   2. `.chunk(String)` contains zero harmony channel markers.
func runHarmonyReasoningCheck(modelPath: String, maxNew: Int) async throws {
    let modelDir = URL(fileURLWithPath: modelPath)
    print("\n=== BENCH_HARMONY_CHECK: Gemma-4 harmony reasoning channel ===")
    let loadStart = CFAbsoluteTimeGetCurrent()
    let context = try await MLXLMCommon.loadModel(
        from: modelDir, using: #huggingFaceTokenizerLoader())
    print(String(format: "Load: %.2fs", CFAbsoluteTimeGetCurrent() - loadStart))
    print("Model: \(type(of: context.model))")
    print("Reasoning stamp: \(context.configuration.reasoningParserName ?? "nil")")

    // Matches tpae's 2026-04-20 2:59 PM screenshot trigger:
    // "can you create a README for my game". Open-ended tasks with
    // ambiguity ("what game?") reliably elicit Gemma-4's
    // `<|channel>thought\n…<channel|>` block. Override via env.
    let promptText = ProcessInfo.processInfo.environment[
        "BENCH_HARMONY_PROMPT"]
        ?? "Can you create a README for my game?"
    let messages: [[String: any Sendable]] = [
        ["role": "user", "content": promptText]
    ]
    let promptTokens = try context.tokenizer.applyChatTemplate(
        messages: messages)
    let promptIds = MLXArray(promptTokens.map { Int32($0) })
        .reshaped(1, promptTokens.count)

    let input = LMInput(text: LMInput.Text(tokens: promptIds))
    nonisolated(unsafe) let ctxSendable = context
    nonisolated(unsafe) let sendable = input
    var chunkText = ""
    var reasoningText = ""
    var chunkCount = 0
    var reasoningCount = 0

    do {
        let engine = BatchEngine(context: ctxSendable, maxBatchSize: 1)
        let fallback = GenerateParameters(maxTokens: maxNew, prefillStepSize: 512)
        var params = GenerateParameters(
            generationConfig: context.configuration.generationDefaults,
            fallback: fallback)
        params.maxTokens = maxNew
        params.prefillStepSize = 512
        let stream = await engine.generate(input: sendable, parameters: params)

        for await ev in stream {
            switch ev {
            case .chunk(let c):
                chunkText += c
                chunkCount += 1
            case .reasoning(let r):
                reasoningText += r
                reasoningCount += 1
            case .toolCall, .info:
                break
            }
        }
        await engine.shutdown()
    }
    // Give actor-owned teardown a scheduling point before process exit.
    // Some JANG/JANGTQ models carry compiled helper functions whose
    // deinit erases MLX compiler-cache entries; doing that after async main has
    // started C++ static finalizers can race MLX's compiler-cache teardown.
    try? await Task.sleep(nanoseconds: 50_000_000)

    print("chunks=\(chunkCount) reasoningDeltas=\(reasoningCount)")
    print(".chunk preview:")
    print("  \"\(chunkText.prefix(300))\"")
    print(".reasoning preview:")
    print("  \"\(reasoningText.prefix(300))\"")

    // Harmony markers that must NOT leak into .chunk.
    let markers = ["<|channel>", "<channel|>", "<|channel|>"]
    for m in markers {
        if chunkText.contains(m) {
            fputs("FAIL: .chunk leaked harmony marker: \"\(m)\"\n", stderr)
            exit(1)
        }
    }
    if reasoningCount == 0 {
        // WARN not FAIL — some prompts elicit no reasoning; this still
        // validates the no-leak invariant. But for the main regression
        // goal (Bug A from tpae's screenshot), we want to see deltas.
        fputs("WARN: zero .reasoning deltas — prompt may not elicit CoT.\n", stderr)
    }
    print("PASS harmony markers absent from .chunk.")
    print("=== BENCH_HARMONY_CHECK: passed ===")
}

// MARK: - Qwen enable_thinking reasoning check (2026-04-20 fix B)

/// Real-model regression for Bug B (Qwen3.6 `<think>\n` prefill).
/// Loads a Qwen 3.x model with enable_thinking=true, and asserts:
///   1. .reasoning fires.
///   2. .chunk has zero `<think>` or `</think>` markers.
func runQwenThinkingReasoningCheck(modelPath: String, maxNew: Int) async throws {
    let modelDir = URL(fileURLWithPath: modelPath)
    print("\n=== BENCH_QWEN_THINKING_CHECK: Qwen3.x prefilled-think channel ===")
    var chunkText = ""
    var reasoningText = ""
    var chunkCount = 0
    var reasoningCount = 0

    do {
        let loadStart = CFAbsoluteTimeGetCurrent()
        let context = try await MLXLMCommon.loadModel(
            from: modelDir, using: #huggingFaceTokenizerLoader())
        print(String(format: "Load: %.2fs", CFAbsoluteTimeGetCurrent() - loadStart))
        print("Model: \(type(of: context.model))")
        print("Reasoning stamp: \(context.configuration.reasoningParserName ?? "nil")")

        let promptText =
            "Please think through this briefly and then answer: " +
            "What's 2 + 2?"
        let messages: [[String: any Sendable]] = [
            ["role": "user", "content": promptText]
        ]
        let promptTokens: [Int]
        do {
            promptTokens = try context.tokenizer.applyChatTemplate(
                messages: messages,
                tools: nil,
                additionalContext: ["enable_thinking": true])
        } catch {
            promptTokens = try context.tokenizer.applyChatTemplate(messages: messages)
        }
        let promptIds = MLXArray(promptTokens.map { Int32($0) })
            .reshaped(1, promptTokens.count)

        let input = LMInput(text: LMInput.Text(tokens: promptIds))
        nonisolated(unsafe) let ctxSendable = context
        nonisolated(unsafe) let sendable = input

        do {
            let engine = BatchEngine(context: ctxSendable, maxBatchSize: 1)
            let params = GenerateParameters(
                maxTokens: maxNew, temperature: 0, prefillStepSize: 512)
            let stream = await engine.generate(input: sendable, parameters: params)

            for await ev in stream {
                switch ev {
                case .chunk(let c):
                    chunkText += c
                    chunkCount += 1
                case .reasoning(let r):
                    reasoningText += r
                    reasoningCount += 1
                case .toolCall, .info:
                    break
                }
            }
            await engine.shutdown()
        }
    }
    // Give model/engine teardown a scheduling point before process exit.
    // Qwen JANGTQ carries compiled helper functions whose deinit erases
    // MLX compiler-cache entries; doing that after async main has started
    // C++ static finalizers can race MLX's compiler-cache teardown.
    try? await Task.sleep(nanoseconds: 50_000_000)

    print("chunks=\(chunkCount) reasoningDeltas=\(reasoningCount)")
    print(".chunk preview: \"\(chunkText.prefix(300))\"")
    print(".reasoning preview: \"\(reasoningText.prefix(300))\"")

    if chunkText.contains("<think>") || chunkText.contains("</think>") {
        fputs("FAIL: .chunk leaked <think> markers — startInReasoning broken.\n", stderr)
        exit(1)
    }
    if reasoningCount == 0 {
        fputs("WARN: zero .reasoning deltas — prompt may not elicit CoT.\n", stderr)
    }
    print("PASS no <think> leakage in .chunk.")
    print("=== BENCH_QWEN_THINKING_CHECK: passed ===")
}

// MARK: - DSV4 chat-template kwargs round-trip

/// Verify the DSV4 tokenizer template or Swift fallback threads the two
/// DSV4 kwargs — `enable_thinking` (bool) and `reasoning_effort`
/// ('max'/None) — through the upstream `applyChatTemplate` path.
///
/// The DSV4 template:
/// - With `enable_thinking=true`  appends `<｜Assistant｜><think>` (open).
/// - With `enable_thinking=false` appends `<｜Assistant｜></think>` so the
///   parser starts in visible-answer mode.
/// - With `enable_thinking=true` + `reasoning_effort='max'` prepends a
///   fixed REASONING_EFFORT_MAX preface immediately after BOS.
///
/// Loads the bundle's tokenizer (real HF path), applies the template
/// four ways, decodes back to text, and asserts the expected markers.
/// Pure tokenizer/template exercise — no model forward, no GPU.
func runDSV4TemplateKwargsCheck(modelPath: String) async throws {
    let modelDir = URL(fileURLWithPath: modelPath)
    print("\n=== BENCH_DSV4_TEMPLATE_KWARGS: enable_thinking + reasoning_effort ===")
    print("Loading tokenizer only (no model forward)...")
    let (tokenizer, tokenizerDir) = try await loadBenchTokenizer(from: modelDir)
    print("Tokenizer dir: \(tokenizerDir.path)")

    let messages: [[String: any Sendable]] = [
        ["role": "user", "content": "hi"]
    ]

    func render(_ ctx: [String: any Sendable]?) throws -> String {
        let ids = try tokenizer.applyChatTemplate(
            messages: messages, tools: nil, additionalContext: ctx)
        return tokenizer.decode(tokenIds: ids, skipSpecialTokens: false)
    }

    let chatNoEffort = try render(["enable_thinking": false])
    let thinkNoEffort = try render(["enable_thinking": true])
    let chatMaxEffort = try render([
        "enable_thinking": false, "reasoning_effort": "max"
    ])
    let thinkMaxEffort = try render([
        "enable_thinking": true, "reasoning_effort": "max"
    ])

    var failures: [String] = []
    func check(_ label: String, _ ok: Bool, _ why: String) {
        let mark = ok ? "PASS" : "FAIL"
        print("  [\(mark)] \(label): \(why)")
        if !ok { failures.append(label) }
    }

    let preface = "Reasoning Effort: Absolute maximum"

    check("chat-mode request closes thinking",
        chatNoEffort.hasSuffix("</think>"),
        "tail = …\(String(chatNoEffort.suffix(40)))")
    check("thinking-mode tail (open <think>)",
        thinkNoEffort.hasSuffix("<think>"),
        "tail = …\(String(thinkNoEffort.suffix(40)))")
    check("max-effort preface absent without effort kwarg",
        !chatNoEffort.contains(preface) && !thinkNoEffort.contains(preface),
        "neither chat nor thinking carry preface")
    check("max-effort preface absent in chat+max",
        !chatMaxEffort.contains(preface),
        "preface suppressed while enable_thinking=false")
    check("max-effort preface present in thinking+max",
        thinkMaxEffort.contains(preface),
        "preface present")
    check("max-effort variants preserve mode-specific tail",
        chatMaxEffort.hasSuffix("</think>") && thinkMaxEffort.hasSuffix("<think>"),
        "tails preserved under reasoning_effort=max")

    if !failures.isEmpty {
        fputs("BENCH_DSV4_TEMPLATE_KWARGS: \(failures.count) failures: \(failures)\n", stderr)
        exit(1)
    }
    print("=== BENCH_DSV4_TEMPLATE_KWARGS: passed ===")
}

// MARK: - Generic chat-template smoke

// MARK: - Model config / JANG metadata smoke

func runConfigSmoke(modelPath: String) async throws {
    let modelDir = URL(fileURLWithPath: modelPath)
    print("\n=== BENCH_CONFIG_SMOKE: model/JANG metadata ===")
    print("Model: \(modelPath)")

    let config = try loadJSONObject(modelDir.appending(path: "config.json"))
    let generationConfig = try? loadJSONObject(modelDir.appending(path: "generation_config.json"))
    let jangConfig = try? loadJSONObject(modelDir.appending(path: "jang_config.json"))
    let index = try? loadJSONObject(modelDir.appending(path: "model.safetensors.index.json"))
    let weightMap = index?["weight_map"] as? [String: Any]
    let weightKeys = weightMap.map { Array($0.keys) } ?? []

    let modelType = jsonString(config["model_type"]) ?? "nil"
    let textModelType = jsonString(jsonAt(config, ["text_config", "model_type"]))
    let dispatchHint: String = {
        if FileManager.default.fileExists(
            atPath: modelDir.appending(path: "config_omni.json").path)
        {
            return "nemotron_h_omni"
        }
        return textModelType ?? modelType
    }()

    let layers = firstJSONInt(config, [
        ["num_hidden_layers"], ["n_layers"], ["num_layers"],
        ["text_config", "num_hidden_layers"],
    ])
    let hidden = firstJSONInt(config, [
        ["hidden_size"], ["d_model"], ["text_config", "hidden_size"],
    ])
    let heads = firstJSONInt(config, [
        ["num_attention_heads"], ["n_heads"], ["text_config", "num_attention_heads"],
    ])
    let kvHeads = firstJSONInt(config, [
        ["num_key_value_heads"], ["n_kv_heads"], ["text_config", "num_key_value_heads"],
    ])
    let attnListCount = (config["attn_type_list"] as? [Any])?.count

    let configEOS = jsonInts(config["eos_token_id"])
    let generationEOS = jsonInts(generationConfig?["eos_token_id"])
    let configBOS = jsonInts(config["bos_token_id"])
    let generationBOS = jsonInts(generationConfig?["bos_token_id"])
    var effectiveEOS = generationEOS.isEmpty ? configEOS : generationEOS
    if modelType == "deepseek_v4" {
        effectiveEOS = Array(Set(effectiveEOS).union([1, 128803, 128804])).sorted()
    }
    let effectiveBOS = generationBOS.isEmpty ? configBOS : generationBOS

    let (tokenizer, tokenizerDir) = try await loadBenchTokenizer(from: modelDir)
    print("Tokenizer dir: \(tokenizerDir.path)")
    let tokenizerEOS = tokenizer.eosTokenId
    let tokenizerBOS = tokenizer.bosToken.flatMap { tokenizer.convertTokenToId($0) }

    let tqPackedCount = weightKeys.filter { $0.hasSuffix(".tq_packed") }.count
    let tqNormsCount = weightKeys.filter { $0.hasSuffix(".tq_norms") }.count
    let tqBitsCount = weightKeys.filter { $0.hasSuffix(".tq_bits") }.count
    let weightFormat =
        (jsonString(config["weight_format"]) ?? jsonString(jangConfig?["weight_format"]) ?? "nil")
            .lowercased()
    let profile = jsonString(jangConfig?["profile"]) ?? "nil"
    let sidecarExists = FileManager.default.fileExists(
        atPath: modelDir.appending(path: "jangtq_runtime.safetensors").path)
    let routed = resolveRoutedBits(
        config["mxtq_bits"]
            ?? jsonAt(config, ["quantization", "mxtq_bits"])
            ?? jsonAt(config, ["quantization", "routed_expert_bits"])
            ?? jangConfig?["mxtq_bits"]
            ?? jsonAt(jangConfig ?? [:], ["quantization", "mxtq_bits"])
    )
    let expectedJANGTQ =
        weightFormat == "mxtq"
        || profile.lowercased().contains("jangtq")
        || sidecarExists
        || tqPackedCount > 0
    let bosInEOS = !Set(effectiveBOS).isDisjoint(with: Set(effectiveEOS))
    let tokenizerEOSCovered = tokenizerEOS.map { effectiveEOS.contains($0) } ?? false
    let tokenizerBOSCovered = tokenizerBOS.map { effectiveBOS.contains($0) } ?? false

    var failures: [String] = []
    var warnings: [String] = []
    if modelType == "nil" { failures.append("missing model_type") }
    if let layers, layers <= 0 { failures.append("non-positive layer count") }
    if layers == nil { failures.append("missing layer count") }
    if let attnListCount, let layers, attnListCount != layers {
        failures.append("attn_type_list count \(attnListCount) != layers \(layers)")
    }
    if effectiveEOS.isEmpty { failures.append("empty effective EOS stop set") }
    if let tokenizerEOS, !effectiveEOS.contains(tokenizerEOS) {
        warnings.append("tokenizer EOS \(tokenizerEOS) not in effective EOS \(effectiveEOS)")
    }
    if bosInEOS {
        warnings.append("BOS appears in EOS set")
    }
    if let tokenizerBOS, !effectiveBOS.isEmpty, !effectiveBOS.contains(tokenizerBOS) {
        warnings.append("tokenizer BOS \(tokenizerBOS) != configured BOS \(effectiveBOS)")
    }
    if expectedJANGTQ {
        if !sidecarExists {
            failures.append("JANGTQ detected but jangtq_runtime.safetensors missing")
        }
        if tqPackedCount == 0 {
            warnings.append("JANGTQ detected but no .tq_packed keys in index")
        }
        if routed.uniform == nil && routed.gateUp == nil && routed.down == nil {
            warnings.append("JANGTQ detected but routed bit metadata not explicit")
        }
    }
    if modelType == "deepseek_v4", !effectiveEOS.contains(128804) {
        failures.append("DSV4 effective EOS missing Assistant role token 128804")
    }

    print(
        "CONFIG_SMOKE status=\(failures.isEmpty ? "PASS" : "FAIL") modelType=\(modelType) dispatch=\(dispatchHint) layers=\(layers.map(String.init) ?? "nil") hidden=\(hidden.map(String.init) ?? "nil") heads=\(heads.map(String.init) ?? "nil") kvHeads=\(kvHeads.map(String.init) ?? "nil") attnList=\(attnListCount.map(String.init) ?? "nil") weightFormat=\(weightFormat) profile=\(profile) sidecar=\(sidecarExists) tqPacked=\(tqPackedCount) tqNorms=\(tqNormsCount) tqBits=\(tqBitsCount) routedBits=\(routed.description) eosConfig=\(configEOS) eosGen=\(generationEOS) eosEffective=\(effectiveEOS) tokenizerEOS=\(tokenizerEOS.map(String.init) ?? "nil") tokenizerEOSCovered=\(tokenizerEOSCovered) bosConfig=\(configBOS) bosGen=\(generationBOS) bosEffective=\(effectiveBOS) tokenizerBOS=\(tokenizerBOS.map(String.init) ?? "nil") tokenizerBOSCovered=\(tokenizerBOSCovered) bosInEOS=\(bosInEOS)"
    )
    if !warnings.isEmpty {
        print("CONFIG_SMOKE warnings=\(warnings)")
    }
    if !failures.isEmpty {
        fputs("BENCH_CONFIG_SMOKE: \(failures.count) failure(s): \(failures)\n", stderr)
        exit(1)
    }
    print("=== BENCH_CONFIG_SMOKE: passed ===")
}

func runZayaContract(modelPath: String) async throws {
    let modelDir = URL(fileURLWithPath: modelPath)
    print("\n=== BENCH_ZAYA_CONTRACT: ZAYA CCA/JANG bundle contract ===")
    print("Model: \(modelPath)")

    let config = try loadJSONObject(modelDir.appending(path: "config.json"))
    let generationConfig = try? loadJSONObject(modelDir.appending(path: "generation_config.json"))
    let jangConfig = try? loadJSONObject(modelDir.appending(path: "jang_config.json"))
    let tokenizerConfig = try? loadJSONObject(modelDir.appending(path: "tokenizer_config.json"))
    let index = try loadJSONObject(modelDir.appending(path: "model.safetensors.index.json"))
    let weightMap = index["weight_map"] as? [String: Any] ?? [:]
    let weightKeys = Array(weightMap.keys)

    let modelType = jsonString(config["model_type"]) ?? "nil"
    let isZayaVL = modelType == "zaya1_vl"
    let layers = jsonInt(config["num_hidden_layers"])
    let hidden = jsonInt(config["hidden_size"])
    let heads = jsonInt(config["num_attention_heads"])
    let groups = jsonInt(config["num_query_groups"])
    let cca = (config["cca"] as? Bool) ?? false
    let ccaHeads = jsonInt(config["cca_num_q_heads"])
    let kvChannels = jsonInt(config["kv_channels"]) ?? jsonInt(config["head_dim"])
    let cacheSubtype = jsonString(jangConfig?["cache_subtype"])
    let cacheType = jsonString(jsonAt(jangConfig ?? [:], ["capabilities", "cache_type"]))
    let expertLayout =
        jsonString(jangConfig?["expert_layout"])
        ?? jsonString(jsonAt(config, ["quantization", "expert_layout"]))
        ?? jsonString(config["zaya_expert_layout"])
    let visionLora = (config["vision_lora"] as? Bool) ?? false
    let weightFormat =
        (jsonString(config["weight_format"]) ?? jsonString(jangConfig?["weight_format"]) ?? "nil")
            .lowercased()

    let configEOS = jsonInts(config["eos_token_id"])
    let generationEOS = jsonInts(generationConfig?["eos_token_id"])
    let effectiveEOS = generationEOS.isEmpty ? configEOS : generationEOS
    let (tokenizer, tokenizerDir) = try await loadBenchTokenizer(from: modelDir)
    let tokenizerEOS = tokenizer.eosTokenId
    let tokenizerChatTemplate = tokenizerConfig?["chat_template"] is String
    let templateFile = FileManager.default.fileExists(
        atPath: modelDir.appending(path: "chat_template.jinja").path)

    let attnLayers = layerSet(weightKeys, marker: ".self_attn.")
    let moeLayers = layerSet(weightKeys, marker: ".zaya_block.")
    let expectedLayerCount = isZayaVL ? 40 : 80
    let expectedHeads = isZayaVL ? 8 : 16
    let expectedAttn = isZayaVL
        ? Set(0..<expectedLayerCount)
        : Set(stride(from: 0, to: expectedLayerCount, by: 2))
    let expectedMoE = isZayaVL
        ? Set(0..<expectedLayerCount)
        : Set(stride(from: 1, to: expectedLayerCount, by: 2))
    let expectedPathCount = 40
    let conv0 = weightKeys.filter { $0.hasSuffix(".self_attn.qkv.conv_qk.0.weight") }.count
    let conv1 = weightKeys.filter { $0.hasSuffix(".self_attn.qkv.conv_qk.1.weight") }.count
    let temp = weightKeys.filter { $0.hasSuffix(".self_attn.qkv.temp") }.count
    let qProj = weightKeys.filter { $0.hasSuffix(".self_attn.qkv.linear_q.weight") }.count
    let kProj = weightKeys.filter { $0.hasSuffix(".self_attn.qkv.linear_k.weight") }.count
    let v1Proj = weightKeys.filter { $0.hasSuffix(".self_attn.qkv.val_proj1.weight") }.count
    let v2Proj = weightKeys.filter { $0.hasSuffix(".self_attn.qkv.val_proj2.weight") }.count

    let tqPacked = weightKeys.filter { $0.hasSuffix(".tq_packed") }.count
    let tqNorms = weightKeys.filter { $0.hasSuffix(".tq_norms") }.count
    let tqBits = weightKeys.filter { $0.hasSuffix(".tq_bits") }.count
    let localExperts = weightKeys.filter { $0.contains(".local_experts.") }.count
    let localExpertLoRA = weightKeys.filter {
        $0.contains(".local_experts.") && $0.contains(".lora_")
    }.count
    let localExpertNonLoRA = localExperts - localExpertLoRA
    let sidecar = FileManager.default.fileExists(
        atPath: modelDir.appending(path: "jangtq_runtime.safetensors").path)
    let tqInFeatures = jangConfig?["tq_in_features"] as? [String: Any]
    let tqInFeatureCount = tqInFeatures?.count ?? 0
    let tqInFeatureBad = tqInFeatures?.filter { jsonInt($0.value) != 2048 }.count ?? 0
    let routed = resolveRoutedBits(config["mxtq_bits"] ?? jangConfig?["mxtq_bits"])

    var failures: [String] = []
    var warnings: [String] = []

    if modelType != "zaya" && !isZayaVL {
        failures.append("model_type \(modelType) not in {zaya,zaya1_vl}")
    }
    if layers != expectedLayerCount {
        failures.append(
            "num_hidden_layers \(layers.map(String.init) ?? "nil") != \(expectedLayerCount)")
    }
    if hidden != 2048 { failures.append("hidden_size \(hidden.map(String.init) ?? "nil") != 2048") }
    if heads != expectedHeads {
        failures.append(
            "num_attention_heads \(heads.map(String.init) ?? "nil") != \(expectedHeads)")
    }
    if groups != 2 { failures.append("num_query_groups \(groups.map(String.init) ?? "nil") != 2") }
    if !cca { failures.append("cca flag is not true") }
    let effectiveCCAHeads = ccaHeads ?? (isZayaVL ? heads : nil)
    if effectiveCCAHeads != 8 {
        failures.append(
            "cca_num_q_heads/head fallback \(effectiveCCAHeads.map(String.init) ?? "nil") != 8")
    }
    if kvChannels != 128 { failures.append("kv_channels \(kvChannels.map(String.init) ?? "nil") != 128") }
    if cacheSubtype != "zaya_cca" {
        failures.append("jang_config.cache_subtype \(cacheSubtype ?? "nil") != zaya_cca")
    }
    if cacheType != "hybrid" {
        failures.append("jang_config.capabilities.cache_type \(cacheType ?? "nil") != hybrid")
    }
    if expertLayout != "split_switch_mlp" {
        failures.append("expert_layout \(expertLayout ?? "nil") != split_switch_mlp")
    }
    if attnLayers != expectedAttn {
        failures.append("attention layer set mismatch count=\(attnLayers.count)")
    }
    if moeLayers != expectedMoE {
        failures.append("MoE layer set mismatch count=\(moeLayers.count)")
    }
    for (label, count) in [
        ("conv_qk.0", conv0), ("conv_qk.1", conv1), ("temp", temp),
        ("linear_q", qProj), ("linear_k", kProj),
        ("val_proj1", v1Proj), ("val_proj2", v2Proj),
    ] where count != expectedPathCount {
        failures.append("\(label) count \(count) != \(expectedPathCount)")
    }
    if effectiveEOS.isEmpty {
        failures.append("empty effective EOS")
    }
    if let tokenizerEOS, !effectiveEOS.contains(tokenizerEOS) {
        failures.append("tokenizer EOS \(tokenizerEOS) not in effective EOS \(effectiveEOS)")
    }
    if !tokenizerChatTemplate && !templateFile {
        failures.append("missing tokenizer chat_template and chat_template.jinja")
    }

    switch weightFormat {
    case "mxtq":
        if !sidecar { failures.append("mxtq bundle missing jangtq_runtime.safetensors") }
        if tqPacked != 120 || tqNorms != 120 || tqBits != 120 {
            failures.append("mxtq TQ group counts packed/norms/bits = \(tqPacked)/\(tqNorms)/\(tqBits), expected 120/120/120")
        }
        if isZayaVL {
            if localExpertNonLoRA != 0 {
                failures.append("non-LoRA local_experts keys remain: \(localExpertNonLoRA)")
            }
            if visionLora && localExpertLoRA != 2560 {
                failures.append("vision LoRA local_experts count \(localExpertLoRA) != 2560")
            }
        } else if localExperts != 0 {
            failures.append("local_experts keys remain: \(localExperts)")
        }
        if tqInFeatureCount != 120 {
            failures.append("tq_in_features count \(tqInFeatureCount) != 120")
        }
        if tqInFeatureBad != 0 {
            failures.append("tq_in_features has \(tqInFeatureBad) non-2048 entries")
        }
        let routedUniformOK = routed.uniform == 2 || routed.uniform == 4
        let routedMixedOK =
            (routed.gateUp == 2 || routed.gateUp == 4)
            && (routed.down == 2 || routed.down == 4)
        if !routedUniformOK && !routedMixedOK {
            failures.append("routed bits \(routed.description) not in {2,4}")
        }
    case "mxfp4":
        if sidecar { warnings.append("mxfp4 bundle has unexpected JANGTQ sidecar") }
        if tqPacked != 0 || tqNorms != 0 || tqBits != 0 {
            failures.append("mxfp4 bundle has TQ codebook keys packed/norms/bits = \(tqPacked)/\(tqNorms)/\(tqBits)")
        }
    default:
        warnings.append("unrecognized weight_format \(weightFormat)")
    }

    print(
        "ZAYA_CONTRACT status=\(failures.isEmpty ? "PASS" : "FAIL") modelType=\(modelType) tokenizerDir=\(tokenizerDir.path) weightFormat=\(weightFormat) cacheSubtype=\(cacheSubtype ?? "nil") cacheType=\(cacheType ?? "nil") attnLayers=\(attnLayers.count) moeLayers=\(moeLayers.count) conv0=\(conv0) conv1=\(conv1) temp=\(temp) tqPacked=\(tqPacked) tqNorms=\(tqNorms) tqBits=\(tqBits) tqInFeatures=\(tqInFeatureCount) localExpertLoRA=\(localExpertLoRA) localExpertNonLoRA=\(localExpertNonLoRA) routedBits=\(routed.description) sidecar=\(sidecar) eosEffective=\(effectiveEOS) tokenizerEOS=\(tokenizerEOS.map(String.init) ?? "nil") template=\(tokenizerChatTemplate || templateFile)"
    )
    if !warnings.isEmpty { print("ZAYA_CONTRACT warnings=\(warnings)") }
    if !failures.isEmpty {
        fputs("BENCH_ZAYA_CONTRACT: \(failures.count) failure(s): \(failures)\n", stderr)
        exit(1)
    }
    print("=== BENCH_ZAYA_CONTRACT: passed ===")
}

func runZayaTopK(modelPath: String) async throws {
    let modelDir = URL(fileURLWithPath: modelPath)
    let env = ProcessInfo.processInfo.environment
    let prompt = env["BENCH_ZAYA_TOPK_PROMPT"] ?? "What is the capital of France?"
    let rawPrompt = env["BENCH_ZAYA_TOPK_RAW"] == "1"
    let enableThinking = env["BENCH_ZAYA_TOPK_THINKING"] == "1"
    let topK = Int(env["BENCH_ZAYA_TOPK_K"] ?? "10") ?? 10

    print("\n=== BENCH_ZAYA_TOPK: next-token parity probe ===")
    print("Model: \(modelPath)")
    print("Mode: \(rawPrompt ? "raw" : "chat") enableThinking=\(enableThinking)")

    let loadStart = CFAbsoluteTimeGetCurrent()
    let context = try await MLXLMCommon.loadModel(
        from: modelDir, using: #huggingFaceTokenizerLoader())
    print(String(format: "Load: %.2fs", CFAbsoluteTimeGetCurrent() - loadStart))

    let promptTokens: [Int]
    if rawPrompt {
        promptTokens = context.tokenizer.encode(text: prompt)
    } else {
        let messages: [[String: any Sendable]] = [
            ["role": "user", "content": prompt]
        ]
        promptTokens = try context.tokenizer.applyChatTemplate(
            messages: messages,
            tools: nil,
            additionalContext: ["enable_thinking": enableThinking])
    }

    print("Prompt tokens: \(promptTokens)")
    print("Prompt rendered: \(context.tokenizer.decode(tokenIds: promptTokens, skipSpecialTokens: false).debugDescription)")

    let promptIds = MLXArray(promptTokens.map { Int32($0) })
        .reshaped(1, promptTokens.count)
    let cache = context.model.newCache(parameters: nil)
    let logits = context.model(promptIds, cache: cache)
    let last = logits[0, promptTokens.count - 1, 0...]
    let order = argSort(-last)
    MLX.eval(order, last)

    var rows: [String] = []
    for i in 0..<min(topK, last.size) {
        let id = Int(order[i].item(Int32.self))
        let value = last[id].item(Float.self)
        let text = context.tokenizer.decode(tokenIds: [id], skipSpecialTokens: false)
            .replacingOccurrences(of: "\n", with: "\\n")
        rows.append(String(format: "#%02d id=%d logit=%.4f text=%@",
            i + 1, id, value, text))
    }
    print(rows.joined(separator: "\n"))
    print("=== BENCH_ZAYA_TOPK: done ===")
}

func runZayaMoEBits(modelPath: String) async throws {
    let modelDir = URL(fileURLWithPath: modelPath)
    print("\n=== BENCH_ZAYA_MOE_BITS: resolved JANGTQ MoE bit probe ===")
    print("Model: \(modelPath)")

    let configData = try Data(contentsOf: modelDir.appending(path: "config.json"))
    if let vlConfig = try? JSONDecoder.json5().decode(Zaya1VLConfiguration.self, from: configData)
    {
        print(
            "Decoded ZAYA1-VL config: weightFormat=\(vlConfig.weightFormat ?? "nil") " +
            "gateUp=\(vlConfig.routedExpertGateUpBits.map(String.init) ?? "nil") " +
            "down=\(vlConfig.routedExpertDownBits.map(String.init) ?? "nil") " +
            "seed=\(vlConfig.mxtqSeed.map(String.init) ?? "nil")"
        )
    }

    let loadStart = CFAbsoluteTimeGetCurrent()
    let context = try await MLXLMCommon.loadModel(
        from: modelDir, using: #huggingFaceTokenizerLoader())
    print(String(format: "Load: %.2fs", CFAbsoluteTimeGetCurrent() - loadStart))

    print("Dynamic model: \(type(of: context.model))")

    var rows: [String] = []
    let module = context.model as Module
    for (name, child) in module.namedModules() {
        if let layer = child as? TurboQuantSwitchGLU {
            rows.append(
                "\(name): input=\(layer.inputDims) hidden=\(layer.hiddenDims) " +
                "experts=\(layer.numExperts) gateUp=\(layer.gateUpBits) " +
                "down=\(layer.downBits) seed=\(layer.mxtqSeed)"
            )
        }
    }

    print("TurboQuantSwitchGLU modules reflected: \(rows.count)")
    for row in rows.prefix(12) {
        print(row)
    }
    if rows.count > 12 {
        print("... \(rows.count - 12) additional module(s) omitted")
    }
    print("=== BENCH_ZAYA_MOE_BITS: done ===")
}

func runZayaTQKernelProbe(modelPath: String) async throws {
    let modelDir = URL(fileURLWithPath: modelPath)
    let env = ProcessInfo.processInfo.environment
    let layer = Int(env["BENCH_ZAYA_TQ_KERNEL_PROBE_LAYER"] ?? "0") ?? 0
    let expertIDs = (env["BENCH_ZAYA_TQ_KERNEL_PROBE_EXPERTS"] ?? "0,7,15")
        .split(separator: ",")
        .compactMap { Int($0.trimmingCharacters(in: .whitespacesAndNewlines)) }

    print("\n=== BENCH_ZAYA_TQ_KERNEL_PROBE: actual tensor kernel parity ===")
    print("Model: \(modelPath)")
    print("Layer: \(layer) experts: \(expertIDs)")

    let config = try loadJSONObject(modelDir.appending(path: "config.json"))
    guard let hiddenSize = config["hidden_size"] as? Int else {
        throw NSError(
            domain: "RunBench.ZayaTQKernelProbe", code: 1,
            userInfo: [NSLocalizedDescriptionKey: "config.json missing hidden_size"])
    }

    let indexURL = modelDir.appending(path: "model.safetensors.index.json")
    let index = try loadJSONObject(indexURL)
    guard let weightMap = index["weight_map"] as? [String: String] else {
        throw NSError(
            domain: "RunBench.ZayaTQKernelProbe", code: 2,
            userInfo: [NSLocalizedDescriptionKey: "model.safetensors.index.json missing weight_map"])
    }

    var shardCache: [String: [String: MLXArray]] = [:]
    func tensor(_ key: String) throws -> MLXArray {
        guard let shard = weightMap[key] else {
            throw NSError(
                domain: "RunBench.ZayaTQKernelProbe", code: 3,
                userInfo: [NSLocalizedDescriptionKey: "missing tensor in index: \(key)"])
        }
        if shardCache[shard] == nil {
            shardCache[shard] = try MLX.loadArrays(url: modelDir.appending(path: shard))
        }
        guard let value = shardCache[shard]?[key] else {
            throw NSError(
                domain: "RunBench.ZayaTQKernelProbe", code: 4,
                userInfo: [NSLocalizedDescriptionKey: "tensor \(key) absent from shard \(shard)"])
        }
        return value
    }

    let sidecarURL = modelDir.appending(path: "jangtq_runtime.safetensors")
    guard FileManager.default.fileExists(atPath: sidecarURL.path) else {
        throw NSError(
            domain: "RunBench.ZayaTQKernelProbe", code: 5,
            userInfo: [NSLocalizedDescriptionKey: "missing jangtq_runtime.safetensors"])
    }
    let sidecar = try MLX.loadArrays(url: sidecarURL)

    let prefix = "model.layers.\(layer).zaya_block.experts.switch_mlp"
    let gatePacked = try tensor("\(prefix).gate_proj.tq_packed")
    let gateNorms = try tensor("\(prefix).gate_proj.tq_norms")
    let upPacked = try tensor("\(prefix).up_proj.tq_packed")
    let upNorms = try tensor("\(prefix).up_proj.tq_norms")
    let downPacked = try tensor("\(prefix).down_proj.tq_packed")
    let downNorms = try tensor("\(prefix).down_proj.tq_norms")

    let hiddenDims = gatePacked.dim(1)
    let numExperts = gatePacked.dim(0)
    let gateBits = inferTQBits(inFeatures: hiddenSize, packedCols: gatePacked.dim(-1))
    let upBits = inferTQBits(inFeatures: hiddenSize, packedCols: upPacked.dim(-1))
    let downBits = inferTQBits(inFeatures: hiddenDims, packedCols: downPacked.dim(-1))
    guard gateBits == upBits else {
        throw NSError(
            domain: "RunBench.ZayaTQKernelProbe", code: 6,
            userInfo: [NSLocalizedDescriptionKey:
                "gate/up bits differ: gate=\(gateBits) up=\(upBits)"])
    }

    guard let gateCodebook = sidecar["codebook.\(hiddenSize).\(gateBits)"],
          let downCodebook = sidecar["codebook.\(hiddenDims).\(downBits)"]
    else {
        throw NSError(
            domain: "RunBench.ZayaTQKernelProbe", code: 7,
            userInfo: [NSLocalizedDescriptionKey:
                "sidecar missing codebook.\(hiddenSize).\(gateBits) or codebook.\(hiddenDims).\(downBits)"])
    }

    MLX.eval(
        gatePacked, gateNorms, upPacked, upNorms, downPacked, downNorms,
        gateCodebook, downCodebook)

    let gatePackedCPU = gatePacked.asArray(UInt32.self)
    let upPackedCPU = upPacked.asArray(UInt32.self)
    let downPackedCPU = downPacked.asArray(UInt32.self)
    let gateNormsCPU = gateNorms.asArray(Float.self)
    let upNormsCPU = upNorms.asArray(Float.self)
    let downNormsCPU = downNorms.asArray(Float.self)
    let gateCodebookCPU = gateCodebook.asArray(Float.self)
    let downCodebookCPU = downCodebook.asArray(Float.self)

    let x = deterministicProbeVector(count: hiddenSize)
    let xArray = MLXArray(x).reshaped([1, hiddenSize])
    print(
        "Resolved: hidden=\(hiddenSize) intermediate=\(hiddenDims) experts=\(numExperts) " +
        "gateBits=\(gateBits) upBits=\(upBits) downBits=\(downBits)")

    var failures: [String] = []
    for expert in expertIDs {
        guard expert >= 0 && expert < numExperts else {
            failures.append("expert \(expert) out of range 0..<\(numExperts)")
            continue
        }

        let rhs = MLXArray([UInt32(expert)])

        let gateMetal = JANGTQKernels.gatherTQTopK(
            xRot: xArray,
            packed: gatePacked,
            norms: gateNorms,
            codebook: gateCodebook,
            rhsIndices: rhs,
            batchTokens: 1,
            K: 1,
            inFeatures: hiddenSize,
            outFeatures: hiddenDims,
            bits: gateBits)
        let gateCPU = tqMatmulReference(
            x: x,
            packed: gatePackedCPU,
            norms: gateNormsCPU,
            codebook: gateCodebookCPU,
            expert: expert,
            numExperts: numExperts,
            inFeatures: hiddenSize,
            outFeatures: hiddenDims,
            bits: gateBits)

        let upMetal = JANGTQKernels.gatherTQTopK(
            xRot: xArray,
            packed: upPacked,
            norms: upNorms,
            codebook: gateCodebook,
            rhsIndices: rhs,
            batchTokens: 1,
            K: 1,
            inFeatures: hiddenSize,
            outFeatures: hiddenDims,
            bits: upBits)
        let upCPU = tqMatmulReference(
            x: x,
            packed: upPackedCPU,
            norms: upNormsCPU,
            codebook: gateCodebookCPU,
            expert: expert,
            numExperts: numExperts,
            inFeatures: hiddenSize,
            outFeatures: hiddenDims,
            bits: upBits)

        let fusedMetal = JANGTQKernels.fusedGateUpSwiGLU(
            xRot: xArray,
            packedGate: gatePacked,
            normsGate: gateNorms,
            packedUp: upPacked,
            normsUp: upNorms,
            codebook: gateCodebook,
            rhsIndices: rhs,
            batchTokens: 1,
            K: 1,
            inFeatures: hiddenSize,
            outFeatures: hiddenDims,
            bits: gateBits)
        let fusedCPU = zip(gateCPU, upCPU).map { gate, up -> Float in
            let silu = gate / Float(1.0 + Foundation.exp(Double(-gate)))
            return silu * up
        }

        let downInput = MLXArray(fusedCPU).reshaped([1, hiddenDims])
        let downMetal = JANGTQKernels.gatherTQ(
            xRot: downInput,
            packed: downPacked,
            norms: downNorms,
            codebook: downCodebook,
            rhsIndices: rhs,
            nRows: 1,
            inFeatures: hiddenDims,
            outFeatures: hiddenSize,
            bits: downBits)
        let downCPU = tqMatmulReference(
            x: fusedCPU,
            packed: downPackedCPU,
            norms: downNormsCPU,
            codebook: downCodebookCPU,
            expert: expert,
            numExperts: numExperts,
            inFeatures: hiddenDims,
            outFeatures: hiddenSize,
            bits: downBits)

        MLX.eval(gateMetal, upMetal, fusedMetal, downMetal)
        let gateDiff = maxAbsDiff(gateMetal.asArray(Float.self), gateCPU)
        let upDiff = maxAbsDiff(upMetal.asArray(Float.self), upCPU)
        let fusedDiff = maxAbsDiff(fusedMetal.asArray(Float.self), fusedCPU)
        let downDiff = maxAbsDiff(downMetal.asArray(Float.self), downCPU)

        print(String(
            format: "expert=%02d gate_max=%.6f up_max=%.6f fused_max=%.6f down_max=%.6f",
            expert, gateDiff, upDiff, fusedDiff, downDiff))

        if gateDiff > 0.02 { failures.append("expert \(expert) gate diff \(gateDiff)") }
        if upDiff > 0.02 { failures.append("expert \(expert) up diff \(upDiff)") }
        if fusedDiff > 0.05 { failures.append("expert \(expert) fused diff \(fusedDiff)") }
        if downDiff > 0.02 { failures.append("expert \(expert) down diff \(downDiff)") }
    }

    if !failures.isEmpty {
        fputs("BENCH_ZAYA_TQ_KERNEL_PROBE: \(failures.count) failure(s): \(failures)\n", stderr)
        exit(1)
    }
    print("=== BENCH_ZAYA_TQ_KERNEL_PROBE: passed ===")
}

private func inferTQBits(inFeatures: Int, packedCols: Int) -> Int {
    guard inFeatures > 0 else { return 0 }
    return max(1, (packedCols * 32) / inFeatures)
}

private func deterministicProbeVector(count: Int) -> [Float] {
    (0 ..< count).map { index in
        Float(((index * 37 + 17) % 211) - 105) / 64.0
    }
}

private func tqMatmulReference(
    x: [Float],
    packed: [UInt32],
    norms: [Float],
    codebook: [Float],
    expert: Int,
    numExperts: Int,
    inFeatures: Int,
    outFeatures: Int,
    bits: Int
) -> [Float] {
    precondition(expert >= 0 && expert < numExperts)
    let valsPerU32 = 32 / bits
    let mask = UInt32((1 << bits) - 1)
    let packedCols = (inFeatures + valsPerU32 - 1) / valsPerU32
    let expertPackedBase = expert * outFeatures * packedCols
    let expertNormBase = expert * outFeatures
    var output = [Float](repeating: 0, count: outFeatures)
    for out in 0 ..< outFeatures {
        var acc: Float = 0
        let rowBase = expertPackedBase + out * packedCols
        for packIndex in 0 ..< packedCols {
            let word = packed[rowBase + packIndex]
            let inputBase = packIndex * valsPerU32
            for slot in 0 ..< valsPerU32 {
                let inputIndex = inputBase + slot
                if inputIndex >= inFeatures { break }
                let codeIndex = Int((word >> UInt32(slot * bits)) & mask)
                acc += x[inputIndex] * codebook[codeIndex]
            }
        }
        output[out] = acc * norms[expertNormBase + out]
    }
    return output
}

private func maxAbsDiff(_ lhs: [Float], _ rhs: [Float]) -> Float {
    guard lhs.count == rhs.count else { return .infinity }
    return zip(lhs, rhs).map { abs($0 - $1) }.max() ?? 0
}

private func loadJSONObject(_ url: URL) throws -> [String: Any] {
    let data = try Data(contentsOf: url)
    guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
        throw NSError(
            domain: "RunBench.ConfigSmoke", code: 1,
            userInfo: [NSLocalizedDescriptionKey: "not a JSON object: \(url.path)"])
    }
    return json
}

private func layerSet(_ keys: [String], marker: String) -> Set<Int> {
    Set(keys.compactMap { key -> Int? in
        guard key.contains(marker) else { return nil }
        let parts = key.split(separator: ".")
        guard parts.count > 2, parts[0] == "model", parts[1] == "layers" else {
            return nil
        }
        return Int(parts[2])
    })
}

private func jsonAt(_ object: [String: Any], _ path: [String]) -> Any? {
    var current: Any? = object
    for key in path {
        current = (current as? [String: Any])?[key]
    }
    return current
}

private func firstJSONInt(_ object: [String: Any], _ paths: [[String]]) -> Int? {
    for path in paths {
        if let value = jsonInt(jsonAt(object, path)) { return value }
    }
    return nil
}

private func jsonString(_ value: Any?) -> String? {
    value as? String
}

private func jsonInt(_ value: Any?) -> Int? {
    if let int = value as? Int { return int }
    if let number = value as? NSNumber { return number.intValue }
    return nil
}

private func jsonInts(_ value: Any?) -> [Int] {
    if let int = jsonInt(value) { return [int] }
    return (value as? [Any])?.compactMap(jsonInt) ?? []
}

private struct RoutedBitsDescription {
    var uniform: Int?
    var gateUp: Int?
    var down: Int?

    var description: String {
        if gateUp != nil || down != nil {
            return "gateUp:\(gateUp.map(String.init) ?? "nil"),down:\(down.map(String.init) ?? "nil")"
        }
        return uniform.map(String.init) ?? "nil"
    }
}

private func resolveRoutedBits(_ value: Any?) -> RoutedBitsDescription {
    if let int = jsonInt(value) {
        return .init(uniform: int, gateUp: nil, down: nil)
    }
    guard let dict = value as? [String: Any] else {
        return .init(uniform: nil, gateUp: nil, down: nil)
    }
    if let routedInt = jsonInt(dict["routed_expert"] ?? dict["routed"]) {
        return .init(uniform: routedInt, gateUp: nil, down: nil)
    }
    if let routed = (dict["routed_expert"] ?? dict["routed"]) as? [String: Any] {
        return .init(
            uniform: nil,
            gateUp: jsonInt(routed["gate_up_proj"] ?? routed["gate_proj"] ?? routed["up_proj"]),
            down: jsonInt(routed["down_proj"]))
    }
    return .init(
        uniform: nil,
        gateUp: jsonInt(dict["gate_up_proj"] ?? dict["gate_proj"] ?? dict["up_proj"]),
        down: jsonInt(dict["down_proj"]))
}

/// Match the production LLM/VLM factory tokenizer path for tokenizer-only
/// bench modes: JANG source-tokenizer fallback first, then tokenizer_class
/// substitution for classes swift-transformers cannot instantiate directly.
private func loadBenchTokenizer(from modelDir: URL) async throws -> (any MLXLMCommon.Tokenizer, URL) {
    let jangResolvedDir = JangLoader.resolveTokenizerDirectory(for: modelDir)
    let templateResolvedDir = JangLoader.resolveChatTemplateSidecarSubstitution(
        for: jangResolvedDir)
    let tokenizerDir = JangLoader.resolveTokenizerClassSubstitution(for: templateResolvedDir)
    let loader = #huggingFaceTokenizerLoader()
    return (try await loader.load(from: tokenizerDir), tokenizerDir)
}

/// Tokenizer-only smoke for model-family chat-template compatibility.
/// It catches missing templates, swift-jinja parser/runtime failures,
/// lost tool schemas, and raw Jinja leakage before a full model load.
func runTemplateSmoke(modelPath: String) async throws {
    struct TemplateSmokeCase {
        let label: String
        let messages: [[String: any Sendable]]
        let tools: [[String: any Sendable]]?
        let context: [String: any Sendable]?
    }

    let modelDir = URL(fileURLWithPath: modelPath)
    print("\n=== BENCH_TEMPLATE_SMOKE: chat-template/Jinja/fallback ===")
    print("Model: \(modelPath)")
    print("Loading tokenizer only (no model forward)...")

    let (tokenizer, tokenizerDir) = try await loadBenchTokenizer(from: modelDir)
    print("Tokenizer dir: \(tokenizerDir.path)")
    print("Tokenizer: bos=\(tokenizer.bosToken ?? "nil") eos=\(tokenizer.eosToken ?? "nil")")
    let directTemplate: Template? = {
        let candidate = tokenizerDir.appendingPathComponent("chat_template.jinja")
        let source: String?
        if let fileSource = try? String(contentsOf: candidate, encoding: .utf8) {
            source = fileSource
        } else {
            let configURL = tokenizerDir.appendingPathComponent("tokenizer_config.json")
            if let data = try? Data(contentsOf: configURL),
               let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
               let configSource = json["chat_template"] as? String {
                source = configSource
            } else {
                source = nil
            }
        }
        guard let source else { return nil }
        return try? Template(source, with: .init(lstripBlocks: true, trimBlocks: true))
    }()

    let plainMessages: [[String: any Sendable]] = [
        ["role": "system", "content": "You are concise."],
        ["role": "user", "content": "Say OK."],
    ]
    let multiTurnMessages: [[String: any Sendable]] = [
        ["role": "system", "content": "Keep answers short."],
        ["role": "user", "content": "Say alpha."],
        ["role": "assistant", "content": "alpha"],
        ["role": "user", "content": "Now say beta."],
    ]
    let reasoningHistoryMessages: [[String: any Sendable]] = [
        ["role": "user", "content": "What is 2+2?"],
        ["role": "assistant", "reasoning_content": "2+2=4", "content": "4"],
        ["role": "user", "content": "Now answer 3+1 with one digit."],
    ]

    let weatherParams: [String: any Sendable] = [
        "type": "object",
        "properties": [
            "city": ["type": "string", "description": "City name"] as [String: any Sendable],
        ] as [String: any Sendable],
        "required": ["city"],
    ]
    let weatherFn: [String: any Sendable] = [
        "name": "get_weather",
        "description": "Get current weather for a city.",
        "parameters": weatherParams,
    ]
    let weatherTool: [String: any Sendable] = [
        "type": "function",
        "function": weatherFn,
    ]
    let timeParams: [String: any Sendable] = [
        "type": "object",
        "properties": [
            "timezone": ["type": "string", "description": "IANA timezone"] as [String: any Sendable],
        ] as [String: any Sendable],
        "required": ["timezone"],
    ]
    let timeFn: [String: any Sendable] = [
        "name": "get_time",
        "description": "Get current time in a timezone.",
        "parameters": timeParams,
    ]
    let timeTool: [String: any Sendable] = [
        "type": "function",
        "function": timeFn,
    ]
    func osaurusSizedTool(_ index: Int) -> [String: any Sendable] {
        let params: [String: any Sendable] = [
            "type": "object",
            "properties": [
                "path": [
                    "type": "string",
                    "description": "Workspace-relative path or absolute path selected by the user.",
                ] as [String: any Sendable],
                "query": [
                    "type": "string",
                    "description": "Natural language request, command text, search pattern, or code fragment.",
                ] as [String: any Sendable],
                "mode": [
                    "type": "string",
                    "enum": ["read", "write", "append", "replace", "search"],
                    "description": "Operation mode.",
                ] as [String: any Sendable],
                "options": [
                    "type": "object",
                    "properties": [
                        "recursive": ["type": "boolean"] as [String: any Sendable],
                        "max_results": ["type": "integer"] as [String: any Sendable],
                        "globs": [
                            "type": "array",
                            "items": ["type": "string"] as [String: any Sendable],
                        ] as [String: any Sendable],
                    ] as [String: any Sendable],
                    "additionalProperties": false,
                ] as [String: any Sendable],
            ] as [String: any Sendable],
            "required": ["path", "query"],
            "additionalProperties": false,
        ]
        return [
            "type": "function",
            "function": [
                "name": "osaurus_probe_tool_\(index)",
                "description": String(
                    repeating:
                        "Synthetic Osaurus-sized sandbox/file/memory tool used to measure native chat-template rendering cost. ",
                    count: 4),
                "parameters": params,
            ] as [String: any Sendable],
        ]
    }
    let osaurusSizedTools = (0..<9).map(osaurusSizedTool)

    let cases: [TemplateSmokeCase] = [
        .init(label: "plain", messages: plainMessages, tools: nil, context: nil),
        .init(label: "thinking_false", messages: plainMessages, tools: nil,
              context: ["enable_thinking": false]),
        .init(label: "thinking_true", messages: plainMessages, tools: nil,
              context: ["enable_thinking": true]),
        .init(label: "reasoning_max", messages: plainMessages, tools: nil,
              context: ["enable_thinking": true, "reasoning_effort": "max"]),
        .init(label: "tools_thinking_true", messages: [
            ["role": "user", "content": "Call get_weather for Tokyo and get_time for Asia/Tokyo."],
        ], tools: [weatherTool, timeTool], context: ["enable_thinking": true]),
        .init(label: "osaurus_sized_tools_thinking_true", messages: [
            ["role": "system", "content": String(repeating: "You are a local Osaurus agent. Keep answers concise. ", count: 30)],
            ["role": "user", "content": "Use the available local tools only if needed. Say hello."],
            ["role": "assistant", "reasoning_content": "The user is greeting.", "content": "Hi."],
            ["role": "user", "content": "Now answer again briefly."],
        ], tools: osaurusSizedTools, context: ["enable_thinking": true]),
        .init(label: "multi_turn_off", messages: multiTurnMessages, tools: nil,
              context: ["enable_thinking": false]),
        .init(label: "reasoning_history", messages: reasoningHistoryMessages, tools: nil,
              context: ["enable_thinking": true]),
    ]

    func clippedTail(_ text: String, count: Int = 160) -> String {
        String(text.suffix(count))
            .replacingOccurrences(of: "\n", with: "\\n")
            .replacingOccurrences(of: "\"", with: "\\\"")
    }

    var failures: [String] = []
    for testCase in cases {
        do {
            let start = CFAbsoluteTimeGetCurrent()
            let ids = try tokenizer.applyChatTemplate(
                messages: testCase.messages,
                tools: testCase.tools,
                additionalContext: testCase.context)
            let renderMs = Int((CFAbsoluteTimeGetCurrent() - start) * 1000)
            let text = tokenizer.decode(tokenIds: ids, skipSpecialTokens: false)
            var directProfile = "direct=unavailable"
            if let directTemplate {
                var rawContext: [String: Any] = [
                    "messages": testCase.messages,
                    "add_generation_prompt": true,
                ]
                rawContext["bos_token"] = tokenizer.bosToken ?? ""
                rawContext["eos_token"] = tokenizer.eosToken ?? ""
                if let tools = testCase.tools {
                    rawContext["tools"] = tools
                }
                if let context = testCase.context {
                    for (key, value) in context {
                        rawContext[key] = value
                    }
                }
                if rawContext["thinking"] == nil,
                   let enableThinking = rawContext["enable_thinking"]
                {
                    rawContext["thinking"] = enableThinking
                }
                var jinjaValues: [String: Value] = [:]
                let valueStart = CFAbsoluteTimeGetCurrent()
                for (key, value) in rawContext {
                    jinjaValues[key] = try Value(any: value)
                }
                let valueMs = Int((CFAbsoluteTimeGetCurrent() - valueStart) * 1000)
                let directRenderStart = CFAbsoluteTimeGetCurrent()
                let rendered = try directTemplate.render(jinjaValues)
                let directRenderMs = Int((CFAbsoluteTimeGetCurrent() - directRenderStart) * 1000)
                let encodeStart = CFAbsoluteTimeGetCurrent()
                let directIds = tokenizer.encode(text: rendered, addSpecialTokens: false)
                let encodeMs = Int((CFAbsoluteTimeGetCurrent() - encodeStart) * 1000)
                let directToolMention = testCase.tools == nil
                    || rendered.contains("get_weather")
                    || rendered.contains("get_time")
                    || rendered.contains("osaurus_probe_tool_0")
                directProfile = "valueMs=\(valueMs) directRenderMs=\(directRenderMs) encodeMs=\(encodeMs) directIds=\(directIds.count) directChars=\(rendered.count) directToolMention=\(directToolMention) directTail=\"\(clippedTail(rendered, count: 80))\""
            }
            let hasJinjaLeak = text.contains("{{") || text.contains("{%") || text.contains("{#")
            let hasThinkOpen = text.contains("<think>") || text.contains("<|channel|>analysis")
                || text.contains("<channel>analysis") || text.contains("[MODEL_SETTINGS]")
            let hasThinkClose = text.contains("</think>") || text.contains("<|end|>")
                || text.contains("</channel>")
            let toolMention = testCase.tools == nil
                || text.contains("get_weather")
                || text.contains("get_time")
                || text.contains("osaurus_probe_tool_0")
            let ok = !ids.isEmpty && !text.isEmpty && !hasJinjaLeak && toolMention
            print(
                "TEMPLATE_SMOKE label=\(testCase.label) status=\(ok ? "PASS" : "FAIL") ms=\(renderMs) \(directProfile) ids=\(ids.count) chars=\(text.count) tools=\(testCase.tools?.count ?? 0) thinkOpen=\(hasThinkOpen) thinkClose=\(hasThinkClose) toolMention=\(toolMention) tail=\"\(clippedTail(text))\""
            )
            if !ok {
                var reasons: [String] = []
                if ids.isEmpty { reasons.append("empty ids") }
                if text.isEmpty { reasons.append("empty decode") }
                if hasJinjaLeak { reasons.append("raw Jinja leaked") }
                if !toolMention { reasons.append("tool schema/name missing") }
                failures.append("\(testCase.label): \(reasons.joined(separator: ", "))")
            }
        } catch {
            print("TEMPLATE_SMOKE label=\(testCase.label) status=FAIL error=\(error)")
            failures.append("\(testCase.label): \(error)")
        }
    }

    if !failures.isEmpty {
        fputs("BENCH_TEMPLATE_SMOKE: \(failures.count) failure(s): \(failures)\n", stderr)
        exit(1)
    }

    print("=== BENCH_TEMPLATE_SMOKE: passed ===")
}

// MARK: - Orphan-slot consumer-cancellation reproducer

/// Mimics the osaurus-reported pattern: consumer breaks the for-await
/// loop early (Task.isCancelled), engine slot keeps stepping
/// internally, second request fires and collides with the orphan
/// slot's Metal pipelines mid-encode.
///
/// Pre-`continuation.onTermination` fix: request B crashed inside
/// `BatchEngine.stepPrefill` with
/// `notifyExternalReferencesNonZeroOnDealloc`.
/// Post-fix: termination handler reaps slot A; request B prefills
/// against a clean state.
///
/// Uses `BatchEngine` with `CacheCoordinator(usePagedCache=true,
/// enableDiskCache=true)` matching the osaurus config so the cache-
/// hit path actually fires for request B.
func runOrphanSlotRepro(modelPath: String, maxNew: Int) async throws {
    let modelDir = URL(fileURLWithPath: modelPath)
    print("\n=== BENCH_ORPHAN_SLOT_REPRO: consumer-cancel → reuse-cache pattern ===")
    let loadStart = CFAbsoluteTimeGetCurrent()
    let context = try await MLXLMCommon.loadModel(
        from: modelDir, using: #huggingFaceTokenizerLoader())
    print(String(format: "Load: %.2fs", CFAbsoluteTimeGetCurrent() - loadStart))
    print("Model: \(type(of: context.model))")

    let diskDir = FileManager.default.temporaryDirectory
        .appendingPathComponent("vmlx-orphan-repro-\(UUID().uuidString)")
    try FileManager.default.createDirectory(
        at: diskDir, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: diskDir) }

    var coordCfg = CacheCoordinatorConfig()
    coordCfg.usePagedCache = true
    coordCfg.enableDiskCache = true
    coordCfg.diskCacheDir = diskDir
    coordCfg.modelKey = modelDir.lastPathComponent
    coordCfg.pagedBlockSize = 64
    coordCfg.maxCacheBlocks = 512
    let coord = CacheCoordinator(config: coordCfg)

    nonisolated(unsafe) let ctx = context
    let engine = BatchEngine(
        context: ctx, maxBatchSize: 4, cacheCoordinator: coord)

    let userQuery = "Name one fruit."
    let messages: [[String: any Sendable]] = [
        ["role": "user", "content": userQuery]
    ]
    let promptTokens = try context.tokenizer.applyChatTemplate(
        messages: messages, tools: nil, additionalContext: nil)
    print("Prompt tokens: \(promptTokens.count)")

    func buildInput() -> LMInput {
        let arr = MLXArray(promptTokens.map { Int32($0) })
            .reshaped(1, promptTokens.count)
        return LMInput(text: LMInput.Text(tokens: arr))
    }

    let params = GenerateParameters(
        maxTokens: maxNew, temperature: 0.7, topP: 0.8,
        prefillStepSize: 512)

    // ─── Request A: consumer breaks after 4 tokens (simulates Task.isCancelled) ───
    print("\n[Request A] starting (consumer will break after 4 tokens)...")
    let aStart = CFAbsoluteTimeGetCurrent()
    let aInput = buildInput()
    nonisolated(unsafe) let aSend = aInput
    let aStream = await engine.generate(input: aSend, parameters: params)
    var aTokenCount = 0
    var aBroken = false
    for await event in aStream {
        if case .chunk = event { aTokenCount += 1 }
        if case .reasoning = event { aTokenCount += 1 }
        if aTokenCount >= 4 && !aBroken {
            aBroken = true
            print("  [Request A] consumer breaks after 4 tokens")
            break  // ← THIS is the pattern that orphans the slot pre-fix
        }
    }
    let aDur = CFAbsoluteTimeGetCurrent() - aStart
    print(String(format: "  [Request A] consumer exited at %.2fs", aDur))

    // Give the slot a moment to react — onTermination -> cancel(id) is async.
    try? await Task.sleep(nanoseconds: 200_000_000)

    // ─── Request B: same prompt — would hit warm cache and collide ───
    print("\n[Request B] starting (same prompt; cache-hit path)...")
    let bStart = CFAbsoluteTimeGetCurrent()
    let bInput = buildInput()
    nonisolated(unsafe) let bSend = bInput
    let bStream = await engine.generate(input: bSend, parameters: params)
    var bTokenCount = 0
    var bFinishReason: String = "(no info)"
    for await event in bStream {
        if case .chunk = event { bTokenCount += 1 }
        if case .reasoning = event { bTokenCount += 1 }
        if case .info(let info) = event {
            switch info.stopReason {
            case .stop: bFinishReason = "stop (EOS)"
            case .length: bFinishReason = "length"
            case .cancelled: bFinishReason = "cancelled"
            default: bFinishReason = "other"
            }
        }
    }
    let bDur = CFAbsoluteTimeGetCurrent() - bStart
    print(String(format: "  [Request B] completed in %.2fs", bDur))
    print("  [Request B] tokens: \(bTokenCount), finish: \(bFinishReason)")

    if bTokenCount > 0 {
        print("\n=== BENCH_ORPHAN_SLOT_REPRO: PASSED — request B completed without crash ===")
    } else {
        print("\n=== BENCH_ORPHAN_SLOT_REPRO: FAIL — request B produced 0 tokens ===")
        exit(1)
    }
}

// MARK: - Thinking-loop diagnostic probe

/// Probe the model's `</think>` emission behavior on a validation-style
/// prompt that's known to trigger self-refinement loops in
/// reasoning-trained models ("give me 20 random digits"). Reports:
///   - whether `</think>` was ever emitted
///   - reasoning-token count vs content-token count
///   - the LAST 200 chars of reasoning + FIRST 200 chars of content
///   - finish reason
///
    /// Sampling uses the bundle's `generation_config.json` by default.
    /// The probe prints the effective values so a failed run cannot be
    /// mistaken for production parity if a caller overrides them.
///
/// Used to A/B JANGTQ bits=2 vs JANGTQ4 bits=4 on the same task to
/// isolate whether 4-bit quantization compresses the EOS / `</think>`
/// margin enough to cause infinite self-refinement loops.
func runThinkingLoopProbe(modelPath: String, maxNew: Int) async throws {
    let modelDir = URL(fileURLWithPath: modelPath)
    print("\n=== BENCH_THINK_LOOP_PROBE: validation-task </think> emission ===")
    let loadStart = CFAbsoluteTimeGetCurrent()
    let (context, jangPressRuntime) = try await MLXLMCommon.loadModel(
        from: modelDir,
        using: #huggingFaceTokenizerLoader(),
        loadConfiguration: .default)
    _ = jangPressRuntime
    print(String(format: "Load: %.2fs", CFAbsoluteTimeGetCurrent() - loadStart))
    print("Model: \(type(of: context.model))")
    print(String(format: "maxTokens budget: %d", maxNew))

    // Validation-style prompt — known to trigger self-refinement loops
    // in Qwen3.x / DeepSeek-V4 / MiniMax-M2 reasoning-trained models.
    let env = ProcessInfo.processInfo.environment
    let promptTokens: [Int]
    let syntheticHistoryTurns: Int
    let syntheticToolCount: Int
    let enableThinking: Bool
    if let promptFile = env["BENCH_THINK_RENDERED_PROMPT_FILE"],
       !promptFile.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
        let raw = try String(contentsOfFile: promptFile, encoding: .utf8)
        let rendered: String
        if let bodyStart = raw.range(of: "\n\n")?.upperBound {
            rendered = String(raw[bodyStart...])
        } else {
            rendered = raw
        }
        promptTokens = context.tokenizer.encode(
            text: rendered,
            addSpecialTokens: false)
        syntheticHistoryTurns = -1
        syntheticToolCount = -1
        enableThinking = rendered.contains("<think>")
        print("rendered prompt file: \(promptFile)")
    } else {
        let userQuery = env["BENCH_THINK_PROMPT"]
            ?? "Give me a random 20-digit number. Only return the number itself."
        var messages: [[String: any Sendable]] = []
        if let systemPrompt = env["BENCH_THINK_SYSTEM"],
           !systemPrompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            messages.append(["role": "system", "content": systemPrompt])
        }
        syntheticHistoryTurns = max(0, Int(env["BENCH_THINK_HISTORY_TURNS"] ?? "0") ?? 0)
        if syntheticHistoryTurns > 0 {
            for index in 0..<syntheticHistoryTurns {
                messages.append([
                    "role": "user",
                    "content": index == 0 ? "hi" : "what are u and what can u do",
                ])
                messages.append([
                    "role": "assistant",
                    "content": index == 0
                        ? "Hello! How can I help you today?"
                        : "I'm Osaurus, a local macOS assistant. I can help inspect files, answer questions, and use tools when needed.",
                    "reasoning_content": "The prior user message was simple, so answer directly without tools.",
                ])
            }
        }
        messages.append(["role": "user", "content": userQuery])

        syntheticToolCount = max(0, Int(env["BENCH_THINK_TOOLS"] ?? "0") ?? 0)
        let syntheticTools: [[String: any Sendable]]? = syntheticToolCount > 0
            ? (0..<syntheticToolCount).map { index in
                let properties: [String: any Sendable] = [
                    "query": [
                        "type": "string",
                        "description": "User request, file path, command, or lookup text for tool \(index).",
                    ] as [String: any Sendable],
                    "limit": [
                        "type": "integer",
                        "description": "Maximum result count.",
                    ] as [String: any Sendable],
                ]
                let parameters: [String: any Sendable] = [
                    "type": "object",
                    "properties": properties,
                    "required": ["query"],
                ]
                let function: [String: any Sendable] = [
                    "name": "osaurus_probe_tool_\(index)",
                    "description": "Synthetic osaurus-style diagnostic tool \(index). Used only to test whether a larger tool roster changes reasoning close behavior.",
                    "parameters": parameters,
                ]
                return [
                    "type": "function",
                    "function": function,
                ] as [String: any Sendable]
            }
            : nil
        // Default enable_thinking=true; set THINK=0 env to test the
        // chat-no-think workaround on the same prompt.
        let envThink = env["THINK"]
        enableThinking = (envThink ?? "1") != "0"
        promptTokens = try context.tokenizer.applyChatTemplate(
            messages: messages,
            tools: syntheticTools,
            additionalContext: ["enable_thinking": enableThinking])
        print("system message: \(messages.first?["role"] as? String == "system" ? "YES" : "NO")")
    }
    print("enable_thinking: \(enableThinking)")
    print("synthetic history turns: \(syntheticHistoryTurns)")
    print("synthetic tools: \(syntheticToolCount)")
    print("\nPrompt tokens: \(promptTokens.count)")
    print(
        "Prompt rendered tail (last 200 chars): "
            + context.tokenizer.decode(
                tokenIds: Array(promptTokens.suffix(80)),
                skipSpecialTokens: false
            ).debugDescription)

    let promptArr = MLXArray(promptTokens.map { Int32($0) })
        .reshaped(1, promptTokens.count)
    let input = LMInput(text: LMInput.Text(tokens: promptArr))
    nonisolated(unsafe) let send = input
    nonisolated(unsafe) let ctx = context

    // Set up reasoning parser to split the stream.
    let promptTail = context.tokenizer.decode(
        tokenIds: Array(promptTokens.suffix(20)), skipSpecialTokens: false)
    let stamp = context.configuration.reasoningParserName ?? "think_xml"
    var parser = ReasoningParser.forPrompt(stampName: stamp, promptTail: promptTail)
    print("Reasoning stamp: \(stamp)")

    var fallbackParams = GenerateParameters(
        maxTokens: maxNew,
        prefillStepSize: 512)
    if env["BENCH_THINK_COMPILED"] == "1" {
        fallbackParams.enableCompiledBatchDecode = true
    }
    let params = GenerateParameters(
        generationConfig: context.configuration.generationDefaults,
        fallback: fallbackParams)
    print(
        String(
            format:
                "Sampling: maxTokens=%d temp=%.3f topP=%.3f topK=%d minP=%.3f rep=%@ autoCloseBias=%@",
            params.maxTokens ?? -1,
            Double(params.temperature),
            Double(params.topP),
            params.topK,
            Double(params.minP),
            params.repetitionPenalty.map { String(format: "%.3f", Double($0)) } ?? "nil",
            (ProcessInfo.processInfo.environment["BENCH_THINK_NO_AUTO_BIAS"] == "1") ? "disabled" : "default"))
    var reasoningOut = ""
    var contentOut = ""
    var rawTokenCount = 0
    var sawCloseTag = false
    var finishReason: String = "unknown"
    let startTime = CFAbsoluteTimeGetCurrent()
    let coordinator: CacheCoordinator?
    if env["BENCH_THINK_PROD_COORD"] == "1" {
        let cacheRoot = env["BENCH_THINK_CACHE_DIR"]
            ?? FileManager.default.homeDirectoryForCurrentUser
                .appendingPathComponent(".osaurus/cache/kv_v2")
                .path
        let modelName = URL(fileURLWithPath: modelPath).lastPathComponent
        let scopedKey = env["BENCH_THINK_MODEL_KEY"]
            ?? "\(modelName)|kv=fp16"
        let cfg = CacheCoordinatorConfig(
            usePagedCache: true,
            enableDiskCache: true,
            diskCacheDir: URL(fileURLWithPath: cacheRoot, isDirectory: true),
            ssmMaxEntries: 50,
            enableSSMReDerive: false,
            modelKey: scopedKey,
            defaultKVMode: .none,
            defaultMaxKVSize: 65536,
            longPromptMultiplier: 2.0)
        let coord = CacheCoordinator(config: cfg)
        if env["BENCH_THINK_HYBRID"] == "1" {
            coord.setHybrid(true)
        }
        coordinator = coord
        print("Coordinator: enabled cacheDir=\(cacheRoot) modelKey=\(scopedKey) hybrid=\(env["BENCH_THINK_HYBRID"] == "1")")
    } else {
        coordinator = nil
    }
    let engine = BatchEngine(
        context: ctx,
        maxBatchSize: 1,
        cacheCoordinator: coordinator)
    let path = env["BENCH_THINK_PATH"]?.lowercased() ?? "submit"
    print("engine path: \(path)")
    if path == "generate" {
        let stream = await engine.generate(input: send, parameters: params)
        for await event in stream {
            switch event {
            case .chunk(let text):
                contentOut += text
            case .reasoning(let text):
                reasoningOut += text
            case .toolCall:
                break
            case .info(let info):
                rawTokenCount = max(rawTokenCount, info.generationTokenCount)
                switch info.stopReason {
                case .stop: finishReason = "stop (EOS)"
                case .length: finishReason = "length (max_tokens)"
                case .cancelled: finishReason = "cancelled"
                default: finishReason = "other"
                }
                if info.unclosedReasoning {
                    finishReason += " / unclosedReasoning"
                }
            }
        }
        parser = nil
    } else {
        let (_, stream) = await engine.submit(input: send, parameters: params)
        for await event in stream {
            switch event {
            case .token(let id):
                rawTokenCount += 1
                let piece = context.tokenizer.decode(
                    tokenIds: [id], skipSpecialTokens: false)
                // Watch for raw </think> emission BEFORE the parser strips it.
                if piece.contains("</think>") || (parser != nil && {
                    // Parser will consume </think> on next feed call —
                    // detect via sentinel before the buffer drains.
                    return false
                }()) {
                    sawCloseTag = true
                }
                // Feed the piece through the parser if available.
                if var p = parser {
                    let segs = p.feed(piece)
                    parser = p
                    for s in segs {
                        switch s {
                        case .reasoning(let r): reasoningOut += r
                        case .content(let c): contentOut += c
                        }
                    }
                } else {
                    contentOut += piece
                }
            case .info(let info):
                switch info.stopReason {
                case .stop: finishReason = "stop (EOS)"
                case .length: finishReason = "length (max_tokens)"
                case .cancelled: finishReason = "cancelled"
                default: finishReason = "other"
                }
            }
        }
    }
    // Snapshot state BEFORE flush — flush() resets insideReasoning.
    let parserStateBeforeFlush = parser?.isInsideReasoning ?? false
    if var p = parser {
        let trail = p.flush()
        parser = p
        for s in trail {
            switch s {
            case .reasoning(let r): reasoningOut += r
            case .content(let c): contentOut += c
            }
        }
    }
    let dt = CFAbsoluteTimeGetCurrent() - startTime

    // Heuristic: a non-empty content stream means parser observed
    // </think> in the wire output. (The token-level sentinel is
    // imprecise because </think> can be split across tokens; the
    // parser's state transition is authoritative.)
    if !contentOut.isEmpty {
        sawCloseTag = true
    }

    print("")
    print("=== Summary ===")
    print(String(format: "  decoded:    %d tokens in %.2fs (%.1f tok/s)",
        rawTokenCount, dt, Double(rawTokenCount) / dt))
    print(String(format: "  reasoning:  %d chars", reasoningOut.count))
    print(String(format: "  content:    %d chars", contentOut.count))
    print("  </think>:   \(sawCloseTag ? "YES — parser transitioned to content" : "NO — model never closed reasoning")")
    print("  finish:     \(finishReason)")
    print("  parser-pre-flush insideReasoning: \(parserStateBeforeFlush)  ← maps to GenerateCompletionInfo.unclosedReasoning")
    print("")
    print("=== Last 300 chars of reasoning ===")
    let rTail =
        reasoningOut.count > 300
        ? String(reasoningOut.suffix(300)) : reasoningOut
    print(rTail)
    print("")
    print("=== First 300 chars of content ===")
    let cHead =
        contentOut.count > 300
        ? String(contentOut.prefix(300)) : contentOut
    print(cHead.isEmpty ? "(empty)" : cHead)
    print("")
    print("=== BENCH_THINK_LOOP_PROBE: done ===")
}

// MARK: - Laguna Osaurus loop probe

func runLagunaLoopProbe(modelPath: String, maxNew: Int) async throws {
    let env = ProcessInfo.processInfo.environment
    let modelDir = URL(fileURLWithPath: modelPath)
    let maxTokens = max(maxNew, 512)
    let temperature = Float(env["LAGUNA_TEMP"] ?? "0") ?? 0
    let topP = Float(env["LAGUNA_TOP_P"] ?? "1.0") ?? 1.0
    let rep = Float(env["LAGUNA_REP"] ?? "1.15") ?? 1.15
    let repCtx = Int(env["LAGUNA_REP_CTX"] ?? "256") ?? 256

    print("\n=== BENCH_LAGUNA_LOOP: Osaurus Laguna no-thinking loop probe ===")
    print("Model: \(modelDir.path)")
    print(String(format:
        "Params: maxTokens=%d temp=%.2f topP=%.2f rep=%.2f repCtx=%d",
        maxTokens, Double(temperature), Double(topP), Double(rep), repCtx))

    let loadStart = CFAbsoluteTimeGetCurrent()
    let context = try await MLXLMCommon.loadModel(
        from: modelDir, using: #huggingFaceTokenizerLoader())
    print(String(format: "Load: %.2fs  Model: %@",
        CFAbsoluteTimeGetCurrent() - loadStart,
        String(describing: type(of: context.model))))
    print("Reasoning stamp: \(context.configuration.reasoningParserName ?? "nil")")
    print("EOS IDs: \(context.configuration.eosTokenIds.sorted())")

    nonisolated(unsafe) let ctx = context
    let engine = BatchEngine(context: ctx, maxBatchSize: 1)

    let fileTreePrompt = try lagunaPromptFixture(env: env)

    let params = GenerateParameters(
        maxTokens: maxTokens,
        temperature: temperature,
        topP: topP,
        repetitionPenalty: rep,
        repetitionContextSize: repCtx,
        prefillStepSize: 512)

    struct ProbeResult {
        var thinking: Bool
        var text = ""
        var reasoning = ""
        var chunks = 0
        var reasoningDeltas = 0
        var genTokens = 0
        var apiGenSec = 0.0
        var infoTokps = 0.0
        var wallTokps = 0.0
        var totalSec = 0.0
        var finish = "unknown"
        var loop = false
        var unclosedReasoning = false
        var leaks: [String] = []
    }

    func lagunaMarkerLeaks(in text: String) -> [String] {
        [
            "<|reserved_token_",
            "<|tool_call_begin|>",
            "<|tool_call_argument_begin|>",
            "<|tool_call_end|>",
            "<|tool_calls_section_begin|>",
            "<|tool_calls_section_end|>",
            "<|im_start|>",
            "<|im_end|>",
            "〈|EOS|〉",
            "<assistant>",
            "</assistant>",
            "<tool_call>",
            "</tool_call>",
        ].filter { text.contains($0) }
    }

    func runCase(thinking: Bool) async throws -> ProbeResult {
        var ui = UserInput(prompt: fileTreePrompt)
        ui.additionalContext = ["enable_thinking": thinking]
        let input = try await ctx.processor.prepare(input: ui)
        let promptIds = input.text.tokens.reshaped(-1)
            .asArray(Int32.self)
            .map { Int($0) }
        let promptTail = ctx.tokenizer.decode(
            tokenIds: Array(promptIds.suffix(80)),
            skipSpecialTokens: false)
        print("\n--- enable_thinking=\(thinking) ---")
        print("Prompt tokens: \(promptIds.count)")
        print("Prompt tail: \(promptTail.debugDescription)")

        nonisolated(unsafe) let send = input
        let t0 = CFAbsoluteTimeGetCurrent()
        let stream = await engine.generate(input: send, parameters: params)
        var result = ProbeResult(thinking: thinking)
        for await event in stream {
            switch event {
            case .chunk(let c):
                result.text += c
                result.chunks += 1
            case .reasoning(let r):
                result.reasoning += r
                result.reasoningDeltas += 1
            case .toolCall:
                break
            case .info(let info):
                result.genTokens = info.generationTokenCount
                result.apiGenSec = info.generateTime
                result.infoTokps = info.tokensPerSecond
                switch info.stopReason {
                case .stop: result.finish = "stop"
                case .length: result.finish = "length"
                case .cancelled: result.finish = "cancelled"
                default: result.finish = "other"
                }
                result.unclosedReasoning = info.unclosedReasoning
            }
        }
        result.totalSec = CFAbsoluteTimeGetCurrent() - t0
        result.wallTokps = result.totalSec > 0
            ? Double(result.genTokens) / result.totalSec
            : 0
        result.loop = lagunaLoopHeuristic(result.text + " " + result.reasoning)
        result.leaks = lagunaMarkerLeaks(in: result.text)
        return result
    }

    let off = try await runCase(thinking: false)
    let on = try await runCase(thinking: true)
    await engine.shutdown()

    func printResult(_ r: ProbeResult) {
        let visible = r.text.trimmingCharacters(in: .whitespacesAndNewlines)
        let previewSource = visible.isEmpty
            ? "[empty visible; reasoning chars=\(r.reasoning.count)]"
            : visible
        let preview = previewSource.count > 500
            ? String(previewSource.prefix(500)) + "..."
            : previewSource
        print(String(format:
            "\n[%@] finish=%@ total=%.2fs apiGen=%.2fs genTok=%d wallTokps=%.1f infoTokps=%.1f chunks=%d reasoningDeltas=%d text=%d reasoning=%d loop=%@ unclosedReasoning=%@ leaks=%@",
            r.thinking ? "thinking=ON" : "thinking=OFF",
            r.finish, r.totalSec, r.apiGenSec, r.genTokens,
            r.wallTokps, r.infoTokps, r.chunks, r.reasoningDeltas,
            r.text.count, r.reasoning.count, r.loop ? "YES" : "NO",
            r.unclosedReasoning ? "YES" : "NO",
            r.leaks.isEmpty ? "none" : r.leaks.joined(separator: ",")))
        print(preview.replacingOccurrences(of: "\n", with: "\\n"))
    }

    printResult(off)
    printResult(on)

    if off.loop {
        print("\nDiagnosis: thinking-off generated a repeated visible-content pattern.")
        print("Next A/B: rerun with LAGUNA_REP_CTX=256 and compare loop=false/tokps.")
    }
    if on.text.isEmpty && !on.reasoning.isEmpty {
        print("\nDiagnosis: thinking-on produced reasoning without visible content.")
        print("That means the model did not close </think> within the token budget.")
    }
    var failures: [String] = []
    if off.text.isEmpty {
        failures.append("thinking-off produced no visible content")
    }
    if off.finish != "stop" {
        failures.append("thinking-off finished with \(off.finish), expected stop")
    }
    if off.loop {
        failures.append("thinking-off loop heuristic fired")
    }
    if !off.leaks.isEmpty {
        failures.append("thinking-off marker leaks: \(off.leaks.joined(separator: ","))")
    }
    if on.text.isEmpty {
        failures.append("thinking-on produced no visible content at \(maxTokens) tokens")
    }
    if on.finish != "stop" {
        failures.append("thinking-on finished with \(on.finish), expected stop")
    }
    if on.loop {
        failures.append("thinking-on loop heuristic fired")
    }
    if on.unclosedReasoning {
        failures.append("thinking-on ended with unclosed reasoning")
    }
    if !on.leaks.isEmpty {
        failures.append("thinking-on marker leaks: \(on.leaks.joined(separator: ","))")
    }
    if !failures.isEmpty {
        throw NSError(
            domain: "BENCH_LAGUNA_LOOP",
            code: 1,
            userInfo: [NSLocalizedDescriptionKey: failures.joined(separator: "; ")])
    }
    print("\n=== BENCH_LAGUNA_LOOP: done ===")
}

private func lagunaLoopHeuristic(_ text: String) -> Bool {
    let normalized = text
        .lowercased()
        .replacingOccurrences(of: #"\s+"#, with: " ", options: .regularExpression)
        .trimmingCharacters(in: .whitespacesAndNewlines)
    var previousScalar: UnicodeScalar?
    var repeatedScalarCount = 0
    for scalar in normalized.unicodeScalars {
        if CharacterSet.whitespacesAndNewlines.contains(scalar) {
            previousScalar = nil
            repeatedScalarCount = 0
            continue
        }
        if scalar == previousScalar {
            repeatedScalarCount += 1
            if repeatedScalarCount >= 64 { return true }
        } else {
            previousScalar = scalar
            repeatedScalarCount = 1
        }
    }
    let words = normalized.split(separator: " ").map(String.init)
    guard words.count >= 18 else { return false }
    let maxWidth = min(32, words.count / 3)
    if maxWidth >= 6 {
        for width in 6...maxWidth {
            let tail = Array(words.suffix(width * 3))
            let a = tail[0..<width]
            let b = tail[width..<(width * 2)]
            let c = tail[(width * 2)..<(width * 3)]
            if Array(a) == Array(b), Array(b) == Array(c) {
                return true
            }
        }
    }
    let lines = normalized
        .split(separator: ".")
        .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
        .filter { $0.count >= 32 }
    var counts: [String: Int] = [:]
    for line in lines {
        counts[line, default: 0] += 1
        if counts[line, default: 0] >= 3 {
            return true
        }
    }
    return false
}

// MARK: - No-hidden-guard sampling probe

/// Behavioral proof for the no-hidden-guards policy. This is a diagnostic
/// harness only: it sends explicit request parameters, prints the effective
/// values, and fails on loops, visible reasoning marker leaks, or empty visible
/// content. It does not alter production defaults or family behavior.
func runNoGuardSamplingProbe(modelPath: String, maxNew: Int) async throws {
    let modelDir = URL(fileURLWithPath: modelPath)
    let modelName = modelDir.lastPathComponent
    print("\n=== BENCH_NO_GUARD_SAMPLING — \(modelName) ===")

    let loadStart = CFAbsoluteTimeGetCurrent()
    let context = try await MLXLMCommon.loadModel(
        from: modelDir, using: #huggingFaceTokenizerLoader())
    print(String(format: "Load: %.2fs  Model: %@",
        CFAbsoluteTimeGetCurrent() - loadStart,
        String(describing: type(of: context.model))))
    print("Tool format: \(context.configuration.toolCallFormat.map{"\($0)"} ?? "json")")
    print("Reasoning stamp: \(context.configuration.reasoningParserName ?? "nil")")
    print("Tokenizer BOS: \(context.tokenizer.bosToken ?? "nil")")

    struct Case {
        var label: String
        var prompt: String
        var maxTokens: Int
        var temperature: Float
        var topP: Float
        var topK: Int
        var minP: Float
        var repetitionPenalty: Float?
        var enableThinking: Bool?
    }

    var cases: [Case] = [
        Case(
            label: "A_hi_greedy_no_rep",
            prompt: "say hi",
            maxTokens: max(64, min(maxNew, 96)),
            temperature: 0,
            topP: 1,
            topK: 0,
            minP: 0,
            repetitionPenalty: nil,
            enableThinking: nil),
        Case(
            label: "B_star_story_temp06_rep1_think_on",
            prompt: "tell me a 2-sentence story about a star",
            maxTokens: max(maxNew, 256),
            temperature: 0.6,
            topP: 1,
            topK: 0,
            minP: 0,
            repetitionPenalty: 1.0,
            enableThinking: true),
    ]
    if modelName.lowercased().contains("ling")
        || modelName.lowercased().contains("bailing")
    {
        cases.append(Case(
            label: "C_ling_russian_threejs_temp07",
            prompt:
                "Привет. Напиши краткий план одной HTML/Three.js игры: охотник бежит по лесу, стреляет из дробовика, а кабаны и лоси появляются как враги. Ответь по-русски структурировано в 5 пунктах; каждый пункт должен быть полным коротким предложением, без повторения одного слова или символа.",
            maxTokens: max(maxNew, 420),
            temperature: 0.7,
            topP: 1,
            topK: 0,
            minP: 0,
            repetitionPenalty: nil,
            enableThinking: false))
    }

    nonisolated(unsafe) let ctx = context
    let engine = BatchEngine(context: ctx, maxBatchSize: 1)

    struct Result {
        var text = ""
        var reasoning = ""
        var toolCalls = 0
        var genTokens = 0
        var genSec = 0.0
        var totalSec = 0.0
        var ttftSec = 0.0
        var stop = "unknown"
        var unclosedReasoning = false
    }

    func stopLabel(_ reason: GenerateStopReason) -> String {
        switch reason {
        case .stop: return "stop"
        case .length: return "length"
        case .cancelled: return "cancelled"
        default: return "other"
        }
    }

    func preview(_ text: String, limit: Int) -> String {
        let collapsed = text.replacingOccurrences(of: "\n", with: "\\n")
        return collapsed.count > limit
            ? String(collapsed.prefix(limit)) + "..."
            : collapsed
    }

    func markerLeaks(in text: String) -> [String] {
        [
            "<think>", "</think>",
            "[THINK]", "[/THINK]",
            "<|channel>", "<channel|>",
            "<|channel|>analysis", "<|channel|>final",
            "<|message|>",
        ].filter { text.contains($0) }
    }

    func occurrenceCount(_ needle: String, in haystack: String) -> Int {
        guard !needle.isEmpty else { return 0 }
        var count = 0
        var search = haystack[...]
        while let range = search.range(of: needle) {
            count += 1
            search = search[range.upperBound...]
        }
        return count
    }

    func run(_ c: Case) async throws -> Result {
        var input = UserInput(prompt: c.prompt)
        if let thinking = c.enableThinking {
            input.additionalContext = ["enable_thinking": thinking]
        }
        let prepared = try await ctx.processor.prepare(input: input)
        let promptTokens = prepared.text.tokens.reshaped(-1).asArray(Int32.self)
        let tail = ctx.tokenizer.decode(
            tokenIds: Array(promptTokens.suffix(80)).map(Int.init),
            skipSpecialTokens: false)
        print("\n--- \(c.label) ---")
        print("Prompt tokens: \(promptTokens.count)")
        print("Prompt tail: \(tail.debugDescription)")
        let params = GenerateParameters(
            maxTokens: c.maxTokens,
            temperature: c.temperature,
            topP: c.topP,
            topK: c.topK,
            minP: c.minP,
            repetitionPenalty: c.repetitionPenalty,
            prefillStepSize: 512)
        print(String(format:
            "Sampling: maxTokens=%d temp=%.3f topP=%.3f topK=%d minP=%.3f rep=%@ enable_thinking=%@",
            params.maxTokens ?? -1,
            Double(params.temperature),
            Double(params.topP),
            params.topK,
            Double(params.minP),
            params.repetitionPenalty.map { String(format: "%.3f", Double($0)) } ?? "nil",
            c.enableThinking.map(String.init) ?? "omitted"))

        nonisolated(unsafe) let send = prepared
        var result = Result()
        let start = CFAbsoluteTimeGetCurrent()
        let stream = await engine.generate(input: send, parameters: params)
        for await event in stream {
            switch event {
            case .chunk(let text):
                if result.ttftSec == 0 {
                    result.ttftSec = CFAbsoluteTimeGetCurrent() - start
                }
                result.text += text
            case .reasoning(let text):
                if result.ttftSec == 0 {
                    result.ttftSec = CFAbsoluteTimeGetCurrent() - start
                }
                result.reasoning += text
            case .toolCall:
                result.toolCalls += 1
            case .info(let info):
                result.genTokens = info.generationTokenCount
                result.genSec = info.generateTime
                result.stop = stopLabel(info.stopReason)
                result.unclosedReasoning = info.unclosedReasoning
            }
        }
        result.totalSec = CFAbsoluteTimeGetCurrent() - start
        return result
    }

    var failures: [String] = []
    do {
        for c in cases {
            let r = try await run(c)
            let visible = r.text.trimmingCharacters(in: .whitespacesAndNewlines)
            let display = visible.isEmpty
                ? "[empty visible; reasoning chars=\(r.reasoning.count)]"
                : visible
            let combined = r.text + "\n" + r.reasoning
            let loop = lagunaLoopHeuristic(combined)
            let leaks = markerLeaks(in: r.text)
            let bosCount = context.tokenizer.bosToken.map {
                occurrenceCount($0, in: combined)
            } ?? 0
            let tokps = r.genSec > 0 ? Double(r.genTokens) / r.genSec : 0
            print(String(format:
                "Result: stop=%@ unclosedReasoning=%@ genTokens=%d genSec=%.3f tokps=%.1f ttft_ms=%.0f textChars=%d reasoningChars=%d tools=%d loop=%@ bosRepeats=%d leaks=%@",
                r.stop,
                r.unclosedReasoning ? "YES" : "NO",
                r.genTokens,
                r.genSec,
                tokps,
                r.ttftSec * 1000,
                r.text.count,
                r.reasoning.count,
                r.toolCalls,
                loop ? "YES" : "NO",
                bosCount,
                leaks.isEmpty ? "none" : leaks.joined(separator: ",")))
            print("First 400 visible: \"\(preview(display, limit: 400))\"")
            print("Last 200 visible: \"\(preview(String(display.suffix(200)), limit: 240))\"")

            if visible.isEmpty {
                failures.append("\(c.label): no visible content")
            }
            if loop {
                failures.append("\(c.label): repeated-output loop heuristic fired")
            }
            if bosCount >= 3 {
                failures.append("\(c.label): BOS token repeated \(bosCount)x")
            }
            if !leaks.isEmpty {
                failures.append("\(c.label): visible reasoning markers \(leaks.joined(separator: ","))")
            }
            if c.enableThinking == true && r.unclosedReasoning {
                failures.append("\(c.label): reasoning did not close before stop")
            }
            if c.label == "A_hi_greedy_no_rep" {
                let lower = visible.lowercased()
                if !(lower.contains("hi") || lower.contains("hello")) {
                    failures.append("\(c.label): visible output did not greet")
                }
            }
            if c.label == "B_star_story_temp06_rep1_think_on" {
                let lower = visible.lowercased()
                let mentionsStellarSubject =
                    lower.contains("star")
                    || lower.contains("sun")
                    || lower.contains("stellar")
                if !mentionsStellarSubject {
                    failures.append("\(c.label): visible story did not mention a stellar subject")
                }
            }
            if c.label == "C_ling_russian_threejs_temp07" {
                let lower = visible.lowercased()
                if !(lower.contains("three.js") || lower.contains("html")
                    || lower.contains("игр"))
                {
                    failures.append("\(c.label): output did not stay on the Three.js game task")
                }
            }
        }
    } catch {
        await engine.shutdown()
        throw error
    }

    await engine.shutdown()
    if !failures.isEmpty {
        throw NSError(domain: "BENCH_NO_GUARD_SAMPLING", code: 1,
            userInfo: [NSLocalizedDescriptionKey:
                failures.joined(separator: "; ")])
    }
    print("\n=== BENCH_NO_GUARD_SAMPLING: passed ===")
}

private func lagunaPromptFixture(env: [String: String]) throws -> String {
    if let promptFile = env["LAGUNA_PROMPT_FILE"], !promptFile.isEmpty {
        return try String(contentsOfFile: promptFile, encoding: .utf8)
    }

    if let treePath = env["LAGUNA_TREE_PATH"], !treePath.isEmpty {
        let root = URL(fileURLWithPath: treePath).standardizedFileURL
        let limit = Int(env["LAGUNA_TREE_LIMIT"] ?? "240") ?? 240
        var entries: [String] = []
        if let enumerator = FileManager.default.enumerator(
            at: root,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles, .skipsPackageDescendants])
        {
            for case let url as URL in enumerator {
                let path = url.standardizedFileURL.path
                guard path.hasPrefix(root.path) else { continue }
                var rel = String(path.dropFirst(root.path.count))
                if rel.hasPrefix("/") { rel.removeFirst() }
                if rel.isEmpty { continue }
                if rel.hasPrefix(".build/") || rel.hasPrefix("DerivedData/") {
                    continue
                }
                let values = try? url.resourceValues(forKeys: [.isDirectoryKey])
                let isDir = values?.isDirectory ?? false
                entries.append("- \(rel)\(isDir ? "/" : "")")
                if entries.count >= max(1, limit) { break }
            }
        }
        entries.sort()
        return """
        For each file in this folder, summarize in one line what it does.

        file_tree path: \(root.path)
        \(entries.joined(separator: "\n"))
        """
    }

    return """
    For each file in this folder, summarize in one line what it does.

    file_tree path:
    .
    - eval_test_images/ - Core testing infrastructure
    - eval_tasks/ - Task management and validation
    - eval_outputs/ - Output management
    - validators/ - Validation helpers and scoring
    - runners/ - CLI and benchmark runners
    - reports/ - Generated reports
    """
}

// MARK: - DSV4 FIM vs Chat coherence probe

// MARK: - DSV4 production coherence gate

/// Generic reasoning multi-turn gate for Harmony/Qwen/think-XML-style models.
/// It is intentionally a live BatchEngine harness rather than a unit parser
/// test: the goal is to prove a growing chat transcript, prior assistant
/// reasoning_content, bundle generation defaults, and stream routing together.
func runReasoningTurnMatrix(modelPath: String, maxNew: Int) async throws {
    let modelDir = URL(fileURLWithPath: modelPath)
    let modelName = modelDir.lastPathComponent
    let env = ProcessInfo.processInfo.environment
    let budget = max(maxNew, 256)
    let effortBudget = max(
        Int(env["BENCH_REASONING_EFFORT_MAX_TOKENS"] ?? "1024") ?? 1024,
        budget)
    let cacheRoot = env["BENCH_REASONING_CACHE_DIR"] ??
        "/tmp/vmlx-reasoning-turn-matrix/\(modelName)"
    try? FileManager.default.createDirectory(
        atPath: cacheRoot, withIntermediateDirectories: true)

    print("\n=== BENCH_REASONING_TURN_MATRIX — \(modelName) ===")
    print("Cache dir: \(cacheRoot)")
    let rss0 = currentRSSMiB()
    let loadStart = CFAbsoluteTimeGetCurrent()
    let context = try await MLXLMCommon.loadModel(
        from: modelDir, using: #huggingFaceTokenizerLoader())
    let loadSec = CFAbsoluteTimeGetCurrent() - loadStart
    let rss1 = currentRSSMiB()
    print(String(format: "Load: %.2fs  Model: %@  RSS +%.0f MiB",
        loadSec, String(describing: type(of: context.model)), rss1 - rss0))
    print("Reasoning stamp: \(context.configuration.reasoningParserName ?? "nil")")
    print("Tool format: \(context.configuration.toolCallFormat.map{"\($0)"} ?? "json")")

    let ctx = context
    let cfg = CacheCoordinatorConfig(
        usePagedCache: true, enableDiskCache: true,
        pagedBlockSize: 64, maxCacheBlocks: 512,
        diskCacheMaxGB: 4.0,
        diskCacheDir: URL(fileURLWithPath: cacheRoot),
        ssmMaxEntries: 32, modelKey: modelName)
    let coordinator = CacheCoordinator(config: cfg)
    let engine = BatchEngine(
        context: ctx, maxBatchSize: 1, cacheCoordinator: coordinator)

    let randomSeed = env["BENCH_REASONING_SEED"].flatMap(UInt64.init)
    let fallback = GenerateParameters(
        maxTokens: budget, randomSeed: randomSeed, prefillStepSize: 512)
    var params = GenerateParameters(
        generationConfig: context.configuration.generationDefaults,
        fallback: fallback)
    params.maxTokens = budget
    params.randomSeed = randomSeed
    print(String(format:
        "Sampling: maxTokens=%d temp=%.3f topP=%.3f topK=%d minP=%.3f rep=%@ seed=%@",
        params.maxTokens ?? -1,
        Double(params.temperature),
        Double(params.topP),
        params.topK,
        Double(params.minP),
        params.repetitionPenalty.map { String(format: "%.3f", Double($0)) } ?? "nil",
        params.randomSeed.map(String.init) ?? "nil"))

    func additional(thinking: Bool, effort: String? = nil) -> [String: any Sendable] {
        var context: [String: any Sendable] = ["enable_thinking": thinking]
        if let effort {
            context["reasoning_effort"] = effort
        }
        return context
    }

    func promptTail(thinking: Bool, effort: String? = nil) async throws -> String {
        let ui = UserInput(
            chat: [.user("Probe reasoning template routing.")],
            additionalContext: additional(thinking: thinking, effort: effort))
        let input = try await ctx.processor.prepare(input: ui)
        let ids = input.text.tokens.reshaped(-1).asArray(Int32.self).map { Int($0) }
        return ctx.tokenizer.decode(
            tokenIds: Array(ids.suffix(128)), skipSpecialTokens: false)
    }

    func tailHasReasoningRail(_ text: String) -> Bool {
        text.contains("<think>")
            || text.contains("<|think|>")
            || text.contains("[THINK]")
            || text.contains("<|channel>")
            || text.contains("<|channel|>analysis")
            || text.contains("<channel|>")
    }

    let thinkingOnTail = try await promptTail(thinking: true)
    let thinkingOffTail = try await promptTail(thinking: false)
    let reasoningPromptToggleActive =
        thinkingOnTail != thinkingOffTail && tailHasReasoningRail(thinkingOnTail)
    print(
        "Reasoning prompt toggle: \(reasoningPromptToggleActive ? "active" : "not-template-active")"
            + " onTail=\(thinkingOnTail.suffix(96).debugDescription)"
            + " offTail=\(thinkingOffTail.suffix(96).debugDescription)")

    final class ReasoningState {
        var thinkingOnRoutedCount = 0
    }
    let reasoningState = ReasoningState()

    struct MatrixAnswer {
        var label: String
        var text: String
        var reasoning: String
        var promptTokens: Int
        var completionTokens: Int
        var ttftMs: Int
        var wall: Double
        var tokps: Double
        var stop: String
        var unclosedReasoning: Bool
    }

    func stopString(_ info: GenerateCompletionInfo?) -> String {
        guard let info else { return "nil" }
        switch info.stopReason {
        case .stop: return "stop"
        case .length: return "length"
        case .cancelled: return "cancelled"
        }
    }

    func hasRawControlMarker(_ text: String) -> String? {
        [
            "<think>", "</think>",
            "[THINK]", "[/THINK]",
            "<|channel>", "<channel|>",
            "<|channel|>analysis", "<|channel|>final",
            "<|message|>", "<|end|>",
        ].first { text.contains($0) }
    }

    func assistantMessage(from answer: MatrixAnswer) -> Chat.Message {
        Chat.Message(
            role: .assistant,
            content: answer.text.trimmingCharacters(in: .whitespacesAndNewlines),
            reasoningContent: answer.reasoning.isEmpty ? nil : answer.reasoning)
    }

    func ask(
        label: String,
        chat: [Chat.Message],
        thinking: Bool,
        effort: String? = nil,
        maxTokens: Int? = nil
    ) async throws -> MatrixAnswer {
        let ui = UserInput(
            chat: chat,
            additionalContext: additional(thinking: thinking, effort: effort))
        let prepared = try await ctx.processor.prepare(input: ui)
        let promptTokens = prepared.text.tokens.size
        nonisolated(unsafe) let sendable = prepared
        let t0 = CFAbsoluteTimeGetCurrent()
        var turnParams = params
        if let maxTokens {
            turnParams.maxTokens = maxTokens
        }
        let stream = await engine.generate(input: sendable, parameters: turnParams)
        var text = ""
        var reasoning = ""
        var info: GenerateCompletionInfo?
        var ttft: Double?
        var chunks = 0
        for await event in stream {
            switch event {
            case .chunk(let c):
                if ttft == nil { ttft = CFAbsoluteTimeGetCurrent() - t0 }
                text += c
                chunks += 1
            case .reasoning(let r):
                if ttft == nil { ttft = CFAbsoluteTimeGetCurrent() - t0 }
                reasoning += r
                chunks += 1
            case .info(let i):
                info = i
            default:
                break
            }
        }
        let wall = CFAbsoluteTimeGetCurrent() - t0
        let completionTokens = info?.generationTokenCount ?? chunks
        let genSec = info?.generateTime ?? wall
        let tokps = genSec > 0 ? Double(completionTokens) / genSec : 0
        let answer = MatrixAnswer(
            label: label,
            text: text,
            reasoning: reasoning,
            promptTokens: promptTokens,
            completionTokens: completionTokens,
            ttftMs: Int((ttft ?? 0) * 1000),
            wall: wall,
            tokps: tokps,
            stop: stopString(info),
            unclosedReasoning: info?.unclosedReasoning ?? false)
        let visible = text.trimmingCharacters(in: .whitespacesAndNewlines)
        let previewSource = visible.isEmpty
            ? "[empty visible; reasoning chars=\(reasoning.count)]"
            : visible
        let preview = String(previewSource.prefix(180))
            .replacingOccurrences(of: "\n", with: "\\n")
        print(String(format:
            "TURN label=%@ thinking=%@ effort=%@ prompt=%d completion=%d ttft=%dms wall=%.2fs tokps=%.2f stop=%@ unclosedReasoning=%@ textChars=%d reasoningChars=%d sample=\"%@\"",
            label, thinking.description, effort ?? "nil",
            promptTokens, completionTokens, answer.ttftMs, wall, tokps,
            answer.stop, answer.unclosedReasoning.description,
            text.count, reasoning.count, preview))
        return answer
    }

    func fail(_ code: Int, _ message: String) throws -> Never {
        throw NSError(
            domain: "BENCH_REASONING_TURN_MATRIX",
            code: code,
            userInfo: [NSLocalizedDescriptionKey: message])
    }

    func requireCommon(_ answer: MatrixAnswer) throws {
        if answer.stop == "length" {
            try fail(20, "\(answer.label): stopped by length")
        }
        if answer.unclosedReasoning {
            try fail(21, "\(answer.label): ended inside reasoning")
        }
        if let marker = hasRawControlMarker(answer.text) {
            try fail(22, "\(answer.label): visible chunk leaked \(marker)")
        }
        if answer.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            try fail(23, "\(answer.label): empty visible output")
        }
    }

    func requireThinkingOff(_ answer: MatrixAnswer) throws {
        try requireCommon(answer)
        if !answer.reasoning.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            try fail(24, "\(answer.label): reasoning emitted while thinking off")
        }
    }

    func requireThinkingOn(_ answer: MatrixAnswer) throws {
        try requireCommon(answer)
        if !answer.reasoning.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            reasoningState.thinkingOnRoutedCount += 1
        }
    }

    func requireContains(_ answer: MatrixAnswer, _ terms: [String]) throws {
        let lower = answer.text.lowercased()
        for term in terms where !lower.contains(term.lowercased()) {
            try fail(26, "\(answer.label): missing visible term \(term)")
        }
    }

    var transcript: [Chat.Message] = [
        .system("You are a concise assistant. Follow the user's requested format."),
        .user("Remember this exact phrase: copper-lantern. Reply with only: saved.")
    ]
    let turn1 = try await ask(
        label: "turn1-off-save", chat: transcript, thinking: false)
    try requireThinkingOff(turn1)
    try requireContains(turn1, ["saved"])
    transcript.append(assistantMessage(from: turn1))

    transcript.append(.user("What exact phrase did I ask you to remember? Answer with only that phrase."))
    let turn2 = try await ask(
        label: "turn2-on-recall", chat: transcript, thinking: true)
    try requireThinkingOn(turn2)
    try requireContains(turn2, ["copper", "lantern"])
    transcript.append(assistantMessage(from: turn2))

    transcript.append(.user("Now answer only the color of a clear daytime sky."))
    let turn3 = try await ask(
        label: "turn3-off-after-reasoning", chat: transcript, thinking: false)
    try requireThinkingOff(turn3)
    transcript.append(assistantMessage(from: turn3))

    transcript.append(.user("Compute 12 + 8. Reply with only the number."))
    let turn4 = try await ask(
        label: "turn4-on-math-max", chat: transcript,
        thinking: true, effort: "max")
    try requireThinkingOn(turn4)
    try requireContains(turn4, ["20"])

    let efforts = (env["BENCH_REASONING_EFFORTS"] ?? "low,medium,high,max")
        .split(separator: ",")
        .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
        .filter { !$0.isEmpty }
    for effort in efforts {
        let answer = try await ask(
            label: "effort-\(effort)",
            chat: [.user("Say one short sentence about why cache boundaries matter.")],
            thinking: true,
            effort: effort,
            maxTokens: effortBudget)
        try requireThinkingOn(answer)
    }

    if reasoningPromptToggleActive && reasoningState.thinkingOnRoutedCount == 0 {
        try fail(25, "thinking-on prompt rail was active but no ON row routed reasoning")
    }

    let snapshot = coordinator.snapshotStats()
    let paged = snapshot.pagedStats.map {
        "hits=\($0.cacheHits),misses=\($0.cacheMisses),allocated=\($0.allocatedBlocks),free=\($0.freeBlocks),evictions=\($0.evictions)"
    } ?? "disabled"
    let disk = snapshot.diskStats.map {
        "hits=\($0.hits),misses=\($0.misses),stores=\($0.stores),maxBytes=\($0.maxSizeBytes)"
    } ?? "disabled"
    let ssm = snapshot.ssmStats
    print(
        "REASONING_CACHE_STATS hybrid=\(snapshot.isHybrid) pagedIncompatible=\(snapshot.isPagedIncompatible) paged{\(paged)} disk{\(disk)} ssm{hits=\(ssm.hits),misses=\(ssm.misses),reDerives=\(ssm.reDerives)}"
    )
    print("=== BENCH_REASONING_TURN_MATRIX: PASS ===")
}

/// Production DSV4 chat coherence gate. Unlike the generic
/// `BENCH_BATCH_CHAT` row, this uses `UserInput(chat:)` for every turn so the
/// DSV4 Jinja template, `enable_thinking`, and `reasoning_effort` kwargs are
/// exercised exactly like osaurus will call them.
func runDSV4CoherenceGate(modelPath: String, maxNew: Int) async throws {
    let modelDir = URL(fileURLWithPath: modelPath)
    let env = ProcessInfo.processInfo.environment
    let rowFilter = (env["BENCH_DSV4_ROW"] ?? "all").lowercased()
    let longRepeat = Int(env["BENCH_DSV4_LONG_REPEAT"] ?? "220") ?? 220
    let longMaxNew = Int(env["BENCH_DSV4_LONG_MAX_TOKENS"] ?? "\(max(96, maxNew))")
        ?? max(96, maxNew)
    let chatMaxNew = Int(env["BENCH_DSV4_CHAT_MAX_TOKENS"] ?? "\(max(160, maxNew))")
        ?? max(160, maxNew)
    // DSV4 max-effort prompts need enough budget to close the think_xml block
    // at bundle-derived sampling defaults; too small a cap can length-stop
    // before visible text and create a false coherence failure.
    let reasoningMaxDefault = max(1024, maxNew)
    let reasoningMaxNew = Int(env["BENCH_DSV4_REASONING_MAX_TOKENS"] ?? "\(reasoningMaxDefault)")
        ?? reasoningMaxDefault
    let dsv4Temperature = env["BENCH_DSV4_TEMP"].flatMap(Float.init)
    let dsv4TopP = env["BENCH_DSV4_TOP_P"].flatMap(Float.init)
    let dsv4TopK = env["BENCH_DSV4_TOP_K"].flatMap(Int.init)
    let dsv4MinP = env["BENCH_DSV4_MIN_P"].flatMap(Float.init)
    let dsv4Seed = env["BENCH_DSV4_SEED"].flatMap(UInt64.init)
    let dsv4RepetitionPenaltyOverride =
        env["BENCH_DSV4_REPETITION_PENALTY"].flatMap(Float.init)
    let dsv4MaxRepetitionPenaltyOverride =
        env["BENCH_DSV4_MAX_REPETITION_PENALTY"].flatMap(Float.init)
    let dsv4RepetitionContext =
        Int(env["BENCH_DSV4_REPETITION_CONTEXT"] ?? "64") ?? 64
    let useCacheCoordinator =
        (env["BENCH_DSV4_CACHE"] ?? "on").lowercased() != "off"
    let launchKey = env["BENCH_DSV4_KEY"] ?? "sapphire-42"
    let systemText = env["BENCH_DSV4_SYSTEM"] ??
        "You are a concise assistant. Answer directly."
    let includeSystem =
        (env["BENCH_DSV4_NO_SYSTEM"] ?? "0") != "1"

    do {
    print("\n=== BENCH_DSV4_COHERENCE: production chat coherence gate ===")
    print("Loading with real HuggingFace tokenizer...")
    let loadStart = CFAbsoluteTimeGetCurrent()
    let context = try await MLXLMCommon.loadModel(
        from: modelDir, using: #huggingFaceTokenizerLoader())
    print(String(format: "Load: %.2fs", CFAbsoluteTimeGetCurrent() - loadStart))
    print("Model: \(type(of: context.model))")
    print("Reasoning stamp: \(context.configuration.reasoningParserName ?? "nil")")
    if let defaults = context.configuration.generationDefaults {
        print(String(format:
            "Generation defaults: temp=%@ topP=%@ topK=%@ minP=%@ rep=%@ doSample=%@",
            defaults.temperature.map { String(format: "%.3f", Double($0)) } ?? "nil",
            defaults.topP.map { String(format: "%.3f", Double($0)) } ?? "nil",
            defaults.topK.map(String.init) ?? "nil",
            defaults.minP.map { String(format: "%.3f", Double($0)) } ?? "nil",
            defaults.repetitionPenalty.map { String(format: "%.3f", Double($0)) } ?? "nil",
            defaults.doSample.map(String.init) ?? "nil"))
    } else {
        print("Generation defaults: nil")
    }

    nonisolated(unsafe) let ctx = context
    let engine: BatchEngine
    let coordinator: CacheCoordinator?
    if useCacheCoordinator {
        var cfg = CacheCoordinatorConfig()
        cfg.usePagedCache = true
        cfg.enableDiskCache = false
        cfg.pagedBlockSize = 256
        cfg.maxCacheBlocks = 1024
        cfg.modelKey = modelDir.lastPathComponent
        let coord = CacheCoordinator(config: cfg)
        coordinator = coord
        engine = BatchEngine(
            context: ctx, maxBatchSize: 1, cacheCoordinator: coord)
        print("DSV4 cache coordinator: on")
    } else {
        coordinator = nil
        engine = BatchEngine(context: ctx, maxBatchSize: 1)
        print("DSV4 cache coordinator: off")
    }

    func shouldRun(_ row: String) -> Bool {
        rowFilter == "all" || rowFilter == row
    }

    struct DSV4Answer {
        let label: String
        let text: String
        let reasoning: String
        let promptTokens: Int
        let chunks: Int
        let info: GenerateCompletionInfo?

        var visible: String { text }
    }

    func hasRawReasoningMarker(_ s: String) -> Bool {
        s.contains("<think>") || s.contains("</think>")
    }

    func fail(_ code: Int, _ message: String) throws -> Never {
        throw NSError(
            domain: "BENCH_DSV4_COHERENCE",
            code: code,
            userInfo: [NSLocalizedDescriptionKey: message])
    }

    func printBlock(_ label: String, _ text: String) {
        print("--- \(label)_BEGIN ---")
        print(text)
        print("--- \(label)_END ---")
    }

    func ask(
        label: String,
        chat: [Chat.Message],
        enableThinking: Bool,
        reasoningEffort: String? = nil,
        maxTokens: Int
    ) async throws -> DSV4Answer {
        var ui = UserInput(chat: chat)
        var additional: [String: any Sendable] = [
            "enable_thinking": enableThinking
        ]
        if let reasoningEffort {
            additional["reasoning_effort"] = reasoningEffort
        }
        ui.additionalContext = additional

        let lm = try await ctx.processor.prepare(input: ui)
        let promptTokens = lm.text.tokens.size
        print("  \(label): prepared prompt=\(promptTokens) maxNew=\(maxTokens)")
        if (env["BENCH_DSV4_DEBUG_PROMPT"] ?? "0") == "1" {
            let ids = lm.text.tokens.reshaped(-1).asArray(Int32.self).map { Int($0) }
            let tail = ctx.tokenizer.decode(
                tokenIds: Array(ids.suffix(256)), skipSpecialTokens: false)
            print("  \(label): prompt tail = \(tail.debugDescription)")
        }
        nonisolated(unsafe) let sendable = lm

        let fallbackParams = GenerateParameters(
            maxTokens: maxTokens,
            randomSeed: dsv4Seed,
            prefillStepSize: 512)
        var params = GenerateParameters(
            generationConfig: ctx.configuration.generationDefaults,
            fallback: fallbackParams)
        // The row owns decode budget and any explicit diagnostic overrides.
        // Sampling defaults otherwise stay bundle-derived.
        params.maxTokens = maxTokens
        params.randomSeed = dsv4Seed
        params.prefillStepSize = 512
        if let dsv4Temperature {
            params.temperature = dsv4Temperature
        }
        if let dsv4TopP {
            params.topP = dsv4TopP
        }
        if let dsv4TopK {
            params.topK = dsv4TopK
        }
        if let dsv4MinP {
            params.minP = dsv4MinP
        }
        let requestedPenaltyOverride =
            reasoningEffort == "max"
                ? dsv4MaxRepetitionPenaltyOverride
                : dsv4RepetitionPenaltyOverride
        if let requestedPenaltyOverride {
            params.repetitionPenalty =
                requestedPenaltyOverride == 1.0 ? nil : requestedPenaltyOverride
        }
        params.repetitionContextSize = dsv4RepetitionContext
        params.enableCompiledBatchDecode = false

        let t0 = CFAbsoluteTimeGetCurrent()
        let stream = await engine.generate(input: sendable, parameters: params)
        var text = ""
        var reasoning = ""
        var info: GenerateCompletionInfo?
        var chunks = 0
        for await event in stream {
            switch event {
            case .chunk(let c):
                text += c
                chunks += 1
            case .reasoning(let r):
                reasoning += r
                chunks += 1
            case .info(let i):
                info = i
            default:
                break
            }
        }
        let wall = CFAbsoluteTimeGetCurrent() - t0
        let visible = text
        let sampleSource = visible.isEmpty ? "[empty visible; reasoning chars=\(reasoning.count)]" : visible
        let sample = String(sampleSource.prefix(180)).replacingOccurrences(of: "\n", with: " ")
        let infoSuffix: String
        let tokps: Double
        if let info {
            tokps = wall > 0 ? Double(info.generationTokenCount) / wall : 0
            infoSuffix = " stop=\(info.stopReason) gen=\(info.generationTokenCount) unclosedReasoning=\(info.unclosedReasoning)"
        } else {
            tokps = 0
            infoSuffix = " stop=nil gen=nil unclosedReasoning=nil"
        }
        print(String(format:
            "  %@: prompt=%d chunks=%d wall=%.2fs tokps=%.2f text=%d reasoning=%d%@ sample=\"%@\"",
            label, promptTokens, chunks, wall, tokps, text.count, reasoning.count, infoSuffix, sample))
        if hasRawReasoningMarker(text) {
            try fail(10, "\(label): raw <think> marker leaked into .chunk")
        }
        if visible.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            try fail(11, "\(label): empty visible stream")
        }
        return DSV4Answer(
            label: label, text: text, reasoning: reasoning,
            promptTokens: promptTokens, chunks: chunks, info: info)
    }

    func containsAll(_ haystack: String, _ needles: [String]) -> Bool {
        let lower = haystack.lowercased()
        return needles.allSatisfy { lower.contains($0.lowercased()) }
    }

    func stoppedByLength(_ answer: DSV4Answer) -> Bool {
        guard let info = answer.info else { return false }
        if case .length = info.stopReason { return true }
        return false
    }

    func occurrenceCount(_ needle: String, in haystack: String) -> Int {
        guard !needle.isEmpty else { return 0 }
        var count = 0
        var remainder = haystack[...]
        while let range = remainder.range(of: needle) {
            count += 1
            remainder = remainder[range.upperBound...]
        }
        return count
    }

    // 1. Production multi-turn chat with thinking disabled.
    if shouldRun("chat") {
        print("\n[DSV4 chat multi-turn, thinking off]")
        var chat: [Chat.Message] = includeSystem
            ? [.system(systemText)]
            : []
        chat.append(.user("Remember this exact launch key: \(launchKey). Reply with only: saved."))
        let t1 = try await ask(
            label: "turn1-save", chat: chat, enableThinking: false,
            maxTokens: max(48, min(maxNew, 96)))
        printBlock("DSV4_T1_TEXT", t1.text)
        if !t1.reasoning.isEmpty { printBlock("DSV4_T1_REASONING", t1.reasoning) }
        if t1.info?.unclosedReasoning == true {
            try fail(22, "turn1 ended inside reasoning despite enable_thinking=false")
        }
        if stoppedByLength(t1) {
            try fail(25, "turn1 stopped by length instead of EOS/stop")
        }
        chat.append(.assistant(t1.visible))

        chat.append(.user("What exact launch key did I ask you to remember? Answer with only the key."))
        let t2 = try await ask(
            label: "turn2-recall", chat: chat, enableThinking: false,
            maxTokens: chatMaxNew)
        printBlock("DSV4_T2_TEXT", t2.text)
        if !t2.reasoning.isEmpty { printBlock("DSV4_T2_REASONING", t2.reasoning) }
        if t2.info?.unclosedReasoning == true {
            try fail(23, "turn2 ended inside reasoning despite enable_thinking=false")
        }
        if stoppedByLength(t2) {
            try fail(26, "turn2 stopped by length instead of EOS/stop")
        }
        let t2Visible = t2.visible.trimmingCharacters(in: .whitespacesAndNewlines)
        let t2Lower = t2Visible.lowercased()
        if !t2Lower.contains(launchKey.lowercased()) {
            try fail(20, "turn2 did not recall exact \(launchKey)")
        }
        if t2Lower.contains("sappberry") || t2Visible.count > 80 {
            try fail(
                28,
                "turn2 did not follow exact-only recall; chars=\(t2Visible.count)")
        }
        chat.append(.assistant(t2.visible))

        chat.append(.user("Is sapphire usually associated with blue? Answer yes or no."))
        let t3 = try await ask(
            label: "turn3-followup", chat: chat, enableThinking: false,
            maxTokens: max(48, min(maxNew, 96)))
        printBlock("DSV4_T3_TEXT", t3.text)
        if !t3.reasoning.isEmpty { printBlock("DSV4_T3_REASONING", t3.reasoning) }
        if t3.info?.unclosedReasoning == true {
            try fail(24, "turn3 ended inside reasoning despite enable_thinking=false")
        }
        if stoppedByLength(t3) {
            try fail(27, "turn3 stopped by length instead of EOS/stop")
        }
        if !containsAll(t3.visible, ["yes"]) && !t3.visible.lowercased().contains("blue") {
            try fail(21, "turn3 did not coherently answer the sapphire/blue follow-up")
        }
    }

    // 2. Reasoning off/on/max-effort routing. The assertion is stream-level:
    // no raw markers in .chunk; thinking modes must produce non-empty routed
    // reasoning or answer content.
    if shouldRun("reasoning") {
        print("\n[DSV4 reasoning modes]")
        let off = try await ask(
            label: "reasoning-off",
            chat: [.user("Q: What is 7 + 5? Answer with just the number.\nA:")],
            enableThinking: false,
            maxTokens: max(32, min(maxNew, 64)))
        printBlock("DSV4_REASONING_OFF_TEXT", off.text)
        if off.info?.unclosedReasoning == true {
            try fail(33, "reasoning-off ended inside reasoning")
        }
        if !off.visible.contains("12") {
            try fail(30, "reasoning-off answer did not contain 12")
        }

        let on = try await ask(
            label: "reasoning-on",
            chat: [.user("Think briefly, then answer: what is 7 + 5?")],
            enableThinking: true,
            maxTokens: max(96, min(maxNew, 160)))
        printBlock("DSV4_REASONING_ON_TEXT", on.text)
        if !on.reasoning.isEmpty { printBlock("DSV4_REASONING_ON_REASONING", on.reasoning) }
        if on.info?.unclosedReasoning == true {
            try fail(34, "reasoning-on did not close its reasoning block")
        }
        if stoppedByLength(on) {
            try fail(39, "reasoning-on stopped by length instead of EOS/stop")
        }
        if on.reasoning.isEmpty && on.text.isEmpty {
            try fail(31, "reasoning-on emitted neither reasoning nor answer")
        }

        let maxEffort = try await ask(
            label: "reasoning-max",
            chat: [.user("Q: What is 7 + 5? Use careful reasoning, then answer with just the number.\nA:")],
            enableThinking: true,
            reasoningEffort: "max",
            maxTokens: reasoningMaxNew)
        printBlock("DSV4_REASONING_MAX_TEXT", maxEffort.text)
        if !maxEffort.reasoning.isEmpty {
            printBlock("DSV4_REASONING_MAX_REASONING", maxEffort.reasoning)
        }
        if maxEffort.info?.unclosedReasoning == true {
            try fail(35, "reasoning-max did not close its reasoning block")
        }
        if stoppedByLength(maxEffort) {
            try fail(37, "reasoning-max stopped by length instead of EOS/stop")
        }
        let repeatedFinal = occurrenceCount(
            "the final answer is 12", in: maxEffort.text.lowercased())
        if repeatedFinal >= 3 {
            try fail(38, "reasoning-max repeated final answer \(repeatedFinal)x")
        }
        if !maxEffort.visible.contains("12") {
            try fail(36, "reasoning-max answer did not contain 12")
        }
        if maxEffort.visible.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            try fail(32, "reasoning-max emitted empty output")
        }
    }

    // 3. Long-context semantic recall. Keep the decisive fact near the front
    // and ask only after a large filler body; this fails if DSV4 is effectively
    // local-window-only.
    if shouldRun("long") {
        print("\n[DSV4 long-context semantic recall]")
        let filler = (0..<max(1, longRepeat)).map { i in
            "Filler line \(i): routine deployment note about packaging, telemetry, cache cleanup, and harmless status bookkeeping."
        }.joined(separator: "\n")
        let longPrompt = """
        Read the operations briefing and answer the final question using the briefing.

        Critical launch note near the beginning: the secret codename is CERULEAN RIVER, and the launch city is OSLO.

        \(filler)

        Final question: what are the secret codename and launch city? Answer with just those two values.
        """
        let long = try await ask(
            label: "long-context-recall",
            chat: [.user(longPrompt)],
            enableThinking: false,
            maxTokens: longMaxNew)
        printBlock("DSV4_LONG_CONTEXT_TEXT", long.text)
        if !long.reasoning.isEmpty { printBlock("DSV4_LONG_CONTEXT_REASONING", long.reasoning) }
        if long.info?.unclosedReasoning == true {
            try fail(41, "long-context recall ended inside reasoning despite enable_thinking=false")
        }
        if !containsAll(long.visible, ["cerulean", "river", "oslo"]) {
            try fail(
                40,
                "long-context answer missed CERULEAN RIVER / OSLO; promptTokens=\(long.promptTokens)")
        }
    }

    if let snapshot = coordinator?.snapshotStats() {
        let paged = snapshot.pagedStats.map {
            "hits=\($0.cacheHits),misses=\($0.cacheMisses),allocated=\($0.allocatedBlocks),free=\($0.freeBlocks),evictions=\($0.evictions)"
        } ?? "disabled"
        let disk = snapshot.diskStats.map {
            "hits=\($0.hits),misses=\($0.misses),stores=\($0.stores),maxBytes=\($0.maxSizeBytes)"
        } ?? "disabled"
        let ssm = snapshot.ssmStats
        print(
            "DSV4_CACHE_STATS hybrid=\(snapshot.isHybrid) pagedIncompatible=\(snapshot.isPagedIncompatible) paged{\(paged)} disk{\(disk)} ssm{hits=\(ssm.hits),misses=\(ssm.misses),reDerives=\(ssm.reDerives)}"
        )
    }

    await engine.shutdown()
    }
    try? await Task.sleep(nanoseconds: 100_000_000)
    print("\n=== BENCH_DSV4_COHERENCE: PASS ===")
}

func runDSV4AgenticToolCheck(modelPath: String, maxNew: Int) async throws {
    let modelDir = URL(fileURLWithPath: modelPath)
    let env = ProcessInfo.processInfo.environment
    let maxTokens = max(96, min(maxNew, 192))
    let cacheDir = URL(fileURLWithPath: env["BENCH_DSV4_AGENTIC_CACHE_DIR"]
        ?? "/tmp/vmlx-dsv4-agentic-cache/\(modelDir.lastPathComponent)")

    print("\n=== BENCH_DSV4_AGENTIC_TOOL: DSML tool + chat history coherence ===")
    let loadStart = CFAbsoluteTimeGetCurrent()
    let context = try await MLXLMCommon.loadModel(
        from: modelDir, using: #huggingFaceTokenizerLoader())
    print(String(format: "Load: %.2fs", CFAbsoluteTimeGetCurrent() - loadStart))
    print("Model: \(type(of: context.model))")
    print("Tool format: \(context.configuration.toolCallFormat.map { "\($0)" } ?? "json (default)")")
    print("Reasoning stamp: \(context.configuration.reasoningParserName ?? "nil")")
    guard context.configuration.toolCallFormat == .dsml else {
        throw NSError(
            domain: "BENCH_DSV4_AGENTIC_TOOL",
            code: 1,
            userInfo: [
                NSLocalizedDescriptionKey:
                    "DSV4 agentic row requires DSML tool-call format, got \(context.configuration.toolCallFormat.map { "\($0)" } ?? "nil")",
            ])
    }
    guard context.configuration.reasoningParserName == "think_xml" else {
        throw NSError(
            domain: "BENCH_DSV4_AGENTIC_TOOL",
            code: 2,
            userInfo: [
                NSLocalizedDescriptionKey:
                    "DSV4 agentic row requires think_xml reasoning parser, got \(context.configuration.reasoningParserName ?? "nil")",
            ])
    }

    nonisolated(unsafe) let ctx = context
    var coordConfig = CacheCoordinatorConfig()
    coordConfig.usePagedCache = true
    coordConfig.enableDiskCache = true
    coordConfig.diskCacheDir = cacheDir
    coordConfig.diskCacheMaxGB = 4
    coordConfig.pagedBlockSize = 256
    coordConfig.maxCacheBlocks = 1024
    coordConfig.modelKey = "\(modelDir.lastPathComponent)|dsv4-agentic-tool"
    let coordinator = CacheCoordinator(config: coordConfig)
    let engine = BatchEngine(context: ctx, maxBatchSize: 1, cacheCoordinator: coordinator)
    defer { Task { await engine.shutdown() } }

    let fallbackParams = GenerateParameters(maxTokens: maxTokens, prefillStepSize: 512)
    var params = GenerateParameters(
        generationConfig: ctx.configuration.generationDefaults,
        fallback: fallbackParams)
    params.maxTokens = maxTokens
    params.prefillStepSize = 512
    params.enableCompiledBatchDecode = false
    print(String(format:
        "Sampling: maxTokens=%d temp=%.3f topP=%.3f topK=%d minP=%.3f rep=%@ seed=%@",
        params.maxTokens ?? -1,
        Double(params.temperature),
        Double(params.topP),
        params.topK,
        Double(params.minP),
        params.repetitionPenalty.map { String(format: "%.3f", Double($0)) } ?? "nil",
        params.randomSeed.map(String.init) ?? "nil"))

    let lookupParams: [String: any Sendable] = [
        "type": "object",
        "properties": [
            "launch_id": [
                "type": "string",
                "description": "Launch identifier to look up.",
            ] as [String: any Sendable],
        ] as [String: any Sendable],
        "required": ["launch_id"],
    ]
    let lookupTool: ToolSpec = [
        "type": "function",
        "function": [
            "name": "lookup_launch_status",
            "description": "Look up launch readiness status.",
            "parameters": lookupParams,
        ] as [String: any Sendable],
    ]

    struct DSV4ToolTurn {
        var text = ""
        var reasoning = ""
        var toolCalls: [ToolCall] = []
        var promptTokens = 0
        var genTokens = 0
        var tokps = 0.0
        var stop = "nil"
        var unclosedReasoning = false
    }

    func fail(_ code: Int, _ message: String) throws -> Never {
        throw NSError(
            domain: "BENCH_DSV4_AGENTIC_TOOL",
            code: code,
            userInfo: [NSLocalizedDescriptionKey: message])
    }

    func markerLeaks(in text: String) -> [String] {
        [
            "<think>", "</think>", "<tool_call>", "<|tool_call>",
            "[TOOL_CALLS]", "<｜tool▁calls｜>", "<｜tool▁call▁begin｜>",
            "<\u{FF5C}DSML\u{FF5C}tool_calls>", "<\u{FF5C}DSML\u{FF5C}invoke",
        ].filter { text.contains($0) }
    }

    func printBlock(_ label: String, _ text: String) {
        print("--- \(label)_BEGIN ---")
        print(text)
        print("--- \(label)_END ---")
    }

    func runTurn(
        label: String,
        chat: [Chat.Message],
        tools: [ToolSpec]? = nil,
        enableThinking: Bool
    ) async throws -> DSV4ToolTurn {
        let input = try await ctx.processor.prepare(input: UserInput(
            chat: chat,
            tools: tools,
            additionalContext: ["enable_thinking": enableThinking]))
        let promptTokens = input.text.tokens.size
        nonisolated(unsafe) let sendable = input
        print("  \(label): prepared prompt=\(promptTokens) tools=\(tools?.count ?? 0)")

        var result = DSV4ToolTurn(promptTokens: promptTokens)
        let start = CFAbsoluteTimeGetCurrent()
        let stream = await engine.generate(input: sendable, parameters: params)
        for await event in stream {
            switch event {
            case .chunk(let chunk):
                result.text += chunk
            case .reasoning(let reasoning):
                result.reasoning += reasoning
            case .toolCall(let call):
                result.toolCalls.append(call)
            case .info(let info):
                result.genTokens = info.generationTokenCount
                result.stop = "\(info.stopReason)"
                result.unclosedReasoning = info.unclosedReasoning
            default:
                break
            }
        }
        let elapsed = max(CFAbsoluteTimeGetCurrent() - start, 0.001)
        result.tokps = Double(result.genTokens) / elapsed
        let preview = (result.text.isEmpty ? "[empty]" : result.text)
            .replacingOccurrences(of: "\n", with: "\\n")
        print(String(format:
            "  %@: text=%d reasoning=%d toolCalls=%d gen=%d stop=%@ unclosed=%@ tokps=%.2f sample=\"%@\"",
            label, result.text.count, result.reasoning.count, result.toolCalls.count,
            result.genTokens, result.stop, result.unclosedReasoning ? "true" : "false",
            result.tokps, String(preview.prefix(180))))
        if !markerLeaks(in: result.text).isEmpty {
            try fail(10, "\(label): raw parser marker leaked into visible chunk")
        }
        if result.unclosedReasoning {
            try fail(11, "\(label): stream ended with unclosed reasoning")
        }
        return result
    }

    var chat: [Chat.Message] = [
        .system("You are a concise assistant. Use tools when requested."),
        .user("Use lookup_launch_status for launch_id DSV4-77. Emit only the tool call."),
    ]
    let toolTurn = try await runTurn(
        label: "turn1-tool-call",
        chat: chat,
        tools: [lookupTool],
        enableThinking: false)
    printBlock("DSV4_AGENTIC_T1_TEXT", toolTurn.text)
    if !toolTurn.reasoning.isEmpty {
        printBlock("DSV4_AGENTIC_T1_REASONING", toolTurn.reasoning)
    }
    guard !toolTurn.toolCalls.isEmpty else {
        try fail(20, "turn1 did not emit a structured .toolCall")
    }
    let toolCalls = toolTurn.toolCalls
    let toolText = toolTurn.text
    let toolCall = toolCalls[0]
    let toolCallId = toolCall.id ?? "call_0_\(toolCall.function.name)"
    guard toolCall.function.name == "lookup_launch_status" else {
        try fail(21, "unexpected tool name \(toolCall.function.name)")
    }
    let toolArgs = toolCall.function.arguments.mapValues { $0.anyValue }
    print("DSV4_AGENTIC_TOOL_CALL name=\(toolCall.function.name) id=\(toolCallId) args=\(toolArgs)")
    let toolResultJSON = #"{"launch_id":"DSV4-77","status":"green","window":"04:30Z"}"#
    chat.append(.assistant(toolText, toolCalls: toolCalls))
    chat.append(.tool(toolResultJSON, toolCallId: toolCallId))
    chat.append(.user("Summarize the launch status and window from the tool result. Answer in one short sentence."))

    let summaryTurn = try await runTurn(
        label: "turn2-tool-result-summary",
        chat: chat,
        enableThinking: false)
    printBlock("DSV4_AGENTIC_T2_TEXT", summaryTurn.text)
    if !summaryTurn.reasoning.isEmpty {
        printBlock("DSV4_AGENTIC_T2_REASONING", summaryTurn.reasoning)
    }
    let summaryLower = summaryTurn.text.lowercased()
    if !summaryLower.contains("green") || !summaryTurn.text.contains("04:30") {
        try fail(30, "turn2 did not use tool result status/window")
    }
    chat.append(.assistant(summaryTurn.text))
    chat.append(.user("What launch id did the tool result describe? Answer with only the id."))

    let recallTurn = try await runTurn(
        label: "turn3-tool-history-recall",
        chat: chat,
        enableThinking: false)
    printBlock("DSV4_AGENTIC_T3_TEXT", recallTurn.text)
    if !recallTurn.text.uppercased().contains("DSV4-77") {
        try fail(31, "turn3 did not recall DSV4-77 from tool history")
    }

    let snapshot = coordinator.snapshotStats()
    let paged = snapshot.pagedStats.map {
        "hits=\($0.cacheHits),misses=\($0.cacheMisses),allocated=\($0.allocatedBlocks),free=\($0.freeBlocks),evictions=\($0.evictions)"
    } ?? "disabled"
    let disk = snapshot.diskStats.map {
        "hits=\($0.hits),misses=\($0.misses),stores=\($0.stores),maxBytes=\($0.maxSizeBytes)"
    } ?? "disabled"
    let ssm = snapshot.ssmStats
    print(
        "DSV4_AGENTIC_CACHE_STATS hybrid=\(snapshot.isHybrid) pagedIncompatible=\(snapshot.isPagedIncompatible) paged{\(paged)} disk{\(disk)} ssm{hits=\(ssm.hits),misses=\(ssm.misses),reDerives=\(ssm.reDerives)}"
    )
    guard snapshot.isPagedIncompatible else {
        try fail(40, "DSV4 agentic row did not mark cache as paged-incompatible")
    }
    guard let diskStats = snapshot.diskStats else {
        try fail(41, "DSV4 agentic row did not enable disk L2 cache")
    }
    guard diskStats.stores > 0 else {
        try fail(42, "DSV4 agentic row did not store any disk L2 cache entries")
    }
    guard diskStats.hits > 0 else {
        try fail(43, "DSV4 agentic row did not observe any disk L2 cache hits")
    }
    await engine.shutdown()
    print("=== BENCH_DSV4_AGENTIC_TOOL: PASS ===")
}

/// Side-by-side decode on the same simple factual prompt across DSV4's
/// three prompt-construction modes. Loads the model ONCE and runs the
/// forward pass through each mode. No assertions — prints decoded
/// output so a human can read whether each mode actually answers.
///
/// The reasoning_parser is NOT applied — we want raw model output
/// including any `<think>...</think>` envelope.
func runDSV4FIMvsChat(modelPath: String, maxNew: Int) async throws {
    let modelDir = URL(fileURLWithPath: modelPath)
    print("\n=== BENCH_DSV4_FIM_VS_CHAT: FIM vs chat coherence ===")
    let loadStart = CFAbsoluteTimeGetCurrent()
    let context = try await MLXLMCommon.loadModel(
        from: modelDir, using: #huggingFaceTokenizerLoader())
    print(String(format: "Load: %.2fs", CFAbsoluteTimeGetCurrent() - loadStart))
    print("Model: \(type(of: context.model))")

    let env = ProcessInfo.processInfo.environment
    let prompt = env["BENCH_DSV4_FIM_PROMPT"] ?? "The capital of France is"
    let question = env["BENCH_DSV4_FIM_QUESTION"]
        ?? "Q: Briefly, what is the capital of France?\nA:"

    func decode(_ tokens: [Int]) -> String {
        context.tokenizer.decode(
            tokenIds: tokens, skipSpecialTokens: false)
    }

    func runMode(label: String, promptTokens: [Int]) async throws {
        print("\n[\(label)]  prompt tokens: \(promptTokens.count)")
        print("  prompt rendered: \(decode(promptTokens).debugDescription)")

        nonisolated(unsafe) let ctx = context
        let engine = BatchEngine(context: ctx, maxBatchSize: 1)
        let promptArr = MLXArray(promptTokens.map { Int32($0) })
            .reshaped(1, promptTokens.count)
        let input = LMInput(text: LMInput.Text(tokens: promptArr))
        nonisolated(unsafe) let send = input
        let params = GenerateParameters(
            maxTokens: maxNew, temperature: 0, prefillStepSize: 512)
        let stream = await engine.submit(input: send, parameters: params)

        var generated: [Int] = []
        let startTime = CFAbsoluteTimeGetCurrent()
        for await event in stream.1 {
            if case .token(let id) = event { generated.append(id) }
        }
        let dt = CFAbsoluteTimeGetCurrent() - startTime
        let text = decode(generated)
        let tps = dt > 0 ? Double(generated.count) / dt : 0
        print(String(format: "  decoded %d tokens in %.2fs (%.1f tok/s)",
            generated.count, dt, tps))
        print("  ---")
        print(text)
        print("  ---")
    }

    // Mode 1: FIM — raw prompt, no chat template, no markers.
    let fimTokens = context.tokenizer.encode(text: prompt)
    try await runMode(label: "FIM (raw)", promptTokens: fimTokens)

    // Mode 2: CHAT no-think — applyChatTemplate enable_thinking=false.
    let messages: [[String: any Sendable]] = [
        ["role": "user", "content": question]
    ]
    let chatNoThink = try context.tokenizer.applyChatTemplate(
        messages: messages,
        tools: nil,
        additionalContext: ["enable_thinking": false])
    try await runMode(label: "CHAT enable_thinking=false",
        promptTokens: chatNoThink)

    // Mode 3: CHAT thinking — applyChatTemplate enable_thinking=true.
    let chatThink = try context.tokenizer.applyChatTemplate(
        messages: messages,
        tools: nil,
        additionalContext: ["enable_thinking": true])
    try await runMode(label: "CHAT enable_thinking=true",
        promptTokens: chatThink)

    print("\n=== BENCH_DSV4_FIM_VS_CHAT: done — read outputs above ===")
}

// MARK: - Qwen3.6 multi-turn + tool-call leak check (exact tpae scenario)

/// Replays the EXACT pattern from tpae's 2026-04-20 3:02-3:04 PM
/// screenshots: Qwen3.6 with `enable_thinking=true`, three turns,
/// includes a synthetic tool-response role. Asserts that across ALL
/// turns, `.chunk(String)` never contains `<think>` or `</think>`
/// markers — the bug tpae reported was "thinking bleeds into
/// content" after a tool call. If the fix works per-request (each
/// request construction builds a fresh parser with
/// startInReasoning=true), every turn must be clean.
func runQwenMultiturnToolCheck(modelPath: String, maxNew: Int) async throws {
    let modelDir = URL(fileURLWithPath: modelPath)
    print("\n=== BENCH_QWEN_MULTITURN_TOOL: Qwen3.x 3-turn + tool call ===")
    let loadStart = CFAbsoluteTimeGetCurrent()
    let context = try await MLXLMCommon.loadModel(
        from: modelDir, using: #huggingFaceTokenizerLoader())
    print(String(format: "Load: %.2fs", CFAbsoluteTimeGetCurrent() - loadStart))
    print("Model: \(type(of: context.model))")
    print("Reasoning stamp: \(context.configuration.reasoningParserName ?? "nil")")

    struct TurnResult {
        let idx: Int
        let requiresToolCall: Bool
        let promptTokens: Int
        let chunks: Int
        let reasoningDeltas: Int
        let toolCalls: Int
        let tokps: Double
        let chunkSample: String
        let reasoningSample: String
        let leakedThink: Bool
        let emptyVisible: Bool
    }

    let weatherParams: [String: any Sendable] = [
        "type": "object",
        "properties": [
            "location": ["type": "string", "description": "City or region name"] as [String: any Sendable],
        ] as [String: any Sendable],
        "required": ["location"],
    ]
    let weatherTool: [String: any Sendable] = [
        "type": "function",
        "function": [
            "name": "get_weather",
            "description": "Get current weather for a location.",
            "parameters": weatherParams,
        ] as [String: any Sendable],
    ]

    // Three concrete parser surfaces. Keep the prompts bounded so the row
    // proves parser/tool behavior instead of failing because an open-ended
    // prompt spends the whole token cap in the reasoning channel.
    let turns: [(
        label: String,
        messages: [Chat.Message],
        tools: [[String: any Sendable]]?,
        enableThinking: Bool,
        requiresToolCall: Bool
    )] = [
        (
            "Turn 1 — thinking visible answer",
            [
                .user("What is 7+8-11? Reply with just the number.")
            ],
            nil,
            true,
            false
        ),
        (
            "Turn 2 — structured tool call",
            [
                .user("Call get_weather for location Tokyo. Emit only the tool call.")
            ],
            [weatherTool],
            false,
            true
        ),
        (
            "Turn 3 — thinking off visible answer",
            [
                .user("Reply with exactly qwen-visible-ok.")
            ],
            nil,
            false,
            false
        ),
    ]

    var results: [TurnResult] = []
    var anyLeak = false

    for (idx, turn) in turns.enumerated() {
        let prepared = try await context.processor.prepare(input: UserInput(
            chat: turn.messages,
            tools: turn.tools,
            additionalContext: ["enable_thinking": turn.enableThinking]))
        let input = prepared.withToolSchemas(turn.tools)
        let promptTokens = input.text.tokenIds ?? []
        let promptTokenCount = promptTokens.isEmpty ? input.text.tokens.size : promptTokens.count
        if turn.requiresToolCall && input.toolSchemas?.isEmpty != false {
            fputs("\nFAIL: tool-required turn \(idx + 1) lost prepared tool schemas before decode\n", stderr)
            exit(4)
        }
        nonisolated(unsafe) let ctxSendable = context
        nonisolated(unsafe) let sendable = input

        let engine = BatchEngine(context: ctxSendable, maxBatchSize: 1)
        let fallback = GenerateParameters(maxTokens: maxNew, prefillStepSize: 512)
        var params = GenerateParameters(
            generationConfig: context.configuration.generationDefaults,
            fallback: fallback)
        params.maxTokens = maxNew
        params.prefillStepSize = 512
        let stream = await engine.generate(input: sendable, parameters: params)

        var chunkText = ""
        var reasoningText = ""
        var chunkCount = 0
        var reasoningCount = 0
        var toolCallCount = 0
        let streamStart = CFAbsoluteTimeGetCurrent()
        for await ev in stream {
            switch ev {
            case .chunk(let c):
                chunkText += c
                chunkCount += 1
            case .reasoning(let r):
                reasoningText += r
                reasoningCount += 1
            case .toolCall:
                toolCallCount += 1
            case .info:
                break
            }
        }
        let elapsed = max(CFAbsoluteTimeGetCurrent() - streamStart, 0.001)
        let generatedDeltas = chunkCount + reasoningCount
        let tokps = Double(generatedDeltas) / elapsed
        // Check every envelope pattern we support — whichever the model's
        // family uses. If ANY leaks, the test fails.
        let leakedMarkers = [
            "<think>", "</think>",           // Qwen/DeepSeek/GLM/MiniMax/Nemotron
            "<|channel>", "<channel|>",      // Gemma-4 harmony
        ]
        let leaked = leakedMarkers.contains { chunkText.contains($0) }
        if leaked { anyLeak = true }
        let emptyVisible = chunkText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        let r = TurnResult(
            idx: idx + 1,
            requiresToolCall: turn.requiresToolCall,
            promptTokens: promptTokenCount,
            chunks: chunkCount,
            reasoningDeltas: reasoningCount,
            toolCalls: toolCallCount,
            tokps: tokps,
            chunkSample: String(chunkText.prefix(160)),
            reasoningSample: String(reasoningText.prefix(160)),
            leakedThink: leaked,
            emptyVisible: emptyVisible)
        results.append(r)
        print("\n[\(turn.label)] promptTokens=\(promptTokenCount)")
        print(String(format:
            "  chunks=%d reasoning=%d toolCalls=%d tokps=%.1f leakedThinkMarkers=%@ emptyVisible=%@",
            chunkCount, reasoningCount, toolCallCount, tokps,
            leaked ? "true" : "false", emptyVisible ? "true" : "false"))
        print("  .chunk: \"\(r.chunkSample)\"")
        print("  .reasoning: \"\(r.reasoningSample)\"")
    }

    print("\n=== Turn-by-turn summary ===")
    for r in results {
        print(String(format:
            "  Turn %d: prompt=%d, chunks=%d, reasoning=%d, toolCalls=%d, tokps=%.1f, leak=%@, emptyVisible=%@",
            r.idx, r.promptTokens, r.chunks, r.reasoningDeltas, r.toolCalls,
            r.tokps, r.leakedThink ? "true" : "false", r.emptyVisible ? "true" : "false"))
    }
    if anyLeak {
        fputs("\nFAIL: at least one turn leaked reasoning envelope markers in .chunk\n", stderr)
        exit(1)
    }
    if results.contains(where: { $0.emptyVisible && $0.toolCalls == 0 }) {
        fputs("\nFAIL: at least one non-tool turn produced reasoning-only output with no visible chunk\n", stderr)
        exit(2)
    }
    if let missingTool = results.first(where: { $0.requiresToolCall && $0.toolCalls == 0 }) {
        fputs("\nFAIL: tool-required turn \(missingTool.idx) produced no structured .toolCall event\n", stderr)
        exit(3)
    }
    print("\nPASS: all turns have zero reasoning envelope markers in .chunk")
    print("PASS: no non-tool reasoning-only empty-visible turns")
    print("PASS: all tool-required turns emitted structured .toolCall events")
    print("=== BENCH_QWEN_MULTITURN_TOOL: passed ===")
}

// MARK: - Perf micro-bench (BENCH_PERF=1)

/// Deterministic decode tok/s micro-bench. Runs warmup + measurement
/// turns, computes median tok/s from the library's own `.info`
/// `generationTime`. Grep-friendly one-liner output per run:
///
///   PERF model=<name> variant=<label> genTokens=N genSec=F tokps=F
///
/// Default mode uses temperature 0 and a fixed token budget so the only
/// variable is the decode hot path. Set BENCH_PERF_USE_GENERATION_CONFIG=1
/// to seed sampling from the bundle's generation_config.json, with explicit
/// BENCH_PERF_TEMP/TOP_P/TOP_K/MIN_P/REPETITION_PENALTY env overrides still
/// taking final precedence. Set BENCH_PERF_SEED to make stochastic rows
/// reproducible.
func runPerfBench(
    modelPath: String,
    maxNew: Int,
    variant: String,
    warmup: Int,
    runs: Int,
    useTokenIterator: Bool = false
) async throws {
    let modelDir = URL(fileURLWithPath: modelPath)
    let env = ProcessInfo.processInfo.environment
    let modelName = modelDir.lastPathComponent
    let useJangPressLoad = env["BENCH_PERF_JANGPRESS"] == "1"
    let useMmap = env["BENCH_PERF_MMAP"] != "0"
        let pathLabel = useTokenIterator ? "iter" : "batch"
        var perfLine = ""

        do {
        let rssBeforeLoad = currentRSSMiB()
        let footprintBeforeLoad = currentPhysFootprintMiB()
        print(String(format:
            "PERF_MEMORY label=before_load rss_mib=%.0f footprint_mib=%.0f",
            rssBeforeLoad, footprintBeforeLoad))
        var peakRSSMiB = rssBeforeLoad
        var peakFootprintMiB = footprintBeforeLoad
        let loadStart = CFAbsoluteTimeGetCurrent()
        let context: ModelContext
        let jangPressRuntime: JangPressRuntime?
        let nativeMTPRequestedAtLoad =
            env["BENCH_PERF_NATIVE_MTP_DEPTH"] != nil
            || env["BENCH_PERF_LOAD_NATIVE_MTP"] == "1"
        if useJangPressLoad {
            let loaded = try await MLXLMCommon.loadModel(
                from: modelDir,
                using: #huggingFaceTokenizerLoader(),
                loadConfiguration: LoadConfiguration(
                    jangPress: .enabled(coldFraction: 0.70),
                    maxResidentBytes: .fraction(0.70),
                    memoryLimit: .fraction(0.70),
                    useMmapSafetensors: useMmap,
                    nativeMTP: nativeMTPRequestedAtLoad))
            context = loaded.0
            jangPressRuntime = loaded.1
        } else if nativeMTPRequestedAtLoad {
            let loaded = try await MLXLMCommon.loadModel(
                from: modelDir,
                using: #huggingFaceTokenizerLoader(),
                loadConfiguration: LoadConfiguration(
                    jangPress: .disabled,
                    maxResidentBytes: .unlimited,
                    memoryLimit: .unlimited,
                    useMmapSafetensors: useMmap,
                    nativeMTP: true))
            context = loaded.0
            jangPressRuntime = loaded.1
        } else {
            if useMmap {
                let loaded = try await MLXLMCommon.loadModel(
                    from: modelDir,
                    using: #huggingFaceTokenizerLoader(),
                    loadConfiguration: LoadConfiguration(useMmapSafetensors: true))
                context = loaded.0
                jangPressRuntime = loaded.1
            } else {
                context = try await MLXLMCommon.loadModel(
                    from: modelDir, using: #huggingFaceTokenizerLoader())
                jangPressRuntime = nil
            }
        }
        let rssAfterLoad = currentRSSMiB()
        let footprintAfterLoad = currentPhysFootprintMiB()
        peakRSSMiB = max(peakRSSMiB, rssAfterLoad)
        peakFootprintMiB = max(peakFootprintMiB, footprintAfterLoad)
        defer {
            if let jangPressRuntime {
                JangPressActivation.deactivate(jangPressRuntime)
            }
        }
        let loadSec = CFAbsoluteTimeGetCurrent() - loadStart
        print(String(format:
            "PERF_MEMORY label=after_load rss_mib=%.0f footprint_mib=%.0f loadSec=%.2f",
            rssAfterLoad, footprintAfterLoad, loadSec))

        let promptText = env[
            "BENCH_PERF_PROMPT"]
            ?? "Write one long paragraph describing ocean waves. Be verbose and detailed."
        let messages: [[String: any Sendable]] = [
            ["role": "user", "content": promptText]
        ]

        let promptTokens: [Int]
        let enableThinking = (env["BENCH_PERF_ENABLE_THINKING"] ?? "0") == "1"
        do {
            promptTokens = try context.tokenizer.applyChatTemplate(
                messages: messages,
                tools: nil,
                additionalContext: ["enable_thinking": enableThinking])
        } catch {
            promptTokens = try context.tokenizer.applyChatTemplate(messages: messages)
        }
        print("PERF_TEMPLATE enable_thinking=\(enableThinking)")
        let promptIds = MLXArray(promptTokens.map { Int32($0) })
            .reshaped(1, promptTokens.count)

        let graphStats = try await decodeGraphStatsIfRequested(
            context: context,
            promptIds: promptIds,
            parameters: GenerateParameters(
                maxTokens: 1, temperature: 0, prefillStepSize: 512))
        if let graphStats {
            print(
                "GRAPH_STATS decodeNodes=\(graphStats.nodes) asType=\(graphStats.asType)"
            )
        }

        let perfCacheCoordinator: CacheCoordinator?
        if env["BENCH_PERF_CACHE_COORDINATOR"] == "1" {
            let modelKey = "\(modelName)|perf-cache-coordinator"
            let diskDir = env["BENCH_PERF_CACHE_DIR"].map {
                URL(fileURLWithPath: ($0 as NSString).expandingTildeInPath)
            }
            let config = CacheCoordinatorConfig(
                usePagedCache: env["BENCH_PERF_CACHE_PAGED"] != "0",
                enableDiskCache: env["BENCH_PERF_CACHE_DISK"] == "1",
                diskCacheMaxGB: Float(env["BENCH_PERF_CACHE_DISK_MAX_GB"] ?? "1") ?? 1,
                diskCacheDir: diskDir,
                enableSSMReDerive: env["BENCH_PERF_CACHE_SSM_REDERIVE"] == "1",
                modelKey: modelKey)
            let coordinator = CacheCoordinator(config: config)
            if env["BENCH_PERF_CACHE_HYBRID"] == "1" {
                coordinator.setHybrid(true)
            }
            perfCacheCoordinator = coordinator
            print(
                "PERF_CACHE_COORDINATOR enabled paged=\(config.usePagedCache) disk=\(config.enableDiskCache) hybrid=\(coordinator.isHybrid) modelKey=\(modelKey)"
            )
        } else {
            perfCacheCoordinator = nil
        }

        nonisolated(unsafe) let ctxSendable = context
        let engine = BatchEngine(
            context: ctxSendable,
            maxBatchSize: 1,
            cacheCoordinator: perfCacheCoordinator)

        struct PerfTurnResult {
            var genTokens = 0
            var genSec = 0.0
            var ttftSec = 0.0
            var promptSec = 0.0
            var text = ""
            var reasoning = ""
            var toolCalls = 0
            var stopReason = "unknown"
            var unclosedReasoning = false
            var rssMiB = 0.0
            var footprintMiB = 0.0

            var tokps: Double {
                genSec > 0 ? Double(genTokens) / genSec : 0
            }

            var firstDecodeSecEstimate: Double {
                max(0, ttftSec - promptSec)
            }

            var tailTokpsEstimate: Double {
                guard genTokens > 1 else { return 0 }
                let tailSec = genSec - firstDecodeSecEstimate
                return tailSec > 0 ? Double(genTokens - 1) / tailSec : 0
            }

            func promptTokensPerSecond(promptTokenCount: Int) -> Double {
                promptSec > 0 ? Double(promptTokenCount) / promptSec : 0
            }
        }

        func stopLabel(_ reason: GenerateStopReason) -> String {
            switch reason {
            case .stop: return "stop"
            case .length: return "length"
            case .cancelled: return "cancelled"
            }
        }

        func compactPreview(_ text: String, limit: Int = 240) -> String {
            let oneLine = text
                .replacingOccurrences(of: "\n", with: "\\n")
                .replacingOccurrences(of: "\t", with: " ")
            return oneLine.count > limit
                ? String(oneLine.prefix(limit)) + "..."
                : oneLine
        }

        func markerLeaks(in text: String) -> [String] {
            [
                "<think>", "</think>", "<tool_call>", "<|tool_call>",
                "[TOOL_CALLS]", "<|channel|>", "<|im_end|>",
                "〈|EOS|〉", "<｜Assistant｜>", "<｜User｜>",
            ].filter { text.contains($0) }
        }

        let useGenerationConfigSampling = env["BENCH_PERF_USE_GENERATION_CONFIG"] == "1"
        let samplingSource = useGenerationConfigSampling
            ? "bundle-defaults"
            : "explicit-env"
        let perfSeed = env["BENCH_PERF_SEED"].flatMap(UInt64.init)
        let perfSeedLabel = perfSeed.map(String.init) ?? "nil"

        func oneTurn(_ label: String) async throws -> PerfTurnResult {
            let input = LMInput(text: LMInput.Text(tokens: promptIds))
            nonisolated(unsafe) let sendable = input
            let perfRepetitionContext = Int(env["BENCH_PERF_REPETITION_CONTEXT"] ?? "20") ?? 20
            let printPhaseSnapshot = env["BENCH_PERF_PHASE_SNAPSHOT"] == "1"
            if printPhaseSnapshot {
                if NativeMTPPhaseDiagnostics.enabled {
                    NativeMTPPhaseDiagnostics.reset()
                }
                if NativeMTPGDNReplayDiagnostics.enabled {
                    NativeMTPGDNReplayDiagnostics.reset()
                }
            }
            var params: GenerateParameters
            if useGenerationConfigSampling {
                let fallback = GenerateParameters(
                    maxTokens: maxNew,
                    temperature: 0,
                    topP: 1,
                    topK: 0,
                    minP: 0,
                    randomSeed: perfSeed,
                    repetitionPenalty: nil,
                    repetitionContextSize: perfRepetitionContext,
                    prefillStepSize: 512)
                params = GenerateParameters(generationConfig: context.configuration.generationDefaults,
                    fallback: fallback)
                params.maxTokens = maxNew
                params.prefillStepSize = 512
                params.repetitionContextSize = perfRepetitionContext
            } else {
                let perfTemperature = Float(env["BENCH_PERF_TEMP"] ?? "0") ?? 0
                let perfTopP = Float(env["BENCH_PERF_TOP_P"] ?? "1.0") ?? 1.0
                let perfTopK = Int(env["BENCH_PERF_TOP_K"] ?? "0") ?? 0
                let perfMinP = Float(env["BENCH_PERF_MIN_P"] ?? "0") ?? 0
                let perfRepetitionPenalty = env["BENCH_PERF_REPETITION_PENALTY"]
                    .flatMap(Float.init)
                params = GenerateParameters(
                    maxTokens: maxNew,
                    temperature: perfTemperature,
                    topP: perfTopP,
                    topK: perfTopK,
                    minP: perfMinP,
                    randomSeed: perfSeed,
                    repetitionPenalty: perfRepetitionPenalty,
                    repetitionContextSize: perfRepetitionContext,
                    prefillStepSize: 512)
            }
            if let perfSeed {
                params.randomSeed = perfSeed
            }
            if let override = env["BENCH_PERF_TEMP"].flatMap(Float.init) {
                params.temperature = override
            }
            if let override = env["BENCH_PERF_TOP_P"].flatMap(Float.init) {
                params.topP = override
            }
            if let override = env["BENCH_PERF_TOP_K"].flatMap(Int.init) {
                params.topK = override
            }
            if let override = env["BENCH_PERF_MIN_P"].flatMap(Float.init) {
                params.minP = override
            }
            if let override = env["BENCH_PERF_REPETITION_PENALTY"].flatMap(Float.init) {
                params.repetitionPenalty = override
            }
            if env["BENCH_PERF_COMPILED"] == "1" {
                params.enableCompiledDecode = useTokenIterator
                params.enableCompiledBatchDecode = !useTokenIterator
                params.compiledMaxCacheLength =
                    Int(env["BENCH_PERF_COMPILED_MAX_LEN"] ?? "4096") ?? 4096
            }
            if let nativeMTPDepth = env["BENCH_PERF_NATIVE_MTP_DEPTH"].flatMap(Int.init) {
                params.draftStrategy = .nativeMTP(depth: nativeMTPDepth)
            }
            switch (env["BENCH_PERF_KV_MODE"] ?? "none").lowercased() {
            case "tq", "tq44", "turboquant":
                params.kvMode = .turboQuant(keyBits: 4, valueBits: 4)
                params.quantizedKVStart = Int(env["BENCH_PERF_TQ_START"] ?? "8") ?? 8
            case "tq33":
                params.kvMode = .turboQuant(keyBits: 3, valueBits: 3)
                params.quantizedKVStart = Int(env["BENCH_PERF_TQ_START"] ?? "8") ?? 8
            case "tq88":
                params.kvMode = .turboQuant(keyBits: 8, valueBits: 8)
                params.quantizedKVStart = Int(env["BENCH_PERF_TQ_START"] ?? "8") ?? 8
            default:
                break
            }
            var result = PerfTurnResult()
            let start = CFAbsoluteTimeGetCurrent()
            let whichPath = env["BENCH_PERF_PATH"] ?? "batch"
            if useTokenIterator {
                nonisolated(unsafe) let ctxLocal = ctxSendable
                let stream = try MLXLMCommon.generate(
                    input: sendable, parameters: params, context: ctxLocal)
                for await ev in stream {
                    switch ev {
                    case .chunk(let chunk):
                        if result.ttftSec == 0 {
                            result.ttftSec = CFAbsoluteTimeGetCurrent() - start
                        }
                        result.text += chunk
                    case .reasoning(let reasoning):
                        if result.ttftSec == 0 {
                            result.ttftSec = CFAbsoluteTimeGetCurrent() - start
                        }
                        result.reasoning += reasoning
                    case .toolCall:
                        if result.ttftSec == 0 {
                            result.ttftSec = CFAbsoluteTimeGetCurrent() - start
                        }
                        result.toolCalls += 1
                    case .info(let info):
                        result.genTokens = info.generationTokenCount
                        result.promptSec = info.promptTime
                        result.genSec = info.generateTime
                        result.stopReason = stopLabel(info.stopReason)
                        result.unclosedReasoning = info.unclosedReasoning
                    }
                }
            } else if whichPath == "submit" {
                let (_, stream) = await engine.submit(
                    input: sendable, parameters: params)
                var rawTokens: [Int] = []
                for await ev in stream {
                    switch ev {
                    case .token(let token):
                        if result.ttftSec == 0 {
                            result.ttftSec = CFAbsoluteTimeGetCurrent() - start
                        }
                        rawTokens.append(token)
                    case .info(let info):
                        result.genTokens = info.generationTokenCount
                        result.promptSec = info.promptTime
                        result.genSec = info.generateTime
                        result.stopReason = stopLabel(info.stopReason)
                        result.unclosedReasoning = info.unclosedReasoning
                    }
                }
                result.text = ctxSendable.tokenizer.decode(
                    tokenIds: rawTokens, skipSpecialTokens: false)
            } else {
                let stream = await engine.generate(input: sendable, parameters: params)
                for await ev in stream {
                    switch ev {
                    case .chunk(let chunk):
                        if result.ttftSec == 0 {
                            result.ttftSec = CFAbsoluteTimeGetCurrent() - start
                        }
                        result.text += chunk
                    case .reasoning(let reasoning):
                        if result.ttftSec == 0 {
                            result.ttftSec = CFAbsoluteTimeGetCurrent() - start
                        }
                        result.reasoning += reasoning
                    case .toolCall:
                        if result.ttftSec == 0 {
                            result.ttftSec = CFAbsoluteTimeGetCurrent() - start
                        }
                        result.toolCalls += 1
                    case .info(let info):
                        result.genTokens = info.generationTokenCount
                        result.promptSec = info.promptTime
                        result.genSec = info.generateTime
                        result.stopReason = stopLabel(info.stopReason)
                        result.unclosedReasoning = info.unclosedReasoning
                    }
                }
            }
            result.rssMiB = currentRSSMiB()
            result.footprintMiB = currentPhysFootprintMiB()
            peakRSSMiB = max(peakRSSMiB, result.rssMiB)
            peakFootprintMiB = max(peakFootprintMiB, result.footprintMiB)
            if printPhaseSnapshot {
                let phaseSummary = NativeMTPPhaseDiagnostics.enabled
                    ? NativeMTPPhaseDiagnostics.summary(limit: 12)
                    : "disabled"
                let gdnReplay = NativeMTPGDNReplayDiagnostics.enabled
                    ? NativeMTPGDNReplayDiagnostics.snapshot(reset: true)
                    : NativeMTPGDNReplaySnapshot(calls: 0, prefixStates: 0, seconds: 0)
                if NativeMTPPhaseDiagnostics.enabled {
                    _ = NativeMTPPhaseDiagnostics.snapshot(reset: true)
                }
                print(String(format:
                    "  PERF_PHASE label=%@ phaseDiag=%@ gdnReplayCalls=%d gdnReplayStates=%d gdnReplaySec=%.3f",
                    label, phaseSummary, gdnReplay.calls, gdnReplay.prefixStates,
                    gdnReplay.seconds))
            }
            if label.hasPrefix("run") {
                let visible = result.text.isEmpty ? result.reasoning : result.text
                let leaks = markerLeaks(in: result.text).joined(separator: ",")
                let loop = lagunaLoopHeuristic(visible)
                print(String(format:
                    "  PERF_RUN label=%@ samplingSource=%@ seed=%@ ttft_ms=%.0f prompt_ms=%.0f prompt_tps=%.0f first_decode_ms=%.0f genTokens=%d genSec=%.3f tokps=%.1f tail_tokps_est=%.1f rss_mib=%.0f footprint_mib=%.0f temp=%.2f topP=%.2f topK=%d minP=%.2f rep=%@ stop=%@ unclosedReasoning=%@ textChars=%d reasoningChars=%d toolCalls=%d loop=%@ leaks=%@",
                    label, samplingSource, perfSeedLabel,
                    result.ttftSec * 1000,
                    result.promptSec * 1000,
                    result.promptTokensPerSecond(promptTokenCount: promptTokens.count),
                    result.firstDecodeSecEstimate * 1000,
                    result.genTokens, result.genSec, result.tokps,
                    result.tailTokpsEstimate,
                    result.rssMiB, result.footprintMiB,
                    Double(params.temperature), Double(params.topP),
                    params.topK, Double(params.minP),
                    params.repetitionPenalty.map { String(format: "%.2f", Double($0)) } ?? "nil",
                    result.stopReason,
                    result.unclosedReasoning ? "YES" : "NO",
                    result.text.count, result.reasoning.count,
                    result.toolCalls, loop ? "YES" : "NO",
                    leaks.isEmpty ? "none" : leaks))
                if !result.reasoning.isEmpty {
                    print("    REASONING_PREVIEW \"\(compactPreview(result.reasoning))\"")
                }
                print("    TEXT_PREVIEW \"\(compactPreview(result.text.isEmpty ? result.reasoning : result.text))\"")
                if env["BENCH_PERF_FULL_TEXT"] == "1" {
                    printDecodedOutput(
                        label: label,
                        text: result.text.isEmpty ? result.reasoning : result.text)
                }
            }
            if let snapshot = perfCacheCoordinator?.snapshotStats() {
                let paged = snapshot.pagedStats.map {
                    "hits=\($0.cacheHits),misses=\($0.cacheMisses),allocated=\($0.allocatedBlocks),free=\($0.freeBlocks),evictions=\($0.evictions)"
                } ?? "disabled"
                let disk = snapshot.diskStats.map {
                    "hits=\($0.hits),misses=\($0.misses),stores=\($0.stores),maxBytes=\($0.maxSizeBytes)"
                } ?? "disabled"
                let ssm = snapshot.ssmStats
                print(
                    "PERF_CACHE_STATS label=\(label) hybrid=\(snapshot.isHybrid) pagedIncompatible=\(snapshot.isPagedIncompatible) paged{\(paged)} disk{\(disk)} ssm{hits=\(ssm.hits),misses=\(ssm.misses),reDerives=\(ssm.reDerives)}"
                )
            }
            return result
        }

        for i in 0..<warmup {
            _ = try await oneTurn("warmup\(i)")
        }

        var tokps: [Double] = []
        var lastGenTokens = 0
        var lastGenSec = 0.0
        var lastResult = PerfTurnResult()
        for i in 0..<runs {
            let result = try await oneTurn("run\(i)")
            tokps.append(result.tokps)
            lastGenTokens = result.genTokens
            lastGenSec = result.genSec
            lastResult = result
        }
        await engine.shutdown()

        let median = tokps.sorted()[tokps.count / 2]
        let best = tokps.max() ?? 0
        let lastVisible = lastResult.text.isEmpty
            ? lastResult.reasoning
            : lastResult.text
        let leaks = markerLeaks(in: lastResult.text)
        let loop = lagunaLoopHeuristic(lastVisible)
        let head = String(
            gitShortHead(modelDir: FileManager.default.currentDirectoryPath))
        perfLine = String(format:
            "PERF model=%@ variant=%@ path=%@ samplingSource=%@ seed=%@ kvMode=%@ jangpress=%@ mmap=%@ commit=%@ loadSec=%.2f promptTokens=%d peak_rss_mib=%.0f peak_footprint_mib=%.0f graphNodes=%@ asType=%@ genTokens=%d genSec=%.3f tokps_median=%.1f tokps_best=%.1f runs=%@ stop=%@ unclosedReasoning=%@ loop=%@ leaks=%@",
            modelName, variant, pathLabel, samplingSource, perfSeedLabel,
            env["BENCH_PERF_KV_MODE"] ?? "none",
            useJangPressLoad ? "on" : "off",
            useMmap ? "on" : "off",
            head, loadSec, promptTokens.count,
            peakRSSMiB, peakFootprintMiB,
            graphStats.map { String($0.nodes) } ?? "na",
            graphStats.map { String($0.asType) } ?? "na",
            lastGenTokens, lastGenSec,
            median, best,
            tokps.map { String(format: "%.1f", $0) }.joined(separator: ","),
            lastResult.stopReason,
            lastResult.unclosedReasoning ? "YES" : "NO",
            loop ? "YES" : "NO",
            leaks.isEmpty ? "none" : leaks.joined(separator: ","))
    }

    try? await Task.sleep(nanoseconds: 50_000_000)
    print(perfLine)
}

private struct DecodeGraphStats {
    let nodes: Int
    let asType: Int
}

private func decodeGraphStatsIfRequested(
    context: ModelContext,
    promptIds: MLXArray,
    parameters: GenerateParameters
) async throws -> DecodeGraphStats? {
    guard ProcessInfo.processInfo.environment["BENCH_GRAPH_STATS"] == "1" else {
        return nil
    }

    let cache = context.model.newCache(parameters: parameters)
    let prefillInput = LMInput(text: LMInput.Text(tokens: promptIds))
    switch try context.model.prepare(
        prefillInput, cache: cache, windowSize: parameters.prefillStepSize)
    {
    case .tokens(let text):
        let stepText: LMInput.Text =
            text.tokens.ndim == 1 ? text[text: .newAxis] : text
        let out = context.model(stepText, cache: cache, state: nil)
        MLX.eval(out.logits, cache)
    case .logits(let out):
        MLX.eval(out.logits, cache)
    }

    let seed = promptIds.reshaped(-1)[-1][.newAxis, .newAxis]
    let decodeOut = context.model(
        LMInput.Text(tokens: seed), cache: cache, state: nil)
    return graphStats(for: decodeOut.logits)
}

private func graphStats(for array: MLXArray) -> DecodeGraphStats? {
    var nodes: Int32 = -1
    var asType: Int32 = -1
    let ptr = unsafeBitCast(array.ctx, to: UnsafeMutableRawPointer?.self)
    let rc = vmlx_graph_stats(ptr, &nodes, &asType)
    guard rc == 0 else { return nil }
    return DecodeGraphStats(nodes: Int(nodes), asType: Int(asType))
}

/// Get short git HEAD for the tree rooted at `modelDir`. Purely for
/// reporting — failure returns "?".
fileprivate func gitShortHead(modelDir: String) -> String {
    let p = Process()
    p.launchPath = "/usr/bin/git"
    p.arguments = ["-C", modelDir, "rev-parse", "--short", "HEAD"]
    p.currentDirectoryPath = modelDir
    let pipe = Pipe()
    p.standardOutput = pipe
    p.standardError = Pipe()
    do {
        try p.run()
        p.waitUntilExit()
        let d = pipe.fileHandleForReading.readDataToEndOfFile()
        return String(data: d, encoding: .utf8)?
            .trimmingCharacters(in: .whitespacesAndNewlines) ?? "?"
    } catch {
        return "?"
    }
}

// MARK: - Crash fuzz (2026-04-23)
//
// tpae reported a reproducible `_assertionFailure` on Qwen 3.6 27B via
// osaurus 0.17.3. The osaurus repo ships no source, so we can't map the
// stack. This runner loads the real 27B once and fires a sequence of
// adversarial request patterns that osaurus is likely to produce. Each
// scenario prints `SCENARIO N START` before and `SCENARIO N DONE`
// after — the last "START" line on a crash names the culprit.

func runCrashFuzz(modelPath: String, maxNew: Int) async throws {
    let modelDir = URL(fileURLWithPath: modelPath)
    print("\n=== BENCH_CRASH_FUZZ — tpae Qwen3.6-27B repro probe ===")
    print("Loading with real HuggingFace tokenizer from \(modelPath) ...")
    let loadStart = CFAbsoluteTimeGetCurrent()
    let context = try await MLXLMCommon.loadModel(
        from: modelDir, using: #huggingFaceTokenizerLoader())
    print(String(format: "Load: %.2fs", CFAbsoluteTimeGetCurrent() - loadStart))
    print("Model: \(type(of: context.model))")
    nonisolated(unsafe) let ctx = context

    // Short-decode-budget baseline — each scenario should finish in
    // seconds, not minutes. Keep `maxNew` honest but low.
    let budget = min(maxNew, 48)

    // Helper: drain a submitted stream to completion. Separated from
    // prepare/submit so Swift 6 strict concurrency doesn't trip on
    // `ctx` capture inside TaskGroup closures.
    @Sendable func drain(
        _ stream: AsyncStream<BatchGeneration>,
        cap: Int
    ) async -> (tokens: [Int], stop: GenerateStopReason?) {
        var ids: [Int] = []
        var stop: GenerateStopReason? = nil
        for await e in stream {
            switch e {
            case .token(let id): ids.append(id)
            case .info(let info): stop = info.stopReason
            }
            if ids.count >= cap { break }
        }
        return (ids, stop)
    }

    // Pre-prepare inputs on the main task, then pass sendable LMInputs
    // into TaskGroup closures. Sidesteps the `ctx` non-Sendable capture.
    func prepareInputs(_ prompts: [String]) async throws -> [LMInput] {
        var out: [LMInput] = []
        for p in prompts {
            out.append(try await ctx.processor.prepare(
                input: UserInput(prompt: p)))
        }
        return out
    }

    // Scenario 1: B=1 baseline — prove the engine is healthy at all.
    do {
        print("\nSCENARIO 1 START: B=1 baseline")
        let engine = BatchEngine(context: ctx, maxBatchSize: 1)
        let params = GenerateParameters(
            maxTokens: budget, temperature: 0, prefillStepSize: 512)
        let inputs = try await prepareInputs(["The capital of France is"])
        nonisolated(unsafe) let send = inputs[0]
        let (_, s) = await engine.submit(input: send, parameters: params)
        let r = await drain(s, cap: budget)
        print("SCENARIO 1 DONE: \(r.tokens.count) tokens, stop=\(r.stop.map{"\($0)"} ?? "nil")")
    }

    // Scenario 2: B=4 concurrent distinct prompts.
    do {
        print("\nSCENARIO 2 START: B=4 concurrent")
        let engine = BatchEngine(context: ctx, maxBatchSize: 4)
        let params = GenerateParameters(
            maxTokens: budget, temperature: 0, prefillStepSize: 512)
        let inputs = try await prepareInputs([
            "Name one color.",
            "Name one animal.",
            "Name one country.",
            "Name one number.",
        ])
        var streams: [AsyncStream<BatchGeneration>] = []
        for input in inputs {
            nonisolated(unsafe) let send = input
            let (_, s) = await engine.submit(input: send, parameters: params)
            streams.append(s)
        }
        await withTaskGroup(of: (Int, Int).self) { group in
            for (i, s) in streams.enumerated() {
                group.addTask {
                    let r = await drain(s, cap: budget)
                    return (i, r.tokens.count)
                }
            }
            for await r in group {
                print("  slot \(r.0): \(r.1) tokens")
            }
        }
        print("SCENARIO 2 DONE")
    }

    // Scenario 3: B=4 with mid-stream cancellation on two slots.
    // Matches osaurus's "client disconnected" pattern — HTTP Task
    // cancelled while BatchEngine is mid-decode. The surviving slots
    // must finish cleanly.
    do {
        print("\nSCENARIO 3 START: B=4 with cancel on 2 of 4")
        let engine = BatchEngine(context: ctx, maxBatchSize: 4)
        let params = GenerateParameters(
            maxTokens: max(budget, 96), temperature: 0, prefillStepSize: 512)
        let inputs = try await prepareInputs([
            "Describe water briefly.",
            "Describe fire briefly.",
            "Describe wind briefly.",
            "Describe earth briefly.",
        ])
        var ids: [BatchRequestID] = []
        var streams: [AsyncStream<BatchGeneration>] = []
        for input in inputs {
            nonisolated(unsafe) let send = input
            let (uuid, s) = await engine.submit(input: send, parameters: params)
            ids.append(uuid); streams.append(s)
        }
        let perSlotCap = max(budget, 96)
        await withTaskGroup(of: (Int, Int).self) { group in
            for (i, s) in streams.enumerated() {
                group.addTask {
                    let r = await drain(s, cap: perSlotCap)
                    return (i, r.tokens.count)
                }
            }
            let idsSnapshot = ids
            group.addTask {
                try? await Task.sleep(nanoseconds: 120_000_000)
                await engine.cancel(idsSnapshot[1])
                await engine.cancel(idsSnapshot[2])
                return (-1, 0)
            }
            for await r in group where r.0 >= 0 {
                print("  slot \(r.0): \(r.1) tokens")
            }
        }
        print("SCENARIO 3 DONE")
    }

    // Scenario 4: every slot maxTokens=1 — EOS-on-first-token ballpark.
    // Post-105ff8b this should finish cleanly with `.length` stops (or
    // `.stop` if first token happens to be EOS); pre-fix this was the
    // known force-unwrap.
    do {
        print("\nSCENARIO 4 START: B=4 maxTokens=1 each")
        let engine = BatchEngine(context: ctx, maxBatchSize: 4)
        let params = GenerateParameters(
            maxTokens: 1, temperature: 0, prefillStepSize: 512)
        let inputs = try await prepareInputs(["Hi", "Hello", "Hey", "Yo"])
        var streams: [AsyncStream<BatchGeneration>] = []
        for input in inputs {
            nonisolated(unsafe) let send = input
            let (_, s) = await engine.submit(input: send, parameters: params)
            streams.append(s)
        }
        await withTaskGroup(of: Int.self) { group in
            for s in streams {
                group.addTask {
                    let r = await drain(s, cap: 1)
                    return r.tokens.count
                }
            }
            var total = 0
            for await c in group { total += c }
            print("  total tokens across 4 slots: \(total)")
        }
        print("SCENARIO 4 DONE")
    }

    // Scenario 5: single-token prompt. Tokenizer returns one non-BOS id
    // → prefill with 1 token is a cold-start edge case for the hybrid
    // SSM path (no multi-token prefill; the Mamba scan runs on T=1).
    do {
        print("\nSCENARIO 5 START: single-token prompt")
        let engine = BatchEngine(context: ctx, maxBatchSize: 1)
        let params = GenerateParameters(
            maxTokens: budget, temperature: 0, prefillStepSize: 512)
        let inputs = try await prepareInputs(["a"])
        nonisolated(unsafe) let send = inputs[0]
        let (_, s) = await engine.submit(input: send, parameters: params)
        let r = await drain(s, cap: budget)
        print("SCENARIO 5 DONE: \(r.tokens.count) tokens, stop=\(r.stop.map{"\($0)"} ?? "nil")")
    }

    // Scenario 6: submit, then abandon the stream (consumer drops).
    // If BatchEngine relies on backpressure from the consumer to free
    // slot state, an abandoned stream could wedge or trap.
    do {
        print("\nSCENARIO 6 START: submit + drop consumer")
        let engine = BatchEngine(context: ctx, maxBatchSize: 2)
        let params = GenerateParameters(
            maxTokens: budget, temperature: 0, prefillStepSize: 512)
        let input6 = try await ctx.processor.prepare(
            input: UserInput(prompt: "Count to ten."))
        nonisolated(unsafe) let sendable6 = input6
        let (uuid6, s6) = await engine.submit(
            input: sendable6, parameters: params)
        _ = s6 // drop the stream on the floor
        try? await Task.sleep(nanoseconds: 300_000_000)
        await engine.cancel(uuid6)
        print("SCENARIO 6 DONE")
    }

    // Scenario 7: back-to-back identical prompts on one engine. Second
    // submit should hit the coordinator cache. Exercises SSM seed +
    // paged KV hit + potential disk-write race.
    do {
        print("\nSCENARIO 7 START: identical prompt twice back-to-back")
        let engine = BatchEngine(context: ctx, maxBatchSize: 1)
        let params = GenerateParameters(
            maxTokens: budget, temperature: 0, prefillStepSize: 512)
        for pass in 1...2 {
            let inputs = try await prepareInputs(["List three prime numbers."])
            nonisolated(unsafe) let send = inputs[0]
            let (_, s) = await engine.submit(input: send, parameters: params)
            let r = await drain(s, cap: budget)
            print("  pass \(pass): \(r.tokens.count) tokens, stop=\(r.stop.map{"\($0)"} ?? "nil")")
        }
        print("SCENARIO 7 DONE")
    }

    // Scenario 8: stop-string that matches the first plausible output.
    // `extraStopStrings = [" "]` — the first whitespace after the prompt
    // will trip the matcher and close the slot quickly. Exercises the
    // StopStringMatcher integration under B=4.
    do {
        print("\nSCENARIO 8 START: B=4 with aggressive stop-string")
        let engine = BatchEngine(context: ctx, maxBatchSize: 4)
        var params = GenerateParameters(
            maxTokens: budget, temperature: 0, prefillStepSize: 512)
        params.extraStopStrings = ["\n", "."]
        let inputs = try await prepareInputs([
            "Say hi.", "Say hello.", "Say hey.", "Say yo.",
        ])
        var streams: [AsyncStream<BatchGeneration>] = []
        for input in inputs {
            nonisolated(unsafe) let send = input
            let (_, s) = await engine.submit(input: send, parameters: params)
            streams.append(s)
        }
        await withTaskGroup(of: Int.self) { group in
            for s in streams {
                group.addTask {
                    let r = await drain(s, cap: budget)
                    return r.tokens.count
                }
            }
            var total = 0
            for await c in group { total += c }
            print("  total: \(total) tokens across 4 slots")
        }
        print("SCENARIO 8 DONE")
    }

    // Scenario 9: wildly different prompt lengths in the same batch.
    // Short prompts + a 500-token prompt → pad/mask alignment stress.
    do {
        print("\nSCENARIO 9 START: B=2 mixed short + long prompts")
        let engine = BatchEngine(context: ctx, maxBatchSize: 2)
        let params = GenerateParameters(
            maxTokens: budget, temperature: 0, prefillStepSize: 512)
        let longPrompt = String(repeating: "The quick brown fox jumps. ", count: 60)
        let inputs = try await prepareInputs(["Hi.", longPrompt])
        var streams: [(String, AsyncStream<BatchGeneration>)] = []
        for (label, input) in zip(["short", "long"], inputs) {
            nonisolated(unsafe) let send = input
            let (_, s) = await engine.submit(input: send, parameters: params)
            streams.append((label, s))
        }
        await withTaskGroup(of: (String, Int).self) { group in
            for (label, s) in streams {
                group.addTask {
                    let r = await drain(s, cap: budget)
                    return (label, r.tokens.count)
                }
            }
            for await r in group {
                print("  \(r.0): \(r.1) tokens")
            }
        }
        print("SCENARIO 9 DONE")
    }

    // Scenario 10: rapid sequential submits on a shared engine — the
    // osaurus server handles requests one at a time in the common case
    // where the client opens a new HTTP request per turn. Cache writes
    // from turn N overlap with turn N+1's submit — potential race on
    // the coordinator's disk write queue.
    do {
        print("\nSCENARIO 10 START: 5 rapid sequential submits")
        let engine = BatchEngine(context: ctx, maxBatchSize: 1)
        let params = GenerateParameters(
            maxTokens: budget, temperature: 0, prefillStepSize: 512)
        let inputs = try await prepareInputs([
            "One.", "Two.", "Three.", "Four.", "Five.",
        ])
        for (i, input) in inputs.enumerated() {
            nonisolated(unsafe) let send = input
            let (_, s) = await engine.submit(input: send, parameters: params)
            let r = await drain(s, cap: budget)
            print("  turn \(i+1): \(r.tokens.count) tokens")
        }
        print("SCENARIO 10 DONE")
    }

    print("\n=== BENCH_CRASH_FUZZ: all scenarios finished without crash ===")
}

// MARK: - Crash fuzz V2 — generate() pipeline, multi-turn (2026-04-23)
//
// Targets the NaiveStreamingDetokenizer + ReasoningParser +
// ToolCallProcessor + StopStringMatcher chain on the
// `BatchEngine.generate(input:parameters:)` path — the same path
// osaurus uses and the one that reproduced tpae's crash.
// Each scenario drives REAL chat-template rendering and expects the
// generate() stream to close cleanly. Printing `SCENARIO N START`
// before and `SCENARIO N DONE` after makes the last line before a
// crash name the culprit.

func runCrashFuzzV2(modelPath: String, maxNew: Int) async throws {
    let modelDir = URL(fileURLWithPath: modelPath)
    print("\n=== BENCH_CRASH_FUZZ_V2 — generate() pipeline stress ===")
    print("Loading with real HuggingFace tokenizer from \(modelPath) ...")
    let loadStart = CFAbsoluteTimeGetCurrent()
    let context = try await MLXLMCommon.loadModel(
        from: modelDir, using: #huggingFaceTokenizerLoader())
    print(String(format: "Load: %.2fs", CFAbsoluteTimeGetCurrent() - loadStart))
    print("Model: \(type(of: context.model))")
    nonisolated(unsafe) let ctx = context

    let budget = min(maxNew, 96)
    let engine = BatchEngine(context: ctx, maxBatchSize: 1)
    let params = GenerateParameters(
        maxTokens: budget, temperature: 0, prefillStepSize: 512)

    // Run a single prompt through `generate()` — the full production
    // pipeline (detokenizer → reasoning parser → tool processor →
    // stop-string matcher). Returns accumulated text chunks + whether
    // anything streamed at all. If any inner stage traps, we crash
    // here with the SCENARIO line as the breadcrumb.
    @Sendable func runGenerate(
        _ userInput: UserInput, label: String
    ) async throws -> (chunks: String, reasoning: String, toolCalls: Int) {
        let input = try await ctx.processor.prepare(input: userInput)
        nonisolated(unsafe) let send = input
        let stream = await engine.generate(input: send, parameters: params)
        var chunks = ""
        var reasoning = ""
        var tools = 0
        for await event in stream {
            switch event {
            case .chunk(let c): chunks += c
            case .reasoning(let r): reasoning += r
            case .toolCall: tools += 1
            case .info: break
            }
        }
        let preview = chunks.count > 80 ? String(chunks.prefix(80)) + "..." : chunks
        print("  \(label): chunks=\(chunks.count) reasoning=\(reasoning.count) tools=\(tools) -> \"\(preview)\"")
        return (chunks, reasoning, tools)
    }

    // Scenario 1: emoji-heavy output — forces byte-level BPE to
    // complete multi-byte graphemes across token boundaries. This is
    // the class that tripped the tpae crash pre-fix.
    do {
        print("\nSCENARIO 1 START: emoji-heavy output")
        _ = try await runGenerate(
            UserInput(prompt: "Write exactly five emojis separated by spaces. Only emojis, nothing else."),
            label: "emoji")
        print("SCENARIO 1 DONE")
    }

    // Scenario 2: Unicode-dense multilingual output — cleanUp
    // substitutions + grapheme-cluster edge cases compound here.
    do {
        print("\nSCENARIO 2 START: multilingual output")
        _ = try await runGenerate(
            UserInput(prompt: "Say \"hello\" in Japanese, Korean, Chinese, Arabic, Hindi."),
            label: "ml")
        print("SCENARIO 2 DONE")
    }

    // Scenario 3: punctuation-dense output — exercises the
    // cleanUpTokenizationSpaces substitutions specifically
    // (" ." → ".", " n't" → "n't", " 's" → "'s").
    do {
        print("\nSCENARIO 3 START: contractions + punctuation dense")
        _ = try await runGenerate(
            UserInput(prompt: "Write one sentence that uses: don't, won't, isn't, it's, she's, we've, they're, I'm. All in one sentence."),
            label: "punct")
        print("SCENARIO 3 DONE")
    }

    // Scenario 4: reasoning ON then OFF on the same engine. Verifies
    // the reasoning parser resets cleanly between turns AND that
    // special-token boundaries (Qwen 3.6 has `<|im_end|>`, channel
    // markers, etc.) don't trip the detokenizer.
    do {
        print("\nSCENARIO 4 START: reasoning ON, then OFF, on the same engine")
        let onPrompt = UserInput(prompt: "What is 7 * 8? Think step by step.")
        let offPrompt = UserInput(prompt: "What color is the sky? /no_think")
        _ = try await runGenerate(onPrompt, label: "ON")
        _ = try await runGenerate(offPrompt, label: "OFF")
        print("SCENARIO 4 DONE")
    }

    // Scenario 5: very short output that likely ends on an EOS-family
    // special token. This class is what originally inspired the
    // special-token collapse test case.
    do {
        print("\nSCENARIO 5 START: single-word answer")
        _ = try await runGenerate(
            UserInput(prompt: "Reply with exactly one word: yes or no. Is the sky blue?"),
            label: "oneword")
        print("SCENARIO 5 DONE")
    }

    // Scenario 6: `\u{fffd}`-prone content — ask for characters whose
    // UTF-8 encoding is likely to split across tokens. Incomplete
    // grapheme completion collapsing multi-replacement tails into a
    // single emoji is the third decode-shrinkage path.
    do {
        print("\nSCENARIO 6 START: UTF-8 multibyte continuation")
        _ = try await runGenerate(
            UserInput(prompt: "Write this exact string and nothing else: 你好世界 🌏 🚀 café naïve résumé"),
            label: "utf8")
        print("SCENARIO 6 DONE")
    }

    // Scenario 7: 10 rapid back-to-back turns on the same engine,
    // alternating short / emoji / code prompts. Exhausts cross-turn
    // cache reuse + re-entry into the generate() pipeline.
    do {
        print("\nSCENARIO 7 START: 10 rapid back-to-back turns")
        let prompts = [
            "Count to three.",
            "Three emojis: ",
            "```python\nprint('hi')\n```",
            "¿Hola, cómo estás?",
            "🚀🎉🔥",
            "42",
            "Say 'done'.",
            "一",
            ".",
            "!",
        ]
        for (i, p) in prompts.enumerated() {
            _ = try await runGenerate(
                UserInput(prompt: p),
                label: "t\(i+1)")
        }
        print("SCENARIO 7 DONE")
    }

    print("\n=== BENCH_CRASH_FUZZ_V2: all scenarios finished without crash ===")
}

// MARK: - Official final-pass multi-turn (2026-04-23)
//
// One-model harness designed to be looped over several models by the
// shell driver. Each scenario reports per-turn timing, tok/s, reasoning
// / chunk / tool-call counts, peak RSS in MiB, and content validation
// where applicable. All scenarios run through `engine.generate(...)`
// — the production path — so every turn exercises the full pipeline
// (detokenizer → reasoning parser → tool-call processor → stop-string
// matcher). Stats are printed in a compact single-line format per turn
// for easy shell post-processing.

/// Current resident set size in MiB, via `mach_task_basic_info`. Used
/// as a cheap peak-memory tracker across turns.
fileprivate func currentRSSMiB() -> Double {
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

/// Current physical footprint in MiB. This mirrors Activity Monitor's
/// process footprint better than RSS for mmap-backed model diagnostics.
fileprivate func currentPhysFootprintMiB() -> Double {
    #if canImport(Darwin)
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
    #else
    return -1
    #endif
}

func runOfficialMultiTurn(modelPath: String, maxNew: Int) async throws {
    let modelDir = URL(fileURLWithPath: modelPath)
    let modelName = modelDir.lastPathComponent
    print("\n=== BENCH_OFFICIAL — \(modelName) ===")
    let rssBefore = currentRSSMiB()
    print(String(format: "RSS before load: %.0f MiB", rssBefore))
    let loadStart = CFAbsoluteTimeGetCurrent()
    let context = try await loadBenchModelContext(from: modelDir)
    let loadSec = CFAbsoluteTimeGetCurrent() - loadStart
    let rssAfterLoad = currentRSSMiB()
    print(String(format: "Load: %.2fs  Model: %@  RSS +%.0f MiB -> %.0f MiB",
        loadSec, String(describing: type(of: context.model)),
        rssAfterLoad - rssBefore, rssAfterLoad))
    print("Tool format: \(context.configuration.toolCallFormat.map { "\($0)" } ?? "json (default)")")
    print("Reasoning stamp: \(context.configuration.reasoningParserName ?? "nil")")

    nonisolated(unsafe) let ctx = context
    let engine = BatchEngine(context: ctx, maxBatchSize: 1)
    let params = GenerateParameters(
        maxTokens: max(maxNew, 96), temperature: 0, prefillStepSize: 512)

    // Use reference boxes so the nested @Sendable closure can mutate
    // counts without tripping Swift 6's strict-concurrency capture
    // rules. Closures run sequentially on the caller's task; the box
    // isn't a concurrency tool, just a capture-mode tool.
    final class Stats: @unchecked Sendable {
        var peakRSS: Double
        var passCount = 0
        var failCount = 0
        init(peakRSS: Double) { self.peakRSS = peakRSS }
    }
    let stats = Stats(peakRSS: rssAfterLoad)

    // One turn through generate(). Captures timing, content, peak RSS.
    // Returns the text/reasoning/toolCall counts so scenario-specific
    // validation can run on them.
    func runTurn(
        label: String, prompt: String, thinking: Bool?,
        tools: [[String: any Sendable]]? = nil,
        validate: (_ text: String, _ reasoning: String, _ tools: Int) -> (ok: Bool, why: String)
    ) async throws {
        var userInput = UserInput(prompt: prompt)
        if let thinking {
            userInput.additionalContext = ["enable_thinking": thinking]
        }
        if let tools {
            userInput.tools = tools
        }
        let t0 = CFAbsoluteTimeGetCurrent()
        let input = try await ctx.processor.prepare(input: userInput)
        nonisolated(unsafe) let send = input
        let stream = await engine.generate(input: send, parameters: params)
        var text = ""
        var reasoning = ""
        var toolCalls = 0
        var chunks = 0
        var reasoningDeltas = 0
        var ttft: Double?
        for await event in stream {
            switch event {
            case .chunk(let c):
                if ttft == nil { ttft = CFAbsoluteTimeGetCurrent() - t0 }
                text += c; chunks += 1
            case .reasoning(let r):
                if ttft == nil { ttft = CFAbsoluteTimeGetCurrent() - t0 }
                reasoning += r; reasoningDeltas += 1
            case .toolCall: toolCalls += 1
            case .info: break
            }
        }
        let total = CFAbsoluteTimeGetCurrent() - t0
        let now = currentRSSMiB()
        if now > stats.peakRSS { stats.peakRSS = now }

        // Approximate decode tok/s as (chunks+reasoningDeltas)/total —
        // deltas are per-token in the generate() pipeline. Not exact
        // (detokenizer may buffer multi-byte chars across multiple
        // tokens, emitting fewer events than tokens), but close enough
        // for relative comparison across turns.
        let deltas = chunks + reasoningDeltas
        let tokps = total > 0 ? Double(deltas) / total : 0
        let ttftMs = Int((ttft ?? 0) * 1000)
        let preview = text.isEmpty ? "[empty visible; reasoning chars=\(reasoning.count)]" : text
        let short = preview.count > 100 ? String(preview.prefix(100)) + "…" : preview

        let v = validate(text, reasoning, toolCalls)
        let status = v.ok ? "PASS" : "FAIL"
        if v.ok { stats.passCount += 1 } else { stats.failCount += 1 }

        print(String(format:
            "  [%@] %@  ttft=%4dms total=%5.2fs tokps=%5.1f chunks=%4d reasoning=%4d tools=%d rss=%.0fMiB -> %@%@",
            status, label, ttftMs, total, tokps, chunks, reasoning.count, toolCalls, now,
            v.ok ? "" : "WHY=\(v.why) ",
            "\"\(short.replacingOccurrences(of: "\n", with: "\\n"))\""))
    }

    // S1: Math problem — reasoning ON. Validate answer text contains "4".
    try await runTurn(
        label: "S1 reasoning=ON  math 7+8-11", prompt: "Compute 7 + 8 - 11. Respond with just the number.",
        thinking: true
    ) { text, reasoning, _ in
        // The answer must be visible content. Reasoning-only output is a
        // red row, not a fallback pass.
        if text.contains("4") {
            return (true, "")
        }
        return (false, "answer not found")
    }

    // S2: Same prompt — verifies repeat-turn reproducibility. Hybrid cache
    // topology is validated separately by runBatchEngineCacheHit so this matrix
    // does not overclaim cache reuse from timing alone.
    try await runTurn(
        label: "S2 reasoning=ON  math 7+8-11 (repeat prompt)",
        prompt: "Compute 7 + 8 - 11. Respond with just the number.",
        thinking: true
    ) { text, reasoning, _ in
        if text.contains("4") {
            return (true, "")
        }
        return (false, "answer not found on repeat prompt")
    }

    // S3: thinking=OFF — should produce mostly .chunk events, minimal
    // .reasoning. Ask for a source-reference-satisfiable answer.
    try await runTurn(
        label: "S3 reasoning=OFF math 2+2",
        prompt: "Compute 2 + 2. Respond with just the number.",
        thinking: false
    ) { text, reasoning, _ in
        if text.contains("4") {
            return (true, "")
        }
        return (false, "answer not found")
    }

    // S4: Multi-tool-call prompt. Two tools; ask the model to call
    // both. Different formats (xmlFunction, Mistral inline, harmony,
    // Pythonic) have different match criteria, so we accept as PASS
    // if the generation completed without crash and the pipeline
    // produced ≥1 tool call OR ≥1 content chunk (some models prefer
    // to inline the answer rather than call tools when the question
    // is answerable from prior knowledge).
    // Nested dictionaries need explicit `[String: any Sendable]`
    // annotations so nested literals don't widen to `Any`.
    let weatherParams: [String: any Sendable] = [
        "type": "object",
        "properties": [
            "city": ["type": "string", "description": "City name"] as [String: any Sendable],
        ] as [String: any Sendable],
        "required": ["city"],
    ]
    let weatherFn: [String: any Sendable] = [
        "name": "get_weather",
        "description": "Get current weather for a city.",
        "parameters": weatherParams,
    ]
    let weatherTool: [String: any Sendable] = [
        "type": "function",
        "function": weatherFn,
    ]
    let timeParams: [String: any Sendable] = [
        "type": "object",
        "properties": [
            "timezone": ["type": "string", "description": "IANA zone"] as [String: any Sendable],
        ] as [String: any Sendable],
        "required": ["timezone"],
    ]
    let timeFn: [String: any Sendable] = [
        "name": "get_time",
        "description": "Get current time in a timezone.",
        "parameters": timeParams,
    ]
    let timeTool: [String: any Sendable] = [
        "type": "function",
        "function": timeFn,
    ]
    try await runTurn(
        label: "S4 multi-tool-call",
        prompt: "I need BOTH the current weather in Tokyo AND the current time in Asia/Tokyo. Call the appropriate tools.",
        thinking: true, tools: [weatherTool, timeTool]
    ) { text, reasoning, tools in
        // Accept any of: ≥1 tool call extracted OR the model produced
        // substantive output discussing both entities (fallback path
        // when the tokenizer emits tool-call syntax in a format our
        // parser doesn't recognize for this model). Primary metric:
        // no crash + non-empty output.
        if tools >= 1 { return (true, "extracted \(tools) tool call(s)") }
        if text.count >= 10 {
            return (true, "no tool call but \(text.count) visible chars emitted")
        }
        return (false, "empty output, no tool call")
    }

    // S5: UTF-8 multilingual stress — the shrinkage-prone class that
    // originally tripped tpae's crash. Validate expected UTF-8 survives
    // instead of accepting any non-empty response.
    try await runTurn(
        label: "S5 utf8 inclusion",
        prompt: "Write one short sentence that includes both words café and 你好.",
        thinking: false
    ) { text, reasoning, _ in
        if !text.lowercased().contains("café") || !text.contains("你好") {
            return (false, "missing expected UTF-8 words")
        }
        return (true, "")
    }

    // S6: 5-turn rapid sequential chat. Each turn is a plain prompt
    // (no tools). Tests cross-turn cache reuse + repeated enter/exit
    // of the generate() pipeline + detokenizer stability under churn.
    let rapidPrompts = [
        "Name a country.",
        "Name a color.",
        "Name a fruit.",
        "Name an animal.",
        "Name a day of the week.",
    ]
    for (i, p) in rapidPrompts.enumerated() {
        try await runTurn(
            label: "S6.\(i+1) rapid",
            prompt: p, thinking: false
        ) { text, reasoning, _ in
            if text.isEmpty { return (false, "empty visible output") }
            return (true, "")
        }
    }

    print(String(format:
        "\n=== BENCH_OFFICIAL summary: model=%@ pass=%d fail=%d peakRSS=%.0fMiB loadSec=%.2f ===",
        modelName, stats.passCount, stats.failCount, stats.peakRSS, loadSec))

    if stats.failCount > 0 {
        throw NSError(domain: "BENCH_OFFICIAL", code: 1, userInfo: [
            NSLocalizedDescriptionKey: "\(stats.failCount) scenario(s) failed validation",
        ])
    }
}

// MARK: - Production exhaustive matrix (2026-04-23)
//
// The most comprehensive real-model test we run. For a single model,
// exercises every production path vmlx-swift-lm ships: tool-call round
// trips, reasoning ON→OFF alternation, L2 disk cache hit, SSM state
// re-derive on hybrid models, TurboQuant runtime sidecar on JANGTQ
// bundles. Every scenario validates content — a pass requires the
// right answer or the right tool_call schema, not just "stream
// didn't crash".
//
// Environment knobs:
//   BENCH_PROD_CACHE_DIR   L2 disk cache root (default
//                          /tmp/vmlx-prod-cache/<model-basename>)
//   BENCH_MAX_TOKENS       per-turn decode cap (default 64)
//   BENCH_PROD_SKIP_TOOLS  skip S3 tool round-trip for hostile formats
//                          (e.g. pure-JSON models during triage)
//
// Exit code 0 = every scenario PASS with correct content. Non-zero
// = at least one scenario flunked its content predicate; the
// summary line names which.


// MARK: - Production exhaustive matrix (2026-04-23, v2 — prompt-based)
//
// Prompt-string based to avoid the chat-template apply hang from v1.
// Uses the same text-history accumulation VLBench.runMixedMultiTurn
// relies on. Validates content per scenario.
//
// Coverage per invocation:
//   S1  reasoning=ON  math (validate "4" in output)
//   S2  SAME prompt as S1 — paged cache hit, TTFT drops
//   S3  reasoning=OFF math (validate "4")
//   S4  reasoning ON→OFF→ON alternation within one engine
//   S5  UTF-8 multilingual inclusion (validate expected words)
//   S6  SSM-seed: hybrid-SSM-only models — identical prefix
//       continuation should reuse paged KV + SSM companion cache
//   S7  L2 disk: rerun with fresh process via env override
//       (scripted by shell loop, not here)

func runProdMatrix(modelPath: String, maxNew: Int) async throws {
    let modelDir = URL(fileURLWithPath: modelPath)
    let modelName = modelDir.lastPathComponent
    let env = ProcessInfo.processInfo.environment
    let budget = max(maxNew, 96)

    print("\n=== BENCH_PROD — \(modelName) ===")
    let cacheRoot = env["BENCH_PROD_CACHE_DIR"] ??
        "/tmp/vmlx-prod-cache/\(modelName)"
    try? FileManager.default.createDirectory(
        atPath: cacheRoot, withIntermediateDirectories: true)
    print("Cache dir: \(cacheRoot)")

    let nativeMTPDepth = env["BENCH_PROD_NATIVE_MTP_DEPTH"].flatMap(Int.init)
    let rss0 = currentRSSMiB()
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
        context = try await MLXLMCommon.loadModel(
            from: modelDir, using: #huggingFaceTokenizerLoader())
    }
    let loadSec = CFAbsoluteTimeGetCurrent() - loadStart
    let rss1 = currentRSSMiB()
    print(String(format: "Load: %.2fs  Model: %@  RSS +%.0f MiB",
        loadSec, String(describing: type(of: context.model)),
        rss1 - rss0))
    print("Tool format: \(context.configuration.toolCallFormat.map{"\($0)"} ?? "json")")
    print("Reasoning stamp: \(context.configuration.reasoningParserName ?? "nil")")

    // BatchEngine with the standard default CacheCoordinator wiring
    // (nil passes through to the per-prompt cache — this matches the
    // production path the osaurus server uses). If BENCH_PROD_COORD=1
    // is set, attach an explicit L2-disk coordinator.
    nonisolated(unsafe) let ctx = context
    let coordinator: CacheCoordinator?
    if (env["BENCH_PROD_COORD"] ?? "0") == "1" {
        let cfg = CacheCoordinatorConfig(
            usePagedCache: true, enableDiskCache: true,
            pagedBlockSize: 64, maxCacheBlocks: 512,
            diskCacheMaxGB: 4.0,
            diskCacheDir: URL(fileURLWithPath: cacheRoot),
            ssmMaxEntries: 32, modelKey: modelName)
        let coord = CacheCoordinator(config: cfg)
        if env["BENCH_PROD_CACHE_HYBRID"] == "1" {
            coord.setHybrid(true)
        }
        coordinator = coord
        print("Coordinator: enabled (L2 disk at \(cacheRoot), hybrid=\(coord.isHybrid))")
    } else {
        coordinator = nil
    }
    let engine = BatchEngine(
        context: ctx, maxBatchSize: 1, cacheCoordinator: coordinator)
    let greedy = (env["BENCH_PROD_GREEDY"] ?? "0") == "1"
    let randomSeed = env["BENCH_PROD_SEED"].flatMap(UInt64.init)
    var params: GenerateParameters
    if greedy {
        params = GenerateParameters(
            maxTokens: budget, temperature: 0, topP: 1, topK: 0,
            minP: 0, randomSeed: randomSeed, repetitionPenalty: nil,
            prefillStepSize: 512)
    } else {
        let fallback = GenerateParameters(
            maxTokens: budget, randomSeed: randomSeed, prefillStepSize: 512)
        params = GenerateParameters(
            generationConfig: context.configuration.generationDefaults,
            fallback: fallback)
        // The live gate owns decode budget. Bundle generation_config drives
        // sampling, but a huge max_new_tokens must not make matrix rows unbounded.
        params.maxTokens = budget
        params.randomSeed = randomSeed
    }
    if let nativeMTPDepth {
        params.draftStrategy = .nativeMTP(depth: nativeMTPDepth)
    }
    print(String(format:
        "Sampling: mode=%@ mtpDepth=%@ maxTokens=%d temp=%.3f topP=%.3f topK=%d minP=%.3f rep=%@ seed=%@",
        greedy ? "explicit-greedy" : "bundle-defaults",
        nativeMTPDepth.map(String.init) ?? "off",
        params.maxTokens ?? -1,
        Double(params.temperature),
        Double(params.topP),
        params.topK,
        Double(params.minP),
        params.repetitionPenalty.map { String(format: "%.3f", Double($0)) } ?? "nil",
        params.randomSeed.map(String.init) ?? "nil"))

    func promptTail(thinking: Bool) async throws -> String {
        var probe = UserInput(prompt: "Probe reasoning toggle.")
        probe.additionalContext = ["enable_thinking": thinking]
        let input = try await ctx.processor.prepare(input: probe)
        let ids = input.text.tokens.reshaped(-1).asArray(Int32.self).map { Int($0) }
        return ctx.tokenizer.decode(
            tokenIds: Array(ids.suffix(96)), skipSpecialTokens: false)
    }

    func hasReasoningPromptRail(_ text: String) -> Bool {
        text.contains("<think>")
            || text.contains("[THINK]")
            || text.contains("<|channel>")
            || text.contains("<|channel|>analysis")
    }

    let thinkingOnPromptTail = try await promptTail(thinking: true)
    let thinkingOffPromptTail = try await promptTail(thinking: false)
    let reasoningPromptToggleActive =
        thinkingOnPromptTail != thinkingOffPromptTail
        && hasReasoningPromptRail(thinkingOnPromptTail)
    print(
        "Reasoning prompt toggle: \(reasoningPromptToggleActive ? "active" : "not-template-active")"
            + " onTail=\(thinkingOnPromptTail.suffix(80).debugDescription)"
            + " offTail=\(thinkingOffPromptTail.suffix(80).debugDescription)")

    final class Stats: @unchecked Sendable {
        var peakRSS: Double; var pass = 0; var fail = 0
        var ttftByLabel: [String: Int] = [:]
        init(peakRSS: Double) { self.peakRSS = peakRSS }
    }
    let stats = Stats(peakRSS: rss1)

    struct TurnResult {
        var text = ""
        var reasoning = ""
        var tools = 0
        var ttftMs = 0
        var totalSec = 0.0
        var tokps = 0.0
        var promptSec = 0.0
        var genTokens = 0
        var genSec = 0.0
        var stop = "unknown"
    }

    func runTurn(
        label: String, prompt: String, thinking: Bool?,
        extraTools: [[String: any Sendable]]? = nil,
        validate: (TurnResult) -> (ok: Bool, why: String)
    ) async throws {
        var userInput = UserInput(prompt: prompt)
        if let thinking {
            userInput.additionalContext = ["enable_thinking": thinking]
        }
        if let extraTools {
            userInput.tools = extraTools
        }
        let t0 = CFAbsoluteTimeGetCurrent()
        let input = try await ctx.processor.prepare(input: userInput)
        nonisolated(unsafe) let send = input
        let stream = await engine.generate(input: send, parameters: params)
        var r = TurnResult()
        var ttft: Double?
        var deltas = 0
        for await ev in stream {
            switch ev {
            case .chunk(let c):
                if ttft == nil { ttft = CFAbsoluteTimeGetCurrent() - t0 }
                r.text += c; deltas += 1
            case .reasoning(let rs):
                if ttft == nil { ttft = CFAbsoluteTimeGetCurrent() - t0 }
                r.reasoning += rs; deltas += 1
            case .toolCall: r.tools += 1
            case .info(let info):
                r.promptSec = info.promptTime
                r.genTokens = info.generationTokenCount
                r.genSec = info.generateTime
                switch info.stopReason {
                case .stop:
                    r.stop = "stop"
                case .length:
                    r.stop = "length"
                case .cancelled:
                    r.stop = "cancelled"
                }
            }
        }
        r.totalSec = CFAbsoluteTimeGetCurrent() - t0
        r.tokps = r.genSec > 0 ? Double(r.genTokens) / r.genSec :
            (r.totalSec > 0 ? Double(deltas) / r.totalSec : 0)
        r.ttftMs = Int((ttft ?? 0) * 1000)
        stats.ttftByLabel[label] = r.ttftMs
        let now = currentRSSMiB()
        if now > stats.peakRSS { stats.peakRSS = now }

        let v = validate(r)
        let status = v.ok ? "PASS" : "FAIL"
        if v.ok { stats.pass += 1 } else { stats.fail += 1 }
        let visible = r.text.trimmingCharacters(in: .whitespacesAndNewlines)
        let preview = visible.isEmpty
            ? "[empty visible; reasoning chars=\(r.reasoning.count)]"
            : visible
        let short = preview.count > 120 ? String(preview.prefix(120)) + "…" : preview
        print(String(format:
            "  [%@] %@  ttft=%4dms prompt=%4.0fms total=%5.2fs genTokens=%d tokps=%5.1f stop=%@ chunks=%4d reasoning=%4d tools=%d rss=%.0fMiB %@-> \"%@\"",
            status, label, r.ttftMs, r.promptSec * 1000,
            r.totalSec, r.genTokens, r.tokps, r.stop,
            r.text.count, r.reasoning.count, r.tools, now,
            v.ok ? "" : "WHY=\(v.why) ",
            short.replacingOccurrences(of: "\n", with: "\\n")))
    }

    func visibleReasoningMarkerLeak(_ text: String) -> String? {
        [
            "<think>", "</think>",
            "[THINK]", "[/THINK]",
            "<|channel>", "<channel|>",
            "<|channel|>analysis", "<|channel|>final",
            "<|message|>",
        ].first { text.contains($0) }
    }

    func requireNoVisibleReasoningMarkers(_ r: TurnResult) -> (Bool, String) {
        if let marker = visibleReasoningMarkerLeak(r.text) {
            return (false, "visible chunk leaked reasoning marker \(marker)")
        }
        return (true, "")
    }

    func requireVisibleAnswer(_ r: TurnResult, contains expected: String) -> (Bool, String) {
        let noLeak = requireNoVisibleReasoningMarkers(r)
        if !noLeak.0 { return noLeak }
        if reasoningPromptToggleActive &&
            r.reasoning.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        {
            return (false, "reasoning ON produced no .reasoning deltas")
        }
        let visible = r.text.trimmingCharacters(in: .whitespacesAndNewlines)
        if visible.isEmpty {
            return (false, "answer only appeared in reasoning / no visible chunk")
        }
        if !r.text.contains(expected) {
            if r.reasoning.contains(expected) {
                return (false, "'\(expected)' only appeared in reasoning")
            }
            return (false, "no '\(expected)' in visible output")
        }
        return (true, "")
    }

    func requireReasoningOffVisible(_ r: TurnResult) -> (Bool, String) {
        let noLeak = requireNoVisibleReasoningMarkers(r)
        if !noLeak.0 { return noLeak }
        if !r.reasoning.isEmpty {
            return (false, "reasoning emitted while enable_thinking=false")
        }
        if r.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return (false, "empty visible output")
        }
        return (true, "")
    }

    // ──────────────── S1  reasoning=ON math ────────────────
    try await runTurn(
        label: "S1 think=ON math(7+8-11)",
        prompt: "Compute 7 + 8 - 11. Respond with just the number.",
        thinking: true
    ) { r in
        requireVisibleAnswer(r, contains: "4")
    }

    // ──────────────── S2  same prompt → cache hit ────────────────
    try await runTurn(
        label: "S2 cache-hit think=ON same prompt",
        prompt: "Compute 7 + 8 - 11. Respond with just the number.",
        thinking: true
    ) { r in
        let contentOK = requireVisibleAnswer(r, contains: "4")
        if !contentOK.0 { return contentOK }
        let ttftS1 = stats.ttftByLabel["S1 think=ON math(7+8-11)"] ?? 99999
        // Soft check: warn on no speedup, don't fail
        if r.ttftMs >= Int(Double(ttftS1) * 0.95) {
            return (true, "warn: no TTFT speedup vs S1 (\(ttftS1)→\(r.ttftMs)ms)")
        }
        return (true, "")
    }

    // ──────────────── S3  reasoning=OFF math ────────────────
    try await runTurn(
        label: "S3 think=OFF math(2+2)",
        prompt: "Compute 2 + 2. Respond with just the number.",
        thinking: false
    ) { r in
        let visible = requireReasoningOffVisible(r)
        if !visible.0 { return visible }
        if !r.text.contains("4") {
            return (false, "no '4' in visible output")
        }
        return (true, "")
    }

    // ──────────────── S4  reasoning ON→OFF→ON alternation ────────────────
    try await runTurn(
        label: "S4.1 flip-back think=ON math(12+3)",
        prompt: "Compute 12 + 3. Respond with just the number.",
        thinking: true
    ) { r in
        requireVisibleAnswer(r, contains: "15")
    }
    try await runTurn(
        label: "S4.2 flip think=OFF name",
        prompt: "Question: Name the planet Mars. Answer with Mars only:",
        thinking: false
    ) { r in
        let visible = requireReasoningOffVisible(r)
        if !visible.0 { return visible }
        if !r.text.lowercased().contains("mars") {
            return (false, "no 'Mars' in visible output")
        }
        return (true, "")
    }
    try await runTurn(
        label: "S4.3 flip-back think=ON math(5 times 4)",
        prompt: "What is 5 times 4? Answer with only the number.",
        thinking: true
    ) { r in
        requireVisibleAnswer(r, contains: "20")
    }

    // ──────────────── S5  UTF-8 inclusion ────────────────
    try await runTurn(
        label: "S5 utf8 inclusion",
        prompt: "Write one short sentence that includes the exact literal strings café and 你好. Do not translate either string.",
        thinking: false
    ) { r in
        let visible = requireReasoningOffVisible(r)
        if !visible.0 { return visible }
        if !r.text.lowercased().contains("café") || !r.text.contains("你好") {
            return (false, "missing expected UTF-8 words in visible output")
        }
        return (true, "")
    }

    print(String(format:
        "\n=== BENCH_PROD summary: model=%@ pass=%d fail=%d peakRSS=%.0fMiB loadSec=%.2f ===",
        modelName, stats.pass, stats.fail, stats.peakRSS, loadSec))
    if let snapshot = coordinator?.snapshotStats() {
        let paged = snapshot.pagedStats.map {
            "hits=\($0.cacheHits),misses=\($0.cacheMisses),allocated=\($0.allocatedBlocks),free=\($0.freeBlocks),evictions=\($0.evictions)"
        } ?? "disabled"
        let disk = snapshot.diskStats.map {
            "hits=\($0.hits),misses=\($0.misses),stores=\($0.stores),maxBytes=\($0.maxSizeBytes)"
        } ?? "disabled"
        let ssm = snapshot.ssmStats
        print(
            "PROD_CACHE_STATS hybrid=\(snapshot.isHybrid) pagedIncompatible=\(snapshot.isPagedIncompatible) paged{\(paged)} disk{\(disk)} ssm{hits=\(ssm.hits),misses=\(ssm.misses),reDerives=\(ssm.reDerives)}"
        )
    }
    if stats.fail > 0 {
        throw NSError(domain: "BENCH_PROD", code: 1, userInfo: [
            NSLocalizedDescriptionKey: "\(stats.fail) scenario(s) failed validation",
        ])
    }
}
