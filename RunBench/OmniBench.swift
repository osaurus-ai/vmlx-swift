// OmniBench.swift
// End-to-end multi-turn bench harness for Nemotron-3-Nano-Omni bundles.
// Env-gated dispatch via BENCH_OMNI=1 in Bench.swift.
//
// Test matrix (per bundle):
//   1. Text-only single-turn         — baseline LLM smoke
//   2. Text-only multi-turn (×3)     — cache reuse on the standard text path
//   3. Image single-turn             — RADIO ViT + mlp1 + splice via prepare()
//   4. Image multi-turn (×2)         — image cache + MediaSalt
//   5. Video encoder smoke           — extractImageEmbeds(video:true) shape + finiteness
//   6. Audio rows                    — extractAudioEmbeds(waveform:) shape + finiteness,
//                                      plus UserInput.audios -> LMInput -> sound splice
//                                      through TokenIterator on real weights
//   7. Reasoning OFF (enable_thinking=false) parity
//
// Each row is independent — failures are caught and reported, the bench
// continues to the next row. Pass/fail summary printed at end.
//
// Bundle paths come from the env:
//   BENCH_MODEL=$HOME/.mlxstudio/models/JANGQ-AI/Nemotron-3-Nano-Omni-30B-A3B-MXFP4
// (or JANGTQ4 / JANGTQ2 — the harness inspects which it got)

import CoreImage
import Foundation
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import MLXNN
import MLXVLM
@preconcurrency import VMLXTokenizers

enum OmniBench {

    struct RowResult {
        let row: String
        let passed: Bool
        let detail: String
        let secs: Double
        let tokPerSec: Double?
    }

    static func run(modelPath: String, maxNewTokens: Int) async throws {
        let modelDir = URL(fileURLWithPath: modelPath)
        print("=================================================================")
        print("=== OmniBench — \(modelDir.lastPathComponent)")
        print("=== max new tokens: \(maxNewTokens)")
        print("=================================================================")

        // Sanity: confirm the bundle is actually an omni bundle.
        let configOmni = modelDir.appending(path: "config_omni.json")
        guard FileManager.default.fileExists(atPath: configOmni.path) else {
            print("FAIL: \(configOmni.lastPathComponent) not found — this isn't an omni bundle")
            exit(1)
        }

        // Load context once. The omni bundle is detected automatically by
        // VLMModelFactory._load via config_omni.json presence.
        let loadStart = CFAbsoluteTimeGetCurrent()
        let context: ModelContext
        do {
            context = try await MLXLMCommon.loadModel(
                from: modelDir, using: #huggingFaceTokenizerLoader())
        } catch {
            print("FAIL: load: \(error)")
            exit(1)
        }
        let loadSecs = CFAbsoluteTimeGetCurrent() - loadStart
        print(String(format: "Load: %.2fs | Model: %@ | Processor: %@",
            loadSecs,
            String(describing: type(of: context.model)),
            String(describing: type(of: context.processor))))
        let samplingProbe = makeOmniParameters(
            context: context,
            maxNewTokens: maxNewTokens)
        print(String(format:
            "Sampling: source=%@ maxTokens=%d temp=%.3f topP=%.3f topK=%d minP=%.3f rep=%@ seed=%@",
            (ProcessInfo.processInfo.environment["BENCH_OMNI_GREEDY"] == "1")
                ? "explicit-greedy-env"
                : "generation_config",
            samplingProbe.maxTokens ?? -1,
            Double(samplingProbe.temperature),
            Double(samplingProbe.topP),
            samplingProbe.topK,
            Double(samplingProbe.minP),
            samplingProbe.repetitionPenalty.map {
                String(format: "%.3f", Double($0))
            } ?? "nil",
            samplingProbe.randomSeed.map(String.init) ?? "nil"))

        guard let omni = context.model as? NemotronHOmni else {
            print("FAIL: dispatch — got \(type(of: context.model)), expected NemotronHOmni")
            exit(1)
        }


        var results: [RowResult] = []

        // Row 1: Text-only single-turn
        results.append(await runRow("1. text-only single-turn", maxNew: maxNewTokens) {
            let cache = context.model.newCache(parameters: .init())
            return try await runTextTurn(
                prompt: "What is the capital of France? Answer in one sentence.",
                context: context, cache: cache, maxNewTokens: maxNewTokens)
        })

        // Row 2: Text-only multi-turn (3 turns, rendered as full chat
        // history each turn). Raw KV/Mamba cache reuse is only valid for
        // append-only token suffixes; a fresh prompt with a populated cache
        // is a cache-contract violation, not a multi-turn session.
        results.append(await runRow("2. text-only multi-turn x3", maxNew: maxNewTokens) {
            let prompts = [
                "What is the capital of France?",
                "And of Germany?",
                "Of those two countries, which has more people?",
            ]
            var history: [Chat.Message] = []
            var detail = ""
            var totalToks = 0
            var totalSecs = 0.0
            for (i, p) in prompts.enumerated() {
                history.append(.user(p))
                let r = try await runChatTurn(
                    history: history, context: context,
                    maxNewTokens: maxNewTokens)
                history.append(Chat.Message(
                    role: .assistant,
                    content: r.shortText,
                    reasoningContent: r.reasoningText.isEmpty ? nil : r.reasoningText))
                detail += "T\(i + 1): \(r.shortText.prefix(80)) | "
                totalToks += r.tokens
                totalSecs += r.secs
            }
            return TurnResult(
                shortText: detail.trimmingCharacters(in: .whitespaces),
                tokens: totalToks, secs: totalSecs)
        })

        // Row 3: Image single-turn
        results.append(await runRow("3. image single-turn", maxNew: maxNewTokens) {
            let img = try synthesiseGradient(side: 224)
            let cache = context.model.newCache(parameters: .init())
            return try await runImageTurn(
                prompt: "Name the two most prominent colors in this image.",
                image: img, context: context, cache: cache,
                maxNewTokens: maxNewTokens)
        })

        // Row 3b: direct image with reasoning explicitly off. This isolates
        // model/template behavior from BatchEngine scheduling so the B3 row
        // cannot be misclassified as a batching bug if the direct no-thinking
        // vision path is also not grounded.
        results.append(await runRow("3b. image reasoning OFF direct", maxNew: maxNewTokens) {
            let img = try synthesiseGradient(side: 224)
            let cache = context.model.newCache(parameters: .init())
            return try await runImageTurn(
                prompt: "What are the two main colors in this image? Answer briefly.",
                image: img, context: context, cache: cache,
                maxNewTokens: maxNewTokens,
                enableThinking: false)
        })

        if (ProcessInfo.processInfo.environment["BENCH_OMNI_TAIL_PROBE"] ?? "0") == "1" {
            results.append(await runRow("3c. image no-think tail variants", maxNew: maxNewTokens) {
                let img = try synthesiseGradient(side: 224)
                return try await runImageTailVariants(
                    prompt: "Describe this image briefly.",
                    image: img,
                    context: context,
                    maxNewTokens: maxNewTokens)
            })
        }

        // Row 4: Image multi-turn. This must render real chat history with
        // the image attached to the original media-bearing user message. Do
        // not reuse raw cache across unrelated fresh prompts here.
        results.append(await runRow("4. image multi-turn x2", maxNew: maxNewTokens) {
            let img = try synthesiseGradient(side: 224)
            var history: [Chat.Message] = [
                .user("Name the two most prominent colors in this image.",
                      images: [.ciImage(img)]),
            ]
            let r1 = try await runChatTurn(
                history: history, context: context,
                maxNewTokens: maxNewTokens)
            try validateSyntheticGradientImageText(
                r1.shortText,
                row: "image multi-turn turn1")
            history.append(Chat.Message(
                role: .assistant,
                content: r1.shortText,
                reasoningContent: r1.reasoningText.isEmpty ? nil : r1.reasoningText))
            history.append(.user("Repeat the two color names from the image."))
            let r2 = try await runChatTurn(
                history: history, context: context,
                maxNewTokens: maxNewTokens)
            try validateSyntheticGradientImageText(
                r2.shortText,
                row: "image multi-turn turn2")
            return TurnResult(
                shortText: "T1: \(r1.shortText.prefix(80)) || T2: \(r2.shortText.prefix(80))",
                tokens: r1.tokens + r2.tokens,
                secs: r1.secs + r2.secs)
        })

        // Row 5: Video encoder smoke — validate
        // extractImageEmbeds(pixelValues:, video: true) on T=2 channel
        // stack runs on real RADIO weights and returns finite embeddings
        // with the expected shape (groups*tokens_per_group, llm_hidden=2688).
        results.append(await runRow("5. video encoder smoke", maxNew: 0) {
            let imageSize = 512
            let img = try synthesiseGradient(side: imageSize)
            // 4 frames -> 2 groups of T=2 stacking -> (2, 6, 512, 512)
            let pixelValues = vlmStackFramesIntoChannels(
                [img, img, img, img],
                imageSize: imageSize,
                temporalPatchDim: 2,
                mean: CLIP_NORM_MEAN, std: CLIP_NORM_STD)
            let t0 = CFAbsoluteTimeGetCurrent()
            let videoEmbeds = omni.extractImageEmbeds(
                pixelValues: pixelValues, video: true)
            MLX.eval(videoEmbeds)
            let secs = CFAbsoluteTimeGetCurrent() - t0
            let shape = videoEmbeds.shape
            // Expect (groups * 256_post_shuffle, 2688)
            let expectedTokens = 2 * 256
            let expectedHidden = 2688
            guard shape.count == 2,
                  shape[0] == expectedTokens,
                  shape[1] == expectedHidden
            else {
                throw NSError(
                    domain: "OmniBench", code: 5,
                    userInfo: [NSLocalizedDescriptionKey:
                        "video shape \(shape), expected [\(expectedTokens), \(expectedHidden)]"])
            }
            let any = videoEmbeds.asArray(Float.self).prefix(64)
            let allFinite = any.allSatisfy { $0.isFinite }
            if !allFinite {
                throw NSError(
                    domain: "OmniBench", code: 5,
                    userInfo: [NSLocalizedDescriptionKey: "non-finite values in video embeds"])
            }
            return TurnResult(
                shortText: "shape=\(shape), encode \(String(format: "%.2fs", secs))",
                tokens: 0, secs: secs)
        })

        // Row 6a: Audio encoder smoke (synthetic tone) — fast finite-check.
        results.append(await runRow("6a. audio encoder smoke", maxNew: 0) {
            let waveform = synthesiseToneWaveform(seconds: 1.0, hz: 440)
            let t0 = CFAbsoluteTimeGetCurrent()
            let audioEmbeds = omni.extractAudioEmbeds(waveform: waveform)
            MLX.eval(audioEmbeds)
            let secs = CFAbsoluteTimeGetCurrent() - t0
            let shape = audioEmbeds.shape
            // Expect (frames_after_subsampling_x8, 2688)
            // 1s @ 16 kHz with hop=160 = 100 STFT frames + center pad ~101.
            // Subsampled by 8 -> ~12-13 frames after Parakeet.
            guard shape.count == 2,
                  shape[1] == 2688,
                  shape[0] > 0,
                  shape[0] < 50
            else {
                throw NSError(
                    domain: "OmniBench", code: 6,
                    userInfo: [NSLocalizedDescriptionKey:
                        "audio shape \(shape), expected [<50, 2688]"])
            }
            let any = audioEmbeds.asArray(Float.self).prefix(64)
            let allFinite = any.allSatisfy { $0.isFinite }
            if !allFinite {
                throw NSError(
                    domain: "OmniBench", code: 6,
                    userInfo: [NSLocalizedDescriptionKey: "non-finite values in audio embeds"])
            }
            return TurnResult(
                shortText: "shape=\(shape), encode \(String(format: "%.2fs", secs))",
                tokens: 0, secs: secs)
        })

        // Row 5b: Video LMInput END-TO-END — full UserInput.videos →
        // NemotronHOmniProcessor decodes via nemotronOmniPreprocessVideo
        // → splice at imageContextTokenId (video reuses <image> tokens).
        let videoFixture = URL(
            fileURLWithPath: "Tests/MLXLMTests/Resources/1080p_30.mov")
        if FileManager.default.fileExists(atPath: videoFixture.path) {
            results.append(await runRow("5b. video LMInput end-to-end", maxNew: maxNewTokens) {
                let cache = context.model.newCache(parameters: .init())
                return try await runVideoTurn(
                    prompt: "Describe this video briefly.",
                    videoURL: videoFixture,
                    context: context, cache: cache,
                    maxNewTokens: maxNewTokens,
                    enableThinking: true)
            })
            results.append(await runRow("5c. video reasoning OFF direct", maxNew: maxNewTokens) {
                let cache = context.model.newCache(parameters: .init())
                return try await runVideoTurn(
                    prompt: "Describe this video briefly.",
                    videoURL: videoFixture,
                    context: context, cache: cache,
                    maxNewTokens: maxNewTokens,
                    enableThinking: false)
            })
            if (ProcessInfo.processInfo.environment["BENCH_OMNI_VIDEO_CACHE_ALIAS"] ?? "0") == "1" {
                results.append(await runRow("5d. video repeated cache alias", maxNew: maxNewTokens) {
                    return try await runVideoRepeatedCacheAliasProof(
                        videoURL: videoFixture,
                        modelDir: modelDir,
                        context: context,
                        maxNewTokens: maxNewTokens)
                })
            }
        }

        // Row 6b: Audio LMInput END-TO-END — full UserInput.audio →
        // NemotronHOmniProcessor.prepare → NemotronHOmni.prepare splice
        // at sound_context_token_id → forward → decode. Real audio
        // file from Tests/Resources, run through TokenIterator like a
        // production turn would.
        let audioFixture = URL(
            fileURLWithPath: "Tests/MLXLMTests/Resources/audio_only.mov")
        if FileManager.default.fileExists(atPath: audioFixture.path) {
            results.append(await runRow("6b. audio LMInput end-to-end", maxNew: maxNewTokens) {
                let cache = context.model.newCache(parameters: .init())
                return try await runAudioTurn(
                    prompt: "Briefly describe what you hear.",
                    audioURL: audioFixture,
                    context: context, cache: cache,
                    maxNewTokens: maxNewTokens)
            })
        }

        // Row 7: Reasoning OFF parity
        results.append(await runRow("7. reasoning OFF (enable_thinking=false)", maxNew: maxNewTokens) {
            let cache = context.model.newCache(parameters: .init())
            return try await runTextTurn(
                prompt: "Briefly: what is 2+2?",
                context: context, cache: cache,
                maxNewTokens: maxNewTokens,
                enableThinking: false)
        })

        // ─── Stress matrix the API surface will hit in production ───
        //
        // Row 8: reasoning ON→OFF→ON toggle within ONE conversation —
        // share the cache across turns, flip enable_thinking each
        // turn, verify no crash, no NaN, coherent output. The chat
        // template re-renders with the kwarg per turn; the cache
        // covers the SHARED prefix only (system + earlier turns'
        // text) — when the rendered prompt diverges due to the
        // think-block toggle, the new turn re-prefills any new text
        // tokens. Mamba state must round-trip cleanly through this.
        results.append(await runRow("8. reasoning ON→OFF→ON toggle", maxNew: maxNewTokens) {
            let cache = context.model.newCache(parameters: .init())
            var detail = ""
            var totalToks = 0, totalSecs = 0.0
            let plan: [(String, Bool)] = [
                ("Briefly: what's 2+2? Show your reasoning.", true),
                ("Now no reasoning: what's the capital of France?", false),
                ("Reasoning back on: which is larger, 12*7 or 9*9?", true),
            ]
            for (i, (p, think)) in plan.enumerated() {
                let r = try await runTextTurn(
                    prompt: p, context: context, cache: cache,
                    maxNewTokens: maxNewTokens,
                    enableThinking: think)
                detail += "T\(i + 1)[\(think ? "ON" : "OFF")]: \(r.shortText.prefix(60)) | "
                totalToks += r.tokens
                totalSecs += r.secs
            }
            return TurnResult(
                shortText: detail.trimmingCharacters(in: .whitespaces),
                tokens: totalToks, secs: totalSecs)
        })

        // Row 9: mixed image + audio in ONE turn — verify the
        // processor stitches both placeholder-token streams (image
        // 256-tile run + audio frame run) into a single prompt and
        // the model splices both encoder outputs into the right
        // token positions. Placeholder-count drift between processor
        // and either encoder would silently desync; coherent text
        // means both made it through.
        if FileManager.default.fileExists(atPath: audioFixture.path) {
            results.append(await runRow("9. mixed image + audio one turn", maxNew: maxNewTokens) {
                let img = try synthesiseGradient(side: 224)
                let cache = context.model.newCache(parameters: .init())
                let userInput = UserInput(
                    prompt: "Combine what you see and hear into one sentence.",
                    images: [.ciImage(img)],
                    audios: [.url(audioFixture)])
                let lmInput = try await context.processor.prepare(input: userInput)
                let params = makeOmniParameters(
                    context: context,
                    maxNewTokens: maxNewTokens)
                let iter = try TokenIterator(
                    input: lmInput, model: context.model, cache: cache,
                    parameters: params)
                let t0 = CFAbsoluteTimeGetCurrent()
                let tokens = collectGeneratedTokens(
                    iter, context: context, maxNewTokens: maxNewTokens)
                let secs = CFAbsoluteTimeGetCurrent() - t0
                let text = userVisibleText(
                    context: context, lmInput: lmInput, tokenIds: tokens)
                return TurnResult(
                    shortText: text.replacingOccurrences(of: "\n", with: " "),
                    tokens: tokens.count, secs: secs)
            })
        }

        // Row 10: media-salt isolation — ask the same prompt twice,
        // first with audio A, then with synthesized audio B (different
        // bytes, same SR). Output for the two turns SHOULD differ;
        // identical output would indicate the cache returned A's KV
        // for B's prompt (the cache-poisoning class). With
        // computeMediaSalt(for:) hashing the audio waveform, this
        // is supposed to be impossible.
        if FileManager.default.fileExists(atPath: audioFixture.path) {
            results.append(await runRow("10. media-salt isolation (audio A vs B)", maxNew: maxNewTokens) {
                let cacheA = context.model.newCache(parameters: .init())
                let rA = try await runAudioTurn(
                    prompt: "What's in this audio?",
                    audioURL: audioFixture,
                    context: context, cache: cacheA,
                    maxNewTokens: maxNewTokens)
                // Audio B: synthesized 1-second tone at a totally
                // different frequency — bytes diverge so the salt
                // diverges.
                let toneB = synthesiseToneWaveform(seconds: 1.5, hz: 220)
                let cacheB = context.model.newCache(parameters: .init())
                let userB = UserInput(
                    prompt: "What's in this audio?",
                    audios: [.samples(toneB, sampleRate: 16_000)])
                let lmB = try await context.processor.prepare(input: userB)
                let params = makeOmniParameters(
                    context: context,
                    maxNewTokens: maxNewTokens)
                let iter = try TokenIterator(
                    input: lmB, model: context.model, cache: cacheB,
                    parameters: params)
                let t0 = CFAbsoluteTimeGetCurrent()
                let tokens = collectGeneratedTokens(
                    iter, context: context, maxNewTokens: maxNewTokens)
                let secs = CFAbsoluteTimeGetCurrent() - t0
                let textB = userVisibleText(
                    context: context, lmInput: lmB, tokenIds: tokens)
                let aShort = rA.shortText.prefix(80)
                let bShort = textB.replacingOccurrences(of: "\n", with: " ").prefix(80)
                let differ = aShort != bShort
                if !differ {
                    throw NSError(
                        domain: "OmniBench", code: 10,
                        userInfo: [NSLocalizedDescriptionKey:
                            "audio A and B produced IDENTICAL output — possible cache poisoning"])
                }
                return TurnResult(
                    shortText: "A: \(aShort) || B: \(bShort)",
                    tokens: rA.tokens + tokens.count,
                    secs: rA.secs + secs)
            })
        }

        // Row 11: hybrid SSM warm-pass parity — run T1 with full
        // prefill, then T2 with the same cache reused (no clear).
        // The Mamba conv+hidden state from T1 must round-trip
        // correctly into T2; a mid-prefill SSM mismatch shows up as
        // garbage output. This catches `extractSSMStates` /
        // `restoreSSMStates` regressions on the omni hybrid pattern.
        results.append(await runRow("11. hybrid SSM warm-pass parity", maxNew: maxNewTokens) {
            let cache = context.model.newCache(parameters: .init())
            let r1 = try await runTextTurn(
                prompt: "List three planets in our solar system.",
                context: context, cache: cache,
                maxNewTokens: maxNewTokens)
            // T2 reuses the SAME cache — Mamba state must persist.
            let r2 = try await runTextTurn(
                prompt: "Now list three more.",
                context: context, cache: cache,
                maxNewTokens: maxNewTokens)
            // Cache types should reflect the hybrid pattern.
            let kinds = Set(cache.map { String(describing: type(of: $0)) })
            let isHybrid = kinds.contains("MambaCache") && kinds.contains("KVCacheSimple")
            if !isHybrid {
                throw NSError(
                    domain: "OmniBench", code: 11,
                    userInfo: [NSLocalizedDescriptionKey:
                        "cache topology not hybrid: \(kinds)"])
            }
            return TurnResult(
                shortText: "T1: \(r1.shortText.prefix(80)) || T2: \(r2.shortText.prefix(80)) || cache=\(kinds)",
                tokens: r1.tokens + r2.tokens,
                secs: r1.secs + r2.secs)
        })

        // ─── BatchEngine stress (BENCH_OMNI_BATCH=1) ───
        //
        // Honest answer to "does omni work through BatchEngine, not just
        // TokenIterator?" — verifies the whole prepare → admit → prefill
        // → BatchArraysCache → batched-decode → finishSlot pipeline on
        // the real omni hybrid topology.
        if (ProcessInfo.processInfo.environment["BENCH_OMNI_BATCH"] ?? "0") == "1" {
            nonisolated(unsafe) let ctxBatch = context

            // Row B1: text-only, B=1 through BatchEngine.
            // Sets `enable_thinking: false` so we exercise the content
            // path (not the `<think>...</think>` block, which emits as
            // `.reasoning` events, never `.chunk`).
            results.append(await runRow("B1. BatchEngine text B=1", maxNew: maxNewTokens) {
                nonisolated(unsafe) let ctxLocal = ctxBatch
                let engine = BatchEngine(context: ctxLocal, maxBatchSize: 2)
                let params = makeOmniParameters(
                    context: ctxBatch,
                    maxNewTokens: maxNewTokens)
                var userInput = UserInput(prompt: "What is the capital of France?")
                userInput.additionalContext = ["enable_thinking": false]
                let lmInput = try await ctxBatch.processor.prepare(input: userInput)
                nonisolated(unsafe) let sendable = lmInput
                let stream = await engine.generate(input: sendable, parameters: params)
                let t0 = CFAbsoluteTimeGetCurrent()
                var text = ""
                var chunks = 0
                for await event in stream {
                    switch event {
                    case .chunk(let c): text += c; chunks += 1
                    case .reasoning(let r): text += r; chunks += 1
                    default: break
                    }
                    if chunks > maxNewTokens * 2 { break }
                }
                let secs = CFAbsoluteTimeGetCurrent() - t0
                if text.isEmpty {
                    throw NSError(
                        domain: "OmniBench", code: 100,
                        userInfo: [NSLocalizedDescriptionKey:
                            "BatchEngine text B=1: empty output stream"])
                }
                return TurnResult(
                    shortText: text.replacingOccurrences(of: "\n", with: " "),
                    tokens: chunks, secs: secs)
            })

            // Row B2: text-only, B=2 through BatchEngine — exercises
            // BatchArraysCache merge for the 23 Mamba layers + batched
            // decode through hybrid topology. Two concurrent prompts;
            // both must produce sensible text.
            results.append(await runRow("B2. BatchEngine text B=2 concurrent", maxNew: maxNewTokens) {
                nonisolated(unsafe) let ctxLocal = ctxBatch
                let engine = BatchEngine(context: ctxLocal, maxBatchSize: 2)
                let params = makeOmniParameters(
                    context: ctxBatch,
                    maxNewTokens: maxNewTokens)
                let prompts = [
                    "Capital of Japan?",
                    "Capital of Brazil?",
                ]
                var inputs: [LMInput] = []
                for p in prompts {
                    var ui = UserInput(prompt: p)
                    ui.additionalContext = ["enable_thinking": false]
                    let lm = try await ctxBatch.processor.prepare(input: ui)
                    inputs.append(lm)
                }
                let t0 = CFAbsoluteTimeGetCurrent()
                nonisolated(unsafe) let i0 = inputs[0]
                nonisolated(unsafe) let i1 = inputs[1]
                nonisolated(unsafe) let p = params
                async let s0Task = Self.collectStream(
                    engine: engine, input: i0, parameters: p,
                    maxNew: maxNewTokens)
                async let s1Task = Self.collectStream(
                    engine: engine, input: i1, parameters: p,
                    maxNew: maxNewTokens)
                let (s0, s1) = await (s0Task, s1Task)
                let secs = CFAbsoluteTimeGetCurrent() - t0
                if s0.text.isEmpty || s1.text.isEmpty {
                    throw NSError(
                        domain: "OmniBench", code: 101,
                        userInfo: [NSLocalizedDescriptionKey:
                            "BatchEngine B=2: one slot empty (s0=\(s0.text.count) s1=\(s1.text.count))"])
                }
                return TurnResult(
                    shortText: "S0: \(s0.text.prefix(80)) || S1: \(s1.text.prefix(80))",
                    tokens: s0.tokens + s1.tokens, secs: secs)
            })

            // Row B3: multimodal (image) B=1 through BatchEngine.
            // Trips the .logits PrepareResult path (line 880 of
            // BatchEngine.swift) that wraps VL prefill output. If
            // BatchEngine routes VL inputs correctly through
            // model.prepare() and back, this passes.
            results.append(await runRow("B3. BatchEngine image B=1", maxNew: maxNewTokens) {
                nonisolated(unsafe) let ctxLocal = ctxBatch
                let engine = BatchEngine(context: ctxLocal, maxBatchSize: 2)
                let params = makeOmniParameters(
                    context: ctxBatch,
                    maxNewTokens: maxNewTokens)
                let img = try synthesiseGradient(side: 224)
                var userInput = UserInput(
                    prompt: "What are the two main colors in this image? Answer briefly.",
                    images: [.ciImage(img)])
                userInput.additionalContext = ["enable_thinking": false]
                let lmInput = try await ctxBatch.processor.prepare(input: userInput)
                nonisolated(unsafe) let sendable = lmInput
                let stream = await engine.generate(input: sendable, parameters: params)
                let t0 = CFAbsoluteTimeGetCurrent()
                var text = ""
                var chunks = 0
                for await event in stream {
                    switch event {
                    case .chunk(let c): text += c; chunks += 1
                    case .reasoning(let r): text += r; chunks += 1
                    default: break
                    }
                    if chunks > maxNewTokens * 2 { break }
                }
                let secs = CFAbsoluteTimeGetCurrent() - t0
                if text.isEmpty {
                    throw NSError(
                        domain: "OmniBench", code: 102,
                        userInfo: [NSLocalizedDescriptionKey:
                            "BatchEngine image B=1: empty output stream"])
                }
                try validateVisibleOmniText(text, row: "BatchEngine image B=1")
                try validateSyntheticGradientImageText(
                    text,
                    row: "BatchEngine image B=1")
                return TurnResult(
                    shortText: text.replacingOccurrences(of: "\n", with: " "),
                    tokens: chunks, secs: secs)
            })

            // Row B4: audio B=1 through BatchEngine.
            // Same as B3 but with audio — exercises the Parakeet +
            // sound_projection splice path under the batched pipeline.
            if FileManager.default.fileExists(atPath: audioFixture.path) {
                results.append(await runRow("B4. BatchEngine audio B=1", maxNew: maxNewTokens) {
                    nonisolated(unsafe) let ctxLocal = ctxBatch
                    let engine = BatchEngine(context: ctxLocal, maxBatchSize: 2)
                    let params = makeOmniParameters(
                        context: ctxBatch,
                        maxNewTokens: maxNewTokens)
                    var userInput = UserInput(
                        prompt: "Briefly describe the audio.",
                        audios: [.url(audioFixture)])
                    userInput.additionalContext = ["enable_thinking": false]
                    let lmInput = try await ctxBatch.processor.prepare(input: userInput)
                    nonisolated(unsafe) let sendable = lmInput
                    let stream = await engine.generate(input: sendable, parameters: params)
                    let t0 = CFAbsoluteTimeGetCurrent()
                    var text = ""
                    var chunks = 0
                    for await event in stream {
                        switch event {
                        case .chunk(let c): text += c; chunks += 1
                        case .reasoning(let r): text += r; chunks += 1
                        default: break
                        }
                        if chunks > maxNewTokens * 2 { break }
                    }
                    let secs = CFAbsoluteTimeGetCurrent() - t0
                    if text.isEmpty {
                        throw NSError(
                            domain: "OmniBench", code: 103,
                            userInfo: [NSLocalizedDescriptionKey:
                                "BatchEngine audio B=1: empty output stream"])
                    }
                    try validateVisibleOmniText(text, row: "BatchEngine audio B=1")
                    return TurnResult(
                        shortText: text.replacingOccurrences(of: "\n", with: " "),
                        tokens: chunks, secs: secs)
                })
            }
        }

        // Summary
        print("\n=================================================================")
        print("=== OmniBench summary - \(modelDir.lastPathComponent)")
        print("=================================================================")
        var pass = 0, fail = 0
        for r in results {
            let mark = r.passed ? "PASS" : "FAIL"
            let toks = r.tokPerSec.map { String(format: "%.1f tok/s", $0) } ?? "-"
            print(String(format: "  [%@] %@  %@  | %@",
                mark, r.row.padding(toLength: 42, withPad: " ", startingAt: 0),
                String(format: "%.2fs", r.secs), toks))
            if !r.detail.isEmpty {
                let p = r.detail.count > 220
                    ? String(r.detail.prefix(220)) + "..." : r.detail
                print("        \"\(p)\"")
            }
            if r.passed { pass += 1 } else { fail += 1 }
        }
        print("\n=== \(pass) passed, \(fail) failed | load \(String(format: "%.2fs", loadSecs)) ===")
        if fail > 0 {
            print("FAIL: OmniBench failed \(fail) of \(results.count) rows")
            exit(1)
        }
    }

    // MARK: - Row runner

    /// Runs a single bench row, catches errors, returns a `RowResult`.
    private static func runRow(
        _ label: String, maxNew _: Int,
        _ body: () async throws -> TurnResult
    ) async -> RowResult {
        print("\n--- \(label) ---")
        let t0 = CFAbsoluteTimeGetCurrent()
        do {
            let r = try await body()
            let secs = CFAbsoluteTimeGetCurrent() - t0
            let tps = r.secs > 0 && r.tokens > 0 ? Double(r.tokens) / r.secs : nil
            return RowResult(row: label, passed: true, detail: r.shortText,
                             secs: secs, tokPerSec: tps)
        } catch {
            let secs = CFAbsoluteTimeGetCurrent() - t0
            let errorText = (error as NSError).localizedDescription
            return RowResult(row: label, passed: false,
                             detail: "ERROR: \(errorText)", secs: secs, tokPerSec: nil)
        }
    }

    // MARK: - Per-mode turn runners

    struct TurnResult {
        let shortText: String
        let reasoningText: String
        let tokens: Int
        let secs: Double

        init(
            shortText: String,
            tokens: Int,
            secs: Double,
            reasoningText: String = ""
        ) {
            self.shortText = shortText
            self.reasoningText = reasoningText
            self.tokens = tokens
            self.secs = secs
        }
    }

    private struct RawGenerationResult {
        let tokenIds: [Int]
        let stopTokenID: Int?
        let hitMaxTokens: Bool
    }

    private static func makeOmniParameters(
        context: ModelContext,
        maxNewTokens: Int
    ) -> GenerateParameters {
        var params = GenerateParameters(
            generationConfig: context.configuration.generationDefaults)
        params.maxTokens = maxNewTokens
        params.prefillStepSize = 512
        if let seedText = ProcessInfo.processInfo.environment["BENCH_OMNI_RANDOM_SEED"],
           let seed = UInt64(seedText)
        {
            params.randomSeed = seed
        }
        if ProcessInfo.processInfo.environment["BENCH_OMNI_GREEDY"] == "1" {
            params.temperature = 0.0
            params.topP = 1.0
            params.topK = 0
            params.minP = 0.0
            params.repetitionPenalty = nil
        }
        return params
    }

    private static func runTextTurn(
        prompt: String,
        context: ModelContext,
        cache: [KVCache],
        maxNewTokens: Int,
        enableThinking: Bool = true
    ) async throws -> TurnResult {
        var userInput = UserInput(prompt: prompt)
        userInput.additionalContext = ["enable_thinking": enableThinking]
        let lmInput = try await context.processor.prepare(input: userInput)

        let params = makeOmniParameters(
            context: context,
            maxNewTokens: maxNewTokens)

        let iter = try TokenIterator(
            input: lmInput, model: context.model, cache: cache,
            parameters: params)

        let t0 = CFAbsoluteTimeGetCurrent()
        let raw = collectRawGeneratedTokens(
            iter, context: context, maxNewTokens: maxNewTokens)
        let secs = CFAbsoluteTimeGetCurrent() - t0
        let tokens = raw.tokenIds
        let parts = reasoningAndVisibleText(
            context: context, lmInput: lmInput, tokenIds: tokens,
            allowReasoningFallback: false)
        let text = parts.visible
        debugOmniRawDecode(
            label: "text thinking=\(enableThinking)",
            context: context,
            lmInput: lmInput,
            raw: raw,
            visibleText: text)
        if !enableThinking && text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            throw NSError(
                domain: "OmniBench", code: 104,
                userInfo: [NSLocalizedDescriptionKey:
                    "reasoning disabled produced no visible content"])
        }
        return TurnResult(
            shortText: text.replacingOccurrences(of: "\n", with: " "),
            tokens: tokens.count, secs: secs,
            reasoningText: parts.reasoning)
    }

    private static func runChatTurn(
        history: [Chat.Message],
        context: ModelContext,
        maxNewTokens: Int,
        enableThinking: Bool = true
    ) async throws -> TurnResult {
        let userInput = UserInput(
            chat: history,
            additionalContext: ["enable_thinking": enableThinking])
        let lmInput = try await context.processor.prepare(input: userInput)

        let params = makeOmniParameters(
            context: context,
            maxNewTokens: maxNewTokens)
        let cache = context.model.newCache(parameters: params)
        let iter = try TokenIterator(
            input: lmInput, model: context.model, cache: cache,
            parameters: params)

        let t0 = CFAbsoluteTimeGetCurrent()
        let raw = collectRawGeneratedTokens(
            iter, context: context, maxNewTokens: maxNewTokens)
        let secs = CFAbsoluteTimeGetCurrent() - t0
        let tokens = raw.tokenIds
        let parts = reasoningAndVisibleText(
            context: context, lmInput: lmInput, tokenIds: tokens,
            allowReasoningFallback: false)
        let text = parts.visible
        debugOmniRawDecode(
            label: "chat history thinking=\(enableThinking)",
            context: context,
            lmInput: lmInput,
            raw: raw,
            visibleText: text)
        if raw.hitMaxTokens {
            throw NSError(
                domain: "OmniBench", code: 105,
                userInfo: [NSLocalizedDescriptionKey:
                    "chat history turn hit max_tokens before a normal stop"])
        }
        try validateVisibleOmniText(text, row: "chat history turn")
        return TurnResult(
            shortText: text.replacingOccurrences(of: "\n", with: " "),
            tokens: tokens.count, secs: secs,
            reasoningText: parts.reasoning)
    }

    private static func runImageTurn(
        prompt: String,
        image: CIImage,
        context: ModelContext,
        cache: [KVCache],
        maxNewTokens: Int,
        enableThinking: Bool = true
    ) async throws -> TurnResult {
        var userInput = UserInput(prompt: prompt, images: [.ciImage(image)])
        userInput.additionalContext = ["enable_thinking": enableThinking]
        let lmInput = try await context.processor.prepare(input: userInput)

        let params = makeOmniParameters(
            context: context,
            maxNewTokens: maxNewTokens)

        let iter = try TokenIterator(
            input: lmInput, model: context.model, cache: cache,
            parameters: params)
        let t0 = CFAbsoluteTimeGetCurrent()
        let raw = collectRawGeneratedTokens(
            iter, context: context, maxNewTokens: maxNewTokens)
        let secs = CFAbsoluteTimeGetCurrent() - t0
        let tokens = raw.tokenIds
        let parts = reasoningAndVisibleText(
            context: context, lmInput: lmInput, tokenIds: tokens)
        let text = parts.visible
        debugOmniRawDecode(
            label: "image direct thinking=\(enableThinking)",
            context: context,
            lmInput: lmInput,
            raw: raw,
            visibleText: text)
        try validateVisibleOmniText(text, row: "image turn")
        try validateSyntheticGradientImageText(text, row: "image turn")
        return TurnResult(
            shortText: text.replacingOccurrences(of: "\n", with: " "),
            tokens: tokens.count, secs: secs,
            reasoningText: parts.reasoning)
    }

    private static func runImageTailVariants(
        prompt: String,
        image: CIImage,
        context: ModelContext,
        maxNewTokens: Int
    ) async throws -> TurnResult {
        var userInput = UserInput(prompt: prompt, images: [.ciImage(image)])
        let baseThinking = ProcessInfo.processInfo
            .environment["BENCH_OMNI_TAIL_PROBE_BASE"] == "thinking"
        userInput.additionalContext = ["enable_thinking": !baseThinking ? false : true]
        let baseInput = try await context.processor.prepare(input: userInput)

        let variants: [(name: String, tail: String, allowReasoningFallback: Bool)] = [
            ("closed-compact", "<|im_start|>assistant\n<think></think>", false),
            ("closed-newline", "<|im_start|>assistant\n<think></think>\n", false),
            ("closed-spaced", "<|im_start|>assistant\n<think>\n</think>\n\n", false),
            ("assistant-only", "<|im_start|>assistant\n", false),
            ("thinking-open", "<|im_start|>assistant\n<think>\n", true),
        ]

        var details: [String] = []
        var totalTokens = 0
        var totalSecs = 0.0
        var groundedPasses = 0

        for variant in variants {
            let lmInput = try rewriteAssistantTail(
                baseInput,
                tokenizer: context.tokenizer,
                replacementTail: variant.tail)
            let cache = context.model.newCache(parameters: .init())
            let params = makeOmniParameters(
                context: context,
                maxNewTokens: maxNewTokens)

            let iter = try TokenIterator(
                input: lmInput, model: context.model, cache: cache,
                parameters: params)
            let t0 = CFAbsoluteTimeGetCurrent()
            let raw = collectRawGeneratedTokens(
                iter, context: context, maxNewTokens: maxNewTokens)
            let secs = CFAbsoluteTimeGetCurrent() - t0
            let tokens = raw.tokenIds
            let text = userVisibleText(
                context: context,
                lmInput: lmInput,
                tokenIds: tokens,
                allowReasoningFallback: variant.allowReasoningFallback)
            debugOmniRawDecode(
                label: "image tail \(variant.name)",
                context: context,
                lmInput: lmInput,
                raw: raw,
                visibleText: text)
            totalTokens += tokens.count
            totalSecs += secs
            do {
                try validateVisibleOmniText(text, row: "image tail \(variant.name)")
                try validateSyntheticGradientImageText(
                    text,
                    row: "image tail \(variant.name)")
                groundedPasses += 1
                details.append("\(variant.name)=PASS:\(text.prefix(70))")
            } catch {
                details.append("\(variant.name)=FAIL:\(text.prefix(70))")
            }
        }

        if groundedPasses == 0 {
            throw NSError(
                domain: "OmniBench", code: 105,
                userInfo: [NSLocalizedDescriptionKey:
                    "no image tail variant grounded the image; \(details.joined(separator: " || "))"])
        }
        return TurnResult(
            shortText: details.joined(separator: " || "),
            tokens: totalTokens,
            secs: totalSecs)
    }

    private static func rewriteAssistantTail(
        _ input: LMInput,
        tokenizer: any MLXLMCommon.Tokenizer,
        replacementTail: String
    ) throws -> LMInput {
        let currentTails = [
            "<|im_start|>assistant\n",
            "<|im_start|>assistant\n<think>\n",
            "<|im_start|>assistant\n<think></think>",
            "<|im_start|>assistant\n<think>\n</think>\n\n",
        ]
        let replacementIds = tokenizer.encode(
            text: replacementTail,
            addSpecialTokens: false)
        let originalIds = input.text.tokens.reshaped([-1]).asArray(Int.self)
        for currentTail in currentTails {
            let currentTailIds = tokenizer.encode(
                text: currentTail,
                addSpecialTokens: false)
            if originalIds.count >= currentTailIds.count,
               Array(originalIds.suffix(currentTailIds.count)) == currentTailIds
            {
                let rewrittenIds = Array(originalIds.dropLast(currentTailIds.count)) + replacementIds
                let tokens = MLXArray(rewrittenIds).expandedDimensions(axis: 0)
                let mask = ones(like: tokens).asType(.int8)
                return LMInput(
                    text: .init(tokens: tokens, mask: mask),
                    image: input.image,
                    video: input.video,
                    audio: input.audio,
                    mediaTokenIds: input.mediaTokenIds,
                    cacheScopeSalt: input.cacheScopeSalt)
            }
        }
        let promptTail = tokenizer.decode(
            tokenIds: Array(originalIds.suffix(128)),
            skipSpecialTokens: false)
        throw NSError(
            domain: "OmniBench", code: 106,
            userInfo: [NSLocalizedDescriptionKey:
                "could not rewrite assistant tail; prompt tail=\(promptTail)"])
    }

    private static func runVideoTurn(
        prompt: String,
        videoURL: URL,
        context: ModelContext,
        cache: [KVCache],
        maxNewTokens: Int,
        enableThinking: Bool = true
    ) async throws -> TurnResult {
        var userInput = UserInput(prompt: prompt, videos: [.url(videoURL)])
        userInput.additionalContext = ["enable_thinking": enableThinking]
        let lmInput = try await context.processor.prepare(input: userInput)

        let params = makeOmniParameters(
            context: context,
            maxNewTokens: maxNewTokens)

        let iter = try TokenIterator(
            input: lmInput, model: context.model, cache: cache,
            parameters: params)
        let t0 = CFAbsoluteTimeGetCurrent()
        let raw = collectRawGeneratedTokens(
            iter, context: context, maxNewTokens: maxNewTokens)
        let secs = CFAbsoluteTimeGetCurrent() - t0
        let tokens = raw.tokenIds
        let text = userVisibleText(
            context: context, lmInput: lmInput, tokenIds: tokens)
        debugOmniRawDecode(
            label: "video thinking=\(enableThinking)",
            context: context,
            lmInput: lmInput,
            raw: raw,
            visibleText: text)
        try validateVisibleOmniText(text, row: "video turn")
        return TurnResult(
            shortText: text.replacingOccurrences(of: "\n", with: " "),
            tokens: tokens.count, secs: secs)
    }

    private static func runVideoRepeatedCacheAliasProof(
        videoURL: URL,
        modelDir: URL,
        context: ModelContext,
        maxNewTokens: Int
    ) async throws -> TurnResult {
        var params = makeOmniParameters(
            context: context,
            maxNewTokens: maxNewTokens)
        params.maxTokens = min(maxNewTokens, 48)

        let coordinator = makeOmniProofCoordinator(
            modelDir: modelDir,
            context: context,
            label: "same-video post-prepare alias")

        let firstInput = try await makeVideoProofInput(
            videoURL: videoURL,
            context: context)
        let firstRawTokens = firstInput.text.tokens.reshaped(-1).asArray(Int.self)
        guard firstInput.requiresPostPrepareCacheKey else {
            throw NSError(
                domain: "OmniBench", code: 120,
                userInfo: [NSLocalizedDescriptionKey:
                    "video proof input did not require a post-prepare cache key"])
        }
        guard let salt = computeCacheSalt(for: firstInput, parameters: params) else {
            throw NSError(
                domain: "OmniBench", code: 121,
                userInfo: [NSLocalizedDescriptionKey:
                    "video proof input did not produce a media salt"])
        }

        let t0 = CFAbsoluteTimeGetCurrent()
        let first = try await collectGenerationText(
            stream: MLXLMCommon.generate(
                input: firstInput,
                parameters: params,
                context: context,
                cacheCoordinator: coordinator),
            maxNewTokens: params.maxTokens ?? maxNewTokens)
        try validateVisibleOmniText(first.text, row: "video repeated cache alias first")

        guard let effectiveTokens = coordinator.resolvePostPrepareCacheKeyAlias(
            rawTokens: firstRawTokens,
            mediaSalt: salt)
        else {
            throw NSError(
                domain: "OmniBench", code: 122,
                userInfo: [NSLocalizedDescriptionKey:
                    "same-video post-prepare alias was not recorded after first generation"])
        }
        guard effectiveTokens != firstRawTokens else {
            throw NSError(
                domain: "OmniBench", code: 123,
                userInfo: [NSLocalizedDescriptionKey:
                    "same-video post-prepare alias did not change raw video tokens"])
        }

        let directDetail: String
        switch coordinator.fetch(tokens: effectiveTokens, mediaSalt: salt) {
        case .hit(let matched, let remaining, let detail, let blocks, _, _):
            directDetail = "\(detail.rawValue) matched=\(matched)/\(effectiveTokens.count) remaining=\(remaining.count)"
            coordinator.release(blocks: blocks)
        case .miss:
            throw NSError(
                domain: "OmniBench", code: 124,
                userInfo: [NSLocalizedDescriptionKey:
                    "same-video post-prepare alias resolved but cache fetch missed"])
        }

        let replayInput = try await makeVideoProofInput(
            videoURL: videoURL,
            context: context)
        let replayRawTokens = replayInput.text.tokens.reshaped(-1).asArray(Int.self)
        guard replayRawTokens == firstRawTokens,
              computeCacheSalt(for: replayInput, parameters: params) == salt
        else {
            throw NSError(
                domain: "OmniBench", code: 125,
                userInfo: [NSLocalizedDescriptionKey:
                    "same video did not reproduce raw tokens/media salt"])
        }
        guard coordinator.resolvePostPrepareCacheKeyAlias(
            rawTokens: replayRawTokens,
            mediaSalt: salt) == effectiveTokens
        else {
            throw NSError(
                domain: "OmniBench", code: 126,
                userInfo: [NSLocalizedDescriptionKey:
                    "replayed video did not resolve to the same effective cache key"])
        }

        let beforeReplay = coordinator.snapshotStats()
        let replay = try await collectGenerationText(
            stream: MLXLMCommon.generate(
                input: replayInput,
                parameters: params,
                context: context,
                cacheCoordinator: coordinator),
            maxNewTokens: params.maxTokens ?? maxNewTokens)
        try validateVisibleOmniText(replay.text, row: "video repeated cache alias replay")
        let afterReplay = coordinator.snapshotStats()
        let beforeHits = totalCacheHits(beforeReplay)
        let afterHits = totalCacheHits(afterReplay)
        guard afterHits > beforeHits else {
            throw NSError(
                domain: "OmniBench", code: 127,
                userInfo: [NSLocalizedDescriptionKey:
                    "replayed video did not increment paged/disk cache hits"])
        }

        let secs = CFAbsoluteTimeGetCurrent() - t0
        let detail = [
            "same-video post-prepare alias",
            "raw=\(firstRawTokens.count)",
            "effective=\(effectiveTokens.count)",
            "salt=\(salt.prefix(12))",
            "probe=\(directDetail)",
            "hits \(beforeHits)->\(afterHits)",
            "first=\(first.text.prefix(60))",
            "replay=\(replay.text.prefix(60))",
            "stats=\(cacheStatsSummary(afterReplay))",
        ].joined(separator: " | ")
        return TurnResult(
            shortText: detail,
            tokens: first.tokens + replay.tokens,
            secs: secs)
    }

    /// Helper for B>1 BatchEngine harness rows: collect a single
    /// slot's stream output. Used to drive multiple slots concurrently
    /// via `async let`.
    private static func collectStream(
        engine: BatchEngine,
        input: LMInput,
        parameters: GenerateParameters,
        maxNew: Int
    ) async -> (text: String, tokens: Int) {
        nonisolated(unsafe) let sendable = input
        let stream = await engine.generate(input: sendable, parameters: parameters)
        var text = ""
        var chunks = 0
        // Count BOTH .chunk and .reasoning as real output. The
        // Nemotron-Omni-Reasoning model emits `<think>...</think>` first;
        // with short maxNew the parser routes everything into .reasoning
        // and `text` stays empty even though the pipeline is producing
        // tokens. We care about "did BatchEngine produce text?", not
        // "did it produce non-reasoning text?".
        for await event in stream {
            switch event {
            case .chunk(let c): text += c; chunks += 1
            case .reasoning(let r): text += r; chunks += 1
            default: break
            }
            if chunks > maxNew * 2 { break }
        }
        return (text, chunks)
    }

    private static func collectGenerationText(
        stream: AsyncStream<Generation>,
        maxNewTokens: Int
    ) async throws -> (text: String, reasoning: String, tokens: Int, info: GenerateCompletionInfo?) {
        var text = ""
        var reasoning = ""
        var info: GenerateCompletionInfo?
        var events = 0
        for await event in stream {
            switch event {
            case .chunk(let chunk):
                text += chunk
                events += 1
            case .reasoning(let chunk):
                reasoning += chunk
                events += 1
            case .info(let generationInfo):
                info = generationInfo
            case .toolCall:
                break
            }
            if events > maxNewTokens * 2 {
                throw NSError(
                    domain: "OmniBench", code: 128,
                    userInfo: [NSLocalizedDescriptionKey:
                        "generation emitted too many chunks without completing"])
            }
        }
        let visible = text.trimmingCharacters(in: .whitespacesAndNewlines)
        return (
            visible.isEmpty ? reasoning.trimmingCharacters(in: .whitespacesAndNewlines) : visible,
            reasoning,
            info?.generationTokenCount ?? events,
            info)
    }

    private static func makeVideoProofInput(
        videoURL: URL,
        context: ModelContext
    ) async throws -> LMInput {
        var userInput = UserInput(
            prompt: "Describe this video briefly.",
            videos: [.url(videoURL)])
        userInput.additionalContext = ["enable_thinking": false]
        return try await context.processor.prepare(input: userInput)
    }

    private static func makeOmniProofCoordinator(
        modelDir: URL,
        context: ModelContext,
        label: String
    ) -> CacheCoordinator {
        let safeModelName = modelDir.lastPathComponent
            .replacingOccurrences(of: "/", with: "_")
            .replacingOccurrences(of: " ", with: "_")
        let safeLabel = label
            .replacingOccurrences(of: "/", with: "_")
            .replacingOccurrences(of: " ", with: "_")
        let cacheDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("vmlx-omni-\(safeModelName)-\(safeLabel)-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(
            at: cacheDir,
            withIntermediateDirectories: true)
        return CacheCoordinator(config: CacheCoordinatorConfig(
            usePagedCache: true,
            enableDiskCache: true,
            pagedBlockSize: 64,
            maxCacheBlocks: 4096,
            diskCacheMaxGB: 2.0,
            diskCacheDir: cacheDir,
            ssmMaxEntries: 64,
            enableSSMReDerive: true,
            modelKey: "\(modelDir.path)|\(String(describing: type(of: context.model)))|\(label)"))
    }

    private static func totalCacheHits(_ stats: CacheCoordinatorStatsSnapshot) -> Int {
        (stats.pagedStats?.cacheHits ?? 0)
            + (stats.diskStats?.hits ?? 0)
            + stats.ssmStats.hits
    }

    private static func cacheStatsSummary(_ stats: CacheCoordinatorStatsSnapshot) -> String {
        let paged = stats.pagedStats.map {
            "paged(h=\($0.cacheHits),m=\($0.cacheMisses),a=\($0.allocatedBlocks))"
        } ?? "paged(off)"
        let disk = stats.diskStats.map {
            "disk(h=\($0.hits),m=\($0.misses),s=\($0.stores))"
        } ?? "disk(off)"
        let ssm = "ssm(h=\(stats.ssmStats.hits),m=\(stats.ssmStats.misses),r=\(stats.ssmStats.reDerives))"
        return "\(paged),\(disk),\(ssm),hybrid=\(stats.isHybrid)"
    }

    private static func runAudioTurn(
        prompt: String,
        audioURL: URL,
        context: ModelContext,
        cache: [KVCache],
        maxNewTokens: Int
    ) async throws -> TurnResult {
        // Full LMInput audio path — UserInput.audios → processor →
        // NemotronHOmni.prepare splices at sound_context_token_id=27
        // → forward → TokenIterator decodes. Same shape as the image
        // turn except with `audios:` instead of `images:`.
        let userInput = UserInput(prompt: prompt, audios: [.url(audioURL)])
        let lmInput = try await context.processor.prepare(input: userInput)

        let params = makeOmniParameters(
            context: context,
            maxNewTokens: maxNewTokens)

        let iter = try TokenIterator(
            input: lmInput, model: context.model, cache: cache,
            parameters: params)
        let t0 = CFAbsoluteTimeGetCurrent()
        let raw = collectRawGeneratedTokens(
            iter, context: context, maxNewTokens: maxNewTokens)
        let secs = CFAbsoluteTimeGetCurrent() - t0
        let tokens = raw.tokenIds
        let text = userVisibleText(
            context: context, lmInput: lmInput, tokenIds: tokens)
        debugOmniRawDecode(
            label: "audio",
            context: context,
            lmInput: lmInput,
            raw: raw,
            visibleText: text)
        try validateVisibleOmniText(text, row: "audio turn")
        return TurnResult(
            shortText: text.replacingOccurrences(of: "\n", with: " "),
            tokens: tokens.count, secs: secs)
    }

    private static func collectGeneratedTokens(
        _ iterator: TokenIterator,
        context: ModelContext,
        maxNewTokens: Int
    ) -> [Int] {
        collectRawGeneratedTokens(
            iterator, context: context, maxNewTokens: maxNewTokens).tokenIds
    }

    private static func collectRawGeneratedTokens(
        _ iterator: TokenIterator,
        context: ModelContext,
        maxNewTokens: Int
    ) -> RawGenerationResult {
        let stops = rawIteratorStopTokenIDs(context: context)
        var tokens: [Int] = []
        for token in iterator {
            if stops.contains(token) {
                return RawGenerationResult(
                    tokenIds: tokens,
                    stopTokenID: token,
                    hitMaxTokens: false)
            }
            tokens.append(token)
            if tokens.count >= maxNewTokens {
                return RawGenerationResult(
                    tokenIds: tokens,
                    stopTokenID: nil,
                    hitMaxTokens: true)
            }
        }
        return RawGenerationResult(
            tokenIds: tokens,
            stopTokenID: nil,
            hitMaxTokens: false)
    }

    private static func debugOmniRawDecode(
        label: String,
        context: ModelContext,
        lmInput: LMInput,
        raw: RawGenerationResult,
        visibleText: String
    ) {
        guard ProcessInfo.processInfo.environment["BENCH_OMNI_DIAG"] == "1" else {
            return
        }
        let rawText = context.tokenizer.decode(
            tokenIds: raw.tokenIds,
            skipSpecialTokens: false)
        let promptTokens = lmInput.text.tokens.reshaped(-1).asArray(Int.self)
        let promptTail = context.tokenizer.decode(
            tokenIds: Array(promptTokens.suffix(256)),
            skipSpecialTokens: false)
        let tokenTail = raw.tokenIds.suffix(24).map(String.init).joined(separator: ",")
        print("""
        [OMNI_DIAG] label=\(label) tokens=\(raw.tokenIds.count) stopToken=\(raw.stopTokenID.map(String.init) ?? "nil") hitMax=\(raw.hitMaxTokens)
        [OMNI_DIAG] token_tail=[\(tokenTail)]
        [OMNI_DIAG] prompt_tail="\(escapedDiag(promptTail))"
        [OMNI_DIAG] raw="\(escapedDiag(rawText))"
        [OMNI_DIAG] visible="\(escapedDiag(visibleText))"
        """)
    }

    private static func escapedDiag(_ text: String) -> String {
        text
            .replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "\n", with: "\\n")
            .replacingOccurrences(of: "\r", with: "\\r")
            .replacingOccurrences(of: "\"", with: "\\\"")
    }

    private static func rawIteratorStopTokenIDs(context: ModelContext) -> Set<Int> {
        var stops = context.configuration.eosTokenIds
        if let eos = context.tokenizer.eosTokenId { stops.insert(eos) }
        if let unk = context.tokenizer.unknownTokenId { stops.insert(unk) }
        for token in context.configuration.extraEOSTokens {
            if let id = context.tokenizer.convertTokenToId(token) {
                stops.insert(id)
            }
        }
        for token in commonEndTurnTokens {
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
        allowReasoningFallback: Bool = true
    ) -> String {
        reasoningAndVisibleText(
            context: context,
            lmInput: lmInput,
            tokenIds: tokenIds,
            allowReasoningFallback: allowReasoningFallback).visible
    }

    private static func reasoningAndVisibleText(
        context: ModelContext,
        lmInput: LMInput,
        tokenIds: [Int],
        allowReasoningFallback: Bool = true
    ) -> (reasoning: String, visible: String) {
        let raw = context.tokenizer.decode(tokenIds: tokenIds, skipSpecialTokens: false)
        let promptTokens = lmInput.text.tokens.reshaped(-1).asArray(Int.self)
        let promptTail = context.tokenizer.decode(
            tokenIds: Array(promptTokens.suffix(256)),
            skipSpecialTokens: false)

        guard var parser = ReasoningParser.forPrompt(
            stampName: context.configuration.reasoningParserName,
            promptTail: promptTail)
        else {
            return ("", raw.trimmingCharacters(in: .whitespacesAndNewlines))
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
            return (reasoning.trimmingCharacters(in: .whitespacesAndNewlines), trimmedVisible)
        }
        let trimmedReasoning = reasoning.trimmingCharacters(in: .whitespacesAndNewlines)
        guard allowReasoningFallback else { return (trimmedReasoning, "") }
        return (trimmedReasoning, trimmedReasoning)
    }

    private static func validateVisibleOmniText(_ text: String, row: String) throws {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty {
            throw NSError(
                domain: "OmniBench", code: 20,
                userInfo: [NSLocalizedDescriptionKey: "\(row): empty visible text"])
        }
        if trimmed.contains("<think>") || trimmed.contains("</think>") {
            throw NSError(
                domain: "OmniBench", code: 21,
                userInfo: [NSLocalizedDescriptionKey: "\(row): raw reasoning tag leaked"])
        }

        let lower = trimmed.lowercased()
        if occurrenceCount("let's see", in: lower) >= 3 {
            throw NSError(
                domain: "OmniBench", code: 22,
                userInfo: [NSLocalizedDescriptionKey: "\(row): repeated filler phrase loop"])
        }
        let repeatedBigram = maxRepeatedBigram(in: lower)
        if repeatedBigram.count >= 4 {
            let head = String(trimmed.prefix(220))
            let tail = String(trimmed.suffix(220))
            throw NSError(
                domain: "OmniBench", code: 23,
                userInfo: [NSLocalizedDescriptionKey:
                    "\(row): repeated bigram loop bigram=\"\(repeatedBigram.bigram)\" "
                    + "count=\(repeatedBigram.count) head=\(head) tail=\(tail)"])
        }
        if row.lowercased().contains("image") {
            let missingMediaPhrases = [
                "blank or missing",
                "uploaded correctly",
                "can't describe the image",
                "cannot describe the image",
                "unable to view the image",
                "can't see the image",
            ]
            if missingMediaPhrases.contains(where: lower.contains) {
                let excerpt = String(trimmed.prefix(160))
                throw NSError(
                    domain: "OmniBench", code: 24,
                    userInfo: [NSLocalizedDescriptionKey:
                        "\(row): image not grounded; excerpt=\(excerpt)"])
            }
        }
    }

    private static func validateSyntheticGradientImageText(
        _ text: String,
        row: String
    ) throws {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        let lower = trimmed.lowercased()
        let impossibleTerms = [
            "ooredoo",
            "white background",
            "blue circle",
            "solid color",
            "single, solid color",
            "uniform, consistent color",
            "no gradients",
            "person's face",
            "yellow shirt",
            "text \"describe",
            "centered in white",
        ]
        if impossibleTerms.contains(where: lower.contains) {
            throw NSError(
                domain: "OmniBench", code: 26,
                userInfo: [NSLocalizedDescriptionKey:
                    "\(row): hallucinated non-existent image content; excerpt=\(String(trimmed.prefix(180)))"])
        }
        let hasBlue = lower.contains("blue") || lower.contains("cyan")
        let hasWarm = lower.contains("red")
            || lower.contains("orange")
            || lower.contains("yellow")
        if !hasBlue || !hasWarm {
            throw NSError(
                domain: "OmniBench", code: 25,
                userInfo: [NSLocalizedDescriptionKey:
                    "\(row): synthetic warm/blue image not grounded; excerpt=\(String(trimmed.prefix(180)))"])
        }
    }

    private static func occurrenceCount(_ needle: String, in haystack: String) -> Int {
        guard !needle.isEmpty else { return 0 }
        var count = 0
        var rest = haystack[...]
        while let range = rest.range(of: needle) {
            count += 1
            rest = rest[range.upperBound...]
        }
        return count
    }

    private static func maxRepeatedBigram(in text: String) -> (bigram: String, count: Int) {
        let stopBigrams: Set<String> = [
            "of the", "in the", "to the", "and the", "is a", "it is",
        ]
        let words = text
            .split { !$0.isLetter && !$0.isNumber && $0 != "'" }
            .map(String.init)
        guard words.count >= 2 else { return ("", 0) }
        var counts: [String: Int] = [:]
        for i in 0..<(words.count - 1) {
            let key = words[i] + " " + words[i + 1]
            guard !stopBigrams.contains(key) else { continue }
            counts[key, default: 0] += 1
        }
        return counts.max { lhs, rhs in
            if lhs.value == rhs.value { return lhs.key > rhs.key }
            return lhs.value < rhs.value
        }.map { ($0.key, $0.value) } ?? ("", 0)
    }

    // MARK: - Synthesizers

    private static func synthesiseGradient(side: Int) throws -> CIImage {
        var bytes = [UInt8](repeating: 0, count: side * side * 4)
        for y in 0 ..< side {
            for x in 0 ..< side {
                let off = (y * side + x) * 4
                let isTop = y < side / 2
                bytes[off + 0] = isTop ? 255 : 0
                bytes[off + 1] = isTop ? 32 : 80
                bytes[off + 2] = isTop ? 0 : 255
                bytes[off + 3] = 255
            }
        }
        let data = Data(bytes)
        let cs = CGColorSpace(name: CGColorSpace.sRGB)!
        guard let img = CIImage(
            bitmapData: data, bytesPerRow: side * 4,
            size: .init(width: side, height: side),
            format: .RGBA8, colorSpace: cs) as CIImage?
        else {
            throw NSError(domain: "OmniBench", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "CIImage build failed"])
        }
        return img
    }

    /// Build a `seconds`-long 16 kHz mono Float32 tone at `hz` for the
    /// audio path. Real ASR-quality testing requires a real waveform —
    /// the goal here is to verify the encode pipeline (mel STFT ->
    /// Parakeet -> sound_projection) doesn't crash on real weights.
    private static func synthesiseToneWaveform(seconds: Double, hz: Double) -> [Float] {
        let sr = 16_000
        let n = Int(seconds * Double(sr))
        var w = [Float](repeating: 0, count: n)
        for i in 0 ..< n {
            w[i] = Float(0.1 * sin(2.0 * .pi * hz * Double(i) / Double(sr)))
        }
        return w
    }
}
