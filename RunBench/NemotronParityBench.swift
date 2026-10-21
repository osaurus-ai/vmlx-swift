import Foundation
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import MLXVLM
import VMLXTokenizers

/// BENCH_PARITY_FIXTURE=<fixture.json> BENCH_MODEL=<bundle>: numeric parity of
/// this runtime against a reference fixture produced by the Python runtime from
/// the SAME bundle (format: `swift-parity/make_fixture.py`, Nemotron 3.5).
///
/// Per text case:
/// 1. prompt ids: the bundle chat template rendered here vs the fixture's ids;
/// 2. teacher-forced: one forward over prompt + the fixture's greedy ids, then
///    for every position top-1 agreement and the KL(ref || swift) restricted to
///    the reference's top-20 ids (both renormalized over that set);
/// 3. greedy: this runtime's own greedy continuation vs the fixture's.
///
/// Fails closed: a case with no positions or an empty reference is INVALID,
/// never a pass. Diagnostic only — not a speed or app-behavior measurement.
enum NemotronParityBench {
    struct CaseResult {
        let name: String
        let promptMatch: Bool
        let positions: Int
        let top1Agree: Double
        let klMean: Double
        let klMax: Double
        let greedyPrefixMatch: Int
        let greedyCount: Int
        var visionMatches: Bool? = nil
    }

    static func run(modelDir: URL, fixturePath: String, imagePath: String?) async throws {
        let data = try Data(contentsOf: URL(fileURLWithPath: fixturePath))
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
            let cases = root["cases"] as? [[String: Any]], !cases.isEmpty
        else {
            throw NSError(domain: "NemotronParity", code: 2, userInfo: [
                NSLocalizedDescriptionKey: "Parity fixture has no cases"])
        }

        let loadStart = CFAbsoluteTimeGetCurrent()
        // BENCH_PARITY_LLM_ONLY=1 loads the text model through the LLM
        // factory even when the bundle also routes to a VLM (config_omni.json),
        // so the language path can be scored before the vision path exists.
        let context: ModelContext
        if ProcessInfo.processInfo.environment["BENCH_PARITY_LLM_ONLY"] == "1" {
            context = try await LLMModelFactory.shared.load(
                from: modelDir, using: #huggingFaceTokenizerLoader())
        } else {
            context = try await MLXLMCommon.loadModel(
                from: modelDir,
                using: #huggingFaceTokenizerLoader(),
                loadConfiguration: .osaurusProduction
            )
        }
        print(String(format: "PARITY load: %.2fs model=%@", CFAbsoluteTimeGetCurrent() - loadStart,
            String(describing: type(of: context.model))))

        let requestedCases = ProcessInfo.processInfo.environment["BENCH_PARITY_CASES"]
            .map { Set($0.split(separator: ",").map(String.init)) }
        var results: [CaseResult] = []
        for c in cases {
            if let requestedCases, let name = c["name"] as? String,
                !requestedCases.contains(name) { continue }

            guard let name = c["name"] as? String,
                let messagesAny = c["messages"] as? [[String: Any]],
                let refPrompt = (c["prompt_ids"] as? [Int]),
                let refGreedy = (c["greedy_ids"] as? [Int]),
                let refTopIds = c["top20_ids"] as? [[Int]],
                let refTopLp = c["top20_logprobs"] as? [[Double]],
                !refPrompt.isEmpty, refTopIds.count == refPrompt.count + refGreedy.count
            else {
                print("PARITY INVALID: malformed case \(c["name"] ?? "?")")
                continue
            }
            let isImage = c["vision"] is [String: Any]
            if isImage {
                guard let imagePath else {
                    print("PARITY SKIP \(name): image case needs BENCH_PARITY_IMAGE")
                    continue
                }
                if let r = try await runImageCase(
                    context: context, name: name, c: c, imagePath: imagePath,
                    refPrompt: refPrompt, refGreedy: refGreedy, refTopIds: refTopIds, refTopLp: refTopLp)
                {
                    results.append(r)
                }
                continue
            }

            let kwargs = (c["chat_template_kwargs"] as? [String: Any]) ?? [:]
            let tools = (c["tools"] as? [[String: Any]])?.map { sendableDictionary($0) }
            let messages = messagesAny.map { sendableDictionary($0) }
            let rendered = try context.tokenizer.applyChatTemplate(
                messages: messages, tools: tools,
                additionalContext: sendableDictionary(kwargs))
            let promptMatch = rendered == refPrompt
            if !promptMatch {
                let firstDiff = zip(rendered, refPrompt).enumerated().first { $0.element.0 != $0.element.1 }?.offset
                print("PARITY \(name) prompt MISMATCH swift=\(rendered.count) ref=\(refPrompt.count) first_diff=\(firstDiff.map(String.init) ?? "len")")
                print("  swift text: \(context.tokenizer.decode(tokenIds: rendered, skipSpecialTokens: false).debugDescription)")
                print("  ref   text: \(context.tokenizer.decode(tokenIds: refPrompt, skipSpecialTokens: false).debugDescription)")
            }

            // Teacher-forced over the REFERENCE ids so positions line up even
            // when the template differs.
            let full = refPrompt + refGreedy
            let logits = context.model(
                LMInput.Text(tokens: MLXArray(full.map { Int32($0) })[.newAxis, .ellipsis]),
                cache: nil, state: nil
            ).logits.asType(.float32)
            let r = score(name: name, logits: logits[0], refTopIds: refTopIds, refTopLp: refTopLp)

            // Own greedy continuation from the reference prompt.
            let greedy = greedyDecode(
                model: context.model, prompt: refPrompt, maxTokens: refGreedy.count,
                stop: [2, 11])
            let prefix = zip(greedy, refGreedy).prefix { $0 == $1 }.count
            print("PARITY \(name) greedy swift: \(context.tokenizer.decode(tokenIds: greedy, skipSpecialTokens: false).debugDescription)")
            results.append(CaseResult(
                name: name, promptMatch: promptMatch, positions: r.positions, top1Agree: r.top1,
                klMean: r.klMean, klMax: r.klMax, greedyPrefixMatch: prefix, greedyCount: refGreedy.count))
        }

        print("PARITY SUMMARY bundle=\(modelDir.lastPathComponent)")
        print("| case | prompt ids | positions | top-1 agree | KL mean (top-20) | KL max | greedy prefix |")
        print("|---|---|---|---|---|---|---|")
        for r in results {
            print(String(format: "| %@ | %@ | %d | %.2f%% | %.2e | %.3f | %d/%d |",
                r.name, r.promptMatch ? "identical" : "DIFF", r.positions, r.top1Agree * 100,
                r.klMean, r.klMax, r.greedyPrefixMatch, r.greedyCount))
        }
        let expectedNames = Set(cases.compactMap { $0["name"] as? String })
        let requiredNames = requestedCases ?? expectedNames
        let actualNames = Set(results.map(\.name))
        let failed = results.filter {
            !$0.promptMatch || $0.positions == 0 || $0.top1Agree < 0.99
                || !$0.klMean.isFinite || $0.klMean > 1e-3
                || $0.greedyPrefixMatch < min(16, $0.greedyCount)
                || $0.visionMatches == false
        }
        guard !results.isEmpty, actualNames == requiredNames, failed.isEmpty else {
            throw NSError(domain: "NemotronParity", code: 1, userInfo: [
                NSLocalizedDescriptionKey:
                    "Parity acceptance failed; failed=\(failed.map(\.name)), missing=\(requiredNames.subtracting(actualNames).sorted()). See measured rows above."
            ])
        }
        print("PARITY ACCEPTANCE PASS cases=\(results.count)")
    }

    private static func runImageCase(
        context: ModelContext, name: String, c: [String: Any], imagePath: String,
        refPrompt: [Int], refGreedy: [Int], refTopIds: [[Int]], refTopLp: [[Double]]
    ) async throws -> CaseResult? {
        let kwargs = (c["chat_template_kwargs"] as? [String: Any]) ?? [:]
        let text = ((c["messages"] as? [[String: Any]])?.first?["content"] as? [[String: Any]])?
            .compactMap { $0["text"] as? String }.joined() ?? ""
        let input = UserInput(
            chat: [.user(text, images: [.url(URL(fileURLWithPath: imagePath))])],
            additionalContext: sendableDictionary(kwargs))
        let prepared = try await context.processor.prepare(input: input)
        let ids = prepared.text.tokens.asArray(Int32.self).map(Int.init)
        let promptMatch = ids == refPrompt
        if !promptMatch {
            print("PARITY \(name) prompt MISMATCH swift=\(ids.count) ref=\(refPrompt.count)")
            print("  swift head: \(Array(ids.prefix(16))) tail: \(Array(ids.suffix(16)))")
            print("  ref   head: \(Array(refPrompt.prefix(16))) tail: \(Array(refPrompt.suffix(16)))")
            return CaseResult(
                name: name, promptMatch: false, positions: 0, top1Agree: 0, klMean: .nan,
                klMax: .nan, greedyPrefixMatch: 0, greedyCount: refGreedy.count)
        }
        // The fixture scores one teacher-forced forward over prompt + answer.
        // Match that shape; sequential cached forwards have different reduction
        // shapes and are tested separately by the independent continuation.
        let cache = context.model.newCache(parameters: nil)
        let teacherInput = LMInput(
            text: .init(tokens: MLXArray((refPrompt + refGreedy).map(Int32.init))[.newAxis, .ellipsis]),
            image: prepared.image, mediaTokenIds: prepared.mediaTokenIds)
        let first = try context.model.prepare(teacherInput, cache: cache, windowSize: nil)
        var lastLogits: MLXArray
        switch first {
        case .logits(let out):
            lastLogits = out.logits
        case .tokens(let remaining):
            lastLogits = context.model(remaining, cache: cache, state: nil).logits
        }
        let teacherRows = lastLogits.dim(1)
        var visionMatches = false
        if let omni = context.model as? NemotronHOmni, let pixels = prepared.image?.pixels,
            let vision = c["vision"] as? [String: Any]
        {
            let embeds = omni.extractImageEmbeds(pixelValues: pixels, frames: prepared.image?.frames)
                .asType(.float32)
            if let referencePath = ProcessInfo.processInfo.environment["BENCH_PARITY_VISION_ARRAYS"] {
                let reference = try loadArrays(url: URL(fileURLWithPath: referencePath))
                if let refPixels = reference["pixels"], let refRows = reference["rows"] {
                    print("PARITY vision arrays pixel_max=\(abs(pixels.asType(.float32) - refPixels).max().item(Float.self)) row_max=\(abs(embeds - refRows).max().item(Float.self))")
                    try save(arrays: ["pixels": pixels, "rows": embeds],
                        url: URL(fileURLWithPath: referencePath).deletingLastPathComponent()
                            .appendingPathComponent("vision-swift.safetensors"))
                }
            }
            let first8 = embeds[0 ..< 8]
            let sums = first8.sum(axis: -1)
            let l2 = sqrt((first8 * first8).sum(axis: -1))
            let mean = embeds.mean()
            let absMean = abs(embeds).mean()
            eval(sums, l2, mean, absMean)
            let refSums = (vision["row_sum_first8"] as? [Double]) ?? []
            let refL2 = (vision["row_l2_first8"] as? [Double]) ?? []
            let sw = sums.asArray(Float.self).map(Double.init)
            let swL2 = l2.asArray(Float.self).map(Double.init)
            let l2Rel = zip(swL2, refL2).map { abs($0 - $1) / max(1e-6, abs($1)) }.max() ?? .nan
            let sumAbs = zip(sw, refSums).map { abs($0 - $1) }.max() ?? .nan
            let sumRel = zip(sw, refSums).map { abs($0 - $1) / max(1e-6, abs($1)) }.max() ?? .nan
            let refAbsMean = (vision["all_rows_absmean"] as? Double) ?? .nan
            let absMeanRel = abs(Double(absMean.item(Float.self)) - refAbsMean) / max(1e-6, abs(refAbsMean))
            visionMatches = embeds.dim(0) == (vision["n_rows"] as? Int)
                && refSums.count == 8 && refL2.count == 8
                && l2Rel <= 0.02 && sumRel <= 0.02 && absMeanRel <= 0.01
            print(String(format: "PARITY %@ vision rows=%d (ref %d) first8 L2 max_rel=%.4f sum max_abs=%.3f absmean swift=%.6f ref=%.6f (rel %.4f) mean swift=%.6f ref=%.6f",
                name, embeds.dim(0), (vision["n_rows"] as? Int) ?? -1, l2Rel, sumAbs,
                Double(absMean.item(Float.self)), refAbsMean,
                abs(Double(absMean.item(Float.self)) - refAbsMean) / refAbsMean,
                Double(mean.item(Float.self)), (vision["all_rows_mean"] as? Double) ?? .nan))
            print("PARITY \(name) vision first8 sums swift=\(sw.map { String(format: "%.2f", $0) }) ref=\(refSums.map { String(format: "%.2f", $0) })")
        }
        let offset = refPrompt.count + refGreedy.count - teacherRows
        let r = score(
            name: name, logits: lastLogits[0].asType(.float32),
            refTopIds: Array(refTopIds[offset...]), refTopLp: Array(refTopLp[offset...]))
        // Independent continuation: every next forward consumes OUR token,
        // never the fixture token. Teacher-forced argmax is not generation.
        let greedyCache = context.model.newCache(parameters: nil)
        let prefillStart = CFAbsoluteTimeGetCurrent()
        let greedyPrepared = try context.model.prepare(prepared, cache: greedyCache, windowSize: nil)
        var nextLogits: MLXArray
        switch greedyPrepared {
        case .logits(let output): nextLogits = output.logits
        case .tokens(let remaining):
            nextLogits = context.model(remaining, cache: greedyCache, state: nil).logits
        }
        eval(nextLogits)
        let prefillSeconds = CFAbsoluteTimeGetCurrent() - prefillStart
        let decodeStart = CFAbsoluteTimeGetCurrent()
        var greedy: [Int] = []
        for _ in 0 ..< refGreedy.count {
            let token = argMax(nextLogits[0, -1], axis: -1).item(Int.self)
            greedy.append(token)
            if token == 2 || token == 11 { break }
            nextLogits = context.model(
                LMInput.Text(tokens: MLXArray([Int32(token)])[.newAxis, .ellipsis]),
                cache: greedyCache, state: nil).logits
        }
        let decodeSeconds = CFAbsoluteTimeGetCurrent() - decodeStart
        let prefix = zip(greedy, refGreedy).prefix { $0 == $1 }.count
        print("PARITY \(name) independent greedy: \(context.tokenizer.decode(tokenIds: greedy, skipSpecialTokens: false).debugDescription)")
        print(String(format: "PARITY %@ diagnostic prefill_seconds=%.3f decode_tokens=%d decode_seconds=%.3f tokens_per_second=%.3f stop=%@",
            name, prefillSeconds, greedy.count, decodeSeconds,
            Double(greedy.count) / max(decodeSeconds, 1e-9),
            [2, 11].contains(greedy.last ?? -1) ? "eos" : "fixture_cap"))
        return CaseResult(
            name: name, promptMatch: true, positions: r.positions, top1Agree: r.top1,
            klMean: r.klMean, klMax: r.klMax, greedyPrefixMatch: prefix, greedyCount: refGreedy.count,
            visionMatches: visionMatches)
    }

    private static func score(
        name: String, logits: MLXArray, refTopIds: [[Int]], refTopLp: [[Double]]
    ) -> (positions: Int, top1: Double, klMean: Double, klMax: Double) {
        let n = min(logits.dim(0), refTopIds.count)
        guard n > 0 else { return (0, 0, .nan, .nan) }
        let lp = logits - logSumExp(logits, axis: -1, keepDims: true)
        let idx = MLXArray(refTopIds.prefix(n).flatMap { $0.map { Int32($0) } }).reshaped(n, -1)
        let picked = takeAlong(lp[0 ..< n], idx, axis: -1)
        let top1 = argMax(lp[0 ..< n], axis: -1)
        eval(picked, top1)
        let swiftTop = picked.asArray(Float.self)
        let swiftArg = top1.asArray(Int32.self)
        let k = refTopIds[0].count
        let verbose = ProcessInfo.processInfo.environment["BENCH_PARITY_VERBOSE"] == "1"
        var agree = 0
        var decisive = 0
        var decisiveAgree = 0
        var klSum = 0.0
        var klMax = 0.0
        for i in 0 ..< n {
            if Int(swiftArg[i]) == refTopIds[i][0] { agree += 1 }
            let ref = refTopLp[i]
            // The reference log-probs come from bf16 logits (0.125-0.25 steps,
            // exact ties): only positions whose top-1 leads by >= 0.5 nats
            // have a well-defined argmax to compare.
            if ref.count > 1, ref[0] - ref[1] >= 0.5 {
                decisive += 1
                if Int(swiftArg[i]) == refTopIds[i][0] { decisiveAgree += 1 }
            }
            let refLse = log(ref.map { exp($0) }.reduce(0, +))
            let sw = (0 ..< k).map { Double(swiftTop[i * k + $0]) }
            let swLse = log(sw.map { exp($0) }.reduce(0, +))
            var kl = 0.0
            for j in 0 ..< k {
                let p = ref[j] - refLse
                let q = sw[j] - swLse
                kl += exp(p) * (p - q)
            }
            klSum += kl
            klMax = max(klMax, kl)
            if verbose {
                print(String(format: "PARITY_POS %@ %d kl=%.4f ref_top=%d swift_top=%d", name, i, kl,
                    refTopIds[i][0], Int(swiftArg[i])))
            }
        }
        print(String(format: "PARITY %@ teacher-forced positions=%d top1=%.2f%% decisive_top1=%d/%d kl_mean=%.3e kl_max=%.4f",
            name, n, Double(agree) / Double(n) * 100, decisiveAgree, decisive, klSum / Double(n), klMax))
        return (n, Double(agree) / Double(n), klSum / Double(n), klMax)
    }

    private static func greedyDecode(
        model: any LanguageModel, prompt: [Int], maxTokens: Int, stop: Set<Int>
    ) -> [Int] {
        let cache = model.newCache(parameters: nil)
        let prefillStart = CFAbsoluteTimeGetCurrent()
        var logits = model(
            LMInput.Text(tokens: MLXArray(prompt.map { Int32($0) })[.newAxis, .ellipsis]),
            cache: cache, state: nil
        ).logits
        eval(logits)
        let prefillSeconds = CFAbsoluteTimeGetCurrent() - prefillStart
        let decodeStart = CFAbsoluteTimeGetCurrent()
        var out: [Int] = []
        for _ in 0 ..< maxTokens {
            let t = argMax(logits[0, -1], axis: -1).item(Int.self)
            out.append(t)
            if stop.contains(t) { break }
            logits = model(
                LMInput.Text(tokens: MLXArray([Int32(t)])[.newAxis, .ellipsis]),
                cache: cache, state: nil
            ).logits
        }
        let decodeSeconds = CFAbsoluteTimeGetCurrent() - decodeStart
        print(String(format: "PARITY text diagnostic prefill_seconds=%.3f decode_tokens=%d decode_seconds=%.3f tokens_per_second=%.3f stop=%@",
            prefillSeconds, out.count, decodeSeconds, Double(out.count) / max(decodeSeconds, 1e-9),
            stop.contains(out.last ?? -1) ? "eos" : "fixture_cap"))
        return out
    }

    private static func sendableValue(_ value: Any) -> any Sendable {
        switch value {
        case let d as [String: Any]: return sendableDictionary(d)
        case let a as [Any]: return a.map { sendableValue($0) }
        case let s as String: return s
        case let n as NSNumber:
            if CFGetTypeID(n) == CFBooleanGetTypeID() { return n.boolValue }
            if CFNumberIsFloatType(n) { return n.doubleValue }
            return n.intValue
        default: return String(describing: value)
        }
    }

    private static func sendableDictionary(_ d: [String: Any]) -> [String: any Sendable] {
        d.mapValues { sendableValue($0) }
    }
}
