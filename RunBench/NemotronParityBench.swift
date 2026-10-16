import Foundation
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import MLXNN
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
    }

    static func run(modelDir: URL, fixturePath: String, imagePath: String?) async throws {
        let data = try Data(contentsOf: URL(fileURLWithPath: fixturePath))
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
            let cases = root["cases"] as? [[String: Any]], !cases.isEmpty
        else {
            print("PARITY INVALID: fixture has no cases")
            return
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
            ).0
        }
        print(String(format: "PARITY load: %.2fs model=%@", CFAbsoluteTimeGetCurrent() - loadStart,
            String(describing: type(of: context.model))))
        if let module = context.model as? Module {
            var counts: [String: Int] = [:]
            for (path, m) in module.leafModules().flattened()
            where path.contains("vision_model") || path.contains("mlp1")
                || path.contains("vision_final")
            {
                let key: String
                if let q = m as? QuantizedLinear {
                    key = "\(path.contains("mlp1") ? "mlp1" : "vision") QuantizedLinear b\(q.bits) g\(q.groupSize) w\(q.weight.shape)"
                } else if let l = m as? Linear {
                    key = "\(path.contains("mlp1") ? "mlp1" : "vision") Linear \(l.weight.dtype) \(l.weight.shape)"
                } else {
                    key = "\(path.split(separator: ".").prefix(1).joined()) \(type(of: m))"
                }
                counts[key, default: 0] += 1
            }
            for (k, v) in counts.sorted(by: { $0.key < $1.key }) { print("PARITY vision-module \(k) x\(v)") }
        }

        var results: [CaseResult] = []
        for c in cases {
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
            if let only = ProcessInfo.processInfo.environment["BENCH_PARITY_ONLY"],
                !only.split(separator: ",").contains(Substring(name))
            {
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
        if results.isEmpty { print("PARITY INVALID: no case scored") }
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
        // Prefill the multimodal prompt, then teacher-force the reference
        // continuation one token at a time through the same cache.
        let cache = context.model.newCache(parameters: nil)
        var rows: [MLXArray] = []
        let first = try context.model.prepare(prepared, cache: cache, windowSize: nil)
        var lastLogits: MLXArray
        switch first {
        case .logits(let out):
            lastLogits = out.logits
        case .tokens(let remaining):
            lastLogits = context.model(remaining, cache: cache, state: nil).logits
        }
        rows.append(lastLogits[0].asType(.float32))
        let prefillRows = lastLogits.dim(1)
        if let omni = context.model as? NemotronHOmni, let pixels = prepared.image?.pixels,
            let vision = c["vision"] as? [String: Any]
        {
            let embeds = omni.extractImageEmbeds(pixelValues: pixels, frames: prepared.image?.frames)
                .asType(.float32)
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
            let refAbsMean = (vision["all_rows_absmean"] as? Double) ?? .nan
            print(String(format: "PARITY %@ vision rows=%d (ref %d) first8 L2 max_rel=%.4f sum max_abs=%.3f absmean swift=%.6f ref=%.6f (rel %.4f) mean swift=%.6f ref=%.6f",
                name, embeds.dim(0), (vision["n_rows"] as? Int) ?? -1, l2Rel, sumAbs,
                Double(absMean.item(Float.self)), refAbsMean,
                abs(Double(absMean.item(Float.self)) - refAbsMean) / refAbsMean,
                Double(mean.item(Float.self)), (vision["all_rows_mean"] as? Double) ?? .nan))
            print("PARITY \(name) vision first8 sums swift=\(sw.map { String(format: "%.2f", $0) }) ref=\(refSums.map { String(format: "%.2f", $0) })")
            // BENCH_PARITY_VISION_REF=<safetensors with pixel_values / proj from
            // the Python tower>: split preprocessing error from tower error.
            if let refPath = ProcessInfo.processInfo.environment["BENCH_PARITY_VISION_REF"],
                let ref = try? loadArrays(url: URL(fileURLWithPath: refPath)),
                let refPixels = ref["pixel_values"], let refProj = ref["proj"]
            {
                let swiftPixels = pixels.ndim == 4 ? pixels : pixels.reshaped(refPixels.shape)
                let pixDiff = abs(swiftPixels.asType(.float32) - refPixels).max()
                let pixMeanDiff = abs(swiftPixels.asType(.float32) - refPixels).mean()
                let stages = omni.visionStagesForParity(pixelValues: refPixels.asType(.bfloat16))
                for stage in ["tower", "ln"] {
                    guard let mine = stages[stage]?.asType(.float32), let theirs = ref[stage] else { continue }
                    let a = mine.reshaped(-1, mine.dim(-1)), b = theirs.reshaped(-1, theirs.dim(-1))
                    let c = (a * b).sum(axis: -1) / (sqrt((a * a).sum(axis: -1)) * sqrt((b * b).sum(axis: -1)))
                    let rel = abs(a - b).mean() / abs(b).mean()
                    eval(c, rel)
                    print(String(format: "PARITY vision stage %@ (ref pixels) rel_abs_err=%.4f cos min=%.5f mean=%.5f absmean swift=%.4f ref=%.4f",
                        stage, rel.item(Float.self), c.min().item(Float.self), c.mean().item(Float.self),
                        abs(a).mean().item(Float.self), abs(b).mean().item(Float.self)))
                }
                let sp = pixels.asType(.float32)
                eval(sp)
                print(String(format: "PARITY vision pixel stats swift mean=%.4f absmean=%.4f ref mean=%.4f absmean=%.4f",
                    sp.mean().item(Float.self), abs(sp).mean().item(Float.self),
                    refPixels.mean().item(Float.self), abs(refPixels).mean().item(Float.self)))
                let fromRef = omni.extractImageEmbeds(pixelValues: refPixels.asType(.bfloat16))
                    .asType(.float32)
                let projDiff = abs(fromRef - refProj).mean() / abs(refProj).mean()
                let cos = (fromRef * refProj).sum(axis: -1)
                    / (sqrt((fromRef * fromRef).sum(axis: -1)) * sqrt((refProj * refProj).sum(axis: -1)))
                eval(pixDiff, pixMeanDiff, projDiff, cos)
                print(String(format: "PARITY %@ vision pixels swift-vs-ref max_abs=%.4f mean_abs=%.5f shape swift=%@ ref=%@",
                    name, pixDiff.item(Float.self), pixMeanDiff.item(Float.self),
                    "\(pixels.shape)", "\(refPixels.shape)"))
                print(String(format: "PARITY %@ vision tower(ref pixels) rel_abs_err=%.4f cos min=%.5f mean=%.5f",
                    name, projDiff.item(Float.self), cos.min().item(Float.self), cos.mean().item(Float.self)))
            }
        }
        var greedy: [Int] = []
        var own = argMax(lastLogits[0, -1], axis: -1).item(Int.self)
        for (i, t) in refGreedy.enumerated() {
            greedy.append(own)
            let out = context.model(
                LMInput.Text(tokens: MLXArray([Int32(t)])[.newAxis, .ellipsis]), cache: cache, state: nil
            ).logits
            if i < refGreedy.count - 1 {
                rows.append(out[0].asType(.float32))
            }
            own = argMax(out[0, -1], axis: -1).item(Int.self)
        }
        // Only the positions this path produced: the prompt prefill returns
        // either all rows or only the last; score what lines up.
        let stacked = concatenated(rows, axis: 0)
        // Prefill rows cover the last `prefillRows` prompt positions.
        let offset = refPrompt.count - prefillRows
        let r = score(
            name: name, logits: stacked, refTopIds: Array(refTopIds[offset...]),
            refTopLp: Array(refTopLp[offset...]))
        let prefix = zip(greedy, refGreedy).prefix { $0 == $1 }.count
        print("PARITY \(name) greedy swift(teacher-forced argmax): \(context.tokenizer.decode(tokenIds: greedy, skipSpecialTokens: false).debugDescription)")
        return CaseResult(
            name: name, promptMatch: true, positions: r.positions, top1Agree: r.top1,
            klMean: r.klMean, klMax: r.klMax, greedyPrefixMatch: prefix, greedyCount: refGreedy.count)
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
        var logits = model(
            LMInput.Text(tokens: MLXArray(prompt.map { Int32($0) })[.newAxis, .ellipsis]),
            cache: cache, state: nil
        ).logits
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
