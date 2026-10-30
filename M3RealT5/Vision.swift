// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// MiniMax-M3.1 image input on the real bundle against vMLX Python's graded 12-image set (campaign scripts
// `mm31t/vis_set.py`, `vis_gen.py`, `vis_eval.py`; the r3 run's `gen.json`, `rows.jsonl`, `media.npz`, reference
// logits, converted by `swift_vis_prep.py` into `$T5_VIS/vis_media.safetensors` and `vis_ref.safetensors`):
//
//   visprep  (tokenizer only, no weights) the input processor on every item: prompt tokens == Python's prompt ids,
//            bfloat16 pixel values == Python's bit for bit, grids equal.
//   visfeat  + the vision stack: features (all images of an item in one call, as Python) vs Python's, bfloat16 bits.
//   viseval  teacher-forced on Python's rows (prompt + the bundle's Python answer) with Swift's features spliced:
//            per position the log-probs at the source reference's top-128, the own argmax, the next-token log-prob,
//            the rank of the reference argmax and the think / skip log-probs; `swift_vis_eval.py` scores them as
//            vis_eval does.
//   visgen   the serving path (input processor -> TokenIterator, greedy, disk prefix cache, the bundle's DSpark drafter
//            requested): each item's answer (graded by `swift_vis_grade.py`), then text follow-ups after image turns:
//            reuse, the prefilled suffix holds no image token, DSpark tokens == a cold plain run.

import Foundation
import MLX
import MLXHuggingFace
import MLXLMCommon
import VMLXTokenizers

@testable import MLXLLM

struct VisItem {
    let id: String
    let q: String
    let images: [String]
    let promptTokens: Int
}

let visDir = URL(fileURLWithPath: env["T5_VIS"] ?? "\(NSHomeDirectory())/m31swift/data/vis")

func visItems() throws -> [VisItem] {
    let gen = try JSONSerialization.jsonObject(with: Data(contentsOf: visDir.appendingPathComponent("gen.json")))
        as! [[String: Any]]
    return gen.map { g in
        let id = g["id"] as! String
        let images = id == "two_images" ? ["two_a", "two_b"] : [id]
        return VisItem(
            id: id, q: g["q"] as! String,
            images: images.map { visDir.appendingPathComponent("images/\($0).png").path },
            promptTokens: (g["prompt_tokens"] as! NSNumber).intValue)
    }
}

func visRows() throws -> [String: [Int]] {
    var out: [String: [Int]] = [:]
    let text = try String(contentsOf: visDir.appendingPathComponent("rows.jsonl"), encoding: .utf8)
    for line in text.split(separator: "\n") where !line.isEmpty {
        let row = try JSONSerialization.jsonObject(with: Data(line.utf8)) as! [String: Any]
        out[row["id"] as! String] = (row["ids"] as! [NSNumber]).map(\.intValue)
    }
    return out
}

func visMessages(_ item: VisItem) -> [Chat.Message] {
    [.user(item.q, images: item.images.map { .url(URL(fileURLWithPath: $0)) })]
}

func bits(_ x: MLXArray) -> MLXArray { x.asType(.bfloat16).view(dtype: .uint16) }

/// Prompt tokens, pixel values and grids of the input processor vs Python's.
func visCompareInput(_ item: VisItem, _ input: LMInput, rows: [String: [Int]], media: [String: MLXArray])
    -> [String: Any]
{
    let key = "vis:\(item.id)"
    let ids = tokenIds(input)
    let pyPrompt = Array(rows[key]!.prefix(item.promptTokens))
    var r: [String: Any] = [
        "prompt_tokens": ids.count, "python_prompt_tokens": pyPrompt.count, "tokens_equal": ids == pyPrompt,
        "first_token_diff": firstDiff(ids, pyPrompt) as Any, "image_tokens": ids.filter { $0 == 200_025 }.count,
    ]
    guard let image = input.image else { r["error"] = "no image payload"; return r }
    let pv = media["\(key)|pv"]!
    r["pixels_shape"] = image.pixels.shape
    r["pixels_equal"] = image.pixels.shape == pv.shape && arrayEqual(bits(image.pixels), bits(pv)).item(Bool.self)
    let grid = media["\(key)|grid"]!.asArray(Int32.self).map(Int.init)
    r["grids"] = (image.frames ?? []).map { [$0.t, $0.h, $0.w] }
    r["grids_equal"] = (image.frames ?? []).flatMap { [$0.t, $0.h, $0.w] } == grid
    return r
}

func taskVisPrep() async throws {
    // Tokenizer + the factory's input processor for this bundle, without weights.
    let dir = bundleURL()
    let tokenizer = try await #huggingFaceTokenizerLoader().load(from: dir)
    let configData = try Data(contentsOf: dir.appendingPathComponent("config.json"))
    let vision = try MiniMaxM3VisionConfiguration.decode(configuration: configData)!
    let processor = LLMUserInputProcessor(
        tokenizer: tokenizer, configuration: ModelConfiguration(directory: dir), modelType: "minimax_m3_vl",
        messageGenerator: DefaultMessageGenerator(),
        miniMaxM3Vision: MiniMaxM3VisionPromptConfiguration(
            processor: MiniMaxM3ImageProcessorConfiguration.load(directory: dir), imageTokenId: vision.imageTokenId))
    let rows = try visRows()
    let media = try loadArrays(url: visDir.appendingPathComponent("vis_media.safetensors"))
    var out: [[String: Any]] = []
    for item in try visItems() {
        let t0 = nowMs()
        let input = try processor.prepare(input: UserInput(chat: visMessages(item)))
        var r = visCompareInput(item, input, rows: rows, media: media)
        r["id"] = item.id
        r["prepare_ms"] = nowMs() - t0
        log("\(item.id): tokens \(r["tokens_equal"]!) pixels \(r["pixels_equal"]!) grids \(r["grids_equal"]!) "
            + "(\(r["prompt_tokens"]!) tokens, \(r["image_tokens"]!) image, \(String(format: "%.0f", r["prepare_ms"] as! Double)) ms)")
        out.append(r)
    }
    let all = out.allSatisfy { ($0["tokens_equal"] as? Bool) == true && ($0["pixels_equal"] as? Bool) == true
        && ($0["grids_equal"] as? Bool) == true }
    log("visprep: \(all ? "ALL EQUAL" : "DIFFERENCES") over \(out.count) items")
    try writeJSON(["items": out, "all_equal": all], "visprep_\(tag).json")
}

func taskVisFeatures() async throws {
    let (model, context, loadMs) = try await loadModel(realTokenizer: true)
    guard let vision = model.vision else { fatalError("the bundle loaded without its vision stack") }
    log("loaded \(String(format: "%.1f", loadMs / 1000)) s; vision stack built; TF32 env \(env["MLX_ENABLE_TF32"] ?? "default")")
    let rows = try visRows()
    let media = try loadArrays(url: visDir.appendingPathComponent("vis_media.safetensors"))
    var out: [[String: Any]] = []
    for item in try visItems() {
        let input = try await prepare(context, visMessages(item))
        var r = visCompareInput(item, input, rows: rows, media: media)
        r["id"] = item.id
        let image = input.image!
        let t0 = nowMs()
        let features = vision(image.pixels, grids: image.frames ?? [])
        eval(features)
        r["tower_ms"] = nowMs() - t0
        r["dtype"] = "\(features.dtype)"
        let py = media["vis:\(item.id)|feat"]!
        let swiftBF = features.asType(.bfloat16)
        let same = (bits(swiftBF) .== bits(py)).asType(.float32).mean().item(Float.self)
        let diff = abs(swiftBF.asType(.float32) - py.asType(.float32))
        let f32diff = abs(features.asType(.float32) - py.asType(.float32)).max().item(Float.self)
        let a = swiftBF.asType(.float32), b = py.asType(.float32)
        let cos = ((a * b).sum(axis: -1) / (sqrt((a * a).sum(axis: -1)) * sqrt((b * b).sum(axis: -1)))).min()
            .item(Float.self)
        r["features_rows"] = features.dim(0)
        r["bf16_identical_fraction"] = same
        r["bf16_max_abs_diff"] = diff.max().item(Float.self)
        r["f32_vs_python_bf16_max_abs_diff"] = f32diff
        r["max_abs_feature"] = abs(b).max().item(Float.self)
        r["min_row_cosine"] = cos
        log("\(item.id): bf16 identical \(String(format: "%.6f", same)), max |diff| \(r["bf16_max_abs_diff"]!), "
            + "min cos \(cos), tower \(String(format: "%.0f", r["tower_ms"] as! Double)) ms; tokens \(r["tokens_equal"]!) "
            + "pixels \(r["pixels_equal"]!)")
        out.append(r)
        Memory.clearCache()
    }
    try writeJSON(["items": out, "tf32_env": env["MLX_ENABLE_TF32"] ?? "default"], "visfeat_\(tag).json")
}

func taskVisEval() async throws {
    let (model, context, loadMs) = try await loadModel(realTokenizer: true)
    guard let vision = model.vision else { fatalError("the bundle loaded without its vision stack") }
    log("loaded \(String(format: "%.1f", loadMs / 1000)) s; TF32 env \(env["MLX_ENABLE_TF32"] ?? "default")")
    let rows = try visRows()
    let ref = try loadArrays(url: visDir.appendingPathComponent("vis_ref.safetensors"))
    var arrays: [String: MLXArray] = [:]
    var summary: [[String: Any]] = []
    for item in try visItems() {
        let key = "vis:\(item.id)"
        let ids = rows[key]!
        let input = try await prepare(context, visMessages(item))
        let idsArray = MLXArray(ids.map { Int32($0) }).reshaped(1, ids.count)
        guard let splice = try MiniMaxM3ImageSplice.make(tokens: ids, start: 0, image: input.image!, vision: vision),
            let embeddings = splice.embeddings(for: 0 ..< ids.count, ids: idsArray, embed: { model.model.embedTokens($0) })
        else { fatalError("\(item.id): no image tokens in the row") }
        let t0 = nowMs()
        let cache = model.newCache(parameters: nil)
        let h = model.model(idsArray, cache: cache, inputEmbeddings: embeddings)
        let lp = logProbs(model.lmHead!(h))[0]  // [n, V] float32
        let topIdx = ref["\(key)|top_idx"]!.asType(.int32)
        let refArgmax = ref["\(key)|argmax"]!.asType(.int32)
        let next = MLXArray(ids.dropFirst().map { Int32($0) })
        let q = takeAlong(lp, topIdx, axis: -1)
        let sa = argMax(lp, axis: -1).asType(.int32)
        let tl = takeAlong(lp[..<(ids.count - 1)], next.expandedDimensions(axis: -1), axis: -1).squeezed(axis: -1)
        let rank = (lp .> takeAlong(lp, refArgmax.expandedDimensions(axis: -1), axis: -1)).asType(.int32).sum(axis: -1)
        let dec = concatenated([lp[0..., 200_059 ..< 200_060], lp[0..., 200_060 ..< 200_061]], axis: -1)
        eval(q, sa, tl, rank, dec)
        arrays["\(key)|q"] = q
        arrays["\(key)|sa"] = sa
        arrays["\(key)|tl"] = tl
        arrays["\(key)|rank"] = rank
        arrays["\(key)|dec"] = dec
        let ms = nowMs() - t0
        log("\(item.id): \(ids.count) tokens teacher-forced in \(String(format: "%.0f", ms)) ms")
        summary.append(["id": item.id, "tokens": ids.count, "ms": ms])
        Memory.clearCache()
    }
    let url = outDir.appendingPathComponent("viseval_\(tag).safetensors")
    try save(arrays: arrays, url: url)
    log("wrote \(url.path)")
    try writeJSON(["rows": summary, "tf32_env": env["MLX_ENABLE_TF32"] ?? "default"], "viseval_\(tag).json")
}

final class VisPrefillRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var _prefills: [AlignedPrefillPrefillRecord] = []
    var prefills: [AlignedPrefillPrefillRecord] { lock.withLock { _prefills } }
    func install() {
        AlignedPrefill.hooks.prefillObserver = { [self] r in lock.withLock { _prefills.append(r) } }
    }
}

/// Split a generated text into (reasoning, visible answer) the way a chat client sends it back.
func splitThinking(_ text: String) -> (String?, String) {
    let clean = text.replacingOccurrences(of: "[e~[", with: "")
    guard let close = clean.range(of: "</mm:think>") else { return (nil, clean.trimmingCharacters(in: .whitespacesAndNewlines)) }
    let reasoning = clean[..<close.lowerBound].replacingOccurrences(of: "<mm:think>", with: "")
    return (reasoning.trimmingCharacters(in: .whitespacesAndNewlines),
            String(clean[close.upperBound...]).trimmingCharacters(in: .whitespacesAndNewlines))
}

func taskVisGen() async throws {
    let (model, context, loadMs) = try await loadModel(realTokenizer: true)
    guard model.vision != nil else { fatalError("the bundle loaded without its vision stack") }
    let drafter = try attachDrafter(model, bundleURL())
    log("loaded \(String(format: "%.1f", loadMs / 1000)) s; drafter \(drafter?.lastPathComponent ?? "none"); TF32 env "
        + "\(env["MLX_ENABLE_TF32"] ?? "default")")
    let maxTokens = Int(env["T5_VIS_MAX"] ?? "2048")!
    let diskDir = outDir.appendingPathComponent("diskcache-vis-\(UUID().uuidString)")
    defer { try? FileManager.default.removeItem(at: diskDir) }
    var config = VMLXServerRuntimeSettings().cacheCoordinatorConfig(modelKey: "minimax-m31-vis", diskCacheDirectory: diskDir)
    config.enableDiskCache = true
    let coordinator = CacheCoordinator(config: config)
    coordinator.setGenPromptSuffixTokens(generationPromptSuffix(context.tokenizer))
    let recorder = VisPrefillRecorder()
    recorder.install()
    let rows = try visRows()
    let only = env["T5_VIS_ONLY"].map { Set($0.split(separator: ",").map(String.init)) }

    var items: [[String: Any]] = []
    var answers: [String: (VisItem, String)] = [:]
    for item in try visItems() where only?.contains(item.id) ?? true {
        let input = try await prepare(context, visMessages(item))
        let ids = tokenIds(input)
        let before = recorder.prefills.count
        let (run, iterator) = try greedy(model, input, maxTokens: maxTokens, drafter: drafter, coordinator: coordinator)
        var it = iterator
        it.storeCacheAfterGeneration(generatedTokenIds: run.tokens, includeGeneratedBoundary: false)
        let text = context.tokenizer.decode(tokenIds: run.tokens, skipSpecialTokens: false)
        answers[item.id] = (item, text)
        let pyGen = Array(rows["vis:\(item.id)"]!.dropFirst(item.promptTokens))
        let prefill = recorder.prefills[before...].first
        let r: [String: Any] = [
            "id": item.id, "q": item.q, "prompt_tokens": ids.count, "image_tokens": ids.filter { $0 == 200_025 }.count,
            "gen_tokens": run.tokens.count, "ended": run.tokens.last == eosToken, "think": run.tokens.first == 200_059,
            "text": text, "prefill_ms": run.prefillMs, "decode_tok_s": run.decodeTokS,
            "prefill_tok_s": Double(ids.count) / (run.prefillMs / 1000), "dspark": statsJSON(run.stats),
            "python_tokens_first_diff": firstDiff(run.tokens, pyGen) as Any, "python_gen_tokens": pyGen.count,
            "same_as_python": run.tokens == pyGen,
            "prefill_pieces": prefill.map { $0.pieces.map { [$0.lowerBound, $0.upperBound] } } as Any,
        ]
        log("\(item.id): \(run.tokens.count) tokens (\(run.tokens.last == eosToken ? "eos" : "max")), same as Python "
            + "\(run.tokens == pyGen) (first diff \(firstDiff(run.tokens, pyGen).map(String.init) ?? "-")), DSpark "
            + "\(run.stats == nil ? "off" : "ON"); \(splitThinking(text).1.prefix(140).debugDescription)")
        items.append(r)
        Memory.clearCache()
    }

    // Text follow-ups after an image turn: the image turn's rows reused, text-only suffix, DSpark == plain.
    let followUps: [(String, String)] = [
        ("ocr_invoice", "Who is the invoice billed to?"),
        ("table_temps", "Which city is coldest on Monday, and at what temperature?"),
        ("two_images", "How many circles does the second image have?"),
        ("bar_chart", "Which category has the shortest bar, and what is its value?"),
    ]
    var follow: [[String: Any]] = []
    for (id, question) in followUps {
        guard let (item, text) = answers[id] else { continue }
        let (reasoning, answer) = splitThinking(text)
        let messages = visMessages(item) + [
            Chat.Message(role: .assistant, content: answer, reasoningContent: reasoning), .user(question),
        ]
        let input = try await prepare(context, messages)
        let ids = tokenIds(input)
        let before = recorder.prefills.count
        let (warm, warmIterator) = try greedy(
            model, input, maxTokens: maxTokens, drafter: drafter, coordinator: coordinator)
        var it = warmIterator
        it.storeCacheAfterGeneration(generatedTokenIds: warm.tokens, includeGeneratedBoundary: false)
        let prefill = recorder.prefills[before...].first
        let (cold, _) = try greedy(model, input, maxTokens: maxTokens, drafter: nil, coordinator: nil)
        let reused = prefill?.start ?? -1
        let suffixHasImage = reused >= 0 && ids[reused...].contains(200_025)
        let warmText = context.tokenizer.decode(tokenIds: warm.tokens, skipSpecialTokens: false)
        let r: [String: Any] = [
            "id": id, "q": question, "prompt_tokens": ids.count, "reused": reused,
            "image_run_end": (ids.lastIndex(of: 200_025) ?? -1) + 1, "suffix_has_image_token": suffixHasImage,
            "warm_equals_cold": warm.tokens == cold.tokens, "first_diff": firstDiff(warm.tokens, cold.tokens) as Any,
            "warm_prefill_ms": warm.prefillMs, "cold_prefill_ms": cold.prefillMs, "dspark": statsJSON(warm.stats),
            "warm_decode_tok_s": warm.decodeTokS, "cold_decode_tok_s": cold.decodeTokS, "text": warmText,
        ]
        log("follow-up \(id): reused \(reused) / \(ids.count) (image run ends \(r["image_run_end"]!)), suffix image "
            + "\(suffixHasImage), warm == cold \(warm.tokens == cold.tokens), DSpark \(warm.stats == nil ? "off" : "ON") "
            + "(\(warm.stats?.specSteps ?? 0) spec rounds), first event \(String(format: "%.0f", warm.prefillMs)) vs "
            + "\(String(format: "%.0f", cold.prefillMs)) ms; \(splitThinking(warmText).1.prefix(120).debugDescription)")
        follow.append(r)
        Memory.clearCache()
    }
    try writeJSON(["items": items, "follow_ups": follow, "tf32_env": env["MLX_ENABLE_TF32"] ?? "default"],
        "visgen_\(tag).json")
}
