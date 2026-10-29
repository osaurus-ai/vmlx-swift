// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Test harness (not shipped): MiniMax-M3.1 Swift runtime on real-weight 5-layer JANGHT bundles.
//
//   T5_BUNDLE=<bundle> T5_OUT=<dir> [T5_TOKENS=<tokens.safetensors>] [T5_LENS=8269,32845] \
//       .build/release/M3RealT5 load|parity|partial|speed|floordiv|gathersort
//
// parity   prefill in 2048-token chunks then 16 teacher-forced decode steps, per MSA path (T5_PATHS=dense,auto):
//          log-probs of the last chunk + decode steps (lp_N), and for every position the argmax (am_N) and the
//          log-prob of the next token (tlp_N); written to $T5_OUT/swift_<T5_TAG>_<path>.safetensors.
// partial  cache partial hit on real weights: A+B prefilled, trimmed / restored to A, then C (+ decode); must be
//          bit-identical to a cache that only ever saw A, then C (same chunk schedule).
// speed    TokenIterator (greedy, prefill step 2048): cold prefill pp/s and decode tok/s (64 steps), MSA paths
//          interleaved, 1 discarded + T5_PROBES probes.
// floordiv integer floorDivide vs a shift and the composed op's primitives, per call, dependent chains.
// gathersort SwitchLayers.gatherSort as shipped vs a floorDivide-free variant: bit identity + per-call cost.

import Foundation
import MLX
import MLXHuggingFace
import MLXLMCommon
import MLXNN
import VMLXTokenizers

@testable import MLXLLM

let env = ProcessInfo.processInfo.environment
let command = CommandLine.arguments.count > 1 ? CommandLine.arguments[1] : "help"

func log(_ s: String) {
    print("[t5] \(s)")
    fflush(stdout)
}

func nowMs() -> Double { Double(DispatchTime.now().uptimeNanoseconds) / 1e6 }

func median(_ v: [Double]) -> Double {
    let s = v.sorted()
    return s.count % 2 == 1 ? s[s.count / 2] : (s[s.count / 2 - 1] + s[s.count / 2]) / 2
}

func physFootprintMiB() -> Double {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
    let kr = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
        }
    }
    return kr == KERN_SUCCESS ? Double(info.phys_footprint) / 1_048_576 : -1
}

func mib(_ bytes: Int) -> Double { Double(bytes) / 1_048_576 }

func setMSA(_ path: String) {
    if path == "auto" { unsetenv("VMLX_MINIMAX_MSA") } else { setenv("VMLX_MINIMAX_MSA", path, 1) }
}

func writeJSON(_ object: Any, _ name: String) throws {
    let url = outDir.appendingPathComponent(name)
    let data = try JSONSerialization.data(withJSONObject: object, options: [.prettyPrinted, .sortedKeys])
    try data.write(to: url)
    log("wrote \(url.path)")
}

struct NullTokenizer: MLXLMCommon.Tokenizer {
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }
    func encode(text: String, addSpecialTokens: Bool) -> [Int] { text.utf8.map { Int($0) } }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { nil }
    func applyChatTemplate(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [] }
}

struct NullTokenizerLoader: MLXLMCommon.TokenizerLoader {
    func load(from directory: URL) async throws -> any MLXLMCommon.Tokenizer { NullTokenizer() }
}

let outDir = URL(fileURLWithPath: env["T5_OUT"] ?? FileManager.default.currentDirectoryPath)
let tag = env["T5_TAG"] ?? "t5"
let lens = (env["T5_LENS"] ?? "8269,32845").split(separator: ",").map { Int($0)! }
let chunk = Int(env["T5_CHUNK"] ?? "2048")!
let decodeSteps = Int(env["T5_DECODE"] ?? "16")!

func bundleURL() -> URL {
    guard let path = env["T5_BUNDLE"] else { fatalError("T5_BUNDLE not set") }
    return URL(fileURLWithPath: path)
}

func loadTokens() throws -> [Int32] {
    guard let path = env["T5_TOKENS"] else { fatalError("T5_TOKENS not set") }
    let arrays = try loadArrays(url: URL(fileURLWithPath: path))
    return arrays["tokens"]!.asType(.int32).asArray(Int32.self)
}

func loadModel(realTokenizer: Bool = false) async throws -> (MiniMaxM3Model, ModelContext, Double) {
    let t0 = nowMs()
    let context: ModelContext
    if realTokenizer {
        context = try await LLMModelFactory.shared.load(
            from: bundleURL(), using: #huggingFaceTokenizerLoader())
    } else {
        context = try await LLMModelFactory.shared.load(from: bundleURL(), using: NullTokenizerLoader())
    }
    let t = nowMs() - t0
    guard let model = context.model as? MiniMaxM3Model else {
        fatalError("loaded \(type(of: context.model)), not MiniMaxM3Model")
    }
    return (model, context, t)
}

/// log-softmax rows in float32
func logProbs(_ logits: MLXArray) -> MLXArray {
    let x = logits.asType(.float32)
    return x - logSumExp(x, axis: -1, keepDims: true)
}

struct Probe {
    var lastLP: [MLXArray] = []  // last chunk rows + decode rows
    var argmax: [MLXArray] = []
    var targetLP: [MLXArray] = []
}

/// Feeds `tokens[start ..< end]` in `chunk`-token pieces into `cache`; records argmax / next-token log-prob for every
/// position (targets `tokens[p + 1]`), and the full log-probs of the final piece when `keepLast`.
func feed(
    _ model: MiniMaxM3Model, _ cache: [KVCache], _ tokens: [Int32], _ start: Int, _ end: Int,
    _ probe: inout Probe, keepLast: Bool
) {
    var s = start
    while s < end {
        let e = min(s + chunk, end)
        let x = MLXArray(Array(tokens[s ..< e])).reshaped(1, e - s)
        let lp = logProbs(model(x, cache: cache)[0])
        let target = MLXArray(Array(tokens[(s + 1) ..< (e + 1)])).reshaped(-1, 1)
        let am = argMax(lp, axis: -1).asType(.int32)
        let tlp = takeAlong(lp, target, axis: -1).reshaped(-1)
        if keepLast && e == end {
            eval(am, tlp, lp)
            probe.lastLP.append(lp)
        } else {
            eval(am, tlp)
        }
        probe.argmax.append(am)
        probe.targetLP.append(tlp)
        s = e
    }
}

func teacherForcedDecode(_ model: MiniMaxM3Model, _ cache: [KVCache], _ tokens: [Int32], from n: Int, steps: Int, _ probe: inout Probe) {
    for t in 0 ..< steps {
        let x = MLXArray([tokens[n + t]]).reshaped(1, 1)
        let lp = logProbs(model(x, cache: cache)[0])
        let am = argMax(lp, axis: -1).asType(.int32)
        let tlp = takeAlong(lp, MLXArray([tokens[n + t + 1]]).reshaped(1, 1), axis: -1).reshaped(-1)
        eval(am, tlp, lp)
        probe.lastLP.append(lp)
        probe.argmax.append(am)
        probe.targetLP.append(tlp)
    }
}

// MARK: - load

func taskLoad() async throws {
    Memory.peakMemory = 0
    let before = physFootprintMiB()
    let (model, context, loadMs) = try await loadModel(realTokenizer: true)
    let afterLoad = physFootprintMiB()
    let activeAfterLoad = mib(Memory.activeMemory)
    var report: [String: Any] = [
        "bundle": bundleURL().path, "load_ms": loadMs, "phys_footprint_before_MiB": before,
        "phys_footprint_after_load_MiB": afterLoad, "mlx_active_after_load_MiB": activeAfterLoad,
        "model_type": String(describing: type(of: context.model)),
        "tool_call_format": String(describing: context.configuration.toolCallFormat),
        "reasoning_parser": context.configuration.reasoningParserName ?? "nil",
    ]
    // Routed experts / routers.
    var moe: [String] = []
    for (index, layer) in model.model.layers.enumerated() {
        if let block = layer.mlp as? MiniMaxM3SparseMoeBlock {
            moe.append(
                "\(index):\(type(of: block.experts)) router=\(block.gate.weight.dtype) quantized=\(block.gate is QuantizedLinear) bias=\(block.eScoreCorrectionBias.dtype)"
            )
        } else {
            moe.append("\(index):\(type(of: layer.mlp))")
        }
    }
    report["layers"] = moe
    report["sparse_layers"] = model.model.layers.map { $0.selfAttn.indexer != nil }

    // Every quantized module carries its plan entry; nothing else is quantized.
    let config = try JSONSerialization.jsonObject(
        with: Data(contentsOf: bundleURL().appendingPathComponent("config.json"))) as! [String: Any]
    let plan = config["quantization"] as! [String: Any]
    var expected: [String: String] = [:]
    var routed = Set<String>()
    for (path, entry) in plan {
        guard let entry = entry as? [String: Any] else { continue }
        if entry["mode"] as? String == "jangt" {
            routed.insert(path)
        } else {
            expected[path] = "\(entry["mode"] as? String ?? "affine")/\(entry["bits"]!)/\(entry["group_size"]!)"
        }
    }
    var actual: [String: String] = [:]
    for (path, module) in model.namedModules() {
        if let q = module as? Quantized {
            actual[path] = "\(q.mode.rawValue)/\(q.bits)/\(q.groupSize)"
        }
    }
    let planMismatch = Set(expected.keys).union(actual.keys).filter { expected[$0] != actual[$0] }.sorted()
    report["plan_entries"] = expected.count
    report["plan_routed"] = routed.count
    report["plan_mismatch"] = planMismatch.map { "\($0): plan \(expected[$0] ?? "-") module \(actual[$0] ?? "-")" }

    // Every text tensor of the bundle is bound as stored (values compared), banks are mapped, vision dropped.
    let index = try JSONSerialization.jsonObject(
        with: Data(contentsOf: bundleURL().appendingPathComponent("model.safetensors.index.json"))) as! [String: Any]
    let weightMap = index["weight_map"] as! [String: String]
    let parameters = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
    let isVision: (String) -> Bool = {
        $0.hasPrefix("vision_tower.") || $0.hasPrefix("multi_modal_projector.") || $0.hasPrefix("patch_merge_mlp.")
    }
    let banked = Set(weightMap.keys.filter { $0.contains(".jt_") })
    let text = Set(weightMap.keys.filter { !$0.contains(".jt_") && !isVision($0) })
    report["index_tensors"] = weightMap.count
    report["vision_tensors_dropped"] = weightMap.keys.filter(isVision).count
    report["banked_tensors"] = banked.count
    report["banked_not_in_exclusion_set"] = banked.subtracting(model.customTensorNames).sorted()
    report["unbound_text_tensors"] = text.subtracting(parameters.keys).sorted()
    report["parameters_not_from_bundle"] = Set(parameters.keys).subtracting(text).sorted()
    var valueMismatch: [String] = []
    var compared = 0
    for shard in Set(weightMap.values).sorted() {
        let stored = try loadArrays(url: bundleURL().appendingPathComponent(shard))
        for (key, value) in stored where text.contains(key) {
            guard let bound = parameters[key] else { continue }
            if bound.dtype != value.dtype || bound.shape != value.shape || !arrayEqual(bound, value).item(Bool.self) {
                valueMismatch.append("\(key) bound \(bound.dtype) \(bound.shape) stored \(value.dtype) \(value.shape)")
            }
            compared += 1
        }
    }
    report["values_compared"] = compared
    report["value_mismatch"] = valueMismatch
    report["phys_footprint_after_compare_MiB"] = physFootprintMiB()

    // A first forward: 3011 tokens (> 2048: MSA selects) + 4 decode steps, finite.
    let probeLen = 3011
    let ids = (0 ..< probeLen + 5).map { Int32(($0 * 7919 + 13) % 200000) }
    var probe = Probe()
    let cache = model.newCache(parameters: nil)
    let t0 = nowMs()
    feed(model, cache, ids, 0, probeLen, &probe, keepLast: true)
    teacherForcedDecode(model, cache, ids, from: probeLen, steps: 4, &probe)
    report["first_forward_ms"] = nowMs() - t0
    let lp = concatenated(probe.lastLP, axis: 0)
    report["first_forward_finite"] = isFinite(lp).all().item(Bool.self)
    report["cache_types"] = Array(Set(cache.map { String(describing: type(of: $0)) }))
    report["cache_offsets"] = cache.map { $0.offset }
    report["mlx_active_after_forward_MiB"] = mib(Memory.activeMemory)
    report["mlx_peak_MiB"] = mib(Memory.peakMemory)
    report["phys_footprint_after_forward_MiB"] = physFootprintMiB()

    // Tokenizer round trip.
    let sample = "The capital of France is Paris. 17 + 25 = 42."
    let encoded = context.tokenizer.encode(text: sample, addSpecialTokens: false)
    report["tokenizer_sample_ids"] = encoded
    report["tokenizer_roundtrip"] = context.tokenizer.decode(tokenIds: encoded, skipSpecialTokens: false) == sample

    // The generic entry point (what an app calls without naming a factory).
    do {
        let t1 = nowMs()
        let generic = try await MLXLMCommon.loadModel(from: bundleURL(), using: NullTokenizerLoader())
        report["generic_loadModel"] = "\(type(of: generic.model)) in \(Int(nowMs() - t1)) ms"
    } catch {
        report["generic_loadModel"] = "FAILED: \(error)"
    }
    try writeJSON(report, "load_\(tag).json")
    log("load \(tag): \(Int(loadMs)) ms, plan mismatches \(planMismatch.count), unbound \((report["unbound_text_tensors"] as! [String]).count), value mismatches \(valueMismatch.count)")
}

// MARK: - parity

func taskParity() async throws {
    let tokens = try loadTokens()
    let (model, _, loadMs) = try await loadModel()
    log("loaded in \(Int(loadMs)) ms; tokens \(tokens.count)")
    let paths = (env["T5_PATHS"] ?? "dense,auto").split(separator: ",").map(String.init)
    PathSpy.shared.install()
    var trace: [String: Any] = [:]
    for path in paths {
        setMSA(path)
        var arrays: [String: MLXArray] = [:]
        for n in lens {
            _ = PathSpy.shared.drain()
            precondition(tokens.count >= n + decodeSteps + 1)
            var probe = Probe()
            let cache = model.newCache(parameters: nil)
            let t0 = nowMs()
            feed(model, cache, tokens, 0, n, &probe, keepLast: true)
            teacherForcedDecode(model, cache, tokens, from: n, steps: decodeSteps, &probe)
            arrays["lp_\(n)"] = concatenated(probe.lastLP, axis: 0)
            arrays["am_\(n)"] = concatenated(probe.argmax, axis: 0)
            arrays["tlp_\(n)"] = concatenated(probe.targetLP, axis: 0)
            eval(arrays["lp_\(n)"]!, arrays["am_\(n)"]!, arrays["tlp_\(n)"]!)
            let picks = PathSpy.shared.drain()
            trace["\(path)_\(n)"] = picks
            log("parity \(path) n=\(n): \(Int(nowMs() - t0)) ms, offsets \(Set(cache.map { $0.offset })), paths \(Dictionary(picks.map { ($0, 1) }, uniquingKeysWith: +))")
            Memory.clearCache()
        }
        try save(arrays: arrays, url: outDir.appendingPathComponent("swift_\(tag)_\(path).safetensors"))
    }
    unsetenv("VMLX_MINIMAX_MSA")
    try writeJSON(trace, "swift_paths_\(tag).json")
}

// MARK: - partial hit

func taskPartial() async throws {
    let tokens = try loadTokens()
    let (model, _, _) = try await loadModel()
    let la = Int(env["T5_PARTIAL_A"] ?? "8269")!
    let lb = Int(env["T5_PARTIAL_B"] ?? "3001")!
    let lc = Int(env["T5_PARTIAL_C"] ?? "4133")!
    let cStart = Int(env["T5_PARTIAL_C_START"] ?? "20000")!
    // Sequence for the C runs: A then C's tokens (C taken from elsewhere in the corpus), then decode tokens.
    let seqAC = Array(tokens[0 ..< la]) + Array(tokens[cStart ..< (cStart + lc + decodeSteps + 1)])
    let seqAB = Array(tokens[0 ..< (la + lb + 1)])
    var report: [String: Any] = ["A": la, "B": lb, "C": lc, "C_start": cStart, "chunk": chunk, "decode": decodeSteps]
    let paths = (env["T5_PATHS"] ?? "auto,dense").split(separator: ",").map(String.init)

    func runC(_ cache: [KVCache]) -> MLXArray {
        var probe = Probe()
        feed(model, cache, seqAC, la, la + lc, &probe, keepLast: true)
        teacherForcedDecode(model, cache, seqAC, from: la + lc, steps: decodeSteps, &probe)
        let lp = concatenated(probe.lastLP, axis: 0)
        let am = concatenated(probe.argmax, axis: 0)
        let tl = concatenated(probe.targetLP, axis: 0)
        eval(lp, am, tl)
        return concatenated([lp.flattened(), am.asType(.float32), tl], axis: 0)
    }
    func prefillA() -> [KVCache] {
        let cache = model.newCache(parameters: nil)
        var probe = Probe()
        feed(model, cache, seqAC, 0, la, &probe, keepLast: false)
        return cache
    }
    func addB(_ cache: [KVCache]) {
        var probe = Probe()
        feed(model, cache, seqAB, la, la + lb, &probe, keepLast: false)
    }
    for path in paths {
        setMSA(path)
        var r: [String: Any] = [:]
        // fresh: A then C
        let fresh = runC(prefillA())
        // trim: A + B, trim B
        var cache = prefillA()
        addB(cache)
        let trimmed = cache.map { $0.trim(lb) }
        r["trim_returned"] = Array(Set(trimmed))
        r["offsets_after_trim"] = Array(Set(cache.map { $0.offset }))
        let trimRun = runC(cache)
        // restore: snapshot (copy) after A, B into the live cache, C on the snapshot
        cache = prefillA()
        let snapshot = cache.map { $0.copy() }
        addB(cache)
        let restoreRun = runC(snapshot)
        // state round trip: state taken after A + B and trimmed, set into new caches (disk-restore shape)
        cache = prefillA()
        addB(cache)
        let restored: [KVCache] = cache.map { c in
            let sliced = c.state.map { $0[0..., 0..., ..<la, 0...] * 1 }
            if c is MiniMaxM3SparseCache {
                let copy = MiniMaxM3SparseCache()
                copy.state = sliced
                return copy
            } else {
                let copy = KVCacheSimple()
                copy.state = sliced
                return copy
            }
        }
        let stateRun = runC(restored)
        // control: one B token left in the cache (trim lb - 1) must differ
        cache = prefillA()
        addB(cache)
        _ = cache.map { $0.trim(lb - 1) }
        var controlProbe = Probe()
        let seqControl = Array(seqAB[0 ..< (la + 1)]) + Array(seqAC[la ..< seqAC.count])
        feed(model, cache, seqControl, la + 1, la + 1 + lc, &controlProbe, keepLast: true)
        let controlLP = concatenated(controlProbe.lastLP, axis: 0)
        // The control's last-chunk rows sit one position later; compare its log-probs at the same C tokens.
        let freshLP = fresh[0 ..< controlLP.size].reshaped(controlLP.shape)
        let controlDiff = abs(controlLP - freshLP).max().item(Float.self)
        for (name, run) in [("trim", trimRun), ("restore_copy", restoreRun), ("state_roundtrip", stateRun)] {
            let d = abs(run - fresh).max().item(Float.self)
            r["\(name)_maxabs"] = d
            r["\(name)_bit_identical"] = d == 0 && arrayEqual(run, fresh).item(Bool.self)
        }
        r["control_one_stale_token_maxabs"] = controlDiff
        r["fresh_values"] = fresh.size
        report[path] = r
        log("partial \(path): \(r)")
        Memory.clearCache()
    }
    unsetenv("VMLX_MINIMAX_MSA")
    try writeJSON(report, "partial_\(tag).json")
}

// MARK: - speed

func taskSpeed() async throws {
    let tokens = try loadTokens()
    let (model, _, _) = try await loadModel()
    let probes = Int(env["T5_PROBES"] ?? "3")!
    let steps = Int(env["T5_STEPS"] ?? "64")!
    let paths = (env["T5_PATHS"] ?? "dense,auto").split(separator: ",").map(String.init)
    var report: [String: Any] = ["probes": probes, "discarded": 1, "steps": steps, "chunk": chunk]
    for n in lens {
        var pp: [String: [Double]] = [:], tg: [String: [Double]] = [:], pf: [String: [Double]] = [:]
        var firstTokens: [String: [Int]] = [:]
        for i in 0 ... probes {
            let order = i % 2 == 0 ? paths : paths.reversed()
            for path in order {
                setMSA(path)
                Memory.clearCache()
                let prompt = MLXArray(Array(tokens[0 ..< n]))
                var parameters = GenerateParameters(maxTokens: steps + 1, temperature: 0, prefillStepSize: chunk)
                parameters.maxTokens = steps + 1
                let t0 = nowMs()
                var iterator = try TokenIterator(prompt: prompt, model: model, parameters: parameters)
                guard let first = iterator.next() else { fatalError("no first token") }
                let t1 = nowMs()
                var generated = [first]
                for _ in 0 ..< steps {
                    guard let token = iterator.next() else { break }
                    generated.append(token)
                }
                let t2 = nowMs()
                if i > 0 {
                    pp[path, default: []].append(Double(n) / ((t1 - t0) / 1000))
                    tg[path, default: []].append(Double(generated.count - 1) / ((t2 - t1) / 1000))
                    pf[path, default: []].append(t1 - t0)
                }
                if firstTokens[path] == nil { firstTokens[path] = Array(generated.prefix(16)) }
                log("speed n=\(n) probe \(i) \(path): prefill \(Int(t1 - t0)) ms, \(generated.count - 1) decode in \(Int(t2 - t1)) ms")
            }
        }
        var row: [String: Any] = [:]
        for path in paths {
            row[path] = [
                "pp_s": ["runs": pp[path]!, "median": median(pp[path]!)],
                "tok_s": ["runs": tg[path]!, "median": median(tg[path]!)],
                "prefill_ms_median": median(pf[path]!), "first16": firstTokens[path]!,
            ]
        }
        report["\(n)"] = row
        log("speed n=\(n): \(row)")
    }
    unsetenv("VMLX_MINIMAX_MSA")
    try writeJSON(report, "speed_\(tag).json")
}

// MARK: - floorDivide

func taskFloorDiv() throws {
    let iters = Int(env["T5_FD_ITERS"] ?? "50")!
    let probes = Int(env["T5_PROBES"] ?? "7")!
    var report: [String: Any] = ["iters": iters, "probes": probes]
    let four = MLXArray(Int32(4)), three = MLXArray(Int32(3)), two = MLXArray(Int32(2)), zero = MLXArray(Int32(0))
    // Each op maps x -> x (values kept) through the op under test, so a chain of calls is a dependency chain.
    let ops: [(String, (MLXArray) -> MLXArray)] = [
        ("floorDivide_int32", { x in floorDivide(x * four + three, four) }),
        ("rightShift_int32", { x in rightShift(x * four + three, two) }),
        ("method_floorDivide_int32", { x in (x * four + three).floorDivide(4) }),
        ("remainder_int32", { x in x + remainder(x, four) * zero }),
        ("less_int32", { x in x + (x .< zero).asType(.int32) }),
        ("notEqual_int32", { x in x + (x .!= x).asType(.int32) }),
        ("logicalAnd", { x in x + logicalAnd(x .< zero, x .< zero).asType(.int32) }),
        ("subtract_int32", { x in x - zero }),
        ("floorDivide_float32", { x in floorDivide((x * four + three).asType(.float32), MLXArray(Float(4))).asType(.int32) }),
        ("baseline_mul_add", { x in x * MLXArray(Int32(1)) + zero }),
    ]
    for size in [1, 64, 4096, 65536] {
        var row: [String: Any] = [:]
        let x0 = MLXArray(0 ..< Int32(size))
        eval(x0)
        var evalMs: [String: [Double]] = [:], buildMs: [String: [Double]] = [:]
        for p in 0 ... probes {
            let order = p % 2 == 0 ? ops : ops.reversed()
            for (name, op) in order {
                Stream.defaultStream(.gpu).synchronize()
                let t0 = nowMs()
                var x = x0
                for _ in 0 ..< iters { x = op(x) }
                let t1 = nowMs()
                eval(x)
                let t2 = nowMs()
                if p > 0 {
                    buildMs[name, default: []].append((t1 - t0) / Double(iters))
                    evalMs[name, default: []].append((t2 - t1) / Double(iters))
                }
                precondition(arrayEqual(x, x0).item(Bool.self), "\(name) changed values")
            }
        }
        for (name, _) in ops {
            row[name] = ["eval_ms_per_call": median(evalMs[name]!), "build_ms_per_call": median(buildMs[name]!)]
        }
        // One op per eval (a call alone in its own graph).
        var single: [String: [Double]] = [:]
        for p in 0 ... probes {
            for (name, op) in ops where name == "floorDivide_int32" || name == "rightShift_int32" || name == "baseline_mul_add" {
                var t: [Double] = []
                for _ in 0 ..< 20 {
                    let t0 = nowMs()
                    let y = op(x0)
                    eval(y)
                    t.append(nowMs() - t0)
                }
                if p > 0 { single[name, default: []].append(median(t)) }
            }
        }
        for (name, v) in single { row["single_eval_ms_\(name)"] = median(v) }
        report["\(size)"] = row
        log("floordiv size \(size): \(row)")
    }
    // CPU stream: the same chain on the CPU device.
    try Device.withDefaultDevice(.cpu) {
        let x0 = MLXArray(0 ..< Int32(4096))
        eval(x0)
        var t: [Double] = []
        for p in 0 ... probes {
            let t0 = nowMs()
            var x = x0
            for _ in 0 ..< iters { x = floorDivide(x * four + three, four) }
            eval(x)
            if p > 0 { t.append((nowMs() - t0) / Double(iters)) }
        }
        report["cpu_floorDivide_int32_4096_ms_per_call"] = median(t)
    }
    try writeJSON(report, "floordiv_\(tag).json")
}

// MARK: - gatherSort

/// SwitchLayers.gatherSort as shipped before the prototype (integer floorDivide).
func gatherSortFloorDivide(x: MLXArray, indices: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
    let m = indices.dim(-1)
    let flat = indices.flattened()
    let order = argSort(flat)
    let inverseOrder = argSort(order)
    return (x.flattened(start: 0, end: -3)[order.floorDivide(m)], flat[order], inverseOrder)
}

/// gatherSort with the token index computed without integer floorDivide: a shift when the route count is a power of
/// two, else a gather from the repeated token ids (exact for any count).
func gatherSortNoDivide(x: MLXArray, indices: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
    let m = indices.dim(-1)
    let flat = indices.flattened()
    let order = argSort(flat)
    let inverseOrder = argSort(order)
    let token: MLXArray
    if m & (m - 1) == 0 {
        token = rightShift(order, MLXArray(UInt32(m.trailingZeroBitCount)))
    } else {
        let tokens = flat.size / m
        token = take(repeated(MLXArray.arange(tokens), count: m), order)
    }
    return (x.flattened(start: 0, end: -3)[token], flat[order], inverseOrder)
}

func taskGatherSort() throws {
    let probes = Int(env["T5_PROBES"] ?? "7")!
    var report: [String: Any] = [:]
    for (tokens, k) in [(1, 4), (16, 4), (512, 4), (2048, 4), (2048, 8), (2048, 6), (37, 6)] {
        let key = MLXRandom.key(UInt64(tokens * 31 + k))
        let indices = MLXRandom.randInt(Int32(0) ..< Int32(128), [1, tokens, k], key: key)
        let x = expandedDimensions(MLXRandom.normal([1, tokens, 64], key: MLXRandom.key(7)).asType(.bfloat16), axes: [-2, -3])
        let (a0, a1, a2) = gatherSortFloorDivide(x: x, indices: indices)
        let (b0, b1, b2) = gatherSort(x: x, indices: indices)  // the library's (prototype on the branch)
        let (c0, c1, c2) = gatherSortNoDivide(x: x, indices: indices)
        eval(c0, c1, c2)
        precondition(arrayEqual(b0, c0).item(Bool.self) && arrayEqual(b1, c1).item(Bool.self) && arrayEqual(b2, c2).item(Bool.self))
        eval(a0, a1, a2, b0, b1, b2)
        let identical =
            a0.shape == b0.shape && arrayEqual(a0, b0).item(Bool.self) && arrayEqual(a1, b1).item(Bool.self)
            && arrayEqual(a2, b2).item(Bool.self) && a0.dtype == b0.dtype && a1.dtype == b1.dtype
        var ta: [Double] = [], tb: [Double] = []
        for p in 0 ... probes {
            for which in (p % 2 == 0 ? [0, 1] : [1, 0]) {
                var t: [Double] = []
                for _ in 0 ..< 20 {
                    let t0 = nowMs()
                    let r = which == 0 ? gatherSortFloorDivide(x: x, indices: indices) : gatherSort(x: x, indices: indices)
                    eval(r.0, r.1, r.2)
                    t.append(nowMs() - t0)
                }
                if p > 0 { (which == 0 ? { ta.append(median(t)) } : { tb.append(median(t)) })() }
            }
        }
        report["T\(tokens)_k\(k)"] = [
            "bit_identical": identical, "floorDivide_ms": median(ta), "library_ms": median(tb),
        ]
        log("gathersort T\(tokens) k\(k): identical \(identical) floorDivide \(median(ta)) ms library \(median(tb)) ms")
    }
    try writeJSON(report, "gathersort_\(tag).json")
}

// MARK: - main

try FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)
log("command \(command), MLX_ENABLE_TF32=\(env["MLX_ENABLE_TF32"] ?? "unset"), tag \(tag)")
switch command {
case "load": try await taskLoad()
case "parity": try await taskParity()
case "partial": try await taskPartial()
case "speed": try await taskSpeed()
case "floordiv": try taskFloorDiv()
case "gathersort": try taskGatherSort()
case "layers": try await taskLayers()
case "bload": try await taskBundleLoad()
case "evalrows": try await taskEvalRows()
case "dspark": try await taskDSpark()
case "think": try await taskThink()
case "cacheloop": try await taskCacheLoop()
case "longctx": try await taskLongContext()
case "profile": try await taskProfile()
case "vprofile": try await taskVerifyProfile()
case "stepab": try await taskStepAB()
case "vsplit": try await taskVerifySplit()
case "serving": try await taskServing()
case "servingloop": try await taskServingLoop()
case "servingdoc": try await taskServingDoc()
case "reload": try await taskReload()
case "msainfo": log(msaDiagnostics())
case "fdloop":
    // A long loop of one integer floorDivide per eval (for sampling the process from outside).
    let n = Int(env["T5_FD_LOOP"] ?? "3000")!
    let op = env["T5_FD_OP"] ?? "floorDivide"
    let x0 = MLXArray(0 ..< Int32(4096))
    eval(x0)
    let t0 = nowMs()
    for _ in 0 ..< n {
        let y: MLXArray
        switch op {
        case "rightShift": y = rightShift(x0 * MLXArray(Int32(4)) + MLXArray(Int32(3)), MLXArray(Int32(2)))
        case "fdExplicitGPU":
            let g = StreamOrDevice.gpu
            y = floorDivide(add(multiply(x0, MLXArray(Int32(4)), stream: g), MLXArray(Int32(3)), stream: g), MLXArray(Int32(4)), stream: g)
        case "fdRunWith":
            var r = x0
            StreamOrDevice.default.stream.runWith {
                r = floorDivide(x0 * MLXArray(Int32(4)) + MLXArray(Int32(3)), MLXArray(Int32(4)))
            }
            y = r
        case "fdComposed":
            // floor_divide's composition without its Divide: remainder, compares, logical_and, subtract.
            let a = x0 * MLXArray(Int32(4)) + MLXArray(Int32(3)), b = MLXArray(Int32(4)), zero = MLXArray(Int32(0))
            let rem = remainder(a, b)
            let step = logicalAnd(rem .!= zero, (a .< zero) .!= (b .< zero))
            y = rightShift(a, MLXArray(Int32(2))) - step.asType(.int32)
        case "floorDivideU32": y = floorDivide(x0.asType(.uint32) * MLXArray(UInt32(4)), MLXArray(UInt32(4)))
        case "floorDivideI16": y = floorDivide(x0.asType(.int16) * MLXArray(Int16(4)), MLXArray(Int16(4)))
        case "floorDivideI64": y = floorDivide(x0.asType(.int64) * MLXArray(Int64(4)), MLXArray(Int64(4)))
        default: y = floorDivide(x0 * MLXArray(Int32(4)) + MLXArray(Int32(3)), MLXArray(Int32(4)))
        }
        eval(y)
    }
    log("fdloop \(op): \((nowMs() - t0) / Double(n)) ms per eval")
default: log("usage: M3RealT5 load|parity|partial|speed|floordiv|gathersort")
}
log("done")
