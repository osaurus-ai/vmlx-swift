// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import Testing

@testable import MLXLLM

/// MiniMax-M3.1 MSA attention microbenchmark on synthetic tensors (no model, no weights): the Swift counterpart of the
/// vMLX Python runtime's `bench/minimax_m3_msa_bench.py`, same method, so the two tables compare directly.
///
/// One sparse layer's attention at the released geometry (64 query heads, 4 KV groups, head dim 128, 4 index heads of
/// 128, blocks of 128, top-16, local 1), bf16, through the layer's own selection and `attend`, with the path forced
/// (`dense` = the mask path, `sparse` = the block-sparse kernels for the call's size):
/// - decode: one query row at the end of Sk cached keys (K / V are views of step-padded buffers, as served);
/// - prefill: an L-row chunk at the end of Sk keys.
///
/// Per call, median [min-max] over probes; a probe is `iters` calls, each depending on the previous one's output as
/// consecutive layers do, after `warmup` discarded probes. "eval" = evaluating the built graph (GPU work and command
/// encoding), "wall" = building + evaluating. Paths alternate inside every probe (order reversed every other probe),
/// so other GPU load hits each path alike. "attn" times the attention given the selection; "layer" adds the selection.
///
/// `VMLX_MINIMAX_MSA_BENCH_SELECT=1` compares the SELECTION paths instead (the Python bench's `--select`):
/// `reference` (the score-sheet ops) vs `kernel` (the admitted selection kernel), attention routed as served (auto).
/// "sel" times the indexer's selection alone (block scores + top-k), "layer" selection + attention; decode at 8k / 32k /
/// 128k and a 2048-row chunk at 8k / 32k / 128k by default.
///
/// Run it from a release build: a debug build times MLX's unoptimized C++ graph encoding, not the kernels. The test
/// target has `#if DEBUG` helpers, hence `-DDEBUG`:
///
///     swift build -c release --target MLXLMTests -Xswiftc -enable-testing -Xswiftc -DDEBUG
///     VMLX_MINIMAX_MSA_BENCH=1 [VMLX_MINIMAX_MSA_BENCH_MODE=decode|prefill] [VMLX_MINIMAX_MSA_BENCH_SK=8192,32768]
///     [VMLX_MINIMAX_MSA_BENCH_PROBES=7] [VMLX_MINIMAX_MSA_BENCH_JSON=<file>] \
///         swift test -c release --skip-build -Xswiftc -enable-testing -Xswiftc -DDEBUG --disable-xctest \
///         --filter MiniMaxM3MSABenchmarkTests
@Suite(
    "MiniMax-M3.1 MSA microbenchmark (synthetic)", .serialized,
    .enabled(if: ProcessInfo.processInfo.environment["VMLX_MINIMAX_MSA_BENCH"] == "1"))
struct MiniMaxM3MSABenchmarkTests {
    static let heads = 64, groups = 4, headDim = 128, indexDim = 128
    static let environment = ProcessInfo.processInfo.environment

    struct Row: Encodable {
        let mode: String, sk: Int, rows: Int, groups: Int, path: String
        let attnEvalMs: [Double], attnWallMs: [Double], layerEvalMs: [Double], layerWallMs: [Double]
        let attnEvalProbes: [Double], layerEvalProbes: [Double]
        let peakMiBAttn: Int, peakMiBLayer: Int
    }

    static func summary(_ t: [Double]) -> [Double] {
        let s = t.sorted()
        let median = s.count % 2 == 1 ? s[s.count / 2] : (s[s.count / 2 - 1] + s[s.count / 2]) / 2
        return [median, s.first!, s.last!]
    }

    static func synchronize() { Stream.defaultStream(.gpu).synchronize() }

    /// (wall ms, eval ms) per call for `iters` dependent calls.
    static func probe(_ iters: Int, _ call: (Int, MLXArray?) -> MLXArray) -> (Double, Double) {
        synchronize()
        let t0 = DispatchTime.now().uptimeNanoseconds
        var previous: MLXArray?
        for i in 0 ..< iters { previous = call(i, previous) }
        let t1 = DispatchTime.now().uptimeNanoseconds
        eval(previous!)
        synchronize()
        let t2 = DispatchTime.now().uptimeNanoseconds
        return (Double(t2 - t0) / Double(iters) / 1e6, Double(t2 - t1) / Double(iters) / 1e6)
    }

    struct SelectRow: Encodable {
        let mode: String, sk: Int, rows: Int, groups: Int, path: String, arithmetic: String
        let selEvalMs: [Double], selWallMs: [Double], layerEvalMs: [Double], layerWallMs: [Double]
        let selEvalProbes: [Double], layerEvalProbes: [Double]
        let peakMiBSel: Int, peakMiBLayer: Int
    }

    @Test("selection reference vs kernel: selection and layer, decode 8k / 32k / 128k, 2048-row chunk 8k / 32k / 128k",
          .enabled(if: ProcessInfo.processInfo.environment["VMLX_MINIMAX_MSA_BENCH_SELECT"] == "1"))
    func selectionBenchmark() throws {
        let probes = Int(Self.environment["VMLX_MINIMAX_MSA_BENCH_PROBES"] ?? "") ?? 7
        let warmup = Int(Self.environment["VMLX_MINIMAX_MSA_BENCH_WARMUP"] ?? "") ?? 1
        let modes = (Self.environment["VMLX_MINIMAX_MSA_BENCH_MODE"] ?? "decode,prefill").split(separator: ",")
        let skOverride = Self.environment["VMLX_MINIMAX_MSA_BENCH_SK"].map {
            $0.split(separator: ",").compactMap { Int($0) }
        }
        let layer = try MiniMaxM3MSABlockSparseTests.attention(groups: Self.groups)
        let indexer = try #require(layer.indexer)
        let paths: [MiniMaxM3MSA.SelectRequest] = [.reference, .kernel]
        let arithmetic = MiniMaxM3MSA.selectArithmetic(.bfloat16, headDim: Self.indexDim)?.rawValue ?? "none"
        var results: [SelectRow] = []
        try Device.withDefaultDevice(.gpu) {
            for mode in modes {
                let decode = mode == "decode"
                let rows = decode ? 1 : 2048
                let iters = decode ? 32 : 1
                for sk in skOverride ?? [8192, 32768, 131072] {
                    MLXRandom.seed(UInt64(sk))
                    let step = 256
                    let capacity = (sk + step - 1) / step * step + step
                    let kbuf = MLXRandom.normal([1, Self.groups, capacity, Self.headDim]).asType(.bfloat16)
                    let vbuf = MLXRandom.normal([1, Self.groups, capacity, Self.headDim]).asType(.bfloat16)
                    let ibuf = MLXRandom.normal([1, 1, capacity, Self.indexDim]).asType(.bfloat16)
                    let q = (MLXRandom.normal([1, Self.heads, rows, Self.headDim]) * Float(0.3)).asType(.bfloat16)
                    let iq = MLXRandom.normal([1, Self.groups, rows, Self.indexDim]).asType(.bfloat16)
                    eval(kbuf, vbuf, ibuf, q, iq)
                    let (k, v, ik) = (
                        kbuf[0..., 0..., ..<sk], vbuf[0..., 0..., ..<sk], ibuf[0..., 0..., ..<sk]
                    )
                    let offset = sk - rows
                    let mask: MLXFast.ScaledDotProductAttentionMaskMode = rows > 1 ? .causal : .none
                    let eps = MLXArray(Float(1e-3)).asType(.bfloat16)
                    let selectOnly: (Int, MLXArray?) -> MLXArray = { _, previous in
                        let iqi = previous.map { iq + eps * $0[.ellipsis, ..<1].asType(iq.dtype) } ?? iq
                        return indexer.select(queries: iqi, keys: ik, offset: offset)!.topIndices
                    }
                    let fullLayer: (Int, MLXArray?) -> MLXArray = { _, previous in
                        let iqi = previous.map { iq + eps * $0[0..., ..<Self.groups, 0..., ..<Self.indexDim] } ?? iq
                        let selection = indexer.select(queries: iqi, keys: ik, offset: offset)
                        let qi = previous.map { q + eps * $0 } ?? q
                        return layer.attend(qi, k, v, selection: selection, mask: mask)
                    }
                    var times: [String: [Double]] = [:]
                    var peaks: [String: Int] = [:]
                    for n in 0 ..< (warmup + probes) {
                        let order = n % 2 == 0 ? paths : paths.reversed()
                        for path in order {
                            MiniMaxM3MSASelectTests.withSelect(path) {
                                MiniMaxM3MSABlockSparseTests.withRequest(.auto) {
                                    for (part, call) in [("sel", selectOnly), ("layer", fullLayer)] {
                                        GPU.resetPeakMemory()
                                        let (wall, evalTime) = Self.probe(iters, call)
                                        peaks["\(path) \(part)"] = Memory.peakMemory / (1 << 20)
                                        if n >= warmup {
                                            times["\(path) \(part) wall", default: []].append(wall)
                                            times["\(path) \(part) eval", default: []].append(evalTime)
                                        }
                                    }
                                }
                            }
                        }
                    }
                    for path in paths {
                        let p = path.rawValue
                        let row = SelectRow(
                            mode: String(mode), sk: sk, rows: rows, groups: Self.groups, path: p,
                            arithmetic: arithmetic,
                            selEvalMs: Self.summary(times["\(path) sel eval"]!),
                            selWallMs: Self.summary(times["\(path) sel wall"]!),
                            layerEvalMs: Self.summary(times["\(path) layer eval"]!),
                            layerWallMs: Self.summary(times["\(path) layer wall"]!),
                            selEvalProbes: times["\(path) sel eval"]!, layerEvalProbes: times["\(path) layer eval"]!,
                            peakMiBSel: peaks["\(path) sel"]!, peakMiBLayer: peaks["\(path) layer"]!)
                        results.append(row)
                        func f(_ s: [Double]) -> String { String(format: "%8.3f [%.3f-%.3f]", s[0], s[1], s[2]) }
                        print(
                            String(format: "[MSA select bench] %@ Sk=%6d L=%4d G=%d %@", String(mode), sk, rows,
                                Self.groups, p.padding(toLength: 9, withPad: " ", startingAt: 0))
                                + "  sel-eval \(f(row.selEvalMs))  sel-wall \(f(row.selWallMs))"
                                + "  layer-eval \(f(row.layerEvalMs))  layer-wall \(f(row.layerWallMs))"
                                + "  peak \(row.peakMiBSel)/\(row.peakMiBLayer) MiB (\(arithmetic))")
                    }
                    Memory.clearCache()
                }
            }
        }
        #expect(!results.isEmpty)
        if let path = Self.environment["VMLX_MINIMAX_MSA_BENCH_JSON"] {
            let encoder = JSONEncoder()
            encoder.keyEncodingStrategy = .convertToSnakeCase
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            try encoder.encode(results).write(to: URL(fileURLWithPath: path))
        }
    }

    @Test("dense vs sparse: attention and layer, decode 8k / 32k / 128k, 2048-row prefill chunk 8k / 32k",
          .enabled(if: ProcessInfo.processInfo.environment["VMLX_MINIMAX_MSA_BENCH_SELECT"] != "1"))
    func benchmark() throws {
        let probes = Int(Self.environment["VMLX_MINIMAX_MSA_BENCH_PROBES"] ?? "") ?? 7
        let warmup = Int(Self.environment["VMLX_MINIMAX_MSA_BENCH_WARMUP"] ?? "") ?? 1
        let modes = (Self.environment["VMLX_MINIMAX_MSA_BENCH_MODE"] ?? "decode,prefill").split(separator: ",")
        let skOverride = Self.environment["VMLX_MINIMAX_MSA_BENCH_SK"].map {
            $0.split(separator: ",").compactMap { Int($0) }
        }
        let layer = try MiniMaxM3MSABlockSparseTests.attention(groups: Self.groups)
        let indexer = try #require(layer.indexer)
        let paths: [MiniMaxM3MSA.Request] = [.dense, .sparse]
        var results: [Row] = []
        try Device.withDefaultDevice(.gpu) {
            for mode in modes {
                let decode = mode == "decode"
                let rows = decode ? 1 : 2048
                let iters = decode ? 32 : 1
                for sk in skOverride ?? (decode ? [8192, 32768, 131072] : [8192, 32768]) {
                    MLXRandom.seed(UInt64(sk))
                    let step = 256
                    let capacity = (sk + step - 1) / step * step + step
                    let kbuf = MLXRandom.normal([1, Self.groups, capacity, Self.headDim]).asType(.bfloat16)
                    let vbuf = MLXRandom.normal([1, Self.groups, capacity, Self.headDim]).asType(.bfloat16)
                    let ibuf = MLXRandom.normal([1, 1, capacity, Self.indexDim]).asType(.bfloat16)
                    let q = (MLXRandom.normal([1, Self.heads, rows, Self.headDim]) * Float(0.3)).asType(.bfloat16)
                    let iq = MLXRandom.normal([1, Self.groups, rows, Self.indexDim]).asType(.bfloat16)
                    eval(kbuf, vbuf, ibuf, q, iq)
                    let (k, v, ik) = (
                        kbuf[0..., 0..., ..<sk], vbuf[0..., 0..., ..<sk], ibuf[0..., 0..., ..<sk]
                    )
                    let offset = sk - rows
                    let selection0 = try #require(indexer.select(queries: iq, keys: ik, offset: offset))
                    eval(selection0.topIndices)
                    let mask: MLXFast.ScaledDotProductAttentionMaskMode = rows > 1 ? .causal : .none
                    let eps = MLXArray(Float(1e-3)).asType(.bfloat16)
                    func perturbed(_ previous: MLXArray?) -> MLXArray { previous.map { q + eps * $0 } ?? q }
                    let attnOnly: (Int, MLXArray?) -> MLXArray = { _, previous in
                        layer.attend(perturbed(previous), k, v, selection: selection0, mask: mask)
                    }
                    let fullLayer: (Int, MLXArray?) -> MLXArray = { _, previous in
                        let iqi = previous.map { iq + eps * $0[0..., ..<Self.groups, 0..., ..<Self.indexDim] } ?? iq
                        let selection = indexer.select(queries: iqi, keys: ik, offset: offset)
                        return layer.attend(perturbed(previous), k, v, selection: selection, mask: mask)
                    }
                    var times: [String: [Double]] = [:]
                    var peaks: [String: Int] = [:]
                    for n in 0 ..< (warmup + probes) {
                        let order = n % 2 == 0 ? paths : paths.reversed()
                        for path in order {
                            MiniMaxM3MSABlockSparseTests.withRequest(path) {
                                for (part, call) in [("attn", attnOnly), ("layer", fullLayer)] {
                                    GPU.resetPeakMemory()
                                    let (wall, evalTime) = Self.probe(iters, call)
                                    peaks["\(path) \(part)"] = Memory.peakMemory / (1 << 20)
                                    if n >= warmup {
                                        times["\(path) \(part) wall", default: []].append(wall)
                                        times["\(path) \(part) eval", default: []].append(evalTime)
                                    }
                                }
                            }
                        }
                    }
                    for path in paths {
                        let p = path.rawValue
                        let row = Row(
                            mode: String(mode), sk: sk, rows: rows, groups: Self.groups, path: p,
                            attnEvalMs: Self.summary(times["\(path) attn eval"]!),
                            attnWallMs: Self.summary(times["\(path) attn wall"]!),
                            layerEvalMs: Self.summary(times["\(path) layer eval"]!),
                            layerWallMs: Self.summary(times["\(path) layer wall"]!),
                            attnEvalProbes: times["\(path) attn eval"]!, layerEvalProbes: times["\(path) layer eval"]!,
                            peakMiBAttn: peaks["\(path) attn"]!, peakMiBLayer: peaks["\(path) layer"]!)
                        results.append(row)
                        func f(_ s: [Double]) -> String { String(format: "%8.3f [%.3f-%.3f]", s[0], s[1], s[2]) }
                        print(
                            String(format: "[MSA bench] %@ Sk=%6d L=%4d G=%d %@", String(mode), sk, rows, Self.groups,
                                p.padding(toLength: 6, withPad: " ", startingAt: 0))
                                + "  attn-eval \(f(row.attnEvalMs))  attn-wall \(f(row.attnWallMs))"
                                + "  layer-eval \(f(row.layerEvalMs))  layer-wall \(f(row.layerWallMs))"
                                + "  peak \(row.peakMiBAttn)/\(row.peakMiBLayer) MiB")
                    }
                    Memory.clearCache()
                }
            }
        }
        #expect(!results.isEmpty)
        if let path = Self.environment["VMLX_MINIMAX_MSA_BENCH_JSON"] {
            let encoder = JSONEncoder()
            encoder.keyEncodingStrategy = .convertToSnakeCase
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            try encoder.encode(results).write(to: URL(fileURLWithPath: path))
        }
    }
}
