// SPDX-License-Identifier: Apache-2.0
// Lane matmul runtime (kernel sources in LaneQMMSources.swift). Port of vMLX Python
// vmlx_engine/metal/lane_qmm.py (4554485c1); see that file and the round-2 audit R2-21 for the
// measurements behind every choice here.

import Foundation
import MLX
import MLXNN
import MLXRandom

extension LaneQMM {

    // MARK: kernels

    /// One compiled kernel per (variant, template constants). Template integers are baked into the source
    /// as `constexpr int` lines (as the Python port does), so `matmul2d_descriptor` sees compile-time values
    /// and each constant set gets its own name (MLX caches compiled kernels by name).
    private static let lock = NSLock()
    nonisolated(unsafe) private static var compiled: [String: MLXFast.MLXFastKernel] = [:]

    private static func kernel(
        _ variant: String, body: String, inputs: [String], outputs: [String],
        constants: [(String, Int)]
    ) -> MLXFast.MLXFastKernel {
        let prefix = constants.map { "  constexpr int \($0.0) = \($0.1);\n" }.joined()
        let source = prefix + body
        let key = variant + "|" + constants.map { "\($0.0)=\($0.1)" }.joined(separator: ",")
        lock.lock()
        defer { lock.unlock() }
        if let k = compiled[key] { return k }
        var hasher = Hasher()
        hasher.combine(source)
        let name = "vmlx_lane_\(variant)_\(UInt(bitPattern: hasher.finalize()))"
        let k = MLXFast.metalKernel(
            name: name, inputNames: inputs, outputNames: outputs,
            source: source, header: header)
        compiled[key] = k
        return k
    }

    // MARK: helpers

    /// K slices for an (n, k) weight: fixed by the shape, never by the row count (row independence).
    static func splitK(n: Int, k: Int) -> Int {
        let tiles = (n + nt - 1) / nt
        var sk = 1
        while sk < 8 && tiles * sk < 1024 && (k / 64) / (sk * 2) >= 8 { sk *= 2 }
        return sk
    }

    /// (N, K/GS) scales and biases -> (K/GS, N, 2) bf16 pairs, group-major.
    public static func packScales(_ scales: MLXArray, _ biases: MLXArray) -> MLXArray {
        stacked([scales.T, biases.T], axis: -1).asType(.bfloat16)
    }

    /// MLX's packed (N, K*bits/32) weight regrouped [N/NT][K/group][NT columns x a group's words]: same bytes.
    public static func tileWeight(_ w: MLXArray, bits: Int, group: Int = 64) -> MLXArray {
        let n = w.dim(0)
        let kw = w.dim(1)
        let words = group * bits / 32
        return contiguous(
            w.reshaped([n / nt, nt, kw / words, words]).transposed(0, 2, 1, 3)
                .reshaped([n, kw]))
    }

    /// ``tileWeight`` undone: MLX's packed layout again.
    public static func untileWeight(_ w: MLXArray, bits: Int, group: Int = 64) -> MLXArray {
        let n = w.dim(0)
        let kw = w.dim(1)
        let words = group * bits / 32
        return contiguous(
            w.reshaped([n / nt, kw / words, nt, words]).transposed(0, 2, 1, 3)
                .reshaped([n, kw]))
    }

    nonisolated(unsafe) private static var availability: Bool?

    /// Metal 4 tensor ops compile here and give the right answer (checked once per process).
    public static func available() -> Bool {
        lock.lock()
        if let a = availability {
            lock.unlock()
            return a
        }
        lock.unlock()
        let ok: Bool = {
            let w = (MLXRandom.normal([64, 128]) * 0.05).asType(.bfloat16)
            let (q, s, b) = MLX.quantized(w, groupSize: 64, bits: 4)
            guard let b else { return false }
            let x = MLXRandom.normal([3, 128]).asType(.bfloat16)
            let y = laneMatmul(
                x, weight: q, sbt: packScales(s.asType(.bfloat16), b.asType(.bfloat16)),
                bits: 4, group: 64, tiled: false)
            let ref = matmul(
                x.asType(.float32),
                dequantized(q, scales: s, biases: b, groupSize: 64, bits: 4).asType(.float32).T)
            return allClose(y.asType(.float32), ref, rtol: 0.05, atol: 0.25).item(Bool.self)
        }()
        lock.lock()
        availability = ok
        lock.unlock()
        return ok
    }

    // MARK: matmul

    /// x (..., K) bf16 times packed `weight` (N, K*bits/32) transposed; at most `maxRows` rows.
    public static func laneMatmul(
        _ x: MLXArray, weight: MLXArray, sbt: MLXArray, bits: Int,
        group: Int = 64, tiled: Bool
    ) -> MLXArray {
        let k = x.dim(-1)
        let n = weight.dim(0)
        let lead = Array(x.shape.dropLast())
        let x2 = x.reshaped([-1, k])
        let m = x2.dim(0)
        precondition(m <= maxRows, "lane matmul takes at most \(maxRows) rows")
        let mp = 16 * ((m + 15) / 16)
        let kg = k / group
        let mdims = MLXArray([Int32(m), Int32(mp)] + [Int32](repeating: 0, count: 14))
        let xs = kernel(
            "xsum", body: xsumSource, inputs: ["X", "mdims"], outputs: ["XS"],
            constants: [("K", k), ("GS", group)])(
                [x2, mdims], grid: (kg, mp, 1), threadGroup: (Swift.min(kg, 256), 1, 1),
                outputShapes: [[kg, mp]], outputDTypes: [.float32])[0]
        let sk = splitK(n: n, k: k)
        let block = mp <= rowBlock ? mp : rowBlock
        let edge = mp % block != 0 ? 1 : 0
        let y: MLXArray
        if bits == 8, group == 64, direct8Enabled {
            y =
                kernel(
                    tiled ? "main8_tiled" : "main8", body: tiled ? main8TiledSource : main8Source,
                    inputs: ["X", "XS", "Wq", "SBt", "mdims"], outputs: ["Y"],
                    constants: [
                        ("TMR", block / 16), ("N", n), ("K", k), ("NT", nt), ("SK", sk),
                        ("GS", group), ("EDGE", edge),
                    ])(
                    [x2, xs, weight, sbt, mdims],
                    grid: (((n + nt - 1) / nt) * 32 * sk, (mp + block - 1) / block, 1),
                    threadGroup: (32 * sk, 1, 1), outputShapes: [[m, n]], outputDTypes: [.bfloat16])[
                    0]
        } else if bits != 4 {
            precondition(bits > 4 && group == 64, "lane bytes kernel: 5/6/8-bit, groups of 64")
            y =
                kernel(
                    "bytes", body: bytesSource, inputs: ["X", "XS", "Wq", "SBt", "mdims"],
                    outputs: ["Y"],
                    constants: [
                        ("TMR", block / 16), ("N", n), ("K", k), ("NT", nt), ("SK", sk),
                        ("BITS", bits), ("TILED", tiled ? 1 : 0),
                    ])(
                    [x2, xs, weight, sbt, mdims],
                    grid: (((n + nt - 1) / nt) * 32 * sk, (mp + block - 1) / block, 1),
                    threadGroup: (32 * sk, 1, 1), outputShapes: [[m, n]], outputDTypes: [.bfloat16])[
                    0]
        } else {
            y =
                kernel(
                    tiled ? "main_tiled" : "main", body: tiled ? mainTiledSource : mainSource,
                    inputs: ["X", "XS", "Wq", "SBt", "mdims"], outputs: ["Y"],
                    constants: [
                        ("TMR", block / 16), ("N", n), ("K", k), ("NT", nt), ("SK", sk),
                        ("GS", group), ("EDGE", edge),
                    ])(
                    [x2, xs, weight, sbt, mdims],
                    grid: (((n + nt - 1) / nt) * 32 * sk, (mp + block - 1) / block, 1),
                    threadGroup: (32 * sk, 1, 1), outputShapes: [[m, n]], outputDTypes: [.bfloat16])[
                    0]
        }
        return y.reshaped(lead + [n])
    }

    /// 8-bit weights straight from device memory (`main8*`); `VMLX_LANE_QMM_DIRECT8=0` keeps the
    /// widening bytes kernel.
    nonisolated(unsafe) static var direct8Enabled =
        ProcessInfo.processInfo.environment["VMLX_LANE_QMM_DIRECT8"] != "0"

    // MARK: install

    /// Whether the lane matmul reads this QuantizedLinear (affine, bf16 scales, supported width/group).
    static func takes(_ q: QuantizedLinear) -> Bool {
        guard type(of: q) == QuantizedLinear.self, q.mode == .affine, readableBits.contains(q.bits),
            q.bits >= 4,
            (q.bits == 4 ? [32, 64] : [64]).contains(q.groupSize),
            q.scales.dtype == .bfloat16, let b = q.biases, b.dtype == .bfloat16,
            q.weight.dtype == .uint32, q.weight.ndim == 2
        else { return false }
        let k = q.weight.dim(1) * 32 / q.bits
        return k % 64 == 0 && q.weight.dim(0) % 4 == 0
    }

    nonisolated(unsafe) private static var installedTargets: Set<ObjectIdentifier> = []
    nonisolated(unsafe) private static var laneFlatTargets: [ObjectIdentifier: Bool] = [:]

    /// Whether `target`'s verify cost is lane-flat (set by ``installForDFlash2Target``).
    public static func isLaneFlat(_ target: Any) -> Bool {
        guard let module = target as? Module else { return false }
        lock.lock()
        defer { lock.unlock() }
        return laneFlatTargets[ObjectIdentifier(module)] ?? false
    }
    nonisolated(unsafe) public private(set) static var lastInstallReport: InstallReport?

    /// Install on a DFlash2 target once per process (VMLX_LANE_QMM=0 opts out).
    nonisolated(unsafe) private static var installedDrafters: Set<ObjectIdentifier> = []

    /// Opt-in (`VMLX_LANE_QMM_DRAFTER=1`): route a DFlash2 drafter's quantized projections through the lane
    /// matmul, once per drafter, at load.
    ///
    /// Every draft forward is an 8-16-row block, where MLX's quantized matmul leaves its 1-row bandwidth
    /// behind (27B MLP shape, 4-bit g64 at ~925 MHz: 1 row 0.10 ms, 8 rows 0.27 ms, 16 rows 0.18-0.46 ms;
    /// lane 0.12-0.13 ms at 8-16 rows). Qwen3.8-27B JANG_4D greedy code probes, warm: draft time per
    /// request -45..-65 %, end-to-end +7-8 % (long 15k-token context ~71.5 -> ~77.5 tok/s, 3 rounds).
    ///
    /// Opt-in because it is not output-neutral on targets whose verify is not row-exact (27B): the
    /// drafter's rounding changes which rows are verified together, and a near-tie can then resolve
    /// differently. Greedy text stays deterministic but differs from the stock drafter's.
    public static func installForDFlash2Drafter(_ drafter: Any) {
        guard ProcessInfo.processInfo.environment["VMLX_LANE_QMM_DRAFTER"] == "1",
            ProcessInfo.processInfo.environment["VMLX_LANE_QMM"] != "0",
            let module = drafter as? Module
        else { return }
        lock.lock()
        let fresh = installedDrafters.insert(ObjectIdentifier(module)).inserted
        lock.unlock()
        guard fresh else { return }
        let start = Date()
        let report = install(model: module, tile: true)
        // Compile the lane kernels for the drafter's shapes now (8- and 16-row blocks): left to the first
        // request, compilation cost it ~10 % (27B: 72-77 vs 80-85 tok/s with the install already at load).
        var warmed = Set<String>()
        for (_, m) in module.leafModules().flattened() {
            guard let lane = m as? LaneQuantizedLinear else { continue }
            let k = lane.weight.dim(1) * 32 / lane.bits
            let key = "\(lane.weight.dim(0))x\(k)b\(lane.bits)t\(lane.laneTiled)"
            guard warmed.insert(key).inserted else { continue }
            for rows in [8, 16] {
                MLX.eval(lane(MLXArray.zeros([rows, k], dtype: .bfloat16)))
            }
        }
        FileHandle.standardError.write(Data(String(
            format: "[LaneQMM] DFlash2 drafter lane=%d tiled=%d skipped=%d warmed=%d shapes (%.2fs)\n",
            report.lane, report.tiled, report.skipped, warmed.count,
            Date().timeIntervalSince(start)).utf8))
    }

    public static func installForDFlash2Target(_ target: Any) {
        guard ProcessInfo.processInfo.environment["VMLX_LANE_QMM"] != "0",
            let module = target as? Module
        else { return }
        let id = ObjectIdentifier(module)
        lock.lock()
        let fresh = installedTargets.insert(id).inserted
        lock.unlock()
        guard fresh else { return }
        let start = Date()
        // Tiling policy. Several Swift S=1 paths read projection weights directly into their own fused
        // kernels (Qwen35 GDN: the fused in_proj_qkv/z/b/a decode projection and the compiled out_proj tail),
        // and those must keep MLX's packed layout. Default "safe" tiles every other lane module (attention,
        // dense MLP: 27B JANG_4D 8-row forward 74.0 -> 67.4 ms with everything tiled). VMLX_LANE_QMM_TILED=0
        // tiles nothing; =1 tiles everything (diagnostic: corrupts S=1 output on models with direct readers).
        let tiling = ProcessInfo.processInfo.environment["VMLX_LANE_QMM_TILED"]
        let report = install(
            model: module, tile: tiling != "0",
            tileFilter: { path in
                tiling == "1" || !directWeightReaderPaths.contains { path.contains($0) }
            })
        lastInstallReport = report
        // Lane-flat = every packed (uint32) weight of the language model now runs on the lane, so verify
        // cost is ~flat in rows. JANGH codebook MLPs (Qwen3.8-27B JANGH2) keep it False.
        var flat = report.lane > 0
        for (path, m) in module.leafModules().flattened() {
            if path.hasPrefix("vision") || path.contains("vision_tower") || path.contains("visual")
            {
                continue
            }
            // Dense JANGH codebook projections (Qwen3.8-27B JANGH2 MLPs) do not run on the lane; their bank
            // tensors live outside module parameters, so the uint32 scan below cannot see them.
            if m is JANGHDenseLinear {
                flat = false
                break
            }
            if m is LaneQuantizedLinear || m is Embedding { continue }
            if m.parameters().flattened().contains(where: { $0.1.dtype == .uint32 }) {
                flat = false
                break
            }
        }
        lock.lock()
        laneFlatTargets[id] = flat
        lock.unlock()
        if ProcessInfo.processInfo.environment["VMLX_LANE_QMM_LOG"] == "1" {
            FileHandle.standardError.write(
                Data(
                    "[LaneQMM] DFlash2 target: \(report.lane) projections (\(report.tiled) tiled), skipped \(report.skipped), \(String(format: "%.2f", Date().timeIntervalSince(start)))s\n"
                        .utf8))
        }
    }

    public struct InstallReport: Sendable {
        public var lane = 0
        public var tiled = 0
        public var skipped = 0
    }

    /// Module paths whose packed `.weight` is read directly by fused S=1 kernels (never tiled by default).
    static let directWeightReaderPaths = ["linear_attn.in_proj", "linear_attn.out_proj"]

    /// Below this many rows an UNTILED lane module runs MLX's quantized matmul (qmv), which is faster
    /// there (27B JANG_4D 1-row forward: qmv 48.9 ms vs lane 61.4 ms). VMLX_LANE_QMM_MIN_ROWS overrides.
    static let minRows =
        Int(ProcessInfo.processInfo.environment["VMLX_LANE_QMM_MIN_ROWS"] ?? "") ?? 4

    /// Route `model`'s plain affine QuantizedLinear layers through the lane matmul (per-module swap; other
    /// models in the process keep MLX). Tiles weights in ~2 GB batches so the old layout is freed promptly.
    @discardableResult
    public static func install(
        model: Module, tile: Bool = true, tileFilter: (String) -> Bool = { _ in true }
    ) -> InstallReport {
        var report = InstallReport()
        guard available() else { return report }
        var updates: [(String, Module)] = []
        var pending: [MLXArray] = []
        var pendingBytes = 0
        let leaves = model.leafModules().flattened()
        // Language model only (as in Python): vision towers keep MLX.
        let languageOnly = leaves.contains { $0.0.hasPrefix("language_model.") }
        for (path, m) in leaves {
            if languageOnly && !path.hasPrefix("language_model.") { continue }
            guard let q = m as? QuantizedLinear, !(q is LaneQuantizedLinear) else { continue }
            guard takes(q), let qb = q.biases else {
                report.skipped += 1
                continue
            }
            let sbt = packScales(q.scales, qb)
            var w = q.weight
            var isTiled = false
            if tile, tileFilter(path), w.dim(0) % nt == 0 {
                w = tileWeight(w, bits: q.bits, group: q.groupSize)
                isTiled = true
                report.tiled += 1
            }
            let lane = LaneQuantizedLinear(
                weight: w, bias: q.bias, scales: q.scales, biases: qb,
                groupSize: q.groupSize, bits: q.bits, sbt: sbt, tiled: isTiled)
            updates.append((path, lane))
            pending.append(contentsOf: [sbt, w])
            pendingBytes += sbt.nbytes + 2 * w.nbytes
            report.lane += 1
            if pendingBytes >= 2 << 30 {
                eval(pending)
                pending.removeAll()
                pendingBytes = 0
            }
        }
        if !pending.isEmpty { eval(pending) }
        if !updates.isEmpty { model.update(modules: ModuleChildren.unflattened(updates)) }
        Memory.clearCache()
        return report
    }
}

/// QuantizedLinear whose <= 128-row bf16 calls run the lane matmul (untiled modules: >= `LaneQMM.minRows` rows;
/// fewer rows take MLX's qmv on the unchanged weight). Wider calls (prompt prefill chunks) and
/// other dtypes run MLX's own quantized matmul on the MLX-layout weight, rebuilt per call when tiled (one
/// weight copy per call: ~2-3 % of a 2048-row chunk, measured in Python).
public final class LaneQuantizedLinear: QuantizedLinear {
    let laneSBT: MLXArray
    let laneTiled: Bool

    init(
        weight: MLXArray, bias: MLXArray?, scales: MLXArray, biases: MLXArray, groupSize: Int,
        bits: Int,
        sbt: MLXArray, tiled: Bool
    ) {
        self.laneSBT = sbt
        self.laneTiled = tiled
        super.init(
            weight: weight, bias: bias, scales: scales, biases: biases, groupSize: groupSize,
            bits: bits, mode: .affine)
    }

    public override func callAsFunction(_ x: MLXArray) -> MLXArray {
        var rows = 1
        for d in x.shape.dropLast() { rows *= d }
        var y: MLXArray
        if rows <= LaneQMM.maxRows && x.dtype == .bfloat16 && (laneTiled || rows >= LaneQMM.minRows)
        {
            y = LaneQMM.laneMatmul(
                x, weight: weight, sbt: laneSBT, bits: bits, group: groupSize,
                tiled: laneTiled)
        } else {
            let w = laneTiled ? LaneQMM.untileWeight(weight, bits: bits, group: groupSize) : weight
            y = quantizedMM(
                x, w, scales: scales, biases: biases, transpose: true, groupSize: groupSize,
                bits: bits, mode: .affine)
        }
        if let bias { y = y + bias }
        return y
    }
}
