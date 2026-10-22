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
    /// Scales from a packed (K/group, N, 2) `sbt` (see ``packScales``).
    public static func laneMatmul(
        _ x: MLXArray, weight: MLXArray, sbt: MLXArray, bits: Int,
        group: Int = 64, tiled: Bool
    ) -> MLXArray {
        laneMatmul(x, weight: weight, scaleSource: .packed(sbt), bits: bits, group: group, tiled: tiled)
    }

    /// Same matmul reading the module's OWN (N, K/scaleGroup) bf16 `scales` and `biases` in place: no
    /// packed copy. Bit-identical to the `sbt` form (same values, same arithmetic).
    public static func laneMatmul(
        _ x: MLXArray, weight: MLXArray, scales: MLXArray, biases: MLXArray, scaleGroup: Int,
        bits: Int, group: Int = 64, tiled: Bool
    ) -> MLXArray {
        laneMatmul(
            x, weight: weight, scaleSource: .direct(scales, biases, scaleGroup: scaleGroup), bits: bits,
            group: group, tiled: tiled)
    }

    enum ScaleSource {
        case packed(MLXArray)
        case direct(MLXArray, MLXArray, scaleGroup: Int)
    }

    static func laneMatmul(
        _ x: MLXArray, weight: MLXArray, scaleSource: ScaleSource, bits: Int,
        group: Int, tiled: Bool
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
        // Scale operands: the packed SBt, or S and B in place (+ their group geometry as constants).
        let scaleInputs: [String]
        let scaleArrays: [MLXArray]
        let scaleConstants: [(String, Int)]
        let suffix: String
        switch scaleSource {
        case .packed(let sbt):
            scaleInputs = ["SBt"]; scaleArrays = [sbt]; scaleConstants = []; suffix = ""
        case .direct(let scales, let biases, let scaleGroup):
            precondition(scaleGroup % group == 0, "lane: scale group must be a multiple of the lane group")
            scaleInputs = ["S", "B"]; scaleArrays = [scales, biases]
            scaleConstants = [("SKG", k / scaleGroup), ("SREP", scaleGroup / group)]
            suffix = "_direct"
        }
        func source(_ body: String) -> String { suffix.isEmpty ? body : directScales(body) }
        let inputs = ["X", "XS", "Wq"] + scaleInputs + ["mdims"]
        let args = [x2, xs, weight] + scaleArrays + [mdims]
        let y: MLXArray
        if bits == 8, group == 64 || group == 128, direct8Enabled {
            y =
                kernel(
                    (tiled ? "main8_tiled" : "main8") + suffix,
                    body: source(tiled ? main8TiledSource : main8Source),
                    inputs: inputs, outputs: ["Y"],
                    constants: [
                        ("TMR", block / 16), ("N", n), ("K", k), ("NT", nt), ("SK", sk),
                        ("GS", group), ("EDGE", edge),
                    ] + scaleConstants)(
                    args,
                    grid: (((n + nt - 1) / nt) * 32 * sk, (mp + block - 1) / block, 1),
                    threadGroup: (32 * sk, 1, 1), outputShapes: [[m, n]], outputDTypes: [.bfloat16])[
                    0]
        } else if bits != 4 {
            precondition(bits > 4 && group == 64, "lane bytes kernel: 5/6/8-bit, groups of 64")
            y =
                kernel(
                    "bytes" + suffix, body: source(bytesSource), inputs: inputs,
                    outputs: ["Y"],
                    constants: [
                        ("TMR", block / 16), ("N", n), ("K", k), ("NT", nt), ("SK", sk),
                        ("BITS", bits), ("TILED", tiled ? 1 : 0),
                    ] + scaleConstants)(
                    args,
                    grid: (((n + nt - 1) / nt) * 32 * sk, (mp + block - 1) / block, 1),
                    threadGroup: (32 * sk, 1, 1), outputShapes: [[m, n]], outputDTypes: [.bfloat16])[
                    0]
        } else {
            y =
                kernel(
                    (tiled ? "main_tiled" : "main") + suffix,
                    body: source(tiled ? mainTiledSource : mainSource),
                    inputs: inputs, outputs: ["Y"],
                    constants: [
                        ("TMR", block / 16), ("N", n), ("K", k), ("NT", nt), ("SK", sk),
                        ("GS", group), ("EDGE", edge),
                    ] + scaleConstants)(
                    args,
                    grid: (((n + nt - 1) / nt) * 32 * sk, (mp + block - 1) / block, 1),
                    threadGroup: (32 * sk, 1, 1), outputShapes: [[m, n]], outputDTypes: [.bfloat16])[
                    0]
        }
        return y.reshaped(lead + [n])
    }

    /// 8-bit weights straight from device memory (`main8*`); `VMLX_LANE_QMM_DIRECT8=0` keeps the
    /// widening bytes kernel.
    /// `VMLX_LANE_QMM_SBT=1` / `=0`: force the packed SBt copy on / off for every lane module (A/B).
    /// Unset: the target's ``LaneQMMTargetPolicy/laneQMMPackedScales`` (packed for drafters and for
    /// targets without a policy).
    static let packedScalesOverride: Bool? = {
        switch ProcessInfo.processInfo.environment["VMLX_LANE_QMM_SBT"] {
        case "1": return true
        case "0": return false
        default: return nil
        }
    }()

    nonisolated(unsafe) static var direct8Enabled =
        ProcessInfo.processInfo.environment["VMLX_LANE_QMM_DIRECT8"] != "0"

    // MARK: install

    /// Group size the lane kernels run `q` at, or nil when it is not lane-readable. The packed weight
    /// does not depend on the group size (MLX packs along K; scales live apart), so an affine group of
    /// 128 is exactly two groups of 64 sharing one scale/bias. 4-bit and direct 8-bit kernels index
    /// scales by GS and read 128 natively; the 5/6-bit bytes kernel is fixed at 64 and gets each 128
    /// group's scale and bias twice (identical dequantization).
    static func laneGroup(for q: QuantizedLinear) -> Int? {
        switch (q.bits, q.groupSize) {
        case (4, 32), (4, 64), (4, 128): return q.groupSize
        case (5, 64), (6, 64), (8, 64): return 64
        case (8, 128): return direct8Enabled ? 128 : 64
        case (5, 128), (6, 128): return 64
        default: return nil
        }
    }

    /// Whether the lane matmul reads this QuantizedLinear (affine, bf16 scales, supported width/group).
    static func takes(_ q: QuantizedLinear) -> Bool {
        guard type(of: q) == QuantizedLinear.self, q.mode == .affine, readableBits.contains(q.bits),
            q.bits >= 4, laneGroup(for: q) != nil,
            q.scales.dtype == .bfloat16, let b = q.biases, b.dtype == .bfloat16,
            q.weight.dtype == .uint32, q.weight.ndim == 2
        else { return false }
        let k = q.weight.dim(1) * 32 / q.bits
        return k % 64 == 0 && q.weight.dim(0) % 4 == 0
    }

    nonisolated(unsafe) private static var installedTargets: Set<ObjectIdentifier> = []
    nonisolated(unsafe) private static var laneFlatTargets: [ObjectIdentifier: Bool] = [:]

    nonisolated(unsafe) private static var installedDrafters: Set<ObjectIdentifier> = []

    /// Install on a DFlash2 drafter once per process (`VMLX_LANE_QMM_DRAFTER=0` keeps MLX: A/B control).
    ///
    /// Every draft forward is an 8-16-row block, where MLX's quantized matmul leaves its 1-row bandwidth
    /// behind (27B MLP shape, 4-bit g64, 925 MHz: 1 row 0.10 ms, 8 rows 0.27 ms, 16 rows 0.18-0.46 ms;
    /// lane 0.12-0.13 ms at 8-16 rows). The drafter's numerics never reach the target's tokens: only
    /// how many proposals are accepted.
    public static func installForDFlash2Drafter(_ drafter: Any) {
        guard ProcessInfo.processInfo.environment["VMLX_LANE_QMM"] != "0",
            ProcessInfo.processInfo.environment["VMLX_LANE_QMM_DRAFTER"] != "0",
            let module = drafter as? Module
        else { return }
        lock.lock()
        let fresh = installedDrafters.insert(ObjectIdentifier(module)).inserted
        lock.unlock()
        guard fresh else { return }
        let report = install(model: module, tile: true, untiledMinRows: minRows)
        FileHandle.standardError.write(Data(
            "[LaneQMM] drafter lane=\(report.lane) tiled=\(report.tiled) skipped=\(report.skipped)\n".utf8))
    }

    /// Whether `target`'s verify cost is lane-flat (set by ``installForDFlash2Target``).
    public static func isLaneFlat(_ target: Any) -> Bool {
        guard let module = target as? Module else { return false }
        lock.lock()
        defer { lock.unlock() }
        return laneFlatTargets[ObjectIdentifier(module)] ?? false
    }
    nonisolated(unsafe) public private(set) static var lastInstallReport: InstallReport?

    /// Install on a DFlash2 target once per process (VMLX_LANE_QMM=0 opts out). A target conforming
    /// to ``LaneQMMTargetPolicy`` chooses tiling and the untiled row threshold from its own measured
    /// row costs; other targets keep the defaults below.
    public static func installForDFlash2Target(_ target: Any) {
        guard ProcessInfo.processInfo.environment["VMLX_LANE_QMM"] != "0",
            let module = target as? Module,
            (target as? LaneQMMTargetPolicy)?.laneQMMEnabled ?? true
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
        //
        // A tiled weight is a re-laid-out COPY that replaces the module's weight. It is only tiled when that
        // replacement frees the original (`WeightBacking.copyWouldReplace`). Under the shipped memory-safety
        // loader every weight is a window into a mapped safetensors shard, which the replacement cannot
        // release: measured 2026-10-11 on Qwen3.8-27B JANG_4D + bundled DFlash2, idle live memory after two
        // turns 41.3 GB tiled vs 28.4 GB untiled (21.6 GB model) for ~8-10 % DFlash2 decode. DO NOT drop the
        // replace check: resident memory must stay at the model's size (Eric, 2026-10-11).
        let policy = target as? LaneQMMTargetPolicy
        let tiling = ProcessInfo.processInfo.environment["VMLX_LANE_QMM_TILED"]
            ?? (policy.map { $0.laneQMMTiled ? nil : "0" } ?? nil)
        let report = install(
            model: module, tile: tiling != "0",
            tileFilter: { path in
                tiling == "1" || !readsRawPackedWeight(path)
            },
            untiledMinRows: policy?.laneQMMUntiledMinRows ?? minRows,
            // SPEED-CRITICAL (Qwen3.8-27B DFlash2): the 27B keeps the PACKED scale copy — reading
            // its scales in place measured 2x per verify matmul and DFlash2 fell to AR speed
            // (dbf4459a7). Per-target choice; do not flip the default for every model.
            packedScales: policy?.laneQMMPackedScales ?? true)
        lastInstallReport = report
        // Lane-flat = every packed (uint32) weight of the language model now runs on the lane, so verify
        // cost is ~flat in rows. JANGH codebook MLPs (Qwen3.8-27B JANGH2) keep it False.
        var flat = report.lane > 0
        for (path, m) in module.leafModules().flattened() {
            if path.hasPrefix("vision") || path.contains("vision_tower") || path.contains("visual")
            {
                continue
            }
            // Dense and routed custom banks do not run on the lane. Their tensors can live outside
            // Module.parameters, so the uint32 scan cannot establish lane coverage. Ornith JANGH2
            // otherwise looked flat and lost the narrow adaptive candidate despite row-scaled expert
            // verification cost. Keep the Qwen27B dense exception and include opaque routed banks.
            if m is JANGHDenseLinear || m is any WeightedRoutedExpertLayer {
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
    // Qwen3.5/MiniCPM's compiled shared expert and quantized router also
    // read .weight directly. Owned loading makes tiling eligible; permuting
    // these weights would feed the fused AR/probe path a different matrix,
    // corrupt its cache, and make subsequent DSpark tokens repeat.
    // Retain lane dispatch with MLX packing for these modules; other eligible
    // projections still tile. Do not disable the fused reader or speculation.
    static let directWeightReaderPaths = [
        "linear_attn.in_proj", "linear_attn.out_proj",
        "mlp.shared_expert.",
    ]

    static func readsRawPackedWeight(_ path: String) -> Bool {
        // Router `mlp.gate` is a module name, not a prefix: dense `mlp.gate_proj`
        // must remain eligible for tiling (notably Qwen 27B DFlash2).
        path == "mlp.gate" || path.hasSuffix(".mlp.gate")
            || directWeightReaderPaths.contains { path.contains($0) }
    }

    /// Below this many rows an UNTILED lane module runs MLX's quantized matmul (qmv), which is faster
    /// there (27B JANG_4D 1-row forward: qmv 48.9 ms vs lane 61.4 ms). VMLX_LANE_QMM_MIN_ROWS overrides.
    public static let minRows =
        Int(ProcessInfo.processInfo.environment["VMLX_LANE_QMM_MIN_ROWS"] ?? "") ?? 4

    /// Route `model`'s plain affine QuantizedLinear layers through the lane matmul (per-module swap; other
    /// models in the process keep MLX). Tiles weights in ~2 GB batches so the old layout is freed promptly.
    @discardableResult
    public static func install(
        model: Module, tile: Bool = true, tileFilter: (String) -> Bool = { _ in true },
        untiledMinRows: Int = minRows, packedScales policyPacked: Bool = true
    ) -> InstallReport {
        install(model: model, tile: tile, tileFilter: tileFilter,
                untiledMinRows: untiledMinRows, packedScales: policyPacked,
                evaluationBatchBytes: 2 << 30)
    }

    static func install(
        model: Module, tile: Bool, tileFilter: (String) -> Bool,
        untiledMinRows: Int, packedScales policyPacked: Bool,
        evaluationBatchBytes: Int
    ) -> InstallReport {
        let packedScales = packedScalesOverride ?? policyPacked
        var report = InstallReport()
        guard available() else { return report }
        var updates: [(String, Module)] = []
        var pending: [MLXArray] = []
        var pendingBytes = 0
        // Consume each strong leaf reference after visiting it. Otherwise this
        // traversal itself keeps the complete original bank alive, even after
        // replacing modules and evaluating the new layouts in bounded batches.
        var leaves = model.leafModules().flattened().map { ($0.0, Optional($0.1)) }
        // Language model only (as in Python): vision towers keep MLX.
        let languageOnly = leaves.contains { $0.0.hasPrefix("language_model.") }
        for index in leaves.indices {
            let path = leaves[index].0
            guard let m = leaves[index].1 else { continue }
            defer { leaves[index].1 = nil }
            if languageOnly && !path.hasPrefix("language_model.") { continue }
            guard let q = m as? QuantizedLinear, !(q is LaneQuantizedLinear) else { continue }
            guard takes(q), let qb = q.biases else {
                report.skipped += 1
                continue
            }
            let group = laneGroup(for: q)!
            let repeats = q.groupSize / group
            // Packed SBt = a (K/GS, N, 2) bf16 COPY of the module's scales and biases, kept next to the
            // originals (+1.6 GB on Qwen3.8-27B JANG_4D). Without it the kernels read scales in place
            // (`directScales`, bit-identical, ~2x slower per verify matmul). Per-target choice, see
            // `LaneQMMTargetPolicy.laneQMMPackedScales`.
            let sbt: MLXArray? =
                !packedScales ? nil
                : repeats == 1
                    ? packScales(q.scales, qb)
                    : packScales(
                        repeated(q.scales, count: repeats, axis: 1), repeated(qb, count: repeats, axis: 1))
            var w = q.weight
            var isTiled = false
            if tile, tileFilter(path), w.dim(0) % nt == 0, WeightBacking.copyWouldReplace([w]) {
                w = tileWeight(w, bits: q.bits, group: group)
                isTiled = true
                report.tiled += 1
            }
            let lane = LaneQuantizedLinear(
                weight: w, bias: q.bias, scales: q.scales, biases: qb,
                groupSize: q.groupSize, bits: q.bits, sbt: sbt, tiled: isTiled, laneGroup: group,
                untiledMinRows: untiledMinRows)
            updates.append((path, lane))
            pending.append(w)
            pendingBytes += 2 * w.nbytes
            if let sbt {
                pending.append(sbt)
                pendingBytes += sbt.nbytes
            }
            report.lane += 1
            if pendingBytes >= evaluationBatchBytes {
                eval(pending)
                pending.removeAll()
                pendingBytes = 0
                // Evaluation alone does not release the original modules.
                // Install this batch now, before allocating the next one.
                model.update(modules: ModuleChildren.unflattened(updates))
                updates.removeAll()
                Memory.clearCache()
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
    /// `VMLX_QUANTIZED_OUTPUT_PROMOTE=1` (as for `QuantizedLinear`): keep MLX's promoted output.
    /// Read once: this module runs ~250 times per token.
    static let propagatePromotedOutput =
        ProcessInfo.processInfo.environment["VMLX_QUANTIZED_OUTPUT_PROMOTE"] == "1"

    /// Packed (K/group, N, 2) scales, or nil (default): the kernels read `scales`/`biases` in place.
    let laneSBT: MLXArray?
    let laneTiled: Bool
    /// Group size of `laneSBT` and of the tiled layout (``LaneQMM/laneGroup(for:)``); the module's own
    /// `groupSize`/`scales` stay the bundle's, for MLX's matmul on wide calls.
    let laneGroup: Int
    /// Untiled modules run MLX's matmul below this many rows (``LaneQMM/minRows`` unless the target's
    /// ``LaneQMMTargetPolicy`` measured a different crossover).
    let laneUntiledMinRows: Int

    init(
        weight: MLXArray, bias: MLXArray?, scales: MLXArray, biases: MLXArray, groupSize: Int,
        bits: Int,
        sbt: MLXArray?, tiled: Bool, laneGroup: Int? = nil, untiledMinRows: Int = LaneQMM.minRows
    ) {
        self.laneSBT = sbt
        self.laneTiled = tiled
        self.laneGroup = laneGroup ?? groupSize
        self.laneUntiledMinRows = untiledMinRows
        super.init(
            weight: weight, bias: bias, scales: scales, biases: biases, groupSize: groupSize,
            bits: bits, mode: .affine)
    }

    public override func callAsFunction(_ x: MLXArray) -> MLXArray {
        var rows = 1
        for d in x.shape.dropLast() { rows *= d }
        var y: MLXArray
        if rows <= LaneQMM.maxRows && x.dtype == .bfloat16 && (laneTiled || rows >= laneUntiledMinRows)
        {
            if let laneSBT {
                y = LaneQMM.laneMatmul(
                    x, weight: weight, sbt: laneSBT, bits: bits, group: laneGroup,
                    tiled: laneTiled)
            } else {
                y = LaneQMM.laneMatmul(
                    x, weight: weight, scales: scales, biases: biases!, scaleGroup: groupSize,
                    bits: bits, group: laneGroup, tiled: laneTiled)
            }
        } else {
            let w = laneTiled ? LaneQMM.untileWeight(weight, bits: bits, group: laneGroup) : weight
            y = quantizedMM(
                x, w, scales: scales, biases: biases, transpose: true, groupSize: groupSize,
                bits: bits, mode: .affine)
        }
        if let bias { y = y + bias }
        // Same dtype contract as `QuantizedLinear`: a half-precision activation stays half. Against
        // fp16 scales MLX promotes `quantizedMM` to fp32, and the lane kernel writes bf16; either
        // would otherwise change the residual stream and KV cache dtype of a model whose decode
        // runs through this module (fp32 doubles every KV byte).
        let inputDType = x.dtype
        if !Self.propagatePromotedOutput, y.dtype != inputDType,
            inputDType == .bfloat16 || inputDType == .float16
        {
            y = y.asType(inputDType)
        }
        return y
    }
}

/// Lane matmul placement measured per target family. Tiled modules run the lane for every row
/// count, so a target whose one-row step is a plain QuantizedLinear decode pays the lane's 1-row cost
/// on ordinary AR for the rest of the process; such targets install untiled and start the lane where
/// it beats MLX's matmul.
public protocol LaneQMMTargetPolicy {
    var laneQMMTiled: Bool { get }
    var laneQMMUntiledMinRows: Int { get }
    /// Whether lane modules keep a packed (K/group, N, 2) copy of their scales and biases (faster
    /// kernels, extra memory = 1/8 of the 4-bit weight bytes) or read them in place. Default true; a
    /// target sets false where the in-place read measured the same end-to-end speed (see
    /// `LaneQMMSources.directScales`).
    var laneQMMPackedScales: Bool { get }
    /// Whether the lane matmul may be installed on this target at all. Default true; a target whose verify
    /// output the lane kernels change sets false (GLM-5.3: see `Glm5Next.laneQMMEnabled`).
    var laneQMMEnabled: Bool { get }
}

extension LaneQMMTargetPolicy {
    public var laneQMMPackedScales: Bool { true }
    public var laneQMMEnabled: Bool { true }
}
