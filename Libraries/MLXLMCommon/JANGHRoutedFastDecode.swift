import Foundation
import MLX
import MLXFast

/// Routed-expert JANGH decode kernels that are bitwise equal to `JANGHProjectionKernel` (fc1 / up) and
/// `JANGHWeightedDownKernel` (fc2 / down + routing sum), measured as the dominant cost of a Nemotron 3.5
/// Bolt decode step: its 2- and 3-bit codebook experts (9 of 240 per layer, K 1,024 -> 2,048 -> 1,024)
/// ran at ~0.37 ms per MoE layer — ~30 GB/s for ~12 MB of packed weights.
///
/// Two causes, both removed without changing a single float operation:
/// - every code evaluated the codebook polynomial `u * fma(beta, u*u, alpha)` (and fc1 re-loaded and
///   re-shifted its packed word per code). Here each lane loads its packed words once per (row, block)
///   and reads the level from a table holding the exact float bit patterns of that polynomial (the
///   same table `JANGHDenseFastQMV` uses), then runs the reference's fma chain in the reference order;
/// - the weighted-down kernel looped over all routes inside each threadgroup, so a one-token decode
///   launched only N/8 threadgroups (128 for Bolt's fc2) for 9 experts' work. Here pass A computes
///   every (route, row) dot product in its own threadgroup, and pass B forms
///   `total += score * dot * scale` per row in route order, exactly as the reference loop did.
///
/// Both are admitted per immutable bank owner after a bitwise comparison with the reference kernels on
/// the real tensors (`JANGHDenseFastAdmission`); `VMLX_JANGH_ROUTED_FAST=0` disables.
enum JANGHRoutedFastDecode {
    static let enabled = RuntimeEnvironment.value("VMLX_JANGH_ROUTED_FAST") != "0"
    /// Rows per simdgroup (bench override `VMLX_JANGH_ROUTED_ROWS`, 4 / 8 / 16).
    static let rows = Int(RuntimeEnvironment.value("VMLX_JANGH_ROUTED_ROWS") ?? "") ?? 4
    /// Level table in program-scope constant memory (bench override `VMLX_JANGH_ROUTED_CONST=0`).
    static let constantTable = RuntimeEnvironment.value("VMLX_JANGH_ROUTED_CONST") != "0"

    static func levelTable(bits: Int, alpha: Double, beta: Double) -> (table: String, signature: String) {
        let a = Float(alpha)
        let b = Float(beta)
        let center = Float((1 << bits) - 1) / 2
        let levels = (0 ..< (1 << bits)).map { code -> UInt32 in
            let u = Float(code) - center
            return (u * a.addingProduct(b, u * u)).bitPattern
        }
        return (
            levels.map { "as_type<float>(0x\(String($0, radix: 16))u)" }.joined(separator: ", "),
            levels.map { String($0, radix: 16) }.joined()
        )
    }

    /// Code extraction for a lane's 16 consecutive values from words loaded once (same integer
    /// codes as the reference's per-code word/shift/mask).
    static func codes(bits: Int) -> String {
        var lines: [String] = []
        let units = bits == 3 ? 2 : 16 * bits / 32
        lines.append("uint qw[\(Swift.max(units, 1))];")
        lines.append("for (uint j = 0; j < \(units)u; ++j) qw[j] = packed[row + qbase + j];")
        if bits == 3 { lines.append("uint qstart = (cbase * 3u) & 31u;") }
        lines.append("uint code[16];")
        for i in 0 ..< 16 {
            if bits == 3 {
                lines.append("{ uint qb = qstart + \(i * 3)u; uint c = qb < 32u ? (qw[0] >> qb) : (qw[1] >> (qb - 32u)); if (qb > 29u && qb < 32u) c |= qw[1] << (32u - qb); code[\(i)] = c & 7u; }")
            } else {
                let bit = i * bits, word = bit / 32, shift = bit % 32
                var unpacked = "(qw[\(word)] >> \(shift)u)"
                if shift + bits > 32 { unpacked = "(\(unpacked) | (qw[\(word + 1)] << \(32 - shift)u))" }
                lines.append("code[\(i)] = \(unpacked) & \((1 << bits) - 1)u;")
            }
        }
        return lines.joined(separator: "\n")
    }

    static func eligible(k: Int, n: Int, bits: Int) -> Bool {
        enabled && [2, 3, 4].contains(bits) && k > 0 && k.isMultiple(of: 512) && n > 0
            && n.isMultiple(of: 2 * rows)
    }
}

/// fc1 / up: twin of `JANGHProjectionKernel`'s qmv (activation already in the projection basis).
final class JANGHRoutedFastQMV {
    private let kernel: MLXFast.MLXFastKernel
    let bits: Int
    let rowsPerSimdgroup: Int

    /// `rowsPerSimdgroup` rows share each lane's x values (the reference kernel uses 4); the level
    /// table lives in program-scope constant memory unless `constantTable` is false.
    init?(bits: Int, alpha: Double, beta: Double, rowsPerSimdgroup: Int = JANGHRoutedFastDecode.rows,
          constantTable: Bool = JANGHRoutedFastDecode.constantTable) {
        guard [2, 3, 4].contains(bits), [4, 8, 16].contains(rowsPerSimdgroup) else { return nil }
        self.bits = bits
        self.rowsPerSimdgroup = rowsPerSimdgroup
        let (table, signature) = JANGHRoutedFastDecode.levelTable(bits: bits, alpha: alpha, beta: beta)
        let lv = "LV_\(bits)_\(signature)"
        let header = constantTable ? "constant float \(lv)[\(1 << bits)] = {\(table)};\n" : ""
        let local = constantTable ? "" : "const float \(lv)[\(1 << bits)] = {\(table)};"
        let source = """
            \(local)
            uint lane = thread_index_in_simdgroup;
            uint row0 = threadgroup_position_in_grid.y * (2u * RPS) + simdgroup_index_in_threadgroup * RPS;
            uint dispatch = threadgroup_position_in_grid.z;
            uint expert = indices[dispatch];
            float accum[RPS];
            for (uint r = 0; r < RPS; ++r) accum[r] = 0.0f;
            if (expert >= EXPERTS) {
                if (lane == 0) for (uint r = 0; r < RPS; ++r)
                    out[size_t(dispatch) * N + row0 + r] = as_type<float>(0x7fc00000u);
                return;
            }
            for (uint block = 0; block < K; block += 512u) {
                float values[16];
                uint cbase = block + lane * 16u;
                for (uint i = 0; i < 16; ++i)
                    values[i] = float(x[size_t(dispatch / XDIV) * K + cbase + i]);
                uint qbase = (cbase * BITS) >> 5u;
                for (uint r = 0; r < RPS; ++r) {
                    size_t row = (size_t(expert) * N + row0 + r) * WORDS;
                    \(JANGHRoutedFastDecode.codes(bits: bits))
                    float partial = 0.0f;
                    for (uint i = 0; i < 16; ++i) partial = fma(values[i], \(lv)[code[i]], partial);
                    accum[r] += partial;
                }
            }
            for (uint r = 0; r < RPS; ++r) {
                float total = simd_sum(accum[r]);
                if (lane == 0)
                    out[size_t(dispatch) * N + row0 + r] = total * float(scales[size_t(expert) * N + row0 + r]);
            }
            """
        kernel = MLXFast.metalKernel(
            name: "jangh_routed_fast_qmv_b\(bits)_r\(rowsPerSimdgroup)_c\(constantTable ? 1 : 0)_" + signature,
            inputNames: ["x", "packed", "scales", "indices"], outputNames: ["out"], source: source,
            header: header, ensureRowContiguous: false)
    }

    /// Same contract as `JANGHProjectionKernel.project` for an already-rotated input.
    func project(_ x: MLXArray, packed: MLXArray, scales: MLXArray, indices: MLXArray) -> MLXArray {
        let k = x.dim(1)
        let n = packed.dim(1)
        let flat = indices.flattened()
        return kernel(
            [contiguous(x), packed, scales, contiguous(flat)],
            template: [
                ("K", k), ("N", n), ("EXPERTS", packed.dim(0)), ("WORDS", k * bits / 32),
                ("BITS", bits), ("XDIV", flat.size / x.dim(0)), ("RPS", rowsPerSimdgroup),
            ],
            grid: (64, n / (2 * rowsPerSimdgroup), flat.size), threadGroup: (64, 1, 1),
            outputShapes: [[flat.size, n]], outputDTypes: [.float32])[0]
    }
}

/// fc2 / down + routing: twin of `JANGHWeightedDownKernel` split into route-parallel dots (pass A)
/// and an in-order routing reduction (pass B).
final class JANGHRoutedFastWeightedDown {
    private let dots: MLXFast.MLXFastKernel
    private let reduce: MLXFast.MLXFastKernel
    let bits: Int

    let rowsPerSimdgroup: Int

    init?(bits: Int, alpha: Double, beta: Double, rowsPerSimdgroup: Int = JANGHRoutedFastDecode.rows,
          constantTable: Bool = JANGHRoutedFastDecode.constantTable) {
        guard [2, 3, 4].contains(bits), [4, 8, 16].contains(rowsPerSimdgroup) else { return nil }
        self.bits = bits
        self.rowsPerSimdgroup = rowsPerSimdgroup
        let (table, signature) = JANGHRoutedFastDecode.levelTable(bits: bits, alpha: alpha, beta: beta)
        let lv = "LV_\(bits)_\(signature)"
        let header = constantTable ? "constant float \(lv)[\(1 << bits)] = {\(table)};\n" : ""
        let local = constantTable ? "" : "const float \(lv)[\(1 << bits)] = {\(table)};"
        dots = MLXFast.metalKernel(
            name: "jangh_routed_fast_down_dots_b\(bits)_r\(rowsPerSimdgroup)_c\(constantTable ? 1 : 0)_" + signature,
            inputNames: ["hidden", "packed", "indices"], outputNames: ["out"],
            source: """
                \(local)
                uint lane = thread_index_in_simdgroup;
                uint row0 = threadgroup_position_in_grid.y * (2u * RPS) + simdgroup_index_in_threadgroup * RPS;
                size_t dispatch = threadgroup_position_in_grid.z;
                uint expert = indices[dispatch];
                if (expert >= EXPERTS) return;  // pass B turns the whole token into NaN
                float accum[RPS];
                for (uint r = 0; r < RPS; ++r) accum[r] = 0.0f;
                for (uint block = 0; block < H; block += 512u) {
                    float values[16];
                    uint cbase = block + lane * 16u;
                    for (uint i = 0; i < 16; ++i) values[i] = hidden[dispatch * H + cbase + i];
                    uint qbase = (cbase * BITS) >> 5u;
                    for (uint r = 0; r < RPS; ++r) {
                        size_t row = (size_t(expert) * N + row0 + r) * WORDS;
                        \(JANGHRoutedFastDecode.codes(bits: bits))
                        for (uint i = 0; i < 16; ++i) accum[r] = fma(values[i], \(lv)[code[i]], accum[r]);
                    }
                }
                for (uint r = 0; r < RPS; ++r) {
                    float dot = simd_sum(accum[r]);
                    if (lane == 0) out[dispatch * N + row0 + r] = dot;
                }
                """, header: header, ensureRowContiguous: false)
        reduce = MLXFast.metalKernel(
            name: "jangh_routed_fast_down_reduce",
            inputNames: ["dots", "scales", "indices", "scores"], outputNames: ["out"],
            source: """
                uint row = thread_position_in_grid.x;
                uint token = thread_position_in_grid.y;
                if (row >= N) return;
                float total = 0;
                for (uint route = 0; route < ROUTES; ++route) {
                    size_t dispatch = size_t(token) * ROUTES + route;
                    uint expert = indices[dispatch];
                    if (expert >= EXPERTS) {
                        out[size_t(token) * N + row] = T(as_type<float>(0x7fc00000u));
                        return;
                    }
                    float dot = dots[dispatch * N + row];
                    total += scores[dispatch] * dot * float(scales[size_t(expert) * N + row]);
                }
                out[size_t(token) * N + row] = T(total);
                """, ensureRowContiguous: false)
    }

    /// Same contract as `JANGHWeightedDownKernel.projectPreparedHidden` (hidden already prepared).
    func project(
        _ hidden: MLXArray, packed: MLXArray, scales: MLXArray, indices: MLXArray, scores: MLXArray,
        outputDType: DType
    ) -> MLXArray {
        let h = hidden.dim(1)
        let n = packed.dim(1)
        let tokens = indices.dim(0)
        let routes = indices.dim(1)
        let idx = contiguous(indices)
        let d = dots(
            [contiguous(hidden), packed, idx],
            template: [
                ("H", h), ("N", n), ("EXPERTS", packed.dim(0)), ("WORDS", h * bits / 32),
                ("BITS", bits), ("RPS", rowsPerSimdgroup),
            ],
            grid: (64, n / (2 * rowsPerSimdgroup), tokens * routes), threadGroup: (64, 1, 1),
            outputShapes: [[tokens * routes, n]], outputDTypes: [.float32])[0]
        return reduce(
            [d, scales, idx, contiguous(scores)],
            template: [
                ("T", outputDType), ("N", n), ("ROUTES", routes), ("EXPERTS", packed.dim(0)),
            ],
            grid: (n, tokens, 1), threadGroup: (Swift.min(256, n), 1, 1),
            outputShapes: [[tokens, n]], outputDTypes: [outputDType])[0]
    }
}

/// Diagnostic (RunBench BENCH_JANGH_ROUTED_MICRO=1): GPU time of the routed decode kernels at a given
/// shape on synthetic banks, against MLX's affine 4-bit gather_qmv at the same shape. Not used by models.
public enum JANGHRoutedFastBench {
    public static func run(experts: Int = 240, routes: Int = 9, k: Int = 1024, n: Int = 2048, bits: Int = 2) -> [String] {
        var lines: [String] = []
        func time(_ reps: Int = 200, _ body: () -> MLXArray) -> Double {
            MLX.eval(body())
            var s: [Double] = []
            for _ in 0 ..< 5 {
                let t0 = CFAbsoluteTimeGetCurrent()
                var outs: [MLXArray] = []
                for _ in 0 ..< reps { outs.append(body()) }
                MLX.eval(outs)
                s.append((CFAbsoluteTimeGetCurrent() - t0) / Double(reps) * 1e6)
            }
            s.sort(); return s[2]
        }
        let words = k * bits / 32
        let packed = MLXRandom.randInt(0 ..< Int32.max, [experts, n, words]).asType(.uint32)
        let scales = (MLXRandom.uniform(low: 0.5, high: 1.5, [experts, n])).asType(.float16)
        let x = MLXRandom.normal([1, k]).asType(.bfloat16)
        let idx = MLXArray((0 ..< routes).map { UInt32(($0 * 37) % experts) }).reshaped(1, routes)
        let scores = MLXRandom.uniform(low: 0, high: 1, [1, routes]).asType(.float32)
        MLX.eval(packed, scales, x, idx, scores)
        let mb = Double(routes * n * words * 4) / 1e6
        for rows in [4, 8, 16] {
            guard let fc1 = JANGHRoutedFastQMV(bits: bits, alpha: 1.0, beta: 0.1, rowsPerSimdgroup: rows),
                let fc2 = JANGHRoutedFastWeightedDown(bits: bits, alpha: 1.0, beta: 0.1, rowsPerSimdgroup: rows)
            else { continue }
            let t1 = time { fc1.project(x, packed: packed, scales: scales, indices: idx) }
            let hidden = MLXRandom.normal([routes, k]).asType(.float32)
            MLX.eval(hidden)
            let t2 = time { fc2.project(hidden, packed: packed, scales: scales, indices: idx, scores: scores, outputDType: .bfloat16) }
            lines.append(String(format: "[ROUTED-MICRO] bits=%d rows=%d fc1(K=%d,N=%d,R=%d) %.1f us (%.0f GB/s)  down %.1f us (%.0f GB/s)", bits, rows, k, n, routes, t1, mb / t1 * 1e3, t2, mb / t2 * 1e3))
        }
        // Dependent chain of 40 layers (each layer's output feeds the next), as in one decode step.
        if k == 1024, n == 2048,
            let fc1 = JANGHRoutedFastQMV(bits: bits, alpha: 1.0, beta: 0.1, rowsPerSimdgroup: 4),
            let fc2 = JANGHRoutedFastWeightedDown(bits: bits, alpha: 1.0, beta: 0.1, rowsPerSimdgroup: 4)
        {
            let packed2 = MLXRandom.randInt(0 ..< Int32.max, [experts, k, n * bits / 32]).asType(.uint32)
            let scales2 = (MLXRandom.uniform(low: 0.5, high: 1.5, [experts, k])).asType(.float16)
            MLX.eval(packed2, scales2)
            func chain(glue: Bool) -> Double {
                time(10) {
                    var h = x
                    for _ in 0 ..< 40 {
                        var mid = fc1.project(h, packed: packed, scales: scales, indices: idx)
                        if glue {
                            let r = maximum(mid.asType(.float32), MLXArray(Float(0)))
                            mid = r * r
                        }
                        h = fc2.project(mid, packed: packed2, scales: scales2, indices: idx, scores: scores, outputDType: .bfloat16) * 1e-3
                    }
                    return h
                } / 40
            }
            lines.append(String(format: "[ROUTED-MICRO] bits=%d dependent chain per layer: kernels-only %.1f us, with relu2 glue %.1f us", bits, chain(glue: false), chain(glue: true)))
        }
        // MLX affine 4-bit gather_qmv on the same [E, N, K] bank shape, for scale.
        let wf = MLXRandom.normal([experts, n, k]).asType(.bfloat16)
        let (w, s4, b4) = MLX.quantized(wf, groupSize: 64, bits: 4)
        MLX.eval(w, s4)
        let xr = x.reshaped(1, 1, 1, k)
        let tg = time { gatherQuantizedMM(xr, w, scales: s4, biases: b4, rhsIndices: idx.reshaped(1, 1, routes), transpose: true, groupSize: 64, bits: 4) }
        let mb4 = Double(routes * n * k / 2) / 1e6
        lines.append(String(format: "[ROUTED-MICRO] mlx affine4 gather_qmv %.1f us (%.0f GB/s, %.1f MB)", tg, mb4 / tg * 1e3, mb4))
        return lines
    }
}
