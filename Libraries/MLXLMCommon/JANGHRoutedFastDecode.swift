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
            && n.isMultiple(of: 8)
    }
}

/// fc1 / up: twin of `JANGHProjectionKernel`'s qmv (activation already in the projection basis).
final class JANGHRoutedFastQMV {
    private let kernel: MLXFast.MLXFastKernel
    let bits: Int

    init?(bits: Int, alpha: Double, beta: Double) {
        guard [2, 3, 4].contains(bits) else { return nil }
        self.bits = bits
        let (table, signature) = JANGHRoutedFastDecode.levelTable(bits: bits, alpha: alpha, beta: beta)
        let source = """
            const float LV[\(1 << bits)] = {\(table)};
            uint lane = thread_index_in_simdgroup;
            uint row0 = threadgroup_position_in_grid.y * 8u + simdgroup_index_in_threadgroup * 4u;
            uint dispatch = threadgroup_position_in_grid.z;
            uint expert = indices[dispatch];
            float accum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            if (expert >= EXPERTS) {
                if (lane == 0) for (uint r = 0; r < 4; ++r)
                    out[size_t(dispatch) * N + row0 + r] = as_type<float>(0x7fc00000u);
                return;
            }
            for (uint block = 0; block < K; block += 512u) {
                float values[16];
                uint cbase = block + lane * 16u;
                for (uint i = 0; i < 16; ++i)
                    values[i] = float(x[size_t(dispatch / XDIV) * K + cbase + i]);
                uint qbase = (cbase * BITS) >> 5u;
                for (uint r = 0; r < 4; ++r) {
                    size_t row = (size_t(expert) * N + row0 + r) * WORDS;
                    \(JANGHRoutedFastDecode.codes(bits: bits))
                    float partial = 0.0f;
                    for (uint i = 0; i < 16; ++i) partial = fma(values[i], LV[code[i]], partial);
                    accum[r] += partial;
                }
            }
            for (uint r = 0; r < 4; ++r) {
                float total = simd_sum(accum[r]);
                if (lane == 0)
                    out[size_t(dispatch) * N + row0 + r] = total * float(scales[size_t(expert) * N + row0 + r]);
            }
            """
        kernel = MLXFast.metalKernel(
            name: "jangh_routed_fast_qmv_b\(bits)_" + signature,
            inputNames: ["x", "packed", "scales", "indices"], outputNames: ["out"], source: source,
            ensureRowContiguous: false)
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
                ("BITS", bits), ("XDIV", flat.size / x.dim(0)),
            ],
            grid: (64, n / 8, flat.size), threadGroup: (64, 1, 1),
            outputShapes: [[flat.size, n]], outputDTypes: [.float32])[0]
    }
}

/// fc2 / down + routing: twin of `JANGHWeightedDownKernel` split into route-parallel dots (pass A)
/// and an in-order routing reduction (pass B).
final class JANGHRoutedFastWeightedDown {
    private let dots: MLXFast.MLXFastKernel
    private let reduce: MLXFast.MLXFastKernel
    let bits: Int

    init?(bits: Int, alpha: Double, beta: Double) {
        guard [2, 3, 4].contains(bits) else { return nil }
        self.bits = bits
        let (table, signature) = JANGHRoutedFastDecode.levelTable(bits: bits, alpha: alpha, beta: beta)
        dots = MLXFast.metalKernel(
            name: "jangh_routed_fast_down_dots_b\(bits)_" + signature,
            inputNames: ["hidden", "packed", "indices"], outputNames: ["out"],
            source: """
                const float LV[\(1 << bits)] = {\(table)};
                uint lane = thread_index_in_simdgroup;
                uint row0 = threadgroup_position_in_grid.y * 8u + simdgroup_index_in_threadgroup * 4u;
                size_t dispatch = threadgroup_position_in_grid.z;
                uint expert = indices[dispatch];
                if (expert >= EXPERTS) return;  // pass B turns the whole token into NaN
                float accum[4] = {0, 0, 0, 0};
                for (uint block = 0; block < H; block += 512u) {
                    float values[16];
                    uint cbase = block + lane * 16u;
                    for (uint i = 0; i < 16; ++i) values[i] = hidden[dispatch * H + cbase + i];
                    uint qbase = (cbase * BITS) >> 5u;
                    for (uint r = 0; r < 4; ++r) {
                        size_t row = (size_t(expert) * N + row0 + r) * WORDS;
                        \(JANGHRoutedFastDecode.codes(bits: bits))
                        for (uint i = 0; i < 16; ++i) accum[r] = fma(values[i], LV[code[i]], accum[r]);
                    }
                }
                for (uint r = 0; r < 4; ++r) {
                    float dot = simd_sum(accum[r]);
                    if (lane == 0) out[dispatch * N + row0 + r] = dot;
                }
                """, ensureRowContiguous: false)
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
                ("BITS", bits),
            ],
            grid: (64, n / 8, tokens * routes), threadGroup: (64, 1, 1),
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
