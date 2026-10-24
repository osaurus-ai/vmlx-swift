import Foundation
import MLX
import MLXFast

#if canImport(CryptoKit)
    import CryptoKit
#else
    import Crypto
#endif

/// Executable JANGH building blocks. Not registered as a model loader.
/// QMV returns F32, matching the reference's projection accumulation boundary.
final class JANGHProjectionKernel {
    let identity: String
    private let bits: Int
    private let rotation: JANGHFormatContract.Rotation
    private let qmv: MLXFast.MLXFastKernel
    private let h32: MLXFast.MLXFastKernel
    private let fuseFloatRotation: Bool

    /// Dense decode keeps H32 in float registers through the dot. The default
    /// preserves the existing activation-dtype rotation contract for other users.
    init(contract: JANGHFormatContract, module: String, fuseFloatRotation: Bool = false) throws {
        guard let projection = contract.projections[module],
            let book = contract.codebooks[projection.bits]
        else { throw JANGHFormatContract.ValidationError.invalid("missing JANGH projection") }
        bits = projection.bits
        rotation = projection.rotation
        self.fuseFloatRotation = fuseFloatRotation
        let version = fuseFloatRotation ? "jangh-qmv-dense-float-h32-v2" : "jangh-qmv-v1"
        let description =
            "\(version)|\(bits)|\(rotation.rawValue)|\(book.alpha.bitPattern)|\(book.beta.bitPattern)"
        identity = SHA256.hash(data: Data(description.utf8)).map { String(format: "%02x", $0) }
            .joined()
        let alpha = String(Float(book.alpha)) + "f"
        let beta = String(Float(book.beta)) + "f"
        let center = String(Float((1 << bits) - 1) / 2) + "f"
        // Each lane holds 16 consecutive values. Four local butterfly stages
        // and one lane-pair shuffle form H32 without an intermediate BF16/F16
        // store. K%32 guarantees both lanes of every live pair are present;
        // inactive pairs carry zeros and participate in the same shuffle.
        let fusedRotation = fuseFloatRotation && rotation == .hadamard32 ? """
            for (uint stage = 1u; stage < 16u; stage <<= 1u) {
                for (uint i = 0; i < 16u; ++i) {
                    if ((i & stage) == 0u) {
                        float a = values[i], b = values[i + stage];
                        values[i] = a + b;
                        values[i + stage] = a - b;
                    }
                }
            }
            for (uint i = 0; i < 16u; ++i) {
                float other = simd_shuffle_xor(values[i], 1u);
                values[i] = ((lane & 1u) ? other - values[i] : values[i] + other)
                    * 0.17677669529663687f;
            }
            """ : ""
        let source = """
            uint lane = thread_index_in_simdgroup;
            uint row0 = threadgroup_position_in_grid.y * 8u + simdgroup_index_in_threadgroup * 4u;
            uint dispatch = threadgroup_position_in_grid.z;
            uint expert = indices[dispatch];
            float accum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            if (expert >= EXPERTS) {
                if (lane == 0) for (uint r = 0; r < 4; ++r)
                    if (row0 + r < N) out[size_t(dispatch) * N + row0 + r] = as_type<float>(0x7fc00000u);
                return;
            }
            for (uint block = 0; block < K; block += 512u) {
                float values[16];
                for (uint i = 0; i < 16; ++i) {
                    uint column = block + lane * 16u + i;
                    values[i] = column < K ? float(x[size_t(dispatch / XDIV) * K + column]) : 0.0f;
                }
                \(fusedRotation)
                for (uint r = 0; r < 4; ++r) {
                    if (row0 + r >= N) continue;
                    size_t row = (size_t(expert) * N + row0 + r) * WORDS;
                    float partial = 0.0f;
                    for (uint i = 0; i < 16; ++i) {
                        uint column = block + lane * 16u + i;
                        if (column >= K) continue;
                        uint bit = column * BITS;
                        uint shift = bit & 31u;
                        uint word = bit >> 5u;
                        uint code = packed[row + word] >> shift;
                        if (shift + BITS > 32u) code |= packed[row + word + 1] << (32u - shift);
                        code &= (1u << BITS) - 1u;
                        float u = float(code) - \(center);
                        float level = u * fma(\(beta), u * u, \(alpha));
                        partial = fma(values[i], level, partial);
                    }
                    accum[r] += partial;
                }
            }
            for (uint r = 0; r < 4; ++r) {
                float total = simd_sum(accum[r]);
                if (lane == 0 && row0 + r < N)
                    out[size_t(dispatch) * N + row0 + r] = total * float(scales[size_t(expert) * N + row0 + r]);
            }
            """
        qmv = MLXFast.metalKernel(
            name: "jangh_qmv_" + identity,
            inputNames: ["x", "packed", "scales", "indices"], outputNames: ["out"], source: source,
            ensureRowContiguous: false)
        h32 = MLXFast.metalKernel(
            name: "jangh_h32_v1", inputNames: ["x"], outputNames: ["out"],
            source: """
                uint lane = thread_index_in_simdgroup;
                size_t offset = size_t(threadgroup_position_in_grid.z) * K
                    + size_t(threadgroup_position_in_grid.y) * 32u + lane;
                float value = float(x[offset]);
                for (ushort stage = 1; stage < 32; stage <<= 1) {
                    float other = simd_shuffle_xor(value, stage);
                    value = (lane & stage) ? other - value : value + other;
                }
                out[offset] = T(value * 0.17677669529663687f);
                """)
    }

    func hadamard32(_ input: MLXArray) throws -> MLXArray {
        guard input.ndim > 0, input.dim(-1) > 0, input.dim(-1).isMultiple(of: 32),
            [.float16, .bfloat16, .float32].contains(input.dtype)
        else { throw JANGHFormatContract.ValidationError.invalid("invalid H32 activation") }
        let width = input.dim(-1)
        guard input.size > 0 else { return input }
        return h32(
            [contiguous(input)], template: [("T", input.dtype), ("K", width)],
            grid: (32, width / 32, input.size / width), threadGroup: (32, 1, 1),
            outputShapes: [input.shape], outputDTypes: [input.dtype])[0]
    }

    /// Input [tokens,K], packed [experts,N,K*bits/32], row scales [experts,N].
    /// Routes are grouped by input token. Invalid expert IDs produce NaN without
    /// reading outside the bank; trusted router callers should never produce them.
    func project(_ input: MLXArray, packed: MLXArray, scales: MLXArray, indices: MLXArray) throws
        -> MLXArray
    {
        guard input.ndim == 2, packed.ndim == 3, scales.ndim == 2,
            input.dim(0) > 0, input.dim(1) > 0, input.dim(1).isMultiple(of: 32),
            packed.dim(0) > 0, packed.dim(1) > 0,
            packed.dtype == .uint32, scales.dtype == .float16, indices.dtype == .uint32,
            indices.size > 0, indices.size.isMultiple(of: input.dim(0)),
            [.float16, .bfloat16, .float32].contains(input.dtype)
        else {
            throw JANGHFormatContract.ValidationError.invalid("invalid JANGH projection tensors")
        }
        let width = input.dim(1)
        let bitWidth = width.multipliedReportingOverflow(by: bits)
        guard !bitWidth.overflow, bitWidth.partialValue <= Int(UInt32.max),
            packed.shape == [packed.dim(0), packed.dim(1), bitWidth.partialValue / 32],
            scales.shape == [packed.dim(0), packed.dim(1)]
        else { throw JANGHFormatContract.ValidationError.invalid("invalid JANGH packed geometry") }
        try JANGHBankLayout.requireReadyRowContiguous(packed, role: "packed projection")
        try JANGHBankLayout.requireReadyRowContiguous(scales, role: "projection scales")
        let x = rotation == .hadamard32 && !fuseFloatRotation ? try hadamard32(input) : input
        return qmv(
            [
                contiguous(x), packed, scales,
                contiguous(indices.flattened()),
            ],
            template: [
                ("K", width), ("N", packed.dim(1)), ("EXPERTS", packed.dim(0)),
                ("WORDS", bitWidth.partialValue / 32), ("BITS", bits),
                ("XDIV", indices.size / input.dim(0)),
            ],
            grid: (64, (packed.dim(1) + 7) / 8, indices.size), threadGroup: (64, 1, 1),
            outputShapes: [[indices.size, packed.dim(1)]], outputDTypes: [.float32])[0]
    }
}

/// Fast single-expert (dense) JANGH decode QMV for 2- and 4-bit codebooks (Qwen3.8-27B JANGH2 MLP, 2026-10-06).
///
/// WHY: `JANGHProjectionKernel`'s qmv decodes every code with its own word load, shift/mask and the cubic codebook
/// formula `u * fma(beta, u*u, alpha)` — fine for routed experts (640-wide banks, top-10), but for one dense
/// 17408x5120 bank per projection it is compute-bound: 27B JANGH2 AR ran 12 tok/s vs 23 tok/s on the 4-bit
/// JANG_4D. This kernel keeps the SAME per-row statement order (x values, then for each of 4 rows a partial fma chain
/// over the lane's 16 values, `accum += partial`, simd_sum, `* scale`) but loads the lane's packed words once per
/// (row, block) into registers and reads levels from a table holding the exact bit patterns of the formula's
/// float result. Admitted per immutable projection and input variant after a bitwise comparison with `JANGHProjectionKernel`
/// (`JANGHDenseFastAdmission`); otherwise the caller keeps the original kernel.
final class JANGHDenseFastQMV {
    private let kernel: MLXFast.MLXFastKernel
    let bits: Int

    init?(bits: Int, alpha: Double, beta: Double) {
        guard bits == 2 || bits == 4 else { return nil }
        self.bits = bits
        let a = Float(alpha)
        let b = Float(beta)
        let center = Float((1 << bits) - 1) / 2
        // Same float operations as the reference kernel: u = float(code) - center; level = u * fma(beta, u*u, alpha).
        let levels = (0 ..< (1 << bits)).map { code -> UInt32 in
            let u = Float(code) - center
            let level = u * a.addingProduct(b, u * u)
            return level.bitPattern
        }
        let table = levels.map { "as_type<float>(0x\(String($0, radix: 16))u)" }.joined(
            separator: ", ")
        let laneWords = 16 * bits / 32
        let source = """
            const float LV[\(1 << bits)] = {\(table)};
            uint lane = thread_index_in_simdgroup;
            uint row0 = threadgroup_position_in_grid.y * 8u + simdgroup_index_in_threadgroup * 4u;
            uint t = threadgroup_position_in_grid.z;
            float accum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            for (uint block = 0; block < K; block += 512u) {
                float values[16];
                for (uint i = 0; i < 16u; ++i) values[i] = float(x[size_t(t) * K + block + lane * 16u + i]);
                uint w0 = ((block + lane * 16u) * BITS) >> 5u;
                for (uint r = 0; r < 4u; ++r) {
                    size_t row = size_t(row0 + r) * WORDS;
                    uint wv[\(laneWords)];
                    for (uint j = 0; j < \(laneWords)u; ++j) wv[j] = packed[row + w0 + j];
                    float partial = 0.0f;
                    for (uint i = 0; i < 16u; ++i) {
                        uint bit = i * BITS;
                        uint code = (wv[bit >> 5u] >> (bit & 31u)) & ((1u << BITS) - 1u);
                        partial = fma(values[i], LV[code], partial);
                    }
                    accum[r] += partial;
                }
            }
            for (uint r = 0; r < 4u; ++r) {
                float total = simd_sum(accum[r]);
                if (lane == 0) out[size_t(t) * N + row0 + r] = total * float(scales[row0 + r]);
            }
            """
        kernel = MLXFast.metalKernel(
            name: "jangh_dense_fast_qmv_b\(bits)_" + levels.map { String($0, radix: 16) }.joined(),
            inputNames: ["x", "packed", "scales"], outputNames: ["out"], source: source,
            ensureRowContiguous: true)
    }

    func project(_ x: MLXArray, packed: MLXArray, scales: MLXArray) -> MLXArray {
        let k = x.dim(1)
        let n = packed.dim(1)
        return kernel(
            [x, packed, scales],
            template: [("K", k), ("N", n), ("WORDS", k * bits / 32), ("BITS", bits)],
            grid: (64, n / 8, x.dim(0)), threadGroup: (64, 1, 1),
            outputShapes: [[x.dim(0), n]], outputDTypes: [.float32])[0]
    }

    static func eligible(k: Int, n: Int, bits: Int) -> Bool {
        (bits == 2 || bits == 4) && k > 0 && k.isMultiple(of: 512) && n > 0 && n.isMultiple(of: 8)
    }
}

/// An admission belongs to one immutable projection/bank owner, never a process-wide
/// shape. Geometry/dtype variants are keyed separately by the owner.
final class JANGHDenseFastAdmission {
    private let lock = NSLock()
    private var verdicts: [String: Bool] = [:]
    private let isEnabled: Bool
    static let enabled = RuntimeEnvironment.value("VMLX_JANGH_DENSE_FAST") != "0"

    init(enabled: Bool) { isEnabled = enabled }

    static func bitwiseEqual(_ got: MLXArray, _ want: MLXArray) -> Bool {
        guard got.shape == want.shape, got.dtype == want.dtype else { return false }
        let bits: DType
        switch got.dtype {
        case .float32: bits = .uint32
        case .float16, .bfloat16: bits = .uint16
        default: return false
        }
        return all(got.view(dtype: bits) .== want.view(dtype: bits)).item(Bool.self)
    }

    func admits(key: String, compare: () throws -> (MLXArray, MLXArray)) -> Bool {
        guard isEnabled, !CompiledDecodeTrace.isActive else { return false }
        lock.lock()
        defer { lock.unlock() }
        if let verdict = verdicts[key] { return verdict }
        let equal: Bool
        if let (got, want) = try? compare() {
            equal = Self.bitwiseEqual(got, want)
        } else {
            equal = false
        }
        verdicts[key] = equal
        RuntimeDiagnostics.write(
            Data("[JANGH] projection admission \(key) admitted=\(equal)\n".utf8))
        return equal
    }
}
