import CryptoKit
import Foundation
import MLX
import MLXFast

/// Decode-only experimental primitive; not installed by any model factory.
/// Prepared inputs and hidden outputs stay F32 across H32/activation boundaries.
final class JANGHFusedGateUpKernel {
    let identity: String
    private let gateBits: Int
    private let upBits: Int
    private let inputRotation: JANGHFormatContract.Rotation
    private let outputRotation: JANGHFormatContract.Rotation
    private let fused: MLXFast.MLXFastKernel
    private let h32: MLXFast.MLXFastKernel

    init(
        contract: JANGHFormatContract, gateModule: String, upModule: String,
        outputRotation: JANGHFormatContract.Rotation
    ) throws {
        guard let gate = contract.projections[gateModule], let up = contract.projections[upModule],
            gate.rotation == up.rotation,
            let gateBook = contract.codebooks[gate.bits], let upBook = contract.codebooks[up.bits]
        else { throw JANGHFormatContract.ValidationError.invalid("incompatible fused JANGH projections") }
        gateBits = gate.bits
        upBits = up.bits
        inputRotation = gate.rotation
        self.outputRotation = outputRotation
        let description = "jangh-fused-gu-v1|\(gate.bits)|\(up.bits)|\(gate.rotation.rawValue)|"
            + "\(outputRotation.rawValue)|\(gateBook.alpha.bitPattern)|\(gateBook.beta.bitPattern)|"
            + "\(upBook.alpha.bitPattern)|\(upBook.beta.bitPattern)"
        identity = SHA256.hash(data: Data(description.utf8)).map { String(format: "%02x", $0) }.joined()
        func projection(_ name: String, bits: Int, book: JANGHFormatContract.Codebook) -> String {
            let center = Float((1 << bits) - 1) / 2
            return """
                size_t base_\(name) = (size_t(expert) * N + row0 + r) * WORDS_\(name);
                float partial_\(name) = 0.0f;
                for (uint i = 0; i < 16; ++i) {
                    uint column = block + lane * 16u + i;
                    if (column >= K) continue;
                    uint bit = column * \(bits)u;
                    uint shift = bit & 31u;
                    uint word = bit >> 5u;
                    uint code = packed_\(name)[base_\(name) + word] >> shift;
                    if (shift + \(bits)u > 32u)
                        code |= packed_\(name)[base_\(name) + word + 1] << (32u - shift);
                    code &= (1u << \(bits)u) - 1u;
                    float u = float(code) - \(center)f;
                    float level = u * fma(\(Float(book.beta))f, u * u, \(Float(book.alpha))f);
                    partial_\(name) = fma(values[i], level, partial_\(name));
                }
                accum_\(name)[r] += partial_\(name);
                """
        }
        let source = """
            uint lane = thread_index_in_simdgroup;
            uint sg = simdgroup_index_in_threadgroup;
            uint row0 = threadgroup_position_in_grid.y * ROWS + sg * 4u;
            uint dispatch = threadgroup_position_in_grid.z;
            uint expert = indices[dispatch];
            // Expert is uniform across every thread in this threadgroup: no partial barrier exit.
            if (expert >= EXPERTS) {
                if (lane == 0) for (uint r = 0; r < 4; ++r)
                    if (row0 + r < N) out[size_t(dispatch) * N + row0 + r] = as_type<float>(0x7fc00000u);
                return;
            }
            float accum_g[4] = {0, 0, 0, 0};
            float accum_u[4] = {0, 0, 0, 0};
            for (uint block = 0; block < K; block += 512u) {
                float values[16];
                for (uint i = 0; i < 16; ++i) {
                    uint column = block + lane * 16u + i;
                    values[i] = column < K ? float(x[size_t(dispatch / XDIV) * K + column]) : 0.0f;
                }
                for (uint r = 0; r < 4; ++r) {
                    if (row0 + r >= N) continue;
                    \(projection("g", bits: gate.bits, book: gateBook))
                    \(projection("u", bits: up.bits, book: upBook))
                }
            }
            threadgroup float hidden[32];
            for (uint r = 0; r < 4; ++r) {
                float g = simd_sum(accum_g[r]);
                float u = simd_sum(accum_u[r]);
                if (lane == 0 && row0 + r < N) {
                    g *= float(scales_g[size_t(expert) * N + row0 + r]);
                    u *= float(scales_u[size_t(expert) * N + row0 + r]);
                    float limit = limit_value[0];
                    if (limit > 0.0f) { g = metal::min(g, limit); u = metal::clamp(u, -limit, limit); }
                    float h = (g / (1.0f + metal::fast::exp(-g))) * u;
                    if (ROT_OUT) hidden[sg * 4u + r] = h;
                    else out[size_t(dispatch) * N + row0 + r] = h;
                }
            }
            if (ROT_OUT) {
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (sg == 0) {
                    float value = hidden[lane];
                    for (ushort stage = 1; stage < 32; stage <<= 1) {
                        float other = simd_shuffle_xor(value, stage);
                        value = (lane & stage) ? other - value : value + other;
                    }
                    out[size_t(dispatch) * N + threadgroup_position_in_grid.y * 32u + lane]
                        = value * 0.17677669529663687f;
                }
            }
            """
        fused = MLXFast.metalKernel(
            name: "jangh_fused_gu_" + identity,
            inputNames: ["x", "packed_g", "scales_g", "packed_u", "scales_u", "indices", "limit_value"],
            outputNames: ["out"], source: source)
        h32 = MLXFast.metalKernel(
            name: "jangh_decode_h32_f32_v1", inputNames: ["x"], outputNames: ["out"],
            source: """
                uint lane = thread_index_in_simdgroup;
                size_t offset = size_t(threadgroup_position_in_grid.z) * K
                    + size_t(threadgroup_position_in_grid.y) * 32u + lane;
                float value = float(x[offset]);
                for (ushort stage = 1; stage < 32; stage <<= 1) {
                    float other = simd_shuffle_xor(value, stage);
                    value = (lane & stage) ? other - value : value + other;
                }
                out[offset] = value * 0.17677669529663687f;
                """)
    }

    /// Mirrors the reference's fused decode branch: H32 promotes to F32, without
    /// a BF16/F16 cast between rotation and projection. No rotation preserves dtype.
    func prepareInputForFusedDecode(_ input: MLXArray) throws -> MLXArray {
        try validateInput(input)
        guard inputRotation == .hadamard32 else { return input }
        return h32(
            [contiguous(input)], template: [("K", input.dim(1))],
            grid: (32, input.dim(1) / 32, input.dim(0)), threadGroup: (32, 1, 1),
            outputShapes: [input.shape], outputDTypes: [.float32])[0]
    }

    private func validateInput(_ input: MLXArray) throws {
        guard input.ndim == 2, input.dim(0) > 0, input.dim(1) > 0,
            input.dim(1).isMultiple(of: 32),
            [.float16, .bfloat16, .float32].contains(input.dtype)
        else { throw JANGHFormatContract.ValidationError.invalid("invalid fused JANGH input") }
    }

    /// Input is already in the gate/up basis; never rotates it a second time.
    /// Output H32, if selected, rotates the down projection INPUT in F32.
    func activatePreparedInput(
        _ input: MLXArray, gatePacked: MLXArray, gateScales: MLXArray,
        upPacked: MLXArray, upScales: MLXArray, indices: MLXArray, limit: Float?
    ) throws -> MLXArray {
        try validateInput(input)
        if let limit, !limit.isFinite || limit <= 0 {
            throw JANGHFormatContract.ValidationError.invalid("invalid SwiGLU limit")
        }
        guard gatePacked.ndim == 3, upPacked.ndim == 3,
            gatePacked.dim(0) > 0, gatePacked.dim(1) > 0,
            gatePacked.dtype == .uint32, upPacked.dtype == .uint32,
            gateScales.dtype == .float16, upScales.dtype == .float16,
            indices.dtype == .uint32, indices.size > 0,
            indices.size.isMultiple(of: input.dim(0))
        else { throw JANGHFormatContract.ValidationError.invalid("invalid fused JANGH tensors") }
        let k = input.dim(1), n = gatePacked.dim(1), experts = gatePacked.dim(0)
        let gWidth = k.multipliedReportingOverflow(by: gateBits)
        let uWidth = k.multipliedReportingOverflow(by: upBits)
        guard !gWidth.overflow, !uWidth.overflow,
            gWidth.partialValue <= Int(UInt32.max), uWidth.partialValue <= Int(UInt32.max),
            gatePacked.shape == [experts, n, gWidth.partialValue / 32],
            upPacked.shape == [experts, n, uWidth.partialValue / 32],
            gateScales.shape == [experts, n], upScales.shape == [experts, n],
            outputRotation != .hadamard32 || n.isMultiple(of: 32)
        else { throw JANGHFormatContract.ValidationError.invalid("invalid fused JANGH geometry") }
        let rotated = outputRotation == .hadamard32
        let rows = rotated ? 32 : 8
        let threads = rotated ? 256 : 64
        return fused(
            [contiguous(input), contiguous(gatePacked), contiguous(gateScales),
             contiguous(upPacked), contiguous(upScales), contiguous(indices.flattened()),
             MLXArray([limit ?? 0])],
            template: [("K", k), ("N", n), ("EXPERTS", experts), ("ROWS", rows),
                       ("WORDS_g", gWidth.partialValue / 32), ("WORDS_u", uWidth.partialValue / 32),
                       ("XDIV", indices.size / input.dim(0)), ("ROT_OUT", rotated)],
            grid: (threads, (n + rows - 1) / rows, indices.size), threadGroup: (threads, 1, 1),
            outputShapes: [[indices.size, n]], outputDTypes: [.float32])[0]
    }
}
