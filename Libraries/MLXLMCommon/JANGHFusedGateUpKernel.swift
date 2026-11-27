import Foundation
import MLX
import MLXFast

#if canImport(CryptoKit)
    import CryptoKit
#else
    import Crypto
#endif

/// Decode-only experimental primitive; not installed by any model factory.
/// Prepared inputs and hidden outputs stay F32 across H32/activation boundaries.
final class JANGHFusedGateUpKernel {
    let identity: String
    private let gateBits: Int
    private let upBits: Int
    let inputRotation: JANGHFormatContract.Rotation
    private let outputRotation: JANGHFormatContract.Rotation
    private let fused: MLXFast.MLXFastKernel
    private let fusedYZ: MLXFast.MLXFastKernel
    private let h32: MLXFast.MLXFastKernel

    init(
        contract: JANGHFormatContract, gateModule: String, upModule: String,
        outputRotation: JANGHFormatContract.Rotation, activation: JANGTActivation = .silu
    ) throws {
        guard let gate = contract.projections[gateModule], let up = contract.projections[upModule],
            gate.rotation == up.rotation,
            let gateBook = contract.codebooks[gate.bits], let upBook = contract.codebooks[up.bits]
        else {
            throw JANGHFormatContract.ValidationError.invalid(
                "incompatible fused JANGH projections")
        }
        gateBits = gate.bits
        upBits = up.bits
        inputRotation = gate.rotation
        self.outputRotation = outputRotation
        let description =
            "jangh-fused-gu-v3-split|\(gate.bits)|\(up.bits)|\(gate.rotation.rawValue)|"
            + "\(outputRotation.rawValue)|\(gateBook.alpha.bitPattern)|\(gateBook.beta.bitPattern)|"
            + "\(upBook.alpha.bitPattern)|\(upBook.beta.bitPattern)" + activation.nameTag
        identity = SHA256.hash(data: Data(description.utf8)).map { String(format: "%02x", $0) }
            .joined()
        func projection(_ name: String, bits: Int, book: JANGHFormatContract.Codebook, accumName: String) -> String {
            let dot = JANGHDecodeQDot.source(
                bits: bits, packed: "packed_\(name)", rowBase: "base_\(name)",
                columnBase: "block + lane * 16u", values: "values",
                accumulator: "partial_\(name)", alpha: Float(book.alpha), beta: Float(book.beta))
            return """
                size_t base_\(name) = (size_t(expert) * N + row0 + r) * WORDS_\(name);
                float partial_\(name) = 0.0f;
                if (block + lane * 16u < K) {
                    \(dot)
                }
                \(accumName)[r] += partial_\(name);
                """
        }

        let source: String
        if outputRotation == .hadamard32 {
            source = """
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
                    _Pragma("clang loop unroll(full)")
                    for (uint i = 0; i < 16; ++i) {
                        uint column = block + lane * 16u + i;
                        values[i] = column < K ? float(x[size_t(dispatch / XDIV) * K + column]) : 0.0f;
                    }
                    _Pragma("clang loop unroll(full)")
                    for (uint r = 0; r < 4; ++r) {
                        if (row0 + r >= N) continue;
                        \(projection("g", bits: gate.bits, book: gateBook, accumName: "accum_g"))
                        \(projection("u", bits: up.bits, book: upBook, accumName: "accum_u"))
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
                        hidden[sg * 4u + r] = \(activation.metalValue());
                    }
                }
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
                """
        } else {
            // Split-simdgroup fused gate+up: sg0 decodes gate (4 rows) and sg1 decodes up (same 4 rows),
            // passing gate via 16-byte threadgroup memory. Halves live accumulators to prevent register
            // spills under multi-row verify occupancy while staying bit-exact.
            source = """
                uint lane = thread_index_in_simdgroup;
                uint sg = simdgroup_index_in_threadgroup;
                uint row0 = threadgroup_position_in_grid.y * 4u;
                uint dispatch = threadgroup_position_in_grid.z;
                uint expert = indices[dispatch];
                // Expert is uniform across every thread in this threadgroup: no partial barrier exit.
                if (expert >= EXPERTS) {
                    if (sg == 0u && lane == 0u) for (uint r = 0; r < 4; ++r)
                        if (row0 + r < N) out[size_t(dispatch) * N + row0 + r] = as_type<float>(0x7fc00000u);
                    return;
                }
                float accum[4] = {0, 0, 0, 0};
                if (sg == 0u) {
                    for (uint block = 0; block < K; block += 512u) {
                        float values[16];
                        _Pragma("clang loop unroll(full)")
                        for (uint i = 0; i < 16; ++i) {
                            uint column = block + lane * 16u + i;
                            values[i] = column < K ? float(x[size_t(dispatch / XDIV) * K + column]) : 0.0f;
                        }
                        _Pragma("clang loop unroll(full)")
                        for (uint r = 0; r < 4; ++r) {
                            if (row0 + r >= N) continue;
                            \(projection("g", bits: gate.bits, book: gateBook, accumName: "accum"))
                        }
                    }
                } else {
                    for (uint block = 0; block < K; block += 512u) {
                        float values[16];
                        _Pragma("clang loop unroll(full)")
                        for (uint i = 0; i < 16; ++i) {
                            uint column = block + lane * 16u + i;
                            values[i] = column < K ? float(x[size_t(dispatch / XDIV) * K + column]) : 0.0f;
                        }
                        _Pragma("clang loop unroll(full)")
                        for (uint r = 0; r < 4; ++r) {
                            if (row0 + r >= N) continue;
                            \(projection("u", bits: up.bits, book: upBook, accumName: "accum"))
                        }
                    }
                }
                threadgroup float g_sh[4];
                float val[4];
                _Pragma("clang loop unroll(full)")
                for (uint r = 0; r < 4; ++r) {
                    float s = simd_sum(accum[r]);
                    if (lane == 0u && row0 + r < N) {
                        float sc = (sg == 0u) ? float(scales_g[size_t(expert) * N + row0 + r])
                                              : float(scales_u[size_t(expert) * N + row0 + r]);
                        val[r] = s * sc;
                    }
                }
                if (sg == 0u && lane == 0u) {
                    _Pragma("clang loop unroll(full)")
                    for (uint r = 0; r < 4; ++r) g_sh[r] = val[r];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (sg == 1u && lane == 0u) {
                    float limit = limit_value[0];
                    _Pragma("clang loop unroll(full)")
                    for (uint r = 0; r < 4; ++r) {
                        if (row0 + r < N) {
                            float g = g_sh[r], u = val[r];
                            if (limit > 0.0f) { g = metal::min(g, limit); u = metal::clamp(u, -limit, limit); }
                            out[size_t(dispatch) * N + row0 + r] = \(activation.metalValue());
                        }
                    }
                }
                """
        }
        fused = MLXFast.metalKernel(
            name: "jangh_fused_gu_" + identity,
            inputNames: [
                "x", "packed_g", "scales_g", "packed_u", "scales_u", "indices", "limit_value",
            ],
            outputNames: ["out"], source: source, ensureRowContiguous: false)
        fusedYZ = MLXFast.metalKernel(
            name: "jangh_fused_gu_yz_" + identity,
            inputNames: [
                "x", "packed_g", "scales_g", "packed_u", "scales_u", "indices", "limit_value",
            ],
            outputNames: ["out"], source: JANGTKernelSource.swapGridYZ(source), ensureRowContiguous: false)
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
        upPacked: MLXArray, upScales: MLXArray, indices: MLXArray, limit: Float?,
        slotFastest: Bool = false
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
        let k = input.dim(1)
        let n = gatePacked.dim(1)
        let experts = gatePacked.dim(0)
        let gWidth = k.multipliedReportingOverflow(by: gateBits)
        let uWidth = k.multipliedReportingOverflow(by: upBits)
        guard !gWidth.overflow, !uWidth.overflow,
            gWidth.partialValue <= Int(UInt32.max), uWidth.partialValue <= Int(UInt32.max),
            gatePacked.shape == [experts, n, gWidth.partialValue / 32],
            upPacked.shape == [experts, n, uWidth.partialValue / 32],
            gateScales.shape == [experts, n], upScales.shape == [experts, n],
            outputRotation != .hadamard32 || n.isMultiple(of: 32)
        else { throw JANGHFormatContract.ValidationError.invalid("invalid fused JANGH geometry") }
        try JANGHBankLayout.requireReadyRowContiguous(gatePacked, role: "gate packed")
        try JANGHBankLayout.requireReadyRowContiguous(gateScales, role: "gate scales")
        try JANGHBankLayout.requireReadyRowContiguous(upPacked, role: "up packed")
        try JANGHBankLayout.requireReadyRowContiguous(upScales, role: "up scales")
        let rotated = outputRotation == .hadamard32
        let rows = rotated ? 32 : 4
        let threads = rotated ? 256 : 64
        let tiles = (n + rows - 1) / rows
        let yz = slotFastest && input.dim(0) > 1
        let kExec = yz ? fusedYZ : fused
        let grid = yz ? (threads, indices.size, tiles) : (threads, tiles, indices.size)
        return kExec(
            [
                contiguous(input), gatePacked, gateScales,
                upPacked, upScales, contiguous(indices.flattened()),
                MLXArray([limit ?? 0]),
            ],
            template: [
                ("K", k), ("N", n), ("EXPERTS", experts), ("ROWS", rows),
                ("WORDS_g", gWidth.partialValue / 32), ("WORDS_u", uWidth.partialValue / 32),
                ("XDIV", indices.size / input.dim(0)),
            ],
            grid: grid, threadGroup: (threads, 1, 1),
            outputShapes: [[indices.size, n]], outputDTypes: [.float32])[0]
    }
}
