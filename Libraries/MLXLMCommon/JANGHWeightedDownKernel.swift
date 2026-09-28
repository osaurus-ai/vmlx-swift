import Foundation
import MLX
import MLXFast

#if canImport(CryptoKit)
    import CryptoKit
#else
    import Crypto
#endif

/// Experimental routed decode primitive. No loader or model factory enables it.
/// Hidden rows must already be in this down projection's input basis, in F32.
/// Accumulates projection and routing in F32, casting only the final token rows.
final class JANGHWeightedDownKernel {
    let identity: String
    let inputRotation: JANGHFormatContract.Rotation
    private let bits: Int
    private let kernel: MLXFast.MLXFastKernel

    init(contract: JANGHFormatContract, module: String) throws {
        guard module.hasSuffix(".down_proj"), let projection = contract.projections[module],
            let book = contract.codebooks[projection.bits]
        else { throw JANGHFormatContract.ValidationError.invalid("missing JANGH down projection") }
        bits = projection.bits
        inputRotation = projection.rotation
        let description =
            "jangh-weighted-down-v2|\(bits)|\(inputRotation.rawValue)|"
            + "\(book.alpha.bitPattern)|\(book.beta.bitPattern)"
        identity = SHA256.hash(data: Data(description.utf8)).map { String(format: "%02x", $0) }
            .joined()
        let center = Float((1 << bits) - 1) / 2
        kernel = MLXFast.metalKernel(
            name: "jangh_weighted_down_" + identity,
            inputNames: ["hidden", "packed", "scales", "indices", "scores"], outputNames: ["out"],
            source: """
                uint lane = thread_index_in_simdgroup;
                uint sg = simdgroup_index_in_threadgroup;
                uint row0 = threadgroup_position_in_grid.y * 8u + sg * 4u;
                uint token = threadgroup_position_in_grid.z;
                float total[4] = {0, 0, 0, 0};
                for (uint route = 0; route < ROUTES; ++route) {
                    size_t dispatch = size_t(token) * ROUTES + route;
                    uint expert = indices[dispatch];
                    // Uniform across the threadgroup, before any packed/scale access.
                    // Even a zero-weight invalid route is invalid, never an OOB read.
                    if (expert >= EXPERTS) {
                        if (lane == 0) for (uint r = 0; r < 4; ++r)
                            if (row0 + r < N) out[size_t(token) * N + row0 + r]
                                = T(as_type<float>(0x7fc00000u));
                        return;
                    }
                    float accum[4] = {0, 0, 0, 0};
                    for (uint block = 0; block < H; block += 512u) {
                        for (uint i = 0; i < 16; ++i) {
                            uint column = block + lane * 16u + i;
                            if (column >= H) continue;
                            float value = hidden[dispatch * H + column];
                            uint bit = column * \(bits)u;
                            uint shift = bit & 31u;
                            uint word = bit >> 5u;
                            for (uint r = 0; r < 4; ++r) {
                                if (row0 + r >= N) continue;
                                size_t base = (size_t(expert) * N + row0 + r) * WORDS;
                                uint code = packed[base + word] >> shift;
                                if (shift + \(bits)u > 32u)
                                    code |= packed[base + word + 1] << (32u - shift);
                                code &= (1u << \(bits)u) - 1u;
                                float u = float(code) - \(center)f;
                                float level = u * fma(\(Float(book.beta))f, u * u, \(Float(book.alpha))f);
                                accum[r] = fma(value, level, accum[r]);
                            }
                        }
                    }
                    for (uint r = 0; r < 4; ++r) {
                        float dot = simd_sum(accum[r]);
                        if (row0 + r < N)
                            total[r] += scores[dispatch] * dot * float(scales[size_t(expert) * N + row0 + r]);
                    }
                }
                if (lane == 0) for (uint r = 0; r < 4; ++r)
                    if (row0 + r < N) out[size_t(token) * N + row0 + r] = T(total[r]);
                """, ensureRowContiguous: false)
    }

    /// `hidden`: [T*R,H] F32, `indices`/`scores`: [T,R] U32/F32.
    /// `preparedBasis` is an explicit caller assertion; this API never rotates.
    /// Packed weights: [E,N,H*bits/32] U32; per-row scales: [E,N] F16.
    /// Invalid expert IDs return NaN for that entire token, including zero-score IDs.
    func projectPreparedHidden(
        _ hidden: MLXArray, preparedBasis: JANGHFormatContract.Rotation,
        packed: MLXArray, scales: MLXArray, indices: MLXArray, scores: MLXArray,
        outputDType: DType
    ) throws -> MLXArray {
        guard preparedBasis == inputRotation,
            hidden.ndim == 2, hidden.dtype == .float32,
            hidden.dim(0) > 0, hidden.dim(1) > 0, hidden.dim(1).isMultiple(of: 32),
            indices.ndim == 2, indices.dtype == .uint32,
            indices.dim(0) > 0, indices.dim(1) > 0,
            scores.shape == indices.shape, scores.dtype == .float32,
            packed.ndim == 3, packed.dtype == .uint32,
            packed.dim(0) > 0, packed.dim(1) > 0, scales.dtype == .float16,
            [.float16, .bfloat16, .float32].contains(outputDType)
        else {
            throw JANGHFormatContract.ValidationError.invalid("invalid JANGH weighted-down tensors")
        }
        let h = hidden.dim(1)
        let n = packed.dim(1)
        let experts = packed.dim(0)
        let width = h.multipliedReportingOverflow(by: bits)
        let dispatches = indices.dim(0).multipliedReportingOverflow(by: indices.dim(1))
        guard !width.overflow, !dispatches.overflow,
            width.partialValue <= Int(UInt32.max), n <= Int(UInt32.max) - 7,
            experts <= Int(UInt32.max), indices.dim(0) <= Int(UInt32.max),
            indices.dim(1) < Int(UInt32.max),
            hidden.dim(0) == dispatches.partialValue,
            packed.shape == [experts, n, width.partialValue / 32], scales.shape == [experts, n]
        else {
            throw JANGHFormatContract.ValidationError.invalid(
                "invalid JANGH weighted-down geometry")
        }
        try JANGHBankLayout.requireReadyRowContiguous(packed, role: "down packed")
        try JANGHBankLayout.requireReadyRowContiguous(scales, role: "down scales")
        return kernel(
            [contiguous(hidden), packed, scales, contiguous(indices), contiguous(scores)],
            template: [
                ("T", outputDType), ("H", h), ("N", n), ("EXPERTS", experts),
                ("ROUTES", indices.dim(1)),
                ("WORDS", width.partialValue / 32),
            ],
            grid: (64, (n + 7) / 8, indices.dim(0)), threadGroup: (64, 1, 1),
            outputShapes: [[indices.dim(0), n]], outputDTypes: [outputDType])[0]
    }
}
