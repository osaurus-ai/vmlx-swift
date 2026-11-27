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
    private let kernelYZ: MLXFast.MLXFastKernel
    private let parallelKernel: MLXFast.MLXFastKernel
    private let parallelKernelYZ: MLXFast.MLXFastKernel

    init(contract: JANGHFormatContract, module: String) throws {
        // `.fc2` is the down projection of a gate-less relu2 stack (Nemotron-H).
        guard module.hasSuffix(".down_proj") || module.hasSuffix(".fc2"),
            let projection = contract.projections[module],
            let book = contract.codebooks[projection.bits]
        else { throw JANGHFormatContract.ValidationError.invalid("missing JANGH down projection") }
        bits = projection.bits
        inputRotation = projection.rotation
        let description =
            "jangh-weighted-down-v3-word|\(bits)|\(inputRotation.rawValue)|"
            + "\(book.alpha.bitPattern)|\(book.beta.bitPattern)"
        identity = SHA256.hash(data: Data(description.utf8)).map { String(format: "%02x", $0) }
            .joined()
        let dot = JANGHDecodeQDot.source(
            bits: bits, packed: "packed", rowBase: "base", columnBase: "block + lane * 16u",
            values: "values", accumulator: "accum[r]", alpha: Float(book.alpha), beta: Float(book.beta))
        let baseSource = """
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
                        if (block + lane * 16u >= H) continue;
                        float values[16];
                        _Pragma("clang loop unroll(full)")
                        for (uint i = 0; i < 16; ++i)
                            values[i] = hidden[dispatch * H + block + lane * 16u + i];
                        _Pragma("clang loop unroll(full)")
                        for (uint r = 0; r < 4; ++r) {
                            if (row0 + r >= N) continue;
                            size_t base = (size_t(expert) * N + row0 + r) * WORDS;
                            \(dot)
                        }
                    }

                    for (uint r = 0; r < 4; ++r) {
                        float dot = simd_sum(accum[r]);
                        if (row0 + r < N) {
                            float projected = dot * float(scales[size_t(expert) * N + row0 + r]);
                            if (ROUND_BF16) projected = float(bfloat16_t(projected));
                            total[r] += scores[dispatch] * projected;
                        }
                    }
                }
                if (lane == 0) for (uint r = 0; r < 4; ++r)
                    if (row0 + r < N) out[size_t(token) * N + row0 + r] = T(total[r]);
                """
        kernel = MLXFast.metalKernel(
            name: "jangh_weighted_down_" + identity,
            inputNames: ["hidden", "packed", "scales", "indices", "scores"], outputNames: ["out"],
            source: baseSource, ensureRowContiguous: false)
        kernelYZ = MLXFast.metalKernel(
            name: "jangh_weighted_down_yz_" + identity,
            inputNames: ["hidden", "packed", "scales", "indices", "scores"], outputNames: ["out"],
            source: JANGTKernelSource.swapGridYZ(baseSource), ensureRowContiguous: false)
        // Parallel-slot weighted down (T <= 8, ROUTES >= 8, N % 4 == 0):
        // PSG simdgroups per TG split ROUTES across 4 output rows and reduce via threadgroup memory.
        let parallelSource = """
                uint lane = thread_index_in_simdgroup;
                uint sg = simdgroup_index_in_threadgroup;
                uint row0 = threadgroup_position_in_grid.y * 4u;
                uint token = threadgroup_position_in_grid.z;
                // Uniform invalid-expert validation across all routes before any bank read or barrier.
                for (uint route = 0; route < ROUTES; ++route) {
                    if (indices[size_t(token) * ROUTES + route] >= EXPERTS) {
                        if (sg == 0u && lane == 0u) for (uint r = 0; r < 4; ++r)
                            if (row0 + r < N) out[size_t(token) * N + row0 + r]
                                = T(as_type<float>(0x7fc00000u));
                        return;
                    }
                }
                threadgroup float sg_buf[PSG][4];
                float total[4] = {0, 0, 0, 0};
                uint per_sg = ROUTES / PSG;
                uint k_start = sg * per_sg;
                for (uint j = 0; j < per_sg; ++j) {
                    uint route = k_start + j;
                    size_t dispatch = size_t(token) * ROUTES + route;
                    uint expert = indices[dispatch];
                    float accum[4] = {0, 0, 0, 0};
                    for (uint block = 0; block < H; block += 512u) {
                        if (block + lane * 16u >= H) continue;
                        float values[16];
                        _Pragma("clang loop unroll(full)")
                        for (uint i = 0; i < 16; ++i)
                            values[i] = hidden[dispatch * H + block + lane * 16u + i];
                        _Pragma("clang loop unroll(full)")
                        for (uint r = 0; r < 4; ++r) {
                            if (row0 + r >= N) continue;
                            size_t base = (size_t(expert) * N + row0 + r) * WORDS;
                            \(dot)
                        }
                    }
                    _Pragma("clang loop unroll(full)")
                    for (uint r = 0; r < 4; ++r) {
                        float d = simd_sum(accum[r]);
                        if (lane == 0u && row0 + r < N) {
                            float projected = d * float(scales[size_t(expert) * N + row0 + r]);
                            if (ROUND_BF16) projected = float(bfloat16_t(projected));
                            total[r] += scores[dispatch] * projected;
                        }
                    }
                }
                if (lane == 0u) {
                    _Pragma("clang loop unroll(full)")
                    for (uint r = 0; r < 4; ++r) sg_buf[sg][r] = total[r];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (sg == 0u && lane == 0u) {
                    _Pragma("clang loop unroll(full)")
                    for (uint r = 0; r < 4; ++r) {
                        if (row0 + r < N) {
                            float acc = sg_buf[0][r];
                            _Pragma("clang loop unroll(full)")
                            for (uint s = 1; s < PSG; ++s) acc += sg_buf[s][r];
                            out[size_t(token) * N + row0 + r] = T(acc);
                        }
                    }
                }
                """
        parallelKernel = MLXFast.metalKernel(
            name: "jangh_weighted_down_par_" + identity,
            inputNames: ["hidden", "packed", "scales", "indices", "scores"], outputNames: ["out"],
            source: parallelSource, ensureRowContiguous: false)
        parallelKernelYZ = MLXFast.metalKernel(
            name: "jangh_weighted_down_par_yz_" + identity,
            inputNames: ["hidden", "packed", "scales", "indices", "scores"], outputNames: ["out"],
            source: JANGTKernelSource.swapGridYZ(parallelSource), ensureRowContiguous: false)
    }

    /// `hidden`: [T*R,H] F32, `indices`/`scores`: [T,R] U32/F32.
    /// `preparedBasis` is an explicit caller assertion; this API never rotates.
    /// Packed weights: [E,N,H*bits/32] U32; per-row scales: [E,N] F16.
    /// Invalid expert IDs return NaN for that entire token, including zero-score IDs.
    func projectPreparedHidden(
        _ hidden: MLXArray, preparedBasis: JANGHFormatContract.Rotation,
        packed: MLXArray, scales: MLXArray, indices: MLXArray, scores: MLXArray,
        outputDType: DType, roundRouteToBFloat16: Bool = false, slotFastest: Bool = false
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
        let tokens = indices.dim(0)
        let routes = indices.dim(1)
        let yz = slotFastest && tokens > 1
        if tokens <= 8 && routes >= 8 && n.isMultiple(of: 4)
            && (routes.isMultiple(of: 5) || routes.isMultiple(of: 4))
        {
            let psg = routes.isMultiple(of: 5) ? 5 : 4
            let threads = psg * 32
            let kExec = yz ? parallelKernelYZ : parallelKernel
            let grid = yz ? (threads, tokens, n / 4) : (threads, n / 4, tokens)
            return kExec(
                [contiguous(hidden), packed, scales, contiguous(indices), contiguous(scores)],
                template: [
                    ("T", outputDType), ("H", h), ("N", n), ("EXPERTS", experts),
                    ("ROUTES", routes), ("PSG", psg),
                    ("WORDS", width.partialValue / 32),
                    ("ROUND_BF16", roundRouteToBFloat16),
                ],
                grid: grid, threadGroup: (threads, 1, 1),
                outputShapes: [[tokens, n]], outputDTypes: [outputDType])[0]
        }
        let kExec = yz ? kernelYZ : kernel
        let grid = yz ? (64, tokens, (n + 7) / 8) : (64, (n + 7) / 8, tokens)
        return kExec(
            [contiguous(hidden), packed, scales, contiguous(indices), contiguous(scores)],
            template: [
                ("T", outputDType), ("H", h), ("N", n), ("EXPERTS", experts),
                ("ROUTES", routes),
                ("WORDS", width.partialValue / 32),
                ("ROUND_BF16", roundRouteToBFloat16),
            ],
            grid: grid, threadGroup: (64, 1, 1),
            outputShapes: [[tokens, n]], outputDTypes: [outputDType])[0]
    }
}
