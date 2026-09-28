#if canImport(CryptoKit)
    import CryptoKit
#else
    import Crypto
#endif
import Foundation
import MLX
import MLXFast

/// Executable JANGH building blocks. Not registered as a model loader.
/// QMV returns F32, matching the reference's projection accumulation boundary.
final class JANGHProjectionKernel {
    let identity: String
    private let bits: Int
    private let rotation: JANGHFormatContract.Rotation
    private let qmv: MLXFast.MLXFastKernel
    private let h32: MLXFast.MLXFastKernel

    init(contract: JANGHFormatContract, module: String) throws {
        guard let projection = contract.projections[module],
            let book = contract.codebooks[projection.bits]
        else { throw JANGHFormatContract.ValidationError.invalid("missing JANGH projection") }
        bits = projection.bits
        rotation = projection.rotation
        let description =
            "jangh-qmv-v1|\(bits)|\(rotation.rawValue)|\(book.alpha.bitPattern)|\(book.beta.bitPattern)"
        identity = SHA256.hash(data: Data(description.utf8)).map { String(format: "%02x", $0) }
            .joined()
        let alpha = String(Float(book.alpha)) + "f"
        let beta = String(Float(book.beta)) + "f"
        let center = String(Float((1 << bits) - 1) / 2) + "f"
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
            inputNames: ["x", "packed", "scales", "indices"], outputNames: ["out"], source: source)
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
        let x = rotation == .hadamard32 ? try hadamard32(input) : input
        return qmv(
            [
                contiguous(x), contiguous(packed), contiguous(scales),
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
