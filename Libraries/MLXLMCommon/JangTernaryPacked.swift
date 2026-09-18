// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX

/// Expand PTQ1_0-density trits to native affine 2-bit words. On Metal, decode
/// and validate directly into words without materializing a uint32 per trit.
/// Row chunks bound temporary work (the real lm_head has 1.27B codes);
/// validation per chunk releases its graph before the next.
/// `maximumChunkCodes` targets a code count, with at least one complete row.
/// It is not a process-footprint bound or an allocator-policy override.
/// Scales remain the original float16 array, never rounded/re-quantized.
func expandJangTernaryPacked(
    _ packed: MLXArray, scales: MLXArray, maximumChunkCodes: Int = 1_048_576
) throws -> (weight: MLXArray, scales: MLXArray, biases: MLXArray) {
    guard packed.dtype == .uint8, packed.ndim == 2,
        packed.dim(0) > 0, packed.dim(1) > 0, packed.dim(1) % 26 == 0,
        scales.dtype == .float16, scales.shape == [packed.dim(0), packed.dim(1) / 26],
        maximumChunkCodes > 0
    else {
        throw JangLoaderError.loadFailed(
            "ternary_packed_26b expects uint8 [rows,groups*26] and unchanged float16 [rows,groups] scales"
        )
    }
    let rows = packed.dim(0)
    let groups = packed.dim(1) / 26
    let chunkRows = max(1, maximumChunkCodes / (groups * 128))
    var chunks: [MLXArray] = []
    for start in stride(from: 0, to: rows, by: chunkRows) {
        let end = min(start + chunkRows, rows)
        let bytes = packed[start ..< end, 0...]
        let words: MLXArray
        #if canImport(Metal)
            if Device.defaultDevice() == .gpu {
                let shape = [end - start, groups * 8]
                let output = JangTernaryPackedMetal.expand(
                    [bytes], grid: ((end - start) * groups * 8, 1, 1),
                    threadGroup: (256, 1, 1), outputShapes: [shape, shape],
                    outputDTypes: [.uint32, .uint8])
                // One readback both validates every byte and completes the
                // shared kernel's word output. Invalid input is never returned.
                guard MLX.all(output[1] .== UInt8(0)).item(Bool.self) else {
                    throw JangLoaderError.loadFailed(
                        "ternary_packed_26b contains noncanonical base-3 bytes")
                }
                words = output[0]
            } else {
                words = try expandJangTernaryPackedChunk(bytes, groups: groups)
            }
        #else
            words = try expandJangTernaryPackedChunk(bytes, groups: groups)
        #endif
        chunks.append(words)
    }
    let words = chunks.count == 1 ? chunks[0] : concatenated(chunks, axis: 0)
    let biases = -scales
    MLX.eval(words, biases)
    return (words, scales, biases)
}

/// Portable fallback retains the original integer expansion and validation.
private func expandJangTernaryPackedChunk(_ packed: MLXArray, groups: Int) throws -> MLXArray {
    let rows = packed.dim(0)
    let bytes = packed.asType(.uint32).reshaped(rows, groups, 26)
    let head = bytes[0..., 0..., 0 ..< 25]
    let tail = bytes[0..., 0..., 25 ..< 26]
    guard MLX.all(head .<= 242).item(Bool.self), MLX.all(tail .<= 26).item(Bool.self) else {
        throw JangLoaderError.loadFailed(
            "ternary_packed_26b contains noncanonical base-3 bytes")
    }
    let headCodes = stacked(
        [1, 3, 9, 27, 81].map { remainder(floorDivide(head, $0), 3) }, axis: -1
    ).reshaped(rows, groups, 125)
    let tailCodes = stacked(
        [1, 3, 9].map { remainder(floorDivide(tail, $0), 3) }, axis: -1
    ).reshaped(rows, groups, 3)
    let codes = concatenated([headCodes, tailCodes], axis: -1).reshaped(rows, groups * 8, 16)
    let shifts = MLXArray((0 ..< 16).map { UInt32(2 * $0) })
    let words = (codes << shifts).sum(axis: -1).asType(.uint32)
    MLX.eval(words)
    return words
}

#if canImport(Metal)
    private enum JangTernaryPackedMetal {
        // Shape metadata, not template dimensions: all 402 module shapes
        // reuse one compiled kernel. Row-contiguous normalization also covers
        // sliced/strided input arrays without interpreting padding as bytes.
        static let expand = MLXFast.metalKernel(
            name: "jang_ternary_packed_26b_expand",
            inputNames: ["packed"], outputNames: ["words", "invalid"],
            source: """
                uint element = thread_position_in_grid.x;
                uint count = uint(packed_shape[0]) * uint(packed_shape[1] / 26) * 8u;
                if (element >= count) return;
                uint group = element / 8u;
                uint first = (element % 8u) * 16u;
                constexpr uint powers[5] = {1u, 3u, 9u, 27u, 81u};
                uint word = 0u;
                uint bad = 0u;
                for (uint lane = 0u; lane < 16u; ++lane) {
                    uint code = first + lane;
                    uint byte_index = code / 5u;
                    uint value = uint(packed[group * 26u + byte_index]);
                    bad |= uint(value > (byte_index == 25u ? 26u : 242u));
                    uint trit = (value / powers[code % 5u]) % 3u;
                    word |= trit << (2u * lane);
                }
                words[element] = word;
                invalid[element] = uchar(bad);
                """)
    }
#endif

extension JangTernaryPackedRuntimeContract {
    func expand(weights: inout [String: MLXArray]) throws {
        var expanded = 0
        for path in modulePaths.sorted() {
            guard let packed = weights["\(path).weight"],
                let scales = weights["\(path).scales"], weights["\(path).biases"] == nil
            else {
                throw JangLoaderError.loadFailed(
                    "packed ternary module missing weight/scales or already stores biases: \(path)")
            }
            let result = try expandJangTernaryPacked(packed, scales: scales)
            weights["\(path).weight"] = result.weight
            weights["\(path).biases"] = result.biases
            expanded += 1
        }
        guard expanded == modulePaths.count else {
            throw JangLoaderError.loadFailed(
                "packed ternary expansion count does not match manifest")
        }
        FileHandle.standardError.write(
            Data(
                "[Load] JANG ternary_packed_26b expanded=\(expanded) runtime_bits=2 scales_unchanged=true\n"
                    .utf8))
    }
}
