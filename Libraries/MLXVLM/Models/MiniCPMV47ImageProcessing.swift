// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// MiniCPM-V 4.7 image preprocessing, matching transformers' PIL-backend `MiniCPMV4_6ImageProcessorPil` exactly:
//   1. slice grid selection (`get_sliced_grid`), thumbnail size (`find_best_resize`), refine size
//      (`get_refine_size`) — Python `int()` truncation, `round()` half-to-even, `math.log` ratios;
//   2. PIL `Image.resize(BICUBIC)`: separable, antialiased for downscale, 22-bit fixed-point coefficients and
//      uint8 rounding between the horizontal and vertical pass (Pillow `Resample.c`);
//   3. `x.astype(float64) * (1/255)` narrowed to float32, then `(x - mean) / std` in float32;
//   4. NaViT patch packing: one `[patch, patch, C]` tile per 14x14 patch, crops in `[thumbnail, slices...]`
//      order, each crop's patches row-major.
// The numbers it produces are bit-identical to the reference for untagged 8-bit RGB input (verified in
// MiniCPMV47VisionTests against reference dumps); CoreImage colour management of ICC-tagged inputs is the one
// thing PIL (which ignores profiles) does differently.

import CoreImage
import Foundation
import MLX
import MLXLMCommon

enum MiniCPMV47ImageProcessing {

    // MARK: Python arithmetic

    /// Python `round()`: ties to even.
    static func pyRound(_ x: Double) -> Int { Int(x.rounded(.toNearestOrEven)) }

    /// `ensure_divide(length, divisor) = max(round(length / divisor) * divisor, divisor)`.
    static func ensureDivide(_ length: Double, _ divisor: Int) -> Int {
        max(pyRound(length / Double(divisor)) * divisor, divisor)
    }

    /// `find_best_resize`: returns `(height, width)`.
    static func findBestResize(
        height: Double, width: Double, scaleResolution: Int, patchSize: Int, allowUpscale: Bool
    ) -> (Int, Int) {
        var h = height
        var w = width
        if h * w > Double(scaleResolution * scaleResolution) || allowUpscale {
            let aspect = w / h
            h = Double(Int(Double(scaleResolution) / aspect.squareRoot()))  // int() truncates
            w = Double(Int(h * aspect))
        }
        let bestW = ensureDivide(w, patchSize * 4)
        let bestH = ensureDivide(h, patchSize * 4)
        return (bestH, bestW)
    }

    /// `get_refine_size`: returns `(height, width)` of the refined image whose `grid` tiles are each a best-resize.
    static func refineSize(
        height: Int, width: Int, grid: (rows: Int, cols: Int), scaleResolution: Int, patchSize: Int,
        allowUpscale: Bool
    ) -> (Int, Int) {
        let refineW = ensureDivide(Double(width), grid.cols)
        let refineH = ensureDivide(Double(height), grid.rows)
        let (bestH, bestW) = findBestResize(
            height: Double(refineH) / Double(grid.rows), width: Double(refineW) / Double(grid.cols),
            scaleResolution: scaleResolution, patchSize: patchSize, allowUpscale: allowUpscale)
        return (bestH * grid.rows, bestW * grid.cols)
    }

    /// `get_sliced_grid`: `(rows, cols)` or nil when the image is too small to slice.
    static func slicedGrid(height: Int, width: Int, maxSliceNums: Int, scaleResolution: Int) -> (rows: Int, cols: Int)? {
        let logRatio = log(Double(width) / Double(height))
        let ratio = Double(width * height) / Double(scaleResolution * scaleResolution)
        let multiple = min(Int(ceil(ratio)), maxSliceNums)
        if multiple <= 1 { return nil }
        var best = (rows: 1, cols: 1)
        var minError = Double.infinity
        for numSlices in [multiple - 1, multiple, multiple + 1] {
            if numSlices == 1 || numSlices > maxSliceNums { continue }
            for numRows in 1 ... numSlices where numSlices % numRows == 0 {
                let numCols = numSlices / numRows
                let error = abs(logRatio - log(Double(numCols) / Double(numRows)))
                if error < minError {
                    best = (numRows, numCols)
                    minError = error
                } else if error == minError && numRows > best.rows {
                    best = (numRows, numCols)
                }
            }
        }
        return best
    }

    // MARK: PIL bicubic resample (8-bit, fixed point)

    /// Interleaved 8-bit RGB.
    struct RGB8 {
        var width: Int
        var height: Int
        var data: [UInt8]  // row-major, 3 bytes per pixel

        subscript(y: Int, x: Int, c: Int) -> UInt8 {
            data[(y * width + x) * 3 + c]
        }

        func cropped(x: Int, y: Int, width w: Int, height h: Int) -> RGB8 {
            var out = [UInt8](repeating: 0, count: w * h * 3)
            for row in 0 ..< h {
                let src = ((y + row) * width + x) * 3
                let dst = row * w * 3
                out.replaceSubrange(dst ..< dst + w * 3, with: data[src ..< src + w * 3])
            }
            return RGB8(width: w, height: h, data: out)
        }
    }

    private static let precisionBits: Int32 = 32 - 8 - 2

    /// Keys cubic with a = -0.5 (PIL `bicubic_filter`), support 2.
    private static func bicubic(_ x: Double) -> Double {
        let a = -0.5
        var x = x
        if x < 0 { x = -x }
        if x < 1.0 { return ((a + 2.0) * x - (a + 3.0)) * x * x + 1 }
        if x < 2.0 { return (((x - 5) * x + 8) * x - 4) * a }
        return 0.0
    }

    /// PIL `precompute_coeffs` + `normalize_coeffs_8bpc` for a full-extent box.
    private static func coefficients(inSize: Int, outSize: Int) -> (ksize: Int, bounds: [(Int, Int)], kk: [Int32]) {
        let scale = Double(inSize) / Double(outSize)
        let filterScale = max(scale, 1.0)
        let support = 2.0 * filterScale
        let ksize = Int(ceil(support)) * 2 + 1
        var bounds: [(Int, Int)] = []
        bounds.reserveCapacity(outSize)
        var kk = [Int32](repeating: 0, count: outSize * ksize)
        var prekk = [Double](repeating: 0, count: ksize)
        for xx in 0 ..< outSize {
            let center = (Double(xx) + 0.5) * scale
            let ss = 1.0 / filterScale
            var xmin = Int(center - support + 0.5)
            if xmin < 0 { xmin = 0 }
            var xmax = Int(center + support + 0.5)
            if xmax > inSize { xmax = inSize }
            xmax -= xmin
            var ww = 0.0
            for x in 0 ..< xmax {
                let w = bicubic((Double(x + xmin) - center + 0.5) * ss)
                prekk[x] = w
                ww += w
            }
            for x in 0 ..< xmax where ww != 0.0 { prekk[x] /= ww }
            for x in xmax ..< ksize { prekk[x] = 0 }
            for x in 0 ..< ksize {
                let scaled = prekk[x] * Double(1 << precisionBits)
                kk[xx * ksize + x] = Int32(prekk[x] < 0 ? -0.5 + scaled : 0.5 + scaled)
            }
            bounds.append((xmin, xmax))
        }
        return (ksize, bounds, kk)
    }

    @inline(__always) private static func clip8(_ v: Int32) -> UInt8 {
        let shifted = v >> precisionBits
        return UInt8(clamping: shifted)
    }

    /// `Image.resize((width, height), BICUBIC)` on an 8-bit RGB image: horizontal pass, then vertical.
    static func resizeBicubicPIL(_ image: RGB8, toWidth outW: Int, height outH: Int) -> RGB8 {
        var current = image
        if outW != image.width {
            let (ksize, bounds, kk) = coefficients(inSize: image.width, outSize: outW)
            var out = [UInt8](repeating: 0, count: outW * image.height * 3)
            let half = Int32(1) << (precisionBits - 1)
            for y in 0 ..< image.height {
                let rowBase = y * image.width * 3
                for xx in 0 ..< outW {
                    let (xmin, xmax) = bounds[xx]
                    var s0 = half, s1 = half, s2 = half
                    for x in 0 ..< xmax {
                        let k = kk[xx * ksize + x]
                        let p = rowBase + (x + xmin) * 3
                        s0 &+= Int32(current.data[p]) &* k
                        s1 &+= Int32(current.data[p + 1]) &* k
                        s2 &+= Int32(current.data[p + 2]) &* k
                    }
                    let o = (y * outW + xx) * 3
                    out[o] = clip8(s0)
                    out[o + 1] = clip8(s1)
                    out[o + 2] = clip8(s2)
                }
            }
            current = RGB8(width: outW, height: image.height, data: out)
        }
        if outH != current.height {
            let (ksize, bounds, kk) = coefficients(inSize: current.height, outSize: outH)
            var out = [UInt8](repeating: 0, count: current.width * outH * 3)
            let half = Int32(1) << (precisionBits - 1)
            let w = current.width
            for yy in 0 ..< outH {
                let (ymin, ymax) = bounds[yy]
                for x in 0 ..< w {
                    var s0 = half, s1 = half, s2 = half
                    for y in 0 ..< ymax {
                        let k = kk[yy * ksize + y]
                        let p = ((y + ymin) * w + x) * 3
                        s0 &+= Int32(current.data[p]) &* k
                        s1 &+= Int32(current.data[p + 1]) &* k
                        s2 &+= Int32(current.data[p + 2]) &* k
                    }
                    let o = (yy * w + x) * 3
                    out[o] = clip8(s0)
                    out[o + 1] = clip8(s1)
                    out[o + 2] = clip8(s2)
                }
            }
            current = RGB8(width: w, height: outH, data: out)
        }
        return current
    }

    // MARK: CoreImage -> 8-bit RGB

    /// Render a `CIImage` to 8-bit RGB without colour conversion when the source declares a colour space
    /// (PIL reads stored bytes as-is), assuming sRGB for untagged sources.
    static func rgb8(from image: CIImage) throws -> RGB8 {
        let (height, width) = try QwenVL.intExtent(image.extent.size)
        let colorSpace = image.colorSpace ?? CGColorSpace(name: CGColorSpace.sRGB)!
        let context = CIContext(options: [.workingColorSpace: NSNull(), .outputColorSpace: NSNull()])
        var rgba = [UInt8](repeating: 0, count: width * height * 4)
        rgba.withUnsafeMutableBytes { ptr in
            context.render(
                image, toBitmap: ptr.baseAddress!, rowBytes: width * 4,
                bounds: CGRect(x: image.extent.origin.x, y: image.extent.origin.y, width: CGFloat(width), height: CGFloat(height)),
                format: .RGBA8, colorSpace: colorSpace)
        }
        var rgb = [UInt8](repeating: 0, count: width * height * 3)
        for p in 0 ..< width * height {
            rgb[p * 3] = rgba[p * 4]
            rgb[p * 3 + 1] = rgba[p * 4 + 1]
            rgb[p * 3 + 2] = rgba[p * 4 + 2]
        }
        return RGB8(width: width, height: height, data: rgb)
    }

    // MARK: Normalisation + patch packing

    /// How 8-bit values become normalised floats.
    enum Normalization {
        /// PIL image processor: `x.astype(float64) * (1/255)` narrowed to float32, then `(v - mean) / std`.
        case pil
        /// torchvision video processor (`rescale_and_normalize`, fused): `(float32(x) - mean*255) / (std*255)`.
        case fusedTorch
    }

    /// `[patchesH * patchesW, patch, patch, 3]` float32, patches row-major, each tile `[row, col, channel]`.
    static func normalizedPatches(
        _ image: RGB8, patchSize: Int, mean: [Float], std: [Float], normalization: Normalization = .pil
    ) -> MLXArray {
        precondition(image.height % patchSize == 0 && image.width % patchSize == 0)
        let gridH = image.height / patchSize
        let gridW = image.width / patchSize
        let rescale = 1.0 / 255.0
        let fusedMean = mean.map { $0 * Float(255.0) }
        let fusedStd = std.map { $0 * Float(255.0) }
        var out = [Float](repeating: 0, count: image.width * image.height * 3)
        var o = 0
        for py in 0 ..< gridH {
            for px in 0 ..< gridW {
                for i in 0 ..< patchSize {
                    let y = py * patchSize + i
                    for j in 0 ..< patchSize {
                        let x = px * patchSize + j
                        let p = (y * image.width + x) * 3
                        for c in 0 ..< 3 {
                            switch normalization {
                            case .pil:
                                let v = Float(Double(image.data[p + c]) * rescale)
                                out[o] = (v - mean[c]) / std[c]
                            case .fusedTorch:
                                out[o] = (Float(image.data[p + c]) - fusedMean[c]) / fusedStd[c]
                            }
                            o += 1
                        }
                    }
                }
            }
        }
        return MLXArray(out, [gridH * gridW, patchSize, patchSize, 3])
    }

    // MARK: Full per-image pipeline

    struct ProcessedCrops {
        /// `[N, patch, patch, 3]` for all crops, thumbnail first.
        let patches: MLXArray
        /// Patch grid `(1, h, w)` per crop.
        let grids: [THW]
        /// Slice grid `(rows, cols)`, or nil when the image was not sliced.
        let sliceGrid: (rows: Int, cols: Int)?
    }

    /// Image path by default (PIL bicubic, PIL normalisation); the video path passes torchvision's resize and
    /// the fused normalisation, the slicing math being shared verbatim between the two reference processors.
    static func process(
        _ source: RGB8, maxSliceNums: Int, scaleResolution: Int, patchSize: Int, sliceMode: Bool,
        mean: [Float], std: [Float],
        resize: (RGB8, Int, Int) -> RGB8 = resizeBicubicPIL, normalization: Normalization = .pil
    ) -> ProcessedCrops {
        let (h, w) = (source.height, source.width)
        let grid = sliceMode
            ? slicedGrid(height: h, width: w, maxSliceNums: maxSliceNums, scaleResolution: scaleResolution)
            : nil
        let (sourceH, sourceW) = findBestResize(
            height: Double(h), width: Double(w), scaleResolution: scaleResolution, patchSize: patchSize,
            allowUpscale: grid == nil)
        let thumbnail = resize(source, sourceW, sourceH)
        var parts = [normalizedPatches(thumbnail, patchSize: patchSize, mean: mean, std: std, normalization: normalization)]
        var grids = [THW(1, sourceH / patchSize, sourceW / patchSize)]
        if let grid {
            let (refineH, refineW) = refineSize(
                height: h, width: w, grid: grid, scaleResolution: scaleResolution, patchSize: patchSize,
                allowUpscale: true)
            let refined = resize(source, refineW, refineH)
            let tileH = refineH / grid.rows
            let tileW = refineW / grid.cols
            for row in stride(from: 0, to: refineH, by: tileH) {
                for col in stride(from: 0, to: refineW, by: tileW) {
                    let tile = refined.cropped(x: col, y: row, width: tileW, height: tileH)
                    parts.append(normalizedPatches(tile, patchSize: patchSize, mean: mean, std: std, normalization: normalization))
                    grids.append(THW(1, tileH / patchSize, tileW / patchSize))
                }
            }
        }
        return ProcessedCrops(patches: concatenated(parts), grids: grids, sliceGrid: grid)
    }

    // MARK: Placeholder expansion

    /// Token ids replacing one `<|image_pad|>` placeholder, as `replace_image_token` + `_prepend_local_ids`:
    /// `<image_id>N</image_id><image>pad*thumb</image>` then rows of `<slice>pad*slice</slice>` joined by `\n`.
    static func placeholderTokens(
        crops: ProcessedCrops, localIndex: Int, useImageId: Bool, padTokenId: Int,
        tokensPerPatchDivisor: Int, markers: (imageStart: Int, imageEnd: Int, sliceStart: Int, sliceEnd: Int, newline: Int),
        imageIdStart: Int?, imageIdEnd: Int?, digitTokens: (Int) -> [Int]
    ) -> [Int] {
        var out: [Int] = []
        if useImageId, let imageIdStart, let imageIdEnd {
            out.append(imageIdStart)
            out.append(contentsOf: digitTokens(localIndex))
            out.append(imageIdEnd)
        }
        let thumbCount = crops.grids[0].product / tokensPerPatchDivisor
        out.append(markers.imageStart)
        out.append(contentsOf: repeatElement(padTokenId, count: thumbCount))
        out.append(markers.imageEnd)
        if let grid = crops.sliceGrid, grid.rows > 0, grid.cols > 0 {
            let perSlice = crops.grids.count > 1 ? crops.grids[1].product / tokensPerPatchDivisor : 0
            for row in 0 ..< grid.rows {
                if row > 0 { out.append(markers.newline) }
                for _ in 0 ..< grid.cols {
                    out.append(markers.sliceStart)
                    out.append(contentsOf: repeatElement(padTokenId, count: perSlice))
                    out.append(markers.sliceEnd)
                }
            }
        }
        return out
    }

    /// Replace the k-th single placeholder token with `replacements[k]`, carrying template cache boundaries
    /// through (a boundary inside an expanded span is discarded, like `QwenVL.expandPaddingTokens`).
    static func expand(
        promptTokens: [Int], placeholder: Int, replacements: [[Int]], boundaries: CanonicalChatCacheBoundaries,
        modality: String = "image"
    ) throws -> (tokens: [Int], boundaries: CanonicalChatCacheBoundaries) {
        let positions = promptTokens.indices.filter { promptTokens[$0] == placeholder }
        guard positions.count == replacements.count else {
            throw VLMError.processing(
                "MiniCPM-V 4.7: \(positions.count) \(modality) placeholder(s) in the prompt but \(replacements.count) \(modality)(s)")
        }
        var result: [Int] = []
        result.reserveCapacity(promptTokens.count + replacements.reduce(0) { $0 + $1.count })
        var cursor = 0
        for (position, replacement) in zip(positions, replacements) {
            result.append(contentsOf: promptTokens[cursor ..< position])
            result.append(contentsOf: replacement)
            cursor = position + 1
        }
        result.append(contentsOf: promptTokens[cursor...])
        func remap(_ boundary: Int) -> Int? {
            guard boundary > 0, boundary < promptTokens.count else { return nil }
            var mapped = boundary
            for (position, replacement) in zip(positions, replacements) where position < boundary {
                mapped += replacement.count - 1
            }
            return mapped
        }
        return (
            result,
            CanonicalChatCacheBoundaries(
                all: boundaries.all.compactMap(remap), stable: boundaries.stable.compactMap(remap))
        )
    }
}
