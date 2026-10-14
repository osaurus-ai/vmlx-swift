// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// MiniCPM-V 4.7 video preprocessing, matching transformers' `MiniCPMV4_6VideoProcessor` (torchvision backend)
// as `MiniCPMV4_7Processor` drives it with the bundle defaults (`max_num_frames` 128, `stack_frames` 1,
// `slice_mode` true, `scale_resolution` 448, `patch_size` 14, `max_slice_nums` 9, mean/std 0.5):
//   1. frame sampling (`sample_frames`): one frame per started second, index `int(i * fps)`; clips longer than
//      `max_num_frames` seconds are sampled at 0.1 s and thinned with `np.linspace(..., dtype=int)`;
//   2. every sampled frame is one visual unit, sliced like an image (same `get_sliced_grid` / `find_best_resize` /
//      `get_refine_size`), but resized with torchvision's uint8 antialiased bicubic (ATen's int16 fixed-point
//      separable kernel, uint8 between the horizontal and vertical pass) rather than PIL's 22-bit one, and
//      normalised with the fused `(x - 0.5*255) / (0.5*255)` in float32;
//   3. placeholders per frame: `<image>pad*thumb</image>` then rows of `<slice>pad*k</slice>` joined by `\n`,
//      frames concatenated with nothing in between and no `<image_id>` (`replace_video_token`).
// The reference's `stack_frames > 1` composites (`concat_frames_as_image`) are built but never consumed by its
// own pixel stage (`_preprocess` resizes `videos`, not `visual_units`), so that knob is not ported: the default
// `stack_frames = 1` is the only path with a self-consistent reference, and no bundle sets anything else.
//
// Frames are decoded with AVFoundation (`AVAssetReader`, BGRA output, presentation order); the reference reads
// its frames with OpenCV/PyAV, whose YUV->RGB conversion differs from VideoToolbox's by up to ~150/255 on chroma
// edges. Parity is therefore established on identical decoded frame arrays (see MiniCPMV47VisionTests), and the
// sampled indices depend only on frame count and nominal frame rate, which both decoders agree on.

import AVFoundation
import CoreImage
import Foundation
import MLX
import MLXLMCommon

enum MiniCPMV47VideoProcessing {

    // MARK: Frame sampling

    /// Port of `MiniCPMV4_6VideoProcessor.sample_frames` with `stack_frames == 1`.
    /// `duration` is `totalFrames / fps`, as every transformers video reader computes it.
    static func sampleFrameIndices(totalFrames: Int, fps: Double, maxNumFrames: Int) -> [Int] {
        guard totalFrames > 0, fps > 0 else { return [] }
        let duration = Double(totalFrames) / fps
        let numSeconds = Int(ceil(duration))
        if duration > Double(maxNumFrames) {
            // timestamps = [round(i * 0.1, 1) for i in range(int(duration / 0.1))]
            let count = Int(duration / 0.1)
            var main = (0 ..< count).map { i -> Int in
                let ts = (Double(i) * 0.1 * 10).rounded(.toNearestOrEven) / 10
                return min(Int(ts * fps), totalFrames - 1)
            }
            if main.count > maxNumFrames {
                main = linspaceInt(0, main.count - 1, count: maxNumFrames).map { main[$0] }
            }
            return main
        }
        return (0 ..< numSeconds).map { Int(Double($0) * fps) }
    }

    /// `np.linspace(start, stop, num, dtype=int)`: float64 samples truncated toward zero, last pinned to `stop`.
    static func linspaceInt(_ start: Int, _ stop: Int, count: Int) -> [Int] {
        guard count > 1 else { return count == 1 ? [start] : [] }
        let step = Double(stop - start) / Double(count - 1)
        var out = (0 ..< count).map { Int(Double(start) + Double($0) * step) }
        out[count - 1] = stop
        return out
    }

    // MARK: Decoding

    /// A decoded clip: only the sampled frames are kept, as 8-bit RGB.
    struct SampledVideo {
        let frames: [MiniCPMV47ImageProcessing.RGB8]
        let indices: [Int]
        let totalFrames: Int
        let fps: Double
    }

    /// Frame count and nominal frame rate of the video track. The count comes from the sample table (one
    /// pass over the compressed samples, no decoding), which is what OpenCV's `CAP_PROP_FRAME_COUNT` reports.
    static func frameCount(of asset: AVAsset) async throws -> (total: Int, fps: Double) {
        let tracks = try await asset.loadTracks(withMediaType: .video)
        guard let track = tracks.first else { throw VLMError.noVideoTrackFound }
        guard try await track.load(.isDecodable) else { throw VLMError.videoNotDecodable }
        let fps = Double(try await track.load(.nominalFrameRate))
        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: nil)
        output.alwaysCopiesSampleData = false
        reader.add(output)
        guard reader.startReading() else {
            throw VLMError.processing("MiniCPM-V 4.7 video: cannot read the video track (\(reader.error.map { "\($0)" } ?? "no error"))")
        }
        var total = 0
        while let sample = output.copyNextSampleBuffer() {
            total += CMSampleBufferGetNumSamples(sample)
        }
        if reader.status == .failed {
            throw VLMError.processing("MiniCPM-V 4.7 video: reading the sample table failed (\(reader.error.map { "\($0)" } ?? "no error"))")
        }
        return (total, fps)
    }

    /// Decode the asset in presentation order and keep the frames at `indices` (ascending or not) as RGB.
    static func decodeFrames(of asset: AVAsset, indices: [Int]) async throws -> [MiniCPMV47ImageProcessing.RGB8] {
        let tracks = try await asset.loadTracks(withMediaType: .video)
        guard let track = tracks.first else { throw VLMError.noVideoTrackFound }
        let wanted = Set(indices)
        guard let last = wanted.max() else { return [] }
        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderTrackOutput(
            track: track,
            outputSettings: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA])
        output.alwaysCopiesSampleData = false
        reader.add(output)
        guard reader.startReading() else {
            throw VLMError.processing("MiniCPM-V 4.7 video: cannot decode the video track (\(reader.error.map { "\($0)" } ?? "no error"))")
        }
        var decoded: [Int: MiniCPMV47ImageProcessing.RGB8] = [:]
        var index = 0
        while index <= last, let sample = output.copyNextSampleBuffer() {
            defer { index += 1 }
            guard wanted.contains(index) else { continue }
            guard let pixelBuffer = CMSampleBufferGetImageBuffer(sample) else {
                throw VLMError.processing("MiniCPM-V 4.7 video: frame \(index) has no pixel buffer")
            }
            decoded[index] = rgb8(from: pixelBuffer)
        }
        if reader.status == .failed {
            throw VLMError.processing("MiniCPM-V 4.7 video: decoding failed at frame \(index) (\(reader.error.map { "\($0)" } ?? "no error"))")
        }
        reader.cancelReading()
        return try indices.map { i in
            guard let frame = decoded[i] else {
                throw VLMError.processing("MiniCPM-V 4.7 video: frame \(i) was not decoded (track ended after \(index) frames)")
            }
            return frame
        }
    }

    /// BGRA pixel buffer -> interleaved RGB bytes.
    static func rgb8(from pixelBuffer: CVPixelBuffer) -> MiniCPMV47ImageProcessing.RGB8 {
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        let stride = CVPixelBufferGetBytesPerRow(pixelBuffer)
        let base = CVPixelBufferGetBaseAddress(pixelBuffer)!.assumingMemoryBound(to: UInt8.self)
        var rgb = [UInt8](repeating: 0, count: width * height * 3)
        for y in 0 ..< height {
            let row = y * stride
            var o = y * width * 3
            for x in 0 ..< width {
                let p = row + x * 4
                rgb[o] = base[p + 2]
                rgb[o + 1] = base[p + 1]
                rgb[o + 2] = base[p]
                o += 3
            }
        }
        return .init(width: width, height: height, data: rgb)
    }

    /// Sample and decode one `UserInput.Video`.
    static func sample(_ video: UserInput.Video, maxNumFrames: Int) async throws -> SampledVideo {
        switch video {
        case .avAsset(let asset):
            return try await sample(asset: asset, maxNumFrames: maxNumFrames)
        case .url(let url):
            return try await sample(asset: AVURLAsset(url: url), maxNumFrames: maxNumFrames)
        case .frames(let frames):
            // In-memory frames carry their own timestamps: the frame count is the array length and the frame
            // rate is the mean inter-frame interval, so `duration = total / fps` spans the clip like a file's.
            guard !frames.isEmpty else { throw VLMError.processing("MiniCPM-V 4.7 video: no frames") }
            let span = frames.last!.timeStamp.seconds - frames.first!.timeStamp.seconds
            let fps = frames.count > 1 && span > 0 ? Double(frames.count - 1) / span : 1.0
            let indices = sampleFrameIndices(totalFrames: frames.count, fps: fps, maxNumFrames: maxNumFrames)
            let rgb = try indices.map { try MiniCPMV47ImageProcessing.rgb8(from: frames[$0].frame) }
            return SampledVideo(frames: rgb, indices: indices, totalFrames: frames.count, fps: fps)
        }
    }

    static func sample(asset: AVAsset, maxNumFrames: Int) async throws -> SampledVideo {
        let (total, fps) = try await frameCount(of: asset)
        let indices = sampleFrameIndices(totalFrames: total, fps: fps, maxNumFrames: maxNumFrames)
        let frames = try await decodeFrames(of: asset, indices: indices)
        return SampledVideo(frames: frames, indices: indices, totalFrames: total, fps: fps)
    }

    // MARK: torchvision uint8 antialiased bicubic (ATen separable kernel)

    private static func bicubic(_ x: Double) -> Double {
        let a = -0.5
        let x = abs(x)
        if x < 1.0 { return ((a + 2.0) * x - (a + 3.0)) * x * x + 1 }
        if x < 2.0 { return (((x - 5) * x + 8) * x - 4) * a }
        return 0.0
    }

    /// `HelperInterpBase::_compute_index_ranges_int16_weights`: PIL-style double weights, then int16 quantisation
    /// at the largest precision `p < 22` with `round(w_max * 2^(p+1)) < 2^15`.
    static func int16Coefficients(inSize: Int, outSize: Int) -> (bounds: [(Int, Int)], weights: [Int32], stride: Int, precision: Int) {
        let scale = Double(inSize) / Double(outSize)
        let support = scale >= 1.0 ? 2.0 * scale : 2.0
        let invscale = scale >= 1.0 ? 1.0 / scale : 1.0
        let maxInterp = Int(ceil(support)) * 2 + 1
        var bounds: [(Int, Int)] = []
        bounds.reserveCapacity(outSize)
        var weights = [Double](repeating: 0, count: outSize * maxInterp)
        var wMax = 0.0
        for i in 0 ..< outSize {
            let center = scale * (Double(i) + 0.5)
            let xmin = max(Int(center - support + 0.5), 0)
            var xsize = min(Int(center + support + 0.5), inSize) - xmin
            xsize = min(xsize, maxInterp)
            var total = 0.0
            for j in 0 ..< xsize {
                let w = bicubic((Double(j + xmin) - center + 0.5) * invscale)
                weights[i * maxInterp + j] = w
                total += w
            }
            if total > 0 {
                for j in 0 ..< xsize {
                    weights[i * maxInterp + j] /= total
                    wMax = max(wMax, weights[i * maxInterp + j])
                }
            }
            bounds.append((xmin, xsize))
        }
        var precision = 0
        while precision < 22 {
            let next = Int(0.5 + wMax * Double(1 << (precision + 1)))
            if next >= (1 << 15) { break }
            precision += 1
        }
        let factor = Double(1 << precision)
        let quantized = weights.map { w -> Int32 in
            let v = w * factor
            return Int32(Int16(truncatingIfNeeded: Int(v < 0 ? -0.5 + v : 0.5 + v)))
        }
        return (bounds, quantized, maxInterp, precision)
    }

    /// `tvF.resize(uint8 CHW, [outH, outW], BICUBIC, antialias=True)` on CPU: horizontal pass, then vertical,
    /// each `clamp((sum(src * w_int16) + 2^(p-1)) >> p, 0, 255)` with uint8 rounding in between.
    static func resizeBicubicTorchAA(
        _ image: MiniCPMV47ImageProcessing.RGB8, toWidth outW: Int, height outH: Int
    ) -> MiniCPMV47ImageProcessing.RGB8 {
        var current = image
        if outW != image.width {
            let (bounds, kk, ksize, p) = int16Coefficients(inSize: image.width, outSize: outW)
            let half = Int32(1) << Int32(p - 1)
            var out = [UInt8](repeating: 0, count: outW * image.height * 3)
            for y in 0 ..< image.height {
                let rowBase = y * image.width * 3
                for xx in 0 ..< outW {
                    let (xmin, xsize) = bounds[xx]
                    var s0 = half, s1 = half, s2 = half
                    for j in 0 ..< xsize {
                        let k = kk[xx * ksize + j]
                        let src = rowBase + (xmin + j) * 3
                        s0 &+= Int32(current.data[src]) &* k
                        s1 &+= Int32(current.data[src + 1]) &* k
                        s2 &+= Int32(current.data[src + 2]) &* k
                    }
                    let o = (y * outW + xx) * 3
                    out[o] = UInt8(clamping: s0 >> Int32(p))
                    out[o + 1] = UInt8(clamping: s1 >> Int32(p))
                    out[o + 2] = UInt8(clamping: s2 >> Int32(p))
                }
            }
            current = .init(width: outW, height: image.height, data: out)
        }
        if outH != current.height {
            let (bounds, kk, ksize, p) = int16Coefficients(inSize: current.height, outSize: outH)
            let half = Int32(1) << Int32(p - 1)
            let w = current.width
            var out = [UInt8](repeating: 0, count: w * outH * 3)
            for yy in 0 ..< outH {
                let (ymin, ysize) = bounds[yy]
                for x in 0 ..< w {
                    var s0 = half, s1 = half, s2 = half
                    for j in 0 ..< ysize {
                        let k = kk[yy * ksize + j]
                        let src = ((ymin + j) * w + x) * 3
                        s0 &+= Int32(current.data[src]) &* k
                        s1 &+= Int32(current.data[src + 1]) &* k
                        s2 &+= Int32(current.data[src + 2]) &* k
                    }
                    let o = (yy * w + x) * 3
                    out[o] = UInt8(clamping: s0 >> Int32(p))
                    out[o + 1] = UInt8(clamping: s1 >> Int32(p))
                    out[o + 2] = UInt8(clamping: s2 >> Int32(p))
                }
            }
            current = .init(width: w, height: outH, data: out)
        }
        return current
    }

    // MARK: Per-frame pipeline

    /// Thumbnail + slices for one frame, torchvision resize and fused normalisation (`rescale_and_normalize`:
    /// `(float32(x) - mean*255) / (std*255)`).
    static func process(
        frame: MiniCPMV47ImageProcessing.RGB8, maxSliceNums: Int, scaleResolution: Int, patchSize: Int, sliceMode: Bool,
        mean: [Float], std: [Float]
    ) -> MiniCPMV47ImageProcessing.ProcessedCrops {
        MiniCPMV47ImageProcessing.process(
            frame, maxSliceNums: maxSliceNums, scaleResolution: scaleResolution, patchSize: patchSize,
            sliceMode: sliceMode, mean: mean, std: std,
            resize: resizeBicubicTorchAA, normalization: .fusedTorch)
    }

    /// `replace_video_token`: the frames' placeholders back to back, no `<image_id>`.
    static func placeholderTokens(
        frames: [MiniCPMV47ImageProcessing.ProcessedCrops], padTokenId: Int, tokensPerPatchDivisor: Int,
        markers: (imageStart: Int, imageEnd: Int, sliceStart: Int, sliceEnd: Int, newline: Int)
    ) -> [Int] {
        frames.flatMap { crops in
            MiniCPMV47ImageProcessing.placeholderTokens(
                crops: crops, localIndex: 0, useImageId: false, padTokenId: padTokenId,
                tokensPerPatchDivisor: tokensPerPatchDivisor, markers: markers, imageIdStart: nil, imageIdEnd: nil,
                digitTokens: { _ in [] })
        }
    }
}
