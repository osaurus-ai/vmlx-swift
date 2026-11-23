// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Testing

@testable import MLXVLM

/// An image or a video frame smaller than the patch factor is resized, as the reference does. It is
/// not rejected. The expected sizes come from `smart_resize` in transformers main (8108baccf,
/// 2026-11-16): `qwen2_vl/image_processing_qwen2_vl.py` for images and
/// `qwen3_vl/video_processing_qwen3_vl.py` for video. Before this fix, a 17-pixel-high image in
/// ocrbench_v2 made every Qwen3.5 and Qwen3-VL run fail with "Height: 17 must be larger than factor".
/// The inputs keep each rounding step away from an integer, so Float and Double give the same result.
struct QwenVLTinyImageResizeTests {
    @Test("An image lower than the factor is scaled up through min_pixels")
    func imageLowerThanFactorScalesUp() throws {
        // Qwen3-VL: patch 16 x merge 2, min_pixels 65536, max_pixels 16777216.
        let size = try QwenVL.targetSize(
            height: 17, width: 640, factor: 32, minPixels: 65536, maxPixels: 16_777_216)
        #expect(size.0 == 64 && size.1 == 1600, "got \(size)")
    }

    @Test("An image lower than the factor becomes one patch row when the pixel budget allows it")
    func imageLowerThanFactorRoundsToOneRow() throws {
        // Qwen2-VL: patch 14 x merge 2, min_pixels 3136, max_pixels 12845056.
        let size = try QwenVL.targetSize(
            height: 17, width: 640, factor: 28, minPixels: 3136, maxPixels: 12_845_056)
        #expect(size.0 == 28 && size.1 == 644, "got \(size)")
    }

    @Test("An image lower than half the factor rounds to zero rows, then min_pixels scales it up")
    func imageLowerThanHalfTheFactor() throws {
        let size = try QwenVL.targetSize(
            height: 12, width: 700, factor: 28, minPixels: 3136, maxPixels: 12_845_056)
        #expect(size.0 == 28 && size.1 == 448, "got \(size)")
    }

    @Test("An image of normal size keeps its size")
    func normalImageIsUnchanged() throws {
        let image = try QwenVL.targetSize(
            height: 480, width: 640, factor: 32, minPixels: 65536, maxPixels: 16_777_216)
        #expect(image.0 == 480 && image.1 == 640, "got \(image)")
    }

    @Test("An image side of zero is an error, not a division by zero")
    func zeroImageSideIsAnError() {
        #expect(throws: (any Error).self) {
            try QwenVL.targetSize(
                height: 0, width: 640, factor: 32, minPixels: 65536, maxPixels: 16_777_216)
        }
    }

    @Test("An image aspect ratio over 200 is still an error, as in the reference")
    func extremeImageAspectRatioIsStillAnError() {
        #expect(throws: (any Error).self) {
            try QwenVL.targetSize(
                height: 1, width: 300, factor: 32, minPixels: 65536, maxPixels: 16_777_216)
        }
    }

    // Qwen3VLProcessor exists only where CoreImage and AVFoundation do, so not on Linux.
    #if canImport(CoreImage) && canImport(AVFoundation)
        @Test("A video frame smaller than the factor is scaled up before it is resized")
        func videoFrameSmallerThanFactorScalesUp() throws {
            let short = try Qwen3VLProcessor.videoTargetSize(
                frameCount: 4, height: 17, width: 640, temporalFactor: 2, factor: 32,
                minPixels: 4096, maxPixels: 25_165_824)
            #expect(short.0 == 32 && short.1 == 1216, "got \(short)")
            let narrow = try Qwen3VLProcessor.videoTargetSize(
                frameCount: 4, height: 10, width: 1000, temporalFactor: 2, factor: 28,
                minPixels: 4096, maxPixels: 25_165_824)
            #expect(narrow.0 == 28 && narrow.1 == 2800, "got \(narrow)")
        }

        @Test(
            "A video frame of normal size keeps its size; zero and extreme aspect ratios are errors"
        )
        func videoFrameEdgeCases() throws {
            let frame = try Qwen3VLProcessor.videoTargetSize(
                frameCount: 4, height: 480, width: 640, temporalFactor: 2, factor: 32,
                minPixels: 4096, maxPixels: 25_165_824)
            #expect(frame.0 == 480 && frame.1 == 640, "got \(frame)")
            #expect(throws: (any Error).self) {
                try Qwen3VLProcessor.videoTargetSize(
                    frameCount: 4, height: 17, width: 0, temporalFactor: 2, factor: 32,
                    minPixels: 4096, maxPixels: 25_165_824)
            }
            #expect(throws: (any Error).self) {
                try Qwen3VLProcessor.videoTargetSize(
                    frameCount: 4, height: 1, width: 300, temporalFactor: 2, factor: 32,
                    minPixels: 4096, maxPixels: 25_165_824)
            }
        }
    #endif
}
