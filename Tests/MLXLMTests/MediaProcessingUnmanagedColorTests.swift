// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Qwen3-VL preprocessing must see an image's stored RGB values, as the reference PIL processor
// does (`Image.open(...).convert("RGB")` ignores the ICC profile), not CoreImage's linear,
// gamut-converted working-space values.

import CoreImage
import Foundation
import MLX
import Testing

@testable import MLXVLM

@Suite("Unmanaged colour preprocessing", .serialized)
struct MediaProcessingUnmanagedColorTests {
    /// 8x8 solid blocks of distinct colours, tagged with `colorSpace`.
    private func taggedPNG(_ colorSpace: String) throws -> (Data, [UInt8], Int, Int) {
        let w = 64, h = 32
        let space = try #require(CGColorSpace(name: colorSpace as CFString))
        var raw = [UInt8](repeating: 0, count: w * h * 4)
        for y in 0 ..< h {
            for x in 0 ..< w {
                let i = (y * w + x) * 4
                raw[i] = UInt8((x / 8 * 37 + 20) % 256)
                raw[i + 1] = UInt8((y / 8 * 71 + 50) % 256)
                raw[i + 2] = UInt8((x / 8 * 13 + y / 8 * 29 + 90) % 256)
                raw[i + 3] = 255
            }
        }
        let provider = try #require(CGDataProvider(data: Data(raw) as CFData))
        let image = try #require(
            CGImage(
                width: w, height: h, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: w * 4,
                space: space, bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent))
        let data = NSMutableData()
        let destination = try #require(
            CGImageDestinationCreateWithData(data, "public.png" as CFString, 1, nil))
        CGImageDestinationAddImage(destination, image, nil)
        #expect(CGImageDestinationFinalize(destination))
        return (data as Data, raw, w, h)
    }

    @Test(
        "unmanaged render equals the stored values, normalized PIL-style",
        arguments: ["kCGColorSpaceITUR_2020_sRGBGamma", "kCGColorSpaceDisplayP3", "kCGColorSpaceSRGB"])
    func storedValues(colorSpace: String) throws {
        let (data, raw, w, h) = try taggedPNG(colorSpace)
        let image = try #require(CIImage(data: data))
        let normalized = MediaProcessing.normalize(
            MediaProcessing.resampleBicubic(image, to: CGSize(width: w, height: h)),
            mean: (0.5, 0.5, 0.5), std: (0.5, 0.5, 0.5))
        let pixels = MediaProcessing.asMLXArrayUnmanaged(normalized)
        #expect(pixels.shape == [1, 3, h, w])
        // Block centres (bicubic at scale 1 is exact away from block edges).
        var maxError: Float = 0
        for y in stride(from: 4, to: h, by: 8) {
            for x in stride(from: 4, to: w, by: 8) {
                for c in 0 ..< 3 {
                    let expected = (Float(raw[(y * w + x) * 4 + c]) / 255 - 0.5) / 0.5
                    maxError = max(maxError, abs(pixels[0, c, y, x].item(Float.self) - expected))
                }
            }
        }
        #expect(maxError < 2e-3, "max |unmanaged - reference| = \(maxError)")
        // The managed render is what Qwen3-VL used before: off by far more on wide-gamut sources.
        if colorSpace != "kCGColorSpaceSRGB" {
            let managed = MediaProcessing.asMLXArray(normalized)
            #expect(MLX.max(abs(managed - pixels)).item(Float.self) > 0.1)
        }
    }
}
