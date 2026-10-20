// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// The processor's pixels are the media cache key (`computeMediaSalt`) and the model's input, so
// the same image bytes must always produce the same pixel tensor: serially, and while another
// render runs concurrently (the app prepares several requests at once).

import CoreImage
import Foundation
import MLX
import Testing

@testable import MLXVLM

@Suite("Media preprocessing determinism", .serialized)
struct MediaProcessingDeterminismTests {
    /// A PNG tagged with a wide-gamut (ITU-R 2020) profile, like screenshots and many app exports.
    private func taggedPNG(colorSpace: CFString, width: Int = 512, height: Int = 256) throws -> Data {
        let space = try #require(CGColorSpace(name: colorSpace))
        let context = try #require(
            CGContext(
                data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 0,
                space: space, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue))
        for y in stride(from: 0, to: height, by: 8) {
            for x in stride(from: 0, to: width, by: 8) {
                context.setFillColor(
                    red: CGFloat((x * 7 + y) % 256) / 255, green: CGFloat((y * 3) % 256) / 255,
                    blue: CGFloat((x + y * 5) % 256) / 255, alpha: 1)
                context.fill(CGRect(x: x, y: y, width: 8, height: 8))
            }
        }
        let image = try #require(context.makeImage())
        let data = NSMutableData()
        let destination = try #require(
            CGImageDestinationCreateWithData(data, "public.png" as CFString, 1, nil))
        CGImageDestinationAddImage(destination, image, nil)
        #expect(CGImageDestinationFinalize(destination))
        return data as Data
    }

    /// The Qwen VL image path: resample to the target grid, normalize, render to float32.
    private func pixels(_ data: Data) throws -> [Float] {
        let image = try #require(CIImage(data: data))
        let resampled = MediaProcessing.resampleBicubic(image, to: CGSize(width: 512, height: 256))
        let normalized = MediaProcessing.normalize(
            resampled, mean: (0.5, 0.5, 0.5), std: (0.5, 0.5, 0.5))
        let array = MediaProcessing.asMLXArray(normalized)
        MLX.eval(array)
        return array.asArray(Float.self)
    }

    @Test(
        "same bytes give the same pixels, serially and under concurrent renders",
        arguments: ["kCGColorSpaceITUR_2020_sRGBGamma", "kCGColorSpaceDisplayP3", "kCGColorSpaceSRGB"])
    func deterministic(colorSpace: String) throws {
        let data = try taggedPNG(colorSpace: colorSpace as CFString)
        let reference = try pixels(data)
        for _ in 0 ..< 4 { #expect(try pixels(data) == reference) }
        let results = OSAllocatedUnfairLockBox<[Float]>([])
        DispatchQueue.concurrentPerform(iterations: 6) { _ in
            if let value = try? pixels(data) { results.append(value) }
        }
        let concurrent = results.value
        #expect(concurrent.count == 6)
        for value in concurrent { #expect(value == reference) }
    }
}

/// Minimal lock-protected collector for the concurrent renders above.
private final class OSAllocatedUnfairLockBox<T>: @unchecked Sendable {
    private let lock = NSLock()
    private var stored: [T] = []
    init(_ initial: [T]) { stored = initial }
    func append(_ value: T) { lock.lock(); stored.append(value); lock.unlock() }
    var value: [T] { lock.lock(); defer { lock.unlock() }; return stored }
}
