// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import CoreImage
import Foundation
import ImageIO
import MLX
import MLXLMCommon
import Testing

@testable import MLXVLM

@Suite("Gemma unified stored RGB contract", .serialized)
struct Gemma4UnifiedPixelContractTests {
    private struct ImageTokenizer: Tokenizer {
        func encode(text: String, addSpecialTokens: Bool) -> [Int] { [258880] }
        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
        func convertTokenToId(_ token: String) -> Int? { token == "<|image|>" ? 258880 : nil }
        func convertIdToToken(_ id: Int) -> String? { nil }
        var bosToken: String? { nil }
        var eosToken: String? { nil }
        var unknownToken: String? { nil }
        func applyChatTemplate(
            messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?
        ) throws -> [Int] { [258880] }
    }

    @Test(
        arguments: [
            "kCGColorSpaceSRGB", "kCGColorSpaceDisplayP3", "kCGColorSpaceITUR_2020_sRGBGamma",
        ], [UInt8(255), UInt8(128), UInt8(0)])
    func productionProcessorStoredRGB(colorSpace: String, alpha: UInt8) async throws {
        try await MLXMetalTestLock.withLock {
            // 5 x 14 pooled patches: the real processor chooses the same dimensions,
            // isolating colour from resize interpolation and patch grouping.
            let w = 240
            let h = 672
            let space = try #require(CGColorSpace(name: colorSpace as CFString))
            var raw = [UInt8](repeating: 255, count: w * h * 4)
            for y in 0 ..< h {
                for x in 0 ..< w {
                    let i = (y * w + x) * 4
                    raw[i] = UInt8((x / 48 * 37 + 20) % 256)
                    raw[i + 1] = UInt8((y / 48 * 71 + 50) % 256)
                    raw[i + 2] = UInt8((x / 48 * 13 + y / 48 * 29 + 90) % 256)
                    raw[i + 3] = alpha
                }
            }
            let provider = try #require(CGDataProvider(data: Data(raw) as CFData))
            let image = try #require(
                CGImage(
                    width: w, height: h, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: w * 4,
                    space: space,
                    bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue),
                    provider: provider, decode: nil, shouldInterpolate: false,
                    intent: .defaultIntent))
            let data = NSMutableData()
            let dest = try #require(
                CGImageDestinationCreateWithData(data, "public.png" as CFString, 1, nil))
            CGImageDestinationAddImage(dest, image, nil)
            #expect(CGImageDestinationFinalize(dest))
            let ci = try #require(CIImage(data: data as Data))
            let config = try JSONDecoder().decode(
                Gemma4ProcessorConfiguration.self,
                from: Data(
                    """
                    {"processor_class":"Gemma4UnifiedProcessor","image_processor":{
                    "image_processor_type":"Gemma4UnifiedImageProcessor","max_soft_tokens":70,
                    "patch_size":16,"pooling_kernel_size":3},"image_seq_length":70}
                    """.utf8))
            let processor = Gemma4Processor(config, tokenizer: ImageTokenizer())
            let input = try await processor.prepare(
                input: UserInput(
                    prompt: .text("Read this image."), images: [.ciImage(ci)]))
            let pixels = try #require(input.image?.pixels)
            let repeated = try await processor.prepare(
                input: UserInput(
                    prompt: .text("Read this image."), images: [.ciImage(ci)]))
            let salt = try #require(computeMediaSalt(for: input))
            #expect(salt == computeMediaSalt(for: repeated))
            #expect(pixels.shape == [1, 3, h, w])
            guard pixels.shape == [1, 3, h, w] else { return }
            let values = pixels.asArray(Float.self)
            var maxError: Float = 0
            for y in stride(from: 24, to: h, by: 48) {
                for x in stride(from: 24, to: w, by: 48) {
                    for c in 0 ..< 3 {
                        let stored = Float(raw[(y * w + x) * 4 + c])
                        let a = Float(alpha) / 255
                        let expected = (stored * a + 255 * (1 - a)).rounded() / 255
                        maxError = max(maxError, abs(values[c * h * w + y * w + x] - expected))
                    }
                }
            }
            #expect(
                maxError < 2e-3,
                "production Gemma processor stored RGB error \(maxError), \(colorSpace), alpha \(alpha)"
            )
        }
    }
}
