import XCTest
@testable import MLXVLM

final class NemotronImageResamplingTests: XCTestCase {
    func testFloatBicubicMatchesTorchAntialiasReference() {
        // torch 2.14.1, F.interpolate(float32 RGB, mode="bicubic",
        // align_corners=False, antialias=True). Values are R; G/B differ by 3/6.
        let cases: [(Int, Int, Int, Int, [Float])] = [
            (5, 3, 3, 5, [
                -0.21160911, 14.603775, 29.419155,
                16.953272, 31.768658, 46.584034,
                48.184616, 63, 77.815376,
                79.415962, 94.231346, 109.046722,
                96.580841, 111.396225, 126.211609]),
            (3, 5, 7, 2, [
                22.267618, 23.905857, 27.570675, 32.133003, 36.695332, 40.360149, 41.998386,
                84.001610, 85.639862, 89.304672, 93.866997, 98.429337, 102.094154, 103.732384]),
            (7, 4, 3, 2, [43.426620, 63.966942, 84.507271, 158.492752, 179.033066, 199.573380]),
        ]
        for (width, height, outWidth, outHeight, expected) in cases {
            let input = (0 ..< width * height * 3).map { UInt8(($0 * 3) % 256) }
            let output = NemotronImageResampling.resize(input, width: width, height: height,
                targetWidth: outWidth, targetHeight: outHeight)
            XCTAssertEqual(output.count, expected.count * 3)
            for (pixel, red) in expected.enumerated() {
                for channel in 0 ..< 3 {
                    XCTAssertEqual(output[pixel * 3 + channel], red + Float(channel * 3), accuracy: 0.0001,
                        "\(width)x\(height) -> \(outWidth)x\(outHeight), pixel=\(pixel), channel=\(channel)")
                }
            }
        }
    }

    func testUnscaledPixelsRetainTheirOriginalBytes() {
        let input: [UInt8] = [0, 128, 255, 1, 32, 240, 9, 4, 99, 220, 33, 19]
        XCTAssertEqual(NemotronImageResampling.resize(input, width: 2, height: 2,
            targetWidth: 2, targetHeight: 2), input.map(Float.init))
    }
}
