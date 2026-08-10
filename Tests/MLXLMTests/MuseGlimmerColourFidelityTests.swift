import CoreImage
import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import MLXVLM

/// The live model described a red/green/blue banded image as "a greyscale
/// ramp" — it saw the bands but not their colour. That points at the pixel
/// pipeline rather than the weights, so this reads the channel values straight
/// out of the patch tensor.
///
/// Row layout from `QwenVL.patchify` is `[channel][temporal][h][w]`, so channel
/// `c` occupies a contiguous block of `temporal * patch * patch` values.
@Suite("Muse Glimmer colour fidelity")
struct MuseGlimmerColourFidelityTests {

    static let imagePath = "/tmp/raptorproof/vl-bands.png"
    static var imageExists: Bool { FileManager.default.fileExists(atPath: imagePath) }

    @Test("the three bands keep distinct channel signatures", .enabled(if: imageExists))
    func bandsKeepColour() throws {
        let json = """
            {
              "image_processor": {
                "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5],
                "patch_size": 14, "merge_size": 2, "temporal_patch_size": 2,
                "max_image_tokens": 4096
              }
            }
            """
        let config = try JSONDecoder().decode(
            MuseGlimmerProcessorConfiguration.self, from: Data(json.utf8))
        let processor = MuseGlimmerProcessor(config, tokenizer: ColourStubTokenizer())

        let image = CIImage(contentsOf: URL(fileURLWithPath: Self.imagePath))!
        let (patches, frame) = try processor.preprocess(images: [image], processing: nil)
        eval(patches)
        let values = patches.asArray(Float.self)

        let perChannel = config.temporalPatchSize * config.patchSize * config.patchSize
        let rowWidth = 3 * perChannel
        #expect(patches.dim(1) == rowWidth)

        // Mean of each channel for a given patch row.
        func channelMeans(row: Int) -> (r: Float, g: Float, b: Float) {
            let base = row * rowWidth
            func mean(_ c: Int) -> Float {
                let start = base + c * perChannel
                var total: Float = 0
                for i in start ..< (start + perChannel) { total += values[i] }
                return total / Float(perChannel)
            }
            return (mean(0), mean(1), mean(2))
        }

        // Patch rows are grid-ordered, so the first row of patches is the top
        // band and the last is the bottom.
        let top = channelMeans(row: 0)
        let bottom = channelMeans(row: frame.t * frame.h * frame.w - 1)
        print("[colour] top   r=\(top.r) g=\(top.g) b=\(top.b)")
        print("[colour] bottom r=\(bottom.r) g=\(bottom.g) b=\(bottom.b)")

        // Source image: top band is red (220,30,30), bottom is blue (40,70,220).
        // If colour survives, red dominates at the top and blue at the bottom.
        #expect(top.r > top.b, "top band lost red dominance (r=\(top.r) b=\(top.b))")
        #expect(bottom.b > bottom.r, "bottom band lost blue dominance")

        // And the channels must not be collapsed to a single grey value.
        let topSpread = max(top.r, top.g, top.b) - min(top.r, top.g, top.b)
        #expect(topSpread > 0.2, "channels collapsed to grey (spread=\(topSpread))")
    }
}

private struct ColourStubTokenizer: Tokenizer {
    func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { nil }
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }
    func applyChatTemplate(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [] }
}
