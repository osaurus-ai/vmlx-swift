import CoreImage
import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import MLXVLM

/// End-to-end image path on a REAL file: processor resize/patchify/grid →
/// vision tower → merged tokens. The unit tests above feed the tower synthetic
/// patches, so they cannot catch a processor that produces the wrong grid, and
/// the model cannot catch a placeholder count that disagrees with the feature
/// count — that mismatch is silent until the scatter, and by then the shapes
/// no longer say which side was wrong.
@Suite("Muse Glimmer VL pipeline")
struct MuseGlimmerVLPipelineTests {

    static let imagePath = "/tmp/raptorproof/vl-bands.png"
    static var imageExists: Bool { FileManager.default.fileExists(atPath: imagePath) }

    static func processorConfig(maxTokens: Int = 4096) throws
        -> MuseGlimmerProcessorConfiguration
    {
        let json = """
            {
              "image_processor": {
                "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5],
                "patch_size": 14, "merge_size": 2, "temporal_patch_size": 2,
                "max_image_tokens": \(maxTokens)
              },
              "video_processor": { "fps": 2.0 }
            }
            """
        return try JSONDecoder().decode(
            MuseGlimmerProcessorConfiguration.self, from: Data(json.utf8))
    }

    static func visionConfig() throws -> MuseGlimmerVisionConfiguration {
        // Small tower, REAL patch geometry (14 / 2 / merge 2) so the grid math
        // the processor produces is what the tower consumes.
        let json = """
            {
              "hidden_size": 32, "intermediate_size": 64,
              "num_attention_heads": 2, "num_hidden_layers": 4,
              "patch_size": 14, "patch_temporal": 2, "merge_size": 2,
              "pos_emb_height": 8, "pos_emb_width": 8, "layer_norm_eps": 1e-5
            }
            """
        return try JSONDecoder().decode(
            MuseGlimmerVisionConfiguration.self, from: Data(json.utf8))
    }

    @Test("a real image patchifies to a grid the tower consumes", .enabled(if: imageExists))
    func realImageThroughProcessorAndTower() throws {
        let pConfig = try Self.processorConfig()
        let vConfig = try Self.visionConfig()
        let processor = MuseGlimmerProcessor(pConfig, tokenizer: StubTokenizer())

        let image = CIImage(contentsOf: URL(fileURLWithPath: Self.imagePath))!
        let (patches, frame) = try processor.preprocess(images: [image], processing: nil)
        eval(patches)

        // Patch rows must equal the grid, and the width must be the flattened
        // patch volume the linear embedder expects (1176 on the real bundle).
        #expect(patches.dim(0) == frame.t * frame.h * frame.w)
        #expect(patches.dim(1) == pConfig.temporalPatchSize * 3
            * pConfig.patchSize * pConfig.patchSize)
        // Both grid axes must be whole merge blocks or the 2x2 merge drops rows.
        #expect(frame.h % pConfig.mergeSize == 0, "grid h=\(frame.h) not merge-aligned")
        #expect(frame.w % pConfig.mergeSize == 0, "grid w=\(frame.w) not merge-aligned")
        // Token budget honoured.
        let mergedTokens = (frame.h / pConfig.mergeSize) * (frame.w / pConfig.mergeSize)
            * frame.t
        #expect(mergedTokens <= pConfig.maxImageTokens,
            "\(mergedTokens) merged tokens exceeds the \(pConfig.maxImageTokens) budget")

        // The tower consumes exactly what the processor produced.
        let tower = MuseGlimmerVisionModel(vConfig)
        let out = tower(patches.asType(.float32), frames: [frame])
        eval(out)

        #expect(out.dim(0) == mergedTokens,
            "tower emitted \(out.dim(0)) tokens, processor implies \(mergedTokens)")
        #expect(out.dim(1) == vConfig.hiddenSize * vConfig.mergeSize * vConfig.mergeSize)
        #expect(out.asArray(Float.self).allSatisfy { $0.isFinite })
    }

    @Test("a tighter token budget shrinks the grid instead of overflowing",
        .enabled(if: imageExists))
    func tokenBudgetShrinksTheGrid() throws {
        let image = CIImage(contentsOf: URL(fileURLWithPath: Self.imagePath))!
        let big = try Self.processorConfig(maxTokens: 4096)
        let small = try Self.processorConfig(maxTokens: 64)

        let (_, bigFrame) = try MuseGlimmerProcessor(big, tokenizer: StubTokenizer())
            .preprocess(images: [image], processing: nil)
        let (_, smallFrame) = try MuseGlimmerProcessor(small, tokenizer: StubTokenizer())
            .preprocess(images: [image], processing: nil)

        let bigTokens = (bigFrame.h / 2) * (bigFrame.w / 2)
        let smallTokens = (smallFrame.h / 2) * (smallFrame.w / 2)
        #expect(smallTokens <= 64, "budget ignored: \(smallTokens) tokens")
        #expect(smallTokens < bigTokens, "tighter budget did not shrink the grid")
        // Still merge-aligned after the shrink — the floor must not round to 0.
        #expect(smallFrame.h % 2 == 0 && smallFrame.w % 2 == 0)
        #expect(smallFrame.h > 0 && smallFrame.w > 0)
    }
}

/// Minimal tokenizer: `preprocess` never touches it, but `MuseGlimmerProcessor`
/// requires one at construction.
private struct StubTokenizer: Tokenizer {
    func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { nil }
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }
    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [] }
}
