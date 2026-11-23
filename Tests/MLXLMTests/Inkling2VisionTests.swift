// Copyright © 2026 Osaurus AI. All rights reserved.

import Foundation
import MLX
@testable import MLXLLM
import Testing

@Suite("Inkling-2 vision input", .serialized)
struct Inkling2VisionTests {
    let img = 200054

    @Test("each placeholder expands to one per tile; boundaries follow")
    func expansion() throws {
        let raw = [1, 2, img, 3, img, 4]
        let e = try Inkling2VisionPrompt.expand(tokens: raw, imageTokenId: img, counts: [3, 2])
        #expect(e.tokens == [1, 2, img, img, img, 3, img, img, 4])
        #expect(e.rawToExpanded == [0, 1, 2, 5, 6, 8, 9])
        #expect(Inkling2VisionPrompt.remapBoundaries([2, 4, 6], rawToExpanded: e.rawToExpanded) == [2, 6, 9])
        #expect(throws: Inkling2VisionError.self) {
            _ = try Inkling2VisionPrompt.expand(tokens: raw, imageTokenId: img, counts: [3])
        }
    }

    @Test("spans: whole prompt, a suffix after a restore, and a restore that splits an image")
    func spans() throws {
        let grids = [(rows: 1, cols: 3), (rows: 2, cols: 1)]  // 3 and 2 tiles
        let full = [9, img, img, img, 8, img, img, 7]
        let a = try Inkling2VisionLayout.spans(tokens: full, imageTokenId: img, grids: grids)
        #expect(a.map(\.start) == [1, 5] && a.map(\.length) == [3, 2])
        #expect(a.map(\.tileStart) == [0, 3] && a.map(\.skippedTiles) == [0, 0])
        // Restored through the first image: only the second remains.
        let b = try Inkling2VisionLayout.spans(tokens: [8, img, img, 7], imageTokenId: img, grids: grids)
        #expect(b.count == 1 && b[0].tileStart == 3 && b[0].skippedTiles == 0 && b[0].length == 2)
        // Restored into the middle of the first image: its last tile, then the second image.
        let c = try Inkling2VisionLayout.spans(tokens: [img, 8, img, img, 7], imageTokenId: img, grids: grids)
        #expect(c.count == 2)
        #expect(c[0].tileStart == 0 && c[0].skippedTiles == 2 && c[0].length == 1)
        #expect(c[1].tileStart == 3 && c[1].skippedTiles == 0 && c[1].length == 2)
        #expect(try Inkling2VisionLayout.spans(tokens: [1, 2], imageTokenId: img, grids: grids).isEmpty)
        #expect(throws: Inkling2VisionError.self) {
            _ = try Inkling2VisionLayout.spans(tokens: Array(repeating: img, count: 6), imageTokenId: img, grids: grids)
        }
    }

    @Test("preprocessing: tile-major layout, -1 padding, mean/std 0.5")
    func preprocessing() {
        // 3 x 2 image, tile 2: rows = 1, cols = 2 (padded to 4 wide).
        var rgba = [UInt8](repeating: 255, count: 3 * 2 * 4)
        for i in 0 ..< 6 { rgba[i * 4] = UInt8(i * 40) }
        let p = Inkling2ImagePreprocessor.process(rgba: rgba, width: 3, height: 2, premultiplied: false, tile: 2)
        #expect(p.rows == 1 && p.cols == 2 && p.tiles.count == 2 * 2 * 2 * 3)
        func r(_ v: Int) -> Float { (Float(v) / 255 - 0.5) / 0.5 }
        // Tile 0 holds x 0-1, tile 1 holds x 2 and padding.
        #expect(p.tiles[0] == r(0))  // (y0, x0)
        #expect(p.tiles[3] == r(40))  // (y0, x1)
        #expect(p.tiles[6] == r(120))  // (y1, x0)
        let t1 = 2 * 2 * 3
        #expect(p.tiles[t1] == r(80))  // tile 1 (y0, x2)
        #expect(p.tiles[t1 + 3] == Float((-1.0 / 255 - 0.5) / 0.5))  // padding
    }

    @Test("bilinear weights match align_corners=False")
    func bilinear() {
        let w = Inkling2VisionModel.bilinearWeights(out: 4, in: 2).asArray(Float.self)
        // src = (o + 0.5) * 0.5 - 0.5 = -0.25, 0.25, 0.75, 1.25 -> clamp 0 / interpolate / clamp top.
        #expect(w == [1, 0, 0.75, 0.25, 0.25, 0.75, 0, 1])
        let same = Inkling2VisionModel.bilinearWeights(out: 3, in: 3).asArray(Float.self)
        #expect(same == [1, 0, 0, 0, 1, 0, 0, 0, 1])
    }

    @Test("vision tensors of every layout map to one tree")
    func visionKeys() {
        let k = Inkling2LoopModel.visionKey
        #expect(k("model.visual.backbone.layers.3.ln_1.weight") == "vision_tower.layers.3.layernorm_before.weight")
        #expect(k("model.visual.backbone.layers.3.ls_2.gamma") == "vision_tower.layers.3.lambda_2")
        #expect(k("model.visual.backbone.layers.3.mlp.c_fc.bias") == "vision_tower.layers.3.mlp.fc1.bias")
        #expect(k("model.visual.backbone.ln_pre.weight") == "vision_tower.pre_layernorm.weight")
        #expect(k("model.visual.backbone.embeddings.position_embedding") == "vision_tower.embeddings.position_embedding")
        #expect(k("model.visual.proj_in.weight") == "vision_tower.merger.linear_1.weight")
        #expect(k("model.vision_tower.embeddings.position_embedding.weight") == "vision_tower.embeddings.position_embedding")
        #expect(k("model.vision_tower.merger.linear_2.weight") == "vision_tower.merger.linear_2.weight")
        #expect(k("vision_tower.proj_out.weight") == "vision_tower.merger.linear_2.weight")
        #expect(k("vision_tower.backbone.layers.0.self_attn.q_proj.weight") == "vision_tower.layers.0.self_attn.q_proj.weight")
        #expect(k("vision_tower.layers.0.lambda_1") == "vision_tower.layers.0.lambda_1")
        #expect(k("model.layers.0.mlp.gate.weight") == nil)
    }
}
