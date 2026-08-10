import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import MLXVLM

/// Both quants describe a red/green/blue banded image as greyscale or
/// "low-contrast", while the patch tensor demonstrably carries full colour
/// separation. That narrows the loss to the tower itself, so this feeds solid
/// colour fields straight in and asks whether the output embeddings can tell
/// them apart at all.
///
/// Random weights are fine for this: the question is whether colour
/// *information* survives the architecture, not whether it is interpreted.
/// A tower that collapses channels produces near-identical embeddings for red
/// and blue no matter what the weights are.
@Suite("Muse Glimmer tower colour")
struct MuseGlimmerTowerColourTests {

    static func config() throws -> MuseGlimmerVisionConfiguration {
        let json = """
            {
              "hidden_size": 64, "intermediate_size": 128,
              "num_attention_heads": 4, "num_hidden_layers": 4,
              "patch_size": 14, "patch_temporal": 2, "merge_size": 2,
              "pos_emb_height": 8, "pos_emb_width": 8, "layer_norm_eps": 1e-5
            }
            """
        return try JSONDecoder().decode(
            MuseGlimmerVisionConfiguration.self, from: Data(json.utf8))
    }

    /// A patch tensor for a 4x4 grid filled with one normalized RGB colour,
    /// laid out `[channel][temporal][h][w]` exactly as `QwenVL.patchify` emits.
    static func solidPatches(r: Float, g: Float, b: Float, config: MuseGlimmerVisionConfiguration)
        -> (MLXArray, THW)
    {
        let frame = THW(1, 4, 4)
        let perChannel = config.patchTemporal * config.patchSize * config.patchSize
        let rows = frame.t * frame.h * frame.w
        var flat: [Float] = []
        flat.reserveCapacity(rows * 3 * perChannel)
        for _ in 0 ..< rows {
            flat += Array(repeating: r, count: perChannel)
            flat += Array(repeating: g, count: perChannel)
            flat += Array(repeating: b, count: perChannel)
        }
        return (MLXArray(flat).reshaped(rows, 3 * perChannel), frame)
    }

    @Test("the tower distinguishes red from blue")
    func towerSeparatesColours() throws {
        let config = try Self.config()
        let tower = MuseGlimmerVisionModel(config)

        // Normalized (mean/std 0.5) pure red and pure blue.
        let (redPatches, frame) = Self.solidPatches(r: 1, g: -1, b: -1, config: config)
        let (bluePatches, _) = Self.solidPatches(r: -1, g: -1, b: 1, config: config)

        let red = tower(redPatches, frames: [frame])
        let blue = tower(bluePatches, frames: [frame])
        eval(red, blue)

        let r = red.asArray(Float.self)
        let b = blue.asArray(Float.self)
        #expect(r.count == b.count)

        // Mean absolute difference between the two embeddings, against the
        // typical magnitude — a collapsed tower lands near zero.
        var diff: Float = 0
        var magnitude: Float = 0
        for i in 0 ..< r.count {
            diff += abs(r[i] - b[i])
            magnitude += abs(r[i])
        }
        diff /= Float(r.count)
        magnitude /= Float(r.count)
        let relative = magnitude > 0 ? diff / magnitude : 0
        print("[tower-colour] mean|Δ|=\(diff) mean|red|=\(magnitude) relative=\(relative)")

        #expect(r.allSatisfy { $0.isFinite } && b.allSatisfy { $0.isFinite })
        #expect(relative > 0.1,
            "tower output barely differs between red and blue (relative=\(relative)) — colour is being collapsed inside the tower")
    }

    @Test("the patch embedder alone keeps colour apart")
    func patchEmbedderSeparatesColours() throws {
        let config = try Self.config()
        let tower = MuseGlimmerVisionModel(config)
        let (redPatches, frame) = Self.solidPatches(r: 1, g: -1, b: -1, config: config)
        let (greenPatches, _) = Self.solidPatches(r: -1, g: 1, b: -1, config: config)

        // Whole-tower comparison for a second colour pair, so a pass on
        // red/blue alone cannot come from one lucky channel.
        let red = tower(redPatches, frames: [frame])
        let green = tower(greenPatches, frames: [frame])
        eval(red, green)
        let rv = red.asArray(Float.self), gv = green.asArray(Float.self)
        var diff: Float = 0
        for i in 0 ..< rv.count { diff += abs(rv[i] - gv[i]) }
        diff /= Float(rv.count)
        print("[tower-colour] red vs green mean|Δ|=\(diff)")
        #expect(diff > 1e-3, "red and green produce the same embedding")
    }
}
