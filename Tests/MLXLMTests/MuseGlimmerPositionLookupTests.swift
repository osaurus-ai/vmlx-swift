import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXVLM

/// At a 32x32 patch grid the position table is sampled one-to-one — the
/// bilinear weights collapse to 1 on the exact cell — so every row of the
/// embedder's output must equal a specific row of `position_embedding_table`,
/// with no interpolation to hide behind.
///
/// That makes the index arithmetic checkable exactly. `patch_embedding` has no
/// bias, so feeding zero patches leaves the position term alone in the output.
/// Rows arrive grouped by merge block — `(hb, wb, di, dj)` — while the table is
/// row-major over the raster grid, and getting that correspondence wrong is
/// invisible in every shape and magnitude check.
@Suite("Muse Glimmer position lookup")
struct MuseGlimmerPositionLookupTests {

    static let bundle = URL(fileURLWithPath: NSHomeDirectory())
        .appendingPathComponent("models/JANGQ-AI/Muse-Glimmer-30B-JANG_6M")

    static var enabled: Bool {
        ProcessInfo.processInfo.environment["MUSE_VL_LIVE"] == "1"
            && FileManager.default.fileExists(
                atPath: bundle.appendingPathComponent("config.json").path)
    }

    @Test("every patch gets the position table row for its grid cell",
        .enabled(if: enabled))
    func lookupIsExact() throws {
        struct Wrapper: Decodable {
            let visionConfiguration: MuseGlimmerVisionConfiguration
            enum CodingKeys: String, CodingKey { case visionConfiguration = "vision_config" }
        }
        let data = try Data(contentsOf: Self.bundle.appendingPathComponent("config.json"))
        let config = try JSONDecoder().decode(Wrapper.self, from: data).visionConfiguration

        var weights: [String: MLXArray] = [:]
        var table: MLXArray?
        for f in try FileManager.default.contentsOfDirectory(
            at: Self.bundle, includingPropertiesForKeys: nil)
        where f.pathExtension == "safetensors" {
            guard let arrays = try? MLX.loadArrays(url: f) else { continue }
            for (k, v) in arrays where k.hasPrefix("model.vision_tower.") {
                let short = String(k.dropFirst("model.vision_tower.".count))
                weights[short] = v.asType(.float32)
                if short == "patch_embedder.position_embedding_table.weight" {
                    table = v.asType(.float32)
                }
            }
        }
        let positionTable = try #require(table)
        let tower = MuseGlimmerVisionModel(config)
        try tower.update(
            parameters: ModuleParameters.unflattened(weights),
            verify: Module.VerifyUpdate.all)
        eval(tower)

        let grid = 32
        let per = config.patchTemporal * config.patchSize * config.patchSize
        let zeros = MLXArray.zeros([grid * grid, 3 * per], dtype: .float32)
        let frame = THW(1, grid, grid)

        var embedded: MLXArray?
        _ = tower(zeros, frames: [frame]) { stage, h in
            if stage == "embedder" { embedded = h }
        }
        let actual = try #require(embedded)
        eval(actual)
        print("[poslookup] embedder output \(actual.shape), table \(positionTable.shape)")

        // The tower runs over the raster grid, so row k is simply grid cell k
        // and the table is indexed row-major. At a 32x32 grid that is the
        // identity — any departure from it is an ordering bug.
        var expectedIndices: [Int32] = []
        for row in 0 ..< grid {
            for col in 0 ..< grid {
                expectedIndices.append(Int32(row * config.posEmbWidth + col))
            }
        }
        let expected = positionTable[MLXArray(expectedIndices), 0...]
        eval(expected)

        let delta = MLX.abs(actual.asType(.float32) - expected).max()
        delta.eval()
        let worst = delta.item(Float.self)
        print("[poslookup] max|actual - expected| = \(worst)")

        // If the arithmetic is off, report how far: a pure transpose shows up
        // as matching the transposed index set instead.
        var transposed: [Int32] = []
        for row in 0 ..< grid {
            for col in 0 ..< grid {
                transposed.append(Int32(col * config.posEmbWidth + row))
            }
        }
        let tDelta = MLX.abs(
            actual.asType(.float32) - positionTable[MLXArray(transposed), 0...]).max()
        tDelta.eval()
        print("[poslookup] max|actual - TRANSPOSED expected| = \(tDelta.item(Float.self))")

        #expect(worst < 1e-4,
            "the position term does not match the table row for each raster cell (max delta \(worst)) — the index arithmetic in interpolatedPositions is wrong")
    }
}
