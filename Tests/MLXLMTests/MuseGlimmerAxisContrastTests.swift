import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXVLM

/// Live probes show Muse Glimmer reporting vertical position correctly (a bar
/// in the top quarter reads as "top", one in the bottom reads as "bottom") but
/// losing horizontal position entirely — a left bar and a right bar both read
/// as "right", while Qwen3.6 gets all four right.
///
/// This reproduces that asymmetry at the tower, where it can be localized. Two
/// splits of the same picture — top/bottom and left/right — should separate
/// their halves about equally. Qwen3.6 runs the identical measurement, because
/// a ratio means nothing without knowing what a working tower scores.
@Suite("Muse Glimmer axis contrast")
struct MuseGlimmerAxisContrastTests {

    static let muse = URL(fileURLWithPath: NSHomeDirectory())
        .appendingPathComponent("models/JANGQ-AI/Muse-Glimmer-30B-JANG_6M")
    static let qwen = URL(fileURLWithPath: NSHomeDirectory())
        .appendingPathComponent("models/dealign.ai/Qwen3.6-27B-JANG_4M-CRACK")

    static var enabled: Bool {
        ProcessInfo.processInfo.environment["MUSE_VL_LIVE"] == "1"
            && FileManager.default.fileExists(
                atPath: muse.appendingPathComponent("config.json").path)
    }

    /// Patch rows for a 32x32 grid split either horizontally or vertically.
    static func split(vertical: Bool, merge: Int, per: Int, grid: Int) -> MLXArray {
        let cells = grid / merge
        var flat: [Float] = []
        for hb in 0 ..< cells {
            for wb in 0 ..< cells {
                for di in 0 ..< merge {
                    for dj in 0 ..< merge {
                        let row = hb * merge + di, col = wb * merge + dj
                        let dark = vertical ? row < grid / 2 : col < grid / 2
                        flat += Array(repeating: dark ? Float(-1) : Float(1), count: 3 * per)
                    }
                }
            }
        }
        return MLXArray(flat).reshaped(grid * grid, 3 * per)
    }

    /// Separation between the two halves, in units of the spread inside one
    /// half — so vertical and horizontal are directly comparable.
    static func separation(_ out: MLXArray, vertical: Bool, cells: Int) -> Float {
        let rows = out.dim(0)
        var groupA: [Int32] = [], groupB: [Int32] = []
        for i in 0 ..< rows {
            let inFirst = vertical ? (i / cells) < cells / 2 : (i % cells) < cells / 2
            if inFirst { groupA.append(Int32(i)) } else { groupB.append(Int32(i)) }
        }
        let a = out[MLXArray(groupA), 0...]
        let b = out[MLXArray(groupB), 0...]
        let across = MLX.abs(a.mean(axis: 0) - b.mean(axis: 0)).mean()
        let ma = a.mean(axis: 0)
        let within = MLX.sqrt((a - ma).square().mean(axis: 0)).mean()
        eval(across, within)
        return across.item(Float.self) / max(within.item(Float.self), 1e-6)
    }

    @Test("both axes survive the tower as well as they do in a working one",
        .enabled(if: enabled))
    func axesAreSymmetric() throws {
        struct MuseWrapper: Decodable {
            let visionConfiguration: MuseGlimmerVisionConfiguration
            enum CodingKeys: String, CodingKey { case visionConfiguration = "vision_config" }
        }
        let museConfig = try JSONDecoder().decode(
            MuseWrapper.self,
            from: Data(contentsOf: Self.muse.appendingPathComponent("config.json"))
        ).visionConfiguration
        var museWeights: [String: MLXArray] = [:]
        for f in try FileManager.default.contentsOfDirectory(
            at: Self.muse, includingPropertiesForKeys: nil)
        where f.pathExtension == "safetensors" {
            guard let arrays = try? MLX.loadArrays(url: f) else { continue }
            for (k, v) in arrays where k.hasPrefix("model.vision_tower.") {
                museWeights[String(k.dropFirst("model.vision_tower.".count))] = v.asType(.float32)
            }
        }
        let museTower = MuseGlimmerVisionModel(museConfig)
        try museTower.update(
            parameters: ModuleParameters.unflattened(museWeights),
            verify: Module.VerifyUpdate.all)
        eval(museTower)

        let grid = 32
        let musePer = museConfig.patchTemporal * museConfig.patchSize * museConfig.patchSize
        let museCells = grid / museConfig.mergeSize
        let frame = THW(1, grid, grid)
        let museV = museTower(
            Self.split(vertical: true, merge: museConfig.mergeSize, per: musePer, grid: grid),
            frames: [frame])
        let museH = museTower(
            Self.split(vertical: false, merge: museConfig.mergeSize, per: musePer, grid: grid),
            frames: [frame])
        eval(museV, museH)
        let mV = Self.separation(museV, vertical: true, cells: museCells)
        let mH = Self.separation(museH, vertical: false, cells: museCells)
        print("[axis] Muse   vertical=\(mV)  horizontal=\(mH)  ratio V/H=\(mV / max(mH, 1e-6))")

        // Control.
        guard FileManager.default.fileExists(
            atPath: Self.qwen.appendingPathComponent("config.json").path)
        else {
            Issue.record("control bundle missing; no baseline for these numbers")
            return
        }
        struct QwenWrapper: Decodable {
            let visionConfiguration: Qwen3VLConfiguration.VisionConfiguration
            enum CodingKeys: String, CodingKey { case visionConfiguration = "vision_config" }
        }
        let qwenConfig = try JSONDecoder().decode(
            QwenWrapper.self,
            from: Data(contentsOf: Self.qwen.appendingPathComponent("config.json"))
        ).visionConfiguration
        var qwenWeights: [String: MLXArray] = [:]
        for f in try FileManager.default.contentsOfDirectory(
            at: Self.qwen, includingPropertiesForKeys: nil)
        where f.pathExtension == "safetensors" {
            guard let arrays = try? MLX.loadArrays(url: f) else { continue }
            for (k, v) in arrays where k.hasPrefix("vision_tower.") {
                qwenWeights[String(k.dropFirst("vision_tower.".count))] = v.asType(.float32)
            }
        }
        let qwenTower = Qwen3VLVision.VisionModel(qwenConfig)
        try qwenTower.update(
            parameters: ModuleParameters.unflattened(qwenTower.sanitize(weights: qwenWeights)),
            verify: Module.VerifyUpdate.all)
        eval(qwenTower)

        let qwenPer = qwenConfig.temporalPatchSize * qwenConfig.patchSize * qwenConfig.patchSize
        let qwenCells = grid / qwenConfig.spatialMergeSize
        let (qV, _) = qwenTower(
            Self.split(vertical: true, merge: qwenConfig.spatialMergeSize, per: qwenPer, grid: grid),
            gridTHW: [frame])
        let (qH, _) = qwenTower(
            Self.split(vertical: false, merge: qwenConfig.spatialMergeSize, per: qwenPer, grid: grid),
            gridTHW: [frame])
        eval(qV, qH)
        let cV = Self.separation(qV, vertical: true, cells: qwenCells)
        let cH = Self.separation(qH, vertical: false, cells: qwenCells)
        print("[axis] Qwen   vertical=\(cV)  horizontal=\(cH)  ratio V/H=\(cV / max(cH, 1e-6))")

        // A working tower treats the two axes alike. Muse favouring one axis by
        // far more than the control does is the tower-level form of the live
        // finding that left/right is unreadable while top/bottom is fine.
        let museRatio = mV / max(mH, 1e-6)
        let qwenRatio = cV / max(cH, 1e-6)
        #expect(museRatio < qwenRatio * 3,
            "Muse separates the vertical split \(museRatio)x better than the horizontal one, against \(qwenRatio)x for the control — horizontal position is being lost inside the vision tower")
    }
}
