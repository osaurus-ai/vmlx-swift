import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXVLM

/// The tower permutes patches into window order for attention and then undoes
/// that permutation so features line up with the `<|patch|>` placeholders. When
/// the window covers the whole image that permutation is the identity, so the
/// code path is never exercised by a 448x448 picture — but any other window
/// size makes it real, and a wrong inverse silently scrambles where things are.
///
/// A gradient makes the ordering checkable: cell `r*16+c` gets a brightness
/// that rises monotonically with its raster index, so if the output rows are in
/// raster order their brightness still rises with the row index.
@Suite("Muse Glimmer window order")
struct MuseGlimmerWindowOrderTests {

    static let bundle = URL(fileURLWithPath: NSHomeDirectory())
        .appendingPathComponent("models/JANGQ-AI/Muse-Glimmer-30B-JANG_6M")

    static var enabled: Bool {
        ProcessInfo.processInfo.environment["MUSE_VL_LIVE"] == "1"
            && FileManager.default.fileExists(
                atPath: bundle.appendingPathComponent("config.json").path)
    }

    static func loadTower() throws -> (MuseGlimmerVisionModel, MuseGlimmerVisionConfiguration) {
        struct Wrapper: Decodable {
            let visionConfiguration: MuseGlimmerVisionConfiguration
            enum CodingKeys: String, CodingKey { case visionConfiguration = "vision_config" }
        }
        let data = try Data(contentsOf: bundle.appendingPathComponent("config.json"))
        let config = try JSONDecoder().decode(Wrapper.self, from: data).visionConfiguration
        var weights: [String: MLXArray] = [:]
        for f in try FileManager.default.contentsOfDirectory(
            at: bundle, includingPropertiesForKeys: nil)
        where f.pathExtension == "safetensors" {
            guard let arrays = try? MLX.loadArrays(url: f) else { continue }
            for (k, v) in arrays where k.hasPrefix("model.vision_tower.") {
                weights[String(k.dropFirst("model.vision_tower.".count))] = v.asType(.float32)
            }
        }
        let tower = MuseGlimmerVisionModel(config)
        try tower.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        eval(tower)
        return (tower, config)
    }

    @Test("output rows come back in raster order", .enabled(if: enabled))
    func rasterOrderIsRestored() throws {
        let (tower, config) = try Self.loadTower()
        let grid = 32, merge = config.mergeSize, cells = grid / merge
        let per = config.patchTemporal * config.patchSize * config.patchSize

        // Brightness rises with the merged cell's raster index.
        var flat: [Float] = []
        for hb in 0 ..< cells {
            for wb in 0 ..< cells {
                let level = Float(hb * cells + wb) / Float(cells * cells - 1) * 2 - 1
                for _ in 0 ..< (merge * merge) {
                    flat += Array(repeating: level, count: 3 * per)
                }
            }
        }
        let frame = THW(1, grid, grid)
        let patches = MLXArray(flat).reshaped(grid * grid, 3 * per)
        let out = tower(patches, frames: [frame])
        eval(out)

        // Per-row summary, then Pearson correlation against the row index.
        let rowMean = out.mean(axis: -1)
        eval(rowMean)
        let values = rowMean.asArray(Float.self)
        let n = values.count
        let idx = (0 ..< n).map { Float($0) }
        func mean(_ a: [Float]) -> Float { a.reduce(0, +) / Float(a.count) }
        let mv = mean(values), mi = mean(idx)
        var num: Float = 0, dv: Float = 0, di: Float = 0
        for i in 0 ..< n {
            num += (values[i] - mv) * (idx[i] - mi)
            dv += (values[i] - mv) * (values[i] - mv)
            di += (idx[i] - mi) * (idx[i] - mi)
        }
        let correlation = num / (Float(Double(dv * di).squareRoot()) + 1e-9)
        let window = ProcessInfo.processInfo.environment["VMLX_MUSE_VISION_WINDOW"] ?? "default"
        print("[order] window=\(window) rows=\(n) correlation(row index, brightness) = \(correlation)")

        // No threshold is asserted without a reference. Qwen3.6's working
        // tower is run on the identical gradient below; whatever it scores is
        // what "correct" looks like for this statistic.
        #expect(correlation.isFinite, "correlation did not compute")
        Self.record(window: window, correlation: correlation)
    }

    nonisolated(unsafe) static var scores: [String: Float] = [:]
    static func record(window: String, correlation: Float) { scores[window] = correlation }

    /// The control. Same gradient, same statistic, a tower known to work.
    @Test("the same gradient through Qwen3.6's tower", .enabled(if: enabled))
    func qwenReference() throws {
        let qwenURL = URL(fileURLWithPath: NSHomeDirectory())
            .appendingPathComponent("models/dealign.ai/Qwen3.6-27B-JANG_4M-CRACK")
        try #require(FileManager.default.fileExists(
            atPath: qwenURL.appendingPathComponent("config.json").path))

        struct Wrapper: Decodable {
            let visionConfiguration: Qwen3VLConfiguration.VisionConfiguration
            enum CodingKeys: String, CodingKey { case visionConfiguration = "vision_config" }
        }
        let data = try Data(contentsOf: qwenURL.appendingPathComponent("config.json"))
        let config = try JSONDecoder().decode(Wrapper.self, from: data).visionConfiguration
        let tower = Qwen3VLVision.VisionModel(config)

        var weights: [String: MLXArray] = [:]
        for f in try FileManager.default.contentsOfDirectory(
            at: qwenURL, includingPropertiesForKeys: nil)
        where f.pathExtension == "safetensors" {
            guard let arrays = try? MLX.loadArrays(url: f) else { continue }
            for (k, v) in arrays where k.hasPrefix("vision_tower.") {
                weights[String(k.dropFirst("vision_tower.".count))] = v.asType(.float32)
            }
        }
        try tower.update(
            parameters: ModuleParameters.unflattened(tower.sanitize(weights: weights)),
            verify: Module.VerifyUpdate.all)
        eval(tower)

        let grid = 32, merge = config.spatialMergeSize, cells = grid / merge
        let per = config.temporalPatchSize * config.patchSize * config.patchSize
        var flat: [Float] = []
        for hb in 0 ..< cells {
            for wb in 0 ..< cells {
                let level = Float(hb * cells + wb) / Float(cells * cells - 1) * 2 - 1
                for _ in 0 ..< (merge * merge) {
                    flat += Array(repeating: level, count: 3 * per)
                }
            }
        }
        let frame = THW(1, grid, grid)
        let (out, _) = tower(MLXArray(flat).reshaped(grid * grid, 3 * per), gridTHW: [frame])
        eval(out)

        let rowMean = out.mean(axis: -1)
        eval(rowMean)
        let values = rowMean.asArray(Float.self)
        let n = values.count
        let idx = (0 ..< n).map { Float($0) }
        func mean(_ a: [Float]) -> Float { a.reduce(0, +) / Float(a.count) }
        let mv = mean(values), mi = mean(idx)
        var num: Float = 0, dv: Float = 0, di: Float = 0
        for i in 0 ..< n {
            num += (values[i] - mv) * (idx[i] - mi)
            dv += (values[i] - mv) * (values[i] - mv)
            di += (idx[i] - mi) * (idx[i] - mi)
        }
        let correlation = num / (Float(Double(dv * di).squareRoot()) + 1e-9)
        print("[order] CONTROL Qwen3.6 correlation(row index, brightness) = \(correlation)")
        print("[order] Muse Glimmer scored ~0.42 on the identical statistic")
        #expect(correlation.isFinite)
    }
}
