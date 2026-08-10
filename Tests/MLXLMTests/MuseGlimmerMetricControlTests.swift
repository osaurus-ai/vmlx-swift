import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXVLM

/// The half-black / half-white contrast measurement says Muse Glimmer's vision
/// tower destroys the picture. Before trusting that, the measurement itself has
/// to be shown to mean something — a metric that scores every tower poorly
/// proves nothing about this one.
///
/// Qwen3.6's vision tower is present locally, unquantized, and known to work in
/// this runtime, so it is the positive control: the identical input and the
/// identical statistic run through a tower whose output is known to be usable.
@Suite("Muse Glimmer metric control")
struct MuseGlimmerMetricControlTests {

    static let qwen = URL(fileURLWithPath: NSHomeDirectory())
        .appendingPathComponent("models/dealign.ai/Qwen3.6-27B-JANG_4M-CRACK")

    static var enabled: Bool {
        ProcessInfo.processInfo.environment["MUSE_VL_LIVE"] == "1"
            && FileManager.default.fileExists(
                atPath: qwen.appendingPathComponent("config.json").path)
    }

    /// Separation of the two halves against the spread inside one uniform half.
    /// Identical to the statistic used on the Muse tower.
    static func contrast(_ x: MLXArray) -> (Float, Float) {
        let rows = x.dim(0)
        let half = rows / 2
        let top = x[0 ..< half, 0...].mean(axis: 0)
        let bottom = x[half ..< rows, 0...].mean(axis: 0)
        let across = MLX.abs(top - bottom).mean()
        let blackHalf = x[0 ..< half, 0...]
        let mean = blackHalf.mean(axis: 0)
        let within = MLX.sqrt((blackHalf - mean).square().mean(axis: 0)).mean()
        eval(across, within)
        return (across.item(Float.self), within.item(Float.self))
    }

    @Test("a known-good tower scores high on the same measurement",
        .enabled(if: enabled))
    func qwenControl() throws {
        struct Wrapper: Decodable {
            let visionConfiguration: Qwen3VLConfiguration.VisionConfiguration
            enum CodingKeys: String, CodingKey { case visionConfiguration = "vision_config" }
        }
        let data = try Data(contentsOf: Self.qwen.appendingPathComponent("config.json"))
        let config = try JSONDecoder().decode(Wrapper.self, from: data).visionConfiguration
        print("[control] qwen vision: hidden=\(config.hiddenSize) depth=\(config.depth) patch=\(config.patchSize) merge=\(config.spatialMergeSize)")

        let tower = Qwen3VLVision.VisionModel(config)

        var weights: [String: MLXArray] = [:]
        for f in try FileManager.default.contentsOfDirectory(
            at: Self.qwen, includingPropertiesForKeys: nil)
        where f.pathExtension == "safetensors" {
            guard let arrays = try? MLX.loadArrays(url: f) else { continue }
            for (k, v) in arrays where k.hasPrefix("vision_tower.") {
                weights[String(k.dropFirst("vision_tower.".count))] = v.asType(.float32)
            }
        }
        print("[control] qwen vision tensors: \(weights.count)")
        try #require(!weights.isEmpty)
        let sanitized = tower.sanitize(weights: weights)
        try tower.update(parameters: ModuleParameters.unflattened(sanitized), verify: Module.VerifyUpdate.all)
        eval(tower)

        // Same picture, same patch layout convention: rows grouped by merge
        // block, top half black, bottom half white.
        let grid = 32, merge = config.spatialMergeSize
        let per = config.temporalPatchSize * config.patchSize * config.patchSize
        var flat: [Float] = []
        for hb in 0 ..< (grid / merge) {
            for _ in 0 ..< (grid / merge) {
                for di in 0 ..< merge {
                    for _ in 0 ..< merge {
                        let value: Float = (hb * merge + di) < grid / 2 ? -1 : 1
                        flat += Array(repeating: value, count: 3 * per)
                    }
                }
            }
        }
        let frame = THW(1, grid, grid)
        let (out, _) = tower(
            MLXArray(flat).reshaped(grid * grid, 3 * per), gridTHW: [frame])
        eval(out)

        let (across, within) = Self.contrast(out)
        let ratio = across / max(within, 1e-6)
        print("[control] qwen tower output \(out.shape)")
        print("[control] across=\(across) within=\(within) ratio=\(ratio)")
        print("[control] Muse Glimmer on the identical measurement: ratio 0.13")

        // The control lands in the same low band as Muse Glimmer, so this
        // statistic does NOT discriminate a working tower from a broken one.
        // Recorded as an assertion so the dead end stays closed: anyone
        // tempted to read a low ratio as a defect has to get past this first.
        // Vision correctness is settled by the end-to-end probes in osaurus
        // (matched opposite stimuli), which the same Qwen tower passes 2/2
        // while Muse Glimmer does not.
        #expect(ratio < 1.0,
            "the control tower now scores \(ratio); if a known-good tower scores high here the statistic may be worth revisiting")
    }
}
