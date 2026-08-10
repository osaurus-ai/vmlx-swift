import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXVLM

/// The tower's output does not distinguish a black half from a white half, yet
/// content enters the patch embedder ~56x stronger than the position term. So
/// the picture is destroyed somewhere in the 50 blocks. This walks the stack
/// and reports, at every stage, how strongly the two halves are still
/// separated — the layer where that collapses is the layer with the bug.
@Suite("Muse Glimmer layer bisect")
struct MuseGlimmerLayerBisectTests {

    static let bundle = URL(fileURLWithPath: NSHomeDirectory())
        .appendingPathComponent("models/JANGQ-AI/Muse-Glimmer-30B-JANG_6M")

    static var enabled: Bool {
        ProcessInfo.processInfo.environment["MUSE_VL_LIVE"] == "1"
            && FileManager.default.fileExists(
                atPath: bundle.appendingPathComponent("config.json").path)
    }

    @Test("locate the layer where the picture dies", .enabled(if: enabled))
    func bisect() throws {
        struct Wrapper: Decodable {
            let visionConfiguration: MuseGlimmerVisionConfiguration
            enum CodingKeys: String, CodingKey { case visionConfiguration = "vision_config" }
        }
        let data = try Data(contentsOf: Self.bundle.appendingPathComponent("config.json"))
        let config = try JSONDecoder().decode(Wrapper.self, from: data).visionConfiguration

        var tower: [String: MLXArray] = [:]
        for f in try FileManager.default.contentsOfDirectory(
            at: Self.bundle, includingPropertiesForKeys: nil)
        where f.pathExtension == "safetensors" {
            guard let arrays = try? MLX.loadArrays(url: f) else { continue }
            for (k, v) in arrays where k.hasPrefix("model.vision_tower.") {
                tower[String(k.dropFirst("model.vision_tower.".count))] = v.asType(.float32)
            }
        }
        let module = MuseGlimmerVisionModel(config)
        try module.update(parameters: ModuleParameters.unflattened(tower), verify: .all)
        eval(module)

        let grid = 32, merge = config.mergeSize
        let per = config.patchTemporal * config.patchSize * config.patchSize
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
        let patches = MLXArray(flat).reshaped(grid * grid, 3 * per)

        // At every stage the rows are still one-per-patch in merge-block order,
        // so the first half of the rows is the black half of the picture.
        func contrast(_ x: MLXArray) -> (Float, Float) {
            let rows = x.dim(0), width = x.dim(1)
            let half = rows / 2
            let top = x[0 ..< half, 0...].mean(axis: 0)
            let bottom = x[half ..< rows, 0...].mean(axis: 0)
            let across = MLX.abs(top - bottom).mean()
            // Spread inside the black half: what "different" looks like when
            // the content is identical.
            let blackHalf = x[0 ..< half, 0...]
            let mean = blackHalf.mean(axis: 0)
            let within = MLX.sqrt((blackHalf - mean).square().mean(axis: 0)).mean()
            eval(across, within)
            _ = width
            return (across.item(Float.self), within.item(Float.self))
        }

        var lines: [String] = []
        var firstCollapse: String?
        _ = module(patches, frames: [frame]) { name, h in
            let (across, within) = contrast(h)
            let ratio = across / max(within, 1e-6)
            if name == "embedder" || name == "ln_pre" || name.hasSuffix("0")
                || name.hasSuffix("5") || name == "layer49"
            {
                lines.append("[bisect] \(name): across=\(across) within=\(within) ratio=\(ratio)")
            }
            if ratio < 0.5, firstCollapse == nil, name.hasPrefix("layer") {
                firstCollapse = "\(name) (ratio \(ratio))"
            }
        }
        for l in lines { print(l) }
        print("[bisect] first stage with ratio < 0.5: \(firstCollapse ?? "none")")
        // Diagnostic only. The ratio falling through the stack is NOT evidence
        // of a defect — Qwen3.6's working tower ends in the same band (see
        // MuseGlimmerMetricControl). This records the per-layer profile so a
        // future fix can be compared against it; the assertion is limited to
        // what is actually defensible.
        #expect(!lines.isEmpty, "the probe never fired")
    }
}
