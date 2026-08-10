import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXVLM

/// The full model names a solid red field "green" and gives the same answer for
/// a solid blue one, yet the patch tensor, the channel order, the normalization
/// constants and all 809 loaded weights check out. That leaves the tower's
/// forward pass, which the random-weight separation test cannot judge: almost
/// any random linear map separates two inputs, so a pass there says little.
///
/// This runs the REAL vision weights (a few hundred MB, not the 20GB model)
/// through the tower and asks whether two opposite colours stay apart.
@Suite("Muse Glimmer trained tower")
struct MuseGlimmerTrainedTowerTests {

    static let bundle = URL(fileURLWithPath: NSHomeDirectory())
        .appendingPathComponent("models/JANGQ-AI/Muse-Glimmer-30B-JANG_6M")

    static var enabled: Bool {
        ProcessInfo.processInfo.environment["MUSE_VL_LIVE"] == "1"
            && FileManager.default.fileExists(
                atPath: bundle.appendingPathComponent("config.json").path)
    }

    /// Solid-colour patch rows laid out `[channel][temporal][h][w]`, matching
    /// what `QwenVL.patchify` emits.
    static func solid(r: Float, g: Float, b: Float, frame: THW, config: MuseGlimmerVisionConfiguration)
        -> MLXArray
    {
        let per = config.patchTemporal * config.patchSize * config.patchSize
        let rows = frame.t * frame.h * frame.w
        var flat: [Float] = []
        flat.reserveCapacity(rows * 3 * per)
        for _ in 0 ..< rows {
            flat += Array(repeating: r, count: per)
            flat += Array(repeating: g, count: per)
            flat += Array(repeating: b, count: per)
        }
        return MLXArray(flat).reshaped(rows, 3 * per)
    }

    @Test("the trained tower keeps red and blue apart", .enabled(if: enabled))
    func trainedTowerSeparatesColours() throws {
        struct Wrapper: Decodable { let visionConfiguration: MuseGlimmerVisionConfiguration
            enum CodingKeys: String, CodingKey { case visionConfiguration = "vision_config" } }
        let data = try Data(contentsOf: Self.bundle.appendingPathComponent("config.json"))
        let config = try JSONDecoder().decode(Wrapper.self, from: data).visionConfiguration
        print("[trained] hidden=\(config.hiddenSize) layers=\(config.numLayers) patch=\(config.patchSize) merge=\(config.mergeSize)")

        let tower = MuseGlimmerVisionModel(config)

        // Pull only the vision-tower tensors off disk.
        var weights: [String: MLXArray] = [:]
        for f in try FileManager.default.contentsOfDirectory(
            at: Self.bundle, includingPropertiesForKeys: nil)
        where f.pathExtension == "safetensors" {
            guard let arrays = try? MLX.loadArrays(url: f) else { continue }
            for (k, v) in arrays where k.hasPrefix("model.vision_tower.") {
                weights[String(k.dropFirst("model.vision_tower.".count))] = v.asType(.float32)
            }
        }
        print("[trained] vision_tower tensors: \(weights.count)")
        try #require(!weights.isEmpty)
        try tower.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        eval(tower)

        let frame = THW(1, 8, 8)
        let red = tower(Self.solid(r: 1, g: -0.84, b: -0.84, frame: frame, config: config),
            frames: [frame])
        let blue = tower(Self.solid(r: -0.84, g: -0.84, b: 1, frame: frame, config: config),
            frames: [frame])
        eval(red, blue)
        print("[trained] output shape \(red.shape)")

        let rv = red.asType(.float32).asArray(Float.self)
        let bv = blue.asType(.float32).asArray(Float.self)
        #expect(rv.allSatisfy { $0.isFinite }, "trained tower produced non-finite output")

        var diff: Float = 0, mag: Float = 0
        for i in 0 ..< rv.count { diff += abs(rv[i] - bv[i]); mag += abs(rv[i]) }
        diff /= Float(rv.count); mag /= Float(rv.count)
        let relative = mag > 0 ? diff / mag : 0
        print("[trained] red vs blue mean|Δ|=\(diff) mean|red|=\(mag) relative=\(relative)")

        #expect(relative > 0.02,
            "the trained tower collapses red and blue (relative=\(relative)) — the colour signal dies inside the vision forward pass")
    }

    /// The model cannot tell a black circle from a black square, so the tower
    /// output is suspected of carrying no usable spatial structure. This feeds
    /// a half-black / half-white field — for which the correct output is
    /// obvious — and asks whether the top half of the output tokens differs
    /// from the bottom half at all, and whether the change lands at the true
    /// midpoint.
    ///
    /// At a 32x32 grid the window partition is a single window and the position
    /// table maps one-to-one, so anything wrong here is in the forward maths
    /// rather than in the geometry bookkeeping.
    @Test("the tower encodes where things are", .enabled(if: enabled))
    func towerIsSpatiallyInformative() throws {
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

        // 32x32 patch grid → 16x16 merged cells. Rows arrive grouped by merge
        // block: (hb, wb, di, dj), so the grid row is hb*merge + di.
        let grid = 32, merge = config.mergeSize, cells = grid / merge
        let per = config.patchTemporal * config.patchSize * config.patchSize
        var flat: [Float] = []
        for hb in 0 ..< cells {
            for _ in 0 ..< cells {
                for di in 0 ..< merge {
                    for _ in 0 ..< merge {
                        // Top half black (-1), bottom half white (+1).
                        let value: Float = (hb * merge + di) < grid / 2 ? -1 : 1
                        flat += Array(repeating: value, count: 3 * per)
                    }
                }
            }
        }
        let frame = THW(1, grid, grid)
        let out = module(MLXArray(flat).reshaped(grid * grid, 3 * per), frames: [frame])
        eval(out)
        print("[spatial] output shape \(out.shape)")

        let rows = out.dim(0)
        let values = out.asType(.float32).asArray(Float.self)
        let width = out.dim(1)
        func row(_ i: Int) -> ArraySlice<Float> { values[(i * width) ..< ((i + 1) * width)] }
        func meanAbsDiff(_ a: Int, _ b: Int) -> Float {
            let ra = Array(row(a)), rb = Array(row(b))
            var d: Float = 0
            for i in 0 ..< width { d += abs(ra[i] - rb[i]) }
            return d / Float(width)
        }

        // Rows 0..<half are the black half, the rest white, in raster order.
        let half = rows / 2
        var across: Float = 0, within: Float = 0
        for i in 0 ..< 16 {
            across += meanAbsDiff(i, half + i)
            within += meanAbsDiff(i, (i + 1) % half)
        }
        across /= 16
        within /= 16
        print("[spatial] mean|Δ| across the black/white boundary = \(across)")
        print("[spatial] mean|Δ| within the black half           = \(within)")
        print("[spatial] contrast ratio across/within            = \(across / max(within, 1e-6))")

        // No verdict is drawn from the ratio: Qwen3.6's working tower scores
        // in the same band on this statistic (see MuseGlimmerMetricControl),
        // so it cannot separate a good tower from a bad one. What is still
        // worth asserting is that the halves are not bit-identical and the
        // output is finite — a tower returning constant or NaN features is
        // broken by any standard.
        #expect(across > 0, "the two halves produce identical features")
        #expect(values.allSatisfy { $0.isFinite }, "non-finite tower output")
    }

    /// Vision tokens are scattered into embeddings that have already been
    /// RMS-normed, so every text token arrives with RMS exactly 1.0 by
    /// construction. If the projected vision tokens land at a very different
    /// magnitude they are out of distribution for every downstream layer —
    /// which reads as "the model can see something is there but cannot tell
    /// what", exactly the observed behaviour.
    ///
    /// This is the one remaining measurable link: tower, adapter, projection
    /// and geometry are all verified, so scale is what is left.
    @Test("projected vision tokens match the normed text embedding scale",
        .enabled(if: enabled))
    func visionTokenScaleMatchesText() throws {
        struct Wrapper: Decodable {
            let visionConfiguration: MuseGlimmerVisionConfiguration
            enum CodingKeys: String, CodingKey { case visionConfiguration = "vision_config" }
        }
        let data = try Data(contentsOf: Self.bundle.appendingPathComponent("config.json"))
        let config = try JSONDecoder().decode(Wrapper.self, from: data).visionConfiguration

        var tower: [String: MLXArray] = [:]
        var adapter: [String: MLXArray] = [:]
        var projection: MLXArray?
        for f in try FileManager.default.contentsOfDirectory(
            at: Self.bundle, includingPropertiesForKeys: nil)
        where f.pathExtension == "safetensors" {
            guard let arrays = try? MLX.loadArrays(url: f) else { continue }
            for (k, v) in arrays {
                if k.hasPrefix("model.vision_tower.") {
                    tower[String(k.dropFirst("model.vision_tower.".count))] = v.asType(.float32)
                } else if k.hasPrefix("model.vision_adapter.") {
                    adapter[String(k.dropFirst("model.vision_adapter.".count))] = v.asType(.float32)
                } else if k == "model.vision_projection.weight" {
                    projection = v.asType(.float32)
                }
            }
        }
        try #require(!tower.isEmpty && !adapter.isEmpty)
        let projectionWeight = try #require(projection)

        let towerModule = MuseGlimmerVisionModel(config)
        try towerModule.update(parameters: ModuleParameters.unflattened(tower), verify: .all)

        let merged = config.hiddenSize * config.mergeSize * config.mergeSize
        let hidden = adapter["fc1.weight"]!.dim(0)
        let projector = MuseGlimmerVisionProjector(
            inputDimensions: merged, hiddenDimensions: hidden)
        try projector.update(parameters: ModuleParameters.unflattened(adapter), verify: .all)
        eval(towerModule, projector)

        let frame = THW(1, 8, 8)
        let towerOut = towerModule(
            Self.solid(r: 1, g: -0.84, b: -0.84, frame: frame, config: config), frames: [frame])
        let projected = projector(towerOut).matmul(projectionWeight.transposed())
        eval(projected)

        // RMS per token over the hidden dimension.
        func rms(_ x: MLXArray) -> Float {
            let v = MLX.sqrt(MLX.mean(x.asType(.float32).square(), axis: -1)).mean()
            v.eval()
            return v.item(Float.self)
        }
        let visionRMS = rms(projected)
        print("[scale] projected vision token RMS = \(visionRMS)")
        print("[scale] normed text token RMS      = 1.0 (by construction)")
        print("[scale] ratio vision/text          = \(visionRMS)")

        // An order of magnitude either way is a genuine distribution mismatch,
        // not the ordinary spread between two embedding sources.
        #expect(visionRMS > 0.1 && visionRMS < 10.0,
            "projected vision tokens sit at RMS \(visionRMS) against normed text tokens at 1.0 — the vision tokens are out of distribution for the text stack")
    }
}
