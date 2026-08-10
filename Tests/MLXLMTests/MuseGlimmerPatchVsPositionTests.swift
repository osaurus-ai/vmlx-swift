import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import MLXVLM

/// The trained tower separates two whole-image colours but does not separate a
/// black half from a white half — content is being swamped by something. The
/// patch embedder adds exactly two things together:
///
///     embeddings = patch_embedding(patches) + position
///
/// so if one term dwarfs the other, that sum is where the picture is lost.
@Suite("Muse Glimmer patch vs position")
struct MuseGlimmerPatchVsPositionTests {

    static let bundle = URL(fileURLWithPath: NSHomeDirectory())
        .appendingPathComponent("models/JANGQ-AI/Muse-Glimmer-30B-JANG_6M")

    static var enabled: Bool {
        ProcessInfo.processInfo.environment["MUSE_VL_LIVE"] == "1"
            && FileManager.default.fileExists(
                atPath: bundle.appendingPathComponent("config.json").path)
    }

    @Test("patch content is not swamped by the position table", .enabled(if: enabled))
    func relativeMagnitudes() throws {
        var patchWeight: MLXArray?
        var positionTable: MLXArray?
        for f in try FileManager.default.contentsOfDirectory(
            at: Self.bundle, includingPropertiesForKeys: nil)
        where f.pathExtension == "safetensors" {
            guard let arrays = try? MLX.loadArrays(url: f) else { continue }
            for (k, v) in arrays {
                if k.hasSuffix("patch_embedder.patch_embedding.weight") {
                    patchWeight = v.asType(.float32)
                } else if k.hasSuffix("patch_embedder.position_embedding_table.weight") {
                    positionTable = v.asType(.float32)
                }
            }
        }
        let pw = try #require(patchWeight)
        let pt = try #require(positionTable)
        print("[mag] patch_embedding.weight \(pw.shape)  position_table \(pt.shape)")

        func rms(_ x: MLXArray) -> Float {
            let v = MLX.sqrt(MLX.mean(x.square()))
            v.eval()
            return v.item(Float.self)
        }

        let per = 2 * 14 * 14
        for (label, value) in [("black", Float(-1)), ("white", Float(1))] {
            let patch = MLXArray.full([3 * per], values: MLXArray(value))
            let embedded = pw.matmul(patch)
            print("[mag] patch embedding RMS (\(label)) = \(rms(embedded))")
        }
        let posRMS = rms(pt)
        print("[mag] position table RMS = \(posRMS)")

        let black = pw.matmul(MLXArray.full([3 * per], values: MLXArray(Float(-1))))
        let white = pw.matmul(MLXArray.full([3 * per], values: MLXArray(Float(1))))
        let contentGap = rms(black - white)
        print("[mag] |black − white| content gap RMS = \(contentGap)")
        print("[mag] content gap ÷ position RMS      = \(contentGap / posRMS)")

        // If a full black-to-white swing moves the embedding far less than the
        // position term already varies, position dominates the sum and the
        // picture cannot survive it.
        // Control: the same two magnitudes from Qwen3.6, whose tower resolves
        // both axes. Horizontal position can only reach the language model
        // through this embedding — vertical survives without it, because token
        // order already encodes rows — so its scale relative to content is the
        // thing to compare.
        let qwen = URL(fileURLWithPath: NSHomeDirectory())
            .appendingPathComponent("models/dealign.ai/Qwen3.6-27B-JANG_4M-CRACK")
        if FileManager.default.fileExists(atPath: qwen.appendingPathComponent("config.json").path) {
            var qPatch: MLXArray?
            var qPos: MLXArray?
            for f in try FileManager.default.contentsOfDirectory(
                at: qwen, includingPropertiesForKeys: nil)
            where f.pathExtension == "safetensors" {
                guard let arrays = try? MLX.loadArrays(url: f) else { continue }
                for (k, v) in arrays {
                    if k.hasSuffix("vision_tower.patch_embed.proj.weight") { qPatch = v.asType(.float32) }
                    if k.hasSuffix("vision_tower.pos_embed.weight") { qPos = v.asType(.float32) }
                }
            }
            if let qPatch, let qPos {
                let flatPatch = qPatch.reshaped(qPatch.dim(0), -1)
                let ones = MLXArray.full([flatPatch.dim(1)], values: MLXArray(Float(1)))
                let qGap = rms(flatPatch.matmul(ones)) * 2
                let qPosRMS = rms(qPos)
                print("[mag] CONTROL Qwen content gap=\(qGap) position RMS=\(qPosRMS) ratio=\(qGap / qPosRMS)")
                print("[mag] Muse ratio for comparison = \(contentGap / posRMS)")
            }
        }

        #expect(contentGap > posRMS * 0.1,
            "a black→white swing (\(contentGap)) is tiny next to the position table (\(posRMS)) — content is swamped in patch_embedding + position")
    }
}
