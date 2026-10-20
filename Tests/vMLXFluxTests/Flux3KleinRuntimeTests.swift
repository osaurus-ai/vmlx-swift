import Foundation
import XCTest
@preconcurrency import MLX
@testable import vMLXFluxKit
@testable import vMLXFluxModels

/// Explicit local GPU qualification. Run each mode in a fresh process to release
/// model weights and Metal allocations between generation/edit/reference rows.
final class Flux3KleinRuntimeTests: XCTestCase {
    func testNativeParity() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let path = env["FLUX3_RUNTIME_BUNDLE"], let output = env["FLUX3_RUNTIME_OUT"] else {
            throw XCTSkip("Set FLUX3_RUNTIME_BUNDLE and FLUX3_RUNTIME_OUT for an authorized GPU run")
        }
        let root = URL(fileURLWithPath: path)
        let out = URL(fileURLWithPath: output)
        try FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)
        let bundle = try Flux3Bundle(modelPath: root)
        XCTAssertEqual(bundle.stamp.quantization.bits, 4, "Small qualification uses the 4-bit bundle")
        let sources = (env["FLUX3_RUNTIME_REFERENCES"] ?? "").split(separator: "\n").map { URL(fileURLWithPath: String($0)) }
        let width = env["FLUX3_RUNTIME_WIDTH"].flatMap(Int.init)
        let height = env["FLUX3_RUNTIME_HEIGHT"].flatMap(Int.init)
        let steps = env["FLUX3_RUNTIME_STEPS"].flatMap(Int.init) ?? 4
        let seed = env["FLUX3_RUNTIME_SEED"].flatMap(UInt64.init) ?? 42
        let prompt = env["FLUX3_RUNTIME_PROMPT"] ?? "A red ceramic teapot on a wooden table, soft morning window light, detailed photograph"
        let started = Date()
        let pipeline = try await Flux3Pipeline(modelPath: root, bundle: bundle)
        let loadSeconds = Date().timeIntervalSince(started)
        let operationStart = Date()
        let references = sources.isEmpty ? nil : try pipeline.references(sources, width: width, height: height)
        let w = references?.width ?? width ?? 512, h = references?.height ?? height ?? 512
        try bundle.limits.validateEditOrGenerateForQualification(width: w, height: h, steps: steps, sources: sources.count, output: out)
        let (image, effective) = try pipeline.sample(prompt: prompt, width: w, height: h, steps: steps,
            seed: seed, references: references, preview: nil, observe: { name, value in
                try MLX.save(arrays: [name: value], url: out.appendingPathComponent("swift-\(name).safetensors"))
            }, progress: { step, total, _ in print("[Flux3Parity] step=\(step)/\(total)") })
        let url = try vMLXFluxKit.ImageIO.writePNG(image, outputDir: out, prefix: "swift")
        let seconds = Date().timeIntervalSince(operationStart)
        let pixels = image.asArray(Float.self)
        XCTAssertTrue(pixels.allSatisfy(\.isFinite))
        XCTAssertGreaterThan((pixels.max() ?? 0) - (pixels.min() ?? 0), 0.05, "Reject a flat decoded image")
        let record: [String: Any] = ["bundle": path, "prompt": prompt, "seed": seed, "steps": steps,
            "width": effective.width, "height": effective.height, "references": sources.map(\.path),
            "load_seconds": loadSeconds, "operation_seconds": seconds, "images_per_second": 1 / seconds,
            "png": url.path, "active_bytes": Memory.activeMemory, "peak_bytes": Memory.peakMemory]
        try JSONSerialization.data(withJSONObject: record, options: [.prettyPrinted, .sortedKeys])
            .write(to: out.appendingPathComponent("swift-result.json"))
        // The test process exits after this mode. Cached Metal allocations are also released now.
        Memory.clearCache()
    }
}

private extension FluxModelLimits {
    func validateEditOrGenerateForQualification(width: Int, height: Int, steps: Int, sources: Int, output: URL) throws {
        if sources == 0 {
            try validate(ImageGenRequest(prompt: "qualification", width: width, height: height, steps: steps,
                guidance: 1, outputDir: output))
        } else {
            try validateEdit(width: width, height: height, steps: steps, guidance: 1, format: .png,
                sourceCount: sources, hasMask: false, strength: nil)
        }
    }
}
