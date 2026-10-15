import Foundation
@preconcurrency import MLX
import vMLXFlux

/// Measures request workspace, not image quality or application performance.
/// Uses a fresh process per bundle; callers select only missing calibration sizes.
func runImageMemoryCheck(_ arguments: [String], estimateOnly: Bool = false) async throws {
    guard arguments.count >= 4 else {
        throw FluxError.invalidRequest("memory-check MODEL_DIRECTORY FAMILY OUTPUT_DIRECTORY SIZE[,SIZE] [EDIT_SOURCE]")
    }
    let model = URL(fileURLWithPath: arguments[0])
    let family = arguments[1]
    let output = URL(fileURLWithPath: arguments[2])
    let sizes = try arguments[3].split(separator: ",").map { value -> Int in
        guard let size = Int(value), size > 0 else { throw FluxError.invalidRequest("invalid calibration size") }
        return size
    }
    let engine = FluxEngine()
    let limits = try await engine.limits(for: model, name: family)
    if !estimateOnly { try await engine.load(name: family, modelPath: model) }
    for size in sizes {
        Memory.clearCache()
        Memory.peakMemory = 0
        let start = Date()
        let estimate: Int
        let events: AsyncThrowingStream<ImageGenEvent, Error>?
        if arguments.count > 4 {
            let request = ImageEditRequest(prompt: "Make the teapot blue.",
                sourceImage: URL(fileURLWithPath: arguments[4]), width: size, height: size,
                steps: limits.fixedSteps ?? 2, guidance: 4, seed: 7, outputDir: output)
            estimate = try await engine.estimatedPeakBytes(for: request, modelURL: model, name: family)
            if estimateOnly { events = nil } else { events = await engine.edit(request) }
        } else {
            let request = ImageGenRequest(prompt: "A red ceramic teapot on a wooden table.",
                width: size, height: size, steps: limits.fixedSteps ?? 2,
                guidance: family == "qwen-image-2.1-turbo" ? 1 : 4, seed: 7, outputDir: output)
            estimate = try await engine.estimatedPeakBytes(for: request, modelURL: model, name: family)
            if estimateOnly { events = nil } else { events = await engine.generate(request) }
        }
        guard let events else {
            print("ESTIMATE family=\(family) size=\(size) estimate=\(estimate)")
            continue
        }
        var completed = false
        for try await event in events {
            switch event {
            case .completed(let url, _, _):
                completed = true
                print("MEMORY family=\(family) size=\(size) estimate=\(estimate) peak=\(Memory.peakMemory) active=\(Memory.activeMemory) relativeError=\(Double(estimate) / Double(max(1, Memory.peakMemory)) - 1) elapsed=\(Date().timeIntervalSince(start)) path=\(url.path)")
                fflush(stdout)
            case .failed(let message, _): throw FluxError.invalidRequest(message)
            case .cancelled: throw CancellationError()
            default: break
            }
        }
        guard completed else { throw FluxError.invalidRequest("calibration produced no completion") }
    }
    await engine.unload()
}
