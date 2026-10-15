import Foundation
@preconcurrency import MLX
import vMLXFlux

/// Exercise each advertised Qwen 2.1 sampler against its real non-Turbo bundle.
func runSamplingCheck(_ arguments: [String]) async throws {
    guard arguments.count == 2 else {
        throw FluxError.invalidRequest("sampling-check MODEL_DIRECTORY OUTPUT_DIRECTORY")
    }
    let model = URL(fileURLWithPath: arguments[0]), output = URL(fileURLWithPath: arguments[1])
    VMLXFluxModels.registerAll()
    let engine = FluxEngine()
    let limits = try await engine.limits(for: model, name: "qwen-image-2.1")
    try await engine.load(name: "qwen-image-2.1", modelPath: model)
    var images: [String: Data] = [:]
    for sampler in limits.samplers {
        let request = ImageGenRequest(prompt: "A red ceramic teapot on a wooden table, natural daylight, photographic.",
            width: 512, height: 512, steps: 8, guidance: 4, seed: 7, outputDir: output,
            sampler: sampler, scheduler: "linear", shift: 1)
        var completed = false
        let start = Date()
        for try await event in await engine.generate(request) {
            switch event {
            case .completed(let url, _, let effective):
                guard effective?.sampler == sampler, effective?.scheduler == "linear", effective?.shift == 1 else {
                    throw FluxError.invalidRequest("sampling effective parameters mismatch")
                }
                images[sampler] = try Data(contentsOf: url)
                print("Sampler=\(sampler) image=\(url.path) elapsed=\(Date().timeIntervalSince(start)) peak=\(Memory.peakMemory)")
                fflush(stdout)
                completed = true
            case .failed(let message, _): throw FluxError.invalidRequest(message)
            case .cancelled: throw CancellationError()
            default: break
            }
        }
        guard completed else { throw FluxError.invalidRequest("missing sampler output") }
    }
    guard images.count == limits.samplers.count, Set(images.values).count == images.count else {
        throw FluxError.invalidRequest("different samplers did not produce distinct images")
    }
    print("PASS all advertised samplers produce distinct images and report the actual scheduler/shift")
    await engine.unload()
}
