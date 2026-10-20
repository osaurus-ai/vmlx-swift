import Foundation
import CryptoKit
import ImageIO
@preconcurrency import MLX
import vMLXFlux

/// Real native public entry points; one process per row releases all GPU resources.
/// No server, network, sampler overrides, or estimate-based admission.
func runFlux3Check(_ arguments: [String]) async throws {
    guard arguments.count >= 3 else {
        throw FluxError.invalidRequest("flux3-check MODEL_DIRECTORY OUTPUT_DIRECTORY generate|edit|unsized|multi [REFERENCE ...]")
    }
    let model = URL(fileURLWithPath: arguments[0]), output = URL(fileURLWithPath: arguments[1])
    let mode = arguments[2], sources = arguments.dropFirst(3).map { URL(fileURLWithPath: $0) }
    guard ["generate", "edit", "unsized", "multi"].contains(mode),
          (mode == "generate" ? sources.isEmpty : mode == "multi" ? (2...10).contains(sources.count) : sources.count == 1) else {
        throw FluxError.invalidRequest("generate needs no reference; edit/unsized need one; multi needs 2...10 ordered references")
    }
    try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
    let receipt = output.appendingPathComponent("swift-\(mode)-seed42.json")
    let genPrompt = "A cozy corner bakery storefront in the early morning, a wooden sign above the door reading 'MORNING GLORY BAKERY' in cream-colored letters, a window full of fresh bread and pastries, warm light inside, wet cobblestone street, photograph"
    let editPrompt = "A cozy corner bakery storefront in the early morning, a wooden sign above the door reading 'OSAURUS CAFE' in cream-colored letters, a bright red striped awning over the window, a window full of fresh bread and pastries, warm light inside, wet cobblestone street, photograph"
    let prompt = mode == "generate" ? genPrompt : editPrompt
    var record: [String: Any] = ["model": model.path, "mode": mode, "seed": 42, "steps": 4,
        "prompt": prompt, "sources": sources.map(\.path), "scheduler": "flow-map",
        "requested_width": mode == "unsized" ? NSNull() : 512,
        "requested_height": mode == "unsized" ? NSNull() : 512,
        "negative_prompt": mode == "generate" ? "ignored by distilled model" : "not present in native edit DTO"]
    func save() throws {
        try JSONSerialization.data(withJSONObject: record, options: [.prettyPrinted, .sortedKeys]).write(to: receipt, options: .atomic)
    }
    VMLXFluxModels.registerAll()
    let engine = FluxEngine()
    let start = Date()
    do {
        try await engine.load(name: "flux3-image-klein", modelPath: model)
        record["load_seconds"] = Date().timeIntervalSince(start)
        let runStart = Date()
        let stream: AsyncThrowingStream<ImageGenEvent, Error>
        if mode == "generate" {
            stream = await engine.generate(ImageGenRequest(prompt: prompt, negativePrompt: "ignored by distilled model",
                width: 512, height: 512, steps: 4, guidance: 1, seed: 42, outputDir: output))
        } else {
            stream = await engine.edit(try ImageEditRequest(prompt: prompt, sourceImages: sources,
                width: mode == "unsized" ? nil : 512, height: mode == "unsized" ? nil : 512,
                steps: 4, guidance: 1, seed: 42, outputDir: output))
        }
        var completed = false
        for try await event in stream {
            switch event {
            case .completed(let url, let seed, let effective), .imageCompleted(let url, let seed, _, let effective):
                let elapsed = Date().timeIntervalSince(runStart)
                let named = output.appendingPathComponent("swift-\(mode)-seed42.png")
                let bytes = try Data(contentsOf: url)
                try bytes.write(to: named, options: .atomic)
                guard seed == 42, effective?.steps == 4, effective?.scheduler == "flow-map" else {
                    throw FluxError.invalidRequest("FLUX 3 effective parameters do not match request")
                }
                if mode != "unsized", effective?.width != 512 || effective?.height != 512 {
                    throw FluxError.invalidRequest("FLUX 3 output dimensions do not match request")
                }
                record["effective_parameters"] = try effective.map { try JSONSerialization.jsonObject(with: JSONEncoder().encode($0)) }
                record["operation_seconds"] = elapsed
                record["images_per_second"] = 1 / elapsed
                record["total_seconds"] = Date().timeIntervalSince(start)
                record["peak_mlx_bytes"] = Memory.peakMemory
                record["output"] = named.path
                record["sha256"] = SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined()
                completed = true
            case .failed(let message, _): throw FluxError.invalidRequest(message)
            case .cancelled: throw CancellationError()
            default: break
            }
        }
        guard completed else { throw FluxError.invalidRequest("FLUX 3 stream ended without an image") }
        record["status"] = "completed"
        await engine.unload()
        Memory.clearCache()
        record["mlx_active_bytes_after_unload"] = Memory.activeMemory
        try save()
        print("FLUX3_CHECK \(receipt.path)")
    } catch {
        record["status"] = "failed"
        record["error"] = String(describing: error)
        record["total_seconds"] = Date().timeIntervalSince(start)
        await engine.unload()
        Memory.clearCache()
        try save()
        throw error
    }
}
