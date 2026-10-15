import Foundation
import AVFoundation
@preconcurrency import MLX
import vMLXFlux

/// Real local video runner. Uses the production loader, sampler and MP4 writer.
func runVideoProbe(_ arguments: [String]) async throws {
    guard arguments.count.isMultiple(of: 2) else {
        throw NSError(domain: "VideoProbe", code: 1, userInfo: [NSLocalizedDescriptionKey:
            "video --model PATH --output PATH [--prompt TEXT --width 832 --height 480 --frames 49 --fps 16 --steps 50 --seed 7 --source PATH --cancel-step N]"])
    }
    var options: [String: String] = [:]
    let allowed: Set<String> = ["--model", "--output", "--prompt", "--width", "--height", "--frames", "--fps", "--steps", "--seed", "--source", "--cancel-step"]
    for i in stride(from: 0, to: arguments.count, by: 2) {
        guard allowed.contains(arguments[i]), options[arguments[i]] == nil else {
            throw NSError(domain: "VideoProbe", code: 1, userInfo: [NSLocalizedDescriptionKey: "Unknown or duplicate argument \(arguments[i])"])
        }
        options[arguments[i]] = arguments[i + 1]
    }
    guard let path = options["--model"], let output = options["--output"] else {
        throw NSError(domain: "VideoProbe", code: 1, userInfo: [NSLocalizedDescriptionKey: "--model and --output are required"])
    }
    func integer(_ name: String, _ fallback: Int) throws -> Int {
        guard let raw = options[name] else { return fallback }
        guard let n = Int(raw), n > 0 else { throw NSError(domain: "VideoProbe", code: 1, userInfo: [NSLocalizedDescriptionKey: "Invalid \(name)"]) }
        return n
    }
    let model = URL(fileURLWithPath: path), directory = URL(fileURLWithPath: output)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    let interrupt = FluxInterrupt()
    let cancelStep = try integer("--cancel-step", Int.max)
    let seed: UInt64
    if let raw = options["--seed"] {
        guard let value = UInt64(raw) else { throw NSError(domain: "VideoProbe", code: 1) }
        seed = value
    } else { seed = 7 }
    let request = VideoGenRequest(prompt: options["--prompt"] ?? "A red sailboat moves across a calm blue lake in daylight.",
        width: try integer("--width", 832), height: try integer("--height", 480),
        numFrames: try integer("--frames", 49), fps: try integer("--fps", 16),
        steps: try integer("--steps", 50), guidance: 5, seed: seed, outputDir: directory,
        sourceImage: options["--source"].map { URL(fileURLWithPath: $0) }, interrupt: interrupt)
    VMLXFluxVideo.registerAll()
    let engine = FluxEngine()
    let started = Date()
    try await engine.load(name: "wan-2.1", modelPath: model)
    print("WAN loaded seconds=\(Date().timeIntervalSince(started)) active=\(Memory.activeMemory)")
    fflush(stdout)
    var terminal = false
    var cancelledAt: Date?
    for try await event in await engine.generateVideo(request, interrupt: interrupt) {
        switch event {
        case .step(let step, let total, let eta):
            print("WAN step=\(step)/\(total) eta=\(eta ?? -1) active=\(Memory.activeMemory) peak=\(Memory.peakMemory)")
            fflush(stdout)
            if step >= cancelStep, cancelledAt == nil { cancelledAt = Date(); interrupt.cancel() }
        case .preview(_, let step): print("WAN preview=\(step)")
        case .completed(let url, let actualSeed, let fps, let frames, let effective):
            let duration = try await AVURLAsset(url: url).load(.duration).seconds
            guard abs(duration - Double(frames) / Double(fps)) <= 1.0 / Double(fps) else {
                throw NSError(domain: "VideoProbe", code: 2, userInfo: [NSLocalizedDescriptionKey: "MP4 duration mismatch"])
            }
            print("WAN completed path=\(url.path) seed=\(actualSeed) frames=\(frames) fps=\(fps) duration=\(duration) elapsed=\(Date().timeIntervalSince(started)) effective=\(String(describing: effective))")
            terminal = true
        case .cancelled:
            print("WAN cancelled latency=\(cancelledAt.map { Date().timeIntervalSince($0) } ?? -1)")
            terminal = true
        case .failed(let message, _): throw NSError(domain: "VideoProbe", code: 3, userInfo: [NSLocalizedDescriptionKey: message])
        }
    }
    guard terminal else { throw NSError(domain: "VideoProbe", code: 4, userInfo: [NSLocalizedDescriptionKey: "Missing terminal event"]) }
    await engine.unload()
}
