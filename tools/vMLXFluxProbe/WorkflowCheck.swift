import Foundation
@preconcurrency import MLX
import vMLXFlux

/// Integration against installed weights, never synthetic tensors or pipelines.
func runWorkflowCheck(_ arguments: [String]) async throws {
    guard (3...4).contains(arguments.count) else {
        throw FluxError.invalidRequest("workflow-check MODEL_ROOT OUTPUT_DIRECTORY ZIMAGE_ADAPTER_FILE [all|stages|adapters|previews|memory1024|hires]")
    }
    let mode = arguments.count == 4 ? arguments[3] : "all"
    guard ["all", "stages", "adapters", "previews", "memory1024", "hires"].contains(mode) else {
        throw FluxError.invalidRequest("unknown workflow check mode")
    }
    let root = URL(fileURLWithPath: arguments[0]), output = URL(fileURLWithPath: arguments[1])
    try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
    VMLXFluxModels.registerAll()
    let engine = FluxEngine()
    func report(_ message: String) { print(message); fflush(stdout) }
    func require(_ condition: Bool, _ message: String) throws {
        guard condition else { throw FluxError.invalidRequest("workflow check: \(message)") }
    }
    let prompt = "A red ceramic teapot on a wooden kitchen table, natural daylight, photographic."
    func generate(_ request: ImageGenRequest) async throws -> (URL, Int, Double) {
        let started = Date()
        var result: URL?, previews = 0
        for try await event in await engine.generate(request) {
            switch event {
            case .completed(let url, _, _): result = url
            case .imageCompleted(let url, _, _, _): result = url
            case .failed(let message, _): throw FluxError.invalidRequest(message)
            case .cancelled: throw CancellationError()
            case .preview(let data, let step):
                try require(data.count < 65_536, "oversized preview")
                try data.write(to: output.appendingPathComponent("preview-\(step).png"))
                previews += 1
            case .step: break
            }
        }
        guard let result else { throw FluxError.invalidRequest("no generated image") }
        return (result, previews, Date().timeIntervalSince(started))
    }
    func sample(_ conditioning: FluxConditioning, _ latent: FluxLatent, seed: UInt64,
                interrupt: FluxInterrupt? = nil, cancelAtStep: Int? = nil) async throws -> FluxLatent? {
        for try await event in await engine.sample(conditioning, latent: latent,
            steps: 8, guidance: 1, seed: seed, interrupt: interrupt) {
            switch event {
            case .completed(let result, _): return result
            case .failed(let message, _): throw FluxError.invalidRequest(message)
            case .step(let step, _, _): if step == cancelAtStep { interrupt?.cancel() }
            case .cancelled: return nil
            case .preview: break
            }
        }
        throw FluxError.invalidRequest("sample has no terminal event")
    }
    if mode == "hires" {
        let qwen = root.appendingPathComponent("Qwen-Image-2.1-Turbo-mflux-4bit")
        try await engine.load(name: "qwen-image-2.1-turbo", modelPath: qwen)
        let request = ImageGenRequest(prompt: prompt, width: 1536, height: 1536,
            steps: 8, guidance: 1, seed: 7, outputDir: output)
        Memory.peakMemory = 0
        _ = try await generate(request)
        let freshPeak = Memory.peakMemory
        report("Hires fresh peak=\(freshPeak)")
        let conditioning = try await engine.encodePrompt(model: qwen, prompt: prompt)
        Memory.peakMemory = 0
        let empty = try await engine.emptyLatent(model: qwen, width: 768, height: 768)
        guard let first = try await sample(conditioning, empty, seed: 7) else { throw CancellationError() }
        await engine.release([.latent(empty)])
        let resized = try await engine.resize(first, width: 1536, height: 1536)
        await engine.release([.latent(first)])
        var final: FluxLatent?
        for try await event in await engine.sample(conditioning, latent: resized,
            steps: 8, guidance: 1, seed: 7, denoise: 0.45) {
            switch event {
            case .completed(let value, _): final = value
            case .failed(let message, _): throw FluxError.invalidRequest(message)
            case .cancelled: throw CancellationError()
            default: break
            }
        }
        guard let final else { throw FluxError.invalidRequest("no hires latent") }
        await engine.release([.latent(resized), .conditioning(conditioning)])
        let files = try await engine.decode(final, outputDir: output)
        let peak = Memory.peakMemory
        report("Hires output=\(files[0].path) width=\(final.width) height=\(final.height) peak=\(peak) freshPeak=\(freshPeak)")
        let expectedBytes = try await engine.retainedBytes(for: .latent(final))
        let activeBefore = Memory.activeMemory
        await engine.release([.latent(final)])
        let activeAfter = Memory.activeMemory
        report("Lifetime before=\(activeBefore) after=\(activeAfter) expectedDrop=\(expectedBytes)")
        try require(activeBefore - activeAfter >= expectedBytes, "released latent bytes remain active")
        try require(final.width == 1536 && final.height == 1536, "hires dimensions")
        try require(peak < freshPeak, "hires peak is not below fresh generation")
        await engine.unload()
        return
    }
    if mode == "all" || mode == "stages" {
    let qwen = root.appendingPathComponent("Qwen-Image-2.1-Turbo-mflux-4bit")
    try await engine.load(name: "qwen-image-2.1-turbo", modelPath: qwen)
    let qwenRequest = ImageGenRequest(prompt: prompt, width: 512, height: 512,
        steps: 8, guidance: 1, seed: 7, outputDir: output)
    let baseline = try await generate(qwenRequest)
    let conditioning = try await engine.encodePrompt(model: qwen, prompt: prompt)
    let cached = try await engine.encodePrompt(model: qwen, prompt: prompt)
    try require(conditioning == cached, "prompt conditioning was not reused")
    let empty = try await engine.emptyLatent(model: qwen, width: 512, height: 512)
    guard let sampled = try await sample(conditioning, empty, seed: 7) else { throw CancellationError() }
    let stageFiles = try await engine.decode(sampled, outputDir: output)
    try require(try Data(contentsOf: baseline.0) == Data(contentsOf: stageFiles[0]), "stage/generate seed-7 byte mismatch")
    report("PASS stage parity and conditioning reuse")
    let other = try await engine.encodePrompt(model: qwen, prompt: "A blue boat at sea.")
    let mixed = try await engine.mix([(conditioning, 1), (other, 0)])
    guard let mixedLatent = try await sample(mixed, empty, seed: 7) else { throw CancellationError() }
    let mixedFiles = try await engine.decode(mixedLatent, outputDir: output)
    try require(try Data(contentsOf: baseline.0) == Data(contentsOf: mixedFiles[0]), "mix 1/0 byte mismatch")
    report("PASS zero-weight conditioning mix")
    let interrupt = FluxInterrupt()
    let cancelled = try await sample(conditioning, empty, seed: 8, interrupt: interrupt, cancelAtStep: 1)
    try require(cancelled == nil, "sample ignored cancellation")
    guard let recovered = try await sample(conditioning, empty, seed: 8) else { throw CancellationError() }
    let recoveredFiles = try await engine.decode(recovered, outputDir: output)
    var second = qwenRequest; second.seed = 8
    let secondBaseline = try await generate(second)
    try require(try Data(contentsOf: recoveredFiles[0]) == Data(contentsOf: secondBaseline.0), "cancel changed input latent")
    report("PASS cancellation preserves latent and seed-8 reuse parity")
    await engine.release([.latent(empty), .latent(sampled), .latent(mixedLatent), .latent(recovered),
                          .conditioning(conditioning), .conditioning(other), .conditioning(mixed)])
    do {
        _ = try await engine.decode(empty, outputDir: output)
        throw FluxError.invalidRequest("released handle remained usable")
    } catch FluxError.staleHandle { report("PASS stale handle") }
    await engine.unload()
    }
    if mode == "stages" { return }

    try await engine.load(name: "z-image-turbo", modelPath: root.appendingPathComponent("Z-Image-Turbo-mflux-8bit"))
    var request = ImageGenRequest(prompt: prompt, width: 512, height: 512,
        steps: 20, guidance: 0, seed: 7, outputDir: output)
    if mode == "memory1024" { request.width = 1024; request.height = 1024 }
    Memory.peakMemory = 0
    let estimate = try await engine.estimatedPeakBytes(for: request)
    let base = try await generate(request)
    report("ZImage base width=\(request.width) elapsed=\(base.2) estimate=\(estimate) peak=\(Memory.peakMemory) active=\(Memory.activeMemory)")
    if mode == "memory1024" { await engine.unload(); return }
    var controlTime = base.2
    if mode == "all" || mode == "adapters" {
    let adapter = URL(fileURLWithPath: arguments[2])
    request.adapters = [FluxAdapter(url: adapter, scale: 0)]
    let zero = try await generate(request)
    try require(try Data(contentsOf: base.0) == Data(contentsOf: zero.0), "zero adapter changed output")
    request.adapters = [FluxAdapter(url: adapter, scale: 0.5)]
    let adapted = try await generate(request)
    let loads = await engine.adapterLoadCount
    let replay = try await generate(request)
    let replayLoads = await engine.adapterLoadCount
    try require(loads == replayLoads && loads == 1, "adapter reloaded")
    try require(try Data(contentsOf: adapted.0) == Data(contentsOf: replay.0), "adapter replay mismatch")
    try require(try Data(contentsOf: base.0) != Data(contentsOf: adapted.0), "nonzero adapter did not affect output")
    request.adapters = []
    let restored = try await generate(request)
    try require(try Data(contentsOf: base.0) == Data(contentsOf: restored.0), "adapter removal did not restore base")
    report("PASS real adapter zero, cache, replay, removal")
    controlTime = restored.2
    }
    if mode == "adapters" { await engine.unload(); return }
    request.previewInterval = 5
    let preview = try await generate(request)
    try require(try Data(contentsOf: base.0) == Data(contentsOf: preview.0), "previews changed final image")
    report("Preview count=\(preview.1) elapsed=\(preview.2) control=\(controlTime) ratio=\(preview.2/controlTime)")
    try require((3...4).contains(preview.1), "preview count outside 3...4")
    await engine.unload()
}
