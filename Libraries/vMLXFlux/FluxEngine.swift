import Foundation
@_exported import vMLXFluxKit
@_exported import vMLXFluxModels
@_exported import vMLXFluxVideo

/// Top-level facade for vmlx-flux. One import, one actor, one API.
///
/// The engine loads a FluxModel on demand via `load(_:)`, then dispatches
/// generation / edit / upscale / video requests to the right backend. It
/// streams progress events as they arrive from the scheduler so the
/// calling app (vMLX) can render live step counters + partial previews.
///
/// Threading: actor-isolated. MLX ops are not thread-safe across the
/// same allocator, and the generation loop has a persistent latent buffer,
/// so every entry point goes through the actor's executor.
public actor FluxEngine {

    // MARK: - State

    /// Currently loaded model (nil if no model is resident).
    public internal(set) var loaded: LoadedModel?

    /// Cached model registry lookup — `ModelRegistry.lookup(name:)`
    /// is O(1) but we keep a local copy for logging.
    public internal(set) var lastLoadedName: String?
    var loadedModelURL: URL?
    var adapterState = FluxAdapterState()
    var stageConditionings: [UUID: StoredFluxConditioning] = [:]
    var stageLatents: [UUID: StoredFluxLatent] = [:]
    var stagePromptCache: [StagePromptKey: FluxConditioning] = [:]
    private var operationBusy = false
    private var operationWaiters: [CheckedContinuation<Void, Never>] = []
    public var adapterLoadCount: Int { adapterState.loadCount }

    func acquireOperation() async {
        if !operationBusy { operationBusy = true; return }
        await withCheckedContinuation { operationWaiters.append($0) }
    }

    func releaseOperation() {
        if operationWaiters.isEmpty { operationBusy = false }
        else { operationWaiters.removeFirst().resume() }
    }

    // MARK: - Init

    public init() {}

    /// Query implemented request constraints without loading model weights.
    /// Supply the canonical name when a bundle directory has a custom name.
    public func limits(for modelURL: URL, name: String? = nil) throws -> FluxModelLimits {
        VMLXFluxModels.registerAll()
        guard let entry = ModelRegistry.lookupFuzzy(name: name ?? modelURL.lastPathComponent) else {
            throw FluxError.unknownModel(name ?? modelURL.lastPathComponent)
        }
        guard let query = entry.limits else {
            throw FluxError.notImplemented("request limits for \(entry.name)")
        }
        return try query(modelURL)
    }

    /// Metadata-only preflight, also usable before `load` through the modelURL overload.
    public func estimatedPeakBytes(for request: ImageGenRequest) throws -> Int {
        guard let loadedModelURL, let loaded else { throw FluxError.notLoaded }
        return try estimatedPeakBytes(for: request, modelURL: loadedModelURL, name: loaded.name)
    }

    public func estimatedPeakBytes(for request: ImageGenRequest, modelURL: URL, name: String? = nil) throws -> Int {
        let family = try estimateFamily(modelURL, name: name)
        let constraints = try limits(for: modelURL, name: family)
        try constraints.validate(request)
        return try ImageMemoryEstimator.estimate(model: modelURL, family: family,
            width: request.width, height: request.height,
            parallelImages: min(request.batchSize, constraints.maximumParallelImages), guidance: request.guidance,
            previewMaxPixel: request.previewInterval == nil ? nil : request.previewMaxPixel).peakBytes
    }

    public func estimatedPeakBytes(for request: ImageEditRequest) throws -> Int {
        guard let loadedModelURL, let loaded else { throw FluxError.notLoaded }
        return try estimatedPeakBytes(for: request, modelURL: loadedModelURL, name: loaded.name)
    }

    public func estimatedPeakBytes(for request: ImageEditRequest, modelURL: URL, name: String? = nil) throws -> Int {
        let family = try estimateFamily(modelURL, name: name)
        try limits(for: modelURL, name: family).validate(request)
        let width: Int, height: Int
        if family == "qwen-image-edit" {
            let plan = try QwenImageEditPreprocessPlan(sourceImages: request.sourceImages,
                requestedWidth: request.width, requestedHeight: request.height,
                steps: request.steps, guidance: request.guidance)
            width = plan.outputWidth; height = plan.outputHeight
        } else {
            let size = try ImageIO.dimensions(of: request.sourceImages.last ?? request.sourceImage)
            let ratio = Double(size.width) / Double(size.height)
            let w = max(32, Int((sqrt(1024 * 1024 * ratio) / 32).rounded(.toNearestOrEven)) * 32)
            let h = max(32, Int((Double(w) / ratio / 32).rounded(.toNearestOrEven)) * 32)
            width = request.width ?? w; height = request.height ?? h
        }
        return try ImageMemoryEstimator.estimate(model: modelURL, family: family, width: width, height: height,
            sourcePixels: request.sourceImages.count * 1024 * 1024, guidance: request.guidance,
            previewMaxPixel: request.previewInterval == nil ? nil : request.previewMaxPixel).peakBytes
    }

    public func estimatedPeakBytes(for request: UpscaleRequest) throws -> Int {
        guard let loadedModelURL, let loaded else { throw FluxError.notLoaded }
        return try estimatedPeakBytes(for: request, modelURL: loadedModelURL, name: loaded.name)
    }

    public func estimatedPeakBytes(for request: UpscaleRequest, modelURL: URL, name: String? = nil) throws -> Int {
        let family = try estimateFamily(modelURL, name: name)
        guard family == "seedvr2" else { throw FluxError.wrongModelKind(expected: "seedvr2", actual: family) }
        let size = try ImageIO.dimensions(of: request.sourceImage)
        try SeedVR2BundleMetadata.validateUpscale(sourceWidth: size.width, sourceHeight: size.height,
            scale: request.scale, format: request.outputFormat)
        let width = (size.width * request.scale + 15) / 16 * 16
        let height = (size.height * request.scale + 15) / 16 * 16
        return try ImageMemoryEstimator.estimate(model: modelURL, family: family, width: width, height: height).peakBytes
    }

    private func estimateFamily(_ model: URL, name: String?) throws -> String {
        VMLXFluxModels.registerAll()
        guard let entry = ModelRegistry.lookupFuzzy(name: name ?? model.lastPathComponent) else {
            throw FluxError.unknownModel(name ?? model.lastPathComponent)
        }
        return entry.name
    }

    // MARK: - Load / unload

    /// Load a model by canonical name. Mirrors the Python mflux
    /// `ModelConfig.from_name(_:)` resolution with the `SUPPORTED_MODELS`
    /// dict from `vmlx_engine/image_gen.py`.
    ///
    /// Parameters:
    /// - `name`: canonical model key — `"flux1-schnell"`, `"flux2-klein"`,
    ///   `"z-image-turbo"`, `"qwen-image"`, `"qwen-image-edit"`,
    ///   `"flux1-kontext"`, `"flux1-fill"`, `"seedvr2"`, `"fibo"`.
    /// - `modelPath`: local directory containing weights. We NEVER
    ///   silently download from HuggingFace — the caller (vMLX
    ///   `DownloadManager`) must stage the weights first.
    /// - `quantize`: bit width (4, 8) or nil for full precision.
    public func load(
        name: String,
        modelPath: URL,
        quantize: Int? = nil
    ) async throws {
        await acquireOperation()
        defer { releaseOperation() }
        try Task.checkCancellation()
        guard let entry = ModelRegistry.lookup(name: name) else {
            throw FluxError.unknownModel(name)
        }
        let nextAdapters = FluxAdapterState()
        let model = try await FluxAdapterScope.$current.withValue(nextAdapters) {
            try await entry.loader(modelPath, quantize)
        }
        clearStageHandles()
        adapterState = nextAdapters
        self.loaded = LoadedModel(
            name: name,
            kind: entry.kind,
            model: model
        )
        self.lastLoadedName = name
        self.loadedModelURL = modelPath
    }

    /// Resolve and load a local image-generation bundle from
    /// `~/.mlxstudio/models/image` or a caller-supplied model store.
    ///
    /// This keeps the "no silent downloads" rule intact while letting vMLX
    /// callers use canonical model names instead of hard-coded local paths.
    public func load(
        name: String,
        from store: MLXStudioModelStore = MLXStudioModelStore()
    ) async throws -> LocalFluxModel {
        VMLXFluxModels.registerAll()
        VMLXFluxVideo.registerAll()
        guard let local = try store.resolve(name: name) else {
            throw FluxError.localModelNotFound(name, store.root)
        }
        guard local.canEnterNativeLoadPath else {
            throw FluxError.localModelIncomplete(
                local.directory,
                reasons: local.blockedReasons)
        }
        guard let canonicalName = local.canonicalName else {
            throw FluxError.unknownModel(local.directoryName)
        }
        try await load(
            name: canonicalName,
            modelPath: local.directory,
            quantize: local.quantizationBits)
        return local
    }

    /// Unload the current model and release weights.
    public func unload() async {
        await acquireOperation()
        defer { releaseOperation() }
        clearStageHandles()
        self.loaded = nil
        self.loadedModelURL = nil
        self.adapterState = FluxAdapterState()
    }

    // MARK: - Generate (text-to-image)

    /// Run a text-to-image generation. Returns an AsyncThrowingStream of
    /// progress events — the final `.completed(url:)` carries the
    /// written image path.
    public func generate(
        _ request: ImageGenRequest
    ) -> AsyncThrowingStream<ImageGenEvent, Error> {
        var request = request
        let interrupt = request.interrupt ?? FluxInterrupt()
        request.interrupt = interrupt
        let job = request
        return AsyncThrowingStream { continuation in
            Task { [weak self] in
                guard let self else { continuation.finish(); return }
                do {
                    try await self.performGenerate(job, continuation: continuation)
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { reason in
                if case .cancelled = reason { interrupt.cancel() }
            }
        }
    }

    private func performGenerate(
        _ request: ImageGenRequest,
        continuation: AsyncThrowingStream<ImageGenEvent, Error>.Continuation
    ) async throws {
        await acquireOperation()
        defer { adapterState.clear(); releaseOperation() }
        try Task.checkCancellation()
        guard let loaded else { throw FluxError.notLoaded }
        guard let generator = loaded.model as? ImageGenerator else {
            throw FluxError.wrongModelKind(
                expected: "ImageGenerator",
                actual: String(describing: type(of: loaded.model))
            )
        }
        try await FluxAdapterScope.$current.withValue(adapterState) {
            if request.adapters.contains(where: { $0.scale != 0 }),
                let preparer = loaded.model as? FluxAdapterPreparing {
                try await preparer.prepareAdapterLayers()
            }
            try adapterState.install(request.adapters) { _ in
                continuation.yield(.step(step: 0, total: 0, etaSeconds: nil))
            }
            try await FluxInterruptScope.$current.withValue(request.interrupt) {
                try FluxInterruptScope.check()
                for try await event in generator.generate(request) {
                    continuation.yield(event)
                }
            }
        }
    }

    // MARK: - Edit (image-to-image)

    /// Run an image edit request (Kontext / Fill / Qwen-Image-Edit).
    public func edit(
        _ request: ImageEditRequest
    ) -> AsyncThrowingStream<ImageGenEvent, Error> {
        var request = request
        let interrupt = request.interrupt ?? FluxInterrupt()
        request.interrupt = interrupt
        let job = request
        return AsyncThrowingStream { continuation in
            Task { [weak self] in
                guard let self else { continuation.finish(); return }
                do {
                    try await self.performEdit(job, continuation: continuation)
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { reason in
                if case .cancelled = reason { interrupt.cancel() }
            }
        }
    }

    private func performEdit(
        _ request: ImageEditRequest,
        continuation: AsyncThrowingStream<ImageGenEvent, Error>.Continuation
    ) async throws {
        await acquireOperation()
        defer { adapterState.clear(); releaseOperation() }
        try Task.checkCancellation()
        guard let loaded else { throw FluxError.notLoaded }
        guard let editor = loaded.model as? ImageEditor else {
            throw FluxError.wrongModelKind(
                expected: "ImageEditor",
                actual: String(describing: type(of: loaded.model))
            )
        }
        try await FluxAdapterScope.$current.withValue(adapterState) {
            if request.adapters.contains(where: { $0.scale != 0 }),
                let preparer = loaded.model as? FluxAdapterPreparing {
                try await preparer.prepareAdapterLayers()
            }
            try adapterState.install(request.adapters) { _ in
                continuation.yield(.step(step: 0, total: 0, etaSeconds: nil))
            }
            try await FluxInterruptScope.$current.withValue(request.interrupt) {
                try FluxInterruptScope.check()
                for try await event in editor.edit(request) {
                    continuation.yield(event)
                }
            }
        }
    }

    // MARK: - Upscale (SeedVR2)

    /// Run an upscale request. Emits the same progress events as gen/edit.
    public func upscale(
        _ request: UpscaleRequest,
        interrupt: FluxInterrupt? = nil
    ) -> AsyncThrowingStream<ImageGenEvent, Error> {
        var request = request
        let interrupt = interrupt ?? request.interrupt ?? FluxInterrupt()
        request.interrupt = interrupt
        let job = request
        return AsyncThrowingStream { continuation in
            continuation.onTermination = { reason in
                if case .cancelled = reason { interrupt.cancel() }
            }
            Task { [weak self] in
                guard let self else { continuation.finish(); return }
                do {
                    try await self.performUpscale(job, continuation: continuation)
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    private func performUpscale(
        _ request: UpscaleRequest,
        continuation: AsyncThrowingStream<ImageGenEvent, Error>.Continuation
    ) async throws {
        await acquireOperation()
        defer { releaseOperation() }
        try Task.checkCancellation()
        guard let loaded else { throw FluxError.notLoaded }
        guard let upscaler = loaded.model as? ImageUpscaler else {
            throw FluxError.wrongModelKind(
                expected: "ImageUpscaler",
                actual: String(describing: type(of: loaded.model))
            )
        }
        for try await event in upscaler.upscale(request) {
            continuation.yield(event)
        }
    }

    // MARK: - Video

    public func generateVideo(_ request: VideoGenRequest, interrupt: FluxInterrupt? = nil)
        -> AsyncThrowingStream<VideoGenEvent, Error> {
        var request = request
        let interrupt = interrupt ?? request.interrupt ?? FluxInterrupt()
        request.interrupt = interrupt
        let job = request
        return AsyncThrowingStream { continuation in
            Task {
                await self.acquireOperation()
                defer { self.releaseOperation(); continuation.finish() }
                do {
                    try Task.checkCancellation()
                    try interrupt.check()
                    guard let generator = self.loaded?.model as? any VideoGenerator else {
                        throw FluxError.wrongModelKind(expected: "VideoGenerator", actual: self.loaded?.name ?? "unloaded")
                    }
                    for try await event in generator.generate(job) { continuation.yield(event) }
                } catch is CancellationError { continuation.yield(.cancelled) }
                catch { continuation.finish(throwing: error) }
            }
            continuation.onTermination = { reason in
                if case .cancelled = reason { interrupt.cancel() }
            }
        }
    }

}

/// A loaded model + its canonical name + kind (for fast dispatch).
public struct LoadedModel: Sendable {
    public let name: String
    public let kind: ModelKind
    public let model: any FluxModel
}
