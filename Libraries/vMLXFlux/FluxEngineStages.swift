import Foundation
@preconcurrency import MLX
import vMLXFluxKit

struct StoredFluxConditioning {
    let value: FluxStageConditioning
    let adapters: [FluxAdapter]
    let adapterIdentity: [String]
}
struct StoredFluxLatent {
    let value: MLXArray
    let width: Int
    let height: Int
}

extension FluxEngine {
    func clearStageHandles() {
        stageConditionings.removeAll()
        stageLatents.removeAll()
        stagePromptCache.removeAll()
    }

    /// Caller owns the operation lease. Load only once and release the old
    /// model first; stage calls must not transiently retain two weight copies.
    func stageModel(for url: URL) async throws -> any FluxStageModel {
        if loadedModelURL?.standardizedFileURL != url.standardizedFileURL {
            VMLXFluxModels.registerAll()
            guard let entry = ModelRegistry.lookupFuzzy(name: url.lastPathComponent) else {
                throw FluxError.unknownModel(url.lastPathComponent)
            }
            loaded = nil
            loadedModelURL = nil
            clearStageHandles()
            adapterState = FluxAdapterState()
            let model = try await FluxAdapterScope.$current.withValue(adapterState) {
                try await entry.loader(url, nil)
            }
            loaded = LoadedModel(name: entry.name, kind: entry.kind, model: model)
            loadedModelURL = url
            lastLoadedName = entry.name
        }
        guard let model = loaded?.model as? any FluxStageModel else {
            throw FluxError.notImplemented("stage API for \(loaded?.name ?? url.lastPathComponent)")
        }
        return model
    }

    public func encodePrompt(model: URL, prompt: String, negativePrompt: String? = nil,
                             adapters: [FluxAdapter] = []) async throws -> FluxConditioning {
        await acquireOperation()
        defer { adapterState.clear(); releaseOperation() }
        try Task.checkCancellation()
        let pipeline = try await stageModel(for: model)
        return try FluxAdapterScope.$current.withValue(adapterState) {
            try adapterState.install(adapters) { _ in }
            // Content identity is validated by install, so modified adapter files
            // cannot reuse conditioning from the previous content at that URL.
            let key = StagePromptKey(prompt: prompt, negative: negativePrompt,
                adapterIdentity: adapterState.identity)
            if let handle = stagePromptCache[key], stageConditionings[handle.id] != nil { return handle }
            let value = try pipeline.stageEncodePrompt(prompt, negativePrompt: negativePrompt)
            let handle = FluxConditioning()
            stageConditionings[handle.id] = StoredFluxConditioning(value: value, adapters: adapters, adapterIdentity: adapterState.identity)
            stagePromptCache[key] = handle
            return handle
        }
    }

    public func encodeImage(model: URL, image: URL, width: Int? = nil, height: Int? = nil) async throws -> FluxLatent {
        await acquireOperation()
        defer { releaseOperation() }
        try Task.checkCancellation()
        let pipeline = try await stageModel(for: model)
        let dimensions = try ImageIO.dimensions(of: image)
        let w = width ?? dimensions.width, h = height ?? dimensions.height
        try validateStageSize(width: w, height: h, pipeline: pipeline)
        return storeStageLatent(try pipeline.stageEncodeImage(image, width: w, height: h), width: w, height: h)
    }

    public func emptyLatent(model: URL, width: Int, height: Int, batch: Int = 1) async throws -> FluxLatent {
        await acquireOperation()
        defer { releaseOperation() }
        try Task.checkCancellation()
        let pipeline = try await stageModel(for: model)
        try validateStageSize(width: width, height: height, pipeline: pipeline)
        guard (1...8).contains(batch) else {
            throw FluxError.invalidField(field: "batch", value: "\(batch)", allowed: "1...8")
        }
        let value = MLXArray.zeros([batch, height / pipeline.stageSpatialScale,
            width / pipeline.stageSpatialScale, pipeline.stageChannels], dtype: .float32)
        eval(value)
        return storeStageLatent(value, width: width, height: height)
    }

    public func mix(_ parts: [(FluxConditioning, Float)]) async throws -> FluxConditioning {
        await acquireOperation()
        defer { releaseOperation() }
        guard !parts.isEmpty, parts.allSatisfy({ $0.1.isFinite }) else {
            throw FluxError.invalidRequest("mix requires finite weights and at least one conditioning")
        }
        let stored = try parts.map { part -> (StoredFluxConditioning, Float) in
            guard let value = stageConditionings[part.0.id] else { throw FluxError.staleHandle }
            return (value, part.1)
        }
        let used = stored.filter { $0.1 != 0 }
        guard let first = used.first else { throw FluxError.invalidRequest("mix weights cannot all be zero") }
        guard used.allSatisfy({ $0.0.adapters == first.0.adapters && $0.0.adapterIdentity == first.0.adapterIdentity }) else {
            throw FluxError.invalidRequest("mixed conditioning must use the same adapter set")
        }
        var arrays = first.0.value.arrays
        if first.1 != 1 { arrays = arrays.mapValues { $0 * first.1 } }
        for (item, weight) in used.dropFirst() {
            guard Set(item.value.arrays.keys) == Set(arrays.keys) else {
                throw FluxError.invalidRequest("conditioning payloads differ")
            }
            for key in arrays.keys.sorted() {
                var a = arrays[key]!, b = item.value.arrays[key]!
                guard a.ndim == 3, b.ndim == 3, a.dim(0) == b.dim(0), a.dim(2) == b.dim(2) else {
                    throw FluxError.invalidRequest("conditioning shapes differ for \(key)")
                }
                // Prompt length varies. Preserve token positions and pad the
                // shorter embedding with zeros before the requested weighted sum.
                let length = max(a.dim(1), b.dim(1))
                if a.dim(1) < length { a = concatenated([a, MLXArray.zeros([a.dim(0), length - a.dim(1), a.dim(2)], dtype: a.dtype)], axis: 1) }
                if b.dim(1) < length { b = concatenated([b, MLXArray.zeros([b.dim(0), length - b.dim(1), b.dim(2)], dtype: b.dtype)], axis: 1) }
                arrays[key] = a + b * weight
            }
        }
        eval(Array(arrays.values))
        let handle = FluxConditioning()
        stageConditionings[handle.id] = StoredFluxConditioning(
            value: FluxStageConditioning(arrays: arrays), adapters: first.0.adapters, adapterIdentity: first.0.adapterIdentity)
        return handle
    }

    public func sample(_ conditioning: FluxConditioning, latent: FluxLatent,
                       steps: Int, guidance: Float, seed: UInt64,
                       sampler: String? = nil, scheduler: String? = nil, shift: Float? = nil,
                       denoise: Float = 1, interrupt: FluxInterrupt? = nil)
        -> AsyncThrowingStream<FluxSampleEvent, Error> {
        let interrupt = interrupt ?? FluxInterrupt()
        let options = FluxStageSampling(steps: steps, guidance: guidance, seed: seed,
            sampler: sampler, scheduler: scheduler, shift: shift, denoise: denoise)
        return AsyncThrowingStream { continuation in
            let task = Task {
                await self.performStageSample(conditioning, latent: latent, options: options,
                    interrupt: interrupt, continuation: continuation)
            }
            continuation.onTermination = { reason in
                if case .cancelled = reason { interrupt.cancel(); task.cancel() }
            }
        }
    }

    private func performStageSample(_ conditioning: FluxConditioning, latent: FluxLatent,
                                   options: FluxStageSampling, interrupt: FluxInterrupt,
                                   continuation: AsyncThrowingStream<FluxSampleEvent, Error>.Continuation) async {
        await acquireOperation()
        defer { adapterState.clear(); releaseOperation(); continuation.finish() }
        do {
            try Task.checkCancellation()
            try interrupt.check()
            guard let condition = stageConditionings[conditioning.id], let source = stageLatents[latent.id] else {
                throw FluxError.staleHandle
            }
            guard let pipeline = loaded?.model as? any FluxStageModel else { throw FluxError.staleHandle }
            try adapterState.install(condition.adapters) { _ in
                continuation.yield(.step(step: 0, total: 0, etaSeconds: nil))
            }
            guard adapterState.identity == condition.adapterIdentity else {
                throw FluxError.invalidRequest("adapter contents changed; encode the prompt again")
            }
            var outputs: [MLXArray] = []
            var effective: EffectiveImageParameters?
            for index in 0..<source.value.dim(0) {
                try interrupt.check()
                var imageOptions = options
                imageOptions.seed = options.seed &+ UInt64(index)
                let (output, actual) = try pipeline.stageSample(condition.value,
                    latent: source.value[index..<(index + 1)], width: source.width, height: source.height,
                    options: imageOptions, interrupt: interrupt) { step, total, eta in
                        continuation.yield(.step(step: index * total + step,
                            total: source.value.dim(0) * total, etaSeconds: eta))
                    }
                eval(output)
                outputs.append(output)
                if effective == nil { effective = actual }
            }
            try interrupt.check()
            guard let effective else { throw FluxError.invalidRequest("empty sample batch") }
            let value = outputs.count == 1 ? outputs[0] : concatenated(outputs, axis: 0)
            let handle = storeStageLatent(value, width: source.width, height: source.height)
            continuation.yield(.completed(handle, effective: effective))
        } catch is CancellationError { continuation.yield(.cancelled) }
        catch { continuation.finish(throwing: error) }
    }

    public func decode(_ latent: FluxLatent, outputDir: URL, format: ImageFormat = .png) async throws -> [URL] {
        await acquireOperation()
        defer { releaseOperation() }
        guard format == .png else {
            throw FluxError.invalidField(field: "format", value: format.rawValue, allowed: "png")
        }
        guard let source = stageLatents[latent.id], let pipeline = loaded?.model as? any FluxStageModel else {
            throw FluxError.staleHandle
        }
        var files: [URL] = []
        do {
            for index in 0..<source.value.dim(0) {
                try Task.checkCancellation()
                let rgb = try pipeline.stageDecode(source.value[index..<(index + 1)])
                files.append(try ImageIO.writePNG(rgb, outputDir: outputDir, prefix: "stage"))
            }
            return files
        } catch {
            for file in files { try? FileManager.default.removeItem(at: file) }
            throw error
        }
    }

    public func resize(_ latent: FluxLatent, width: Int, height: Int, method: String = "bilinear") async throws -> FluxLatent {
        await acquireOperation()
        defer { releaseOperation() }
        guard let source = stageLatents[latent.id], let pipeline = loaded?.model as? any FluxStageModel else {
            throw FluxError.staleHandle
        }
        try validateStageSize(width: width, height: height, pipeline: pipeline)
        guard ["nearest", "bilinear", "bislerp"].contains(method) else {
            throw FluxError.invalidField(field: "method", value: method, allowed: "nearest, bilinear, bislerp")
        }
        let targetH = height / pipeline.stageSpatialScale, targetW = width / pipeline.stageSpatialScale
        var value = source.value
        for (axis, target) in [(1, targetH), (2, targetW)] where value.dim(axis) != target {
            let size = value.dim(axis)
            if method == "nearest" {
                let indices = (0..<target).map { Int32(min(size - 1, $0 * size / target)) }
                value = value.take(MLXArray(indices), axis: axis)
            } else {
                let positions = (0..<target).map { max(0, min(Float(size - 1), (Float($0) + 0.5) * Float(size) / Float(target) - 0.5)) }
                let lower = positions.map { Int32(floor($0)) }
                let upper = lower.map { min(Int32(size - 1), $0 + 1) }
                var ratioShape = [1, 1, 1, 1]; ratioShape[axis] = target
                let ratio = MLXArray(zip(positions, lower).map { $0 - Float($1) }).reshaped(ratioShape)
                let a = value.take(MLXArray(lower), axis: axis).asType(.float32)
                let b = value.take(MLXArray(upper), axis: axis).asType(.float32)
                let linear = a * (1 - ratio) + b * ratio
                if method == "bilinear" { value = linear.asType(value.dtype) }
                else {
                    let an = sqrt(sum(a * a, axis: -1, keepDims: true))
                    let bn = sqrt(sum(b * b, axis: -1, keepDims: true))
                    let dot = sum(a * b, axis: -1, keepDims: true) / maximum(an * bn, Float(1e-20))
                    let angle = acos(clip(dot, min: -0.999999, max: 0.999999))
                    let unit = a / maximum(an, Float(1e-20)) * (sin((1 - ratio) * angle) / sin(angle))
                        + b / maximum(bn, Float(1e-20)) * (sin(ratio * angle) / sin(angle))
                    let spherical = unit * (an * (1 - ratio) + bn * ratio)
                    let singular = logicalOr(abs(dot) .> Float(0.9995), minimum(an, bn) .< Float(1e-10))
                    value = which(singular, linear, spherical).asType(value.dtype)
                }
            }
        }
        eval(value)
        return storeStageLatent(value, width: width, height: height)
    }

    /// Estimated peak active bytes for sampling with live handles retained.
    /// Uses the same family estimator (calibration is still pending) as request-level generation.
    public func estimatedPeakBytes(for conditioning: FluxConditioning, latent: FluxLatent,
                                   options: FluxStageSampling) throws -> Int {
        guard stageConditionings[conditioning.id] != nil, let source = stageLatents[latent.id],
              let loaded, let loadedModelURL else { throw FluxError.staleHandle }
        let request = ImageGenRequest(prompt: "", width: source.width, height: source.height,
            steps: options.steps, guidance: options.guidance, seed: options.seed,
            outputDir: loadedModelURL, sampler: options.sampler, scheduler: options.scheduler, shift: options.shift)
        let pipeline = try estimatedPeakBytes(for: request, modelURL: loadedModelURL, name: loaded.name)
        return try stagePeakWithRetainedBytes(pipeline)
    }

    public func estimatedPeakBytes(forDecoding latent: FluxLatent) throws -> Int {
        guard let source = stageLatents[latent.id], let loaded, let loadedModelURL else { throw FluxError.staleHandle }
        let estimate = try ImageMemoryEstimator.estimate(model: loadedModelURL, family: loaded.name,
            width: source.width, height: source.height)
        return try stagePeakWithRetainedBytes(estimate.weightBytes + estimate.decodingBytes)
    }

    private func stagePeakWithRetainedBytes(_ pipeline: Int) throws -> Int {
        var bytes = pipeline
        let arrays = stageLatents.values.map(\.value)
            + stageConditionings.values.flatMap { Array($0.value.arrays.values) }
        for array in arrays {
            let (sum, overflow) = bytes.addingReportingOverflow(array.nbytes)
            guard !overflow else { throw FluxError.invalidRequest("stage memory size overflows") }
            bytes = sum
        }
        return bytes
    }

    /// Logical tensor bytes owned by a handle. Shared buffers remain active
    /// until their final owner is released; this is not an allocator-cache size.
    public func retainedBytes(for handle: FluxStageHandle) throws -> Int {
        switch handle {
        case .latent(let handle):
            guard let value = stageLatents[handle.id] else { throw FluxError.staleHandle }
            return value.value.nbytes
        case .conditioning(let handle):
            guard let value = stageConditionings[handle.id] else { throw FluxError.staleHandle }
            return value.value.arrays.values.reduce(0) { $0 + $1.nbytes }
        }
    }

    public func release(_ handles: [FluxStageHandle]) async {
        await acquireOperation()
        defer { releaseOperation() }
        for handle in handles {
            switch handle {
            case .conditioning(let value):
                stageConditionings.removeValue(forKey: value.id)
                stagePromptCache = stagePromptCache.filter { $0.value.id != value.id }
            case .latent(let value): stageLatents.removeValue(forKey: value.id)
            }
        }
        // Arrays are evaluated when stored. Dropping the final owner releases
        // active buffers here; allocator cache remains under the host's policy.
    }

    private func storeStageLatent(_ value: MLXArray, width: Int, height: Int) -> FluxLatent {
        let handle = FluxLatent(width: width, height: height)
        stageLatents[handle.id] = StoredFluxLatent(value: value, width: width, height: height)
        return handle
    }

    private func validateStageSize(width: Int, height: Int, pipeline: any FluxStageModel) throws {
        guard width > 0, height > 0, width % pipeline.stageSpatialScale == 0,
              height % pipeline.stageSpatialScale == 0 else {
            throw FluxError.invalidRequest("stage size must be positive multiples of \(pipeline.stageSpatialScale)")
        }
        if let loadedModelURL, let loaded {
            let constraints = try limits(for: loadedModelURL, name: loaded.name)
            let area = width.multipliedReportingOverflow(by: height)
            guard !area.overflow, width >= constraints.minDimension, height >= constraints.minDimension,
                  constraints.maxDimension.map({ width <= $0 && height <= $0 }) ?? true,
                  constraints.maxPixels.map({ area.partialValue <= $0 }) ?? true,
                  width % constraints.dimensionMultiple == 0, height % constraints.dimensionMultiple == 0 else {
                throw FluxError.invalidRequest("stage size does not satisfy the model's dimension limits")
            }
        }
    }
}

struct StagePromptKey: Hashable {
    let prompt: String
    let negative: String?
    let adapterIdentity: [String]
}
