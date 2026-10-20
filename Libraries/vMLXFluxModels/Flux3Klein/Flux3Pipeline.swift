import Foundation
@preconcurrency import MLX
import MLXRandom
import vMLXFluxKit

final class Flux3Pipeline: @unchecked Sendable {
    let bundle: Flux3Bundle
    let transformer: Flux3Transformer
    let textEncoder: Flux3TextEncoder
    let vae: Flux3VAE

    init(modelPath: URL, bundle: Flux3Bundle) async throws {
        self.bundle = bundle
        // Caller has completed metadata preflight before this first tensor load.
        try Task.checkCancellation()
        try FluxInterruptScope.check()
        _ = try Flux3WeightManifest.validate(modelPath: modelPath, bundle: bundle)
        let loaded = try WeightLoader.load(from: modelPath)
        try Task.checkCancellation()
        try FluxInterruptScope.check()
        let store = MFluxStore(loaded)
        transformer = try Flux3Transformer(store: store, cfg: bundle.transformerConfig)
        vae = try Flux3VAE(store: store)
        textEncoder = try await Flux3TextEncoder(store: store, modelPath: modelPath, bundle: bundle)
    }

    struct References {
        let tokens: MLXArray?
        let ids: [[Float]]
        let width, height: Int
        let crop: Flux3ReferencePlan.Box?
    }

    func references(_ sources: [URL], width: Int?, height: Int?) throws -> References {
        var prepared: [Flux3Pixels] = []
        var firstPlan: Flux3ReferencePlan?
        let single = sources.count == 1
        for source in sources {
            try FluxInterruptScope.check()
            let original = try Flux3ImageReader.read(source)
            let size = Flux3ReferencePlan.Size(original.width, original.height)
            let plan = try single ? Flux3ReferencePlan.single(size, limits: bundle.stamp.limits)
                : Flux3ReferencePlan.multiple(size, limits: bundle.stamp.limits)
            if firstPlan == nil { firstPlan = plan }
            prepared.append(try original.prepared(plan: plan, single: single))
        }
        guard let firstPlan else { throw FluxError.invalidRequest("FLUX 3 editing requires references") }
        let output = firstPlan.output(width: width, height: height, single: single)
        try bundle.limits.validateEdit(width: output.canvas.width, height: output.canvas.height, steps: 1,
            guidance: 1, format: .png, sourceCount: sources.count, hasMask: false, strength: nil)
        var tokens: [MLXArray] = [], ids: [[Float]] = []
        for (index, image) in prepared.enumerated() {
            try FluxInterruptScope.check()
            let encoded = try vae.encodeTokens(MLXArray(image.normalizedNCHW, [1, 3, image.height, image.width]))
            try withError { _ in eval(encoded) }
            tokens.append(encoded)
            ids += Flux3PromptPlan.imageIDs(height: image.height / 16, width: image.width / 16, layer: -(index + 1))
        }
        return References(tokens: concatenated(tokens, axis: 1), ids: ids, width: output.canvas.width,
            height: output.canvas.height, crop: output.crop)
    }

    func sample(prompt: String, width: Int, height: Int, steps: Int, seed: UInt64,
                references: References?, preview: ImagePreviewEmitter?,
                observe: ((String, MLXArray) throws -> Void)? = nil,
                progress: (Int, Int, Double?) -> Void) throws -> (MLXArray, EffectiveImageParameters) {
        try FluxInterruptScope.check()
        let start = Date()
        let text = try textEncoder.encode(prompt)
        let h = height / 16, w = width / 16
        // Per-call key, no global random-state mutation. Euler state remains Float32.
        var latent = MLXRandom.normal([1, h * w, 128], key: MLXRandom.key(seed)).asType(.bfloat16).asType(.float32)
        // Internal parity hook: absent in ordinary generation; never changes tensors.
        try observe?("text", text.tokens)
        try observe?("noise", latent)
        if let tokens = references?.tokens { try observe?("references", tokens) }
        let imageIDs = Flux3PromptPlan.imageIDs(height: h, width: w)
        let schedule = try Flux3Schedule(steps: steps, alpha: bundle.scheduler.alpha)
        let denoiseStart = Date()
        for i in 0..<steps {
            try FluxInterruptScope.check()
            let velocity = try transformer(latent, imageIDs: imageIDs, text: text.tokens, textIDs: text.ids,
                timestep: schedule.timesteps[i], targetTimestep: schedule.timesteps[i + 1],
                references: references?.tokens, referenceIDs: references?.ids ?? [])
            if i == 0 { try observe?("step0-velocity", velocity) }
            latent = latent + (schedule.timesteps[i + 1] - schedule.timesteps[i]) * velocity
            try withError { _ in eval(latent) }
            try FluxInterruptScope.check()
            let elapsed = Date().timeIntervalSince(denoiseStart)
            progress(i + 1, steps, elapsed / Double(i + 1) * Double(steps - i - 1))
            if let preview {
                preview.emit(step: i + 1) {
                    let spatial = latent.reshaped([1, h, w, 128]).transposed(0, 3, 1, 2)
                    let reduced = preview.reducedLatent(spatial, scale: 16)
                    let ph = reduced.dim(2), pw = reduced.dim(3)
                    let packed = reduced.transposed(0, 2, 3, 1).reshaped([1, ph * pw, 128])
                    guard let raw = try? vae.decodeTokens(packed, height: ph * 16, width: pw * 16, materializeStages: false) else { return nil }
                    return clip(raw / 2 + 0.5, min: Float(0), max: Float(1)).asType(.float32)
                }
            }
        }
        let denoiseSeconds = Date().timeIntervalSince(denoiseStart)
        try observe?("latents", latent)
        var image = try vae.decodeTokens(latent, height: height, width: width)
        image = clip(image / 2 + 0.5, min: Float(0), max: Float(1)).asType(.float32)
        if let crop = references?.crop { image = image[0..., 0..., crop.top..<crop.bottom, crop.left..<crop.right] }
        try withError { _ in eval(image) }
        try FluxInterruptScope.check()
        let elapsed = Date().timeIntervalSince(start)
        print("[Flux3] steps=\(steps) prompt_sample_decode_seconds=\(elapsed) denoise_steps_per_second=\(Double(steps) / denoiseSeconds) guidance=none scheduler=flow-map")
        return (image, EffectiveImageParameters(width: image.dim(3), height: image.dim(2), steps: steps,
            guidance: 1, seed: seed, sampler: "euler", scheduler: "flow-map"))
    }
}
