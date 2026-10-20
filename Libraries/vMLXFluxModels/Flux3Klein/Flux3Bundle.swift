import Foundation
import vMLXFluxKit

/// Configuration-only preflight. Never opens a tensor or allocates an MLX array.
public struct Flux3Bundle: Sendable {
    public struct Stamp: Decodable, Sendable {
        public struct Defaults: Decodable, Sendable {
            public let steps: Int
            public let guidance: Float?
            public let width: Int
            public let height: Int
        }
        public struct Limits: Decodable, Sendable {
            public let sizeMultiple, minSide, maxSide, maxReferenceImages: Int
            public let singleReferenceMaxPixels, multiReferenceMaxPixelsEach: Int
        }
        public struct Features: Decodable, Sendable {
            public let hiddenStateLayers: [Int]
            public let padMultiple, maxTokensMflux, maxTokensReference: Int
            public let chatTemplate: String
        }
        public struct Capabilities: Decodable, Sendable {
            public let textToImage, imageEdit, multipleSourceImages, negativePrompt, guidance: Bool
        }
        public struct Quantization: Decodable, Sendable {
            public let library, mode: String
            public let bits, groupSize: Int
        }
        public let family, variant, mfluxModel: String
        public let defaults: Defaults
        public let limits: Limits
        public let textFeatures: Features
        public let capabilities: Capabilities
        public let quantization: Quantization
    }
    public struct Scheduler: Decodable, Sendable {
        public let numInferenceSteps: Int
        public let alpha: Float
        public let flowMap: Bool
        let className: String
        enum CodingKeys: String, CodingKey {
            case numInferenceSteps, alpha, flowMap
            case className = "_className"
        }
    }
    public let stamp: Stamp
    public let scheduler: Scheduler
    let transformerConfig: Flux3TransformerConfig

    public init(modelPath: URL) throws {
        do {
            let decoder = JSONDecoder()
            decoder.keyDecodingStrategy = .convertFromSnakeCase
            struct Index: Decodable { let osaurusImage: Stamp }
            stamp = try decoder.decode(Index.self, from: Data(contentsOf: modelPath.appendingPathComponent("model_index.json"))).osaurusImage
            scheduler = try decoder.decode(Scheduler.self, from: Data(contentsOf: modelPath.appendingPathComponent("scheduler/scheduler_config.json")))
            transformerConfig = try decoder.decode(Flux3TransformerConfig.self, from: Data(contentsOf: modelPath.appendingPathComponent("transformer/config.json")))
            var reasons: [String] = []
            let t = transformerConfig
            if t.numLayers != 5 || t.numSingleLayers != 28 || t.attentionHeadDim != 128 || t.numAttentionHeads != 24
                || t.inChannels != 128 || t.condInChannels != 128 || t.jointAttentionDim != 20480
                || t.timestepChannels != 256 || t.mlpRatio != 3 || t.axesDimsRope != [32, 32, 32, 32]
                || !t.flowMap || t.ropeTheta != 10000 {
                reasons.append("unsupported FLUX 3 transformer architecture")
            }
            for (component, expectedClass) in [("transformer", "Flux3Transformer2DModel"), ("vae", "AutoencoderKLFlux2")] {
                let raw = try JSONSerialization.jsonObject(with: Data(contentsOf: modelPath.appendingPathComponent("\(component)/config.json"))) as? [String: Any]
                if raw?["_class_name"] as? String != expectedClass { reasons.append("wrong \(component) class") }
            }
            if stamp.family != "flux3-image-klein" || stamp.variant != "distilled" || stamp.mfluxModel != stamp.family {
                reasons.append("requires flux3-image-klein distilled stamp")
            }
            if scheduler.className != "Flux3FlowMapScheduler" || !scheduler.flowMap || !scheduler.alpha.isFinite || scheduler.alpha <= 0 {
                reasons.append("requires a positive-alpha Flux3FlowMapScheduler")
            }
            let l = stamp.limits
            if l.sizeMultiple != 16 || l.minSide != 64 || l.maxSide != 4096 || l.maxReferenceImages != 10
                || l.singleReferenceMaxPixels != 4194304 || l.multiReferenceMaxPixelsEach != 1048576 {
                reasons.append("unsupported reference/size contract")
            }
            let f = stamp.textFeatures
            if f.hiddenStateLayers != [4, 8, 12, 16, 20, 24, 28, 32] || f.padMultiple != 80 || f.maxTokensMflux != 4096 {
                reasons.append("unsupported text feature contract")
            }
            let c = stamp.capabilities
            if !c.textToImage || !c.imageEdit || !c.multipleSourceImages || c.negativePrompt || c.guidance || stamp.defaults.guidance != nil {
                reasons.append("requires guidance-distilled generation and editing")
            }
            let q = stamp.quantization
            if q.library != "mflux" || q.mode != "affine" || ![4, 6, 8].contains(q.bits) || q.groupSize != 64 {
                reasons.append("unsupported mflux quantization contract")
            }
            if stamp.defaults.steps < 1 || scheduler.numInferenceSteps != stamp.defaults.steps {
                reasons.append("invalid or inconsistent default steps")
            }
            for side in [stamp.defaults.width, stamp.defaults.height] where side < l.minSide || side > l.maxSide || side % l.sizeMultiple != 0 {
                reasons.append("invalid default dimensions")
            }
            let fm = FileManager.default
            for file in ["text_encoder/tokenizer.json", "text_encoder/tokenizer_config.json", "text_encoder/chat_template.jinja", "transformer/config.json", "vae/config.json"] {
                if !fm.fileExists(atPath: modelPath.appendingPathComponent(file).path) { reasons.append("missing \(file)") }
            }
            struct Language: Decodable, Equatable {
                let numHiddenLayers, hiddenSize, intermediateSize, numAttentionHeads: Int
                let numKeyValueHeads, headDim, vocabSize: Int
                let ropeTheta, rmsNormEps: Float
                let attentionBias: Bool
                let hiddenAct: String
            }
            struct Text: Decodable { let textConfig: Language? }
            let textData = try Data(contentsOf: modelPath.appendingPathComponent("text_encoder/config.json"))
            let language = try decoder.decode(Language.self, from: textData)
            let nested = try decoder.decode(Text.self, from: textData).textConfig
            if language.numHiddenLayers != 32 || language.hiddenSize != 2560 || language.intermediateSize != 9728
                || language.numAttentionHeads != 32 || language.numKeyValueHeads != 8 || language.headDim != 128
                || language.vocabSize != 151936 || language.ropeTheta != 5_000_000 || language.rmsNormEps != 1e-6
                || language.attentionBias || language.hiddenAct != "silu" || nested.map({ $0 != language }) == true {
                reasons.append("unsupported or inconsistent 32-layer Qwen3-VL language configuration")
            }
            struct VAE: Decodable {
                let latentChannels, layersPerBlock, normNumGroups, inChannels, outChannels: Int
                let blockOutChannels, patchSize: [Int]
                let decoderBlockOutChannels: [Int]?
                let batchNormEps: Float
                let useQuantConv, usePostQuantConv, midBlockAddAttention: Bool
                let actFn: String
            }
            let vae = try decoder.decode(VAE.self, from: Data(contentsOf: modelPath.appendingPathComponent("vae/config.json")))
            if vae.latentChannels != 32 || vae.layersPerBlock != 2 || vae.normNumGroups != 32
                || vae.inChannels != 3 || vae.outChannels != 3 || vae.blockOutChannels != [128, 256, 512, 512]
                || vae.decoderBlockOutChannels.map({ $0 != vae.blockOutChannels }) == true
                || vae.patchSize != [2, 2] || vae.batchNormEps != 1e-4
                || !vae.useQuantConv || !vae.usePostQuantConv || !vae.midBlockAddAttention || vae.actFn != "silu" {
                reasons.append("unsupported FLUX 3 VAE architecture")
            }
            // Resolve every indexed shard inside its own component, before WeightLoader.
            struct Weights: Decodable { let weightMap: [String: String] }
            for component in ["transformer", "text_encoder", "vae"] {
                let root = modelPath.appendingPathComponent(component)
                let index = try decoder.decode(Weights.self, from: Data(contentsOf: root.appendingPathComponent("model.safetensors.index.json")))
                if index.weightMap.isEmpty { reasons.append("empty \(component) index") }
                let sentinels: [String: [String]] = [
                    "transformer": ["target_timestep_embedder.linear_1.weight", "mode_blocks.image.4.ff.linear_out.weight", "single_transformer_blocks.27.attn.to_q.weight"],
                    "text_encoder": ["embed_tokens.weight", "layers.31.self_attn.q_proj.weight"],
                    "vae": ["bn.running_mean", "bn.running_var", "quant_conv.weight", "post_quant_conv.weight"],
                ]
                for key in sentinels[component, default: []] where index.weightMap[key] == nil {
                    reasons.append("missing \(component) indexed weight \(key)")
                }
                for shard in Set(index.weightMap.values) {
                    if shard.contains("/") || shard.contains("\\") || !shard.hasSuffix(".safetensors") {
                        reasons.append("invalid \(component) shard path")
                    } else if !fm.fileExists(atPath: root.appendingPathComponent(shard).path) {
                        reasons.append("missing \(component)/\(shard)")
                    }
                }
            }
            if !reasons.isEmpty { throw FluxError.localModelIncomplete(modelPath, reasons: reasons) }
        } catch let error as FluxError { throw error }
        catch { throw FluxError.localModelIncomplete(modelPath, reasons: ["FLUX 3 bundle metadata: \(error)"]) }
    }

    public var limits: FluxModelLimits {
        FluxModelLimits(dimensionMultiple: stamp.limits.sizeMultiple, minDimension: stamp.limits.minSide,
            maxDimension: stamp.limits.maxSide, minimumSteps: 1,
            supportsNegativePrompt: false, supportsEditNegativePrompt: false, ignoresNegativePrompt: true, supportsStrength: false,
            supportsMask: false, maxSourceImages: stamp.limits.maxReferenceImages, supportedFormats: [.png],
            samplers: ["euler"], schedulers: ["flow-map"], defaultSampler: "euler", defaultScheduler: "flow-map")
    }
}
