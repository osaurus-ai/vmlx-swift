// Architecture port from NaiveAI/Naive-N0.5-Flash, revision 0235b3b5.
// Kept unregistered until cache/batch and real-bundle gates are complete.
import Foundation
import MLX
import MLXLMCommon
import MLXNN

final class NaiveN05FlashIndexer: Module {
    let config: NaiveN05ArchitectureContract
    let wq: Linear
    let wk: Linear
    @ModuleInfo(key: "k_norm") var keyNorm: LayerNorm
    @ModuleInfo(key: "weights_proj") var weightsProjection: Linear
    init(_ c: NaiveN05ArchitectureContract) {
        config = c
        wq = Linear(c.hiddenDimensions, c.indexerHeads * c.indexerDimensions, bias: false)
        wk = Linear(c.hiddenDimensions, c.indexerDimensions, bias: false)
        _keyNorm.wrappedValue = LayerNorm(dimensions: c.indexerDimensions, eps: 1e-5, affine: true, bias: true)
        _weightsProjection.wrappedValue = Linear(c.hiddenDimensions, c.indexerHeads, bias: false)
    }
    func project(_ states: MLXArray, positions: MLXArray, rotaryTables: NaiveN05FlashMath.RotaryTables? = nil) -> (MLXArray, MLXArray, MLXArray) {
        let c = config
        var q = wq(states).reshaped(states.dim(0), states.dim(1), c.indexerHeads, c.indexerDimensions).transposed(0, 2, 1, 3)
        var k = keyNorm(wk(states)).expandedDimensions(axis: 1)
        q = NaiveN05FlashMath.rotary(q, positions: positions, dimensions: c.fullAttention.rotaryDimensions, theta: c.fullAttention.ropeTheta, tables: rotaryTables)
        k = NaiveN05FlashMath.rotary(k, positions: positions, dimensions: c.fullAttention.rotaryDimensions, theta: c.fullAttention.ropeTheta, tables: rotaryTables)
        if c.indexerPrecision == .fp8E4M3 {
            q = NaiveN05FlashMath.roundIndexerFP8(q)
            k = NaiveN05FlashMath.roundIndexerFP8(k)
        }
        let weights = weightsProjection(states) * Float(1 / sqrt(Double(c.indexerHeads)))
        // Cache representation is F32 also for bf16 mode; values are unchanged.
        return (q.asType(.float32), k.asType(.float32), weights.asType(.float32))
    }
    func scores(query: MLXArray, keys: MLXArray, weights: MLXArray) -> MLXArray {
        let scores = maximum(matmul(query, keys.swappedAxes(-1, -2)), 0)
        return (scores * weights.transposed(0, 2, 1).expandedDimensions(axis: -1)).sum(axis: 1)
    }
}

final class NaiveN05FlashAttention: Module {
    let config: NaiveN05ArchitectureContract
    let geometry: NaiveN05ArchitectureContract.Attention
    let sliding: Bool
    let allowedMaskGPUArange: Bool
    @ModuleInfo(key: "q_proj") var query: Linear
    @ModuleInfo(key: "k_proj") var key: Linear
    @ModuleInfo(key: "v_proj") var value: Linear
    @ModuleInfo(key: "o_proj") var output: Linear
    let indexer: NaiveN05FlashIndexer?
    @ParameterInfo(key: "attention_sink_bias") var sink: MLXArray?
    init(_ c: NaiveN05ArchitectureContract, layer: Int, allowedMaskGPUArange: Bool = false) {
        config = c
        self.allowedMaskGPUArange = allowedMaskGPUArange
        sliding = c.attentionKinds[layer] == .sliding
        geometry = sliding ? c.slidingAttention : c.fullAttention
        let g = geometry
        _query.wrappedValue = Linear(c.hiddenDimensions, g.heads * g.keyDimensions, bias: c.attentionBias)
        _key.wrappedValue = Linear(c.hiddenDimensions, g.kvHeads * g.keyDimensions, bias: c.attentionBias)
        _value.wrappedValue = Linear(c.hiddenDimensions, g.kvHeads * g.valueDimensions, bias: c.attentionBias)
        _output.wrappedValue = Linear(g.heads * g.valueDimensions, c.hiddenDimensions, bias: false)
        _sink.wrappedValue = g.hasSink ? MLXArray.zeros([g.heads]) : nil
        indexer = sliding ? nil : NaiveN05FlashIndexer(c)
    }
    func callAsFunction(_ states: MLXArray, positions: MLXArray, padding: MLXArray, cache: NaiveN05FlashCache?, rotaryTables: NaiveN05FlashMath.RotaryTables? = nil) throws -> MLXArray {
        let g = geometry, batch = states.dim(0), length = states.dim(1)
        let past = cache?.offset ?? 0
        let keyOffset = cache?.keyOffset ?? 0
        var q = query(states).reshaped(batch, length, g.heads, g.keyDimensions).transposed(0, 2, 1, 3)
        var k = key(states).reshaped(batch, length, g.kvHeads, g.keyDimensions).transposed(0, 2, 1, 3)
        var v = value(states).reshaped(batch, length, g.kvHeads, g.valueDimensions).transposed(0, 2, 1, 3)
        q = NaiveN05FlashMath.rotary(q, positions: positions, dimensions: g.rotaryDimensions, theta: g.ropeTheta, tables: rotaryTables)
        k = NaiveN05FlashMath.rotary(k, positions: positions, dimensions: g.rotaryDimensions, theta: g.ropeTheta, tables: rotaryTables)
        let projectedIndex = indexer?.project(states, positions: positions, rotaryTables: rotaryTables)
        var indexKeys = projectedIndex?.1
        if let cache {
            let full = try cache.append(keys: k, values: v, indexer: indexKeys)
            k = full.0; v = full.1; indexKeys = full.2
        }
        var allowed = NaiveN05FlashMath.allowedMask(padding: padding, queryOffset: past, length: length,
            keyOffset: keyOffset, keyLength: k.dim(2), window: sliding ? config.window : nil,
            gpuArange: allowedMaskGPUArange)
        if let indexer, let projectedIndex, let indexKeys {
            let scores = indexer.scores(query: projectedIndex.0, keys: indexKeys, weights: projectedIndex.2)
            allowed = NaiveN05FlashMath.sparseMask(scores: scores, allowed: allowed, topK: config.indexerTopK)
        }
        let result = NaiveN05FlashMath.attention(query: q, key: k, value: v, allowed: allowed,
            sink: sink, valueScale: config.valueScale.map(Float.init))
        return output(result.transposed(0, 2, 1, 3).reshaped(batch, length, -1))
    }
}

final class NaiveN05FlashRouter: Module {
    let config: NaiveN05ArchitectureContract
    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "e_score_correction_bias") var correction: MLXArray
    init(_ c: NaiveN05ArchitectureContract) {
        config = c
        _weight.wrappedValue = MLXArray.zeros([c.expertCount, c.hiddenDimensions], dtype: .float32)
        _correction.wrappedValue = MLXArray.zeros([c.expertCount], dtype: .float32)
    }
    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        let probabilities = sigmoid(matmul(x.asType(.float32), weight.asType(.float32).T))
        let indices = argPartition(-(probabilities + correction.asType(.float32)), kth: config.routes - 1, axis: -1)[.ellipsis, ..<config.routes]
        var scores = takeAlong(probabilities, indices, axis: -1)
        if config.normalizeRoutes { scores = scores / (scores.sum(axis: -1, keepDims: true) + 1e-20) }
        return (indices, scores * Float(config.routingScale))
    }
}

final class NaiveN05FlashDenseMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    init(_ c: NaiveN05ArchitectureContract) {
        _gate.wrappedValue = Linear(c.hiddenDimensions, c.denseDimensions, bias: false)
        _up.wrappedValue = Linear(c.hiddenDimensions, c.denseDimensions, bias: false)
        _down.wrappedValue = Linear(c.denseDimensions, c.hiddenDimensions, bias: false)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { down(silu(gate(x)) * up(x)) }
}

final class NaiveN05FlashMoE: Module, UnaryLayer {
    let routedAdviceLayerIndex: Int?
    let gate: NaiveN05FlashRouter
    // Separate projections keep original gate/up/down names and affine loading.
    @ModuleInfo(key: "switch_mlp") var experts: Module
    init(_ c: NaiveN05ArchitectureContract, experts: (Module & WeightedRoutedExpertLayer)? = nil,
         layerIndex: Int? = nil) {
        routedAdviceLayerIndex = layerIndex
        gate = NaiveN05FlashRouter(c)
        if let experts {
            _experts.wrappedValue = experts
        } else {
            _experts.wrappedValue = SwitchGLU(inputDims: c.hiddenDimensions,
                hiddenDims: c.expertDimensions, numExperts: c.expertCount)
        }
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (indices, scores) = gate(x)
        if let layer = routedAdviceLayerIndex {
            // Preserve explicit advisor enablement and its bounded readback policy.
            JangPressCanonicalExpertAdvisor.shared.observe(layer: layer, indices: indices)
        }
        // Custom format construction supplies its complete weighted operation.
        // The ordinary affine branch retains Naive-specific contribution rounding.
        guard let affine = experts as? SwitchGLU else {
            return (experts as! any WeightedRoutedExpertLayer).callRouted(x, indices: indices, scores: scores)
        }
        let selected = affine(x, indices)
        let weighted = (selected.asType(.float32) * scores.expandedDimensions(axis: -1)).asType(x.dtype)
        return weighted.sum(axis: -2).asType(x.dtype)
    }
}

final class NaiveN05FlashDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: NaiveN05FlashAttention
    @ModuleInfo(key: "input_layernorm") var inputNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postNorm: RMSNorm
    let mlp: UnaryLayer
    init(_ c: NaiveN05ArchitectureContract, layer: Int, routed: (Module & WeightedRoutedExpertLayer)?,
         allowedMaskGPUArange: Bool) {
        _attention.wrappedValue = NaiveN05FlashAttention(c, layer: layer,
            allowedMaskGPUArange: allowedMaskGPUArange)
        _inputNorm.wrappedValue = RMSNorm(dimensions: c.hiddenDimensions, eps: Float(c.normEpsilon))
        _postNorm.wrappedValue = RMSNorm(dimensions: c.hiddenDimensions, eps: Float(c.normEpsilon))
        mlp = c.routedLayers[layer] ? NaiveN05FlashMoE(c, experts: routed, layerIndex: layer) : NaiveN05FlashDenseMLP(c)
    }
    func callAsFunction(_ x: MLXArray, positions: MLXArray, padding: MLXArray, cache: NaiveN05FlashCache?, rotaryTables: NaiveN05FlashMath.RotaryTables? = nil) throws -> MLXArray {
        let h = try x + attention(inputNorm(x), positions: positions, padding: padding, cache: cache, rotaryTables: rotaryTables)
        return h + mlp(postNorm(h))
    }
}

final class NaiveN05FlashBackbone: Module {
    @ModuleInfo(key: "embed_tokens") var embedding: Embedding
    let layers: [NaiveN05FlashDecoderLayer]
    let norm: RMSNorm
    init(_ c: NaiveN05ArchitectureContract, routedFactory: NaiveN05FlashModel.RoutedFactory?,
         allowedMaskGPUArange: Bool) throws {
        _embedding.wrappedValue = Embedding(embeddingCount: c.vocabularySize, dimensions: c.hiddenDimensions)
        layers = try (0..<c.layerCount).map { layer in
            let routed = c.routedLayers[layer] ? try routedFactory?(layer, c) : nil
            return NaiveN05FlashDecoderLayer(c, layer: layer, routed: routed,
                allowedMaskGPUArange: allowedMaskGPUArange)
        }
        norm = RMSNorm(dimensions: c.hiddenDimensions, eps: Float(c.normEpsilon))
    }
}

/// Unregistered reference graph. The runtime bridge separately admits unpadded
/// single-sequence input and validates model-owned companion caches.
final class NaiveN05FlashModel: Module {
    typealias RoutedFactory = (Int, NaiveN05ArchitectureContract) throws -> (Module & WeightedRoutedExpertLayer)
    let config: NaiveN05ArchitectureContract
    /// Exact Int32 mask positions; captured once, with no cache-policy change.
    let allowedMaskGPUArange: Bool
    /// Exact apply-only rotary policy; captured once for qualified hardware.
    let fusedRotaryApply: Bool
    let excludedSafetensorsKeys: Set<String>
    let model: NaiveN05FlashBackbone
    @ModuleInfo(key: "lm_head") var head: Linear
    init(_ c: NaiveN05ArchitectureContract, routedFactory: RoutedFactory? = nil,
        excludedSafetensorsKeys: Set<String> = [], allowedMaskGPUArange: Bool? = nil,
        fusedRotaryApply: Bool? = nil) throws {
        let canonicalExclusions = Set(c.routedLayers.indices.filter { c.routedLayers[$0] }.flatMap { layer in
            ["gate_proj", "up_proj", "down_proj"].flatMap { role in
                ["tq2_packed", "tq2_scales"].map {
                    "model.layers.\(layer).mlp.switch_mlp.\(role).\($0)"
                }
            }
        })
        guard excludedSafetensorsKeys.isEmpty
            || (routedFactory != nil && excludedSafetensorsKeys == canonicalExclusions)
        else { throw NaiveN05FlashCache.Failure.invalidGeometry }
        self.excludedSafetensorsKeys = excludedSafetensorsKeys
        config = c
        let allowedMaskGPUArange = allowedMaskGPUArange
            ?? NaiveN05FlashMath.allowedMaskGPUArangeRequested(environment: ProcessInfo.processInfo.environment)
        self.allowedMaskGPUArange = allowedMaskGPUArange
        self.fusedRotaryApply = fusedRotaryApply
            ?? NaiveN05FusedRotaryApply.modelRequested(c, environment: ProcessInfo.processInfo.environment)
        model = try NaiveN05FlashBackbone(c, routedFactory: routedFactory,
            allowedMaskGPUArange: allowedMaskGPUArange)
        _head.wrappedValue = Linear(c.hiddenDimensions, c.vocabularySize, bias: false)
    }
    func newCache() -> [NaiveN05FlashCache] {
        config.attentionKinds.map { kind in
            NaiveN05FlashCache(window: kind == .sliding ? config.window : nil, requiresIndexer: kind == .sparse)
        }
    }
    func callAsFunction(_ tokens: MLXArray, padding suppliedPadding: MLXArray? = nil, positions suppliedPositions: MLXArray? = nil, cache: [NaiveN05FlashCache]? = nil) throws -> MLXArray {
        let past = cache?.first?.offset ?? 0
        guard tokens.ndim == 2, cache == nil || (cache!.count == config.layerCount && cache!.allSatisfy { $0.offset == past }) else { throw NaiveN05FlashCache.Failure.invalidGeometry }
        let padding = suppliedPadding ?? MLXArray.ones([tokens.dim(0), past + tokens.dim(1)], dtype: .bool)
        guard padding.shape == [tokens.dim(0), past + tokens.dim(1)] else { throw NaiveN05FlashCache.Failure.invalidGeometry }
        let positions = suppliedPositions ?? maximum(cumsum(padding.asType(.int32), axis: -1) - 1, 0)[0..., past...]
        guard positions.shape == tokens.shape else { throw NaiveN05FlashCache.Failure.invalidGeometry }
        let rotaryTables = NaiveN05FlashMath.RotaryTables(
            positions: positions, fusedApply: fusedRotaryApply)
        var h = model.embedding(tokens)
        for (i, layer) in model.layers.enumerated() { h = try layer(h, positions: positions, padding: padding, cache: cache?[i], rotaryTables: rotaryTables) }
        return head(model.norm(h))
    }
}

extension NaiveN05FlashModel: SafetensorsLoadKeyExcluding {
    func excludeFromGenericSafetensorsLoad(key: String) -> Bool {
        excludedSafetensorsKeys.contains(key)
    }
    var requiresExactTensorMmapBuffers: Bool { !excludedSafetensorsKeys.isEmpty }
}
