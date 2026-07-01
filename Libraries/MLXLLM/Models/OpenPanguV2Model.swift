// Copyright © 2026 Osaurus.
//
// OpenPangu-v2 decoder layer + inner/outer model. Wraps the attention (MLA +
// convs + prepended sinks, in OpenPanguV2.swift) and the MoE/dense MLP with:
//   • MHC 4-stream hyper-connections (attn_mhc_module / mlp_mhc_module per layer
//     + global merge_mhc_module) — see OpenPanguV2MHC.swift.
//   • Sandwich norm: input_layernorm → attn → post_attention_layernorm, then
//     pre_mlp_layernorm → mlp → post_mlp_layernorm; plus block_post_layernorm on
//     the 9 `block_post_layernorm_idx` layers.
//   • Per-layer hybrid attention: DSA (full + indexer) vs SWA (sliding window)
//     is realized entirely in the cache + mask (OpenPanguV2Cache / per-layer
//     window). The DSA lightning indexer itself is wired in a later pass.
//
// The residual stream is (B, L, mhcNumStream, hiddenSize) throughout the stack;
// the model tiles the embedding into `mhcNumStream` copies at the bottom and
// collapses back with merge_mhc before the final norm — mirroring
// DeepseekV4ModelInner (hc_head) exactly.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Decoder layer

final class OpenPanguV2DecoderLayer: Module {
    let layerIdx: Int

    @ModuleInfo(key: "self_attn") var selfAttn: OpenPanguV2Attention
    var mlp: UnaryLayer
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
    @ModuleInfo(key: "pre_mlp_layernorm") var preMlpLayerNorm: RMSNorm
    @ModuleInfo(key: "post_mlp_layernorm") var postMlpLayerNorm: RMSNorm
    /// Present only on the 9 `block_post_layernorm_idx` layers (nil elsewhere).
    @ModuleInfo(key: "block_post_layernorm") var blockPostLayerNorm: RMSNorm?

    @ModuleInfo(key: "attn_mhc_module") var attnMHC: OpenPanguV2MHCModule
    @ModuleInfo(key: "mlp_mhc_module") var mlpMHC: OpenPanguV2MHCModule

    init(_ config: OpenPanguV2Configuration, layerIdx: Int) {
        self.layerIdx = layerIdx
        self._selfAttn.wrappedValue = OpenPanguV2Attention(config)
        // first_k_dense_replace layers (0,1) are dense; the rest are MoE.
        if layerIdx >= config.firstKDenseReplace {
            self.mlp = OpenPanguV2MoE(config)
        } else {
            self.mlp = OpenPanguV2MLP(
                hiddenSize: config.hiddenSize, intermediateSize: config.intermediateSize)
        }
        let eps = config.rmsNormEps
        let h = config.hiddenSize
        self._inputLayerNorm.wrappedValue = RMSNorm(dimensions: h, eps: eps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(dimensions: h, eps: eps)
        self._preMlpLayerNorm.wrappedValue = RMSNorm(dimensions: h, eps: eps)
        self._postMlpLayerNorm.wrappedValue = RMSNorm(dimensions: h, eps: eps)
        if config.blockPostLayernormIdx.contains(layerIdx) {
            // block_post_layernorm normalizes the FLATTENED 4-stream residual
            // (weight is [mhcNumStream*hidden] = 10240, not per-stream 2560).
            self._blockPostLayerNorm.wrappedValue = RMSNorm(
                dimensions: config.mhcNumStream * h, eps: eps)
        }
        self._attnMHC.wrappedValue = OpenPanguV2MHCModule(config)
        self._mlpMHC.wrappedValue = OpenPanguV2MHCModule(config)
    }

    /// `h`: (B, L, mhcNumStream, hiddenSize).
    func callAsFunction(
        _ h: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: OpenPanguV2Cache?
    ) -> MLXArray {
        // ---- Attention (MHC collapse → sandwich norm → attn → expand) ----
        let residualA = h
        let (xA, postA, combA) = attnMHC.collapse(h)
        let attnOut = postAttentionLayerNorm(
            selfAttn(inputLayerNorm(xA), mask: mask, cache: cache))
        var hOut = attnMHC.expand(
            blockOut: attnOut, residual: residualA, post: postA, comb: combA)

        // ---- MLP (MHC collapse → sandwich norm → mlp → expand) ----
        let residualF = hOut
        let (xF, postF, combF) = mlpMHC.collapse(hOut)
        let mlpOut = postMlpLayerNorm(mlp(preMlpLayerNorm(xF)))
        if ProcessInfo.processInfo.environment["OPENPANGU_MHC_TRACE"] != nil {
            FileHandle.standardError.write(Data(
                "[LYR \(layerIdx)] xF=\(xF.shape) mlpOut=\(mlpOut.shape) postF=\(postF.shape) combF=\(combF.shape) residF=\(residualF.shape)\n".utf8))
        }
        hOut = mlpMHC.expand(
            blockOut: mlpOut, residual: residualF, post: postF, comb: combF)

        // ---- Optional per-block post-norm (per-stream over hidden axis) ----
        if ProcessInfo.processInfo.environment["OPENPANGU_MHC_TRACE"] != nil {
            FileHandle.standardError.write(Data(
                "[LYR \(layerIdx)] afterMLPexpand=\(hOut.shape) hasBPLN=\(blockPostLayerNorm != nil)\n".utf8))
        }
        if let bpln = blockPostLayerNorm {
            // Norm the flattened 4-stream residual (weight is [4*hidden]) then
            // restore the (B,L,stream,hidden) shape.
            let (B, L, S, H) = (hOut.dim(0), hOut.dim(1), hOut.dim(2), hOut.dim(3))
            hOut = bpln(hOut.reshaped(B, L, S * H)).reshaped(B, L, S, H)
        }
        return hOut
    }
}

// MARK: - Inner model

public final class OpenPanguV2ModelInner: Module {
    let config: OpenPanguV2Configuration
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    fileprivate var layers: [OpenPanguV2DecoderLayer]
    @ModuleInfo(key: "merge_mhc_module") var mergeMHC: OpenPanguV2MergeMHC
    @ModuleInfo(key: "norm") var norm: RMSNorm

    init(_ config: OpenPanguV2Configuration) {
        self.config = config
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        self.layers = (0..<config.numHiddenLayers).map {
            OpenPanguV2DecoderLayer(config, layerIdx: $0)
        }
        self._mergeMHC.wrappedValue = OpenPanguV2MergeMHC(config)
        self._norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        // embed: (B, L) → (B, L, H) → tile to (B, L, mhcNumStream, H).
        var h = embedTokens(inputs).expandedDimensions(axis: -2)
        h = repeated(h, count: config.mhcNumStream, axis: -2)

        // Flattened view for mask sizing (createAttentionMask reads dim(1)=L).
        let hFlat = h.reshaped(h.dim(0), h.dim(1), -1)

        for (i, layer) in layers.enumerated() {
            let opCache = cache?[i] as? OpenPanguV2Cache
            // DSA layers: full causal. SWA layers: sliding window.
            let window: Int? = config.isSlidingLayer(i) ? config.slidingWindowFor(i) : nil
            let mask = createAttentionMask(
                h: hFlat, cache: opCache, windowSize: window, returnArray: true)
            h = layer(h, mask: mask, cache: opCache)
        }

        // Collapse the mhcNumStream streams → (B, L, H), then final norm.
        return norm(mergeMHC(h))
    }
}

// MARK: - Outer model

public final class OpenPanguV2Model: Module, LLMModel, KVCacheDimensionProvider, LoRAModel {
    public var kvHeads: [Int]
    let config: OpenPanguV2Configuration
    public var model: OpenPanguV2ModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    public init(_ config: OpenPanguV2Configuration) {
        self.config = config
        // Attention caches the fully-expanded K/V (numAttentionHeads), not the
        // compressed latent — so the allocator must size for all heads.
        self.kvHeads = Array(repeating: config.numAttentionHeads, count: config.numHiddenLayers)
        self.model = OpenPanguV2ModelInner(config)
        self._lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabSize, bias: false)
    }

    /// Per-layer hybrid cache: DSA (full attention + indexer pool) layers get an
    /// unbounded `KVCacheSimple`; SWA layers get a `RotatingKVCache` sized to the
    /// layer's sliding window. Both are wrapped by `OpenPanguV2Cache`, which also
    /// carries the 3 causal-conv states (path-dependent) + the indexer pool.
    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        (0..<config.numHiddenLayers).map { layerIdx in
            if config.isSlidingLayer(layerIdx) {
                let win = config.slidingWindowFor(layerIdx)
                return OpenPanguV2Cache(
                    kv: RotatingKVCache(maxSize: win, keep: 0),
                    isDSA: false, slidingWindow: win)
            }
            return OpenPanguV2Cache(
                kv: KVCacheSimple(), isDSA: true, slidingWindow: 0)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        lmHead(model(inputs, cache: cache))
    }

    /// Remap the openpangu-v2 (JANG_2L) bundle key names to the module attribute
    /// paths. The bundle is already in MLX-swift layout — routed experts ship
    /// pre-stacked as `mlp.switch_mlp.{gate,up,down}_proj.*` (matching our
    /// `SwitchGLU`), phi is a quantized `Linear` (`attn_mhc_module.phi.*`), and the
    /// self_attn/MoE projections + embed/lm_head are quantized `Linear`/`Embedding`
    /// (loaded by the standard quant substitution). Only three transforms differ:
    ///   • depthwise conv weight `[C,1,k]` (PyTorch) → `[C,k,1]` (MLX Conv1d), and
    ///     route `*.qa_conv.weight` to the wrapped path `*.qa_conv.conv.weight`.
    ///   • `mlp.e_score_correction_bias` → `mlp.gate.e_score_correction_bias`
    ///     (the router bias ships one level up from where our gate holds it).
    ///   • Drop MTP layers (>= numHiddenLayers) and the DSA `self_attn.indexer.*`
    ///     (both are later passes; the indexer module isn't wired yet).
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        let convSuffixes = ["qa_conv", "compresskv_conv", "o_conv"]

        // phi's packed dim is bit-width AMBIGUOUS (e.g. 2-bit/gs128 vs 8-bit/gs32
        // unpack to the same [24,640]+[24,80] shapes), so the JANG shape-walk
        // mis-infers it and corrupts the projection (mixes → malformed). We KNOW
        // phi's logical input dim (mhcNumStream*hiddenSize), so dequantize it to
        // dense fp16 here — bits = packedCols*32/logicalIn is then exact — and drop
        // the scales/biases so the walk leaves the dense weight alone. Merge phi is
        // already fp16 (no scales) → passes through untouched.
        let phiLogicalIn = config.mhcNumStream * config.hiddenSize

        for (key, value) in weights {
            // Drop MTP layers (46,47,48) and the DSA indexer (later passes).
            if let li = Self.layerIndex(of: key), li >= config.numHiddenLayers { continue }
            if key.contains(".indexer.") { continue }
            if key.contains("rotary_emb.inv_freq") { continue }

            // MHC phi: dequantize when quantized (attn/mlp), pass through fp16
            // (merge), and rename `<base>.phi.weight` → `<base>.phi` to match the
            // raw `phi` param (so the config quant dict can't substitute it).
            if key.contains("mhc_module.phi.") {
                if key.hasSuffix(".phi.scales") || key.hasSuffix(".phi.biases") { continue }
                if key.hasSuffix(".phi.weight") {
                    let base = String(key.dropLast(".weight".count))  // …phi
                    if let scales = weights["\(base).scales"], let packed = value.shape.last {
                        let bits = packed * 32 / phiLogicalIn
                        let gs = phiLogicalIn / (scales.shape.last ?? 1)
                        out[base] = dequantized(
                            value, scales: scales, biases: weights["\(base).biases"],
                            groupSize: gs, bits: bits)
                    } else {
                        out[base] = value  // merge phi (fp16, no scales)
                    }
                    continue
                }
            }

            // Depthwise conv weight axis reorder + wrap-path route.
            if convSuffixes.contains(where: { key.hasSuffix(".self_attn.\($0).weight") }) {
                let prefix = String(key.dropLast(".weight".count))  // …self_attn.<base>
                out["\(prefix).conv.weight"] = value.ndim == 3 ? value.transposed(0, 2, 1) : value
                continue
            }

            // Router bias lives at `mlp.e_score_correction_bias`; our gate holds it
            // at `mlp.gate.e_score_correction_bias`.
            if key.hasSuffix(".mlp.e_score_correction_bias") {
                let prefix = String(key.dropLast(".e_score_correction_bias".count))
                out["\(prefix).gate.e_score_correction_bias"] = value
                continue
            }

            out[key] = value
        }
        return out
    }

    /// Extract the decoder-layer index from a `model.layers.N.*` key, if any.
    private static func layerIndex(of key: String) -> Int? {
        guard let r = key.range(of: "model.layers.") else { return nil }
        let rest = key[r.upperBound...]
        let digits = rest.prefix { $0.isNumber }
        return digits.isEmpty ? nil : Int(digits)
    }

    public var loraLayers: [Module] {
        model.layers
    }
}
