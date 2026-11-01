// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXFast
import MLXLMCommon
import MLXNN

// MiniMax-M3 / M3.1 DSpark: the block drafter a bundle ships in `dspark/` (`DSparkMiniMaxDraftModel`) and the target
// side that feeds and verifies it. Port of the vMLX Python runtime (perf/minimax-m3-msa-select 630929ed):
// `vmlx_engine/dspark/drafter.py` (`minimax_config`, `DSparkDrafter` in its MiniMax flavor, `_load_minimax_drafter`,
// `minimax_tensor_name`, the pre-quantized form `write_quantized_minimax_drafter` writes) and
// `vmlx_engine/dspark/minimax_target.py` (taps, verify, rollback; the verify is MiniMaxM3Verify.swift).
//
// The drafter (DFlash block drafter + DSpark Markov / confidence heads, MiniMax layers):
// - context: the residual stream after the target layers `dspark_target_layer_ids` (before the final norm),
//   concatenated -> `fc` -> `hidden_norm` ((1 + w) RMS norm); each draft layer projects it with its own k / v
//   projections (k with the per-head (1 + w) `k_norm` and partial RoPE over `rotary_dim` at the context positions);
// - block: [anchor, mask x (span - 1)] embedded by the TARGET's embedding (`dspark_noise_token_id` is the mask);
//   layers of (1 + w) norms, per-head q / k norms, partial RoPE, a dense SwiGLU-OAI FFN, and sliding-window attention:
//   a block row at position p reads context keys at positions > p - `sliding_window` plus the whole block
//   (non-causal);
// - logits: final norm -> the TARGET's lm_head; block row i predicts the token after it;
// - Markov head: logits_i += markov_w2 . markov_w1[prev_i] (prev_0 = the anchor, prev_i = the token drafted at
//   i - 1), on the top-K base-logit candidates of each row (`VMLX_DSPARK_MARKOV_CANDIDATES`, default 32; 0 = the
//   full vocabulary); confidence head sigmoid(proj([h_i ; markov_w1[prev_i]])) (not read by the Swift controller).
//
// Per request (`MiniMaxM3DSparkSession`): the context K / V of the last `sliding_window` committed positions, rebuilt
// from the tap rows the target cache carries (`MiniMaxM3SparseCache.dsparkTail`, recorded by every forward while a
// drafter is attached), keyed by absolute position: an in-place restore, a stretch of AR steps or a prefix-cache hit
// all leave the cache as the single source of truth. A verify forward runs every row on decode arithmetic
// (MiniMaxM3Verify.swift); a rejection trims every cache lane (K / V, index keys, tap tail): all are append-only and
// anchored to absolute positions, so the next rows overwrite the trimmed tail exactly.
//
// Draft width: Adaptive (`DSparkDecoder`, the DeepSeek-V4.1 controller) picks every step's width. The ladder top is
// Python's `ladder_top`: twice the trained block (the target verifies any width exactly, so there is no verify
// coverage limit); the trained block is only where the width starts. `VMLX_DSPARK_MAX_DRAFT` (ladder top) and
// `VMLX_DSPARK_WIDTH` (pin every step) are diagnostic levers; no bundle stamp or config field caps the width.

// MARK: - Config

/// The MiniMax-M3.1 draft config (`minimax_config` in the Python runtime).
struct MiniMaxM3DSparkConfig: Sendable, Equatable {
    var hiddenSize: Int
    var layers: Int
    var heads: Int
    var kvHeads: Int
    var headDim: Int
    var rotaryDim: Int
    var rmsNormEps: Float
    var vocabularySize: Int
    var intermediateSize: Int
    var swigluAlpha: Float
    var swigluLimit: Float
    var slidingWindow: Int
    var blockSize: Int
    var ropeTheta: Float
    var maskTokenId: Int
    var targetLayerIds: [Int]
    var markovRank: Int

    enum ConfigError: Error, CustomStringConvertible {
        case unsupported(String)
        case missing(String)
        var description: String {
            switch self {
            case .unsupported(let what): return "MiniMax DSpark drafter not supported: \(what)"
            case .missing(let key): return "MiniMax DSpark drafter config has no \(key)"
            }
        }
    }

    /// `DSparkMiniMaxDraftModel`, or a text config carrying `dspark_block_size`.
    static func isMiniMax(_ raw: [String: Any]) -> Bool {
        let architectures = (raw["architectures"] as? [Any])?.compactMap { $0 as? String } ?? []
        let text = raw["text_config"] as? [String: Any] ?? raw
        return architectures.contains("DSparkMiniMaxDraftModel") || text["dspark_block_size"] != nil
    }

    init(raw: [String: Any]) throws {
        let t = raw["text_config"] as? [String: Any] ?? raw
        let dsc = t["dspark_config"] as? [String: Any] ?? [:]
        func value(_ key: String, _ nested: String? = nil) -> Any? {
            if let v = t[key], !(v is NSNull) { return v }
            if let nested, let v = dsc[nested], !(v is NSNull) { return v }
            return nil
        }
        func int(_ v: Any?) -> Int? { (v as? NSNumber)?.intValue ?? (v as? Int) }
        func double(_ v: Any?) -> Double? { (v as? NSNumber)?.doubleValue ?? (v as? Double) }
        func bool(_ v: Any?) -> Bool? { (v as? NSNumber)?.boolValue ?? (v as? Bool) }
        func required(_ key: String, _ nested: String? = nil) throws -> Int {
            guard let v = int(value(key, nested)) else { throw ConfigError.missing(key) }
            return v
        }
        var problems: [String] = []
        if (value("dspark_markov_head_type", "markov_head_type") as? String ?? "vanilla") != "vanilla" {
            problems.append("markov head is not vanilla")
        }
        if !(bool(value("dspark_use_dense_ffn", "use_dense_ffn")) ?? true) { problems.append("MoE draft FFN") }
        if bool(value("dspark_hybrid_context_fusion", "hybrid_context_fusion")) ?? false {
            problems.append("hybrid context fusion")
        }
        if ((t["layer_types"] as? [Any]) ?? []).contains(where: { ($0 as? String) != "sliding_attention" }) {
            problems.append("non-sliding draft layers")
        }
        let confidence = bool(raw["enable_confidence_head"] ?? t["enable_confidence_head"]) ?? false
        let withMarkov = bool(raw["confidence_head_with_markov"] ?? t["confidence_head_with_markov"]) ?? true
        if !(confidence && withMarkov) { problems.append("confidence head without the Markov latent") }
        if !problems.isEmpty { throw ConfigError.unsupported(problems.joined(separator: ", ")) }

        headDim = try required("head_dim")
        hiddenSize = try required("hidden_size")
        layers = try required("num_hidden_layers")
        heads = try required("num_attention_heads")
        kvHeads = try required("num_key_value_heads")
        if let r = int(t["rotary_dim"]), r > 0 {
            rotaryDim = r
        } else {
            rotaryDim = Int(Double(headDim) * (double(t["partial_rotary_factor"]) ?? 1.0))
        }
        rmsNormEps = Float(double(t["rms_norm_eps"]) ?? 1e-6)
        vocabularySize = try required("vocab_size")
        if let v = int(t["dspark_ffn_hidden_size"]), v > 0 {
            intermediateSize = v
        } else if let v = int(dsc["ffn_hidden_size"]), v > 0 {
            intermediateSize = v
        } else {
            intermediateSize = try required("dense_intermediate_size")
        }
        swigluAlpha = Float(double(t["swiglu_alpha"]) ?? 1.702)
        swigluLimit = Float(double(t["swiglu_limit"]) ?? 7.0)
        slidingWindow = try required("sliding_window")
        blockSize = try required("dspark_block_size")
        guard let theta = double(t["rope_theta"]) else { throw ConfigError.missing("rope_theta") }
        ropeTheta = Float(theta)
        maskTokenId = try required("dspark_noise_token_id", "mask_token_id")
        guard let taps = (t["dspark_target_layer_ids"] as? [Any])?.compactMap({ int($0) }), !taps.isEmpty else {
            throw ConfigError.missing("dspark_target_layer_ids")
        }
        targetLayerIds = taps
        markovRank = try required("dspark_markov_rank", "markov_rank")
    }
}

// MARK: - Modules (live paths = the Python drafter's module paths)

final class MiniMaxM3DSparkAttention: Module {
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: MiniMaxM3RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: MiniMaxM3RMSNorm

    init(_ c: MiniMaxM3DSparkConfig) {
        _qProj.wrappedValue = Linear(c.hiddenSize, c.heads * c.headDim, bias: false)
        _kProj.wrappedValue = Linear(c.hiddenSize, c.kvHeads * c.headDim, bias: false)
        _vProj.wrappedValue = Linear(c.hiddenSize, c.kvHeads * c.headDim, bias: false)
        _oProj.wrappedValue = Linear(c.heads * c.headDim, c.hiddenSize, bias: false)
        _qNorm.wrappedValue = MiniMaxM3RMSNorm(dimensions: c.headDim, eps: c.rmsNormEps)
        _kNorm.wrappedValue = MiniMaxM3RMSNorm(dimensions: c.headDim, eps: c.rmsNormEps)
    }
}

final class MiniMaxM3DSparkLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: MiniMaxM3DSparkAttention
    @ModuleInfo(key: "mlp") var mlp: MiniMaxM3MLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: MiniMaxM3RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: MiniMaxM3RMSNorm

    init(_ c: MiniMaxM3DSparkConfig) {
        _selfAttn.wrappedValue = MiniMaxM3DSparkAttention(c)
        _mlp.wrappedValue = MiniMaxM3MLP(
            dimensions: c.hiddenSize, hiddenDimensions: c.intermediateSize, alpha: c.swigluAlpha,
            limit: c.swigluLimit)
        _inputLayerNorm.wrappedValue = MiniMaxM3RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = MiniMaxM3RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
    }
}

final class MiniMaxM3DSparkMarkov: Module {
    @ModuleInfo(key: "markov_w1") var w1: Embedding
    @ModuleInfo(key: "markov_w2") var w2: Linear

    init(vocabulary: Int, rank: Int) {
        _w1.wrappedValue = Embedding(embeddingCount: vocabulary, dimensions: rank)
        _w2.wrappedValue = Linear(rank, vocabulary, bias: false)
    }
}

final class MiniMaxM3DSparkConfidence: Module {
    @ModuleInfo(key: "proj") var proj: Linear

    init(dimensions: Int) {
        _proj.wrappedValue = Linear(dimensions, 1, bias: true)
    }
}

/// Context K / V of the drafter's layers for positions `start ..< end` (`[1, kvHeads, end - start, headDim]` each).
struct MiniMaxM3DSparkContext {
    var keys: [MLXArray]
    var values: [MLXArray]
    var start: Int
    var end: Int
    var count: Int { end - start }

    static func empty(at position: Int) -> Self { Self(keys: [], values: [], start: position, end: position) }

    /// Append rows for positions `end ..< end + n` (`MiniMaxM3DSparkDrafter.contextRows`).
    mutating func append(_ rows: [(MLXArray, MLXArray)]) {
        guard let n = rows.first?.0.dim(2), n > 0 else { return }
        if keys.isEmpty {
            keys = rows.map(\.0)
            values = rows.map(\.1)
        } else {
            keys = zip(keys, rows).map { concatenated([$0, $1.0], axis: 2) }
            values = zip(values, rows).map { concatenated([$0, $1.1], axis: 2) }
        }
        end += n
    }

    /// Keep only the last `n` rows (a sliding-window drafter never reads further back).
    mutating func keepLast(_ n: Int) {
        guard n > 0, count > n else { return }
        let drop = count - n
        keys = keys.map { $0[0..., 0..., drop...] }
        values = values.map { $0[0..., 0..., drop...] }
        start += drop
    }
}

final class MiniMaxM3DSparkDrafter: Module {
    let config: MiniMaxM3DSparkConfig
    let layers: [MiniMaxM3DSparkLayer]
    @ModuleInfo(key: "fc") var fc: Linear
    @ModuleInfo(key: "hidden_norm") var hiddenNorm: MiniMaxM3RMSNorm
    @ModuleInfo(key: "norm") var norm: MiniMaxM3RMSNorm
    @ModuleInfo(key: "markov_head") var markovHead: MiniMaxM3DSparkMarkov
    @ModuleInfo(key: "confidence_head") var confidenceHead: MiniMaxM3DSparkConfidence

    init(_ c: MiniMaxM3DSparkConfig) {
        config = c
        layers = (0 ..< c.layers).map { _ in MiniMaxM3DSparkLayer(c) }
        _fc.wrappedValue = Linear(c.targetLayerIds.count * c.hiddenSize, c.hiddenSize, bias: false)
        _hiddenNorm.wrappedValue = MiniMaxM3RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
        _norm.wrappedValue = MiniMaxM3RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
        _markovHead.wrappedValue = MiniMaxM3DSparkMarkov(vocabulary: c.vocabularySize, rank: c.markovRank)
        _confidenceHead.wrappedValue = MiniMaxM3DSparkConfidence(dimensions: c.hiddenSize + c.markovRank)
    }

    private func rope(_ x: MLXArray, _ offset: Int) -> MLXArray {
        MLXFast.RoPE(
            x, dimensions: config.rotaryDim, traditional: false, base: config.ropeTheta, scale: 1, offset: offset)
    }

    /// Per-layer context K / V `[1, kvHeads, C, headDim]` of tap features `[1, C, nTaps * hidden]` at positions
    /// `position ..< position + C` (Python `add_context`).
    func contextRows(_ features: MLXArray, position: Int) -> [(MLXArray, MLXArray)] {
        let count = features.dim(1)
        let ctx = hiddenNorm(fc(features))
        return layers.map { layer in
            let a = layer.selfAttn
            let k = a.kNorm(a.kProj(ctx).reshaped(1, count, config.kvHeads, config.headDim)).transposed(0, 2, 1, 3)
            let v = a.vProj(ctx).reshaped(1, count, config.kvHeads, config.headDim).transposed(0, 2, 1, 3)
            return (rope(k, position), v)
        }
    }

    /// Keys for a block at positions `start ..< start + length` under the sliding window (Python `_sliding`): `k` / `v`
    /// hold the context rows of positions `base ..< start` followed by the block rows; context positions > p - window
    /// for the row at p, and the whole block (non-causal). nil mask = every row reads every key held.
    static func sliding(
        _ k: MLXArray, _ v: MLXArray, start: Int, length: Int, window: Int, dtype: DType, base: Int
    ) -> (MLXArray, MLXArray, MLXArray?) {
        if start + length - 1 - window < base { return (k, v, nil) }
        let first = max(base, start + 1 - window)
        let ks = k[0..., 0..., (first - base)...]
        let vs = v[0..., 0..., (first - base)...]
        let keyPositions = concatenated(
            [arange(first, start), MLXArray.full([length], values: MLXArray(Int32(start + length)))], axis: 0)
        let queryPositions = arange(start, start + length).reshaped(length, 1)
        let keep = keyPositions.reshaped(1, -1) .> (queryPositions - Int32(window))
        let mask = MLX.where(keep, MLXArray(Float(0)), MLXArray(-Float.infinity)).asType(dtype)
        return (ks, vs, mask.reshaped(1, 1, length, -1))
    }

    /// The block's final-norm hidden `[1, L, hidden]` for block embeddings `[1, L, hidden]` at positions
    /// `start ..< start + L` over `context` (which must end at `start`). `window` overrides the config's sliding
    /// window (tests: controls; 0 = every context row held is visible).
    func block(
        _ embeddings: MLXArray, start: Int, context: MiniMaxM3DSparkContext, window: Int? = nil
    ) -> MLXArray {
        let window = window ?? config.slidingWindow
        precondition(context.end == start, "DSpark drafter context ends at \(context.end), block starts at \(start)")
        let length = embeddings.dim(1)
        var h = embeddings
        for (li, layer) in layers.enumerated() {
            let a = layer.selfAttn
            let x = layer.inputLayerNorm(h)
            let q = rope(
                a.qNorm(a.qProj(x).reshaped(1, length, config.heads, config.headDim)).transposed(0, 2, 1, 3), start)
            let k = rope(
                a.kNorm(a.kProj(x).reshaped(1, length, config.kvHeads, config.headDim)).transposed(0, 2, 1, 3),
                start)
            let v = a.vProj(x).reshaped(1, length, config.kvHeads, config.headDim).transposed(0, 2, 1, 3)
            var keys = context.keys.isEmpty ? k : concatenated([context.keys[li], k], axis: 2)
            var values = context.values.isEmpty ? v : concatenated([context.values[li], v], axis: 2)
            var mask: MLXFast.ScaledDotProductAttentionMaskMode = .none
            if window > 0 {
                let (ks, vs, m) = Self.sliding(
                    keys, values, start: start, length: length, window: window, dtype: q.dtype, base: context.start)
                keys = ks
                values = vs
                if let m { mask = .array(m) }
            }
            let o = MLXFast.scaledDotProductAttention(
                queries: q, keys: keys, values: values, scale: pow(Float(config.headDim), -0.5), mask: mask)
            h = h + a.oProj(o.transposed(0, 2, 1, 3).reshaped(1, length, -1))
            h = h + layer.mlp(layer.postAttentionLayerNorm(h))
        }
        return norm(h)
    }

    /// Rows `index` (any shape) of a Linear / Embedding weight; a quantized module dequantizes only those rows (BF16).
    /// The Markov / confidence heads are read this way, so pre-quantized heads draft correctly (Python `_rows`).
    static func rows(_ module: Module, _ index: MLXArray?) -> MLXArray {
        func dequantize(_ w: MLXArray, _ s: MLXArray, _ b: MLXArray?, _ g: Int, _ bits: Int, _ mode: QuantizationMode)
            -> MLXArray
        {
            guard let index else {
                return dequantized(w, scales: s, biases: b, groupSize: g, bits: bits, mode: mode, dtype: .bfloat16)
            }
            let flat = index.reshaped(-1)
            let out = dequantized(
                w[flat], scales: s[flat], biases: b.map { $0[flat] }, groupSize: g, bits: bits, mode: mode,
                dtype: .bfloat16)
            return out.reshaped(index.shape + [-1])
        }
        if let q = module as? QuantizedLinear {
            return dequantize(q.weight, q.scales, q.biases, q.groupSize, q.bits, q.mode)
        }
        if let q = module as? QuantizedEmbedding {
            return dequantize(q.weight, q.scales, q.biases, q.groupSize, q.bits, q.mode)
        }
        let weight = (module as? Linear)?.weight ?? (module as! Embedding).weight
        return index.map { weight[$0] } ?? weight
    }

    /// The Markov chain over block rows 0 ..< count (Python `draft`): tokens `[count]` (uint32) and, when asked,
    /// confidences `[count]` float32. `candidates` K (0 or >= the vocabulary: full vocabulary) rescores only the top-K
    /// base logits of each row. With `noise` (`[>= count, vocab]` Gumbel rows), step i proposes
    /// argmax(score * inverseTemperature + noise[i]) (coupled sampled drafting), else argmax(score).
    func markovDraft(
        hidden: MLXArray, anchor: Int, head: (MLXArray) -> MLXArray, count: Int, candidates: Int,
        noise: MLXArray? = nil, inverseTemperature: Float = 1, confidences wantConfidences: Bool = false
    ) -> (MLXArray, MLXArray?) {
        let w1 = markovHead.w1
        let w2 = markovHead.w2
        let logits = head(hidden[0..., ..<count, 0...])[0].asType(.float32)  // [count, V]
        var confW: MLXArray?
        var confB: MLXArray?
        if wantConfidences {
            confW = Self.rows(confidenceHead.proj, nil).asType(.float32)
            confB = confidenceHead.proj.bias!.asType(.float32)
        }
        func confidence(_ i: Int, _ latent: MLXArray) -> MLXArray? {
            guard let confW, let confB else { return nil }
            let feature = concatenated([hidden[0, i].asType(.float32), latent])
            return sigmoid(matmul(feature, confW[0]) + confB[0])
        }
        var tokens: [MLXArray] = []
        var confs: [MLXArray] = []
        let draftVocabulary = w2.weight.dim(0)
        if candidates > 0 && candidates < draftVocabulary {
            let k = candidates
            let cand = argPartition(-logits, kth: k - 1, axis: -1)[0..., ..<k]  // [count, K]
            let base = takeAlong(logits, cand, axis: -1)
            var previous = MLXArray(Int32(anchor))
            for i in 0 ..< count {
                let latent = Self.rows(w1, previous).asType(.float32)  // [r]
                var row = base[i] + matmul(Self.rows(w2, cand[i]).asType(.float32), latent)  // [K]
                if let c = confidence(i, latent) { confs.append(c) }
                if let noise {
                    row = row * inverseTemperature + take(noise[i], cand[i], axis: 0)
                }
                let token = take(cand[i], argMax(row), axis: 0)
                tokens.append(token.reshaped(1))
                previous = token
            }
        } else {
            var previous = MLXArray([Int32(anchor)])
            for i in 0 ..< count {
                let latent = w1(previous)  // [1, r]
                var row = logits[i] + w2(latent)[0].asType(.float32)
                if let c = confidence(i, latent[0].asType(.float32)) { confs.append(c) }
                if let noise { row = row * inverseTemperature + noise[i] }
                let token = argMax(row).reshaped(1)
                tokens.append(token)
                previous = token
            }
        }
        return (concatenated(tokens), wantConfidences ? stacked(confs) : nil)
    }
}

// MARK: - Loader

enum MiniMaxM3DSparkLoader {
    enum LoadError: Error, CustomStringConvertible {
        case notADrafter(URL)
        case noWeights(URL)
        case unexpected(String)

        var description: String {
            switch self {
            case .notADrafter(let url): "\(url.path) is not a MiniMax DSpark drafter (config.json)"
            case .noWeights(let url): "\(url.path) holds no safetensors"
            case .unexpected(let detail): "MiniMax DSpark drafter does not load: \(detail)"
            }
        }
    }

    /// A MiniMax drafter tensor name under the drafter's module names: the release's
    /// `language_model.model.dspark.layers.N.decoder_layer.*` / `final_norm.*` names are mapped; names that already
    /// are module paths pass unchanged (Python `minimax_tensor_name`).
    static func tensorName(_ name: String) -> String {
        var n = name
        let prefix = "language_model.model.dspark."
        if n.hasPrefix(prefix) { n = String(n.dropFirst(prefix.count)) }
        n = n.replacingOccurrences(of: ".decoder_layer.", with: ".")
        if n.hasPrefix("final_norm.") { n = "norm." + n.dropFirst("final_norm.".count) }
        return n
    }

    static func rawConfig(_ directory: URL) throws -> [String: Any] {
        guard let data = try? Data(contentsOf: directory.appendingPathComponent("config.json")),
            let raw = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { throw LoadError.notADrafter(directory) }
        return raw
    }

    /// `X.weight` (e4m3 bytes) + `X.weight_scale_inv` (e8m0) pairs become `X.weight` (viewed uint32) + `X.scales`
    /// (MLX's mxfp8 layout, same bytes), in place; returns the module paths X (Python `mxfp8_pairs_to_mlx`).
    static func decodeMXFP8Pairs(_ weights: inout [String: MLXArray]) -> Set<String> {
        var paths = Set<String>()
        for key in weights.keys.sorted() where key.hasSuffix(".weight_scale_inv") {
            let base = String(key.dropLast(".weight_scale_inv".count))
            guard let w = weights[base + ".weight"] else { continue }
            weights[base + ".scales"] = weights.removeValue(forKey: key)
            weights[base + ".weight"] = w.view(dtype: .uint32)
            paths.insert(base)
        }
        return paths
    }

    static func entry(_ value: Any?, defaults: (Int, Int, QuantizationMode)? = nil)
        -> (groupSize: Int, bits: Int, mode: QuantizationMode)?
    {
        guard let e = value as? [String: Any] else { return nil }
        guard let g = (e["group_size"] as? NSNumber)?.intValue ?? defaults?.0,
            let b = (e["bits"] as? NSNumber)?.intValue ?? defaults?.1
        else { return nil }
        let modeName = e["mode"] as? String ?? "affine"
        guard let mode = QuantizationMode(rawValue: modeName) else { return nil }
        return (g, b, mode)
    }

    /// A MiniMax-M3.1 `dspark/` directory (Python `_load_minimax_drafter`), either
    /// - as released (no `quantization` block): MXFP8 linears (`weight_scale_inv`) kept as stored, everything else as
    ///   stored (BF16); or
    /// - pre-quantized (`config.json["quantization"]`, mlx-lm's per-module form keyed by live module paths): every module
    ///   with an entry gets exactly that entry (affine `{group_size, bits}` or `{mode: mxfp8, ...}`); a module with
    ///   `.scales` but no entry gets the block's top-level `group_size` / `bits` / `mode`; MXFP8 pairs still in the
    ///   release form are MXFP8.
    /// Then a strict load: every module parameter set with its declared shape, no tensor left over.
    static func load(directory: URL) throws -> (MiniMaxM3DSparkConfig, MiniMaxM3DSparkDrafter) {
        let raw = try rawConfig(directory)
        guard MiniMaxM3DSparkConfig.isMiniMax(raw) else { throw LoadError.notADrafter(directory) }
        let config = try MiniMaxM3DSparkConfig(raw: raw)
        let files = try FileManager.default.contentsOfDirectory(atPath: directory.path)
            .filter { $0.hasSuffix(".safetensors") && !$0.hasPrefix("._") }.sorted()
        guard !files.isEmpty else { throw LoadError.noWeights(directory) }
        var weights: [String: MLXArray] = [:]
        for file in files {
            for (name, value) in try loadArrays(url: directory.appendingPathComponent(file)) {
                let renamed = tensorName(name)
                guard weights[renamed] == nil else { throw LoadError.unexpected("duplicate tensor \(renamed)") }
                weights[renamed] = value
            }
        }
        let drafter = MiniMaxM3DSparkDrafter(config)
        let mxfp8 = decodeMXFP8Pairs(&weights)
        let quant = raw["quantization"] as? [String: Any]
        let top: (Int, Int, QuantizationMode)? = quant.flatMap { q in
            guard let g = (q["group_size"] as? NSNumber)?.intValue, let b = (q["bits"] as? NSNumber)?.intValue,
                let mode = QuantizationMode(rawValue: q["mode"] as? String ?? "affine")
            else { return nil }
            return (g, b, mode)
        }
        quantize(model: drafter) { path, module -> (groupSize: Int, bits: Int, mode: QuantizationMode)? in
            if mxfp8.contains(path) { return (32, 8, .mxfp8) }
            guard let quant else { return nil }
            if let e = entry(quant[path]) { return e }
            guard module is Linear || module is Embedding, weights[path + ".scales"] != nil, let top else {
                return nil
            }
            return top
        }
        do {
            try drafter.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        } catch {
            throw LoadError.unexpected("\(error)")
        }
        eval(drafter)
        return (config, drafter)
    }
}

// MARK: - Head (weights shared by every request)

/// The loaded drafter plus the target's embedding and output head. Immutable after load; per-request state is in
/// `MiniMaxM3DSparkSession`.
final class MiniMaxM3DSparkHead: @unchecked Sendable {
    let config: MiniMaxM3DSparkConfig
    let drafter: MiniMaxM3DSparkDrafter
    let embed: (MLXArray) -> MLXArray
    let head: (MLXArray) -> MLXArray
    /// Markov rescoring candidates per row (`VMLX_DSPARK_MARKOV_CANDIDATES`, default 32; 0 = full vocabulary).
    let candidates: Int

    init(
        config: MiniMaxM3DSparkConfig, drafter: MiniMaxM3DSparkDrafter, embed: @escaping (MLXArray) -> MLXArray,
        head: @escaping (MLXArray) -> MLXArray, candidates: Int = MiniMaxM3DSpark.markovCandidates
    ) {
        self.config = config
        self.drafter = drafter
        self.embed = embed
        self.head = head
        self.candidates = candidates
    }

    /// Draft `count` tokens after `anchor` at `start` (the anchor's position) over `context` (ending at `start`): one
    /// block of max(trained block, count) rows, then the Markov chain over its first `count` rows. Unevaluated.
    func propose(
        anchor: Int, start: Int, count: Int, context: MiniMaxM3DSparkContext, noise: MLXArray? = nil,
        inverseTemperature: Float = 1
    ) -> MLXArray {
        let span = max(config.blockSize, count)
        var ids = [Int32](repeating: Int32(config.maskTokenId), count: span)
        ids[0] = Int32(anchor)
        let hidden = drafter.block(embed(MLXArray(ids).reshaped(1, span)), start: start, context: context)
        return drafter.markovDraft(
            hidden: hidden, anchor: anchor, head: head, count: count, candidates: candidates, noise: noise,
            inverseTemperature: inverseTemperature
        ).0
    }
}

// MARK: - Runtime helpers

enum MiniMaxM3DSpark {
    /// The cache list's DSpark holder: its first sparse-attention layer cache (the tap tail and the verify flag).
    static func tapHolder(_ cache: [KVCache]?) -> MiniMaxM3SparseCache? {
        guard let cache else { return nil }
        for layer in cache {
            if let sparse = layer as? MiniMaxM3SparseCache { return sparse }
        }
        return nil
    }

    /// True for a forward a DSpark session armed as a verify: one sequence of several rows.
    static func verifies(_ inputs: MLXArray, holder: MiniMaxM3SparseCache) -> Bool {
        holder.dsparkVerifyArmed && inputs.ndim == 2 && inputs.dim(0) == 1 && inputs.dim(1) > 1
    }

    static func environmentInt(_ name: String) -> Int? {
        guard let raw = RuntimeEnvironment.value(name)?.trimmingCharacters(in: .whitespaces), !raw.isEmpty,
            let value = Int(raw), value >= 1
        else { return nil }
        return value
    }

    /// `VMLX_DSPARK_MARKOV_CANDIDATES` (default 32; 0 = the full vocabulary).
    static var markovCandidates: Int {
        guard let raw = RuntimeEnvironment.value("VMLX_DSPARK_MARKOV_CANDIDATES"), let v = Int(raw) else { return 32 }
        return max(0, v)
    }

    /// The widest draft Adaptive may choose (Python `ladder_top`): twice the trained block (the verify is exact at any
    /// width, so no verify coverage limits it); `VMLX_DSPARK_MAX_DRAFT` replaces it (diagnostic lever).
    static func ladderTop(block: Int) -> Int {
        if let pinned = environmentInt("VMLX_DSPARK_MAX_DRAFT") {
            NSLog("[MiniMax-M3 DSpark] VMLX_DSPARK_MAX_DRAFT=\(pinned) limits the draft ladder (diagnostic lever)")
            return pinned
        }
        return max(1, 2 * block)
    }

    /// `VMLX_DSPARK_WIDTH`: every step's width pinned (diagnostic lever); nil lets Adaptive choose.
    static var pinnedWidth: Int? { environmentInt("VMLX_DSPARK_WIDTH") }
}

// MARK: - Session (per-request drafter context)

final class MiniMaxM3DSparkSession: DSparkDraftSession {
    let head: MiniMaxM3DSparkHead
    private(set) var context: MiniMaxM3DSparkContext

    init(head: MiniMaxM3DSparkHead) {
        self.head = head
        context = .empty(at: 0)
    }

    /// Bring the context to committed end `end` (exclusive) from the cache's tap tail: positions already held stay,
    /// only new rows are projected; at most `sliding_window` rows are kept. A tail that does not describe the cache up
    /// to `end` leaves an empty context positioned at `end` (drafts only: the target verifies every token).
    func sync(end: Int, tail: HybridPoolHiddenTail?) {
        let window = head.config.slidingWindow
        guard let tail, tail.endPosition == end, end > 0, !tail.isEmpty else {
            context = .empty(at: end)
            return
        }
        if context.end > end || context.end < tail.startPosition || (context.count == 0 && context.end != end) {
            context = .empty(at: max(tail.startPosition, end - window))
        }
        if context.end < end, let rows = tail.rows(from: context.end, to: end) {
            context.append(head.drafter.contextRows(rows.expandedDimensions(axis: 0), position: context.end))
            context.keepLast(window)
            asyncEval(context.keys + context.values)
        }
    }

    private func proposal(
        seed: Int, cache: [KVCache], count: Int, noise: MLXArray?, inverseTemperature: Float
    ) -> [Int] {
        let holder = MiniMaxM3DSpark.tapHolder(cache)
        let end = holder?.offset ?? (cache.first?.offset ?? 0)
        sync(end: end, tail: holder?.dsparkTail)
        let tokens = head.propose(
            anchor: seed, start: end, count: count, context: context, noise: noise,
            inverseTemperature: inverseTemperature)
        eval(tokens)
        return tokens.asArray(UInt32.self).map(Int.init)
    }

    func draft(seed: Int, cache: [KVCache]) -> [Int] {
        draft(seed: seed, cache: cache, count: head.config.blockSize)
    }

    func draft(seed: Int, cache: [KVCache], count: Int) -> [Int] {
        proposal(seed: seed, cache: cache, count: max(1, count), noise: nil, inverseTemperature: 1)
    }

    func draft(seed: Int, cache: [KVCache], noise: MLXArray, inverseTemperature: Float) -> [Int] {
        draft(seed: seed, cache: cache, count: head.config.blockSize, noise: noise, inverseTemperature: inverseTemperature)
    }

    func draft(seed: Int, cache: [KVCache], count: Int, noise: MLXArray, inverseTemperature: Float) -> [Int] {
        proposal(
            seed: seed, cache: cache, count: max(1, count), noise: noise[..<max(1, count)],
            inverseTemperature: inverseTemperature)
    }

    var vocabularySize: Int? { head.config.vocabularySize }

    /// A rollback is a trim of every lane: as cheap for a zero-accept step as for any other.
    var commitsAnyPrefix: Bool { true }

    /// Arms the cache list's holder: the next multi-row forward is a verify (every row on decode arithmetic).
    func beginVerifyCapture(cache: [KVCache]) -> Bool {
        guard let holder = MiniMaxM3DSpark.tapHolder(cache) else { return false }
        holder.dsparkVerifyArmed = true
        holder.dsparkVerifyStart = holder.offset
        return true
    }

    /// Keep the first `keep` rows of the verify forward in every cache lane (Python `rollback`): K / V, index keys and
    /// the tap tail are append-only and anchored to absolute positions, so a trim is exact wherever it lands (mid a
    /// 128-key block or across one); the next rows overwrite the trimmed tail. Lockstep of the K / V and index lanes is
    /// checked.
    func commitVerifiedPrefix(cache: [KVCache], checkpoint: SpeculativeCacheListCheckpoint, keep: Int) -> Bool {
        guard let holder = MiniMaxM3DSpark.tapHolder(cache), let start = holder.dsparkVerifyStart else { return false }
        let total = holder.offset - start
        guard keep >= 1, keep <= total, cache.allSatisfy({ $0.offset == holder.offset && $0.isTrimmable }) else {
            return false
        }
        let trim = total - keep
        guard trim > 0 else { return true }
        for (i, layer) in cache.enumerated() {
            let trimmed = layer.trim(trim)
            precondition(trimmed == trim, "DSpark rollback: layer \(i) cache holds fewer than \(trim) verify rows")
            if let sparse = layer as? MiniMaxM3SparseCache {
                precondition(
                    sparse.indexRows == sparse.offset,
                    "DSpark rollback: layer \(i) index-key lane \(sparse.indexRows) != K/V lane \(sparse.offset)")
            }
        }
        return true
    }

    func endVerifyCapture(cache: [KVCache]) {
        guard let holder = MiniMaxM3DSpark.tapHolder(cache) else { return }
        holder.dsparkVerifyArmed = false
        holder.dsparkVerifyStart = nil
    }
}

// MARK: - MiniMaxM3Model conformance

extension MiniMaxM3Model: DSparkSpeculativeModel {
    private var attachedHead: MiniMaxM3DSparkHead? {
        dsparkAttachLock.lock()
        defer { dsparkAttachLock.unlock() }
        return dsparkHead
    }

    public var dsparkBlockSize: Int? { attachedHead?.config.blockSize }

    public var dsparkLadderTop: Int? { attachedHead.map { MiniMaxM3DSpark.ladderTop(block: $0.config.blockSize) } }

    public var dsparkPinnedWidth: Int? {
        guard attachedHead != nil, let width = MiniMaxM3DSpark.pinnedWidth else { return nil }
        NSLog("[MiniMax-M3 DSpark] VMLX_DSPARK_WIDTH=\(width) pins every step's width (diagnostic lever)")
        return width
    }

    /// Load the bundle's `dspark/` drafter once (later calls are no-ops). False when the directory does not hold a
    /// MiniMax DSpark drafter or `VMLX_DSPARK=0`; throws when it does but cannot serve this model (geometry, weights).
    public func attachDSparkDraft(directory: URL) throws -> Bool {
        if DSparkEnvironment.miniMaxDisabled() { return false }
        dsparkAttachLock.lock()
        defer { dsparkAttachLock.unlock() }
        if dsparkHead != nil { return true }
        guard let raw = try? MiniMaxM3DSparkLoader.rawConfig(directory), MiniMaxM3DSparkConfig.isMiniMax(raw) else {
            return false
        }
        let start = Date()
        let (config, drafter) = try MiniMaxM3DSparkLoader.load(directory: directory)
        // The drafter reads the target's taps, embedding and output head: its geometry must be the target's.
        guard config.hiddenSize == configuration.hiddenSize else {
            throw MiniMaxM3DSparkLoader.LoadError.unexpected(
                "drafter hidden_size \(config.hiddenSize) != target \(configuration.hiddenSize)")
        }
        guard config.vocabularySize == configuration.vocabularySize else {
            throw MiniMaxM3DSparkLoader.LoadError.unexpected(
                "drafter vocabulary \(config.vocabularySize) != target \(configuration.vocabularySize) (its Markov "
                    + "head rescores the target's logits)")
        }
        guard config.targetLayerIds.allSatisfy({ (0 ..< configuration.hiddenLayers).contains($0) }) else {
            throw MiniMaxM3DSparkLoader.LoadError.unexpected(
                "drafter taps layers \(config.targetLayerIds); the target has \(configuration.hiddenLayers)")
        }
        let head = MiniMaxM3DSparkHead(
            config: config, drafter: drafter,
            embed: { [unowned self] ids in self.model.embedTokens(ids) },
            head: { [unowned self] h in self.lmHead.map { $0(h) } ?? self.model.embedTokens.asLinear(h) })
        dsparkHead = head
        model.dsparkTailCapacity = config.slidingWindow
        model.dsparkTapLayers = config.targetLayerIds
        NSLog(
            "[MiniMax-M3 DSpark] drafter attached from \(directory.path): \(config.layers) layers, block "
                + "\(config.blockSize), taps \(config.targetLayerIds), window \(config.slidingWindow), draft ladder "
                + "1...\(MiniMaxM3DSpark.ladderTop(block: config.blockSize)) (Adaptive), "
                + String(format: "%.1f s", Date().timeIntervalSince(start)))
        return true
    }

    public func makeDSparkDraftSession() -> (any DSparkDraftSession)? {
        attachedHead.map { MiniMaxM3DSparkSession(head: $0) }
    }

    public func dsparkCacheUnsupportedReason(_ cache: [KVCache]) -> String? {
        guard MiniMaxM3DSpark.tapHolder(cache) != nil else {
            return "the cache has no MiniMax-M3 sparse-attention layer (DSpark keeps its taps there)"
        }
        guard cache.allSatisfy(\.isTrimmable), SpeculativeCacheListCheckpoint.supports(cache) else {
            return "a cache layer cannot be checkpointed or trimmed in place"
        }
        return nil
    }
}
