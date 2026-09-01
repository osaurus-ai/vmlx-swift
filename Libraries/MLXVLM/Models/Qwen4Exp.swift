// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

public struct Qwen4ExpConfiguration: Codable, Sendable {
    struct JangMetadata: Codable, Sendable {
        let normConvention: String?
        enum CodingKeys: String, CodingKey { case normConvention = "norm_convention" }
    }
    struct TextExtras: Codable, Sendable {
        var dtype: String?
        var mambaSSMDType: String?
        var mtpNumHiddenLayers = 0
        var layerTypes: [String]?
        var outputGateType = "sigmoid"
        var hcCount = 4
        var hcLowrank = 320
        var pleLayerIds = [2]
        var pleEmbedDim = 2560
        var pleConvKernelSize = 4
        var ngramSize = 3
        var headsPerNgram = 8
        var ngramVocabSizeBase = 20_000_000
        var makeNgramVocabSizeDivisibleBy = 128
        var seed = 1234
        var splitNgramParts = 128
        var indexerNHeads = 4
        var indexerKVHeads = 1
        var indexerHeadDim = 128
        var indexerBudget = 2048
        var indexerCompressRatio = 4
        var mropeSection = [11, 11, 10]

        enum CodingKeys: String, CodingKey {
            case dtype
            case mambaSSMDType = "mamba_ssm_dtype"
            case mtpNumHiddenLayers = "mtp_num_hidden_layers"
            case layerTypes = "layer_types"
            case outputGateType = "output_gate_type"
            case hcCount = "hc_count"
            case hcLowrank = "hc_lowrank"
            case pleLayerIds = "ple_layer_ids"
            case pleEmbedDim = "ple_embed_dim"
            case pleConvKernelSize = "ple_conv_kernel_size"
            case ngramSize = "ngram_size"
            case headsPerNgram = "heads_per_ngram"
            case ngramVocabSizeBase = "ngram_vocab_size_base"
            case makeNgramVocabSizeDivisibleBy = "make_ngram_vocab_size_divisible_by"
            case seed
            case splitNgramParts = "split_ngram_parts"
            case indexerNHeads = "indexer_n_heads"
            case indexerKVHeads = "indexer_kv_heads"
            case indexerHeadDim = "indexer_head_dim"
            case indexerBudget = "indexer_budget"
            case indexerCompressRatio = "indexer_compress_ratio"
            case mropeSection = "mrope_section"
        }

        init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            dtype = try c.decodeIfPresent(String.self, forKey: .dtype)
            mambaSSMDType = try c.decodeIfPresent(String.self, forKey: .mambaSSMDType)
            mtpNumHiddenLayers = try c.decodeIfPresent(Int.self, forKey: .mtpNumHiddenLayers) ?? 0
            layerTypes = try c.decodeIfPresent([String].self, forKey: .layerTypes)
            outputGateType = try c.decodeIfPresent(String.self, forKey: .outputGateType) ?? "sigmoid"
            hcCount = try c.decodeIfPresent(Int.self, forKey: .hcCount) ?? 4
            hcLowrank = try c.decodeIfPresent(Int.self, forKey: .hcLowrank) ?? 320
            pleLayerIds = try c.decodeIfPresent([Int].self, forKey: .pleLayerIds) ?? [2]
            pleEmbedDim = try c.decodeIfPresent(Int.self, forKey: .pleEmbedDim) ?? 2560
            pleConvKernelSize = try c.decodeIfPresent(Int.self, forKey: .pleConvKernelSize) ?? 4
            ngramSize = try c.decodeIfPresent(Int.self, forKey: .ngramSize) ?? 3
            headsPerNgram = try c.decodeIfPresent(Int.self, forKey: .headsPerNgram) ?? 8
            ngramVocabSizeBase = try c.decodeIfPresent(Int.self, forKey: .ngramVocabSizeBase) ?? 20_000_000
            makeNgramVocabSizeDivisibleBy =
                try c.decodeIfPresent(Int.self, forKey: .makeNgramVocabSizeDivisibleBy) ?? 128
            seed = try c.decodeIfPresent(Int.self, forKey: .seed) ?? 1234
            splitNgramParts = try c.decodeIfPresent(Int.self, forKey: .splitNgramParts) ?? 128
            indexerNHeads = try c.decodeIfPresent(Int.self, forKey: .indexerNHeads) ?? 4
            indexerKVHeads = try c.decodeIfPresent(Int.self, forKey: .indexerKVHeads) ?? 1
            indexerHeadDim = try c.decodeIfPresent(Int.self, forKey: .indexerHeadDim) ?? 128
            indexerBudget = try c.decodeIfPresent(Int.self, forKey: .indexerBudget) ?? 2048
            indexerCompressRatio =
                try c.decodeIfPresent(Int.self, forKey: .indexerCompressRatio) ?? 4
            mropeSection =
                try c.decodeIfPresent([Int].self, forKey: .mropeSection) ?? [11, 11, 10]
        }
    }

    let base: Qwen35Configuration
    let extras: TextExtras
    let jangMetadata: JangMetadata?
    let quantizationContainer: BaseConfiguration.QuantizationContainer?

    enum CodingKeys: String, CodingKey {
        case text = "text_config"
        case jangMetadata = "jang_config"
        case quantizationContainer = "quantization"
    }

    public init(from decoder: Decoder) throws {
        base = try Qwen35Configuration(from: decoder)
        let container = try decoder.container(keyedBy: CodingKeys.self)
        extras = try container.decode(TextExtras.self, forKey: .text)
        jangMetadata = try container.decodeIfPresent(JangMetadata.self, forKey: .jangMetadata)
        quantizationContainer = try container.decodeIfPresent(
            BaseConfiguration.QuantizationContainer.self,
            forKey: .quantizationContainer)
    }

    public func encode(to encoder: Encoder) throws {
        try base.encode(to: encoder)
    }

    /// Compute dtype declared by the active model bundle. This is deliberately
    /// optional: an absent or unknown declaration must not silently become a
    /// family-wide FP16/BF16 rewrite.
    var declaredComputeDType: DType? {
        switch extras.dtype?
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        {
        case "bfloat16", "bf16": .bfloat16
        case "float16", "fp16", "half": .float16
        case "float32", "fp32", "float": .float32
        default: nil
        }
    }

    /// The row-equivalent verifier workaround is proven only for the released
    /// 4M checkpoint's complete routed-expert bank. A local q4 sub-block is not
    /// enough to identify that topology: 4S is mixed q2/q3/q4, and the private
    /// MTP head can itself carry q4 tensors. Require every trunk layer's three
    /// routed projections to resolve to affine q4/g64 from bundle metadata.
    var hasUniformQ4G64TrunkRoutedExperts: Bool {
        guard base.textConfiguration.hiddenLayers > 0,
            let quantization = quantizationContainer?.perLayerQuantization
        else { return false }

        for layer in 0 ..< base.textConfiguration.hiddenLayers {
            for projection in ["gate_proj", "up_proj", "down_proj"] {
                let path = "language_model.layers.\(layer).mlp.switch_mlp.\(projection)"
                guard let resolved = quantization.quantization(layer: path),
                    resolved.bits == 4,
                    resolved.groupSize == 64,
                    resolved.mode == .affine
                else { return false }
            }
        }
        return true
    }
}

private enum Qwen4ExpDTypeTrace {
    private static let lock = NSLock()
    nonisolated(unsafe) private static var textAuditClaimed = false
    nonisolated(unsafe) private static var headAuditClaimed = false

    static func claimTextAudit() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        guard !textAuditClaimed else { return false }
        textAuditClaimed = true
        return true
    }

    static func claimHeadAudit() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        guard !headAuditClaimed else { return false }
        headAuditClaimed = true
        return true
    }

    static func logTextAudit(
        rawEmbedding: DType, computeEmbedding: DType, layerDTypes: [DType],
        finalMixer: DType, declared: DType?
    ) {
        let mismatchedLayers = layerDTypes.enumerated().compactMap { index, dtype in
            dtype == declared ? nil : "\(index):\(dtype)"
        }
        let uniqueLayerDTypes = Set(layerDTypes.map(String.init(describing:))).sorted()
        let layerSummary = uniqueLayerDTypes.joined(separator: ",")
        let mismatchSummary = mismatchedLayers.isEmpty
            ? "none" : mismatchedLayers.joined(separator: ",")
        let declaredSummary = declared.map { String(describing: $0) } ?? "unspecified"
        let message = "[Qwen4Exp] live_dtype_boundaries embedding_raw=\(rawEmbedding)"
            + " embedding_compute=\(computeEmbedding) layers=\(layerDTypes.count)"
            + " layer_output_dtypes=\(layerSummary)"
            + " mismatched_layers=\(mismatchSummary)"
            + " final_mixer=\(finalMixer) declared=\(declaredSummary)\n"
        FileHandle.standardError.write(Data(message.utf8))
    }

    static func logHeadAudit(input: DType, rawLogits: DType, returnedLogits: DType) {
        FileHandle.standardError.write(Data(
            ("[Qwen4Exp] live_dtype_head head_input=\(input) raw_logits=\(rawLogits)"
                + " returned_logits=\(returnedLogits)\n").utf8))
    }
}

private final class Qwen4ExpGroupedRMSNorm: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    let groupSize: Int
    let eps: Float

    init(dimensions: Int, groupSize: Int, eps: Float) {
        self.groupSize = groupSize
        self.eps = eps
        _weight.wrappedValue = MLXArray.ones([dimensions])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let original = x.shape
        let grouped = x.reshaped(Array(original.dropLast()) + [-1, groupSize])
        let normalized = MLXFast.rmsNorm(grouped, weight: MLXArray.mlxNone, eps: eps)
        return normalized.reshaped(original) * weight
    }
}

/// Qwen3.8 Flash Next executes two mHC mixers in every decoder layer.  Keep
/// the weights as graph inputs so one compiled decode graph is shared across
/// all layers instead of retaining 96 layer-specific constant graphs.
private enum Qwen4ExpCompiledMHC {
    typealias Region = @Sendable ([MLXArray]) -> [MLXArray]

    private static let enabled: Bool = {
        let value = RuntimeEnvironment.value("VMLX_QWEN4_EXP_COMPILE_MHC")
            ?? "1"
        return value != "0" && value.lowercased() != "false"
    }()
    private static let allowTwoTokenVerify: Bool =
        RuntimeEnvironment.flag("VMLX_QWEN4_EXP_COMPILE_MHC_S2")
    private static let lock = NSLock()
    nonisolated(unsafe) private static var regions: [String: Region] = [:]
    nonisolated(unsafe) private static var didReport = false

    static func supportsShape(_ hyper: MLXArray) -> Bool {
        let sequenceLength = hyper.dim(-2)
        return sequenceLength == 1 || (allowTwoTokenVerify && sequenceLength == 2)
    }

    static func callDense(
        hyper: MLXArray,
        normWeight: MLXArray,
        downWeight: MLXArray,
        downOutputSize: Int,
        upWeight: MLXArray,
        hcCount: Int,
        hiddenSize: Int,
        eps: Float
    ) -> (MLXArray, MLXArray)? {
        guard enabled, !CompiledDecodeTrace.isActive, supportsShape(hyper),
            hyper.dtype == .bfloat16,
            normWeight.dtype == .bfloat16,
            downWeight.dtype == .bfloat16,
            upWeight.dtype == .bfloat16
        else { return nil }

        let key = [
            "dense", String(hcCount), String(hiddenSize), String(eps),
            String(downOutputSize), "S\(hyper.dim(-2))",
        ].joined(separator: "|")

        lock.lock()
        var region = regions[key]
        if region == nil {
            region = vmlxTrustedCompile { (args: [MLXArray]) -> [MLXArray] in
                let hyper = args[0]
                let original = hyper.shape
                let grouped = hyper.reshaped(
                    Array(original.dropLast()) + [-1, hiddenSize])
                let normalized = (
                    MLXFast.rmsNorm(grouped, weight: MLXArray.mlxNone, eps: eps)
                        .reshaped(original) * args[1]
                )
                let combined = matmul(normalized, args[2].transposed())
                let parts = MLX.split(combined, indices: [downOutputSize], axis: -1)
                let activated = silu(parts[0] / Float(hcCount))
                let projected = matmul(activated, args[3].transposed())
                let weights = sigmoid(projected).reshaped(
                    Array(projected.shape.dropLast()) + [hcCount, hiddenSize])
                let normalizedGroups = normalized.reshaped(
                    Array(normalized.shape.dropLast()) + [hcCount, hiddenSize])
                let mixed = (weights * normalizedGroups).mean(axis: -2)
                    .asType(hyper.dtype)
                let injection = (2 * sigmoid(parts[1] / Float(hcCount)))
                    .asType(hyper.dtype)
                return [mixed, injection]
            }
            regions[key] = region
        }
        reportLocked()
        lock.unlock()

        let outputs = region!([hyper, normWeight, downWeight, upWeight])
        guard outputs.count == 2 else { return nil }
        return (outputs[0], outputs[1])
    }

    static func call(
        hyper: MLXArray,
        normWeight: MLXArray,
        downWeight: MLXArray,
        downScales: MLXArray,
        downBiases: MLXArray,
        downGroupSize: Int,
        downBits: Int,
        downMode: QuantizationMode,
        downOutputSize: Int,
        upWeight: MLXArray,
        upScales: MLXArray,
        upBiases: MLXArray,
        upGroupSize: Int,
        upBits: Int,
        upMode: QuantizationMode,
        hcCount: Int,
        hiddenSize: Int,
        eps: Float
    ) -> (MLXArray, MLXArray)? {
        guard enabled, !CompiledDecodeTrace.isActive, supportsShape(hyper),
            hyper.dtype == .bfloat16,
            normWeight.dtype == .bfloat16,
            downScales.dtype == .bfloat16, downBiases.dtype == .bfloat16,
            upScales.dtype == .bfloat16, upBiases.dtype == .bfloat16
        else { return nil }

        let key = [
            String(hcCount), String(hiddenSize), String(eps),
            String(downGroupSize), String(downBits), String(describing: downMode),
            String(downOutputSize), String(upGroupSize), String(upBits),
            String(describing: upMode), "S\(hyper.dim(-2))",
        ].joined(separator: "|")

        lock.lock()
        var region = regions[key]
        if region == nil {
            region = vmlxTrustedCompile { (args: [MLXArray]) -> [MLXArray] in
                let hyper = args[0]
                let original = hyper.shape
                let grouped = hyper.reshaped(
                    Array(original.dropLast()) + [-1, hiddenSize])
                let normalized = (
                    MLXFast.rmsNorm(grouped, weight: MLXArray.mlxNone, eps: eps)
                        .reshaped(original) * args[1]
                )
                let combined = quantizedMM(
                    normalized, args[2], scales: args[3], biases: args[4],
                    transpose: true, groupSize: downGroupSize, bits: downBits,
                    mode: downMode)
                let parts = MLX.split(combined, indices: [downOutputSize], axis: -1)
                let activated = silu(parts[0] / Float(hcCount))
                let projected = quantizedMM(
                    activated, args[5], scales: args[6], biases: args[7],
                    transpose: true, groupSize: upGroupSize, bits: upBits,
                    mode: upMode)
                let weights = sigmoid(projected).reshaped(
                    Array(projected.shape.dropLast()) + [hcCount, hiddenSize])
                let normalizedGroups = normalized.reshaped(
                    Array(normalized.shape.dropLast()) + [hcCount, hiddenSize])
                let mixed = (weights * normalizedGroups).mean(axis: -2)
                    .asType(hyper.dtype)
                let injection = (2 * sigmoid(parts[1] / Float(hcCount)))
                    .asType(hyper.dtype)
                return [mixed, injection]
            }
            regions[key] = region
        }
        reportLocked()
        lock.unlock()

        let outputs = region!([
            hyper, normWeight,
            downWeight, downScales, downBiases,
            upWeight, upScales, upBiases,
        ])
        guard outputs.count == 2 else { return nil }
        return (outputs[0], outputs[1])
    }

    private static func reportLocked() {
        guard !didReport else { return }
        didReport = true
        FileHandle.standardError.write(Data(
            "[Qwen4Exp] compiled_mhc_decode=active shared_weight_inputs=true dtype=bfloat16\n".utf8))
    }

}

private final class Qwen4ExpGatedResidual: Module {
    private static let fuseDecodeInputs =
        RuntimeEnvironment.value("VMLX_QWEN4_EXP_FUSE_DECODE_INPUTS") != "0"
    private static let fusionDiagnosticLock = NSLock()
    private nonisolated(unsafe) static var didReportInputFusion = false

    private enum FusedInputProjection {
        case dense(weight: MLXArray)
        case affine(
            weight: MLXArray,
            scales: MLXArray,
            biases: MLXArray,
            groupSize: Int,
            bits: Int,
            mode: QuantizationMode)
    }

    let hcCount: Int
    let hiddenSize: Int
    let combines: Bool
    @ModuleInfo(key: "hc_norm") var norm: Qwen4ExpGroupedRMSNorm
    @ModuleInfo(key: "input_mix_weight_down") var mixDown: Linear
    @ModuleInfo(key: "input_mix_weight_up") var mixUp: Linear
    @ModuleInfo(key: "block_inject_weight") var inject: Linear?
    private var attemptedInputFusion = false
    private var fusedInputProjection: FusedInputProjection?

    init(_ config: Qwen4ExpConfiguration, combines: Bool = true) {
        let text = config.base.textConfiguration
        let extras = config.extras
        hcCount = extras.hcCount
        hiddenSize = text.hiddenSize
        self.combines = combines
        let width = hcCount * hiddenSize
        _norm.wrappedValue = Qwen4ExpGroupedRMSNorm(
            dimensions: width, groupSize: hiddenSize, eps: text.rmsNormEps)
        _mixDown.wrappedValue = Linear(width, extras.hcLowrank, bias: false)
        _mixUp.wrappedValue = Linear(extras.hcLowrank, width, bias: false)
        if combines { _inject.wrappedValue = Linear(width, hcCount, bias: false) }
        super.init()
    }

    func mix(_ hyper: MLXArray) -> (MLXArray, MLXArray?) {
        var fused: FusedInputProjection?
        if Self.fuseDecodeInputs, combines, Qwen4ExpCompiledMHC.supportsShape(hyper),
            !CompiledDecodeTrace.isActive
        {
            fused = fusedProjection(for: hyper)
        }
        if case .affine(
            let downWeight, let downScales, let downBiases,
            let downGroupSize, let downBits, let downMode) = fused,
            let up = mixUp as? QuantizedLinear,
            let upBiases = up.biases,
            let compiled = Qwen4ExpCompiledMHC.call(
                hyper: hyper,
                normWeight: norm.weight,
                downWeight: downWeight,
                downScales: downScales,
                downBiases: downBiases,
                downGroupSize: downGroupSize,
                downBits: downBits,
                downMode: downMode,
                downOutputSize: mixDown.shape.0,
                upWeight: up.weight,
                upScales: up.scales,
                upBiases: upBiases,
                upGroupSize: up.groupSize,
                upBits: up.bits,
                upMode: up.mode,
                hcCount: hcCount,
                hiddenSize: hiddenSize,
                eps: norm.eps)
        {
            return compiled
        }
        if case .dense(let downWeight) = fused,
            !(mixUp is QuantizedLinear), mixUp.bias == nil,
            let compiled = Qwen4ExpCompiledMHC.callDense(
                hyper: hyper,
                normWeight: norm.weight,
                downWeight: downWeight,
                downOutputSize: mixDown.shape.0,
                upWeight: mixUp.weight,
                hcCount: hcCount,
                hiddenSize: hiddenSize,
                eps: norm.eps)
        {
            return compiled
        }

        let normalized = norm(hyper)
        let mixedDown: MLXArray
        let injected: MLXArray?
        if let fused
        {
            let combined: MLXArray
            switch fused {
            case .dense(let weight):
                combined = matmul(normalized, weight.transposed())
            case .affine(
                let weight, let scales, let biases, let groupSize, let bits, let mode):
                combined = Qwen4ExpBF16Affine.dense(
                    normalized, weight, scales: scales, biases: biases,
                    groupSize: groupSize, bits: bits, mode: mode)
            }
            let parts = MLX.split(combined, indices: [mixDown.shape.0], axis: -1)
            mixedDown = parts[0]
            injected = parts[1]
        } else {
            mixedDown = mixDown(normalized)
            injected = inject.map { $0(normalized) }
        }
        var weights = silu(mixedDown / Float(hcCount))
        weights = sigmoid(mixUp(weights))
            .reshaped(Array(weights.shape.dropLast()) + [hcCount, hiddenSize])
        let grouped = normalized.reshaped(Array(normalized.shape.dropLast()) + [hcCount, hiddenSize])
        let mixed = (weights * grouped).mean(axis: -2).asType(hyper.dtype)
        let injection = injected.map {
            (2 * sigmoid($0 / Float(hcCount))).asType(hyper.dtype)
        }
        return (mixed, injection)
    }

    private func fusedProjection(for input: MLXArray) -> FusedInputProjection? {
        if !attemptedInputFusion {
            attemptedInputFusion = true
            var fusionKind = "none"
            if let down = mixDown as? QuantizedLinear,
                let injection = inject as? QuantizedLinear,
                let downBiases = down.biases, let injectionBiases = injection.biases,
                down.bias == nil, injection.bias == nil,
                down.groupSize == injection.groupSize,
                down.bits == injection.bits, down.mode == injection.mode,
                down.scales.dtype == injection.scales.dtype,
                downBiases.dtype == injectionBiases.dtype,
                down.scales.dtype == downBiases.dtype
            {
                let weight = concatenated([down.weight, injection.weight], axis: 0)
                let scales = concatenated([down.scales, injection.scales], axis: 0)
                let biases = concatenated([downBiases, injectionBiases], axis: 0)
                MLX.eval(weight, scales, biases)
                fusedInputProjection = .affine(
                    weight: weight, scales: scales, biases: biases,
                    groupSize: down.groupSize, bits: down.bits, mode: down.mode)
                fusionKind = "affine"
            } else if let injection = inject,
                !(mixDown is QuantizedLinear), !(injection is QuantizedLinear),
                mixDown.bias == nil, injection.bias == nil,
                mixDown.weight.dtype == injection.weight.dtype,
                Array(mixDown.weight.shape.dropFirst())
                    == Array(injection.weight.shape.dropFirst())
            {
                let weight = concatenated([mixDown.weight, injection.weight], axis: 0)
                MLX.eval(weight)
                fusedInputProjection = .dense(weight: weight)
                fusionKind = "dense"
            } else {
                return nil
            }
            Self.fusionDiagnosticLock.lock()
            if !Self.didReportInputFusion {
                Self.didReportInputFusion = true
                FileHandle.standardError.write(Data(
                    "[Qwen4Exp] fused_hyper_decode_input_projections=active kind=\(fusionKind) dtype=\(input.dtype)\n".utf8))
            }
            Self.fusionDiagnosticLock.unlock()
        }
        return fusedInputProjection
    }

    func combine(_ hyper: MLXArray, block: MLXArray, injection: MLXArray) -> MLXArray {
        let value = expandedDimensions(block, axis: -2) * expandedDimensions(injection, axis: -1)
        return (hyper + value.reshaped(hyper.shape)).asType(hyper.dtype)
    }
}

private final class Qwen4ExpPLE: Module {
    private static let profileHost = RuntimeEnvironment.flag("VMLX_QWEN4_EXP_PROFILE_PLE")
    private static let traceHistory = RuntimeEnvironment.flag("VMLX_QWEN4_EXP_TRACE_PLE_HISTORY")
    private static let traceState = RuntimeEnvironment.flag("VMLX_QWEN4_EXP_TRACE_PLE_STATE")
    let config: Qwen4ExpConfiguration
    let pleLayerIndex: Int
    let multipliers: [Int64]
    let sizes: [Int]
    let offsets: [Int]
    let contextLength: Int
    let headCount: Int
    let headDimension: Int
    let stateLength: Int
    private var table: Qwen4ExpNGramTable?
    private var hostProfileCount = 0
    private var historyTraceCount = 0

    final class Prefetch: @unchecked Sendable {
        let batch: Int
        let length: Int
        let history: [[Int64]]
        let rows: [Int64]
        private let group = DispatchGroup()
        private var result: Result<[Float], Error>?

        init(
            table: Qwen4ExpNGramTable,
            batch: Int,
            length: Int,
            history: [[Int64]],
            rows: [Int64]
        ) {
            self.batch = batch
            self.length = length
            self.history = history
            self.rows = rows
            group.enter()
            DispatchQueue.global(qos: .userInitiated).async { [self] in
                result = Result { try table.gather(rows) }
                group.leave()
            }
        }

        func values() throws -> [Float] {
            group.wait()
            return try result!.get()
        }
    }

    @ModuleInfo(key: "key_proj") var keyProj: Linear
    @ModuleInfo(key: "value_proj") var valueProj: Linear
    @ModuleInfo(key: "norm_key") var normKey: Qwen4ExpGroupedRMSNorm
    @ModuleInfo(key: "norm_query") var normQuery: Qwen4ExpGroupedRMSNorm
    @ModuleInfo(key: "norm_conv") var normConv: Qwen4ExpGroupedRMSNorm
    @ParameterInfo(key: "conv1d_weight") var convWeight: MLXArray

    init(_ config: Qwen4ExpConfiguration, pleLayerIndex: Int) {
        self.config = config
        self.pleLayerIndex = pleLayerIndex
        let text = config.base.textConfiguration
        let extras = config.extras
        contextLength = extras.ngramSize - 1
        headCount = (extras.ngramSize - 1) * extras.headsPerNgram
        headDimension = extras.pleEmbedDim / headCount
        stateLength = (extras.pleConvKernelSize - 1) * extras.ngramSize
        multipliers = Qwen4ExpNGramHash.layerMultipliers(
            unigramVocabSize: text.vocabularySize, ngramSize: extras.ngramSize,
            pleLayerIndex: pleLayerIndex, seed: extras.seed).map { Int64(bitPattern: $0) }
        let layout = Qwen4ExpNGramHash.headVocabLayout(
            ngramHeads: headCount, pleLayerIndex: pleLayerIndex,
            ngramVocabSizeBase: extras.ngramVocabSizeBase)
        sizes = layout.sizes
        offsets = layout.offsets
        let hyperWidth = text.hiddenSize * extras.hcCount
        _keyProj.wrappedValue = Linear(extras.pleEmbedDim, hyperWidth, bias: false)
        _valueProj.wrappedValue = Linear(extras.pleEmbedDim, text.hiddenSize, bias: false)
        _normKey.wrappedValue = Qwen4ExpGroupedRMSNorm(
            dimensions: hyperWidth, groupSize: text.hiddenSize, eps: text.rmsNormEps)
        _normQuery.wrappedValue = Qwen4ExpGroupedRMSNorm(
            dimensions: hyperWidth, groupSize: text.hiddenSize, eps: text.rmsNormEps)
        _normConv.wrappedValue = Qwen4ExpGroupedRMSNorm(
            dimensions: hyperWidth, groupSize: text.hiddenSize, eps: text.rmsNormEps)
        _convWeight.wrappedValue = MLXArray.zeros([hyperWidth, extras.pleConvKernelSize])
        super.init()
    }

    func configureTable(modelDirectory: URL, layerIndex: Int) throws {
        table = try Qwen4ExpNGramTable(
            modelDirectory: modelDirectory, layerIndex: layerIndex,
            shardCount: config.extras.splitNgramParts)
        guard table?.dimensions == headDimension else {
            throw Qwen4ExpNGramTableError.invalidTensor(
                "PLE", "table head dimension does not match config")
        }
    }

    func prefetch(inputIds: MLXArray, cache: MambaCache?) -> Prefetch {
        guard let table else {
            fatalError("qwen4_exp PLE table was not configured by VLMModelFactory")
        }
        let batch = inputIds.dim(0)
        let length = inputIds.dim(1)
        let hostStarted = CFAbsoluteTimeGetCurrent()
        let flat = inputIds.asArray(Int32.self).map(Int64.init)
        let tokenReadFinished = CFAbsoluteTimeGetCurrent()
        var history = [[Int64]]()
        history.reserveCapacity(batch)
        let priorFlat = cache?[2]?.asArray(Int32.self).map(Int64.init)
        for b in 0 ..< batch {
            let prior: [Int64]
            if let priorFlat {
                prior = Array(priorFlat[b * contextLength ..< (b + 1) * contextLength])
            } else {
                prior = [Int64](repeating: Int64(config.base.eosTokenId?.first ?? 248_044), count: contextLength)
            }
            history.append(prior + Array(flat[b * length ..< (b + 1) * length]))
        }
        if Self.traceHistory, pleLayerIndex == 0, length == 1,
            historyTraceCount < 8, let first = history.first
        {
            historyTraceCount += 1
            FileHandle.standardError.write(Data(
                ("[Qwen4ExpPLEHistory] sample=\(historyTraceCount)"
                    + " prior=\(Array(first.dropLast())) token=\(first.last ?? -1)\n").utf8))
            if Self.traceState, let cache {
                func magnitude(_ value: MLXArray?) -> Float {
                    guard let value else { return -1 }
                    let result = abs(value.asType(.float32)).sum()
                    MLX.eval(result)
                    return result.item(Float.self)
                }
                FileHandle.standardError.write(Data(String(format:
                    "[Qwen4ExpPLEState] sample=%d offset=%d conv=%.9g recurrent=%.9g ple_conv=%.9g\n",
                    historyTraceCount, cache.offset,
                    magnitude(cache[0]), magnitude(cache[1]), magnitude(cache[3])).utf8))
            }
        }
        let historyFinished = CFAbsoluteTimeGetCurrent()
        let hashed = Qwen4ExpNGramHash.hashHistory(
            history, multipliers: multipliers, headVocabSizes: sizes, headOffsets: offsets,
            ngramSize: config.extras.ngramSize, headsPerNgram: config.extras.headsPerNgram,
            eosTokenId: Int64(config.base.eosTokenId?.first ?? 248_044))
        let rows = hashed.flatMap { Array($0.suffix(length)).flatMap { $0 } }
        let hashFinished = CFAbsoluteTimeGetCurrent()
        let prefetch = Prefetch(
            table: table, batch: batch, length: length, history: history, rows: rows)
        if Self.profileHost, length == 1, hostProfileCount < 8 {
            FileHandle.standardError.write(Data(String(format:
                "[Qwen4ExpPLEProfile] prefetch_start token_read=%.3fms history=%.3fms hash=%.3fms rows=%d\n",
                (tokenReadFinished - hostStarted) * 1000,
                (historyFinished - tokenReadFinished) * 1000,
                (hashFinished - historyFinished) * 1000,
                rows.count).utf8))
        }
        return prefetch
    }

    func callAsFunction(
        _ hidden: MLXArray, inputIds: MLXArray, cache: MambaCache?,
        prefetch suppliedPrefetch: Prefetch? = nil,
        preloadedEmbedding: MLXArray? = nil,
        recordPrefixCommitStates: Bool = false
    ) -> MLXArray {
        let prefetch = preloadedEmbedding == nil
            ? (suppliedPrefetch ?? self.prefetch(inputIds: inputIds, cache: cache))
            : nil
        let batch = preloadedEmbedding?.dim(0) ?? prefetch!.batch
        let length = preloadedEmbedding?.dim(1) ?? prefetch!.length
        let embedding: MLXArray
        if let preloadedEmbedding {
            embedding = preloadedEmbedding.asType(hidden.dtype)
        } else {
            let gatherStarted = CFAbsoluteTimeGetCurrent()
            let values: [Float]
            do { values = try prefetch!.values() }
            catch { fatalError("qwen4_exp PLE lookup failed: \(error)") }
            let gatherFinished = CFAbsoluteTimeGetCurrent()
            embedding = MLXArray(values)
                .reshaped(batch, length, headCount * headDimension)
                .asType(hidden.dtype)
            let arrayFinished = CFAbsoluteTimeGetCurrent()
            if Self.profileHost, length == 1, hostProfileCount < 8 {
                hostProfileCount += 1
                FileHandle.standardError.write(Data(String(format:
                    "[Qwen4ExpPLEProfile] sample=%d join_wait=%.3fms mlx_array=%.3fms rows=%d\n",
                    hostProfileCount,
                    (gatherFinished - gatherStarted) * 1000,
                    (arrayFinished - gatherFinished) * 1000,
                    prefetch!.rows.count).utf8))
            }
        }
        if let cache, !recordPrefixCommitStates {
            if preloadedEmbedding != nil {
                let prior = cache[2] ?? MLXArray.full(
                    [batch, contextLength],
                    values: MLXArray(Int32(config.base.eosTokenId?.first ?? 248_044)))
                let history = concatenated(
                    [prior.asType(.int32), inputIds.asType(.int32)], axis: 1)
                cache[2] = history[0..., (-contextLength)...]
            } else {
                let tails = prefetch!.history.flatMap {
                    Array($0.suffix(contextLength)).map(Int32.init)
                }
                cache[2] = MLXArray(tails).reshaped(batch, contextLength)
            }
        }

        let text = config.base.textConfiguration
        let extras = config.extras
        let key = normKey(keyProj(embedding)).reshaped(batch, length, extras.hcCount, text.hiddenSize)
        let value = valueProj(embedding)
        let query = normQuery(hidden).reshaped(batch, length, extras.hcCount, text.hiddenSize)
        var gate = (key * query).sum(axis: -1, keepDims: true) / sqrt(Float(text.hiddenSize))
        gate = sqrt(maximum(abs(gate), MLXArray(Float(1e-6)))) * sign(gate)
        // `maximum(abs(gate), 1e-6)` intentionally evaluates the nonlinear
        // gate in F32. Do not let that precision island promote the PLE value
        // stream, residual hyper-state, and every later affine projection to
        // F32. The bundle-declared compute dtype owns the PLE output just as it
        // owns the trunk; only the recurrent GDN state remains F32.
        var gated = (sigmoid(gate) * expandedDimensions(value, axis: -2))
            .reshaped(batch, length, extras.hcCount * text.hiddenSize)
            .asType(hidden.dtype)

        let state = cache?[3] ?? MLXArray.zeros(
            [batch, stateLength, extras.hcCount * text.hiddenSize], dtype: hidden.dtype)
        let normalizedGated = normConv(gated)
        let full = concatenated([state, normalizedGated], axis: 1)
        if let cache {
            if recordPrefixCommitStates {
                // Staged native-MTP verification must leave committed PLE
                // history and convolution state untouched until acceptance is
                // known. Slots 4/5 retain the verifier inputs required to
                // reconstruct exactly the accepted prefix after the target
                // attention caches have been trimmed.
                cache[4] = inputIds
                cache[5] = normalizedGated
            } else {
                cache[3] = full[0..., (-stateLength)..., 0...]
                cache[4] = nil
                cache[5] = nil
            }
        }
        var convolved = MLXArray.zeros(gated.shape, dtype: gated.dtype)
        for tap in 0 ..< extras.pleConvKernelSize {
            let start = tap * extras.ngramSize
            convolved = convolved + full[0..., start ..< (start + length), 0...]
                * convWeight[0..., tap]
        }
        gated = gated + silu(convolved).asType(gated.dtype)
        return gated.asType(hidden.dtype)
    }

    /// Resolve the disk-backed PLE rows before entering an MLX compile
    /// transform.  The returned tensor is tiny (one 2560-wide embedding for
    /// decode) and becomes an explicit input to the compiled trunk.
    func compiledDecodeEmbedding(inputIds: MLXArray, cache: MambaCache?) -> MLXArray {
        let prefetch = prefetch(inputIds: inputIds, cache: cache)
        let values: [Float]
        do { values = try prefetch.values() }
        catch { fatalError("qwen4_exp PLE lookup failed: \(error)") }
        return MLXArray(values).reshaped(
            prefetch.batch, prefetch.length, headCount * headDimension)
    }

    func commitVerifyStaged(
        cache: MambaCache, acceptedInputs: Int, blockLength: Int
    ) -> Bool {
        guard acceptedInputs > 0, acceptedInputs <= blockLength,
            let stagedIds = cache[4], let stagedConv = cache[5],
            stagedIds.dim(1) == blockLength, stagedConv.dim(1) == blockLength
        else { return false }

        let acceptedIds = stagedIds[0..., 0 ..< acceptedInputs]
        let priorIds = cache[2]
            ?? MLXArray.full(
                [stagedIds.dim(0), contextLength],
                values: MLXArray(Int32(config.base.eosTokenId?.first ?? 248_044)))
        cache[2] = concatenated([priorIds, acceptedIds], axis: 1)[
            0..., (-contextLength)...]

        let priorConv = cache[3]
            ?? MLXArray.zeros(
                [stagedConv.dim(0), stateLength, stagedConv.dim(-1)],
                dtype: stagedConv.dtype)
        let acceptedConv = stagedConv[0..., 0 ..< acceptedInputs, 0...]
        cache[3] = concatenated([priorConv, acceptedConv], axis: 1)[
            0..., (-stateLength)..., 0...]
        cache[4] = nil
        cache[5] = nil
        return true
    }
}

private final class Qwen4ExpQSAIndexer: Module {
    let extras: Qwen4ExpConfiguration.TextExtras
    let hiddenSize: Int
    let eps: Float
    let rotary: Qwen35Language.RotaryEmbedding
    @ModuleInfo(key: "index_qk_proj") var projection: Linear
    @ModuleInfo(key: "q_layernorm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_layernorm") var kNorm: RMSNorm

    init(_ config: Qwen4ExpConfiguration) {
        extras = config.extras
        let text = config.base.textConfiguration
        hiddenSize = text.hiddenSize
        eps = text.rmsNormEps
        _projection.wrappedValue = Linear(
            text.hiddenSize, (extras.indexerNHeads + 1) * extras.indexerHeadDim, bias: false)
        _qNorm.wrappedValue = RMSNorm(dimensions: extras.indexerHeadDim, eps: text.rmsNormEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: extras.indexerHeadDim, eps: text.rmsNormEps)
        // Real 3-channel M-RoPE sections. For text-only positions (all three
        // channels equal) this is numerically identical to the previous
        // collapsed [Int.max/4, 0, 0] section; with media present the h/w
        // channels carry the vision grid geometry, matching the reference
        // runtime's mrope_section=[11, 11, 10] contract.
        rotary = Qwen35Language.RotaryEmbedding(
            dim: Int(Float(text.headDim ?? 256) * text.partialRotaryFactor),
            base: text.ropeTheta, mropeSection: config.extras.mropeSection)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, cache: QSAKVCache?) -> MLXArray? {
        let B = x.dim(0), S = x.dim(1)
        precondition(B == 1, "qwen4_exp QSA currently supports batch size 1")
        let past = cache?.offset ?? 0
        let qk = projection(x)
        let split = MLX.split(
            qk, indices: [extras.indexerNHeads * extras.indexerHeadDim], axis: -1)
        let rawKeys = split[1]
        let allKeys = cache?.updateIndexerKeys(rawKeys) ?? rawKeys
        let total = allKeys.dim(1)

        // The native QSA budget keeps every token while the context is at or
        // below this boundary.  Persist the raw index lane above, but avoid
        // pooling, normalizing, rotating, sorting, and materializing a mask
        // whose result would be all-visible anyway.
        guard total > extras.indexerBudget else { return nil }

        var query = qNorm(split[0].reshaped(B, S, extras.indexerNHeads, extras.indexerHeadDim))
            .transposed(0, 2, 1, 3)
        let blocks = total / extras.indexerCompressRatio
        guard blocks > 0 else { return nil }
        var pooled = allKeys[0..., ..<(blocks * extras.indexerCompressRatio), 0...]
            .reshaped(B, blocks, extras.indexerCompressRatio, extras.indexerHeadDim)
            .asType(.float32).mean(axis: 2).asType(allKeys.dtype)
        pooled = kNorm(pooled)
        let qPositions = MLXArray(past ..< (past + S)).asType(.int32).reshaped(1, S)
        let blockPositions = MLXArray(stride(
            from: 0, to: blocks * extras.indexerCompressRatio,
            by: extras.indexerCompressRatio)).asType(.int32).reshaped(1, blocks)
        let (qCos, qSin) = rotary(x: query, positionIds: qPositions)
        query = Qwen35Language.applyMultimodalRotaryPosEmb(
            q: query, k: query, cos: qCos, sin: qSin).0
        let pooled4 = expandedDimensions(pooled, axis: 1)
        let (kCos, kSin) = rotary(x: pooled4, positionIds: blockPositions)
        pooled = Qwen35Language.applyMultimodalRotaryPosEmb(
            q: pooled4, k: pooled4, cos: kCos, sin: kSin).0[0..., 0, 0..., 0...]
        return Qwen4ExpQSA.selectedTokenMask(
            query: query, pooledKeys: expandedDimensions(pooled, axis: 1),
            pastLen: past, compressRatio: extras.indexerCompressRatio,
            blockTopK: extras.indexerBudget / extras.indexerCompressRatio,
            keyLen: total)
    }
}

private final class Qwen4ExpAttention: Module {
    let text: Qwen35Configuration.TextConfiguration
    let scale: Float
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm
    @ModuleInfo(key: "indexer") var indexer: Qwen4ExpQSAIndexer
    let rotary: Qwen35Language.RotaryEmbedding

    init(_ config: Qwen4ExpConfiguration) {
        text = config.base.textConfiguration
        let headDim = text.headDim ?? (text.hiddenSize / text.attentionHeads)
        scale = pow(Float(headDim), -0.5)
        _qProj.wrappedValue = Linear(text.hiddenSize, text.attentionHeads * headDim * 2, bias: false)
        _kProj.wrappedValue = Linear(text.hiddenSize, text.kvHeads * headDim, bias: false)
        _vProj.wrappedValue = Linear(text.hiddenSize, text.kvHeads * headDim, bias: false)
        _oProj.wrappedValue = Linear(text.attentionHeads * headDim, text.hiddenSize, bias: false)
        _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: text.rmsNormEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: text.rmsNormEps)
        _indexer.wrappedValue = Qwen4ExpQSAIndexer(config)
        rotary = Qwen35Language.RotaryEmbedding(
            dim: Int(Float(headDim) * text.partialRotaryFactor),
            base: text.ropeTheta, mropeSection: config.extras.mropeSection)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, cache: QSAKVCache?,
        positionIds explicitPositions: MLXArray? = nil,
        positionOffset: Int = 0
    ) -> MLXArray {
        let B = x.dim(0), S = x.dim(1)
        let past = cache?.offset ?? 0
        let sparseMask = indexer(x, cache: cache)
        let headDim = text.headDim ?? (text.hiddenSize / text.attentionHeads)
        let qg = qProj(x).reshaped(B, S, text.attentionHeads, headDim * 2)
            .split(parts: 2, axis: -1)
        var query = qNorm(qg[0]).transposed(0, 2, 1, 3)
        let gate = qg[1].reshaped(B, S, -1)
        var key = kNorm(kProj(x).reshaped(B, S, text.kvHeads, headDim))
            .transposed(0, 2, 1, 3)
        let value = vProj(x).reshaped(B, S, text.kvHeads, headDim).transposed(0, 2, 1, 3)
        // Media prefill passes explicit 3-channel M-RoPE positions from
        // getRopeIndex; decode after media continues from past + ropeDelta.
        // Text-only keeps the sequential cache-offset positions (offset 0).
        let positions = explicitPositions
            ?? MLXArray((past + positionOffset) ..< (past + positionOffset + S))
                .asType(.int32).reshaped(1, S)
        let (cos, sin) = rotary(x: value, positionIds: positions)
        (query, key) = Qwen35Language.applyMultimodalRotaryPosEmb(
            q: query, k: key, cos: cos, sin: sin)

        // Pass an explicit pre-update causal mask. Falling through to
        // `attentionWithCacheUpdate`'s cache-derived mask asks the cache for
        // a mask after `update()` advanced its offset, which double-counts
        // this segment (S=69 produced a bogus [69,138] mask).
        let mask: MLXFast.ScaledDotProductAttentionMaskMode
        if S == 1 {
            // The cache contains no future keys during single-token decode.
            // Below the QSA budget this is the optimized no-mask SDPA path;
            // above it, the sparse selector alone is the complete mask.
            mask = sparseMask.map { .array($0) } ?? .none
        } else {
            let keyLength = past + S
            let queryPositions = MLXArray(past ..< (past + S)).asType(.int32).reshaped(S, 1)
            let keyPositions = MLXArray(0 ..< keyLength).asType(.int32).reshaped(1, keyLength)
            var causal = MLX.lessEqual(keyPositions, queryPositions).reshaped(1, 1, S, keyLength)
            if let sparseMask { causal = MLX.logicalAnd(causal, sparseMask) }
            mask = .array(causal)
        }
        let output = attentionWithCacheUpdate(
            queries: query, keys: key, values: value, cache: cache,
            scale: scale, mask: mask)
            .transposed(0, 2, 1, 3).reshaped(B, S, -1)
        return oProj(output * sigmoid(gate))
    }
}

private final class Qwen4ExpDecoderLayer: Module {
    private static let profiledLayer = RuntimeEnvironment.value("VMLX_QWEN4_EXP_PROFILE_LAYER").flatMap(Int.init)
    private static let profileLimit = Int(RuntimeEnvironment.value("VMLX_QWEN4_EXP_PROFILE_LIMIT") ?? "4") ?? 4
    private static let profileSequenceLength = Int(RuntimeEnvironment.value("VMLX_QWEN4_EXP_PROFILE_SEQUENCE") ?? "1") ?? 1

    let layerIndex: Int
    let isLinear: Bool
    let ple: Qwen4ExpPLE?
    @ModuleInfo(key: "linear_attn") var linearAttention: Qwen35Language.GatedDeltaNet?
    @ModuleInfo(key: "self_attn") var attention: Qwen4ExpAttention?
    @ModuleInfo(key: "mlp") var mlp: Qwen35Language.SparseMoeBlock
    @ModuleInfo(key: "attn_hyper_connection") var attentionResidual: Qwen4ExpGatedResidual
    @ModuleInfo(key: "mlp_hyper_connection") var mlpResidual: Qwen4ExpGatedResidual
    @ModuleInfo(key: "ple") var pleModule: Qwen4ExpPLE?
    private var profileCount = 0

    init(_ config: Qwen4ExpConfiguration, layerIndex: Int) {
        self.layerIndex = layerIndex
        let text = config.base.textConfiguration
        let types = config.extras.layerTypes ?? (0 ..< text.hiddenLayers).map {
            ($0 + 1) % text.fullAttentionInterval == 0 ? "full_attention" : "linear_attention"
        }
        isLinear = types[layerIndex] == "linear_attention"
        if isLinear {
            _linearAttention.wrappedValue = Qwen35Language.GatedDeltaNet(
                text, outputGateSigmoid: config.extras.outputGateType == "sigmoid",
                fuseDecodeInputProjections: true)
        } else {
            _attention.wrappedValue = Qwen4ExpAttention(config)
        }
        _mlp.wrappedValue = Qwen35Language.SparseMoeBlock(
            text, layerIdx: layerIndex, allowFusedGateUpCache: false,
            compileDecodeRegions: true,
            decodeEquivalentVerifierRows: config.hasUniformQ4G64TrunkRoutedExperts)
        _attentionResidual.wrappedValue = Qwen4ExpGatedResidual(config)
        _mlpResidual.wrappedValue = Qwen4ExpGatedResidual(config)
        if let pleIndex = config.extras.pleLayerIds.firstIndex(of: layerIndex + 1) {
            let module = Qwen4ExpPLE(config, pleLayerIndex: pleIndex)
            self.ple = module
            _pleModule.wrappedValue = module
        } else {
            self.ple = nil
        }
        super.init()
    }

    func callAsFunction(
        _ input: MLXArray, inputIds: MLXArray, cache: KVCache?,
        plePrefetch: Qwen4ExpPLE.Prefetch? = nil,
        pleEmbedding: MLXArray? = nil,
        recordPrefixCommitStates: Bool = false,
        positionIds: MLXArray? = nil,
        positionOffset: Int = 0
    ) -> MLXArray {
        if Self.profiledLayer == layerIndex,
            inputIds.size == Self.profileSequenceLength,
            profileCount < Self.profileLimit
        {
            // Isolate this layer from the lazy graph produced by preceding
            // layers.  Without this barrier, the first timed component also
            // pays for every unevaluated dependency feeding `input`, which
            // misattributes upstream work to `attention_mix`.
            MLX.eval(input)
            var timings: [(String, Double)] = []
            func evaluated<T>(_ label: String, _ body: () -> (T, [MLXArray])) -> T {
                let started = CFAbsoluteTimeGetCurrent()
                let (result, arrays) = body()
                MLX.eval(arrays)
                timings.append((label, (CFAbsoluteTimeGetCurrent() - started) * 1000))
                return result
            }

            var hyper = input
            if let ple {
                hyper = evaluated("ple") {
                    let result = hyper + ple(
                        hyper, inputIds: inputIds, cache: cache as? MambaCache,
                        prefetch: plePrefetch,
                        preloadedEmbedding: pleEmbedding,
                        recordPrefixCommitStates: recordPrefixCommitStates)
                    return (result, [result])
                }
            }
            let attentionMix = evaluated("attention_mix") {
                let result = attentionResidual.mix(hyper)
                return (result, [result.0] + (result.1.map { [$0] } ?? []))
            }
            let attentionOutput = evaluated(isLinear ? "gdn" : "qsa") {
                let result = isLinear
                    ? linearAttention!(
                        attentionMix.0, cache: cache as? MambaCache,
                        recordPrefixCommitStates: recordPrefixCommitStates)
                    : attention!(
                        attentionMix.0, cache: cache as? QSAKVCache,
                        positionIds: positionIds, positionOffset: positionOffset)
                return (result, [result])
            }
            hyper = evaluated("attention_combine") {
                let result = attentionResidual.combine(
                    hyper, block: attentionOutput, injection: attentionMix.1!)
                return (result, [result])
            }
            let mlpMix = evaluated("mlp_mix") {
                let result = mlpResidual.mix(hyper)
                return (result, [result.0] + (result.1.map { [$0] } ?? []))
            }
            let mlpOutput = evaluated("moe") {
                let result = mlp(mlpMix.0)
                return (result, [result])
            }
            let result = evaluated("mlp_combine") {
                let result = mlpResidual.combine(
                    hyper, block: mlpOutput, injection: mlpMix.1!)
                return (result, [result])
            }
            profileCount += 1
            let fields = timings.map { String(format: "%@=%.3fms", $0.0, $0.1) }
                .joined(separator: " ")
            FileHandle.standardError.write(Data(
                ("[Qwen4ExpProfile] layer=\(layerIndex)"
                    + " sequence_length=\(inputIds.size) sample=\(profileCount)"
                    + " \(fields)\n").utf8))
            return result
        }

        var hyper = input
        if let ple {
            hyper = hyper + ple(
                hyper, inputIds: inputIds, cache: cache as? MambaCache,
                prefetch: plePrefetch,
                preloadedEmbedding: pleEmbedding,
                recordPrefixCommitStates: recordPrefixCommitStates)
        }
        let (attentionInput, inject) = attentionResidual.mix(hyper)
        let attentionOutput = isLinear
            ? linearAttention!(
                attentionInput, cache: cache as? MambaCache,
                recordPrefixCommitStates: recordPrefixCommitStates)
            : attention!(
                attentionInput, cache: cache as? QSAKVCache,
                positionIds: positionIds, positionOffset: positionOffset)
        hyper = attentionResidual.combine(hyper, block: attentionOutput, injection: inject!)
        let (mlpInput, mlpInject) = mlpResidual.mix(hyper)
        return mlpResidual.combine(hyper, block: mlp(mlpInput), injection: mlpInject!)
    }
}

/// The released Qwen4Exp MTP head is not a Qwen35 residual block. It owns one
/// full QSA layer with the same two hyper-connection mixers and sparse MoE as
/// the trunk, but no PLE and no GDN state.
private final class Qwen4ExpMTPDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: Qwen4ExpAttention
    @ModuleInfo(key: "mlp") var mlp: Qwen35Language.SparseMoeBlock
    @ModuleInfo(key: "attn_hyper_connection") var attentionResidual: Qwen4ExpGatedResidual
    @ModuleInfo(key: "mlp_hyper_connection") var mlpResidual: Qwen4ExpGatedResidual

    init(_ config: Qwen4ExpConfiguration, layerIndex: Int) {
        _attention.wrappedValue = Qwen4ExpAttention(config)
        _mlp.wrappedValue = Qwen35Language.SparseMoeBlock(
            config.base.textConfiguration, layerIdx: layerIndex,
            allowFusedGateUpCache: false, compileDecodeRegions: true,
            decodeEquivalentVerifierRows: false)
        _attentionResidual.wrappedValue = Qwen4ExpGatedResidual(config)
        _mlpResidual.wrappedValue = Qwen4ExpGatedResidual(config)
        super.init()
    }

    func callAsFunction(_ input: MLXArray, inputIds: MLXArray, cache: QSAKVCache?)
        -> MLXArray
    {
        var hyper = input
        let (attentionInput, injection) = attentionResidual.mix(hyper)
        hyper = attentionResidual.combine(
            hyper, block: attention(attentionInput, cache: cache),
            injection: injection!)
        let (mlpInput, mlpInjection) = mlpResidual.mix(hyper)
        return mlpResidual.combine(
            hyper, block: mlp(mlpInput), injection: mlpInjection!)
    }
}

/// Native four-stream Qwen4Exp next-token head.
///
/// `hiddenStates` is the trunk's pre-mixer `[B,T,4*D]` state. The released
/// head globally normalizes that vector, projects each reshaped stream with
/// the shared `fc_hidden`, adds the projected next-token embedding to every
/// stream, executes its private full-QSA layer, and collapses only for logits.
private final class Qwen4ExpMTPZeroCenteredRMSNorm: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray

    private let eps: Float

    init(dimensions: Int, eps: Float) {
        self._weight.wrappedValue = MLXArray.zeros([dimensions])
        self.eps = eps
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        // Qwen4Exp stores only the two MTP fusion-norm tensors around zero.
        // Keep the reduction in F32, apply the checkpoint as `1 + weight`,
        // and return to the bundle-owned compute dtype before either affine.
        let inputF32 = input.asType(.float32)
        let variance = (inputF32 * inputF32).mean(axis: -1, keepDims: true)
        return (inputF32 * rsqrt(variance + eps)
            * (1 + weight.asType(.float32))).asType(input.dtype)
    }
}

private final class Qwen4ExpMTPModule: Module {
    @ModuleInfo(key: "pre_fc_norm_hidden")
    var preFCNormHidden: Qwen4ExpMTPZeroCenteredRMSNorm
    @ModuleInfo(key: "pre_fc_norm_embedding")
    var preFCNormEmbedding: Qwen4ExpMTPZeroCenteredRMSNorm
    @ModuleInfo(key: "fc_embedding") var fcEmbedding: Linear
    @ModuleInfo(key: "fc_hidden") var fcHidden: Linear
    @ModuleInfo(key: "layers") var layers: [Qwen4ExpMTPDecoderLayer]
    @ModuleInfo(key: "hyper_connection_mixer") var mixer: Qwen4ExpGatedResidual

    private let hiddenSize: Int
    private let hcCount: Int

    init(_ config: Qwen4ExpConfiguration) {
        let text = config.base.textConfiguration
        hiddenSize = text.hiddenSize
        hcCount = config.extras.hcCount
        _preFCNormHidden.wrappedValue = Qwen4ExpMTPZeroCenteredRMSNorm(
            dimensions: hiddenSize * hcCount, eps: text.rmsNormEps)
        _preFCNormEmbedding.wrappedValue = Qwen4ExpMTPZeroCenteredRMSNorm(
            dimensions: hiddenSize, eps: text.rmsNormEps)
        _fcEmbedding.wrappedValue = Linear(hiddenSize, hiddenSize, bias: false)
        _fcHidden.wrappedValue = Linear(hiddenSize, hiddenSize, bias: false)
        _layers.wrappedValue = (0 ..< config.extras.mtpNumHiddenLayers).map {
            Qwen4ExpMTPDecoderLayer(config, layerIndex: $0)
        }
        _mixer.wrappedValue = Qwen4ExpGatedResidual(config, combines: false)
        super.init()
    }

    func preMixerHidden(
        hiddenStates: MLXArray, nextTokenIds: MLXArray,
        embedding: Embedding, cache: [KVCache]?
    ) -> MLXArray {
        let expected = hiddenSize * hcCount
        precondition(
            hiddenStates.dim(-1) == expected,
            "qwen4_exp MTP requires trunk pre-mixer four-stream hidden")
        let embedded = fcEmbedding(preFCNormEmbedding(embedding(nextTokenIds)))
        let normalized = preFCNormHidden(hiddenStates).reshaped(
            Array(hiddenStates.shape.dropLast()) + [hcCount, hiddenSize])
        var hidden = fcHidden(normalized) + expandedDimensions(embedded, axis: -2)
        hidden = hidden.reshaped(hiddenStates.shape)
        for (index, layer) in layers.enumerated() {
            hidden = layer(
                hidden, inputIds: nextTokenIds,
                cache: cache?[index] as? QSAKVCache)
        }
        return hidden
    }

    func mixed(_ preMixerHidden: MLXArray) -> MLXArray {
        mixer.mix(preMixerHidden).0
    }

    func makeCache() -> [KVCache] {
        layers.map { _ in QSAKVCache() as KVCache }
    }
}

private final class Qwen4ExpTextModel: Module {
    struct ForwardResult {
        let mixed: MLXArray
        let preMixer: MLXArray
    }

    let config: Qwen4ExpConfiguration
    @ModuleInfo(key: "embed_tokens") var embedding: Embedding
    @ModuleInfo(key: "layers") var layers: [Qwen4ExpDecoderLayer]
    @ModuleInfo(key: "hyper_connection_mixer") var mixer: Qwen4ExpGatedResidual

    init(_ config: Qwen4ExpConfiguration) {
        self.config = config
        let text = config.base.textConfiguration
        _embedding.wrappedValue = Embedding(
            embeddingCount: text.vocabularySize, dimensions: text.hiddenSize)
        _layers.wrappedValue = (0 ..< text.hiddenLayers).map {
            Qwen4ExpDecoderLayer(config, layerIndex: $0)
        }
        _mixer.wrappedValue = Qwen4ExpGatedResidual(config, combines: false)
        super.init()
    }

    func forward(
        _ inputIds: MLXArray, embeddings: MLXArray? = nil, cache: [KVCache]?,
        pleEmbeddings: [Int: MLXArray]? = nil,
        recordPrefixCommitStates: Bool = false,
        positionIds: MLXArray? = nil,
        positionOffset: Int = 0
    ) -> ForwardResult {
        let auditDTypes = Qwen4ExpDTypeTrace.claimTextAudit()
        var plePrefetches: [Int: Qwen4ExpPLE.Prefetch] = [:]
        for (index, layer) in layers.enumerated() {
            if pleEmbeddings?[index] == nil, let ple = layer.ple {
                plePrefetches[index] = ple.prefetch(
                    inputIds: inputIds, cache: cache?[index] as? MambaCache)
            }
        }
        var hidden = embeddings ?? embedding(inputIds)
        let rawEmbeddingDType = hidden.dtype
        if let computeDType = config.declaredComputeDType,
            hidden.dtype != computeDType
        {
            hidden = hidden.asType(computeDType)
        }
        let computeEmbeddingDType = hidden.dtype
        var layerDTypes: [DType] = []
        if auditDTypes { layerDTypes.reserveCapacity(layers.count) }
        hidden = tiled(hidden, repetitions: [1, 1, config.extras.hcCount])
        for (index, layer) in layers.enumerated() {
            hidden = layer(
                hidden, inputIds: inputIds, cache: cache?[index],
                plePrefetch: plePrefetches[index],
                pleEmbedding: pleEmbeddings?[index],
                recordPrefixCommitStates: recordPrefixCommitStates,
                positionIds: positionIds,
                positionOffset: positionOffset)
            if inputIds.dim(1) > 1, plePrefetches[index + 1] != nil {
                // Commit the resident layer immediately so its Metal work can
                // overlap the already-running SSD row prefetch during prefill.
                // Decode row reads finish before the PLE layer needs them; a
                // one-token barrier here only splits every decode graph into a
                // second command buffer and reduces steady-state throughput.
                asyncEval(hidden)
            }
            if auditDTypes { layerDTypes.append(hidden.dtype) }
        }
        let result = mixer.mix(hidden).0
        if auditDTypes {
            Qwen4ExpDTypeTrace.logTextAudit(
                rawEmbedding: rawEmbeddingDType, computeEmbedding: computeEmbeddingDType,
                layerDTypes: layerDTypes, finalMixer: result.dtype,
                declared: config.declaredComputeDType)
        }
        return ForwardResult(mixed: result, preMixer: hidden)
    }

    func callAsFunction(
        _ inputIds: MLXArray, embeddings: MLXArray? = nil, cache: [KVCache]?,
        positionIds: MLXArray? = nil, positionOffset: Int = 0
    ) -> MLXArray {
        forward(
            inputIds, embeddings: embeddings, cache: cache,
            positionIds: positionIds, positionOffset: positionOffset
        ).mixed
    }
}

protocol Qwen4ExpModelDirectoryConfigurable: AnyObject {
    func configure(modelDirectory: URL) throws
}

public final class Qwen4Exp: Module, VLMModel, Qwen4ExpModelDirectoryConfigurable,
    SafetensorsLoadKeyExcluding, NativeMTPModel, DFlash2StagedVerifyRollbackModel,
    CompiledDecodeExternalInputModel
{
    /// QSA index selection and its path-dependent cache currently require a
    /// single sequence. Keep concurrent requests queued until a B-wide QSA
    /// cache/indexer contract is implemented and proven.
    public var maximumSupportedDecodeBatchSize: Int? { 1 }

    public let config: Qwen4ExpConfiguration
    @ModuleInfo(key: "language_model") private var textModel: Qwen4ExpTextModel
    @ModuleInfo(key: "lm_head") private var head: Linear
    @ModuleInfo(key: "mtp") private var mtp: Qwen4ExpMTPModule?
    @ModuleInfo(key: "visual") private var visionModel: Qwen3VLVision.VisionModel

    /// M-RoPE delta established by the most recent media prefill, keyed by the
    /// conversation's cache identity so concurrent sessions do not cross.
    /// Decode positions after a media prefill continue at
    /// `cache.offset + delta`, matching the reference runtime's
    /// `_rope_deltas` contract. Text-only conversations keep delta 0.
    private let ropeDeltaLock = NSLock()
    nonisolated(unsafe) private var ropeDeltas: [ObjectIdentifier: Int] = [:]

    public init(_ config: Qwen4ExpConfiguration) {
        self.config = config
        _textModel.wrappedValue = Qwen4ExpTextModel(config)
        _head.wrappedValue = Linear(
            config.base.textConfiguration.hiddenSize,
            config.base.textConfiguration.vocabularySize, bias: false)
        if config.extras.mtpNumHiddenLayers > 0 {
            _mtp.wrappedValue = Qwen4ExpMTPModule(config)
        }
        _visionModel.wrappedValue = Qwen3VLVision.VisionModel(config.base.visionConfiguration)
        super.init()
    }

    private func setRopeDelta(_ delta: Int, for cache: [KVCache]?) {
        guard let first = cache?.first else { return }
        ropeDeltaLock.lock()
        defer { ropeDeltaLock.unlock() }
        if ropeDeltas.count > 128 { ropeDeltas.removeAll() }
        ropeDeltas[ObjectIdentifier(first as AnyObject)] = delta
    }

    private func ropeDelta(for cache: [KVCache]?) -> Int {
        guard let first = cache?.first else { return 0 }
        ropeDeltaLock.lock()
        defer { ropeDeltaLock.unlock() }
        return ropeDeltas[ObjectIdentifier(first as AnyObject)] ?? 0
    }

    public var vocabularySize: Int { config.base.vocabSize }
    public var loraLayers: [Module] { textModel.layers }
    var usesResidentNativeAffineRoutedExperts: Bool {
        textModel.layers.allSatisfy { $0.mlp.switchMLP is SwitchGLU }
    }
    public var requiresExactTensorMmapBuffers: Bool { true }
    public var requiresResidentSafetensorsWeights: Bool { true }

    func configure(modelDirectory: URL) throws {
        for (index, layer) in textModel.layers.enumerated() {
            try layer.ple?.configureTable(modelDirectory: modelDirectory, layerIndex: index)
        }
    }

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        textModel.layers.map { layer in
            if layer.isLinear {
                return MambaCache(slots: layer.ple == nil ? 2 : 6) as KVCache
            }
            return QSAKVCache() as KVCache
        }
    }

    public func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        let inputIds = input.text.tokens

        guard input.image != nil || input.video != nil else {
            setRopeDelta(0, for: cache)
            return .logits(LMOutput(logits: callAsFunction(inputIds, cache: cache)))
        }

        // Media prefill: encode pixels through the bundled quantized vision
        // tower, scatter image rows onto image placeholders and video rows
        // onto video placeholders (never interleaved), and derive 3-channel
        // M-RoPE positions plus the rope delta that decode continues from.
        let visionDType = visionModel.patchEmbed.proj.weight.dtype
        var pixelParts: [MLXArray] = []
        var imageFrames: [THW]?
        var videoFrames: [THW]?
        if let image = input.image {
            pixelParts.append(image.pixels.asType(visionDType))
            imageFrames = image.frames
        }
        if let video = input.video {
            pixelParts.append(video.pixels.asType(visionDType))
            videoFrames = video.frames
        }
        var frames: [THW] = []
        if let imageFrames { frames.append(contentsOf: imageFrames) }
        if let videoFrames { frames.append(contentsOf: videoFrames) }
        guard !pixelParts.isEmpty, !frames.isEmpty else {
            throw ModelFactoryError.unsupportedModelType(
                "qwen4_exp media input carried no pixels/frames")
        }

        let textEmbeds = textModel.embedding(inputIds)
        let (visionHidden, _) = visionModel(concatenated(pixelParts), gridTHW: frames)
        let features = visionHidden.asType(textEmbeds.dtype)

        let mergeSize = config.base.visionConfiguration.spatialMergeSize
        let divisor = max(1, mergeSize * mergeSize)
        let imageRowCount = (imageFrames ?? []).reduce(0) { $0 + $1.product / divisor }
        let mergedEmbeddings = try Self.scatterMediaFeatures(
            features: features,
            imageRowCount: imageRowCount,
            inputEmbeds: textEmbeds,
            inputIds: inputIds,
            imageTokenId: config.base.imageTokenIndex,
            videoTokenId: config.base.videoTokenIndex)

        let (positionIds, deltas) = Qwen3VLLanguage.getRopeIndex(
            inputIds: inputIds,
            imageGridTHW: imageFrames,
            videoGridTHW: videoFrames,
            spatialMergeSize: mergeSize,
            imageTokenId: config.base.imageTokenIndex,
            videoTokenId: config.base.videoTokenIndex,
            visionStartTokenId: config.base.visionStartTokenId,
            attentionMask: nil)
        setRopeDelta(deltas.reshaped(-1)[0].item(Int.self), for: cache)

        let headInput = textModel(
            inputIds, embeddings: mergedEmbeddings, cache: cache,
            positionIds: positionIds)
        return .logits(LMOutput(logits: projectToLogits(headInput)))
    }

    /// Fixed per-modality scatter: rows [0, imageRowCount) are image-batch
    /// rows and land only on image placeholders; the remainder are video rows
    /// and land only on video placeholders. A total-count mismatch fails
    /// loudly instead of silently mis-assigning features.
    private static func scatterMediaFeatures(
        features: MLXArray,
        imageRowCount: Int,
        inputEmbeds: MLXArray,
        inputIds: MLXArray,
        imageTokenId: Int,
        videoTokenId: Int
    ) throws -> MLXArray {
        let imageMask = (inputIds .== MLXArray(imageTokenId))
        let videoMask = (inputIds .== MLXArray(videoTokenId))
        let totalPlaceholders = (imageMask .|| videoMask).sum().item(Int.self)
        let featureRows = features.dim(0)
        guard totalPlaceholders == featureRows else {
            throw ModelFactoryError.unsupportedModelType(
                "qwen4_exp media features (\(featureRows) rows) do not match "
                    + "placeholder tokens (\(totalPlaceholders))")
        }

        let originalShape = inputEmbeds.shape
        let width = inputEmbeds.dim(-1)
        var result = inputEmbeds.flattened()

        func scatter(mask: MLXArray, rows: MLXArray) throws {
            let flags = mask.flattened().asType(.bool).asArray(Bool.self)
            var positions: [Int] = []
            positions.reserveCapacity(rows.dim(0))
            for (index, flag) in flags.enumerated() where flag { positions.append(index) }
            guard !positions.isEmpty else { return }
            guard positions.count == rows.dim(0) else {
                throw ModelFactoryError.unsupportedModelType(
                    "qwen4_exp per-modality media rows (\(rows.dim(0))) do not "
                        + "match placeholders (\(positions.count))")
            }
            var flatIndices: [UInt32] = []
            flatIndices.reserveCapacity(positions.count * width)
            for position in positions {
                let base = position * width
                for offset in 0 ..< width { flatIndices.append(UInt32(base + offset)) }
            }
            result[MLXArray(flatIndices)] = rows.flattened()
        }

        if imageRowCount > 0 {
            try scatter(mask: imageMask, rows: features[0 ..< imageRowCount])
        }
        if featureRows > imageRowCount {
            try scatter(mask: videoMask, rows: features[imageRowCount ..< featureRows])
        }
        return result.reshaped(originalShape)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let headInput = textModel(
            inputs, cache: cache, positionOffset: ropeDelta(for: cache))
        return projectToLogits(headInput)
    }

    public func compiledDecodeExternalInputs(
        inputIds: MLXArray, cache: [KVCache]
    ) -> [MLXArray] {
        textModel.layers.enumerated().compactMap { index, layer in
            layer.ple?.compiledDecodeEmbedding(
                inputIds: inputIds[.newAxis],
                cache: cache[index] as? MambaCache)
        }
    }

    public func compiledDecodeForward(
        inputIds: MLXArray, externalInputs: [MLXArray], cache: [KVCache]
    ) -> MLXArray {
        let pleLayerIndices = textModel.layers.indices.filter {
            textModel.layers[$0].ple != nil
        }
        precondition(
            externalInputs.count == pleLayerIndices.count,
            "qwen4_exp compiled PLE input count mismatch")
        var pleEmbeddings: [Int: MLXArray] = [:]
        for (offset, layerIndex) in pleLayerIndices.enumerated() {
            pleEmbeddings[layerIndex] = externalInputs[offset]
        }
        let forward = textModel.forward(
            inputIds[.newAxis], cache: cache,
            pleEmbeddings: pleEmbeddings)
        return projectToLogits(forward.mixed)
    }

    private func projectToLogits(_ headInput: MLXArray) -> MLXArray {
        let logits = head(headInput)
        let result: MLXArray
        guard let computeDType = config.declaredComputeDType,
            logits.dtype != computeDType
        else {
            result = logits
            if Qwen4ExpDTypeTrace.claimHeadAudit() {
                Qwen4ExpDTypeTrace.logHeadAudit(
                    input: headInput.dtype, rawLogits: logits.dtype, returnedLogits: result.dtype)
            }
            return result
        }
        result = logits.asType(computeDType)
        if Qwen4ExpDTypeTrace.claimHeadAudit() {
            Qwen4ExpDTypeTrace.logHeadAudit(
                input: headInput.dtype, rawLogits: logits.dtype, returnedLogits: result.dtype)
        }
        return result
    }

    // MARK: - NativeMTPModel

    public var nativeMTPAvailable: Bool { mtp != nil }

    public func makeNativeMTPCache() -> [KVCache] {
        mtp?.makeCache() ?? []
    }

    public func nativeBackboneForward(
        _ inputs: MLXArray, cache: [KVCache]?
    ) -> NativeMTPForwardResult {
        let forward = textModel.forward(inputs, cache: cache)
        return NativeMTPForwardResult(
            logits: projectToLogits(forward.mixed),
            hiddenStates: forward.preMixer)
    }

    public func nativeBackboneMTPVerifyForward(
        _ inputs: MLXArray, cache: [KVCache]?
    ) -> NativeMTPForwardResult {
        let forward = textModel.forward(
            inputs, cache: cache, recordPrefixCommitStates: true)
        return NativeMTPForwardResult(
            logits: projectToLogits(forward.mixed),
            hiddenStates: forward.preMixer)
    }

    public func nativeMTPForward(
        hiddenStates: MLXArray, nextTokenIds: MLXArray, cache: [KVCache]?
    ) -> NativeMTPForwardResult {
        guard let mtp else {
            fatalError("Qwen4Exp nativeMTPForward called without an MTP module")
        }
        let preMixer = mtp.preMixerHidden(
            hiddenStates: hiddenStates, nextTokenIds: nextTokenIds,
            embedding: textModel.embedding, cache: cache)
        return NativeMTPForwardResult(
            logits: projectToLogits(mtp.mixed(preMixer)),
            hiddenStates: preMixer)
    }

    // MARK: - Native MTP staged verifier rollback

    public func commitVerifiedBlock(cache: [KVCache], acceptedInputs: Int) -> Bool {
        for (index, layer) in textModel.layers.enumerated() where layer.isLinear {
            guard index < cache.count, let mamba = cache[index] as? MambaCache,
                let gdn = layer.linearAttention,
                gdn.commitVerifyStash(cache: mamba, acceptedInputs: acceptedInputs),
                layer.ple == nil
            else { return false }
        }
        return true
    }

    public func commitStagedVerifiedBlock(
        cache: [KVCache], acceptedInputs: Int, blockLength: Int
    ) -> Bool {
        for (index, layer) in textModel.layers.enumerated() where layer.isLinear {
            guard index < cache.count, let mamba = cache[index] as? MambaCache,
                let gdn = layer.linearAttention,
                gdn.commitVerifyStaged(
                    cache: mamba, acceptedInputs: acceptedInputs,
                    blockLength: blockLength)
            else { return false }
            if let ple = layer.ple,
                !ple.commitVerifyStaged(
                    cache: mamba, acceptedInputs: acceptedInputs,
                    blockLength: blockLength)
            { return false }
        }
        return true
    }

    public func excludeFromGenericSafetensorsLoad(key: String) -> Bool {
        key.contains("ngram_embedding.shards")
            || key.contains("ngram_heads_")
            || key.contains("layer_multipliers")
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var output: [String: MLXArray] = [:]
        output.reserveCapacity(weights.count)
        for (originalKey, originalValue) in weights {
            if (originalKey.hasPrefix("mtp.") && mtp == nil)
                || originalKey.contains("ngram_embedding.shards")
                || originalKey.contains("ngram_heads_") || originalKey.contains("layer_multipliers")
            { continue }
            var key = originalKey
            var value = originalValue
            if key.hasPrefix("model.language_model.") {
                key.removeFirst("model.".count)
            }
            if key.hasPrefix("model.mtp.") {
                key.removeFirst("model.".count)
            }
            if key.hasPrefix("mtp."),
                key.contains(".mlp.experts.gate_up_proj.")
            {
                let parts = MLX.split(value, parts: 2, axis: 1)
                let suffix = String(key.split(separator: ".").last!)
                let stem = key.replacingOccurrences(
                    of: ".mlp.experts.gate_up_proj.\(suffix)",
                    with: ".mlp.switch_mlp")
                output[stem + ".gate_proj.\(suffix)"] = parts[0]
                output[stem + ".up_proj.\(suffix)"] = parts[1]
                continue
            }
            if key.hasPrefix("mtp."), key.contains(".mlp.experts.down_proj.") {
                key = key.replacingOccurrences(
                    of: ".mlp.experts.down_proj.",
                    with: ".mlp.switch_mlp.down_proj.")
            }
            if key.contains(".ple.ple_embedding.") {
                key = key.replacingOccurrences(of: ".ple.ple_embedding.", with: ".ple.")
            }
            // The bundle stores the vision patch embed in HF Conv3d layout
            // [out, in, t, h, w]; MLX conv expects channels-last
            // [out, t, h, w, in]. Same guard as Qwen3VL's sanitize so an
            // already-MLX-format tensor passes through untouched.
            if key == "visual.patch_embed.proj.weight",
                value.ndim == 5,
                value.dim(-1) != config.base.visionConfiguration.inChannels
            {
                value = value.transposed(0, 2, 3, 4, 1)
            }
            if key.hasSuffix(".conv1d.weight"), value.ndim == 3, value.dim(1) == 1 {
                value = value.squeezed(axis: 1).transposed(1, 0)
            }
            if key.hasSuffix(".ple.conv1d.weight"), value.ndim == 3 {
                key = key.replacingOccurrences(of: ".ple.conv1d.weight", with: ".ple.conv1d_weight")
                value = value.squeezed(axis: 1)
            }
            let normAlreadyShifted =
                config.jangMetadata?.normConvention?.lowercased() == "runtime_plus1_applied"
            if !normAlreadyShifted,
                key.contains("hc_norm.weight") || key.contains(".ple.norm_")
                    || key.contains(".q_norm.weight") || key.contains(".k_norm.weight")
                    || key.contains("layernorm.weight")
            { value = value + 1 }
            output[key] = value
        }

        // Record the checkpoint storage dtype before the loader materializes
        // the runtime BF16 parameter bank. text_config.dtype owns the
        // residual, projection, and logits stream; the GDN recurrent state is
        // still explicitly governed by mamba_ssm_dtype.
        var affineMetadataCount = 0
        var affineMetadataDTypes = Set<String>()
        for (key, value) in output
        where key.hasSuffix(".scales") || key.hasSuffix(".biases") {
            let stem = String(key.dropLast(7))
            guard output[stem + ".weight"]?.dtype == .uint32 else { continue }
            affineMetadataCount += 1
            affineMetadataDTypes.insert(String(describing: value.dtype))
        }
        let computeDType = config.declaredComputeDType.map(String.init(describing:))
            ?? "unresolved"
        let storageDTypes = affineMetadataDTypes.sorted().joined(separator: ",")
        FileHandle.standardError.write(Data(
            "[Qwen4Exp] declared_compute_dtype=\(computeDType) affine_metadata_checkpoint_dtype=\(storageDTypes) affine_metadata_count=\(affineMetadataCount)\n".utf8))

        return output
    }
}
