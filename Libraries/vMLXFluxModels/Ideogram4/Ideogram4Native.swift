import Foundation
@preconcurrency import MLX
import MLXNN
import MLXRandom
import VMLXTokenizers
import vMLXFluxKit

enum Ideogram4Rotary {
    static func rotateHalf(_ x: MLXArray) -> MLXArray {
        let half = x.dim(-1) / 2
        return concatenated([-x[.ellipsis, half ..< x.dim(-1)], x[.ellipsis, 0 ..< half]], axis: -1)
    }
}

struct Ideogram4PromptInputs {
    let tokenIDs: MLXArray
    let textPositionIDs: MLXArray
    let positionIDs: MLXArray
    let segmentIDs: MLXArray
    let indicator: MLXArray
    let attentionMask: MLXArray
    let numImageTokens: Int
    let gridHeight: Int
    let gridWidth: Int
    let maxTextTokens: Int

    init(tokenIDs ids: [Int32], width: Int, height: Int) throws {
        guard !ids.isEmpty else {
            throw FluxError.invalidRequest("Ideogram prompt must not be empty")
        }
        guard width % 16 == 0, height % 16 == 0 else {
            throw FluxError.invalidRequest("Ideogram width/height must be divisible by 16")
        }
        let gridHeight = height / 16
        let gridWidth = width / 16
        let imageTokens = gridHeight * gridWidth
        let textTokens = ids.count
        let total = textTokens + imageTokens

        var tokenValues = [Int32](repeating: 0, count: total)
        var textPositionValues = [Int32](repeating: 0, count: total * 3)
        var positionValues = [Int32](repeating: 0, count: total * 3)
        var segmentValues = [Int32](repeating: -1, count: total)
        var indicatorValues = [Int32](repeating: 0, count: total)
        var attentionValues = [Int32](repeating: 0, count: total)

        for index in 0 ..< textTokens {
            tokenValues[index] = ids[index]
            for axis in 0 ..< 3 {
                textPositionValues[index * 3 + axis] = Int32(index)
                positionValues[index * 3 + axis] = Int32(index)
            }
            segmentValues[index] = 1
            indicatorValues[index] = 3
            attentionValues[index] = 1
        }

        for row in 0 ..< gridHeight {
            for column in 0 ..< gridWidth {
                let imageIndex = row * gridWidth + column
                let tokenIndex = textTokens + imageIndex
                let base = tokenIndex * 3
                positionValues[base] = 65536
                positionValues[base + 1] = Int32(65536 + row)
                positionValues[base + 2] = Int32(65536 + column)
                segmentValues[tokenIndex] = 1
                indicatorValues[tokenIndex] = 2
            }
        }

        self.tokenIDs = MLXArray(tokenValues, [1, total])
        self.textPositionIDs = MLXArray(textPositionValues, [1, total, 3])
        self.positionIDs = MLXArray(positionValues, [1, total, 3])
        self.segmentIDs = MLXArray(segmentValues, [1, total])
        self.indicator = MLXArray(indicatorValues, [1, total])
        self.attentionMask = MLXArray(attentionValues, [1, total])
        self.numImageTokens = imageTokens
        self.gridHeight = gridHeight
        self.gridWidth = gridWidth
        self.maxTextTokens = textTokens
    }

    func negative(llmFeaturesDim: Int) -> Ideogram4NegativeInputs {
        Ideogram4NegativeInputs(
            positionIDs: positionIDs[0..., maxTextTokens..., 0...],
            segmentIDs: segmentIDs[0..., maxTextTokens...],
            indicator: indicator[0..., maxTextTokens...],
            llmFeatures: MLXArray.zeros([1, numImageTokens, llmFeaturesDim], dtype: .float32))
    }
}

struct Ideogram4NegativeInputs {
    let positionIDs: MLXArray
    let segmentIDs: MLXArray
    let indicator: MLXArray
    let llmFeatures: MLXArray
}

struct Ideogram4Scheduler {
    let tValues: [Float]
    let sValues: [Float]
    let guidanceValues: [Float]

    init(
        steps: Int,
        width: Int,
        height: Int,
        mu: Double = 0,
        std: Double = 1.75,
        guidance: Float? = nil
    ) throws {
        guard steps > 0 else {
            throw FluxError.invalidRequest("Ideogram steps must be greater than zero")
        }
        let schedule = Self.schedule(steps: steps, width: width, height: height, mu: mu, std: std)
        self.tValues = Array(schedule.dropFirst())
        self.sValues = Array(schedule.dropLast())
        if let preset = Self.presetGuidanceValues(steps: steps, guidance: guidance) {
            self.guidanceValues = preset
        } else if let guidance {
            self.guidanceValues = [Float](repeating: guidance, count: steps)
        } else {
            self.guidanceValues = [Float](repeating: 7, count: steps)
        }
    }

    private static func presetGuidanceValues(steps: Int, guidance: Float?) -> [Float]? {
        guard guidance == nil || guidance == 7 else { return nil }
        switch steps {
        case 20:
            return [Float](repeating: 3, count: 2) + [Float](repeating: 7, count: 18)
        case 48:
            return [Float](repeating: 3, count: 3) + [Float](repeating: 7, count: 45)
        case 12:
            return [Float](repeating: 3, count: 1) + [Float](repeating: 7, count: 11)
        default:
            return nil
        }
    }

    private static func schedule(steps: Int, width: Int, height: Int, mu: Double, std: Double) -> [Float] {
        let pixels = Double(width * height)
        let knownPixels = Double(512 * 512)
        let mean = mu + 0.5 * log(pixels / knownPixels)
        let tMin = 1.0 / (1.0 + exp(0.5 * 18.0))
        let tMax = 1.0 / (1.0 + exp(0.5 * -15.0))
        return (0 ... steps).map { index in
            let p = Double(index) / Double(steps)
            let z: Double
            if p <= 0 {
                z = -Double.infinity
            } else if p >= 1 {
                z = Double.infinity
            } else {
                z = inverseNormalCDF(p)
            }
            let y = mean + std * z
            let shifted = 1.0 - (1.0 / (1.0 + exp(-y)))
            return Float(min(max(shifted, tMin), tMax))
        }
    }

    // Acklam inverse-normal approximation, enough to match mflux NormalDist.inv_cdf
    // for scheduler construction.
    private static func inverseNormalCDF(_ p: Double) -> Double {
        let a: [Double] = [
            -3.969683028665376e+01, 2.209460984245205e+02,
            -2.759285104469687e+02, 1.383577518672690e+02,
            -3.066479806614716e+01, 2.506628277459239e+00,
        ]
        let b: [Double] = [
            -5.447609879822406e+01, 1.615858368580409e+02,
            -1.556989798598866e+02, 6.680131188771972e+01,
            -1.328068155288572e+01,
        ]
        let c: [Double] = [
            -7.784894002430293e-03, -3.223964580411365e-01,
            -2.400758277161838e+00, -2.549732539343734e+00,
            4.374664141464968e+00, 2.938163982698783e+00,
        ]
        let d: [Double] = [
            7.784695709041462e-03, 3.224671290700398e-01,
            2.445134137142996e+00, 3.754408661907416e+00,
        ]
        let pLow = 0.02425
        let pHigh = 1 - pLow
        if p < pLow {
            let q = sqrt(-2 * log(p))
            return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
                / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
        }
        if p > pHigh {
            let q = sqrt(-2 * log(1 - p))
            return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
                / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
        }
        let q = p - 0.5
        let r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    }
}

enum Ideogram4Latents {
    private static let shift: [Float] = [
        0.01984364, 0.10149707, 0.29689495, 0.27188619, -0.21445648, -0.15979549, 0.05021099, -0.15083604,
        -0.15360136, -0.20131799, 0.01922352, 0.0622626, 0.10140969, -0.06739428, 0.3758261, -0.233712,
        0.35164491, -0.02590912, -0.0271935, -0.10833897, -0.1476848, -0.01130957, -0.2298372, 0.23526423,
        -0.10893522, 0.11957631, 0.04047799, 0.3134589, -0.17225064, -0.18646109, -0.34691978, -0.03571246,
        0.02583857, 0.10190072, 0.28402294, 0.26952152, -0.21634675, -0.17938656, 0.04358909, -0.15007621,
        -0.1548502, -0.18971131, 0.02710861, 0.05609494, 0.10697846, -0.06854968, 0.38167698, -0.24269937,
        0.35705471, -0.03063305, -0.02946109, -0.11244286, -0.14336038, -0.01362137, -0.21863696, 0.23228983,
        -0.11739769, 0.11693044, 0.02563311, 0.31356594, -0.17420591, -0.19006285, -0.34905377, -0.04025005,
        0.01924137, 0.07652984, 0.2995608, 0.2628057, -0.22011674, -0.12715361, 0.04879879, -0.14075719,
        -0.15935895, -0.2123584, 0.01974813, 0.05523547, 0.10011992, -0.06428964, 0.37781868, -0.21491644,
        0.34254215, -0.03153528, -0.0310082, -0.10761415, -0.14730405, -0.02475182, -0.2285588, 0.2515081,
        -0.10445128, 0.12446, 0.07062869, 0.30880162, -0.18016875, -0.18869164, -0.34533499, -0.0129177,
        0.02578168, 0.07993659, 0.28642181, 0.26038408, -0.22459419, -0.14820155, 0.04059549, -0.14043529,
        -0.16111187, -0.2020305, 0.02602069, 0.04852717, 0.10432153, -0.06309942, 0.38402443, -0.22397003,
        0.34814481, -0.03774432, -0.03381438, -0.11245691, -0.14128767, -0.02853208, -0.21752016, 0.24872463,
        -0.11399775, 0.1222687, 0.05620835, 0.309178, -0.18065738, -0.19401479, -0.34495114, -0.01760592,
    ]

    private static let scale: [Float] = [
        1.63933691, 1.70204478, 1.73642566, 1.90004803, 1.6675316, 1.69059584, 1.56853198, 1.62314944,
        1.89106626, 1.58086668, 1.60822129, 1.60962993, 1.63322129, 1.56074359, 1.73419528, 1.7919265,
        1.64040632, 1.66802808, 1.60390303, 1.75480492, 1.63187587, 1.64334594, 1.61722884, 1.60146046,
        1.63459219, 1.55291476, 1.68771497, 1.68415657, 1.78966054, 1.66631641, 1.65626686, 1.65976433,
        1.63487607, 1.69513249, 1.72933756, 1.91310663, 1.67035057, 1.72286863, 1.56719251, 1.61934825,
        1.88628859, 1.56911539, 1.59455129, 1.60829869, 1.62470611, 1.56052853, 1.73677003, 1.77563606,
        1.63732541, 1.66370527, 1.59508952, 1.75153949, 1.63029275, 1.64517667, 1.61659342, 1.59722044,
        1.64103121, 1.5408531, 1.68610394, 1.67772755, 1.78998563, 1.66621713, 1.65458955, 1.66041308,
        1.64710857, 1.68163503, 1.74000294, 1.92784786, 1.67411194, 1.67395548, 1.57406532, 1.62199356,
        1.87618195, 1.5584375, 1.57438785, 1.61711053, 1.63094305, 1.55644029, 1.73124302, 1.80666627,
        1.6463621, 1.65932006, 1.60816188, 1.75682671, 1.64695873, 1.63121722, 1.61380832, 1.60478651,
        1.63396035, 1.53505068, 1.65534289, 1.67132281, 1.80317197, 1.6767314, 1.65700938, 1.68426259,
        1.65339716, 1.67540638, 1.73298504, 1.94067348, 1.67893609, 1.70635117, 1.5730906, 1.61928553,
        1.87148809, 1.56244866, 1.56697152, 1.61584394, 1.62759496, 1.55480378, 1.73484107, 1.79055143,
        1.64688773, 1.66121492, 1.60135887, 1.75254572, 1.64798332, 1.62989921, 1.61381592, 1.60792883,
        1.63939668, 1.53075757, 1.65371318, 1.66801185, 1.80029087, 1.67591476, 1.65655173, 1.68533454,
    ]

    static func validateDimensions(width: Int, height: Int) throws {
        for (label, value) in [("width", width), ("height", height)] {
            guard value >= 256, value <= 2048 else {
                throw FluxError.invalidRequest("Ideogram \(label) must be in [256, 2048], got \(value)")
            }
            guard value % 16 == 0 else {
                throw FluxError.invalidRequest("Ideogram \(label) must be a multiple of 16, got \(value)")
            }
        }
    }

    static func createNoise(seed: UInt64?, width: Int, height: Int) -> MLXArray {
        MLXRandom.normal([1, (height / 16) * (width / 16), 128], key: seed.map(MLXRandom.key))
            .asType(.float32)
    }

    static func unpack(_ latents: MLXArray, width: Int, height: Int) throws -> MLXArray {
        let gridHeight = height / 16
        let gridWidth = width / 16
        guard latents.shape == [1, gridHeight * gridWidth, 128] else {
            throw FluxError.invalidRequest("Ideogram latents shape \(latents.shape) does not match [1, \(gridHeight * gridWidth), 128]")
        }
        let shiftArray = MLXArray(shift, [1, 1, 128]).asType(latents.dtype)
        let scaleArray = MLXArray(scale, [1, 1, 128]).asType(latents.dtype)
        var h = latents * scaleArray + shiftArray
        h = h.reshaped([1, gridHeight, gridWidth, 2, 2, 32])
        h = h.transposed(0, 5, 1, 3, 2, 4)
        return h.reshaped([1, 32, gridHeight * 2, gridWidth * 2]).asType(.bfloat16)
    }
}

private final class Ideogram4Tokenizer {
    private let tokenizer: any VMLXTokenizers.Tokenizer
    private let maxLength = 2048

    init(modelPath: URL) async throws {
        tokenizer = try await AutoTokenizer.from(modelFolder: modelPath.appendingPathComponent("tokenizer"), strict: false)
    }

    func ids(_ prompt: String) throws -> [Int32] {
        let trimmed = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            throw FluxError.invalidRequest("Ideogram prompt must not be empty")
        }
        let formatted = "<|im_start|>user\n\(trimmed)<|im_end|>\n<|im_start|>assistant\n"
        let encoded = tokenizer.encode(text: formatted, addSpecialTokens: false)
        guard encoded.count <= maxLength else {
            throw FluxError.invalidRequest("Ideogram prompt has \(encoded.count) tokens, exceeds max_length=\(maxLength)")
        }
        return encoded.map(Int32.init)
    }
}

private final class Ideogram4TextEncoder {
    private let embedTokens: MFluxEmbedding
    private let layers: [Ideogram4TextLayer]
    private let finalNorm: MFluxRMSNorm

    static let hidden = 4096
    static let heads = 32
    static let kvHeads = 8
    static let headDim = 128
    static let intermediate = 12288
    static let theta: Float = 5_000_000
    static let tapLayers = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 35]

    init(store: MFluxStore) throws {
        embedTokens = try store.embedding("text_encoder", "language_model.embed_tokens", dimensions: Self.hidden)
        layers = try (0 ..< 36).map { try Ideogram4TextLayer(store: store, index: $0) }
        finalNorm = try store.rmsNorm("text_encoder", "language_model.norm", eps: 1e-6)
    }

    func promptFeatures(inputs: Ideogram4PromptInputs) -> MLXArray {
        var h = embedTokens(inputs.tokenIDs)
        let seq = h.dim(1)
        let (cos, sin) = Self.ropeCosSin(positionIDs: inputs.textPositionIDs[0..., 0..., 0], dtype: h.dtype)
        let mask = Self.causalPaddingMask(attentionMask: inputs.attentionMask, seq: seq, dtype: h.dtype)
        var captures: [MLXArray] = []
        for (index, layer) in layers.enumerated() {
            h = layer(h, cos: cos, sin: sin, mask: mask)
            if Self.tapLayers.contains(index) {
                captures.append(h)
            }
        }
        _ = finalNorm(h)
        let stackedLayers = stacked(captures, axis: 0).transposed(1, 2, 3, 0)
        let features = stackedLayers.reshaped([1, seq, Self.hidden * Self.tapLayers.count])
        let mask3D = inputs.attentionMask.asType(features.dtype).expandedDimensions(axis: -1)
        return (features * mask3D).asType(.float32)
    }

    private static func causalPaddingMask(attentionMask: MLXArray, seq: Int, dtype: DType) -> MLXArray {
        let negInf = MLXArray(-Float.greatestFiniteMagnitude, dtype: dtype)
        let padding = MLX.where(
            attentionMask .== MLXArray(Int32(1)),
            MLXArray.zeros(attentionMask.shape, dtype: dtype),
            MLXArray.full(attentionMask.shape, values: negInf, dtype: dtype)
        ).reshaped([1, 1, 1, seq])
        let idx = MLXArray(0 ..< Int32(seq))
        let causal = idx.expandedDimensions(axis: 0) .> idx.expandedDimensions(axis: 1)
        let causalMask = MLX.where(
            causal,
            MLXArray.full([seq, seq], values: negInf, dtype: dtype),
            MLXArray.zeros([seq, seq], dtype: dtype)
        ).reshaped([1, 1, seq, seq])
        return causalMask + padding
    }

    private static func ropeCosSin(positionIDs: MLXArray, dtype: DType) -> (MLXArray, MLXArray) {
        let half = headDim / 2
        let invFreq = MLXArray((0 ..< half).map { Float(1) / pow(theta, Float(2 * $0) / Float(headDim)) })
        let freqs = positionIDs.asType(.float32).expandedDimensions(axis: -1) * invFreq.reshaped([1, 1, half])
        let emb = concatenated([freqs, freqs], axis: -1)
        return (cos(emb).asType(dtype), sin(emb).asType(dtype))
    }
}

private final class Ideogram4TextLayer {
    private let inputNorm: MFluxRMSNorm
    private let postNorm: MFluxRMSNorm
    private let qNorm: MFluxRMSNorm
    private let kNorm: MFluxRMSNorm
    private let qProj: MFluxLinear
    private let kProj: MFluxLinear
    private let vProj: MFluxLinear
    private let oProj: MFluxLinear
    private let gate: MFluxLinear
    private let up: MFluxLinear
    private let down: MFluxLinear

    init(store: MFluxStore, index: Int) throws {
        let p = "language_model.layers.\(index)"
        inputNorm = try store.rmsNorm("text_encoder", "\(p).input_layernorm", eps: 1e-6)
        postNorm = try store.rmsNorm("text_encoder", "\(p).post_attention_layernorm", eps: 1e-6)
        qNorm = try store.rmsNorm("text_encoder", "\(p).self_attn.q_norm", eps: 1e-6)
        kNorm = try store.rmsNorm("text_encoder", "\(p).self_attn.k_norm", eps: 1e-6)
        qProj = try store.linear("text_encoder", "\(p).self_attn.q_proj", inputDimensions: 4096, outputDimensions: 4096)
        kProj = try store.linear("text_encoder", "\(p).self_attn.k_proj", inputDimensions: 4096, outputDimensions: 1024)
        vProj = try store.linear("text_encoder", "\(p).self_attn.v_proj", inputDimensions: 4096, outputDimensions: 1024)
        oProj = try store.linear("text_encoder", "\(p).self_attn.o_proj", inputDimensions: 4096, outputDimensions: 4096)
        gate = try store.linear("text_encoder", "\(p).mlp.gate_proj", inputDimensions: 4096, outputDimensions: 12288)
        up = try store.linear("text_encoder", "\(p).mlp.up_proj", inputDimensions: 4096, outputDimensions: 12288)
        down = try store.linear("text_encoder", "\(p).mlp.down_proj", inputDimensions: 12288, outputDimensions: 4096)
    }

    func callAsFunction(_ hidden: MLXArray, cos: MLXArray, sin: MLXArray, mask: MLXArray) -> MLXArray {
        var h = hidden + attention(inputNorm(hidden), cos: cos, sin: sin, mask: mask)
        h = h + down(silu(gate(postNorm(h))) * up(postNorm(h)))
        return h
    }

    private func attention(_ x: MLXArray, cos: MLXArray, sin: MLXArray, mask: MLXArray) -> MLXArray {
        let seq = x.dim(1)
        var q = qProj(x).reshaped([1, seq, 32, 128])
        var k = kProj(x).reshaped([1, seq, 8, 128])
        let v = vProj(x).reshaped([1, seq, 8, 128])
        q = qNorm(q).transposed(0, 2, 1, 3)
        k = kNorm(k).transposed(0, 2, 1, 3)
        let vt = v.transposed(0, 2, 1, 3)
        q = Self.applyRope(q, cos: cos, sin: sin)
        k = Self.applyRope(k, cos: cos, sin: sin)
        let kr = repeatKV(k, n: 4)
        let vr = repeatKV(vt, n: 4)
        let att = MLX.scaledDotProductAttention(
            queries: q.asType(.float32),
            keys: kr.asType(.float32),
            values: vr.asType(.float32),
            scale: Float(1.0 / sqrt(128.0)),
            mask: mask)
        let merged = att.asType(x.dtype).transposed(0, 2, 1, 3).reshaped([1, seq, 4096])
        return oProj(merged)
    }

    private func repeatKV(_ x: MLXArray, n: Int) -> MLXArray {
        if n == 1 { return x }
        let expanded = x.reshaped([x.dim(0), x.dim(1), 1, x.dim(2), x.dim(3)])
        return concatenated(Array(repeating: expanded, count: n), axis: 2)
            .reshaped([x.dim(0), x.dim(1) * n, x.dim(2), x.dim(3)])
    }

    private static func applyRope(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let c = cos.expandedDimensions(axis: 1)
        let s = sin.expandedDimensions(axis: 1)
        return x * c + Ideogram4Rotary.rotateHalf(x) * s
    }
}

private final class Ideogram4ScalarEmbed {
    private let mlpIn: MFluxLinear
    private let mlpOut: MFluxLinear
    private let dim = 4608

    init(store: MFluxStore, component: String) throws {
        mlpIn = try store.linear(component, "t_embedding.mlp_in", inputDimensions: dim, outputDimensions: dim, bias: true)
        mlpOut = try store.linear(component, "t_embedding.mlp_out", inputDimensions: dim, outputDimensions: dim, bias: true)
    }

    func callAsFunction(_ t: Float) -> MLXArray {
        mlpOut(silu(mlpIn(Self.embedding(t, dim: dim))))
    }

    private static func embedding(_ t: Float, dim: Int) -> MLXArray {
        let half = dim / 2
        let scaled = 1e4 * t
        let freq = log(Float(1e4)) / Float(half - 1)
        let frequencies = (0 ..< half).map { exp(Float($0) * -freq) }
        let values = frequencies.map { sin(scaled * $0) } + frequencies.map { cos(scaled * $0) }
        return MLXArray(values, [1, dim])
    }
}

private final class Ideogram4Transformer {
    private let inputProj: MFluxLinear
    private let llmCondNorm: MFluxRMSNorm
    private let llmCondProj: MFluxLinear
    private let timeEmbedding: Ideogram4ScalarEmbed
    private let adalnProj: MFluxLinear
    private let imageIndicatorEmbedding: MLXArray
    private let layers: [Ideogram4TransformerBlock]
    private let finalLayer: Ideogram4FinalLayer

    init(store: MFluxStore, component: String) throws {
        inputProj = try store.linear(component, "input_proj", inputDimensions: 128, outputDimensions: 4608, bias: true)
        llmCondNorm = try store.rmsNorm(component, "llm_cond_norm", eps: 1e-6)
        llmCondProj = try store.linear(component, "llm_cond_proj", inputDimensions: 53248, outputDimensions: 4608, bias: true)
        timeEmbedding = try Ideogram4ScalarEmbed(store: store, component: component)
        adalnProj = try store.linear(component, "adaln_proj", inputDimensions: 4608, outputDimensions: 512, bias: true)
        imageIndicatorEmbedding = try store.tensor(component, "embed_image_indicator.weight")
        layers = try (0 ..< 34).map { try Ideogram4TransformerBlock(store: store, component: component, index: $0) }
        finalLayer = try Ideogram4FinalLayer(store: store, component: component)
    }

    func callAsFunction(
        llmFeatures: MLXArray,
        x: MLXArray,
        timestep: Float,
        positionIDs: MLXArray,
        segmentIDs: MLXArray,
        indicator: MLXArray
    ) -> MLXArray {
        let llmMask = (indicator .== MLXArray(Int32(3))).asType(x.dtype).expandedDimensions(axis: -1)
        let imageMask = (indicator .== MLXArray(Int32(2))).asType(x.dtype).expandedDimensions(axis: -1)
        var h = inputProj(x * imageMask) * imageMask
        let llm = llmCondProj(llmCondNorm(llmFeatures * llmMask)) * llmMask
        h = h + llm
        let indicatorIDs = (indicator .== MLXArray(Int32(2))).asType(.int32)
        h = h + imageIndicatorEmbedding[indicatorIDs]

        var adalnInput = timeEmbedding(timestep)
        adalnInput = silu(adalnProj(adalnInput.expandedDimensions(axis: 1)))
        let (cos, sin) = Ideogram4MRoPE.cosSin(positionIDs: positionIDs, dtype: h.dtype)
        for layer in layers {
            h = layer(h, segmentIDs: segmentIDs, cos: cos, sin: sin, adalnInput: adalnInput)
        }
        return finalLayer(h, c: adalnInput).asType(.float32)
    }
}

private enum Ideogram4MRoPE {
    static func cosSin(positionIDs: MLXArray, dtype: DType) -> (MLXArray, MLXArray) {
        let headDim = 256
        let half = headDim / 2
        let invFreq = MLXArray((0 ..< half).map { Float(1) / pow(Float(5_000_000), Float(2 * $0) / Float(headDim)) })
        let pos = positionIDs.asType(.float32)
        let freqs = (0 ..< 3).map { axis in
            pos[0..., 0..., axis].expandedDimensions(axis: -1) * invFreq.reshaped([1, 1, half])
        }
        var selector = [Int32](repeating: 0, count: half)
        for index in stride(from: 1, to: 20 * 3, by: 3) { selector[index] = 1 }
        for index in stride(from: 2, to: 20 * 3, by: 3) { selector[index] = 2 }
        var selected: [MLXArray] = []
        selected.reserveCapacity(half)
        for (index, axis) in selector.enumerated() {
            selected.append(freqs[Int(axis)][0..., 0..., index].expandedDimensions(axis: -1))
        }
        let freq = concatenated(selected, axis: -1)
        let emb = concatenated([freq, freq], axis: -1)
        return (cos(emb).asType(dtype), sin(emb).asType(dtype))
    }

    static func apply(_ q: MLXArray, _ k: MLXArray, cos: MLXArray, sin: MLXArray) -> (MLXArray, MLXArray) {
        let c = cos.expandedDimensions(axis: 1)
        let s = sin.expandedDimensions(axis: 1)
        return (q * c + Ideogram4Rotary.rotateHalf(q) * s, k * c + Ideogram4Rotary.rotateHalf(k) * s)
    }
}

private final class Ideogram4TransformerBlock {
    private let attention: Ideogram4Attention
    private let feedForward: Ideogram4MLP
    private let attentionNorm1: MFluxRMSNorm
    private let ffnNorm1: MFluxRMSNorm
    private let attentionNorm2: MFluxRMSNorm
    private let ffnNorm2: MFluxRMSNorm
    private let adalnModulation: MFluxLinear

    init(store: MFluxStore, component: String, index: Int) throws {
        let p = "layers.\(index)"
        attention = try Ideogram4Attention(store: store, component: component, prefix: "\(p).attention")
        feedForward = try Ideogram4MLP(store: store, component: component, prefix: "\(p).feed_forward")
        attentionNorm1 = try store.rmsNorm(component, "\(p).attention_norm1", eps: 1e-5)
        ffnNorm1 = try store.rmsNorm(component, "\(p).ffn_norm1", eps: 1e-5)
        attentionNorm2 = try store.rmsNorm(component, "\(p).attention_norm2", eps: 1e-5)
        ffnNorm2 = try store.rmsNorm(component, "\(p).ffn_norm2", eps: 1e-5)
        adalnModulation = try store.linear(component, "\(p).adaln_modulation", inputDimensions: 512, outputDimensions: 18432, bias: true)
    }

    func callAsFunction(
        _ x0: MLXArray,
        segmentIDs: MLXArray,
        cos: MLXArray,
        sin: MLXArray,
        adalnInput: MLXArray
    ) -> MLXArray {
        let mod = adalnModulation(adalnInput)
        let scaleMSA = MLXArray(Float(1)) + mod[0..., 0..., 0 ..< 4608]
        let gateMSA = tanh(mod[0..., 0..., 4608 ..< 9216])
        let scaleMLP = MLXArray(Float(1)) + mod[0..., 0..., 9216 ..< 13824]
        let gateMLP = tanh(mod[0..., 0..., 13824 ..< 18432])
        let attnOut = attention(attentionNorm1(x0) * scaleMSA, segmentIDs: segmentIDs, cos: cos, sin: sin)
        var x = x0 + gateMSA * attentionNorm2(attnOut)
        let ffnOut = feedForward(ffnNorm1(x) * scaleMLP)
        x = x + gateMLP * ffnNorm2(ffnOut)
        return x
    }
}

private final class Ideogram4Attention {
    private let qkv: MFluxLinear
    private let normQ: MFluxRMSNorm
    private let normK: MFluxRMSNorm
    private let out: MFluxLinear
    private let heads = 18
    private let headDim = 256

    init(store: MFluxStore, component: String, prefix: String) throws {
        qkv = try store.linear(component, "\(prefix).qkv", inputDimensions: 4608, outputDimensions: 13824)
        normQ = try store.rmsNorm(component, "\(prefix).norm_q", eps: 1e-5)
        normK = try store.rmsNorm(component, "\(prefix).norm_k", eps: 1e-5)
        out = try store.linear(component, "\(prefix).o", inputDimensions: 4608, outputDimensions: 4608)
    }

    func callAsFunction(_ x: MLXArray, segmentIDs: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let seq = x.dim(1)
        let projected = qkv(x).reshaped([1, seq, 3, heads, headDim])
        var q = normQ(projected[0..., 0..., 0, 0..., 0...]).transposed(0, 2, 1, 3)
        var k = normK(projected[0..., 0..., 1, 0..., 0...]).transposed(0, 2, 1, 3)
        let v = projected[0..., 0..., 2, 0..., 0...].transposed(0, 2, 1, 3)
        (q, k) = Ideogram4MRoPE.apply(q, k, cos: cos, sin: sin)
        let sameSegment = segmentIDs.expandedDimensions(axis: 2) .== segmentIDs.expandedDimensions(axis: 1)
        let mask = MLX.where(
            sameSegment.expandedDimensions(axis: 1),
            MLXArray.zeros([1, 1, seq, seq], dtype: .float32),
            MLXArray.full(
                [1, 1, seq, seq],
                values: MLXArray(-Float.greatestFiniteMagnitude, dtype: .float32),
                dtype: .float32))
        let attended = MLX.scaledDotProductAttention(
            queries: q.asType(.float32),
            keys: k.asType(.float32),
            values: v.asType(.float32),
            scale: Float(1.0 / sqrt(256.0)),
            mask: mask)
        let merged = attended.asType(x.dtype).transposed(0, 2, 1, 3).reshaped([1, seq, 4608])
        return out(merged)
    }
}

private final class Ideogram4MLP {
    private let w1: MFluxLinear
    private let w2: MFluxLinear
    private let w3: MFluxLinear

    init(store: MFluxStore, component: String, prefix: String) throws {
        w1 = try store.linear(component, "\(prefix).w1", inputDimensions: 4608, outputDimensions: 12288)
        w2 = try store.linear(component, "\(prefix).w2", inputDimensions: 12288, outputDimensions: 4608)
        w3 = try store.linear(component, "\(prefix).w3", inputDimensions: 4608, outputDimensions: 12288)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        w2(silu(w1(x)) * w3(x))
    }
}

private final class Ideogram4FinalLayer {
    private let linear: MFluxLinear
    private let adalnModulation: MFluxLinear

    init(store: MFluxStore, component: String) throws {
        linear = try store.linear(component, "final_layer.linear", inputDimensions: 4608, outputDimensions: 128, bias: true)
        adalnModulation = try store.linear(component, "final_layer.adaln_modulation", inputDimensions: 512, outputDimensions: 4608, bias: true)
    }

    func callAsFunction(_ x: MLXArray, c: MLXArray) -> MLXArray {
        let scale = MLXArray(Float(1)) + adalnModulation(silu(c))
        return linear(ideogramLayerNormNoAffine(x) * scale)
    }
}

private func ideogramLayerNormNoAffine(_ x: MLXArray, eps: Float = 1e-6) -> MLXArray {
    let xf = x.asType(.float32)
    let meanValue = mean(xf, axis: -1, keepDims: true)
    let centered = xf - meanValue
    let variance = mean(centered * centered, axis: -1, keepDims: true)
    return (centered * rsqrt(variance + MLXArray(eps))).asType(x.dtype)
}

private final class Ideogram4VAE {
    private let postQuant: MFluxConv2D
    private let convIn: MFluxConv2D
    private let midRes0: Ideogram4VAEResnet
    private let midAttention: Ideogram4VAEAttention
    private let midRes1: Ideogram4VAEResnet
    private let upBlocks: [(resnets: [Ideogram4VAEResnet], upsampler: Ideogram4VAEUpsampler?)]
    private let normOut: MFluxGroupNorm
    private let convOut: MFluxConv2D

    init(store: MFluxStore) throws {
        postQuant = try store.conv2d("vae", "post_quant_conv")
        convIn = try store.conv2d("vae", "decoder.conv_in", padding: 1)
        midRes0 = try Ideogram4VAEResnet(store: store, prefix: "decoder.mid_block.resnets.0")
        midAttention = try Ideogram4VAEAttention(store: store, prefix: "decoder.mid_block.attentions.0")
        midRes1 = try Ideogram4VAEResnet(store: store, prefix: "decoder.mid_block.resnets.1")
        var blocks: [(resnets: [Ideogram4VAEResnet], upsampler: Ideogram4VAEUpsampler?)] = []
        for index in 0 ..< 4 {
            let resnets = try (0 ..< 3).map {
                try Ideogram4VAEResnet(store: store, prefix: "decoder.up_blocks.\(index).resnets.\($0)")
            }
            let upsampler = index < 3
                ? try Ideogram4VAEUpsampler(store: store, prefix: "decoder.up_blocks.\(index).upsamplers.0")
                : nil
            blocks.append((resnets, upsampler))
        }
        upBlocks = blocks
        normOut = try store.groupNorm("vae", "decoder.conv_norm_out")
        convOut = try store.conv2d("vae", "decoder.conv_out", padding: 1)
    }

    func decode(_ latents: MLXArray) -> MLXArray {
        var h = postQuant(latents.transposed(0, 2, 3, 1)).transposed(0, 3, 1, 2)
        h = convIn(h.transposed(0, 2, 3, 1)).transposed(0, 3, 1, 2)
        h = midRes1(midAttention(midRes0(h)))
        for block in upBlocks {
            for resnet in block.resnets {
                h = resnet(h)
            }
            if let upsampler = block.upsampler {
                h = upsampler(h)
            }
        }
        h = normOut(h.transposed(0, 2, 3, 1)).transposed(0, 3, 1, 2)
        h = silu(h)
        h = convOut(h.transposed(0, 2, 3, 1)).transposed(0, 3, 1, 2)
        return VAEDecoder.postprocess(h)
    }
}

private final class Ideogram4VAEResnet {
    private let norm1: MFluxGroupNorm
    private let conv1: MFluxConv2D
    private let norm2: MFluxGroupNorm
    private let conv2: MFluxConv2D
    private let shortcut: MFluxConv2D?

    init(store: MFluxStore, prefix: String) throws {
        norm1 = try store.groupNorm("vae", "\(prefix).norm1")
        conv1 = try store.conv2d("vae", "\(prefix).conv1", padding: 1)
        norm2 = try store.groupNorm("vae", "\(prefix).norm2")
        conv2 = try store.conv2d("vae", "\(prefix).conv2", padding: 1)
        shortcut = store.hasKey("vae", "\(prefix).conv_shortcut.weight")
            ? try store.conv2d("vae", "\(prefix).conv_shortcut")
            : nil
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let nhwc = input.transposed(0, 2, 3, 1)
        var h = conv1(silu(norm1(nhwc)))
        h = conv2(silu(norm2(h)))
        let residual = shortcut?(nhwc) ?? nhwc
        return (residual + h).transposed(0, 3, 1, 2)
    }
}

private final class Ideogram4VAEAttention {
    private let groupNorm: MFluxGroupNorm
    private let toQ: MFluxLinear
    private let toK: MFluxLinear
    private let toV: MFluxLinear
    private let toOut: MFluxLinear
    private let channels = 512

    init(store: MFluxStore, prefix: String) throws {
        groupNorm = try store.groupNorm("vae", "\(prefix).group_norm")
        toQ = try store.linear("vae", "\(prefix).to_q", inputDimensions: channels, outputDimensions: channels, bias: true)
        toK = try store.linear("vae", "\(prefix).to_k", inputDimensions: channels, outputDimensions: channels, bias: true)
        toV = try store.linear("vae", "\(prefix).to_v", inputDimensions: channels, outputDimensions: channels, bias: true)
        toOut = try store.linear("vae", "\(prefix).to_out.0", inputDimensions: channels, outputDimensions: channels, bias: true)
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let nhwc = input.transposed(0, 2, 3, 1)
        let batch = nhwc.dim(0)
        let height = nhwc.dim(1)
        let width = nhwc.dim(2)
        let normed = groupNorm(nhwc.asType(.float32)).asType(input.dtype)
        let q = toQ(normed).reshaped([batch, height * width, 1, channels]).transposed(0, 2, 1, 3)
        let k = toK(normed).reshaped([batch, height * width, 1, channels]).transposed(0, 2, 1, 3)
        let v = toV(normed).reshaped([batch, height * width, 1, channels]).transposed(0, 2, 1, 3)
        let attended = MLX.scaledDotProductAttention(
            queries: q.asType(.float32),
            keys: k.asType(.float32),
            values: v.asType(.float32),
            scale: Float(1.0 / sqrt(512.0)),
            mask: nil)
        let out = attended.asType(input.dtype).transposed(0, 2, 1, 3).reshaped([batch, height, width, channels])
        return (nhwc + toOut(out)).transposed(0, 3, 1, 2)
    }
}

private final class Ideogram4VAEUpsampler {
    private let conv: MFluxConv2D

    init(store: MFluxStore, prefix: String) throws {
        conv = try store.conv2d("vae", "\(prefix).conv", padding: 1)
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let up = repeated(repeated(input, count: 2, axis: 2), count: 2, axis: 3)
        return conv(up.transposed(0, 2, 3, 1)).transposed(0, 3, 1, 2)
    }
}

final class Ideogram4Pipeline {
    private let textEncoder: Ideogram4TextEncoder
    private let conditionalTransformer: Ideogram4Transformer
    private let unconditionalTransformer: Ideogram4Transformer
    private let vae: Ideogram4VAE
    private let tokenizer: Ideogram4Tokenizer

    init(modelPath: URL, loadedWeights: LoadedWeights) async throws {
        let store = MFluxStore(loadedWeights)
        tokenizer = try await Ideogram4Tokenizer(modelPath: modelPath)
        textEncoder = try Ideogram4TextEncoder(store: store)
        conditionalTransformer = try Ideogram4Transformer(store: store, component: "transformer")
        unconditionalTransformer = try Ideogram4Transformer(store: store, component: "unconditional_transformer")
        vae = try Ideogram4VAE(store: store)
    }

    func generate(
        prompt: String,
        width: Int,
        height: Int,
        steps: Int,
        guidance: Float,
        seed: UInt64?,
        progress: (Int, Int, Double?) -> Void
    ) throws -> MLXArray {
        try Ideogram4Latents.validateDimensions(width: width, height: height)
        let tokenIDs = try tokenizer.ids(prompt)
        let inputs = try Ideogram4PromptInputs(tokenIDs: tokenIDs, width: width, height: height)
        let llmFeatures = textEncoder.promptFeatures(inputs: inputs)
        let negative = inputs.negative(llmFeaturesDim: llmFeatures.dim(2))
        let textPadding = MLXArray.zeros([1, inputs.maxTextTokens, 128], dtype: .float32)
        var latents = Ideogram4Latents.createNoise(seed: seed, width: width, height: height)
        let scheduler = try Ideogram4Scheduler(
            steps: steps,
            width: width,
            height: height,
            guidance: guidance)

        let start = Date()
        for step in 0 ..< steps {
            let scheduleIndex = steps - 1 - step
            let t = scheduler.tValues[scheduleIndex]
            let s = scheduler.sValues[scheduleIndex]
            let g = scheduler.guidanceValues[scheduleIndex]
            let positiveInput = concatenated([textPadding, latents], axis: 1)
            let positiveVelocity = conditionalTransformer(
                llmFeatures: llmFeatures,
                x: positiveInput,
                timestep: t,
                positionIDs: inputs.positionIDs,
                segmentIDs: inputs.segmentIDs,
                indicator: inputs.indicator)[0..., inputs.maxTextTokens..., 0...]
            let negativeVelocity = unconditionalTransformer(
                llmFeatures: negative.llmFeatures,
                x: latents,
                timestep: t,
                positionIDs: negative.positionIDs,
                segmentIDs: negative.segmentIDs,
                indicator: negative.indicator)
            let velocity = MLXArray(g) * positiveVelocity + (MLXArray(Float(1)) - MLXArray(g)) * negativeVelocity
            latents = latents + velocity * MLXArray(s - t)
            eval(latents)
            let elapsed = Date().timeIntervalSince(start)
            progress(step + 1, steps, elapsed / Double(step + 1) * Double(steps - step - 1))
        }
        let unpacked = try Ideogram4Latents.unpack(latents, width: width, height: height)
        return vae.decode(unpacked)
    }
}
