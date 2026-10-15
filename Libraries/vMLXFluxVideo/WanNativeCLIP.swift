// Architecture reference: ml-explore/mlx-examples/video/wan2.1/clip.py.
import Foundation
@preconcurrency import MLX
import MLXNN
import vMLXFluxKit

final class WanNativeCLIP {
    let weights: WanNativeWeights
    init(arrays: [String: MLXArray]) throws {
        var mapped: [String: MLXArray] = [:]
        for (key, value) in arrays where key.hasPrefix("visual.") {
            var name = String(key.dropFirst(7))
            if name == "conv1.weight" { name = "patch_embedding.weight" }
            if name == "class_embedding" { name = "cls_embedding" }
            if name == "positional_embedding" || name == "pos_embedding" { name = "position_embedding" }
            name = name.replacingOccurrences(of: "ln_pre.", with: "pre_norm.")
                .replacingOccurrences(of: "transformer.resblocks.", with: "transformer.")
                .replacingOccurrences(of: ".ln_1.", with: ".norm1.")
                .replacingOccurrences(of: ".ln_2.", with: ".norm2.")
                .replacingOccurrences(of: ".attn.in_proj_weight", with: ".attn.to_qkv.weight")
                .replacingOccurrences(of: ".attn.in_proj_bias", with: ".attn.to_qkv.bias")
                .replacingOccurrences(of: ".attn.out_proj.", with: ".attn.proj.")
                .replacingOccurrences(of: ".mlp.c_fc.", with: ".mlp.0.")
                .replacingOccurrences(of: ".mlp.c_proj.", with: ".mlp.2.")
            mapped[name] = value
        }
        weights = WanNativeWeights(mapped)
        _ = try weights.tensor("patch_embedding.weight")
        _ = try weights.tensor("cls_embedding")
    }

    func encode(_ image: URL, interrupt: FluxInterrupt?) throws -> MLXArray {
        let pixels = try ImageIO.readRGBValues(image, width: 224, height: 224, normalization: .zeroToOne)
        var x = MLXArray(pixels, [1, 3, 224, 224]).transposed(0, 2, 3, 1)
        x = (x - MLXArray([0.48145466, 0.4578275, 0.40821073] as [Float]))
            / MLXArray([0.26862954, 0.26130258, 0.27577711] as [Float])
        x = try weights.conv2("patch_embedding", x, stride: 14).reshaped([1, 256, 1280])
        x = try concatenated([weights.tensor("cls_embedding").reshaped([1, 1, 1280]), x], axis: 1)
            + weights.tensor("position_embedding").reshaped([1, 257, 1280])
        x = try weights.norm("pre_norm", x, eps: 1e-5)
        // Wan consumes the penultimate visual block, not a pooled CLIP output.
        for i in 0..<31 {
            try interrupt?.check()
            let p = "transformer.\(i)"
            let qkv = try weights.linear(p + ".attn.to_qkv", weights.norm(p + ".norm1", x, eps: 1e-5))
                .reshaped([1, 257, 3, 16, 80])
            let q = qkv[0..., 0..., 0, 0..., 0...].transposed(0, 2, 1, 3)
            let k = qkv[0..., 0..., 1, 0..., 0...].transposed(0, 2, 1, 3)
            let v = qkv[0..., 0..., 2, 0..., 0...].transposed(0, 2, 1, 3)
            let y = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v,
                scale: 1 / sqrt(Float(80)), mask: nil).transposed(0, 2, 1, 3).reshaped([1, 257, 1280])
            x = x + (try weights.linear(p + ".attn.proj", y))
            x = x + (try weights.linear(p + ".mlp.2",
                gelu(weights.linear(p + ".mlp.0", weights.norm(p + ".norm2", x, eps: 1e-5)))))
            eval(x)
        }
        return x
    }
}
