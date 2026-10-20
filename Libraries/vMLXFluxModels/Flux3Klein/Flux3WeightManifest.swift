import Foundation
import vMLXFluxKit

/// Safetensors metadata preflight, before MLX opens any tensor buffers. Validates
/// every native graph binding against the exact saved mflux affine/NHWC layout.
enum Flux3WeightManifest {
    struct Tensor: Equatable {
        let dtype: String
        let shape: [Int]
    }
    struct Audit: Equatable {
        let tensors: [String: Int]
        let affineLayers: [String: Int]
    }

    static func readShard(_ url: URL, bits: Int) throws -> [String: Tensor] {
        func invalid(_ reason: String) -> FluxError { .invalidRequest("\(url.lastPathComponent): \(reason)") }
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        let fileSize = try handle.seekToEnd()
        try handle.seek(toOffset: 0)
        guard let prefix = try handle.read(upToCount: 8), prefix.count == 8 else { throw invalid("truncated safetensors prefix") }
        let headerSize = prefix.enumerated().reduce(UInt64(0)) { $0 | UInt64($1.element) << ($1.offset * 8) }
        // Safetensors format bound, also enforced by mlx/io/safetensors.cpp.
        // This is not an image-size or memory-admission heuristic.
        let maximumHeaderLength: UInt64 = 100_000_000
        guard fileSize >= 8, headerSize <= maximumHeaderLength, headerSize <= fileSize - 8, headerSize <= UInt64(Int.max),
              let header = try handle.read(upToCount: Int(headerSize)), header.count == Int(headerSize),
              let object = try JSONSerialization.jsonObject(with: header) as? [String: Any] else {
            throw invalid("invalid safetensors header")
        }
        guard let metadata = object["__metadata__"] as? [String: String],
              metadata["quantization_level"] == String(bits),
              metadata["mflux_version"]?.isEmpty == false else {
            throw invalid("missing or inconsistent mflux quantization metadata")
        }
        let bufferSize = fileSize - 8 - headerSize
        var tensors: [String: Tensor] = [:]
        var ranges: [(UInt64, UInt64)] = []
        for (key, value) in object where key != "__metadata__" {
            guard let raw = value as? [String: Any], let dtype = raw["dtype"] as? String,
                  let shape = raw["shape"] as? [Int], let offsets = raw["data_offsets"] as? [Int],
                  offsets.count == 2, offsets[0] >= 0, offsets[1] >= offsets[0],
                  UInt64(offsets[1]) <= bufferSize else { throw invalid("invalid tensor descriptor \(key)") }
            let width: Int
            switch dtype {
            case "BF16": width = 2
            case "F32", "U32": width = 4
            default: throw invalid("unsupported \(key) dtype \(dtype)")
            }
            var bytes = width
            for dimension in shape {
                let product = bytes.multipliedReportingOverflow(by: dimension)
                guard dimension >= 0, !product.overflow else { throw invalid("invalid tensor shape \(key)") }
                bytes = product.partialValue
            }
            guard offsets[1] - offsets[0] == bytes else { throw invalid("tensor byte count mismatch \(key)") }
            tensors[key] = Tensor(dtype: dtype, shape: shape)
            ranges.append((UInt64(offsets[0]), UInt64(offsets[1])))
        }
        var end: UInt64 = 0
        for range in ranges.sorted(by: { $0.0 < $1.0 }) {
            guard range.0 == end else { throw invalid("overlapping or unindexed tensor data") }
            end = range.1
        }
        guard end == bufferSize else { throw invalid("truncated or unindexed tensor buffer") }
        return tensors
    }

    static func validate(modelPath: URL, bundle: Flux3Bundle) throws -> Audit {
        struct Index: Decodable { let weightMap: [String: String] }
        let decoder = JSONDecoder(); decoder.keyDecodingStrategy = .convertFromSnakeCase
        let expected = expectedTensors(bits: bundle.stamp.quantization.bits)
        var counts: [String: Int] = [:]
        var affineCounts: [String: Int] = [:]
        var reasons: [String] = []
        for component in ["transformer", "text_encoder", "vae"] {
            let root = modelPath.appendingPathComponent(component)
            let index = try decoder.decode(Index.self, from: Data(contentsOf: root.appendingPathComponent("model.safetensors.index.json")))
            var actual: [String: Tensor] = [:]
            for shard in Set(index.weightMap.values).sorted() {
                guard !shard.contains("/"), !shard.contains("\\") else {
                    throw FluxError.invalidRequest("non-local safetensors shard path")
                }
                let tensors = try readShard(root.appendingPathComponent(shard), bits: bundle.stamp.quantization.bits)
                for (key, tensor) in tensors {
                    guard index.weightMap[key] == shard, actual[key] == nil else {
                        throw FluxError.invalidRequest("\(component) header/index mismatch for \(key)")
                    }
                    actual[key] = tensor
                }
            }
            if Set(index.weightMap.keys) != Set(actual.keys) { reasons.append("\(component) indexed tensors missing from shards") }
            let wanted = expected[component, default: [:]]
            for key in wanted.keys.sorted() where actual[key] != wanted[key] {
                reasons.append("\(component).\(key) expected \(String(describing: wanted[key])), got \(String(describing: actual[key]))")
            }
            let extra = Set(actual.keys).subtracting(wanted.keys)
            if !extra.isEmpty { reasons.append("unsupported \(component) tensor keys: \(extra.sorted().joined(separator: ", "))") }
            counts[component] = actual.count
            affineCounts[component] = actual.keys.filter { $0.hasSuffix(".scales") }.count
        }
        if !reasons.isEmpty { throw FluxError.localModelIncomplete(modelPath, reasons: reasons) }
        return Audit(tensors: counts, affineLayers: affineCounts)
    }

    /// The graph currently supports the stamped 3072-wide transformer, Qwen3-VL
    /// 4B language tower, and 32-channel Flux2 VAE. Shape changes need a port.
    static func expectedTensors(bits: Int) -> [String: [String: Tensor]] {
        var all: [String: [String: Tensor]] = [:]
        func tensor(_ component: String, _ name: String, _ shape: [Int], dtype: String = "BF16") {
            all[component, default: [:]][name] = Tensor(dtype: dtype, shape: shape)
        }
        func affine(_ component: String, _ name: String, _ input: Int, _ output: Int, bias: Bool = false) {
            tensor(component, "\(name).weight", [output, input * bits / 32], dtype: "U32")
            tensor(component, "\(name).scales", [output, input / 64])
            tensor(component, "\(name).biases", [output, input / 64])
            if bias { tensor(component, "\(name).bias", [output]) }
        }
        for prefix in ["timestep_embedder", "target_timestep_embedder"] {
            affine("transformer", "\(prefix).linear_1", 256, 3072)
            affine("transformer", "\(prefix).linear_2", 3072, 3072)
        }
        for mode in ["image", "image_cond", "txt"] {
            for stage in ["early", "single"] { affine("transformer", "\(stage)_stream_modulations.\(mode).linear", 3072, 9216) }
        }
        for mode in ["image", "image_cond"] { affine("transformer", "x_embedders.\(mode)", 128, 3072) }
        affine("transformer", "context_embedder", 20480, 3072)
        affine("transformer", "norm_out.linear", 3072, 6144)
        affine("transformer", "proj_out", 3072, 128)
        let blocks = ["mode_blocks.image", "mode_blocks.image_cond", "txt_mode_blocks"].flatMap { prefix in
            (0..<5).map { "\(prefix).\($0)" }
        } + (0..<28).map { "single_transformer_blocks.\($0)" }
        for prefix in blocks {
            for projection in ["to_q", "to_k", "to_v", "to_out"] { affine("transformer", "\(prefix).attn.\(projection)", 3072, 3072) }
            for norm in ["norm_q", "norm_k"] { tensor("transformer", "\(prefix).attn.\(norm).weight", [128]) }
            affine("transformer", "\(prefix).ff.linear_in", 3072, 18432)
            affine("transformer", "\(prefix).ff.linear_out", 9216, 3072)
        }
        affine("text_encoder", "embed_tokens", 2560, 151936)
        for i in 0..<32 {
            let p = "layers.\(i)"
            for norm in ["input_layernorm", "post_attention_layernorm"] { tensor("text_encoder", "\(p).\(norm).weight", [2560]) }
            for norm in ["q_norm", "k_norm"] { tensor("text_encoder", "\(p).self_attn.\(norm).weight", [128]) }
            affine("text_encoder", "\(p).self_attn.q_proj", 2560, 4096)
            for projection in ["k_proj", "v_proj"] { affine("text_encoder", "\(p).self_attn.\(projection)", 2560, 1024) }
            affine("text_encoder", "\(p).self_attn.o_proj", 4096, 2560)
            for projection in ["gate_proj", "up_proj"] { affine("text_encoder", "\(p).mlp.\(projection)", 2560, 9728) }
            affine("text_encoder", "\(p).mlp.down_proj", 9728, 2560)
        }
        tensor("text_encoder", "norm.weight", [2560])
        tensor("text_encoder", "rotary_emb.inv_freq", [64], dtype: "F32")
        func vaeNorm(_ name: String, _ channels: Int) {
            tensor("vae", "\(name).weight", [channels]); tensor("vae", "\(name).bias", [channels])
        }
        func conv(_ name: String, _ input: Int, _ output: Int, kernel: Int = 3) {
            tensor("vae", "\(name).weight", [output, kernel, kernel, input]); tensor("vae", "\(name).bias", [output])
        }
        func resnet(_ name: String, _ input: Int, _ output: Int) {
            vaeNorm("\(name).norm1", input); vaeNorm("\(name).norm2", output)
            conv("\(name).conv1", input, output); conv("\(name).conv2", output, output)
            if input != output { conv("\(name).conv_shortcut", input, output, kernel: 1) }
        }
        conv("encoder.conv_in", 3, 128); conv("encoder.conv_out", 512, 64)
        conv("decoder.conv_in", 32, 512); conv("decoder.conv_out", 128, 3)
        conv("quant_conv", 64, 64, kernel: 1); conv("post_quant_conv", 32, 32, kernel: 1)
        vaeNorm("encoder.conv_norm_out", 512); vaeNorm("decoder.conv_norm_out", 128)
        for mode in ["encoder", "decoder"] {
            for i in 0..<2 { resnet("\(mode).mid_block.resnets.\(i)", 512, 512) }
            let p = "\(mode).mid_block.attentions.0"
            vaeNorm("\(p).group_norm", 512)
            for projection in ["to_q", "to_k", "to_v", "to_out"] { affine("vae", "\(p).\(projection)", 512, 512, bias: true) }
        }
        var input = 128
        for (i, output) in [128, 256, 512, 512].enumerated() {
            for j in 0..<2 { resnet("encoder.down_blocks.\(i).resnets.\(j)", j == 0 ? input : output, output) }
            if i < 3 { conv("encoder.down_blocks.\(i).downsamplers.0.conv", output, output) }
            input = output
        }
        input = 512
        for (i, output) in [512, 512, 256, 128].enumerated() {
            for j in 0..<3 { resnet("decoder.up_blocks.\(i).resnets.\(j)", j == 0 ? input : output, output) }
            if i < 3 { conv("decoder.up_blocks.\(i).upsamplers.0.conv", output, output) }
            input = output
        }
        tensor("vae", "bn.running_mean", [128]); tensor("vae", "bn.running_var", [128])
        return all
    }
}
