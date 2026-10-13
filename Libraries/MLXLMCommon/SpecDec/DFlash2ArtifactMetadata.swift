import Foundation

/// Header-only validation of the exact parameter topology consumed by DFlash2Loader.
/// This never constructs a module or reads tensor payloads.
public enum DFlash2ArtifactMetadata {
    public static func requiredShapes(configData: Data) throws -> [String: [Int]] {
        guard let root = try JSONSerialization.jsonObject(with: configData) as? [String: Any],
            let heads = root["num_attention_heads"] as? Int, heads > 0,
            let layers = root["num_hidden_layers"] as? Int, layers > 0, layers <= 1024
        else { throw DFlash2LoadError.targetMismatch("invalid attention dimensions") }
        let c = try DFlash2Configuration(json: root)
        guard c.isRunnable, c.hiddenSize > 0, c.numHiddenLayers > 0,
            c.headDim > 0, c.numKeyValueHeads > 0, c.intermediateSize > 0,
            c.vocabSize > 0,
            !c.isDFlash2
                || (c.dflash.convGroupSize > 0 && c.hiddenSize % c.dflash.convGroupSize == 0),
            !c.targetLayerIds.isEmpty, c.targetLayerIds.allSatisfy({ $0 >= 0 })
        else { throw DFlash2LoadError.targetMismatch("invalid drafter dimensions") }
        func product(_ dimensions: Int...) throws -> Int {
            var value = 1
            for dimension in dimensions {
                let result = value.multipliedReportingOverflow(by: dimension)
                guard dimension > 0, !result.overflow else {
                    throw DFlash2LoadError.targetMismatch("drafter dimension overflow")
                }
                value = result.partialValue
            }
            return value
        }
        let h = c.hiddenSize
        let d = c.headDim
        let rank = c.dflash.selectorRank
        var shapes: [String: [Int]] = [
            "norm.weight": [h], "hidden_norm.weight": [h],
            "fc.weight": [h, try product(c.targetLayerIds.count, h)],
        ]
        if c.isDFlash2 {
            shapes["candidate_selector.predecessor_codebook.weight"] = [c.vocabSize, rank]
            shapes["candidate_selector.successor_codebook.weight"] = [c.vocabSize, rank]
            shapes["candidate_selector.hidden_projection.weight"] = [rank, h]
        }
        if c.isDSpark {
            shapes["markov_head.markov_w1.weight"] = [c.vocabSize, c.markovRank]
            shapes["markov_head.markov_w2.weight"] = [c.vocabSize, c.markovRank]
            if c.hasConfidenceHead {
                let width = h.addingReportingOverflow(c.markovRank)
                guard !width.overflow else {
                    throw DFlash2LoadError.targetMismatch("drafter confidence dimension overflow")
                }
                shapes["confidence_head.proj.weight"] = [1, width.partialValue]
                shapes["confidence_head.proj.bias"] = [1]
            }
        }
        for i in 0 ..< c.numHiddenLayers {
            let p = "layers.\(i)."
            for (name, shape) in [
                "self_attn.q_proj.weight": [try product(c.numAttentionHeads, d), h],
                "self_attn.k_proj.weight": [try product(c.numKeyValueHeads, d), h],
                "self_attn.v_proj.weight": [try product(c.numKeyValueHeads, d), h],
                "self_attn.o_proj.weight": [h, try product(c.numAttentionHeads, d)],
                "self_attn.q_norm.weight": [d], "self_attn.k_norm.weight": [d],
                "input_layernorm.weight": [h], "post_attention_layernorm.weight": [h],
                "mlp.gate_proj.weight": [c.intermediateSize, h],
                "mlp.up_proj.weight": [c.intermediateSize, h],
                "mlp.down_proj.weight": [h, c.intermediateSize],
            ] { shapes[p + name] = shape }
            for name in ["attention_conv", "mlp_conv"] where c.isDFlash2 {
                shapes[p + name + ".base_kernel"] = [2, c.dflash.convKernelSize, h]
                shapes[p + name + ".kernel_projection.weight"] = [
                    try product(2, c.dflash.convKernelSize, h / c.dflash.convGroupSize), h,
                ]
            }
        }
        // Validate storage dimensions too, before any later packed-width arithmetic.
        for shape in shapes.values {
            var elements = 4
            for dimension in shape { elements = try product(elements, dimension) }
        }
        return shapes
    }

    /// Returns a diagnostic on missing/mismatched tensors, including packed affine metadata.
    public static func rejectionReason(at directory: URL) -> String? {
        do {
            let data = try Data(contentsOf: directory.appendingPathComponent("config.json"))
            let expected = try requiredShapes(configData: data)
            let root = try JSONSerialization.jsonObject(with: data) as! [String: Any]
            let quant = root["quantization"] as? [String: Any]
            var tensors: [String: (shape: [Int], dtype: String)] = [:]
            guard
                let files = FileManager.default.enumerator(
                    at: directory, includingPropertiesForKeys: nil)
            else { return "Cannot enumerate drafter weights" }
            for case let file as URL in files where file.pathExtension == "safetensors" {
                let handle = try FileHandle(forReadingFrom: file)
                defer { try? handle.close() }
                guard let sizeBytes = try handle.read(upToCount: 8), sizeBytes.count == 8 else {
                    return "Invalid safetensors header"
                }
                let size = sizeBytes.enumerated().reduce(UInt64(0)) {
                    $0 | (UInt64($1.element) << UInt64(8 * $1.offset))
                }
                guard size > 0, size <= 64 * 1024 * 1024,
                    let bytes = try handle.read(upToCount: Int(size)), bytes.count == Int(size),
                    let header = try JSONSerialization.jsonObject(with: bytes) as? [String: Any]
                else { return "Invalid safetensors header" }
                let fileSize = try handle.seekToEnd()
                for (rawName, value) in header where rawName != "__metadata__" {
                    var name = rawName
                    if [
                        "candidate_selector.predecessor_codebook",
                        "candidate_selector.successor_codebook",
                    ].contains(name) {
                        name += ".weight"
                    }
                    guard tensors[name] == nil, let entry = value as? [String: Any],
                        let shape = entry["shape"] as? [Int], shape.allSatisfy({ $0 > 0 }),
                        let dtype = entry["dtype"] as? String,
                        let offsets = entry["data_offsets"] as? [Int], offsets.count == 2,
                        offsets[0] >= 0, offsets[1] >= offsets[0],
                        UInt64(offsets[1]) <= fileSize - 8 - size
                    else { return "Malformed, duplicate or truncated tensor: \(name)" }
                    let sizes = ["F16": 2, "BF16": 2, "F32": 4, "U32": 4]
                    guard let width = sizes[dtype] else {
                        return "Unsupported drafter dtype: \(dtype)"
                    }
                    var count = width
                    for dimension in shape {
                        let product = count.multipliedReportingOverflow(by: dimension)
                        guard !product.overflow else { return "Invalid tensor dimensions" }
                        count = product.partialValue
                    }
                    guard count == offsets[1] - offsets[0] else {
                        return "Incorrect tensor byte length: \(name)"
                    }
                    tensors[name] = (shape, dtype)
                }
            }
            var allowed = Set(expected.keys)
            for (name, shape) in expected {
                guard let tensor = tensors[name] else { return "Missing drafter tensor: \(name)" }
                let prefix = name.hasSuffix(".weight") ? String(name.dropLast(7)) : name
                if let scales = tensors[prefix + ".scales"] {
                    guard shape.count == 2, tensor.dtype == "U32",
                        let quant, (quant["mode"] as? String ?? "affine") == "affine",
                        let bits = quant["bits"] as? Int, [2, 3, 4, 5, 6, 8].contains(bits),
                        let group = quant["group_size"] as? Int, group > 0,
                        shape[1] % group == 0,
                        !shape[1].multipliedReportingOverflow(by: bits).overflow,
                        shape[1].multipliedReportingOverflow(by: bits).partialValue % 32 == 0,
                        tensor.shape == [
                            shape[0],
                            shape[1].multipliedReportingOverflow(by: bits).partialValue / 32,
                        ],
                        scales.shape == [shape[0], shape[1] / group],
                        ["F16", "BF16", "F32"].contains(scales.dtype),
                        let biases = tensors[prefix + ".biases"], biases.shape == scales.shape,
                        biases.dtype == scales.dtype
                    else { return "Incompatible packed drafter tensor: \(name)" }
                    allowed.insert(prefix + ".scales")
                    allowed.insert(prefix + ".biases")
                } else if tensor.shape != shape || !["F16", "BF16", "F32"].contains(tensor.dtype) {
                    return "Incompatible drafter tensor: \(name)"
                }
            }
            if let extra = Set(tensors.keys).subtracting(allowed).sorted().first {
                return "Unexpected drafter tensor: \(extra)"
            }
            return nil
        } catch { return "Drafter metadata cannot be validated: \(error.localizedDescription)" }
    }
}
