// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Cmlx
import Foundation
import MLX

/// Validated, immutable descriptors for native affine/MXFP4 expert slices.
/// Every projection/companion is resolved independently: companions can live
/// in different shards. Mapping a slice never exposes a whole bank to Metal.
public final class MixedQuantizedExpertCatalog: Sendable {
    public struct InvalidBundle: LocalizedError, Sendable {
        public let reason: String
        public var errorDescription: String? { reason }
    }

    public struct Projection {
        public let weight: MLXArray
        public let scales: MLXArray
        public let biases: MLXArray?
        public let bits: Int
        public let groupSize: Int
        public let mode: QuantizationMode
    }

    public struct Expert {
        public let gate: Projection
        public let up: Projection
        public let down: Projection
    }

    private struct Index: Decodable {
        let weightMap: [String: String]
        enum CodingKeys: String, CodingKey { case weightMap = "weight_map" }
    }

    private struct Tensor: Decodable {
        let dtype: String
        let shape: [Int]
        let dataOffsets: [UInt64]
        enum CodingKeys: String, CodingKey {
            case dtype, shape
            case dataOffsets = "data_offsets"
        }
    }

    private struct Region: Sendable {
        let url: URL
        let name: String
        let shape: [Int]
        let dtype: DType
        let offset: UInt64
        let length: UInt64
    }

    private struct Spec: Sendable {
        let weight: Region
        let scales: Region
        let biases: Region?
        let bits: Int
        let groupSize: Int
    }

    private struct Layer: Sendable {
        let gate: Spec
        let up: Spec
        let down: Spec
    }

    public let expertCount: Int
    private let layers: [Int: Layer]

    /// Reads bounded headers only. It does not create any MLX weight buffers.
    public init(directory: URL, layerIndices: [Int], expertCount: Int,
                inputDimensions: Int, hiddenDimensions: Int) throws {
        func invalid(_ message: String) -> InvalidBundle { InvalidBundle(reason: message) }
        guard expertCount > 0, inputDimensions > 0, hiddenDimensions > 0,
            !layerIndices.isEmpty, Set(layerIndices).count == layerIndices.count,
            layerIndices.allSatisfy({ $0 >= 0 }) else {
            throw invalid("Invalid mixed expert dimensions or layer indices")
        }
        let root = directory.resolvingSymlinksInPath().standardizedFileURL
        let index = try JSONDecoder().decode(Index.self, from: Data(contentsOf:
            root.appendingPathComponent("model.safetensors.index.json")))
        var headers: [URL: (UInt64, UInt64, [String: Tensor])] = [:]

        func header(_ filename: String) throws -> (URL, UInt64, UInt64, [String: Tensor]) {
            guard !filename.isEmpty, filename == (filename as NSString).lastPathComponent,
                filename.hasSuffix(".safetensors") else { throw invalid("Invalid shard name: \(filename)") }
            let url = root.appendingPathComponent(filename).resolvingSymlinksInPath().standardizedFileURL
            guard url.deletingLastPathComponent() == root else { throw invalid("Shard escapes bundle: \(filename)") }
            if let cached = headers[url] { return (url, cached.0, cached.1, cached.2) }
            let file = try FileHandle(forReadingFrom: url)
            defer { try? file.close() }
            let fileSize = try file.seekToEnd()
            try file.seek(toOffset: 0)
            guard let prefix = try file.read(upToCount: 8), prefix.count == 8 else {
                throw invalid("Truncated shard header: \(filename)")
            }
            let size = prefix.enumerated().reduce(UInt64(0)) { $0 | UInt64($1.element) << ($1.offset * 8) }
            guard size > 0, size <= 64 * 1024 * 1024, fileSize >= 8, size <= fileSize - 8,
                let data = try file.read(upToCount: Int(size)), data.count == Int(size),
                var object = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                throw invalid("Invalid or oversized shard header: \(filename)")
            }
            object.removeValue(forKey: "__metadata__")
            let tensors = try JSONDecoder().decode([String: Tensor].self,
                from: JSONSerialization.data(withJSONObject: object))
            headers[url] = (8 + size, fileSize, tensors)
            return (url, 8 + size, fileSize, tensors)
        }

        func region(_ name: String) throws -> Region {
            guard let file = index.weightMap[name] else { throw invalid("Missing expert tensor: \(name)") }
            let (url, start, fileSize, tensors) = try header(file)
            guard let tensor = tensors[name], tensor.shape.count == 3,
                tensor.shape.allSatisfy({ $0 > 0 && $0 <= Int32.max }),
                tensor.shape[0] == expertCount, tensor.dataOffsets.count == 2 else {
                throw invalid("Invalid expert shape or offsets: \(name)")
            }
            let dtype: DType
            switch tensor.dtype {
            case "U32": dtype = .uint32
            case "U8": dtype = .uint8
            case "F16": dtype = .float16
            case "BF16": dtype = .bfloat16
            default: throw invalid("Unsupported expert dtype: \(name): \(tensor.dtype)")
            }
            var bytes = UInt64(dtype.size)
            for dimension in tensor.shape {
                let result = bytes.multipliedReportingOverflow(by: UInt64(dimension))
                guard !result.overflow else { throw invalid("Expert byte size overflow: \(name)") }
                bytes = result.partialValue
            }
            let low = tensor.dataOffsets[0], high = tensor.dataOffsets[1]
            guard low <= high, high <= fileSize - start, high - low == bytes,
                bytes <= Int.max, (start + low) % UInt64(dtype.size) == 0 else {
                throw invalid("Expert payload bounds, length, or alignment mismatch: \(name)")
            }
            return Region(url: url, name: name, shape: tensor.shape, dtype: dtype,
                          offset: start + low, length: bytes)
        }

        func projection(_ layer: Int, _ name: String) throws -> Spec {
            let stem = "model.layers.\(layer).mlp.switch_mlp.\(name)"
            let weight = try region(stem + ".weight"), scales = try region(stem + ".scales")
            let biases = try index.weightMap[stem + ".biases"].map { _ in try region(stem + ".biases") }
            let input = name == "down_proj" ? hiddenDimensions : inputDimensions
            let output = name == "down_proj" ? inputDimensions : hiddenDimensions
            guard weight.dtype == .uint32, weight.shape[1] == output, scales.shape[1] == output,
                weight.shape[2] * 32 % input == 0, input % scales.shape[2] == 0 else {
                throw invalid("Expert projection geometry mismatch: \(stem)")
            }
            let bits = weight.shape[2] * 32 / input, group = input / scales.shape[2]
            if let biases {
                guard [2, 4, 8].contains(bits), [32, 64, 128].contains(group),
                    [.float16, .bfloat16].contains(scales.dtype),
                    biases.dtype == scales.dtype, biases.shape == scales.shape else {
                    throw invalid("Invalid native affine expert companions: \(stem)")
                }
            } else {
                guard bits == 4, group == 32, scales.dtype == .uint8 else {
                    throw invalid("Invalid native MXFP4 expert companions: \(stem)")
                }
            }
            return Spec(weight: weight, scales: scales, biases: biases, bits: bits, groupSize: group)
        }
        var built: [Int: Layer] = [:]
        for layer in layerIndices {
            try Task.checkCancellation()
            built[layer] = try Layer(gate: projection(layer, "gate_proj"),
                up: projection(layer, "up_proj"), down: projection(layer, "down_proj"))
        }
        self.expertCount = expertCount
        self.layers = built
    }

    public func loadExpert(layer: Int, index: Int) throws -> Expert {
        guard (0..<expertCount).contains(index), let spec = layers[layer] else {
            throw InvalidBundle(reason: "Invalid expert selection: layer=\(layer), expert=\(index)")
        }
        func map(_ region: Region) throws -> MLXArray {
            let length = region.length / UInt64(expertCount)
            let offset = region.offset + UInt64(index) * length
            var shape = region.shape.dropFirst().map(Int32.init)
            var array = mlx_array_new()
            do {
                let status = try withError {
                    region.url.withUnsafeFileSystemRepresentation { path -> Int32 in
                        guard let path else { return 1 }
                        return mlx_array_new_mmap_file_region(&array, path, offset, Int(length),
                            &shape, Int32(shape.count), region.dtype.cmlxDtype)
                    }
                }
                guard status == 0 else {
                    throw InvalidBundle(reason: "Unable to map expert region: \(region.name)[\(index)]")
                }
                return MLXArray(array)
            } catch {
                mlx_array_free(array)
                throw error
            }
        }
        func load(_ spec: Spec) throws -> Projection {
            try Projection(weight: map(spec.weight), scales: map(spec.scales),
                biases: spec.biases.map(map), bits: spec.bits, groupSize: spec.groupSize,
                mode: spec.biases == nil ? .mxfp4 : .affine)
        }
        return try Expert(gate: load(spec.gate), up: load(spec.up), down: load(spec.down))
    }
}
