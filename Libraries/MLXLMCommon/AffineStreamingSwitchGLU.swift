// Copyright © 2026 Apple Inc.

import Cmlx
import Foundation
import MLX
import MLXNN

/// Errors raised before a file-backed affine expert is admitted to the model
/// graph. These are deliberately structural: a malformed bundle must fail at
/// load instead of silently falling back to a full resident expert bank.
public enum AffineStreamingExpertError: Error, CustomStringConvertible {
    case missingIndex(URL)
    case malformedIndex(URL)
    case missingTensor(String)
    case invalidTensor(String, String)
    case mmapFailed(String)
    case notConfigured
    case invalidExpert(layer: Int, expert: Int)

    public var description: String {
        switch self {
        case .missingIndex(let url): return "missing safetensors index: \(url.path)"
        case .malformedIndex(let url): return "malformed safetensors index: \(url.path)"
        case .missingTensor(let name): return "missing affine routed tensor: \(name)"
        case .invalidTensor(let name, let reason):
            return "invalid affine routed tensor \(name): \(reason)"
        case .mmapFailed(let name): return "unable to mmap affine routed tensor region: \(name)"
        case .notConfigured: return "affine routed expert catalog is not configured"
        case .invalidExpert(let layer, let expert):
            return "invalid routed expert \(expert) for layer \(layer)"
        }
    }
}

/// Header-only catalog for Qwen4's stacked native-affine expert tensors.
///
/// The generic loader must exclude these tensors. At runtime this catalog maps
/// one exact expert slice at a time with `mlx_array_new_mmap_file_region`, so a
/// Metal command never receives the 512-expert backing buffer that caused the
/// entire routed bank to become Wired Memory.
public final class AffineStreamingExpertCatalog: @unchecked Sendable {
    public enum Projection: String, CaseIterable, Sendable {
        case gate = "gate_proj"
        case up = "up_proj"
        case down = "down_proj"
    }

    public struct ProjectionArrays {
        public let weight: MLXArray
        public let scales: MLXArray
        public let biases: MLXArray
        public let groupSize: Int
        public let bits: Int
    }

    public struct ExpertArrays {
        public let gate: ProjectionArrays
        public let up: ProjectionArrays
        public let down: ProjectionArrays
    }

    private struct IndexFile: Decodable {
        let weightMap: [String: String]
        enum CodingKeys: String, CodingKey { case weightMap = "weight_map" }
    }

    private struct Region: Sendable {
        let url: URL
        let name: String
        let dtype: DType
        let shape: [Int]
        let dataOffset: UInt64
        let dataLength: UInt64
    }

    private struct ProjectionSpec: Sendable {
        let weight: Region
        let scales: Region
        let biases: Region
        let groupSize: Int
        let bits: Int
    }

    private struct LayerSpec: Sendable {
        let gate: ProjectionSpec
        let up: ProjectionSpec
        let down: ProjectionSpec
    }

    public let numExperts: Int
    public let inputDims: Int
    public let hiddenDims: Int

    private let lock = NSLock()
    private var layers: [Int: LayerSpec] = [:]
    private var configuredDirectory: URL?

    public init(numExperts: Int, inputDims: Int, hiddenDims: Int) {
        self.numExperts = numExperts
        self.inputDims = inputDims
        self.hiddenDims = hiddenDims
    }

    /// Exact key predicate used by Qwen4's generic-loader exclusion gate.
    public static func isRoutedTensorKey(_ originalKey: String) -> Bool {
        var key = originalKey
        if key.hasPrefix("model.") { key.removeFirst("model.".count) }
        let parts = key.split(separator: ".", omittingEmptySubsequences: false)
        guard parts.count == 7,
            parts[0] == "language_model", parts[1] == "layers",
            Int(parts[2]) != nil, parts[3] == "mlp", parts[4] == "switch_mlp",
            ["gate_proj", "up_proj", "down_proj"].contains(String(parts[5])),
            ["weight", "scales", "biases"].contains(String(parts[6]))
        else { return false }
        return true
    }

    public func configure(modelDirectory: URL, layerCount: Int) throws {
        let resolved = modelDirectory.resolvingSymlinksInPath()
        lock.lock()
        if configuredDirectory == resolved, layers.count == layerCount {
            lock.unlock()
            return
        }
        lock.unlock()

        let indexURL = resolved.appendingPathComponent("model.safetensors.index.json")
        guard let data = try? Data(contentsOf: indexURL) else {
            throw AffineStreamingExpertError.missingIndex(indexURL)
        }
        guard let index = try? JSONDecoder().decode(IndexFile.self, from: data) else {
            throw AffineStreamingExpertError.malformedIndex(indexURL)
        }

        var shards: [URL: JangPressShard] = [:]
        var built: [Int: LayerSpec] = [:]
        built.reserveCapacity(layerCount)

        func canonicalName(layer: Int, projection: Projection, suffix: String) -> String {
            "language_model.layers.\(layer).mlp.switch_mlp.\(projection.rawValue).\(suffix)"
        }

        func locate(_ canonical: String) throws -> Region {
            let indexedName: String
            let filename: String
            if let found = index.weightMap[canonical] {
                indexedName = canonical
                filename = found
            } else if let found = index.weightMap["model.\(canonical)"] {
                indexedName = "model.\(canonical)"
                filename = found
            } else {
                throw AffineStreamingExpertError.missingTensor(canonical)
            }
            let url = resolved.appendingPathComponent(filename)
            let shard: JangPressShard
            if let existing = shards[url] {
                shard = existing
            } else {
                shard = try JangPressShard(path: url)
                shards[url] = shard
            }
            guard let descriptor = shard.descriptor(for: indexedName),
                let dtype = Self.dtype(named: descriptor.dtype)
            else {
                throw AffineStreamingExpertError.invalidTensor(
                    indexedName, "missing descriptor or unsupported dtype")
            }
            let expectedBytes = try Self.checkedByteCount(
                shape: descriptor.shape, dtype: dtype, name: indexedName)
            guard descriptor.dataLength == expectedBytes,
                descriptor.dataOffset <= shard.fileSize,
                descriptor.dataLength <= shard.fileSize - descriptor.dataOffset,
                descriptor.dataOffset % UInt64(dtype.size) == 0
            else {
                throw AffineStreamingExpertError.invalidTensor(
                    indexedName, "shape byte count, file bounds, or dtype alignment mismatch")
            }
            return Region(
                url: url, name: indexedName, dtype: dtype, shape: descriptor.shape,
                dataOffset: descriptor.dataOffset, dataLength: descriptor.dataLength)
        }

        func projectionSpec(layer: Int, projection: Projection) throws -> ProjectionSpec {
            let weight = try locate(canonicalName(layer: layer, projection: projection, suffix: "weight"))
            let scales = try locate(canonicalName(layer: layer, projection: projection, suffix: "scales"))
            let biases = try locate(canonicalName(layer: layer, projection: projection, suffix: "biases"))
            let inFeatures = projection == .down ? hiddenDims : inputDims
            let outFeatures = projection == .down ? inputDims : hiddenDims
            let label = weight.name
            guard weight.dtype == .uint32, scales.dtype == .float16,
                biases.dtype == .float16,
                weight.shape.count == 3, scales.shape.count == 3,
                biases.shape == scales.shape,
                weight.shape[0] == numExperts, scales.shape[0] == numExperts,
                weight.shape[1] == outFeatures, scales.shape[1] == outFeatures,
                weight.shape[2] > 0, scales.shape[2] > 0
            else {
                throw AffineStreamingExpertError.invalidTensor(
                    label, "expected stacked U32 weight and matching F16 affine companions")
            }
            let packedBits = weight.shape[2].multipliedReportingOverflow(by: 32)
            guard !packedBits.overflow, packedBits.partialValue % inFeatures == 0 else {
                throw AffineStreamingExpertError.invalidTensor(label, "non-integral packed bit width")
            }
            let bits = packedBits.partialValue / inFeatures
            // MLX's native affine quantized kernels accept 2/4/8-bit packed
            // weights. PLE's independent SSD row decoder supports odd widths,
            // but routed matmul must reject those rather than reaching Metal.
            guard [2, 4, 8].contains(bits),
                inFeatures % scales.shape[2] == 0
            else {
                throw AffineStreamingExpertError.invalidTensor(
                    label, "unsupported bit width or affine group geometry")
            }
            let groupSize = inFeatures / scales.shape[2]
            return ProjectionSpec(
                weight: weight, scales: scales, biases: biases,
                groupSize: groupSize, bits: bits)
        }

        for layer in 0 ..< layerCount {
            built[layer] = try LayerSpec(
                gate: projectionSpec(layer: layer, projection: .gate),
                up: projectionSpec(layer: layer, projection: .up),
                down: projectionSpec(layer: layer, projection: .down))
        }

        lock.lock()
        configuredDirectory = resolved
        layers = built
        lock.unlock()

        let mappedBytes = shards.values.reduce(UInt64(0)) { $0 + $1.fileSize }
        FileHandle.standardError.write(Data(
            "[Qwen4AffineExperts] exact-region catalog ready layers=\(layerCount) experts=\(numExperts) files=\(shards.count) indexed_file_bytes=\(mappedBytes) cache=file-backed\n".utf8))
    }

    public func loadExpert(layer: Int, expert: Int) throws -> ExpertArrays {
        guard expert >= 0, expert < numExperts else {
            throw AffineStreamingExpertError.invalidExpert(layer: layer, expert: expert)
        }
        lock.lock()
        let layerSpec = layers[layer]
        let configured = configuredDirectory != nil
        lock.unlock()
        guard configured else { throw AffineStreamingExpertError.notConfigured }
        guard let layerSpec else {
            throw AffineStreamingExpertError.invalidExpert(layer: layer, expert: expert)
        }
        return try ExpertArrays(
            gate: loadProjection(layerSpec.gate, expert: expert),
            up: loadProjection(layerSpec.up, expert: expert),
            down: loadProjection(layerSpec.down, expert: expert))
    }

    private func loadProjection(_ spec: ProjectionSpec, expert: Int) throws -> ProjectionArrays {
        ProjectionArrays(
            weight: try mapExpertRegion(spec.weight, expert: expert),
            scales: try mapExpertRegion(spec.scales, expert: expert),
            biases: try mapExpertRegion(spec.biases, expert: expert),
            groupSize: spec.groupSize,
            bits: spec.bits)
    }

    private func mapExpertRegion(_ region: Region, expert: Int) throws -> MLXArray {
        guard region.shape.first == numExperts,
            region.dataLength % UInt64(numExperts) == 0
        else {
            throw AffineStreamingExpertError.invalidTensor(
                region.name, "expert dimension does not divide payload")
        }
        let length = region.dataLength / UInt64(numExperts)
        let offsetResult = UInt64(expert).multipliedReportingOverflow(by: length)
        guard !offsetResult.overflow,
            region.dataOffset <= UInt64.max - offsetResult.partialValue
        else {
            throw AffineStreamingExpertError.invalidTensor(region.name, "expert offset overflow")
        }
        let offset = region.dataOffset + offsetResult.partialValue
        let shape = Array(region.shape.dropFirst())
        var cShape = shape.map(Int32.init)
        var result = mlx_array_new()
        let rc = region.url.withUnsafeFileSystemRepresentation { path -> Int32 in
            guard let path else { return 1 }
            return mlx_array_new_mmap_file_region(
                &result, path, offset, Int(length), &cShape, Int32(cShape.count),
                region.dtype.cmlxDtype)
        }
        guard rc == 0 else {
            mlx_array_free(result)
            throw AffineStreamingExpertError.mmapFailed("\(region.name)[\(expert)]")
        }
        return MLXArray(result)
    }

    private static func dtype(named name: String) -> DType? {
        switch name.uppercased() {
        case "U32": return .uint32
        case "F16": return .float16
        case "BF16": return .bfloat16
        case "F32": return .float32
        default: return nil
        }
    }

    private static func checkedByteCount(shape: [Int], dtype: DType, name: String) throws -> UInt64 {
        var count = UInt64(dtype.size)
        for dimension in shape {
            guard dimension >= 0 else {
                throw AffineStreamingExpertError.invalidTensor(name, "negative dimension")
            }
            let next = count.multipliedReportingOverflow(by: UInt64(dimension))
            guard !next.overflow else {
                throw AffineStreamingExpertError.invalidTensor(name, "shape byte count overflow")
            }
            count = next.partialValue
        }
        return count
    }
}

/// Qwen4-only native affine MoE path backed by exact safetensors expert slices.
/// The cache retains file mappings, not copied tensors, and is strictly bounded
/// per layer. Each chunk is evaluated before returning so mappings from old
/// routing decisions can be evicted and unpinned before the next layer runs.
public final class AffineStreamingSwitchGLU: Module {
    private struct CachedExpert {
        let arrays: AffineStreamingExpertCatalog.ExpertArrays
        var access: UInt64
    }

    public let inputDims: Int
    public let hiddenDims: Int
    public let numExperts: Int
    public let layerIndex: Int
    public let cacheExpertLimit: Int
    public let prefillChunkSize: Int

    private let catalog: AffineStreamingExpertCatalog
    private let lock = NSLock()
    private var cache: [Int: CachedExpert] = [:]
    private var accessClock: UInt64 = 0

    public init(
        inputDims: Int,
        hiddenDims: Int,
        numExperts: Int,
        layerIndex: Int,
        catalog: AffineStreamingExpertCatalog,
        cacheExpertLimit: Int = 16,
        prefillChunkSize: Int = 4
    ) {
        self.inputDims = inputDims
        self.hiddenDims = hiddenDims
        self.numExperts = numExperts
        self.layerIndex = layerIndex
        self.catalog = catalog
        self.cacheExpertLimit = max(1, cacheExpertLimit)
        self.prefillChunkSize = max(1, prefillChunkSize)
        super.init()
    }

    public var cachedExpertCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return cache.count
    }

    public func reduced(_ x: MLXArray, indices: MLXArray, scores: MLXArray) -> MLXArray {
        let totalTokens = x.size / inputDims
        let kSlots = indices.dim(-1)
        let xFlat = x.reshaped([totalTokens, inputDims])
        let indicesFlat = indices.reshaped([totalTokens, kSlots])
        let scoresFlat = scores.reshaped([totalTokens, kSlots])
        var chunks: [MLXArray] = []
        chunks.reserveCapacity((totalTokens + prefillChunkSize - 1) / prefillChunkSize)

        var start = 0
        while start < totalTokens {
            let end = min(totalTokens, start + prefillChunkSize)
            let chunkIndices = indicesFlat[start ..< end, 0...]
            let ids = chunkIndices.reshaped([-1]).asArray(Int32.self).map(Int.init)
            guard !ids.isEmpty, ids.allSatisfy({ $0 >= 0 && $0 < numExperts }) else {
                fatalError("[Qwen4AffineExperts] invalid routed IDs at layer \(layerIndex)")
            }
            let unique = Array(Set(ids)).sorted()
            let selected = unique.map { expert($0) }
            var remap: [Int: Int32] = [:]
            remap.reserveCapacity(unique.count)
            for (local, global) in unique.enumerated() { remap[global] = Int32(local) }
            let localIndices = MLXArray(
                ids.map { remap[$0]! }, chunkIndices.shape).asType(.uint32)

            func bank(
                _ projection: (AffineStreamingExpertCatalog.ExpertArrays)
                    -> AffineStreamingExpertCatalog.ProjectionArrays,
                _ value: (AffineStreamingExpertCatalog.ProjectionArrays) -> MLXArray
            ) -> MLXArray {
                let arrays = selected.map { value(projection($0)) }
                return arrays.count == 1
                    ? expandedDimensions(arrays[0], axis: 0)
                    : stacked(arrays, axis: 0)
            }

            let gate0 = selected[0].gate
            let up0 = selected[0].up
            let down0 = selected[0].down
            let input = expandedDimensions(xFlat[start ..< end, 0...], axes: [-2, -3])
            let up = MLX.gatherQuantizedMM(
                input, bank({ $0.up }, { $0.weight }),
                scales: bank({ $0.up }, { $0.scales }),
                biases: bank({ $0.up }, { $0.biases }), rhsIndices: localIndices,
                transpose: true, groupSize: up0.groupSize, bits: up0.bits,
                mode: .affine)
            let gate = MLX.gatherQuantizedMM(
                input, bank({ $0.gate }, { $0.weight }),
                scales: bank({ $0.gate }, { $0.scales }),
                biases: bank({ $0.gate }, { $0.biases }), rhsIndices: localIndices,
                transpose: true, groupSize: gate0.groupSize, bits: gate0.bits,
                mode: .affine)
            let activated = silu(gate) * up
            let down = MLX.gatherQuantizedMM(
                activated, bank({ $0.down }, { $0.weight }),
                scales: bank({ $0.down }, { $0.scales }),
                biases: bank({ $0.down }, { $0.biases }), rhsIndices: localIndices,
                transpose: true, groupSize: down0.groupSize, bits: down0.bits,
                mode: .affine)
            let routed = squeezed(down, axis: -2)
            let reduced = (routed * scoresFlat[start ..< end, 0...][.ellipsis, .newAxis])
                .sum(axis: -2)
            MLX.eval(reduced)
            chunks.append(reduced)
            start = end
        }

        let result = chunks.count == 1 ? chunks[0] : concatenated(chunks, axis: 0)
        // This path REQUIRES f16 scales, so gatherQuantizedMM promotes bf16
        // activations to fp32 (promote_types) and the whole MoE block leaked
        // fp32 into the residual stream. Pin to the activation dtype.
        return result.reshaped(x.shape).asType(x.dtype)
    }

    private func expert(_ index: Int) -> AffineStreamingExpertCatalog.ExpertArrays {
        lock.lock()
        accessClock &+= 1
        if var cached = cache[index] {
            cached.access = accessClock
            cache[index] = cached
            lock.unlock()
            return cached.arrays
        }
        lock.unlock()

        let loaded: AffineStreamingExpertCatalog.ExpertArrays
        do {
            loaded = try catalog.loadExpert(layer: layerIndex, expert: index)
        } catch {
            fatalError("[Qwen4AffineExperts] \(error)")
        }

        lock.lock()
        accessClock &+= 1
        if let raced = cache[index] {
            lock.unlock()
            return raced.arrays
        }
        cache[index] = CachedExpert(arrays: loaded, access: accessClock)
        while cache.count > cacheExpertLimit,
            let victim = cache.min(by: { $0.value.access < $1.value.access })?.key
        {
            cache.removeValue(forKey: victim)
        }
        let count = cache.count
        lock.unlock()

        if accessClock == 2 || accessClock % 256 == 0 {
            FileHandle.standardError.write(Data(
                "[Qwen4AffineExperts] layer=\(layerIndex) mapped_expert=\(index) cache=\(count)/\(cacheExpertLimit) backend=exact-mmap-region mode=affine\n".utf8))
        }
        return loaded
    }
}
