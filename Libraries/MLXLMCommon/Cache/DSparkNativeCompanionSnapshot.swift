import Foundation
import MLX

/// Diagnostic full-native companion carried inside its target SSD payload.
/// This deliberately has no separate cache, quota, delta chain or publication.
public struct DSparkNativeCompanionSnapshot {
    static let metadataKey = "__dspark_native_reference_v1__"
    static var enabled: Bool {
        ProcessInfo.processInfo.environment["VMLX_DSPARK_NATIVE_COMPANION_REFERENCE"] == "1"
    }

    struct Header: Codable {
        var version: Int = 1
        var generation: String = UUID().uuidString
        let targetIdentity: String
        let draftIdentity: String
        let prefixIdentity: String
        let boundary: Int
        let captureLayers: [Int]
        let heads: Int
        let headDimension: Int
        let layerCount: Int
        let embeddingDType: String
        let layerDTypes: [String]
    }

    let header: Header
    let arrays: [String: MLXArray]

    static func capture(cache: [KVCache], config: DFlash2Configuration,
                        draftIdentity: String, targetIdentity: String,
                        embeddingDType: DType, tokens: [Int], mediaSalt: String?) -> Self? {
        guard config.isDSpark, config.layerTypes.allSatisfy({ $0 == "full_attention" }),
              !tokens.isEmpty, cache.count == config.numHiddenLayers,
              cache.allSatisfy({ type(of: $0) == KVCacheSimple.self && $0.offset == tokens.count })
        else { return nil }
        // KVCacheSimple.copy owns the valid state, so later append/rollback
        // cannot mutate the snapshot waiting for the target's SSD publication.
        let frozen = cache.map { $0.copy() }
        var payload: [String: MLXArray] = [:]
        var dtypes: [String] = []
        for (index, layer) in frozen.enumerated() {
            let state = layer.state
            guard state.count == 2, valid(state[0], boundary: tokens.count, config: config),
                  valid(state[1], boundary: tokens.count, config: config),
                  state[0].dtype == state[1].dtype else { return nil }
            payload["dspark_ref_\(index)_k"] = state[0]
            payload["dspark_ref_\(index)_v"] = state[1]
            dtypes.append(String(describing: state[0].dtype))
        }
        let header = Header(targetIdentity: targetIdentity, draftIdentity: draftIdentity,
            prefixIdentity: DiskCache.hashTokens(tokens, modelKey: targetIdentity, mediaSalt: mediaSalt),
            boundary: tokens.count, captureLayers: config.targetLayerIds, heads: config.numKeyValueHeads,
            headDimension: config.headDim, layerCount: config.numHiddenLayers,
            embeddingDType: String(describing: embeddingDType), layerDTypes: dtypes)
        guard let data = try? JSONEncoder().encode(header) else { return nil }
        payload[metadataKey] = MLXArray(data.map { Int32($0) })
        MLX.eval(Array(payload.values))
        return Self(header: header, arrays: payload)
    }

    func matches(tokens: [Int], targetIdentity: String?, mediaSalt: String?) -> Bool {
        header.targetIdentity == targetIdentity && header.boundary == tokens.count
            && header.prefixIdentity == DiskCache.hashTokens(tokens, modelKey: targetIdentity, mediaSalt: mediaSalt)
    }

    static func restore(arrays: [String: MLXArray], config: DFlash2Configuration,
                        draftIdentity: String, targetIdentity: String,
                        embeddingDType: DType, tokens: [Int], mediaSalt: String?) -> [KVCache]? {
        guard config.isDSpark, config.layerTypes.allSatisfy({ $0 == "full_attention" }),
              let encoded = arrays[metadataKey], encoded.dtype == .int32,
              encoded.ndim == 1, encoded.size > 0, encoded.size <= 64 * 1024,
              encoded.asArray(Int32.self).allSatisfy({ (0...255).contains($0) }),
              let header = try? JSONDecoder().decode(Header.self,
                  from: Data(encoded.asArray(Int32.self).map { UInt8($0) })),
              header.version == 1, UUID(uuidString: header.generation) != nil,
              header.targetIdentity == targetIdentity, header.draftIdentity == draftIdentity,
              header.boundary == tokens.count, header.boundary > 0,
              header.prefixIdentity == DiskCache.hashTokens(tokens, modelKey: targetIdentity, mediaSalt: mediaSalt),
              header.captureLayers == config.targetLayerIds,
              header.heads == config.numKeyValueHeads, header.headDimension == config.headDim,
              header.layerCount == config.numHiddenLayers,
              header.layerDTypes.count == config.numHiddenLayers,
              header.embeddingDType == String(describing: embeddingDType)
        else { return nil }
        var result: [KVCache] = []
        for index in 0..<config.numHiddenLayers {
            guard let keys = arrays["dspark_ref_\(index)_k"],
                  let values = arrays["dspark_ref_\(index)_v"],
                  valid(keys, boundary: header.boundary, config: config),
                  valid(values, boundary: header.boundary, config: config),
                  keys.dtype == values.dtype,
                  String(describing: keys.dtype) == header.layerDTypes[index]
            else { return nil }
            let cache = KVCacheSimple()
            cache.state = [keys, values]
            result.append(cache)
        }
        return result
    }

    private static func valid(_ array: MLXArray, boundary: Int, config: DFlash2Configuration) -> Bool {
        array.shape == [1, config.numKeyValueHeads, boundary, config.headDim]
            && [.float16, .bfloat16, .float32].contains(array.dtype)
    }
}
