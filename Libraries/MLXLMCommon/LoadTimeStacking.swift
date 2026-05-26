import Foundation
import MLX

private func loadTimeMachStacksEnabled() -> Bool {
    let env = ProcessInfo.processInfo.environment
    let raw = env["MLXPRESS_LOADTIME_MACH_STACKS"]
        ?? env["JANGPRESS_LOADTIME_MACH_STACKS"]
        ?? "0"
    let normalized = raw.lowercased()
    return normalized == "1"
        || normalized == "true"
        || normalized == "yes"
        || normalized == "on"
}

private func loadTimeMachReleaseAfterStack() -> Bool {
    let env = ProcessInfo.processInfo.environment
    let raw = env["MLXPRESS_LOADTIME_MACH_RELEASE_AFTER_STACK"]
        ?? env["JANGPRESS_LOADTIME_MACH_RELEASE_AFTER_STACK"]
        ?? "1"
    let normalized = raw.lowercased()
    return !(normalized == "0"
        || normalized == "false"
        || normalized == "no"
        || normalized == "off")
}

private func loadTimeMachCompressPercent() -> Int {
    let env = ProcessInfo.processInfo.environment
    let raw = env["MLXPRESS_LOADTIME_MACH_COMPRESS_PCT"]
        ?? env["JANGPRESS_LOADTIME_MACH_COMPRESS_PCT"]
        ?? "70"
    return max(0, min(100, Int(raw) ?? 70))
}

private func loadTimeStackMaterializationEnabled() -> Bool {
    let env = ProcessInfo.processInfo.environment
    let raw = env["MLXPRESS_LOADTIME_MATERIALIZE_STACKS"]
        ?? env["JANGPRESS_LOADTIME_MATERIALIZE_STACKS"]
        ?? "1"
    let normalized = raw.lowercased()
    return !(normalized == "0"
        || normalized == "false"
        || normalized == "no"
        || normalized == "off")
}

/// Stack load-time per-expert tensors and immediately materialize the result.
///
/// MLX stack operations are lazy. During model `sanitize(...)`, that can leave
/// a retained graph pointing at every per-expert input tensor until the final
/// model eval. Large MoE/JANGTQ bundles can have tens of thousands of
/// per-expert tensors, so the lazy graph doubles the peak footprint and can
/// crash before the model finishes loading. Use this helper for weight-loading
/// restacks, not for small runtime stacks in forward passes.
public func loadTimeMaterializedStacked(_ arrays: [MLXArray], axis: Int = 0) -> MLXArray {
    let result = MLX.stacked(arrays, axis: axis)
    guard loadTimeStackMaterializationEnabled() else {
        return result
    }
    MLXCacheIOLock.withSerializedMLXCacheIO {
        MLX.eval(result)
        Stream.gpu.synchronize()
        MLX.Memory.clearCache()
    }
    return result
}

private final class MLXPressLoadTimeMachStackStore: @unchecked Sendable {
    static let shared = MLXPressLoadTimeMachStackStore()

    private let lock = NSLock()
    private var cache: JangPressMachCache?
    private var didLogEnablement = false

    func stack(_ arrays: [MLXArray], axis: Int, label: String) -> MLXArray? {
        guard loadTimeMachStacksEnabled() else { return nil }

        let stacked = MLX.stacked(arrays, axis: axis)
        MLX.eval(stacked)
        let arrayData = stacked.asData(access: .copy)
        let layer = Self.layerIndex(from: label) ?? 0
        let component = label

        do {
            let machArray: MLXArray = try arrayData.data.withUnsafeBytes { raw in
                guard raw.baseAddress != nil || raw.count == 0 else {
                    throw JangPressMachError.mmapFailed(EINVAL)
                }
                let cache = try self.cacheLocked()
                _ = try cache.register(
                    layer: layer,
                    expert: 0,
                    component: component,
                    bytes: raw)
                let array = try cache.array(
                    layer: layer,
                    expert: 0,
                    component: component,
                    shape: arrayData.shape,
                    dtype: arrayData.dType)
                if loadTimeMachReleaseAfterStack() {
                    cache.release(layer: layer, expert: 0, components: [component])
                }
                return array
            }
            MLX.Memory.clearCache()
            return machArray
        } catch {
            FileHandle.standardError.write(Data(
                "[MLXPressMachStack] failed label=\(label): \(error); falling back to MLX stack\n".utf8))
            MLX.Memory.clearCache()
            return nil
        }
    }

    private func cacheLocked() throws -> JangPressMachCache {
        lock.lock()
        defer { lock.unlock() }
        if let cache { return cache }
        if !didLogEnablement {
            let message = "[MLXPressMachStack] load-time Mach-backed routed stacks enabled "
                + "compressPct=\(loadTimeMachCompressPercent()) "
                + "releaseAfterStack=\(loadTimeMachReleaseAfterStack())\n"
            FileHandle.standardError.write(Data(message.utf8))
            didLogEnablement = true
        }
        let created = JangPressMachCache(config: .init(
            enablePrefetch: false,
            enableDiskRefault: false,
            manualCompressPercent: loadTimeMachCompressPercent()))
        cache = created
        return created
    }

    private static func layerIndex(from label: String) -> Int? {
        let marker = "model.layers."
        guard let range = label.range(of: marker) else { return nil }
        var digits = ""
        var index = range.upperBound
        while index < label.endIndex {
            let char = label[index]
            guard char >= "0" && char <= "9" else { break }
            digits.append(char)
            index = label.index(after: index)
        }
        return Int(digits)
    }
}

/// Stack load-time routed expert tensors into purgeable Mach VM and return an
/// MLXArray that points at that region. This is opt-in and intentionally used
/// only for diagnostics until every family proves coherent low-footprint decode.
public func loadTimeMachBackedStacked(
    _ arrays: [MLXArray],
    axis: Int = 0,
    label: String
) -> MLXArray? {
    MLXPressLoadTimeMachStackStore.shared.stack(arrays, axis: axis, label: label)
}
