import Foundation
import Cmlx
import MLX
import MLXNN

#if canImport(Darwin)
    import Darwin
#elseif canImport(Glibc)
    import Glibc
#endif

public enum JANGTQStreamingExperts {
    public static var isEnabled: Bool {
        let env = ProcessInfo.processInfo.environment
        let raw =
            env["MLXPRESS_STREAMING_EXPERTS"]?.lowercased()
            ?? env["JANGPRESS_STREAMING_EXPERTS"]?.lowercased()
            ?? "0"
        return raw == "1" || raw == "true" || raw == "yes" || raw == "on"
    }

    public static var residentExpertsEnabled: Bool {
        let env = ProcessInfo.processInfo.environment
        let raw =
            env["MLXPRESS_RESIDENT_EXPERTS"]?
            .lowercased()
            ?? env["JANGPRESS_RESIDENT_EXPERTS"]?.lowercased()
            ?? "0"
        return raw == "1" || raw == "true" || raw == "yes" || raw == "on"
    }

    public static var usesActiveExpertModule: Bool {
        isEnabled || residentExpertsEnabled
    }

    public static func configureModelDirectory(_ modelDirectory: URL) {
        JANGTQStreamingExpertStore.shared.configureModelDirectory(modelDirectory)
    }

    public static func clearConfiguredModelDirectory() {
        JANGTQStreamingExpertStore.shared.clearConfiguredModelDirectory()
    }

    public static func configuredModelDirectoryForDiagnostics() -> URL? {
        JANGTQStreamingExpertStore.shared.configuredModelDirectoryForDiagnostics()
    }

    public static func resetResidentTensors() {
        JANGTQStreamingExpertStore.shared.resetResidentTensors()
    }

    public static func registerResidentTensor(
        layerIdx: Int,
        expertIdx: Int,
        projectionName: String,
        suffixName: String,
        array: MLXArray
    ) {
        guard let projection = StreamingProjection(rawValue: projectionName),
            let suffix = StreamingSuffix(rawValue: suffixName)
        else { return }
        JANGTQStreamingExpertStore.shared.registerResidentTensor(
            layerIdx: layerIdx,
            expertIdx: expertIdx,
            projection: projection,
            suffix: suffix,
            array: array)
    }

    public static var materializeBeforeRouterReadback: Bool {
        let env = ProcessInfo.processInfo.environment
        let raw =
            env["MLXPRESS_STREAMING_MATERIALIZE_ROUTER_INPUT"]?
            .lowercased()
            ?? env["JANGPRESS_STREAMING_MATERIALIZE_ROUTER_INPUT"]?.lowercased()
            ?? "1"
        return raw != "0" && raw != "false" && raw != "no" && raw != "off"
    }

    public static func hasStreamableExperts(in modelDirectory: URL) -> Bool {
        let resolved = modelDirectory.resolvingSymlinksInPath()
        guard let index = try? JANGTQStreamingExpertIndex.build(modelDirectory: resolved) else {
            return false
        }
        return !index.layers.isEmpty
    }

    public static func hasUnstackedStreamableExperts(in modelDirectory: URL) -> Bool {
        let resolved = modelDirectory.resolvingSymlinksInPath()
        guard let index = try? JANGTQStreamingExpertIndex.build(modelDirectory: resolved) else {
            return false
        }
        return index.layers.values.contains { !$0.experts.isEmpty }
    }

    public static func isStreamableRoutedTensorKey(_ key: String) -> Bool {
        guard key.hasSuffix(".tq_packed") || key.hasSuffix(".tq_norms") else {
            return false
        }
        let patterns = [
            #"^(?:language_model\.)?model\.layers\.\d+\.mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)\.(tq_packed|tq_norms)$"#,
            #"^layers\.\d+\.ffn\.experts\.\d+\.(w1|w2|w3)\.(tq_packed|tq_norms)$"#,
            #"^(?:language_model\.)?model\.layers\.\d+\.(?:mlp|block_sparse_moe)\.experts\.\d+\.(w1|w2|w3)\.(tq_packed|tq_norms)$"#,
            #"^backbone\.layers\.\d+\.mixer\.experts\.\d+\.(up_proj|down_proj)\.(tq_packed|tq_norms)$"#,
            #"^(?:language_model\.)?model\.layers\.\d+\.(?:mlp|block_sparse_moe)\.switch_mlp\.(gate_proj|up_proj|down_proj)\.(tq_packed|tq_norms)$"#,
            #"^(?:language_model\.)?model\.layers\.\d+\.(?:mlp\.)?zaya_block\.experts\.switch_mlp\.(gate_proj|up_proj|down_proj)\.(tq_packed|tq_norms)$"#,
            #"^layers\.\d+\.ffn\.switch_mlp\.(w1|w2|w3)\.(tq_packed|tq_norms)$"#,
            #"^backbone\.layers\.\d+\.mixer\.switch_mlp\.(up_proj|down_proj|fc1|fc2)\.(tq_packed|tq_norms)$"#,
        ]
        return patterns.contains { pattern in
            key.range(of: pattern, options: .regularExpression) != nil
        }
    }

    public static func stackedOffsetDescriptor(
        in modelDirectory: URL,
        layerIdx: Int,
        projectionName: String,
        suffixName: String
    ) -> JANGTQStackedOffsetDescriptor? {
        guard let projection = StreamingProjection(rawValue: projectionName),
              let suffix = StreamingSuffix(rawValue: suffixName),
              let index = try? JANGTQStreamingExpertIndex.build(
                modelDirectory: modelDirectory.resolvingSymlinksInPath())
        else { return nil }
        return index.stackedOffsetDescriptor(
            layerIdx: layerIdx,
            projection: projection,
            suffix: suffix)
    }

    public static func stackedOffsetDescriptors(
        in modelDirectory: URL,
        layerIdx: Int,
        projectionName: String,
        suffixName: String
    ) -> [JANGTQStackedOffsetDescriptor] {
        guard let projection = StreamingProjection(rawValue: projectionName),
              let suffix = StreamingSuffix(rawValue: suffixName),
              let index = try? JANGTQStreamingExpertIndex.build(
                modelDirectory: modelDirectory.resolvingSymlinksInPath())
        else { return [] }
        return index.stackedOffsetDescriptors(
            layerIdx: layerIdx,
            projection: projection,
            suffix: suffix)
    }
}

public struct JANGTQStackedOffsetDescriptor: Sendable, Equatable {
    public let layerIdx: Int
    public let projectionName: String
    public let suffixName: String
    public let fileURL: URL
    public let spanOffset: UInt64
    public let spanByteCount: Int
    public let expertByteCount: Int
    public let expertByteOffsets: [UInt64]
    public let logicalShape: [Int]
    public let dtype: String
    public let storageLayout: String

    public var expertCount: Int {
        expertByteOffsets.count
    }
}

public struct JANGTQActiveOffsetWindow: Sendable, Equatable {
    public let start: UInt64
    public let end: UInt64
    public let experts: [Int]

    public var byteCount: UInt64 {
        end > start ? end - start : 0
    }
}

extension JANGTQStackedOffsetDescriptor {
    public func activeExpertByteWindows(
        activeExperts: Set<Int>,
        elementByteSize: Int,
        maxGapBytes: UInt64? = nil
    ) -> [JANGTQActiveOffsetWindow] {
        guard !activeExperts.isEmpty, elementByteSize > 0 else { return [] }

        var ranges: [(expert: Int, start: UInt64, end: UInt64)] = []
        for (expertIdx, byteOffset) in expertByteOffsets.enumerated() {
            guard byteOffset != UInt64.max,
                activeExperts.contains(expertIdx)
            else { continue }
            let absoluteStart = spanOffset + byteOffset
            let absoluteEnd = absoluteStart + UInt64(expertByteCount)
            guard absoluteEnd > absoluteStart,
                (absoluteEnd - absoluteStart) % UInt64(elementByteSize) == 0
            else { continue }
            ranges.append((expertIdx, absoluteStart, absoluteEnd))
        }
        guard !ranges.isEmpty else { return [] }

        let sortedRanges = ranges.sorted {
            if $0.start == $1.start { return $0.expert < $1.expert }
            return $0.start < $1.start
        }

        guard let maxGapBytes else {
            return sortedRanges.map { range in
                JANGTQActiveOffsetWindow(
                    start: range.start,
                    end: range.end,
                    experts: [range.expert])
            }
        }

        var windows: [JANGTQActiveOffsetWindow] = []
        var currentStart = sortedRanges[0].start
        var currentEnd = sortedRanges[0].end
        var currentExperts = [sortedRanges[0].expert]

        for range in sortedRanges.dropFirst() {
            let gap = range.start > currentEnd ? range.start - currentEnd : 0
            if gap <= maxGapBytes {
                currentEnd = max(currentEnd, range.end)
                currentExperts.append(range.expert)
                continue
            }

            windows.append(JANGTQActiveOffsetWindow(
                start: currentStart,
                end: currentEnd,
                experts: currentExperts.sorted()))
            currentStart = range.start
            currentEnd = range.end
            currentExperts = [range.expert]
        }

        windows.append(JANGTQActiveOffsetWindow(
            start: currentStart,
            end: currentEnd,
            experts: currentExperts.sorted()))
        return windows
    }
}

public struct MLXPressStreamingProfileRow: Sendable {
    public let name: String
    public let count: Int
    public let seconds: Double
    public let bytes: Int

    public var milliseconds: Double {
        seconds * 1000
    }

    public var averageMilliseconds: Double {
        milliseconds / Double(max(1, count))
    }

    public var bytesMB: Double {
        Double(bytes) / (1024 * 1024)
    }

    public var bytesPerCallMB: Double {
        bytesMB / Double(max(1, count))
    }

    public var bandwidthMBps: Double {
        seconds > 0 ? bytesMB / seconds : 0
    }
}

public struct MLXPressStreamingProfileSnapshot: Sendable {
    public let totalSeconds: Double
    public let rows: [MLXPressStreamingProfileRow]

    public var isEmpty: Bool {
        rows.isEmpty
    }
}

public enum MLXPressStreamingProfile {
    public static var isEnabled: Bool {
        MLXPressStreamingProfileState.shared.isEnabled
    }

    public static func time<T>(_ name: String, bytes: Int = 0, _ body: () -> T) -> T {
        MLXPressStreamingProfileState.shared.time(name, bytes: bytes, body)
    }

    public static func record(_ name: String, seconds: Double = 0, bytes: Int = 0) {
        MLXPressStreamingProfileState.shared.record(name, seconds: seconds, bytes: bytes)
    }

    public static func dump(reason: String) {
        MLXPressStreamingProfileState.shared.dump(reason: reason)
    }

    public static func snapshot(maxRows: Int? = nil) -> MLXPressStreamingProfileSnapshot {
        MLXPressStreamingProfileState.shared.snapshot(maxRows: maxRows)
    }
}

public struct MLXPressActiveExpertTraceExpert: Sendable {
    public let expertIdx: Int
    public let count: Int
}

public struct MLXPressActiveExpertTraceLayer: Sendable {
    public let layerIdx: Int
    public let calls: Int
    public let tokenCount: Int
    public let routedSlots: Int
    public let uniqueExpertTouches: Int
    public let repeatedWithinChunk: Int
    public let consecutiveReuseTouches: Int
    public let topExperts: [MLXPressActiveExpertTraceExpert]

    public var averageUniqueExpertsPerCall: Double {
        Double(uniqueExpertTouches) / Double(max(1, calls))
    }

    public var consecutiveReuseRate: Double {
        Double(consecutiveReuseTouches) / Double(max(1, uniqueExpertTouches))
    }
}

public struct MLXPressActiveExpertTraceSnapshot: Sendable {
    public let totalCalls: Int
    public let totalTokens: Int
    public let totalRoutedSlots: Int
    public let totalUniqueExpertTouches: Int
    public let totalRepeatedWithinChunk: Int
    public let totalConsecutiveReuseTouches: Int
    public let layers: [MLXPressActiveExpertTraceLayer]

    public var isEmpty: Bool {
        totalCalls == 0
    }

    public var consecutiveReuseRate: Double {
        Double(totalConsecutiveReuseTouches) / Double(max(1, totalUniqueExpertTouches))
    }
}

public struct MLXPressActiveSliceResidencySnapshot: Sendable {
    public let budgetBytes: UInt64
    public let tensorBudgetBytes: UInt64
    public let bankBudgetBytes: UInt64
    public let residentBytes: UInt64
    public let tensorResidentBytes: UInt64
    public let sliceResidentBytes: UInt64
    public let bankResidentBytes: UInt64
    public let entries: Int
    public let tensorEntries: Int
    public let sliceEntries: Int
    public let bankEntries: Int
    public let hits: Int
    public let tensorHits: Int
    public let sliceHits: Int
    public let bankHits: Int
    public let misses: Int
    public let tensorMisses: Int
    public let sliceMisses: Int
    public let bankMisses: Int
    public let hitBytes: UInt64
    public let tensorHitBytes: UInt64
    public let sliceHitBytes: UInt64
    public let bankHitBytes: UInt64
    public let missBytes: UInt64
    public let tensorMissBytes: UInt64
    public let sliceMissBytes: UInt64
    public let bankMissBytes: UInt64
    public let stores: Int
    public let tensorStores: Int
    public let sliceStores: Int
    public let bankStores: Int
    public let evictions: Int
    public let tensorEvictions: Int
    public let sliceEvictions: Int
    public let bankEvictions: Int

    public var isActive: Bool {
        budgetBytes > 0 || hits > 0 || misses > 0 || stores > 0 || evictions > 0
    }

    public var requests: Int {
        hits + misses
    }

    public var hitRate: Double {
        Double(hits) / Double(max(1, requests))
    }

    public var byteHitRate: Double {
        Double(hitBytes) / Double(max(1, hitBytes + missBytes))
    }
}

public enum MLXPressActiveExpertTrace {
    public static var isEnabled: Bool {
        MLXPressActiveExpertTraceState.shared.isEnabled
    }

    public static func record(
        layerIdx: Int,
        expertIndices: [Int],
        tokenCount: Int,
        kSlots: Int
    ) {
        MLXPressActiveExpertTraceState.shared.record(
            layerIdx: layerIdx,
            expertIndices: expertIndices,
            tokenCount: tokenCount,
            kSlots: kSlots)
    }

    public static func dump(reason: String) {
        MLXPressActiveExpertTraceState.shared.dump(reason: reason)
    }

    public static func snapshot(
        maxLayers: Int? = nil,
        maxExpertsPerLayer: Int? = nil
    ) -> MLXPressActiveExpertTraceSnapshot {
        MLXPressActiveExpertTraceState.shared.snapshot(
            maxLayers: maxLayers,
            maxExpertsPerLayer: maxExpertsPerLayer)
    }
}

public enum MLXPressActiveSliceResidency {
    public static func snapshot() -> MLXPressActiveSliceResidencySnapshot {
        JANGTQStreamingExpertStore.shared.activeSliceResidencySnapshot()
    }
}

private final class MLXPressStreamingProfileState: @unchecked Sendable {
    private struct Stat {
        var count: Int = 0
        var seconds: Double = 0
        var bytes: Int = 0
    }

    static let shared = MLXPressStreamingProfileState()

    let isEnabled: Bool
    private let printEvery: Int
    private let maxRows: Int
    private let lock = NSLock()
    private var stats: [String: Stat] = [:]
    private var events: Int = 0

    private init() {
        let env = ProcessInfo.processInfo.environment
        let raw =
            env["MLXPRESS_STREAMING_PROFILE"]?
            .lowercased()
            ?? env["JANGPRESS_STREAMING_PROFILE"]?.lowercased()
            ?? "0"
        self.isEnabled = raw == "1" || raw == "true" || raw == "yes" || raw == "on"
        self.printEvery = max(
            0,
            Int(
                env["MLXPRESS_STREAMING_PROFILE_EVERY"]
                    ?? env["JANGPRESS_STREAMING_PROFILE_EVERY"]
                    ?? "120"
            ) ?? 120)
        self.maxRows = max(
            1,
            Int(
                env["MLXPRESS_STREAMING_PROFILE_TOP"]
                    ?? env["JANGPRESS_STREAMING_PROFILE_TOP"]
                    ?? "16"
            ) ?? 16)
    }

    func time<T>(_ name: String, bytes: Int = 0, _ body: () -> T) -> T {
        guard isEnabled else { return body() }
        let start = Date.timeIntervalSinceReferenceDate
        let value = body()
        record(name, seconds: Date.timeIntervalSinceReferenceDate - start, bytes: bytes)
        return value
    }

    func record(_ name: String, seconds: Double = 0, bytes: Int = 0) {
        guard isEnabled else { return }
        let output: String?
        lock.lock()
        var stat = stats[name] ?? Stat()
        stat.count += 1
        stat.seconds += seconds
        stat.bytes += bytes
        stats[name] = stat
        events += 1
        if printEvery > 0, events % printEvery == 0 {
            output = summaryLocked(reason: "events=\(events)")
        } else {
            output = nil
        }
        lock.unlock()
        if let output {
            FileHandle.standardError.write(Data(output.utf8))
        }
    }

    func dump(reason: String) {
        guard isEnabled else { return }
        lock.lock()
        let output = summaryLocked(reason: reason)
        lock.unlock()
        FileHandle.standardError.write(Data(output.utf8))
    }

    func snapshot(maxRows: Int? = nil) -> MLXPressStreamingProfileSnapshot {
        guard isEnabled else {
            return MLXPressStreamingProfileSnapshot(totalSeconds: 0, rows: [])
        }
        lock.lock()
        let snapshot = snapshotLocked(maxRows: maxRows ?? self.maxRows)
        lock.unlock()
        return snapshot
    }

    private func snapshotLocked(maxRows: Int) -> MLXPressStreamingProfileSnapshot {
        let totalSeconds = stats.values.reduce(0) { $0 + $1.seconds }
        let rows = stats
            .sorted {
                if $0.value.seconds == $1.value.seconds {
                    return $0.key < $1.key
                }
                return $0.value.seconds > $1.value.seconds
            }
            .prefix(max(1, maxRows))
            .map { key, stat in
                MLXPressStreamingProfileRow(
                    name: key,
                    count: stat.count,
                    seconds: stat.seconds,
                    bytes: stat.bytes)
            }
        return MLXPressStreamingProfileSnapshot(totalSeconds: totalSeconds, rows: rows)
    }

    private func summaryLocked(reason: String) -> String {
        let snapshot = snapshotLocked(maxRows: maxRows)
        let rows = snapshot.rows
            .map { row -> String in
                return String(
                    format: "%@ count=%d total=%.1fms avg=%.3fms bytes=%.2fMB bytes/call=%.2fMB bw=%.2fMB/s",
                    row.name,
                    row.count,
                    row.milliseconds,
                    row.averageMilliseconds,
                    row.bytesMB,
                    row.bytesPerCallMB,
                    row.bandwidthMBps)
            }
            .joined(separator: " | ")
        return String(
            format: "[MLXPressStreamingProfile] %@ total=%.1fms %@\n",
            reason, snapshot.totalSeconds * 1000, rows)
    }
}

private final class MLXPressActiveExpertTraceState: @unchecked Sendable {
    private struct LayerStat {
        var calls: Int = 0
        var tokenCount: Int = 0
        var routedSlots: Int = 0
        var uniqueExpertTouches: Int = 0
        var repeatedWithinChunk: Int = 0
        var consecutiveReuseTouches: Int = 0
        var expertCounts: [Int: Int] = [:]
        var previousUniqueExperts: Set<Int> = []
    }

    static let shared = MLXPressActiveExpertTraceState()

    let isEnabled: Bool
    private let maxLayers: Int
    private let maxExpertsPerLayer: Int
    private let lock = NSLock()
    private var layers: [Int: LayerStat] = [:]

    private init() {
        let env = ProcessInfo.processInfo.environment
        let raw =
            env["MLXPRESS_ACTIVE_EXPERT_TRACE"]?
            .lowercased()
            ?? env["JANGPRESS_ACTIVE_EXPERT_TRACE"]?.lowercased()
            ?? "0"
        self.isEnabled = raw == "1" || raw == "true" || raw == "yes" || raw == "on"
        self.maxLayers = max(
            1,
            Int(env["MLXPRESS_ACTIVE_EXPERT_TRACE_TOP_LAYERS"] ?? "16") ?? 16)
        self.maxExpertsPerLayer = max(
            1,
            Int(env["MLXPRESS_ACTIVE_EXPERT_TRACE_TOP_EXPERTS"] ?? "8") ?? 8)
    }

    func record(
        layerIdx: Int,
        expertIndices: [Int],
        tokenCount: Int,
        kSlots: Int
    ) {
        guard isEnabled, !expertIndices.isEmpty else { return }
        let unique = Set(expertIndices)
        lock.lock()
        var stat = layers[layerIdx] ?? LayerStat()
        stat.calls += 1
        stat.tokenCount += tokenCount
        stat.routedSlots += max(0, tokenCount * kSlots)
        stat.uniqueExpertTouches += unique.count
        stat.repeatedWithinChunk += max(0, expertIndices.count - unique.count)
        stat.consecutiveReuseTouches += stat.previousUniqueExperts.intersection(unique).count
        for expert in expertIndices {
            stat.expertCounts[expert, default: 0] += 1
        }
        stat.previousUniqueExperts = unique
        layers[layerIdx] = stat
        lock.unlock()
    }

    func dump(reason: String) {
        guard isEnabled else { return }
        let snapshot = snapshot(maxLayers: maxLayers, maxExpertsPerLayer: maxExpertsPerLayer)
        guard !snapshot.isEmpty else { return }
        let layerSummary = snapshot.layers.map { layer in
            let experts = layer.topExperts
                .map { "\($0.expertIdx):\($0.count)" }
                .joined(separator: ",")
            return "L\(layer.layerIdx)(calls=\(layer.calls),slots=\(layer.routedSlots),unique=\(layer.uniqueExpertTouches),reuse=\(layer.consecutiveReuseTouches),top=[\(experts)])"
        }.joined(separator: " | ")
        let output =
            "[MLXPressActiveExpertTrace] \(reason) calls=\(snapshot.totalCalls) tokens=\(snapshot.totalTokens) slots=\(snapshot.totalRoutedSlots) uniqueTouches=\(snapshot.totalUniqueExpertTouches) consecutiveReuse=\(snapshot.totalConsecutiveReuseTouches) reuseRate=\(String(format: "%.3f", snapshot.consecutiveReuseRate)) \(layerSummary)\n"
        FileHandle.standardError.write(Data(output.utf8))
    }

    func snapshot(
        maxLayers: Int? = nil,
        maxExpertsPerLayer: Int? = nil
    ) -> MLXPressActiveExpertTraceSnapshot {
        guard isEnabled else {
            return MLXPressActiveExpertTraceSnapshot(
                totalCalls: 0,
                totalTokens: 0,
                totalRoutedSlots: 0,
                totalUniqueExpertTouches: 0,
                totalRepeatedWithinChunk: 0,
                totalConsecutiveReuseTouches: 0,
                layers: [])
        }
        lock.lock()
        let snapshot = snapshotLocked(
            maxLayers: maxLayers ?? self.maxLayers,
            maxExpertsPerLayer: maxExpertsPerLayer ?? self.maxExpertsPerLayer)
        lock.unlock()
        return snapshot
    }

    private func snapshotLocked(
        maxLayers: Int,
        maxExpertsPerLayer: Int
    ) -> MLXPressActiveExpertTraceSnapshot {
        let totalCalls = layers.values.reduce(0) { $0 + $1.calls }
        let totalTokens = layers.values.reduce(0) { $0 + $1.tokenCount }
        let totalRoutedSlots = layers.values.reduce(0) { $0 + $1.routedSlots }
        let totalUniqueExpertTouches = layers.values.reduce(0) { $0 + $1.uniqueExpertTouches }
        let totalRepeatedWithinChunk = layers.values.reduce(0) { $0 + $1.repeatedWithinChunk }
        let totalConsecutiveReuseTouches = layers.values.reduce(0) {
            $0 + $1.consecutiveReuseTouches
        }
        let layerRows = layers
            .sorted {
                if $0.value.routedSlots == $1.value.routedSlots {
                    return $0.key < $1.key
                }
                return $0.value.routedSlots > $1.value.routedSlots
            }
            .prefix(max(1, maxLayers))
            .map { layerIdx, stat in
                let topExperts = stat.expertCounts
                    .sorted {
                        if $0.value == $1.value {
                            return $0.key < $1.key
                        }
                        return $0.value > $1.value
                    }
                    .prefix(max(1, maxExpertsPerLayer))
                    .map { expertIdx, count in
                        MLXPressActiveExpertTraceExpert(expertIdx: expertIdx, count: count)
                    }
                return MLXPressActiveExpertTraceLayer(
                    layerIdx: layerIdx,
                    calls: stat.calls,
                    tokenCount: stat.tokenCount,
                    routedSlots: stat.routedSlots,
                    uniqueExpertTouches: stat.uniqueExpertTouches,
                    repeatedWithinChunk: stat.repeatedWithinChunk,
                    consecutiveReuseTouches: stat.consecutiveReuseTouches,
                    topExperts: topExperts)
            }
        return MLXPressActiveExpertTraceSnapshot(
            totalCalls: totalCalls,
            totalTokens: totalTokens,
            totalRoutedSlots: totalRoutedSlots,
            totalUniqueExpertTouches: totalUniqueExpertTouches,
            totalRepeatedWithinChunk: totalRepeatedWithinChunk,
            totalConsecutiveReuseTouches: totalConsecutiveReuseTouches,
            layers: layerRows)
    }
}

private func mlXPressStreamingTokenChunkSize() -> Int {
    let env = ProcessInfo.processInfo.environment
    let raw =
        env["MLXPRESS_STREAMING_TOKEN_CHUNK_SIZE"]
        ?? env["MLXPRESS_STREAMING_TOKEN_CHUNK"]
        ?? env["VMLX_JANGTQ_PREFILL_STEP"]
        ?? "16"
    return max(1, Int(raw) ?? 16)
}

private func mlXPressStreamingReduceTokenChunkSize() -> Int {
    let env = ProcessInfo.processInfo.environment
    let raw =
        env["MLXPRESS_STREAMING_REDUCE_TOKEN_CHUNK_SIZE"]
        ?? env["MLXPRESS_STREAMING_REDUCE_TOKEN_CHUNK"]
        ?? "4"
    return max(1, Int(raw) ?? 4)
}

private func mlXPressStreamingFastReduceMaxTokens() -> Int {
    let env = ProcessInfo.processInfo.environment
    let raw =
        env["MLXPRESS_STREAMING_FAST_REDUCE_MAX_TOKENS"]
        ?? env["MLXPRESS_STREAMING_FAST_REDUCE_TOKEN_LIMIT"]
        ?? "4"
    return max(0, Int(raw) ?? 4)
}

private func mlXPressStreamingEvaluateEachLayerEnabled() -> Bool {
    ProcessInfo.processInfo.environment["MLXPRESS_STREAMING_EVAL_EACH_LAYER"] != "0"
}

private func mlXPressStreamingEvalLayerStride() -> Int {
    let env = ProcessInfo.processInfo.environment
    let raw =
        env["MLXPRESS_STREAMING_EVAL_LAYER_STRIDE"]
        ?? env["JANGPRESS_STREAMING_EVAL_LAYER_STRIDE"]
        ?? "1"
    return max(1, Int(raw) ?? 1)
}

private func mlXPressStreamingShouldEvaluateLayer(_ layerIdx: Int) -> Bool {
    guard mlXPressStreamingEvaluateEachLayerEnabled() else { return false }
    let stride = mlXPressStreamingEvalLayerStride()
    return stride <= 1 || ((layerIdx + 1) % stride == 0)
}

private func mlXPressStreamingMegabyteLimitBytes(_ names: [String]) -> Int? {
    let env = ProcessInfo.processInfo.environment
    for name in names {
        guard let raw = env[name], let mb = Double(raw) else { continue }
        return Int(max(0, mb) * 1024 * 1024)
    }
    return nil
}

private func mlXPressStreamingBoolEnv(_ names: [String], defaultValue: Bool) -> Bool {
    let env = ProcessInfo.processInfo.environment
    for name in names {
        guard let raw = env[name]?.lowercased() else { continue }
        if raw == "1" || raw == "true" || raw == "yes" || raw == "on" {
            return true
        }
        if raw == "0" || raw == "false" || raw == "no" || raw == "off" {
            return false
        }
    }
    return defaultValue
}

private final class MLXPressStreamingAllocatorEvalScheduler: @unchecked Sendable {
    static let shared = MLXPressStreamingAllocatorEvalScheduler()

    private let peakLimitBytes: Int?
    private let activeLimitBytes: Int?
    private let resetPeakAfterEval: Bool
    private let lock = NSLock()
    private var initialPeakResetDone = false

    private init() {
        self.peakLimitBytes = mlXPressStreamingMegabyteLimitBytes([
            "MLXPRESS_STREAMING_EVAL_MLX_PEAK_MB",
            "JANGPRESS_STREAMING_EVAL_MLX_PEAK_MB",
        ])
        self.activeLimitBytes = mlXPressStreamingMegabyteLimitBytes([
            "MLXPRESS_STREAMING_EVAL_MLX_ACTIVE_MB",
            "JANGPRESS_STREAMING_EVAL_MLX_ACTIVE_MB",
        ])
        self.resetPeakAfterEval = mlXPressStreamingBoolEnv(
            [
                "MLXPRESS_STREAMING_EVAL_RESET_PEAK",
                "JANGPRESS_STREAMING_EVAL_RESET_PEAK",
            ],
            defaultValue: true)
    }

    private var isEnabled: Bool {
        peakLimitBytes != nil || activeLimitBytes != nil
    }

    func shouldMaterialize(layerIdx: Int, phase: String) -> Bool {
        guard isEnabled else { return false }
        resetPeakBeforeFirstPressureCheck()

        let activeBytes = max(0, MLX.Memory.activeMemory)
        let peakBytes = max(0, MLX.Memory.peakMemory)
        let activeOverLimit = activeLimitBytes.map { activeBytes >= $0 } ?? false
        let peakOverLimit = peakLimitBytes.map { peakBytes >= $0 } ?? false
        guard activeOverLimit || peakOverLimit else { return false }

        MLXPressStreamingProfile.record(
            "scheduler.allocator_eval",
            bytes: max(activeBytes, peakBytes))
        MLXPressStreamingProfile.record("scheduler.allocator_eval.layer_\(layerIdx).\(phase)")
        return true
    }

    func didMaterialize(force: Bool, layerIdx: Int, phase: String) {
        guard isEnabled else { return }
        let activeBytes = max(0, MLX.Memory.activeMemory)
        let peakBytes = max(0, MLX.Memory.peakMemory)
        MLXPressStreamingProfile.record(
            force ? "scheduler.forced_eval" : "scheduler.allocator_eval_completed",
            bytes: max(activeBytes, peakBytes))
        MLXPressStreamingProfile.record("scheduler.eval.layer_\(layerIdx).\(phase)")
        if resetPeakAfterEval {
            MLX.Memory.peakMemory = 0
            MLXPressStreamingProfile.record("scheduler.allocator_peak_reset")
        }
    }

    private func resetPeakBeforeFirstPressureCheck() {
        guard resetPeakAfterEval else { return }
        lock.lock()
        let shouldReset = !initialPeakResetDone
        if shouldReset {
            initialPeakResetDone = true
        }
        lock.unlock()

        guard shouldReset else { return }
        MLX.Memory.peakMemory = 0
        MLXPressStreamingProfile.record("scheduler.allocator_peak_reset")
    }
}

private func mlXPressStreamingMaterializeIfNeeded(
    _ array: MLXArray,
    force: Bool,
    layerIdx: Int,
    phase: String,
    profileName: String,
    releaseMachColdTiles: Bool = false
) {
    let pressureEval = force
        ? false
        : MLXPressStreamingAllocatorEvalScheduler.shared.shouldMaterialize(
            layerIdx: layerIdx,
            phase: phase)
    guard force || pressureEval else { return }
    MLXPressStreamingProfile.time(profileName) {
        MLX.eval(array)
        MLX.Memory.clearCache()
    }
    MLXPressStreamingAllocatorEvalScheduler.shared.didMaterialize(
        force: force,
        layerIdx: layerIdx,
        phase: phase)
    if releaseMachColdTiles {
        JANGTQStreamingExpertStore.shared.releaseMachColdTiles()
    }
}

private func mlXPressStreamingActiveOffsetFileURLs(
    _ spanGroups: [[StreamingOffsetSpan]],
    activeExperts: Set<Int>?
) -> Set<URL>? {
    guard let activeExperts, !activeExperts.isEmpty else { return nil }
    var urls = Set<URL>()
    for spans in spanGroups {
        for span in spans where !span.presentExperts.isDisjoint(with: activeExperts) {
            urls.insert(span.fileURL)
        }
    }
    return urls.isEmpty ? nil : urls
}

private func mlXPressStreamingFilterOffsetSpans(
    _ spans: [StreamingOffsetSpan],
    activeFileURLs: Set<URL>?
) -> [StreamingOffsetSpan] {
    guard let activeFileURLs, !activeFileURLs.isEmpty else { return spans }
    let filtered = spans.filter { activeFileURLs.contains($0.fileURL) }
    let skippedBytes = spans.reduce(0) { total, span in
        activeFileURLs.contains(span.fileURL) ? total : total + span.byteCount
    }
    if skippedBytes > 0 {
        MLXPressStreamingProfile.record("tensor.offset_span_active_filter", bytes: skippedBytes)
    }
    return filtered
}

private func mlXPressStreamingExpertCacheBudgetBytes() -> Int {
    let env = ProcessInfo.processInfo.environment
    if let raw =
        env["MLXPRESS_STREAMING_EXPERT_CACHE_MB"]
        ?? env["JANGPRESS_STREAMING_EXPERT_CACHE_MB"]
    {
        return max(0, Int(raw) ?? 0) * 1024 * 1024
    }
    return 0
}

private func mlXPressStreamingBankCacheBudgetBytes() -> Int {
    let env = ProcessInfo.processInfo.environment
    if let raw =
        env["MLXPRESS_STREAMING_BANK_CACHE_MB"]
        ?? env["JANGPRESS_STREAMING_BANK_CACHE_MB"]
    {
        return max(0, Int(raw) ?? 0) * 1024 * 1024
    }
    return 0
}

private func mlXPressStreamingOffsetSpanCacheBudgetBytes() -> Int {
    let env = ProcessInfo.processInfo.environment
    if let raw =
        env["MLXPRESS_STREAMING_OFFSET_SPAN_CACHE_MB"]
        ?? env["JANGPRESS_STREAMING_OFFSET_SPAN_CACHE_MB"]
    {
        return max(0, Int(raw) ?? 0) * 1024 * 1024
    }
    return 0
}

private func mlXPressStreamingBankLoadEnabled() -> Bool {
    let env = ProcessInfo.processInfo.environment
    let raw =
        env["MLXPRESS_STREAMING_BANK_LOAD"]?
        .lowercased()
        ?? env["JANGPRESS_STREAMING_BANK_LOAD"]?.lowercased()
        ?? "1"
    return raw != "0" && raw != "false" && raw != "no" && raw != "off"
}

private func mlXPressStreamingDirectStackedEnabled() -> Bool {
    let env = ProcessInfo.processInfo.environment
    let raw =
        env["MLXPRESS_STREAMING_DIRECT_STACKED"]?
        .lowercased()
        ?? env["JANGPRESS_STREAMING_DIRECT_STACKED"]?.lowercased()
        ?? "0"
    return raw == "1" || raw == "true" || raw == "yes" || raw == "on"
}

private func mlXPressStreamingOffsetKernelsEnabled() -> Bool {
    let env = ProcessInfo.processInfo.environment
    let raw =
        env["MLXPRESS_STREAMING_OFFSET_KERNELS"]?
        .lowercased()
        ?? env["JANGPRESS_STREAMING_OFFSET_KERNELS"]?.lowercased()
        ?? "0"
    return raw == "1" || raw == "true" || raw == "yes" || raw == "on"
}

private func mlXPressStreamingOffsetActiveShardFilterEnabled() -> Bool {
    let env = ProcessInfo.processInfo.environment
    let raw =
        env["MLXPRESS_STREAMING_OFFSET_ACTIVE_SHARD_FILTER"]?
        .lowercased()
        ?? env["JANGPRESS_STREAMING_OFFSET_ACTIVE_SHARD_FILTER"]?.lowercased()
        ?? "0"
    return raw == "1" || raw == "true" || raw == "yes" || raw == "on"
}

private func mlXPressStreamingOffsetActiveWindowCoalesceBytes() -> UInt64? {
    let env = ProcessInfo.processInfo.environment
    guard let raw =
        env["MLXPRESS_STREAMING_OFFSET_ACTIVE_WINDOW_COALESCE_MB"]
        ?? env["JANGPRESS_STREAMING_OFFSET_ACTIVE_WINDOW_COALESCE_MB"]
    else { return nil }
    let normalized = raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
    if normalized.isEmpty || normalized == "0" || normalized == "false"
        || normalized == "off" || normalized == "no" || normalized == "none"
    {
        return nil
    }
    guard let mb = Double(normalized), mb > 0 else { return nil }
    let bytes = mb * 1024.0 * 1024.0
    guard bytes.isFinite, bytes > 0 else { return nil }
    return UInt64(min(bytes, Double(UInt64.max)).rounded(.down))
}

private func mlXPressStreamingScoredDownKernelsEnabled() -> Bool {
    let env = ProcessInfo.processInfo.environment
    let raw =
        env["MLXPRESS_STREAMING_SCORED_DOWN_KERNELS"]?
        .lowercased()
        ?? env["JANGPRESS_STREAMING_SCORED_DOWN_KERNELS"]?.lowercased()
        ?? "0"
    return raw == "1" || raw == "true" || raw == "yes" || raw == "on"
}

private func mlXPressStreamingSourceOrderReadsEnabled() -> Bool {
    let env = ProcessInfo.processInfo.environment
    let raw =
        env["MLXPRESS_STREAMING_SOURCE_ORDER_READS"]?
        .lowercased()
        ?? env["JANGPRESS_STREAMING_SOURCE_ORDER_READS"]?.lowercased()
        ?? "0"
    return raw == "1" || raw == "true" || raw == "yes" || raw == "on"
}

private func mlXPressStreamingFNoCacheOverride() -> Bool? {
    let env = ProcessInfo.processInfo.environment
    guard
        let raw = (
            env["MLXPRESS_STREAMING_F_NOCACHE"]
                ?? env["JANGPRESS_STREAMING_F_NOCACHE"]
        )?.lowercased()
    else { return nil }
    if raw == "1" || raw == "true" || raw == "yes" || raw == "on" {
        return true
    }
    if raw == "0" || raw == "false" || raw == "no" || raw == "off" {
        return false
    }
    return nil
}

private func mlXPressStreamingNoCacheThresholdBytes() -> UInt64 {
    let env = ProcessInfo.processInfo.environment
    if let raw = env["MLXPRESS_STREAMING_F_NOCACHE_THRESHOLD_BYTES"],
       let bytes = UInt64(raw)
    {
        return bytes
    }
    if let raw = env["MLXPRESS_STREAMING_F_NOCACHE_THRESHOLD_GB"],
       let gb = Double(raw)
    {
        return UInt64(max(0, gb) * 1024 * 1024 * 1024)
    }
    return UInt64(Double(ProcessInfo.processInfo.physicalMemory) * 0.70)
}

private func mlXPressStreamingUseFNoCache(for index: JANGTQStreamingExpertIndex?) -> Bool {
    if let override = mlXPressStreamingFNoCacheOverride() {
        return override
    }
    guard let index else { return true }
    return index.totalTensorBytes > mlXPressStreamingNoCacheThresholdBytes()
}

private func mlXPressStreamingMachActiveTensorsEnabled() -> Bool {
    let env = ProcessInfo.processInfo.environment
    let raw =
        env["MLXPRESS_STREAMING_MACH_ACTIVE_TENSORS"]?
        .lowercased()
        ?? env["JANGPRESS_STREAMING_MACH_ACTIVE_TENSORS"]?.lowercased()
        ?? "0"
    return raw == "1" || raw == "true" || raw == "yes" || raw == "on"
}

private func mlXPressStreamingMachOffsetSpansEnabled() -> Bool {
    let env = ProcessInfo.processInfo.environment
    let raw =
        env["MLXPRESS_STREAMING_MACH_OFFSET_SPANS"]?
        .lowercased()
        ?? env["JANGPRESS_STREAMING_MACH_OFFSET_SPANS"]?.lowercased()
        ?? "0"
    return raw == "1" || raw == "true" || raw == "yes" || raw == "on"
}

private func mlXPressStreamingMachFullOffsetSpansEnabled() -> Bool {
    let env = ProcessInfo.processInfo.environment
    let raw =
        env["MLXPRESS_STREAMING_MACH_FULL_OFFSET_SPANS"]?
        .lowercased()
        ?? env["JANGPRESS_STREAMING_MACH_FULL_OFFSET_SPANS"]?.lowercased()
        ?? "0"
    return raw == "1" || raw == "true" || raw == "yes" || raw == "on"
}

private func mlXPressStreamingMachPrewarmFullOffsetSpansEnabled() -> Bool {
    let env = ProcessInfo.processInfo.environment
    let raw =
        env["MLXPRESS_STREAMING_MACH_PREWARM_FULL_OFFSET_SPANS"]?
        .lowercased()
        ?? env["JANGPRESS_STREAMING_MACH_PREWARM_FULL_OFFSET_SPANS"]?.lowercased()
        ?? "0"
    return raw == "1" || raw == "true" || raw == "yes" || raw == "on"
}

private func mlXPressStreamingMachFullOffsetLazyEnabled() -> Bool {
    let env = ProcessInfo.processInfo.environment
    let raw =
        env["MLXPRESS_STREAMING_MACH_FULL_OFFSET_LAZY"]?
        .lowercased()
        ?? env["JANGPRESS_STREAMING_MACH_FULL_OFFSET_LAZY"]?.lowercased()
        ?? "0"
    return raw == "1" || raw == "true" || raw == "yes" || raw == "on"
}

private func mlXPressStreamingMachPrewarmMaxLayers() -> Int? {
    let env = ProcessInfo.processInfo.environment
    guard let raw =
        env["MLXPRESS_STREAMING_MACH_PREWARM_MAX_LAYERS"]
        ?? env["JANGPRESS_STREAMING_MACH_PREWARM_MAX_LAYERS"]
    else { return nil }
    let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
    if trimmed.isEmpty || trimmed == "all" || trimmed == "none" || trimmed == "0" {
        return nil
    }
    guard let value = Int(trimmed), value > 0 else { return nil }
    return value
}

private func mlXPressStreamingMachPrewarmMaxBytes() -> UInt64? {
    let env = ProcessInfo.processInfo.environment
    if let raw =
        env["MLXPRESS_STREAMING_MACH_PREWARM_MAX_GB"]
        ?? env["JANGPRESS_STREAMING_MACH_PREWARM_MAX_GB"],
        let gb = Double(raw),
        gb > 0
    {
        return UInt64((gb * 1024.0 * 1024.0 * 1024.0).rounded(.down))
    }
    if let raw =
        env["MLXPRESS_STREAMING_MACH_PREWARM_MAX_MB"]
        ?? env["JANGPRESS_STREAMING_MACH_PREWARM_MAX_MB"],
        let mb = Double(raw),
        mb > 0
    {
        return UInt64((mb * 1024.0 * 1024.0).rounded(.down))
    }
    return nil
}

private func mlXPressStreamingMachOffsetSpanBudgetBytes() -> Int {
    let env = ProcessInfo.processInfo.environment
    if let raw =
        env["MLXPRESS_STREAMING_MACH_OFFSET_SPAN_BUDGET_MB"]
        ?? env["JANGPRESS_STREAMING_MACH_OFFSET_SPAN_BUDGET_MB"]
    {
        return max(0, Int(raw) ?? 0) * 1024 * 1024
    }
    return 0
}

private func mlXPressMachCompressPercent() -> Int {
    let env = ProcessInfo.processInfo.environment
    let raw =
        env["MLXPRESS_MACH_COMPRESS_PCT"]
        ?? env["JANGPRESS_MACH_COMPRESS_PCT"]
        ?? env["MLXPRESS"]
        ?? env["JANGPRESS"]
        ?? "70"
    guard let pct = Int(raw) else { return 70 }
    return max(0, min(95, pct))
}

private enum StreamingProjection: String, CaseIterable {
    case gate = "gate_proj"
    case up = "up_proj"
    case down = "down_proj"
}

private enum StreamingSuffix: String, CaseIterable {
    case packed = "tq_packed"
    case norms = "tq_norms"
}

private struct StreamingTensorRef {
    var name: String
    var fileURL: URL
    var offset: UInt64
    var byteCount: Int
    var dtype: String
    var shape: [Int]
}

private struct StreamingExpertRef {
    var tensors: [StreamingProjection: [StreamingSuffix: StreamingTensorRef]]
}

private struct StreamingLayerRef {
    var experts: [Int: StreamingExpertRef]
    var stacked: [StreamingProjection: [StreamingSuffix: StreamingTensorRef]]
}

private struct StreamingTensorCacheKey: Hashable {
    var layerIdx: Int
    var expertIdx: Int
    var projection: StreamingProjection
    var suffix: StreamingSuffix
}

private struct StreamingStackedTensorCacheKey: Hashable {
    var layerIdx: Int
    var projection: StreamingProjection
    var suffix: StreamingSuffix
}

private struct StreamingActiveBankCacheKey: Hashable {
    var layerIdx: Int
    var projection: StreamingProjection
    var suffix: StreamingSuffix
    var experts: [Int]
}

private struct StreamingOffsetSpanCacheKey: Hashable {
    var fileURL: URL
    var offset: UInt64
    var byteCount: Int
    var dtype: String
}

private struct StreamingTensorCacheEntry {
    var array: MLXArray
    var byteCount: Int
    var lastUse: UInt64
}

private struct StreamingActiveBankCacheEntry {
    var array: MLXArray
    var byteCount: Int
    var lastUse: UInt64
}

private struct StreamingOffsetSpanCacheEntry {
    var span: StreamingOffsetSpan
    var byteCount: Int
    var lastUse: UInt64
}

private struct StreamingOffsetSpan {
    var fileURL: URL
    var array: MLXArray
    var offsets: MLXArray
    var byteCount: Int
    var storageLayout: String
    var presentExperts: Set<Int>
}

private struct StreamingGateUpOffsetSpanGroup {
    var gatePacked: StreamingOffsetSpan
    var gateNorms: StreamingOffsetSpan
    var upPacked: StreamingOffsetSpan
    var upNorms: StreamingOffsetSpan
}

private struct StreamingDownOffsetSpanGroup {
    var packed: StreamingOffsetSpan
    var norms: StreamingOffsetSpan
}

private struct StreamingGateUpOffsetSpanGroupKey: Hashable {
    var gatePacked: URL
    var gateNorms: URL
    var upPacked: URL
    var upNorms: URL
    var experts: [Int]
}

private struct StreamingDownOffsetSpanGroupKey: Hashable {
    var packed: URL
    var norms: URL
    var experts: [Int]
}

private func mlXPressStreamingOffsetSpanMap(
    _ spans: [StreamingOffsetSpan]
) -> [Int: StreamingOffsetSpan] {
    var mapped: [Int: StreamingOffsetSpan] = [:]
    for span in spans {
        for expert in span.presentExperts {
            mapped[expert] = span
        }
    }
    return mapped
}

private func mlXPressStreamingGateUpOffsetSpanGroups(
    gatePacked: [StreamingOffsetSpan],
    gateNorms: [StreamingOffsetSpan],
    upPacked: [StreamingOffsetSpan],
    upNorms: [StreamingOffsetSpan]
) -> [StreamingGateUpOffsetSpanGroup] {
    let gatePackedByExpert = mlXPressStreamingOffsetSpanMap(gatePacked)
    let gateNormsByExpert = mlXPressStreamingOffsetSpanMap(gateNorms)
    let upPackedByExpert = mlXPressStreamingOffsetSpanMap(upPacked)
    let upNormsByExpert = mlXPressStreamingOffsetSpanMap(upNorms)
    var groups: [StreamingGateUpOffsetSpanGroupKey: StreamingGateUpOffsetSpanGroup] = [:]
    for expert in gatePackedByExpert.keys.sorted() {
        guard
            let gatePackedSpan = gatePackedByExpert[expert],
            let gateNormsSpan = gateNormsByExpert[expert],
            let upPackedSpan = upPackedByExpert[expert],
            let upNormsSpan = upNormsByExpert[expert]
        else { continue }
        let experts = gatePackedSpan.presentExperts
            .intersection(gateNormsSpan.presentExperts)
            .intersection(upPackedSpan.presentExperts)
            .intersection(upNormsSpan.presentExperts)
            .sorted()
        guard !experts.isEmpty else { continue }
        let key = StreamingGateUpOffsetSpanGroupKey(
            gatePacked: gatePackedSpan.fileURL,
            gateNorms: gateNormsSpan.fileURL,
            upPacked: upPackedSpan.fileURL,
            upNorms: upNormsSpan.fileURL,
            experts: experts)
        groups[key] = StreamingGateUpOffsetSpanGroup(
            gatePacked: gatePackedSpan,
            gateNorms: gateNormsSpan,
            upPacked: upPackedSpan,
            upNorms: upNormsSpan)
    }
    return groups.keys
        .sorted {
            if $0.gatePacked.path != $1.gatePacked.path {
                return $0.gatePacked.path < $1.gatePacked.path
            }
            if $0.gateNorms.path != $1.gateNorms.path {
                return $0.gateNorms.path < $1.gateNorms.path
            }
            if $0.upPacked.path != $1.upPacked.path {
                return $0.upPacked.path < $1.upPacked.path
            }
            if $0.upNorms.path != $1.upNorms.path {
                return $0.upNorms.path < $1.upNorms.path
            }
            return $0.experts.lexicographicallyPrecedes($1.experts)
        }
        .compactMap { groups[$0] }
}

private func mlXPressStreamingDownOffsetSpanGroups(
    packed: [StreamingOffsetSpan],
    norms: [StreamingOffsetSpan]
) -> [StreamingDownOffsetSpanGroup] {
    let packedByExpert = mlXPressStreamingOffsetSpanMap(packed)
    let normsByExpert = mlXPressStreamingOffsetSpanMap(norms)
    var groups: [StreamingDownOffsetSpanGroupKey: StreamingDownOffsetSpanGroup] = [:]
    for expert in packedByExpert.keys.sorted() {
        guard
            let packedSpan = packedByExpert[expert],
            let normsSpan = normsByExpert[expert]
        else { continue }
        let experts = packedSpan.presentExperts
            .intersection(normsSpan.presentExperts)
            .sorted()
        guard !experts.isEmpty else { continue }
        let key = StreamingDownOffsetSpanGroupKey(
            packed: packedSpan.fileURL,
            norms: normsSpan.fileURL,
            experts: experts)
        groups[key] = StreamingDownOffsetSpanGroup(
            packed: packedSpan,
            norms: normsSpan)
    }
    return groups.keys
        .sorted {
            if $0.packed.path != $1.packed.path {
                return $0.packed.path < $1.packed.path
            }
            if $0.norms.path != $1.norms.path {
                return $0.norms.path < $1.norms.path
            }
            return $0.experts.lexicographicallyPrecedes($1.experts)
        }
        .compactMap { groups[$0] }
}

private struct StreamingSliceCacheEntry {
    var data: Data
    var byteCount: Int
    var lastUse: UInt64
}

private struct StreamingResidentTensorEntry {
    var array: MLXArray
    var byteCount: Int
}

private struct StreamingMachTensorEntry {
    var byteCount: Int
    var lastUse: UInt64
    var isOffsetSpan: Bool
}

private struct StreamingStackedSliceReadResult {
    var data: Data
    var fileReadBytes: Int
    var cacheHitBytes: Int

    var assembledBytes: Int {
        data.count
    }
}

private struct StreamingStackedSliceReadTarget {
    var expertIdx: Int
    var slots: [Int]
}

private struct StreamingStackedSliceReadPlan {
    var firstExpertIdx: Int
    var expertCount: Int
    var targets: [StreamingStackedSliceReadTarget]
}

private func makeStackedSliceReadPlan(expertIndices: [Int]) -> [StreamingStackedSliceReadPlan] {
    guard !expertIndices.isEmpty else { return [] }

    var slotsByExpert: [Int: [Int]] = [:]
    for (slot, expert) in expertIndices.enumerated() {
        slotsByExpert[expert, default: []].append(slot)
    }

    let sortedExperts = slotsByExpert.keys.sorted()
    var plans: [StreamingStackedSliceReadPlan] = []
    plans.reserveCapacity(sortedExperts.count)

    var first = sortedExperts[0]
    var previous = first
    var targets: [StreamingStackedSliceReadTarget] = [
        StreamingStackedSliceReadTarget(expertIdx: first, slots: slotsByExpert[first] ?? [])
    ]

    func flush() {
        plans.append(
            StreamingStackedSliceReadPlan(
                firstExpertIdx: first,
                expertCount: previous - first + 1,
                targets: targets))
    }

    for expert in sortedExperts.dropFirst() {
        if expert == previous + 1 {
            targets.append(
                StreamingStackedSliceReadTarget(
                    expertIdx: expert,
                    slots: slotsByExpert[expert] ?? []))
            previous = expert
            continue
        }

        flush()
        first = expert
        previous = expert
        targets = [
            StreamingStackedSliceReadTarget(
                expertIdx: expert,
                slots: slotsByExpert[expert] ?? [])
        ]
    }
    flush()
    return plans
}

private final class JANGTQStreamingExpertIndex: @unchecked Sendable {
    let modelDirectory: URL
    let layers: [Int: StreamingLayerRef]
    let totalTensorBytes: UInt64

    init(modelDirectory: URL, layers: [Int: StreamingLayerRef]) {
        self.modelDirectory = modelDirectory
        self.layers = layers
        self.totalTensorBytes = Self.computeTotalTensorBytes(layers)
    }

    static func build(modelDirectory: URL) throws -> JANGTQStreamingExpertIndex {
        let fm = FileManager.default
        let files = try fm.contentsOfDirectory(
            at: modelDirectory,
            includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles]
        )
        .filter { $0.pathExtension == "safetensors" }
        .filter { $0.lastPathComponent != "jangtq_runtime.safetensors" }
        .sorted { $0.lastPathComponent < $1.lastPathComponent }

        var layers: [Int: StreamingLayerRef] = [:]
        for file in files {
            let header = try readSafetensorsHeader(file)
            for (key, value) in header.tensors {
                guard let dtype = value["dtype"] as? String,
                    let shape = value["shape"] as? [Int],
                    let offsets = value["data_offsets"] as? [UInt64],
                    offsets.count == 2,
                    offsets[1] >= offsets[0]
                else { continue }

                let ref = StreamingTensorRef(
                    name: key,
                    fileURL: file,
                    offset: header.dataBase + offsets[0],
                    byteCount: Int(offsets[1] - offsets[0]),
                    dtype: dtype,
                    shape: shape)

                if let match = matchPerExpertTQKey(key) {
                    var layer = layers[match.layer] ?? StreamingLayerRef(experts: [:], stacked: [:])
                    var expert = layer.experts[match.expert] ?? StreamingExpertRef(tensors: [:])
                    var projection = expert.tensors[match.projection] ?? [:]
                    projection[match.suffix] = ref
                    expert.tensors[match.projection] = projection
                    layer.experts[match.expert] = expert
                    layers[match.layer] = layer
                    continue
                }

                if let match = matchStackedTQKey(key) {
                    var layer = layers[match.layer] ?? StreamingLayerRef(experts: [:], stacked: [:])
                    var projection = layer.stacked[match.projection] ?? [:]
                    projection[match.suffix] = ref
                    layer.stacked[match.projection] = projection
                    layers[match.layer] = layer
                }
            }
        }
        return JANGTQStreamingExpertIndex(modelDirectory: modelDirectory, layers: layers)
    }

    func stackedOffsetDescriptor(
        layerIdx: Int,
        projection: StreamingProjection,
        suffix: StreamingSuffix
    ) -> JANGTQStackedOffsetDescriptor? {
        guard let layer = layers[layerIdx] else { return nil }

        if let ref = layer.stacked[projection]?[suffix] {
            guard let expertCount = ref.shape.first,
                expertCount > 0,
                ref.byteCount % expertCount == 0
            else { return nil }

            let expertByteCount = ref.byteCount / expertCount
            return JANGTQStackedOffsetDescriptor(
                layerIdx: layerIdx,
                projectionName: projection.rawValue,
                suffixName: suffix.rawValue,
                fileURL: ref.fileURL,
                spanOffset: ref.offset,
                spanByteCount: ref.byteCount,
                expertByteCount: expertByteCount,
                expertByteOffsets: (0 ..< expertCount).map {
                    UInt64($0 * expertByteCount)
                },
                logicalShape: ref.shape,
                dtype: ref.dtype,
                storageLayout: "stacked-contiguous")
        }

        let entries = layer.experts.compactMap { expertIdx, expert -> (Int, StreamingTensorRef)? in
            guard let ref = expert.tensors[projection]?[suffix] else { return nil }
            return (expertIdx, ref)
        }
        .sorted { $0.0 < $1.0 }
        guard !entries.isEmpty else { return nil }

        for (expected, entry) in entries.enumerated() where entry.0 != expected {
            return nil
        }

        guard let first = entries.first?.1 else { return nil }
        for (_, ref) in entries {
            guard ref.fileURL == first.fileURL,
                ref.dtype == first.dtype,
                ref.shape == first.shape,
                ref.byteCount == first.byteCount
            else { return nil }
        }

        let spanStart = entries.reduce(UInt64.max) { min($0, $1.1.offset) }
        let spanEnd = entries.reduce(UInt64.min) {
            max($0, $1.1.offset + UInt64(max(0, $1.1.byteCount)))
        }
        guard spanStart != UInt64.max,
            spanEnd >= spanStart,
            spanEnd - spanStart <= UInt64(Int.max)
        else { return nil }

        return JANGTQStackedOffsetDescriptor(
            layerIdx: layerIdx,
            projectionName: projection.rawValue,
            suffixName: suffix.rawValue,
            fileURL: first.fileURL,
            spanOffset: spanStart,
            spanByteCount: Int(spanEnd - spanStart),
            expertByteCount: first.byteCount,
            expertByteOffsets: entries.map { $0.1.offset - spanStart },
            logicalShape: [entries.count] + first.shape,
            dtype: first.dtype,
            storageLayout: "expert-major-single-file-offsets")
    }

    func stackedOffsetDescriptors(
        layerIdx: Int,
        projection: StreamingProjection,
        suffix: StreamingSuffix
    ) -> [JANGTQStackedOffsetDescriptor] {
        if let single = stackedOffsetDescriptor(
            layerIdx: layerIdx,
            projection: projection,
            suffix: suffix)
        {
            return [single]
        }

        guard let layer = layers[layerIdx] else { return [] }
        let entries = layer.experts.compactMap { expertIdx, expert -> (Int, StreamingTensorRef)? in
            guard let ref = expert.tensors[projection]?[suffix] else { return nil }
            return (expertIdx, ref)
        }
        .sorted { $0.0 < $1.0 }
        guard !entries.isEmpty else { return [] }
        for (expected, entry) in entries.enumerated() where entry.0 != expected {
            return []
        }

        guard let first = entries.first?.1 else { return [] }
        for (_, ref) in entries {
            guard ref.dtype == first.dtype,
                ref.shape == first.shape,
                ref.byteCount == first.byteCount
            else { return [] }
        }

        let expertCount = entries.count
        let grouped = Dictionary(grouping: entries) { $0.1.fileURL }
        guard grouped.count > 1 else { return [] }

        return grouped.keys.sorted { $0.path < $1.path }.compactMap { fileURL in
            guard let refs = grouped[fileURL], !refs.isEmpty else { return nil }
            let spanStart = refs.reduce(UInt64.max) { min($0, $1.1.offset) }
            let spanEnd = refs.reduce(UInt64.min) {
                max($0, $1.1.offset + UInt64(max(0, $1.1.byteCount)))
            }
            guard spanStart != UInt64.max,
                spanEnd >= spanStart,
                spanEnd - spanStart <= UInt64(Int.max)
            else { return nil }

            var byteOffsets = [UInt64](repeating: UInt64.max, count: expertCount)
            for (expertIdx, ref) in refs {
                byteOffsets[expertIdx] = ref.offset - spanStart
            }
            return JANGTQStackedOffsetDescriptor(
                layerIdx: layerIdx,
                projectionName: projection.rawValue,
                suffixName: suffix.rawValue,
                fileURL: fileURL,
                spanOffset: spanStart,
                spanByteCount: Int(spanEnd - spanStart),
                expertByteCount: first.byteCount,
                expertByteOffsets: byteOffsets,
                logicalShape: [expertCount] + first.shape,
                dtype: first.dtype,
                storageLayout: "expert-major-multi-file-offsets")
        }
    }

    func hasAlignedOffsetShardGroups(layerIdx: Int) -> Bool {
        func fileSet(_ projection: StreamingProjection, _ suffix: StreamingSuffix) -> Set<URL> {
            Set(
                stackedOffsetDescriptors(
                    layerIdx: layerIdx,
                    projection: projection,
                    suffix: suffix
                ).map(\.fileURL))
        }

        let gatePacked = fileSet(.gate, .packed)
        let gateNorms = fileSet(.gate, .norms)
        let upPacked = fileSet(.up, .packed)
        let upNorms = fileSet(.up, .norms)
        let downPacked = fileSet(.down, .packed)
        let downNorms = fileSet(.down, .norms)

        guard !gatePacked.isEmpty,
            !downPacked.isEmpty
        else { return false }

        return gatePacked == gateNorms
            && gatePacked == upPacked
            && gatePacked == upNorms
            && downPacked == downNorms
    }

    func hasOffsetDispatchCoverage(layerIdx: Int) -> Bool {
        func expertSet(_ projection: StreamingProjection, _ suffix: StreamingSuffix) -> Set<Int> {
            var experts = Set<Int>()
            for descriptor in stackedOffsetDescriptors(
                layerIdx: layerIdx,
                projection: projection,
                suffix: suffix)
            {
                for (expertIdx, byteOffset) in descriptor.expertByteOffsets.enumerated()
                    where byteOffset != UInt64.max
                {
                    experts.insert(expertIdx)
                }
            }
            return experts
        }

        let sets = StreamingProjection.allCases.flatMap { projection in
            StreamingSuffix.allCases.map { suffix in
                expertSet(projection, suffix)
            }
        }
        guard let first = sets.first, !first.isEmpty else { return false }
        return sets.allSatisfy { $0 == first }
    }

    private static func computeTotalTensorBytes(_ layers: [Int: StreamingLayerRef]) -> UInt64 {
        var total: UInt64 = 0
        for layer in layers.values {
            for expert in layer.experts.values {
                for suffixes in expert.tensors.values {
                    for ref in suffixes.values {
                        total += UInt64(max(0, ref.byteCount))
                    }
                }
            }
            for suffixes in layer.stacked.values {
                for ref in suffixes.values {
                    total += UInt64(max(0, ref.byteCount))
                }
            }
        }
        return total
    }

    private struct HeaderRead {
        var dataBase: UInt64
        var tensors: [String: [String: Any]]
    }

    private static func readSafetensorsHeader(_ url: URL) throws -> HeaderRead {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        let prefix = try handle.read(upToCount: 8) ?? Data()
        guard prefix.count == 8 else {
            throw NSError(
                domain: "JANGTQStreamingExperts", code: 1,
                userInfo: [NSLocalizedDescriptionKey: "short safetensors header: \(url.path)"])
        }
        let headerLength = prefix.withUnsafeBytes {
            UInt64(littleEndian: $0.loadUnaligned(as: UInt64.self))
        }
        let headerData = try handle.read(upToCount: Int(headerLength)) ?? Data()
        guard headerData.count == Int(headerLength) else {
            throw NSError(
                domain: "JANGTQStreamingExperts", code: 2,
                userInfo: [NSLocalizedDescriptionKey: "truncated safetensors header: \(url.path)"])
        }
        let json = try JSONSerialization.jsonObject(with: headerData)
        guard let dict = json as? [String: Any] else {
            throw NSError(
                domain: "JANGTQStreamingExperts", code: 3,
                userInfo: [NSLocalizedDescriptionKey: "invalid safetensors header: \(url.path)"])
        }
        var tensors: [String: [String: Any]] = [:]
        for (key, value) in dict where key != "__metadata__" {
            guard var entry = value as? [String: Any] else { continue }
            if let rawShape = entry["shape"] as? [NSNumber] {
                entry["shape"] = rawShape.map(\.intValue)
            }
            if let rawOffsets = entry["data_offsets"] as? [NSNumber] {
                entry["data_offsets"] = rawOffsets.map(\.uint64Value)
            }
            tensors[key] = entry
        }
        return HeaderRead(dataBase: 8 + headerLength, tensors: tensors)
    }

    private struct KeyMatch {
        var layer: Int
        var expert: Int
        var projection: StreamingProjection
        var suffix: StreamingSuffix
    }

    private struct StackedKeyMatch {
        var layer: Int
        var projection: StreamingProjection
        var suffix: StreamingSuffix
    }

    private static func matchPerExpertTQKey(_ key: String) -> KeyMatch? {
        let patterns: [(String, [String: StreamingProjection])] = [
            (
                #"^(?:language_model\.)?model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.(tq_packed|tq_norms)$"#,
                [
                    "gate_proj": .gate,
                    "up_proj": .up,
                    "down_proj": .down,
                ]
            ),
            (
                #"^layers\.(\d+)\.ffn\.experts\.(\d+)\.(w1|w2|w3)\.(tq_packed|tq_norms)$"#,
                [
                    "w1": .gate,
                    "w2": .down,
                    "w3": .up,
                ]
            ),
            (
                #"^(?:language_model\.)?model\.layers\.(\d+)\.(?:mlp|block_sparse_moe)\.experts\.(\d+)\.(w1|w2|w3)\.(tq_packed|tq_norms)$"#,
                [
                    "w1": .gate,
                    "w2": .down,
                    "w3": .up,
                ]
            ),
            (
                #"^backbone\.layers\.(\d+)\.mixer\.experts\.(\d+)\.(up_proj|down_proj)\.(tq_packed|tq_norms)$"#,
                [
                    "up_proj": .up,
                    "down_proj": .down,
                ]
            ),
        ]
        for (pattern, projectionMap) in patterns {
            guard let regex = try? NSRegularExpression(pattern: pattern) else { continue }
            let nsRange = NSRange(key.startIndex ..< key.endIndex, in: key)
            guard let match = regex.firstMatch(in: key, range: nsRange),
                match.numberOfRanges == 5,
                let layerRange = Range(match.range(at: 1), in: key),
                let expertRange = Range(match.range(at: 2), in: key),
                let projectionRange = Range(match.range(at: 3), in: key),
                let suffixRange = Range(match.range(at: 4), in: key),
                let layer = Int(key[layerRange]),
                let expert = Int(key[expertRange]),
                let projection = projectionMap[String(key[projectionRange])],
                let suffix = StreamingSuffix(rawValue: String(key[suffixRange]))
            else { continue }
            return KeyMatch(layer: layer, expert: expert, projection: projection, suffix: suffix)
        }
        return nil
    }

    private static func matchStackedTQKey(_ key: String) -> StackedKeyMatch? {
        let patterns: [(String, [String: StreamingProjection])] = [
            (
                #"^(?:language_model\.)?model\.layers\.(\d+)\.(?:mlp|block_sparse_moe)\.switch_mlp\.(gate_proj|up_proj|down_proj)\.(tq_packed|tq_norms)$"#,
                [
                    "gate_proj": .gate,
                    "up_proj": .up,
                    "down_proj": .down,
                ]
            ),
            (
                #"^(?:language_model\.)?model\.layers\.(\d+)\.(?:mlp\.)?zaya_block\.experts\.switch_mlp\.(gate_proj|up_proj|down_proj)\.(tq_packed|tq_norms)$"#,
                [
                    "gate_proj": .gate,
                    "up_proj": .up,
                    "down_proj": .down,
                ]
            ),
            (
                #"^layers\.(\d+)\.ffn\.switch_mlp\.(w1|w2|w3)\.(tq_packed|tq_norms)$"#,
                [
                    "w1": .gate,
                    "w2": .down,
                    "w3": .up,
                ]
            ),
            (
                #"^backbone\.layers\.(\d+)\.mixer\.switch_mlp\.(up_proj|down_proj|fc1|fc2)\.(tq_packed|tq_norms)$"#,
                [
                    "up_proj": .up,
                    "down_proj": .down,
                    "fc1": .up,
                    "fc2": .down,
                ]
            ),
        ]
        for (pattern, projectionMap) in patterns {
            guard let regex = try? NSRegularExpression(pattern: pattern) else { continue }
            let nsRange = NSRange(key.startIndex ..< key.endIndex, in: key)
            guard let match = regex.firstMatch(in: key, range: nsRange),
                match.numberOfRanges == 4,
                let layerRange = Range(match.range(at: 1), in: key),
                let projectionRange = Range(match.range(at: 2), in: key),
                let suffixRange = Range(match.range(at: 3), in: key),
                let layer = Int(key[layerRange]),
                let projection = projectionMap[String(key[projectionRange])],
                let suffix = StreamingSuffix(rawValue: String(key[suffixRange]))
            else { continue }
            return StackedKeyMatch(layer: layer, projection: projection, suffix: suffix)
        }
        return nil
    }
}

public final class StreamingTurboQuantSwitchReLUSquaredMLP: Module {
    @ModuleInfo(key: "fc1") public var fc1: TurboQuantSwitchLinear
    @ModuleInfo(key: "fc2") public var fc2: TurboQuantSwitchLinear

    private let layerIdx: Int
    private let inputDims: Int
    private let hiddenDims: Int
    private let evaluateEachLayer: Bool
    private let tokenChunkSize: Int

    public init(
        inputDims: Int,
        hiddenDims: Int,
        numExperts: Int,
        bits: Int = 2,
        seed: Int = 42,
        layerIdx: Int
    ) {
        self.layerIdx = layerIdx
        self.inputDims = inputDims
        self.hiddenDims = hiddenDims
        self.evaluateEachLayer = mlXPressStreamingShouldEvaluateLayer(layerIdx)
        self.tokenChunkSize = mlXPressStreamingTokenChunkSize()
        self._fc1.wrappedValue = TurboQuantSwitchLinear(
            inFeatures: inputDims,
            outFeatures: hiddenDims,
            numExperts: numExperts,
            bits: bits,
            seed: seed)
        self._fc2.wrappedValue = TurboQuantSwitchLinear(
            inFeatures: hiddenDims,
            outFeatures: inputDims,
            numExperts: numExperts,
            bits: bits,
            seed: seed)
        super.init()
        _ = JANGTQStreamingExpertStore.shared.index()
    }

    public func callAsFunction(_ x: MLXArray, _ indices: MLXArray) -> MLXArray {
        let totalTokens = x.size / inputDims
        let kSlots = indices.dim(-1)
        let xFlat = x.reshaped([totalTokens, inputDims])
        let indicesFlat = indices.reshaped([totalTokens, kSlots])
        let allIndexValues = indicesFlat.reshaped([-1]).asArray(Int32.self).map(Int.init)
        let chunkSize = max(1, tokenChunkSize)
        if totalTokens <= chunkSize {
            let chunk = callChunk(
                xFlat: xFlat,
                indicesFlat: indicesFlat,
                indexValues: allIndexValues,
                tokenCount: totalTokens,
                kSlots: kSlots)
            var outShape = Array(indices.shape)
            outShape.append(inputDims)
            return chunk.reshaped(outShape)
        }

        var chunks: [MLXArray] = []
        chunks.reserveCapacity((totalTokens + chunkSize - 1) / chunkSize)
        var start = 0
        while start < totalTokens {
            let end = min(start + chunkSize, totalTokens)
            let valueStart = start * kSlots
            let valueEnd = end * kSlots
            chunks.append(
                callChunk(
                    xFlat: xFlat[start ..< end, 0...],
                    indicesFlat: indicesFlat[start ..< end, 0...],
                    indexValues: Array(allIndexValues[valueStart ..< valueEnd]),
                    tokenCount: end - start,
                    kSlots: kSlots))
            start = end
        }
        let joined = concatenated(chunks, axis: 0)
        var outShape = Array(indices.shape)
        outShape.append(inputDims)
        return joined.reshaped(outShape)
    }

    private func callChunk(
        xFlat: MLXArray,
        indicesFlat: MLXArray,
        indexValues: [Int]? = nil,
        tokenCount: Int,
        kSlots: Int
    ) -> MLXArray {
        let indexValues =
            indexValues
            ?? indicesFlat.reshaped([-1]).asArray(Int32.self).map(Int.init)
        let uniqueExperts = Array(Set(indexValues)).sorted()
        guard !uniqueExperts.isEmpty else {
            fatalError("[MLXPressStreaming] empty routed expert set in layer \(layerIdx)")
        }
        MLXPressActiveExpertTrace.record(
            layerIdx: layerIdx,
            expertIndices: indexValues,
            tokenCount: tokenCount,
            kSlots: kSlots)
        let useDirectStacked = JANGTQStreamingExpertStore.shared.canUseDirectStacked(
            layerIdx: layerIdx)

        func stack(_ projection: StreamingProjection, _ suffix: StreamingSuffix) -> MLXArray? {
            MLXPressStreamingProfile.time("stack.\(projection.rawValue).\(suffix.rawValue)") {
                if useDirectStacked {
                    guard let direct = JANGTQStreamingExpertStore.shared.loadDirectStack(
                        layerIdx: layerIdx,
                        projection: projection,
                        suffix: suffix)
                    else {
                        fatalError(
                            "[MLXPressStreaming] direct stacked tensor load failed for layer \(layerIdx) \(projection.rawValue).\(suffix.rawValue)")
                    }
                    return direct
                }
                let bankByteCount = JANGTQStreamingExpertStore.shared.activeBankByteCount(
                    layerIdx: layerIdx,
                    expertIndices: uniqueExperts,
                    projection: projection,
                    suffix: suffix)
                if let bankByteCount,
                    let cached = JANGTQStreamingExpertStore.shared.cachedActiveBank(
                        layerIdx: layerIdx,
                        expertIndices: uniqueExperts,
                        projection: projection,
                        suffix: suffix,
                        byteCount: bankByteCount)
                {
                    return cached
                }
                if let bank = JANGTQStreamingExpertStore.shared.loadStack(
                    layerIdx: layerIdx,
                    expertIndices: uniqueExperts,
                    projection: projection,
                    suffix: suffix)
                {
                    if let bankByteCount {
                        return JANGTQStreamingExpertStore.shared.storeActiveBank(
                            bank,
                            layerIdx: layerIdx,
                            expertIndices: uniqueExperts,
                            projection: projection,
                            suffix: suffix,
                            byteCount: bankByteCount)
                    }
                    return bank
                }
                var arrays: [MLXArray] = []
                arrays.reserveCapacity(uniqueExperts.count)
                for expert in uniqueExperts {
                    guard
                        let array = JANGTQStreamingExpertStore.shared.load(
                            layerIdx: layerIdx,
                            expertIdx: expert,
                            projection: projection,
                            suffix: suffix)
                    else { return nil }
                    arrays.append(array)
                }
                let bank: MLXArray
                if arrays.count == 1 {
                    bank = arrays[0].expandedDimensions(axis: 0)
                } else {
                    bank = MLX.stacked(arrays, axis: 0)
                }
                if let bankByteCount {
                    return JANGTQStreamingExpertStore.shared.storeActiveBank(
                        bank,
                        layerIdx: layerIdx,
                        expertIndices: uniqueExperts,
                        projection: projection,
                        suffix: suffix,
                        byteCount: bankByteCount)
                }
                return bank
            }
        }

        guard
            let signsIn = JANGTQRuntimeCache.shared.signs(
                inFeatures: inputDims, seed: fc1.mxtqSeed),
            let signsInter = JANGTQRuntimeCache.shared.signs(
                inFeatures: hiddenDims, seed: fc2.mxtqSeed),
            let cbIn = JANGTQRuntimeCache.shared.codebook(inFeatures: inputDims, bits: fc1.bits),
            let cbInter = JANGTQRuntimeCache.shared.codebook(inFeatures: hiddenDims, bits: fc2.bits)
        else {
            fatalError(
                "[MLXPressStreaming] missing active Nemotron JANGTQ tensors for layer \(layerIdx)")
        }

        var remap: [Int: Int32] = [:]
        if !useDirectStacked {
            for (local, expert) in uniqueExperts.enumerated() {
                remap[expert] = Int32(local)
            }
        }
        let rhsIndexValues: [Int32] = useDirectStacked
            ? indexValues.map { Int32($0) }
            : indexValues.map { remap[$0] ?? 0 }
        let rhsIndices = MLXArray(rhsIndexValues, indicesFlat.shape).asType(.uint32).reshaped([-1])

        guard let fc1Packed = stack(.up, .packed),
            let fc1Norms = stack(.up, .norms)
        else {
            fatalError(
                "[MLXPressStreaming] missing active Nemotron JANGTQ fc1 tensors for layer \(layerIdx)"
            )
        }
        let xRot1 = JANGTQKernels.hadamardRotate(xFlat, signs: signsIn, dim: inputDims)
        var hidden = JANGTQKernels.gatherTQTopK(
            xRot: xRot1,
            packed: fc1Packed,
            norms: fc1Norms,
            codebook: cbIn,
            rhsIndices: rhsIndices,
            batchTokens: tokenCount,
            K: kSlots,
            inFeatures: inputDims,
            outFeatures: hiddenDims,
            bits: fc1.bits)
        let relu = MLX.maximum(hidden, MLXArray(0, dtype: hidden.dtype))
        hidden = relu * relu
        mlXPressStreamingMaterializeIfNeeded(
            hidden,
            force: evaluateEachLayer,
            layerIdx: layerIdx,
            phase: "relu_hidden",
            profileName: "relu_hidden.eval")

        guard let fc2Packed = stack(.down, .packed),
            let fc2Norms = stack(.down, .norms)
        else {
            fatalError(
                "[MLXPressStreaming] missing active Nemotron JANGTQ fc2 tensors for layer \(layerIdx)"
            )
        }

        let xRot2 = JANGTQKernels.hadamardRotate(hidden, signs: signsInter, dim: hiddenDims)
        let out = JANGTQKernels.gatherTQ(
            xRot: xRot2,
            packed: fc2Packed,
            norms: fc2Norms,
            codebook: cbInter,
            rhsIndices: rhsIndices,
            nRows: tokenCount * kSlots,
            inFeatures: hiddenDims,
            outFeatures: inputDims,
            bits: fc2.bits)

        let shaped = out.reshaped([tokenCount, kSlots, inputDims]).asType(xFlat.dtype)
        mlXPressStreamingMaterializeIfNeeded(
            shaped,
            force: evaluateEachLayer,
            layerIdx: layerIdx,
            phase: "relu_down",
            profileName: "relu_down.eval")
        return shaped
    }
}

private final class JANGTQStreamingExpertStore: @unchecked Sendable {
    static let shared = JANGTQStreamingExpertStore()
    private let lock = NSLock()
    private var configuredModelDirectory: URL?
    private var cachedIndex: JANGTQStreamingExpertIndex?
    private let tensorCacheBudgetBytes = mlXPressStreamingExpertCacheBudgetBytes()
    private let bankCacheBudgetBytes = mlXPressStreamingBankCacheBudgetBytes()
    private let offsetSpanCacheBudgetBytes = mlXPressStreamingOffsetSpanCacheBudgetBytes()
    private let bankLoadEnabled = mlXPressStreamingBankLoadEnabled()
    private let directStackedEnabled = mlXPressStreamingDirectStackedEnabled()
    private let offsetKernelsEnabled = mlXPressStreamingOffsetKernelsEnabled()
    private let machActiveTensorsEnabled = mlXPressStreamingMachActiveTensorsEnabled()
    private let machOffsetSpansEnabled = mlXPressStreamingMachOffsetSpansEnabled()
    private let machFullOffsetSpansEnabled = mlXPressStreamingMachFullOffsetSpansEnabled()
    private let machPrewarmFullOffsetSpansEnabled =
        mlXPressStreamingMachPrewarmFullOffsetSpansEnabled()
    private let machFullOffsetLazyEnabled = mlXPressStreamingMachFullOffsetLazyEnabled()
    private let machOffsetSpanBudgetBytes = mlXPressStreamingMachOffsetSpanBudgetBytes()
    private let machCompressPercent = mlXPressMachCompressPercent()
    private let machCache = JangPressMachCache(
        config: JangPressMachConfig(
            enableDiskRefault: true,
            manualCompressPercent: mlXPressMachCompressPercent()))
    private var tensorCacheBytes = 0
    private var tensorCacheClock: UInt64 = 0
    private var tensorCache: [StreamingTensorCacheKey: StreamingTensorCacheEntry] = [:]
    private var tensorCacheHits = 0
    private var tensorCacheMisses = 0
    private var tensorCacheHitBytes: UInt64 = 0
    private var tensorCacheMissBytes: UInt64 = 0
    private var tensorCacheStores = 0
    private var tensorCacheEvictions = 0
    private var activeBankCacheBytes = 0
    private var activeBankCache: [StreamingActiveBankCacheKey: StreamingActiveBankCacheEntry] = [:]
    private var activeBankCacheHits = 0
    private var activeBankCacheMisses = 0
    private var activeBankCacheHitBytes: UInt64 = 0
    private var activeBankCacheMissBytes: UInt64 = 0
    private var activeBankCacheStores = 0
    private var activeBankCacheEvictions = 0
    private var offsetSpanCacheBytes = 0
    private var offsetSpanCache: [StreamingOffsetSpanCacheKey: StreamingOffsetSpanCacheEntry] = [:]
    private var offsetSpanCacheHits = 0
    private var offsetSpanCacheMisses = 0
    private var offsetSpanCacheHitBytes: UInt64 = 0
    private var offsetSpanCacheMissBytes: UInt64 = 0
    private var offsetSpanCacheStores = 0
    private var offsetSpanCacheEvictions = 0
    private var residentTensorBytes = 0
    private var residentTensors: [StreamingTensorCacheKey: StreamingResidentTensorEntry] = [:]
    private var machRegisteredTensors: [StreamingTensorCacheKey: StreamingMachTensorEntry] = [:]
    private var machRegisteredTensorBytes = 0
    private var machOffsetSpanRegisteredBytes = 0
    private var machTensorHits = 0
    private var machTensorMisses = 0
    private var machTensorHitBytes: UInt64 = 0
    private var machTensorMissBytes: UInt64 = 0
    private var machTensorStores = 0
    private var machTensorEvictions = 0
    private var machOffsetBudgetSkips = 0
    private var directStackedTensorCache: [StreamingStackedTensorCacheKey: MLXArray] = [:]
    private var directStackedShardCache: [String: [String: MLXArray]] = [:]
    private var sliceCacheBytes = 0
    private var sliceCache: [StreamingTensorCacheKey: StreamingSliceCacheEntry] = [:]
    private var sliceCacheHits = 0
    private var sliceCacheMisses = 0
    private var sliceCacheHitBytes: UInt64 = 0
    private var sliceCacheMissBytes: UInt64 = 0
    private var sliceCacheStores = 0
    private var sliceCacheEvictions = 0
    private var readPolicyLogged = false
    private var didPrewarmFullOffsetSpans = false
    #if canImport(Darwin)
        private var cachedFileDescriptors: [String: Int32] = [:]
    #endif

    func configureModelDirectory(_ modelDirectory: URL) {
        let resolved = modelDirectory.resolvingSymlinksInPath()
        var changed = false
        lock.lock()
        if configuredModelDirectory != resolved {
            configuredModelDirectory = resolved
            clearModelScopedCachesLocked()
            changed = true
        }
        lock.unlock()
        if changed {
            machCache.removeAll()
        }
    }

    func clearConfiguredModelDirectory() {
        lock.lock()
        configuredModelDirectory = nil
        clearModelScopedCachesLocked()
        lock.unlock()
        machCache.removeAll()
    }

    func configuredModelDirectoryForDiagnostics() -> URL? {
        lock.lock()
        let directory = configuredModelDirectory
        lock.unlock()
        return directory
    }

    private func clearModelScopedCachesLocked() {
        cachedIndex = nil
        tensorCache.removeAll(keepingCapacity: false)
        tensorCacheBytes = 0
        activeBankCache.removeAll(keepingCapacity: false)
        activeBankCacheBytes = 0
        offsetSpanCache.removeAll(keepingCapacity: false)
        offsetSpanCacheBytes = 0
        residentTensors.removeAll(keepingCapacity: false)
        residentTensorBytes = 0
        machRegisteredTensors.removeAll(keepingCapacity: false)
        machRegisteredTensorBytes = 0
        machOffsetSpanRegisteredBytes = 0
        directStackedTensorCache.removeAll(keepingCapacity: false)
        directStackedShardCache.removeAll(keepingCapacity: false)
        sliceCache.removeAll(keepingCapacity: false)
        sliceCacheBytes = 0
        readPolicyLogged = false
        didPrewarmFullOffsetSpans = false
    }

    func index() -> JANGTQStreamingExpertIndex? {
        lock.lock()
        if let cachedIndex {
            lock.unlock()
            return cachedIndex
        }
        let configuredModelDirectory = configuredModelDirectory
        lock.unlock()

        let modelDirectory: URL
        if let configuredModelDirectory {
            modelDirectory = configuredModelDirectory
        } else if let path =
            ProcessInfo.processInfo.environment["MLXPRESS_MODEL_DIR"]
            ?? ProcessInfo.processInfo.environment["JANGPRESS_MODEL_DIR"],
            !path.isEmpty
        {
            modelDirectory = URL(fileURLWithPath: path).resolvingSymlinksInPath()
        } else {
            return nil
        }
        guard let built = try? JANGTQStreamingExpertIndex.build(modelDirectory: modelDirectory)
        else {
            return nil
        }
        lock.lock()
        cachedIndex = built
        lock.unlock()
        let layers = built.layers.count
        let experts = built.layers.values.map { $0.experts.count }.max() ?? 0
        let stacked = built.layers.values.filter { !$0.stacked.isEmpty }.count
        let cacheMB = tensorCacheBudgetBytes / (1024 * 1024)
        let bankCacheMB = bankCacheBudgetBytes / (1024 * 1024)
        let offsetSpanCacheMB = offsetSpanCacheBudgetBytes / (1024 * 1024)
        let noCache = mlXPressStreamingUseFNoCache(for: built)
        let streamableGB = Double(built.totalTensorBytes) / (1024 * 1024 * 1024)
        let thresholdGB = Double(mlXPressStreamingNoCacheThresholdBytes()) / (1024 * 1024 * 1024)
        FileHandle.standardError.write(
            Data(
                "[MLXPressStreaming] indexed active-expert JANGTQ tensors layers=\(layers) experts=\(experts) stackedLayers=\(stacked)\n"
                    .utf8))
        FileHandle.standardError.write(
            Data("[MLXPressStreaming] active-expert tensor cache budget=\(cacheMB) MB\n".utf8))
        FileHandle.standardError.write(
            Data("[MLXPressStreaming] active-expert bank cache budget=\(bankCacheMB) MB\n".utf8))
        FileHandle.standardError.write(
            Data("[MLXPressStreaming] offset-span cache budget=\(offsetSpanCacheMB) MB\n".utf8))
        FileHandle.standardError.write(
            Data(
                String(
                    format: "[MLXPressStreaming] active-expert read cache policy=%@ streamable=%.2fGB threshold=%.2fGB\n",
                    noCache ? "F_NOCACHE" : "os-cache",
                    streamableGB,
                    thresholdGB
                ).utf8))
        if directStackedEnabled {
            FileHandle.standardError.write(
                Data("[MLXPressStreaming] direct stacked mmap dispatch enabled\n".utf8))
        }
        if offsetKernelsEnabled {
            FileHandle.standardError.write(
                Data("[MLXPressStreaming] offset-addressed active expert kernels enabled\n".utf8))
        }
        if machActiveTensorsEnabled {
            FileHandle.standardError.write(
                Data(
                    "[MLXPressStreaming] Mach purgeable active tensor cache enabled compressPct=\(machCompressPercent)\n"
                        .utf8))
        }
        if machOffsetSpansEnabled {
            let budgetMB = machOffsetSpanBudgetBytes / (1024 * 1024)
            FileHandle.standardError.write(
                Data(
                    "[MLXPressStreaming] Mach purgeable offset-span cache enabled compressPct=\(machCompressPercent) budget=\(budgetMB) MB\n"
                        .utf8))
            if machFullOffsetSpansEnabled {
                FileHandle.standardError.write(
                    Data(
                        "[MLXPressStreaming] Mach full routed-span residency enabled prewarm=\(machPrewarmFullOffsetSpansEnabled) lazy=\(machFullOffsetLazyEnabled)\n"
                            .utf8))
            }
        }
        prewarmFullOffsetSpansIfNeeded(index: built)
        return built
    }

    func resetResidentTensors() {
        lock.lock()
        residentTensors.removeAll(keepingCapacity: false)
        residentTensorBytes = 0
        machRegisteredTensors.removeAll(keepingCapacity: false)
        machRegisteredTensorBytes = 0
        machOffsetSpanRegisteredBytes = 0
        didPrewarmFullOffsetSpans = false
        lock.unlock()
        machCache.removeAll()
    }

    private func prewarmFullOffsetSpansIfNeeded(index: JANGTQStreamingExpertIndex) {
        guard machOffsetSpansEnabled,
            machFullOffsetSpansEnabled,
            machPrewarmFullOffsetSpansEnabled
        else { return }

        lock.lock()
        if didPrewarmFullOffsetSpans {
            lock.unlock()
            return
        }
        didPrewarmFullOffsetSpans = true
        lock.unlock()

        let maxLayers = mlXPressStreamingMachPrewarmMaxLayers()
        let maxBytes = mlXPressStreamingMachPrewarmMaxBytes()
        let layerIds = index.layers.keys.sorted()
        let selectedLayerIds = maxLayers.map { Array(layerIds.prefix($0)) } ?? layerIds
        var prewarmed = 0
        var prewarmedBytes: UInt64 = 0
        var stoppedByBudget = false
        let start = Date.timeIntervalSinceReferenceDate

        FileHandle.standardError.write(
            Data(
                "[MLXPressStreaming] Mach full routed-span prewarm start layers=\(selectedLayerIds.count)\n"
                    .utf8))

        outer: for layerIdx in selectedLayerIds {
            for projection in StreamingProjection.allCases {
                for suffix in StreamingSuffix.allCases {
                    let descriptors = index.stackedOffsetDescriptors(
                        layerIdx: layerIdx,
                        projection: projection,
                        suffix: suffix)
                    for descriptor in descriptors {
                        if let maxBytes, prewarmedBytes >= maxBytes {
                            stoppedByBudget = true
                            break outer
                        }
                        guard let elementByteSize = elementByteSize(from: descriptor.dtype),
                            descriptor.spanByteCount % elementByteSize == 0
                        else { continue }
                        if let maxBytes,
                            prewarmedBytes + UInt64(max(0, descriptor.spanByteCount)) > maxBytes
                        {
                            stoppedByBudget = true
                            break outer
                        }
                        let elementCount = descriptor.spanByteCount / elementByteSize
                        guard makeMachFullOffsetSpanArray(
                            from: descriptor,
                            shape: [elementCount]) != nil
                        else { continue }
                        let key = machFullOffsetKey(
                            layerIdx: descriptor.layerIdx,
                            projection: projection,
                            suffix: suffix,
                            descriptor: descriptor)
                        machCache.release(
                            layer: key.layerIdx,
                            expert: key.expertIdx,
                            components: [machComponentName(for: key)])
                        prewarmed += 1
                        prewarmedBytes += UInt64(max(0, descriptor.spanByteCount))
                        MLXPressStreamingProfile.record(
                            "tensor.mach_full_offset_prewarm",
                            bytes: descriptor.spanByteCount)
                    }
                }
            }
        }

        let seconds = Date.timeIntervalSinceReferenceDate - start
        MLXPressStreamingProfile.record(
            stoppedByBudget
                ? "tensor.mach_full_offset_prewarm_budget_stop"
                : "tensor.mach_full_offset_prewarm_done",
            seconds: seconds,
            bytes: Int(min(prewarmedBytes, UInt64(Int.max))))
        FileHandle.standardError.write(
            Data(
                String(
                    format: "[MLXPressStreaming] Mach full routed-span prewarm %@ spans=%d bytes=%.2fGB seconds=%.2f\n",
                    stoppedByBudget ? "stopped" : "done",
                    prewarmed,
                    Double(prewarmedBytes) / (1024.0 * 1024.0 * 1024.0),
                    seconds
                ).utf8))
    }

    func registerResidentTensor(
        layerIdx: Int,
        expertIdx: Int,
        projection: StreamingProjection,
        suffix: StreamingSuffix,
        array: MLXArray
    ) {
        let key = StreamingTensorCacheKey(
            layerIdx: layerIdx,
            expertIdx: expertIdx,
            projection: projection,
            suffix: suffix)
        let byteCount = array.nbytes
        lock.lock()
        if let existing = residentTensors[key] {
            residentTensorBytes -= existing.byteCount
        }
        residentTensors[key] = StreamingResidentTensorEntry(
            array: array,
            byteCount: byteCount)
        residentTensorBytes += byteCount
        lock.unlock()
    }

    func canUseDirectStacked(layerIdx: Int) -> Bool {
        guard directStackedEnabled,
            let layer = index()?.layers[layerIdx],
            !layer.stacked.isEmpty
        else { return false }
        for projection in StreamingProjection.allCases {
            guard layer.stacked[projection]?[.packed] != nil,
                layer.stacked[projection]?[.norms] != nil
            else { return false }
        }
        return true
    }

    func canUseOffsetDispatch(layerIdx: Int) -> Bool {
        guard offsetKernelsEnabled,
            mmapActiveTensorsEnabled(),
            let index = index()
        else { return false }
        guard index.hasOffsetDispatchCoverage(layerIdx: layerIdx) else {
            MLXPressStreamingProfile.record("offset_dispatch_incomplete_coverage")
            return false
        }
        if !index.hasAlignedOffsetShardGroups(layerIdx: layerIdx) {
            MLXPressStreamingProfile.record("offset_dispatch_unaligned_shards")
            MLXPressStreamingProfile.record("offset_dispatch_flexible_shard_groups")
        }
        for projection in StreamingProjection.allCases {
            for suffix in StreamingSuffix.allCases {
                guard !index.stackedOffsetDescriptors(
                    layerIdx: layerIdx,
                    projection: projection,
                    suffix: suffix).isEmpty
                else { return false }
            }
        }
        return true
    }

    func loadOffsetSpans(
        layerIdx: Int,
        projection: StreamingProjection,
        suffix: StreamingSuffix,
        activeExperts: Set<Int>? = nil
    ) -> [StreamingOffsetSpan] {
        guard offsetKernelsEnabled,
            let descriptors = index()?.stackedOffsetDescriptors(
                layerIdx: layerIdx,
                projection: projection,
                suffix: suffix),
            !descriptors.isEmpty
        else { return [] }
        let spans = descriptors.flatMap {
            makeOffsetSpans(from: $0, activeExperts: activeExperts)
        }
        if activeExperts != nil {
            return spans
        }
        return spans.count == descriptors.count ? spans : []
    }

    private func makeOffsetSpans(
        from descriptor: JANGTQStackedOffsetDescriptor,
        activeExperts: Set<Int>? = nil
    ) -> [StreamingOffsetSpan] {
        guard
            let elementByteSize = elementByteSize(from: descriptor.dtype),
            descriptor.spanByteCount % elementByteSize == 0
        else { return [] }

        if machOffsetSpansEnabled,
            machFullOffsetSpansEnabled,
            let span = makeFullMachOffsetSpan(from: descriptor, elementByteSize: elementByteSize)
        {
            return [span]
        }

        if activeExperts == nil {
            guard let span = makeOffsetSpanWindow(
                from: descriptor,
                spanOffset: descriptor.spanOffset,
                spanByteCount: descriptor.spanByteCount,
                selectedExperts: Set(
                    descriptor.expertByteOffsets.enumerated().compactMap { expertIdx, byteOffset in
                        byteOffset == UInt64.max ? nil : expertIdx
                    }),
                elementByteSize: elementByteSize)
            else { return [] }
            return [span]
        }

        let windows = activeExpertWindows(
            descriptor: descriptor,
            activeExperts: activeExperts ?? [],
            elementByteSize: elementByteSize)
        guard !windows.isEmpty else { return [] }

        let mappedBytes = windows.reduce(0) { $0 + Int($1.end - $1.start) }
        let skippedBytes = max(0, descriptor.spanByteCount - mappedBytes)
        if skippedBytes > 0 {
            MLXPressStreamingProfile.record(
                "tensor.offset_span_active_window",
                bytes: skippedBytes)
        }
        if windows.count > 1 {
            MLXPressStreamingProfile.record(
                "tensor.offset_span_active_segments",
                bytes: mappedBytes)
        }

        return windows.compactMap { window in
            makeOffsetSpanWindow(
                from: descriptor,
                spanOffset: window.start,
                spanByteCount: Int(window.end - window.start),
                selectedExperts: Set(window.experts),
                elementByteSize: elementByteSize)
        }
    }

    private func activeExpertWindows(
        descriptor: JANGTQStackedOffsetDescriptor,
        activeExperts: Set<Int>,
        elementByteSize: Int
    ) -> [JANGTQActiveOffsetWindow] {
        let windows = descriptor.activeExpertByteWindows(
            activeExperts: activeExperts,
            elementByteSize: elementByteSize,
            maxGapBytes: mlXPressStreamingOffsetActiveWindowCoalesceBytes())
        let selectedExpertCount = windows.reduce(0) { $0 + $1.experts.count }
        if selectedExpertCount > windows.count {
            MLXPressStreamingProfile.record(
                "tensor.offset_span_active_coalesced_experts",
                bytes: selectedExpertCount - windows.count)
        }
        return windows
    }

    private func makeFullMachOffsetSpan(
        from descriptor: JANGTQStackedOffsetDescriptor,
        elementByteSize: Int
    ) -> StreamingOffsetSpan? {
        guard descriptor.spanByteCount > 0,
            descriptor.spanByteCount % elementByteSize == 0
        else { return nil }
        let selectedExperts = Set(
            descriptor.expertByteOffsets.enumerated().compactMap { expertIdx, byteOffset in
                byteOffset == UInt64.max ? nil : expertIdx
            })
        guard !selectedExperts.isEmpty else { return nil }
        let cacheKey = StreamingOffsetSpanCacheKey(
            fileURL: descriptor.fileURL,
            offset: descriptor.spanOffset,
            byteCount: descriptor.spanByteCount,
            dtype: descriptor.dtype)
        if let cached = cachedOffsetSpan(for: cacheKey) {
            return cached
        }

        var elementOffsets: [UInt32] = []
        elementOffsets.reserveCapacity(descriptor.expertByteOffsets.count)
        var presentExperts = Set<Int>()
        for (expertIdx, byteOffset) in descriptor.expertByteOffsets.enumerated() {
            if byteOffset == UInt64.max {
                elementOffsets.append(UInt32.max)
                continue
            }
            guard byteOffset % UInt64(elementByteSize) == 0 else { return nil }
            let elementOffset = byteOffset / UInt64(elementByteSize)
            guard elementOffset <= UInt64(UInt32.max) else { return nil }
            elementOffsets.append(UInt32(elementOffset))
            presentExperts.insert(expertIdx)
        }
        guard !presentExperts.isEmpty else { return nil }

        let elementCount = descriptor.spanByteCount / elementByteSize
        let machKey = machFullOffsetKey(for: descriptor)
        guard machFullOffsetLazyEnabled || isMachRegistered(key: machKey) else {
            return nil
        }
        guard let array = makeMachFullOffsetSpanArray(
            from: descriptor,
            shape: [elementCount])
        else { return nil }
        MLXPressStreamingProfile.record(
            "tensor.mach_full_offset_span.\(descriptor.storageLayout)",
            bytes: descriptor.spanByteCount)
        let span = StreamingOffsetSpan(
            fileURL: descriptor.fileURL,
            array: array,
            offsets: MLXArray(elementOffsets),
            byteCount: descriptor.spanByteCount,
            storageLayout: "mach-full-\(descriptor.storageLayout)",
            presentExperts: presentExperts)
        return storeOffsetSpan(span, key: cacheKey)
    }

    private func makeOffsetSpanWindow(
        from descriptor: JANGTQStackedOffsetDescriptor,
        spanOffset: UInt64,
        spanByteCount: Int,
        selectedExperts: Set<Int>,
        elementByteSize: Int
    ) -> StreamingOffsetSpan? {
        guard !selectedExperts.isEmpty,
            spanByteCount > 0,
            spanByteCount % elementByteSize == 0
        else { return nil }

        let cacheKey = StreamingOffsetSpanCacheKey(
            fileURL: descriptor.fileURL,
            offset: spanOffset,
            byteCount: spanByteCount,
            dtype: descriptor.dtype)
        let machEligible = machOffsetSpansEnabled
            && selectedExperts.count == 1
            && spanByteCount == descriptor.expertByteCount
        if !machEligible, let cached = cachedOffsetSpan(for: cacheKey) {
            return cached
        }

        var elementOffsets: [UInt32] = []
        elementOffsets.reserveCapacity(descriptor.expertByteOffsets.count)
        var presentExperts = Set<Int>()
        for (expertIdx, byteOffset) in descriptor.expertByteOffsets.enumerated() {
            if byteOffset == UInt64.max || !selectedExperts.contains(expertIdx) {
                elementOffsets.append(UInt32.max)
                continue
            }
            presentExperts.insert(expertIdx)
            let absoluteOffset = descriptor.spanOffset + byteOffset
            guard absoluteOffset >= spanOffset else { return nil }
            let relativeByteOffset = absoluteOffset - spanOffset
            guard relativeByteOffset % UInt64(elementByteSize) == 0 else { return nil }
            let elementOffset = relativeByteOffset / UInt64(elementByteSize)
            guard elementOffset <= UInt64(UInt32.max) else { return nil }
            elementOffsets.append(UInt32(elementOffset))
        }
        guard !presentExperts.isEmpty else { return nil }

        let elementCount = spanByteCount / elementByteSize
        let shape = [elementCount]
        let machArray = machEligible
            ? makeMachOffsetSpanArray(
                from: descriptor,
                spanOffset: spanOffset,
                spanByteCount: spanByteCount,
                expertIdx: presentExperts.first!,
                shape: shape)
            : nil
        let array = machArray ?? MLXPressStreamingProfile.time(
            "tensor.offset_span_mmap_array",
            bytes: spanByteCount
        ) { () -> MLXArray? in
            makeMmapArray(
                from: descriptor.fileURL,
                offset: spanOffset,
                count: spanByteCount,
                shape: shape,
                dtype: descriptor.dtype)
        }
        guard let array else { return nil }
        MLXPressStreamingProfile.record(
            "tensor.offset_span.\(descriptor.storageLayout)",
            bytes: spanByteCount)
        let span = StreamingOffsetSpan(
            fileURL: descriptor.fileURL,
            array: array,
            offsets: MLXArray(elementOffsets),
            byteCount: spanByteCount,
            storageLayout: descriptor.storageLayout,
            presentExperts: presentExperts)
        return machArray == nil ? storeOffsetSpan(span, key: cacheKey) : span
    }

    private func makeMachOffsetSpanArray(
        from descriptor: JANGTQStackedOffsetDescriptor,
        spanOffset: UInt64,
        spanByteCount: Int,
        expertIdx: Int,
        shape: [Int]
    ) -> MLXArray? {
        guard let projection = StreamingProjection(rawValue: descriptor.projectionName),
            let suffix = StreamingSuffix(rawValue: descriptor.suffixName)
        else { return nil }
        let key = StreamingTensorCacheKey(
            layerIdx: descriptor.layerIdx,
            expertIdx: expertIdx,
            projection: projection,
            suffix: suffix)
        let ref = StreamingTensorRef(
            name: "\(descriptor.projectionName).\(descriptor.suffixName)#offset.\(expertIdx)",
            fileURL: descriptor.fileURL,
            offset: spanOffset,
            byteCount: spanByteCount,
            dtype: descriptor.dtype,
            shape: shape)
        return loadMachOffsetSpan(ref: ref, key: key, shape: shape)
    }

    private func makeMachFullOffsetSpanArray(
        from descriptor: JANGTQStackedOffsetDescriptor,
        shape: [Int]
    ) -> MLXArray? {
        guard let projection = StreamingProjection(rawValue: descriptor.projectionName),
            let suffix = StreamingSuffix(rawValue: descriptor.suffixName)
        else { return nil }
        let key = machFullOffsetKey(
            layerIdx: descriptor.layerIdx,
            projection: projection,
            suffix: suffix,
            descriptor: descriptor)
        let ref = StreamingTensorRef(
            name: "\(descriptor.projectionName).\(descriptor.suffixName)#full-offset.\(descriptor.spanOffset)",
            fileURL: descriptor.fileURL,
            offset: descriptor.spanOffset,
            byteCount: descriptor.spanByteCount,
            dtype: descriptor.dtype,
            shape: shape)
        return loadMachBackedTensor(
            ref: ref,
            key: key,
            shape: shape,
            isOffsetSpan: true,
            readProfileName: "tensor.mach_full_offset_read",
            registerProfileName: "tensor.mach_full_offset_register",
            hitProfileName: "tensor.mach_full_offset_hit",
            arrayProfileName: "tensor.mach_full_offset_array",
            useNoCacheForRegistration: false)
    }

    private func machFullOffsetKey(
        for descriptor: JANGTQStackedOffsetDescriptor
    ) -> StreamingTensorCacheKey {
        guard let projection = StreamingProjection(rawValue: descriptor.projectionName),
            let suffix = StreamingSuffix(rawValue: descriptor.suffixName)
        else {
            return StreamingTensorCacheKey(
                layerIdx: descriptor.layerIdx,
                expertIdx: machFullOffsetSyntheticExpertIndex(for: descriptor),
                projection: .gate,
                suffix: .packed)
        }
        return machFullOffsetKey(
            layerIdx: descriptor.layerIdx,
            projection: projection,
            suffix: suffix,
            descriptor: descriptor)
    }

    private func machFullOffsetKey(
        layerIdx: Int,
        projection: StreamingProjection,
        suffix: StreamingSuffix,
        descriptor: JANGTQStackedOffsetDescriptor
    ) -> StreamingTensorCacheKey {
        StreamingTensorCacheKey(
            layerIdx: layerIdx,
            expertIdx: machFullOffsetSyntheticExpertIndex(for: descriptor),
            projection: projection,
            suffix: suffix)
    }

    private func isMachRegistered(key: StreamingTensorCacheKey) -> Bool {
        lock.lock()
        let registered = machRegisteredTensors[key] != nil
        lock.unlock()
        return registered
    }

    private func machFullOffsetSyntheticExpertIndex(
        for descriptor: JANGTQStackedOffsetDescriptor
    ) -> Int {
        let key = "\(descriptor.fileURL.path)#\(descriptor.spanOffset)#\(descriptor.spanByteCount)"
        var hash: UInt64 = 14_695_981_039_346_656_037
        for byte in key.utf8 {
            hash ^= UInt64(byte)
            hash &*= 1_099_511_628_211
        }
        let positive = Int(hash % UInt64(Int32.max - 1)) + 1
        return -positive
    }

    private func cachedOffsetSpan(for key: StreamingOffsetSpanCacheKey) -> StreamingOffsetSpan? {
        guard offsetSpanCacheBudgetBytes > 0 else { return nil }
        lock.lock()
        defer { lock.unlock() }
        guard var entry = offsetSpanCache[key], entry.byteCount == key.byteCount else {
            offsetSpanCacheMisses += 1
            offsetSpanCacheMissBytes += UInt64(max(0, key.byteCount))
            return nil
        }
        tensorCacheClock &+= 1
        entry.lastUse = tensorCacheClock
        offsetSpanCache[key] = entry
        offsetSpanCacheHits += 1
        offsetSpanCacheHitBytes += UInt64(max(0, key.byteCount))
        MLXPressStreamingProfile.record("tensor.offset_span_cache_hit", bytes: key.byteCount)
        return entry.span
    }

    private func storeOffsetSpan(
        _ span: StreamingOffsetSpan,
        key: StreamingOffsetSpanCacheKey
    ) -> StreamingOffsetSpan {
        guard offsetSpanCacheBudgetBytes > 0,
            key.byteCount <= offsetSpanCacheBudgetBytes
        else {
            return span
        }
        MLXPressStreamingProfile.record("tensor.offset_span_cache_store", bytes: key.byteCount)
        lock.lock()
        tensorCacheClock &+= 1
        if let existing = offsetSpanCache[key] {
            offsetSpanCacheBytes -= existing.byteCount
        }
        offsetSpanCache[key] = StreamingOffsetSpanCacheEntry(
            span: span,
            byteCount: key.byteCount,
            lastUse: tensorCacheClock)
        offsetSpanCacheBytes += key.byteCount
        offsetSpanCacheStores += 1
        evictResidencyToBudgetLocked()
        lock.unlock()
        return span
    }

    func loadDirectStack(
        layerIdx: Int,
        projection: StreamingProjection,
        suffix: StreamingSuffix
    ) -> MLXArray? {
        guard directStackedEnabled else { return nil }
        guard let layer = index()?.layers[layerIdx],
            let ref = layer.stacked[projection]?[suffix]
        else { return nil }

        let cacheKey = StreamingStackedTensorCacheKey(
            layerIdx: layerIdx,
            projection: projection,
            suffix: suffix)
        lock.lock()
        if let cached = directStackedTensorCache[cacheKey] {
            lock.unlock()
            MLXPressStreamingProfile.record("tensor.direct_stacked_mmap_hit")
            return cached
        }
        lock.unlock()

        let loaded = MLXPressStreamingProfile.time("tensor.direct_stacked_mmap_load") {
            loadMmapTensor(ref)
        }
        guard let loaded else { return nil }
        lock.lock()
        directStackedTensorCache[cacheKey] = loaded
        lock.unlock()
        return loaded
    }

    func loadStack(
        layerIdx: Int,
        expertIndices: [Int],
        projection: StreamingProjection,
        suffix: StreamingSuffix
    ) -> MLXArray? {
        guard bankLoadEnabled, !expertIndices.isEmpty else { return nil }
        guard let layer = index()?.layers[layerIdx],
            let ref = layer.stacked[projection]?[suffix],
            let expertCount = ref.shape.first,
            expertCount > 0,
            ref.byteCount % expertCount == 0
        else { return nil }

        let expertByteCount = ref.byteCount / expertCount
        let expertShape = Array(ref.shape.dropFirst())
        guard !expertShape.isEmpty else { return nil }
        for expert in expertIndices where expert < 0 || expert >= expertCount {
            return nil
        }

        let readStart = Date.timeIntervalSinceReferenceDate
        let readResult = readStackedSlices(
            from: ref,
            layerIdx: layerIdx,
            expertIndices: expertIndices,
            projection: projection,
            suffix: suffix,
            expertByteCount: expertByteCount)
        let readSeconds = Date.timeIntervalSinceReferenceDate - readStart
        guard let readResult else { return nil }
        MLXPressStreamingProfile.record(
            "tensor.stacked_bank_read",
            seconds: readSeconds,
            bytes: readResult.fileReadBytes)
        if readResult.cacheHitBytes > 0 {
            MLXPressStreamingProfile.record(
                "tensor.active_slice_cache_hit",
                bytes: readResult.cacheHitBytes)
        }
        if readResult.fileReadBytes < readResult.assembledBytes {
            MLXPressStreamingProfile.record(
                "tensor.stacked_bank_cached_assemble",
                bytes: readResult.assembledBytes)
        }
        var shape = [expertIndices.count]
        shape.append(contentsOf: expertShape)
        let array = MLXPressStreamingProfile.time(
            "tensor.stacked_bank_array",
            bytes: expertByteCount * expertIndices.count
        ) {
            makeArray(data: readResult.data, shape: shape, dtype: ref.dtype)
        }
        guard let array else { return nil }
        return array
    }

    func activeBankByteCount(
        layerIdx: Int,
        expertIndices: [Int],
        projection: StreamingProjection,
        suffix: StreamingSuffix
    ) -> Int? {
        guard !expertIndices.isEmpty else { return nil }
        var residentTotal = 0
        var allResident = true
        lock.lock()
        for expert in expertIndices {
            let key = StreamingTensorCacheKey(
                layerIdx: layerIdx,
                expertIdx: expert,
                projection: projection,
                suffix: suffix)
            guard let entry = residentTensors[key] else {
                allResident = false
                break
            }
            residentTotal += entry.byteCount
        }
        lock.unlock()
        if allResident {
            return residentTotal
        }

        guard let layer = index()?.layers[layerIdx] else { return nil }
        var total = 0
        for expert in expertIndices {
            if let ref = layer.experts[expert]?.tensors[projection]?[suffix] {
                total += ref.byteCount
                continue
            }
            guard let ref = layer.stacked[projection]?[suffix],
                let expertCount = ref.shape.first,
                expertCount > 0,
                expert >= 0,
                expert < expertCount,
                ref.byteCount % expertCount == 0
            else { return nil }
            total += ref.byteCount / expertCount
        }
        return total
    }

    func cachedActiveBank(
        layerIdx: Int,
        expertIndices: [Int],
        projection: StreamingProjection,
        suffix: StreamingSuffix,
        byteCount: Int
    ) -> MLXArray? {
        guard bankCacheBudgetBytes > 0, !expertIndices.isEmpty else { return nil }
        let key = StreamingActiveBankCacheKey(
            layerIdx: layerIdx,
            projection: projection,
            suffix: suffix,
            experts: expertIndices)
        lock.lock()
        defer { lock.unlock() }
        guard var entry = activeBankCache[key], entry.byteCount == byteCount else {
            activeBankCacheMisses += 1
            activeBankCacheMissBytes += UInt64(max(0, byteCount))
            return nil
        }
        tensorCacheClock &+= 1
        entry.lastUse = tensorCacheClock
        activeBankCache[key] = entry
        activeBankCacheHits += 1
        activeBankCacheHitBytes += UInt64(max(0, byteCount))
        MLXPressStreamingProfile.record("tensor.active_bank_cache_hit", bytes: byteCount)
        return entry.array
    }

    func storeActiveBank(
        _ array: MLXArray,
        layerIdx: Int,
        expertIndices: [Int],
        projection: StreamingProjection,
        suffix: StreamingSuffix,
        byteCount: Int
    ) -> MLXArray {
        guard bankCacheBudgetBytes > 0,
            !expertIndices.isEmpty,
            byteCount <= bankCacheBudgetBytes
        else {
            return array
        }
        MLXPressStreamingProfile.record("tensor.active_bank_cache_store", bytes: byteCount)
        let key = StreamingActiveBankCacheKey(
            layerIdx: layerIdx,
            projection: projection,
            suffix: suffix,
            experts: expertIndices)
        lock.lock()
        tensorCacheClock &+= 1
        if let existing = activeBankCache[key] {
            activeBankCacheBytes -= existing.byteCount
        }
        activeBankCache[key] = StreamingActiveBankCacheEntry(
            array: array,
            byteCount: byteCount,
            lastUse: tensorCacheClock)
        activeBankCacheBytes += byteCount
        activeBankCacheStores += 1
        evictResidencyToBudgetLocked()
        lock.unlock()
        return array
    }

    private func loadMmapTensor(_ ref: StreamingTensorRef) -> MLXArray? {
        let path = ref.fileURL.path
        lock.lock()
        if let arrays = directStackedShardCache[path],
            let array = arrays[ref.name]
        {
            lock.unlock()
            return array
        }
        lock.unlock()

        let arrays: [String: MLXArray]
        do {
            arrays = try withMmapSafetensorsEnvForDirectStackedLoad {
                try MLX.loadArrays(url: ref.fileURL)
            }
        } catch {
            FileHandle.standardError.write(
                Data(
                    "[MLXPressStreaming] direct stacked mmap load failed file=\(path) error=\(error)\n"
                        .utf8))
            return nil
        }

        guard let array = arrays[ref.name] else {
            return nil
        }
        lock.lock()
        directStackedShardCache[path] = arrays
        lock.unlock()
        return array
    }

    private func withMmapSafetensorsEnvForDirectStackedLoad<T>(
        _ body: () throws -> T
    ) rethrows -> T {
        #if canImport(Darwin) || canImport(Glibc)
        let mmapKey = "MLX_SAFETENSORS_MMAP"
        let vmlxMmapKey = "VMLINUX_MMAP_SAFETENSORS"
        let tensorKey = "MLX_SAFETENSORS_MMAP_TENSOR_BUFFERS"
        let priorMmap = getenv(mmapKey).map { String(cString: $0) }
        let priorVmlxMmap = getenv(vmlxMmapKey).map { String(cString: $0) }
        let priorTensor = getenv(tensorKey).map { String(cString: $0) }
        setenv(mmapKey, "1", 1)
        setenv(vmlxMmapKey, "1", 1)
        setenv(tensorKey, "1", 1)
        defer {
            if let priorMmap {
                setenv(mmapKey, priorMmap, 1)
            } else {
                unsetenv(mmapKey)
            }
            if let priorVmlxMmap {
                setenv(vmlxMmapKey, priorVmlxMmap, 1)
            } else {
                unsetenv(vmlxMmapKey)
            }
            if let priorTensor {
                setenv(tensorKey, priorTensor, 1)
            } else {
                unsetenv(tensorKey)
            }
        }
        #endif
        return try body()
    }

    func load(
        layerIdx: Int,
        expertIdx: Int,
        projection: StreamingProjection,
        suffix: StreamingSuffix
    ) -> MLXArray? {
        let cacheKey = StreamingTensorCacheKey(
            layerIdx: layerIdx,
            expertIdx: expertIdx,
            projection: projection,
            suffix: suffix)
        if let resident = residentTensor(for: cacheKey) {
            return resident
        }

        guard let layer = index()?.layers[layerIdx] else { return nil }

        if let ref = layer.experts[expertIdx]?.tensors[projection]?[suffix] {
            if let machArray = loadMachTensor(ref: ref, key: cacheKey, shape: ref.shape) {
                return machArray
            }
            if let cached = cachedTensor(for: cacheKey, byteCount: ref.byteCount) {
                MLXPressStreamingProfile.record("tensor.cache_hit", bytes: ref.byteCount)
                return cached
            }
            let mmapArray = MLXPressStreamingProfile.time(
                "tensor.mmap_array",
                bytes: ref.byteCount
            ) { () -> MLXArray? in
                makeMmapArray(
                    from: ref.fileURL,
                    offset: ref.offset,
                    count: ref.byteCount,
                    shape: ref.shape,
                    dtype: ref.dtype)
            }
            if let array = mmapArray {
                return storeTensor(array, key: cacheKey, byteCount: ref.byteCount)
            }
            let data = MLXPressStreamingProfile.time("tensor.read", bytes: ref.byteCount) {
                readBytes(from: ref.fileURL, offset: ref.offset, count: ref.byteCount)
            }
            guard let data else { return nil }
            let array = MLXPressStreamingProfile.time(
                "tensor.array", bytes: ref.byteCount
            ) {
                makeArray(data: data, shape: ref.shape, dtype: ref.dtype)
            }
            guard let array else { return nil }
            return storeTensor(array, key: cacheKey, byteCount: ref.byteCount)
        }

        guard let ref = layer.stacked[projection]?[suffix],
            expertIdx >= 0,
            let expertCount = ref.shape.first,
            expertIdx < expertCount,
            expertCount > 0,
            ref.byteCount % expertCount == 0
        else { return nil }

        let expertByteCount = ref.byteCount / expertCount
        let expertOffset = ref.offset + UInt64(expertIdx * expertByteCount)
        let expertShape = Array(ref.shape.dropFirst())
        guard !expertShape.isEmpty else { return nil }
        let expertRef = StreamingTensorRef(
            name: "\(ref.name)#expert.\(expertIdx)",
            fileURL: ref.fileURL,
            offset: expertOffset,
            byteCount: expertByteCount,
            dtype: ref.dtype,
            shape: expertShape)
        if let machArray = loadMachTensor(ref: expertRef, key: cacheKey, shape: expertShape) {
            return machArray
        }
        if let cached = cachedTensor(for: cacheKey, byteCount: expertByteCount) {
            MLXPressStreamingProfile.record("tensor.cache_hit", bytes: expertByteCount)
            return cached
        }
        let mmapArray = MLXPressStreamingProfile.time(
            "tensor.stacked_mmap_array",
            bytes: expertByteCount
        ) { () -> MLXArray? in
            makeMmapArray(
                from: ref.fileURL,
                offset: expertOffset,
                count: expertByteCount,
                shape: expertShape,
                dtype: ref.dtype)
        }
        if let array = mmapArray {
            return storeTensor(array, key: cacheKey, byteCount: expertByteCount)
        }
        let data = MLXPressStreamingProfile.time("tensor.stacked_read", bytes: expertByteCount) {
            readBytes(from: ref.fileURL, offset: expertOffset, count: expertByteCount)
        }
        guard let data else { return nil }
        let array = MLXPressStreamingProfile.time(
            "tensor.stacked_array", bytes: expertByteCount
        ) {
            makeArray(data: data, shape: expertShape, dtype: ref.dtype)
        }
        guard let array else { return nil }
        return storeTensor(array, key: cacheKey, byteCount: expertByteCount)
    }

    private func residentTensor(for key: StreamingTensorCacheKey) -> MLXArray? {
        lock.lock()
        let entry = residentTensors[key]
        lock.unlock()
        guard let entry else { return nil }
        MLXPressStreamingProfile.record("tensor.resident_hit", bytes: entry.byteCount)
        return entry.array
    }

    private func loadMachTensor(
        ref: StreamingTensorRef,
        key: StreamingTensorCacheKey,
        shape: [Int]
    ) -> MLXArray? {
        guard machActiveTensorsEnabled else { return nil }
        return loadMachBackedTensor(
            ref: ref,
            key: key,
            shape: shape,
            isOffsetSpan: false,
            readProfileName: "tensor.mach_read",
            registerProfileName: "tensor.mach_register",
            hitProfileName: "tensor.mach_hit",
            arrayProfileName: "tensor.mach_array")
    }

    private func loadMachOffsetSpan(
        ref: StreamingTensorRef,
        key: StreamingTensorCacheKey,
        shape: [Int]
    ) -> MLXArray? {
        guard machOffsetSpansEnabled else { return nil }
        return loadMachBackedTensor(
            ref: ref,
            key: key,
            shape: shape,
            isOffsetSpan: true,
            readProfileName: "tensor.mach_offset_read",
            registerProfileName: "tensor.mach_offset_register",
            hitProfileName: "tensor.mach_offset_hit",
            arrayProfileName: "tensor.mach_offset_array")
    }

    private func loadMachBackedTensor(
        ref: StreamingTensorRef,
        key: StreamingTensorCacheKey,
        shape: [Int],
        isOffsetSpan: Bool,
        readProfileName: String,
        registerProfileName: String,
        hitProfileName: String,
        arrayProfileName: String,
        useNoCacheForRegistration: Bool = true
    ) -> MLXArray? {
        guard let dtype = dtype(from: ref.dtype) else { return nil }

        let component = machComponentName(for: key)
        if isOffsetSpan,
           machOffsetSpanBudgetBytes > 0,
           ref.byteCount > machOffsetSpanBudgetBytes
        {
            lock.lock()
            machOffsetBudgetSkips += 1
            lock.unlock()
            MLXPressStreamingProfile.record("tensor.mach_offset_budget_skip", bytes: ref.byteCount)
            return nil
        }

        lock.lock()
        let alreadyRegistered: Bool
        if var entry = machRegisteredTensors[key] {
            tensorCacheClock &+= 1
            entry.lastUse = tensorCacheClock
            machRegisteredTensors[key] = entry
            alreadyRegistered = true
        } else {
            alreadyRegistered = false
        }
        lock.unlock()

        if !alreadyRegistered {
            let evicted: [(StreamingTensorCacheKey, Int)]
            lock.lock()
            evicted = isOffsetSpan
                ? evictMachOffsetSpansToBudgetLocked(pendingByteCount: ref.byteCount, protecting: key)
                : []
            if isOffsetSpan,
               machOffsetSpanBudgetBytes > 0,
               machOffsetSpanRegisteredBytes + ref.byteCount > machOffsetSpanBudgetBytes
            {
                machOffsetBudgetSkips += 1
                lock.unlock()
                recordMachEvictions(evicted)
                MLXPressStreamingProfile.record(
                    "tensor.mach_offset_budget_skip",
                    bytes: ref.byteCount)
                return nil
            }
            machTensorMisses += 1
            machTensorMissBytes += UInt64(max(0, ref.byteCount))
            lock.unlock()
            recordMachEvictions(evicted)

            guard registerMachBackedTensor(
                ref: ref,
                key: key,
                component: component,
                readProfileName: readProfileName,
                useNoCache: useNoCacheForRegistration)
            else {
                return nil
            }
            lock.lock()
            tensorCacheClock &+= 1
            if let old = machRegisteredTensors[key] {
                machRegisteredTensorBytes = max(0, machRegisteredTensorBytes - old.byteCount)
                if old.isOffsetSpan {
                    machOffsetSpanRegisteredBytes = max(
                        0,
                        machOffsetSpanRegisteredBytes - old.byteCount)
                }
            } else {
                machTensorStores += 1
            }
            machRegisteredTensors[key] = StreamingMachTensorEntry(
                byteCount: ref.byteCount,
                lastUse: tensorCacheClock,
                isOffsetSpan: isOffsetSpan)
            machRegisteredTensorBytes += ref.byteCount
            if isOffsetSpan {
                machOffsetSpanRegisteredBytes += ref.byteCount
            }
            lock.unlock()
            MLXPressStreamingProfile.record(registerProfileName, bytes: ref.byteCount)
        } else {
            lock.lock()
            machTensorHits += 1
            machTensorHitBytes += UInt64(max(0, ref.byteCount))
            lock.unlock()
            MLXPressStreamingProfile.record(hitProfileName, bytes: ref.byteCount)
        }

        let start = Date.timeIntervalSinceReferenceDate
        do {
            let array = try machCache.array(
                layer: key.layerIdx,
                expert: key.expertIdx,
                component: component,
                shape: shape,
                dtype: dtype)
            MLXPressStreamingProfile.record(
                arrayProfileName,
                seconds: Date.timeIntervalSinceReferenceDate - start,
                bytes: ref.byteCount)
            return array
        } catch {
            FileHandle.standardError.write(
                Data(
                    "[MLXPressStreaming] Mach tensor acquire failed layer=\(key.layerIdx) expert=\(key.expertIdx) component=\(component) error=\(error)\n"
                        .utf8))
            return nil
        }
    }

    private func evictMachOffsetSpansToBudgetLocked(
        pendingByteCount: Int,
        protecting protectedKey: StreamingTensorCacheKey
    ) -> [(StreamingTensorCacheKey, Int)] {
        guard machOffsetSpanBudgetBytes > 0 else { return [] }
        var evicted: [(StreamingTensorCacheKey, Int)] = []
        while machOffsetSpanRegisteredBytes + pendingByteCount > machOffsetSpanBudgetBytes {
            guard let victim = machRegisteredTensors
                .filter({ $0.key != protectedKey && $0.value.isOffsetSpan })
                .min(by: { lhs, rhs in
                    if lhs.value.lastUse == rhs.value.lastUse {
                        return machComponentName(for: lhs.key) < machComponentName(for: rhs.key)
                    }
                    return lhs.value.lastUse < rhs.value.lastUse
                })
            else { break }
            machRegisteredTensors.removeValue(forKey: victim.key)
            machRegisteredTensorBytes = max(0, machRegisteredTensorBytes - victim.value.byteCount)
            machOffsetSpanRegisteredBytes = max(
                0,
                machOffsetSpanRegisteredBytes - victim.value.byteCount)
            machTensorEvictions += 1
            evicted.append((victim.key, victim.value.byteCount))
        }
        return evicted
    }

    private func recordMachEvictions(_ evicted: [(StreamingTensorCacheKey, Int)]) {
        for (victim, bytes) in evicted {
            _ = machCache.remove(
                layer: victim.layerIdx,
                expert: victim.expertIdx,
                component: machComponentName(for: victim))
            MLXPressStreamingProfile.record("tensor.mach_offset_evict", bytes: bytes)
        }
    }

    private func registerMachBackedTensor(
        ref: StreamingTensorRef,
        key: StreamingTensorCacheKey,
        component: String,
        readProfileName: String,
        useNoCache: Bool
    ) -> Bool {
        #if canImport(Darwin)
            if let fd = cachedDarwinFileDescriptor(for: ref.fileURL, useNoCache: useNoCache) {
                let start = Date.timeIntervalSinceReferenceDate
                do {
                    _ = try machCache.registerFilled(
                        layer: key.layerIdx,
                        expert: key.expertIdx,
                        component: component,
                        byteCount: ref.byteCount,
                        diskURL: ref.fileURL,
                        diskOffset: ref.offset
                    ) { target, count in
                        let ok = preadDarwinFully(
                            fd: fd,
                            into: target,
                            count: count,
                            offset: off_t(ref.offset))
                        if !ok {
                            throw JangPressMachError.mmapFailed(errno == 0 ? EIO : errno)
                        }
                    }
                    MLXPressStreamingProfile.record(
                        readProfileName,
                        seconds: Date.timeIntervalSinceReferenceDate - start,
                        bytes: ref.byteCount)
                    return true
                } catch {
                    FileHandle.standardError.write(
                        Data(
                            "[MLXPressStreaming] Mach tensor direct registration failed layer=\(key.layerIdx) expert=\(key.expertIdx) component=\(component) error=\(error)\n"
                                .utf8))
                    return false
                }
            }
        #endif

        let data = MLXPressStreamingProfile.time(readProfileName, bytes: ref.byteCount) {
            readBytes(from: ref.fileURL, offset: ref.offset, count: ref.byteCount)
        }
        guard let data else { return false }
        do {
            _ = try data.withUnsafeBytes {
                try machCache.register(
                    layer: key.layerIdx,
                    expert: key.expertIdx,
                    component: component,
                    bytes: $0,
                    diskURL: ref.fileURL,
                    diskOffset: ref.offset)
            }
        } catch {
            FileHandle.standardError.write(
                Data(
                    "[MLXPressStreaming] Mach tensor registration failed layer=\(key.layerIdx) expert=\(key.expertIdx) component=\(component) error=\(error)\n"
                        .utf8))
            return false
        }
        return true
    }

    private func machComponentName(for key: StreamingTensorCacheKey) -> String {
        if key.expertIdx < 0 {
            return "\(key.projection.rawValue).\(key.suffix.rawValue).full.\(-key.expertIdx)"
        }
        return "\(key.projection.rawValue).\(key.suffix.rawValue)"
    }

    func releaseMachColdTiles() {
        guard machActiveTensorsEnabled || machOffsetSpansEnabled else { return }
        let released = machCache.releaseColdTiles(compressPercent: machCompressPercent)
        if released > 0 {
            MLXPressStreamingProfile.record("tensor.mach_release_cold")
        }
    }

    private func cachedTensor(
        for key: StreamingTensorCacheKey,
        byteCount: Int
    ) -> MLXArray? {
        guard tensorCacheBudgetBytes > 0 else { return nil }
        lock.lock()
        defer { lock.unlock() }
        guard var entry = tensorCache[key] else {
            tensorCacheMisses += 1
            tensorCacheMissBytes += UInt64(max(0, byteCount))
            return nil
        }
        tensorCacheClock &+= 1
        entry.lastUse = tensorCacheClock
        tensorCache[key] = entry
        tensorCacheHits += 1
        tensorCacheHitBytes += UInt64(max(0, entry.byteCount))
        return entry.array
    }

    private func storeTensor(
        _ array: MLXArray,
        key: StreamingTensorCacheKey,
        byteCount: Int
    ) -> MLXArray {
        guard tensorCacheBudgetBytes > 0, byteCount <= tensorCacheBudgetBytes else {
            return array
        }
        MLXPressStreamingProfile.record("tensor.cache_store", bytes: byteCount)
        lock.lock()
        tensorCacheClock &+= 1
        if let existing = tensorCache[key] {
            tensorCacheBytes -= existing.byteCount
        }
        tensorCache[key] = StreamingTensorCacheEntry(
            array: array,
            byteCount: byteCount,
            lastUse: tensorCacheClock)
        tensorCacheBytes += byteCount
        tensorCacheStores += 1

        evictResidencyToBudgetLocked()
        lock.unlock()
        return array
    }

    private func readStackedSlices(
        from ref: StreamingTensorRef,
        layerIdx: Int,
        expertIndices: [Int],
        projection: StreamingProjection,
        suffix: StreamingSuffix,
        expertByteCount: Int
    ) -> StreamingStackedSliceReadResult? {
        if tensorCacheBudgetBytes > 0 {
            return readStackedSlicesWithResidency(
                from: ref,
                layerIdx: layerIdx,
                expertIndices: expertIndices,
                projection: projection,
                suffix: suffix,
                expertByteCount: expertByteCount)
        }
        #if canImport(Darwin)
            if let data = readStackedSlicesNoCacheDarwin(
                from: ref.fileURL,
                baseOffset: ref.offset,
                expertIndices: expertIndices,
                expertByteCount: expertByteCount)
            {
                return StreamingStackedSliceReadResult(
                    data: data,
                    fileReadBytes: data.count,
                    cacheHitBytes: 0)
            }
        #endif

        guard let handle = try? FileHandle(forReadingFrom: ref.fileURL) else { return nil }
        defer { try? handle.close() }
        var data = Data(count: expertByteCount * expertIndices.count)
        do {
            for (slot, expert) in expertIndices.enumerated() {
                let sourceOffset = ref.offset + UInt64(expert * expertByteCount)
                try handle.seek(toOffset: sourceOffset)
                let chunk = try handle.read(upToCount: expertByteCount) ?? Data()
                guard chunk.count == expertByteCount else { return nil }
                let targetStart = slot * expertByteCount
                let targetEnd = targetStart + expertByteCount
                data.replaceSubrange(targetStart ..< targetEnd, with: chunk)
            }
            return StreamingStackedSliceReadResult(
                data: data,
                fileReadBytes: data.count,
                cacheHitBytes: 0)
        } catch {
            return nil
        }
    }

    private func readStackedSlicesWithResidency(
        from ref: StreamingTensorRef,
        layerIdx: Int,
        expertIndices: [Int],
        projection: StreamingProjection,
        suffix: StreamingSuffix,
        expertByteCount: Int
    ) -> StreamingStackedSliceReadResult? {
        let totalCount = expertByteCount * expertIndices.count
        var data = Data(count: totalCount)
        var fileReadBytes = 0
        var cacheHitBytes = 0

        for (slot, expert) in expertIndices.enumerated() {
            let key = StreamingTensorCacheKey(
                layerIdx: layerIdx,
                expertIdx: expert,
                projection: projection,
                suffix: suffix)
            let sliceData: Data
            if let cached = cachedSliceData(for: key, byteCount: expertByteCount) {
                sliceData = cached
                cacheHitBytes += expertByteCount
            } else {
                let sourceOffset = ref.offset + UInt64(expert * expertByteCount)
                guard let read = readBytes(
                    from: ref.fileURL,
                    offset: sourceOffset,
                    count: expertByteCount)
                else { return nil }
                sliceData = read
                fileReadBytes += expertByteCount
                storeSliceData(read, key: key, byteCount: expertByteCount)
            }
            guard sliceData.count == expertByteCount else { return nil }
            let targetStart = slot * expertByteCount
            let targetEnd = targetStart + expertByteCount
            data.replaceSubrange(targetStart ..< targetEnd, with: sliceData)
        }

        return StreamingStackedSliceReadResult(
            data: data,
            fileReadBytes: fileReadBytes,
            cacheHitBytes: cacheHitBytes)
    }

    private func cachedSliceData(
        for key: StreamingTensorCacheKey,
        byteCount: Int
    ) -> Data? {
        guard tensorCacheBudgetBytes > 0 else { return nil }
        lock.lock()
        defer { lock.unlock() }
        guard var entry = sliceCache[key], entry.byteCount == byteCount else {
            sliceCacheMisses += 1
            sliceCacheMissBytes += UInt64(max(0, byteCount))
            return nil
        }
        tensorCacheClock &+= 1
        entry.lastUse = tensorCacheClock
        sliceCache[key] = entry
        sliceCacheHits += 1
        sliceCacheHitBytes += UInt64(max(0, byteCount))
        return entry.data
    }

    private func storeSliceData(
        _ data: Data,
        key: StreamingTensorCacheKey,
        byteCount: Int
    ) {
        guard tensorCacheBudgetBytes > 0, byteCount <= tensorCacheBudgetBytes else {
            return
        }
        MLXPressStreamingProfile.record("tensor.active_slice_cache_store", bytes: byteCount)
        lock.lock()
        tensorCacheClock &+= 1
        if let existing = sliceCache[key] {
            sliceCacheBytes -= existing.byteCount
        }
        sliceCache[key] = StreamingSliceCacheEntry(
            data: data,
            byteCount: byteCount,
            lastUse: tensorCacheClock)
        sliceCacheBytes += byteCount
        sliceCacheStores += 1
        evictResidencyToBudgetLocked()
        lock.unlock()
    }

    private func evictResidencyToBudgetLocked() {
        while tensorCacheBytes + sliceCacheBytes > tensorCacheBudgetBytes {
            let tensorVictim = tensorCache.min { $0.value.lastUse < $1.value.lastUse }
            let sliceVictim = sliceCache.min { $0.value.lastUse < $1.value.lastUse }

            if let tensorVictim,
                sliceVictim.map({ tensorVictim.value.lastUse <= $0.value.lastUse }) ?? true
            {
                tensorCache.removeValue(forKey: tensorVictim.key)
                tensorCacheBytes -= tensorVictim.value.byteCount
                tensorCacheEvictions += 1
                continue
            }

            if let sliceVictim {
                sliceCache.removeValue(forKey: sliceVictim.key)
                sliceCacheBytes -= sliceVictim.value.byteCount
                sliceCacheEvictions += 1
                continue
            }

            break
        }

        while activeBankCacheBytes > bankCacheBudgetBytes {
            guard let victim = activeBankCache.min(by: { $0.value.lastUse < $1.value.lastUse })
            else { break }
            activeBankCache.removeValue(forKey: victim.key)
            activeBankCacheBytes -= victim.value.byteCount
            activeBankCacheEvictions += 1
        }

        while offsetSpanCacheBytes > offsetSpanCacheBudgetBytes {
            guard let victim = offsetSpanCache.min(by: { $0.value.lastUse < $1.value.lastUse })
            else { break }
            offsetSpanCache.removeValue(forKey: victim.key)
            offsetSpanCacheBytes -= victim.value.byteCount
            offsetSpanCacheEvictions += 1
        }
    }

    func activeSliceResidencySnapshot() -> MLXPressActiveSliceResidencySnapshot {
        lock.lock()
        let totalHits = tensorCacheHits + sliceCacheHits + activeBankCacheHits
            + offsetSpanCacheHits + machTensorHits
        let totalMisses = tensorCacheMisses + sliceCacheMisses + activeBankCacheMisses
            + offsetSpanCacheMisses + machTensorMisses
        let totalHitBytes = tensorCacheHitBytes + sliceCacheHitBytes + activeBankCacheHitBytes
            + offsetSpanCacheHitBytes + machTensorHitBytes
        let totalMissBytes = tensorCacheMissBytes + sliceCacheMissBytes + activeBankCacheMissBytes
            + offsetSpanCacheMissBytes + machTensorMissBytes
        let totalStores = tensorCacheStores + sliceCacheStores + activeBankCacheStores
            + offsetSpanCacheStores + machTensorStores
        let totalEvictions = tensorCacheEvictions + sliceCacheEvictions + activeBankCacheEvictions
            + offsetSpanCacheEvictions + machTensorEvictions
        let snapshot = MLXPressActiveSliceResidencySnapshot(
            budgetBytes: UInt64(max(
                0,
                tensorCacheBudgetBytes + bankCacheBudgetBytes + offsetSpanCacheBudgetBytes
                    + machOffsetSpanBudgetBytes)),
            tensorBudgetBytes: UInt64(max(
                0,
                tensorCacheBudgetBytes + offsetSpanCacheBudgetBytes + machOffsetSpanBudgetBytes)),
            bankBudgetBytes: UInt64(max(0, bankCacheBudgetBytes)),
            residentBytes: UInt64(max(
                0,
                tensorCacheBytes + residentTensorBytes + machRegisteredTensorBytes + sliceCacheBytes
                    + activeBankCacheBytes + offsetSpanCacheBytes)),
            tensorResidentBytes: UInt64(max(
                0,
                tensorCacheBytes + residentTensorBytes + machRegisteredTensorBytes
                    + offsetSpanCacheBytes)),
            sliceResidentBytes: UInt64(max(0, sliceCacheBytes)),
            bankResidentBytes: UInt64(max(0, activeBankCacheBytes)),
            entries: tensorCache.count + residentTensors.count + machRegisteredTensors.count
                + sliceCache.count + activeBankCache.count + offsetSpanCache.count,
            tensorEntries: tensorCache.count + residentTensors.count + machRegisteredTensors.count
                + offsetSpanCache.count,
            sliceEntries: sliceCache.count,
            bankEntries: activeBankCache.count,
            hits: totalHits,
            tensorHits: tensorCacheHits + offsetSpanCacheHits + machTensorHits,
            sliceHits: sliceCacheHits,
            bankHits: activeBankCacheHits,
            misses: totalMisses,
            tensorMisses: tensorCacheMisses + offsetSpanCacheMisses + machTensorMisses,
            sliceMisses: sliceCacheMisses,
            bankMisses: activeBankCacheMisses,
            hitBytes: totalHitBytes,
            tensorHitBytes: tensorCacheHitBytes + offsetSpanCacheHitBytes + machTensorHitBytes,
            sliceHitBytes: sliceCacheHitBytes,
            bankHitBytes: activeBankCacheHitBytes,
            missBytes: totalMissBytes,
            tensorMissBytes: tensorCacheMissBytes + offsetSpanCacheMissBytes + machTensorMissBytes,
            sliceMissBytes: sliceCacheMissBytes,
            bankMissBytes: activeBankCacheMissBytes,
            stores: totalStores,
            tensorStores: tensorCacheStores + offsetSpanCacheStores + machTensorStores,
            sliceStores: sliceCacheStores,
            bankStores: activeBankCacheStores,
            evictions: totalEvictions,
            tensorEvictions: tensorCacheEvictions + offsetSpanCacheEvictions + machTensorEvictions,
            sliceEvictions: sliceCacheEvictions,
            bankEvictions: activeBankCacheEvictions)
        lock.unlock()
        return snapshot
    }

    private func readBytes(from url: URL, offset: UInt64, count: Int) -> Data? {
        #if canImport(Darwin)
            if let data = readBytesNoCacheDarwin(from: url, offset: offset, count: count) {
                return data
            }
        #endif

        guard let handle = try? FileHandle(forReadingFrom: url) else { return nil }
        defer { try? handle.close() }
        do {
            try handle.seek(toOffset: offset)
            let data = try handle.read(upToCount: count) ?? Data()
            return data.count == count ? data : nil
        } catch {
            return nil
        }
    }

    private func mmapActiveTensorsEnabled() -> Bool {
        let env = ProcessInfo.processInfo.environment
        let raw =
            env["MLXPRESS_STREAMING_MMAP_ACTIVE_TENSORS"]?
            .lowercased()
            ?? env["JANGPRESS_STREAMING_MMAP_ACTIVE_TENSORS"]?.lowercased()
            ?? "0"
        return raw == "1" || raw == "true" || raw == "yes" || raw == "on"
    }

    private func makeMmapArray(
        from url: URL,
        offset: UInt64,
        count: Int,
        shape: [Int],
        dtype safetensorsDType: String
    ) -> MLXArray? {
        guard mmapActiveTensorsEnabled(),
              count > 0,
              let dtype = dtype(from: safetensorsDType)
        else { return nil }
        var cShape = shape.map(Int32.init)
        var array = mlx_array_new()
        let rc = url.withUnsafeFileSystemRepresentation { pathPtr -> Int32 in
            guard let pathPtr else { return 1 }
            return mlx_array_new_mmap_file_region(
                &array,
                pathPtr,
                offset,
                count,
                &cShape,
                Int32(cShape.count),
                dtype.cmlxDtype)
        }
        guard rc == 0 else {
            mlx_array_free(array)
            return nil
        }
        return MLXArray(array)
    }

    #if canImport(Darwin)
        private func cachedDarwinFileDescriptor(for url: URL) -> Int32? {
            cachedDarwinFileDescriptor(for: url, useNoCache: true)
        }

        private func cachedDarwinFileDescriptor(for url: URL, useNoCache: Bool) -> Int32? {
            let key = url.path
            let cacheKey = "\(useNoCache ? "nocache" : "cache"):\(key)"
            lock.lock()
            if let cached = cachedFileDescriptors[cacheKey] {
                lock.unlock()
                return cached
            }
            lock.unlock()

            return url.withUnsafeFileSystemRepresentation { pathPtr -> Int32? in
                guard let pathPtr else { return nil }
                let opened = Darwin.open(pathPtr, O_RDONLY)
                guard opened >= 0 else { return nil }

                if useNoCache, shouldUseFNoCacheForDarwinReads() {
                    // Kimi-scale bundles are larger than the RAM warming budget;
                    // keep fallback streaming from filling the OS file cache.
                    // Full Mach routed-span residency uses a cached descriptor
                    // because those reads are explicitly meant to become
                    // compressible resident pages rather than one-shot direct I/O.
                    _ = Darwin.fcntl(opened, F_NOCACHE, 1)
                }

                lock.lock()
                if let existing = cachedFileDescriptors[cacheKey] {
                    lock.unlock()
                    Darwin.close(opened)
                    return existing
                }
                cachedFileDescriptors[cacheKey] = opened
                lock.unlock()
                return opened
            }
        }

        private func shouldUseFNoCacheForDarwinReads() -> Bool {
            let built = index()
            let useNoCache = mlXPressStreamingUseFNoCache(for: built)
            lock.lock()
            let shouldLog = !readPolicyLogged
            if shouldLog {
                readPolicyLogged = true
            }
            lock.unlock()
            if shouldLog, built == nil {
                FileHandle.standardError.write(
                    Data(
                        "[MLXPressStreaming] active-expert read cache policy=F_NOCACHE streamable=unknown\n"
                            .utf8))
            }
            return useNoCache
        }

        private func readBytesNoCacheDarwin(from url: URL, offset: UInt64, count: Int) -> Data? {
            guard count > 0 else { return Data() }
            guard let fd = cachedDarwinFileDescriptor(for: url) else { return nil }
            var data = Data(count: count)
            let bytesRead = data.withUnsafeMutableBytes { buffer -> Int in
                guard let base = buffer.baseAddress else { return -1 }
                var total = 0
                while total < count {
                    let readCount = min(count - total, 64 * 1024 * 1024)
                    let currentOffset = off_t(offset) + off_t(total)
                    let readNow = Darwin.pread(
                        fd,
                        base.advanced(by: total),
                        readCount,
                        currentOffset)
                    if readNow < 0 {
                        return -1
                    }
                    if readNow == 0 {
                        return total
                    }
                    total += readNow
                }
                return total
            }
            return bytesRead == count ? data : nil
        }

        private func preadDarwinFully(
            fd: Int32,
            into target: UnsafeMutableRawPointer,
            count: Int,
            offset: off_t
        ) -> Bool {
            var total = 0
            while total < count {
                let readCount = min(count - total, 64 * 1024 * 1024)
                let readNow = Darwin.pread(
                    fd,
                    target.advanced(by: total),
                    readCount,
                    offset + off_t(total))
                if readNow <= 0 {
                    return false
                }
                total += readNow
            }
            return true
        }

        private func readStackedSlicesNoCacheDarwin(
            from url: URL,
            baseOffset: UInt64,
            expertIndices: [Int],
            expertByteCount: Int
        ) -> Data? {
            guard expertByteCount > 0 else { return Data() }
            guard let fd = cachedDarwinFileDescriptor(for: url) else { return nil }
            let totalCount = expertByteCount * expertIndices.count
            if !mlXPressStreamingSourceOrderReadsEnabled() {
                var data = Data(count: totalCount)
                let success = data.withUnsafeMutableBytes { buffer -> Bool in
                    guard let base = buffer.baseAddress else { return false }
                    for (slot, expert) in expertIndices.enumerated() {
                        let sourceOffset = off_t(baseOffset) + off_t(expert * expertByteCount)
                        let targetBase = base.advanced(by: slot * expertByteCount)
                        if !preadDarwinFully(
                            fd: fd,
                            into: targetBase,
                            count: expertByteCount,
                            offset: sourceOffset)
                        {
                            return false
                        }
                    }
                    return true
                }
                return success ? data : nil
            }

            let readPlan = makeStackedSliceReadPlan(expertIndices: expertIndices)
            var data = Data(count: totalCount)
            let success = data.withUnsafeMutableBytes { buffer -> Bool in
                guard let base = buffer.baseAddress else { return false }
                for range in readPlan {
                    let rangeByteCount = range.expertCount * expertByteCount
                    let sourceOffset =
                        off_t(baseOffset) + off_t(range.firstExpertIdx * expertByteCount)
                    let profileStart = Date.timeIntervalSinceReferenceDate

                    if range.expertCount == 1,
                        range.targets.count == 1,
                        range.targets[0].slots.count == 1,
                        let slot = range.targets[0].slots.first
                    {
                        let targetBase = base.advanced(by: slot * expertByteCount)
                        if !preadDarwinFully(
                            fd: fd,
                            into: targetBase,
                            count: rangeByteCount,
                            offset: sourceOffset)
                        {
                            return false
                        }
                        MLXPressStreamingProfile.record(
                            "tensor.stacked_bank_read_range",
                            seconds: Date.timeIntervalSinceReferenceDate - profileStart,
                            bytes: rangeByteCount)
                        continue
                    }

                    var rangeData = Data(count: rangeByteCount)
                    let rangeRead = rangeData.withUnsafeMutableBytes { rangeBuffer -> Bool in
                        guard let rangeBase = rangeBuffer.baseAddress else { return false }
                        return preadDarwinFully(
                            fd: fd,
                            into: rangeBase,
                            count: rangeByteCount,
                            offset: sourceOffset)
                    }
                    guard rangeRead else { return false }
                    let copied = rangeData.withUnsafeBytes { rangeBuffer -> Bool in
                        guard let rangeBase = rangeBuffer.baseAddress else { return false }
                        for target in range.targets {
                            let rangeExpertOffset =
                                (target.expertIdx - range.firstExpertIdx) * expertByteCount
                            for slot in target.slots {
                                base
                                    .advanced(by: slot * expertByteCount)
                                    .copyMemory(
                                        from: rangeBase.advanced(by: rangeExpertOffset),
                                        byteCount: expertByteCount)
                            }
                        }
                        return true
                    }
                    if !copied {
                        return false
                    }
                    MLXPressStreamingProfile.record(
                        "tensor.stacked_bank_read_range",
                        seconds: Date.timeIntervalSinceReferenceDate - profileStart,
                        bytes: rangeByteCount)
                }
                return true
            }
            return success ? data : nil
        }
    #endif

    private func dtype(from safetensorsDType: String) -> DType? {
        switch safetensorsDType {
        case "U8":
            return .uint8
        case "U32":
            return .uint32
        case "I32":
            return .int32
        case "F16":
            return .float16
        case "BF16":
            return .bfloat16
        case "F32":
            return .float32
        default:
            return nil
        }
    }

    private func elementByteSize(from safetensorsDType: String) -> Int? {
        switch safetensorsDType {
        case "U8":
            return 1
        case "F16", "BF16":
            return 2
        case "U32", "I32", "F32":
            return 4
        default:
            return nil
        }
    }

    private func makeArray(data: Data, shape: [Int], dtype safetensorsDType: String) -> MLXArray? {
        guard let dtype = dtype(from: safetensorsDType) else { return nil }
        return MLXArray(data, shape, dtype: dtype)
    }
}

public final class StreamingTurboQuantSwitchGLU: TurboQuantSwitchGLU {
    private let layerIdx: Int
    private let evaluateEachLayer: Bool
    private let tokenChunkSize: Int
    private let scoredDownKernelsEnabled: Bool

    public init(
        inputDims: Int,
        hiddenDims: Int,
        numExperts: Int,
        gateUpBits: Int,
        downBits: Int,
        seed: Int = 42,
        swigluLimit: Float = 0.0,
        layerIdx: Int
    ) {
        self.layerIdx = layerIdx
        self.evaluateEachLayer = mlXPressStreamingShouldEvaluateLayer(layerIdx)
        self.tokenChunkSize = mlXPressStreamingTokenChunkSize()
        self.scoredDownKernelsEnabled = mlXPressStreamingScoredDownKernelsEnabled()
        super.init(
            inputDims: inputDims,
            hiddenDims: hiddenDims,
            numExperts: numExperts,
            gateUpBits: gateUpBits,
            downBits: downBits,
            seed: seed,
            swigluLimit: swigluLimit)
        _ = JANGTQStreamingExpertStore.shared.index()
    }

    public override func callAsFunction(_ x: MLXArray, _ indices: MLXArray) -> MLXArray {
        let batchTokens = x.size / inputDims
        let kSlots = indices.dim(-1)
        let xFlat = x.reshaped([batchTokens, inputDims])
        let indicesFlat = indices.reshaped([batchTokens, kSlots])
        let useOffsetDispatch = JANGTQStreamingExpertStore.shared.canUseOffsetDispatch(
            layerIdx: layerIdx)
        let shouldReadOffsetIndices = useOffsetDispatch
            && mlXPressStreamingOffsetActiveShardFilterEnabled()
        let allIndexValues: [Int]? = useOffsetDispatch && !shouldReadOffsetIndices
            ? nil
            : MLXPressStreamingProfile.time("router.indices_readback") {
                indicesFlat.reshaped([-1]).asArray(Int32.self).map(Int.init)
            }
        let chunkSize = max(1, tokenChunkSize)
        if batchTokens <= chunkSize {
            let chunk = callChunk(
                xFlat: xFlat,
                indicesFlat: indicesFlat,
                indexValues: allIndexValues,
                tokenCount: batchTokens,
                kSlots: kSlots)
            var outShape = indices.shape
            outShape.append(inputDims)
            return chunk.reshaped(outShape)
        }

        var chunks: [MLXArray] = []
        chunks.reserveCapacity((batchTokens + chunkSize - 1) / chunkSize)
        var start = 0
        while start < batchTokens {
            let end = min(start + chunkSize, batchTokens)
            let valueStart = start * kSlots
            let valueEnd = end * kSlots
            let indexValues = allIndexValues.map {
                Array($0[valueStart ..< valueEnd])
            }
            chunks.append(
                callChunk(
                    xFlat: xFlat[start ..< end, 0...],
                    indicesFlat: indicesFlat[start ..< end, 0...],
                    indexValues: indexValues,
                    tokenCount: end - start,
                    kSlots: kSlots))
            start = end
        }
        let joined = concatenated(chunks, axis: 0)
        var outShape = indices.shape
        outShape.append(inputDims)
        return joined.reshaped(outShape)
    }

    public func reduced(_ x: MLXArray, indices: MLXArray, scores: MLXArray) -> MLXArray {
        let batchTokens = x.size / inputDims
        let kSlots = indices.dim(-1)
        let xFlat = x.reshaped([batchTokens, inputDims])
        let indicesFlat = indices.reshaped([batchTokens, kSlots])
        let scoresFlat = stopGradient(scores.reshaped([batchTokens, kSlots]))
        let useOffsetDispatch = JANGTQStreamingExpertStore.shared.canUseOffsetDispatch(
            layerIdx: layerIdx)
        let shouldReadOffsetIndices = useOffsetDispatch
            && mlXPressStreamingOffsetActiveShardFilterEnabled()
        let allIndexValues: [Int]? = useOffsetDispatch && !shouldReadOffsetIndices
            ? nil
            : MLXPressStreamingProfile.time("router.indices_readback") {
                indicesFlat.reshaped([-1]).asArray(Int32.self).map(Int.init)
            }
        if let allIndexValues, !allIndexValues.isEmpty {
            MLXPressActiveExpertTrace.record(
                layerIdx: layerIdx,
                expertIndices: allIndexValues,
                tokenCount: batchTokens,
                kSlots: kSlots)
        }
        mlXPressStreamingMaterializeIfNeeded(
            scoresFlat,
            force: evaluateEachLayer,
            layerIdx: layerIdx,
            phase: "router_scores",
            profileName: "router.scores_eval")
        let chunkSize = mlXPressStreamingReduceTokenChunkSize()
        let fastReduceMaxTokens = mlXPressStreamingFastReduceMaxTokens()

        func reduceChunkBatchedTopK(start: Int, end: Int) -> MLXArray {
            let tokenCount = end - start
            let valueStart = start * kSlots
            let valueEnd = end * kSlots
            let indexValues = allIndexValues.map {
                Array($0[valueStart ..< valueEnd])
            }
            if useOffsetDispatch && scoredDownKernelsEnabled {
                return MLXPressStreamingProfile.time("reduce.call_chunk_scored") {
                    callChunkScoredOffsets(
                        xFlat: xFlat[start ..< end, 0...],
                        indicesFlat: indicesFlat[start ..< end, 0...],
                        scoresFlat: scoresFlat[start ..< end, 0...],
                        activeExperts: indexValues.map { Set($0) },
                        slotExperts: tokenCount == 1 ? indexValues : nil,
                        tokenCount: tokenCount,
                        kSlots: kSlots)
                }
            }
            let expertOutput = MLXPressStreamingProfile.time("reduce.call_chunk") {
                callChunk(
                    xFlat: xFlat[start ..< end, 0...],
                    indicesFlat: indicesFlat[start ..< end, 0...],
                    indexValues: indexValues,
                    tokenCount: tokenCount,
                    kSlots: kSlots)
            }
            let reduced = MLXPressStreamingProfile.time("reduce.score_sum_build") {
                let scoreChunk = scoresFlat[start ..< end, 0...].asType(expertOutput.dtype)
                return (expertOutput * scoreChunk[.ellipsis, .newAxis]).sum(axis: -2)
            }
            let materialized = stopGradient(reduced)
            mlXPressStreamingMaterializeIfNeeded(
                materialized,
                force: evaluateEachLayer,
                layerIdx: layerIdx,
                phase: "reduce_score_sum",
                profileName: "reduce.score_sum_eval")
            return materialized
        }

        func reduceChunkSerialSlots(start: Int, end: Int) -> MLXArray {
            guard let allIndexValues else {
                return reduceChunkBatchedTopK(start: start, end: end)
            }
            let tokenCount = end - start
            let xChunk = xFlat[start ..< end, 0...]
            var accumulated: MLXArray?
            for slot in 0 ..< kSlots {
                var slotIndexValues: [Int] = []
                slotIndexValues.reserveCapacity(tokenCount)
                for token in start ..< end {
                    slotIndexValues.append(allIndexValues[token * kSlots + slot])
                }
                let expertOutput = MLXPressStreamingProfile.time("reduce.call_chunk") {
                    callChunk(
                        xFlat: xChunk,
                        indicesFlat: indicesFlat[start ..< end, slot ..< (slot + 1)],
                        indexValues: slotIndexValues,
                        tokenCount: tokenCount,
                        kSlots: 1)
                }
                let reducedSlot = MLXPressStreamingProfile.time("reduce.score_sum_build") {
                    let scoreChunk = scoresFlat[start ..< end, slot ..< (slot + 1)]
                        .asType(expertOutput.dtype)
                    return (expertOutput * scoreChunk[.ellipsis, .newAxis])
                        .sum(axis: -2)
                }
                let next = accumulated.map { $0 + reducedSlot } ?? reducedSlot
                mlXPressStreamingMaterializeIfNeeded(
                    next,
                    force: evaluateEachLayer,
                    layerIdx: layerIdx,
                    phase: "reduce_score_sum",
                    profileName: "reduce.score_sum_eval")
                accumulated = stopGradient(next)
            }
            return accumulated ?? MLXArray.zeros([tokenCount, inputDims])
        }

        func reduceChunk(start: Int, end: Int) -> MLXArray {
            let tokenCount = end - start
            if useOffsetDispatch {
                return reduceChunkBatchedTopK(start: start, end: end)
            }
            if fastReduceMaxTokens > 0 && tokenCount <= fastReduceMaxTokens {
                return reduceChunkBatchedTopK(start: start, end: end)
            }
            return reduceChunkSerialSlots(start: start, end: end)
        }

        if batchTokens <= chunkSize {
            return reduceChunk(start: 0, end: batchTokens).reshaped(x.shape)
        }

        var chunks: [MLXArray] = []
        chunks.reserveCapacity((batchTokens + chunkSize - 1) / chunkSize)
        var start = 0
        while start < batchTokens {
            let end = min(start + chunkSize, batchTokens)
            chunks.append(reduceChunk(start: start, end: end))
            start = end
        }
        return concatenated(chunks, axis: 0).reshaped(x.shape)
    }

    private func callChunkScoredOffsets(
        xFlat: MLXArray,
        indicesFlat: MLXArray,
        scoresFlat: MLXArray,
        activeExperts: Set<Int>? = nil,
        slotExperts: [Int]? = nil,
        tokenCount: Int,
        kSlots: Int
    ) -> MLXArray {
        let rhsIndices = indicesFlat.asType(.uint32).reshaped([-1])

        let signsIn = JANGTQRuntimeCache.shared.signs(inFeatures: inputDims, seed: mxtqSeed)
        let signsDn = JANGTQRuntimeCache.shared.signs(inFeatures: hiddenDims, seed: mxtqSeed)
        let cbGate = JANGTQRuntimeCache.shared.codebook(inFeatures: inputDims, bits: gateUpBits)
        let cbDown = JANGTQRuntimeCache.shared.codebook(inFeatures: hiddenDims, bits: downBits)
        guard let signsIn, let signsDn, let cbGate, let cbDown else {
            let missing = [
                signsIn == nil ? "signs.\(inputDims).\(mxtqSeed)" : nil,
                signsDn == nil ? "signs.\(hiddenDims).\(mxtqSeed)" : nil,
                cbGate == nil ? "codebook.\(inputDims).\(gateUpBits)" : nil,
                cbDown == nil ? "codebook.\(hiddenDims).\(downBits)" : nil,
            ].compactMap { $0 }.joined(separator: ", ")
            fatalError(
                "[MLXPressStreaming] missing JANGTQ sidecar array(s) for layer "
                    + "\(layerIdx): \(missing)")
        }

        if tokenCount == 1,
            kSlots > 0,
            kSlots <= 8,
            let slotExperts,
            slotExperts.count == kSlots,
            let slotOutput = callChunkScoredSlotOffsets(
                xFlat: xFlat,
                scoresFlat: scoresFlat,
                slotExperts: slotExperts,
                signsIn: signsIn,
                signsDn: signsDn,
                cbGate: cbGate,
                cbDown: cbDown,
                kSlots: kSlots)
        {
            return slotOutput
        }

        let gatePackedSpans = JANGTQStreamingExpertStore.shared.loadOffsetSpans(
            layerIdx: layerIdx,
            projection: .gate,
            suffix: .packed,
            activeExperts: activeExperts)
        let gateNormSpans = JANGTQStreamingExpertStore.shared.loadOffsetSpans(
            layerIdx: layerIdx,
            projection: .gate,
            suffix: .norms,
            activeExperts: activeExperts)
        let upPackedSpans = JANGTQStreamingExpertStore.shared.loadOffsetSpans(
            layerIdx: layerIdx,
            projection: .up,
            suffix: .packed,
            activeExperts: activeExperts)
        let upNormSpans = JANGTQStreamingExpertStore.shared.loadOffsetSpans(
            layerIdx: layerIdx,
            projection: .up,
            suffix: .norms,
            activeExperts: activeExperts)
        let gateUpGroups = mlXPressStreamingGateUpOffsetSpanGroups(
            gatePacked: gatePackedSpans,
            gateNorms: gateNormSpans,
            upPacked: upPackedSpans,
            upNorms: upNormSpans)
        guard !gateUpGroups.isEmpty else {
            fatalError(
                "[MLXPressStreaming] missing offset-addressed gate/up tensors for layer \(layerIdx)"
            )
        }

        let xAct = MLXPressStreamingProfile.time("gateup.total") {
            let built = MLXPressStreamingProfile.time("gateup.offset_build") {
                let xRot = JANGTQKernels.hadamardRotate(
                    xFlat,
                    signs: signsIn,
                    dim: inputDims)
                let partials = gateUpGroups.map { group -> MLXArray in
                    return JANGTQKernels.fusedGateUpSwiGLUOffsets(
                        xRot: xRot,
                        packedGate: group.gatePacked.array,
                        packedGateOffsets: group.gatePacked.offsets,
                        normsGate: group.gateNorms.array,
                        normsGateOffsets: group.gateNorms.offsets,
                        packedUp: group.upPacked.array,
                        packedUpOffsets: group.upPacked.offsets,
                        normsUp: group.upNorms.array,
                        normsUpOffsets: group.upNorms.offsets,
                        codebook: cbGate,
                        rhsIndices: rhsIndices,
                        batchTokens: tokenCount,
                        K: kSlots,
                        inFeatures: inputDims,
                        outFeatures: hiddenDims,
                        bits: gateUpBits,
                        swigluLimit: swigluLimit)
                }
                return partials.dropFirst().reduce(partials[0]) { $0 + $1 }
            }
            let materialized = stopGradient(built)
            mlXPressStreamingMaterializeIfNeeded(
                materialized,
                force: evaluateEachLayer,
                layerIdx: layerIdx,
                phase: "gateup",
                profileName: "gateup.eval")
            return materialized
        }

        let downPackedSpans = JANGTQStreamingExpertStore.shared.loadOffsetSpans(
            layerIdx: layerIdx,
            projection: .down,
            suffix: .packed,
            activeExperts: activeExperts)
        let downNormSpans = JANGTQStreamingExpertStore.shared.loadOffsetSpans(
            layerIdx: layerIdx,
            projection: .down,
            suffix: .norms,
            activeExperts: activeExperts)
        let downGroups = mlXPressStreamingDownOffsetSpanGroups(
            packed: downPackedSpans,
            norms: downNormSpans)
        guard !downGroups.isEmpty else {
            fatalError(
                "[MLXPressStreaming] missing offset-addressed down tensors for layer \(layerIdx)"
            )
        }

        let out = MLXPressStreamingProfile.time("down.offset_scored_build") {
            let xActRot = JANGTQKernels.hadamardRotate(xAct, signs: signsDn, dim: hiddenDims)
            let scoreValues = scoresFlat.asType(.float32).reshaped([-1])
            let partials = downGroups.map { group -> MLXArray in
                return JANGTQKernels.gatherTQTopKOffsetsScored(
                    xRot: xActRot,
                    packed: group.packed.array,
                    packedOffsets: group.packed.offsets,
                    norms: group.norms.array,
                    normOffsets: group.norms.offsets,
                    codebook: cbDown,
                    rhsIndices: rhsIndices,
                    scores: scoreValues,
                    batchTokens: tokenCount,
                    K: kSlots,
                    inFeatures: hiddenDims,
                    outFeatures: inputDims,
                    bits: downBits)
            }
            return partials.dropFirst().reduce(partials[0]) { $0 + $1 }.asType(xFlat.dtype)
        }
        let materialized = stopGradient(out)
        mlXPressStreamingMaterializeIfNeeded(
            materialized,
            force: evaluateEachLayer,
            layerIdx: layerIdx,
            phase: "down_offset_scored",
            profileName: "down.offset_scored_eval",
            releaseMachColdTiles: true)
        return materialized
    }

    private func callChunkScoredSlotOffsets(
        xFlat: MLXArray,
        scoresFlat: MLXArray,
        slotExperts: [Int],
        signsIn: MLXArray,
        signsDn: MLXArray,
        cbGate: MLXArray,
        cbDown: MLXArray,
        kSlots: Int
    ) -> MLXArray? {
        let activeExperts = Set(slotExperts)
        let gatePacked = exactSlotSpans(
            layerIdx: layerIdx,
            projection: .gate,
            suffix: .packed,
            activeExperts: activeExperts,
            slotExperts: slotExperts)
        let gateNorms = exactSlotSpans(
            layerIdx: layerIdx,
            projection: .gate,
            suffix: .norms,
            activeExperts: activeExperts,
            slotExperts: slotExperts)
        let upPacked = exactSlotSpans(
            layerIdx: layerIdx,
            projection: .up,
            suffix: .packed,
            activeExperts: activeExperts,
            slotExperts: slotExperts)
        let upNorms = exactSlotSpans(
            layerIdx: layerIdx,
            projection: .up,
            suffix: .norms,
            activeExperts: activeExperts,
            slotExperts: slotExperts)
        guard let gatePacked, let gateNorms, let upPacked, let upNorms else {
            MLXPressStreamingProfile.record("slot_offsets_gateup_missing_exact_span")
            return nil
        }

        let xAct = MLXPressStreamingProfile.time("gateup.slot_offsets_total") {
            let built = MLXPressStreamingProfile.time("gateup.slot_offsets_build") {
                let xRot = JANGTQKernels.hadamardRotate(
                    xFlat,
                    signs: signsIn,
                    dim: inputDims)
                return JANGTQKernels.fusedGateUpSwiGLUSlots8(
                    xRot: xRot,
                    packedGate: gatePacked.map(\.array),
                    normsGate: gateNorms.map(\.array),
                    packedUp: upPacked.map(\.array),
                    normsUp: upNorms.map(\.array),
                    codebook: cbGate,
                    batchTokens: 1,
                    K: kSlots,
                    inFeatures: inputDims,
                    outFeatures: hiddenDims,
                    bits: gateUpBits,
                    swigluLimit: swigluLimit)
            }
            let materialized = stopGradient(built)
            mlXPressStreamingMaterializeIfNeeded(
                materialized,
                force: evaluateEachLayer,
                layerIdx: layerIdx,
                phase: "gateup_slot_offsets",
                profileName: "gateup.slot_offsets_eval")
            return materialized
        }

        let downPacked = exactSlotSpans(
            layerIdx: layerIdx,
            projection: .down,
            suffix: .packed,
            activeExperts: activeExperts,
            slotExperts: slotExperts)
        let downNorms = exactSlotSpans(
            layerIdx: layerIdx,
            projection: .down,
            suffix: .norms,
            activeExperts: activeExperts,
            slotExperts: slotExperts)
        guard let downPacked, let downNorms else {
            MLXPressStreamingProfile.record("slot_offsets_down_missing_exact_span")
            return nil
        }

        let out = MLXPressStreamingProfile.time("down.slot_offsets_scored_build") {
            let xActRot = JANGTQKernels.hadamardRotate(xAct, signs: signsDn, dim: hiddenDims)
            let scoreValues = scoresFlat.asType(.float32).reshaped([-1])
            return JANGTQKernels.gatherTQTopKSlots8Scored(
                xRot: xActRot,
                packed: downPacked.map(\.array),
                norms: downNorms.map(\.array),
                codebook: cbDown,
                scores: scoreValues,
                batchTokens: 1,
                K: kSlots,
                inFeatures: hiddenDims,
                outFeatures: inputDims,
                bits: downBits)
                .asType(xFlat.dtype)
        }
        let materialized = stopGradient(out)
        mlXPressStreamingMaterializeIfNeeded(
            materialized,
            force: evaluateEachLayer,
            layerIdx: layerIdx,
            phase: "down_slot_offsets_scored",
            profileName: "down.slot_offsets_scored_eval",
            releaseMachColdTiles: true)
        MLXPressStreamingProfile.record("slot_offsets_decode_fast_path")
        return materialized
    }

    private func exactSlotSpans(
        layerIdx: Int,
        projection: StreamingProjection,
        suffix: StreamingSuffix,
        activeExperts: Set<Int>,
        slotExperts: [Int]
    ) -> [StreamingOffsetSpan]? {
        let spans = JANGTQStreamingExpertStore.shared.loadOffsetSpans(
            layerIdx: layerIdx,
            projection: projection,
            suffix: suffix,
            activeExperts: activeExperts)
        var byExpert: [Int: StreamingOffsetSpan] = [:]
        for span in spans where span.presentExperts.count == 1 {
            if let expert = span.presentExperts.first {
                byExpert[expert] = span
            }
        }
        let ordered = slotExperts.compactMap { byExpert[$0] }
        return ordered.count == slotExperts.count ? ordered : nil
    }

    private func callChunk(
        xFlat: MLXArray,
        indicesFlat: MLXArray,
        indexValues: [Int]? = nil,
        tokenCount: Int,
        kSlots: Int
    ) -> MLXArray {
        let useOffsetDispatch = JANGTQStreamingExpertStore.shared.canUseOffsetDispatch(
            layerIdx: layerIdx)
        let resolvedIndexValues: [Int]
        if let provided = indexValues {
            resolvedIndexValues = provided
        } else if useOffsetDispatch
            && !MLXPressActiveExpertTrace.isEnabled
            && !mlXPressStreamingOffsetActiveShardFilterEnabled()
        {
            resolvedIndexValues = []
        } else {
            resolvedIndexValues = MLXPressStreamingProfile.time("router.indices_readback") {
                indicesFlat.reshaped([-1]).asArray(Int32.self).map(Int.init)
            }
        }
        let uniqueExperts: [Int]
        if useOffsetDispatch {
            uniqueExperts = []
        } else {
            uniqueExperts = MLXPressStreamingProfile.time("router.unique_experts") {
                Array(Set(resolvedIndexValues)).sorted()
            }
            guard !uniqueExperts.isEmpty else {
                fatalError("[MLXPressStreaming] empty routed expert set in layer \(layerIdx)")
            }
        }
        if !resolvedIndexValues.isEmpty {
            MLXPressActiveExpertTrace.record(
                layerIdx: layerIdx,
                expertIndices: resolvedIndexValues,
                tokenCount: tokenCount,
                kSlots: kSlots)
        }
        let useDirectStacked = !useOffsetDispatch
            && JANGTQStreamingExpertStore.shared.canUseDirectStacked(layerIdx: layerIdx)

        func stack(_ projection: StreamingProjection, _ suffix: StreamingSuffix) -> MLXArray? {
            MLXPressStreamingProfile.time("stack.\(projection.rawValue).\(suffix.rawValue)") {
                if useDirectStacked {
                    guard let direct = JANGTQStreamingExpertStore.shared.loadDirectStack(
                        layerIdx: layerIdx,
                        projection: projection,
                        suffix: suffix)
                    else {
                        fatalError(
                            "[MLXPressStreaming] direct stacked tensor load failed for layer \(layerIdx) \(projection.rawValue).\(suffix.rawValue)")
                    }
                    return direct
                }
                let bankByteCount = JANGTQStreamingExpertStore.shared.activeBankByteCount(
                    layerIdx: layerIdx,
                    expertIndices: uniqueExperts,
                    projection: projection,
                    suffix: suffix)
                if let bankByteCount,
                    let cached = JANGTQStreamingExpertStore.shared.cachedActiveBank(
                        layerIdx: layerIdx,
                        expertIndices: uniqueExperts,
                        projection: projection,
                        suffix: suffix,
                        byteCount: bankByteCount)
                {
                    return cached
                }
                if let bank = JANGTQStreamingExpertStore.shared.loadStack(
                    layerIdx: layerIdx,
                    expertIndices: uniqueExperts,
                    projection: projection,
                    suffix: suffix)
                {
                    if let bankByteCount {
                        return JANGTQStreamingExpertStore.shared.storeActiveBank(
                            bank,
                            layerIdx: layerIdx,
                            expertIndices: uniqueExperts,
                            projection: projection,
                            suffix: suffix,
                            byteCount: bankByteCount)
                    }
                    return bank
                }
                var arrays: [MLXArray] = []
                arrays.reserveCapacity(uniqueExperts.count)
                for expert in uniqueExperts {
                    guard
                        let array = JANGTQStreamingExpertStore.shared.load(
                            layerIdx: layerIdx,
                            expertIdx: expert,
                            projection: projection,
                            suffix: suffix)
                    else { return nil }
                    arrays.append(array)
                }
                let bank: MLXArray
                if arrays.count == 1 {
                    bank = arrays[0].expandedDimensions(axis: 0)
                } else {
                    bank = MLX.stacked(arrays, axis: 0)
                }
                if let bankByteCount {
                    return JANGTQStreamingExpertStore.shared.storeActiveBank(
                        bank,
                        layerIdx: layerIdx,
                        expertIndices: uniqueExperts,
                        projection: projection,
                        suffix: suffix,
                        byteCount: bankByteCount)
                }
                return bank
            }
        }

        let signsIn = JANGTQRuntimeCache.shared.signs(inFeatures: inputDims, seed: mxtqSeed)
        let signsDn = JANGTQRuntimeCache.shared.signs(inFeatures: hiddenDims, seed: mxtqSeed)
        let cbGate = JANGTQRuntimeCache.shared.codebook(inFeatures: inputDims, bits: gateUpBits)
        let cbDown = JANGTQRuntimeCache.shared.codebook(inFeatures: hiddenDims, bits: downBits)
        guard let signsIn, let signsDn, let cbGate, let cbDown else {
            let missing = [
                signsIn == nil ? "signs.\(inputDims).\(mxtqSeed)" : nil,
                signsDn == nil ? "signs.\(hiddenDims).\(mxtqSeed)" : nil,
                cbGate == nil ? "codebook.\(inputDims).\(gateUpBits)" : nil,
                cbDown == nil ? "codebook.\(hiddenDims).\(downBits)" : nil,
            ].compactMap { $0 }.joined(separator: ", ")
            fatalError(
                "[MLXPressStreaming] missing JANGTQ sidecar array(s) for layer "
                    + "\(layerIdx): \(missing)")
        }

        var remap: [Int: Int32] = [:]
        if !useDirectStacked && !useOffsetDispatch {
            for (local, expert) in uniqueExperts.enumerated() {
                remap[expert] = Int32(local)
            }
        }
        let rhsIndices: MLXArray
        if useOffsetDispatch {
            rhsIndices = indicesFlat.asType(.uint32).reshaped([-1])
        } else {
            let rhsIndexValues: [Int32] = useDirectStacked
                ? resolvedIndexValues.map { Int32($0) }
                : resolvedIndexValues.map { remap[$0] ?? 0 }
            rhsIndices = MLXArray(rhsIndexValues, indicesFlat.shape).asType(.uint32).reshaped([-1])
        }
        let activeExperts = useOffsetDispatch && !resolvedIndexValues.isEmpty
            ? Set(resolvedIndexValues)
            : nil

        let xAct = MLXPressStreamingProfile.time("gateup.total") {
            if useOffsetDispatch {
                let gatePackedSpans = JANGTQStreamingExpertStore.shared.loadOffsetSpans(
                        layerIdx: layerIdx,
                        projection: .gate,
                        suffix: .packed,
                        activeExperts: activeExperts)
                let gateNormSpans = JANGTQStreamingExpertStore.shared.loadOffsetSpans(
                        layerIdx: layerIdx,
                        projection: .gate,
                        suffix: .norms,
                        activeExperts: activeExperts)
                let upPackedSpans = JANGTQStreamingExpertStore.shared.loadOffsetSpans(
                        layerIdx: layerIdx,
                        projection: .up,
                        suffix: .packed,
                        activeExperts: activeExperts)
                let upNormSpans = JANGTQStreamingExpertStore.shared.loadOffsetSpans(
                        layerIdx: layerIdx,
                        projection: .up,
                        suffix: .norms,
                        activeExperts: activeExperts)
                let gateUpGroups = mlXPressStreamingGateUpOffsetSpanGroups(
                    gatePacked: gatePackedSpans,
                    gateNorms: gateNormSpans,
                    upPacked: upPackedSpans,
                    upNorms: upNormSpans)
                guard !gateUpGroups.isEmpty else {
                    fatalError(
                        "[MLXPressStreaming] missing offset-addressed gate/up tensors for layer \(layerIdx)"
                    )
                }
                let xAct = MLXPressStreamingProfile.time("gateup.offset_build") {
                    let xRot = JANGTQKernels.hadamardRotate(
                        xFlat,
                        signs: signsIn,
                        dim: inputDims)
                    let partials = gateUpGroups.map { group -> MLXArray in
                        return JANGTQKernels.fusedGateUpSwiGLUOffsets(
                            xRot: xRot,
                            packedGate: group.gatePacked.array,
                            packedGateOffsets: group.gatePacked.offsets,
                            normsGate: group.gateNorms.array,
                            normsGateOffsets: group.gateNorms.offsets,
                            packedUp: group.upPacked.array,
                            packedUpOffsets: group.upPacked.offsets,
                            normsUp: group.upNorms.array,
                            normsUpOffsets: group.upNorms.offsets,
                            codebook: cbGate,
                            rhsIndices: rhsIndices,
                            batchTokens: tokenCount,
                            K: kSlots,
                            inFeatures: inputDims,
                            outFeatures: hiddenDims,
                            bits: gateUpBits,
                            swigluLimit: swigluLimit)
                    }
                    return partials.dropFirst().reduce(partials[0]) { $0 + $1 }
                }
                let materialized = stopGradient(xAct)
                mlXPressStreamingMaterializeIfNeeded(
                    materialized,
                    force: evaluateEachLayer,
                    layerIdx: layerIdx,
                    phase: "gateup",
                    profileName: "gateup.eval")
                return materialized
            }
            guard let gatePacked = stack(.gate, .packed),
                let gateNorms = stack(.gate, .norms),
                let upPacked = stack(.up, .packed),
                let upNorms = stack(.up, .norms)
            else {
                fatalError(
                    "[MLXPressStreaming] missing active JANGTQ gate/up tensors for layer \(layerIdx)"
                )
            }
            let xAct = MLXPressStreamingProfile.time("gateup.build") {
                let xRot = JANGTQKernels.hadamardRotate(xFlat, signs: signsIn, dim: inputDims)
                return JANGTQKernels.fusedGateUpSwiGLU(
                    xRot: xRot,
                    packedGate: gatePacked, normsGate: gateNorms,
                    packedUp: upPacked, normsUp: upNorms,
                    codebook: cbGate, rhsIndices: rhsIndices,
                    batchTokens: tokenCount, K: kSlots,
                    inFeatures: inputDims, outFeatures: hiddenDims,
                    bits: gateUpBits,
                    swigluLimit: swigluLimit)
            }
            let materialized = stopGradient(xAct)
            mlXPressStreamingMaterializeIfNeeded(
                materialized,
                force: evaluateEachLayer,
                layerIdx: layerIdx,
                phase: "gateup",
                profileName: "gateup.eval")
            return materialized
        }

        if useOffsetDispatch {
            let downPackedSpans = JANGTQStreamingExpertStore.shared.loadOffsetSpans(
                    layerIdx: layerIdx,
                    projection: .down,
                    suffix: .packed,
                    activeExperts: activeExperts)
            let downNormSpans = JANGTQStreamingExpertStore.shared.loadOffsetSpans(
                    layerIdx: layerIdx,
                    projection: .down,
                    suffix: .norms,
                    activeExperts: activeExperts)
            let downGroups = mlXPressStreamingDownOffsetSpanGroups(
                packed: downPackedSpans,
                norms: downNormSpans)
            guard !downGroups.isEmpty else {
                fatalError(
                    "[MLXPressStreaming] missing offset-addressed down tensors for layer \(layerIdx)"
                )
            }
            let out = MLXPressStreamingProfile.time("down.offset_build") {
                let xActRot = JANGTQKernels.hadamardRotate(xAct, signs: signsDn, dim: hiddenDims)
                let partials = downGroups.map { group -> MLXArray in
                    return JANGTQKernels.gatherTQOffsets(
                        xRot: xActRot,
                        packed: group.packed.array,
                        packedOffsets: group.packed.offsets,
                        norms: group.norms.array,
                        normOffsets: group.norms.offsets,
                        codebook: cbDown,
                        rhsIndices: rhsIndices,
                        nRows: tokenCount * kSlots,
                        inFeatures: hiddenDims,
                        outFeatures: inputDims,
                        bits: downBits)
                }
                let y = partials.dropFirst().reduce(partials[0]) { $0 + $1 }
                return y.reshaped([tokenCount, kSlots, inputDims]).asType(xFlat.dtype)
            }
            let materialized = stopGradient(out)
            mlXPressStreamingMaterializeIfNeeded(
                materialized,
                force: evaluateEachLayer,
                layerIdx: layerIdx,
                phase: "down",
                profileName: "down.eval",
                releaseMachColdTiles: true)
            return materialized
        }

        guard let downPacked = stack(.down, .packed),
            let downNorms = stack(.down, .norms)
        else {
            fatalError(
                "[MLXPressStreaming] missing active JANGTQ down tensors for layer \(layerIdx)")
        }

        let out = MLXPressStreamingProfile.time("down.build") {
            let xActRot = JANGTQKernels.hadamardRotate(xAct, signs: signsDn, dim: hiddenDims)
            let y = JANGTQKernels.gatherTQ(
                xRot: xActRot,
                packed: downPacked, norms: downNorms,
                codebook: cbDown, rhsIndices: rhsIndices,
                nRows: tokenCount * kSlots,
                inFeatures: hiddenDims, outFeatures: inputDims,
                bits: downBits)
            return y.reshaped([tokenCount, kSlots, inputDims]).asType(xFlat.dtype)
        }
        let materialized = stopGradient(out)
        mlXPressStreamingMaterializeIfNeeded(
            materialized,
            force: evaluateEachLayer,
            layerIdx: layerIdx,
            phase: "down",
            profileName: "down.eval",
            releaseMachColdTiles: true)
        return materialized
    }
}
