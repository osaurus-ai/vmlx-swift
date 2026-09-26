// Copyright (c) 2026 Jinho Jang. All rights reserved.
//
// JangPressMachCache is the macOS native-compression primitive for MLXPress.
// It stores routed expert tiles in purgeable anonymous VM regions so Darwin can
// compress or discard dormant pages under memory pressure. The model dispatch
// path must still acquire the selected experts before compute and release cold
// experts after compute; this file provides that primitive without creating
// permanent on-disk stacked tensors.

#if canImport(Darwin)
    import Darwin
#elseif canImport(Glibc)
    import Glibc
#endif
import Foundation
import Cmlx
import MLX
#if canImport(os)
    import os
#endif

private let jangPressMachNoopDestructor: @convention(c) (UnsafeMutableRawPointer?) -> Void = { _ in }
private let jangPressMachColdState =
    VM_PURGABLE_VOLATILE | VM_VOLATILE_GROUP_7 | VM_PURGABLE_BEHAVIOR_LIFO | VM_PURGABLE_NO_AGING

private enum JangPressMachReleasePolicy {
    case volatileLast
    case pageout
}

private func jangPressMachReleasePolicy() -> JangPressMachReleasePolicy {
    let env = ProcessInfo.processInfo.environment
    let raw =
        env["MLXPRESS_MACH_RELEASE_POLICY"]
        ?? env["JANGPRESS_MACH_RELEASE_POLICY"]
        ?? "volatile-last"
    switch raw.lowercased() {
    case "pageout", "compress", "compressed", "compression":
        return .pageout
    default:
        return .volatileLast
    }
}

public enum JangPressMachError: Error, CustomStringConvertible {
    case vmAllocateFailed(kern_return_t)
    case vmPurgableControlFailed(kern_return_t)
    case mmapFailed(Int32)
    case unknownExpert(layer: Int, expert: Int)
    case alreadyDiscarded(layer: Int, expert: Int)

    public var description: String {
        switch self {
        case .vmAllocateFailed(let kr):
            return "vm_allocate failed: kr=\(kr)"
        case .vmPurgableControlFailed(let kr):
            return "vm_purgable_control failed: kr=\(kr)"
        case .mmapFailed(let err):
            return "mmap failed: errno=\(err)"
        case .unknownExpert(let layer, let expert):
            return "unknown expert (layer=\(layer), expert=\(expert))"
        case .alreadyDiscarded(let layer, let expert):
            return "expert tile was discarded and has no disk refault source (layer=\(layer), expert=\(expert))"
        }
    }
}

public struct JangPressMachConfig: Sendable, Equatable {
    /// Fraction of experts that should stay non-volatile when release policies
    /// are driven from the user-facing compression percentage.
    public var alwaysHotFraction: Double

    /// Future scheduler knob. The primitive exposes acquire/release; callers
    /// can use this flag to decide whether to pre-acquire likely next experts.
    public var enablePrefetch: Bool

    /// Allow refilling a tile from a recorded file/offset if Darwin reports the
    /// purgeable region as EMPTY when it is acquired.
    public var enableDiskRefault: Bool

    /// User-facing percentage: 70 means 70% cold, 30% hot.
    public var manualCompressPercent: Int?

    public init(
        alwaysHotFraction: Double = 0.0,
        enablePrefetch: Bool = true,
        enableDiskRefault: Bool = false,
        manualCompressPercent: Int? = nil
    ) {
        if let manualCompressPercent {
            let pct = max(0, min(100, manualCompressPercent))
            self.manualCompressPercent = pct
            self.alwaysHotFraction = Double(100 - pct) / 100.0
        } else {
            self.manualCompressPercent = nil
            self.alwaysHotFraction = max(0.0, min(1.0, alwaysHotFraction))
        }
        self.enablePrefetch = enablePrefetch
        self.enableDiskRefault = enableDiskRefault
    }
}

public struct JangPressTile: @unchecked Sendable {
    public let layerId: Int
    public let expertId: Int
    public let baseAddress: UnsafeMutableRawPointer
    /// Original byte count copied into the region.
    public let byteCount: Int
    /// Page-rounded VM region size.
    public let allocatedSize: Int
    public let diskURL: URL?
    public let diskOffset: UInt64

    fileprivate let region: vm_address_t
    fileprivate var accessCount: UInt64
    fileprivate var lastAccessTick: UInt64
}

public struct JangPressMachStats: Sendable, Equatable {
    public var totalTiles: Int
    public var totalBytesAllocated: Int
    public var totalPayloadBytes: Int
    public var hotPinned: Int
    public var currentlyVolatile: Int
    public var currentlyNonVolatile: Int
    public var acquireCount: UInt64
    public var releaseCount: UInt64
    public var refaultCount: UInt64
    public var discardCount: UInt64
    public var pressureLowCount: UInt64
    public var pressureWarnCount: UInt64
    public var pressureCriticalCount: UInt64
}

public final class JangPressMachCache: @unchecked Sendable {
    private struct TileKey: Hashable, Sendable {
        let layer: Int
        let expert: Int
        let component: String
    }

    private let config: JangPressMachConfig
    private let lock = OSAllocatedUnfairLock()
    private let log = Logger(subsystem: "ai.jangq.vmlx", category: "MLXPressMachCache")

    private var tiles: [TileKey: JangPressTile] = [:]
    private var hotPinned: Set<TileKey> = []
    private var volatileTiles: Set<TileKey> = []
    #if canImport(Darwin)
    private var pressureSource: DispatchSourceMemoryPressure?
    #endif
    private var stats = JangPressMachStats(
        totalTiles: 0,
        totalBytesAllocated: 0,
        totalPayloadBytes: 0,
        hotPinned: 0,
        currentlyVolatile: 0,
        currentlyNonVolatile: 0,
        acquireCount: 0,
        releaseCount: 0,
        refaultCount: 0,
        discardCount: 0,
        pressureLowCount: 0,
        pressureWarnCount: 0,
        pressureCriticalCount: 0)

    public init(config: JangPressMachConfig = .init()) {
        self.config = config
        installPressureMonitor()
    }

    deinit {
        #if canImport(Darwin)
        pressureSource?.cancel()
        #endif
        lock.lock()
        let regions = tiles.values.map { ($0.region, $0.allocatedSize) }
        tiles.removeAll()
        hotPinned.removeAll()
        volatileTiles.removeAll()
        lock.unlock()
        for (region, size) in regions {
            _ = vm_deallocate(mach_task_self_, region, vm_size_t(size))
        }
    }

    @discardableResult
    public func register(
        layer: Int,
        expert: Int,
        bytes: UnsafeRawBufferPointer,
        diskURL: URL? = nil,
        diskOffset: UInt64 = 0
    ) throws -> JangPressTile {
        try register(
            layer: layer,
            expert: expert,
            component: "",
            bytes: bytes,
            diskURL: diskURL,
            diskOffset: diskOffset)
    }

    @discardableResult
    public func register(
        layer: Int,
        expert: Int,
        component: String,
        bytes: UnsafeRawBufferPointer,
        diskURL: URL? = nil,
        diskOffset: UInt64 = 0
    ) throws -> JangPressTile {
        try registerFilled(
            layer: layer,
            expert: expert,
            component: component,
            byteCount: bytes.count,
            diskURL: diskURL,
            diskOffset: diskOffset
        ) { target, _ in
            if let src = bytes.baseAddress, bytes.count > 0 {
                memcpy(target, src, bytes.count)
            }
        }
    }

    @discardableResult
    public func registerFilled(
        layer: Int,
        expert: Int,
        component: String,
        byteCount: Int,
        diskURL: URL? = nil,
        diskOffset: UInt64 = 0,
        fill: (UnsafeMutableRawPointer, Int) throws -> Void
    ) throws -> JangPressTile {
        let pageSize = Int(getpagesize())
        let alignedSize = max(pageSize, ((byteCount + pageSize - 1) / pageSize) * pageSize)

        var address: vm_address_t = 0
        let kr = vm_allocate(
            mach_task_self_,
            &address,
            vm_size_t(alignedSize),
            VM_FLAGS_ANYWHERE | VM_FLAGS_PURGABLE)
        guard kr == KERN_SUCCESS else {
            throw JangPressMachError.vmAllocateFailed(kr)
        }

        let ptr = UnsafeMutableRawPointer(bitPattern: UInt(address))!
        do {
            if byteCount > 0 {
                try fill(ptr, byteCount)
            }
        } catch {
            _ = vm_deallocate(mach_task_self_, address, vm_size_t(alignedSize))
            throw error
        }
        if alignedSize > byteCount {
            memset(ptr.advanced(by: byteCount), 0, alignedSize - byteCount)
        }

        let key = TileKey(layer: layer, expert: expert, component: normalizedComponent(component))
        let tile = JangPressTile(
            layerId: layer,
            expertId: expert,
            baseAddress: ptr,
            byteCount: byteCount,
            allocatedSize: alignedSize,
            diskURL: diskURL,
            diskOffset: diskOffset,
            region: address,
            accessCount: 0,
            lastAccessTick: 0)

        lock.lock()
        if let old = tiles[key] {
            _ = vm_deallocate(mach_task_self_, old.region, vm_size_t(old.allocatedSize))
            stats.totalBytesAllocated -= old.allocatedSize
            stats.totalPayloadBytes -= old.byteCount
            volatileTiles.remove(key)
        } else {
            stats.totalTiles += 1
        }
        tiles[key] = tile
        stats.totalBytesAllocated += alignedSize
        stats.totalPayloadBytes += byteCount
        refreshVolatileCountsLocked()
        lock.unlock()

        return tile
    }

    public func pinHot(layer: Int, experts: [Int]) {
        lock.lock()
        for expert in experts {
            for key in keysLocked(layer: layer, expert: expert) {
                hotPinned.insert(key)
                volatileTiles.remove(key)
            }
        }
        stats.hotPinned = hotPinned.count
        refreshVolatileCountsLocked()
        lock.unlock()
    }

    public func acquire(layer: Int, experts: [Int]) throws -> [JangPressTile] {
        var acquired: [JangPressTile] = []

        lock.lock()
        defer { lock.unlock() }

        for expert in experts {
            let keys = keysLocked(layer: layer, expert: expert)
            guard !keys.isEmpty else {
                throw JangPressMachError.unknownExpert(layer: layer, expert: expert)
            }
            acquired.reserveCapacity(acquired.count + keys.count)
            for key in keys {
                acquired.append(try acquireLocked(key: key))
            }
        }
        refreshVolatileCountsLocked()
        return acquired
    }

    public func acquire(layer: Int, expert: Int, component: String) throws -> JangPressTile {
        lock.lock()
        defer {
            refreshVolatileCountsLocked()
            lock.unlock()
        }

        let key = TileKey(layer: layer, expert: expert, component: normalizedComponent(component))
        guard tiles[key] != nil else {
            throw JangPressMachError.unknownExpert(layer: layer, expert: expert)
        }
        return try acquireLocked(key: key)
    }

    public func array(
        layer: Int,
        expert: Int,
        component: String,
        shape: [Int],
        dtype: DType
    ) throws -> MLXArray {
        let tile = try acquire(layer: layer, expert: expert, component: component)
        let expectedBytes = shape.reduce(1, *) * dtype.size
        precondition(
            expectedBytes == tile.byteCount,
            "shape/dtype byte count \(expectedBytes) does not match tile byte count \(tile.byteCount)")
        var cShape = shape.map(Int32.init)
        let array = mlx_array_new_data_managed_payload(
            tile.baseAddress,
            &cShape,
            Int32(cShape.count),
            dtype.cmlxDtype,
            nil,
            jangPressMachNoopDestructor)
        return MLXArray(array)
    }

    public func release(layer: Int, experts: [Int]) {
        lock.lock()
        defer { lock.unlock() }

        for expert in experts {
            for key in keysLocked(layer: layer, expert: expert) {
                guard let tile = tiles[key], !hotPinned.contains(key) else { continue }
                releaseLocked(key: key, tile: tile)
            }
        }
        refreshVolatileCountsLocked()
    }

    public func release(layer: Int, expert: Int, components: [String]) {
        lock.lock()
        defer { lock.unlock() }

        for component in components {
            let key = TileKey(
                layer: layer,
                expert: expert,
                component: normalizedComponent(component))
            guard let tile = tiles[key], !hotPinned.contains(key) else { continue }
            releaseLocked(key: key, tile: tile)
        }
        refreshVolatileCountsLocked()
    }

    /// Release the coldest fraction of tiles by observed routing frequency.
    /// This is the primitive equivalent of "compressPct": a value of 70 marks
    /// the coldest 70% of non-pinned tiles volatile and keeps the rest hot.
    @discardableResult
    public func releaseColdTiles(compressPercent: Int? = nil) -> Int {
        let pct = max(0, min(100, compressPercent ?? config.manualCompressPercent ?? 0))
        guard pct > 0 else { return 0 }

        lock.lock()
        defer { lock.unlock() }

        let candidates = tiles
            .filter { !hotPinned.contains($0.key) }
            .sorted { lhs, rhs in
                if lhs.value.accessCount == rhs.value.accessCount {
                    return lhs.value.lastAccessTick < rhs.value.lastAccessTick
                }
                return lhs.value.accessCount < rhs.value.accessCount
            }
        let releaseCount = Int((Double(candidates.count) * Double(pct) / 100.0).rounded(.down))
        guard releaseCount > 0 else { return 0 }

        var released = 0
        for (key, tile) in candidates.prefix(releaseCount) {
            if releaseLockedForPolicy(key: key, tile: tile) {
                stats.releaseCount &+= 1
                released += 1
            }
        }
        refreshVolatileCountsLocked()
        return released
    }

    public func snapshot() -> JangPressMachStats {
        lock.lock()
        defer { lock.unlock() }
        return stats
    }

    @discardableResult
    public func remove(layer: Int, expert: Int, component: String) -> Bool {
        let key = TileKey(layer: layer, expert: expert, component: normalizedComponent(component))
        lock.lock()
        guard let tile = tiles.removeValue(forKey: key) else {
            lock.unlock()
            return false
        }
        hotPinned.remove(key)
        volatileTiles.remove(key)
        stats.totalTiles = max(0, stats.totalTiles - 1)
        stats.totalBytesAllocated = max(0, stats.totalBytesAllocated - tile.allocatedSize)
        stats.totalPayloadBytes = max(0, stats.totalPayloadBytes - tile.byteCount)
        refreshVolatileCountsLocked()
        lock.unlock()

        _ = vm_deallocate(mach_task_self_, tile.region, vm_size_t(tile.allocatedSize))
        return true
    }

    public func removeAll() {
        lock.lock()
        let regions = tiles.values.map { ($0.region, $0.allocatedSize) }
        tiles.removeAll(keepingCapacity: false)
        hotPinned.removeAll(keepingCapacity: false)
        volatileTiles.removeAll(keepingCapacity: false)
        stats.totalTiles = 0
        stats.totalBytesAllocated = 0
        stats.totalPayloadBytes = 0
        stats.hotPinned = 0
        stats.currentlyVolatile = 0
        stats.currentlyNonVolatile = 0
        lock.unlock()

        for (region, size) in regions {
            _ = vm_deallocate(mach_task_self_, region, vm_size_t(size))
        }
    }

    private func acquireLocked(key: TileKey) throws -> JangPressTile {
        guard var tile = tiles[key] else {
            throw JangPressMachError.unknownExpert(layer: key.layer, expert: key.expert)
        }

        if !hotPinned.contains(key) {
            var state: Int32 = VM_PURGABLE_NONVOLATILE
            let kr = vm_purgable_control(
                mach_task_self_,
                tile.region,
                VM_PURGABLE_SET_STATE,
                &state)
            guard kr == KERN_SUCCESS else {
                throw JangPressMachError.vmPurgableControlFailed(kr)
            }

            if state == VM_PURGABLE_EMPTY {
                stats.discardCount &+= 1
                guard config.enableDiskRefault, let url = tile.diskURL else {
                    throw JangPressMachError.alreadyDiscarded(
                        layer: key.layer,
                        expert: key.expert)
                }
                try refaultFromDiskLocked(tile: tile, diskURL: url, offset: tile.diskOffset)
                stats.refaultCount &+= 1
            }
            volatileTiles.remove(key)
        }

        tile.accessCount &+= 1
        tile.lastAccessTick = mach_absolute_time()
        tiles[key] = tile
        stats.acquireCount &+= 1
        return tile
    }

    private func keysLocked(layer: Int, expert: Int) -> [TileKey] {
        tiles.keys
            .filter { $0.layer == layer && $0.expert == expert }
            .sorted { lhs, rhs in lhs.component < rhs.component }
    }

    private func normalizedComponent(_ component: String) -> String {
        component.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func releaseLocked(key: TileKey, tile: JangPressTile) {
        if releaseLockedForPolicy(key: key, tile: tile) {
            stats.releaseCount &+= 1
        } else {
            log.warning(
                "release failed layer=\(key.layer) expert=\(key.expert) component=\(key.component)")
        }
    }

    private func releaseLockedForPolicy(key: TileKey, tile: JangPressTile) -> Bool {
        switch jangPressMachReleasePolicy() {
        case .volatileLast:
            var state: Int32 = jangPressMachColdState
            let kr = vm_purgable_control(
                mach_task_self_,
                tile.region,
                VM_PURGABLE_SET_STATE,
                &state)
            if kr == KERN_SUCCESS {
                volatileTiles.insert(key)
                return true
            }
            return false
        case .pageout:
            var state: Int32 = VM_PURGABLE_NONVOLATILE
            let kr = vm_purgable_control(
                mach_task_self_,
                tile.region,
                VM_PURGABLE_SET_STATE,
                &state)
            guard kr == KERN_SUCCESS else { return false }
            volatileTiles.remove(key)
            _ = madvise(tile.baseAddress, tile.allocatedSize, MADV_PAGEOUT)
            return true
        }
    }

    private func refaultFromDiskLocked(
        tile: JangPressTile,
        diskURL: URL,
        offset: UInt64
    ) throws {
        let fd = open(diskURL.path, O_RDONLY)
        guard fd >= 0 else { throw JangPressMachError.mmapFailed(errno) }
        defer { close(fd) }

        var readBytes = 0
        while readBytes < tile.byteCount {
            let readCount = min(tile.byteCount - readBytes, 64 * 1024 * 1024)
            let n = pread(
                fd,
                tile.baseAddress.advanced(by: readBytes),
                readCount,
                off_t(offset) + off_t(readBytes))
            if n <= 0 { throw JangPressMachError.mmapFailed(errno) }
            readBytes += n
        }
    }

    private func refreshVolatileCountsLocked() {
        stats.hotPinned = hotPinned.count
        stats.currentlyVolatile = volatileTiles.count
        stats.currentlyNonVolatile = max(0, tiles.count - volatileTiles.count)
    }

    private func installPressureMonitor() {
        #if canImport(Darwin)
        let queue = DispatchQueue(label: "ai.jangq.vmlx.mlxpress-mach-pressure", qos: .utility)
        let source = DispatchSource.makeMemoryPressureSource(
            eventMask: [.normal, .warning, .critical],
            queue: queue)
        source.setEventHandler { [weak self] in
            guard let self else { return }
            let event = source.data
            let low = event.contains(.normal)
            let warn = event.contains(.warning)
            let critical = event.contains(.critical)
            self.lock.lock()
            if low { self.stats.pressureLowCount &+= 1 }
            if warn { self.stats.pressureWarnCount &+= 1 }
            if critical { self.stats.pressureCriticalCount &+= 1 }
            self.lock.unlock()
        }
        source.activate()
        pressureSource = source
        #endif
    }
}
