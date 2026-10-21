import Foundation

/// One disk-cap calculation for settings, runtime admission and live updates.
/// Owned bytes are the root-wide payload bytes counted by the cache index,
/// including recurrent companions. They are added back to free space so a
/// cache filling itself does not shrink its own quota on the next model load.
public enum DiskCacheCapPolicy {
    public static let automaticFraction = 0.30
    public static let unknownVolumeBytes: Int64 = 10 * 1_073_741_824

    /// Configs use Float GiB for compatibility. Converting their rounded
    /// value directly to Int can trap near Int.max, even for a valid resolved
    /// byte count. Quota arithmetic must saturate rather than crash.
    public static func byteLimit(gigabytes: Float) -> Int {
        let bytes = Double(gigabytes) * 1_073_741_824
        guard !bytes.isNaN, bytes > 0 else { return 0 }
        guard bytes < Double(Int.max) else { return .max }
        return Int(bytes)
    }

    public enum Rule: String, Sendable {
        case explicitPercent, legacyGB, automatic, unknownVolume
    }

    public struct Resolution: Equatable, Sendable {
        public let capBytes: Int64
        public let requestedBytes: Int64
        public let rule: Rule
        /// Kept for source compatibility; explicit limits are no longer clamped.
        public let limitedByHost: Bool
        /// Advisory only: the quota is a ceiling, not a disk reservation.
        public let exceedsAvailableSpace: Bool
        /// Advisory only. Never disables a tier or rejects a request.
        public let lowFreeSpace: Bool
        public var capGB: Double { Double(capBytes) / 1_073_741_824 }
    }

    /// Unknown readings are nil, not zero. Zero free bytes is a real full
    /// volume. An unreadable existing index must not be treated as an empty
    /// cache; callers can retain their current cap when measurements fail.
    public static func resolve(
        percent: Double?, legacyGB: Double?,
        totalBytes: Int64?, freeBytes: Int64?, ownBytes: Int64?,
        previousCapBytes: Int64? = nil
    ) -> Resolution {
        let available: Int64? = freeBytes.flatMap { free in
            ownBytes.map { IndexedBytes.sum(free, $0) }
        }
        let lowFreeSpace = freeBytes.map { $0 < 4 * 1_073_741_824 } ?? false
        let requested: Int64
        let rule: Rule
        if let percent, percent.isFinite, percent > 0, percent <= 100 {
            if let totalBytes, totalBytes > 0 {
                requested = positiveBytes(Double(totalBytes) * percent / 100)
                rule = .explicitPercent
            } else {
                requested = max(1, previousCapBytes ?? unknownVolumeBytes)
                rule = .unknownVolume
            }
        } else if let legacyGB, legacyGB.isFinite, legacyGB > 0 {
            requested = positiveBytes(legacyGB * 1_073_741_824)
            rule = .legacyGB
        } else if let available {
            requested = positiveBytes(Double(available) * automaticFraction)
            rule = .automatic
        } else {
            requested = max(1, previousCapBytes ?? unknownVolumeBytes)
            rule = .unknownVolume
        }

        // An explicit quota is the user's ceiling, not a reservation. Free-space
        // changes must not silently rewrite it or evict a dedicated drive's cache
        // at 25%. The writer still reports actual filesystem failures normally.
        return Resolution(
            capBytes: requested, requestedBytes: requested, rule: rule,
            limitedByHost: false,
            exceedsAvailableSpace: available.map { requested > $0 } ?? false,
            lowFreeSpace: lowFreeSpace)
    }

    private static func positiveBytes(_ value: Double) -> Int64 {
        guard value < Double(Int64.max) else { return .max }
        return max(1, Int64(max(0, value)))
    }
}
