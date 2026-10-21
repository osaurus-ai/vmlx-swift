// Copyright © 2026 Jinho Jang. All rights reserved.
//
// MLXPress is the public name for the routed-weight cold-tier and cache-stack
// runtime. The original JangPress symbols remain as source-compatible aliases
// while host applications migrate.

import Foundation

/// Public MLXPress spelling for the cold-weight policy.
public typealias MLXPressPolicy = JangPressPolicy

/// Public MLXPress spelling for load options.
public typealias MLXPressLoadOptions = JangPressLoadOptions

/// Public MLXPress spelling for the runtime handle.
public typealias MLXPressRuntime = JangPressRuntime

/// Public MLXPress spelling for runtime status.
public typealias MLXPressStatus = JangPressStatus

/// Public MLXPress spelling for mmap-tier configuration.
public typealias MLXPressMmapConfig = JangPressMmapConfig

/// Public MLXPress spelling for the mmap-tier probe.
public typealias MLXPressMmapTier = JangPressMmapTier

/// Public MLXPress spelling for the embed/lm-head tier configuration.
public typealias MLXPressEmbedConfig = JangPressEmbedConfig

/// Public MLXPress spelling for the embed/lm-head tier.
public typealias MLXPressEmbedTier = JangPressEmbedTier

/// Public MLXPress spelling for the native Mach purgeable expert cache.
public typealias MLXPressMachCache = JangPressMachCache

/// Public MLXPress spelling for Mach purgeable expert-cache configuration.
public typealias MLXPressMachConfig = JangPressMachConfig

/// Public MLXPress spelling for Mach purgeable expert-cache errors.
public typealias MLXPressMachError = JangPressMachError

/// Public MLXPress spelling for Mach purgeable expert-cache stats.
public typealias MLXPressMachStats = JangPressMachStats

/// Public MLXPress spelling for a Mach purgeable routed expert tile.
public typealias MLXPressTile = JangPressTile

/// Public MLXPress spelling for prestack/alignment bundle preparation.
public typealias MLXPressPrestacker = JangPressPrestacker

/// Public MLXPress spelling for mmap shard advice.
public typealias MLXPressAdvice = JangPressAdvice

/// Public MLXPress spelling for mmap shard errors.
public typealias MLXPressShardError = JangPressShardError

public extension LoadConfiguration {
    /// Public MLXPress policy spelling. Backed by the legacy `jangPress`
    /// storage field for source compatibility.
    var mlxPress: MLXPressPolicy {
        get { jangPress }
        set { jangPress = newValue }
    }

    /// Opt-in MLXPress with auto-detection (`.auto(envFallback: true)`).
    /// `MLXPRESS=N`/`MLXPRESS=off` are preferred; `JANGPRESS=*` is still
    /// accepted as a compatibility fallback.
    static let experimentalMLXPressAuto = experimentalJangPressAuto

    /// New MLXPress-labeled initializer. The legacy `jangPress:` initializer
    /// remains available for existing callers.
    init(
        mlxPress: MLXPressPolicy,
        maxResidentBytes: ResidentCap = .default,
        memoryLimit: ResidentCap = .default,
        useMmapSafetensors: Bool = true
    ) {
        self.init(
            jangPress: mlxPress,
            maxResidentBytes: maxResidentBytes,
            memoryLimit: memoryLimit,
            useMmapSafetensors: useMmapSafetensors)
    }
}

/// Public MLXPress activation entry point.
public enum MLXPressActivation {
    public static func activate(
        bundleURL: URL,
        options: MLXPressLoadOptions
    ) -> MLXPressRuntime {
        JangPressActivation.activate(bundleURL: bundleURL, options: options)
    }

    public static func deactivate(_ runtime: MLXPressRuntime) {
        JangPressActivation.deactivate(runtime)
    }
}
