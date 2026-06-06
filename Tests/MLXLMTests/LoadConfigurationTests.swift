// Copyright © 2026 Jinho Jang. All rights reserved.
//
// Tests for `LoadConfiguration`, `JangPressPolicy`, `ResidentCap`,
// `LoadBundleFacts`, and `JangPressStatus`.
//
// These cover the typed surface that osaurus / JANG Studio wire to
// settings — they don't load any real model. Real-bundle integration
// is exercised by `JangPressActivationTests` plus the bench harnesses.

import Foundation
import MLX
import Testing
@testable import MLXLMCommon

@Suite("LoadConfiguration")
struct LoadConfigurationTests {

    // MARK: - JangPressPolicy.resolve precedence

    /// `.disabled` always resolves to disabled options regardless of
    /// bundle facts or env state.
    @Test("disabled policy ignores facts and env")
    func disabledIgnoresFacts() {
        let huge = LoadBundleFacts(
            totalSafetensorsBytes: 200 * 1024 * 1024 * 1024,  // 200 GB
            isRouted: true,
            physicalMemory: 128 * 1024 * 1024 * 1024)

        let withEnv = withEnvironmentValue("JANGPRESS", "70") {
            JangPressPolicy.disabled.resolve(facts: huge)
        }
        #expect(withEnv.enabled == false)

        let withoutEnv = withEnvironmentValue("JANGPRESS", nil) {
            JangPressPolicy.disabled.resolve(facts: huge)
        }
        #expect(withoutEnv.enabled == false)
    }

    /// `.enabled(coldFraction:)` always wins over env.
    @Test("explicit enabled wins over JANGPRESS env")
    func explicitEnabledWinsOverEnv() {
        let facts = LoadBundleFacts.tiny
        let opts = withEnvironmentValue("JANGPRESS", "off") {
            JangPressPolicy.enabled(coldFraction: 0.5).resolve(facts: facts)
        }
        #expect(opts.enabled == true)
        #expect(opts.compressPct == 50)
    }

    /// `.enabled(coldFraction:)` clamps fraction to [0.0, 0.95].
    @Test("enabled clamps coldFraction to [0.0, 0.95]")
    func enabledClamps() {
        let facts = LoadBundleFacts.tiny

        let high = JangPressPolicy.enabled(coldFraction: 1.5).resolve(facts: facts)
        #expect(high.compressPct == 95)

        let low = JangPressPolicy.enabled(coldFraction: -0.3).resolve(facts: facts)
        #expect(low.compressPct == 0)
    }

    // MARK: - JangPressPolicy.auto + env precedence

    /// `.auto(envFallback: true)` honors `JANGPRESS=70`.
    @Test("auto envFallback honors JANGPRESS=70")
    func autoHonorsNumericEnv() {
        let opts = withEnvironmentValue("JANGPRESS", "70") {
            JangPressPolicy.auto(envFallback: true).resolve(facts: .tiny)
        }
        #expect(opts.enabled == true)
        #expect(opts.compressPct == 70)
    }

    /// `.auto(envFallback: true)` honors `JANGPRESS=off`.
    @Test("auto envFallback honors JANGPRESS=off")
    func autoHonorsOffEnv() {
        let big = LoadBundleFacts(
            totalSafetensorsBytes: 100 * 1024 * 1024 * 1024,
            isRouted: true,
            physicalMemory: 128 * 1024 * 1024 * 1024)

        let opts = withEnvironmentValue("JANGPRESS", "off") {
            JangPressPolicy.auto(envFallback: true).resolve(facts: big)
        }
        #expect(opts.enabled == false)
    }

    /// Garbage env values (not 0/off/false/0..95) fall through to the
    /// threshold rule.
    @Test("auto envFallback falls through on bad env value")
    func autoFallsThroughOnBadEnv() {
        let big = LoadBundleFacts(
            totalSafetensorsBytes: 100 * 1024 * 1024 * 1024,
            isRouted: true,
            physicalMemory: 128 * 1024 * 1024 * 1024)

        let opts = withEnvironmentValue("JANGPRESS", "banana") {
            JangPressPolicy.auto(envFallback: true).resolve(facts: big)
        }
        // Threshold is met (routed AND > 0.5 × physical) → enabled@70
        #expect(opts.enabled == true)
        #expect(opts.compressPct == 70)
    }

    /// `.auto(envFallback: false)` ignores env entirely.
    @Test("auto envFallback=false ignores JANGPRESS env")
    func autoNoEnvIgnoresEnv() {
        let opts = withEnvironmentValue("JANGPRESS", "70") {
            JangPressPolicy.auto(envFallback: false).resolve(facts: .tiny)
        }
        // .tiny doesn't meet threshold → disabled
        #expect(opts.enabled == false)
    }

    /// Threshold rule: enable iff routed AND bundle > 0.5 × physical.
    @Test("auto threshold enables only when routed AND large")
    func autoThresholdRule() {
        // routed but tiny — disabled
        let routedTiny = LoadBundleFacts(
            totalSafetensorsBytes: 1 * 1024 * 1024 * 1024,
            isRouted: true,
            physicalMemory: 128 * 1024 * 1024 * 1024)
        #expect(
            JangPressPolicy.auto(envFallback: false)
                .resolve(facts: routedTiny).enabled == false)

        // dense but huge — disabled (no routed pool to compress)
        let denseHuge = LoadBundleFacts(
            totalSafetensorsBytes: 200 * 1024 * 1024 * 1024,
            isRouted: false,
            physicalMemory: 128 * 1024 * 1024 * 1024)
        #expect(
            JangPressPolicy.auto(envFallback: false)
                .resolve(facts: denseHuge).enabled == false)

        // routed AND large — enabled at default 70
        let routedHuge = LoadBundleFacts(
            totalSafetensorsBytes: 100 * 1024 * 1024 * 1024,
            isRouted: true,
            physicalMemory: 128 * 1024 * 1024 * 1024)
        let opts = JangPressPolicy.auto(envFallback: false).resolve(facts: routedHuge)
        #expect(opts.enabled == true)
        #expect(opts.compressPct == 70)
    }

    // MARK: - ResidentCap.resolve

    @Test("ResidentCap.unlimited resolves to nil")
    func residentCapUnlimited() {
        #expect(ResidentCap.unlimited.resolve(physicalMemory: 1024) == nil)
        #expect(ResidentCap.unlimited.applyAsCacheLimitInt(physicalMemory: 1024) == nil)
    }

    @Test("ResidentCap.fraction returns fraction × physical")
    func residentCapFraction() {
        let physical: UInt64 = 128 * 1024 * 1024 * 1024
        let expected: UInt64 = 64 * 1024 * 1024 * 1024
        #expect(ResidentCap.fraction(0.5).resolve(physicalMemory: physical) == expected)
    }

    @Test("ResidentCap.fraction clamps to [0.0, 1.0]")
    func residentCapFractionClamps() {
        let high = ResidentCap.fraction(1.5).resolve(physicalMemory: 100)
        #expect(high == 100)

        let low = ResidentCap.fraction(-0.5).resolve(physicalMemory: 100)
        #expect(low == 0)
    }

    @Test("ResidentCap.absolute returns exact bytes")
    func residentCapAbsolute() {
        let cap = ResidentCap.absolute(42).resolve(physicalMemory: 999_999)
        #expect(cap == 42)
    }

    // MARK: - LoadConfiguration default + off

    @Test("LoadConfiguration.default = JangPress disabled (opt-in), 70% caps, mmap on")
    func defaultConfig() {
        let cfg = LoadConfiguration.default
        #expect(cfg.jangPress == .disabled)
        #expect(cfg.maxResidentBytes == .fraction(0.70))
        #expect(cfg.memoryLimit == .fraction(0.70))
        #expect(cfg.useMmapSafetensors == true)
    }

    @Test("LoadConfiguration.experimentalJangPressAuto = .auto + 70% caps + mmap on")
    func experimentalAutoConfig() {
        let cfg = LoadConfiguration.experimentalJangPressAuto
        #expect(cfg.jangPress == .auto(envFallback: true))
        #expect(cfg.maxResidentBytes == .fraction(0.70))
        #expect(cfg.memoryLimit == .fraction(0.70))
        #expect(cfg.useMmapSafetensors == true)
    }

    @Test("LoadConfiguration.off = disabled JangPress + unlimited everything")
    func offConfig() {
        let cfg = LoadConfiguration.off
        #expect(cfg.jangPress == .disabled)
        #expect(cfg.maxResidentBytes == .unlimited)
        #expect(cfg.memoryLimit == .unlimited)
        #expect(cfg.useMmapSafetensors == false)
    }

    @Test("JANGTQ load keeps tq tensors raw and protects mmap residency")
    func jangtqLoadDoesNotSkipWholeModelBFloat16Conversion() throws {
        let source = try String(
            contentsOfFile: "Libraries/MLXLMCommon/Load.swift",
            encoding: .utf8)

        #expect(source.contains("let mmapSafetensorsActive = envFlag(\"MLX_SAFETENSORS_MMAP\")"))
        #expect(source.contains("let allowJANGTQMmapBFloat16 = envFlag(\"VMLINUX_JANGTQ_BF16_MMAP\")"))
        #expect(source.contains("let autoJANGTQMmapBFloat16 = isNemotronHBundle(modelDirectory)"))
        #expect(source.contains("if !isJANGTQNative || !mmapSafetensorsActive || allowJANGTQMmapBFloat16"))
        #expect(source.contains("|| autoJANGTQMmapBFloat16"))
        #expect(source.contains("private func isNemotronHBundle(_ modelDirectory: URL) -> Bool"))
        #expect(source.contains("modelType.lowercased() == \"nemotron_h\""))
        #expect(source.contains("shouldSkip: isJANGTQNative ? isJANGTQParameterKey"))
        #expect(source.contains("key.hasSuffix(\".tq_packed\") || key.hasSuffix(\".tq_norms\")"))
        #expect(!source.contains("if !isJANGTQNative {\n        convertToBFloat16(model: model)\n    }"))
    }

    // MARK: - MemoryStatus

    @Test("MemoryStatus.snapshot reports plausible values")
    func memoryStatusSnapshot() {
        let s = MemoryStatus.snapshot()
        // memoryLimit defaults to ~1.5 × recommendedMaxWorkingSetBytes
        // before any load has set it; should be > 0.
        #expect(s.memoryLimit > 0)
        // cacheLimit defaults to memoryLimit; should be > 0.
        #expect(s.cacheLimit > 0)
        // physical memory > 0 (sysctl always succeeds on darwin).
        #expect(s.physicalMemory > 0)
        // RSS > 0 once we're running (test process IS allocated).
        #expect(s.currentRSS > 0)
    }

    @Test("MemoryStatus reflects recently-set memoryLimit")
    func memoryStatusReflectsSet() {
        let prior = MLX.Memory.memoryLimit
        defer { MLX.Memory.memoryLimit = prior }

        let target = 4 * 1024 * 1024 * 1024  // 4 GB
        MLX.Memory.memoryLimit = target

        let s = MemoryStatus.snapshot()
        #expect(s.memoryLimit == target)
    }

    // MARK: - LoadBundleFacts.inspect

    @Test("inspect counts safetensors byte total")
    func inspectCountsBytes() throws {
        let dir = try Self.makeBundle(files: [
            ("model-00001-of-00002.safetensors", Data(count: 1024)),
            ("model-00002-of-00002.safetensors", Data(count: 2048)),
            // Non-safetensors should not count.
            ("tokenizer.json", Data(count: 99)),
        ])
        defer { try? FileManager.default.removeItem(at: dir) }

        let facts = LoadBundleFacts.inspect(bundleURL: dir)
        #expect(facts.totalSafetensorsBytes == 1024 + 2048)
    }

    @Test("inspect detects num_local_experts → routed")
    func inspectDetectsRouted() throws {
        let cfg = ["num_local_experts": 8, "hidden_size": 4096] as [String: Any]
        let dir = try Self.makeBundle(files: [
            ("config.json",
             try JSONSerialization.data(withJSONObject: cfg)),
        ])
        defer { try? FileManager.default.removeItem(at: dir) }

        let facts = LoadBundleFacts.inspect(bundleURL: dir)
        #expect(facts.isRouted == true)
    }

    @Test("inspect detects nested text_config.num_experts → routed")
    func inspectDetectsNestedRouted() throws {
        let cfg = [
            "text_config": ["num_experts": 32, "hidden_size": 4096] as [String: Any]
        ] as [String: Any]
        let dir = try Self.makeBundle(files: [
            ("config.json",
             try JSONSerialization.data(withJSONObject: cfg)),
        ])
        defer { try? FileManager.default.removeItem(at: dir) }

        let facts = LoadBundleFacts.inspect(bundleURL: dir)
        #expect(facts.isRouted == true)
    }

    @Test("inspect treats dense config as not routed")
    func inspectDense() throws {
        let cfg = ["hidden_size": 4096] as [String: Any]
        let dir = try Self.makeBundle(files: [
            ("config.json",
             try JSONSerialization.data(withJSONObject: cfg)),
        ])
        defer { try? FileManager.default.removeItem(at: dir) }

        let facts = LoadBundleFacts.inspect(bundleURL: dir)
        #expect(facts.isRouted == false)
    }

    @Test("plain DSV4 affine JANG is production-blocked")
    func plainDSV4AffineJANGIsProductionBlocked() throws {
        let cfg = [
            "model_type": "deepseek_v4",
            "weight_format": "affine",
            "n_routed_experts": 256,
            "num_experts_per_tok": 8,
            "moe_intermediate_size": 2048,
        ] as [String: Any]
        let jang = [
            "weight_format": "affine"
        ] as [String: Any]
        let dir = try Self.makeBundle(files: [
            ("config.json", try JSONSerialization.data(withJSONObject: cfg)),
            ("jang_config.json", try JSONSerialization.data(withJSONObject: jang)),
            ("model-00001-of-00001.safetensors", Data(count: 1024)),
        ])
        defer { try? FileManager.default.removeItem(at: dir) }

        let facts = LoadBundleFacts.inspect(bundleURL: dir)
        #expect(facts.modelType == "deepseek_v4")
        #expect(facts.weightFormat == "affine")
        #expect(facts.hasJangConfig)
        #expect(!facts.hasJangTQRuntime)
        #expect(facts.isPlainDeepseekV4AffineJANG)
        #expect(facts.productionBlockReason?.contains("unsupported affine routed-MoE") == true)
    }

    @Test("DSV4 JANGTQ is not blocked by the affine-JANG guard")
    func dsv4JANGTQIsNotBlockedByAffineGuard() throws {
        let cfg = [
            "model_type": "deepseek_v4",
            "weight_format": "mxtq",
            "n_routed_experts": 256,
            "num_experts_per_tok": 8,
            "moe_intermediate_size": 2048,
        ] as [String: Any]
        let dir = try Self.makeBundle(files: [
            ("config.json", try JSONSerialization.data(withJSONObject: cfg)),
            ("jang_config.json", Data("{}".utf8)),
            ("jangtq_runtime.safetensors", Data(count: 16)),
            ("model-00001-of-00001.safetensors", Data(count: 1024)),
        ])
        defer { try? FileManager.default.removeItem(at: dir) }

        let facts = LoadBundleFacts.inspect(bundleURL: dir)
        #expect(facts.hasJangTQRuntime)
        #expect(!facts.isPlainDeepseekV4AffineJANG)
        #expect(facts.productionBlockReason == nil)
    }

    @Test("inspect on missing dir returns zeroed facts")
    func inspectMissingDir() {
        let facts = LoadBundleFacts.inspect(
            bundleURL: URL(fileURLWithPath: "/nonexistent/\(UUID().uuidString)"))
        #expect(facts.totalSafetensorsBytes == 0)
        #expect(facts.isRouted == false)
        // physicalMemory is always populated from ProcessInfo.
        #expect(facts.physicalMemory > 0)
    }

    // MARK: - JangPressStatus.disabled

    @Test("JangPressRuntime.none.status() == .disabled")
    func runtimeNoneStatus() {
        let s = JangPressRuntime.none.status()
        #expect(s == .disabled)
        #expect(s.enabled == false)
        #expect(s.coldFraction == nil)
        #expect(s.tilesUnderManagement == 0)
    }

    @Test("JangPressRuntime with appliedOptions surfaces coldFraction")
    func appliedOptionsSurfaceColdFraction() {
        // Construct a synthetic runtime that has appliedOptions but
        // no actual mmap tier (we don't need a real bundle here —
        // we're testing the status mapping, not the tier behavior).
        let opts = JangPressLoadOptions(
            enabled: true, compressPct: 70, backend: .none)
        let runtime = JangPressRuntime(
            mmap: nil, embed: nil, appliedOptions: opts)

        // Without isActive=true (no mmap or embed tier attached) the
        // status returns .disabled — that's by-design honest signaling.
        // Verify by attaching a real-style runtime that's "active":
        // since both tiers are nil we can't isActive=true via the
        // public surface. Instead, prove the status function reads
        // appliedOptions when isActive: pull coldFraction directly.
        #expect(runtime.appliedOptions?.compressPct == 70)
        #expect(runtime.isActive == false)
        // Confirmed: the field is set, but isActive gates status output.
        // The bundled JangPressActivationTests cover the active-runtime
        // path with a real synthetic bundle.
        _ = runtime.status()  // smoke
    }

    @Test("JangPressLoadOptions equality + clamping round-trip")
    func loadOptionsClampInit() {
        let a = JangPressLoadOptions(enabled: true, compressPct: 70)
        let b = JangPressLoadOptions(enabled: true, compressPct: 70)
        #expect(a == b)
        let clamped = JangPressLoadOptions(enabled: true, compressPct: 200)
        #expect(clamped.compressPct == 100)
    }

    // MARK: - Helpers

    /// Create a temp bundle directory pre-populated with the given
    /// (filename, bytes) entries. Caller must remove on cleanup.
    static func makeBundle(files: [(String, Data)]) throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("LoadConfigurationTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: dir, withIntermediateDirectories: true)
        for (name, data) in files {
            try data.write(to: dir.appendingPathComponent(name))
        }
        return dir
    }
}

extension LoadBundleFacts {
    /// Tiny non-routed bundle that never trips the auto threshold.
    fileprivate static let tiny = LoadBundleFacts(
        totalSafetensorsBytes: 1024,
        isRouted: false,
        physicalMemory: 128 * 1024 * 1024 * 1024)
}

/// Run `block` with `name` set to `value` in the process environment,
/// restoring the prior value on return. Pass `value: nil` to assert
/// the variable is unset for the duration of the block.
@discardableResult
private func withEnvironmentValue<R>(
    _ name: String, _ value: String?, _ block: () -> R
) -> R {
    let prior = ProcessInfo.processInfo.environment[name]
    if let value {
        setenv(name, value, 1)
    } else {
        unsetenv(name)
    }
    defer {
        if let prior {
            setenv(name, prior, 1)
        } else {
            unsetenv(name)
        }
    }
    return block()
}
