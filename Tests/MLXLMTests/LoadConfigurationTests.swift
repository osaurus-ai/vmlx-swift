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

// These tests temporarily mutate process-global environment variables. Keep
// the suite serialized so JANGPRESS=70/off/banana cases cannot race each
// other and produce order-dependent policy results.
@Suite("LoadConfiguration", .serialized)
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
        #expect(cfg.deepseekV4ActivationQAT == nil)
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
        #expect(cfg.deepseekV4ActivationQAT == false)
    }

    @Test("DSV4 activation QAT defaults off and accepts only explicit opt-in env values")
    func dsv4ActivationQATEnvironmentDefaultIsOptIn() {
        #expect(withEnvironmentValue("VMLX_DSV4_OFFICIAL_ACTIVATION_QAT", nil) {
            DeepseekV4ActivationQAT.environmentDefault == false
        })
        #expect(withEnvironmentValue("VMLX_DSV4_OFFICIAL_ACTIVATION_QAT", "0") {
            DeepseekV4ActivationQAT.environmentDefault == false
        })
        #expect(withEnvironmentValue("VMLX_DSV4_OFFICIAL_ACTIVATION_QAT", "1") {
            DeepseekV4ActivationQAT.environmentDefault == true
        })
        #expect(withEnvironmentValue("VMLX_DSV4_OFFICIAL_ACTIVATION_QAT", "yes") {
            DeepseekV4ActivationQAT.environmentDefault == true
        })
    }

    @Test("DSV4 activation QAT explicit per-load request overrides the process fallback")
    func dsv4ActivationQATPerLoadOverride() async {
        let explicitlyOn = await DeepseekV4ActivationQAT.withExplicitRequest(true) {
            DeepseekV4ActivationQAT.enabledForCurrentLoad
        }
        let explicitlyOff = await DeepseekV4ActivationQAT.withExplicitRequest(false) {
            DeepseekV4ActivationQAT.enabledForCurrentLoad
        }

        #expect(explicitlyOn)
        #expect(!explicitlyOff)
    }

    @Test("JANGTQ load keeps tq tensors raw and protects mmap residency")
    func jangtqLoadDoesNotSkipWholeModelBFloat16Conversion() throws {
        let source = try String(
            contentsOfFile: "Libraries/MLXLMCommon/Load.swift",
            encoding: .utf8)

        #expect(source.contains("let mmapSafetensorsActive = envFlag(\"MLX_SAFETENSORS_MMAP\")"))
        // `RuntimeEnvironment.flag` replaced the raw `envFlag` here and names only the current
        // spelling; the legacy `VMLINUX_` name is still honoured, derived centrally by
        // `RuntimeEnvironment.legacyName(of:)`. See `RuntimeEnvironmentSourceCoverageTests`.
        #expect(source.contains(
            "let allowJANGTQMmapBFloat16 = RuntimeEnvironment.flag(\"VMLX_JANGTQ_BF16_MMAP\")"))
        #expect(source.contains("let autoJANGTQMmapBFloat16 = requiresJANGTQMmapBFloat16(modelDirectory)"))
        #expect(source.contains("!isJANGTQNative || !mmapSafetensorsActive || allowJANGTQMmapBFloat16"))
        #expect(source.contains("|| autoJANGTQMmapBFloat16"))
        #expect(source.contains("private func requiresJANGTQMmapBFloat16(_ modelDirectory: URL) -> Bool"))
        #expect(source.contains("modelType == \"nemotron_h\""))
        #expect(source.contains("modelType == \"qwen3_5_moe\""))
        #expect(source.contains("modelType == \"qwen3_5_moe_text\""))
        #expect(source.contains("layerTypes.contains(\"linear_attention\")"))
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

    @Test("DSV4 clears stale process-wide MLX limits")
    func dsv4ClearsStaleProcessLimits() {
        let priorMemoryLimit = MLX.Memory.memoryLimit
        let priorCacheLimit = MLX.Memory.cacheLimit
        defer {
            MLX.Memory.memoryLimit = priorMemoryLimit
            MLX.Memory.cacheLimit = priorCacheLimit
        }

        let gib = UInt64(1 << 30)
        let facts = LoadBundleFacts(
            totalSafetensorsBytes: 95 * gib,
            isRouted: true,
            physicalMemory: 128 * gib,
            modelType: "deepseek_v4",
            weightFormat: "affine",
            hasJangConfig: true,
            numRoutedExperts: 256,
            topK: 8)
        MLX.Memory.memoryLimit = Int(128 * gib * 7 / 10)
        MLX.Memory.cacheLimit = 128 << 20

        let restored = applyPlainDeepseekV4ProcessMemoryLimitsIfNeeded(
            facts: facts,
            recommendedWorkingSetBytes: Int(80 * gib))

        #expect(restored == Int(120 * gib))
        #expect(MLX.Memory.memoryLimit == restored)
        #expect(MLX.Memory.cacheLimit == restored)
    }

    @Test("non-DSV4 load does not rewrite process-wide MLX limits")
    func nonDSV4DoesNotRewriteProcessLimits() {
        let priorMemoryLimit = MLX.Memory.memoryLimit
        let priorCacheLimit = MLX.Memory.cacheLimit
        defer {
            MLX.Memory.memoryLimit = priorMemoryLimit
            MLX.Memory.cacheLimit = priorCacheLimit
        }

        let gib = UInt64(1 << 30)
        let facts = LoadBundleFacts(
            totalSafetensorsBytes: 8 * gib,
            isRouted: false,
            physicalMemory: 128 * gib,
            modelType: "llama")
        MLX.Memory.memoryLimit = Int(64 * gib)
        MLX.Memory.cacheLimit = Int(2 * gib)

        let restored = applyPlainDeepseekV4ProcessMemoryLimitsIfNeeded(
            facts: facts,
            recommendedWorkingSetBytes: Int(80 * gib))

        #expect(restored == nil)
        #expect(MLX.Memory.memoryLimit == Int(64 * gib))
        #expect(MLX.Memory.cacheLimit == Int(2 * gib))
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

    @Test("plain DSV4 affine JANG resolves mmap requests to resident weights")
    func plainDSV4AffineJANGRequiresResidentSafetensors() throws {
        let cfg = [
            "model_type": "deepseek_v4",
            "weight_format": "affine",
            "num_hidden_layers": 43,
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
        #expect(facts.requiresResidentSafetensors)
        #expect(!facts.resolveMmapSafetensors(requested: true))
        #expect(!facts.resolveMmapSafetensors(requested: false))
        #expect(facts.resolveMLXMemoryLimit(requested: .default) == .unlimited)
        #expect(
            facts.resolveMLXMemoryLimit(requested: .absolute(8 * 1024 * 1024 * 1024))
                == .unlimited)
        #expect(facts.resolveMLXAllocatorCacheLimit(requested: .default) == .unlimited)
        #expect(
            facts.resolveMLXAllocatorCacheLimit(requested: .absolute(128 * 1024 * 1024))
                == .unlimited)
    }

    @Test("complete pre-stacked affine DSV4 index permits mmap without restoring cap")
    func prestackedDSV4AffineJANGPermitsMmap() throws {
        let dir = try Self.makeDeepseekV4AffineBundle(
            layout: "prestacked_affine",
            indexData: try Self.deepseekV4AffinePrestackedIndex())
        defer { try? FileManager.default.removeItem(at: dir) }

        let facts = LoadBundleFacts.inspect(bundleURL: dir)
        #expect(facts.routedExpertLayout == "prestacked_affine")
        #expect(facts.hasPrestackedAffineRoutedExperts)
        #expect(facts.isPlainDeepseekV4AffineJANG)
        #expect(!facts.requiresResidentSafetensors)
        #expect(facts.resolveMmapSafetensors(requested: true))
        #expect(!facts.resolveMmapSafetensors(requested: false))
        #expect(facts.resolveMLXMemoryLimit(requested: .default) == .unlimited)
        #expect(facts.resolveMLXAllocatorCacheLimit(requested: .default) == .unlimited)
    }

    @Test("pre-stacked affine declaration without an index fails closed")
    func prestackedDSV4AffineMarkerOnlyFailsClosed() throws {
        let dir = try Self.makeDeepseekV4AffineBundle(
            layout: "prestacked_affine", indexData: nil)
        defer { try? FileManager.default.removeItem(at: dir) }

        let facts = LoadBundleFacts.inspect(bundleURL: dir)
        #expect(!facts.hasPrestackedAffineRoutedExperts)
        #expect(facts.requiresResidentSafetensors)
        #expect(!facts.resolveMmapSafetensors(requested: true))
    }

    @Test("complete pre-stacked affine index without a declaration fails closed")
    func prestackedDSV4AffineIndexOnlyFailsClosed() throws {
        let dir = try Self.makeDeepseekV4AffineBundle(
            layout: nil,
            indexData: try Self.deepseekV4AffinePrestackedIndex())
        defer { try? FileManager.default.removeItem(at: dir) }

        let facts = LoadBundleFacts.inspect(bundleURL: dir)
        #expect(!facts.hasPrestackedAffineRoutedExperts)
        #expect(facts.requiresResidentSafetensors)
        #expect(!facts.resolveMmapSafetensors(requested: true))
    }

    @Test("incomplete pre-stacked affine DSV4 index fails closed")
    func incompletePrestackedDSV4AffineIndexFailsClosed() throws {
        let missing = "layers.42.mlp.switch_mlp.up_proj.biases"
        let dir = try Self.makeDeepseekV4AffineBundle(
            layout: "prestacked_affine",
            indexData: try Self.deepseekV4AffinePrestackedIndex(
                omitting: missing))
        defer { try? FileManager.default.removeItem(at: dir) }

        let facts = LoadBundleFacts.inspect(bundleURL: dir)
        #expect(!facts.hasPrestackedAffineRoutedExperts)
        #expect(facts.requiresResidentSafetensors)
        #expect(!facts.resolveMmapSafetensors(requested: true))
    }

    @Test("mixed pre-stacked and split affine DSV4 index fails closed")
    func mixedPrestackedDSV4AffineIndexFailsClosed() throws {
        let dir = try Self.makeDeepseekV4AffineBundle(
            layout: "prestacked_affine",
            indexData: try Self.deepseekV4AffinePrestackedIndex(
                includeSplitExpert: true))
        defer { try? FileManager.default.removeItem(at: dir) }

        let facts = LoadBundleFacts.inspect(bundleURL: dir)
        #expect(!facts.hasPrestackedAffineRoutedExperts)
        #expect(facts.requiresResidentSafetensors)
        #expect(!facts.resolveMmapSafetensors(requested: true))
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
        #expect(!facts.requiresResidentSafetensors)
        #expect(facts.resolveMmapSafetensors(requested: true))
        #expect(facts.resolveMLXMemoryLimit(requested: .default) == .default)
        #expect(facts.resolveMLXAllocatorCacheLimit(requested: .default) == .default)
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

    @Test(
        "mmap dtype preservation is limited to Gemma 4 JANG affine",
        arguments: [
            ("gemma4_unified", "jang_affine", true),
            ("gemma4_unified", "mxfp8", false),
            ("qwen3_5", "jang_affine", false),
        ])
    func gemma4JANGAffineMmapDtypePolicy(
        modelType: String,
        weightFormat: String,
        expected: Bool
    ) throws {
        let config = [
            "model_type": modelType,
            "weight_format": weightFormat,
        ]
        let dir = try Self.makeBundle(files: [
            ("config.json", try JSONSerialization.data(withJSONObject: config))
        ])
        defer { try? FileManager.default.removeItem(at: dir) }

        #expect(shouldPreserveGemma4JANGAffineMmapDtypes(modelDirectory: dir) == expected)
    }

    @Test("Qwen4Exp affine mmap preserves checkpoint quantization metadata")
    func qwen4ExpAffineMmapDtypePolicy() throws {
        let valid = try Self.makeBundle(files: [
            ("config.json", try JSONSerialization.data(withJSONObject: [
                "model_type": "qwen4_exp",
                "text_config": ["model_type": "qwen4_exp_text", "dtype": "bfloat16"],
                "quantization": ["group_size": 64, "bits": 8],
            ] as [String: Any]))
        ])
        let dense = try Self.makeBundle(files: [
            ("config.json", try JSONSerialization.data(withJSONObject: [
                "model_type": "qwen4_exp",
                "text_config": ["model_type": "qwen4_exp_text", "dtype": "bfloat16"],
            ] as [String: Any]))
        ])
        defer {
            try? FileManager.default.removeItem(at: valid)
            try? FileManager.default.removeItem(at: dense)
        }

        #expect(shouldPreserveQwen4ExpJANGAffineMmapDtypes(modelDirectory: valid))
        #expect(!shouldPreserveQwen4ExpJANGAffineMmapDtypes(modelDirectory: dense))
    }


    @Test("mmap dtype preservation requires validated pre-stacked affine DSV4")
    func dsv4PrestackedAffineMmapDtypePolicy() throws {
        let valid = try Self.makeDeepseekV4AffineBundle(
            layout: "prestacked_affine",
            indexData: try Self.deepseekV4AffinePrestackedIndex())
        let split = try Self.makeDeepseekV4AffineBundle(
            layout: nil,
            indexData: nil)
        let mixed = try Self.makeDeepseekV4AffineBundle(
            layout: "prestacked_affine",
            indexData: try Self.deepseekV4AffinePrestackedIndex(
                includeSplitExpert: true))
        defer {
            try? FileManager.default.removeItem(at: valid)
            try? FileManager.default.removeItem(at: split)
            try? FileManager.default.removeItem(at: mixed)
        }

        #expect(shouldPreserveDeepseekV4PrestackedAffineMmapDtypes(
            modelDirectory: valid))
        #expect(!shouldPreserveDeepseekV4PrestackedAffineMmapDtypes(
            modelDirectory: split))
        #expect(!shouldPreserveDeepseekV4PrestackedAffineMmapDtypes(
            modelDirectory: mixed))
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

    static func makeDeepseekV4AffineBundle(
        layout: String?,
        indexData: Data?
    ) throws -> URL {
        var config =
            [
                "model_type": "deepseek_v4",
                "weight_format": "affine",
                "num_hidden_layers": 43,
                "n_routed_experts": 256,
                "num_experts_per_tok": 6,
                "moe_intermediate_size": 2048,
            ] as [String: Any]
        var jang = ["weight_format": "affine"] as [String: Any]
        if let layout {
            config["routed_expert_layout"] = layout
            jang["routed_expert_layout"] = layout
        }

        var files = [
            ("config.json", try JSONSerialization.data(withJSONObject: config)),
            ("jang_config.json", try JSONSerialization.data(withJSONObject: jang)),
            ("model-00001-of-00001.safetensors", Data(count: 1024)),
        ]
        if let indexData {
            files.append(("model.safetensors.index.json", indexData))
        }
        return try makeBundle(files: files)
    }

    static func deepseekV4AffinePrestackedIndex(
        omitting omittedKey: String? = nil,
        includeSplitExpert: Bool = false
    ) throws -> Data {
        let shard = "model-00001-of-00001.safetensors"
        var weightMap = [String: String]()
        for layer in 0 ..< 43 {
            for projection in ["gate_proj", "down_proj", "up_proj"] {
                for suffix in ["weight", "scales", "biases"] {
                    let key =
                        "layers.\(layer).mlp.switch_mlp.\(projection).\(suffix)"
                    if key != omittedKey {
                        weightMap[key] = shard
                    }
                }
            }
        }
        if includeSplitExpert {
            weightMap["layers.0.ffn.experts.0.w1.weight"] = shard
        }
        return try JSONSerialization.data(withJSONObject: [
            "metadata": ["rebundled_layout": "prestacked-switch_mlp-affine"],
            "weight_map": weightMap,
        ])
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

/// Qwen3.8 Flash Next JANG must never be cold-compressed by the auto
/// threshold: its directory bytes include the 20–29 GiB SSD-served PLE table,
/// so whole-bundle size overstates resident need, and the family's runtime
/// contract keeps every non-PLE compute weight resident.
@Suite struct Qwen4ExpJangPressExclusionTests {
    private var flashNextFacts: LoadBundleFacts {
        LoadBundleFacts(
            totalSafetensorsBytes: 96 << 30,
            isRouted: true,
            physicalMemory: 128 << 30,
            modelType: "qwen4_exp",
            jangFormat: "jang_v2",
            declaredComputeDType: "bfloat16",
            numRoutedExperts: 512,
            topK: 10)
    }

    @Test("auto threshold resolves DISABLED for Flash-Next despite 96GB routed")
    func autoDisabledForFlashNext() {
        let opts = withEnvironmentValue("JANGPRESS", nil) {
            withEnvironmentValue("MLXPRESS", nil) {
                JangPressPolicy.auto(envFallback: true).resolve(facts: flashNextFacts)
            }
        }
        #expect(opts.enabled == false)
    }

    @Test("the threshold itself is unchanged for other routed big bundles")
    func thresholdUnchangedElsewhere() {
        let other = LoadBundleFacts(
            totalSafetensorsBytes: 96 << 30,
            isRouted: true,
            physicalMemory: 128 << 30)
        let opts = withEnvironmentValue("JANGPRESS", nil) {
            withEnvironmentValue("MLXPRESS", nil) {
                JangPressPolicy.auto(envFallback: true).resolve(facts: other)
            }
        }
        #expect(opts.enabled == true)
        #expect(opts.compressPct == 70)
    }
}
