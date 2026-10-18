import Foundation
import XCTest
@testable import MLXLMCommon

final class JANGHLoadPolicyTests: XCTestCase {
    private func config(modelType: String = "glm5_next") -> [String: Any] {
        let book: [String: Any] = ["alpha": 1, "beta": 0, "levels": [-1.5, -0.5, 0.5, 1.5]]
        var quant: [String: Any] = ["mode": "affine", "bits": 8, "group_size": 64]
        for role in ["gate_proj", "up_proj", "down_proj"] {
            quant["model.layers.3.mlp.switch_mlp." + role] = ["mode": "jangtq2", "bits": 2, "rotation": "hadamard32"]
        }
        return ["model_type": modelType, "n_routed_experts": 256, "num_experts_per_tok": 8,
                "quantization": quant,
                "jangtq": ["version": 2, "packing": "lsb-bitstream", "scale_dtype": "float16",
                           "codebook_family": "odd-cubic", "rotation": "hadamard32", "codebooks": ["2": book]]]
    }
    private func inspect(_ config: [String: Any], sidecar: [String: Any]? = nil) throws -> LoadBundleFacts {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try JSONSerialization.data(withJSONObject: config).write(to: directory.appendingPathComponent("config.json"))
        if let sidecar {
            try JSONSerialization.data(withJSONObject: sidecar).write(to: directory.appendingPathComponent("jang_config.json"))
        }
        return LoadBundleFacts.inspect(bundleURL: directory)
    }

    func testValidatedCustomMetadataDoesNotSelectAffineResidentPolicy() throws {
        for type in ["glm5_next", "naive_n05_flash"] {
            let root = config(modelType: type)
            let facts = try inspect(root, sidecar: ["format": "jangtq2", "format_version": 2, "jangtq": root["jangtq"]!])
            XCTAssertEqual(facts.customRoutedFormat, .janghV2)
            XCTAssertFalse(facts.isGlm5NextAffineJANG)
            XCTAssertFalse(facts.requiresResidentSafetensors)
            XCTAssertFalse(facts.requiresUncappedResidentPools)
            XCTAssertTrue(facts.resolveMmapSafetensors(requested: true))
            XCTAssertFalse(facts.resolveMmapSafetensors(requested: false))
            for cap in [ResidentCap.fraction(0.7), .absolute(12345678), .unlimited] {
                XCTAssertEqual(facts.resolveMLXMemoryLimit(requested: cap), cap)
                XCTAssertEqual(facts.resolveMLXAllocatorCacheLimit(requested: cap), cap)
            }
            XCTAssertFalse(JangPressPolicy.disabled.resolve(facts: facts).enabled)
        }
    }

    func testMiniCPMV47JANGH2ShipsUnstackedRoutedBanksAndLoadsResident() throws {
        let root = config(modelType: "minicpmv4_7")
        let facts = try inspect(root, sidecar: ["format": "jangtq2", "format_version": 2, "jangtq": root["jangtq"]!])
        XCTAssertEqual(facts.customRoutedFormat, .janghV2)
        XCTAssertTrue(facts.isMiniCPMV47UnstackedJANGH)
        XCTAssertTrue(facts.requiresResidentSafetensors)
        XCTAssertFalse(facts.resolveMmapSafetensors(requested: true))
        // The measured 24 GB row does not need the uncapped-pool policy the big
        // affine families require; keep the opt-in narrow.
        XCTAssertFalse(facts.requiresUncappedResidentPools)
        for cap in [ResidentCap.fraction(0.7), .absolute(12345678), .unlimited] {
            XCTAssertEqual(facts.resolveMLXMemoryLimit(requested: cap), cap)
            XCTAssertEqual(facts.resolveMLXAllocatorCacheLimit(requested: cap), cap)
        }

        // Other JANGH2 families keep the caller's mmap choice.
        for type in ["glm5_next", "naive_n05_flash"] {
            let otherRoot = config(modelType: type)
            let other = try inspect(otherRoot, sidecar: ["format": "jangtq2", "format_version": 2, "jangtq": otherRoot["jangtq"]!])
            XCTAssertFalse(other.isMiniCPMV47UnstackedJANGH)
            XCTAssertTrue(other.resolveMmapSafetensors(requested: true))
        }

        // A label-only minicpmv4_7 bundle without a validated custom bank must
        // not masquerade into the resident policy.
        let labelOnly = LoadBundleFacts(totalSafetensorsBytes: 12 << 30, isRouted: true,
            physicalMemory: 24 << 30, modelType: "minicpmv4_7", jangFormat: "jangtq2",
            hasJangConfig: true, numRoutedExperts: 64)
        XCTAssertEqual(labelOnly.customRoutedFormat, .none)
        XCTAssertFalse(labelOnly.isMiniCPMV47UnstackedJANGH)
        XCTAssertFalse(labelOnly.requiresResidentSafetensors)
    }

    func testMalformedCustomAndLabelOnlyCannotMasqueradeAsAffine() throws {
        var malformed = config()
        malformed.removeValue(forKey: "jangtq")
        let facts = try inspect(malformed)
        XCTAssertEqual(facts.customRoutedFormat, .invalidJANGHDeclaration)
        XCTAssertFalse(facts.isGlm5NextAffineJANG)
        XCTAssertFalse(facts.requiresUncappedResidentPools)
        let labelOnly = LoadBundleFacts(totalSafetensorsBytes: 96 << 30, isRouted: true,
            physicalMemory: 128 << 30, modelType: "glm5_next", jangFormat: "jangtq2",
            hasJangConfig: true, numRoutedExperts: 256)
        XCTAssertEqual(labelOnly.customRoutedFormat, .none)
        XCTAssertFalse(labelOnly.isGlm5NextAffineJANG)
    }

    func testOrdinaryAffineAndLegacyPoliciesRemainDistinct() throws {
        let affine = LoadBundleFacts(totalSafetensorsBytes: 96 << 30, isRouted: true,
            physicalMemory: 128 << 30, modelType: "glm5_next", weightFormat: "affine",
            hasJangConfig: true, numRoutedExperts: 256)
        XCTAssertTrue(affine.isGlm5NextAffineJANG)
        XCTAssertTrue(affine.requiresResidentSafetensors)
        XCTAssertTrue(affine.requiresUncappedResidentPools)
        let legacy = LoadBundleFacts(totalSafetensorsBytes: 96 << 30, isRouted: true,
            physicalMemory: 128 << 30, modelType: "glm5_next", weightFormat: "mxtq",
            hasJangConfig: true, hasJangTQRuntime: true, numRoutedExperts: 256)
        XCTAssertEqual(legacy.customRoutedFormat, .none)
        XCTAssertFalse(legacy.isGlm5NextAffineJANG)
    }
    func testAutomaticSchedulingBudgetDoesNotDrainAdmittedMappedWeights() throws {
        for type in ["glm5_next", "naive_n05_flash"] {
            var facts = try inspect(config(modelType: type))
            facts.totalSafetensorsBytes = 96 << 30
            facts.physicalMemory = 128 << 30
            let automatic = LoadConfiguration()
            XCTAssertFalse(automatic.memoryLimitWasExplicit)
            XCTAssertEqual(automatic.resolvedSchedulingMemoryLimit(facts: facts,
                recommendedWorkingSetBytes: 107 << 30), .absolute(100 << 30))
            XCTAssertTrue(facts.resolveMmapSafetensors(requested: true))
            XCTAssertFalse(facts.requiresUncappedResidentPools)
            for limit in [ResidentCap.fraction(0.7), .absolute(4 << 30), .unlimited] {
                var explicit = LoadConfiguration(memoryLimit: limit)
                XCTAssertTrue(explicit.memoryLimitWasExplicit)
                XCTAssertEqual(explicit.resolvedSchedulingMemoryLimit(facts: facts,
                    recommendedWorkingSetBytes: 107 << 30), limit)
                explicit = LoadConfiguration()
                explicit.memoryLimit = limit
                XCTAssertTrue(explicit.memoryLimitWasExplicit)
                XCTAssertEqual(explicit.resolvedSchedulingMemoryLimit(facts: facts,
                    recommendedWorkingSetBytes: 107 << 30), limit)
            }
        }
    }

    func testAutomaticSchedulingBudgetClampsScratchRatherThanDroppingBelowWeights() throws {
        var facts = try inspect(config(modelType: "qwen4_exp"))
        facts.totalSafetensorsBytes = 36_971_767_160
        facts.physicalMemory = 48 << 30
        let automatic = LoadConfiguration()
        for ceilingGiB in [36, 37, 38] {
            let ceiling = UInt64(ceilingGiB) << 30
            XCTAssertEqual(automatic.resolvedSchedulingMemoryLimit(facts: facts,
                recommendedWorkingSetBytes: Int(ceiling)), .absolute(ceiling))
        }
        // Plenty of GPU budget: retain the original four-GiB allowance.
        XCTAssertEqual(automatic.resolvedSchedulingMemoryLimit(facts: facts,
            recommendedWorkingSetBytes: 42 << 30),
            .absolute(facts.gpuResidentWeightBytes + (4 << 30)))
        // An explicit user limit is never silently raised.
        let explicit = LoadConfiguration(memoryLimit: .fraction(0.70))
        XCTAssertEqual(explicit.resolvedSchedulingMemoryLimit(facts: facts,
            recommendedWorkingSetBytes: 38 << 30), .fraction(0.70))
        // Missing capacity is not invented by this scheduling policy.
        XCTAssertEqual(automatic.resolvedSchedulingMemoryLimit(facts: facts,
            recommendedWorkingSetBytes: 34 << 30), .default)
    }

    func testAutomaticSchedulingBudgetRefusesOversizedUnknownAndOverflowRows() throws {
        var facts = try inspect(config())
        facts.totalSafetensorsBytes = 96 << 30
        facts.physicalMemory = 128 << 30
        let automatic = LoadConfiguration()
        for workingSet: Int? in [nil, 0, 96 << 30] {
            XCTAssertEqual(automatic.resolvedSchedulingMemoryLimit(facts: facts,
                recommendedWorkingSetBytes: workingSet), .default)
        }
        facts.physicalMemory = 112 << 30
        XCTAssertEqual(automatic.resolvedSchedulingMemoryLimit(facts: facts,
            recommendedWorkingSetBytes: 107 << 30), .default)
        facts.physicalMemory = 128 << 30
        for bytes: UInt64 in [0, 8 << 30, UInt64.max] {
            facts.totalSafetensorsBytes = bytes
            XCTAssertEqual(automatic.resolvedSchedulingMemoryLimit(facts: facts,
                recommendedWorkingSetBytes: 107 << 30), .default)
        }
        let ordinary = LoadBundleFacts(totalSafetensorsBytes: 96 << 30, isRouted: true,
            physicalMemory: 128 << 30, modelType: "naive_n05_flash")
        XCTAssertEqual(automatic.resolvedSchedulingMemoryLimit(facts: ordinary,
            recommendedWorkingSetBytes: 107 << 30), .default)
    }

    func testSafeAutoHostPlanKeepsJANGHMappedAndResolvesSchedulingHeadroom() throws {
        for type in ["glm5_next", "naive_n05_flash"] {
            var facts = try inspect(config(modelType: type))
            facts.totalSafetensorsBytes = 96 << 30
            facts.physicalMemory = 128 << 30
            let host = MemoryStatus(memoryLimit: 0, cacheLimit: 0,
                recommendedWorkingSetBytes: 107 << 30, physicalMemory: 128 << 30,
                currentRSS: 0)
            var settings = VMLXServerRuntimeSettings()
            let plan = settings.resolvedMemorySafetyPlan(bundleFacts: facts, host: host)
            XCTAssertEqual(plan.loadConfiguration.memoryLimit, .absolute(100 << 30))
            XCTAssertTrue(plan.loadConfiguration.useMmapSafetensors)
            XCTAssertEqual(plan.loadConfiguration.maxResidentBytes, .absolute(128 << 20))
            XCTAssertFalse(plan.warnings.contains { $0.contains("loading materialized") })
            settings.memorySafety.customPhysicalMemoryFraction = 0.65
            let explicit = settings.resolvedMemorySafetyPlan(bundleFacts: facts, host: host)
            XCTAssertEqual(explicit.loadConfiguration.memoryLimit, .fraction(0.65))
            XCTAssertTrue(explicit.loadConfiguration.useMmapSafetensors)
            settings.memorySafety.customPhysicalMemoryFraction = nil
            settings.memorySafety.mode = .strict
            let strict = settings.resolvedMemorySafetyPlan(bundleFacts: facts, host: host)
            XCTAssertEqual(strict.loadConfiguration.memoryLimit, .fraction(0.60))
            XCTAssertTrue(strict.loadConfiguration.useMmapSafetensors)
        }
    }

    func testSafeAutoHostPlanDoesNotInventUnknownWorkingSetOrMaterializeCustomBanks() throws {
        var facts = try inspect(config())
        facts.totalSafetensorsBytes = 96 << 30
        facts.physicalMemory = 128 << 30
        let host = MemoryStatus(memoryLimit: 0, cacheLimit: 0,
            recommendedWorkingSetBytes: nil, physicalMemory: 128 << 30, currentRSS: 0)
        let settings = VMLXServerRuntimeSettings()
        let plan = settings.resolvedMemorySafetyPlan(bundleFacts: facts, host: host)
        XCTAssertEqual(plan.loadConfiguration.memoryLimit, .default)
        XCTAssertTrue(plan.loadConfiguration.useMmapSafetensors)
        for mode in [VMLXMemorySafetyMode.balanced, .performance] {
            var settings = settings
            settings.memorySafety.mode = mode
            let plan = settings.resolvedMemorySafetyPlan(bundleFacts: facts, host: host)
            XCTAssertTrue(plan.loadConfiguration.useMmapSafetensors)
            XCTAssertFalse(plan.warnings.contains { $0.contains("loading materialized") })
        }
    }

    func testResidencyPreferencePersistsAndReachesBothLoadResolvers() throws {
        let defaults = try JSONDecoder().decode(VMLXMemorySafetySettings.self, from: Data("{}".utf8))
        XCTAssertTrue(defaults.keepModelInMemory)
        var settings = VMLXServerRuntimeSettings()
        for enabled in [false, true] {
            settings.memorySafety.keepModelInMemory = enabled
            let saved = try JSONEncoder().encode(settings)
            let restored = try JSONDecoder().decode(VMLXServerRuntimeSettings.self, from: saved)
            XCTAssertEqual(restored.memorySafety.keepModelInMemory, enabled)
            XCTAssertEqual(restored.resolvedLoadConfiguration(
                configData: nil, jangConfig: nil, status: nil).keepModelInMemory, enabled)
            XCTAssertEqual(restored.resolvedMemorySafetyPlan().loadConfiguration.keepModelInMemory, enabled)
        }
    }

}
