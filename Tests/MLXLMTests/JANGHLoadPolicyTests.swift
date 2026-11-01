import Foundation
import MLX
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

    /// A Hugging Face hub cache stores every file as a symlink into `blobs/`. The weights must be sized by
    /// their targets: sized by the links (a few dozen bytes each), a 70 GB model was wired as ~10 KB.
    func testInspectSizesSymlinkedShardsByTheirTargets() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let blobs = root.appendingPathComponent("blobs")
        let snapshot = root.appendingPathComponent("snapshots/0123abcd")
        try FileManager.default.createDirectory(at: blobs, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: snapshot, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try Data(count: 4096).write(to: blobs.appendingPathComponent("aaaa"))
        try FileManager.default.createSymbolicLink(
            atPath: snapshot.appendingPathComponent("model-00001-of-00002.safetensors").path,
            withDestinationPath: "../../blobs/aaaa")
        try Data(count: 1000).write(to: snapshot.appendingPathComponent("model-00002-of-00002.safetensors"))
        try JSONSerialization.data(withJSONObject: config()).write(to: snapshot.appendingPathComponent("config.json"))

        XCTAssertEqual(LoadBundleFacts.inspect(bundleURL: snapshot).totalSafetensorsBytes, 4096 + 1000)
    }

    /// The bundle DIRECTORY reached through a symlink (models kept on another volume and linked
    /// into the models folder) must report its target's facts. Listing the unresolved URL throws
    /// ENOTDIR, which read as a 0-byte bundle: no residency budget, so DeepSeek-V4.1 JANG_2K
    /// (197 GiB) on a 256 GiB Mac loaded unwired and timed out on the GPU.
    func testInspectFollowsASymlinkedBundleDirectory() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let real = root.appendingPathComponent("volume/DeepSeek-V4.1-Flash-1017-JANG_2K")
        let link = root.appendingPathComponent("models/DeepSeek-V4.1-Flash-1017-JANG_2K")
        try FileManager.default.createDirectory(at: real, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(
            at: link.deletingLastPathComponent(), withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try Data(count: 4096).write(to: real.appendingPathComponent("model-00001-of-00002.safetensors"))
        // A shard whose only tensor is an SSD n-gram table: counted, then subtracted from the
        // resident weights, through the same symlinked directory.
        let header = try JSONSerialization.data(withJSONObject: [
            "model.layers.0.ple.ngram_embedding.weight": [
                "dtype": "U8", "shape": [100], "data_offsets": [0, 100],
            ],
        ])
        var shard = Data()
        withUnsafeBytes(of: UInt64(header.count).littleEndian) { shard.append(contentsOf: $0) }
        shard.append(header)
        shard.append(Data(count: 100))
        try shard.write(to: real.appendingPathComponent("model-00002-of-00002.safetensors"))
        try JSONSerialization.data(withJSONObject: [
            "model_type": "deepseek_v41", "n_routed_experts": 384, "num_experts_per_tok": 6,
        ]).write(to: real.appendingPathComponent("config.json"))
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: real)

        let direct = LoadBundleFacts.inspect(bundleURL: real)
        let linked = LoadBundleFacts.inspect(bundleURL: link)
        XCTAssertEqual(direct.totalSafetensorsBytes, UInt64(4096 + shard.count))
        XCTAssertEqual(direct.ssdResidentTableBytes, 100)
        XCTAssertEqual(linked, direct)
        XCTAssertEqual(linked.gpuResidentWeightBytes, UInt64(4096 + shard.count - 100))
        XCTAssertEqual(linked.modelType, "deepseek_v41")

        // The residency budget is sized from these bytes: a positive target, not "nothing to wire".
        let gib = 1 << 30
        XCTAssertNotNil(modelResidencyWiredTarget(
            modelBytes: Int(linked.gpuResidentWeightBytes), workingSet: 222 * gib,
            physicalMemory: 256 * gib))
    }

    func testValidatedCustomMetadataDoesNotSelectAffineResidentPolicy() throws {
        for type in ["glm5_next", "naive_n05_flash"] {
            let root = config(modelType: type)
            let facts = try inspect(root, sidecar: ["format": "jangtq2", "format_version": 2, "jangtq": root["jangtq"]!])
            XCTAssertEqual(facts.customRoutedFormat, .janghV2)
            XCTAssertFalse(facts.isGlm5NextAffineJANG)
            XCTAssertFalse(facts.requiresResidentSafetensors)
            XCTAssertFalse(facts.requiresUncappedResidentPools)
            XCTAssertFalse(facts.resolveMmapSafetensors(requested: true))
            XCTAssertFalse(facts.resolveMmapSafetensors(requested: false))
            for cap in [ResidentCap.fraction(0.7), .absolute(12345678), .unlimited] {
                XCTAssertEqual(facts.resolveMLXMemoryLimit(requested: cap), cap)
                XCTAssertEqual(facts.resolveMLXAllocatorCacheLimit(requested: cap), cap)
            }
        }
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
            XCTAssertFalse(facts.resolveMmapSafetensors(requested: true))
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
            recommendedWorkingSetBytes: 107 << 30), .absolute(100 << 30))
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
            XCTAssertFalse(plan.loadConfiguration.useMmapSafetensors)
            XCTAssertEqual(plan.loadConfiguration.maxResidentBytes, .absolute(128 << 20))
            XCTAssertFalse(plan.warnings.contains { $0.contains("loading materialized") })
            settings.memorySafety.customPhysicalMemoryFraction = 0.65
            let explicit = settings.resolvedMemorySafetyPlan(bundleFacts: facts, host: host)
            XCTAssertEqual(explicit.loadConfiguration.memoryLimit, .fraction(0.65))
            XCTAssertFalse(explicit.loadConfiguration.useMmapSafetensors)
            settings.memorySafety.customPhysicalMemoryFraction = nil
            settings.memorySafety.mode = .strict
            let strict = settings.resolvedMemorySafetyPlan(bundleFacts: facts, host: host)
            XCTAssertEqual(strict.loadConfiguration.memoryLimit, .fraction(0.60))
            XCTAssertFalse(strict.loadConfiguration.useMmapSafetensors)
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
        XCTAssertFalse(plan.loadConfiguration.useMmapSafetensors)
        for mode in [VMLXMemorySafetyMode.balanced, .performance] {
            var settings = settings
            settings.memorySafety.mode = mode
            let plan = settings.resolvedMemorySafetyPlan(bundleFacts: facts, host: host)
            XCTAssertFalse(plan.loadConfiguration.useMmapSafetensors)
            XCTAssertFalse(plan.warnings.contains { $0.contains("loading materialized") })
        }
    }

    func testOwnedWeightSchedulingIsSharedAcrossFamilies() {
        for family in ["qwen3_5_moe", "nemotron_h", "minicpmv", "flint", "llama"] {
            let facts = LoadBundleFacts(totalSafetensorsBytes: 96 << 30, isRouted: family != "llama",
                physicalMemory: 128 << 30, modelType: family)
            XCTAssertEqual(LoadConfiguration().resolvedSchedulingMemoryLimit(facts: facts,
                recommendedWorkingSetBytes: 107 << 30), .absolute(100 << 30))
            let explicit = LoadConfiguration(memoryLimit: .absolute(64 << 30))
            XCTAssertEqual(explicit.resolvedSchedulingMemoryLimit(facts: facts,
                recommendedWorkingSetBytes: 107 << 30), .absolute(64 << 30))
            XCTAssertFalse(facts.resolveMmapSafetensors(requested: true))
        }
    }

    func testMissingWorkingSetDoesNotResetExistingNearRAMBudget() {
        let facts = LoadBundleFacts(totalSafetensorsBytes: 94 << 30, isRouted: true,
            physicalMemory: 128 << 30, modelType: "hy_v3", weightFormat: "jang-affine-mixed")
        let settings = VMLXServerRuntimeSettings()
        let unknown = MemoryStatus(memoryLimit: 0, cacheLimit: 0,
            recommendedWorkingSetBytes: nil, physicalMemory: 128 << 30, currentRSS: 0)
        for host in [nil, unknown] {
            let plan = settings.resolvedMemorySafetyPlan(bundleFacts: facts, host: host)
            XCTAssertEqual(plan.loadConfiguration.memoryLimit, .fraction(94.0 / 128.0 + 0.06))
            XCTAssertFalse(plan.loadConfiguration.useMmapSafetensors)
        }
        let known = MemoryStatus(memoryLimit: 0, cacheLimit: 0,
            recommendedWorkingSetBytes: 107 << 30, physicalMemory: 128 << 30, currentRSS: 0)
        XCTAssertEqual(settings.resolvedMemorySafetyPlan(bundleFacts: facts, host: known)
            .loadConfiguration.memoryLimit, .absolute(98 << 30))
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
