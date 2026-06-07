// Pin two related ZAYA contracts via source coverage:
//
// 1. Thinking-toggle declaration. ZAYA capability stamps are trusted.
//    Runtime must not rewrite stale ZAYA bundles or apply family-based
//    default-template overrides. Fix ZAYA bundle metadata at the source;
//    keep automatic thinking-off defaults capability-driven for models
//    stamped `supports_thinking=false` (for example Ling/Bailing).
//
// 2. ZayaCCACache async re-derivation parity with Mamba/Arrays SSM.
//    `ZayaCCACache.conv_state` + `prev_hs` are path-dependent (per Zyphra
//    runtime contract) — same multi-turn contamination problem as Mamba
//    SSM state. Pinned:
//      - `ModelContainer.assignDefaultCacheCoordinator` flips `isHybrid`
//        for Zaya (new detection clause).
//      - `BatchEngine.finishSlot`'s post-prefill `hasSSM` snapshot path
//        includes ZayaCCACache.
//      - `reDeriveSSMStates` / `reDeriveSSMStatesAtBoundaries` recognize
//        ZayaCCACache as a path-dependent layer.
//      - `BatchEngine.admit`'s coordinator-flip gate accepts
//        `family == .zayaCCA`.
//
// Source-coverage style: `LLMUserInputProcessor` is private to MLXLLM;
// `BatchEngine.admit` and `finishSlot` are internal. Pinning the source
// patterns textually catches regressions without needing public API
// exposure or a Metal runner.

import Foundation
import Testing

@Suite("ZAYA: thinking declaration + CCA-cache async re-derivation source coverage")
struct ZayaThinkingAndRederiveContractTests {

    private static func source(_ relativePath: String) throws -> String {
        let repo = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        return try String(contentsOf: repo.appendingPathComponent(relativePath), encoding: .utf8)
    }

    // MARK: - Thinking-toggle declaration source coverage

    @Test("LLMModelFactory.defaultContext does not force ZAYA-family thinking defaults")
    func factoryDoesNotForceZayaFamilyThinkingDefaults() throws {
        let source = try Self.source("Libraries/MLXLLM/LLMModelFactory.swift")

        #expect(!source.contains(#"type.hasPrefix("zaya")"#))
        #expect(!source.contains(#"type.hasPrefix("zyphra")"#))
        #expect(!source.contains(#"capabilities?.supportsThinking == true"#))
        #expect(!source.contains(#"capabilities?.thinkInTemplate == false"#))
        #expect(
            source.contains(#"if capabilities?.supportsThinking == false {"#),
            "LLMModelFactory.defaultContext must keep the supports_thinking=false closing branch for non-thinking models.")
    }

    @Test("LLMModelFactory applies ZAYA chat-template substitution before tokenizer load")
    func factoryAppliesZayaChatTemplateSubstitution() throws {
        let source = try Self.source("Libraries/MLXLLM/LLMModelFactory.swift")

        #expect(source.contains("let templateResolvedDir = JangLoader.resolveChatTemplateSidecarSubstitution("))
        #expect(source.contains("for: jangResolvedDir)"))
        #expect(source.contains("for: templateResolvedDir)"))
    }

    // MARK: - ZayaCCACache rederive integration source coverage

    /// `ModelContainer.enableCachingAsync` must flip `isHybrid=true` for
    /// ZAYA through the shared topology snapshot so the companion cache fires.
    @Test("ModelContainer hybrid auto-detection includes ZayaCCACache")
    func modelContainerDetectsZaya() throws {
        let source = try Self.source("Libraries/MLXLMCommon/ModelContainer.swift")
        #expect(
            source.contains("let topology = await cacheTopologySnapshot()"),
            "ModelContainer.enableCachingAsync must derive cache policy from the live cache topology.")
        #expect(
            source.contains("let isHybrid = topology.requiresSSMCompanionState"),
            "ModelContainer.enableCachingAsync must include ZayaCCACache through ModelCacheTopologySnapshot.requiresSSMCompanionState.")
    }

    /// `BatchEngine.finishSlot`'s post-prefill SSM snapshot must include
    /// ZayaCCACache so the conv_state+prev_hs reach SSMStateCache for
    /// cross-turn restore.
    @Test("BatchEngine post-prefill hasSSM snapshot includes ZayaCCACache")
    func batchEngineFinishSlotSnapshotsZaya() throws {
        let source = try Self.source("Libraries/MLXLMCommon/BatchEngine/BatchEngine.swift")
        #expect(
            source.contains(
                "$0 is MambaCache || $0 is ArraysCache || $0 is ZayaCCACache"),
            "BatchEngine.finishSlot post-prefill hasSSM check must include ZayaCCACache.")
    }

    /// `reDeriveSSMStates` and `reDeriveSSMStatesAtBoundaries` must
    /// recognize ZayaCCA layers as path-dependent.
    @Test("reDeriveSSMStates path-dependent detection includes ZayaCCA")
    func reDeriveDetectsZaya() throws {
        let source = try Self.source("Libraries/MLXLMCommon/Cache/SSMReDerive.swift")
        // Both detection sites must include ZayaCCA.
        let needle =
            #"desc.contains("Mamba") || desc.contains("Arrays") || desc.contains("ZayaCCA")"#
        let hits = source.components(separatedBy: needle).count - 1
        #expect(
            hits >= 2,
            "Both reDeriveSSMStates and reDeriveSSMStatesAtBoundaries must check for ZayaCCA (got \(hits) sites).")
    }

    /// `BatchEngine.admit`'s coordinator-flip gate must accept
    /// `family == .zayaCCA`.
    @Test("BatchEngine.admit coordinator-flip gate accepts .zayaCCA family")
    func admitFlipGateAcceptsZayaCCA() throws {
        let source = try Self.source("Libraries/MLXLMCommon/BatchEngine/BatchEngine.swift")
        #expect(
            source.contains("family == .heterogeneous || family == .mamba || family == .zayaCCA"),
            "BatchEngine admit coordinator-flip gate must include .zayaCCA — without it ZAYA models never flip isHybrid in the live-flip path.")
    }
}
