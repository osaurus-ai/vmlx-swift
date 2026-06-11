// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// CacheCoordinatorConfig.resolveKVPolicy — unit coverage for the
// coordinator-owned KV-sizing contract landed 2026-04-21. Ferebee's
// Osaurus report (200K-char / 55K-token translation crash) motivated
// the contract: the host app delegates KV sizing to vmlx's coordinator
// by setting `defaultKVMode` and/or `defaultMaxKVSize`, and the
// coordinator fills in per-request gaps without overriding explicit
// caller values.

import Testing

@testable import MLXLMCommon

@Suite("CacheCoordinator KV policy")
struct CacheCoordinatorKVPolicyTests {

    @Test("coordinator defaults keep paged RAM cache off")
    func coordinatorDefaultsKeepPagedRAMCacheOff() {
        let config = CacheCoordinatorConfig()

        #expect(!config.usePagedCache)
        #expect(!config.enableDiskCache)
    }

    // MARK: - defaultKVMode

    @Test("explicit request kvMode wins over coordinator default")
    func explicitKVModeWins() {
        let config = CacheCoordinatorConfig(
            defaultKVMode: .turboQuant(keyBits: 3, valueBits: 3)
        )
        let (mode, _) = config.resolveKVPolicy(
            kvMode: .affine(bits: 4),
            maxKVSize: nil,
            promptTokenCount: 100
        )
        #expect(mode == .affine(bits: 4, groupSize: 64))
    }

    @Test("coordinator default kvMode fills nil .none")
    func coordinatorKVModeFillsGap() {
        let config = CacheCoordinatorConfig(
            defaultKVMode: .turboQuant(keyBits: 3, valueBits: 3)
        )
        let (mode, _) = config.resolveKVPolicy(
            kvMode: .none,
            maxKVSize: nil,
            promptTokenCount: 100
        )
        #expect(mode == .turboQuant(keyBits: 3, valueBits: 3))
    }

    @Test("both request and default .none stays .none")
    func bothNoneStaysNone() {
        let config = CacheCoordinatorConfig(defaultKVMode: .none)
        let (mode, _) = config.resolveKVPolicy(
            kvMode: .none,
            maxKVSize: nil,
            promptTokenCount: 100
        )
        #expect(mode == .none)
    }

    // MARK: - defaultMaxKVSize

    @Test("explicit request maxKVSize wins over coordinator default")
    func explicitMaxKVSizeWins() {
        let config = CacheCoordinatorConfig(
            defaultMaxKVSize: 4096,
            longPromptMultiplier: 2.0
        )
        let (_, max) = config.resolveKVPolicy(
            kvMode: .none,
            maxKVSize: 1024,
            promptTokenCount: 50_000
        )
        #expect(max == 1024)
    }

    @Test("default maxKVSize applied only to long prompts")
    func defaultMaxKVSizeLongPrompt() {
        let config = CacheCoordinatorConfig(
            defaultMaxKVSize: 4096,
            longPromptMultiplier: 2.0
        )
        // 8192-token prompt is NOT > 2 × 4096 = 8192 (needs strict >)
        let (_, maxAtThreshold) = config.resolveKVPolicy(
            kvMode: .none,
            maxKVSize: nil,
            promptTokenCount: 8192
        )
        #expect(maxAtThreshold == nil)

        // 8193-token prompt IS > 8192 → cap applied
        let (_, maxAbove) = config.resolveKVPolicy(
            kvMode: .none,
            maxKVSize: nil,
            promptTokenCount: 8193
        )
        #expect(maxAbove == 4096)

        // 55K-token prompt (ferebee's case) → cap applied
        let (_, max55k) = config.resolveKVPolicy(
            kvMode: .none,
            maxKVSize: nil,
            promptTokenCount: 55_000
        )
        #expect(max55k == 4096)
    }

    @Test("default maxKVSize skipped for short prompts")
    func defaultMaxKVSizeShortPrompt() {
        let config = CacheCoordinatorConfig(
            defaultMaxKVSize: 4096,
            longPromptMultiplier: 2.0
        )
        let (_, max) = config.resolveKVPolicy(
            kvMode: .none,
            maxKVSize: nil,
            promptTokenCount: 100
        )
        #expect(max == nil)
    }

    @Test("nil defaultMaxKVSize disables the cap entirely")
    func nilDefaultSkipsPolicy() {
        let config = CacheCoordinatorConfig(defaultMaxKVSize: nil)
        let (_, max) = config.resolveKVPolicy(
            kvMode: .none,
            maxKVSize: nil,
            promptTokenCount: 1_000_000
        )
        #expect(max == nil)
    }

    // MARK: - Combined policy

    @Test("both defaults coexist without interfering")
    func bothDefaultsApply() {
        let config = CacheCoordinatorConfig(
            defaultKVMode: .turboQuant(keyBits: 3, valueBits: 3),
            defaultMaxKVSize: 4096,
            longPromptMultiplier: 2.0
        )
        let (mode, max) = config.resolveKVPolicy(
            kvMode: .none,
            maxKVSize: nil,
            promptTokenCount: 55_000
        )
        #expect(mode == .turboQuant(keyBits: 3, valueBits: 3))
        #expect(max == 4096)
    }

    @Test("custom longPromptMultiplier shifts threshold")
    func customMultiplier() {
        let config = CacheCoordinatorConfig(
            defaultMaxKVSize: 2048,
            longPromptMultiplier: 1.0
        )
        // 2049 > 1 × 2048 → cap
        let (_, maxAbove) = config.resolveKVPolicy(
            kvMode: .none,
            maxKVSize: nil,
            promptTokenCount: 2049
        )
        #expect(maxAbove == 2048)

        // 2048 NOT > 2048 → no cap
        let (_, maxAt) = config.resolveKVPolicy(
            kvMode: .none,
            maxKVSize: nil,
            promptTokenCount: 2048
        )
        #expect(maxAt == nil)
    }

    // MARK: - Ferebee scenario

    @Test("ferebee 55K translation: tq + rotating both engage")
    func ferebeeScenario() {
        // Osaurus 0.17.0 would configure the coordinator this way once
        // it takes the contract seriously: TQ as default (~5× KV memory
        // savings) + 8K window for prompts > 16K tokens.
        let config = CacheCoordinatorConfig(
            defaultKVMode: .turboQuant(keyBits: 3, valueBits: 3),
            defaultMaxKVSize: 8192,
            longPromptMultiplier: 2.0
        )
        // Caller passes neither kvMode nor maxKVSize (UI removed).
        let (mode, max) = config.resolveKVPolicy(
            kvMode: .none,
            maxKVSize: nil,
            promptTokenCount: 55_000
        )
        #expect(mode == .turboQuant(keyBits: 3, valueBits: 3))
        #expect(max == 8192)
    }
}
