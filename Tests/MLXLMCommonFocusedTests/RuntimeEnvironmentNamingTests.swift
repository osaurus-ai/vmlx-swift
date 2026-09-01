// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// The rename is only safe if the legacy spelling keeps working, and only finished if no read
// site is left on the old name alone. Neither is visible from a build, so both are asserted.

import Foundation
import MLXLMCommon
import Testing

@Suite("Runtime environment naming")
struct RuntimeEnvironmentNamingTests {

    @Test("the current spelling is read")
    func currentSpellingRead() {
        #expect(RuntimeEnvironment.value("VMLX_THING", in: ["VMLX_THING": "a"]) == "a")
    }

    /// The whole point of the migration: someone's existing script must not break.
    @Test("the legacy spelling is still honoured")
    func legacySpellingHonoured() {
        #expect(RuntimeEnvironment.value("VMLX_THING", in: ["VMLINUX_THING": "b"]) == "b")
        #expect(RuntimeEnvironment.flag("VMLX_THING", in: ["VMLINUX_THING": "1"]))
    }

    @Test("the current spelling wins when both are set")
    func currentWins() {
        let both = ["VMLX_THING": "new", "VMLINUX_THING": "old"]
        #expect(RuntimeEnvironment.value("VMLX_THING", in: both) == "new")
    }

    /// A name that is not ours gets no derived fallback — otherwise asking for `PATH` would
    /// quietly consult `VMLINUX_PATH`.
    @Test("an unprefixed name gets no legacy fallback")
    func unprefixedHasNoFallback() {
        #expect(RuntimeEnvironment.legacyName(of: "PATH") == nil)
        #expect(RuntimeEnvironment.value("PATH", in: ["VMLINUX_PATH": "x"]) == nil)
        #expect(RuntimeEnvironment.value("PATH", in: ["PATH": "x"]) == "x")
    }

    @Test("flag accepts the spellings the call sites already accepted")
    func flagSpellings() {
        for yes in ["1", "true", "yes", "on", "TRUE", " on "] {
            #expect(RuntimeEnvironment.flag("VMLX_F", in: ["VMLX_F": yes]), "\(yes)")
        }
        for no in ["0", "false", "no", "off", ""] {
            #expect(!RuntimeEnvironment.flag("VMLX_F", in: ["VMLX_F": no]), "\(no)")
        }
        #expect(RuntimeEnvironment.flag("VMLX_F", default: true, in: [:]))
    }

    /// The public constants must NAME the current variable. A consumer reads these to build a
    /// settings UI or documentation, so a legacy value here publishes the old name outward —
    /// which is exactly what this migration is for.
    @Test("public environment-name constants carry the current spelling")
    func publicConstantsAreCurrent() {
        let current: [String] = [
            AccelerationMode.environmentVariable,
            DeepseekV4ReasoningPolicy.rawMaxEnvironmentKey,
            DeepseekV4ReasoningPolicy.forceDirectRailEnvironmentKey,
            RuntimeMoETopKOverride.environmentVariable,
        ]
        for name in current {
            #expect(name.hasPrefix(RuntimeEnvironment.prefix), "\(name) is not a current name")
        }
        // And each keeps its legacy sibling, so the old spelling stays reachable.
        let legacy: [String] = [
            AccelerationMode.legacyEnvironmentVariable,
            DeepseekV4ReasoningPolicy.legacyRawMaxEnvironmentKey,
            DeepseekV4ReasoningPolicy.legacyForceDirectRailEnvironmentKey,
            RuntimeMoETopKOverride.legacyEnvironmentVariable,
        ]
        for (new, old) in zip(current, legacy) {
            #expect(RuntimeEnvironment.legacyName(of: new) == old, "\(new) -> \(old)")
        }
    }

    /// Reading the accelerator through the legacy spelling must still resolve. The read went
    /// through `environmentVariable` alone, so flipping that constant without widening the read
    /// would have dropped every existing user — and nothing else would have said so.
    @Test("the accelerator still resolves from the legacy variable")
    func acceleratorLegacyStillResolves() {
        #expect(
            AccelerationRuntime.requestedMode(
                environment: [AccelerationMode.legacyEnvironmentVariable: "auto"]) == .auto)
        #expect(
            AccelerationRuntime.requestedMode(
                environment: [AccelerationMode.environmentVariable: "auto"]) == .auto)
        #expect(AccelerationRuntime.requestedMode(environment: [:]) == .metal)
    }
}
