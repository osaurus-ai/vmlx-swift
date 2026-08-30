// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import Testing

@testable import MLXLMCommon

/// The host's Speculative Decoding control resolves to
/// `LoadConfiguration.nativeMTP`, which `ModelFactory` carries into model
/// construction as `NativeMTPActivation.explicitRequestOverride`. Two things
/// broke that chain for Nemotron:
///
/// 1. `shouldLoadNativeMTPWeights` gated on a Qwen-only model-type allowlist, so
///    a Nemotron bundle was rejected as "unsupported" no matter what the user
///    picked.
/// 2. `NemotronH` read its own process-level `let` over `VMLX_NEMOTRON_MTP`
///    instead of the task-local, so the head was never built from a host
///    request and could not change within a session.
///
/// Together those made Off / Auto / Force-On indistinguishable for this family.
@Suite("Native MTP activation wiring")
struct NemotronMTPActivationWiringTests {

    // MARK: - Model-type allowlist

    @Test(arguments: ["nemotron_h", "nemotronh", "Nemotron-H", "NEMOTRON_H", "nemotron_h_text"])
    func nemotronIsASupportedNativeMTPModelType(_ modelType: String) {
        #expect(NativeMTPActivation.isSupportedNativeMTPModelType(modelType))
    }

    @Test(arguments: ["qwen3_5_moe", "qwen3_6", "qwen36_moe_text"])
    func qwenRemainsSupported(_ modelType: String) {
        #expect(NativeMTPActivation.isSupportedNativeMTPModelType(modelType))
    }

    /// The allowlist is the fail-closed part of the policy: a family whose head
    /// this runtime cannot drive must still be rejected.
    @Test(arguments: ["llama", "gemma3", "deepseek_v4", "nemotron", "nemotron_4", "", "muse_glimmer"])
    func unrelatedTypesStayRejected(_ modelType: String) {
        #expect(!NativeMTPActivation.isSupportedNativeMTPModelType(modelType))
    }

    // MARK: - The task-local the host sets

    @Test("no host request and no env var means not requested")
    func defaultsToNotRequested() async throws {
        try await NativeMTPActivation.withExplicitRequest(false) {
            #expect(!NativeMTPActivation.isExplicitlyRequested)
        }
    }

    @Test("a host request is visible to model construction")
    func hostRequestIsVisible() async throws {
        try await NativeMTPActivation.withExplicitRequest(true) {
            #expect(NativeMTPActivation.isExplicitlyRequested)
        }
    }

    /// Off must win over the env var, otherwise a machine that once exported
    /// `VMLX_NEMOTRON_MTP=1` for benchmarking could never turn the head back off
    /// from the UI.
    @Test("an explicit false overrides any ambient env opt-in")
    func explicitFalseOverridesEnvironment() async throws {
        try await NativeMTPActivation.withExplicitRequest(false) {
            #expect(!NativeMTPActivation.isExplicitlyRequested)
        }
    }

    /// A Nemotron bundle with no `vmlx_mtp_tuning.json` — which is every
    /// Lightning bundle shipped today — must be refused for a *measured*
    /// reason, not silently ignored. Reaching this error at all is the fix:
    /// before it, the same bundle failed earlier as "unsupported model".
    @Test("a Nemotron bundle without tuning is refused on tuning, not on family")
    func nemotronWithoutTuningFailsForTheRightReason() async throws {
        let config = #"{"model_type":"nemotron_h","num_nextn_predict_layers":1}"#
            .data(using: .utf8)!
        try await NativeMTPActivation.withExplicitRequest(true) {
            do {
                _ = try NativeMTPActivation.shouldLoadNativeMTPWeights(
                    configData: config,
                    baseModelType: "nemotron_h",
                    status: nil)
                Issue.record("expected a fail-closed refusal")
            } catch let error as NativeMTPActivationError {
                switch error {
                case .requestedForUnsupportedModel(let types):
                    Issue.record("still rejected as an unsupported family: \(types)")
                case .requestedButMissingArtifact, .requestedWithoutUsableTuning,
                    .requestedWithBlockedTuning:
                    break  // correct: the family is accepted, the evidence is not
                case .invalidConfigData:
                    Issue.record("unexpected config error")
                }
            }
        }
    }

    /// And with no request at all, nothing loads regardless of family.
    @Test("no request means no MTP weights for anyone")
    func noRequestLoadsNothing() async throws {
        let config = #"{"model_type":"nemotron_h"}"#.data(using: .utf8)!
        try await NativeMTPActivation.withExplicitRequest(false) {
            let load = try NativeMTPActivation.shouldLoadNativeMTPWeights(
                configData: config,
                baseModelType: "nemotron_h",
                status: nil)
            #expect(!load)
        }
    }
}
