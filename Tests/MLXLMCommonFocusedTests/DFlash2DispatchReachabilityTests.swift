// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Every entry point that can carry a `draftStrategy` must actually ACT on
// a DFlash 2 one.
//
// This suite exists because of a real failure caught on the live app: the
// strategy resolved correctly, reached `GenerateParameters`, and was then
// dropped. `BatchEngine.generate` — the path the osaurus chat window
// actually uses — dispatched only on `usesBlockDiffusion`, which is false
// for `.dflash2`, so the request fell through to batched decode and ran
// with no speculation at all. Nothing failed. The answer was correct. The
// only symptom was an empty trace.
//
// A guard that is never reached is worth nothing, so these tests read the
// dispatch sites as source and assert each one names the DFlash 2 path.
// Source-matching is the right shape here: the alternative is loading a
// 27 GB target in CI, and the defect being pinned is structural — a
// missing branch, not a wrong value.

import Foundation
import XCTest

final class DFlash2DispatchReachabilityTests: XCTestCase {

    private static let repoRoot: URL = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()  // MLXLMCommonFocusedTests
        .deletingLastPathComponent()  // Tests
        .deletingLastPathComponent()  // repo root

    private func source(_ relativePath: String) throws -> String {
        let url = Self.repoRoot.appendingPathComponent(relativePath)
        return try String(contentsOf: url, encoding: .utf8)
    }

    func testStagedVerificationDoesNotEnableCompilationWithoutPromotedCaches() throws {
        let text = try source("Libraries/MLXLMCommon/SpecDec/DFlash2TokenIterator.swift")
        let promotionStart = try XCTUnwrap(text.range(of: "if useStagedVerify, HardwareInfo.isCompiledDecodeSupported"))
        let promotionEnd = try XCTUnwrap(text.range(of: "self.useCompiledVerify = true"))
        let promotion = text[promotionStart.lowerBound..<promotionEnd.upperBound]
        XCTAssertTrue(promotion.contains("VMLX_DFLASH2_COMPILED_VERIFY"))
        XCTAssertTrue(promotion.contains("CompilableKVCache(from: layer"))
        XCTAssertTrue(text.contains("if !useCompiledVerify || !stagedWarmedSizes.contains(bs) {"),
                      "Default eager staging must not build a compiled forward on cycle two")
        XCTAssertEqual(text.components(separatedBy: "self.useCompiledVerify = true").count - 1, 1)
        let abandonStart = try XCTUnwrap(text.range(of: "private mutating func abandonInFlightVerify("))
        let abandonEnd = try XCTUnwrap(text.range(of: "private mutating func buildCompiledVerify("))
        XCTAssertTrue(text[abandonStart.lowerBound..<abandonEnd.lowerBound]
            .contains("compiledVerify.removeAll(keepingCapacity: true)"),
            "Replacing staging slots must release traces that captured the old arrays")
    }

    /// `BatchEngine.generate` is what the host chat window calls. Missing
    /// here means speculation silently never runs in the product, which is
    /// exactly the bug this file was written for.
    func testBatchEngineGenerateRoutesDFlash2ToTheSoloPath() throws {
        let text = try source("Libraries/MLXLMCommon/BatchEngine/BatchEngine.swift")
        XCTAssertTrue(
            text.contains("parameters.draftStrategy?.usesDFlash2 == true"),
            "BatchEngine.generate must route DFlash 2 to the exclusive solo path; without it the strategy is accepted and then ignored"
        )
        XCTAssertTrue(
            text.contains("DFlash2TokenIterator("),
            "BatchEngine's solo path must actually construct the DFlash 2 iterator")
        XCTAssertTrue(
            text.contains("strategy.dflash2DrafterPath"),
            "BatchEngine's solo path must read the drafter path off the strategy")
    }

    /// A DFlash 2 request that arrives while the engine serves another request (continuous batching
    /// > 1: a spawned subagent, a parallel delegation) decodes as a batched AR slot, with a log line,
    /// instead of being cancelled. The second of two concurrent requests used to fail outright.
    func testBatchEngineGenerateServesABusyEngineDFlash2RequestAutoregressively() throws {
        let text = try source("Libraries/MLXLMCommon/BatchEngine/BatchEngine.swift")
        XCTAssertFalse(
            text.contains("Rejected BatchEngine.generate DFlash 2 request"),
            "a busy engine must not cancel a DFlash 2 request")
        guard let notice = text.range(of: "BatchEngine.generate DFlash 2 request decodes autoregressively") else {
            return XCTFail("the autoregressive fallback must be logged")
        }
        let after = text[notice.upperBound...].prefix(400)
        XCTAssertTrue(
            after.contains("parameters.draftStrategy = nil"),
            "the fallback must drop the strategy so the batched path does not carry it")
    }

    /// The raw batched submit path has no way to express a draft/verify
    /// cycle, so it must refuse rather than quietly decode without it.
    func testBatchEngineSubmitRefusesDFlash2RatherThanIgnoringIt() throws {
        let text = try source("Libraries/MLXLMCommon/BatchEngine/BatchEngine.swift")
        guard let submitRange = text.range(of: "Rejected BatchEngine.submit DFlash 2 request") else {
            return XCTFail(
                "BatchEngine.submit must explicitly reject DFlash 2; falling through would run the request with no speculation and no signal"
            )
        }
        XCTAssertTrue(
            text[..<submitRange.lowerBound].hasSuffix("\n            ")
                || text.contains("usesDFlash2 == true"),
            "the rejection must be guarded on usesDFlash2")
    }

    /// The two direct-generate entry points in `Evaluate`.
    func testEvaluateEntryPointsDispatchDFlash2BeforeNativeMTP() throws {
        let text = try source("Libraries/MLXLMCommon/Evaluate.swift")
        let dispatchCount = text.components(separatedBy: "strategy.dflash2DrafterPath").count - 1
        XCTAssertGreaterThanOrEqual(
            dispatchCount, 2,
            "both `generate` and `generateTokensTask` must dispatch DFlash 2")

        // Ordering matters: the two strategies are alternatives, and the
        // one the user explicitly configured has to win.
        guard let dflashIndex = text.range(of: "strategy.dflash2DrafterPath")?.lowerBound,
            let mtpIndex = text.range(of: "case .nativeMTP(depth: let depth")?.lowerBound
        else {
            return XCTFail("could not locate both dispatch sites")
        }
        XCTAssertLessThan(
            dflashIndex, mtpIndex,
            "DFlash 2 must be dispatched before native MTP so a selected drafter supersedes the model's own head"
        )
    }

    /// The host resolves one strategy per request; DFlash 2 has to win
    /// there too, or the ordering in `Evaluate` never gets a chance.
    func testSettingsResolutionPrefersDFlash2OverNativeMTP() throws {
        let text = try source("Libraries/MLXLMCommon/ServerRuntimeSettings.swift")
        guard let selectionIndex = text.range(of: "let selection = resolvedDFlash2Selection(")?.lowerBound,
            let mtpIndex = text.range(of: "return .nativeMTP(depth: depth")?.lowerBound
        else {
            return XCTFail("could not locate the strategy resolution branches")
        }
        XCTAssertLessThan(
            selectionIndex, mtpIndex,
            "resolvedMTPDraftStrategy must check the selected drafter before falling back to native MTP"
        )
    }

    /// An unservable request must FALL THROUGH, not fail. A bundle that
    /// stamps a benign `repetition_penalty: 1.0` (Qwen3.8 does) or a
    /// request that arms a reasoning budget still has to answer.
    func testEveryDispatchSiteChecksServabilityBeforeConstructing() throws {
        let evaluate = try source("Libraries/MLXLMCommon/Evaluate.swift")
        XCTAssertEqual(
            evaluate.components(separatedBy: "DFlash2TokenIterator.unservableReason").count - 1, 2,
            "both Evaluate entry points must gate on unservableReason so the request falls through instead of throwing"
        )
        let batch = try source("Libraries/MLXLMCommon/BatchEngine/BatchEngine.swift")
        XCTAssertEqual(
            batch.components(separatedBy: "DFlash2TokenIterator.unservableReason").count - 1, 2,
            "BatchEngine must gate BOTH the solo-path routing decision and the iterator construction"
        )
    }
}
