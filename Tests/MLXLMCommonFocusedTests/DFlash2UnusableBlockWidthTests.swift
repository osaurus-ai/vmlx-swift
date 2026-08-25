// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// A drafted block needs an anchor position plus at least one speculative
// position, so widths below 2 cannot draft. `DFlash2TokenIterator.init`
// has always thrown `blockSizeTooSmall` for them — but throwing is the
// wrong channel. BatchEngine's solo-fast-path catch turns ANY throw into a
// zero-token stream stamped `.cancelled`, so a user who pins Draft Tokens
// Per Step to 1 gets "I wasn't able to generate a response to that. Please
// try rephrasing your request." on every turn, and the real cause goes only
// to os_log — invisible from an unsigned build.
//
// `unservableReason` is the channel that already exists for exactly this:
// a non-nil result makes the caller SKIP the drafter, and the request
// decodes normally. The `maxTokens <= 1` rule is the same shape, mirroring
// `maxTokensTooSmall`. This suite pins that the width rule joins it.

import Foundation
import Testing

@testable import MLXLMCommon

@Suite("DFlash 2 unusable block width")
struct DFlash2UnusableBlockWidthTests {

    private func parameters(blockSize: Int?) -> GenerateParameters {
        var p = GenerateParameters(temperature: 0)
        p.draftStrategy = .dflash2(
            drafterPath: URL(fileURLWithPath: "/tmp/does-not-need-to-exist"),
            blockSize: blockSize)
        return p
    }

    /// The bug as the user met it: width 1 reached `init` and threw.
    @Test func widthBelowTheMinimumIsReportedAsUnservable() throws {
        let reason = try #require(
            DFlash2TokenIterator.unservableReason(parameters(blockSize: 1)),
            "width 1 must be reported here, not thrown from init")
        #expect(reason.contains("below the minimum"))
    }

    @Test func zeroAndNegativeWidthsAreAlsoUnservable() {
        #expect(DFlash2TokenIterator.unservableReason(parameters(blockSize: 0)) != nil)
        #expect(DFlash2TokenIterator.unservableReason(parameters(blockSize: -4)) != nil)
    }

    /// The minimum itself must remain servable — an off-by-one here would
    /// silently disable the narrowest width the runtime supports.
    @Test func theMinimumWidthItselfStaysServable() {
        #expect(
            DFlash2TokenIterator.unservableReason(
                parameters(blockSize: DFlash2TokenIterator.minimumBlockSize)) == nil)
    }

    /// `nil` means "use the checkpoint's trained width", which is resolved
    /// later from the drafter config. Reporting it unservable here would
    /// disable DFlash 2 for every default request.
    @Test func anUnsetWidthIsNotUnservable() {
        #expect(DFlash2TokenIterator.unservableReason(parameters(blockSize: nil)) == nil)
        #expect(DFlash2TokenIterator.unservableReason(parameters(blockSize: 8)) == nil)
    }

    /// The guard in `init` and the rule here must read the same constant, or
    /// a future edit to one silently reopens the throw path the other closed.
    @Test func theGuardAndTheRuleShareOneConstant() throws {
        let source = try String(
            contentsOf: URL(fileURLWithPath: #filePath)
                .deletingLastPathComponent()
                .deletingLastPathComponent()
                .deletingLastPathComponent()
                .appendingPathComponent(
                    "Libraries/MLXLMCommon/SpecDec/DFlash2TokenIterator.swift"),
            encoding: .utf8)
        #expect(
            source.contains("effectiveBlockSize >= Self.minimumBlockSize"),
            "init's guard must use the shared constant, not a bare literal")
        #expect(source.contains("requested < minimumBlockSize"))
    }
}
