//
//  ProposalHeadStampTests.swift
//  MLXLMCommonFocusedTests
//
//  Pins the `vmlx_mtp_proposal_head.json` contract (2026-09-04): the pure
//  eligibility function over every SHIPPED head layout, tolerant load
//  (corrupt / unknown-version → nil, never a throw), atomic write
//  round-trip, and the authoritative-vs-re-derive source matching rule.
//

import Foundation
import XCTest

@testable import MLXLMCommon

final class ProposalHeadStampTests: XCTestCase {

    // MARK: - Eligibility: the pure function over the 7 shipped shapes

    private func layout(
        bits: Int, group: Int, mode: String = "affine", tied: Bool = false
    ) -> ProposalHeadSourceLayout {
        ProposalHeadSourceLayout(bits: bits, groupSize: group, mode: mode, tied: tied)
    }

    func testQ8G64AffineUntiedIsEligibleAtFourBits() {
        // Flash-Next-CRACK-6S / -JANG4M (and the local q8/g64 JANG_2L lineage).
        XCTAssertEqual(
            ProposalHeadVerdict.derive(from: layout(bits: 8, group: 64)),
            .eligible(proposalBits: 4))
    }

    func testLowBitHeadsAreIneligible_nativeHeadAlreadyLowBit() {
        // CRACK-JANG2L q6/g64; 27B 2D q2/g128, 4D q4/g128, 6D q6/g128.
        for (bits, group) in [(6, 64), (2, 128), (4, 128), (6, 128)] {
            XCTAssertEqual(
                ProposalHeadVerdict.derive(from: layout(bits: bits, group: group)),
                .ineligible(reason: "native_head_already_low_bit"),
                "q\(bits)/g\(group) must be ineligible: bits ≤ 6 rule")
        }
    }

    func testUnmeasuredLayoutNamesItsShape() {
        // 27B-MXFP8-CRACK: q8/g32 mxfp8.
        XCTAssertEqual(
            ProposalHeadVerdict.derive(from: layout(bits: 8, group: 32, mode: "mxfp8")),
            .ineligible(reason: "unmeasured_layout_q8_g32"))
        // q8/g128 affine: right bits, wrong group — unmeasured, not eligible.
        XCTAssertEqual(
            ProposalHeadVerdict.derive(from: layout(bits: 8, group: 128)),
            .ineligible(reason: "unmeasured_layout_q8_g128"))
    }

    func testTiedEmbeddingsAlwaysIneligible() {
        // Tied wins over every other property — even a q8/g64 tied head must
        // not get a divergent proposal copy.
        XCTAssertEqual(
            ProposalHeadVerdict.derive(from: layout(bits: 8, group: 64, tied: true)),
            .ineligible(reason: "tied_embeddings"))
        XCTAssertEqual(
            ProposalHeadVerdict.derive(from: layout(bits: 0, group: 0, mode: "none", tied: true)),
            .ineligible(reason: "tied_embeddings"))
    }

    // MARK: - IO: tolerant load, atomic write, round-trip

    private func makeBundleDir() throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("proposal-stamp-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: dir) }
        return dir
    }

    func testWriteThenLoadRoundTripsTheVerdict() throws {
        let dir = try makeBundleDir()
        let source = layout(bits: 8, group: 64)
        let stamp = ProposalHeadStamp(
            family: "qwen4_exp",
            source: source,
            verdict: .derive(from: source),
            basis: "unit-test")
        XCTAssertTrue(stamp.write(toBundleAt: dir))

        let loaded = ProposalHeadStamp.load(fromBundleAt: dir)
        XCTAssertEqual(loaded, stamp)
        XCTAssertEqual(loaded?.verdict, .eligible(proposalBits: 4))
        XCTAssertEqual(loaded?.source, source)

        // The written JSON must use the contract's snake_case keys, so the
        // Python engine and jang-tools read the same file.
        let raw = try String(
            contentsOf: dir.appendingPathComponent(ProposalHeadStamp.fileName),
            encoding: .utf8)
        XCTAssertTrue(raw.contains("\"proposal_bits\""))
        XCTAssertTrue(raw.contains("\"group_size\""))
        XCTAssertTrue(raw.contains("\"eligible\""))
    }

    func testIneligibleStampCarriesReasonNotBits() throws {
        let dir = try makeBundleDir()
        let source = layout(bits: 4, group: 128)
        let stamp = ProposalHeadStamp(
            family: "qwen3_5",
            source: source,
            verdict: .derive(from: source),
            basis: "unit-test")
        XCTAssertTrue(stamp.write(toBundleAt: dir))
        let loaded = ProposalHeadStamp.load(fromBundleAt: dir)
        XCTAssertEqual(loaded?.eligible, false)
        XCTAssertEqual(loaded?.reason, "native_head_already_low_bit")
        XCTAssertNil(loaded?.proposalBits)
    }

    func testMissingCorruptAndUnknownVersionAllReadAsAbsent() throws {
        let dir = try makeBundleDir()
        // Missing.
        XCTAssertNil(ProposalHeadStamp.load(fromBundleAt: dir))
        // Corrupt.
        let url = dir.appendingPathComponent(ProposalHeadStamp.fileName)
        try "not json {{{".write(to: url, atomically: true, encoding: .utf8)
        XCTAssertNil(ProposalHeadStamp.load(fromBundleAt: dir))
        // Unknown version: ignored so a future schema can supersede safely.
        let future: [String: Any] = [
            "version": 99, "family": "qwen4_exp",
            "source": ["bits": 8, "group_size": 64, "mode": "affine", "tied": false],
            "eligible": true, "proposal_bits": 4,
        ]
        let data = try JSONSerialization.data(withJSONObject: future)
        try data.write(to: url)
        XCTAssertNil(ProposalHeadStamp.load(fromBundleAt: dir))
    }

    func testEricStampShapeFromHFDecodes() throws {
        // The exact shape jang-tools ships (Flash-Next eligible stamp).
        let dir = try makeBundleDir()
        let json = """
            {
              "version": 1,
              "family": "qwen4_exp",
              "source": { "bits": 8, "group_size": 64, "mode": "affine", "tied": false },
              "eligible": true,
              "proposal_bits": 4,
              "basis": "settled 2026-09-03 A/B on Qwen3.8-Flash-Next-JANG_4S fixed-D3"
            }
            """
        try json.write(
            to: dir.appendingPathComponent(ProposalHeadStamp.fileName),
            atomically: true, encoding: .utf8)
        let loaded = ProposalHeadStamp.load(fromBundleAt: dir)
        XCTAssertEqual(loaded?.verdict, .eligible(proposalBits: 4))
        XCTAssertEqual(loaded?.family, "qwen4_exp")
        XCTAssertEqual(loaded?.source, layout(bits: 8, group: 64))
    }

    func testSourceMismatchMeansReDerive() throws {
        // The stamp is authoritative ONLY when its source matches the loaded
        // head. A stamped-eligible bundle that was requantized to q6 must be
        // re-derived (and the fresh derivation says ineligible).
        let dir = try makeBundleDir()
        let staleSource = layout(bits: 8, group: 64)
        ProposalHeadStamp(
            family: "qwen4_exp", source: staleSource,
            verdict: .derive(from: staleSource), basis: "stale"
        ).write(toBundleAt: dir)

        let actual = layout(bits: 6, group: 64)
        let stamp = ProposalHeadStamp.load(fromBundleAt: dir)
        XCTAssertNotNil(stamp)
        XCTAssertNotEqual(stamp?.source, actual, "mismatch must void the stamp")
        XCTAssertEqual(
            ProposalHeadVerdict.derive(from: actual),
            .ineligible(reason: "native_head_already_low_bit"))
    }
}

// MARK: - Bootstrap authority + self-healing (the misstamp lesson)

/// Minimal conforming double: reports a fixed "actual loaded head" layout
/// and records installs, standing in for a real model in `ensure`.
private final class HeadDouble: NativeMTPProposalHeadInstalling {
    let layout: ProposalHeadSourceLayout?
    private(set) var installedBits: [Int] = []
    init(layout: ProposalHeadSourceLayout?) { self.layout = layout }
    var nativeMTPProposalHeadFamily: String { "qwen4_exp" }
    var nativeMTPProposalHeadSourceLayout: ProposalHeadSourceLayout? { layout }
    func installNativeMTPProposalHead(bits: Int) { installedBits.append(bits) }
}

extension ProposalHeadStampTests {

    /// The 2L misstamp, end to end: a stamp derived from the CONFIG's tier
    /// default (q6 → "already low bit") sits in a bundle whose ACTUAL loaded
    /// head is q8/g64. The runtime must void the stamp on source mismatch,
    /// re-derive eligibility from the real head, install the proposal copy,
    /// and overwrite the file with measured truth — "the stamp is a cache of
    /// a pure function, never an authority over the weights."
    func testBootstrapSelfHealsAMisstampedBundle() throws {
        let dir = try makeBundleDir()
        let configLie = ProposalHeadSourceLayout(
            bits: 6, groupSize: 64, mode: "affine", tied: false)
        ProposalHeadStamp(
            family: "qwen4_exp", source: configLie,
            verdict: .derive(from: configLie), basis: "misstamp-from-config-default"
        ).write(toBundleAt: dir)

        let model = HeadDouble(
            layout: ProposalHeadSourceLayout(bits: 8, groupSize: 64, mode: "affine", tied: false))
        ProposalHeadBootstrap.ensure(model: model, modelDirectory: dir, isCalibratedBundle: true)

        XCTAssertEqual(model.installedBits, [4], "re-derived verdict must be used in-process")
        let healed = ProposalHeadStamp.load(fromBundleAt: dir)
        XCTAssertEqual(
            healed?.source, model.layout, "stamp must be overwritten with the MEASURED source")
        XCTAssertEqual(healed?.verdict, .eligible(proposalBits: 4))
    }

    /// The inverse authority rule: when the source MATCHES the loaded head,
    /// the stamp's verdict wins even against the runtime's own derivation,
    /// and the file is never rewritten — a jang-tools calibrated verdict
    /// (e.g. deliberately ineligible pending A/B) must not be clobbered.
    func testMatchingStampIsAuthoritativeAndNeverRewritten() throws {
        let dir = try makeBundleDir()
        let source = ProposalHeadSourceLayout(
            bits: 8, groupSize: 64, mode: "affine", tied: false)
        // Hand-build a stamp whose verdict CONTRADICTS the pure rule for
        // this layout: eligible layout, stamped ineligible.
        let url = dir.appendingPathComponent(ProposalHeadStamp.fileName)
        let held = """
            {
              "version": 1,
              "family": "qwen4_exp",
              "source": { "bits": 8, "group_size": 64, "mode": "affine", "tied": false },
              "eligible": false,
              "reason": "held_back_pending_ab",
              "basis": "converter-side hold"
            }
            """
        try held.write(to: url, atomically: true, encoding: .utf8)
        let before = try Data(contentsOf: url)

        let model = HeadDouble(layout: source)
        ProposalHeadBootstrap.ensure(model: model, modelDirectory: dir, isCalibratedBundle: true)

        XCTAssertTrue(model.installedBits.isEmpty, "authoritative ineligible stamp must win")
        let after = try Data(contentsOf: url)
        XCTAssertEqual(before, after, "matching stamp must never be rewritten")
    }

    /// Concurrent first loads race benignly: both derive identical content
    /// and atomic rename leaves a valid file either way.
    func testConcurrentDerivesLeaveAValidIdenticalStamp() async throws {
        let dir = try makeBundleDir()
        let layout = ProposalHeadSourceLayout(
            bits: 8, groupSize: 64, mode: "affine", tied: false)
        await withTaskGroup(of: Void.self) { group in
            for _ in 0 ..< 8 {
                group.addTask {
                    ProposalHeadBootstrap.ensure(
                        model: HeadDouble(layout: layout), modelDirectory: dir,
                        isCalibratedBundle: true)
                }
            }
        }
        let loaded = ProposalHeadStamp.load(fromBundleAt: dir)
        XCTAssertEqual(loaded?.verdict, .eligible(proposalBits: 4))
        XCTAssertEqual(loaded?.source, layout)
    }
}

extension ProposalHeadStampTests {

    /// Uncalibrated (non-JANG) bundles must never be stamped by the runtime:
    /// a plain mlx_lm benchmark quant can carry a q8/g64 head without having
    /// earned an eligible verdict (the eligibility rule's premise is JANG's
    /// AWQ+imatrix calibration). The speed-audit packs are left unstamped on
    /// purpose. An existing source-matching stamp is still honored — placing
    /// one is an explicit human action.
    func testUncalibratedBundleIsNeverStampedButExistingStampIsHonored() throws {
        let dir = try makeBundleDir()
        let layout = ProposalHeadSourceLayout(
            bits: 8, groupSize: 64, mode: "affine", tied: false)

        // No stamp + uncalibrated: no derivation, no write, no install.
        let model = HeadDouble(layout: layout)
        ProposalHeadBootstrap.ensure(
            model: model, modelDirectory: dir, isCalibratedBundle: false)
        XCTAssertNil(ProposalHeadStamp.load(fromBundleAt: dir))
        XCTAssertTrue(model.installedBits.isEmpty)

        // Human-placed matching stamp on the same uncalibrated bundle: honored.
        ProposalHeadStamp(
            family: "qwen4_exp", source: layout,
            verdict: .eligible(proposalBits: 4), basis: "human decision"
        ).write(toBundleAt: dir)
        let second = HeadDouble(layout: layout)
        ProposalHeadBootstrap.ensure(
            model: second, modelDirectory: dir, isCalibratedBundle: false)
        XCTAssertEqual(second.installedBits, [4])
    }
}
