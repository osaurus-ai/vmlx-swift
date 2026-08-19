// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Stage-by-stage comparison against the reference implementation.
//
// The end-to-end parity test says WHETHER the port agrees; this one says
// WHERE it stops agreeing. It is the diagnostic that localised the
// remaining drift to accumulation rather than to a wrong operation: the
// DFlash 2 backbone runs its residual stream up to ~3e6 in bf16 before
// the final RMSNorm pulls it back to ~19, so the last layers are
// genuinely sensitive to summation order, while every individual stage
// still has to match tightly.
//
// Golden file: `scratchpad/probe.py`.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

final class DFlash2StageProbeTests: XCTestCase {

    private static var drafterURL: URL {
        FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("models/Qwen3.8-27B-DFlash2")
    }

    func testStagesMatchReference() throws {
        guard DFlash2Loader.looksLikeDFlash2Drafter(at: Self.drafterURL) else {
            throw XCTSkip("No DFlash 2 drafter at \(Self.drafterURL.path)")
        }
        guard let path = ProcessInfo.processInfo.environment["DFLASH2_PROBE_GOLDEN"],
            FileManager.default.fileExists(atPath: path)
        else {
            throw XCTSkip("Set DFLASH2_PROBE_GOLDEN to probe.safetensors from scratchpad/probe.py")
        }
        let (golden, _) = try loadArraysAndMetadata(url: URL(fileURLWithPath: path))
        let drafter = try DFlash2Loader.load(from: Self.drafterURL)

        let block = golden["block"]!.asType(.int32)
        let targetHidden = golden["target_hidden"]!.asType(.bfloat16)
        let stubA = MLXArray.zeros([1])  // unused; embed comes from the golden file

        _ = stubA
        // The embedded block is taken from the golden file so this test
        // isolates the backbone from the stub-target arithmetic.
        let embedded = golden["embed"]!.asType(.bfloat16)

        let report = drafter.probeStages(
            embedded: embedded, targetHidden: targetHidden, cache: drafter.makeCache())

        func compare(_ key: String, _ got: MLXArray, tolerance: Float) {
            guard let want = golden[key] else {
                XCTFail("golden file has no \(key)")
                return
            }
            XCTAssertEqual(got.shape, want.shape, "\(key) shape")
            let diff = abs(got.asType(.float32) - want).max().item(Float.self)
            let scale = Swift.max(abs(want).max().item(Float.self), 1e-6)
            let relative = diff / scale
            print(
                String(
                    format: "  %-16s absmax=%.4g  maxdiff=%.4g  rel=%.4f%%", (key as NSString).utf8String!,
                    scale, diff, relative * 100))
            XCTAssertLessThan(relative, tolerance, "\(key): relative drift \(relative)")
        }

        // Single stages carry at most one bf16 rounding each.
        compare("hctx", report.context, tolerance: 0.01)
        compare("l0_ln", report.layer0InputNorm, tolerance: 0.01)
        compare("l0_convin", report.layer0ConvInput, tolerance: 0.01)
        compare("l0_kern", report.layer0Kernel, tolerance: 0.01)
        compare("l0_attn", report.layer0Attention, tolerance: 0.02)
        compare("l0_attnconv", report.layer0AttentionConv, tolerance: 0.02)
        compare("l0_post_attn", report.layer0PostAttention, tolerance: 0.02)
        compare("l0_mlpin", report.layer0MLPInput, tolerance: 0.02)
        // Whole layers accumulate; the stream is ~5 orders of magnitude
        // above its post-norm scale by the last one.
        for (index, hidden) in report.perLayer.enumerated() {
            compare("l\(index)", hidden, tolerance: 0.10)
        }
        compare("final", report.final, tolerance: 0.10)
    }
}
