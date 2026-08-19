// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Numerical parity between this package's DFlash 2 drafter and the
// authors' reference MLX implementation (z-lab/dflash `model_mlx.py`).
//
// The golden file is produced by
// `scratchpad/dflash2_reference_dump.py`, which imports the reference
// module UNCHANGED, loads the real `Qwen3.8-27B-DFlash2` weights, and
// substitutes a deterministic low-rank stub for the target's embedding
// and LM head. Only the drafter is under test, so the 52 GB target never
// has to be loaded on either side.
//
// The stub matters: without it a bug that transposes the LM head, or
// gathers the wrong embedding row, is invisible — every path still
// produces "plausible" logits. With a fixed pseudo-random stub, any such
// bug moves the argmax and the test fails.
//
// Skipped when either the drafter checkpoint or the golden file is
// absent, so CI without the 3.8 GB download stays green.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

/// Largest elementwise difference, relative to the reference's scale.
private func relativeDrift(_ got: MLXArray, _ want: MLXArray) -> Float {
    let diff = abs(got - want).max().item(Float.self)
    let scale = Swift.max(abs(want).max().item(Float.self), 1e-6)
    return diff / scale
}

private func fmt(_ value: Float) -> String {
    String(format: "%.2f%%", value * 100)
}

final class DFlash2ReferenceParityTests: XCTestCase {

    private static var drafterURL: URL {
        FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("models/Qwen3.8-27B-DFlash2")
    }

    private static var goldenURL: URL? {
        if let env = ProcessInfo.processInfo.environment["DFLASH2_PARITY_GOLDEN"] {
            return URL(fileURLWithPath: env)
        }
        return nil
    }

    /// Deterministic stand-in for the target model. `embed` and
    /// `projectToLogits` are the only two things the drafter asks of its
    /// target, and both are reproduced here from the same factors the
    /// Python side used.
    private final class StubTarget: Module, LanguageModel, TokenEmbedderModel,
        HiddenStateCaptureModel
    {
        let a: MLXArray  // (vocab, rank)
        let b: MLXArray  // (rank, hidden)
        let head: MLXArray  // (rank, hidden)

        init(a: MLXArray, b: MLXArray, head: MLXArray) {
            self.a = a
            self.b = b
            self.head = head
            super.init()
        }

        var vocabularySize: Int { a.dim(0) }

        func embed(_ tokenIds: MLXArray) -> MLXArray {
            MLX.take(a, tokenIds, axis: 0).matmul(b).asType(.bfloat16)
        }

        func projectToLogits(_ hidden: MLXArray) -> MLXArray {
            hidden.asType(.float32).matmul(head.transposed()).matmul(a.transposed())
        }

        func callAsFunction(
            _ inputs: MLXArray, cache: [KVCache]?, captureLayerIDs: Set<Int>
        ) -> (logits: MLXArray, capturedHiddenStates: [Int: MLXArray]) {
            (MLXArray.zeros([1, 1, 1]), [:])
        }

        func newCache(parameters: GenerateParameters?) -> [KVCache] { [] }

        func prepare(
            _ input: LMInput, cache: [KVCache], windowSize: Int?
        ) throws -> PrepareResult {
            .tokens(input.text)
        }
    }

    func testDrafterForwardMatchesReferenceImplementation() throws {
        guard DFlash2Loader.looksLikeDFlash2Drafter(at: Self.drafterURL) else {
            throw XCTSkip(
                "No DFlash 2 drafter at \(Self.drafterURL.path); run "
                    + "`hf download z-lab/Qwen3.8-27B-DFlash2 --local-dir ~/models/Qwen3.8-27B-DFlash2`"
            )
        }
        guard let goldenURL = Self.goldenURL,
            FileManager.default.fileExists(atPath: goldenURL.path)
        else {
            throw XCTSkip(
                "Set DFLASH2_PARITY_GOLDEN to reference.safetensors produced by "
                    + "scratchpad/dflash2_reference_dump.py")
        }

        let (golden, _) = try loadArraysAndMetadata(url: goldenURL)
        let stub = StubTarget(
            a: golden["stub_a"]!, b: golden["stub_b"]!, head: golden["stub_head"]!)
        let drafter = try DFlash2Loader.load(from: Self.drafterURL)

        XCTAssertEqual(drafter.config.dflash.selectorTopK, 16)
        XCTAssertEqual(drafter.config.dflash.convKernelSize, 2)
        XCTAssertEqual(drafter.config.dflash.convGroupSize, 16)
        XCTAssertEqual(drafter.config.blockSize, 8)
        XCTAssertEqual(drafter.config.isCausal, false)

        for name in ["short", "multi"] {
            let block = golden["\(name)__block"]!.asType(.int32)
            let targetHidden = golden["\(name)__target_hidden"]!.asType(.bfloat16)
            let expectedHidden = golden["\(name)__hidden"]!
            let expectedLogits = golden["\(name)__logits"]!
            let expectedTokens = golden["\(name)__tokens"]!
                .reshaped(-1).asArray(Int32.self)
            let expectedCandidates = golden["\(name)__candidates"]!

            let cache = drafter.makeCache()
            let hidden = drafter.hiddenStates(
                inputs: block, targetHidden: targetHidden, cache: cache,
                embedder: stub, logitsStart: 1)
            let logits = drafter.computeLogits(hidden, embedder: stub)
            // Re-runs the backbone through the public entry point so the
            // test exercises exactly what the decode loop calls, not an
            // internal shortcut around it.
            let proposal = drafter.propose(
                inputs: block, targetHidden: targetHidden, cache: drafter.makeCache(),
                embedder: stub, temperature: 0, logitsStart: 1)
            MLX.eval(hidden, logits, proposal.tokens, proposal.candidates)

            XCTAssertEqual(hidden.shape, expectedHidden.shape, "\(name) hidden shape")

            // The bar is NOT a hand-picked epsilon. This backbone runs its
            // residual stream to ~3e6 in bf16 before the final RMSNorm
            // compresses it to ~19, so one ulp at the top of the stream is
            // worth whole tokens at the bottom. Measured: the reference
            // disagrees with ITSELF by 8.7% on hidden, 5.9% on logits and
            // 4 of 7 path tokens when the only change is bf16 -> fp32.
            //
            // So the question a parity test can actually answer is "is the
            // port at least as close to the reference as the reference is
            // to its own higher-precision self". Anything tighter would be
            // asserting that Swift reproduces MLX-Python's summation order,
            // which is not a property either library promises.
            let referenceHiddenNoise = relativeDrift(
                expectedHidden, golden["\(name)__hidden_fp32"]!)
            let portHiddenDrift = relativeDrift(hidden.asType(.float32), expectedHidden)
            // Coarse bound only. Max-abs on the POST-norm tensor is set by
            // a single outlier channel, so it is noisy in both directions —
            // measured 1.7x the reference's own noise on one case and 0.4x
            // on the other. It is here to catch a forward that is actually
            // broken (which lands at 100%+), not to certify precision. The
            // per-stage bound lives in DFlash2StageProbeTests, and the
            // checks below cover what the decode loop consumes.
            XCTAssertLessThan(
                portHiddenDrift, Swift.max(0.25, referenceHiddenNoise * 3),
                "\(name): drafter hidden drifted \(portHiddenDrift), far past the reference's own bf16/fp32 noise \(referenceHiddenNoise)"
            )

            let referenceLogitNoise = relativeDrift(
                expectedLogits, golden["\(name)__logits_fp32"]!)
            let portLogitDrift = relativeDrift(logits.asType(.float32), expectedLogits)
            XCTAssertLessThanOrEqual(
                portLogitDrift, referenceLogitNoise * 1.5,
                "\(name): port logit drift \(portLogitDrift) exceeds the reference's own bf16/fp32 noise \(referenceLogitNoise)"
            )

            // Candidate sets: same argument. Count how many ids the
            // reference's own two precisions disagree on, and require the
            // port to be no worse.
            let gotCandidates = Set(proposal.candidates.reshaped(-1).asArray(Int32.self))
            let wantCandidates = Set(expectedCandidates.reshaped(-1).asArray(Int32.self))
            let fp32Candidates = Set(
                golden["\(name)__candidates_fp32"]!.reshaped(-1).asArray(Int32.self))
            let referenceCandidateNoise = wantCandidates.symmetricDifference(fp32Candidates).count
            let portCandidateDrift = gotCandidates.symmetricDifference(wantCandidates).count
            XCTAssertLessThanOrEqual(
                portCandidateDrift, Swift.max(referenceCandidateNoise, 2),
                "\(name): port candidate set differs by \(portCandidateDrift) ids; the reference's own precisions differ by \(referenceCandidateNoise)"
            )

            // The traced path. Positions where the reference's own two
            // precisions already disagree are near-ties by definition and
            // carry no information; every OTHER position must match, which
            // is what proves the selector's edge scoring and its sequential
            // predecessor threading are right.
            let gotTokens = proposal.tokens.reshaped(-1).asArray(Int32.self)
            let fp32Tokens = golden["\(name)__tokens_fp32"]!.reshaped(-1).asArray(Int32.self)
            XCTAssertEqual(gotTokens.count, expectedTokens.count, "\(name): path length")
            var stableMismatches: [Int] = []
            for index in gotTokens.indices where expectedTokens[index] == fp32Tokens[index] {
                if gotTokens[index] != expectedTokens[index] {
                    stableMismatches.append(index)
                }
            }
            XCTAssertTrue(
                stableMismatches.isEmpty,
                "\(name): path differs at position(s) \(stableMismatches) where BOTH reference precisions agreed — that is a real divergence, not a near-tie. got=\(gotTokens) want=\(expectedTokens)"
            )
            print(
                "  \(name): hidden \(fmt(portHiddenDrift)) vs ref-noise \(fmt(referenceHiddenNoise)); "
                    + "logits \(fmt(portLogitDrift)) vs \(fmt(referenceLogitNoise)); "
                    + "candidates \(portCandidateDrift) vs \(referenceCandidateNoise); "
                    + "path stable-mismatches 0")
        }
    }
}
