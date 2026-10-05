import Foundation
import MLX
import MLXNN
import XCTest
@testable import MLXLMCommon

/// A routing counterexample through the actual production iterator, not a
/// real-model quality/speed measurement. No weights/functions change between
/// requests: one prompt has bad proposals, the other has exact proposals.
/// Fresh process: AR_SAFETY=0 isolates the acceptance memo from wall-cost
/// governance; ADAPTIVE must remain enabled to exercise the real 16-cycle path.
final class NativeMTPNegativeMemoRequestTests: XCTestCase {
    func testLowAcceptancePromptMustNotDisableLaterAccuratePrompt() throws {
        let env = ProcessInfo.processInfo.environment
        try XCTSkipIf(env["VMLX_NATIVE_MTP_AR_SAFETY"] != "0",
                      "Routing-only counterfactual requires isolated AR governor")
        try XCTSkipIf(env["VMLX_NATIVE_MTP_DISABLE_ADAPTIVE"] == "1",
                      "Must exercise the production warmup/memo decision")
        try XCTSkipIf(env["VMLX_MTP_VERIFY_PREFETCH"] != "0",
                      "Keep verifier submission ownership out of this test")
        for key in ["VMLX_NATIVE_MTP_HYBRID_VERIFY", "VMLINUX_NATIVE_MTP_HYBRID_VERIFY"] {
            try XCTSkipIf(env[key] != nil, "The automatic hybrid warmup must run")
        }
        try FocusedMLXTestSupport.withLock {
            let model = PromptSensitiveMemoTarget()
            XCTAssertNil(NativeMTPHybridWarmupMemo.verdict(for: model))
            let bad = try run(model: model, promptKind: 0)
            let memoAfterBad = NativeMTPHybridWarmupMemo.verdict(for: model)
            let repeatBad = try run(model: model, promptKind: 0)
            let good = try run(model: model, promptKind: 1)
            let memoAfterGood = NativeMTPHybridWarmupMemo.verdict(for: model)
            let repeatGood = try run(model: model, promptKind: 1)
            let freshGood = try run(model: PromptSensitiveMemoTarget(), promptKind: 1)

            // Persist the discriminating observations before ideal-contract
            // assertions fail on the baseline. No direct memo insertion.
            let result: [String: Any] = [
                "schema": 1, "actualProductionIterator": true,
                "sameModelInstance": true, "modelFunctionsUnchanged": true,
                "promptBoundaryHintOnly": true, "actualDiskRestore": false,
                "governorDisabledForIsolation": true, "modelSpeedProof": false,
                "bad": bad.json, "repeatBad": repeatBad.json, "repeatGood": repeatGood.json, "memoAfterBad": memoAfterBad.map { $0 as Any } ?? NSNull(),
                "sameInstanceGood": good.json, "freshInstanceGood": freshGood.json
            ]
            let data = try JSONSerialization.data(withJSONObject: result, options: [.prettyPrinted, .sortedKeys])
            if let path = env["VMLX_MTP_NEGATIVE_MEMO_OUTPUT"] {
                try data.write(to: URL(fileURLWithPath: path), options: .atomic)
            }
            print("NEGATIVE-MEMO-REQUEST " + String(decoding: data, as: UTF8.self))
            XCTAssertNil(memoAfterBad)
            XCTAssertEqual(repeatBad.verifyCalls, 16, "Each failing request gets its own bounded warmup")
            XCTAssertEqual(repeatBad.accepted, 0)
            XCTAssertEqual(repeatBad.tokens, bad.tokens)
            XCTAssertEqual(repeatBad.arTokens, bad.arTokens)
            XCTAssertEqual(memoAfterGood, true)
            XCTAssertEqual(repeatGood.tokens, good.tokens)
            XCTAssertEqual(repeatGood.sequential, 0, "Passing warmup remains reusable")
            XCTAssertGreaterThan(repeatGood.chunk, 0)
            XCTAssertEqual(repeatGood.arTokens, 0)
            XCTAssertEqual(bad.verifyCalls, 16, "Must reach the actual warmup decision")
            XCTAssertEqual(bad.accepted, 0)
            XCTAssertGreaterThan(bad.arTokens, 0, "Current poor request still falls back safely")
            XCTAssertEqual(bad.tokens, Array(repeating: 1, count: 40))
            XCTAssertEqual(good.tokens, freshGood.tokens)
            XCTAssertEqual(good.tokens, Array(repeating: 1, count: 40))
            XCTAssertGreaterThan(freshGood.verifyCalls, 0, "Second prompt is independently speculative")
            XCTAssertEqual(freshGood.accepted, freshGood.verifyCalls)
            XCTAssertEqual(freshGood.arTokens, 0)
            XCTAssertGreaterThan(good.verifyCalls, 0,
                "A content-sensitive negative acceptance memo must not force later prompts to AR")
            XCTAssertEqual(good.arTokens, 0)
        }
    }

    private struct Row {
        let tokens: [Int]
        let verifyCalls: Int
        let accepted: Int
        let arTokens: Int
        let sequential: Int
        let chunk: Int
        let elapsed: Double
        var json: [String: Any] {
            ["tokens": tokens, "verifyCalls": verifyCalls, "acceptedDrafts": accepted,
             "arFallbackTokens": arTokens, "sequentialVerifier": sequential,
             "chunkVerifier": chunk, "elapsedSeconds": elapsed,
             "fixtureTokensPerSecond": Double(tokens.count) / max(elapsed, 1e-9)]
        }
    }

    private func run(model: PromptSensitiveMemoTarget, promptKind: Int32) throws -> Row {
        var parameters = GenerateParameters(maxTokens: 40, temperature: 0)
        parameters.draftStrategy = .nativeMTP(depth: 1)
        parameters.nativeMTPDepthPolicy = .fixed
        // The production grace gate is based on this processor boundary hint,
        // even without a coordinator hit. It suppresses the 12-cycle economic
        // demotion long enough to reach the 16-cycle warmup verdict.
        let input = LMInput(tokens: MLXArray([promptKind, 1, 1]),
                            cachePrefixTokenCounts: [2])
        let start = ProcessInfo.processInfo.systemUptime
        var iterator = try NativeMTPTokenIterator(input: input, model: model,
                                                 parameters: parameters, depth: 1)
        var tokens: [Int] = []
        while let token = iterator.next() { tokens.append(token) }
        return Row(tokens: tokens, verifyCalls: iterator.verifyCalls,
                   accepted: iterator.acceptedByDepth.reduce(0) { $0 + $1.key * $1.value },
                   arTokens: iterator.autoregressiveFallbackTokenCount,
                   sequential: iterator.sequentialVerifierCount,
                   chunk: iterator.chunkVerifierCount,
                   elapsed: ProcessInfo.processInfo.systemUptime - start)
    }
}

/// Minimal deterministic hybrid. The first prompt ID is retained in its
/// request-owned recurrent state and exposed through pre-final-norm hidden.
/// Target always predicts 1; head predicts the context ID (0 or 1).
/// This is a legitimate content-dependent proposal function, not a head whose
/// weights are swapped between requests. Empty head cache is protocol-legal;
/// this test isolates memo routing rather than head-history mathematics.
private final class PromptSensitiveMemoTarget: Module, LanguageModel, NativeMTPModel {
    var nativeMTPAvailable: Bool { true }
    func newCache(parameters: GenerateParameters?) -> [KVCache] { [MambaCache(), KVCacheSimple()] }
    func makeNativeMTPCache() -> [KVCache] { [] }
    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        (cache[0] as! MambaCache)[0] = input.text.tokens.reshaped(-1)[0]
        return .tokens(input.text)
    }
    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        nativeBackboneForward(inputs, cache: cache).logits
    }
    func nativeBackboneForward(_ inputs: MLXArray, cache: [KVCache]?) -> NativeMTPForwardResult {
        let length = inputs.size
        let recurrent = cache![0] as! MambaCache
        recurrent.offset += length
        let kv = cache![1] as! KVCacheSimple
        let input = inputs.reshaped(1, 1, length, 1)
        _ = kv.update(keys: input, values: input)
        let kind = recurrent[0]!.item(Int32.self)
        return .init(logits: logits(token: 1, length: length),
                     hiddenStates: MLXArray.full([1, length, 1], values: MLXArray(Float(kind))))
    }
    func nativeMTPForward(hiddenStates: MLXArray, nextTokenIds: MLXArray,
                          cache: [KVCache]?) -> NativeMTPForwardResult {
        let kind = hiddenStates.reshaped(-1)[0].item(Int.self)
        let length = nextTokenIds.size
        return .init(logits: logits(token: kind, length: length), hiddenStates: hiddenStates)
    }
    private func logits(token: Int, length: Int) -> MLXArray {
        broadcast(MLXArray(token == 1 ? [Float(-100), 100] : [Float(100), -100]),
                  to: [1, length, 2])
    }
}
