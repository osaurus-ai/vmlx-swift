import Foundation
import MLX
import MLXNN
import XCTest
@testable import MLXLMCommon

/// Synthetic deterministic oracle for the ACTUAL iterator/controller. This is
/// not model accuracy or performance evidence. Fresh process with AR safety
/// disabled isolates acceptance demotion from pause/re-entry cache resets.
final class NativeMTPDemotionHeadHistoryTests: XCTestCase {
    func testPureAcceptanceDemotionKeepsConfirmedHeadPairs() throws {
        let env = ProcessInfo.processInfo.environment
        guard env["VMLX_NATIVE_MTP_AR_SAFETY"] == "0",
              env["VMLX_MTP_VERIFY_PREFETCH"] == "0",
              env["VMLX_NATIVE_MTP_DISABLE_ADAPTIVE"] != "1",
              env["VMLX_MTP_ALIGNED_HEAD_CACHE"] != "0",
              env["VMLX_MTP_COMPILED_VERIFY"] != "1",
              env["VMLX_NATIVE_MTP_FORCE_ROLLBACK_REPAIR"] == nil else {
            throw XCTSkip("Requires isolated process: VMLX_NATIVE_MTP_AR_SAFETY=0 VMLX_MTP_VERIFY_PREFETCH=0; adaptive/aligned enabled, compiled verify off, no forced repair")
        }
        try FocusedMLXTestSupport.withLock {
            let model = Demotion119Target()
            var parameters = GenerateParameters(maxTokens: 100, temperature: 0)
            parameters.nativeMTPDepthPolicy = .adaptive(maximumDepth: 3)
            parameters.draftStrategy = .nativeMTP(depth: 3)
            var iterator = try NativeMTPTokenIterator(input: LMInput(tokens: MLXArray([Int32(0)])),
                model: model, parameters: parameters, depth: 3)
            let start = ProcessInfo.processInfo.systemUptime
            var emitted = 0
            while iterator.verifyCalls < 12 && emitted < 80 {
                guard iterator.next() != nil else { XCTFail("Unexpected iterator end"); return }
                emitted += 1
            }
            XCTAssertEqual(iterator.verifyCalls, 12)
            XCTAssertEqual(iterator.acceptedByDepth[1], 12, "Exactly one accepted draft each cycle")
            XCTAssertEqual(model.commits.count, 12)
            // Eleven controls must already pass BEFORE the demotion boundary.
            for row in model.commits.prefix(11) {
                XCTAssertEqual(row.actual, row.expected, "Ordinary rollback/trim lost committed head pairs")
            }
            let last = try XCTUnwrap(model.commits.last)
            print("HEAD-DEMOTION119 cycle=12 expected=\(last.expected.count) actual=\(last.actual.count) refreshes=\(iterator.mtpCacheRefreshCount) fixtureTPS=\(Double(emitted)/max(ProcessInfo.processInfo.systemUptime-start,1e-9)) synthetic=true")
            XCTAssertEqual(model.verifyWidths, Array(repeating: 4, count: 12))
            XCTAssertEqual(last.actual, last.expected, "Pure D3→D2 demotion must preserve already-confirmed pairs")
            // Do not run more cycles after a failed invariant and obscure its cause.
            guard last.actual == last.expected else { return }
            while iterator.verifyCalls < 13 && emitted < 90 {
                guard iterator.next() != nil else { XCTFail("No next cycle after demotion"); return }
                emitted += 1
            }
            XCTAssertEqual(model.verifyWidths.last, 3, "Actual next verifier must use demoted depth")
            XCTAssertEqual(model.commits.last?.actual, model.commits.last?.expected)
        }
    }
}

private final class Demotion119Target: Module, NativeMTPModel, KVCacheDimensionProvider {
    struct Commit { let actual: [Int32]; let expected: [Int32] }
    var kvHeads: [Int] { [1] }
    var nativeMTPAvailable: Bool { true }
    var verifyWidths: [Int] = []
    var commits: [Commit] = []
    private var confirmed: [Int32] = []
    private var headCycle = 0
    private var callInCycle = 0
    func newCache(parameters: GenerateParameters?) -> [KVCache] { [KVCacheSimple()] }
    func makeNativeMTPCache() -> [KVCache] { [KVCacheSimple()] }
    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult { .tokens(input.text) }
    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray { nativeBackboneForward(inputs,cache:cache).logits }
    func nativeBackboneForward(_ inputs: MLXArray, cache: [KVCache]?) -> NativeMTPForwardResult {
        append(inputs, cache)
        return result(inputs, wrong: false)
    }
    func nativeBackboneMTPVerifyForward(_ inputs: MLXArray, cache: [KVCache]?) -> NativeMTPForwardResult {
        verifyWidths.append(inputs.size)
        return nativeBackboneForward(inputs,cache:cache)
    }
    func nativeMTPForward(hiddenStates: MLXArray, nextTokenIds: MLXArray, cache: [KVCache]?) -> NativeMTPForwardResult {
        let cycle = verifyWidths.count
        if cycle != headCycle { headCycle = cycle; callInCycle = 0 }
        if callInCycle == 0 {
            let actual = cache?.first?.state.last?.reshaped(-1).asArray(Int32.self) ?? []
            if cycle > 0 { commits.append(Commit(actual: actual, expected: confirmed)) }
            confirmed.append(contentsOf: nextTokenIds.reshaped(-1).asArray(Int32.self))
        }
        append(nextTokenIds, cache)
        // First draft is correct, second rejects. Independent target always
        // predicts token+1. This fixed fixture schedule is not sampled/model proof.
        let wrong = callInCycle == 1
        callInCycle += 1
        return result(nextTokenIds, wrong: wrong)
    }
    private func append(_ ids: MLXArray, _ cache: [KVCache]?) {
        guard let kv = cache?.first as? KVCacheSimple else { return }
        let x = ids.reshaped(1,1,ids.size,1)
        let updated = kv.update(keys:x, values:x)
        MLX.eval(updated.0,updated.1)
    }
    private func result(_ ids: MLXArray, wrong: Bool) -> NativeMTPForwardResult {
        let tokens = ids.reshaped(-1).asArray(Int32.self)
        let logits: [Float] = tokens.flatMap { token in
            (0..<32).map { $0 == (Int(token) + (wrong ? 2 : 1)) % 32 ? Float(10) : Float(-10) }
        }
        return .init(logits:MLXArray(logits).reshaped(1,tokens.count,32),
                     hiddenStates:ids.asType(.float32).reshaped(1,tokens.count,1))
    }
}
