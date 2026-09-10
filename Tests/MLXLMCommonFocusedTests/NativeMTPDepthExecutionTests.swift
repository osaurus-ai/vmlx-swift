import Foundation
import MLX
import MLXNN
import XCTest
@testable import MLXLMCommon

/// Exercises actual iterator verify dispatch. The zero-weight constant target
/// makes every proposal correct; it is not a model-quality or speed benchmark.
final class NativeMTPDepthExecutionTests: XCTestCase {
    func testGovernorCalibrationDoesNotBuildDiscardedInitialDrafts() throws {
        guard ProcessInfo.processInfo.environment["VMLX_NATIVE_MTP_AR_SAFETY"] != "0" else {
            throw XCTSkip("Requires governor calibration")
        }
        try FocusedMLXTestSupport.withLock {
            for depth in 1...3 {
                for temperature: Float in [0, 1] {
                    let model = DepthDispatchTarget()
                    var parameters = GenerateParameters(maxTokens: 16, temperature: temperature)
                    parameters.randomSeed = 829
                    parameters.draftStrategy = .nativeMTP(depth: depth)
                    parameters.nativeMTPDepthPolicy = .fixed
                    let started = ProcessInfo.processInfo.systemUptime
                    var iterator = try NativeMTPTokenIterator(
                        input: LMInput(tokens: MLXArray([1, 1, 1])), model: model,
                        parameters: parameters, depth: depth)
                    XCTAssertEqual(model.draftCalls, 0, "D\(depth) initial proposals would be discarded")
                    for _ in 0..<3 { XCTAssertEqual(iterator.next(), 1) }
                    XCTAssertEqual(model.draftCalls, 0, "First calibration step must not draft")
                    XCTAssertEqual(iterator.next(), 1)
                    XCTAssertEqual(iterator.autoregressiveFallbackTokenCount, 2)
                    XCTAssertEqual(model.draftCalls, depth, "Prime only after the second calibration step")
                    print("CALIBRATION depth=\(depth) temperature=\(temperature) drafts=\(model.draftCalls) fixtureTokS=\(4 / (ProcessInfo.processInfo.systemUptime - started))")
                }
            }
        }
    }
    func testEarlyStopAbandonsPrefetchedKVRows() throws {
        guard ProcessInfo.processInfo.environment["VMLX_MTP_VERIFY_PREFETCH"] != "0" else {
            throw XCTSkip("Requires prefetch enabled")
        }
        try FocusedMLXTestSupport.withLock {
            let model = DepthDispatchTarget(trackKV: true)
            var parameters = GenerateParameters(maxTokens: 80, temperature: 0)
            parameters.draftStrategy = .nativeMTP(depth: 3)
            parameters.nativeMTPDepthPolicy = .fixed
            var iterator = try NativeMTPTokenIterator(
                input: LMInput(tokens: MLXArray([1, 1, 1])), model: model,
                parameters: parameters, depth: 3)
            let started = ProcessInfo.processInfo.systemUptime
            var tokens: [Int] = []
            while iterator.verifyPrefetchSubmitCount == 0, tokens.count < 40,
                  let token = iterator.next() { tokens.append(token) }
            XCTAssertGreaterThan(iterator.verifyPrefetchSubmitCount, 0)
            let committed = iterator.acceptedByDepth.reduce(0) { $0 + ($1.key + 1) * $1.value }
            let expectedOffset = 3 + 1 + iterator.autoregressiveFallbackTokenCount + committed
            XCTAssertGreaterThan(iterator.cache.last!.offset, expectedOffset,
                                 "The control must have speculative KV rows outstanding")
            iterator.storeCacheAfterGeneration(generatedTokenIds: tokens, includeGeneratedBoundary: false)
            XCTAssertEqual(iterator.verifyPrefetchAbandonedCount, 1)
            let kv = try XCTUnwrap(iterator.cache.last as? KVCacheSimple)
            XCTAssertEqual(kv.offset, expectedOffset)
            let state = try XCTUnwrap(kv.readKV())
            XCTAssertEqual(state.keys.asArray(Int32.self), (0..<expectedOffset).map(Int32.init))
            XCTAssertEqual(state.values.asArray(Int32.self), Array(repeating: 1, count: expectedOffset))
            print("PREFETCH-STOP fixtureTokS=\(Double(tokens.count) / (ProcessInfo.processInfo.systemUptime - started)) restoredOffset=\(kv.offset) diskProof=false")
        }
    }
    func testGovernorPausePreservesCommittedTokensAndBoundsResumeProbe() throws {
        guard ProcessInfo.processInfo.environment["VMLX_NATIVE_MTP_AR_SAFETY"] != "0" else {
            throw XCTSkip("Governor-on fixture requires AR safety enabled")
        }
        try FocusedMLXTestSupport.withLock {
            let model = DepthDispatchTarget(draftDelay: 0.010, backboneDelay: 0.001, trackKV: true)
            var parameters = GenerateParameters(maxTokens: 240, temperature: 0)
            parameters.draftStrategy = .nativeMTP(depth: 3)
            parameters.nativeMTPDepthPolicy = .fixed
            var iterator = try NativeMTPTokenIterator(
                input: LMInput(tokens: MLXArray([1, 1, 1])), model: model,
                parameters: parameters, depth: 3)
            var count = 0
            var previousAR = 0
            var pauses: [Int] = []
            let started = ProcessInfo.processInfo.systemUptime
            while count < 240 {
                let oldTrips = iterator.arSafetyTrips
                let oldDrafts = model.draftCalls
                guard let token = iterator.next() else { break }
                count += 1
                XCTAssertEqual(token, 1)
                if iterator.arSafetyTrips > oldTrips {
                    pauses.append(iterator.verifyCalls)
                    XCTAssertEqual(model.draftCalls, oldDrafts,
                                   "A paused cycle must not build another draft batch")
                }
                if iterator.autoregressiveFallbackTokenCount > previousAR {
                    previousAR = iterator.autoregressiveFallbackTokenCount
                    let committed = iterator.acceptedByDepth.reduce(0) { $0 + ($1.key + 1) * $1.value }
                    XCTAssertEqual(count, 2 + previousAR + committed,
                                   "AR resumed before all verified tokens reached the consumer")
                    let kv = try XCTUnwrap(iterator.cache.last as? KVCacheSimple)
                    XCTAssertEqual(kv.offset, 3 + count - 1)
                    let state = try XCTUnwrap(kv.readKV())
                    XCTAssertEqual(state.keys.asArray(Int32.self), (0..<kv.offset).map(Int32.init))
                    XCTAssertEqual(state.values.asArray(Int32.self), Array(repeating: 1, count: kv.offset))
                    if pauses.count >= 2 { break }
                }
            }
            XCTAssertGreaterThanOrEqual(pauses.count, 2)
            if pauses.count >= 2 { XCTAssertEqual(pauses[1] - pauses[0], 6) }
            XCTAssertGreaterThan(iterator.stagedVerifierCommitCount, 0)
            if ProcessInfo.processInfo.environment["VMLX_MTP_VERIFY_PREFETCH"] != "0" {
                XCTAssertGreaterThan(iterator.verifyPrefetchConsumedCount, 0)
            }
            let elapsed = ProcessInfo.processInfo.systemUptime - started
            print("GOVERNOR-FIXTURE pauses=\(pauses) tokens=\(count) fixtureTokS=\(Double(count) / elapsed) prefetchConsumed=\(iterator.verifyPrefetchConsumedCount) kvPositionProof=true recurrentStateProof=false")
        }
    }
    func testFixedDepthsBoundExecutedVerifyWidths() throws {
        try requireIsolatedDepthRun()
        try FocusedMLXTestSupport.withLock {
            for depth in 1...3 {
                let widths = try run(depth: depth, policy: .fixed)
                XCTAssertFalse(widths.isEmpty)
                XCTAssertTrue(widths.allSatisfy { $0 <= depth + 1 }, "D\(depth): \(widths)")
                XCTAssertTrue(widths.contains(depth + 1))
            }
        }
    }

    func testAdaptiveActuallyPromotesButRespectsCeiling() throws {
        try requireIsolatedDepthRun()
        try FocusedMLXTestSupport.withLock {
            let widths = try run(depth: 1, policy: .adaptive(maximumDepth: 3))
            XCTAssertTrue(widths.contains(2))
            XCTAssertTrue(widths.contains { $0 > 2 }, "No actual promotion: \(widths)")
            XCTAssertTrue(widths.allSatisfy { $0 <= 4 })
        }
    }

    func testSampledIteratorDoesNotSilentlyBecomeGreedy() throws {
        try requireIsolatedDepthRun()
        try FocusedMLXTestSupport.withLock {
            let tokens = try runSampled(disjointDraft: false)
            XCTAssertEqual(Set(tokens), Set([0, 1]))
        }
    }

    func testSampledIteratorRejectsDraftsOutsideTargetSupport() throws {
        try requireIsolatedDepthRun()
        try FocusedMLXTestSupport.withLock {
            let tokens = try runSampled(disjointDraft: true)
            XCTAssertEqual(Set(tokens), Set([0, 1]))
        }
    }

    /// The target assigns equal probability to tokens 0 and 1; top-k removes
    /// the other two. A disjoint proposal must be rejected, then corrected
    /// from the target residual. This exercises iterator sampling, NOT real
    /// model rollback: this fixture's cache commit methods are deliberate no-ops.
    private func runSampled(disjointDraft: Bool) throws -> [Int] {
        let target: [Float] = [1, 1, -100, -100]
        let model = DepthDispatchTarget(
            targetLogits: target,
            draftLogits: disjointDraft ? [-100, -100, 1, 1] : target)
        var parameters = GenerateParameters(maxTokens: 160, temperature: 1, topK: 2)
        parameters.randomSeed = 829
        parameters.draftStrategy = .nativeMTP(depth: 3)
        parameters.nativeMTPDepthPolicy = .fixed
        var iterator = try NativeMTPTokenIterator(
            input: LMInput(tokens: MLXArray([0, 1, 0])), model: model,
            parameters: parameters, depth: 3)
        var tokens: [Int] = []
        let started = ProcessInfo.processInfo.systemUptime
        while tokens.count < 160, let token = iterator.next() { tokens.append(token) }
        let elapsed = ProcessInfo.processInfo.systemUptime - started
        XCTAssertEqual(tokens.count, 160)
        // Sampled hybrid verification is sequential; the staged verify hook
        // observed by verifyWidths is intentionally not used on that path.
        XCTAssertGreaterThan(iterator.verifyMainForwardCount, 0)
        XCTAssertGreaterThan(iterator.mtpForwardCount, 0)
        XCTAssertGreaterThan(iterator.acceptanceProbabilityCount, 0)
        XCTAssertTrue(model.verifyWidths.allSatisfy { $0 <= 4 })
        if disjointDraft {
            XCTAssertGreaterThan(iterator.rejectedCount, 0)
            XCTAssertGreaterThan(iterator.residualCorrectionCount, 0)
        } else {
            XCTAssertEqual(iterator.rejectedCount, 0)
        }
        print("SAMPLED-DISPATCH disjointDraft=\(disjointDraft) support=\(Set(tokens).sorted()) rejections=\(iterator.rejectedCount) residuals=\(iterator.residualCorrectionCount) sequentialVerifies=\(iterator.sequentialVerifierCount) fixtureTokens=\(tokens.count) fixtureTokS=\(Double(tokens.count) / max(elapsed, 1e-9))")
        return tokens
    }

    private func requireIsolatedDepthRun() throws {
        guard ProcessInfo.processInfo.environment["VMLX_NATIVE_MTP_AR_SAFETY"] == "0" else {
            throw XCTSkip("Run this dispatch-only fixture with VMLX_NATIVE_MTP_AR_SAFETY=0; real governor speed qualification is separate")
        }
    }

    private func run(depth: Int, policy: NativeMTPDepthPolicy) throws -> [Int] {
        let model = DepthDispatchTarget()
        var parameters = GenerateParameters(maxTokens: 160, temperature: 0)
        parameters.draftStrategy = .nativeMTP(depth: depth)
        parameters.nativeMTPDepthPolicy = policy
        var iterator = try NativeMTPTokenIterator(
            input: LMInput(tokens: MLXArray([1, 1, 1])), model: model,
            parameters: parameters, depth: depth)
        var count = 0
        let started = ProcessInfo.processInfo.systemUptime
        while let token = iterator.next() {
            XCTAssertEqual(token, 1)
            count += 1
            if count >= 160 { break }
        }
        let elapsed = ProcessInfo.processInfo.systemUptime - started
        XCTAssertEqual(count, 160)
        XCTAssertGreaterThan(iterator.stagedVerifierCommitCount, 0)
        print("DEPTH-DISPATCH requested=\(depth) policy=\(policy) widths=\(Set(model.verifyWidths).sorted()) stagedCommits=\(iterator.stagedVerifierCommitCount) fixtureTokens=\(count) fixtureTokS=\(Double(count) / max(elapsed, 1e-9))")
        return model.verifyWidths
    }
}

private final class DepthDispatchTarget: Module, LanguageModel, NativeMTPModel,
    KVCacheDimensionProvider, DFlash2StagedVerifyRollbackModel, @unchecked Sendable
{
    var kvHeads: [Int] { trackKV ? [1, 1] : [1] }
    var nativeMTPAvailable: Bool { true }
    var verifyWidths: [Int] = []
    private let targetLogits: MLXArray
    private let draftLogits: MLXArray
    private let draftDelay: TimeInterval
    private let backboneDelay: TimeInterval
    private let trackKV: Bool
    var draftCalls = 0
    init(
        targetLogits: [Float] = [-100, 100, -100, -100],
        draftLogits: [Float] = [-100, 100, -100, -100],
        draftDelay: TimeInterval = 0, backboneDelay: TimeInterval = 0, trackKV: Bool = false
    ) {
        self.targetLogits = MLXArray(targetLogits)
        self.draftLogits = MLXArray(draftLogits)
        self.draftDelay = draftDelay
        self.backboneDelay = backboneDelay
        self.trackKV = trackKV
        super.init()
    }
    func newCache(parameters: GenerateParameters?) -> [KVCache] {
        trackKV ? [MambaCache(), KVCacheSimple()] : [MambaCache()]
    }
    func makeNativeMTPCache() -> [KVCache] { [] }
    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        .tokens(input.text)
    }
    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        appendKV(inputs, cache: cache)
        return result(inputs).logits
    }
    func nativeBackboneForward(_ inputs: MLXArray, cache: [KVCache]?) -> NativeMTPForwardResult {
        if backboneDelay > 0 { Thread.sleep(forTimeInterval: backboneDelay) }
        appendKV(inputs, cache: cache)
        return result(inputs)
    }
    func nativeBackboneMTPVerifyForward(_ inputs: MLXArray, cache: [KVCache]?) -> NativeMTPForwardResult {
        verifyWidths.append(inputs.ndim >= 2 ? inputs.dim(1) : inputs.size)
        appendKV(inputs, cache: cache)
        return result(inputs)
    }
    func nativeMTPForward(hiddenStates: MLXArray, nextTokenIds: MLXArray, cache: [KVCache]?) -> NativeMTPForwardResult {
        draftCalls += 1
        if draftDelay > 0 { Thread.sleep(forTimeInterval: draftDelay) }
        return result(nextTokenIds, logits: draftLogits)
    }
    func commitVerifiedBlock(cache: [KVCache], acceptedInputs: Int) -> Bool { true }
    private func appendKV(_ inputs: MLXArray, cache: [KVCache]?) {
        guard trackKV, let kv = cache?.last as? KVCacheSimple else { return }
        let count = inputs.size
        let positions = MLXArray((kv.offset..<(kv.offset + count)).map(Int32.init))
        let arrays = kv.update(keys: positions.reshaped(1, 1, count, 1),
                               values: inputs.reshaped(1, 1, count, 1))
        MLX.eval(arrays.0, arrays.1)
    }
    func commitStagedVerifiedBlock(cache: [KVCache], acceptedInputs: Int, blockLength: Int) -> Bool { true }
    private func result(_ inputs: MLXArray, logits: MLXArray? = nil) -> NativeMTPForwardResult {
        let length = inputs.ndim >= 2 ? inputs.dim(1) : inputs.size
        return .init(logits: broadcast(logits ?? targetLogits, to: [1, length, 4]),
                     hiddenStates: MLXArray.zeros([1, length, 4]))
    }
}
