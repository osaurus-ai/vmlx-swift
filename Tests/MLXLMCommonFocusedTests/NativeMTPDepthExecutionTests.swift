import Foundation
import MLX
import MLXNN
import XCTest
@testable import MLXLMCommon

/// Exercises actual iterator verify dispatch. The zero-weight constant target
/// makes every proposal correct; it is not a model-quality or speed benchmark.
final class NativeMTPDepthExecutionTests: XCTestCase {
    func testSampledRejectionPauseCanProbeAfterProposalQualityChanges() throws {
        guard ProcessInfo.processInfo.environment["VMLX_NATIVE_MTP_AR_SAFETY"] != "0" else {
            throw XCTSkip("Requires the real governor")
        }
        try FocusedMLXTestSupport.withLock {
            let model = DepthDispatchTarget(sequence: true, backboneDelay: 0.003)
            model.setWrongDraft(true)
            var parameters = GenerateParameters(maxTokens: 240, temperature: 1)
            parameters.randomSeed = 829
            parameters.draftStrategy = .nativeMTP(depth: 1)
            var iterator = try NativeMTPTokenIterator(
                input: LMInput(tokens: MLXArray([1, 1, 1])), model: model,
                parameters: parameters, depth: 1)
            let start = ProcessInfo.processInfo.systemUptime
            var tokens: [Int] = []
            var verifiesAtChange = 0
            while tokens.count < 240, let token = iterator.next() {
                tokens.append(token)
                if tokens.count == 96 {
                    verifiesAtChange = iterator.verifyCalls
                    model.setWrongDraft(false)
                }
            }
            XCTAssertEqual(tokens, (0..<240).map { (2 + $0) % 32 })
            XCTAssertGreaterThan(iterator.sequentialVerifierCount, 0)
            XCTAssertEqual(iterator.stagedVerifierCommitCount, 0)
            XCTAssertGreaterThan(iterator.rejectedCount, 0)
            XCTAssertGreaterThan(iterator.autoregressiveFallbackTokenCount, 2)
            XCTAssertGreaterThan(iterator.verifyCalls, verifiesAtChange)
            print("SAMPLED-RECOVERY verifiesBefore=\(verifiesAtChange) after=\(iterator.verifyCalls) fixtureTokS=\(Double(tokens.count) / max(ProcessInfo.processInfo.systemUptime - start, 1e-9)) realModelSpeedProof=false")
        }
    }

    func testLosingDepthsDescendAndChangedRegimeCanRecover() throws {
        guard ProcessInfo.processInfo.environment["VMLX_NATIVE_MTP_AR_SAFETY"] != "0",
              ProcessInfo.processInfo.environment["VMLX_MTP_VERIFY_PREFETCH"] == "0" else {
            throw XCTSkip("Controlled host-cost row: governor on, verify prefetch off; prefetch parity is separate")
        }
        try FocusedMLXTestSupport.withLock {
            let model = DepthDispatchTarget(sequence: true, backboneDelay: 0.004)
            model.setVerifyDelays([2: 0.001, 3: 0.040, 4: 0.080])
            var parameters = GenerateParameters(maxTokens: 480, temperature: 0)
            parameters.draftStrategy = .nativeMTP(depth: 3)
            var iterator = try NativeMTPTokenIterator(
                input: LMInput(tokens: MLXArray([1, 1, 1])), model: model,
                parameters: parameters, depth: 3)
            XCTAssertEqual(model.autoregressiveBackboneCalls, 0, "initial seed must not use the AR specialization")
            let start = ProcessInfo.processInfo.systemUptime
            var tokens: [Int] = []
            var widthsAtChange = 0
            while tokens.count < 480, let token = iterator.next() {
                tokens.append(token)
                if tokens.count == 240 {
                    let widths = model.verifyWidths
                    XCTAssertTrue(widths.contains(4))
                    XCTAssertTrue(widths.contains(3), "D3 must descend through D2")
                    XCTAssertTrue(widths.contains(2), "D1 should be discovered")
                    widthsAtChange = widths.count
                    model.setVerifyDelays([2: 0.001, 3: 0.001, 4: 0.001])
                }
            }
            let recovered = Array(model.verifyWidths.dropFirst(widthsAtChange))
            XCTAssertTrue(recovered.contains(3), "Changed regime must recover D2")
            XCTAssertTrue(recovered.contains(4), "Changed regime must recover D3")
            XCTAssertTrue(model.verifyWidths.allSatisfy { $0 <= 4 })
            XCTAssertEqual(tokens, (0..<480).map { (2 + $0) % 32 })
            print("MTP-LADDER widths=\(model.verifyWidths) recovered=\(recovered) fixtureTokS=\(Double(tokens.count) / max(ProcessInfo.processInfo.systemUptime - start, 1e-9)) realModelSpeedProof=false")
        }
    }

    func testGovernorHandoffPreservesNonconstantAcceptedTokens() throws {
        guard ProcessInfo.processInfo.environment["VMLX_NATIVE_MTP_AR_SAFETY"] != "0" else {
            throw XCTSkip("Requires the real governor")
        }
        try FocusedMLXTestSupport.withLock {
            let model = DepthDispatchTarget(sequence: true, verifyDelay: 0.020)
            var parameters = GenerateParameters(maxTokens: 240, temperature: 0)
            parameters.draftStrategy = .nativeMTP(depth: 3)
            var iterator = try NativeMTPTokenIterator(
                input: LMInput(tokens: MLXArray([1, 1, 1])), model: model,
                parameters: parameters, depth: 3)
            XCTAssertEqual(model.autoregressiveBackboneCalls, 0, "initial seed must not use the AR specialization")
            let start = ProcessInfo.processInfo.systemUptime
            var tokens: [Int] = []
            var previousAR = 0
            while tokens.count < 240, let token = iterator.next() {
                tokens.append(token)
                if iterator.autoregressiveFallbackTokenCount > previousAR {
                    previousAR = iterator.autoregressiveFallbackTokenCount
                    let kv = try XCTUnwrap(iterator.cache.last as? KVCacheSimple)
                    XCTAssertEqual(kv.offset, 3 + tokens.count - 1)
                    let state = try XCTUnwrap(kv.readKV())
                    XCTAssertEqual(state.keys.asArray(Int32.self), (0..<kv.offset).map(Int32.init))
                    XCTAssertEqual(state.values.asArray(Int32.self),
                                   [Int32(1), 1, 1] + tokens.dropLast().map(Int32.init))
                }
            }
            XCTAssertEqual(tokens, (0..<240).map { (2 + $0) % 32 })
            XCTAssertEqual(model.autoregressiveBackboneCalls, iterator.autoregressiveFallbackTokenCount,
                           "only explicit AR steps may use the specialization")
            XCTAssertGreaterThan(iterator.stagedVerifierCommitCount, 0)
            XCTAssertGreaterThan(iterator.arSafetyTrips, 0)
            XCTAssertGreaterThan(iterator.autoregressiveFallbackTokenCount, 2)
            print("GOVERNOR-QUEUE tokens=\(tokens.count) trips=\(iterator.arSafetyTrips) fixtureTokS=\(Double(tokens.count) / max(ProcessInfo.processInfo.systemUptime - start, 1e-9)) realModelSpeedProof=false")
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
        let start = ProcessInfo.processInfo.systemUptime
        while let token = iterator.next() {
            XCTAssertEqual(token, 1)
            count += 1
            if count >= 160 { break }
        }
        XCTAssertEqual(count, 160)
        XCTAssertGreaterThan(iterator.stagedVerifierCommitCount, 0)
        print("DEPTH-DISPATCH requested=\(depth) policy=\(policy) widths=\(Set(model.verifyWidths).sorted()) stagedCommits=\(iterator.stagedVerifierCommitCount) fixtureTokens=\(count) fixtureTokS=\(Double(count) / max(ProcessInfo.processInfo.systemUptime - start, 1e-9)) realModelSpeedProof=false")
        return model.verifyWidths
    }
}

private final class DepthDispatchTarget: Module, LanguageModel, NativeMTPModel,
    NativeMTPAutoregressiveBackboneModel,
    KVCacheDimensionProvider, DFlash2StagedVerifyRollbackModel, @unchecked Sendable
{
    var kvHeads: [Int] { sequence ? [1, 1] : [1] }
    var nativeMTPAvailable: Bool { true }
    private let timingLock = NSLock()
    private var widths: [Int] = []
    var verifyWidths: [Int] { timingLock.withLock { widths } }
    private var delaysByWidth: [Int: TimeInterval] = [:]
    private var wrongDraft = false
    private var arCalls = 0
    var autoregressiveBackboneCalls: Int { timingLock.withLock { arCalls } }
    let sequence: Bool
    let verifyDelay: TimeInterval
    let backboneDelay: TimeInterval
    init(sequence: Bool = false, verifyDelay: TimeInterval = 0, backboneDelay: TimeInterval = 0) {
        self.sequence = sequence
        self.verifyDelay = verifyDelay
        self.backboneDelay = backboneDelay
    }
    func setVerifyDelays(_ values: [Int: TimeInterval]) {
        timingLock.withLock { delaysByWidth = values }
    }
    func setWrongDraft(_ value: Bool) { timingLock.withLock { wrongDraft = value } }
    func newCache(parameters: GenerateParameters?) -> [KVCache] {
        sequence ? [MambaCache(), KVCacheSimple()] : [MambaCache()]
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
    func nativeAutoregressiveBackboneForward(_ inputs: MLXArray, cache: [KVCache]?) -> NativeMTPForwardResult {
        timingLock.withLock { arCalls += 1 }
        return nativeBackboneForward(inputs, cache: cache)
    }
    func nativeBackboneMTPVerifyForward(_ inputs: MLXArray, cache: [KVCache]?) -> NativeMTPForwardResult {
        let width = inputs.ndim >= 2 ? inputs.dim(1) : inputs.size
        let delay = timingLock.withLock {
            widths.append(width)
            return delaysByWidth[width] ?? verifyDelay
        }
        if delay > 0 { Thread.sleep(forTimeInterval: delay) }
        appendKV(inputs, cache: cache)
        return result(inputs)
    }
    func nativeMTPForward(hiddenStates: MLXArray, nextTokenIds: MLXArray, cache: [KVCache]?) -> NativeMTPForwardResult {
        if timingLock.withLock({ wrongDraft }) {
            let length = nextTokenIds.size
            return .init(logits: broadcast(MLXArray([Float(100)] + Array(repeating: Float(-100), count: 31)), to: [1, length, 32]),
                         hiddenStates: MLXArray.zeros([1, length, 4]))
        }
        return result(nextTokenIds)
    }
    func commitVerifiedBlock(cache: [KVCache], acceptedInputs: Int) -> Bool { true }
    func commitStagedVerifiedBlock(cache: [KVCache], acceptedInputs: Int, blockLength: Int) -> Bool { true }
    private func appendKV(_ inputs: MLXArray, cache: [KVCache]?) {
        guard sequence, let kv = cache?.last as? KVCacheSimple else { return }
        let count = inputs.size
        let positions = MLXArray((kv.offset..<(kv.offset + count)).map(Int32.init))
        let arrays = kv.update(keys: positions.reshaped(1, 1, count, 1),
                               values: inputs.reshaped(1, 1, count, 1))
        MLX.eval(arrays.0, arrays.1)
    }
    private func result(_ inputs: MLXArray) -> NativeMTPForwardResult {
        let length = inputs.ndim >= 2 ? inputs.dim(1) : inputs.size
        if sequence {
            let ids = inputs.reshaped(-1).asArray(Int32.self)
            let logits = ids.flatMap { token in
                (0..<32).map { $0 == (Int(token) + 1) % 32 ? Float(100) : Float(-100) }
            }
            return .init(logits: MLXArray(logits).reshaped(1, length, 32),
                         hiddenStates: MLXArray.zeros([1, length, 4]))
        }
        return .init(logits: broadcast(MLXArray([Float(-100), 100, -100, -100]), to: [1, length, 4]),
                     hiddenStates: MLXArray.zeros([1, length, 4]))
    }
}

#if DEBUG
extension NativeMTPDepthExecutionTests {
    func testRepeatedMeanLossInvokesActualAdjacentOrAROwnerPreservingQueue() throws {
        guard ProcessInfo.processInfo.environment["VMLX_NATIVE_MTP_AR_SAFETY"] != "0" else {
            throw XCTSkip("Requires actual governor")
        }
        try FocusedMLXTestSupport.withLock {
            for depth in 1...3 {
                let model = DepthDispatchTarget(sequence: true)
                var parameters = GenerateParameters(maxTokens: 96, temperature: 0)
                parameters.draftStrategy = .nativeMTP(depth: depth)
                var iterator = try NativeMTPTokenIterator(
                    input: LMInput(tokens: MLXArray([1, 1, 1])), model: model,
                    parameters: parameters, depth: depth)
                var now = 100.0
                iterator.testingSeedMeasuredGovernor(depth: depth, now: now)
                let original = iterator.testingGovernorSnapshot
                let pattern = [20.0, 20, 20, 20, 20, 20, 80, 80]
                for duration in pattern {
                    now += duration / 1000
                    iterator.testingFeedGovernorCycle(now: now)
                }
                XCTAssertEqual(iterator.testingGovernorSnapshot.depth, depth)
                XCTAssertFalse(iterator.testingGovernorSnapshot.paused)
                XCTAssertEqual(iterator.testingGovernorSnapshot.pendingLossEnd, 16)
                for (index, duration) in pattern.enumerated() {
                    now += duration / 1000
                    iterator.testingFeedGovernorCycle(now: now)
                    if index < 7 {
                        XCTAssertEqual(iterator.testingGovernorSnapshot.depth, depth)
                        XCTAssertFalse(iterator.testingGovernorSnapshot.paused)
                    }
                }
                let result = iterator.testingGovernorSnapshot
                XCTAssertEqual(result.depth, max(1, depth - 1))
                XCTAssertEqual(result.paused, depth == 1)
                XCTAssertEqual(result.trips, original.trips + (depth == 1 ? 1 : 0))
                XCTAssertNil(result.pendingLossEnd)
                XCTAssertEqual(result.pendingTokens, original.pendingTokens,
                               "governor transition cannot drop or replace committed queued tokens")
            }
        }
    }

    func testActualARReentryClearsMeanLossAndRetainsProbeContract() throws {
        guard ProcessInfo.processInfo.environment["VMLX_NATIVE_MTP_AR_SAFETY"] != "0" else {
            throw XCTSkip("Requires actual governor")
        }
        try FocusedMLXTestSupport.withLock {
            let model = DepthDispatchTarget(sequence: true)
            var parameters = GenerateParameters(maxTokens: 96, temperature: 0)
            parameters.draftStrategy = .nativeMTP(depth: 1)
            var iterator = try NativeMTPTokenIterator(
                input: LMInput(tokens: MLXArray([1, 1, 1])), model: model,
                parameters: parameters, depth: 1)
            var now = 100.0
            iterator.testingSeedMeasuredGovernor(depth: 1, now: now)
            let queued = iterator.testingGovernorSnapshot.pendingTokens
            for duration in Array(repeating: [Double]([20, 20, 20, 20, 20, 20, 80, 80]), count: 2).flatMap({ $0 }) {
                now += duration / 1000
                iterator.testingFeedGovernorCycle(now: now)
            }
            XCTAssertTrue(iterator.testingGovernorSnapshot.paused)
            for index in 0..<16 {
                now += 0.025
                try iterator.testingMeasuredARStep(now: now)
                XCTAssertEqual(iterator.testingGovernorSnapshot.paused, index < 15)
            }
            XCTAssertEqual(iterator.testingGovernorSnapshot.depth, 1)
            XCTAssertEqual(iterator.testingGovernorSnapshot.probeCycles, 6)
            XCTAssertNil(iterator.testingGovernorSnapshot.pendingLossEnd)
            for _ in 0..<6 {
                now += 0.010
                iterator.testingFeedGovernorCycle(now: now)
            }
            XCTAssertEqual(iterator.testingGovernorSnapshot.resumes, 1)
            XCTAssertFalse(iterator.testingGovernorSnapshot.paused)
            XCTAssertEqual(iterator.testingGovernorSnapshot.probeCycles, 0)
            XCTAssertNil(iterator.testingGovernorSnapshot.pendingLossEnd)
            XCTAssertEqual(iterator.testingGovernorSnapshot.pendingTokens, queued)
        }
    }
}
#endif


/// Shape/scope checks only: no MLX arrays, model loading, or GPU work.
final class FlashVerificationScopeAdmissionTests: XCTestCase {
    func testOrdinaryCallsAreNotAdmitted() {
        for rows in 1...9 {
            XCTAssertFalse(FlashVerificationScope.usesRowExactVerification(inputShape: [1, rows, 2560]))
        }
    }

    func testOnlyMatchingSmallVerificationRowsAreAdmitted() {
        for rows in 2...8 {
            FlashVerificationScope.withVerification(inputShape: [1, rows]) {
                XCTAssertTrue(FlashVerificationScope.usesRowExactVerification(inputShape: [1, rows, 2560]))
                for invalid in [[1, 1, 2560], [1, 9, 2560], [2, rows, 2560],
                                [1, rows + 1, 2560], [1, rows, 0], [1, rows, -1],
                                [1, rows], [rows, 2560], [1, rows, 1, 2560], []] {
                    XCTAssertFalse(FlashVerificationScope.usesRowExactVerification(inputShape: invalid))
                }
            }
            XCTAssertFalse(FlashVerificationScope.usesRowExactVerification(inputShape: [1, rows, 2560]))
        }
    }

    func testInvalidVerificationEntryShapesDoNotEstablishScope() {
        for shape in [[], [1], [1, 1], [1, 9], [2, 2], [1, 2, 2560], [0, 2], [1, -2]] {
            FlashVerificationScope.withVerification(inputShape: shape) {
                XCTAssertFalse(FlashVerificationScope.usesRowExactVerification(inputShape: [1, 2, 2560]))
            }
        }
    }

    func testNestedScopeRestoresOuterRows() {
        FlashVerificationScope.withVerification(inputShape: [1, 2]) {
            FlashVerificationScope.withVerification(inputShape: [1, 4]) {
                XCTAssertTrue(FlashVerificationScope.usesRowExactVerification(inputShape: [1, 4, 2560]))
                XCTAssertFalse(FlashVerificationScope.usesRowExactVerification(inputShape: [1, 2, 2560]))
            }
            FlashVerificationScope.withVerification(inputShape: [1, 1]) {
                XCTAssertFalse(FlashVerificationScope.usesRowExactVerification(inputShape: [1, 2, 2560]))
            }
            XCTAssertTrue(FlashVerificationScope.usesRowExactVerification(inputShape: [1, 2, 2560]))
        }
    }

    func testThrowRestoresScope() {
        enum Stop: Error { case expected }
        XCTAssertThrowsError(try FlashVerificationScope.withVerification(inputShape: [1, 2]) {
            XCTAssertTrue(FlashVerificationScope.usesRowExactVerification(inputShape: [1, 2, 2560]))
            throw Stop.expected
        })
        XCTAssertFalse(FlashVerificationScope.usesRowExactVerification(inputShape: [1, 2, 2560]))
    }
}
