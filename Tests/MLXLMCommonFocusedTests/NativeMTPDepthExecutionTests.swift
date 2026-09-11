import Foundation
import MLX
import MLXNN
import XCTest
@testable import MLXLMCommon

/// Exercises actual iterator verify dispatch. The zero-weight constant target
/// makes every proposal correct; it is not a model-quality or speed benchmark.
final class NativeMTPDepthExecutionTests: XCTestCase {
    func testSampledStagedDispatchRequiresModelOptIn() throws {
        try FocusedMLXTestSupport.withLock {
            for enabled in [false, true] {
                let model = DepthDispatchTarget(sequence: true)
                model.nativeMTPSampledStagedVerificationEnabled = enabled
                var parameters = GenerateParameters(maxTokens: 24, temperature: 1)
                parameters.randomSeed = 829
                parameters.draftStrategy = .nativeMTP(depth: 3)
                var iterator = try NativeMTPTokenIterator(
                    input: LMInput(tokens: MLXArray([1, 1, 1])), model: model,
                    parameters: parameters, depth: 3)
                var tokens: [Int] = []
                let started = ProcessInfo.processInfo.systemUptime
                while let token = iterator.next() { tokens.append(token) }
                XCTAssertEqual(tokens, (0..<24).map { (2 + $0) % 32 })
                XCTAssertEqual(iterator.stagedVerifierCommitCount > 0, enabled)
                print("SAMPLED-STAGED optIn=\(enabled) fixtureTokS=\(Double(tokens.count) / max(ProcessInfo.processInfo.systemUptime - started, 1e-9)) dispatchOnly=true realModelSpeedProof=false")
            }
        }
    }

    func testSampledChunkDecisionCarriesEmittedTokenIDs() throws {
        try FocusedMLXTestSupport.withLock {
            for rejectionCase in 0...2 {
                let model = DepthDispatchTarget(sequence: true)
                model.setWrongDraft(rejectionCase == 1)
                if rejectionCase == 2 {
                    // Governor seeding advances the first speculative block.
                    // Put the wrong proposal at its second draft in either mode.
                    let governorOff = ProcessInfo.processInfo.environment["VMLX_NATIVE_MTP_AR_SAFETY"] == "0"
                    model.setWrongDraftInput(governorOff ? 4 : 6)
                }
                var parameters = GenerateParameters(maxTokens: 24, temperature: 1)
                parameters.randomSeed = 829
                // This fixture has no recurrent prefix snapshots. Exercise
                // chunk acceptance with checkpoint/replay, rather than claiming
                // it implements a real hybrid model's prefix-commit protocol.
                parameters.draftStrategy = .nativeMTP(depth: 3, verifierMode: "chunk_replay")
                var iterator = try NativeMTPTokenIterator(
                    input: LMInput(tokens: MLXArray([1, 1, 1])), model: model,
                    parameters: parameters, depth: 3)
                var tokens: [Int] = []
                let started = ProcessInfo.processInfo.systemUptime
                while let token = iterator.next() { tokens.append(token) }
                XCTAssertEqual(tokens, (0..<24).map { (2 + $0) % 32 })
                XCTAssertGreaterThan(iterator.chunkVerifierCount, 0)
                if rejectionCase != 0 {
                    XCTAssertGreaterThan(iterator.residualCorrectionCount, 0)
                } else {
                    XCTAssertGreaterThan(iterator.bonusCount, 0)
                }
                if rejectionCase == 2 {
                    XCTAssertGreaterThan(iterator.acceptedByDepth[1, default: 0], 0,
                        "Must reject after accepting one draft, not just at index zero")
                }
                print("SAMPLED-CHUNK rejectionCase=\(rejectionCase) accepted=\(iterator.acceptedByDepth) fixtureTokS=\(Double(tokens.count) / max(ProcessInfo.processInfo.systemUptime - started, 1e-9)) realModelSpeedProof=false")
            }
        }
    }

    func testProductiveCalibrationResumesPriorDepth() throws {
        guard ProcessInfo.processInfo.environment["VMLX_NATIVE_MTP_AR_SAFETY"] != "0",
              ProcessInfo.processInfo.environment["VMLX_MTP_VERIFY_PREFETCH"] == "0" else {
            throw XCTSkip("Requires governor on and prefetch off for controlled cost observation")
        }
        try FocusedMLXTestSupport.withLock {
            for depth in 1...3 {
                let model = DepthDispatchTarget(sequence: true, backboneDelay: 0.010)
                var parameters = GenerateParameters(maxTokens: 340, temperature: 0)
                parameters.draftStrategy = .nativeMTP(depth: depth)
                var iterator = try NativeMTPTokenIterator(
                    input: LMInput(tokens: MLXArray([1, 1, 1])), model: model,
                    parameters: parameters, depth: depth)
                var tokens: [Int] = []
                var widthsBeforeCalibration: Int?
                var resumedWidth: Int?
                let started = ProcessInfo.processInfo.systemUptime
                while tokens.count < 340, let token = iterator.next() {
                    tokens.append(token)
                    if iterator.autoregressiveFallbackTokenCount > 2,
                       widthsBeforeCalibration == nil {
                        widthsBeforeCalibration = model.verifyWidths.count
                    }
                    if let before = widthsBeforeCalibration,
                       model.verifyWidths.count > before, resumedWidth == nil {
                        resumedWidth = model.verifyWidths[before]
                    }
                }
                XCTAssertEqual(tokens, (0..<340).map { (2 + $0) % 32 })
                XCTAssertNotNil(widthsBeforeCalibration, "Must exercise real productive AR calibration")
                XCTAssertEqual(iterator.autoregressiveFallbackTokenCount, 4,
                    "Expected two seed and two calibration steps, not an unrelated loss recovery")
                XCTAssertEqual(resumedWidth, depth + 1, "Calibration must retain the productive depth")
                print("CALIBRATION-RESUME depth=\(depth) width=\(String(describing: resumedWidth)) fixtureTokS=\(Double(tokens.count) / max(ProcessInfo.processInfo.systemUptime - started, 1e-9)) realModelSpeedProof=false")
            }
        }
    }

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
            // Deliberately separate cost regimes from sub-millisecond dispatch
            // and timer jitter. D3/D2 initially lose to AR, D1 wins; afterward
            // equal verify costs make each wider accepted block cheaper/token.
            let model = DepthDispatchTarget(sequence: true, backboneDelay: 0.020)
            model.setVerifyDelays([2: 0.012, 3: 0.160, 4: 0.320])
            var parameters = GenerateParameters(maxTokens: 480, temperature: 0)
            parameters.draftStrategy = .nativeMTP(depth: 3)
            var iterator = try NativeMTPTokenIterator(
                input: LMInput(tokens: MLXArray([1, 1, 1])), model: model,
                parameters: parameters, depth: 3)
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
                    model.setVerifyDelays([2: 0.012, 3: 0.012, 4: 0.012])
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
    KVCacheDimensionProvider, DFlash2StagedVerifyRollbackModel, @unchecked Sendable
{
    var kvHeads: [Int] { sequence ? [1, 1] : [1] }
    var nativeMTPAvailable: Bool { true }
    var nativeMTPSampledStagedVerificationEnabled = false
    private let timingLock = NSLock()
    private var widths: [Int] = []
    var verifyWidths: [Int] { timingLock.withLock { widths } }
    private var delaysByWidth: [Int: TimeInterval] = [:]
    private var wrongDraft = false
    private var wrongDraftInput: Int?
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
    func setWrongDraftInput(_ value: Int) { timingLock.withLock { wrongDraftInput = value } }
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
        let shouldReject = timingLock.withLock {
            wrongDraft || wrongDraftInput.map { nextTokenIds.reshaped(-1)[0].item(Int.self) == $0 } == true
        }
        if shouldReject {
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
