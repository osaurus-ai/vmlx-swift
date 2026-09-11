import Foundation
import MLX
import MLXNN
import XCTest
@testable import MLXLMCommon

/// Exercises actual iterator verify dispatch. The zero-weight constant target
/// makes every proposal correct; it is not a model-quality or speed benchmark.
final class NativeMTPDepthExecutionTests: XCTestCase {
    func testGroupedProbabilityEvaluationTiming() throws {
        guard ProcessInfo.processInfo.environment["VMLX_BENCH_GROUPED_PROBABILITIES"] == "1" else {
            throw XCTSkip("Opt-in generated-logit component timing")
        }
        try FocusedMLXTestSupport.withLock {
            // Matches the inspected Flash bundle vocabulary, not an allocation
            // of model weights. Production still derives shapes from logits.
            let vocab = 248_320
            var parameters = GenerateParameters(temperature: 1, topP: 0.95, topK: 20)
            parameters.randomSeed = 829
            let sampler = SpeculativeSamplingController(parameters: parameters)
            for count in [2, 3, 4, 6] {
                let values = (0..<(count * vocab)).map { Float(sin(Double($0) * 0.037)) }
                let logits = MLXArray(values).reshaped(count, vocab).asType(.bfloat16)
                MLX.eval(logits)
                func measure(grouped: Bool) -> Double {
                    let start = ProcessInfo.processInfo.systemUptime
                    var rows: [MLXArray] = []
                    for row in 0..<count {
                        let p = sampler.probabilities(logits: logits[row])
                        if !grouped { MLX.eval(p) }
                        rows.append(p)
                    }
                    if grouped { MLX.eval(rows) }
                    return (ProcessInfo.processInfo.systemUptime - start) * 1000
                }
                for _ in 0..<3 { _ = measure(grouped: false); _ = measure(grouped: true) }
                var sequential: [Double] = []
                var grouped: [Double] = []
                for trial in 0..<9 {
                    if trial.isMultiple(of: 2) {
                        sequential.append(measure(grouped: false)); grouped.append(measure(grouped: true))
                    } else {
                        grouped.append(measure(grouped: true)); sequential.append(measure(grouped: false))
                    }
                }
                print("GROUPED-PROBABILITY-BENCH rows=\(count) vocab=\(vocab) sequential_ms=\(sequential) grouped_ms=\(grouped) median_ratio=\(sequential.sorted()[4] / grouped.sorted()[4]) generated_logits=1 synchronized=1 realModelSpeedProof=false")
            }
        }
    }

    func testGroupedProbabilityEvaluationPreservesDistributionsAndRandomDraws() throws {
        try FocusedMLXTestSupport.withLock {
            var parameters = GenerateParameters(temperature: 1, topP: 0.95, topK: 20)
            parameters.randomSeed = 829
            var compared = 0
            var accepted = 0
            var corrected = 0
            for dtype: DType in [.float16, .bfloat16, .float32] {
                for depth in [1, 2, 3, 5] {
                    let count = depth + 1
                    let vocab = 257
                    let values = (0..<(count * vocab)).map {
                        Float(sin(Double($0) * 0.37) * 3 + cos(Double($0) * 0.11))
                    }
                    let logits = MLXArray(values).reshaped(count, vocab).asType(dtype)
                    MLX.eval(logits)
                    let sequential = SpeculativeSamplingController(parameters: parameters)
                    let grouped = SpeculativeSamplingController(parameters: parameters)
                    var reference: [MLXArray] = []
                    var candidate: [MLXArray] = []
                    for row in 0..<count {
                        let p = sequential.probabilities(logits: logits[row])
                        MLX.eval(p)
                        reference.append(p)
                        candidate.append(grouped.probabilities(logits: logits[row]))
                    }
                    MLX.eval(candidate)
                    for row in 0..<count {
                        XCTAssertEqual(reference[row].asArray(Float.self),
                                       candidate[row].asArray(Float.self))
                        let token = MLXArray(0)
                        // Alternating exact proposals and disjoint-support proposals
                        // cover acceptance and residual correction without new draws.
                        let q = row.isMultiple(of: 2) ? reference[row]
                            : MLXArray([Float(1)] + Array(repeating: Float(0), count: vocab - 1))
                        let a = sequential.acceptOrCorrect(draftToken: token,
                            targetProbabilities: reference[row], draftProbabilities: q)
                        let b = grouped.acceptOrCorrect(draftToken: token,
                            targetProbabilities: candidate[row], draftProbabilities: q)
                        XCTAssertEqual(a.accepted, b.accepted)
                        if a.accepted { accepted += 1 } else { corrected += 1 }
                        XCTAssertEqual(a.acceptanceProbability, b.acceptanceProbability)
                        XCTAssertEqual(a.correction?.item(Int.self), b.correction?.item(Int.self))
                        XCTAssertEqual(sequential.sampleFromTarget(probabilities: reference[row]).item(Int.self),
                                       grouped.sampleFromTarget(probabilities: candidate[row]).item(Int.self))
                        compared += 1
                    }
                }
            }
            XCTAssertGreaterThan(accepted, 0)
            XCTAssertGreaterThan(corrected, 0)
            print("GROUPED-PROBABILITY rows=\(compared) exact_distribution_and_rng_control=true realModelSpeedProof=false")
        }
    }

    func testSampledStagingOptInDoesNotEnableUnqualifiedModel() throws {
        try FocusedMLXTestSupport.withLock {
            let model = DepthDispatchTarget()
            var parameters = GenerateParameters(maxTokens: 32, temperature: 1)
            parameters.randomSeed = 829
            parameters.draftStrategy = .nativeMTP(depth: 3)
            var iterator = try NativeMTPTokenIterator(
                input: LMInput(tokens: MLXArray([1, 1, 1])), model: model,
                parameters: parameters, depth: 3, experimentalSampledStaging: true)
            var count = 0
            let start = ProcessInfo.processInfo.systemUptime
            while iterator.next() != nil { count += 1 }
            XCTAssertEqual(count, 32)
            XCTAssertNil(iterator.terminalErrorDescription)
            XCTAssertEqual(iterator.stagedVerifierCommitCount, 0)
            XCTAssertGreaterThan(iterator.sequentialVerifierCount, 0)
            print("SAMPLED-STAGING-EXCLUDED fixtureTokS=\(Double(count) / max(ProcessInfo.processInfo.systemUptime - start, 1e-9)) realModelSpeedProof=false")
        }
    }
    func testSampledAcceptancePauseCanProbeAgain() throws {
        guard ProcessInfo.processInfo.environment["VMLX_NATIVE_MTP_AR_SAFETY"] != "0" else {
            throw XCTSkip("Requires the production AR-safety governor")
        }
        try FocusedMLXTestSupport.withLock {
            let model = DepthDispatchTarget(
                draftLogits: [100, -100, -100, -100], backboneDelay: 0.005,
                trackKV: true)
            var parameters = GenerateParameters(maxTokens: 160, temperature: 1)
            parameters.randomSeed = 829
            parameters.draftStrategy = .nativeMTP(depth: 1)
            parameters.nativeMTPDepthPolicy = .fixed
            var iterator = try NativeMTPTokenIterator(
                input: LMInput(tokens: MLXArray([1, 1, 1])), model: model,
                parameters: parameters, depth: 1)
            let start = ProcessInfo.processInfo.systemUptime
            var count = 0
            var verifiesAtChange = 0
            var previousAR = 0
            while count < 160, let token = iterator.next() {
                count += 1
                XCTAssertEqual(token, 1, "Rejected proposals must never enter target output")
                if iterator.autoregressiveFallbackTokenCount > previousAR {
                    previousAR = iterator.autoregressiveFallbackTokenCount
                    let kv = try XCTUnwrap(iterator.cache.last as? KVCacheSimple)
                    XCTAssertEqual(kv.offset, 3 + count - 1)
                    let state = try XCTUnwrap(kv.readKV())
                    XCTAssertEqual(state.keys.asArray(Int32.self), (0..<kv.offset).map(Int32.init))
                    XCTAssertEqual(state.values.asArray(Int32.self), Array(repeating: 1, count: kv.offset))
                }
                if count == 64 {
                    verifiesAtChange = iterator.verifyCalls
                    model.useTargetAsDraft = true
                }
            }
            XCTAssertEqual(count, 160)
            XCTAssertGreaterThan(iterator.sequentialVerifierCount, 0)
            XCTAssertEqual(iterator.stagedVerifierCommitCount, 0)
            XCTAssertGreaterThan(iterator.rejectedCount, 0)
            XCTAssertGreaterThan(iterator.autoregressiveFallbackTokenCount, 2)
            XCTAssertGreaterThan(iterator.verifyCalls, verifiesAtChange,
                "An acceptance-based AR pause must probe again after proposal quality changes")
            print("SAMPLED-RECOVERY verifiesAtChange=\(verifiesAtChange) finalVerifies=\(iterator.verifyCalls) trips=\(iterator.arSafetyTrips) fixtureTokS=\(Double(count) / (ProcessInfo.processInfo.systemUptime - start)) fullCacheProof=false realModelSpeedProof=false")
        }
    }
    func testSuccessfulResumeUsesRecentlyMeasuredARCostForLaterLoss() throws {
        guard ProcessInfo.processInfo.environment["VMLX_NATIVE_MTP_AR_SAFETY"] != "0" else {
            throw XCTSkip("Requires the production AR-safety governor")
        }
        try FocusedMLXTestSupport.withLock {
            // Controlled host delays, not model performance: expensive startup
            // AR, then cheaper live AR, a winning resume and a subsequent loss.
            let model = DepthDispatchTarget(
                draftDelay: 0.100, backboneDelay: 0.020, verifyDelay: 0.004)
            var parameters = GenerateParameters(maxTokens: 320, temperature: 0)
            parameters.draftStrategy = .nativeMTP(depth: 3)
            parameters.nativeMTPDepthPolicy = .fixed
            var iterator = try NativeMTPTokenIterator(
                input: LMInput(tokens: MLXArray([1, 1, 1])), model: model,
                parameters: parameters, depth: 3)
            var enteredPause = false
            var resumedAt: Int?
            var count = 0
            let start = ProcessInfo.processInfo.systemUptime
            while count < 320, let token = iterator.next() {
                count += 1
                XCTAssertEqual(token, 1)
                if !enteredPause, iterator.arSafetyTrips > 0 {
                    enteredPause = true
                    model.setDelays(draft: 0, backbone: 0.004)
                }
                if resumedAt == nil, iterator.arSafetyResumes > 0 {
                    resumedAt = iterator.verifyCalls
                    // ~16ms/token now loses to live AR ~4ms, but looks cheap
                    // against the stale startup AR ~20ms. Verify width is fixed.
                    model.setDelays(draft: 0.020, backbone: 0.004)
                }
                if let resumedAt,
                    iterator.arSafetyTrips >= 2 || iterator.verifyCalls >= resumedAt + 12
                { break }
            }
            XCTAssertTrue(enteredPause, "Control must enter the initial AR pause")
            XCTAssertNotNil(resumedAt, "Control must complete a winning resume probe")
            XCTAssertGreaterThanOrEqual(iterator.arSafetyTrips, 2,
                "A later loss must use the recent AR measurement, not the startup seed")
            if let resumedAt {
                XCTAssertLessThanOrEqual(iterator.verifyCalls - resumedAt, 12)
            }
            print("GOVERNOR-REENTRY tokens=\(count) fixtureTokS=\(Double(count) / (ProcessInfo.processInfo.systemUptime - start)) trips=\(iterator.arSafetyTrips) resumes=\(iterator.arSafetyResumes) resumedAt=\(String(describing: resumedAt)) finalVerify=\(iterator.verifyCalls) realModelSpeedProof=false")
        }
    }

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
    private let timingLock = NSLock()
    private var draftDelay: TimeInterval
    private var backboneDelay: TimeInterval
    private let verifyDelay: TimeInterval
    private let trackKV: Bool
    var draftCalls = 0
    var useTargetAsDraft = false
    init(
        targetLogits: [Float] = [-100, 100, -100, -100],
        draftLogits: [Float] = [-100, 100, -100, -100],
        draftDelay: TimeInterval = 0, backboneDelay: TimeInterval = 0,
        verifyDelay: TimeInterval = 0, trackKV: Bool = false
    ) {
        self.targetLogits = MLXArray(targetLogits)
        self.draftLogits = MLXArray(draftLogits)
        self.draftDelay = draftDelay
        self.backboneDelay = backboneDelay
        self.verifyDelay = verifyDelay
        self.trackKV = trackKV
        super.init()
    }
    func newCache(parameters: GenerateParameters?) -> [KVCache] {
        trackKV ? [MambaCache(), KVCacheSimple()] : [MambaCache()]
    }
    func makeNativeMTPCache() -> [KVCache] { [] }
    func setDelays(draft: TimeInterval, backbone: TimeInterval) {
        timingLock.withLock {
            draftDelay = draft
            backboneDelay = backbone
        }
    }
    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        .tokens(input.text)
    }
    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        appendKV(inputs, cache: cache)
        return result(inputs).logits
    }
    func nativeBackboneForward(_ inputs: MLXArray, cache: [KVCache]?) -> NativeMTPForwardResult {
        let delay = timingLock.withLock { backboneDelay }
        if delay > 0 { Thread.sleep(forTimeInterval: delay) }
        appendKV(inputs, cache: cache)
        return result(inputs)
    }
    func nativeBackboneMTPVerifyForward(_ inputs: MLXArray, cache: [KVCache]?) -> NativeMTPForwardResult {
        if verifyDelay > 0 { Thread.sleep(forTimeInterval: verifyDelay) }
        verifyWidths.append(inputs.ndim >= 2 ? inputs.dim(1) : inputs.size)
        appendKV(inputs, cache: cache)
        return result(inputs)
    }
    func nativeMTPForward(hiddenStates: MLXArray, nextTokenIds: MLXArray, cache: [KVCache]?) -> NativeMTPForwardResult {
        draftCalls += 1
        let delay = timingLock.withLock { draftDelay }
        if delay > 0 { Thread.sleep(forTimeInterval: delay) }
        return result(nextTokenIds, logits: useTargetAsDraft ? targetLogits : draftLogits)
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
