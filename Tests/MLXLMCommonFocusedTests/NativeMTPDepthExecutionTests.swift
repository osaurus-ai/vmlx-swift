import Foundation
import MLX
import MLXNN
import XCTest
@testable import MLXLMCommon

/// Exercises actual iterator verify dispatch. The zero-weight constant target
/// makes every proposal correct; it is not a model-quality or speed benchmark.
final class NativeMTPDepthExecutionTests: XCTestCase {
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
    var kvHeads: [Int] { [1] }
    var nativeMTPAvailable: Bool { true }
    var verifyWidths: [Int] = []
    private let targetLogits: MLXArray
    private let draftLogits: MLXArray
    init(
        targetLogits: [Float] = [-100, 100, -100, -100],
        draftLogits: [Float] = [-100, 100, -100, -100]
    ) {
        self.targetLogits = MLXArray(targetLogits)
        self.draftLogits = MLXArray(draftLogits)
        super.init()
    }
    func newCache(parameters: GenerateParameters?) -> [KVCache] { [MambaCache()] }
    func makeNativeMTPCache() -> [KVCache] { [] }
    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        .tokens(input.text)
    }
    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray { result(inputs).logits }
    func nativeBackboneForward(_ inputs: MLXArray, cache: [KVCache]?) -> NativeMTPForwardResult {
        result(inputs)
    }
    func nativeBackboneMTPVerifyForward(_ inputs: MLXArray, cache: [KVCache]?) -> NativeMTPForwardResult {
        verifyWidths.append(inputs.ndim >= 2 ? inputs.dim(1) : inputs.size)
        return result(inputs)
    }
    func nativeMTPForward(hiddenStates: MLXArray, nextTokenIds: MLXArray, cache: [KVCache]?) -> NativeMTPForwardResult {
        result(nextTokenIds, logits: draftLogits)
    }
    func commitVerifiedBlock(cache: [KVCache], acceptedInputs: Int) -> Bool { true }
    func commitStagedVerifiedBlock(cache: [KVCache], acceptedInputs: Int, blockLength: Int) -> Bool { true }
    private func result(_ inputs: MLXArray, logits: MLXArray? = nil) -> NativeMTPForwardResult {
        let length = inputs.ndim >= 2 ? inputs.dim(1) : inputs.size
        return .init(logits: broadcast(logits ?? targetLogits, to: [1, length, 4]),
                     hiddenStates: MLXArray.zeros([1, length, 4]))
    }
}
