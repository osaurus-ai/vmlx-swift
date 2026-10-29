import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon
@testable import MLXVLM

/// The runtime's native-MTP top rung (`NativeMTPDepthPolicy.runtimeTopRung`): the only top of the adaptive ladder
/// unless a diagnostic env lever or an explicit fixed depth says otherwise (no hard draft depth, Rule 0).
final class NativeMTPRuntimeTopRungTests: XCTestCase {
    private let noEnv: [String: String] = [:]

    func testRecursiveHeadUsesTheRuntimesRecursiveTopRung() {
        XCTAssertEqual(
            NativeMTPDepthPolicy.runtimeTopRung(model: TopRungFake(block: nil), environment: noEnv),
            NativeMTPDepthPolicy.recursiveHeadTopRung)
    }

    /// A block drafter's top rung is its deepest proposal (block - 1) within the verifier's exact window,
    /// never a host or family constant below it.
    func testBlockDrafterTopRungIsItsBlockWithinTheExactVerifyWindow() {
        let exactTop = FlashVerificationScope.rowExactVerifyRows.upperBound - 1
        XCTAssertEqual(NativeMTPDepthPolicy.verifierTopRung, exactTop)
        for block in [2, 4, 6, 8] {
            XCTAssertEqual(
                NativeMTPDepthPolicy.runtimeTopRung(model: TopRungFake(block: block), environment: noEnv),
                min(block - 1, exactTop), "block \(block)")
        }
        // A block wider than the exact verify window stays within it (correctness bound).
        XCTAssertEqual(
            NativeMTPDepthPolicy.runtimeTopRung(model: TopRungFake(block: 16), environment: noEnv), exactTop)
    }

    func testDeclaredVerifierCapabilityAndTheDiagnosticLever() {
        XCTAssertEqual(NativeMTPDepthPolicy.runtimeTopRung(model: DeclaredCeilingFake(), environment: noEnv), 2)
        // VMLX_MTP_DEPTH_CAP is a diagnostic lever: it replaces the runtime's rung (either way).
        XCTAssertEqual(
            NativeMTPDepthPolicy.runtimeTopRung(model: TopRungFake(block: 8), environment: ["VMLX_MTP_DEPTH_CAP": "3"]), 3)
        XCTAssertEqual(
            NativeMTPDepthPolicy.runtimeTopRung(model: DeclaredCeilingFake(), environment: ["VMLX_MTP_DEPTH_CAP": "6"]), 6)
    }

    /// Nemotron-H (and its Omni wrapper) declare no adaptive ceiling: their former 3 was a measured tuning cap
    /// that also hid the LiLiCorr drafter's block - 1 = 7 rung. Re-adding a conformance fails here.
    func testNemotronDeclaresNoTuningCeiling() {
        XCTAssertFalse((NemotronHModel.self as Any.Type) is any NativeMTPAdaptiveCeilingModel.Type)
        XCTAssertFalse((NemotronHOmni.self as Any.Type) is any NativeMTPAdaptiveCeilingModel.Type)
        // They are block drafters (LiLiCorr) and recursive heads; the runtime resolves their rung.
        XCTAssertTrue((NemotronHModel.self as Any.Type) is any NativeMTPBlockDraftModel.Type)
    }

    /// With the library's default policy, a block drafter reaches its block - 1 rung: the resolved maximum is the
    /// runtime's, not 5 (the default before 2026-10-29, which capped LiLiCorr's block 8 at depth 5).
    func testDefaultPolicyResolvesABlockDrafterToItsBlock() throws {
        let top = NativeMTPDepthPolicy.runtimeTopRung(model: TopRungFake(block: 8), environment: noEnv)
        let resolution = try GenerateParameters().nativeMTPDepthPolicy.resolve(requestedDepth: 3, runtimeCap: top)
        XCTAssertEqual(resolution.initialDepth, 3)
        XCTAssertEqual(resolution.maximumDepth, 7)
    }
}

/// A recursive head (block nil) or a block drafter of `block` positions. Only the capability properties are read.
private final class TopRungFake: Module, LanguageModel, NativeMTPBlockDraftModel, @unchecked Sendable {
    let block: Int?
    init(block: Int?) { self.block = block }
    var nativeMTPAvailable: Bool { true }
    var nativeMTPUsesBlockDrafts: Bool { block != nil }
    var nativeMTPBlockDraftDepthLimit: Int { Swift.max(1, (block ?? 2) - 1) }
    var nativeMTPBlockDraftVocabularySize: Int { 4 }
    func nativeMTPBlockDraft(nextToken: MLXArray, depth: Int, targetCache: [KVCache]) -> MLXArray {
        MLXArray.zeros([depth], type: Int32.self)
    }
    func newCache(parameters: GenerateParameters?) -> [KVCache] { [KVCacheSimple()] }
    func makeNativeMTPCache() -> [KVCache] { [] }
    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult { .tokens(input.text) }
    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray { MLXArray.zeros([1, 1, 4]) }
    func nativeBackboneForward(_ inputs: MLXArray, cache: [KVCache]?) -> NativeMTPForwardResult {
        .init(logits: MLXArray.zeros([1, 1, 4]), hiddenStates: MLXArray.zeros([1, 1, 4]))
    }
    func nativeMTPForward(hiddenStates: MLXArray, nextTokenIds: MLXArray, cache: [KVCache]?) -> NativeMTPForwardResult {
        .init(logits: MLXArray.zeros([1, 1, 4]), hiddenStates: MLXArray.zeros([1, 1, 4]))
    }
}

/// A model that declares a verifier capability of its own (a correctness bound).
private final class DeclaredCeilingFake: NativeMTPAdaptiveCeilingModel {
    var nativeMTPAdaptiveDepthCeiling: Int { 2 }
}
