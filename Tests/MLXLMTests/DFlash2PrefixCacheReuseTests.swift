// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Does DFlash 2 keep cross-turn prefix caching working?
//
// This is not a performance question, it is a correctness-of-plumbing
// one, and it cannot be answered by looking at output: a turn that
// re-prefills a 3k-token prompt it should have restored produces exactly
// the right answer, just slowly. The only direct evidence is which
// tokens the model was actually asked to forward.
//
// So these tests use the same recording fake as
// `HybridStripBoundaryPrefillTests`: a model that appends every token id
// it forwards to an array. Turn 2 must forward ONLY its new suffix.
//
// The hybrid case is the one that matters. Qwen3.8 — the family DFlash 2
// targets — is 48 GatedDeltaNet layers, and hybrid state is
// path-dependent, so the finished prompt cache cannot be trimmed back to
// a reusable boundary. The only reusable checkpoint is the
// generation-suffix-stripped one, captured DURING prefill. An iterator
// that stores only the full prompt stores a boundary the next turn can
// never match, and cross-turn reuse silently drops to zero.

import Foundation
import MLX
@testable import MLXLMCommon
import MLXNN
import XCTest

/// Hybrid target that records forwarded tokens and satisfies everything
/// DFlash 2 asks of a target: per-layer capture, a shared embedding and
/// LM head, and per-step recurrent prefix-state recording.
private final class RecordingDFlash2Target: Module, LanguageModel, HiddenStateCaptureModel,
    TokenEmbedderModel, @unchecked Sendable
{
    private(set) var forwarded: [Int] = []
    let hidden: Int
    let vocab: Int

    init(hidden: Int, vocab: Int) {
        self.hidden = hidden
        self.vocab = vocab
        super.init()
    }

    var vocabularySize: Int { vocab }

    func resetRecording() { forwarded.removeAll() }

    // Hybrid like the real target family (Qwen3.8 = 48 GDN + 16 attention):
    // one recurrent layer plus one attention layer. A pure-Mamba fake is
    // unrepresentative — with no KV layer, a disk hit has nothing to
    // restore and the conservative rollback re-prefills everything, which
    // is TokenIterator's behaviour too, not a DFlash 2 defect.
    func newCache(parameters: GenerateParameters?) -> [KVCache] {
        [MambaCache(), KVCacheSimple()]
    }

    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        .tokens(input.text)
    }

    func embed(_ tokenIds: MLXArray) -> MLXArray {
        let ids = tokenIds.asType(.float32).expandedDimensions(axis: -1)
        return MLX.broadcast(ids / Float(vocab), to: ids.shape.dropLast() + [hidden])
            .asType(.float32)
    }

    func projectToLogits(_ h: MLXArray) -> MLXArray {
        let rows = h.dim(0) * h.dim(1)
        return MLXArray.zeros([h.dim(0), h.dim(1), vocab]) + h.sum(axis: -1, keepDims: true)
            + MLXArray(Float(rows)) * 0
    }

    func callAsFunction(
        _ inputs: MLXArray, cache: [KVCache]?, captureLayerIDs: Set<Int>
    ) -> (logits: MLXArray, capturedHiddenStates: [Int: MLXArray]) {
        callAsFunction(
            inputs, cache: cache, captureLayerIDs: captureLayerIDs,
            recordPrefixCommitStates: false)
    }

    func callAsFunction(
        _ inputs: MLXArray, cache: [KVCache]?, captureLayerIDs: Set<Int>,
        recordPrefixCommitStates: Bool
    ) -> (logits: MLXArray, capturedHiddenStates: [Int: MLXArray]) {
        let ids = inputs.reshaped([-1]).asArray(Int32.self).map(Int.init)
        forwarded.append(contentsOf: ids)

        if let mamba = cache?.first as? MambaCache {
            let base = mamba.state.first?.asArray(Float.self).first ?? 0
            var running = base
            for (step, id) in ids.enumerated() {
                running += Float(id)
                if recordPrefixCommitStates, ids.count > 1 {
                    // One checkpoint per prefix length, exactly what
                    // `commitRecordedPrefix` looks up on rollback.
                    mamba.recordPrefixCommitState(
                        length: step + 1,
                        arrays: [MLXArray([running]).reshaped([1, 1, 1])],
                        offset: mamba.offset + step + 1)
                }
            }
            mamba.state = [MLXArray([running]).reshaped([1, 1, 1])]
            mamba.offset += ids.count
        }

        if let kv = cache?.last as? KVCacheSimple {
            let n = ids.count
            _ = kv.update(
                keys: MLXArray.zeros([1, 1, n, 2]),
                values: MLXArray.zeros([1, 1, n, 2]))
        }

        let h = embed(inputs)
        var captured: [Int: MLXArray] = [:]
        for id in captureLayerIDs { captured[id] = h }
        return (projectToLogits(h), captured)
    }

    var supportsCapturingPrefixCommitRecording: Bool { true }
}

final class DFlash2PrefixCacheReuseTests: XCTestCase {

    private var tempDirs: [URL] = []

    override func tearDown() {
        for dir in tempDirs { try? FileManager.default.removeItem(at: dir) }
        tempDirs.removeAll()
        super.tearDown()
    }

    private let genPromptSuffix = [201, 202, 203]
    private let turn1User = [11, 12, 13, 14, 15, 16, 17]
    private let turn2Extra = [31, 32, 33, 34]
    private let hiddenSize = 8
    private let vocabSize = 256

    private func makeCoordinator(modelKey: String = "recording-dflash2|reasoning=off")
        -> CacheCoordinator
    {
        let diskDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("dflash2-cache-\(UUID().uuidString)")
        tempDirs.append(diskDir)
        let coordinator = CacheCoordinator(config: CacheCoordinatorConfig(
            usePagedCache: true,
            enableDiskCache: true,
            diskCacheDir: diskDir,
            modelKey: modelKey))
        coordinator.setHybrid(true)
        coordinator.setGenPromptSuffixTokens(genPromptSuffix)
        return coordinator
    }

    /// Tiny drafter with the DFlash 2 shape. Weights are the module's own
    /// initialisation — these tests are about which tokens get forwarded,
    /// not about what the drafter predicts.
    private func makeDrafter() throws -> DFlash2DraftModel {
        let json: [String: Any] = [
            "hidden_size": hiddenSize,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "intermediate_size": 16,
            "vocab_size": vocabSize,
            "rms_norm_eps": 1e-6,
            "max_position_embeddings": 4096,
            "num_target_layers": 1,
            "is_causal": false,
            "layer_types": ["full_attention"],
            "rope_parameters": ["rope_theta": 10000, "rope_type": "default"],
            "dflash_config": [
                "block_size": 4,
                "conv_kernel_size": 2,
                "conv_group_size": 2,
                "mask_token_id": 250,
                "selector_rank": 4,
                "selector_top_k": 2,
                "target_layer_ids": [0],
            ],
        ]
        return DFlash2DraftModel(try DFlash2Configuration(json: json))
    }

    private func input(_ tokens: [Int]) -> LMInput {
        LMInput(tokens: MLXArray(tokens.map { Int32($0) }).expandedDimensions(axis: 0))
    }

    private func parameters() -> GenerateParameters {
        var p = GenerateParameters(maxTokens: 2, temperature: 0)
        p.prefillStepSize = 4
        return p
    }

    /// Turn 1 must forward the prompt exactly once — no replay to
    /// reconstruct a boundary after the fact.
    func testPrefillForwardsEachPromptTokenExactlyOnce() throws {
        let lock = lockSerializedMLXTest()
        defer { lock.unlock() }

        let target = RecordingDFlash2Target(hidden: hiddenSize, vocab: vocabSize)
        let drafter = try makeDrafter()
        let coordinator = makeCoordinator()
        let prompt = turn1User + genPromptSuffix

        var iterator = try DFlash2TokenIterator(
            input: input(prompt),
            target: target,
            drafter: drafter,
            blockSize: nil,
            parameters: parameters(),
            cacheCoordinator: coordinator)

        XCTAssertEqual(
            target.forwarded, prompt,
            "prefill must forward the prompt once, in order, with no replay")

        let afterPrefill = target.forwarded
        iterator.storeCacheAfterGeneration(
            generatedTokenIds: [], includeGeneratedBoundary: false)
        XCTAssertEqual(
            target.forwarded, afterPrefill,
            "storing the reusable boundary must not re-derive it through the model")
    }

    /// The one that matters: turn 2 must reuse turn 1's stored prefix and
    /// forward only what is new.
    ///
    /// A hybrid's reusable checkpoint is the generation-suffix-stripped
    /// prefix — turn 2's prompt is `turn1User + <assistant reply> + …`,
    /// which does NOT contain turn 1's full prompt, because the generation
    /// prompt suffix is replaced. Storing only the full prompt therefore
    /// stores something no later turn can ever match.
    func testSecondTurnReusesStoredPrefixInsteadOfReprefilling() throws {
        let lock = lockSerializedMLXTest()
        defer { lock.unlock() }

        let target = RecordingDFlash2Target(hidden: hiddenSize, vocab: vocabSize)
        let drafter = try makeDrafter()
        let coordinator = makeCoordinator()

        // Turn 1.
        let turn1 = turn1User + genPromptSuffix
        var first = try DFlash2TokenIterator(
            input: input(turn1),
            target: target,
            drafter: drafter,
            blockSize: nil,
            parameters: parameters(),
            cacheCoordinator: coordinator)
        _ = first.next()
        first.storeCacheAfterGeneration(
            generatedTokenIds: [], includeGeneratedBoundary: false)

        // Store-side probe first, mirroring HybridStripBoundaryPrefillTests:
        // the stripped boundary must be fetchable before turn 2 can use it.
        let mediaSalt = computeCacheSalt(
            for: input(turn1), parameters: parameters())
        let probe = coordinator.fetch(
            tokens: turn1User + turn2Extra + genPromptSuffix,
            mediaSalt: mediaSalt,
            skipExactDiskBoundary: true)
        guard case .hit(let matched, _, let detail, _, _, _) = probe else {
            return XCTFail("stripped boundary was not stored or is not matchable — fetch missed")
        }
        XCTAssertEqual(matched, turn1User.count, "match must be the stripped prefix (detail \(detail))")

        // Turn 2 continues the conversation: the same user turn, then the
        // assistant's reply, then a new user turn and a fresh generation
        // prompt. Everything up to `turn1User.count` is shared.
        target.resetRecording()
        let turn2 = turn1User + turn2Extra + genPromptSuffix
        _ = try DFlash2TokenIterator(
            input: input(turn2),
            target: target,
            drafter: drafter,
            blockSize: nil,
            parameters: parameters(),
            cacheCoordinator: coordinator)

        XCTAssertLessThan(
            target.forwarded.count, turn2.count,
            """
            Turn 2 re-prefilled all \(turn2.count) tokens — the stored prefix was never \
            matched, so DFlash 2 has disabled cross-turn cache reuse. Forwarded: \
            \(target.forwarded)
            """)
        XCTAssertEqual(
            Array(target.forwarded.suffix(genPromptSuffix.count)), genPromptSuffix,
            "the tail actually forwarded must end at the new generation prompt")
    }

    /// Reasoning effort is part of the cache key (the host stamps it into
    /// `modelKey`). A different effort must NOT restore the other
    /// effort's state — that would silently answer turn 2 from a prefix
    /// built under different instructions.
    func testDifferentReasoningEffortDoesNotReuseTheOtherKeysPrefix() throws {
        let lock = lockSerializedMLXTest()
        defer { lock.unlock() }

        let target = RecordingDFlash2Target(hidden: hiddenSize, vocab: vocabSize)
        let drafter = try makeDrafter()
        let turn1 = turn1User + genPromptSuffix

        let low = makeCoordinator(modelKey: "recording-dflash2|reasoning=low")
        var first = try DFlash2TokenIterator(
            input: input(turn1), target: target, drafter: drafter, blockSize: nil,
            parameters: parameters(), cacheCoordinator: low)
        _ = first.next()
        first.storeCacheAfterGeneration(
            generatedTokenIds: [], includeGeneratedBoundary: false)

        // Same prompt, different effort ⇒ different coordinator scope.
        target.resetRecording()
        let high = makeCoordinator(modelKey: "recording-dflash2|reasoning=high")
        _ = try DFlash2TokenIterator(
            input: input(turn1), target: target, drafter: drafter, blockSize: nil,
            parameters: parameters(), cacheCoordinator: high)

        XCTAssertEqual(
            target.forwarded, turn1,
            "a different reasoning-effort scope must prefill from scratch, not inherit the other scope's cache"
        )
    }

    /// The osaurus live-gate crash recipe: a SAMPLED (non-greedy) dFlash-2
    /// turn generates, stores its cache, and a follow-on request on the
    /// same coordinator builds a second iterator. The app died inside the
    /// second `DFlash2TokenIterator.init` with an `MLXArray.item`
    /// precondition (2026-08-20). Greedy variants of this flow are covered
    /// above and never crashed — the bundle's real defaults
    /// (temp 1.0, top_p 0.95, top_k 20) are what users actually run.
    func testSampledTurnThenRestoreDoesNotCrash() throws {
        let lock = lockSerializedMLXTest()
        defer { lock.unlock() }

        let target = RecordingDFlash2Target(hidden: hiddenSize, vocab: vocabSize)
        let drafter = try makeDrafter()
        let coordinator = makeCoordinator()

        var sampled = GenerateParameters(maxTokens: 6, temperature: 1.0)
        sampled.topP = 0.95
        sampled.topK = 20
        sampled.prefillStepSize = 4

        // Turn 1: sampled generation, then store.
        let turn1 = turn1User + genPromptSuffix
        var first = try DFlash2TokenIterator(
            input: input(turn1), target: target, drafter: drafter, blockSize: nil,
            parameters: sampled, cacheCoordinator: coordinator)
        var generated: [Int] = []
        while let token = first.next(), generated.count < 6 {
            generated.append(token)
        }
        first.storeCacheAfterGeneration(
            generatedTokenIds: generated, includeGeneratedBoundary: false)

        // Follow-on request (the app's auto-title or next turn): same
        // coordinator, extended prompt, same sampled params. Must build —
        // a poisoned stored entry has to read as a miss, never crash.
        let turn2 = turn1User + generated + turn2Extra + genPromptSuffix
        var second = try DFlash2TokenIterator(
            input: input(turn2), target: target, drafter: drafter, blockSize: nil,
            parameters: sampled, cacheCoordinator: coordinator)
        XCTAssertNotNil(second.next(), "follow-on turn must produce a token")
    }
}
