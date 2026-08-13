import Foundation
import MLX
import Testing

@testable import MLXLMCommon

/// The turn-start ("generation-suffix stripped") boundary is the only cross-turn
/// checkpoint a chat prompt is guaranteed to contain: the next turn replaces the
/// trailing generation prompt with the assistant's reply, so the full-prompt key
/// never matches again, but everything before the last turn-start token does.
///
/// Capturing it used to be gated on `coordinator.isHybrid`. That was right while
/// capture meant a retained cache copy or a replayed prefill — only SSM hybrids
/// earned it back. It is wrong now that capture is prefill-time, because the
/// premise behind excluding everyone else ("dense/rotating models reuse via the
/// post-answer boundary") does not hold for reasoning models: hosts strip think
/// blocks when re-rendering history, so the post-answer snapshot can never be a
/// prefix of the next prompt. Measured on DSV4 — `HIT disk boundary=3464
/// remaining=1367` then `MISS all tiers tokens=2644`, i.e. every send
/// re-prefilled the whole previous reply.
///
/// `hybridStripBoundaryIndex` had no coverage at all before this file, so the
/// gate could have been widened or narrowed silently.
@Suite("Turn-start boundary capture is not hybrid-only")
struct TurnStartBoundaryWideningTests {

    /// Qwen-ish turn-start token, matching what `genPromptSuffixTokens.first`
    /// carries for a chat template.
    private static let turnStart = 151_644

    private func makeCoordinator(
        hybrid: Bool,
        disk: Bool = true,
        paged: Bool = false
    ) -> CacheCoordinator {
        var cfg = CacheCoordinatorConfig()
        cfg.usePagedCache = paged
        cfg.enableDiskCache = disk
        cfg.modelKey = "turn-start-boundary-test"
        cfg.diskCacheDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("vmlx_turnstart_test_\(UUID().uuidString)")
        let coordinator = CacheCoordinator(config: cfg)
        coordinator.setHybrid(hybrid)
        coordinator.setGenPromptSuffixTokens([Self.turnStart])
        return coordinator
    }

    /// A chat prompt whose LAST turn-start token sits at index 4.
    private func chatPrompt() -> (tokens: [Int], boundary: Int) {
        let tokens = [1, 2, Self.turnStart, 3, Self.turnStart, 5, 6, 7]
        return (tokens, 4)
    }

    private func input(_ tokens: [Int]) -> LMInput {
        LMInput(tokens: MLXArray(tokens.map { Int32($0) }), tokenIds: tokens)
    }

    // MARK: - The widening

    /// The regression this file exists for. Before the change this returned nil
    /// for every non-SSM model — DSV4 hybrid-pool, Gemma-4 rotating, Qwen dense
    /// — which is precisely the set that pays the multi-turn re-prefill tax.
    @Test("a non-hybrid coordinator now yields the turn-start boundary")
    func nonHybridYieldsBoundary() {
        let (tokens, boundary) = chatPrompt()
        let resolved = TokenIterator.hybridStripBoundaryIndex(
            coordinator: makeCoordinator(hybrid: false),
            promptTokenIds: tokens,
            input: input(tokens))

        #expect(
            resolved == boundary,
            """
            a non-hybrid chat prompt lost its only cross-turn checkpoint, so the \
            next turn re-prefills the entire previous assistant reply
            """)
    }

    /// The behaviour that already worked must be untouched.
    @Test("a hybrid coordinator still yields the same boundary")
    func hybridStillYieldsBoundary() {
        let (tokens, boundary) = chatPrompt()
        #expect(
            TokenIterator.hybridStripBoundaryIndex(
                coordinator: makeCoordinator(hybrid: true),
                promptTokenIds: tokens,
                input: input(tokens)) == boundary)
    }

    /// Hybrid and non-hybrid must agree — the boundary is a property of the
    /// prompt's shape, never of the cache topology.
    @Test("the boundary does not depend on the topology")
    func boundaryIsTopologyIndependent() {
        let (tokens, _) = chatPrompt()
        let asHybrid = TokenIterator.hybridStripBoundaryIndex(
            coordinator: makeCoordinator(hybrid: true),
            promptTokenIds: tokens, input: input(tokens))
        let asDense = TokenIterator.hybridStripBoundaryIndex(
            coordinator: makeCoordinator(hybrid: false),
            promptTokenIds: tokens, input: input(tokens))
        #expect(asHybrid == asDense)
    }

    // MARK: - Guards that must survive the widening

    /// With no tier to land in, the store is discarded, so the caller must not
    /// pay to produce the snapshot. Widening the topology gate must not widen
    /// this one.
    @Test("no persistable tier still yields nothing")
    func noTierYieldsNil() {
        let (tokens, _) = chatPrompt()
        for hybrid in [true, false] {
            #expect(
                TokenIterator.hybridStripBoundaryIndex(
                    coordinator: makeCoordinator(hybrid: hybrid, disk: false, paged: false),
                    promptTokenIds: tokens,
                    input: input(tokens)) == nil)
        }
    }

    /// A raw / non-chat prompt has no turn-start token, so there is no proven
    /// boundary to store and the existing whole-prompt policy stands.
    @Test("a prompt with no turn-start token yields nothing")
    func noTurnStartYieldsNil() {
        let tokens = [1, 2, 3, 4, 5]
        #expect(
            TokenIterator.hybridStripBoundaryIndex(
                coordinator: makeCoordinator(hybrid: false),
                promptTokenIds: tokens,
                input: input(tokens)) == nil)
    }

    /// A turn-start token at index 0 would strip the whole prompt to nothing.
    @Test("a leading turn-start token yields nothing")
    func leadingTurnStartYieldsNil() {
        let tokens = [Self.turnStart, 1, 2, 3]
        #expect(
            TokenIterator.hybridStripBoundaryIndex(
                coordinator: makeCoordinator(hybrid: false),
                promptTokenIds: tokens,
                input: input(tokens)) == nil)
    }

    /// A reusable-prefix warmup publishes the N-1 disk seed instead; capturing
    /// the strip boundary there would store a prefix of a prompt no visible
    /// request will ever render.
    @Test("a reusable-prefix warmup yields nothing")
    func warmupYieldsNil() {
        let (tokens, _) = chatPrompt()
        let warmup = LMInput(
            tokens: MLXArray(tokens.map { Int32($0) }),
            tokenIds: tokens,
            cachePromptIntent: .reusablePrefixWarmup)
        for hybrid in [true, false] {
            #expect(
                TokenIterator.hybridStripBoundaryIndex(
                    coordinator: makeCoordinator(hybrid: hybrid),
                    promptTokenIds: tokens,
                    input: warmup) == nil)
        }
    }

    /// The engines used to each carry their own copy of this predicate, and the
    /// copies disagreed: the batched and MTP engines rejected `stripAt ==
    /// count - 1` while the solo iterator accepted it. A chat template whose
    /// generation prompt is a single token lands exactly there, so the same
    /// prompt got a cross-turn checkpoint on one engine and none on the other
    /// two. All three now call this function, so pinning it pins all of them.
    @Test("a single-token generation prompt still yields its boundary")
    func singleTokenGenerationPromptYieldsBoundary() {
        let tokens = [1, 2, 3, Self.turnStart]
        #expect(
            TokenIterator.hybridStripBoundaryIndex(
                coordinator: makeCoordinator(hybrid: false),
                promptTokenIds: tokens,
                input: input(tokens)) == 3)
    }

    /// No coordinator, no store.
    @Test("no coordinator yields nothing")
    func noCoordinatorYieldsNil() {
        let (tokens, _) = chatPrompt()
        #expect(
            TokenIterator.hybridStripBoundaryIndex(
                coordinator: CacheCoordinator?.none,
                promptTokenIds: tokens,
                input: input(tokens)) == nil)
    }

    // MARK: - The predicate itself

    /// `capturesTurnStartBoundary` must track "is there a tier to land in", for
    /// hybrid and non-hybrid alike — that equivalence is the whole change.
    @Test("capturesTurnStartBoundary tracks tier availability, not topology")
    func predicateTracksTiers() {
        for hybrid in [true, false] {
            let withDisk = makeCoordinator(hybrid: hybrid, disk: true)
            #expect(withDisk.capturesTurnStartBoundary)
            #expect(withDisk.capturesTurnStartBoundary == withDisk.canPersistBoundaries)

            let withoutTiers = makeCoordinator(hybrid: hybrid, disk: false, paged: false)
            #expect(!withoutTiers.capturesTurnStartBoundary)
        }
    }

    /// A paged-incompatible model (DSV4) has its paged tier bypassed, so only
    /// the disk tier can hold the boundary — and it must still be captured,
    /// since DSV4 is the family the widening was measured on.
    @Test("a paged-incompatible model still captures via disk")
    func pagedIncompatibleStillCaptures() {
        let coordinator = makeCoordinator(hybrid: false, disk: true, paged: true)
        coordinator.setPagedIncompatible(true)
        #expect(coordinator.capturesTurnStartBoundary)

        let (tokens, boundary) = chatPrompt()
        #expect(
            TokenIterator.hybridStripBoundaryIndex(
                coordinator: coordinator,
                promptTokenIds: tokens,
                input: input(tokens)) == boundary)
    }
}
