import Foundation
import Testing

@Suite("ChatSession runtime wiring source coverage")
struct ChatSessionRuntimeWiringSourceCoverageTests {
    private static func source(_ relativePath: String) throws -> String {
        let repoRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        return try String(
            contentsOf: repoRoot.appending(path: relativePath),
            encoding: .utf8)
    }

    @Test("streamMap passes container cache coordinator into TokenIterator")
    func streamMapPassesCacheCoordinatorIntoTokenIterator() throws {
        let source = try Self.source("Libraries/MLXLMCommon/ChatSession.swift")
        #expect(source.contains("let cacheCoordinator = model.cacheCoordinator"))
        #expect(source.contains("cacheCoordinator: cacheCoordinator"))
        #expect(source.contains("let iterator = try TokenIterator("))
    }

    /// The branch this pins is the one the bug was: `ChatSession` never consulted
    /// `draftStrategy`, so a caller that set it got plain autoregressive decode with no error and
    /// no log line. Deleting the `else if` restores that silence — and nothing else in-tree would
    /// fail — so this test exists to make the branch's absence loud.
    ///
    /// Source-level rather than behavioural because exercising the dispatch needs a real bundle
    /// with MTP tensors; this pins the wiring, and the guard below pins the safety condition.
    @Test("streamMap dispatches native MTP when draftStrategy requests it")
    func streamMapDispatchesNativeMTPForDraftStrategy() throws {
        let source = try Self.source("Libraries/MLXLMCommon/ChatSession.swift")
        #expect(source.contains("generateParameters.draftStrategy"))
        #expect(source.contains("case .nativeMTP(depth: let depth, verifierMode: _)"))
        #expect(source.contains("let iterator = try NativeMTPTokenIterator("))
    }

    /// Speculation must stay gated on `canUseNativeMTP(for:)` — greedy-only, no media content,
    /// unbounded KV. That predicate is what makes "this cannot change what text is produced" true;
    /// dispatching without it would turn a performance path into a correctness change.
    @Test("native MTP dispatch stays gated on canUseNativeMTP")
    func nativeMTPDispatchIsGated() throws {
        let source = try Self.source("Libraries/MLXLMCommon/ChatSession.swift")
        #expect(source.contains("generateParameters.canUseNativeMTP(for: input)"))
        #expect(source.contains("NativeMTPRuntimeError.modelDoesNotExposeNativeMTP"))
    }

    @Test("streamMap consumer termination cancels the producer task")
    func streamMapTerminationCancelsProducerTask() throws {
        let source = try Self.source("Libraries/MLXLMCommon/ChatSession.swift")
        #expect(source.contains("continuation.onTermination = { _ in"))
        #expect(source.contains("task.cancel()"))
        #expect(source.contains("await task.value"))
    }
}
