import Foundation
import MLX
import MLXLMCommon
import XCTest

final class FinalFlushStopReasonTests: XCTestCase {
    private enum End: Sendable { case cap, cancelled, constraint, mlxError }
    // A single short decoded piece stays in the detokenizer holdback, so
    // STOP is first observed by the matcher during the final flush.
    private struct Pieces: Tokenizer {
        var bosToken: String? { nil }
        var eosToken: String? { nil }
        var unknownToken: String? { nil }
        func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }
        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
            tokenIds.map { _ in "aSTOP" }.joined()
        }
        func convertTokenToId(_ token: String) -> Int? { nil }
        func convertIdToToken(_ id: Int) -> String? { "aSTOP" }
        func applyChatTemplate(
            messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?
        ) throws -> [Int] { [] }
    }
    private struct Iterator: TokenIteratorProtocol {
        let end: End
        var tokenCount = 0
        var generationFailure: GenerationFailure?
        var maxTokens: Int? {
            if case .constraint = end { return 2 }
            return 1
        }
        var promptPrefillTime: TimeInterval { 0 }
        mutating func next() -> Int? {
            if tokenCount == 0 {
                tokenCount = 1
                return 0
            }
            switch end {
            case .cap: break
            case .cancelled: withUnsafeCurrentTask { $0?.cancel() }
            case .constraint:
                generationFailure = GenerationFailure(
                    stage: .decoding, cause: "synthetic constraint failure")
            case .mlxError:
                // Invalid matmul dimensions report through the real MLX.withError
                // boundary in generateLoopTask, without loading a model.
                _ = MLX.matmul(MLXArray.zeros([1, 2]), MLXArray.zeros([3, 1]))
            }
            return nil
        }
    }
    private func run(_ end: End, stops: [String]) async throws -> (String, [GenerateCompletionInfo])
    {
        return try await FocusedMLXTestSupport.withLock {
            let (stream, task) = generateTask(
                promptTokenCount: 3,
                modelConfiguration: ModelConfiguration(id: "synthetic-stop-tail"),
                tokenizer: Pieces(), iterator: Iterator(end: end),
                extraStopStrings: stops, promptTail: "")
            var text = ""
            var infos = [GenerateCompletionInfo]()
            for await event in stream {
                switch event {
                case .chunk(let chunk): text += chunk
                case .info(let info): infos.append(info)
                default: break
                }
            }
            await task.value
            return (text, infos)
        }
    }
    func testStopDiscoveredOnlyAtFinalFlushAtCap() async throws {
        let (text, infos) = try await run(.cap, stops: ["STOP"])
        XCTAssertEqual(text, "a")
        XCTAssertEqual(infos.count, 1)
        let info = try XCTUnwrap(infos.first)
        XCTAssertEqual(info.stopReason, .stop)
        XCTAssertNil(info.generationFailure)
        XCTAssertEqual(info.promptTokenCount, 3)
        XCTAssertEqual(info.generationTokenCount, 1)
    }
    func testNoStopAtCapKeepsLengthAndCompleteText() async throws {
        let (text, infos) = try await run(.cap, stops: [])
        XCTAssertEqual(text, "aSTOP")
        XCTAssertEqual(infos.count, 1)
        let info = try XCTUnwrap(infos.first)
        XCTAssertEqual(info.stopReason, .length)
        XCTAssertNil(info.generationFailure)
        XCTAssertEqual(info.generationTokenCount, 1)
    }
    func testCancellationIsNotRelabeledByFlushedStop() async throws {
        let (text, infos) = try await run(.cancelled, stops: ["STOP"])
        XCTAssertEqual(text, "a")
        XCTAssertEqual(infos.count, 1)
        let info = try XCTUnwrap(infos.first)
        XCTAssertEqual(info.stopReason, .cancelled)
        XCTAssertNil(info.generationFailure)
        XCTAssertEqual(info.generationTokenCount, 1)
    }
    func testConstraintFailureIsNotRelabeledByFlushedStop() async throws {
        let (text, infos) = try await run(.constraint, stops: ["STOP"])
        XCTAssertEqual(text, "a")
        XCTAssertEqual(infos.count, 1)
        let info = try XCTUnwrap(infos.first)
        XCTAssertEqual(info.stopReason, .cancelled)
        XCTAssertEqual(info.generationFailure?.cause, "synthetic constraint failure")
        XCTAssertEqual(info.generationTokenCount, 1)
    }
    func testMLXDecodeFailureIsNotRelabeledByFlushedStop() async throws {
        let (text, infos) = try await run(.mlxError, stops: ["STOP"])
        XCTAssertEqual(text, "a")
        XCTAssertEqual(infos.count, 1)
        let info = try XCTUnwrap(infos.first)
        XCTAssertEqual(info.stopReason, .cancelled)
        XCTAssertNotNil(info.generationFailure)
        XCTAssertEqual(info.generationFailure?.stage, .decoding)
        XCTAssertEqual(info.generationTokenCount, 1)
    }
}
