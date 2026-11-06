import Foundation
import MLX
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

/// Ripple's sliding-layer cache: rows are tagged with their absolute position, so every test can assert which positions
/// a read returns, at lengths that do not divide the 256-row growth step or the window.
@Suite struct RicursiveSlidingKVCacheTests {
    private let window = 32

    private func rows(_ positions: Range<Int>) -> (MLXArray, MLXArray) {
        let tags = MLXArray(positions.map { Float($0) }).reshaped(1, 1, positions.count, 1)
        return (tags, tags + 0.5)
    }

    private func positions(_ array: MLXArray) -> [Int] {
        array.reshaped(-1).asArray(Float.self).map { Int($0) }
    }

    private func filled(_ count: Int, chunk: Int = 7) -> RicursiveSlidingKVCache {
        let cache = RicursiveSlidingKVCache(window: window)
        var at = 0
        while at < count {
            let end = min(at + chunk, count)
            let (k, v) = rows(at ..< end)
            _ = cache.update(keys: k, values: v)
            at = end
        }
        return cache
    }

    @Test func updateReturnsAllHeldRowsInTemporalOrder() {
        let cache = filled(300)
        let (k, v) = rows(300 ..< 303)
        let (keys, values) = cache.update(keys: k, values: v)
        #expect(positions(keys) == Array(0 ..< 303))
        #expect(positions(values) == Array(0 ..< 303))
        #expect(cache.offset == 303)
    }

    @Test func stateHoldsOnlyTheWindowPlusRollbackRows() {
        let cache = filled(301)
        let stored = window - 1 + RicursiveSlidingKVCache.rollbackRows
        #expect(positions(cache.state[0]) == Array((301 - stored) ..< 301))
        #expect(cache.diskCacheStateIdentifier.contains("\(window)"))
    }

    @Test func restoredCacheContinuesAtItsLogicalPosition() {
        let source = filled(301)
        let restored = RicursiveSlidingKVCache(window: window)
        #expect(restored.restoreDiskCacheState(source.state, metadata: source.metaState, offset: 301))
        #expect(restored.offset == 301)
        let (k, v) = rows(301 ..< 305)
        let (keys, _) = restored.update(keys: k, values: v)
        let stored = window - 1 + RicursiveSlidingKVCache.rollbackRows
        #expect(positions(keys) == Array((301 - stored) ..< 305))
    }

    @Test func rollbackPastTheWindowKeepsTheWindowExact() {
        let restored = RicursiveSlidingKVCache(window: window)
        let source = filled(301)
        #expect(restored.restoreDiskCacheState(source.state, metadata: source.metaState, offset: 301))
        let (block, blockValues) = rows(301 ..< 317)
        _ = restored.update(keys: block, values: blockValues)
        #expect(restored.trim(11) == 11)
        #expect(restored.offset == 306)
        let (k, v) = rows(9306 ..< 9307)
        let (keys, _) = restored.update(keys: k, values: v)
        let read = positions(keys)
        #expect(read.last == 9306)
        #expect(Array(read.dropLast().suffix(window - 1)) == Array((306 - (window - 1)) ..< 306))
    }

    @Test func trimNeverShortensTheWindow() {
        let restored = RicursiveSlidingKVCache(window: window)
        let source = filled(301)
        #expect(restored.restoreDiskCacheState(source.state, metadata: source.metaState, offset: 301))
        #expect(restored.trim(RicursiveSlidingKVCache.rollbackRows + 50) == RicursiveSlidingKVCache.rollbackRows)
        #expect(restored.offset == 301 - RicursiveSlidingKVCache.rollbackRows)
    }

    @Test func coldCacheTrimsAnywhere() {
        let cache = filled(301)
        #expect(cache.trim(250) == 250)
        #expect(cache.offset == 51)
    }

    @Test func restoreRefusesAnEntryWithoutItsWindow() {
        let short = RicursiveSlidingKVCache(window: window)
        let (k, v) = rows(0 ..< (window - 2))
        #expect(!short.restoreDiskCacheState([k, v], metadata: short.metaState, offset: 301))
        let id = short.diskCacheStateIdentifier
        #expect(!short.restoreDiskCacheState([k, v], metadata: [id, "301"], offset: 301))
        #expect(!short.restoreDiskCacheState([k, v], metadata: ["other", "301"], offset: 301))
        #expect(short.offset == 0)
    }

    @Test func prefixViewSeesTheRowsBeforeItsBoundary() {
        let cache = filled(301)
        let view = cache.prefixView(rows: 200)
        #expect(view.offset == 200)
        let stored = window - 1 + RicursiveSlidingKVCache.rollbackRows
        #expect(positions(view.state[0]) == Array((200 - stored) ..< 200))
        let (k, v) = rows(301 ..< 302)
        _ = cache.update(keys: k, values: v)
        #expect(positions(view.state[0]).last == 199)
    }

    @Test func copyIsIndependentOfLaterWrites() {
        let cache = filled(301)
        let copy = cache.copy()
        let (k, v) = rows(301 ..< 302)
        _ = cache.update(keys: k, values: v)
        #expect(copy.offset == 301)
        #expect(positions(copy.state[0]).last == 300)
    }
}

/// Ripple's separator contract: parsed text re-rendered by the template (`<think>\n{r}\n</think>\n{c}`,
/// `{c}\n<tool_call>`) reproduces the generated text, for every chunk split.
@Suite struct RicursiveReasoningSeparatorTests {
    private func parse(_ text: String, chunk: Int) -> (reasoning: String, content: String) {
        var parser = ReasoningParser.forPrompt(stampName: "ricursive", promptTail: "<|im_start|>assistant\n")!
        var reasoning = ""
        var content = ""
        func take(_ segments: [ReasoningSegment]) {
            for segment in segments {
                switch segment {
                case .reasoning(let text): reasoning += text
                case .content(let text): content += text
                }
            }
        }
        var rest = Substring(text)
        while !rest.isEmpty {
            take(parser.feed(String(rest.prefix(chunk))))
            rest = rest.dropFirst(chunk)
        }
        take(parser.flush())
        return (reasoning, content)
    }

    @Test(arguments: [1, 2, 3, 5, 7, 8, 9, 11, 64])
    func splitsWithoutSeparators(chunk: Int) {
        let answer = parse("<think>\nline one\n\nline two\n</think>\nThe answer\nis 4.", chunk: chunk)
        #expect(answer.reasoning == "line one\n\nline two")
        #expect(answer.content == "The answer\nis 4.")
        let spaced = parse("<think>\nr\n</think>\n\nAnswer", chunk: chunk)
        #expect(spaced.reasoning == "r")
        #expect(spaced.content == "\nAnswer")
        let plain = parse("No reasoning here.\nSecond line\n", chunk: chunk)
        #expect(plain.reasoning == "")
        #expect(plain.content == "No reasoning here.\nSecond line\n")
    }

    @Test(arguments: [1, 2, 3, 5, 10, 11, 12, 13, 64])
    func toolCallKeepsItsPayloadAndDropsTheTemplateNewline(chunk: Int) {
        let call = "<tool_call>\n{\"name\": \"f\", \"arguments\": {\"s\": \"<think>\\n\"}}\n</tool_call>"
        let bare = parse("<think>\nr\n</think>\n\n" + call, chunk: chunk)
        #expect(bare.reasoning == "r")
        #expect(bare.content == call)
        let text = parse("Checking.\n" + call + "\n" + call, chunk: chunk)
        #expect(text.content == "Checking." + call + call)
    }

    @Test func stampFromModelTypeAndBundleStampResolve() {
        #expect(reasoningStampFromModelType("ricursive") == "ricursive")
        #expect(ReasoningParser.namesAKnownFamily("ricursive"))
    }
}
