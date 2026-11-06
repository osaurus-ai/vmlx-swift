import Testing

@testable import MLXLMCommon

@Suite("Native forwarded stop persistence key")
struct NativeForwardedStopTailTests {
    private func tail(
        generated: [Int] = [10, 11, 12], returned: Int = 4,
        last: Int? = 90, pending: [Int] = [12, 90, 91, 92], index: Int = 2,
        offsets: [Int] = [10, 10], prompt: Int = 5
    ) -> [Int]? {
        NativeMTPTokenIterator.verifiedForwardedStopTail(
            promptTokenCount: prompt, generatedTokenIds: generated,
            returnedTokenCount: returned, lastReturnedTokenId: last,
            pendingTokens: pending, pendingIndex: index, leafOffsets: offsets)
    }

    @Test(
        "recover accepted stop tails at every position, excluding the bonus",
        arguments: 2 ... 8)
    func acceptedQueue(length: Int) {
        let queue = Array(90 ..< 90 + length) + [999]
        for stopIndex in 0 ..< length - 1 {
            let generated = [10, 11] + Array(queue.prefix(stopIndex))
            let result = tail(
                generated: generated, returned: generated.count + 1,
                last: queue[stopIndex], pending: queue, index: stopIndex + 1,
                offsets: [7 + length, 7 + length])
            #expect(result == Array(queue[stopIndex ..< length]))
            #expect(result?.contains(999) == false)
        }
    }

    @Test("mismatched and unforwarded queues fail closed")
    func invalidAlignment() {
        #expect(tail() == [90, 91])
        #expect(tail(offsets: []) == nil)
        #expect(tail(offsets: [10, 9]) == nil)
        #expect(tail(offsets: [8, 8]) == nil)  // exact boundary: existing path
        #expect(tail(offsets: [9, 9]) == nil)  // one stop: existing path
        #expect(tail(offsets: [11, 11]) == nil)  // would include bonus
        #expect(tail(pending: [12, 90, 91, 93, 92]) == nil)  // unaccounted row
        #expect(tail(returned: 3) == nil)
        #expect(tail(returned: 5) == nil)
        #expect(tail(last: nil) == nil)
        #expect(tail(last: 89) == nil)
        #expect(tail(index: 0) == nil)
        #expect(tail(index: 5) == nil)
        #expect(tail(pending: []) == nil)
        #expect(tail(pending: [13, 90, 91, 92]) == nil)
        #expect(tail(generated: [], returned: 1) == nil)
        #expect(tail(prompt: -1) == nil)
    }
}
