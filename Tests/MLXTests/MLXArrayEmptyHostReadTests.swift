@testable import MLX
import Dispatch
import Testing

@Suite("MLXArray empty host reads")
struct MLXArrayEmptyHostReadTests {
    @Test("asArray returns an empty Swift array for a zero-element tensor")
    func emptyAsArray() {
        let empty = MLXArray.zeros([1, 0], type: Int32.self)

        #expect(empty.asArray(Int32.self).isEmpty)
    }

    @Test("asArray returns empty for a zero-length lazy slice")
    func emptySliceAsArray() {
        let tokens = MLXArray([Int32(11), Int32(22)]).reshaped(1, 2)
        let emptySuffix = tokens[0..., 2 ..< 2]

        #expect(emptySuffix.shape == [1, 0])
        #expect(emptySuffix.asArray(Int32.self).isEmpty)
    }

    @Test("evaluated host access holds the stream lock through the read")
    func hostAccessLockScope() {
        nonisolated(unsafe) let array = MLXArray([Int32(7)])
        let enteredRead = DispatchSemaphore(value: 0)
        let releaseRead = DispatchSemaphore(value: 0)
        let readFinished = DispatchSemaphore(value: 0)
        let competingLockAcquired = DispatchSemaphore(value: 0)

        DispatchQueue.global().async {
            array.withEvaluatedHostAccess { _ in
                enteredRead.signal()
                releaseRead.wait()
            }
            readFinished.signal()
        }
        #expect(enteredRead.wait(timeout: .now() + 2) == .success)

        DispatchQueue.global().async {
            MLXArray.withEvalLockForTesting {
                competingLockAcquired.signal()
            }
        }
        #expect(competingLockAcquired.wait(timeout: .now() + 0.05) == .timedOut)

        releaseRead.signal()
        #expect(readFinished.wait(timeout: .now() + 2) == .success)
        #expect(competingLockAcquired.wait(timeout: .now() + 2) == .success)
    }

    @Test("host reads stay valid during concurrent stream maintenance")
    func concurrentHostReads() async {
        await withTaskGroup(of: Bool.self) { group in
            for reader in 0 ..< 8 {
                group.addTask {
                    for iteration in 0 ..< 2_000 {
                        let expected = Int32(reader + iteration)
                        let lazy = MLXArray(Int32(reader)) + MLXArray(Int32(iteration))
                        guard lazy.asArray(Int32.self) == [expected] else { return false }
                    }
                    return true
                }
            }
            group.addTask {
                for _ in 0 ..< 2_000 {
                    Stream.gpu.synchronize()
                    Memory.clearCache()
                }
                return true
            }

            for await passed in group {
                #expect(passed)
            }
        }
    }
}
