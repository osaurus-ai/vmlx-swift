@testable import MLX
import Dispatch
import Metal
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

    // A tensor with no elements has no backing buffer, so its data pointer is
    // null, while its strides need not be zero.

    @Test("asArray and asArrayChecked read [1, 0, 3, 448] as []")
    func emptyRank4HostRead() throws {
        let floats = MLXArray([Float](), [1, 0, 3, 448])
        let ints = MLXArray([Int32](), [1, 0, 3, 448])
        // The case that trapped: strides that span 3 * 448 elements, which a
        // size taken from the strides paired with the null pointer.
        #expect(floats.internalStrides == [0, 3 * 448, 448, 1])
        #expect(ints.internalStrides == [0, 3 * 448, 448, 1])

        try expectEmptyHostReads(floats, as: Float.self)
        try expectEmptyHostReads(ints, as: Int32.self)
    }

    @Test("asArray and asArrayChecked read [0] as []")
    func emptyVectorHostRead() throws {
        try expectEmptyHostReads(MLXArray([Float](), [0]), as: Float.self)
        try expectEmptyHostReads(MLXArray([Int32](), [0]), as: Int32.self)
    }

    @Test("asArray and asArrayChecked read a transposed [2, 0] as []")
    func emptyTransposedHostRead() throws {
        // The transpose's strides are not row-contiguous, so the copy takes
        // its element-by-element path.
        let floats = MLXArray([Float](), [2, 0]).T
        let ints = MLXArray([Int32](), [2, 0]).T
        #expect(floats.shape == [0, 2])

        try expectEmptyHostReads(floats, as: Float.self)
        try expectEmptyHostReads(ints, as: Int32.self)
    }

    /// Both host-read surfaces, in the array's own dtype so that no
    /// conversion runs first.
    private func expectEmptyHostReads<T: HasDType>(_ array: MLXArray, as type: T.Type) throws {
        #expect(array.dtype == T.dtype)
        #expect(array.size == 0)
        #expect(array.asArray(type).isEmpty)
        #expect(try array.asArrayChecked(type).isEmpty)
    }

    // `asMTLBuffer` has no bytes to wrap or copy for an array with no
    // elements either: its no-copy path unwrapped the null data pointer.

    @Test("asMTLBuffer returns nil for [0], with and without a copy")
    func emptyVectorMTLBuffer() throws {
        try expectNoMTLBuffer(MLXArray([Float](), [0]))
    }

    @Test("asMTLBuffer returns nil for [1, 0, 3, 448], with and without a copy")
    func emptyRank4MTLBuffer() throws {
        try expectNoMTLBuffer(MLXArray([Float](), [1, 0, 3, 448]))
    }

    /// The copy first, so that a trap in the no-copy path comes after the
    /// copy's result is recorded.
    private func expectNoMTLBuffer(_ array: MLXArray) throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        #expect(array.size == 0)
        #expect(array.contiguousToDimension() == 0)
        #expect(array.asMTLBuffer(device: device, noCopy: false) == nil)
        #expect(array.asMTLBuffer(device: device, noCopy: true) == nil)
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
