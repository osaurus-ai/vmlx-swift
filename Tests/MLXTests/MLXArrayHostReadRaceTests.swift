import Dispatch
import Foundation
import Metal
import Testing

@testable import MLX

/// Host reads of an array that another thread updates in place.
///
/// `_updateInternal` replaces the C++ array behind `ctx` in place, under
/// `evalLock`, and so frees the old array's description and its buffer. A host
/// read that touches `self` before it takes `evalLock`, or after it lets go
/// (its dtype, a conversion with `asType`, its scalar or its bytes), can read
/// them while the writer frees them.
///
/// Each test runs one writer that replaces the array, again and again, with a
/// fresh one whose elements all equal a counter, and three readers that read it
/// through one surface, for a second. Every read must see one whole array:
/// the right number of elements, all equal, inside the counter's range. A torn
/// or freed read fails that check or crashes the process.
@Suite("MLXArray host reads during in-place updates", .serialized)
struct MLXArrayHostReadRaceTests {

    private final class Box: @unchecked Sendable {
        let array: MLXArray
        init(_ array: MLXArray) { self.array = array }
    }

    private final class Tally: @unchecked Sendable {
        private let lock = NSLock()
        private(set) var reads = 0
        private(set) var bad = 0
        private(set) var firstBad: [Int] = []

        func record(_ values: [Int], count: Int) {
            let whole =
                values.count == count && values.allSatisfy { $0 == values[0] }
                && (0 ..< MLXArrayHostReadRaceTests.limit).contains(values[0])
            lock.withLock {
                reads += 1
                if !whole {
                    bad += 1
                    if firstBad.isEmpty { firstBad = Array(values.prefix(8)) }
                }
            }
        }
    }

    /// Counter values stay below 2^20, so a float32 conversion is exact.
    static let limit = 1 << 20
    static let seconds = 1.0

    /// Race `read` on three threads against a writer, for `seconds`, on an
    /// int32 array of `count` elements. `read` returns the elements it saw.
    private func race(
        count: Int = 4096, _ read: @escaping @Sendable (MLXArray) throws -> [Int]
    ) {
        let savedLimit = Memory.cacheLimit
        Memory.cacheLimit = 0
        defer { Memory.cacheLimit = savedLimit }

        let box = Box(MLXArray(Array(repeating: Int32(0), count: count)))
        let tally = Tally()
        let deadline = Date().addingTimeInterval(Self.seconds)
        let group = DispatchGroup()

        // Writer: no conversion and no graph, only fresh arrays swapped in, so
        // that every swap frees the previous array's description and buffer.
        group.enter()
        DispatchQueue.global(qos: .userInitiated).async {
            var counter: Int32 = 0
            while Date() < deadline {
                counter = (counter + 1) & Int32(Self.limit - 1)
                box.array._updateInternal(MLXArray(Array(repeating: counter, count: count)))
            }
            group.leave()
        }

        for _ in 0 ..< 3 {
            group.enter()
            DispatchQueue.global(qos: .userInitiated).async {
                while Date() < deadline {
                    tally.record((try? read(box.array)) ?? [], count: count)
                }
                group.leave()
            }
        }

        group.wait()
        #expect(tally.reads > 0)
        #expect(tally.bad == 0, "first torn read starts \(tally.firstBad)")
    }

    @Test("asArray with a dtype conversion reads one whole array")
    func asArrayConvertingRace() {
        race { array in array.asArray(Float.self).map { Int($0) } }
    }

    @Test("asArray in the array's own dtype reads one whole array")
    func asArraySameTypeRace() {
        race { array in array.asArray(Int32.self).map { Int($0) } }
    }

    @Test("asArrayChecked with a dtype conversion reads one whole array")
    func asArrayCheckedConvertingRace() {
        race { array in try array.asArrayChecked(Float.self).map { Int($0) } }
    }

    @Test("asData copies one whole array")
    func asDataRace() {
        race { array in
            let data = array.asData(access: .copy)
            guard data.dType == .int32 else { return [] }
            return data.data.withUnsafeBytes { bytes in
                bytes.bindMemory(to: Int32.self).map { Int($0) }
            }
        }
    }

    @Test("item reads one whole scalar, with and without a conversion")
    func itemRace() {
        race(count: 1) { array in [Int(array.item(Int32.self))] }
        race(count: 1) { array in [Int(array.item(Float.self))] }
    }

    @Test("description reads one whole array")
    func descriptionRace() {
        // Six elements: MLX prints all of them up to six, and elides the
        // middle of a longer row with "...".
        race(count: 6) { array in
            let text = array.description
            guard text.hasSuffix("dtype=int32)"), let open = text.firstIndex(of: "["),
                let close = text.firstIndex(of: "]")
            else { return [] }
            return text[text.index(after: open) ..< close].split(separator: ",").map {
                Int($0.trimmingCharacters(in: .whitespaces)) ?? -1
            }
        }
    }

    @Test("asMTLBuffer copies one whole array")
    func asMTLBufferRace() throws {
        nonisolated(unsafe) let device = try #require(MTLCreateSystemDefaultDevice())
        race { array in
            guard let buffer = array.asMTLBuffer(device: device, noCopy: false) else {
                return []
            }
            let elements = buffer.length / MemoryLayout<Int32>.size
            let start = buffer.contents().bindMemory(to: Int32.self, capacity: elements)
            return UnsafeBufferPointer(start: start, count: elements).map { Int($0) }
        }
    }
}
