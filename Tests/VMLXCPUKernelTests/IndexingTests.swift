// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

#if os(Linux)
    import Foundation
    import MLX
    import Testing

    /// Gathers and general copies large enough that #3019 splits them across the pool: an output
    /// of 262144 elements or more, no less than 32768 elements a thread, and at least as many
    /// pieces (indices, leading slices or outer iterations) as threads. Each value is its own
    /// position in its source, exact in float32, so each output is compared bit for bit with the
    /// positions computed on the host.
    @Suite(.serialized) struct IndexingTests {
        /// take along axis 0, as an embedding lookup does: 512 ids into a [1000, 768] table, 393216
        /// elements. The ids come as a vector, as a [16, 32] batch, and as that batch transposed,
        /// whose strides each thread follows to its first id; some are negative, counted from the
        /// end. Then the same ids into a table whose rows are not contiguous (a transposed view),
        /// and 300000 single elements of a vector.
        @Test func gatherAcrossThePool() {
            KernelLock.run {
                let (rows, width, count) = (1000, 768, 512)
                var random = SeededRandom(seed: 71)
                let ids = (0 ..< count).map { _ in
                    Int32(Int(random.next() % UInt64(2 * rows)) - rows)
                }
                // Each id's row, where `position(r, c)` is element (r, c)'s position in its source.
                func expected(_ ids: [Int32], _ position: (Int, Int) -> Int) -> [Double] {
                    ids.flatMap { id in
                        let r = id < 0 ? Int(id) + rows : Int(id)
                        return (0 ..< width).map { Double(position(r, $0)) }
                    }
                }
                let table = MLXArray((0 ..< rows * width).map(Float.init), [rows, width])
                let vector = MLXArray(ids, [count])
                let batch = vector.reshaped([16, 32])
                let byRow = expected(ids) { $0 * width + $1 }
                expectIdentical(take(table, vector, axis: 0), byRow, "ids as a vector")
                expectIdentical(take(table, batch, axis: 0), byRow, "ids as a batch")
                // Element (i, j) of the transposed batch is ids[j * 32 + i].
                let transposedIds = (0 ..< count).map { ids[($0 % 16) * 32 + $0 / 16] }
                expectIdentical(
                    take(table, batch.transposed(), axis: 0),
                    expected(transposedIds) { $0 * width + $1 }, "strided ids")
                // Element (r, c) of the view is element (c, r) of a [768, 1000] array.
                let columns = MLXArray((0 ..< width * rows).map(Float.init), [width, rows])
                expectIdentical(
                    take(columns.transposed(), vector, axis: 0), expected(ids) { $1 * rows + $0 },
                    "rows that are not contiguous")
                let length = 300_000
                let single = (0 ..< length).map { _ in Int32(random.next() % UInt64(length)) }
                expectIdentical(
                    take(
                        MLXArray((0 ..< length).map(Float.init), [length]),
                        MLXArray(single, [length]), axis: 0),
                    single.map(Double.init), "single elements")
            }
        }

        /// takeAlong along the last axis of [512, 1024] (524288 elements, 512 leading rows), with
        /// negative indices among the rest; and along the middle axis of [64, 128, 40] (327680
        /// elements, 64 leading slices), from a contiguous array and from a transposed view, whose
        /// strides each thread follows to its first slice.
        @Test func gatherAlongAnAxisAcrossThePool() {
            KernelLock.run {
                var random = SeededRandom(seed: 72)
                let (rows, columns) = (512, 1024)
                let last = (0 ..< rows * columns).map { _ in
                    Int32(Int(random.next() % UInt64(2 * columns)) - columns)
                }
                expectIdentical(
                    takeAlong(
                        MLXArray((0 ..< rows * columns).map(Float.init), [rows, columns]),
                        MLXArray(last, [rows, columns]), axis: 1),
                    last.enumerated().map { i, j in
                        Double(i / columns * columns + (j < 0 ? Int(j) + columns : Int(j)))
                    }, "the last axis")
                let (a, b, c) = (64, 128, 40)
                let middle = (0 ..< a * b * c).map { _ in Int32(random.next() % UInt64(b)) }
                let indices = MLXArray(middle, [a, b, c])
                // Element (p, q, s) of the contiguous array is at (p * b + q) * c + s. Element
                // (p, q, s) of the view is element (q, p, s) of a [128, 64, 40] array.
                expectIdentical(
                    takeAlong(
                        MLXArray((0 ..< a * b * c).map(Float.init), [a, b, c]), indices, axis: 1),
                    middle.enumerated().map { i, q in
                        Double((i / (b * c) * b + Int(q)) * c + i % c)
                    }, "the middle axis")
                expectIdentical(
                    takeAlong(
                        MLXArray((0 ..< b * a * c).map(Float.init), [b, a, c]).transposed(1, 0, 2),
                        indices, axis: 1),
                    middle.enumerated().map { i, q in
                        Double((Int(q) * a + i / (b * c)) * c + i % c)
                    }, "the middle axis of a view")
            }
        }

        /// The positions, in a contiguous array of `shape`, of the elements of its transpose by
        /// `axes`, in the transpose's row-major order.
        static func transposedPositions(_ shape: [Int], _ axes: [Int]) -> [Double] {
            var strides = [Int](repeating: 1, count: shape.count)
            for d in stride(from: shape.count - 2, through: 0, by: -1) {
                strides[d] = strides[d + 1] * shape[d + 1]
            }
            let (outShape, outStrides) = (axes.map { shape[$0] }, axes.map { strides[$0] })
            var index = [Int](repeating: 0, count: shape.count)
            return (0 ..< shape.reduce(1, *)).map { _ in
                let position = zip(index, outStrides).reduce(0) { $0 + $1.0 * $1.1 }
                var d = index.count - 1
                while d >= 0 {
                    index[d] += 1
                    if index[d] < outShape[d] { break }
                    index[d] = 0
                    d -= 1
                }
                return Double(position)
            }
        }

        /// General copies of a 5-D view, axes (1, 0, 3, 2, 4) of [8, 8, 16, 8, 40]: 327680
        /// elements, and no two axes collapse into one, so #3019 splits its 64 outer iterations,
        /// over two axes, across the pool, each thread seeking its first. contiguous() copies the
        /// view into a contiguous array (a strided source). Concatenating two such views copies
        /// each into strided slices of the result (both sides strided).
        @Test func generalCopiesAcrossThePool() {
            KernelLock.run {
                let shape = [8, 8, 16, 8, 40]
                let axes = [1, 0, 3, 2, 4]
                let count = shape.reduce(1, *)
                let first = MLXArray((0 ..< count).map(Float.init), shape)
                let second = MLXArray((count ..< 2 * count).map(Float.init), shape)
                let positions = Self.transposedPositions(shape, axes)
                expectIdentical(
                    first.transposed(axes: axes).contiguous(), positions, "a strided source")
                // The views are [8, 8, 8, 16, 40], joined along axis 1 into [8, 16, 8, 16, 40]:
                // for each leading index, 8 blocks of the first, then 8 of the second.
                let block = 8 * 16 * 40
                var joined = [Double]()
                for i in 0 ..< 8 {
                    for j in 0 ..< 16 {
                        let start = (i * 8 + j % 8) * block
                        joined += positions[start ..< start + block].map {
                            j < 8 ? $0 : $0 + Double(count)
                        }
                    }
                }
                expectIdentical(
                    concatenated(
                        [first.transposed(axes: axes), second.transposed(axes: axes)], axis: 1),
                    joined, "strided slices")
            }
        }
    }
#endif
