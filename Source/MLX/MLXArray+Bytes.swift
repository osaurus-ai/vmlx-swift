// Copyright © 2024 Apple Inc.

import Cmlx
import Foundation

// MARK: - Backing / Bytes

extension MLXArray {

    /// Return the dimension where the storage is contiguous.
    ///
    /// If this returns 0 then the whole storage is contiguous.  If it returns ndmin + 1 then none of it is contiguous.
    func contiguousToDimension() -> Int {
        let shape = self.shape
        let strides = self.internalStrides

        var expectedStride = 1

        for (dimension, (shape, stride)) in zip(shape, strides).enumerated().reversed() {
            // as long as the actual strides match the expected (contiguous) strides
            // the backing is contiguous in these dimensions
            if stride != expectedStride {
                return dimension + 1
            }
            expectedStride *= shape
        }

        return 0
    }

    /// Return the physical size of the backing (assuming it is evaluated) in elements.  This should
    /// only be used when accessing the backing directly, e.g. via `mlx_array_data_uint8()`
    var physicalSize: Int {
        // nbytes is the logical size of the input, not the physical size
        return zip(self.shape, self.internalStrides)
            .map { Swift.abs($0.0 * $0.1) }
            .max()
            ?? self.size
    }

    func copy(from: UnsafeRawBufferPointer, toContiguous output: UnsafeMutableRawBufferPointer) {
        let contiguousDimension = self.contiguousToDimension()
        let shape = self.shape
        let strides = self.internalStrides

        if contiguousDimension == 0 {
            // entire backing is contiguous
            from.copyBytes(to: output)

        } else {
            // only part of the backing is contiguous (possibly a single element)
            // iterate the non-contiguous parts and copy the contiguous chunks into
            // the output.

            // these are the parts to iterate
            let ndim = contiguousDimension
            let itemSize = self.itemSize

            // the size of each chunk that we copy.  this computes the stride of
            // (contiguousDimension - 1) if it were contiguous
            let destItemSize: Int
            if contiguousDimension == self.ndim {
                // nothing contiguous
                destItemSize = itemSize
            } else {
                destItemSize = strides[contiguousDimension] * shape[contiguousDimension] * itemSize
            }

            // the index of the current source item
            var index = Array.init(repeating: 0, count: ndim)

            // output pointer
            var dest = output.baseAddress!

            while true {
                // compute the source index by multiplying the index by the
                // stride for each dimension

                // note: in the case where the array has negative strides / offset
                // the base pointer we have will have the offset already applied,
                // e.g. asStrided(a, [3, 3], strides: [-3, -1], offset: 8)

                let sourceIndex = zip(index, strides).reduce(0) { $0 + ($1.0 * $1.1) }

                // convert to byte pointer
                let src = from.baseAddress! + sourceIndex * itemSize
                dest.copyMemory(from: src, byteCount: destItemSize)

                // next output address
                dest += destItemSize

                // increment the index
                for dimension in Swift.stride(from: ndim - 1, through: 0, by: -1) {
                    // do we need to "carry" into the next dimension?
                    if index[dimension] == (shape[dimension] - 1) {
                        if dimension == 0 {
                            // all done
                            return
                        }

                        index[dimension] = 0
                    } else {
                        // just increment the dimension and we are done
                        index[dimension] += 1
                        break
                    }
                }
            }

        }
    }

    /// Return the contents as a single contiguous 1d `Swift.Array`.
    ///
    /// Note: because the number of dimensions is dynamic, this cannot produce a multi-dimensional
    /// array.
    ///
    /// ### See Also
    /// - <doc:conversion>
    /// - ``asData(access:)``
    /// - ``asData(noCopy:disambiguate:)``
    /// - ``asMTLBuffer(device:noCopy:)``
    public func asArray<T: HasDType>(_ type: T.Type) -> [T] {
        if type.dtype != self.dtype {
            return self.asType(type).asArray(type)
        }

        // Keep the shared-stream lock through the host copy, not only through
        // `mlx_array_eval`. A concurrent teardown/cache-clear may otherwise
        // enter after eval has returned but before `mlx_array_data_uint8`
        // reads the backing descriptor. Qwen4Exp PLE performs this host read
        // once per decode step, so server cancellation/load churn made that
        // tiny gap a production EXC_BAD_ACCESS in Buffer::raw_ptr().
        return withEvaluatedHostAccess { realized in
            guard realized else {
                reportUnrealizedHostRead("asArray")
                return Self.zeroFilled(count: self.size)
            }
            return copyRealizedHostBytes()
        }
    }

    /// ``asArray(_:)`` that throws instead of returning zero-filled values
    /// when the array could not be evaluated (a Metal allocation,
    /// command-buffer, or kernel failure anywhere in its graph).
    ///
    /// Prefer this on throwing paths that hand the bytes to the user, such
    /// as writing an image to disk: a failed evaluation then fails the
    /// request instead of producing a plausible-looking zero result.
    ///
    /// - Throws: ``MLXError/caught(_:)`` carrying the MLX error message.
    public func asArrayChecked<T: HasDType>(_ type: T.Type) throws -> [T] {
        if type.dtype != self.dtype {
            return try self.asType(type).asArrayChecked(type)
        }
        return try withEvaluatedHostAccess { realized in
            guard realized else {
                throw MLXError.caught(
                    MLXErrorMonitor.lastMessage
                        ?? "array could not be evaluated"
                        + " (shape=\(self.shape), dtype=\(self.dtype))")
            }
            return copyRealizedHostBytes()
        }
    }

    /// Copy the realized backing into a contiguous Swift array. Only valid
    /// inside ``withEvaluatedHostAccess(_:)`` when `realized` is true.
    private func copyRealizedHostBytes<T: HasDType>() -> [T] {
        [T](unsafeUninitializedCapacity: self.size) { destination, initializedCount in
            let source = UnsafeRawBufferPointer(
                start: mlx_array_data_uint8(self.ctx), count: physicalSize * itemSize)
            copy(from: source, toContiguous: UnsafeMutableRawBufferPointer(destination))
            initializedCount = self.size
        }
    }

    /// `count` zero elements. All ``HasDType`` conformers are trivial types
    /// whose zero value is all-zero bytes.
    static func zeroFilled<T: HasDType>(count: Int) -> [T] {
        [T](unsafeUninitializedCapacity: count) { destination, initializedCount in
            UnsafeMutableRawBufferPointer(destination).initializeMemory(
                as: UInt8.self, repeating: 0)
            initializedCount = count
        }
    }

    /// Report a host read that found no realized backing. The eval failure
    /// itself was already dispatched by `mlx_array_eval`; this names the
    /// read that would otherwise have dereferenced a null `array::Data`
    /// (Sentry APPLE-MACOS-1ZE / 2PW / 319 / 2EX / 1X5).
    func reportUnrealizedHostRead(_ surface: String) {
        reportMLXError(
            "[MLXArray.\(surface)] host read of an array that failed to evaluate"
                + " (shape=\(self.shape), dtype=\(self.dtype)); returning zero bytes."
                + " MLX error: \(MLXErrorMonitor.lastMessage ?? "unknown")")
    }

    /// Evaluate under the stream driver lock and report whether the array is
    /// realized. `mlx_array_eval` returns non-zero when the C++ eval threw
    /// (the exception is forwarded to the MLX error handler); the array and
    /// everything downstream of the failed primitive then have no
    /// `array::Data`, and `mlx_array_data_*` would dereference null.
    ///
    /// Must be called with `evalLock` held.
    private func realizeForHostRead() -> Bool {
        mlx_array_eval(ctx) == 0
    }

    /// Evaluate and keep the global MLX stream driver lock held while a
    /// caller reads the realized host backing. Internal so every host-copy
    /// surface can share the same lifetime boundary without exposing the lock.
    ///
    /// `realized` is false when evaluation failed; the body must not touch
    /// `mlx_array_data_*` in that case.
    func withEvaluatedHostAccess<Result>(
        _ body: (_ realized: Bool) throws -> Result
    ) rethrows -> Result {
        try evalLock.withLock {
            let realized = realizeForHostRead()
            return try body(realized)
        }
    }

    /// Evaluate under the stream driver lock and report whether the array has
    /// a realized backing. Callers that read through the mlx-c item/data
    /// accessors outside the lock use this to skip the read on failure.
    func realizedForHostRead() -> Bool {
        evalLock.withLock { realizeForHostRead() }
    }

#if DEBUG
    static func withEvalLockForTesting(_ body: () -> Void) {
        evalLock.withLock(body)
    }
#endif

    /// How to access backing data with ``asData(access:)`` -- this controls how
    /// ``MLXArrayData`` is produced.
    public enum AccessMethod {
        /// Create a contiguous copy of the data backing the ``MLXArray``.  The lifetime of this data
        /// independent of the lifetime of the ``MLXArray``.
        case copy

        /// Return contiguous data from the backing of the ``MLXArray``, avoiding a copy if possible.
        /// The lifetime of the result is valid only while keeping the ``MLXArray`` alive.  This is best
        /// for temporary copies, e.g. creating and writing an image to disk.
        case noCopyIfContiguous

        /// Return a wrapper around the backing of the ``MLXArray``.  This might not be
        /// contiguous -- the caller must examine ``MLXArrayData/strides``
        /// The lifetime of the result is valid only while keeping the ``MLXArray`` alive.  This is best
        /// for temporary copies, e.g. creating and writing an image to disk.
        case noCopy
    }

    /// Container for `Data` backing of ``MLXArray``.
    ///
    /// ### See Also
    /// - ``MLXArray/asData(access:)``
    public struct MLXArrayData {
        /// The bytes backing the ``MLXArray``.
        ///
        /// The ``MLXArray/AccessMethod`` passed to ``MLXArray/asData(access:)``
        /// controls the lifetime and potential layout of this data (e.g. contiguous vs non-contiguous).
        public let data: Data

        /// Dimensions of the data when viewed as ``dType``
        public let shape: [Int]

        /// Strides of the data in terms of the ``dType``
        public let strides: [Int]

        /// The layout of a single items in the ``data``
        public let dType: DType
    }

    /// return a copy of the backing in contiguous layout
    internal func asDataCopy() -> MLXArrayData {
        // An array with no elements has no backing buffer. `physicalSize` is
        // derived from the strides and stays non-zero for a shape like
        // [1, 0, 3, 448], so the null base would pair with a non-zero count —
        // which `UnsafeRawBufferPointer` traps on.
        guard let base = mlx_array_data_uint8(self.ctx) else {
            return MLXArrayData(
                data: Data(),
                shape: self.shape, strides: contiguousStrides(shape: self.shape),
                dType: self.dtype)
        }

        // point into the possibly non-contiguous backing
        let source = UnsafeRawBufferPointer(
            start: base, count: physicalSize * itemSize)

        var data = Data(count: self.nbytes)
        data.withUnsafeMutableBytes { destination in
            copy(from: source, toContiguous: destination)
        }
        return MLXArrayData(
            data: data,
            shape: self.shape, strides: contiguousStrides(shape: self.shape),
            dType: self.dtype)
    }

    /// Return the contents as `Data` bytes in the native ``dtype``.
    ///
    /// > If you use ``AccessMethod/noCopy`` or ``AccessMethod/noCopyIfContiguous`` you
    /// must guarantee that the lifetime of the ``MLXArray`` exceeds the lifetime of the result.
    ///
    /// Callers can specify an ``AccessMethod`` to cause the data to be in _contiguous_ memory
    /// vs. whatever layout the backing actually has.  Callers can use ``MLXArrayData/strides``
    /// if the data is not contiguous.
    ///
    /// By default it will use ``AccessMethod/copy`` and will return a copy of the data in
    /// contiguous memory.
    ///
    /// ### See Also
    /// - <doc:conversion>
    /// - ``asArray(_:)``
    /// - ``asMTLBuffer(device:noCopy:)``
    public func asData(access: AccessMethod = .copy) -> MLXArrayData {
        // Same realization gate as `asArray`: a failed eval leaves no
        // `array::Data`, and every branch below reads `mlx_array_data_uint8`.
        guard realizedForHostRead() else {
            reportUnrealizedHostRead("asData")
            return MLXArrayData(
                data: Data(count: self.nbytes),
                shape: self.shape, strides: contiguousStrides(shape: self.shape),
                dType: self.dtype)
        }

        switch access {
        case .copy:
            return asDataCopy()

        case .noCopyIfContiguous:
            if self.contiguousToDimension() == 0 {
                // the backing is contiguous, we can provide a wrapper
                // for the contents without a copy.
                //
                // An array with no elements has no backing buffer, so
                // `mlx_array_data_uint8` returns null. There are no bytes to
                // wrap, and `Data(bytesNoCopy:)` may not take a null base, so
                // hand back empty `Data` rather than trapping. A zero-element
                // pixel tensor reaches here through `computeMediaSalt`.
                guard let source = UnsafeMutableRawPointer(mutating: mlx_array_data_uint8(self.ctx))
                else {
                    return MLXArrayData(
                        data: Data(),
                        shape: self.shape, strides: contiguousStrides(shape: self.shape),
                        dType: self.dtype)
                }
                let data = Data(
                    bytesNoCopy: source, count: size * itemSize,
                    deallocator: .none)

                return MLXArrayData(
                    data: data,
                    shape: self.shape, strides: contiguousStrides(shape: self.shape),
                    dType: self.dtype)

            } else {
                // not contiguous
                return asDataCopy()
            }

        case .noCopy:
            // Same null backing as `.noCopyIfContiguous` above: no elements
            // means no buffer to reference.
            let data: Data
            if let source = UnsafeMutableRawPointer(mutating: mlx_array_data_uint8(self.ctx)) {
                data = Data(bytesNoCopy: source, count: nbytes, deallocator: .none)
            } else {
                data = Data()
            }

            let strides: [Int]
            if ndim == 0 {
                strides = []
            } else {
                let internalStrides = mlx_array_strides(ctx)!
                strides = (0 ..< ndim).map { Int(internalStrides[$0]) }
            }

            return MLXArrayData(
                data: data,
                shape: self.shape, strides: strides,
                dType: self.dtype)
        }
    }

    /// Return the contents as contiguous bytes in the native ``dtype``.
    ///
    /// > If you can guarantee the lifetime of the ``MLXArray`` will exceed the Data and that
    /// the array will not be mutated (e.g. using indexing or other means) it is possible to pass `noCopy: true`
    /// to reference the backing bytes.
    ///
    /// **Replaced with** ``asData(access:)``
    ///
    /// ### See Also
    /// - <doc:conversion>
    /// - ``asArray(_:)``
    /// - ``asMTLBuffer(device:noCopy:)``
    @available(*, deprecated, message: "use asData(acccess: .copy)")
    public func asData(noCopy: Bool = false, disambiguate: Bool = false) -> Data {
        return asData(access: noCopy ? .noCopyIfContiguous : .copy)
            .data
    }

}

/// Return the strides for contiguous memory
func contiguousStrides(shape: [Int]) -> [Int] {
    var result = [Int]()
    var current = 1
    for d in shape.reversed() {
        result.append(current)
        current *= d
    }
    result.reverse()
    return result
}
