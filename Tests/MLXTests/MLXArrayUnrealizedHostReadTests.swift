import Foundation
import Metal
import MLX
import Testing

/// Host reads of an array whose evaluation failed upstream (Sentry
/// APPLE-MACOS-1ZE, 2PW, 319, 2EX, 1X5).
///
/// A custom Metal kernel with an invalid body fails at eval time (JIT
/// compile) the same way a Metal allocation or command-buffer failure does:
/// `mlx_array_eval` reports the error to the handler and returns non-zero.
/// Any array *downstream* of the failing primitive in the same tape never
/// reaches `set_data`, so its `array::Data` is null and `mlx_array_data_*`
/// dereferences null inside `Buffer::raw_ptr` -- the production crash.
///
/// Before the fix every test in this suite that reads a downstream array
/// killed the test process with SIGSEGV (signal 11).
@Suite("MLXArray unrealized host reads", .serialized)
struct MLXArrayUnrealizedHostReadTests {

    final class MessageBox: @unchecked Sendable {
        private let lock = NSLock()
        private var _messages: [String] = []
        var messages: [String] { lock.withLock { _messages } }
        func append(_ m: String) { lock.withLock { _messages.append(m) } }
    }

    /// A lazily-built array whose own eval throws inside MLX.
    static func failingArray(name: String, shape: [Int] = [4]) -> MLXArray {
        let kernel = MLXFast.metalKernel(
            name: name, inputNames: ["a"], outputNames: ["out"],
            source: "this is not valid metal source;")
        let a = MLXArray.zeros(shape, type: Float.self)
        return kernel(
            [a], grid: (1, 1, 1), threadGroup: (1, 1, 1),
            outputShapes: [shape], outputDTypes: [.float32])[0]
    }

    /// An array that depends on a failing primitive: the tape aborts before
    /// this array is scheduled, so it has no backing at all.
    static func unrealizedArray(name: String, shape: [Int] = [4]) -> MLXArray {
        failingArray(name: name, shape: shape) + MLXArray(Float(1))
    }

    @Test("asArray on an array downstream of a failed eval returns zeros, not a crash")
    func asArrayDownstreamOfFailedEval() {
        let box = MessageBox()
        let target = Self.unrealizedArray(name: "vmlxBrokenKernelAsArrayDownstream")
        let values = withErrorHandler({ box.append($0) }) {
            target.asArray(Float.self)
        }
        #expect(values.count == 4)
        #expect(values.allSatisfy { $0 == 0 })
        // The eval failure itself, then the named host read that skipped the null backing.
        #expect(box.messages.count >= 2)
        #expect(box.messages.contains { $0.contains("[MLXArray.asArray]") })
    }

    @Test("asArray with a dtype conversion on an unrealized array returns zeros")
    func asArrayConvertingDTypeDownstreamOfFailedEval() {
        let box = MessageBox()
        let target = Self.unrealizedArray(name: "vmlxBrokenKernelAsArrayConvert")
        let values = withErrorHandler({ box.append($0) }) {
            target.asArray(Int32.self)
        }
        #expect(values == [0, 0, 0, 0])
    }

    @Test("asData in every access mode returns nbytes of zeros for an unrealized array")
    func asDataDownstreamOfFailedEval() {
        let box = MessageBox()
        let target = Self.unrealizedArray(name: "vmlxBrokenKernelAsData", shape: [2, 3])
        withErrorHandler({ box.append($0) }) {
            for access in [MLXArray.AccessMethod.copy, .noCopy, .noCopyIfContiguous] {
                let data = target.asData(access: access)
                #expect(data.data.count == target.nbytes)
                #expect(data.data.allSatisfy { $0 == 0 })
                #expect(data.shape == [2, 3])
            }
        }
        #expect(box.messages.contains { $0.contains("[MLXArray.asData]") })
    }

    @Test("asArrayChecked throws the MLX error for an unrealized array")
    func asArrayCheckedThrows() {
        let box = MessageBox()
        let target = Self.unrealizedArray(name: "vmlxBrokenKernelChecked")
        withErrorHandler({ box.append($0) }) {
            #expect(throws: MLXError.self) {
                try target.asArrayChecked(Float.self)
            }
        }
        // No "returning zero bytes" report: the checked read throws instead.
        #expect(!box.messages.contains { $0.contains("returning zero bytes") })
    }

    @Test("checked conversion rejects an invalid array without recursive conversion")
    func checkedInvalidConversionThrows() {
        let invalid = withErrorHandler({ _ in }) {
            matmul(MLXArray.zeros([2, 3]), MLXArray.zeros([4, 2]))
        }
        #expect(throws: MLXError.self) {
            try invalid.asArrayChecked(Float.self)
        }
    }

    @Test("asArrayChecked returns the values of a healthy array")
    func asArrayCheckedHealthy() throws {
        let healthy = MLXArray([Float(1), 2, 3]) * MLXArray(Float(2))
        #expect(try healthy.asArrayChecked(Float.self) == [2, 4, 6])
    }

    @Test("item on an unrealized array returns zero")
    func itemDownstreamOfFailedEval() {
        let box = MessageBox()
        let target = Self.unrealizedArray(name: "vmlxBrokenKernelItem", shape: [1])
        let (f, i) = withErrorHandler({ box.append($0) }) {
            (target.item(Float.self), target.item(Int.self))
        }
        #expect(f == 0)
        #expect(i == 0)
        #expect(box.messages.contains { $0.contains("[MLXArray.item]") })
    }

    @Test("asMTLBuffer on an unrealized array returns nil")
    func asMTLBufferDownstreamOfFailedEval() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let box = MessageBox()
        let target = Self.unrealizedArray(name: "vmlxBrokenKernelMTLBuffer")
        let buffer = withErrorHandler({ box.append($0) }) {
            target.asMTLBuffer(device: device, noCopy: true)
        }
        #expect(buffer == nil)
    }

    @Test("MLXErrorMonitor records every dispatched error regardless of handler")
    func errorMonitorSequence() {
        let before = MLXErrorMonitor.sequence()
        #expect(MLXErrorMonitor.message(since: before) == nil)
        let target = Self.unrealizedArray(name: "vmlxBrokenKernelMonitor")
        withErrorHandler({ _ in }) {
            _ = target.asArray(Float.self)
        }
        #expect(MLXErrorMonitor.sequence() > before)
        #expect(MLXErrorMonitor.message(since: before) != nil)
        #expect(MLXErrorMonitor.message(since: MLXErrorMonitor.sequence()) == nil)
    }

    @Test("a healthy host read after a failed one still returns real values (control)")
    func healthyReadAfterFailure() {
        let target = Self.unrealizedArray(name: "vmlxBrokenKernelControl")
        withErrorHandler({ _ in }) {
            _ = target.asArray(Float.self)
        }
        let healthy = MLXArray([Int32(7), 8, 9]) + MLXArray(Int32(1))
        #expect(healthy.asArray(Int32.self) == [8, 9, 10])
        #expect(healthy.asData(access: .copy).data.count == 12)
        #expect(MLXArray(Int32(41)).item(Int.self) + 1 == 42)
    }

    @Test("the failing primitive's own output keeps its allocated buffer (documents the two shapes)")
    func failingPrimitiveOutputHasBuffer() {
        // The custom kernel allocates its output before compiling, so the
        // output itself has a (never written) buffer; only arrays downstream
        // of it lack `Data`. Both shapes must read without crashing.
        let direct = Self.failingArray(name: "vmlxBrokenKernelDirect")
        let values = withErrorHandler({ _ in }) {
            direct.asArray(Float.self)
        }
        #expect(values.count == 4)
    }
}
