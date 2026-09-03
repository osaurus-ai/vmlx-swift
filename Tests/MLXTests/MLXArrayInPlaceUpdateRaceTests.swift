@testable import MLX
import Dispatch
import Testing

/// Regression for the production EXC_BAD_ACCESS in
/// `mlx::core::allocator::Buffer::raw_ptr` (Sentry APPLE-MACOS-1ZE, qwen4_exp).
///
/// Mechanism: `Qwen4ExpPLE.prefetch` host-reads a cache-resident slot array
/// (`cache[2].asArray`) on every decode step, while the recurrent cache updates
/// that SAME persistent slot array in place through the `ArraysCache` subscript
/// setter → `MLXArray._updateInternal` → `mlx_array_set`, which frees the old
/// `ctx`/buffer. Host reads hold `evalLock` across the realized copy; before the
/// fix the in-place update did NOT, so a concurrent update freed the buffer
/// mid-copy and the reader dereferenced a dangling `MTL::Buffer`.
///
/// The reproduction is deterministic on an unfixed build (crashes 8/8): a large
/// array makes the host copy span milliseconds, `cacheLimit = 0` releases the
/// freed buffer immediately instead of recycling it, and the writer issues the
/// bare ctx swap (no `eval`, so it never takes `evalLock` and can run while a
/// reader holds it mid-copy). The fix serializes `_updateInternal` on `evalLock`;
/// with it the reader and the freeing swap can no longer overlap.
@Suite("MLXArray in-place update vs host read race")
struct MLXArrayInPlaceUpdateRaceTests {

    private final class Box: @unchecked Sendable {
        let array: MLXArray
        init(_ array: MLXArray) { self.array = array }
    }

    @Test("host-reading an array while it is updated in place must not crash")
    func inPlaceUpdateDuringHostReadIsSafe() {
        let savedLimit = Memory.cacheLimit
        Memory.cacheLimit = 0
        defer { Memory.cacheLimit = savedLimit }

        let size = 2_000_000          // ~8 MB Int32 backing -> multi-ms copy window
        let iterations = 4000
        let readerThreads = 3

        let box = Box(MLXArray(Array(Int32(0) ..< Int32(size))))
        MLX.eval(box.array)

        let group = DispatchGroup()

        // Writer: bare ctx swap, freeing the old buffer with no evalLock. It must
        // NOT eval `fresh` first — taking evalLock there would serialize against
        // the reader's locked host copy and hide the race (the production shape
        // is the recurrent cache updating the slot with no lock).
        group.enter()
        DispatchQueue.global(qos: .userInitiated).async {
            for index in 0 ..< iterations {
                let base = Int32(truncatingIfNeeded: index)
                let fresh = MLXArray((0 ..< Int32(size)).map { base &+ $0 }) + base
                box.array._updateInternal(fresh)
            }
            group.leave()
        }

        // Readers: host-read the same array (exactly as Qwen4ExpPLE.prefetch does).
        for _ in 0 ..< readerThreads {
            group.enter()
            DispatchQueue.global(qos: .userInitiated).async {
                for _ in 0 ..< iterations {
                    _ = box.array.asArray(Int32.self)
                }
                group.leave()
            }
        }

        group.wait()
        #expect(box.array.size == size)
    }
}
