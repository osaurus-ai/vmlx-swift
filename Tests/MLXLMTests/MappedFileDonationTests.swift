import Cmlx
import CmlxGraphShim
import Foundation
import MLX
import Testing

/// An array over a read-only file mapping must never be donated to an op's output: the op would write into the
/// mapping and fault (ml-explore/mlx#4634). The work runs in a child process, so a fault fails this test instead of
/// ending the test run.
struct MappedFileDonationTests {
    @Test func mappedArrayIsNotDonatedToAnOpOnTheCPU() async {
        await #expect(processExitsWith: .success) {
            try MLXMetalTestLock.withLock {
                let page = Int(getpagesize())
                let url = FileManager.default.temporaryDirectory
                    .appendingPathComponent("mapped-donation-\(UUID().uuidString)")
                try Data(count: 2 * page).write(to: url)  // zeros
                defer { try? FileManager.default.removeItem(at: url) }
                let file = try FileHandle(forReadingFrom: url)
                defer { try? file.close() }
                var handle: UnsafeMutableRawPointer?
                defer { vmlx_mapped_file_release(&handle) }
                let created = try withError { vmlx_mapped_file_create(file.fileDescriptor, &handle) }
                precondition(created == 0, "the file could not be mapped")
                var raw = mlx_array_new()
                var dimensions: [Int32] = [Int32(page / 4)]
                let status = try withError {
                    vmlx_mapped_file_array(&raw.ctx, handle, UInt64(page), page, &dimensions, 1,
                                           Int32(DType.float32.cmlxDtype.rawValue))
                }
                precondition(status == 0, "the mapping failed")
                var mapped: MLXArray? = MLXArray(raw)
                let out = exp(mapped!, stream: .cpu)
                mapped = nil  // the graph now holds the only reference to the mapped array
                eval(out)
                precondition(all(out .== Float(1)).item(Bool.self), "exp(0) is not 1")
                let after = try Data(contentsOf: url)
                precondition(after.allSatisfy { $0 == 0 }, "the op wrote into the file")
            }
        }
    }
}
