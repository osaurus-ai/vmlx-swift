// A safetensors shard that is shorter than its own header declares does not
// fail to load — the weights are memory-mapped, so the tensors whose data falls
// past end-of-file read as zeros on most launches and as leftover garbage on
// some, with the outcome fixed for the life of the mapping.
//
// Observed: a DeepSeek-V4-Flash bundle with 8 of 102 shards short by ~301 MB in
// total. About one launch in five produced gibberish from the very first sampled
// token; the rest silently dropped a shared-expert contribution and looked fine.
// Because the outcome was decided at load and then held for the whole process,
// it presented as a nondeterministic concurrency bug in the MoE routing path and
// survived five A/B legs that each changed a suspected racing component and
// moved the failure rate not at all.
//
// The file length is knowable before a single byte is mapped, so these tests pin
// that we look.

import Foundation
import Testing

@testable import MLXLMCommon

struct TruncatedSafetensorsTests {

    /// Build a safetensors file whose header declares `declaredBytes` of tensor
    /// data while only `actualBytes` are written.
    private func makeShard(
        at url: URL, declaredBytes: Int, actualBytes: Int
    ) throws {
        let header: [String: Any] = [
            "weight": [
                "dtype": "F16",
                "shape": [declaredBytes / 2],
                "data_offsets": [0, declaredBytes],
            ]
        ]
        let headerData = try JSONSerialization.data(withJSONObject: header)
        var out = Data()
        var length = UInt64(headerData.count)
        withUnsafeBytes(of: &length) { out.append(contentsOf: $0) }
        out.append(headerData)
        out.append(Data(repeating: 0, count: actualBytes))
        try out.write(to: url)
    }

    @Test("A shard shorter than its header reports exactly the missing bytes")
    func truncatedShardIsDetected() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("vmlx-trunc-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        let url = dir.appendingPathComponent("model-00001-of-00001.safetensors")
        try makeShard(at: url, declaredBytes: 4096, actualBytes: 1024)

        #expect(safetensorsMissingByteCount(url) == 3072)
    }

    @Test("A complete shard reports no missing bytes")
    func completeShardIsAccepted() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("vmlx-trunc-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        let url = dir.appendingPathComponent("model-00001-of-00001.safetensors")
        try makeShard(at: url, declaredBytes: 4096, actualBytes: 4096)

        #expect(safetensorsMissingByteCount(url) == nil)
    }

    /// A file longer than its header declares is NOT truncation — some writers
    /// pad. Flagging it would turn a working bundle into a hard load failure.
    @Test("A shard longer than its header declares is not treated as truncated")
    func paddedShardIsAccepted() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("vmlx-trunc-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        let url = dir.appendingPathComponent("model-00001-of-00001.safetensors")
        try makeShard(at: url, declaredBytes: 4096, actualBytes: 8192)

        #expect(safetensorsMissingByteCount(url) == nil)
    }

    /// Non-safetensors and unparseable files must stay silent — this check is a
    /// guard against a specific corruption, not a general file validator.
    @Test("An unparseable file is ignored rather than reported as truncated")
    func unparseableFileIsIgnored() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("vmlx-trunc-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        let url = dir.appendingPathComponent("garbage.safetensors")
        try Data(repeating: 0xAB, count: 32).write(to: url)

        #expect(safetensorsMissingByteCount(url) == nil)
    }
}
