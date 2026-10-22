import Foundation
import MLX

/// An immutable checkpoint whose owner is a file, not a retained MLX cache.
/// Capturing runs on the generation owner before decode mutates the live cache.
/// The gathered writer borrows its evaluated buffers only for this synchronous
/// call. No full cache copy or writer queue survives the handoff.
final class DiskPromptSnapshot {
    let tokenCount: Int
    let byteCount: Int
    private let file: FileHandle
    var fileDescriptor: Int32 { file.fileDescriptor }

    private init(tokenCount: Int, byteCount: Int, file: FileHandle) {
        self.tokenCount = tokenCount
        self.byteCount = byteCount
        self.file = file
    }

    deinit { try? file.close() }

    /// Only native full-KV and rotating/SWA states whose complete payload is
    /// self-contained. Recurrent/pool/quantized topologies keep their existing
    /// companion contract until separately qualified; no model-name matching.
    static func supports(_ cache: [KVCache]) -> Bool {
        !cache.isEmpty && cache.allSatisfy { $0 is KVCacheSimple || $0 is RotatingKVCache }
    }

    static func capture(
        cache: [KVCache], tokenCount: Int, directory: URL
    ) throws -> DiskPromptSnapshot {
        try Task.checkCancellation()
        guard tokenCount > 0, supports(cache),
              cacheBoundaryLeafOffsets(cache).allSatisfy({ $0 == tokenCount }) else {
            throw SnapshotError.invalidBoundary
        }
        let workspace = directory.appendingPathComponent(".prompt-workspace-" + UUID().uuidString,
                                                        isDirectory: true)
        try FileManager.default.createDirectory(at: workspace, withIntermediateDirectories: false,
                                               attributes: [.posixPermissions: 0o700])
        var captured = false
        defer { if !captured { try? FileManager.default.removeItem(at: workspace) } }
        let url = workspace.appendingPathComponent("checkpoint.safetensors")
        let arrays = TQDiskSerializer.serialize(cache: cache, ssmStates: nil,
                                               preserveStandardKVStorageDType: true)
        guard !arrays.isEmpty else { throw SnapshotError.incompletePayload }
        MLX.eval(Array(arrays.values))
        guard let tensors = SafetensorsByteWriter.tensors(from: arrays) else {
            throw SnapshotError.incompletePayload
        }
        let bytes = try SafetensorsByteWriter.write(tensors, metadata: ["format": "mlx"],
                                                   to: url, uncached: true)
        try Task.checkCancellation()
        let file = try FileHandle(forUpdating: url)
        // Keep the descriptor, remove the name. Even a crash releases the SSD
        // workspace; quota eviction cannot unlink a checkpoint mid-answer.
        do { try FileManager.default.removeItem(at: workspace) }
        catch { try? file.close(); throw error }
        let result = DiskPromptSnapshot(tokenCount: tokenCount, byteCount: bytes, file: file)
        captured = true
        return result
    }

    /// Restore only when terminal boundary construction needs the checkpoint.
    /// The owned reader bypasses macOS's file cache on Darwin and preserves all
    /// native dtypes. A missing/corrupt file must never be replaced by decode's
    /// newer state under the old prompt key.
    func restore(into freshCache: [KVCache]) throws -> [KVCache] {
        let (arrays, _) = try ResidentSafetensorsReader.load(file: file)
        var cache = freshCache
        guard restoreFromDiskArrays(arrays, into: &cache, requirePromptBoundary: true) == tokenCount,
              cacheBoundaryLeafOffsets(cache).allSatisfy({ $0 == tokenCount }) else {
            throw SnapshotError.incompletePayload
        }
        MLX.eval(cache)
        return cache
    }

    enum SnapshotError: Error { case invalidBoundary, incompletePayload }
}
