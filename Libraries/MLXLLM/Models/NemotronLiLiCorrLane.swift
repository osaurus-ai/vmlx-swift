// Copyright © 2026 Apple Inc.
//
// The LiLiCorr block drafter wired into a NemotronH target as a native-MTP proposer.
//
// Mirrors the vMLX Python lane (02e70c65 / eba39a47 / 3e93c98a / d318155f):
//   * every target forward on a request cache records the drafter's taps (OUTPUTS of the
//     dflash target blocks, before norm_f) keyed by absolute position; a write starting at
//     position s supersedes everything recorded at >= s, so after a rejected verify the
//     rewritten positions hold the committed prefix (latest write wins);
//   * a draft syncs the drafter's context to the target cache's committed position — rows of
//     a rejected verify beyond it are never handed over — then drafts [anchor, mask x (B-1)]
//     at that position and walks the LiLiCorr path on the GPU;
//   * recorded taps and drafter context belong to ONE target cache object: a forward on any
//     other cache (another request, a scratch boundary re-derive) never feeds this request's
//     drafter, and the first draft on a new cache starts from that cache's own taps.

import Foundation
import MLX
import MLXLMCommon

public final class NemotronLiLiCorrLane: @unchecked Sendable {
    public let drafter: NemotronLiLiCorrDrafter
    let attentionCacheIndex: Int
    let layerOrder: [Int]
    /// Positions kept per cache: the drafter's window plus room for one rejected block.
    let keep: Int

    private let lock = NSLock()
    // Recorder (guarded by `lock`).
    private var recorderOwner: ObjectIdentifier?
    private var chunks: [(start: Int, taps: MLXArray)] = []
    private var pendingStart = 0
    private var pendingRows = 0
    private var pendingOwner: ObjectIdentifier?
    private var pendingLayers: [Int: MLXArray] = [:]
    // Drafter context owner.
    private var drafterOwner: ObjectIdentifier?

    public private(set) var draftCount = 0
    public private(set) var headlessDraftCount = 0

    init(drafter: NemotronLiLiCorrDrafter, attentionCacheIndex: Int) {
        self.drafter = drafter
        self.attentionCacheIndex = attentionCacheIndex
        self.layerOrder = drafter.targetLayerIds
        // Saturating: a config without a sliding window reports 1 << 30, and a malformed one
        // could report Int.max; neither may overflow the bound.
        let (sum, overflow) = drafter.window.addingReportingOverflow(drafter.blockSize * 2)
        self.keep = overflow ? Int.max / 2 : sum
    }

    /// The backbone tap that feeds the recorder. Cache-less forwards are not recorded.
    func makeTap() -> NemotronHBlockOutputTap {
        NemotronHBlockOutputTap(
            layerIndices: Set(layerOrder),
            record: { [unowned self] layer, output in
                self.lock.lock()
                self.pendingLayers[layer] = output
                self.lock.unlock()
            },
            begin: { [unowned self] cache, rows in
                guard let cache, self.attentionCacheIndex < cache.count else { return false }
                let attention = cache[self.attentionCacheIndex]
                self.lock.lock()
                self.pendingOwner = ObjectIdentifier(attention as AnyObject)
                self.pendingStart = attention.offset
                self.pendingRows = rows
                self.pendingLayers.removeAll(keepingCapacity: true)
                self.lock.unlock()
                return true
            },
            end: { [unowned self] in
                self.lock.lock()
                defer { self.lock.unlock() }
                self.finishPendingLocked()
            })
    }

    private func finishPendingLocked() {
        guard let owner = pendingOwner, pendingLayers.count == layerOrder.count, pendingRows > 0
        else {
            pendingLayers.removeAll(keepingCapacity: true)
            pendingOwner = nil
            return
        }
        // (1, L, D) per tap layer -> (L, taps * D) in the drafter's layer order.
        var rows = concatenated(layerOrder.map { pendingLayers[$0]![0] }, axis: -1)
        var start = pendingStart
        if rows.dim(0) > keep {
            let drop = rows.dim(0) - keep
            rows = rows[drop...]
            start += drop
        }
        pendingLayers.removeAll(keepingCapacity: true)
        pendingOwner = nil
        if recorderOwner != owner {
            recorderOwner = owner
            chunks.removeAll()
        }
        // A write at `start` supersedes everything recorded at >= start.
        chunks = chunks.compactMap { chunk in
            if chunk.start >= start { return nil }
            let end = chunk.start + chunk.taps.dim(0)
            if end <= start { return chunk }
            return (chunk.start, chunk.taps[0 ..< (start - chunk.start)])
        }
        chunks.append((start, rows))
        // Bound memory to the last `keep` positions.
        let newestEnd = start + rows.dim(0)
        let floor = newestEnd - keep
        chunks = chunks.compactMap { chunk in
            let end = chunk.start + chunk.taps.dim(0)
            if end <= floor { return nil }
            if chunk.start >= floor { return chunk }
            return (floor, chunk.taps[(floor - chunk.start)...])
        }
    }

    /// Recorded taps for `[from, to)` on `owner`, contiguous from the latest position the
    /// recorder can vouch for; returns the actual first position covered.
    private func recordedTapsLocked(owner: ObjectIdentifier, from: Int, to: Int) -> (Int, MLXArray)? {
        guard recorderOwner == owner, to > from else { return nil }
        // Walk back from `to` while chunks stay contiguous.
        var pieces: [MLXArray] = []
        var cursor = to
        for chunk in chunks.reversed() {
            let end = chunk.start + chunk.taps.dim(0)
            guard end >= cursor, chunk.start < cursor else {
                if end < cursor { break }
                continue
            }
            let lo = Swift.max(chunk.start, from)
            pieces.append(chunk.taps[(lo - chunk.start) ..< (cursor - chunk.start)])
            cursor = lo
            if cursor <= from { break }
        }
        guard cursor < to, !pieces.isEmpty else { return nil }
        let joined = pieces.count == 1 ? pieces[0] : concatenated(pieces.reversed(), axis: 0)
        return (cursor, joined)
    }

    /// Draft `depth` tokens after `nextToken` (at the committed position of `targetCache`).
    func draft(
        nextToken: MLXArray, depth: Int, targetCache: [KVCache],
        embed: (MLXArray) -> MLXArray, lmHead: (MLXArray) -> MLXArray
    ) -> MLXArray {
        let attention = targetCache[attentionCacheIndex]
        let owner = ObjectIdentifier(attention as AnyObject)
        let position = attention.offset

        lock.lock()
        if drafterOwner != owner {
            drafterOwner = owner
            drafter.reset(contextEnd: position)
        }
        if drafter.contextEnd > position {
            drafter.truncate(to: position)
        }
        if drafter.contextEnd < position || drafter.lastContext == nil {
            let from = drafter.lastContext == nil ? Swift.max(0, position - drafter.window) : drafter.contextEnd
            if let (first, taps) = recordedTapsLocked(owner: owner, from: from, to: position) {
                if first != drafter.contextEnd { drafter.reset(contextEnd: first) }
                try? drafter.addContext(taps: taps, position: first)
            }
            if drafter.contextEnd != position {
                // No recorded taps reach the anchor (e.g. a prefix restored from SSD before any
                // forward on this cache): draft from an empty context at the anchor.
                drafter.reset(contextEnd: position)
            }
        }
        lock.unlock()

        let block = concatenated(
            [nextToken.reshaped(1).asType(.int32),
             MLXArray(Array(repeating: Int32(drafter.maskTokenId), count: drafter.blockSize - 1))],
            axis: 0
        ).reshaped(1, -1)
        draftCount += 1
        do {
            let hidden = try drafter.draftHidden(blockEmbeddings: embed(block), position: position)
            let useHead = drafter.lastContext != nil
            if !useHead { headlessDraftCount += 1 }
            let (path, top1) = try drafter.propose(
                hidden: hidden, lmHead: lmHead, embed: embed, useLiLiCorr: useHead)
            return (path ?? top1)[0 ..< Swift.min(depth, drafter.blockSize - 1)]
        } catch {
            // Position contract violated (should not happen after the sync above): propose the
            // anchor repeated, which the verifier rejects at row 0 — never a wrong commit.
            return MLXArray(Array(repeating: Int32(drafter.maskTokenId), count: depth))
        }
    }
}

/// Attach the bundle's `lilicorr/` drafter to a NemotronH target (no-op without one).
public enum NemotronLiLiCorrAttachment {
    public static func drafterDirectory(bundle: URL) -> URL? {
        let dir = bundle.appendingPathComponent("lilicorr")
        let config = dir.appendingPathComponent("config.json")
        guard FileManager.default.fileExists(atPath: config.path),
            let data = try? Data(contentsOf: config),
            let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let dflash = json["dflash_config"] as? [String: Any],
            (dflash["projector_type"] as? String)?.lowercased() == "lilicorr"
        else { return nil }
        return dir
    }

    /// Attach when native MTP was explicitly requested for this load, the model has no
    /// recursive MTP head, and the bundle ships a LiLiCorr drafter.
    @discardableResult
    public static func attachIfRequested(to model: NemotronHModel, bundle: URL) throws -> Bool {
        guard NativeMTPActivation.isExplicitlyRequested, !model.hasRecursiveMTPHead,
            let dir = drafterDirectory(bundle: bundle)
        else { return false }
        try model.attachLiLiCorr(drafter: NemotronLiLiCorrDrafter.load(directory: dir))
        return true
    }
}
