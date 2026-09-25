#if canImport(CryptoKit)
    import CryptoKit
#else
    import Crypto
#endif
import Foundation

/// Identity of a canonical, complete-chunk prefill snapshot. This is separate
/// from the normal chat prefix namespace: equal token IDs do not imply equal
/// floating-point cache state when the prefill partition differs.
///
/// Callers must establish provenance before constructing/persisting a seed:
/// aligned offsets alone do not establish that preparation began canonically.
struct CanonicalPrefillCheckpoint: Equatable, Sendable {
    let chunkSize: Int

    init?(chunkSize: Int) {
        guard chunkSize > 0 else { return nil }
        self.chunkSize = chunkSize
    }

    /// The disk cache still supplies its model identity and token hash. Keep
    /// the existing request salt inside this namespace without ambiguous
    /// concatenation, and distinguish nil from an explicitly empty salt.
    func storageSalt(requestSalt: String?) -> String {
        let request = requestSalt.map { "some:\($0.utf8.count):\($0)" } ?? "none"
        let contract = "canonical-prefill-v1:\(chunkSize):\(request)"
        let digest = SHA256.hash(data: Data(contract.utf8))
            .map { String(format: "%02x", $0) }.joined()
        return "canonical-prefill-v1:\(digest)"
    }

    /// A full-chunk seed may skip any number of canonical chunks, not only
    /// the last one. `prepare` must still forward each remaining chunk with
    /// the original chunk size and then evaluate its final nonempty tail.
    /// Empty, unaligned, longer, equal-length and changed-prefix seeds do not
    /// qualify for this continuation path. Exact-boundary fetch is separate.
    func canContinue(seedTokens: [Int], targetTokens: [Int], storedChunkSize: Int?) -> Bool {
        storedChunkSize == chunkSize
            && !seedTokens.isEmpty
            && seedTokens.count % chunkSize == 0
            && seedTokens.count < targetTokens.count
            && targetTokens.starts(with: seedTokens)
    }
}
