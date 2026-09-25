import Testing

@testable import MLXLMCommon

@Suite struct CanonicalPrefillCheckpointTests {
    @Test func identitySeparatesChunkSizeAndRequestSalt() throws {
        let a = try #require(CanonicalPrefillCheckpoint(chunkSize: 512))
        let b = try #require(CanonicalPrefillCheckpoint(chunkSize: 256))
        let salts: [String?] = [nil, "", "none", "some:0:", "a:b", "a", "🙂"]
        let keys = salts.flatMap {
            [a.storageSalt(requestSalt: $0), b.storageSalt(requestSalt: $0)]
        }
        #expect(Set(keys).count == keys.count)
        #expect(a.storageSalt(requestSalt: "a:b") == a.storageSalt(requestSalt: "a:b"))
        #expect(CanonicalPrefillCheckpoint(chunkSize: 0) == nil)
        #expect(CanonicalPrefillCheckpoint(chunkSize: -1) == nil)
    }

    @Test func multipleRemainingChunksPreserveCanonicalPartition() throws {
        let contract = try #require(CanonicalPrefillCheckpoint(chunkSize: 512))
        // The real R36 diagnostic's canonical seed and target lengths, plus
        // exact chunk edges. Verify the actual chunk inputs, not just counts.
        for targetCount in [9729, 10057, 10240, 10241, 10569, 11000] {
            let target = Array(0 ..< targetCount)
            let seed = Array(target.prefix(9728))
            #expect(
                contract.canContinue(seedTokens: seed, targetTokens: target, storedChunkSize: 512))
            func partition(_ tokens: [Int]) -> [[Int]] {
                stride(from: 0, to: tokens.count, by: 512).map {
                    Array(tokens[$0 ..< min($0 + 512, tokens.count)])
                }
            }
            #expect(
                Array(partition(target).dropFirst(19))
                    == partition(Array(target.dropFirst(seed.count))))
        }
    }

    @Test func rejectsUnalignedOrIncompatibleSeeds() throws {
        let contract = try #require(CanonicalPrefillCheckpoint(chunkSize: 512))
        let target = Array(0 ..< 10057)
        #expect(
            !contract.canContinue(
                seedTokens: Array(target.prefix(10030)), targetTokens: target, storedChunkSize: 512)
        )
        #expect(!contract.canContinue(seedTokens: [], targetTokens: target, storedChunkSize: 512))
        #expect(
            !contract.canContinue(
                seedTokens: Array(target.prefix(9728)), targetTokens: target, storedChunkSize: 256))
        // Even an aligned warm snapshot is not a canonical seed without
        // provenance from the original preparation schedule.
        #expect(
            !contract.canContinue(
                seedTokens: Array(target.prefix(9728)), targetTokens: target, storedChunkSize: nil))
        var changed = Array(target.prefix(9728))
        changed[100] = -1
        #expect(
            !contract.canContinue(seedTokens: changed, targetTokens: target, storedChunkSize: 512))
        let exact = Array(target.prefix(9728))
        #expect(!contract.canContinue(seedTokens: exact, targetTokens: exact, storedChunkSize: 512))
        #expect(
            !contract.canContinue(
                seedTokens: exact, targetTokens: Array(target.prefix(512)), storedChunkSize: 512))
    }
}
