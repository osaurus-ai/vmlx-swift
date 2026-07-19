import Foundation
import Testing

@testable import MLXLMCommon

@Suite("TurboQuant live cache transition telemetry")
struct TurboQuantCacheTransitionTelemetryTests {
    @Test("mixed Gemma topology reports eight converted KV and forty preserved rotating layers")
    func mixedGemmaTopology() {
        let before: [any KVCache] =
            (0..<8).map { _ in KVCacheSimple() as any KVCache }
            + (0..<40).map { _ in RotatingKVCache(maxSize: 1_024) as any KVCache }
        let after: [any KVCache] =
            (0..<8).map { _ in TurboQuantKVCache(keyBits: 4, valueBits: 4) as any KVCache }
            + (0..<40).map { _ in RotatingKVCache(maxSize: 1_024) as any KVCache }

        let snapshot = TurboQuantCacheTransitionSnapshot(before: before, after: after)

        #expect(snapshot.before.layerCount == 48)
        #expect(snapshot.before.kvLayerCount == 8)
        #expect(snapshot.before.turboQuantKVLayerCount == 0)
        #expect(snapshot.before.rotatingKVLayerCount == 40)
        #expect(snapshot.after.layerCount == 48)
        #expect(snapshot.after.kvLayerCount == 0)
        #expect(snapshot.after.turboQuantKVLayerCount == 8)
        #expect(snapshot.after.rotatingKVLayerCount == 40)
        #expect(snapshot.convertedTurboQuantKVLayerCount == 8)
    }

    @Test("completion info retains the exact transition snapshot")
    func completionInfoRetainsTransition() {
        let before = ModelCacheTopologySnapshot(
            layerCount: 48,
            kvLayerCount: 8,
            rotatingKVLayerCount: 40
        )
        let after = ModelCacheTopologySnapshot(
            layerCount: 48,
            turboQuantKVLayerCount: 8,
            rotatingKVLayerCount: 40
        )
        let transition = TurboQuantCacheTransitionSnapshot(before: before, after: after)

        let info = GenerateCompletionInfo(
            promptTokenCount: 512,
            generationTokenCount: 32,
            promptTime: 1,
            generationTime: 1,
            turboQuantCompressions: 1,
            turboQuantCacheTransition: transition
        )

        #expect(info.turboQuantCompressions == 1)
        #expect(info.turboQuantCacheTransition == transition)
    }

    @Test("transition snapshot round-trips through Codable")
    func codableRoundTrip() throws {
        let transition = TurboQuantCacheTransitionSnapshot(
            before: ModelCacheTopologySnapshot(
                layerCount: 48,
                kvLayerCount: 8,
                rotatingKVLayerCount: 40
            ),
            after: ModelCacheTopologySnapshot(
                layerCount: 48,
                turboQuantKVLayerCount: 8,
                rotatingKVLayerCount: 40
            )
        )

        let data = try JSONEncoder().encode(transition)
        let decoded = try JSONDecoder().decode(
            TurboQuantCacheTransitionSnapshot.self,
            from: data
        )

        #expect(decoded == transition)
    }
}
