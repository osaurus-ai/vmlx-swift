// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Do SPAWNED AGENTS reuse the prefill cache?
//
// The concern is real-looking: `TextSubagentKind` builds a fresh session id
// for every spawn —
//
//     let sessionId = "spawn-\(agentId)-\(UUID().uuidString)"
//
// so no two spawns ever share one. If disk reuse were keyed on the session,
// every spawned agent would re-prefill its entire system prompt from scratch,
// every time, forever — and it would look fine, because each run individually
// produces correct output. Only the wall-clock would suffer.
//
// It is not session-keyed. `DiskCache.hashTokens(tokens, modelKey:,
// mediaSalt:)` is content-addressed: identical token prefixes on the same
// model collide by construction regardless of who asks. These tests pin that,
// because it is the property the spawn path silently depends on.

import Foundation
import MLX

@testable import MLXLMCommon
import Testing

@Suite("Spawned agents reuse the prefill cache", .serialized)
struct SpawnedAgentPrefixReuseTests {

    private func tempDir() -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("spawn-reuse-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    /// A shared agent system prompt, as two different spawns would tokenize it.
    private let sharedPrefix = Array(1...64)

    /// THE test: spawn A stores, spawn B — a different session entirely —
    /// fetches the same prefix and hits.
    @Test("a second spawn hits the entry the first one stored")
    func secondSpawnHitsTheFirstSpawnsEntry() throws {
        try MLXMetalTestLock.withLock {
            let dir = tempDir()
            defer { try? FileManager.default.removeItem(at: dir) }

            // Two DiskCache instances standing in for two spawns of the same
            // agent. They share only the cache root and the model key — never
            // a session, because spawns never do.
            let spawnA = DiskCache(cacheDir: dir, maxSizeBytes: 64 * 1_048_576, modelKey: "agent-model")
            spawnA.store(tokens: sharedPrefix, arrays: ["k": MLXArray.zeros([4096])])

            let spawnB = DiskCache(cacheDir: dir, maxSizeBytes: 64 * 1_048_576, modelKey: "agent-model")
            let hit = spawnB.fetch(tokens: sharedPrefix)

            #expect(
                hit != nil,
                "a spawned agent re-prefilled a prefix an earlier spawn had already cached")
        }
    }

    /// The property that makes the above work, stated directly: the key does
    /// not contain a session. If a session ever enters it, every spawn becomes
    /// a cold prefill and nothing fails loudly.
    @Test("the cache key is content-addressed, not session-addressed")
    func keyIsContentAddressedNotSessionAddressed() {
        let a = DiskCache.hashTokens(sharedPrefix, modelKey: "agent-model", mediaSalt: nil)
        let b = DiskCache.hashTokens(sharedPrefix, modelKey: "agent-model", mediaSalt: nil)
        #expect(a == b, "identical prefixes on one model must produce one key")

        // And it DOES separate the things that must not collide.
        let otherModel = DiskCache.hashTokens(
            sharedPrefix, modelKey: "different-model", mediaSalt: nil)
        #expect(a != otherModel, "two models would share a cache entry")

        let otherPrefix = DiskCache.hashTokens(
            Array(100...163), modelKey: "agent-model", mediaSalt: nil)
        #expect(a != otherPrefix, "different prompts would share a cache entry")
    }

    /// Different agents on the same model still share a common prefix — the
    /// shared preamble every spawn sends before its own instructions. Reuse
    /// must survive the suffix diverging.
    @Test("spawns sharing a preamble reuse it even when their tails differ")
    func sharedPreambleIsReusedAcrossDifferentAgents() throws {
        try MLXMetalTestLock.withLock {
            let dir = tempDir()
            defer { try? FileManager.default.removeItem(at: dir) }

            let preamble = Array(1...48)
            let researcher = preamble + Array(900...910)
            let reviewer = preamble + Array(800...812)

            let a = DiskCache(cacheDir: dir, maxSizeBytes: 64 * 1_048_576, modelKey: "m")
            a.store(tokens: preamble, arrays: ["k": MLXArray.zeros([2048])])
            a.store(tokens: researcher, arrays: ["k": MLXArray.zeros([2048])])

            let b = DiskCache(cacheDir: dir, maxSizeBytes: 64 * 1_048_576, modelKey: "m")
            // The reviewer's own full prompt is new...
            #expect(b.fetch(tokens: reviewer) == nil, "a prompt never seen must miss")
            // ...but the preamble both agents share is already there.
            #expect(
                b.fetch(tokens: preamble) != nil,
                "the shared preamble was not reusable across agents")
        }
    }

    /// Media salt must still separate entries, or a spawn that attached an
    /// image could collide with a text-only spawn whose tokens matched.
    @Test("media salt still separates otherwise identical prefixes")
    func mediaSaltSeparatesEntries() {
        let plain = DiskCache.hashTokens(sharedPrefix, modelKey: "m", mediaSalt: nil)
        let withImage = DiskCache.hashTokens(sharedPrefix, modelKey: "m", mediaSalt: "img-abc")
        #expect(plain != withImage, "same tokens with different media shared a key")
    }
}
