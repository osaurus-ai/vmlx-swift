// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Cmlx
import Foundation
import MLX
@testable import MLXLMCommon
import Testing

/// A disk-cache fetch must hand back MATERIALIZED arrays.
///
/// `loadArraysAndMetadata` returns lazy `Load` nodes on the CPU stream, one
/// `pread` each. When the engine's restore graph (GPU `asType`, owned copies,
/// SSM seating) is evaluated over them, MLX encodes the GPU side with a
/// cross-stream wait on the read, so a single Metal command buffer spans the
/// SSD read. These tests check materialization, not command-buffer duration
/// or the cause of a particular GPU watchdog failure.
/// Materializing in `fetch` (CPU stream, no GPU encoder) keeps the read out
/// of every later command buffer.
///
/// The finite screen in `DiskCache.fetch` already forced the floating-point
/// tensors, so the property is asserted on an INTEGER tensor too — the one
/// kind the old code left lazy — and on the SSM companion store, which has
/// no screen at all.
@Suite("Disk-cache fetches return materialized arrays")
struct DiskCacheFetchMaterializationTests {

    private static func isAvailable(_ array: MLXArray) -> Bool {
        var value = false
        guard _mlx_array_is_available(&value, array.ctx) == 0 else { return false }
        return value
    }

    @Test("DiskCache.fetch materializes every tensor, integer ones included")
    func kvPayloadIsMaterializedBeforeFetchReturns() async throws {
        try await MLXMetalTestLock.withLock {
            let dir = FileManager.default.temporaryDirectory
                .appendingPathComponent("vmlx-disk-fetch-materialize-\(UUID())")
            defer { try? FileManager.default.removeItem(at: dir) }
            let modelKey = "disk-fetch-materialize-model"
            let tokens = [3, 1, 4, 1, 5, 9, 2, 6]
            let arrays: [String: MLXArray] = [
                "keys": MLXArray.ones([2, 4, 8], dtype: .float16),
                "values": MLXArray.zeros([2, 4, 8], dtype: .float16),
                // Not screened for NaN/Inf: only an explicit materialization
                // makes this one available.
                "offset": MLXArray([Int32(8)]),
            ]
            let expectedBytes = arrays.values.reduce(Int64(0)) { $0 + Int64($1.nbytes) }

            do {
                let writer = DiskCache(cacheDir: dir, maxSizeGB: 0.1, modelKey: modelKey)
                writer.store(tokens: tokens, arrays: arrays)
                #expect(writer.waitForPendingWrites())
            }

            // A fresh instance has no in-flight entry to serve from memory:
            // this fetch reads the file.
            let reader = DiskCache(cacheDir: dir, maxSizeGB: 0.1, modelKey: modelKey)
            let fetched = try #require(reader.fetch(tokens: tokens))
            #expect(fetched.keys.sorted() == ["keys", "offset", "values"])
            let lazy = fetched.keys.sorted().filter { !Self.isAvailable(fetched[$0]!) }
            #expect(
                lazy.isEmpty,
                "fetch returned lazy Load tensors \(lazy); the restore's GPU graph would wait on the disk read")

            let stats = reader.snapshotStats()
            #expect(stats.lastFetchPayloadBytes == expectedBytes)
            #expect(stats.lastFetchMs > 0)
            #expect(fetched["offset"]?.item(Int32.self) == 8)
        }
    }

    @Test("a miss leaves the last-fetch figures untouched")
    func missDoesNotRecordAFetch() async throws {
        try await MLXMetalTestLock.withLock {
            let dir = FileManager.default.temporaryDirectory
                .appendingPathComponent("vmlx-disk-fetch-miss-\(UUID())")
            defer { try? FileManager.default.removeItem(at: dir) }
            let cache = DiskCache(cacheDir: dir, maxSizeGB: 0.1, modelKey: "disk-fetch-miss")
            #expect(cache.fetch(tokens: [7, 7, 7]) == nil)
            let stats = cache.snapshotStats()
            #expect(stats.lastFetchPayloadBytes == 0)
            #expect(stats.lastFetchMs == 0)
        }
    }

    @Test("SSMCompanionDiskStore.fetch materializes every state")
    func companionStatesAreMaterializedBeforeFetchReturns() async throws {
        try await MLXMetalTestLock.withLock {
            let dir = FileManager.default.temporaryDirectory
                .appendingPathComponent("vmlx-ssm-fetch-materialize-\(UUID())")
            defer { try? FileManager.default.removeItem(at: dir) }
            let modelKey = "ssm-fetch-materialize-model"
            let tokens = [11, 12, 13, 14, 15]
            let states = [MLXArray.ones([2, 64], dtype: .float32), MLXArray.zeros([2, 16, 16], dtype: .float32)]

            do {
                let writer = try SSMCompanionDiskStore(cacheDir: dir, modelKey: modelKey, maxBytes: 10_000_000)
                try writer.store(ssmStates: states, tokens: tokens, boundary: tokens.count)
            }

            let reader = try SSMCompanionDiskStore(cacheDir: dir, modelKey: modelKey, maxBytes: 10_000_000)
            let fetched = try #require(reader.fetch(tokens: tokens, boundary: tokens.count))
            #expect(fetched.states.count == states.count)
            #expect(fetched.isComplete)
            let lazy = fetched.states.indices.filter { !Self.isAvailable(fetched.states[$0]) }
            #expect(
                lazy.isEmpty,
                "companion fetch returned lazy states at \(lazy); restoreSSMStates would seat them with GPU copies waiting on the read")
            #expect(fetched.states[0].sum().item(Float.self) == 128)
        }
    }

    @Test("a rejected payload does not replace the last served fetch sample")
    func rejectedPayloadPreservesLastServedSample() async throws {
        try await MLXMetalTestLock.withLock {
            let dir = FileManager.default.temporaryDirectory
                .appendingPathComponent("vmlx-disk-fetch-rejected-\(UUID())")
            defer { try? FileManager.default.removeItem(at: dir) }
            let modelKey = "disk-fetch-rejected"
            let goodTokens = [1, 2, 3]
            let badTokens = [4, 5, 6]
            let cache = DiskCache(cacheDir: dir, maxSizeGB: 0.1, modelKey: modelKey)
            cache.store(tokens: goodTokens, arrays: ["state": MLXArray.ones([8])])
            cache.store(tokens: badTokens, arrays: ["state": MLXArray.ones([16])])
            #expect(cache.waitForPendingPayloadRelease())
            #expect(cache.fetch(tokens: goodTokens) != nil)
            let accepted = cache.snapshotStats()
            let hash = DiskCache.hashTokens(badTokens, modelKey: modelKey, mediaSalt: nil)
            try MLX.save(
                arrays: ["state": MLXArray([Float.infinity] + Array(repeating: Float(0), count: 15))],
                url: dir.appendingPathComponent("\(hash).safetensors"))
            #expect(cache.fetch(tokens: badTokens) == nil)
            #expect(cache.refusedNonFiniteFetches == 1)
            let rejected = cache.snapshotStats()
            #expect(rejected.lastFetchPayloadBytes == accepted.lastFetchPayloadBytes)
            #expect(rejected.lastFetchMs == accepted.lastFetchMs)
        }
    }
}
