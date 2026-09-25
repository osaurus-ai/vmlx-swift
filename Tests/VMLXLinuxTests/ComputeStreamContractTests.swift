// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import Testing

/// The ten cache-path call sites now call `synchronizeComputeStream()`, so the four literals in
/// `BatchEngineGrowingChatCacheSourceTests` that read them no longer show which stream is
/// synchronized. This reads the helper instead and checks which branch holds which call: the
/// `#if os(Linux)` branch synchronizes the default device's stream, and the `#else` branch, which
/// Apple platforms compile, still synchronizes the GPU stream.
@Suite struct ComputeStreamContractTests {
    @Test func helperSynchronizesTheGPUStreamOnApplePlatforms() throws {
        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let source = try String(
            contentsOf: root.appendingPathComponent("Libraries/MLXLMCommon/ComputeStream.swift"),
            encoding: .utf8)
        let ifLinux = try #require(source.range(of: "#if os(Linux)"))
        let orElse = try #require(
            source.range(of: "#else", range: ifLinux.upperBound ..< source.endIndex))
        let endIf = try #require(
            source.range(of: "#endif", range: orElse.upperBound ..< source.endIndex))
        let linuxBranch = source[ifLinux.upperBound ..< orElse.lowerBound]
        let elseBranch = source[orElse.upperBound ..< endIf.lowerBound]
        #expect(linuxBranch.contains("Stream.defaultStream(Device.defaultDevice()).synchronize()"))
        #expect(elseBranch.contains("Stream.gpu.synchronize()"))
    }
}
