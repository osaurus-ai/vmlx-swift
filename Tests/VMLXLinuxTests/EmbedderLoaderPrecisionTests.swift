// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXEmbedders

/// The float32 rule through the real loader, differentially: a tiny BERT checkpoint saved in
/// bfloat16, and the same values saved as float32. Where the rule applies (a CPU backend without a
/// fast low-precision path: a Linux CPU build today), both must load as float32 and embed
/// bit-identically; elsewhere, Apple platforms included, the bfloat16 checkpoint stays bfloat16.
@Suite(.serialized) struct EmbedderLoaderPrecisionTests {
    static let config = #"""
        {"model_type": "bert", "hidden_size": 32, "num_attention_heads": 2,
         "intermediate_size": 64, "num_hidden_layers": 2, "vocab_size": 64,
         "max_position_embeddings": 32, "type_vocab_size": 2}
        """#

    /// The test checks the default rule, so it cannot run while the override is set; it does not
    /// edit the process environment, which every other test shares.
    static let overrideIsSet = CPUPrecisionPolicy.nativeDTypeRequested(
        environment: ProcessInfo.processInfo.environment)

    static func writeCheckpoint(_ weights: [String: MLXArray], to directory: URL) throws {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try Data(config.utf8).write(to: directory.appending(component: "config.json"))
        try save(arrays: weights, url: directory.appending(component: "model.safetensors"))
    }

    @Test(
        .disabled(
            if: overrideIsSet,
            "VMLX_CPU_NATIVE_DTYPE is set; this test checks the default rule"
        ))
    func bf16CheckpointFollowsTheRule() throws {
        let root = FileManager.default.temporaryDirectory
            .appending(component: "vmlx-loader-precision-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }

        let seed = BertModel(
            try JSONDecoder().decode(BertConfiguration.self, from: Data(Self.config.utf8)))
        let rounded = Dictionary(
            uniqueKeysWithValues: seed.parameters().flattened().map {
                ($0.0, $0.1.asType(.bfloat16))
            })
        let bf16Directory = root.appending(component: "bf16")
        let f32Directory = root.appending(component: "f32")
        try Self.writeCheckpoint(rounded, to: bf16Directory)
        try Self.writeCheckpoint(rounded.mapValues { $0.asType(.float32) }, to: f32Directory)

        let fromBF16 = try loadSynchronous(modelDirectory: bf16Directory, modelName: "tiny-bf16")
        let dtypes = Set(fromBF16.parameters().flattened().map { $0.1.dtype })

        let ruleApplies = CPUPrecisionPolicy.shouldWiden(
            buildHasFastDenseLowPrecision: CPUPrecisionPolicy.buildHasFastDenseLowPrecision,
            defaultDevice: Device.defaultDevice().deviceType, nativeDTypeRequested: false)
        #if os(Linux) && !VMLX_CPU_FAST_DENSE_LOW_PRECISION
            // A CPU-only Linux build without VMLX_CPU_FAST_DENSE_LOW_PRECISION is exactly where the
            // rule applies.
            #expect(ruleApplies)
        #endif
        if ruleApplies {
            #expect(dtypes == [.float32])
            let fromF32 = try loadSynchronous(modelDirectory: f32Directory, modelName: "tiny-f32")
            let ids = MLXArray([2, 7, 11, 3] as [Int32]).reshaped(1, 4)
            let a = try #require(
                fromBF16(ids, positionIds: nil, tokenTypeIds: nil, attentionMask: nil).pooledOutput)
            let b = try #require(
                fromF32(ids, positionIds: nil, tokenTypeIds: nil, attentionMask: nil).pooledOutput)
            #expect(a.dtype == .float32)
            #expect(a.asArray(Float.self) == b.asArray(Float.self))
        } else {
            #if os(Linux) && !VMLX_CPU_FAST_DENSE_LOW_PRECISION
                Issue.record("the rule must apply on a CPU-only Linux build")
            #else
                // Nothing widens on Apple platforms.
                #expect(!ruleApplies)
            #endif
            #expect(dtypes == [.bfloat16])
        }
    }
}
