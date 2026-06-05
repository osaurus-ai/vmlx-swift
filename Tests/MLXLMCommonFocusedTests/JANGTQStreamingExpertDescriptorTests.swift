// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
@testable import MLXLLM
@testable import MLXLMCommon
import XCTest

final class JANGTQStreamingExpertDescriptorTests: XCTestCase {
    func testNemotronUltraLayersBlockTypeDecodesToHybridPattern() throws {
        let json = """
        {
          "model_type": "nemotron_h",
          "vocab_size": 131072,
          "hidden_size": 8192,
          "num_hidden_layers": 4,
          "num_attention_heads": 64,
          "num_key_value_heads": 2,
          "mamba_num_heads": 256,
          "mamba_head_dim": 64,
          "ssm_state_size": 128,
          "conv_kernel": 4,
          "n_groups": 8,
          "intermediate_size": 5120,
          "moe_intermediate_size": 5120,
          "moe_shared_expert_intermediate_size": 5120,
          "n_routed_experts": 512,
          "num_experts_per_tok": 22,
          "layers_block_type": ["mamba", "attention", "mlp", "moe"],
          "layer_norm_epsilon": 1e-5,
          "n_group": 8,
          "topk_group": 4
        }
        """

        let config = try JSONDecoder().decode(
            NemotronHConfiguration.self,
            from: Data(json.utf8))

        XCTAssertEqual(config.hybridOverridePattern, "M*-E")
        XCTAssertEqual(config.numHiddenLayers, 4)
        XCTAssertEqual(config.numExpertsPerTok, 22)
    }

    func testNemotronUltraStackedFc1AndFc2TensorsAreStreamable() throws {
        let directory = try makeTemporaryModelDirectory()
        try writeSafetensors(
            at: directory.appendingPathComponent("model.safetensors"),
            tensors: [
                "backbone.layers.1.mixer.switch_mlp.fc1.tq_packed": TensorFixture(
                    dtype: "U8", shape: [4, 2, 4], range: 0 ..< 32),
                "backbone.layers.1.mixer.switch_mlp.fc1.tq_norms": TensorFixture(
                    dtype: "F16", shape: [4, 2], range: 32 ..< 48),
                "backbone.layers.1.mixer.switch_mlp.fc2.tq_packed": TensorFixture(
                    dtype: "U8", shape: [4, 2, 4], range: 48 ..< 80),
                "backbone.layers.1.mixer.switch_mlp.fc2.tq_norms": TensorFixture(
                    dtype: "F16", shape: [4, 2], range: 80 ..< 96),
            ])

        XCTAssertTrue(
            JANGTQStreamingExperts.isStreamableRoutedTensorKey(
                "backbone.layers.1.mixer.switch_mlp.fc1.tq_packed"))
        XCTAssertTrue(
            JANGTQStreamingExperts.isStreamableRoutedTensorKey(
                "backbone.layers.1.mixer.switch_mlp.fc2.tq_norms"))
        XCTAssertTrue(JANGTQStreamingExperts.hasStreamableExperts(in: directory))

        let up = JANGTQStreamingExperts.stackedOffsetDescriptor(
            in: directory,
            layerIdx: 1,
            projectionName: "up_proj",
            suffixName: "tq_packed")
        let down = JANGTQStreamingExperts.stackedOffsetDescriptor(
            in: directory,
            layerIdx: 1,
            projectionName: "down_proj",
            suffixName: "tq_packed")

        XCTAssertEqual(up?.storageLayout, "stacked-contiguous")
        XCTAssertEqual(up?.expertCount, 4)
        XCTAssertEqual(up?.spanByteCount, 32)
        XCTAssertEqual(up?.expertByteCount, 8)
        XCTAssertEqual(down?.storageLayout, "stacked-contiguous")
        XCTAssertEqual(down?.expertCount, 4)
    }

    func testModuleUpdateArrayRecursionSkipsSparseNilEntries() throws {
        let moduleSource = try String(
            contentsOfFile: "Source/MLXNN/Module.swift",
            encoding: .utf8)

        XCTAssertTrue(moduleSource.contains("case (_, .none):\n                            continue"))
    }

    func testDescriptorReportsContiguousStackedTensorLayout() throws {
        let directory = try makeTemporaryModelDirectory()
        try writeSafetensors(
            at: directory.appendingPathComponent("model.safetensors"),
            tensors: [
                "model.layers.0.mlp.switch_mlp.gate_proj.tq_packed": TensorFixture(
                    dtype: "U8", shape: [4, 2, 4], range: 0 ..< 32)
            ])

        let descriptor = JANGTQStreamingExperts.stackedOffsetDescriptor(
            in: directory,
            layerIdx: 0,
            projectionName: "gate_proj",
            suffixName: "tq_packed")

        XCTAssertEqual(descriptor?.storageLayout, "stacked-contiguous")
        XCTAssertEqual(descriptor?.expertCount, 4)
        XCTAssertEqual(descriptor?.spanByteCount, 32)
        XCTAssertEqual(descriptor?.expertByteCount, 8)
        XCTAssertEqual(descriptor?.expertByteOffsets, [0, 8, 16, 24])
        XCTAssertEqual(descriptor?.logicalShape, [4, 2, 4])
    }

    func testDescriptorReportsMiniMaxExpertMajorSingleFileOffsets() throws {
        let directory = try makeTemporaryModelDirectory()
        try writeSafetensors(
            at: directory.appendingPathComponent("model.safetensors"),
            tensors: [
                "layers.1.ffn.experts.0.w1.tq_packed": TensorFixture(
                    dtype: "U8", shape: [2, 4], range: 16 ..< 24),
                "layers.1.ffn.experts.1.w1.tq_packed": TensorFixture(
                    dtype: "U8", shape: [2, 4], range: 216 ..< 224),
                "layers.1.ffn.experts.2.w1.tq_packed": TensorFixture(
                    dtype: "U8", shape: [2, 4], range: 116 ..< 124),
            ])

        let descriptor = JANGTQStreamingExperts.stackedOffsetDescriptor(
            in: directory,
            layerIdx: 1,
            projectionName: "gate_proj",
            suffixName: "tq_packed")

        XCTAssertEqual(descriptor?.storageLayout, "expert-major-single-file-offsets")
        XCTAssertEqual(descriptor?.expertCount, 3)
        XCTAssertEqual(descriptor?.spanByteCount, 208)
        XCTAssertEqual(descriptor?.expertByteCount, 8)
        XCTAssertEqual(descriptor?.expertByteOffsets, [0, 200, 100])
        XCTAssertEqual(descriptor?.logicalShape, [3, 2, 4])
    }

    func testDescriptorRejectsSplitShardExpertMajorRanges() throws {
        let directory = try makeTemporaryModelDirectory()
        try writeSafetensors(
            at: directory.appendingPathComponent("model-00001-of-00002.safetensors"),
            tensors: [
                "layers.2.ffn.experts.0.w1.tq_packed": TensorFixture(
                    dtype: "U8", shape: [2, 4], range: 0 ..< 8)
            ])
        try writeSafetensors(
            at: directory.appendingPathComponent("model-00002-of-00002.safetensors"),
            tensors: [
                "layers.2.ffn.experts.1.w1.tq_packed": TensorFixture(
                    dtype: "U8", shape: [2, 4], range: 0 ..< 8)
            ])

        let descriptor = JANGTQStreamingExperts.stackedOffsetDescriptor(
            in: directory,
            layerIdx: 2,
            projectionName: "gate_proj",
            suffixName: "tq_packed")
        let descriptors = JANGTQStreamingExperts.stackedOffsetDescriptors(
            in: directory,
            layerIdx: 2,
            projectionName: "gate_proj",
            suffixName: "tq_packed")

        XCTAssertNil(descriptor)
        XCTAssertEqual(descriptors.count, 2)
        XCTAssertEqual(descriptors.map(\.storageLayout), [
            "expert-major-multi-file-offsets",
            "expert-major-multi-file-offsets",
        ])
        XCTAssertEqual(descriptors[0].expertByteOffsets, [0, UInt64.max])
        XCTAssertEqual(descriptors[1].expertByteOffsets, [UInt64.max, 0])
    }

    func testActiveOffsetWindowsDefaultKeepsOneWindowPerExpert() throws {
        let descriptor = JANGTQStackedOffsetDescriptor(
            layerIdx: 0,
            projectionName: "gate_proj",
            suffixName: "tq_packed",
            fileURL: URL(fileURLWithPath: "/tmp/model.safetensors"),
            spanOffset: 100,
            spanByteCount: 72,
            expertByteCount: 8,
            expertByteOffsets: [0, 8, 32, UInt64.max, 64],
            logicalShape: [5, 2, 4],
            dtype: "U8",
            storageLayout: "expert-major-single-file-offsets")

        let windows = descriptor.activeExpertByteWindows(
            activeExperts: [0, 1, 2, 4],
            elementByteSize: 1)

        XCTAssertEqual(windows.map(\.start), [100, 108, 132, 164])
        XCTAssertEqual(windows.map(\.end), [108, 116, 140, 172])
        XCTAssertEqual(windows.map { $0.experts }, [[0], [1], [2], [4]])
    }

    func testActiveOffsetWindowsCanExplicitlyCoalesceAdjacentExperts() throws {
        let descriptor = JANGTQStackedOffsetDescriptor(
            layerIdx: 0,
            projectionName: "gate_proj",
            suffixName: "tq_packed",
            fileURL: URL(fileURLWithPath: "/tmp/model.safetensors"),
            spanOffset: 100,
            spanByteCount: 72,
            expertByteCount: 8,
            expertByteOffsets: [0, 8, 32, UInt64.max, 64],
            logicalShape: [5, 2, 4],
            dtype: "U8",
            storageLayout: "expert-major-single-file-offsets")

        let windows = descriptor.activeExpertByteWindows(
            activeExperts: [0, 1, 2, 4],
            elementByteSize: 1,
            maxGapBytes: 0)

        XCTAssertEqual(windows.map(\.start), [100, 132, 164])
        XCTAssertEqual(windows.map(\.end), [116, 140, 172])
        XCTAssertEqual(windows.map { $0.experts }, [[0, 1], [2], [4]])
    }

    func testActiveOffsetWindowsCoalesceWithinConfiguredGapBudget() throws {
        let descriptor = JANGTQStackedOffsetDescriptor(
            layerIdx: 0,
            projectionName: "gate_proj",
            suffixName: "tq_packed",
            fileURL: URL(fileURLWithPath: "/tmp/model.safetensors"),
            spanOffset: 100,
            spanByteCount: 72,
            expertByteCount: 8,
            expertByteOffsets: [0, 8, 32, UInt64.max, 64],
            logicalShape: [5, 2, 4],
            dtype: "U8",
            storageLayout: "expert-major-single-file-offsets")

        let windows = descriptor.activeExpertByteWindows(
            activeExperts: [0, 1, 2, 4],
            elementByteSize: 1,
            maxGapBytes: 16)

        XCTAssertEqual(windows.map(\.start), [100, 164])
        XCTAssertEqual(windows.map(\.end), [140, 172])
        XCTAssertEqual(windows.map { $0.experts }, [[0, 1, 2], [4]])
    }

    private func makeTemporaryModelDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        addTeardownBlock {
            try? FileManager.default.removeItem(at: url)
        }
        return url
    }

    private func writeSafetensors(
        at url: URL,
        tensors: [String: TensorFixture]
    ) throws {
        var header: [String: Any] = [:]
        var byteCount = 0
        for (name, fixture) in tensors {
            header[name] = [
                "dtype": fixture.dtype,
                "shape": fixture.shape,
                "data_offsets": [fixture.range.lowerBound, fixture.range.upperBound],
            ]
            byteCount = max(byteCount, fixture.range.upperBound)
        }

        let headerData = try JSONSerialization.data(
            withJSONObject: header,
            options: [.sortedKeys])
        var data = Data()
        var headerLength = UInt64(headerData.count).littleEndian
        withUnsafeBytes(of: &headerLength) { data.append(contentsOf: $0) }
        data.append(headerData)
        data.append(Data(repeating: 0, count: byteCount))
        try data.write(to: url)
    }
}

private struct TensorFixture {
    var dtype: String
    var shape: [Int]
    var range: Range<Int>
}
