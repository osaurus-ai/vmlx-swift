// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Cmlx
import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXEmbedders

/// A quantization that MLX cannot apply makes the embedder loader throw. MLX raises on a combination
/// of mode, group size and bits that the mode does not support, and on a width that the group size
/// does not divide. The default error handler then ends the process. One real case: an nvfp4
/// checkpoint whose config.json has no "mode", so the loader asks for affine quantization at group
/// size 16.
struct EmbedderQuantizationConfigTests {
    @Test("A quantization that MLX cannot apply makes the loader throw, not end the process")
    func unsupportedQuantizationThrows() async {
        // A child process, so that the failure before the fix fails this test, not the test run.
        // Exit codes: 3, the loader accepted the configuration; 4, the error does not name the layer
        // and the group size; 5, the test's own setup is wrong. A signal is the process ending.
        let result = await #expect(processExitsWith: .success, observing: [\.standardErrorContent])
        {
            let directory = FileManager.default.temporaryDirectory
                .appending(component: "embedder-quantization-\(UUID().uuidString)")
            try FileManager.default.createDirectory(
                at: directory, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: directory) }
            let config = #"""
                {"model_type": "bert", "hidden_size": 32, "num_attention_heads": 2,
                 "intermediate_size": 64, "num_hidden_layers": 1, "vocab_size": 64,
                 "max_position_embeddings": 32, "type_vocab_size": 2,
                 "quantization": {"group_size": 16, "bits": 4}}
                """#
            try Data(config.utf8).write(to: directory.appending(component: "config.json"))

            // One Linear layer carries scales, so the loader quantizes it.
            let model = try ModelType(rawValue: "bert").createModel(
                configuration: Data(config.utf8))
            guard let path = model.leafModules().flattened().first(where: { $0.1 is Linear })?.0
            else { exit(5) }  // the test's own setup is wrong
            let weights = ["\(path).scales": MLXArray.zeros([2, 2])]
            guard model.sanitize(weights: weights).keys.contains("\(path).scales") else { exit(5) }
            try MLX.save(arrays: weights, url: directory.appending(component: "model.safetensors"))

            do {
                _ = try loadSynchronous(modelDirectory: directory, modelName: "quantization-test")
                exit(3)  // the loader accepted group size 16 for affine quantization
            } catch let error as EmbedderError {
                let text = error.errorDescription ?? ""
                // The message must name the layer and the group size.
                guard text.contains(path), text.contains("16") else { exit(4) }
            }
        }
        if let result, !Self.exitedCleanly(result.exitStatus) {
            let stderr = String(decoding: result.standardErrorContent, as: UTF8.self)
            Issue.record("the child process's stderr ends: \(stderr.suffix(600))")
        }
    }

    @Test("The check refuses exactly what MLX's quantize refuses")
    func checkAgreesWithMLX() {
        var disagreements: [String] = []
        for mode in [QuantizationMode.affine, .mxfp4, .mxfp8, .nvfp4] {
            for groupSize in [8, 16, 32, 64, 128, 256] {
                for bits in 0 ... 9 {
                    for columns in [48, 64, 96, 256] {
                        let check =
                            quantizationProblem(
                                groupSize: groupSize, bits: bits, mode: mode, columns: columns)
                            == nil
                        let mlx = Self.mlxAccepts(
                            groupSize: groupSize, bits: bits, mode: mode, columns: columns)
                        if check != mlx {
                            disagreements.append(
                                "\(mode) group \(groupSize) bits \(bits) width \(columns): "
                                    + "check \(check), MLX \(mlx)")
                        }
                    }
                }
            }
        }
        #expect(disagreements.isEmpty, "\(disagreements.prefix(8))")
    }

    /// Whether MLX's quantize accepts the parameters. It calls mlx-c directly: the Swift wrapper
    /// reads the first result even when the call failed.
    static func mlxAccepts(groupSize: Int, bits: Int, mode: QuantizationMode, columns: Int) -> Bool
    {
        let w = MLXArray.zeros([2, columns])
        var result = mlx_vector_array_new()
        defer { mlx_vector_array_free(result) }
        do {
            try withError {
                _ = mlx_quantize(
                    &result, w.ctx,
                    mlx_optional_int(value: Int32(groupSize), has_value: true),
                    mlx_optional_int(value: Int32(bits), has_value: true),
                    mode.rawValue, MLXArray.mlxNone.ctx, StreamOrDevice.default.ctx)
            }
        } catch {
            return false
        }
        return mlx_vector_array_size(result) > 0
    }

    static func exitedCleanly(_ status: ExitStatus) -> Bool {
        if case .exitCode(0) = status { return true }
        return false
    }
}
