// Copyright © 2026 Jinho Jang. All rights reserved.

import Foundation
import MLX
import Testing
@testable import MLXLLM

@Suite("LLMModelFactory startup autodetect")
struct LLMModelFactoryStartupAutodetectTests {
    @Test("JANGTQ sidecar corrects startup metadata without jang_config")
    func jangtqSidecarCorrectsMetadataWithoutJangConfig() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("LLMFactorySidecarAutodetect-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let sidecar = directory.appending(component: "jangtq_runtime.safetensors")
        try MLX.save(
            arrays: ["codebook.128.4": MLXArray([0.0, 1.0] as [Float])],
            metadata: ["format": "mlx"],
            url: sidecar)

        let config = Data(
            """
            {
              "model_type": "minimax_m2",
              "weight_format": "bf16",
              "text_config": {
                "model_type": "minimax_m2_text"
              }
            }
            """.utf8)

        let merged = LLMModelFactory.mergeJANGTQSidecarStartupMetadata(
            config,
            modelDirectory: directory)
        let object = try #require(
            JSONSerialization.jsonObject(with: merged) as? [String: Any])
        let textConfig = try #require(object["text_config"] as? [String: Any])

        #expect(object["weight_format"] as? String == "mxtq")
        #expect(object["mxtq_bits"] as? Int == 4)
        #expect(object["routed_expert_bits"] as? Int == 4)
        #expect(textConfig["weight_format"] as? String == "mxtq")
        #expect(textConfig["mxtq_bits"] as? Int == 4)
        #expect(textConfig["routed_expert_bits"] as? Int == 4)
    }
}
