// Gemma4 VLM Tests — maskedScatter fix validation and config parsing
//
// Tests the fixes for the Range crash in maskedScatter when vision feature count
// doesn't match image token count in the text.

import Foundation
import MLX
import MLXLMCommon
@testable import MLXVLM
import Testing

// MARK: - maskedScatter Unit Tests

/// Replicates the fixed maskedScatter logic for testing
private func maskedScatter(input: MLXArray, mask: MLXArray, source: MLXArray) -> MLXArray? {
    let inputShape = input.shape
    let inputFlat = input.flattened()
    let maskFlat = mask.flattened()
    let sourceFlat = source.flattened()

    let maskValues = maskFlat.asArray(Bool.self)
    let positions = maskValues.enumerated().compactMap { i, v in v ? UInt32(i) : nil }

    guard !positions.isEmpty else { return input }

    let posArray = MLXArray(positions)
    guard sourceFlat.shape[0] == posArray.shape[0] else { return nil }
    inputFlat[posArray] = sourceFlat
    return inputFlat.reshaped(inputShape)
}

@Test func maskedScatterMatchingSizes() {
    MLXMetalTestLock.withLock {
        // 5 tokens, 2 are image tokens, hiddenSize=4
        let input = MLXArray.ones([1, 5, 4])
        let source = MLXArray.zeros([1, 2, 4]) + 42.0

        // Build mask: token positions 1 and 3 are image tokens
        var maskData = [Int32](repeating: 0, count: 5)
        maskData[1] = 1; maskData[3] = 1
        let tokenMask = MLXArray(maskData).reshaped(1, 5).asType(.bool)
        let maskExp = MLX.broadcast(expandedDimensions(tokenMask, axis: -1), to: input.shape)

        let result = maskedScatter(input: input, mask: maskExp, source: source)
        #expect(result != nil, "maskedScatter should succeed with matching sizes")

        if let r = result {
            let flat = r.flattened().asArray(Float.self)
            // token 0 (non-image): should be 1.0
            #expect(flat[0] == 1.0)
            // token 1 (image): should be 42.0
            #expect(flat[4] == 42.0)
            // token 2 (non-image): should be 1.0
            #expect(flat[8] == 1.0)
            // token 3 (image): should be 42.0
            #expect(flat[12] == 42.0)
            // token 4 (non-image): should be 1.0
            #expect(flat[16] == 1.0)
        }
    }
}

@Test func maskedScatterEmptyMask() {
    MLXMetalTestLock.withLock {
        let input = MLXArray.ones([1, 5, 4])
        let source = MLXArray.zeros([1, 2, 4])
        let mask = MLXArray.zeros([1, 5, 4]).asType(.bool)

        let result = maskedScatter(input: input, mask: mask, source: source)
        #expect(result != nil)
        if let r = result {
            let diff = abs(r - input).sum().item(Float.self)
            #expect(diff == 0.0, "Empty mask should return input unchanged")
        }
    }
}

@Test func maskedScatterSizeMismatchDetected() {
    MLXMetalTestLock.withLock {
        // 5 tokens, 3 are image tokens, but source only has 2 features — should fail
        let input = MLXArray.ones([1, 5, 4])
        let source = MLXArray.zeros([1, 2, 4]) + 42.0

        var maskData = [Int32](repeating: 0, count: 5)
        maskData[1] = 1; maskData[2] = 1; maskData[3] = 1  // 3 image positions
        let tokenMask = MLXArray(maskData).reshaped(1, 5).asType(.bool)
        let maskExp = MLX.broadcast(expandedDimensions(tokenMask, axis: -1), to: input.shape)

        // source has 2*4=8 elements but mask has 3*4=12 positions — mismatch
        let result = maskedScatter(input: input, mask: maskExp, source: source)
        #expect(result == nil, "maskedScatter should detect size mismatch")
    }
}

@Test func maskedScatterSingleImageToken() {
    MLXMetalTestLock.withLock {
        // Edge case: exactly 1 image token
        let input = MLXArray.ones([1, 3, 2])
        let source = MLXArray.zeros([1, 1, 2]) + 99.0

        var maskData = [Int32](repeating: 0, count: 3)
        maskData[1] = 1
        let tokenMask = MLXArray(maskData).reshaped(1, 3).asType(.bool)
        let maskExp = MLX.broadcast(expandedDimensions(tokenMask, axis: -1), to: input.shape)

        let result = maskedScatter(input: input, mask: maskExp, source: source)
        #expect(result != nil)
        if let r = result {
            let flat = r.flattened().asArray(Float.self)
            #expect(flat[0] == 1.0)   // token 0
            #expect(flat[2] == 99.0)  // token 1 (image)
            #expect(flat[4] == 1.0)   // token 2
        }
    }
}

@Test func maskedScatterLargeTokenCount() {
    MLXMetalTestLock.withLock {
        // Simulate realistic sizes: 280 image tokens, hidden=16
        let seqLen = 500; let hiddenSize = 16; let numImageTokens = 280
        let input = MLXArray.ones([1, seqLen, hiddenSize])
        let source = MLXArray.zeros([1, numImageTokens, hiddenSize]) + 7.0

        var maskData = [Int32](repeating: 0, count: seqLen)
        for i in 100 ..< (100 + numImageTokens) { maskData[i] = 1 }
        let tokenMask = MLXArray(maskData).reshaped(1, seqLen).asType(.bool)
        let maskExp = MLX.broadcast(expandedDimensions(tokenMask, axis: -1), to: input.shape)

        let result = maskedScatter(input: input, mask: maskExp, source: source)
        #expect(result != nil, "280 image tokens with 280 features should match")

        if let r = result {
            let flat = r.flattened().asArray(Float.self)
            // Non-image position
            #expect(flat[0] == 1.0)
            // First image position
            #expect(flat[100 * hiddenSize] == 7.0)
            // Last image position
            #expect(flat[(100 + numImageTokens - 1) * hiddenSize] == 7.0)
            // After image positions
            #expect(flat[(100 + numImageTokens) * hiddenSize] == 1.0)
        }
    }
}

// MARK: - Config Parsing Tests

@Test func gemma4ConfigDecode() throws {
    let configPath = NSString(string: "~/.cache/huggingface/hub/models--mlx-community--gemma-4-e2b-it-4bit/snapshots/76b6a5af250fa029339a757deeb93716baa8ead0/config.json").expandingTildeInPath
    guard FileManager.default.fileExists(atPath: configPath) else {
        print("SKIP: Gemma4 E2B model not downloaded")
        return
    }
    let data = try Data(contentsOf: URL(fileURLWithPath: configPath))
    let config = try JSONDecoder().decode(Gemma4Configuration.self, from: data)

    #expect(config.imageTokenId == 258880)
    #expect(config.visionConfig.defaultOutputLength == 280)
    #expect(config.visionConfig.poolingKernelSize == 3)
    #expect(config.visionConfig.patchSize == 16)
    #expect(config.textConfig.numHiddenLayers == 35)
    #expect(config.textConfig.numKvSharedLayers == 20)
    #expect(config.textConfig.slidingWindow == 512)
}

@Test func gemma4ProcessorConfigDecode() throws {
    let configPath = NSString(string: "~/.cache/huggingface/hub/models--mlx-community--gemma-4-e2b-it-4bit/snapshots/76b6a5af250fa029339a757deeb93716baa8ead0/processor_config.json").expandingTildeInPath
    guard FileManager.default.fileExists(atPath: configPath) else {
        print("SKIP: Gemma4 E2B model not downloaded")
        return
    }
    let data = try Data(contentsOf: URL(fileURLWithPath: configPath))
    let config = try JSONDecoder().decode(Gemma4ProcessorConfiguration.self, from: data)

    #expect(config.imageSeqLength == 280)
    #expect(config.patchSize == 16)
    #expect(config.poolingKernelSize == 3)
    #expect(config.maxSoftTokens == 280)
}

@Test func gemma4SanitizeSplitsFusedMoEExpertWeights() throws {
    let json = """
    {
      "model_type": "gemma4",
      "text_config": {
        "hidden_size": 8,
        "intermediate_size": 4,
        "moe_intermediate_size": 4,
        "num_hidden_layers": 1,
        "num_experts": 2,
        "top_k_experts": 1,
        "enable_moe_block": true
      },
      "vision_config": {
        "hidden_size": 8,
        "output_proj_dims": 8,
        "default_output_length": 2
      }
    }
    """
    let config = try JSONDecoder().decode(Gemma4Configuration.self, from: Data(json.utf8))
    let model = Gemma4(config)
    let sanitized = model.sanitize(weights: [
        "model.language_model.layers.0.experts.gate_up_proj": MLXArray.zeros([2, 8, 3]),
        "model.language_model.layers.0.experts.down_proj": MLXArray.zeros([2, 3, 4]),
        "language_model.model.layers.0.experts.gate_up_proj.weight": MLXArray.zeros([2, 8, 3]),
        "language_model.model.layers.0.experts.gate_up_proj.scales": MLXArray.zeros([2, 8, 2]),
        "language_model.model.layers.0.experts.down_proj.weight": MLXArray.zeros([2, 3, 4]),
        "language_model.model.layers.0.experts.down_proj.scales": MLXArray.zeros([2, 3, 1]),
    ])

    #expect(sanitized["language_model.model.layers.0.experts.gate_up_proj"] == nil)
    #expect(sanitized["language_model.model.layers.0.experts.down_proj"] == nil)
    #expect(sanitized["language_model.model.layers.0.experts.gate_up_proj.weight"] == nil)
    #expect(sanitized["language_model.model.layers.0.experts.down_proj.weight"] == nil)
    #expect(sanitized["language_model.model.layers.0.experts.switch_glu.gate_proj"]?.shape == [2, 4, 3])
    #expect(sanitized["language_model.model.layers.0.experts.switch_glu.up_proj"]?.shape == [2, 4, 3])
    #expect(sanitized["language_model.model.layers.0.experts.switch_glu.down_proj"]?.shape == [2, 3, 4])
    #expect(sanitized["language_model.model.layers.0.experts.switch_glu.gate_proj.weight"]?.shape == [2, 4, 3])
    #expect(sanitized["language_model.model.layers.0.experts.switch_glu.up_proj.weight"]?.shape == [2, 4, 3])
    #expect(sanitized["language_model.model.layers.0.experts.switch_glu.gate_proj.scales"]?.shape == [2, 4, 2])
    #expect(sanitized["language_model.model.layers.0.experts.switch_glu.up_proj.scales"]?.shape == [2, 4, 2])
    #expect(sanitized["language_model.model.layers.0.experts.switch_glu.down_proj.weight"]?.shape == [2, 3, 4])
    #expect(sanitized["language_model.model.layers.0.experts.switch_glu.down_proj.scales"]?.shape == [2, 3, 1])
}

@Test func imageSeqLengthMatchesVisionOutput() throws {
    let configPath = NSString(string: "~/.cache/huggingface/hub/models--mlx-community--gemma-4-e2b-it-4bit/snapshots/76b6a5af250fa029339a757deeb93716baa8ead0/config.json").expandingTildeInPath
    let procPath = NSString(string: "~/.cache/huggingface/hub/models--mlx-community--gemma-4-e2b-it-4bit/snapshots/76b6a5af250fa029339a757deeb93716baa8ead0/processor_config.json").expandingTildeInPath
    guard FileManager.default.fileExists(atPath: configPath),
          FileManager.default.fileExists(atPath: procPath) else {
        print("SKIP: Gemma4 E2B model not downloaded")
        return
    }
    let modelConfig = try JSONDecoder().decode(
        Gemma4Configuration.self,
        from: Data(contentsOf: URL(fileURLWithPath: configPath)))
    let procConfig = try JSONDecoder().decode(
        Gemma4ProcessorConfiguration.self,
        from: Data(contentsOf: URL(fileURLWithPath: procPath)))

    // This is the root invariant: processor token count must match vision feature count
    #expect(
        procConfig.imageSeqLength == modelConfig.visionConfig.defaultOutputLength,
        "Processor imageSeqLength (\(procConfig.imageSeqLength)) must equal vision defaultOutputLength (\(modelConfig.visionConfig.defaultOutputLength))")
}

@Test func gemma4E4BConfigDecode() throws {
    let base = NSString(string: "~/.cache/huggingface/hub/models--mlx-community--gemma-4-e4b-it-4bit").expandingTildeInPath
    guard let snapshots = try? FileManager.default.contentsOfDirectory(atPath: base + "/snapshots"),
          let first = snapshots.first else {
        print("SKIP: Gemma4 E4B model not downloaded")
        return
    }
    let configPath = base + "/snapshots/" + first + "/config.json"
    let data = try Data(contentsOf: URL(fileURLWithPath: configPath))
    let config = try JSONDecoder().decode(Gemma4Configuration.self, from: data)

    #expect(config.imageTokenId == 258880)
    #expect(config.visionConfig.defaultOutputLength == 280)
    #expect(config.visionConfig.poolingKernelSize == 3)
}
