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

@Test func gemma4Unified12BConfigDecode() throws {
    let data = Data(#"""
    {
      "model_type": "gemma4_unified",
      "image_token_id": 258880,
      "audio_token_id": 258881,
      "video_token_id": 258884,
      "text_config": {
        "attention_bias": false,
        "attention_k_eq_v": true,
        "enable_moe_block": false,
        "final_logit_softcapping": 30.0,
        "global_head_dim": 512,
        "head_dim": 256,
        "hidden_size": 3840,
        "hidden_size_per_layer_input": 0,
        "intermediate_size": 15360,
        "layer_types": [
          "sliding_attention", "sliding_attention", "sliding_attention",
          "sliding_attention", "sliding_attention", "full_attention",
          "sliding_attention", "sliding_attention", "sliding_attention",
          "sliding_attention", "sliding_attention", "full_attention",
          "sliding_attention", "sliding_attention", "sliding_attention",
          "sliding_attention", "sliding_attention", "full_attention",
          "sliding_attention", "sliding_attention", "sliding_attention",
          "sliding_attention", "sliding_attention", "full_attention",
          "sliding_attention", "sliding_attention", "sliding_attention",
          "sliding_attention", "sliding_attention", "full_attention",
          "sliding_attention", "sliding_attention", "sliding_attention",
          "sliding_attention", "sliding_attention", "full_attention",
          "sliding_attention", "sliding_attention", "sliding_attention",
          "sliding_attention", "sliding_attention", "full_attention",
          "sliding_attention", "sliding_attention", "sliding_attention",
          "sliding_attention", "sliding_attention", "full_attention"
        ],
        "model_type": "gemma4_unified_text",
        "moe_intermediate_size": null,
        "num_attention_heads": 16,
        "num_experts": null,
        "num_global_key_value_heads": 1,
        "num_hidden_layers": 48,
        "num_key_value_heads": 8,
        "num_kv_shared_layers": 0,
        "rms_norm_eps": 1e-6,
        "rope_parameters": {
          "full_attention": {
            "partial_rotary_factor": 0.25,
            "rope_theta": 1000000.0,
            "rope_type": "proportional"
          },
          "sliding_attention": {
            "rope_theta": 10000.0,
            "rope_type": "default"
          }
        },
        "sliding_window": 1024,
        "tie_word_embeddings": true,
        "top_k_experts": null,
        "use_double_wide_mlp": false,
        "vocab_size": 262144,
        "vocab_size_per_layer_input": 262144
      },
      "vision_config": {
        "mm_embed_dim": 3840,
        "mm_posemb_size": 1120,
        "model_patch_size": 48,
        "model_type": "gemma4_unified_vision",
        "num_soft_tokens": 280,
        "output_proj_dims": 3840,
        "patch_size": 16,
        "pooling_kernel_size": 3,
        "rms_norm_eps": 1e-6
      }
    }
    """#.utf8)

    let config = try JSONDecoder().decode(Gemma4Configuration.self, from: data)

    #expect(config.modelType == "gemma4_unified")
    #expect(config.imageTokenId == 258880)
    #expect(config.audioTokenId == 258881)
    #expect(config.audioEmbedDim == 640)
    #expect(config.visionSoftTokensPerImage == 280)
    #expect(config.textConfig.numHiddenLayers == 48)
    #expect(config.textConfig.enableMoeBlock == false)
    #expect(config.textConfig.numExperts == 0)
    #expect(config.textConfig.numKeyValueHeads == 8)
    #expect(config.textConfig.numGlobalKeyValueHeads == 1)
    #expect(config.textConfig.globalHeadDim == 512)
    #expect(config.textConfig.slidingWindow == 1024)
    #expect(config.textConfig.layerTypes.filter { $0 == "full_attention" }.count == 8)
    #expect(config.textConfig.layerTypes.filter { $0 == "sliding_attention" }.count == 40)
    #expect(config.visionConfig.usesUnifiedVisionEmbedder)
    #expect(config.visionConfig.hiddenSize == 3840)
    #expect(config.visionConfig.outputProjectionDimensions == 3840)
    #expect(config.visionConfig.modelPatchSize == 48)
    #expect(config.visionConfig.positionEmbeddingSize == 1120)
    #expect(config.visionConfig.defaultOutputLength == 280)
}

@Test func gemma4ProportionalRoPEHandlesAsymmetricGlobalKeyWidth() {
    MLXMetalTestLock.withLock {
        let rope = ProportionalRoPE(
            dimensions: 512,
            base: 1_000_000,
            scalingConfig: [
                "rope_type": .string("proportional"),
                "partial_rotary_factor": .float(0.5),
            ])

        let query = MLXArray.ones([1, 8, 4, 512])
        let key = MLXArray.ones([1, 2, 4, 256])

        let rotatedQuery = rope(query, offset: 0)
        let rotatedKey = rope(key, offset: 0)
        eval(rotatedQuery, rotatedKey)

        #expect(rotatedQuery.shape == query.shape)
        #expect(rotatedKey.shape == key.shape)
    }
}

@Test func gemma4Unified12BAudioConfigDecode() throws {
    let data = Data(#"""
    {
      "model_type": "gemma4_unified",
      "image_token_id": 258880,
      "audio_token_id": 258881,
      "audio_config": {
        "model_type": "gemma4_unified_audio",
        "audio_embed_dim": 640,
        "hidden_size": 640,
        "output_proj_dims": 640
      },
      "text_config": {
        "model_type": "gemma4_unified_text",
        "hidden_size": 3840,
        "num_hidden_layers": 48,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "num_global_key_value_heads": 1,
        "head_dim": 256,
        "global_head_dim": 512,
        "intermediate_size": 15360,
        "vocab_size": 262144
      },
      "vision_config": {
        "model_type": "gemma4_unified_vision",
        "output_proj_dims": 3840
      }
    }
    """#.utf8)

    let config = try JSONDecoder().decode(Gemma4Configuration.self, from: data)

    #expect(config.audioTokenId == 258881)
    #expect(config.audioEmbedDim == 640)
}

@Test func gemma4UnifiedProcessorConfigDecode() throws {
    let data = Data(#"""
    {
      "processor_class": "Gemma4UnifiedProcessor",
      "image_processor": {
        "image_processor_type": "Gemma4UnifiedImageProcessor",
        "max_soft_tokens": 280,
        "patch_size": 16,
        "pooling_kernel_size": 3
      },
      "image_seq_length": 280,
      "audio_seq_length": 750,
      "video_processor": {
        "video_processor_type": "Gemma4UnifiedVideoProcessor",
        "max_soft_tokens": 70,
        "patch_size": 16,
        "pooling_kernel_size": 3
      }
    }
    """#.utf8)

    let config = try JSONDecoder().decode(Gemma4ProcessorConfiguration.self, from: data)

    #expect(config.processorClass == "Gemma4UnifiedProcessor")
    #expect(config.patchSize == 16)
    #expect(config.poolingKernelSize == 3)
    #expect(config.maxSoftTokens == 280)
    #expect(config.imageSeqLength == 280)
    #expect(config.audioSeqLength == 750)
}

@Test func gemma4ProcessorExpandsPreEncodedAudioSoftTokens() throws {
    let source = try String(
        contentsOfFile: "Libraries/MLXVLM/Models/Gemma4.swift",
        encoding: .utf8)

    #expect(source.contains("case .preEncoded(let samples, let sr, let embedding):"))
    #expect(source.contains("tokenCounts.append(embedding.dim(-2))"))
    #expect(source.contains("let audioId = tokenizer.convertTokenToId(\"<|audio|>\") ?? 258881"))
    #expect(source.contains("let beginAudioId = tokenizer.convertTokenToId(\"<|audio>\")"))
    #expect(source.contains("let endAudioId = tokenizer.convertTokenToId(\"<audio|>\")"))
    #expect(source.contains("exp.append(contentsOf: Array(repeating: audioId, count: tokenCount))"))
    #expect(source.contains("audio: processedAudio"))
    #expect(source.contains("preEncodedEmbedding: embedding"))
    #expect(source.contains("Gemma4 raw audio feature extraction is not implemented"))
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
