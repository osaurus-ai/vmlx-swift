// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import MLXVLM
import XCTest

/// Smoke tests for the Nemotron-3-Nano-Omni multimodal Swift wrapper.
///
/// These exercise individual building blocks (projectors, ViT block, conformer
/// block, mel STFT, NVLM tiling) without loading the full 30B bundle so they
/// run quickly in CI. End-to-end validation against a real bundle is gated
/// behind the env var `BENCH_NEMOTRON_OMNI_BUNDLE` and lives in the bench
/// harness, not in unit tests.
final class NemotronHOmniSmokeTests: XCTestCase {

    func testProjectorsForwardShapes() throws {
        let mlp = NemotronHVisionMLPProjector(inDim: 5120, projectorDim: 20480, llmDim: 2688)
        let inX = MLXArray.zeros([2, 256, 5120])
        let outX = mlp(inX)
        XCTAssertEqual(outX.shape, [2, 256, 2688])

        let sp = NemotronHSoundProjector(soundHidden: 1024, projectionHidden: 4096, llmHidden: 2688)
        let inA = MLXArray.zeros([1, 64, 1024])
        let outA = sp(inA)
        XCTAssertEqual(outA.shape, [1, 64, 2688])
    }

    func testVisionProjectorUsesBundleRMSNormAndSquaredReLUContract() throws {
        let projector = NemotronHVisionMLPProjector(
            inDim: 2, projectorDim: 2, llmDim: 1)
        var parameters = ModuleParameters()
        parameters["layer_norm"] = .dictionary([
            "weight": .value(MLXArray.ones([2])),
        ])
        parameters["fc1"] = .dictionary([
            "weight": .value(MLXArray([Float(1), 0, 0, 1], [2, 2])),
        ])
        parameters["fc2"] = .dictionary([
            "weight": .value(MLXArray([Float(1), 1], [1, 2])),
        ])
        try projector.update(parameters: parameters, verify: .all)

        let output = projector(MLXArray([Float(3), 4], [1, 2]))
        MLX.eval(output)
        // RMSNorm([3,4]) is approximately [0.8485,1.1314]; squaring the
        // positive components and summing produces 2.0. LayerNorm+GELU,
        // the broken implementation, produces approximately 0.68 instead.
        XCTAssertEqual(output.item(Float.self), 2.0, accuracy: 1e-4)
    }

    func testRADIOForwardShape() throws {
        // Tiny ViT to keep test fast: 2 blocks, 64 hidden, 8 heads, 16 patch.
        // (The default V3 model is 1280-hidden 32-block; we only validate
        // the forward pipeline shape here.)
        let radio = NemotronHRADIOVisionModel(
            embedDim: 64, numBlocks: 2, numHeads: 8,
            patchSize: 16, numClsTokens: 4, maxGrid: 32)
        let pixels = MLXArray.zeros([1, 3, 128, 128]) // 8×8 patches
        let out = radio(pixels)
        // (B, num_cls + num_patches, embed_dim) = (1, 4 + 64, 64)
        XCTAssertEqual(out.shape, [1, 68, 64])
    }

    func testParakeetSubsamplingShape() throws {
        let sub = NemotronHParakeetSubsampling(hidden: 256, channels: 64)
        // (B, T, n_mels=128) — T must be divisible by 8 after the 3 stride-2 convs.
        let mel = MLXArray.zeros([1, 256, 128])
        let out = sub(mel)
        // After 3× stride-2 reductions on T axis: T/8 = 32 frames.
        // Hidden dim = 256.
        XCTAssertEqual(out.dim(0), 1)
        XCTAssertEqual(out.dim(1), 32)
        XCTAssertEqual(out.dim(2), 256)
    }

    func testParakeetEncoderForward() throws {
        let enc = NemotronHParakeetEncoder(
            hiddenSize: 128, numLayers: 2, numHeads: 4,
            ffHidden: 256, convKernel: 9)
        // Smaller mel for speed.
        let mel = MLXArray.zeros([1, 64, 128])
        let out = enc(mel)
        XCTAssertEqual(out.dim(0), 1)
        XCTAssertEqual(out.dim(1), 8) // 64 / 8 subsampled
        XCTAssertEqual(out.dim(2), 128)
    }

    func testRelShiftSkewing() throws {
        // 2 batches × 2 heads × 3 query tokens × (2*3-1)=5 score columns.
        let scores = MLXArray(
            (0 ..< (2 * 2 * 3 * 5)).map { Float($0) }
        ).reshaped([2, 2, 3, 5])
        let shifted = nemotronOmniRelShift(scores, seqLen: 3)
        XCTAssertEqual(shifted.shape, [2, 2, 3, 3])
    }

    func testPixelShuffleScale05() throws {
        // pixel_shuffle with scale=0.5 (RADIO ps_version=v2):
        //   (B, H, W, C) → (B, H*scale, W*scale, C/scale²)
        // For scale=0.5: spatial halved, channels ×4. 32×32×1280 → 16×16×5120.
        // Tiny smoke: 4×4×16 → 2×2×64
        let inX = MLXArray.zeros([1, 4, 4, 16])
        let outX = nemotronOmniPixelShuffle(inX, scaleFactor: 0.5)
        XCTAssertEqual(outX.shape, [1, 2, 2, 64])
    }

    func testVideoEVSMatchesSourceTokenCount() throws {
        let feats = MLXArray.zeros([16, 256, 8])
        let pruned = nemotronOmniApplyEVS(feats, pruningRate: 0.7)
        XCTAssertEqual(pruned.shape, [1, 1228, 8])
        let targetPruned = nemotronOmniApplyEVS(feats, targetTokenCount: 1024)
        XCTAssertEqual(targetPruned.shape, [1, 1024, 8])
        XCTAssertEqual(
            NemotronHOmniProcessor.videoTokenCountAfterEVS(
                groups: 16, tokensPerGroup: 256, pruningRate: 0.7),
            1228)
        let video = LMInput.ProcessedVideo(
            pixels: MLXArray.zeros([16, 3, 512, 512]),
            frames: nil,
            embeddingTokenCount: 1228)
        XCTAssertEqual(video.embeddingTokenCount, 1228)
    }

    func testMelStftShape() throws {
        // 1 second of 16 kHz silence → expected 101 frames at hop_length=160:
        //   (16000 + 2*256) / 160 = 101.6 → 101 frames before center pad.
        // With center pad of n_fft/2=256: nFrames = 1 + (16000 + 2*256 - 512) / 160 = 100
        // Actual computed below.
        let sr = 16_000
        let waveform = [Float](repeating: 0, count: sr) // 1 second silence
        let mel = nemotronOmniExtractMelFeatures(waveform, sampleRate: sr)
        XCTAssertEqual(mel.dim(0), 1)
        XCTAssertEqual(mel.dim(2), 128) // n_mels
        XCTAssertGreaterThan(mel.dim(1), 50)
        XCTAssertLessThan(mel.dim(1), 200)
    }

    func testMelFilterbankNonZero() throws {
        // Sine-ish wave at 1000 Hz should produce non-flat mel response.
        var w = [Float](repeating: 0, count: 16_000)
        for i in 0 ..< w.count {
            w[i] = sinf(2.0 * .pi * 1000.0 * Float(i) / 16_000.0) * 0.1
        }
        let mel = nemotronOmniExtractMelFeatures(w, normalize: false)
        let arr = mel.asArray(Float.self)
        XCTAssertFalse(arr.allSatisfy { $0 == 0 || $0.isNaN })
    }

    func testRemapMlp1Weights() throws {
        let raw: [String: MLXArray] = [
            "mlp1.0.weight": MLXArray.zeros([5120]),
            "mlp1.1.weight": MLXArray.zeros([20_480, 5120]),
            "mlp1.3.weight": MLXArray.zeros([2688, 20_480]),
            "irrelevant.weight": MLXArray.zeros([1]),
        ]
        let out = remapMlp1Weights(raw)
        // remap returns unprefixed keys — wrapper sanitize prefixes with mlp1.
        XCTAssertEqual(out["layer_norm.weight"]?.shape, [5120])
        XCTAssertNil(out["layer_norm.bias"])
        XCTAssertEqual(out["fc1.weight"]?.shape, [20_480, 5120])
        XCTAssertEqual(out["fc2.weight"]?.shape, [2688, 20_480])
        XCTAssertNil(out["irrelevant.weight"])
    }

    func testRemapSoundProjectionWeights() throws {
        let raw: [String: MLXArray] = [
            "sound_projection.norm.weight": MLXArray.zeros([1024]),
            "sound_projection.linear1.weight": MLXArray.zeros([4096, 1024]),
            "sound_projection.linear2.weight": MLXArray.zeros([2688, 4096]),
            "sound_projection.linear1.bias": MLXArray.zeros([4096]),
            "skip.me": MLXArray.zeros([1]),
        ]
        let out = remapSoundProjectionWeights(raw)
        // remap returns unprefixed keys — wrapper sanitize prefixes with sound_projection.
        XCTAssertEqual(out["norm.weight"]?.shape, [1024])
        XCTAssertEqual(out["linear1.weight"]?.shape, [4096, 1024])
        XCTAssertEqual(out["linear2.weight"]?.shape, [2688, 4096])
        XCTAssertNil(out["skip.me"])
    }

    func testParakeetSubsamplingConv2dTranspose() throws {
        // Source PyTorch shape (256, 1, 3, 3) should map to MLX channels-last
        // (256, 3, 3, 1). remap returns unprefixed keys.
        let raw: [String: MLXArray] = [
            "sound_encoder.encoder.subsampling.layers.0.weight": MLXArray.zeros([256, 1, 3, 3]),
            "sound_encoder.encoder.subsampling.layers.0.bias": MLXArray.zeros([256]),
        ]
        let out = remapParakeetWeights(raw)
        XCTAssertEqual(out["subsampling.layers_0.weight"]?.shape, [256, 3, 3, 1])
    }

    func testParakeetPointwiseConv1dTranspose() throws {
        // Source (2048, 1024, 1) → MLX (2048, 1, 1024). remap unprefixed.
        let raw: [String: MLXArray] = [
            "sound_encoder.encoder.layers.0.conv.pointwise_conv1.weight":
                MLXArray.zeros([2048, 1024, 1])
        ]
        let out = remapParakeetWeights(raw)
        XCTAssertEqual(out["layers.0.conv.pointwise_conv1.weight"]?.shape,
            [2048, 1, 1024])
    }
}
