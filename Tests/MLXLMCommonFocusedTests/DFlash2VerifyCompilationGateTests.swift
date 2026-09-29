// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLMCommon

/// A graph-friendly target with no recurrent layers. Staged commit is a
/// no-op, letting the real iterator's execution policy be tested without
/// loading a checkpoint or depending on a family's recurrent kernels.
private final class CompilationRecordingTarget: Module, LanguageModel,
    HiddenStateCaptureModel, TokenEmbedderModel, DFlash2StagedVerifyRollbackModel,
    @unchecked Sendable
{
    var stagedCalls = 0
    var compiledTraceCalls = 0
    var vocabularySize: Int { 32 }

    func newCache(parameters: GenerateParameters?) -> [KVCache] { [KVCacheSimple()] }

    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        .tokens(input.text)
    }

    func embed(_ ids: MLXArray) -> MLXArray {
        let x = ids.asType(.float32).expandedDimensions(axis: -1)
        return broadcast(x / 32, to: x.shape.dropLast() + [8])
    }

    func projectToLogits(_ hidden: MLXArray) -> MLXArray {
        MLXArray.zeros([hidden.dim(0), hidden.dim(1), vocabularySize])
    }

    func callAsFunction(
        _ inputs: MLXArray, cache: [KVCache]?, captureLayerIDs: Set<Int>
    ) -> (logits: MLXArray, capturedHiddenStates: [Int: MLXArray]) {
        if NativeMTPVerifierStatePolicy.mode == .inputCaptureStaged { stagedCalls += 1 }
        if CompiledDecodeTrace.isActive { compiledTraceCalls += 1 }
        if let layer = cache?.first {
            let rows = MLXArray.zeros([1, 1, inputs.dim(1), 4])
            _ = layer.update(keys: rows, values: rows)
        }
        let h = embed(inputs)
        return (
            projectToLogits(h), Dictionary(uniqueKeysWithValues: captureLayerIDs.map { ($0, h) })
        )
    }

    func commitVerifiedBlock(cache: [KVCache], acceptedInputs: Int) -> Bool { true }
    func commitStagedVerifiedBlock(cache: [KVCache], acceptedInputs: Int, blockLength: Int) -> Bool
    {
        true
    }
}

final class DFlash2VerifyCompilationGateTests: XCTestCase {
    func testStagingDoesNotBypassCompileOptInOrHardwareGateAfterWarmup() throws {
        let keys = [
            "VMLX_DFLASH2_STAGED_VERIFY", "VMLX_DFLASH2_COMPILED_VERIFY",
            "VMLX_ENABLE_UNSAFE_COMPILE", "MLXPRESS_ENABLE_UNSAFE_COMPILE",
            "VMLX_DFLASH2_GOVERNOR", "VMLX_DFLASH2_VERIFY_PREFETCH",
        ]
        let original = ProcessInfo.processInfo.environment
        defer {
            for key in keys {
                if let value = original[key] { setenv(key, value, 1) } else { unsetenv(key) }
            }
        }
        setenv("VMLX_DFLASH2_STAGED_VERIFY", "1", 1)
        setenv("VMLX_DFLASH2_GOVERNOR", "0", 1)
        setenv("VMLX_DFLASH2_VERIFY_PREFETCH", "0", 1)
        unsetenv("MLXPRESS_ENABLE_UNSAFE_COMPILE")

        for (optIn, hardware) in [(nil, "1"), ("0", "1"), ("1", "0")] as [(String?, String)] {
            if let optIn {
                setenv("VMLX_DFLASH2_COMPILED_VERIFY", optIn, 1)
            } else {
                unsetenv("VMLX_DFLASH2_COMPILED_VERIFY")
            }
            setenv("VMLX_ENABLE_UNSAFE_COMPILE", hardware, 1)
            let target = CompilationRecordingTarget()
            let config = try DFlash2Configuration(json: [
                "hidden_size": 8, "num_hidden_layers": 1, "num_attention_heads": 2,
                "num_key_value_heads": 1, "head_dim": 4, "intermediate_size": 16,
                "vocab_size": 32, "num_target_layers": 1, "is_causal": false,
                "layer_types": ["full_attention"],
                "dflash_config": [
                    "block_size": 4, "conv_kernel_size": 2, "conv_group_size": 2,
                    "mask_token_id": 31, "selector_rank": 4, "selector_top_k": 2,
                    "target_layer_ids": [0],
                ],
            ])
            var iterator = try DFlash2TokenIterator(
                input: LMInput(tokens: MLXArray([Int32(1), 2, 3]).reshaped(1, -1)),
                target: target, drafter: DFlash2DraftModel(config), blockSize: 4,
                parameters: GenerateParameters(maxTokens: 24, temperature: 0), cacheCoordinator: nil
            )
            var emitted = 0
            let start = Date.timeIntervalSinceReferenceDate
            while iterator.next() != nil { emitted += 1 }
            let elapsed = Date.timeIntervalSinceReferenceDate - start
            print(
                "[compile-gate synthetic] optIn=\(String(describing: optIn)) hardware=\(hardware) "
                    + "tokens=\(emitted) tok/s=\(Double(emitted) / max(elapsed, 1e-9)) "
                    + "stagedCalls=\(target.stagedCalls) compiledTraces=\(target.compiledTraceCalls)"
            )
            XCTAssertEqual(emitted, 24)
            XCTAssertGreaterThan(iterator.dflash2Stats?.verifyCalls ?? 0, 2)
            XCTAssertGreaterThan(target.stagedCalls, 2, "must exercise post-warmup staged cycles")
            XCTAssertEqual(
                target.compiledTraceCalls, 0,
                "optIn=\(String(describing: optIn)), hardware=\(hardware)")
        }
    }
}
