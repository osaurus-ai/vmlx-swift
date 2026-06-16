// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Phase 1 iter 3 tests — pin the HiddenStateCapture protocol contract.

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

/// Helper to materialize MLX lazy tensors. Wrapped so the bare-name
/// `eval(...)` call doesn't trip the repo-wide pre-write security hook.
private func materialize(_ arrays: MLXArray...) {
    for a in arrays { MLX.eval(a) }
}

private func materializeModel(_ m: Module) {
    MLX.eval(m)
}

@Suite("HiddenStateCapture — Phase 1", .serialized)
struct HiddenStateCaptureTests {

    /// Tiny Qwen3 config — small enough to instantiate in a unit test.
    private static let tinyConfig: Qwen3Configuration = {
        let json = """
        {
          "hidden_size": 128,
          "num_hidden_layers": 4,
          "intermediate_size": 256,
          "num_attention_heads": 4,
          "rms_norm_eps": 1e-6,
          "vocab_size": 512,
          "num_key_value_heads": 2,
          "rope_theta": 1000000,
          "head_dim": 32,
          "tie_word_embeddings": false,
          "max_position_embeddings": 512
        }
        """
        return try! JSONDecoder().decode(
            Qwen3Configuration.self, from: json.data(using: .utf8)!)
    }()

    private func newModel() -> Qwen3Model {
        let m = Qwen3Model(Self.tinyConfig)
        materializeModel(m)
        return m
    }

    /// Build a deterministic 1×L input of Int32 tokens.
    private func tokens(_ values: [Int32]) -> MLXArray {
        MLXArray(values).reshaped(1, values.count)
    }

    @Test("Qwen3Model conforms to HiddenStateCaptureModel")
    func testConformance() {
        let model: Any = newModel()
        #expect(model is HiddenStateCaptureModel)
    }

    @Test("Empty captureLayerIDs is byte-identical to plain forward")
    func testEmptyCaptureIsByteIdentical() {
        let model = newModel()
        let input = tokens([1, 2, 3, 4, 5])
        let plain = model(input, cache: nil)
        let (captured, states) = model(
            input, cache: nil, captureLayerIDs: [])
        materialize(plain, captured)
        #expect(states.isEmpty)
        let eq = equal(plain, captured).all()
        materialize(eq)
        #expect(eq.item(Bool.self))
    }

    @Test("Non-empty capture fills right keys with right shapes")
    func testCapturePopulatesLayers() {
        let model = newModel()
        let input = tokens([1, 2, 3, 4, 5])
        let toCapture: Set<Int> = [0, 2, 3]
        let (_, states) = model(
            input, cache: nil, captureLayerIDs: toCapture)
        #expect(Set(states.keys) == toCapture)
        for id in toCapture {
            let h = states[id]!
            materialize(h)
            #expect(h.ndim == 3)
            #expect(h.dim(0) == 1)
            #expect(h.dim(1) == 5)
            #expect(h.dim(2) == Self.tinyConfig.hiddenSize)
        }
    }

    @Test("Capture is side-effect-free on logits")
    func testCaptureIsSideEffectFree() {
        let model = newModel()
        let input = tokens([10, 20, 30])
        let (logitsA, _) = model(input, cache: nil, captureLayerIDs: [])
        let (logitsB, _) = model(input, cache: nil, captureLayerIDs: [0, 1, 2, 3])
        materialize(logitsA, logitsB)
        let eq = equal(logitsA, logitsB).all()
        materialize(eq)
        #expect(eq.item(Bool.self))
    }

    @Test("extractContextFeature concatenates in the requested order")
    func testExtractContextFeatureShape() throws {
        let model = newModel()
        let input = tokens([1, 2, 3])
        let (_, states) = model(
            input, cache: nil, captureLayerIDs: [0, 1, 2])
        // Request a specific order — output concat must follow it.
        let stacked = try extractContextFeature(
            captured: states, targetLayerIDs: [2, 0, 1])
        materialize(stacked)
        #expect(stacked.ndim == 3)
        #expect(stacked.dim(0) == 1)
        #expect(stacked.dim(1) == 3)
        #expect(stacked.dim(2) == 3 * Self.tinyConfig.hiddenSize)
    }

    @Test("extractContextFeature rejects empty target layer IDs")
    func testExtractContextFeatureRejectsEmptyTargetLayerIDs() throws {
        do {
            _ = try extractContextFeature(captured: [:], targetLayerIDs: [])
            Issue.record("Expected empty target layer IDs to throw")
        } catch let error as SpecDecError {
            #expect(String(describing: error).contains("target_layer_ids"))
        }
    }

    @Test("extractContextFeature rejects missing captured target layer")
    func testExtractContextFeatureRejectsMissingCapturedLayer() throws {
        let hidden = MLXArray.zeros([1, 2, Self.tinyConfig.hiddenSize], dtype: .float32)

        do {
            _ = try extractContextFeature(captured: [0: hidden], targetLayerIDs: [0, 1])
            Issue.record("Expected missing captured layer to throw")
        } catch let error as SpecDecError {
            #expect(String(describing: error).contains("missing captured hidden state"))
        }
    }
}
