// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Pins the sampler output contract that DFlash2TokenIterator.init relies
// on: `sample(logits:)` over a [1, 1, vocab] row must return a 1-element
// integer array. Observed live 2026-08-20: through the osaurus solo path
// the same call produced a 0-dim BOOL scalar and the subsequent
// `.item(Int.self)` precondition took the whole app down.

import Foundation
import MLX
import Testing

@testable import MLXLMCommon

@Suite("TopP sampler output contract", .serialized)
struct TopPSamplerShapeFocusedTests {

    @Test("sample on a [1,1,vocab] row returns one integer token")
    func sampleShapeContract() throws {
        try FocusedMLXTestSupport.withLock {
            // Qwen3.8-27B-JANG_4D generation defaults: temp 1.0, top_p 0.95,
            // top_k 20 — and the real 248320-entry vocabulary row.
            let sampler = TopPSampler(
                temperature: 1.0, topP: 0.95, topK: 20, minP: 0.0)
            for dtype in [DType.float16, DType.bfloat16, DType.float32] {
                let logits = MLXRandom.normal([1, 1, 248320]).asType(dtype)
                let out = sampler.sample(logits: logits)
                MLX.eval(out)
                #expect(out.size == 1, "\(dtype): size \(out.size)")
                #expect(
                    out.dtype == .int32 || out.dtype == .uint32
                        || out.dtype == .int64,
                    "\(dtype): dtype \(out.dtype)")
                let token = out.reshaped(-1)[0].item(Int.self)
                #expect(token >= 0 && token < 248320)
            }
        }
    }
}
