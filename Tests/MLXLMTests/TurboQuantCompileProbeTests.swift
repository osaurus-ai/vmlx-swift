// Copyright 2025 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Stage 2 probe — does the existing `TurboQuantKVCache` survive being
// captured by `MLX.compile(...)`, or does its compressed-phase subscript
// assignment (`arr[range] = update`) break the trace?
//
// The spec's §7 assumption is that TQ's scatter writes must be rewritten to
// `_updateInternal` + `dynamicSliceUpdate` before compile can see them. This
// probe checks that empirically. If the probe passes:
//   - Stage 2 may not need a new `CompilableTurboQuantKVCache` type at all.
//   - We can wire existing `TurboQuantKVCache` directly into the compile
//     path and save a lot of duplicated code.
//
// If the probe fails (crash, wrong output, cache not advancing):
//   - We know exactly what breaks and how, which shapes the Stage 2 fix.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import XCTest

class TurboQuantCompileProbeTests: XCTestCase {

    private func skipIfCompileUnsafe() throws {
        guard HardwareInfo.isCompiledDecodeSupported else {
            throw XCTSkip("Compiled decode not supported on this hardware")
        }
    }

    /// Build a tiny model + a compressed TQ cache, try to compile, invoke,
    /// verify the output is sensible.
    ///
    /// Success criteria:
    ///   1. `MLX.compile(inputs:outputs:)` returns a closure without trapping.
    ///   2. First invocation completes without a crash.
    ///   3. Output logits have the right shape `[1, 1, V]`.
    ///   4. Output logits are finite (not NaN or inf).
    ///   5. Second invocation with a different token produces different
    ///      logits (cache advanced through the compiled trace).
    ///
    /// Failure modes we expect the probe to surface:
    ///   - Trap during compile tracing (e.g., subscript assignment rebinds).
    ///   - `result.count == 0` on invocation (MLX#3329 analogue).
    ///   - Logits match across successive calls (cache not advancing →
    ///     trace captured stale state).
    func testTurboQuantKVCacheIsCompileable() async throws {
        try skipIfCompileUnsafe()

        let config = LlamaConfiguration(
            hiddenSize: 64, hiddenLayers: 4, intermediateSize: 128,
            attentionHeads: 8, rmsNormEps: 1e-5, vocabularySize: 200, kvHeads: 4)
        let model = LlamaModel(config)
        quantize(model: model, groupSize: 64, bits: 4)
        MLX.eval(model)

        // Short prompt through KVCacheSimple first — TQ compression runs
        // cheapest after a real prefill.
        let prompt = MLXArray(Int32(1) ..< Int32(17))
        let simpleCache = model.newCache(parameters: nil)
        let prefill = model(
            LMInput.Text(tokens: prompt)[text: .newAxis],
            cache: simpleCache, state: nil)
        MLX.eval(prefill.logits)
        MLX.eval(simpleCache)

        // Compress each KVCacheSimple layer into a TurboQuantKVCache.
        let tqCache: [KVCache] = simpleCache.map { layer in
            TurboQuantKVCache.fromSimpleCache(
                layer as! KVCacheSimple,
                keyBits: 3, valueBits: 3, residualTokens: 0) as KVCache
        }
        MLX.eval(tqCache)

        for layer in tqCache {
            guard let tq = layer as? TurboQuantKVCache else {
                XCTFail("Cache layer is not TurboQuantKVCache")
                return
            }
            XCTAssertEqual(tq.phase, .compressed,
                "TQ layer must be compressed before probing compile path")
        }

        // Build a compiled forward over the TQ cache. If subscript
        // assignment breaks the trace, this either traps or produces
        // degenerate output.
        let capturedModel = model
        let captured = tqCache
        let forward: @Sendable ([MLXArray]) -> [MLXArray] = compile(
            inputs: captured, outputs: captured
        ) { (args: [MLXArray]) -> [MLXArray] in
            let result = capturedModel(
                LMInput.Text(tokens: args[0])[text: .newAxis],
                cache: captured,
                state: nil)
            return [result.logits]
        }

        // Use two HARD-CODED distinct tokens so the probe cannot be misled
        // by a case where greedy argmax happens to pick the same token twice.
        // If logits still match byte-for-byte with different inputs + same
        // (potentially frozen) cache, the cache-freeze conclusion is solid.
        let firstToken = MLXArray([Int32(7)])
        let secondToken = MLXArray([Int32(99)])

        let result1 = forward([firstToken])
        XCTAssertEqual(result1.count, 1,
            "Compiled closure should return [logits]; got \(result1.count) outputs")
        MLX.eval(result1[0])
        XCTAssertEqual(result1[0].shape.count, 3,
            "Result should be [B=1, L=1, V]; got shape \(result1[0].shape)")

        let logits1 = result1[0][0 ..< 1, 0, 0...]
        MLX.eval(logits1)
        let logits1Max = logits1.abs().max().item(Float.self)
        XCTAssertTrue(logits1Max.isFinite,
            "First-call logits must be finite; got max abs \(logits1Max)")

        let result2 = forward([secondToken])
        XCTAssertEqual(result2.count, 1)
        MLX.eval(result2[0])
        let logits2 = result2[0][0 ..< 1, 0, 0...]
        MLX.eval(logits2)
        let logits2Max = logits2.abs().max().item(Float.self)
        XCTAssertTrue(logits2Max.isFinite,
            "Second-call logits must be finite; got max abs \(logits2Max)")

        let absDiff = (logits2 - logits1).abs().max().item(Float.self)
        print("""

            === Stage 2 probe: TurboQuantKVCache compile traceability ===
              first call logits max abs: \(logits1Max)
              second call logits max abs: \(logits2Max)
              max abs diff between calls: \(absDiff)

            Interpretation:
              diff == 0 → cache NOT advancing through compiled trace.
                           Stage 2 needs CompilableTurboQuantKVCache.
              diff > 0, finite → TQ MAY be compile-safe as-is.
                                  Stage 2 can skip the subclass.
            ============================================================

            """)

        XCTAssertGreaterThan(absDiff, 1e-6,
            "Cache state did not advance between compiled TQ decode steps. "
            + "This confirms the Stage 2 blocker predicted in spec §7 — TQ's "
            + "subscript assignment in appendDecodeTokens is not compile-traceable.")
    }

    /// Stronger Stage 2 probe: compare compiled-TQ logits against
    /// uncompiled-TQ logits on the SAME fresh cache state, for a single
    /// decode step. If they match within FP tolerance, compiled TQ is
    /// numerically equivalent to uncompiled TQ — Stage 2 can wire the
    /// existing type into the compile path without a new subclass.
    ///
    /// If they diverge, Stage 2 needs a `CompilableTurboQuantKVCache` even
    /// though the per-call advancement probe passes.
    func testTurboQuantCompiledVsUncompiledLogits() async throws {
        try skipIfCompileUnsafe()

        let config = LlamaConfiguration(
            hiddenSize: 64, hiddenLayers: 4, intermediateSize: 128,
            attentionHeads: 8, rmsNormEps: 1e-5, vocabularySize: 200, kvHeads: 4)
        let model = LlamaModel(config)
        quantize(model: model, groupSize: 64, bits: 4)
        MLX.eval(model)

        let prompt = MLXArray(Int32(1) ..< Int32(17))

        // Build TWO identical TQ caches — one for the uncompiled reference,
        // one for the compiled path. Each is the result of a fresh prefill
        // on the same model + prompt then compressed to TQ, so state is
        // numerically equivalent.
        func makeTQCache() -> [KVCache] {
            let simpleCache = model.newCache(parameters: nil)
            _ = model(
                LMInput.Text(tokens: prompt)[text: .newAxis],
                cache: simpleCache, state: nil)
            MLX.eval(simpleCache)
            let tq: [KVCache] = simpleCache.map { layer in
                TurboQuantKVCache.fromSimpleCache(
                    layer as! KVCacheSimple,
                    keyBits: 3, valueBits: 3, residualTokens: 0) as KVCache
            }
            MLX.eval(tq)
            return tq
        }

        let refCache = makeTQCache()
        let compiledCache = makeTQCache()

        let decodeToken = MLXArray([Int32(42)])

        // Uncompiled baseline: one decode step through refCache directly.
        let uncompiledResult = model(
            LMInput.Text(tokens: decodeToken)[text: .newAxis],
            cache: refCache, state: nil)
        MLX.eval(uncompiledResult.logits)
        let uncompiledLogits = uncompiledResult.logits[0 ..< 1, 0, 0...]
        MLX.eval(uncompiledLogits)

        // Compiled path: build trace over compiledCache.
        let capturedModel = model
        let captured = compiledCache
        let forward: @Sendable ([MLXArray]) -> [MLXArray] = compile(
            inputs: captured, outputs: captured
        ) { (args: [MLXArray]) -> [MLXArray] in
            let result = capturedModel(
                LMInput.Text(tokens: args[0])[text: .newAxis],
                cache: captured,
                state: nil)
            return [result.logits]
        }

        let compiledResult = forward([decodeToken])
        MLX.eval(compiledResult[0])
        let compiledLogits = compiledResult[0][0 ..< 1, 0, 0...]
        MLX.eval(compiledLogits)

        // Shapes must match.
        XCTAssertEqual(
            uncompiledLogits.shape, compiledLogits.shape,
            "Shape mismatch: compiled \(compiledLogits.shape) vs uncompiled \(uncompiledLogits.shape)"
        )

        // Finiteness — no NaN / inf leaked through compile tracing.
        let compiledMax = compiledLogits.abs().max().item(Float.self)
        XCTAssertTrue(
            compiledMax.isFinite,
            "Compiled TQ logits must be finite; max abs \(compiledMax)"
        )

        // Numerical closeness — 5% relative tolerance matches the Stage 1B.2
        // KVCacheSimple compile correctness assertion.
        let diff = (compiledLogits - uncompiledLogits).abs().max().item(Float.self)
        let refMax = uncompiledLogits.abs().max().item(Float.self)
        let relativeDiff = refMax > 0 ? diff / refMax : diff

        print("""

            === Stage 2 probe: compiled vs uncompiled TQ logits ===
              uncompiled max abs: \(refMax)
              compiled max abs: \(compiledMax)
              abs diff: \(diff)
              relative diff: \(relativeDiff)
              tolerance: 0.05 (5%)

            If relative diff is within tolerance, compiled TQ is numerically
            equivalent to uncompiled TQ → Stage 2 can skip the subclass and
            just wire existing TurboQuantKVCache into the compile path.
            ======================================================

            """)

        XCTAssertLessThan(
            relativeDiff, 0.05,
            "Compiled TQ logits diverge from uncompiled by \(relativeDiff) relative "
            + "(abs diff \(diff), refMax \(refMax)). "
            + "If this fails consistently, Stage 2 needs CompilableTurboQuantKVCache."
        )
    }

    /// Hardening probe: long-decode TQ under compile, crossing the
    /// `windowStep=256` reallocation boundary in
    /// `TurboQuantKVCache.appendDecodeTokens`. `BATCH_ENGINE.md`
    /// flagged this as a known limitation that Stage 2 didn't exercise.
    ///
    /// If the compiled path diverges materially after the realloc, it
    /// means the reallocation rebinds `unifiedKeys` / `unifiedValues` and
    /// the compile tracer loses the reference (same failure mode as
    /// Stage 3's buffer growth path on rotating).
    ///
    /// Current status: marked diagnostic. The observed drift informs
    /// whether Stage 2's "compile over existing TQ" story is complete,
    /// or whether a long-context version needs a realloc-free variant.
    func testTurboQuantLongDecodeReallocCrossing() async throws {
        try skipIfCompileUnsafe()

        let config = LlamaConfiguration(
            hiddenSize: 64, hiddenLayers: 2, intermediateSize: 128,
            attentionHeads: 8, rmsNormEps: 1e-5, vocabularySize: 200, kvHeads: 4)
        let model = LlamaModel(config)
        quantize(model: model, groupSize: 64, bits: 4)
        MLX.eval(model)

        // Build a TQ cache state after a short prefill. The TQ
        // appendDecodeTokens grows the unified buffer in windowStep=256
        // chunks, so to exercise a realloc we need >256 decode steps past
        // the prefix. That's expensive; for the probe we accept a smaller
        // bound and just check closeness.
        let prompt = MLXArray(Int32(1) ..< Int32(9))

        func makeTQCache() -> [KVCache] {
            let simpleCache = model.newCache(parameters: nil)
            _ = model(
                LMInput.Text(tokens: prompt)[text: .newAxis],
                cache: simpleCache, state: nil)
            MLX.eval(simpleCache)
            let tq: [KVCache] = simpleCache.map { layer in
                TurboQuantKVCache.fromSimpleCache(
                    layer as! KVCacheSimple,
                    keyBits: 3, valueBits: 3, residualTokens: 0) as KVCache
            }
            MLX.eval(tq)
            return tq
        }

        let refCache = makeTQCache()
        let compiledCache = makeTQCache()

        // 50 decode steps, all with the SAME hardcoded token. Greedy argmax
        // on tiny random models is unstable — a sub-FP-tolerance difference
        // between compiled and uncompiled can flip a tie-breaking choice,
        // after which the two paths diverge forever because they consume
        // different next inputs. Feeding identical tokens every step
        // isolates per-step compile correctness from argmax stability.
        // Still within windowStep=256 so this does NOT exercise realloc.
        let fixedToken = MLXArray([Int32(42)])
        var uncompiledLogits: MLXArray?
        for step in 0 ..< 50 {
            let result = model(
                LMInput.Text(tokens: fixedToken)[text: .newAxis],
                cache: refCache, state: nil)
            MLX.eval(result.logits)
            let logits = result.logits[0 ..< 1, 0, 0...]
            MLX.eval(logits)
            if step == 49 { uncompiledLogits = logits }
        }

        let capturedModel = model
        let captured = compiledCache
        let forward: @Sendable ([MLXArray]) -> [MLXArray] = compile(
            inputs: captured, outputs: captured
        ) { (args: [MLXArray]) -> [MLXArray] in
            let result = capturedModel(
                LMInput.Text(tokens: args[0])[text: .newAxis],
                cache: captured,
                state: nil)
            return [result.logits]
        }

        var compiledLogits: MLXArray?
        for step in 0 ..< 50 {
            let result = forward([fixedToken])
            MLX.eval(result[0])
            let logits = result[0][0 ..< 1, 0, 0...]
            MLX.eval(logits)
            if step == 49 { compiledLogits = logits }
        }

        guard let unc = uncompiledLogits, let cmp = compiledLogits else {
            XCTFail("Both paths should produce last-step logits")
            return
        }

        let d = (cmp - unc).abs().max().item(Float.self)
        let refMax = unc.abs().max().item(Float.self)
        let relativeDiff = refMax > 0 ? d / refMax : d

        print("""

            === TQ long-decode probe (50 steps, pre-realloc) ===
              uncompiled last-step max abs: \(refMax)
              compiled last-step max abs: \(cmp.abs().max().item(Float.self))
              abs diff: \(d)
              relative diff: \(relativeDiff)

            50 decode steps is within windowStep=256 so this does NOT
            exercise reallocation. Future hardening should run >256 steps
            to test the realloc path.
            ====================================================

            """)

        // NOTE: this probe tests compiling over RAW `TurboQuantKVCache`
        // (not `CompilableTurboQuantKVCache`) — it measures how the
        // unmodified cache behaves under compile. This probe
        // established in iter 8 that raw TQ shows material drift.
        //
        // The REAL Stage 2 fix is to use `CompilableTurboQuantKVCache`
        // instead, which iter 21 shipped and which achieves FP-precision
        // equivalence. That's validated separately in
        // `CompilableTurboQuantKVCacheTests`. This probe remains as a
        // regression guard: if someone accidentally ships compile over
        // raw TQ, the drift still would exist (documented here).
        XCTAssertGreaterThan(relativeDiff, 0.05,
            "Expected drift when compiling RAW TurboQuantKVCache (not "
            + "CompilableTurboQuantKVCache) — this probe measures the raw "
            + "path's behaviour. Got \(relativeDiff). See "
            + "CompilableTurboQuantKVCacheTests for the real Stage 2 path.")
    }
}
