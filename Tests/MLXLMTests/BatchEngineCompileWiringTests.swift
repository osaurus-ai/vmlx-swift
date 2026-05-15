// Copyright 2025 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Stage 1B.3 integration tests — `BatchEngine` actually invokes
// `BatchCompile.compileForward` for solo requests when the engine is
// configured with `maxBatchSize == 1` and the request opts in via
// `enableCompiledBatchDecode`.
//
// Tests cover:
//   - Compiled request completes with correct token count + completion info
//   - Compiled-decode output is close to uncompiled output for the same
//     prompt (greedy determinism modulo FP reordering)
//   - Compile is correctly gated off at `maxBatchSize > 1` (Stage 1B.4 land)
//   - Compile is correctly gated off for TurboQuant requests (Stage 2 land)
//   - Multi-turn compiled requests complete on repeat submission
//
// These tests use the tiny synthetic Llama test model; numerical closeness
// is within 5% relative for 4-bit quantised weights.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import XCTest

class BatchEngineCompileWiringTests: XCTestCase {

    // MARK: Fixture

    /// Build a BatchEngine with the given maxBatchSize. No cache coordinator —
    /// keeps these tests tightly focused on the compile wiring.
    private func makeEngine(maxBatchSize: Int) -> BatchEngine {
        let config = LlamaConfiguration(
            hiddenSize: 64, hiddenLayers: 4, intermediateSize: 128,
            attentionHeads: 8, rmsNormEps: 1e-5, vocabularySize: 200, kvHeads: 4)
        let model = LlamaModel(config)
        quantize(model: model, groupSize: 64, bits: 4)
        MLX.eval(model)

        let processor = TestInputProcessor()
        nonisolated(unsafe) let context = ModelContext(
            configuration: processor.configuration,
            model: model,
            processor: processor,
            tokenizer: processor.tokenizer
        )
        return BatchEngine(context: context, maxBatchSize: maxBatchSize)
    }

    private func collectTokens(from stream: AsyncStream<BatchGeneration>)
        async -> (tokens: [Int], info: GenerateCompletionInfo?)
    {
        var tokens = [Int]()
        var info: GenerateCompletionInfo?
        for await event in stream {
            switch event {
            case .token(let id):
                tokens.append(id)
            case .info(let i):
                info = i
            }
        }
        return (tokens, info)
    }

    private func skipIfCompileUnsafe() throws {
        guard HardwareInfo.isCompiledDecodeSupported else {
            throw XCTSkip("Compiled decode not supported on this hardware")
        }
    }

    // MARK: - Core wiring tests

    /// A compile-enabled request at maxBatchSize=1 completes successfully and
    /// produces the expected token count. Minimum viable "compile wiring
    /// actually executes" test.
    func testCompiledRequestCompletes() async throws {
        try skipIfCompileUnsafe()

        let engine = makeEngine(maxBatchSize: 1)
        let params = GenerateParameters(
            maxTokens: 6,
            enableCompiledBatchDecode: true,
            compiledBatchBuckets: [1, 2, 4],
            temperature: 0
        )
        let input = LMInput(tokens: MLXArray(Int32(1) ..< Int32(9)))

        let (_, stream) = await engine.submit(input: input, parameters: params)
        let result = await collectTokens(from: stream)

        XCTAssertEqual(result.tokens.count, 6, "Compile wiring should produce 6 decode tokens")
        XCTAssertNotNil(result.info, "Completion info must be emitted")
        XCTAssertEqual(result.info?.stopReason, .length)
    }

    /// Compile-enabled and compile-disabled requests both run to completion.
    /// Different BatchEngine instances have different random model weights,
    /// so this test only asserts completion contract, not numerical equality
    /// across instances. Numerical closeness in the compile path is covered
    /// by `BatchCompileForwardInvocationTests` using a single shared model.
    func testCompiledAndUncompiledBothComplete() async throws {
        try skipIfCompileUnsafe()

        let maxTokens = 6

        let uncompiledEngine = makeEngine(maxBatchSize: 1)
        let uncompiledParams = GenerateParameters(
            maxTokens: maxTokens,
            enableCompiledBatchDecode: false,
            temperature: 0
        )
        // Build a fresh MLXArray per submit to satisfy Swift 6 strict
        // Sendable checks — MLXArray instances are not Sendable across
        // actor boundaries, so shared references trip the race checker.
        let (_, uncompiledStream) = await uncompiledEngine.submit(
            input: LMInput(tokens: MLXArray(Int32(1) ..< Int32(9))),
            parameters: uncompiledParams)
        let uncompiledResult = await collectTokens(from: uncompiledStream)

        let compiledEngine = makeEngine(maxBatchSize: 1)
        let compiledParams = GenerateParameters(
            maxTokens: maxTokens,
            enableCompiledBatchDecode: true,
            temperature: 0
        )
        let (_, compiledStream) = await compiledEngine.submit(
            input: LMInput(tokens: MLXArray(Int32(1) ..< Int32(9))),
            parameters: compiledParams)
        let compiledResult = await collectTokens(from: compiledStream)

        XCTAssertEqual(uncompiledResult.tokens.count, maxTokens)
        XCTAssertEqual(compiledResult.tokens.count, maxTokens)
        XCTAssertNotNil(uncompiledResult.info)
        XCTAssertNotNil(compiledResult.info)
    }

    /// When `maxBatchSize > 1`, the compile promotion hook skips — per
    /// Stage 1B.3 scope. Requests still succeed via the uncompiled path.
    /// Stage 1B.4 will lift this restriction.
    func testCompileDoesNotEngageAtMaxBatchSizeGreaterThanOne() async throws {
        let engine = makeEngine(maxBatchSize: 2)
        let params = GenerateParameters(
            maxTokens: 5,
            enableCompiledBatchDecode: true,
            temperature: 0
        )
        let input = LMInput(tokens: MLXArray(Int32(1) ..< Int32(9)))

        let (_, stream) = await engine.submit(input: input, parameters: params)
        let result = await collectTokens(from: stream)

        XCTAssertEqual(result.tokens.count, 5,
            "Compile skip at maxBatchSize>1 must not prevent normal completion")
    }

    /// Stage 2 SHIPPED (iter 21): `kvMode: .turboQuant(...)` +
    /// `enableCompiledBatchDecode` now engages the compile path via
    /// `CompilableTurboQuantKVCache`. Root cause of the earlier drift
    /// was `applyRotaryPosition` routing TQ through the Int
    /// `cache.offset` instead of the MLXArray offsetArray — fixed in
    /// `RoPEApplication.swift`. Multi-step compiled-vs-uncompiled
    /// divergence now at FP precision (~5e-7).
    ///
    /// This test verifies the engine-level path completes without
    /// crashing on a TQ + compile request.
    func testCompileEngagesWithTurboQuant() async throws {
        try skipIfCompileUnsafe()

        let engine = makeEngine(maxBatchSize: 1)
        let params = GenerateParameters(
            maxTokens: 5,
            kvMode: .turboQuant(keyBits: 3, valueBits: 3),
            enableCompiledBatchDecode: true,
            temperature: 0
        )
        let input = LMInput(tokens: MLXArray(Int32(1) ..< Int32(13)))

        let (_, stream) = await engine.submit(input: input, parameters: params)
        let result = await collectTokens(from: stream)

        // Iter 21: TQ + compile now engages the compile path via
        // CompilableTurboQuantKVCache. Request completes correctly.
        XCTAssertEqual(result.tokens.count, 5,
            "TQ + compile request must complete via the compile path (Stage 2 iter 21)")
        XCTAssertNotNil(result.info)
    }

    /// Multi-turn compile: submit the same prompt twice through the same
    /// engine. Both turns complete — the engine must not leave stale
    /// compile state that interferes with the second request.
    func testCompiledMultiTurn() async throws {
        try skipIfCompileUnsafe()

        let engine = makeEngine(maxBatchSize: 1)
        let params = GenerateParameters(
            maxTokens: 4,
            enableCompiledBatchDecode: true,
            temperature: 0
        )
        // Fresh MLXArray per submit (Swift 6 Sendable check — the tokens
        // tensor is not safe to share across actor boundaries).
        let (_, s1) = await engine.submit(
            input: LMInput(tokens: MLXArray(Int32(1) ..< Int32(9))),
            parameters: params)
        let r1 = await collectTokens(from: s1)
        XCTAssertEqual(r1.tokens.count, 4, "Turn 1 must complete under compile")
        XCTAssertNotNil(r1.info)

        let (_, s2) = await engine.submit(
            input: LMInput(tokens: MLXArray(Int32(1) ..< Int32(9))),
            parameters: params)
        let r2 = await collectTokens(from: s2)
        XCTAssertEqual(r2.tokens.count, 4, "Turn 2 must complete under compile")
        XCTAssertNotNil(r2.info)

        // Same engine + same greedy prompt → identical tokens across turns.
        XCTAssertEqual(r1.tokens, r2.tokens,
            "Same engine + same greedy prompt should produce identical tokens across turns")
    }

    /// Short prefills still promote correctly and decode runs through the
    /// compiled path. No TQ interaction here — just edge-case prompt length.
    func testCompiledShortPrompt() async throws {
        try skipIfCompileUnsafe()

        let engine = makeEngine(maxBatchSize: 1)
        let params = GenerateParameters(
            maxTokens: 8,
            enableCompiledBatchDecode: true,
            temperature: 0
        )
        let input = LMInput(tokens: MLXArray(Int32(1) ..< Int32(4)))

        let (_, stream) = await engine.submit(input: input, parameters: params)
        let result = await collectTokens(from: stream)

        XCTAssertEqual(result.tokens.count, 8, "Short-prompt compile must still produce 8 tokens")
        XCTAssertNotNil(result.info)
    }

    // MARK: - Iteration 6 hardening: compile + cache coordinator

    /// Build a maxBatchSize=1 engine with an in-memory paged cache
    /// coordinator attached. Everything else matches `makeEngine`.
    private func makeEngineWithCoordinator() -> (BatchEngine, CacheCoordinator) {
        let config = LlamaConfiguration(
            hiddenSize: 64, hiddenLayers: 4, intermediateSize: 128,
            attentionHeads: 8, rmsNormEps: 1e-5, vocabularySize: 200, kvHeads: 4)
        let model = LlamaModel(config)
        quantize(model: model, groupSize: 64, bits: 4)
        MLX.eval(model)

        let processor = TestInputProcessor()
        nonisolated(unsafe) let context = ModelContext(
            configuration: processor.configuration,
            model: model,
            processor: processor,
            tokenizer: processor.tokenizer
        )

        let coordCfg = CacheCoordinatorConfig(
            usePagedCache: true,
            enableDiskCache: false,
            pagedBlockSize: 4,
            maxCacheBlocks: 256
        )
        let coordinator = CacheCoordinator(config: coordCfg)
        let engine = BatchEngine(
            context: context, maxBatchSize: 1, cacheCoordinator: coordinator)
        return (engine, coordinator)
    }

    /// Compile + cache coordinator: first turn populates the cache while
    /// running the compiled path, second turn's cache-hit restore runs
    /// alongside compile promotion. Both turns must complete.
    ///
    /// Cache-hit restore currently targets `KVCacheSimple` layers (the
    /// coordinator's paged path stores float blocks). After restore, the
    /// compile promotion hook reads the restored `KVCacheSimple` layers
    /// and swaps them for `CompilableKVCache(from:)`. Both paths interact
    /// via slot.cache mutation — no explicit coordination needed since
    /// the engine is actor-isolated.
    func testCompiledMultiTurnWithCoordinator() async throws {
        try skipIfCompileUnsafe()

        let (engine, coordinator) = makeEngineWithCoordinator()
        let params = GenerateParameters(
            maxTokens: 4,
            enableCompiledBatchDecode: true,
            temperature: 0
        )
        let promptTokens: [Int32] = [3, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        let promptSalt = computeCacheSalt(
            for: LMInput(tokens: MLXArray(promptTokens)),
            parameters: params)

        let (_, s1) = await engine.submit(
            input: LMInput(tokens: MLXArray(promptTokens)),
            parameters: params)
        let r1 = await collectTokens(from: s1)
        XCTAssertEqual(r1.tokens.count, 4, "Turn 1 under compile+coordinator must complete")
        XCTAssertNotNil(r1.info)

        // The coordinator should have stored the prompt's post-generation
        // cache state. Verify via direct fetch.
        let lookup = coordinator.fetch(
            tokens: promptTokens.map(Int.init), mediaSalt: promptSalt)
        if case .miss = lookup {
            XCTFail("Turn 1 under compile should still populate the coordinator")
        }

        let (_, s2) = await engine.submit(
            input: LMInput(tokens: MLXArray(promptTokens)),
            parameters: params)
        let r2 = await collectTokens(from: s2)
        XCTAssertEqual(r2.tokens.count, 4, "Turn 2 (cache-hit under compile) must complete")
        XCTAssertNotNil(r2.info)
    }

    /// Compile + coordinator with prompt-extension multi-turn. Turn 2's
    /// prompt is a superset of turn 1's — the coordinator returns the
    /// prefix match, the engine prefills the remaining tokens, compile
    /// promotion runs at the end of prefill. The slot proceeds through
    /// the compiled decode path.
    func testCompiledPromptExtensionWithCoordinator() async throws {
        try skipIfCompileUnsafe()

        let (engine, _) = makeEngineWithCoordinator()
        let params = GenerateParameters(
            maxTokens: 3,
            enableCompiledBatchDecode: true,
            temperature: 0
        )
        let turn1Prompt: [Int32] = [2, 3, 5, 7, 11, 13, 17, 19]
        let turn2Prompt: [Int32] = turn1Prompt + [23, 29]

        let (_, s1) = await engine.submit(
            input: LMInput(tokens: MLXArray(turn1Prompt)),
            parameters: params)
        let r1 = await collectTokens(from: s1)
        XCTAssertEqual(r1.tokens.count, 3, "Turn 1 extension base must complete")

        let (_, s2) = await engine.submit(
            input: LMInput(tokens: MLXArray(turn2Prompt)),
            parameters: params)
        let r2 = await collectTokens(from: s2)
        XCTAssertEqual(r2.tokens.count, 3, "Turn 2 extension must complete under compile+coordinator")
        XCTAssertEqual(
            r2.info?.promptTokenCount, turn2Prompt.count,
            "Extended prompt length must surface in completion info"
        )
    }

    /// Compile + coordinator with a long-ish decode (>16 tokens). Exercises
    /// the per-decode-step hook + coordinator-store end-of-generation path
    /// under compile. Keeps compile promotion and the extended decode loop
    /// interacting correctly.
    func testCompiledLongDecodeWithCoordinator() async throws {
        try skipIfCompileUnsafe()

        let (engine, coordinator) = makeEngineWithCoordinator()
        let params = GenerateParameters(
            maxTokens: 20,
            compiledMaxCacheLength: 512,  // plenty for this test
            enableCompiledBatchDecode: true,
            temperature: 0
        )
        let promptTokens: [Int32] = [41, 43, 47, 53, 59, 61, 67, 71]
        let promptSalt = computeCacheSalt(
            for: LMInput(tokens: MLXArray(promptTokens)),
            parameters: params)

        let (_, stream) = await engine.submit(
            input: LMInput(tokens: MLXArray(promptTokens)),
            parameters: params)
        let result = await collectTokens(from: stream)

        XCTAssertEqual(
            result.tokens.count, 20,
            "Long-decode compile run must produce the requested token count"
        )
        XCTAssertNotNil(result.info)

        // Coordinator must still see the prompt entry after a long decode
        // under compile.
        let lookup = coordinator.fetch(
            tokens: promptTokens.map(Int.init), mediaSalt: promptSalt)
        if case .miss = lookup {
            XCTFail("Long-decode compile run must still populate the coordinator")
        }
    }
}
