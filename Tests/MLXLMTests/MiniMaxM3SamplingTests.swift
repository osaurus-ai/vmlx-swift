// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

/// MiniMax-M3.1 sampling: the bundle's defaults (generation_config.json temperature 1.0 / top_p 0.95, the same values
/// in jang_config chat.sampling_defaults), request values over them, and every filter (temperature, top_p, top_k,
/// min_p) reaching the AR sampler, dSpark's sampled verify and the TokenIterator. Each filter test has a control
/// that shows the filter changes the outcome, so a dropped parameter fails.
@Suite("MiniMax-M3.1 sampling: bundle defaults, request overrides, every filter reaches the sampler", .serialized)
struct MiniMaxM3SamplingTests {
    static let bundle = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        .appendingPathComponent("Fixtures/MiniMaxM3/m31-bundle")

    static func generationConfig() throws -> GenerationConfigFile {
        try JSONDecoder.json5().decode(
            GenerationConfigFile.self, from: Data(contentsOf: bundle.appendingPathComponent("generation_config.json")))
    }

    /// A host's request mapping: bundle defaults, then every value the request carries.
    static func requestParameters(
        temperature: Float? = nil, topP: Float? = nil, topK: Int? = nil, minP: Float? = nil
    ) throws -> GenerateParameters {
        var p = GenerateParameters(generationConfig: try generationConfig())
        if let temperature { p.temperature = temperature }
        if let topP { p.topP = topP }
        if let topK { p.topK = topK }
        if let minP { p.minP = minP }
        return p
    }

    @Test("The shipped generation_config.json and jang_config chat.sampling_defaults: temperature 1.0, top_p 0.95")
    func bundleDefaults() throws {
        let g = try Self.generationConfig()
        #expect(g.temperature == 1.0 && g.topP == 0.95 && g.topK == nil && g.minP == nil && g.doSample == true)
        let p = GenerateParameters(generationConfig: g)
        #expect(p.temperature == 1.0 && p.topP == 0.95 && p.topK == 0 && p.minP == 0)
        // The sampler the defaults build filters (top_p), so it is the TopPSampler at temperature 1.
        #expect(p.sampler() is TopPSampler)
        let jang = try JangLoader.loadConfig(at: Self.bundle)
        let s = try #require(jang.chat?.samplingDefaults)
        #expect(s.temperature == g.temperature && s.topP == g.topP && s.topK == nil && s.minP == nil)
        // A greedy request overrides the bundle's temperature; the bundle's top_p stays (unused at 0).
        let greedy = try Self.requestParameters(temperature: 0)
        #expect(greedy.sampler() is ArgMaxSampler)
        let custom = try Self.requestParameters(temperature: 0.7, topP: 0.8, topK: 40, minP: 0.05)
        #expect(custom.temperature == 0.7 && custom.topP == 0.8 && custom.topK == 40 && custom.minP == 0.05)
    }

    /// A row whose argmax is token 3 and whose mass is spread, so an unfiltered draw often lands elsewhere.
    static func row(vocab: Int = 64) -> MLXArray {
        var v = [Float](repeating: 0, count: vocab)
        for i in 0 ..< vocab { v[i] = Float((i * 37) % 11) * 0.05 }
        v[3] = 1.0
        v[9] = 0.95
        return MLXArray(v).reshaped(1, vocab)
    }

    static func draws(_ p: GenerateParameters, seeds: Range<UInt64> = 0 ..< 48) -> [Int] {
        seeds.map { seed in
            var q = p
            q.randomSeed = seed
            return q.sampler().sample(logits: row()).item(Int.self)
        }
    }

    @Test("AR sampler: top_k, top_p, min_p and temperature each change the draw (controls: without them it differs)")
    func arSamplerHonorsEveryFilter() throws {
        try Device.withDefaultDevice(.cpu) {
            let base = try Self.requestParameters(temperature: 1, topP: 1)
            let free = Self.draws(base)
            #expect(Set(free).count > 4, "control: the unfiltered draws must spread (\(Set(free)))")
            #expect(Self.draws(try Self.requestParameters(temperature: 1, topP: 1, topK: 1)).allSatisfy { $0 == 3 })
            #expect(Set(Self.draws(try Self.requestParameters(temperature: 1, topP: 1, topK: 2))).isSubset(of: [3, 9]))
            #expect(Self.draws(try Self.requestParameters(temperature: 1, topP: 0.01)).allSatisfy { $0 == 3 })
            #expect(Set(Self.draws(try Self.requestParameters(temperature: 1, topP: 1, minP: 0.9))).isSubset(of: [3, 9]))
            #expect(Self.draws(try Self.requestParameters(temperature: 0.001, topP: 1)).allSatisfy { $0 == 3 })
            // The bundle default (top_p 0.95) is applied when the request sends nothing.
            let bundleDraws = Self.draws(try Self.requestParameters())
            let p95 = TopPSampler(temperature: 1, topP: 0.95).filteredLogprobs(Self.row())
            let kept = Set((0 ..< 64).filter { p95[0, $0].item(Float.self) > -Float.infinity })
            #expect(kept.count < 64 && Set(bundleDraws).isSubset(of: kept))
        }
    }

    @Test("dSpark sampled verify uses the same filters (DSparkSampling from the request parameters)")
    func dsparkSamplingHonorsEveryFilter() throws {
        try Device.withDefaultDevice(.cpu) {
            let rows = concatenated(Array(repeating: Self.row(), count: 32), axis: 0)
            func picks(_ p: GenerateParameters) -> [Int] {
                guard let s = DSparkSampling(parameters: p) else { return [] }
                return s.pick(rows, noise: s.noise(first: 100, count: 32, vocab: 64)).asArray(Int32.self).map(Int.init)
            }
            let free = picks(try Self.requestParameters(temperature: 1, topP: 1))
            #expect(free.count == 32 && Set(free).count > 4, "control: unfiltered noise draws spread")
            #expect(picks(try Self.requestParameters(temperature: 1, topP: 1, topK: 1)).allSatisfy { $0 == 3 })
            #expect(picks(try Self.requestParameters(temperature: 1, topP: 0.01)).allSatisfy { $0 == 3 })
            #expect(Set(picks(try Self.requestParameters(temperature: 1, topP: 1, minP: 0.9))).isSubset(of: [3, 9]))
            #expect(DSparkSampling(parameters: try Self.requestParameters(temperature: 0)) == nil)
            // The bundle defaults reach it too.
            let d = try #require(DSparkSampling(parameters: try Self.requestParameters()))
            #expect(d.temperature == 1.0 && d.filters != nil)
        }
    }

    /// Each row the iterator hands its sampler: the argmax (first index), the top-2 logits, and how many entries equal
    /// the max. It returns the logits unchanged, so the sampler draws exactly as it would without it.
    final class TopTwoRecorder: LogitProcessor, @unchecked Sendable {
        struct Row {
            let argmax: Int
            let first: Float
            let second: Float
            let atMax: Int
        }
        var rows: [Row] = []
        func prompt(_ prompt: MLXArray) {}
        func process(logits: MLXArray) -> MLXArray {
            let v = logits.reshaped(-1).asType(.float32).asArray(Float.self)  // exact for bf16 / f16 rows
            let first = v.max()!
            let argmax = v.firstIndex(of: first)!
            let second = v.enumerated().filter { $0.offset != argmax }.map(\.element).max()!
            rows.append(Row(argmax: argmax, first: first, second: second, atMax: v.filter { $0 == first }.count))
            return logits
        }
        func independentCopy() -> Self { self }
        func didSample(token: MLXArray) {}
    }

    /// min_p 1 keeps every token whose probability equals the max, so it equals greedy only where the top-2 logits
    /// differ; at an exact tie it may draw either tied token. The model's logits are bf16, and exact top-2 ties occur:
    /// prompt seed 5 has one at step 10 on MLX with the upstream NAX split-K quantized-matmul route (#4682) and none
    /// without it. The prompt (seed 7) is tie-free in every returned step with and without that route, and the test
    /// first proves that on the iterator's own logits, so a tie fails as a fixture problem, not as a filter bug.
    @Test("TokenIterator on an M3 model: top_k 1 and min_p 1 at temperature 1 decode the greedy tokens")
    func tokenIteratorHonorsFilters() throws {
        try Device.withDefaultDevice(.gpu) {
            let model = try MiniMaxM3DSparkTests.m3Model(seed: 3)
            let prompt = MiniMaxM3DSparkTests.ids(301, seed: 7)
            let input = LMInput(tokens: MLXArray(prompt.map { Int32($0) }))
            func request(_ p: GenerateParameters) -> GenerateParameters {
                var p = p
                p.maxTokens = 12
                p.randomSeed = 7
                return p
            }
            func run(_ p: GenerateParameters) throws -> [Int] {
                var it = try TokenIterator(input: input, model: model, parameters: request(p))
                var out: [Int] = []
                while let t = it.next() { out.append(t) }
                return out
            }
            let greedyParameters = request(try Self.requestParameters(temperature: 0))
            let greedy = try run(greedyParameters)
            #expect(greedy.count == 12)
            // Precondition, on the TokenIterator(parameters:) path itself: the same iterator prepares the prompt again
            // on a fresh cache carrying its aligned-prefill plan, with the recorder as its processor. That run must
            // return the greedy tokens, each step's argmax must be the greedy token, and no step may have an exact tie.
            var recordedIterator = try TokenIterator(input: input, model: model, parameters: greedyParameters)
            let recorder = TopTwoRecorder()
            recordedIterator.cache = model.newCache(parameters: greedyParameters)
            AlignedPrefill.install(recordedIterator.alignedPrefillPlan, into: recordedIterator.cache)
            recordedIterator.processor = recorder
            try recordedIterator.prepare(input: input, windowSize: greedyParameters.prefillStepSize)
            var recorded: [Int] = []
            while let t = recordedIterator.next() { recorded.append(t) }
            try #require(recorded == greedy, "the recorded run left the greedy run's path: \(recorded) vs \(greedy)")
            try #require(recorder.rows.count >= greedy.count)
            for (step, row) in recorder.rows.prefix(greedy.count).enumerated() {
                try #require(row.argmax == greedy[step], "step \(step): argmax \(row.argmax), greedy \(greedy[step])")
                try #require(
                    row.atMax == 1 && row.first > row.second,
                    "fixture: exact top-2 tie at step \(step) (\(row.atMax) at \(row.first)); choose a tie-free prompt")
            }
            #expect(try run(try Self.requestParameters(temperature: 1, topP: 1, topK: 1)) == greedy)
            #expect(try run(try Self.requestParameters(temperature: 1, topP: 1, minP: 1)) == greedy)
            // Control: the same seed without a filter samples something else (the filters are what made it greedy).
            let free = try run(try Self.requestParameters(temperature: 5, topP: 1))
            #expect(free != greedy, "control: unfiltered sampling at temperature 5 reproduced greedy")
        }
    }
}
