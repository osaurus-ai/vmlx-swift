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

    @Test("TokenIterator on an M3 model: top_k 1 and min_p 1 at temperature 1 decode the greedy tokens")
    func tokenIteratorHonorsFilters() throws {
        try Device.withDefaultDevice(.gpu) {
            let model = try MiniMaxM3DSparkTests.m3Model(seed: 3)
            let prompt = MiniMaxM3DSparkTests.ids(301, seed: 5)
            func run(_ p: GenerateParameters) throws -> [Int] {
                var p = p
                p.maxTokens = 12
                p.randomSeed = 7
                var it = try TokenIterator(
                    input: LMInput(tokens: MLXArray(prompt.map { Int32($0) })), model: model, parameters: p)
                var out: [Int] = []
                while let t = it.next() { out.append(t) }
                return out
            }
            let greedy = try run(try Self.requestParameters(temperature: 0))
            #expect(greedy.count == 12)
            #expect(try run(try Self.requestParameters(temperature: 1, topP: 1, topK: 1)) == greedy)
            #expect(try run(try Self.requestParameters(temperature: 1, topP: 1, minP: 1)) == greedy)
            // Control: the same seed without a filter samples something else (the filters are what made it greedy).
            let free = try run(try Self.requestParameters(temperature: 5, topP: 1))
            #expect(free != greedy, "control: unfiltered sampling at temperature 5 reproduced greedy")
        }
    }
}
