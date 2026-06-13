import Foundation
import Testing

@testable import MLXLMCommon

/// Deterministic fuzz proof for streaming tool-call processing.
///
/// Invariant: when generated text contains NO tool call, the reassembled
/// visible output equals the input byte-for-byte — for every tool format, at
/// random chunk sizes. This is the class of bug behind "Gemma + tools scrambles
/// prose" (held tool-marker fragments must never drop/reorder ordinary text)
/// and the inline-JSON whitespace-before-brace drop (a chunk like " {" must keep
/// its leading space). Seeded SplitMix64 makes failures reproducible.
@Suite("ToolCallProcessor prose-fidelity fuzz")
struct ToolCallProcessorFuzzTests {

    private struct RNG {
        var s: UInt64
        init(_ seed: UInt64) { s = seed }
        mutating func next() -> UInt64 {
            s &+= 0x9E37_79B9_7F4A_7C15
            var z = s
            z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
            z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
            return z ^ (z >> 31)
        }
        mutating func int(_ n: Int) -> Int { n <= 0 ? 0 : Int(next() % UInt64(n)) }
        mutating func pick<T>(_ a: [T]) -> T { a[int(a.count)] }
    }

    private let vocab = [
        "call", "calling", "called", "cobblestone", "cactus", "city", "casual", "create",
        "the", "a", "desert", "pool", "complex", "function", "name", "city:", "value",
        "weather", "get", "tool", "use", "{", "}", "(", ")", ":", ",", "- ", "* ", "`code`",
        "\n", "\n\n", "## Heading", "perfect", "masterpiece", "getaway", "architecture",
        "kids'", "family-friendly", "Ritz-Carlton", "spectacular", "and", "with", "gilded",
    ]

    private func randomProse(_ rng: inout RNG, words: Int) -> String {
        var out = ""
        for i in 0..<words {
            if i > 0 { out += " " }
            out += rng.pick(vocab)
        }
        return out
    }

    private func chunked(_ s: String, _ rng: inout RNG) -> [String] {
        var r: [String] = []
        var i = s.startIndex
        while i < s.endIndex {
            let size = 1 + rng.int(5)
            let j = s.index(i, offsetBy: size, limitedBy: s.endIndex) ?? s.endIndex
            r.append(String(s[i..<j]))
            i = j
        }
        return r
    }

    private func visible(_ chunks: [String], _ proc: ToolCallProcessor) -> String {
        var out = ""
        for c in chunks { if let v = proc.processChunk(c) { out += v } }
        if let tail = proc.processEOS() { out += tail }
        return out
    }

    @Test("prose with no tool call survives byte-exact (all formats, random chunking)")
    func proseFidelity() {
        let formats: [ToolCallFormat] = [
            .json, .gemma, .gemma4, .xmlFunction, .glm4, .nemotron, .dsml, .lfm2, .step,
        ]
        var failures = 0
        var first = ""
        for f in formats {
            for seed in UInt64(1)...400 {
                var rng = RNG(seed &* 0x100_0000 &+ UInt64(f.rawValue.count))
                let prose = randomProse(&rng, words: 8 + rng.int(60))
                let proc = ToolCallProcessor(format: f, tools: nil)
                let got = visible(chunked(prose, &rng), proc)
                if got != prose {
                    failures += 1
                    if first.isEmpty {
                        first = "format=\(f.rawValue) seed=\(seed)\n IN:  \(prose.debugDescription)\n OUT: \(got.debugDescription)"
                    }
                }
            }
        }
        #expect(failures == 0, "prose-fidelity fuzz failures=\(failures)\n\(first)")
    }
}
