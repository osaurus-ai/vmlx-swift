import Foundation
import Testing

@testable import MLXLMCommon

/// Regression guard for the "Gemma + tools scrambles prose" bug.
///
/// When a Gemma tool format is active, every generated chunk flows through
/// `ToolCallProcessor`. Gemma's bare-call fallback buffers any trailing
/// "c"/"ca"/"cal"/"call" (a possible start of the `call:` tool marker). When a
/// buffered fragment did NOT continue into a tool call, the held text was
/// neither flushed in order nor cleared, so ordinary prose lost and reordered
/// characters ("cobblestone" -> "obblestone", a trailing "call" dumped at EOS,
/// etc.). Gemma is the only family with bare-call fallback, which is why only
/// Gemma corrupted prose.
@Suite("Gemma tool formats preserve plain prose")
struct Gemma4ProseScrambleTests {

    private func chunked(_ s: String, size: Int) -> [String] {
        var r: [String] = []
        var i = s.startIndex
        while i < s.endIndex {
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

    // Prose dense with `c` words (each a false `call:` prefix) plus whitespace.
    private let prose = """
        Palm Springs is a spectacular desert oasis. Ivory towers with gilded \
        domes rise above cobblestone streets, a masterpiece of architectural \
        elegance and a breathtaking getaway near the Ritz-Carlton.
        """

    @Test("gemma4 passes plain prose through byte-exact at every chunk size")
    func gemma4PreservesProse() {
        for size in [1, 2, 3, 4, 8] {
            let proc = ToolCallProcessor(format: .gemma4, tools: nil)
            let out = visible(chunked(prose, size: size), proc)
            #expect(out == prose, "gemma4 size=\(size) altered prose: \(out.debugDescription)")
            #expect(proc.toolCalls.isEmpty, "gemma4 size=\(size) fabricated a tool call from prose")
        }
    }

    @Test("gemma (legacy) passes plain prose through byte-exact")
    func gemmaLegacyPreservesProse() {
        for size in [1, 2, 3, 4, 8] {
            let proc = ToolCallProcessor(format: .gemma, tools: nil)
            let out = visible(chunked(prose, size: size), proc)
            #expect(out == prose, "gemma size=\(size) altered prose: \(out.debugDescription)")
        }
    }
}
