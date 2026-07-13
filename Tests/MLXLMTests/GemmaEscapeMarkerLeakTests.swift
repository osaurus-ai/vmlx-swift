// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import Testing

@testable import MLXLMCommon

/// Gemma-4's `<|"|>` escape marker is a *structural delimiter* of the bare-call
/// wire format (`call:name{key:<|"|>value<|"|>}`). It is never legitimate
/// argument data.
///
/// Live capture (Qwen-Image-Edit delegation, Gemma-4-12B-it-MXFP8 orchestrating):
/// the model emitted an unbalanced marker and the parser handed the tool a
/// `source_path` ending in `<|"|>`, which the image runtime rejected as a
/// non-existent file. The marker leaked because `takeRawValue` terminates only
/// on `,` `]` `}` — never on the escape marker — so a stray delimiter is
/// swallowed verbatim into an unquoted scalar.
@Suite("Gemma escape marker never leaks into argument values")
struct GemmaEscapeMarkerLeakTests {
    private let parser = Gemma4ToolCallParser()

    private func string(_ call: ToolCall, _ key: String) -> String? {
        guard case .string(let value) = call.function.arguments[key] else { return nil }
        return value
    }

    /// LIVE-OBSERVED: opening marker absent, closing marker present.
    /// The raw scalar swallows the trailing delimiter.
    @Test("stray trailing marker does not leak into a raw scalar")
    func strayTrailingMarkerStripped() throws {
        let wire =
            #"<|tool_call>call:image_edit{source_path:/tmp/a.png<|"|>,prompt:<|"|>make it blue<|"|>}<tool_call|>"#
        let call = try #require(parser.parse(content: wire, tools: nil))

        #expect(call.function.name == "image_edit")
        #expect(string(call, "source_path") == "/tmp/a.png")
        #expect(string(call, "prompt") == "make it blue")
    }

    /// KNOWN AMBIGUITY, deliberately not "fixed": when a string opens with the
    /// marker but never closes it, the delimiter count is odd and *which* marker
    /// is missing is unknowable — the marker is a paired toggle, like a quote.
    /// Any recovery here would be a guess at model intent, so the parser does not
    /// attempt one. What IS enforced universally is the invariant below: whatever
    /// value comes out, it never contains a raw delimiter.
    @Test("no parsed string value ever contains a raw delimiter")
    func noValueEverContainsARawDelimiter() throws {
        let wires = [
            #"<|tool_call>call:f{a:/tmp/a.png<|"|>,b:<|"|>x<|"|>}<tool_call|>"#,
            #"<|tool_call>call:f{a:<|"|>/tmp/a.png<|"|>}<tool_call|>"#,
            #"<|tool_call>call:f{a:<|"|>unterminated}<tool_call|>"#,
            #"<|tool_call>call:f{a:[<|"|>x<|"|>,plain<|"|>]}<tool_call|>"#,
        ]
        for wire in wires {
            guard let call = parser.parse(content: wire, tools: nil) else { continue }
            for (key, value) in call.function.arguments {
                if case .string(let s) = value {
                    #expect(
                        s.contains(#"<|"|>"#) == false,
                        "delimiter leaked into \(key) for wire: \(wire)")
                }
            }
        }
    }

    /// Unterminated string at the end of the body: `parseValue` returns nil,
    /// which `break`s the key/value loop — dropping this argument *and* every
    /// argument after it. A truncated-but-present value is strictly better than
    /// a silently missing one.
    @Test("unterminated trailing string still yields the argument")
    func unterminatedTrailingStringStillParses() throws {
        let wire = #"<|tool_call>call:image_edit{prompt:<|"|>make it blue}<tool_call|>"#
        let call = try #require(parser.parse(content: wire, tools: nil))

        #expect(call.function.name == "image_edit")
        let prompt = try #require(string(call, "prompt"))
        #expect(prompt.contains("make it blue"))
        #expect(prompt.contains(#"<|"|>"#) == false)
    }

    /// Regression guard: the well-formed shape must be untouched by the above.
    @Test("well-formed escaped values are unchanged")
    func wellFormedUnchanged() throws {
        let wire =
            #"<|tool_call>call:image_edit{source_path:<|"|>/tmp/a.png<|"|>,prompt:<|"|>make it blue<|"|>}<tool_call|>"#
        let call = try #require(parser.parse(content: wire, tools: nil))

        #expect(string(call, "source_path") == "/tmp/a.png")
        #expect(string(call, "prompt") == "make it blue")
    }

    /// Regression guard: a raw scalar that merely *contains* an angle bracket or
    /// pipe (but not the full marker) is still ordinary data.
    @Test("escaped values keep lone angle brackets and pipes")
    func loneBracketsSurvive() throws {
        let wire =
            #"<|tool_call>call:shell_run{command:<|"|>echo a | grep b > out.txt<|"|>}<tool_call|>"#
        let call = try #require(parser.parse(content: wire, tools: nil))

        #expect(string(call, "command") == "echo a | grep b > out.txt")
    }
}
