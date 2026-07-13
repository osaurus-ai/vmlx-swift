// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import Testing

/// The stop-string matcher withholds up to `maxStopLen - 1` characters while it
/// disambiguates a possible stop. That held text PRECEDES a tool call in the
/// model's output — but consumers stop forwarding text the moment a `.toolCall`
/// event lands (deliberate no-leak suppression of post-tool prose). So the
/// matcher must be drained BEFORE the event, or the visible answer is silently
/// cut mid-word.
///
/// `Evaluate.swift`'s solo handler does this, and `ToolCallStopMatcherOrderingTests`
/// pins it behaviorally. `BatchEngine` — the path Osaurus actually serves with —
/// shipped without it and reproduced the same truncation for any API client that
/// sets `stop` (LangChain, aider, Cline all do) while the model calls a tool.
///
/// BatchEngine builds its emit closures inline rather than through a reusable
/// handler, so there is no seam to drive without standing up a real engine and
/// model. Pin the invariant in the source instead, the same way
/// `Hy3CompiledDecodeGuardSourceTests` does: the `.toolCall` case must drain the
/// matcher before it yields.
@Suite("BatchEngine drains the stop matcher before .toolCall")
struct BatchStopTailBeforeToolCallSourceTests {

    private static func batchEngineSource() throws -> String {
        // Tests run from the package root.
        let path = "Libraries/MLXLMCommon/BatchEngine/BatchEngine.swift"
        let url = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent(path)
        return try String(contentsOf: url, encoding: .utf8)
    }

    /// Isolate `emitRouted`'s `.toolCall` case and require that it flushes the
    /// stop matcher before yielding the event.
    @Test("the .toolCall branch of emitRouted flushes the stop matcher first")
    func toolCallBranchDrainsStopMatcher() throws {
        let source = try Self.batchEngineSource()

        // Anchor inside `emitRouted` — a doc comment earlier in the file also
        // contains the literal `case .toolCall: break`, and matching that
        // instead makes this guard fail for the wrong reason.
        guard let fn = source.range(of: "func emitRouted(") else {
            Issue.record("`emitRouted` is gone — rewrite this guard")
            return
        }
        let body = source[fn.upperBound...]
        guard let start = body.range(of: "case .toolCall:") else {
            Issue.record("emitRouted's `case .toolCall:` branch is gone — rewrite this guard")
            return
        }
        // The branch runs until the next `case .` at the same level.
        let rest = body[start.upperBound...]
        let end = rest.range(of: "\n                case .")?.lowerBound ?? rest.endIndex
        let branch = String(rest[..<end])

        #expect(
            branch.contains("stopMatcher.flush()"),
            """
            BatchEngine's `.toolCall` branch must drain the stop-string matcher \
            BEFORE yielding the event. Without it, prose the matcher is holding \
            for disambiguation is emitted after the tool call and silently \
            dropped by consumers, cutting the answer mid-word. Branch body was:
            \(branch)
            """)

        // The flush must come before the yield, not after it.
        if let flushAt = branch.range(of: "stopMatcher.flush()"),
            let yieldAt = branch.range(of: "continuation.yield(event)")
        {
            #expect(
                flushAt.lowerBound < yieldAt.lowerBound,
                "the matcher must be drained BEFORE the .toolCall event is yielded")
        } else {
            Issue.record("could not locate both the flush and the yield in the branch")
        }
    }
}
