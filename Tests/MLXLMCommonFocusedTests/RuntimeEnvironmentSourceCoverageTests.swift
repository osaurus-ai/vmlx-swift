// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// The migration is finishable exactly once. This asserts it stays finished: every legacy
// VMLINUX_ literal left in the library must be reachable only as a FALLBACK, never as the sole
// spelling a site reads. Without this, straggler thirty-eight arrives with the next model.

import Foundation
import Testing

@Suite("Runtime environment naming source coverage")
struct RuntimeEnvironmentSourceCoverageTests {

    /// Sites where the legacy name legitimately appears: the accessor that defines the rule, and
    /// the `legacy…` constants that exist precisely to hold it.
    private static let allowedLegacyHolders = [
        "RuntimeEnvironment.swift",          // defines legacyPrefix / legacyName(of:)
        "RuntimeMoETopKOverride.swift",      // legacyEnvironmentVariable
        "RuntimeAcceleration.swift",         // legacyEnvironmentVariable
        "DeepseekV4ReasoningPolicy.swift",   // legacyRawMax… / legacyForceDirectRail…
        "ModelFactory.swift",                // writes both spellings for the C++ consumer
        "JANGTQStreamingExperts.swift",      // writes both spellings for the C++ consumer
    ]

    private func swiftSources(under root: String) -> [(String, String)] {
        let fm = FileManager.default
        guard let e = fm.enumerator(atPath: root) else { return [] }
        var out: [(String, String)] = []
        for case let p as String in e where p.hasSuffix(".swift") {
            if let text = try? String(contentsOfFile: root + "/" + p, encoding: .utf8) {
                out.append((p, text))
            }
        }
        return out
    }

    @Test("no library site reads a legacy name as its only spelling")
    func noLegacyOnlyReads() throws {
        let sources = swiftSources(under: "Libraries")
        try #require(!sources.isEmpty, "found no sources — the path is wrong, not the code")

        var offenders: [String] = []
        for (path, text) in sources {
            let file = path.split(separator: "/").last.map(String.init) ?? path
            if Self.allowedLegacyHolders.contains(file) { continue }
            let lines = text.split(separator: "\n", omittingEmptySubsequences: false).map(String.init)
            for (i, line) in lines.enumerated() {
                guard line.contains("\"VMLINUX_") else { continue }
                let trimmed = line.trimmingCharacters(in: .whitespaces)
                if trimmed.hasPrefix("//") { continue }        // prose may name it

                // The fallback is routinely written across lines:
                //     let raw = env["VMLX_X"]
                //         ?? env["VMLINUX_X"]
                // so the unit is the STATEMENT, not the line — and the counterpart must be the
                // SAME variable, or a neighbouring unrelated VMLX_ read would excuse it.
                let window = lines[max(0, i - 2)...min(lines.count - 1, i + 1)].joined(separator: "\n")
                for name in Self.legacyNames(in: line) where !window.contains("\"VMLX_\(name)\"") {
                    offenders.append("\(path): \(trimmed)")
                }
            }
        }
        #expect(
            offenders.isEmpty,
            "legacy-only read site(s):\n\(offenders.joined(separator: "\n"))")
    }

    /// The bare names inside `"VMLINUX_…"` literals on one line.
    private static func legacyNames(in line: String) -> [String] {
        var out: [String] = []
        var rest = Substring(line)
        while let open = rest.range(of: "\"VMLINUX_") {
            let after = rest[open.upperBound...]
            guard let close = after.firstIndex(of: "\"") else { break }
            out.append(String(after[..<close]))
            rest = after[close...]
        }
        return out
    }

    /// The inverse: the accessor must actually be in use, or the rule above passes vacuously on a
    /// library that simply stopped reading the environment.
    @Test("the accessor is used")
    func accessorIsUsed() throws {
        let uses = swiftSources(under: "Libraries")
            .filter { $0.1.contains("RuntimeEnvironment.value(") || $0.1.contains("RuntimeEnvironment.flag(") }
        #expect(uses.count >= 8, "only \(uses.count) file(s) use the accessor")
    }
}
