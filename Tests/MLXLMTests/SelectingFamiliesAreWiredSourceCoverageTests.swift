// Copyright © 2026 osaurus-eval contributors

import Foundation
import Testing

/// The defect this exists to make impossible: a family can implement `ModelComponentMapping`
/// perfectly — resolve the request, build only the asked-for towers, report the right
/// `modalities` — and still have the request DROPPED, because the registry entry says
/// `create(Config.self, Model.init)` instead of `createSelecting(Config.self,
/// Model.init(_:requesting:))`. That unwiring COMPILES WITH ZERO ERRORS, and a test that
/// constructs the model directly passes just as happily, because it never touches the table.
///
/// So the guard cannot be a behaviour test on any one family — it has to be a statement about
/// the table itself: if a family declares that it selects, its registration must say so.
@Suite("selecting families are registered through the selecting creator")
struct SelectingFamiliesAreWiredSourceCoverageTests {

    private static var repoRoot: URL {
        var root = URL(fileURLWithPath: #filePath)
        for _ in 0 ..< 3 { root.deleteLastPathComponent() }
        return root
    }

    /// Model classes that conform to `ModelComponentMapping`, by file.
    private static func selectingClasses() throws -> [String] {
        let modelsRoot = Self.repoRoot.appending(path: "Libraries/MLXVLM/Models")
        let fm = FileManager.default
        guard let entries = try? fm.contentsOfDirectory(atPath: modelsRoot.path) else { return [] }
        var names: [String] = []
        for entry in entries where entry.hasSuffix(".swift") {
            let text = try String(
                contentsOf: modelsRoot.appending(path: entry), encoding: .utf8)
            guard text.contains("ModelComponentMapping") else { continue }
            // The conforming class is the one whose declaration list mentions the protocol; the
            // declaration may wrap across lines, so scan from each `class` to its opening brace.
            var scan = text.startIndex
            while let decl = text.range(of: "public class ", range: scan ..< text.endIndex) {
                guard let brace = text.range(of: "{", range: decl.upperBound ..< text.endIndex)
                else { break }
                let header = String(text[decl.upperBound ..< brace.lowerBound])
                if header.contains("ModelComponentMapping") {
                    let name = header.prefix { $0.isLetter || $0.isNumber || $0 == "_" }
                    if !name.isEmpty { names.append(String(name)) }
                }
                scan = brace.upperBound
            }
        }
        return names
    }

    @Test("a family that selects is not registered through the plain creator")
    func selectingFamiliesUseTheSelectingCreator() throws {
        let classes = try Self.selectingClasses()
        try #require(
            !classes.isEmpty,
            "found no ModelComponentMapping conformers — the scan is wrong, not the code")

        let factory = try String(
            contentsOf: Self.repoRoot.appending(path: "Libraries/MLXVLM/VLMModelFactory.swift"),
            encoding: .utf8)

        var unwired: [String] = []
        for name in Set(classes) {
            // A registration of this class through the PLAIN creator drops the request.
            if factory.contains("create(\(name)Configuration.self, \(name).init)")
                || factory.contains(", \(name).init)")
            {
                unwired.append(name)
            }
        }
        #expect(
            unwired.isEmpty,
            """
            these families implement ModelComponentMapping but are registered through the plain \
            creator, so the construction request never reaches them: \
            \(unwired.sorted().joined(separator: ", ")). \
            Use createSelecting(…, Model.init(_:requesting:)).
            """)
    }
}
