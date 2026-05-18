import Foundation
import Testing

@Suite("RunBench applicability gates", .serialized)
struct RunBenchApplicabilityFocusedTests {
    @Test("pure paged cache-hit bench skips path-dependent cache topologies")
    func cacheHitBenchSkipsPathDependentCaches() throws {
        let source = try Self.source("RunBench/Bench.swift")
        let body = try Self.functionBody(named: "runBatchEngineCacheHit", in: source)

        #expect(body.contains("cacheRequiresDiskBackedCoordinatorRestore"))
        #expect(body.contains("not applicable"))
    }

    @Test("TurboQuant B2 bench skips path-dependent cache topologies")
    func turboQuantB2BenchSkipsPathDependentCaches() throws {
        let source = try Self.source("RunBench/Bench.swift")
        let body = try Self.functionBody(named: "runBatchEngineTurboQuantB2", in: source)

        #expect(body.contains("cacheRequiresDiskBackedCoordinatorRestore"))
        #expect(body.contains("not applicable"))
    }

    private static func source(_ relativePath: String) throws -> String {
        try String(contentsOfFile: relativePath, encoding: .utf8)
    }

    private static func functionBody(named name: String, in source: String) throws -> Substring {
        guard let start = source.range(of: "func \(name)") else {
            throw TestError.missingFunction(name)
        }
        guard let open = source[start.lowerBound...].firstIndex(of: "{") else {
            throw TestError.missingOpeningBrace(name)
        }
        var depth = 0
        var index = open
        while index < source.endIndex {
            if source[index] == "{" {
                depth += 1
            } else if source[index] == "}" {
                depth -= 1
                if depth == 0 {
                    return source[open...index]
                }
            }
            index = source.index(after: index)
        }
        throw TestError.missingClosingBrace(name)
    }

    enum TestError: Error {
        case missingFunction(String)
        case missingOpeningBrace(String)
        case missingClosingBrace(String)
    }
}
