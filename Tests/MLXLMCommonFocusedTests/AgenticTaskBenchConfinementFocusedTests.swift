// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import Testing

@Suite("Agentic task benchmark shell confinement")
struct AgenticTaskBenchConfinementFocusedTests {
    private func source() throws -> String {
        try String(
            contentsOf: FocusedMLXTestSupport.sourceURL("RunBench/AgenticTaskBench.swift"),
            encoding: .utf8)
    }

    @Test("run_shell is a seatbelt-confined non-login shell")
    func shellUsesTaskScopedSeatbeltProfile() throws {
        let source = try source()
        #expect(source.contains(#"process.executableURL = URL(fileURLWithPath: "/usr/bin/sandbox-exec")"#))
        #expect(source.contains(#"(deny file-write*)"#))
        #expect(source.contains(#"(allow file-write* (subpath "\(task)") (literal "/dev/null"))"#))
        #expect(source.contains(#"(deny file-read* (subpath "\(home)") (subpath "\(temporary)"))"#))
        #expect(source.contains(#""/bin/bash", "-c", command"#))
        #expect(!source.contains(#"process.arguments = ["-lc", command]"#))
        #expect(source.contains("run_shell benchmark confinement is unavailable"))
        #expect(source.contains("Darwin.realpath(url.path, &buffer)"))
        #expect(source.contains("canonicalSandboxPath(taskDirectory)"))
    }

    @Test("run_shell has a bounded wall clock and terminates its process group")
    func shellHasBoundedProcessGroupWatchdog() throws {
        let source = try source()
        #expect(source.contains("BENCH_AGENT_SHELL_TIMEOUT_SECONDS"))
        #expect(source.contains("Darwin.setpgid(process.processIdentifier"))
        #expect(source.contains("Darwin.kill(-process.processIdentifier, SIGTERM)"))
        #expect(source.contains("[terminated: shell command exceeded "))
    }

    @Test("run_shell redirects temporary and home state into the task fixture")
    func shellEnvironmentIsFixtureScoped() throws {
        let source = try source()
        #expect(source.contains(#"environment["HOME"] = dir.path"#))
        #expect(source.contains(#"environment["TMPDIR"] = dir.path"#))
        #expect(source.contains("process.currentDirectoryURL = dir"))
    }

    @Test("agentic mode dispatches before the legacy model load")
    func agenticModeOwnsExactlyOneModelContext() throws {
        let bench = try String(
            contentsOf: FocusedMLXTestSupport.sourceURL("RunBench/Bench.swift"), encoding: .utf8)
        let dispatch = try #require(bench.range(of: #"env["BENCH_AGENTIC_TASKS"]"#))
        let legacyLoad = try #require(bench.range(of: #"print("Loading...")"#))
        #expect(dispatch.lowerBound < legacyLoad.lowerBound)
        #expect(bench.components(separatedBy: #"env["BENCH_AGENTIC_TASKS"]"#).count == 2)
    }

    @Test("agentic score uses bundle sampling and a reproducible seed")
    func scoreUsesBundleGenerationContract() throws {
        let source = try source()
        #expect(source.contains("generationConfig: context.configuration.generationDefaults"))
        #expect(source.contains(#"environment["BENCH_AGENT_SEED"]"#))
        #expect(source.contains("params.randomSeed = seed"))
        #expect(source.contains(#"sampling=\(explicitTemperature == nil ? "bundle-defaults""#))
        #expect(!source.contains(#".flatMap(Float.init) ?? 0"#))
    }

    @Test("independent cases clear only allocator cache and expose active bytes")
    func independentCasesDoNotAccumulateAllocatorResidue() throws {
        let source = try source()
        #expect(source.contains("let activeBeforeClear = MLX.Memory.activeMemory"))
        #expect(source.contains("let cachedBeforeClear = MLX.Memory.cacheMemory"))
        #expect(source.contains("MLX.Memory.clearCache()"))
        #expect(source.contains("memory activeBytes="))
    }
}
