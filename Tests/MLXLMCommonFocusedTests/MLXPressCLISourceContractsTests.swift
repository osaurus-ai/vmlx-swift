import Testing
import Foundation

@Suite("MLXPress CLI source contracts")
struct MLXPressCLISourceContractsTests {
    @Test("omitted sampler flags resolve from bundle generation config")
    func omittedSamplerFlagsResolveFromBundleGenerationConfig() throws {
        let source = try repositoryFile("Sources/MLXPressCLI/main.swift")

        #expect(source.contains("var temperature: Float?"))
        #expect(source.contains("var topP: Float?"))
        #expect(source.contains("session.container.defaultGenerateParameters"))
        #expect(source.contains("if let temperature = options.temperature"))
        #expect(source.contains("if let topP = options.topP"))
        #expect(source.contains("\"temperature\": jsonFloat(resolvedGenerateParameters.temperature)"))
        #expect(source.contains("\"top_p\": jsonFloat(resolvedGenerateParameters.topP)"))
        #expect(!source.contains("var temperature: Float = 0"))
        #expect(!source.contains("var topP: Float = 1"))
    }

    @Test("thinking-on validation guidance rejects private-reasoning-only prompts")
    func thinkingOnValidationGuidanceRejectsPrivateReasoningOnlyPrompts() throws {
        let source = try repositoryFile("Sources/MLXPressCLI/main.swift")

        #expect(source.contains("--thinking is tri-state: omitted leaves the model's template default untouched"))
        #expect(source.contains("var enableThinking: Bool?"))
        #expect(source.contains("if let enableThinking"))
        #expect(!source.contains("var enableThinking = false"))
        #expect(source.contains("thinking-on validation prompts must ask for a visible final answer"))
        #expect(source.contains("Do not use prompts that ask the model to think privately"))
        #expect(source.contains("--min-visible-chars"))
        #expect(source.contains("--fail-on-length-stop"))
    }

    @Test("resident load flag disables safetensors mmap explicitly")
    func residentLoadFlagDisablesSafetensorsMmapExplicitly() throws {
        let source = try repositoryFile("Sources/MLXPressCLI/main.swift")

        #expect(source.contains("var useMmapSafetensors = true"))
        #expect(source.contains(#"case "--resident-load":"#))
        #expect(source.contains(#"useMmapSafetensors = !(try Self.parseBool(args[index], option: "--resident-load"))"#))
        #expect(source.contains(#"case "--enable-resident-load":"#))
        #expect(source.contains("useMmapSafetensors: useMmapSafetensors"))
        #expect(source.contains("--resident-load on disables safetensors mmap"))
    }
}

private func repositoryFile(_ relativePath: String) throws -> String {
    let cwd = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
    let candidates = [
        cwd.appendingPathComponent(relativePath),
        cwd.deletingLastPathComponent().appendingPathComponent(relativePath),
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent(relativePath),
    ]
    for url in candidates {
        if FileManager.default.fileExists(atPath: url.path) {
            return try String(contentsOf: url, encoding: .utf8)
        }
    }
    throw NSError(
        domain: "MLXPressCLISourceContractsTests",
        code: 1,
        userInfo: [NSLocalizedDescriptionKey: "Could not locate \(relativePath)"])
}
