import Foundation
import Testing

@Suite("Generation config factory source coverage")
struct GenerationConfigFactorySourceCoverageTests {
    @Test("LLM and VLM factories carry generation_config defaults into ModelConfiguration")
    func factoriesCarryGenerationConfigDefaults() throws {
        let repoRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let files = [
            "Libraries/MLXLLM/LLMModelFactory.swift",
            "Libraries/MLXVLM/VLMModelFactory.swift",
        ]

        var failures: [String] = []
        for relativePath in files {
            let source = try String(
                contentsOf: repoRoot.appending(path: relativePath),
                encoding: .utf8)
            if !source.contains("generationDefaults: generationConfig") {
                failures.append(relativePath)
            }
        }

        #expect(
            failures.isEmpty,
            "Factories decode generation_config.json but do not carry defaults into ModelConfiguration: \(failures)"
        )
    }

    @Test("VLM JANG load uses quantization container, not deprecated alias")
    func vlmJangLoadUsesQuantizationContainer() throws {
        let repoRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let source = try String(
            contentsOf: repoRoot.appending(path: "Libraries/MLXVLM/VLMModelFactory.swift"),
            encoding: .utf8)

        #expect(source.contains("baseConfig.quantizationContainer?.quantization"))
        #expect(!source.contains("baseConfig.quantization : nil"))
    }
}
