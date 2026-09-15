import Foundation
import MLXLMCommon
import MLXVLM
import Testing

@Suite("Processor registry admission evidence")
struct ProcessorTypeRegistryEvidenceTests {
    private enum ProbeError: Error { case creatorWasCalled }

    @Test("registration and construction use the same table")
    func registrationAndConstruction() async throws {
        let registry = ProcessorTypeRegistry()
        #expect(!registry.containsProcessorType("custom"))
        let initialVersion = registry.registrationVersion
        do {
            _ = try await registry.createModel(
                configuration: Data(), processorType: "custom", tokenizer: TestTokenizer())
            Issue.record("An unregistered processor was constructed")
        } catch ModelFactoryError.unsupportedProcessorType(let type) {
            #expect(type == "custom")
        }
        await registry.registerProcessorType("custom") { _, _ in
            throw ProbeError.creatorWasCalled
        }
        #expect(registry.containsProcessorType("custom"))
        #expect(registry.registrationVersion > initialVersion)
        do {
            _ = try await registry.createModel(
                configuration: Data(), processorType: "custom", tokenizer: TestTokenizer())
            Issue.record("Expected the registered creator's sentinel")
        } catch ProbeError.creatorWasCalled {
            // The live creator lookup agrees with admission's registration lookup.
        }
    }

    @Test(
        "architecture overrides and configured processor selection share one rule",
        arguments: [
            ["mistral3", "PixtralProcessor", "Mistral3Processor"],
            ["ministral3", "PixtralProcessor", "Mistral3Processor"],
            ["nemotron_h_omni", "", "NemotronHOmniProcessor"],
            ["NemotronH_Nano_Omni_Reasoning_V3", "", "NemotronHOmniProcessor"],
            ["qwen3_5", "Qwen3VLProcessor", "Qwen3VLProcessor"],
            ["qwen3_5", "NotAnInstalledProcessor", "NotAnInstalledProcessor"],
        ])
    func processorResolution(row: [String]) {
        let resolved = VLMProcessorTypeRegistry.processorType(
            modelType: row[0], declaredProcessorType: row[1])
        #expect(resolved == row[2])
        #expect(
            VLMProcessorTypeRegistry.shared.containsProcessorType(resolved)
                == (resolved != "NotAnInstalledProcessor"))
    }
}
