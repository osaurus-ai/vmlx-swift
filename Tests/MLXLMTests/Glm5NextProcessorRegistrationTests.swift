import Foundation
import MLXLMCommon
@testable import MLXVLM
import Testing

/// Regression for the GLM-5.3 (`glm5_next`) load/retry loop: the model was
/// registered in the VLM TYPE registry but its processor was NOT registered in
/// `VLMProcessorTypeRegistry`, so the family loaded its ~95 GB of weights and
/// then failed every generation with "Unsupported processor type:
/// Glm5NextProcessor" — which surfaced to users as an endless load/evict loop
/// (RAM sawtooth), no crash, no logged error. This asserts the processor is
/// registered so that never regresses.
@Suite("GLM-5.3 processor registration", .serialized)
struct Glm5NextProcessorRegistrationTests {
    /// Minimal GLM-5.3 tokenizer stand-in: the processor's initialiser only
    /// requires `<|image|>` to resolve; the image/video delimiters are optional.
    struct ImageTokenTokenizer: MLXLMCommon.Tokenizer {
        var bosToken: String? = nil
        var eosToken: String? = nil
        var eosTokenId: Int? = 154_820
        var unknownToken: String? = nil
        var unknownTokenId: Int? = nil

        func encode(text: String, addSpecialTokens: Bool = true) -> [Int] {
            text.utf8.map { Int($0) }
        }

        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
            tokenIds.map(String.init).joined(separator: " ")
        }

        func convertTokenToId(_ token: String) -> Int? {
            switch token {
            case "<|image|>": return 154_854
            case "<|begin_of_image|>": return 154_830
            case "<|end_of_image|>": return 154_831
            case "<|video|>": return 154_855
            default: return nil
            }
        }

        func convertIdToToken(_ id: Int) -> String? { String(id) }

        func applyChatTemplate(
            messages: [[String: any Sendable]],
            tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?
        ) throws -> [Int] {
            encode(text: "user:Describe.")
        }
    }

    static func processorConfigurationData() -> Data {
        """
        {
          "image_processor": {
            "image_mean": [0.48145466, 0.4578275, 0.40821073],
            "image_std": [0.26862954, 0.26130258, 0.27577711],
            "patch_size": 14,
            "merge_size": 2,
            "temporal_patch_size": 2,
            "min_image_tokens": 4,
            "max_image_tokens": 16384
          },
          "video_processor": {
            "image_mean": [0.48145466, 0.4578275, 0.40821073],
            "image_std": [0.26862954, 0.26130258, 0.27577711],
            "patch_size": 14,
            "merge_size": 2,
            "temporal_patch_size": 2,
            "min_image_tokens": 4,
            "max_image_tokens": 16384,
            "fps": 2
          }
        }
        """.data(using: .utf8)!
    }

    @Test("VLM processor registry recognizes Glm5NextProcessor (no 'Unsupported processor type' loop)")
    func registryRecognizesGlm5NextProcessor() async throws {
        let processor = try await VLMProcessorTypeRegistry.shared.createModel(
            configuration: Self.processorConfigurationData(),
            processorType: "Glm5NextProcessor",
            tokenizer: ImageTokenTokenizer())

        #expect(processor is Glm5NextProcessor)
    }

    @Test("The glm5_next model type is served by the VLM factory")
    func vlmFactoryKnowsGlm5Next() {
        #expect(VLMTypeRegistry.supportedModelTypes.contains("glm5_next"))
    }
}
