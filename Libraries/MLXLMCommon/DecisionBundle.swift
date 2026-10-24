import Foundation

public enum DecisionRuntimeError: LocalizedError {
    case invalid(String)
    public var errorDescription: String? {
        if case .invalid(let message) = self { return message }
        return nil
    }
}

/// Admission uses the bundle's explicit contract, never its directory/repository name.
public struct DecisionBundle: Sendable {
    public enum AttentionMode: String, Decodable, Sendable {
        case noncausalFullAttention = "noncausal_full_attention"
        case prefixBidirectional = "prefix_bidir"
    }
    public struct Configuration: Decodable, Sendable {
        public let formatID: String
        public let formatVersion: Int
        public let prompt: String
        public let pooling: String
        public let attentionMode: AttentionMode
        public let codes: [String]
        public let tokenIDs: [Int]
        public let temperature: Float
        public let maxLength: Int?
        public let validatedLength: Int?
        public let systemone: JSONValue?
        enum CodingKeys: String, CodingKey {
            case formatID = "format_id", formatVersion = "format_version", attentionMode = "attention_mode"
            case tokenIDs = "token_ids", maxLength = "max_length", validatedLength = "validated_length"
            case prompt, pooling, codes, temperature, systemone
        }
    }
    public let configuration: Configuration
    public let readoutSHA256: String
    public let readoutShape: [Int]
    public let readoutURL: URL

    public static func isDecisionModel(at directory: URL) -> Bool {
        FileManager.default.fileExists(atPath: directory.appendingPathComponent("decision/decision_config.json").path)
            && declaresDecisionModel(at: directory)
    }

    public static func declaresDecisionModel(at directory: URL) -> Bool {
        guard let data = try? Data(contentsOf: directory.appendingPathComponent("jang_config.json")),
            let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let decision = root["decision"] as? [String: Any]
        else { return false }
        return decision["generates_text"] as? Bool == false
    }

    public init(directory: URL) throws {
        guard Self.isDecisionModel(at: directory) else {
            throw DecisionRuntimeError.invalid("Bundle does not declare a decision runtime")
        }
        configuration = try JSONDecoder().decode(Configuration.self, from: Data(contentsOf:
            directory.appendingPathComponent("decision/decision_config.json")))
        guard configuration.formatID == "d3-code-readout-v1", configuration.formatVersion == 1,
            configuration.prompt == "d3", configuration.pooling == "last",
            configuration.codes.count == 255, Set(configuration.codes).count == 255,
            configuration.tokenIDs.count == 255, Set(configuration.tokenIDs).count == 255,
            configuration.temperature.isFinite, configuration.temperature > 0
        else { throw DecisionRuntimeError.invalid("Unsupported or inconsistent D3 decision configuration") }
        let root = try JSONSerialization.jsonObject(with: Data(contentsOf:
            directory.appendingPathComponent("jang_config.json"))) as? [String: Any]
        guard let decision = root?["decision"] as? [String: Any],
            let readout = decision["readout"] as? [String: Any],
            readout["path"] as? String == "decision/readout.safetensors",
            readout["dtype"] as? String == "float32",
            let shape = readout["shape"] as? [Int], shape.count == 2, shape[0] == 255, shape[1] > 0,
            let hash = readout["sha256"] as? String, hash.count == 64,
            hash.allSatisfy({ $0.isHexDigit }),
            decision["attention_mode"] as? String == configuration.attentionMode.rawValue
        else { throw DecisionRuntimeError.invalid("Invalid D3 readout/attention metadata") }
        readoutShape = shape
        readoutSHA256 = hash.lowercased()
        readoutURL = directory.appendingPathComponent("decision/readout.safetensors")
    }
}
