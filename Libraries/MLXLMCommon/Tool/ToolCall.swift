// Copyright © 2025 Apple Inc.

import Foundation

public struct ToolCall: Hashable, Codable, Sendable {
    /// Represents the function details for a tool call
    public struct Function: Hashable, Codable, Sendable {
        /// The name of the function
        public let name: String

        /// The arguments passed to the function
        public let arguments: [String: JSONValue]

        /// Exact JSON-object text observed at the protocol boundary, when
        /// available. Execution always uses ``arguments``; history encoders
        /// may reuse this only after validating that it decodes to the same
        /// values. Keeping it out of `Codable` preserves the public wire shape.
        public let rawArgumentsJSON: String?

        /// The argument names in the order the model emitted them.
        ///
        /// `arguments` is a Dictionary, so it cannot carry order, and `rawArgumentsJSON` — which
        /// can — is deliberately kept out of `Codable`. That left a re-render after a serialized
        /// history round trip falling back to SORTED order, so a tool call rendered
        /// `path, content` came back `content, path` and every turn after it lost its prefix
        /// boundary. This is the order alone: small, and unlike the raw text it is safe to carry
        /// on the wire, so the boundary survives persistence.
        public let argumentOrder: [String]?

        public init(
            name: String,
            arguments: [String: JSONValue],
            rawArgumentsJSON: String? = nil,
            argumentOrder: [String]? = nil
        ) {
            self.name = name
            self.arguments = arguments
            self.rawArgumentsJSON = rawArgumentsJSON
            self.argumentOrder =
                argumentOrder ?? rawArgumentsJSON.flatMap(Self.topLevelKeyOrder(of:))
        }

        public init(
            name: String,
            arguments: [String: any Sendable],
            rawArgumentsJSON: String? = nil,
            argumentOrder: [String]? = nil
        ) {
            self.name = name
            self.arguments = arguments.mapValues { JSONValue.from($0) }
            self.rawArgumentsJSON = rawArgumentsJSON
            self.argumentOrder =
                argumentOrder ?? rawArgumentsJSON.flatMap(Self.topLevelKeyOrder(of:))
        }

        /// Top-level key names of a JSON object, in source order.
        ///
        /// `JSONSerialization` yields a Dictionary and loses the order, so the order has to come
        /// from the text. Only the keys are scanned; values are skipped by depth and string state,
        /// which is enough to find the commas that separate top-level members.
        static func topLevelKeyOrder(of json: String) -> [String]? {
            let text = json.trimmingCharacters(in: .whitespacesAndNewlines)
            guard text.first == "{", text.last == "}" else { return nil }
            var keys: [String] = []
            var depth = 0
            var inString = false
            var escaped = false
            var current = ""
            var capturing = false
            var expectKey = true
            for ch in text.dropFirst().dropLast() {
                if escaped { if capturing { current.append(ch) }; escaped = false; continue }
                if ch == "\\" { if capturing { current.append(ch) }; escaped = true; continue }
                if inString {
                    if ch == "\"" {
                        inString = false
                        if capturing { keys.append(current); current = ""; capturing = false }
                    } else if capturing {
                        current.append(ch)
                    }
                    continue
                }
                switch ch {
                case "\"":
                    inString = true
                    if depth == 0 && expectKey { capturing = true; current = "" }
                case "{", "[": depth += 1
                case "}", "]": depth -= 1
                case ":": if depth == 0 { expectKey = false }
                case ",": if depth == 0 { expectKey = true }
                default: break
                }
            }
            return keys.isEmpty ? nil : keys
        }

        private enum CodingKeys: String, CodingKey {
            case name
            case arguments
            case argumentOrder
        }

        public init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            self.name = try container.decode(String.self, forKey: .name)
            self.arguments = try container.decode(
                [String: JSONValue].self,
                forKey: .arguments
            )
            self.rawArgumentsJSON = nil
            self.argumentOrder = try container.decodeIfPresent(
                [String].self, forKey: .argumentOrder)
        }

        public func encode(to encoder: Encoder) throws {
            var container = encoder.container(keyedBy: CodingKeys.self)
            try container.encode(name, forKey: .name)
            try container.encode(arguments, forKey: .arguments)
            try container.encodeIfPresent(argumentOrder, forKey: .argumentOrder)
        }
    }

    /// Stable id for correlating a later tool-role result with this call.
    public let id: String?

    /// The function to be called
    public let function: Function

    public init(function: Function, id: String? = nil) {
        self.id = id
        self.function = function
    }

    public init(id: String?, function: Function) {
        self.id = id
        self.function = function
    }
}

extension ToolCall {
    public func execute<Input, Output>(with tool: Tool<Input, Output>) async throws -> Output {
        // Check that the tool name matches the function name
        guard tool.name == function.name else {
            throw ToolError.nameMismatch(toolName: tool.name, functionName: function.name)
        }

        // Convert the JSONValue arguments dictionary to a JSON-encoded Data object
        let jsonObject = function.arguments.mapValues { $0.anyValue }
        let jsonData = try JSONSerialization.data(withJSONObject: jsonObject)

        // Decode the Input type from the JSON data
        let input = try JSONDecoder().decode(Input.self, from: jsonData)

        // Execute the tool's handler with the decoded input
        return try await tool.handler(input)
    }
}

// Define Tool-related errors
public enum ToolError: Error, LocalizedError {
    case nameMismatch(toolName: String, functionName: String)

    public var errorDescription: String? {
        switch self {
        case .nameMismatch(let toolName, let functionName):
            return "Tool name mismatch: expected '\(toolName)' but got '\(functionName)'"
        }
    }
}
