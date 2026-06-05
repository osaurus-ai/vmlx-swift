// Copyright © 2025 Apple Inc.

import Foundation

public typealias ToolSpec = [String: any Sendable]

public func normalizedToolsForChatTemplate(_ tools: [ToolSpec]?) -> [ToolSpec]? {
    tools?.map(normalizedToolForChatTemplate)
}

public func normalizedToolForChatTemplate(_ tool: ToolSpec) -> ToolSpec {
    normalizeChatTemplateValue(tool, inSchemaPosition: false) as? ToolSpec ?? tool
}

private func normalizeChatTemplateValue(
    _ value: any Sendable, inSchemaPosition: Bool
) -> any Sendable {
    if var object = value as? [String: any Sendable] {
        for (key, child) in object {
            if key == "properties", let properties = child as? [String: any Sendable] {
                var normalizedProperties: [String: any Sendable] = [:]
                for (propertyName, propertySchema) in properties {
                    normalizedProperties[propertyName] = normalizeChatTemplateValue(
                        propertySchema, inSchemaPosition: true)
                }
                object[key] = normalizedProperties
            } else if key == "parameters" || key == "items" || key == "additionalProperties"
                || key == "response"
            {
                if key == "additionalProperties", child is Bool {
                    object.removeValue(forKey: key)
                    continue
                }
                object[key] = normalizeChatTemplateValue(child, inSchemaPosition: true)
            } else if key == "oneOf" || key == "anyOf" || key == "allOf" {
                if let entries = child as? [any Sendable] {
                    object[key] =
                        entries.map { normalizeChatTemplateValue($0, inSchemaPosition: true) }
                        as [any Sendable]
                } else {
                    object[key] = normalizeChatTemplateValue(child, inSchemaPosition: true)
                }
            } else {
                object[key] = normalizeChatTemplateValue(
                    child, inSchemaPosition: inSchemaPosition)
            }
        }

        if inSchemaPosition {
            normalizeSchemaType(&object)
        }
        normalizeNullableEnum(&object)
        return object
    }

    if let array = value as? [any Sendable] {
        return array.map {
            normalizeChatTemplateValue($0, inSchemaPosition: inSchemaPosition)
        } as [any Sendable]
    }

    return value
}

private func normalizeSchemaType(_ object: inout [String: any Sendable]) {
    if let type = object["type"] as? String, !type.isEmpty {
        return
    }

    if let entries = stringTypeArray(object["type"]) {
        let nonNull = entries.filter { $0 != "null" }
        if entries.count != nonNull.count {
            object["nullable"] = true
        }
        if let first = nonNull.first, !first.isEmpty {
            object["type"] = first
            return
        }
    } else if object["type"] is NSNull {
        object["nullable"] = true
    }

    object["type"] = inferredSchemaType(from: object)
}

private func inferredSchemaType(from object: [String: any Sendable]) -> String {
    if object["properties"] is [String: any Sendable] {
        return "object"
    }
    if object["items"] != nil {
        return "array"
    }
    return "string"
}

private func normalizeNullableEnum(_ object: inout [String: any Sendable]) {
    guard object["nullable"] as? Bool == true,
          let entries = object["enum"] as? [any Sendable]
    else { return }

    let filtered = entries.filter { entry in
        if entry is NSNull { return false }
        if let string = entry as? String, string == "null" { return false }
        return true
    }
    object["enum"] = filtered
}

private func stringTypeArray(_ value: (any Sendable)?) -> [String]? {
    if let strings = value as? [String] {
        return strings
    }

    if let entries = value as? [any Sendable] {
        var strings: [String] = []
        for entry in entries {
            guard let string = entry as? String else { return nil }
            strings.append(string)
        }
        return strings
    }

    return nil
}

/// Protocol defining the requirements for a tool.
public protocol ToolProtocol: Sendable {
    /// The JSON Schema describing the tool's interface.
    var schema: ToolSpec { get }
}

public struct Tool<Input: Codable, Output: Codable>: ToolProtocol {
    /// The JSON Schema describing the tool's interface.
    public let schema: ToolSpec

    /// The handler for the tool.
    public let handler: @Sendable (Input) async throws -> Output

    /// The name of the tool extracted from the schema
    public var name: String {
        let function = schema["function"] as? [String: any Sendable]
        let name = function?["name"] as? String
        return name ?? ""
    }

    public init(
        name: String,
        description: String,
        parameters: [ToolParameter],
        handler: @Sendable @escaping (Input) async throws -> Output
    ) {
        var properties = [String: any Sendable]()
        var requiredParams = [String]()

        for param in parameters {
            properties[param.name] = param.schema
            if param.isRequired {
                requiredParams.append(param.name)
            }
        }

        self.schema =
            [
                "type": "function",
                "function": [
                    "name": name,
                    "description": description,
                    "parameters": [
                        "type": "object",
                        "properties": properties,
                        "required": requiredParams,
                    ] as [String: any Sendable],
                ] as [String: any Sendable],
            ] as ToolSpec

        self.handler = handler
    }

    public init(schema: ToolSpec, handler: @Sendable @escaping (Input) async throws -> Output) {
        self.schema = schema
        self.handler = handler
    }
}
