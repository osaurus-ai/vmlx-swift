import Foundation

public typealias ToolSpec = [String: any Sendable]

/// INVARIANT: configured chat templates receive the original JSON Schema,
/// including unions, null values, boolean schemas and additionalProperties.
/// Jinja filters own reference-compatible value coercion; the host must not
/// discard constraints or invent types to work around an interpreter error.
/// Historical public names are retained for source compatibility.
public func normalizedToolsForChatTemplate(_ tools: [ToolSpec]?) -> [ToolSpec]? {
    tools
}

public func normalizedToolForChatTemplate(_ tool: ToolSpec) -> ToolSpec {
    tool
}
