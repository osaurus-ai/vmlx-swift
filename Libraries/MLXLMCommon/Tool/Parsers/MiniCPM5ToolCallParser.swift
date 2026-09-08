import Foundation
#if canImport(FoundationXML)
import FoundationXML
#endif

/// Native MiniCPM5 XML transport. No guessed argument types, truncated calls,
/// or execution of examples from the reasoning rail. CDATA is value data,
/// including literal </function>, </param>, and <think> strings within it.
public struct MiniCPM5ToolCallParser: ToolCallParser, Sendable {
    public let startTag: String? = "<function name=\""
    public let endTag: String? = "</function>"
    public var usesCustomEndBoundary: Bool { true }
    public var preservesWhitespaceBeforeToolCalls: Bool { true }

    public init() {}

    public static func matchesTemplate(_ template: String?) -> Bool {
        guard let template else { return false }
        return template.contains("<function name=\"")
            && template.contains("<param name=\"")
            && template.contains("</function>")
            && template.contains("<![CDATA[")
    }

    /// The first real function closer, not one quoted inside a CDATA value.
    public func completeToolCallEnd(in content: String) -> String.Index? {
        var cursor = content.startIndex
        while cursor < content.endIndex {
            let close = content.range(of: "</function>", range: cursor..<content.endIndex)
            let cdata = content.range(of: "<![CDATA[", range: cursor..<content.endIndex)
            if let cdata, close == nil || cdata.lowerBound < close!.lowerBound {
                guard let end = content.range(of: "]]>", range: cdata.upperBound..<content.endIndex)
                else { return nil }
                cursor = end.upperBound
            } else {
                return close?.upperBound
            }
        }
        return nil
    }

    public func parse(content: String, tools: [[String: any Sendable]]?) -> ToolCall? {
        parseEOS(content, tools: tools).first
    }

    public func parseEOS(_ content: String, tools: [[String: any Sendable]]?) -> [ToolCall] {
        var calls: [ToolCall] = []
        var cursor = content.startIndex
        while cursor < content.endIndex {
            let open = content.range(of: "<function name=\"", range: cursor..<content.endIndex)
            let think = content.range(of: "<think>", range: cursor..<content.endIndex)
            if let think, open == nil || think.lowerBound < open!.lowerBound {
                guard let end = content.range(of: "</think>", range: think.upperBound..<content.endIndex)
                else { break }
                cursor = end.upperBound
                continue
            }
            guard let open else { break }
            let remaining = String(content[open.lowerBound...])
            guard let end = completeToolCallEnd(in: remaining) else { break }
            let xml = String(remaining[..<end])
            let delegate = FunctionXMLDelegate()
            let parser = XMLParser(data: Data(xml.utf8))
            parser.shouldResolveExternalEntities = false
            parser.delegate = delegate
            if parser.parse(), !delegate.invalid, let name = delegate.functionName,
                delegate.depth == 0
            {
                var arguments: [String: any Sendable] = [:]
                for key in delegate.order {
                    let value = delegate.values[key] ?? ""
                    arguments[key] = Self.coerce(value, function: name, parameter: key, tools: tools)
                }
                calls.append(ToolCall(function: .init(
                    name: name, arguments: arguments, argumentOrder: delegate.order)))
            }
            cursor = content.index(open.lowerBound, offsetBy: xml.count)
        }
        return calls
    }

    private static func coerce(
        _ raw: String, function: String, parameter: String, tools: [[String: any Sendable]]?
    ) -> any Sendable {
        // An absent schema means string, not JSON guessing: 007 and 3.10
        // must survive parse -> persisted history -> native template exactly.
        let type = getParameterType(funcName: function, paramName: parameter, tools: tools)
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        switch type {
        case "integer":
            if let value = Int(trimmed) { return value }
        case "number":
            if let value = Double(trimmed), value.isFinite { return value }
        case "boolean":
            if trimmed.lowercased() == "true" { return true }
            if trimmed.lowercased() == "false" { return false }
        case "null":
            if trimmed == "null" { return NSNull() }
        case "array", "object":
            if let value = try? JSONSerialization.jsonObject(with: Data(trimmed.utf8)),
                (type == "array" && value is [Any]) || (type == "object" && value is [String: Any])
            { return asSendable(value) }
            if let value = PythonicToolCallParser().parsePythonContainerLiteral(trimmed),
                (type == "array" && value is [Any]) || (type == "object" && value is [String: Any])
            { return value }
        default: break
        }
        // Malformed schema-typed values stay visible to downstream validation.
        return raw
    }
}

private final class FunctionXMLDelegate: NSObject, XMLParserDelegate {
    var functionName: String?
    var values: [String: String] = [:]
    var order: [String] = []
    var depth = 0
    var invalid = false
    private var parameter: String?
    private var value = ""

    func parser(
        _ parser: XMLParser, didStartElement elementName: String, namespaceURI: String?,
        qualifiedName qName: String?, attributes: [String: String]
    ) {
        if depth == 0, elementName == "function", functionName == nil,
            attributes.count == 1, let name = attributes["name"], !name.isEmpty
        {
            functionName = name
        } else if depth == 1, elementName == "param", attributes.count == 1,
            let name = attributes["name"], !name.isEmpty, values[name] == nil
        {
            parameter = name
            value = ""
        } else {
            invalid = true
            parser.abortParsing()
        }
        depth += 1
    }

    func parser(_ parser: XMLParser, foundCharacters string: String) {
        if parameter != nil { value += string }
        else if !string.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { invalid = true }
    }

    func parser(_ parser: XMLParser, foundCDATA CDATABlock: Data) {
        guard parameter != nil, let text = String(data: CDATABlock, encoding: .utf8) else {
            invalid = true
            parser.abortParsing()
            return
        }
        value += text
    }

    func parser(
        _ parser: XMLParser, didEndElement elementName: String, namespaceURI: String?,
        qualifiedName qName: String?
    ) {
        if depth == 2, elementName == "param", let parameter {
            values[parameter] = value
            order.append(parameter)
            self.parameter = nil
        }
        depth -= 1
    }
}
