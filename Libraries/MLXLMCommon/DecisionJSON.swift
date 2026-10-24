import Foundation

/// Order-preserving JSON for decision prompts. Option order assigns answer codes, and nested
/// state/instruction objects must render like the vendor's json.dumps(ensure_ascii=False).
public indirect enum DecisionJSON: Sendable {
    case null
    case bool(Bool)
    case number(String)
    case string(String)
    case array([DecisionJSON])
    case object([(String, DecisionJSON)])

    public subscript(_ key: String) -> DecisionJSON? {
        guard case .object(let entries) = self else { return nil }
        return entries.first(where: { $0.0 == key })?.1
    }
    public var string: String? {
        if case .string(let s) = self { return s }
        return nil
    }
    public var entries: [(String, DecisionJSON)]? {
        if case .object(let v) = self { return v }
        return nil
    }
    public var array: [DecisionJSON]? {
        if case .array(let v) = self { return v }
        return nil
    }
    public var isNull: Bool {
        if case .null = self { return true }
        return false
    }
    public var isTruthy: Bool {
        switch self {
        case .null: return false
        case .bool(let b): return b
        case .number(let n): return Double(n) != 0
        case .string(let s): return !s.isEmpty
        case .array(let a): return !a.isEmpty
        case .object(let o): return !o.isEmpty
        }
    }
    public var canonicalDescription: String {
        if let string { return string }
        let object = try! JSONSerialization.jsonObject(
            with: Data(json.utf8), options: [.fragmentsAllowed])
        let bytes = try! JSONSerialization.data(
            withJSONObject: object,
            options: [.fragmentsAllowed, .sortedKeys, .withoutEscapingSlashes])
        return String(decoding: bytes, as: UTF8.self)
    }
    public var describe: String { string ?? json }
    public var json: String {
        switch self {
        case .null: return "null"
        case .bool(let b): return b ? "true" : "false"
        case .number(let n): return n
        case .string(let s):
            let data = try! JSONSerialization.data(
                withJSONObject: s, options: [.fragmentsAllowed, .withoutEscapingSlashes])
            return String(decoding: data, as: UTF8.self)
        case .array(let values): return "[" + values.map(\.json).joined(separator: ", ") + "]"
        case .object(let entries):
            return "{"
                + entries.map { DecisionJSON.string($0.0).json + ": " + $0.1.json }.joined(
                    separator: ", ") + "}"
        }
    }

    public static func parse(_ data: Data) throws -> Self {
        // Foundation validates UTF-8, escapes, numeric syntax and the complete document first.
        _ = try JSONSerialization.jsonObject(with: data, options: [.fragmentsAllowed])
        var parser = Parser(bytes: Array(data))
        return try parser.value()
    }

    private struct Parser {
        let bytes: [UInt8]
        var at = 0
        mutating func space() {
            while at < bytes.count && [9, 10, 13, 32].contains(bytes[at]) { at += 1 }
        }
        mutating func text() throws -> String {
            let start = at
            at += 1
            while at < bytes.count {
                if bytes[at] == 92 {
                    at += 2
                    continue
                }
                if bytes[at] == 34 {
                    at += 1
                    return try JSONSerialization.jsonObject(
                        with: Data(bytes[start ..< at]), options: [.fragmentsAllowed]) as! String
                }
                at += 1
            }
            throw DecisionRuntimeError.invalid("Unterminated JSON string")
        }
        mutating func value() throws -> DecisionJSON {
            space()
            guard at < bytes.count else { throw DecisionRuntimeError.invalid("Missing JSON value") }
            switch bytes[at] {
            case 34: return .string(try text())
            case 123:
                at += 1
                space()
                var entries: [(String, DecisionJSON)] = []
                while bytes[at] != 125 {
                    let key = try text()
                    space()
                    at += 1
                    let item = try value()
                    space()
                    guard !entries.contains(where: { $0.0 == key }) else {
                        throw DecisionRuntimeError.invalid("Duplicate JSON key: \(key)")
                    }
                    entries.append((key, item))
                    if bytes[at] != 44 { break }
                    at += 1
                    space()
                }
                at += 1
                return .object(entries)
            case 91:
                at += 1
                space()
                var items: [DecisionJSON] = []
                while bytes[at] != 93 {
                    items.append(try value())
                    space()
                    if bytes[at] != 44 { break }
                    at += 1
                    space()
                }
                at += 1
                return .array(items)
            default:
                let start = at
                while at < bytes.count && ![9, 10, 13, 32, 44, 93, 125].contains(bytes[at]) {
                    at += 1
                }
                let scalar = String(decoding: bytes[start ..< at], as: UTF8.self)
                switch scalar {
                case "null": return .null
                case "true": return .bool(true)
                case "false": return .bool(false)
                default: return .number(scalar)
                }
            }
        }
    }
}
