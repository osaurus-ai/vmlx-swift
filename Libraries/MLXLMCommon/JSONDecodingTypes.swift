// Copyright © 2024 Apple Inc.

import Foundation

// MARK: - IntOrIntArray

/// Decodes a JSON value that can be either a single Int or an array of Ints.
/// Used for fields like `eos_token_id` which may appear as `128001` or `[128001, 128008]`.
public struct IntOrIntArray: Codable, Sendable, Equatable {
    public let values: [Int]

    public init(_ values: [Int]) {
        self.values = values
    }

    public init(_ value: Int) {
        self.values = [value]
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let array = try? container.decode([Int].self) {
            self.values = array
        } else if let single = try? container.decode(Int.self) {
            self.values = [single]
        } else {
            throw DecodingError.typeMismatch(
                IntOrIntArray.self,
                DecodingError.Context(
                    codingPath: decoder.codingPath,
                    debugDescription: "Expected Int or [Int]"
                )
            )
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        if values.count == 1 {
            try container.encode(values[0])
        } else {
            try container.encode(values)
        }
    }
}

// MARK: - StringOrNumber

/// Representation of a heterogenous type in a JSON configuration file.
///
/// This can be: a string, a numeric value or an array of numeric values.
/// There are methods to do unwrapping, see e.g. ``asFloat()`` and
/// ``asFloats()`` or callers can switch on the enum.
public enum StringOrNumber: Codable, Equatable, Sendable {
    case string(String)
    case int(Int)
    case float(Float)
    case ints([Int])
    case floats([Float])
    case bool(Bool)

    public init(from decoder: Decoder) throws {
        let values = try decoder.singleValueContainer()

        if let v = try? values.decode(Int.self) {
            self = .int(v)
        } else if let v = try? values.decode(Float.self) {
            self = .float(v)
        } else if let v = try? values.decode([Int].self) {
            self = .ints(v)
        } else if let v = try? values.decode([Float].self) {
            self = .floats(v)
        } else if let v = try? values.decode(Bool.self) {
            self = .bool(v)
        } else {
            let v = try values.decode(String.self)
            self = .string(v)
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .string(let v): try container.encode(v)
        case .int(let v): try container.encode(v)
        case .float(let v): try container.encode(v)
        case .ints(let v): try container.encode(v)
        case .floats(let v): try container.encode(v)
        case .bool(let v): try container.encode(v)
        }
    }

    /// Return the value as an optional array of integers.
    ///
    /// This will not coerce `Float` or `String` to `Int`.
    public func asInts() -> [Int]? {
        switch self {
        case .string(_): nil
        case .int(let v): [v]
        case .float(_): nil
        case .ints(let array): array
        case .floats(_): nil
        case .bool(_): nil
        }
    }

    /// Return the value as an optional integer.
    ///
    /// This will not coerce `Float` or `String` to `Int`.
    public func asInt() -> Int? {
        switch self {
        case .string(_): nil
        case .int(let v): v
        case .float(_): nil
        case .ints(let array): array.count == 1 ? array[0] : nil
        case .floats(_): nil
        case .bool(let bool): bool ? 1 : 0
        }
    }

    /// Return the value as an optional array of floats.
    ///
    /// This will not coerce `Int` or `String` to `Float`.
    public func asFloats() -> [Float]? {
        switch self {
        case .string(_): nil
        case .int(let v): [Float(v)]
        case .float(let float): [float]
        case .ints(let array): array.map { Float($0) }
        case .floats(let array): array
        case .bool(let bool): [bool ? 1.0 : 0.0]
        }
    }

    /// Return the value as an optional float.
    ///
    /// This will not coerce `Int` or `String` to `Float`.
    public func asFloat() -> Float? {
        switch self {
        case .string(_): nil
        case .int(let v): Float(v)
        case .float(let float): float
        case .ints(let array): array.count == 1 ? Float(array[0]) : nil
        case .floats(let array): array.count == 1 ? array[0] : nil
        case .bool(let bool): bool ? 1.0 : 0.0
        }
    }
}

/// A patch dimension that bundles spell either as a bare number or as a `{"height": h, "width": w}`
/// pair. Both forms are real and appear within the same model family: Devstral-Small-2-24B ships
/// `"patch_size": 16`, Magistral-Small-2509 ships `"patch_size": {"height": 14, "width": 14}`.
/// Before this existed the dict form made the processor configuration undecodable, so the model
/// could not load at all.
///
/// A NON-SQUARE pair throws rather than picking an axis. Every caller uses the value for both
/// dimensions — `((width + patch - 1) / patch) * patch` and the same for height — so silently
/// keeping one number would compute a wrong patch grid rather than an approximate one.
public struct IntOrSquareSize: Codable, Sendable, Equatable {
    public let value: Int

    public init(_ value: Int) {
        self.value = value
    }

    private enum SizeKeys: String, CodingKey {
        case height
        case width
    }

    public init(from decoder: Decoder) throws {
        if let single = try? decoder.singleValueContainer().decode(Int.self) {
            self.value = single
            return
        }
        let keyed = try decoder.container(keyedBy: SizeKeys.self)
        let height = try keyed.decodeIfPresent(Int.self, forKey: .height)
        let width = try keyed.decodeIfPresent(Int.self, forKey: .width)
        switch (height, width) {
        case let (h?, w?) where h == w:
            self.value = h
        case let (h?, nil):
            self.value = h
        case let (nil, w?):
            self.value = w
        case let (h?, w?):
            throw DecodingError.dataCorrupted(
                DecodingError.Context(
                    codingPath: decoder.codingPath,
                    debugDescription:
                        "non-square patch size \(h)x\(w); the patch grid math assumes one "
                        + "dimension for both axes, so this cannot be honoured"))
        default:
            throw DecodingError.typeMismatch(
                IntOrSquareSize.self,
                DecodingError.Context(
                    codingPath: decoder.codingPath,
                    debugDescription: "expected a number or a {height, width} object"))
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(value)
    }
}
