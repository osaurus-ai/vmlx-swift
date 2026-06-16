import Foundation

public enum TensorDataPlaneAddressClass: String, Codable, Sendable, Equatable, Hashable {
    case thunderboltLoopback
    case thunderboltDirect
    case tailscaleControl
    case localLoopback
    case privateOther
    case linkLocal
    case unknown

    public var isAllowedForTensorParallelDataPlane: Bool {
        switch self {
        case .thunderboltLoopback, .thunderboltDirect:
            return true
        case .tailscaleControl, .localLoopback, .privateOther, .linkLocal, .unknown:
            return false
        }
    }

    public var isControlPlaneOnly: Bool {
        self == .tailscaleControl
    }
}

public struct TensorDataPlaneAddress: Codable, Sendable, Equatable, Hashable {
    public let rawValue: String
    public let host: String
    public let port: UInt16?
    public let addressClass: TensorDataPlaneAddressClass

    public init(_ rawValue: String) {
        let parsed = Self.parse(rawValue)
        self.rawValue = rawValue
        self.host = parsed.host
        self.port = parsed.port
        self.addressClass = Self.classify(host: parsed.host)
    }

    public var isAllowedForTensorParallelDataPlane: Bool {
        addressClass.isAllowedForTensorParallelDataPlane
    }

    private static func parse(_ rawValue: String) -> (host: String, port: UInt16?) {
        let trimmed = rawValue.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return ("", nil) }

        if let components = URLComponents(string: trimmed),
           components.scheme != nil,
           let host = components.host {
            return (host, components.port.flatMap(UInt16.init))
        }

        if trimmed.hasPrefix("["),
           let closeIndex = trimmed.firstIndex(of: "]") {
            let host = String(trimmed[trimmed.index(after: trimmed.startIndex)..<closeIndex])
            let rest = trimmed[trimmed.index(after: closeIndex)...]
            let port = rest.hasPrefix(":") ? UInt16(rest.dropFirst()) : nil
            return (host, port)
        }

        if trimmed.filter({ $0 == ":" }).count == 1,
           let colon = trimmed.lastIndex(of: ":") {
            let host = String(trimmed[..<colon])
            let port = UInt16(trimmed[trimmed.index(after: colon)...])
            if port != nil {
                return (host, port)
            }
        }

        return (trimmed, nil)
    }

    private static func classify(host: String) -> TensorDataPlaneAddressClass {
        let lower = host.trimmingCharacters(in: .whitespacesAndNewlines)
            .trimmingCharacters(in: CharacterSet(charactersIn: "[]"))
            .lowercased()

        if lower == "localhost" || lower == "::1" {
            return .localLoopback
        }

        let parts = lower.split(separator: ".")
        guard parts.count == 4,
              let a = Int(parts[0]),
              let b = Int(parts[1]),
              let c = Int(parts[2]),
              let d = Int(parts[3]),
              (0...255).contains(a),
              (0...255).contains(b),
              (0...255).contains(c),
              (0...255).contains(d)
        else {
            return .unknown
        }

        switch (a, b, c) {
        case (10, 20, 0):
            return .thunderboltLoopback
        case (10, 10, _):
            return .thunderboltDirect
        case (100, _, _):
            return .tailscaleControl
        case (127, _, _):
            return .localLoopback
        case (169, 254, _):
            return .linkLocal
        default:
            if a == 10
                || (a == 172 && (16...31).contains(b))
                || (a == 192 && b == 168) {
                return .privateOther
            }
            return .unknown
        }
    }
}

public struct TensorDataPlaneFinding: Codable, Sendable, Equatable, Hashable {
    public let address: TensorDataPlaneAddress
    public let level: String
    public let message: String

    public init(address: TensorDataPlaneAddress, level: String, message: String) {
        self.address = address
        self.level = level
        self.message = message
    }
}

public enum TensorDataPlanePolicy {
    public static func findings(
        for rawAddresses: [String],
        role: String = "tensor-parallel data plane"
    ) -> [TensorDataPlaneFinding] {
        rawAddresses.map { raw in
            let address = TensorDataPlaneAddress(raw)
            if address.addressClass == .tailscaleControl {
                return TensorDataPlaneFinding(
                    address: address,
                    level: "error",
                    message: "\(address.host) is Tailscale/control-plane only and must not be used for \(role).")
            }
            if !address.isAllowedForTensorParallelDataPlane {
                return TensorDataPlaneFinding(
                    address: address,
                    level: "warn",
                    message: "\(address.host) is \(address.addressClass.rawValue), not a proven Thunderbolt TP address.")
            }
            return TensorDataPlaneFinding(
                address: address,
                level: "info",
                message: "\(address.host) is accepted for \(role).")
        }
    }

    public static func hasDisallowedDataPlaneAddress(_ rawAddresses: [String]) -> Bool {
        findings(for: rawAddresses).contains { $0.level == "error" || $0.level == "warn" }
    }
}
