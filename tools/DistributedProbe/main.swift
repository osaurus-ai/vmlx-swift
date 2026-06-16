import Darwin
import Foundation
import MLXDistributedCore
import MLXDistributedJACCL

private struct Arguments {
    var json = false
    var peerID: UUID?
    var modes: Set<Mode> = []
    var tlsPort: UInt16?
    var tlsFingerprint: String?
    var rdmaGID: String?
    var rdmaDevices: [String] = []
    var dataPlaneAddresses: [String] = []
    var modelHashes: [String] = []

    init(_ raw: [String]) {
        var index = 1
        while index < raw.count {
            let arg = raw[index]
            switch arg {
            case "--json":
                json = true
            case "--peer-id":
                peerID = Self.value(after: &index, in: raw).flatMap(UUID.init(uuidString:))
            case "--modes":
                modes = Set(Self.value(after: &index, in: raw)
                    .map(Self.parseModes) ?? [])
            case "--tls-port":
                tlsPort = Self.value(after: &index, in: raw).flatMap(UInt16.init)
            case "--tls-fingerprint":
                tlsFingerprint = Self.value(after: &index, in: raw)
            case "--rdma-gid":
                rdmaGID = Self.value(after: &index, in: raw)
            case "--rdma-devices":
                rdmaDevices = Self.value(after: &index, in: raw)?
                    .split(separator: ",").map(String.init) ?? []
            case "--data-plane-addresses", "--tp-hosts":
                dataPlaneAddresses = Self.value(after: &index, in: raw)?
                    .split(separator: ",").map(String.init) ?? []
            case "--models":
                modelHashes = Self.value(after: &index, in: raw)?
                    .split(separator: ",").map(String.init) ?? []
            case "--help", "-h":
                print(Self.help)
                exit(0)
            default:
                break
            }
            index += 1
        }
    }

    private static func value(after index: inout Int, in raw: [String]) -> String? {
        guard index + 1 < raw.count else { return nil }
        index += 1
        return raw[index]
    }

    private static func parseModes(_ value: String) -> [Mode] {
        value.split(separator: ",").compactMap { token in
            switch token.lowercased() {
            case "replica":
                return .replica
            case "pp", "pipeline", "pipelined":
                return .pipelined
            case "tp", "wired":
                return .wired
            default:
                return nil
            }
        }
    }

    static let help = """
    DistributedProbe: local readiness probe for vMLX distributed inference.

    Usage:
      swift run DistributedProbe [--json]
      swift run DistributedProbe --modes replica,pp --tls-port 7901 --tls-fingerprint <64-hex> --models <hash[,hash]>
      swift run DistributedProbe --modes tp --tls-port 7901 --tls-fingerprint <64-hex> --rdma-gid <gid> --rdma-devices <dev[,dev]>
      swift run DistributedProbe --modes tp --data-plane-addresses 10.20.0.1:29500,10.20.0.2:29500

    Notes:
      - This does not start a server and does not prove peer reachability.
      - JACCL/RDMA availability means the local backend can be loaded, not that a TB5 peer is connected.
      - Tailscale 100.x addresses are control-plane only and are rejected for TP data-plane use.
      - TXT output is only emitted when enough endpoint data is supplied to avoid false advertising.
    """
}

private struct NetworkInterface: Codable {
    let name: String
    let address: String
    let family: String
    let flags: [String]
    let hardwarePort: String?

    var isThunderboltCandidate: Bool {
        let lowerPort = hardwarePort?.lowercased() ?? ""
        return lowerPort.contains("thunderbolt") || name == "bridge0"
    }
}

private struct Finding: Codable {
    let level: String
    let message: String
}

private struct ProbeReport: Codable {
    let generatedAt: String
    let hostname: String
    let jacclAvailable: Bool
    let librdmaLoadable: Bool
    let anyDistributedBackendAvailable: Bool
    let rdmaEnvironmentConfigured: Bool
    let mlxDistBackend: String?
    let mlxJacclCoordinator: String?
    let mlxIBVDevicesSet: Bool
    let tensorDataPlaneAddresses: [TensorDataPlaneAddress]
    let tensorDataPlaneFindings: [TensorDataPlaneFinding]
    let interfaces: [NetworkInterface]
    let thunderboltCandidates: [NetworkInterface]
    let txtPreview: [String: String]?
    let findings: [Finding]
}

@main
struct DistributedProbe {
    static func main() {
        let args = Arguments(CommandLine.arguments)
        let interfaces = InterfaceProbe.collect()
        let tbCandidates = interfaces.filter(\.isThunderboltCandidate)
        let env = ProcessInfo.processInfo.environment
        let jaccl = JACCL.isAvailable()
        let librdma = JACCL.librdmaLoadable()
        let anyBackend = JACCL.anyBackendAvailable()
        let ibvSet = !(env["MLX_IBV_DEVICES"] ?? "").isEmpty
        let rdmaConfigured = ibvSet || !args.rdmaDevices.isEmpty
        let rawDataPlaneAddresses = orderedUnique(
            args.dataPlaneAddresses
                + [env["MLX_JACCL_COORDINATOR"]].compactMap { $0 }
        )
        let dataPlaneAddresses = rawDataPlaneAddresses.map(TensorDataPlaneAddress.init)
        let dataPlaneFindings = TensorDataPlanePolicy.findings(for: rawDataPlaneAddresses)
        let txtPreview = makeTXTPreview(
            args: args,
            hostname: currentHostname(),
            jacclAvailable: jaccl,
            rdmaConfigured: rdmaConfigured)
        let findings = makeFindings(
            args: args,
            jacclAvailable: jaccl,
            rdmaConfigured: rdmaConfigured,
            dataPlaneFindings: dataPlaneFindings,
            thunderboltCandidates: tbCandidates,
            txtPreview: txtPreview
        )

        let report = ProbeReport(
            generatedAt: ISO8601DateFormatter().string(from: Date()),
            hostname: currentHostname(),
            jacclAvailable: jaccl,
            librdmaLoadable: librdma,
            anyDistributedBackendAvailable: anyBackend,
            rdmaEnvironmentConfigured: rdmaConfigured,
            mlxDistBackend: env["MLX_DIST_BACKEND"],
            mlxJacclCoordinator: env["MLX_JACCL_COORDINATOR"],
            mlxIBVDevicesSet: ibvSet,
            tensorDataPlaneAddresses: dataPlaneAddresses,
            tensorDataPlaneFindings: dataPlaneFindings,
            interfaces: interfaces,
            thunderboltCandidates: tbCandidates,
            txtPreview: txtPreview,
            findings: findings
        )

        if args.json {
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            if let data = try? encoder.encode(report),
               let string = String(data: data, encoding: .utf8) {
                print(string)
            }
        } else {
            printHuman(report)
        }
    }

    private static func makeTXTPreview(
        args: Arguments,
        hostname: String,
        jacclAvailable: Bool,
        rdmaConfigured: Bool
    ) -> [String: String]? {
        guard !args.modes.isEmpty,
              let tlsPort = args.tlsPort,
              let tlsFingerprint = args.tlsFingerprint,
              !args.modelHashes.isEmpty
        else {
            return nil
        }
        if args.modes.contains(.wired)
            && (!jacclAvailable || !rdmaConfigured || args.rdmaGID == nil || args.rdmaDevices.isEmpty) {
            return nil
        }

        var endpoints: [Endpoint] = [
            .tls(host: hostname, port: tlsPort, fingerprintSHA256: tlsFingerprint)
        ]
        if let gid = args.rdmaGID, !gid.isEmpty {
            endpoints.append(.rdma(gid: gid, devices: args.rdmaDevices))
        }

        let peer = Peer(
            id: args.peerID ?? UUID(),
            hostname: hostname,
            capabilities: PeerCapabilities(modes: args.modes),
            endpoints: endpoints,
            modelHashes: .explicit(args.modelHashes),
            memFreeMiB: nil,
            willingToBeCoordinator: args.modes.contains(.wired)
        )

        return try? TXTSchema.encode(peer)
    }

    private static func makeFindings(
        args: Arguments,
        jacclAvailable: Bool,
        rdmaConfigured: Bool,
        dataPlaneFindings: [TensorDataPlaneFinding],
        thunderboltCandidates: [NetworkInterface],
        txtPreview: [String: String]?
    ) -> [Finding] {
        var out: [Finding] = []
        if thunderboltCandidates.isEmpty {
            out.append(Finding(
                level: "warn",
                message: "No obvious Thunderbolt interface is active in the local interface list. Plug the TB5 cable in and re-run."))
        }
        if jacclAvailable {
            out.append(Finding(
                level: "info",
                message: "JACCL backend loads locally. This is a host capability signal, not peer reachability."))
        } else {
            out.append(Finding(
                level: "warn",
                message: "JACCL backend is not available locally; wired TP/RDMA must stay disabled."))
        }
        if !rdmaConfigured {
            out.append(Finding(
                level: "info",
                message: "MLX_IBV_DEVICES or --rdma-devices is not set; RDMA wiring is not configured yet."))
        }
        for finding in dataPlaneFindings {
            out.append(Finding(level: finding.level, message: finding.message))
        }
        if args.modes.contains(.wired), !jacclAvailable {
            out.append(Finding(
                level: "error",
                message: "Requested tp/wired advertisement, but JACCL is unavailable."))
        }
        if args.modes.contains(.wired), !rdmaConfigured {
            out.append(Finding(
                level: "error",
                message: "Requested tp/wired advertisement, but RDMA devices are not configured."))
        }
        if !args.modes.isEmpty && txtPreview == nil {
            out.append(Finding(
                level: "error",
                message: "TXT preview was not emitted because required endpoint/model fields are missing or unsafe for the requested modes."))
        }
        if !args.modes.isEmpty && args.modelHashes.isEmpty {
            out.append(Finding(
                level: "error",
                message: "--models is required for TXT preview; this probe does not prove the full inventory endpoint needed for dist.models=*."))
        }
        return out
    }

    private static func orderedUnique(_ values: [String]) -> [String] {
        var seen = Set<String>()
        var out: [String] = []
        for value in values {
            let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty, seen.insert(trimmed).inserted else { continue }
            out.append(trimmed)
        }
        return out
    }

    private static func printHuman(_ report: ProbeReport) {
        print("DistributedProbe")
        print("  generated: \(report.generatedAt)")
        print("  host: \(report.hostname)")
        print("  librdma loadable: \(report.librdmaLoadable)")
        print("  JACCL available: \(report.jacclAvailable)")
        print("  any distributed backend: \(report.anyDistributedBackendAvailable)")
        print("  RDMA env configured: \(report.rdmaEnvironmentConfigured)")
        print("  MLX_DIST_BACKEND: \(report.mlxDistBackend ?? "(unset)")")
        print("  MLX_JACCL_COORDINATOR: \(report.mlxJacclCoordinator ?? "(unset)")")
        print("  MLX_IBV_DEVICES set: \(report.mlxIBVDevicesSet)")
        print("")
        print("Tensor data-plane addresses:")
        if report.tensorDataPlaneAddresses.isEmpty {
            print("  none")
        } else {
            for address in report.tensorDataPlaneAddresses {
                let allowed = address.isAllowedForTensorParallelDataPlane ? "allowed" : "blocked"
                print("  \(address.rawValue) host=\(address.host) class=\(address.addressClass.rawValue) \(allowed)")
            }
        }
        print("")
        print("Thunderbolt candidates:")
        if report.thunderboltCandidates.isEmpty {
            print("  none")
        } else {
            for iface in report.thunderboltCandidates {
                print("  \(iface.name) \(iface.address) \(iface.family) port=\(iface.hardwarePort ?? "unknown") flags=\(iface.flags.joined(separator: ","))")
            }
        }
        print("")
        print("TXT preview:")
        if let txt = report.txtPreview {
            for key in txt.keys.sorted() {
                print("  \(key)=\(txt[key] ?? "")")
            }
        } else {
            print("  not emitted")
        }
        print("")
        print("Findings:")
        if report.findings.isEmpty {
            print("  none")
        } else {
            for finding in report.findings {
                print("  [\(finding.level)] \(finding.message)")
            }
        }
    }
}

private enum InterfaceProbe {
    static func collect() -> [NetworkInterface] {
        let ports = HardwarePorts.collect()
        var result: [NetworkInterface] = []
        var head: UnsafeMutablePointer<ifaddrs>?
        guard getifaddrs(&head) == 0, let first = head else {
            return []
        }
        defer { freeifaddrs(first) }

        var cursor: UnsafeMutablePointer<ifaddrs>? = first
        while let current = cursor {
            defer { cursor = current.pointee.ifa_next }
            guard let addr = current.pointee.ifa_addr else { continue }
            let family = Int32(addr.pointee.sa_family)
            guard family == AF_INET || family == AF_INET6 else { continue }

            var host = [CChar](repeating: 0, count: Int(NI_MAXHOST))
            let length = socklen_t(addr.pointee.sa_len)
            let rc = getnameinfo(
                addr,
                length,
                &host,
                socklen_t(host.count),
                nil,
                0,
                NI_NUMERICHOST
            )
            guard rc == 0 else { continue }

            let name = String(cString: current.pointee.ifa_name)
            let flags = decodeFlags(current.pointee.ifa_flags)
            guard flags.contains("up") else { continue }

            result.append(NetworkInterface(
                name: name,
                address: String(cString: host.withUnsafeBufferPointer { $0.baseAddress! }),
                family: family == AF_INET ? "ipv4" : "ipv6",
                flags: flags,
                hardwarePort: ports[name]
            ))
        }

        return result.sorted {
            if $0.name == $1.name { return $0.address < $1.address }
            return $0.name < $1.name
        }
    }

    private static func decodeFlags(_ raw: UInt32) -> [String] {
        var flags: [String] = []
        if raw & UInt32(IFF_UP) != 0 { flags.append("up") }
        if raw & UInt32(IFF_RUNNING) != 0 { flags.append("running") }
        if raw & UInt32(IFF_LOOPBACK) != 0 { flags.append("loopback") }
        if raw & UInt32(IFF_BROADCAST) != 0 { flags.append("broadcast") }
        if raw & UInt32(IFF_POINTOPOINT) != 0 { flags.append("point-to-point") }
        return flags
    }
}

private enum HardwarePorts {
    static func collect() -> [String: String] {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/sbin/networksetup")
        process.arguments = ["-listallhardwareports"]

        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = Pipe()
        do {
            try process.run()
            process.waitUntilExit()
        } catch {
            return [:]
        }
        guard process.terminationStatus == 0 else { return [:] }

        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        guard let output = String(data: data, encoding: .utf8) else { return [:] }

        var currentPort: String?
        var map: [String: String] = [:]
        for line in output.split(separator: "\n").map(String.init) {
            if line.hasPrefix("Hardware Port: ") {
                currentPort = String(line.dropFirst("Hardware Port: ".count))
            } else if line.hasPrefix("Device: "), let port = currentPort {
                let device = String(line.dropFirst("Device: ".count))
                map[device] = port
                currentPort = nil
            }
        }
        return map
    }
}

private func currentHostname() -> String {
    var buffer = [CChar](repeating: 0, count: Int(MAXHOSTNAMELEN))
    if gethostname(&buffer, buffer.count) == 0 {
        return String(cString: buffer.withUnsafeBufferPointer { $0.baseAddress! })
    }
    return "localhost"
}
