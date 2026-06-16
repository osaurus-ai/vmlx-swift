import MLXDistributedCore
import Testing

@Test func thunderboltLoopbackAddressesAreAcceptedForTPDataPlane() {
    let address = TensorDataPlaneAddress("10.20.0.1:29500")

    #expect(address.host == "10.20.0.1")
    #expect(address.port == 29500)
    #expect(address.addressClass == .thunderboltLoopback)
    #expect(address.isAllowedForTensorParallelDataPlane)
}

@Test func thunderboltDirectAddressesAreAcceptedForTPDataPlane() {
    let address = TensorDataPlaneAddress("10.10.6.2")

    #expect(address.addressClass == .thunderboltDirect)
    #expect(address.isAllowedForTensorParallelDataPlane)
}

@Test func tailscaleAddressesAreRejectedForTPDataPlane() {
    let address = TensorDataPlaneAddress("100.93.216.67:29500")
    let findings = TensorDataPlanePolicy.findings(for: ["100.93.216.67:29500"])

    #expect(address.addressClass == .tailscaleControl)
    #expect(!address.isAllowedForTensorParallelDataPlane)
    #expect(findings.count == 1)
    #expect(findings[0].level == "error")
    #expect(findings[0].message.contains("Tailscale/control-plane only"))
}

@Test func privateNonThunderboltAddressesAreNotAcceptedAsProof() {
    let addresses = [
        TensorDataPlaneAddress("10.0.0.4"),
        TensorDataPlaneAddress("192.168.1.20"),
        TensorDataPlaneAddress("172.16.4.8"),
    ]

    #expect(addresses.allSatisfy { $0.addressClass == .privateOther })
    #expect(addresses.allSatisfy { !$0.isAllowedForTensorParallelDataPlane })
}

@Test func localhostAndIPv6LoopbackAreNotMultiRankDataPlaneProof() {
    let ipv4 = TensorDataPlaneAddress("127.0.0.1:29500")
    let ipv6 = TensorDataPlaneAddress("[::1]:29500")

    #expect(ipv4.addressClass == .localLoopback)
    #expect(ipv6.addressClass == .localLoopback)
    #expect(!ipv4.isAllowedForTensorParallelDataPlane)
    #expect(!ipv6.isAllowedForTensorParallelDataPlane)
}

@Test func dataPlanePolicyFlagsAnyNonThunderboltAddress() {
    #expect(!TensorDataPlanePolicy.hasDisallowedDataPlaneAddress([
        "10.20.0.1:29500",
        "10.20.0.2:29500",
        "10.10.1.2",
    ]))

    #expect(TensorDataPlanePolicy.hasDisallowedDataPlaneAddress([
        "10.20.0.1:29500",
        "100.76.234.41:29500",
    ]))
}
