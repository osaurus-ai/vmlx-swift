import Darwin
import Foundation

// Compile alongside RunBench/ProcessMemoryObservation.swift. No model or GPU.
@main
struct MemoryProbeRegression {
    static func main() throws {
        let before = try require(ProcessMemoryObservation.read())
        let bytes = 64 * 1024 * 1024
        let address = mmap(nil, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0)
        guard address != MAP_FAILED, let address else { fatalError("mmap failed") }
        memset(address, 0xa5, bytes)
        let during = try require(ProcessMemoryObservation.read())
        precondition(munmap(address, bytes) == 0)
        let after = try require(ProcessMemoryObservation.read())
        precondition(during.footprintBytes > before.footprintBytes + 48 * 1024 * 1024)
        precondition(after.footprintBytes + 48 * 1024 * 1024 < during.footprintBytes)
        // The old after-turn-only maximum misses this released allocation.
        precondition(after.lifetimePeakFootprintBytes >= during.footprintBytes)
        precondition(after.lifetimePeakFootprintBytes >= before.lifetimePeakFootprintBytes)
        precondition(after.hostWiredBytes != nil && after.hostCompressedBytes != nil)
        print("PASS transient_64mib before=\(before.footprintBytes) during=\(during.footprintBytes) after=\(after.footprintBytes) lifetime_peak=\(after.lifetimePeakFootprintBytes)")
        ProcessMemoryObservation.log(stage: "after_released_transient")
    }

    static func require<T>(_ value: T?) throws -> T {
        guard let value else { throw NSError(domain: "MemoryProbe", code: 1) }
        return value
    }
}
