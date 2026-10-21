import Foundation
#if canImport(Darwin)
import Darwin
#endif

/// Process-ledger counters are not the model's total host-memory cost. In
/// particular, Metal-resident file mappings can raise host wired memory without
/// being charged to this process's physical-footprint ledger. Never add these
/// overlapping counters together or describe a host delta as process ownership.
struct ProcessMemoryObservation {
    let footprintBytes: UInt64
    let lifetimePeakFootprintBytes: UInt64
    let residentBytes: UInt64
    let hostWiredBytes: UInt64?
    let hostCompressedBytes: UInt64?

    static func read() -> Self? {
        #if canImport(Darwin)
        var usage = rusage_info_v4()
        let result = withUnsafeMutablePointer(to: &usage) { pointer in
            pointer.withMemoryRebound(to: rusage_info_t?.self, capacity: 1) {
                proc_pid_rusage(getpid(), RUSAGE_INFO_V4, $0)
            }
        }
        guard result == 0 else { return nil }
        let host = mach_host_self()
        defer { mach_port_deallocate(mach_task_self_, host) }
        var vm = vm_statistics64()
        var count = mach_msg_type_number_t(MemoryLayout<vm_statistics64>.size / MemoryLayout<integer_t>.size)
        let status = withUnsafeMutablePointer(to: &vm) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                host_statistics64(host, HOST_VM_INFO64, $0, &count)
            }
        }
        var page: vm_size_t = 0
        let hostValid = status == KERN_SUCCESS && host_page_size(host, &page) == KERN_SUCCESS
        return Self(
            footprintBytes: usage.ri_phys_footprint,
            lifetimePeakFootprintBytes: usage.ri_lifetime_max_phys_footprint,
            residentBytes: usage.ri_resident_size,
            hostWiredBytes: hostValid ? UInt64(vm.wire_count) * UInt64(page) : nil,
            hostCompressedBytes: hostValid ? UInt64(vm.compressor_page_count) * UInt64(page) : nil)
        #else
        return nil
        #endif
    }

    static func log(stage: String) {
        guard let value = read() else {
            print("PERF_OS_MEMORY stage=\(stage) status=unavailable")
            return
        }
        print("PERF_OS_MEMORY stage=\(stage) footprint_bytes=\(value.footprintBytes) lifetime_peak_footprint_bytes=\(value.lifetimePeakFootprintBytes) rss_bytes=\(value.residentBytes) host_wired_bytes=\(value.hostWiredBytes.map(String.init) ?? "unknown") host_compressed_bytes=\(value.hostCompressedBytes.map(String.init) ?? "unknown") peak_scope=process_lifetime host_scope=whole_machine model_total=not_measured")
    }
}
