import Foundation
import MLX  // brings Cmlx into the link, even though we don't use MLX symbols directly here.
#if canImport(Darwin)
import Darwin
#endif

/// Capability probes for MLX's JACCL distributed backend.
/// The backend is compiled when the macOS SDK provides the verbs headers;
/// other platforms use the upstream unavailable stub. Group lifecycle and
/// collectives are exposed separately by MLXDistributedTP. A successful
/// capability probe does not establish a working RDMA device or peer.
public enum JACCL {

    /// Returns true when the compiled JACCL backend can load librdma and
    /// resolve its required verbs symbols. This does not enumerate devices,
    /// query ports, initialize a group, or prove a peer/collective is ready.
    public static func isAvailable() -> Bool {
        // CmlxDistributedShim exposes `bool vmlx_distributed_is_available(const char* bk)`.
        // Pass "jaccl" to get the JACCL-specific gate (rather than "any").
        return "jaccl".withCString { ptr in
            _vmlx_distributed_is_available(ptr)
        }
    }

    /// Returns true when Apple's RDMA dynamic library can be loaded.
    ///
    /// This is intentionally weaker than `isAvailable()`: it proves the local
    /// host has the RDMA library, not that JACCL can initialize a backend or
    /// reach peers.
    public static func librdmaLoadable() -> Bool {
        #if canImport(Darwin)
        guard let handle = dlopen("librdma.dylib", RTLD_NOW | RTLD_GLOBAL) else {
            return false
        }
        dlclose(handle)
        return true
        #else
        return false
        #endif
    }

    /// Same as `isAvailable()` but probes the global "any backend"
    /// gate, which returns true if any of jaccl/ring/mpi/nccl is
    /// currently available. Useful for detecting whether we're in a
    /// fully-distributed-capable build at all.
    public static func anyBackendAvailable() -> Bool {
        return _vmlx_distributed_is_available(nil)
    }
}

// MARK: - C-symbol forward declarations
//
// The C shim isolates Swift from the upstream C++ distributed group type.

@_silgen_name("vmlx_distributed_is_available")
private func _vmlx_distributed_is_available(_ backend: UnsafePointer<CChar>?) -> Bool
