# MLXDistributedJACCL

Local capability probes for MLX's native JACCL backend. Group lifecycle and
collectives live in `MLXDistributedTP`; the C ABI bridge is
`CmlxDistributedShim`.

## Build selection

`Package.swift` excludes both upstream JACCL entry points from automatic
source discovery. Wrappers in `Source/Cmlx/distributed/` select the real backend
and its five library sources only on macOS when the SDK provides
`infiniband/verbs.h`. Older SDKs, iOS and non-Apple builds select the upstream
unavailable stub. No vendored checkout edits or external package installation
are needed to compile these bindings.

The real backend dynamically loads the operating system's `librdma.dylib`.
Building it does not enable RDMA, grant permissions, configure interfaces or
establish connections. MPI, NCCL and the separate TCP ring backend are not
enabled by this change.

## Readiness contract

- `librdmaLoadable()` proves only that the system library can be loaded.
- `isAvailable()` proves that the compiled JACCL backend loads the library and
  resolves its required verbs symbols. It does not enumerate devices or ports.
- `anyBackendAvailable()` is a library capability check, not a connected-peer
  or tensor-parallel inference check.

Callers must separately check OS configuration, RDMA devices, physical links,
peer identity and compatibility, rank/world configuration and actual
collectives. A Thunderbolt Bridge IP or successful Bonjour discovery alone
cannot establish a TB5/RDMA data path. Never present a size-one fallback group
as successful multi-host inference.

## Local inspection

```sh
swift run -c release DistributedProbe --help
rdma_ctl status
ibv_devices
```

The probe reports interfaces and local backend availability. Its output is
not evidence that model sharding or multi-host inference works. Avoid printing
interface diagnostics publicly without reviewing their contents.

On the development host with Xcode 27, enabling the compiled backend changed
its availability probe from false to true while `rdma_ctl status` still
reported disabled and `ibv_devices` listed no devices. This is the expected
separation between a compiled backend and an operational RDMA session.
