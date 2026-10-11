// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXNN

#if canImport(Darwin)
import Darwin

@_silgen_name("proc_regionfilename")
private func vmlxProcRegionFilename(
    _ pid: Int32, _ address: UInt64, _ buffer: UnsafeMutableRawPointer?, _ bufferSize: UInt32
) -> Int32
#endif

/// Where a weight's bytes live, and therefore whether a runtime re-layout of it costs memory.
///
/// The shipped memory-safety loader (`LoadConfiguration.useMmapSafetensors`, `MLX_SAFETENSORS_MMAP=1`)
/// maps each safetensors shard once and makes every tensor a window into that one mapping. Releasing
/// a single tensor frees nothing while any other tensor from its shard is alive, so a runtime
/// transform that builds a re-laid-out copy of mapped weights (fused decode projections, lane-tiled
/// weights) ADDS the copy to resident memory for the life of the model — memory the host's RAM
/// admission never saw, because it admitted the model by its bundle size. Measured 2026-10-11 on
/// Qwen3.8-27B JANG_4D: +4.1 GB (GatedDeltaNet fused input projections), +12.9 GB (lane tiling).
///
/// A weight that was materialized as its own allocation (a non-mmap load, or a tensor the model's
/// `sanitize` rebuilt) is different: a copy that REPLACES it frees the original, so the transform
/// costs nothing at rest (Flash-Next's fused GDN projection: +0.6 GB with the rows shared, +5 % decode).
///
/// Rule for every runtime weight transform: build the copy only when it replaces the original.
public enum WeightBacking {
    /// True when `array`'s bytes lie inside a file mapping MUCH larger than the array itself — a
    /// whole safetensors shard shared with other tensors — so releasing `array` cannot free them.
    /// A tensor with its own allocation (anonymous memory) or its own small mapping returns false.
    /// Evaluates `array`. False when it cannot tell (non-Darwin, empty array, region lookup failed).
    public static func sharesMappedShard(_ array: MLXArray) -> Bool {
        #if canImport(Darwin)
        guard array.size > 0 else { return false }
        let backing = array.asData(access: .noCopy)
        return backing.data.withUnsafeBytes { raw -> Bool in
            guard let base = raw.baseAddress else { return false }
            let pointer = UInt64(UInt(bitPattern: base))
            var path = [UInt8](repeating: 0, count: Int(MAXPATHLEN))
            let length = path.withUnsafeMutableBytes { buffer in
                vmlxProcRegionFilename(getpid(), pointer, buffer.baseAddress, UInt32(buffer.count))
            }
            guard length > 0 else { return false }  // anonymous: the array owns its allocation
            var address = mach_vm_address_t(pointer)
            var size: mach_vm_size_t = 0
            var info = vm_region_basic_info_data_64_t()
            var count = mach_msg_type_number_t(
                MemoryLayout<vm_region_basic_info_data_64_t>.size / MemoryLayout<natural_t>.size)
            var object: mach_port_t = 0
            let kr = withUnsafeMutablePointer(to: &info) {
                $0.withMemoryRebound(to: Int32.self, capacity: Int(count)) {
                    mach_vm_region(
                        mach_task_self_, &address, &size, VM_REGION_BASIC_INFO_64, $0, &count, &object)
                }
            }
            guard kr == KERN_SUCCESS, address <= pointer else { return false }
            return Int(size) > 2 * array.nbytes + Int(getpagesize())
        }
        #else
        return false
        #endif
    }

    /// Whether a runtime copy of `arrays` may be built: not when any of them shares a mapped shard (the
    /// copy could not replace it). `VMLX_COPY_MAPPED_WEIGHTS=1` restores the old always-copy behavior (A/B only).
    public static func copyWouldReplace(_ arrays: [MLXArray]) -> Bool {
        if ProcessInfo.processInfo.environment["VMLX_COPY_MAPPED_WEIGHTS"] == "1" { return true }
        return !arrays.contains(where: sharesMappedShard)
    }

    /// Make a fused copy the ONLY copy: point each member projection at its block of output rows in
    /// the fused tensors. A row slice of a row-contiguous array is a no-copy view of the same buffer,
    /// so the members' own path (prefill, exact-row verify) reads exactly the bytes it read before and
    /// the members' original storage is released. Only meaningful when ``copyWouldReplace(_:)`` held
    /// for the members; otherwise the originals stay alive in their shard regardless.
    public static func rebindAsRows(
        _ members: [QuantizedLinear], weight: MLXArray, scales: MLXArray, biases: MLXArray
    ) {
        guard ProcessInfo.processInfo.environment["VMLX_GDN_SHARE_FUSED_ROWS"] != "0" else { return }
        var row = 0
        for member in members {
            let rows = member.weight.dim(0)
            _ = member.update(
                parameters: ModuleParameters.unflattened([
                    "weight": weight[row ..< row + rows],
                    "scales": scales[row ..< row + rows],
                    "biases": biases[row ..< row + rows],
                ]))
            row += rows
        }
        MLX.eval(members.flatMap { [$0.weight, $0.scales] + ($0.biases.map { [$0] } ?? []) })
    }

    /// The packed weights of `members` — the bulk of their bytes — for ``copyWouldReplace(_:)``.
    /// Scales and biases are deliberately not consulted: a model can materialize its weights while
    /// keeping its (f16, ~1/8 the size at 4-bit/g64) affine metadata file-backed. Flash-Next does,
    /// and fusing there measured +5 % decode for +0.6 GB (the metadata copy) on a 54 GB model.
    public static func storage(of members: [QuantizedLinear]) -> [MLXArray] {
        members.map(\.weight)
    }
}
