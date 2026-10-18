// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation

/// Installed RAM as seen by memory POLICY (residency target, scheduling limit, allocator-cache sizing).
///
/// `VMLX_EMULATE_PHYSICAL_MEMORY_GIB` replaces it, for diagnosis only: together with an mlock balloon that
/// shrinks the system wire budget, a large Mac then runs exactly the policy a smaller Mac would. That is how the
/// 48 GB Allosaurus rows were reproduced on a 128 GB M5 Max (2026-10-18). It changes no measurement (RSS, wired
/// pages, the wire budget snapshot) — only the RAM size the policies are computed from. Unset in products.
public enum HostPhysicalMemory {
    public static let bytes: UInt64 = {
        let actual = ProcessInfo.processInfo.physicalMemory
        guard let emulated = emulatedBytes(
            RuntimeEnvironment.value("VMLX_EMULATE_PHYSICAL_MEMORY_GIB"))
        else { return actual }
        FileHandle.standardError.write(Data(
            "[Memory] emulated_physical_memory=\(emulated) actual=\(actual) (VMLX_EMULATE_PHYSICAL_MEMORY_GIB, diagnostics only)\n"
                .utf8))
        return emulated
    }()

    /// Reject values that would trap during conversion or overflow downstream Int byte budgets.
    /// Invalid diagnostic input leaves the real physical-memory policy in effect.
    static func emulatedBytes(_ raw: String?) -> UInt64? {
        guard let raw,
            let gib = Double(raw.trimmingCharacters(in: .whitespacesAndNewlines))
        else { return nil }
        let bytes = gib * 1_073_741_824
        guard bytes.isFinite, bytes >= 1, bytes < Double(Int.max) else { return nil }
        return UInt64(bytes)
    }
}
