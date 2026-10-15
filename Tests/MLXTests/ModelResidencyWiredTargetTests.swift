import Testing

@testable import MLX

/// Regression guard for osaurus#2612: raising the model-residency wired limit to
/// the full working set wired non-reclaimable memory a constrained Mac needed,
/// causing flickering, freezes, and kernel restarts on 16 GB machines.
///
/// `modelResidencyWiredTarget` must never return a target that fails to leave a
/// safe reclaim reserve of PHYSICAL RAM, and must decline (return nil) when the
/// model is a large fraction of RAM.
@Suite("Model-residency wired-limit policy")
struct ModelResidencyWiredTargetTests {
    private static let gib = 1024 * 1024 * 1024
    private func gb(_ n: Double) -> Int { Int(n * Double(Self.gib)) }

    // Reserve mirrors the policy: 35% of physical RAM, floored at 8 GB, capped at 24 GB.
    private func osReserve(physical: Int) -> Int {
        // 48-63 GiB: the request is bounded by XNU's own 79% wire cap, so >= 21% always stays unwired.
        physical >= gb(48) && physical < gb(64)
            ? physical - xnuDefaultUserWireLimit(physicalMemory: physical)
            : max(8 * Self.gib, min(physical / 100 * 35, 24 * Self.gib))
    }

    @Test("128 GB Mac: a 91 GB bundle (GLM-5.3-Flash / Naive-N0.5 JANGHT) wires with a 24 GB reserve")
    func bigMacWiresNinetyGigabyteModel() {
        // Measured: left at MLX's default wired limit, ~11 GB of these weights stayed outside the residency
        // set and decode fell to 1.4-1.8 tok/s at random; wired, 25-26.5 tok/s.
        let physical = gb(128)
        let target = modelResidencyWiredTarget(modelBytes: gb(91), workingSet: gb(107), physicalMemory: physical)
        #expect(target != nil)
        if let target {
            #expect(target >= gb(91))
            #expect(target <= gb(107))
            #expect(physical - target >= gb(24))
        }
    }

    @Test("16 GB Mac: a ~9 GB model is NOT wired (the #2612 regression)")
    func constrainedMacDeclinesLargeModel() {
        // Old code returned min(9GB+headroom, workingSet) = the full ~10.6 GB
        // working set here, wiring the machine into a panic. Must be nil now.
        let target = modelResidencyWiredTarget(
            modelBytes: gb(9), workingSet: gb(10.6), physicalMemory: gb(16))
        #expect(target == nil)
    }

    @Test("16 GB Mac: a ~5 GB model wires, but leaves a safe reserve")
    func constrainedMacWiresSmallModelSafely() {
        let physical = gb(16)
        let target = modelResidencyWiredTarget(
            modelBytes: gb(5), workingSet: gb(10.6), physicalMemory: physical)
        #expect(target != nil)
        if let target {
            #expect(physical - target >= osReserve(physical: physical))  // reclaim reserve preserved
            #expect(target <= gb(10.6))                                   // never above working set
            #expect(target >= gb(5))                                      // whole model fits wired
        }
    }

    @Test("128 GB Mac: a 76 GB model IS wired (the residency win is kept)")
    func spaciousMacWiresLargeModel() {
        let physical = gb(128)
        let target = modelResidencyWiredTarget(
            modelBytes: gb(76), workingSet: gb(96), physicalMemory: physical)
        #expect(target != nil)
        if let target {
            #expect(target >= gb(76))
            #expect(physical - target >= osReserve(physical: physical))
            #expect(target <= gb(96))
        }
    }

    @Test("128 GB Mac: a 101 GB bundle that can't fit safely is declined")
    func spaciousMacDeclinesOversizeModel() {
        let target = modelResidencyWiredTarget(
            modelBytes: gb(101), workingSet: gb(96), physicalMemory: gb(128))
        #expect(target == nil)
    }

    @Test("invariant: any returned target preserves the reclaim reserve")
    func returnedTargetAlwaysLeavesReserve() {
        let physicals = [gb(8), gb(16), gb(24), gb(32), gb(48), gb(64), gb(96), gb(128), gb(192)]
        let models = [gb(2), gb(5), gb(9), gb(14), gb(30), gb(48), gb(76), gb(101), gb(140)]
        for physical in physicals {
            let workingSet = physical * 3 / 4  // rough Metal working-set fraction
            for modelBytes in models {
                guard let target = modelResidencyWiredTarget(
                    modelBytes: modelBytes, workingSet: workingSet, physicalMemory: physical)
                else { continue }
                #expect(physical - target >= osReserve(physical: physical))
                #expect(target <= workingSet)
                // 48-63 GiB Macs pin a partial set rather than none when the model is larger
                // (measured: none 0.2 tok/s, partial 18, whole working set 36-41).
                if !(physical >= gb(48) && physical < gb(64)) { #expect(target >= modelBytes) }
            }
        }
    }

    @Test("48 GiB: Allosaurus requests weights + generation overhead, bounded by Metal and the kernel wire cap")
    func allosaurusOn48GiB() {
        // Measured 2026-10-15 under a 48 GB-like wire budget + RAM pressure: no residency 0.2 tok/s
        // (740 GB page-ins), a set capped at the wire headroom 18 tok/s, the whole working set 36-41 tok/s.
        let weights = 36_971_767_160 + 357_581_088
        let wanted = weights + 5 * Self.gib / 2 + Self.gib
        let kernelCap = xnuDefaultUserWireLimit(physicalMemory: gb(48))
        for ceiling in [gb(34), gb(36), gb(37), gb(40)] {
            let target = modelResidencyWiredTarget(
                modelBytes: weights, workingSet: ceiling, physicalMemory: gb(48))
            #expect(target == min(wanted, ceiling, kernelCap), "never nil: none is the 0.2 tok/s failure")
        }
    }

    @Test("zero/invalid inputs leave the default")
    func invalidInputsDecline() {
        #expect(modelResidencyWiredTarget(modelBytes: 0, workingSet: gb(96), physicalMemory: gb(128)) == nil)
        #expect(modelResidencyWiredTarget(modelBytes: gb(5), workingSet: gb(10), physicalMemory: 0) == nil)
    }
}
