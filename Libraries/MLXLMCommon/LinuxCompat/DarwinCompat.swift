// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

// Linux stand-ins for the Darwin-only calls the LM libraries make. Compiled only where Darwin does
// not exist, so Apple builds never see these declarations. Each fails the way its Apple original
// does, so the existing error handling keeps working.

#if !canImport(Darwin) && canImport(Glibc)
    import Foundation
    import Glibc

    // MARK: Mach virtual memory (the JANGPress purgeable-tile cache)
    //
    // Linux has no purgeable memory. Tiles become ordinary anonymous mappings that the kernel never
    // discards, so every tile reports itself as non-volatile and is never refaulted from disk.

    // Public, unlike the rest: `JangPressMachError`'s public cases carry a `kern_return_t`.
    public typealias kern_return_t = Int32
    typealias vm_address_t = UInt
    typealias vm_size_t = UInt
    typealias mach_port_t = UInt32

    let KERN_SUCCESS: kern_return_t = 0
    let KERN_FAILURE: kern_return_t = 5
    let mach_task_self_: mach_port_t = 0

    let VM_FLAGS_ANYWHERE: Int32 = 0x1
    let VM_FLAGS_PURGABLE: Int32 = 0x2
    let VM_PURGABLE_SET_STATE: Int32 = 0
    let VM_PURGABLE_GET_STATE: Int32 = 1
    let VM_PURGABLE_NONVOLATILE: Int32 = 0
    let VM_PURGABLE_VOLATILE: Int32 = 1
    let VM_PURGABLE_EMPTY: Int32 = 2
    let VM_VOLATILE_GROUP_7: Int32 = 0x700
    let VM_PURGABLE_BEHAVIOR_LIFO: Int32 = 0x40
    let VM_PURGABLE_NO_AGING: Int32 = 0x10000

    func vm_allocate(
        _ task: mach_port_t, _ address: UnsafeMutablePointer<vm_address_t>, _ size: vm_size_t,
        _ flags: Int32
    ) -> kern_return_t {
        guard let length = Int(exactly: size) else { return KERN_FAILURE }
        let mapped = mmap(
            nil, length, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0)
        guard let mapped, mapped != UnsafeMutableRawPointer(bitPattern: -1) else {
            return KERN_FAILURE
        }
        address.pointee = UInt(bitPattern: mapped)
        return KERN_SUCCESS
    }

    func vm_deallocate(_ task: mach_port_t, _ address: vm_address_t, _ size: vm_size_t)
        -> kern_return_t
    {
        guard let length = Int(exactly: size) else { return KERN_FAILURE }
        return munmap(UnsafeMutableRawPointer(bitPattern: address), length) == 0
            ? KERN_SUCCESS : KERN_FAILURE
    }

    func vm_purgable_control(
        _ task: mach_port_t, _ address: vm_address_t, _ control: Int32,
        _ state: UnsafeMutablePointer<Int32>
    ) -> kern_return_t {
        // SET_STATE hands back the previous state; nothing is ever purged here.
        state.pointee = VM_PURGABLE_NONVOLATILE
        return KERN_SUCCESS
    }

    func mach_absolute_time() -> UInt64 {
        DispatchTime.now().uptimeNanoseconds
    }

    // MARK: CoreFoundation clock (timing instrumentation)

    package typealias CFAbsoluteTime = Double

    package func CFAbsoluteTimeGetCurrent() -> CFAbsoluteTime {
        Date().timeIntervalSinceReferenceDate
    }

    // MARK: CoreFoundation number introspection (a JSON bool versus a JSON number)

    func CFBooleanGetTypeID() -> UInt { 1 }

    func CFGetTypeID(_ number: NSNumber) -> UInt {
        // swift-corelibs-foundation tags booleans with the "c" type encoding. It tags Int8 (CChar)
        // numbers the same way, but JSONSerialization never produces those, so this is exact for
        // the JSON callers.
        String(cString: number.objCType) == "c" ? 1 : 0
    }

    func CFNumberIsFloatType(_ number: NSNumber) -> Bool {
        let encoding = String(cString: number.objCType)
        return encoding == "d" || encoding == "f"
    }
#endif
