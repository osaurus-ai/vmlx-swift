// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

#if os(Linux)
    import Foundation
    import MLX
    import Testing

    @testable import MLXLMCommon

    /// The Linux stand-ins behave like the Apple APIs they replace.
    @Suite(.serialized) struct LinuxStandInTests {

        /// `JSONValue.from` tells a JSON bool from a number through the CF introspection stand-ins.
        /// Without them, `1` decoded as `true` (the bug the Apple code comments on). `false`, `0`
        /// and `1.0` are the neighbouring cases `Value.swift` warns about.
        @Test func jsonScalarsKeepTheirTypes() throws {
            let parsed = try JSONSerialization.jsonObject(
                with: Data(#"{"t": true, "f": false, "i": 1, "z": 0, "d": 1.5, "w": 1.0}"#.utf8))
            let object = try #require(parsed as? [String: Any])
            #expect(JSONValue.from(try #require(object["t"])) == .bool(true))
            #expect(JSONValue.from(try #require(object["f"])) == .bool(false))
            #expect(JSONValue.from(try #require(object["i"])) == .int(1))
            #expect(JSONValue.from(try #require(object["z"])) == .int(0))
            #expect(JSONValue.from(try #require(object["d"])) == .double(1.5))
            #expect(JSONValue.from(try #require(object["w"])) == .double(1.0))
        }

        @Test func lockSerializesConcurrentIncrements() async {
            let counter = OSAllocatedUnfairLock(initialState: 0)
            await withTaskGroup(of: Void.self) { group in
                for _ in 0 ..< 8 {
                    group.addTask {
                        for _ in 0 ..< 10_000 { counter.withLock { $0 += 1 } }
                    }
                }
            }
            #expect(counter.withLock { $0 } == 80_000)
        }

        @Test func virtualMemoryRoundTrips() throws {
            var address: vm_address_t = 0
            let size = vm_size_t(4 * 4096)
            #expect(vm_allocate(mach_task_self_, &address, size, VM_FLAGS_ANYWHERE) == KERN_SUCCESS)
            let bytes = try #require(UnsafeMutablePointer<UInt8>(bitPattern: UInt(address)))
            bytes[0] = 7
            bytes[Int(size) - 1] = 9
            #expect(bytes[0] == 7 && bytes[Int(size) - 1] == 9)
            var state = VM_PURGABLE_VOLATILE
            #expect(
                vm_purgable_control(mach_task_self_, address, VM_PURGABLE_SET_STATE, &state)
                    == KERN_SUCCESS)
            #expect(state == VM_PURGABLE_NONVOLATILE)
            #expect(vm_deallocate(mach_task_self_, address, size) == KERN_SUCCESS)

            // No mapping of 4 EiB exists: mmap fails, and the stand-in reports it as vm_allocate
            // does, without trapping. A size beyond Int.max fails the same way, before mmap or
            // munmap sees it.
            var huge: vm_address_t = 0
            #expect(
                vm_allocate(mach_task_self_, &huge, vm_size_t(1) << 62, VM_FLAGS_ANYWHERE)
                    == KERN_FAILURE)
            #expect(vm_allocate(mach_task_self_, &huge, .max, VM_FLAGS_ANYWHERE) == KERN_FAILURE)
            #expect(vm_deallocate(mach_task_self_, 0, .max) == KERN_FAILURE)
        }

        /// Stands for the ten former crash sites: on a CPU-only build they aborted with
        /// "Cannot make gpu stream without gpu backend". The work is queued first, so the
        /// synchronization has something to wait for.
        @Test func computeStreamSynchronizesOnACPUOnlyBuild() {
            let doubled = MLXArray([1, 2, 3] as [Float]) * 2
            asyncEval(doubled)
            synchronizeComputeStream()
            #expect(doubled.asArray(Float.self) == [2, 4, 6])
        }

        @Test func gpuStandInReportsNoWorkingSet() {
            #expect(GPU.maxRecommendedWorkingSetBytes() == nil)
        }

        /// MLXLMCommon's `fputs(…, stderr)` reads the `GlibcCompat` stand-in, not Glibc's mutable
        /// global, so the stand-in must be the standard error stream. The test names it through the
        /// module, so it cannot pass by reading Glibc's `stderr` instead.
        @Test func stderrStandInIsTheStandardErrorStream() {
            #expect(fileno(MLXLMCommon.stderr) == STDERR_FILENO)
        }
    }
#endif
